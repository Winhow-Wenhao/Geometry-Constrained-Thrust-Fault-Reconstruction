#!/usr/bin/env python3
"""Callable single-inline UASS-Net inference and 2D fault post-processing.

This module is the importable counterpart of
``demo/fault_process_2d_end_to_end_github.ipynb``.  The numerical defaults and
operation order intentionally preserve the notebook's inline-600 reference
result.  Fixed input hashes, shapes, and stage counts are checked only by the
explicit :func:`validate_inline600_reference` function, so the main pipeline
can also process other regularly sampled SEG-Y inlines.

Point coordinates use the processed array convention throughout:
``(profile_index, sample_index)`` after crossline/sample downsampling.  They
are not SEG-Y header coordinates.

Example, from the repository root::

    python fault_post_processing_2d.py \
        data/segy/inline600.segy \
        model_real.pth \
        --inline 600 \
        --output-npz outputs/fault2d/inline600.npz
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping
import warnings

import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    map_coordinates,
    uniform_filter1d,
)
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


REFERENCE_MODEL_SHA256 = (
    "d154ab68869cfc9c789b94835cb2614c182c7dd0fa3fd56cf2af4bb9b2b638aa"
)
REFERENCE_SEISMIC_SHA256 = (
    "87ed66c6c91839661bf4a7a765175cbd332e0922ce85d63c09b44b91539ff785"
)
REFERENCE_INLINE = 600
REFERENCE_SHAPE = (2080, 1001)
REFERENCE_COUNTS = {
    "candidate_points": 32195,
    "ridge_core_points": 836,
    "ridge_preserved_points": 32163,
    "grouped_points": 2253,
    "first_dbscan_points": 2031,
    "first_dbscan_clusters": 114,
    "first_curvature_points": 2031,
    "first_angle_points": 1968,
    "second_dbscan_points": 1967,
    "merged_clusters": 81,
    "large_cluster_points": 1254,
    "large_clusters": 15,
    "second_curvature_points": 1254,
    "second_angle_points": 1202,
    "interpolated_points": 1927,
    "interpolated_clusters": 15,
}


@dataclass(frozen=True)
class InferenceConfig:
    """Patch inference settings used by the reference notebook."""

    patch_shape: tuple[int, int] = (256, 256)
    overlap: int = 12

    def __post_init__(self) -> None:
        if len(self.patch_shape) != 2:
            raise ValueError("patch_shape must contain two dimensions.")
        if any(int(size) < 16 or int(size) % 16 for size in self.patch_shape):
            raise ValueError("Each patch dimension must be a multiple of 16 and at least 16.")
        if not 0 < int(self.overlap) < min(self.patch_shape):
            raise ValueError("overlap must be positive and smaller than each patch dimension.")


@dataclass(frozen=True)
class PostprocessConfig:
    """Parameters for the complete 2D post-processing sequence.

    The 17/75/8 merge thresholds are the calibrated legacy values in the
    source notebook.  They are deliberately not silently replaced by the
    manuscript table's nominal 15/75/5 values because that would change the
    published reference output.
    """

    probability_threshold: float = 0.1
    gradient_ratio: float = 0.10
    normal_step: float = 1.0
    ridge_support_radius: int = 8
    min_keep_ratio: float = 0.999
    group_max_gap: int = 20

    first_dbscan_eps: float = 10.0
    first_dbscan_min_samples: int = 5

    curvature_tau: float = 0.5
    curvature_delta: float = 1e-3
    curvature_local_window: int = 7
    curvature_smooth_window: int = 5
    curvature_blend_strength: float = 0.01
    curvature_max_shift: float = 0.05
    angle_threshold: float = 90.0
    angle_max_rounds: int = 10

    second_dbscan_eps: float = 10.0
    second_dbscan_min_samples: int = 3

    merge_theta_max: float = 17.0
    merge_distance_max: float = 75.0
    merge_extrapolation_max: float = 8.0

    min_cluster_points: int = 30
    min_cluster_vertical_range: float = 30.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("probability_threshold must be in [0, 1].")
        if self.gradient_ratio < 0.0 or self.normal_step < 0.0:
            raise ValueError("gradient_ratio and normal_step must be non-negative.")
        if self.ridge_support_radius < 0 or not 0.0 <= self.min_keep_ratio <= 1.0:
            raise ValueError("Invalid ridge preservation settings.")
        if self.group_max_gap < 0:
            raise ValueError("group_max_gap must be non-negative.")
        if self.first_dbscan_eps <= 0 or self.first_dbscan_min_samples <= 0:
            raise ValueError("Invalid first DBSCAN settings.")
        if self.second_dbscan_eps <= 0 or self.second_dbscan_min_samples <= 0:
            raise ValueError("Invalid second DBSCAN settings.")
        if self.curvature_delta <= 0 or not 0.0 <= self.curvature_blend_strength <= 1.0:
            raise ValueError("Invalid curvature settings.")
        if self.curvature_max_shift < 0 or self.angle_max_rounds < 0:
            raise ValueError("Invalid smoothing or angle-filter settings.")
        if min(
            self.merge_theta_max,
            self.merge_distance_max,
            self.merge_extrapolation_max,
        ) <= 0:
            raise ValueError("All merge thresholds must be positive.")
        if self.min_cluster_points < 1 or self.min_cluster_vertical_range < 0:
            raise ValueError("Invalid large-cluster filter settings.")


@dataclass(frozen=True)
class SegyInline:
    """One downsampled inline and its native SEG-Y coordinate axes."""

    inline_number: int
    section: np.ndarray
    crossline_values: np.ndarray
    sample_values: np.ndarray
    native_shape: tuple[int, int]
    crossline_stride: int
    sample_stride: int


@dataclass
class InlinePostprocessResult:
    """All intermediate and final arrays from 2D post-processing."""

    candidate_mask: np.ndarray
    candidate_points: np.ndarray
    ridge_core_mask: np.ndarray
    ridge_core_points: np.ndarray
    ridge_preserved_mask: np.ndarray
    ridge_preserved_points: np.ndarray
    grouped_points: np.ndarray
    first_dbscan_raw_labels: np.ndarray
    first_dbscan_points: np.ndarray
    first_dbscan_labels: np.ndarray
    first_curvature_points: np.ndarray
    first_curvature_labels: np.ndarray
    first_angle_points: np.ndarray
    first_angle_labels: np.ndarray
    second_dbscan_raw_labels: np.ndarray
    second_dbscan_keep_mask: np.ndarray
    second_dbscan_points: np.ndarray
    second_dbscan_carried_labels: np.ndarray
    merged_labels: np.ndarray
    large_clusters: dict[int, np.ndarray]
    large_cluster_points: np.ndarray
    large_cluster_labels: np.ndarray
    second_curvature_points: np.ndarray
    second_curvature_labels: np.ndarray
    second_angle_points: np.ndarray
    second_angle_labels: np.ndarray
    interpolated_clusters: dict[int, np.ndarray]
    interpolated_points: np.ndarray
    interpolated_labels: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def stage_counts(self) -> dict[str, int]:
        return {
            "candidate_points": int(len(self.candidate_points)),
            "ridge_core_points": int(len(self.ridge_core_points)),
            "ridge_preserved_points": int(len(self.ridge_preserved_points)),
            "grouped_points": int(len(self.grouped_points)),
            "first_dbscan_points": int(len(self.first_dbscan_points)),
            "first_dbscan_clusters": _cluster_count(self.first_dbscan_labels),
            "first_curvature_points": int(len(self.first_curvature_points)),
            "first_angle_points": int(len(self.first_angle_points)),
            "second_dbscan_points": int(len(self.second_dbscan_points)),
            "merged_clusters": _cluster_count(self.merged_labels),
            "large_cluster_points": int(len(self.large_cluster_points)),
            "large_clusters": _cluster_count(self.large_cluster_labels),
            "second_curvature_points": int(len(self.second_curvature_points)),
            "second_angle_points": int(len(self.second_angle_points)),
            "interpolated_points": int(len(self.interpolated_points)),
            "interpolated_clusters": int(len(self.interpolated_clusters)),
        }

    def clusters_for_3d(self) -> dict[int, np.ndarray]:
        """Return deterministic ``cluster_id -> (N, 2)`` integer arrays."""

        return {
            int(new_label): np.ascontiguousarray(points, dtype=np.int32)
            for new_label, (_, points) in enumerate(sorted(self.interpolated_clusters.items()))
        }


@dataclass
class SingleInlineResult:
    """Complete inference and post-processing result for one inline."""

    inline_number: int
    seismic_section: np.ndarray
    seismic_normalized: np.ndarray
    probability_map: np.ndarray
    postprocess: InlinePostprocessResult
    inference_config: InferenceConfig
    postprocess_config: PostprocessConfig
    crossline_values: np.ndarray | None = None
    sample_values: np.ndarray | None = None
    seismic_path: Path | None = None
    weights_path: Path | None = None
    weights_sha256: str | None = None

    @property
    def stage_counts(self) -> dict[str, int]:
        return self.postprocess.stage_counts

    def clusters_for_3d(self) -> dict[int, np.ndarray]:
        return self.postprocess.clusters_for_3d()


def _empty_points(dtype: np.dtype[Any] | type = np.float64) -> np.ndarray:
    return np.empty((0, 2), dtype=dtype)


def _empty_labels(dtype: np.dtype[Any] | type = np.int64) -> np.ndarray:
    return np.empty(0, dtype=dtype)


def _cluster_count(labels: np.ndarray) -> int:
    labels = np.asarray(labels)
    return int(np.unique(labels).size) if labels.size else 0


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if isinstance(device, torch.device):
        resolved = device
    elif str(device).lower() == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved


def sha256_file(file_path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(file_path).open("rb") as input_file:
        while block := input_file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


class SegyInlineReader:
    """Stream individual inlines from a regular SEG-Y cube.

    The reader uses ``segyio.iline`` and therefore does not materialize the
    full 3D survey in memory.  Returned point coordinates remain indices of
    the downsampled arrays; sampled header axes are exposed as metadata.
    """

    def __init__(
        self,
        file_path: str | Path,
        crossline_stride: int = 2,
        sample_stride: int = 2,
        strict: bool = False,
    ) -> None:
        self.file_path = Path(file_path)
        self.crossline_stride = int(crossline_stride)
        self.sample_stride = int(sample_stride)
        self.strict = bool(strict)
        if self.crossline_stride <= 0 or self.sample_stride <= 0:
            raise ValueError("Downsampling strides must be positive integers.")
        self._file: Any = None
        self.ilines = np.empty(0, dtype=np.int64)
        self.xlines = np.empty(0, dtype=np.int64)
        self.samples = np.empty(0, dtype=np.float64)

    def __enter__(self) -> "SegyInlineReader":
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Seismic input not found: {self.file_path.resolve()}")
        self._file = segyio.open(str(self.file_path), "r", strict=self.strict)
        try:
            self.ilines = np.asarray(self._file.ilines).copy()
            self.xlines = np.asarray(self._file.xlines).copy()
            self.samples = np.asarray(self._file.samples).copy()
            expected_trace_count = len(self.ilines) * len(self.xlines)
            actual_trace_count = int(self._file.tracecount)
            if actual_trace_count != expected_trace_count:
                raise ValueError(
                    f"SEG-Y trace count {actual_trace_count} does not match "
                    f"regular grid size {expected_trace_count}."
                )
            if len(np.unique(self.ilines)) != len(self.ilines):
                raise ValueError("SEG-Y inline headers are not unique.")
        except BaseException:
            self._file.close()
            self._file = None
            raise
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def processed_xlines(self) -> np.ndarray:
        return self.xlines[:: self.crossline_stride].copy()

    @property
    def processed_samples(self) -> np.ndarray:
        return self.samples[:: self.sample_stride].copy()

    def read(self, inline_number: int) -> SegyInline:
        if self._file is None:
            raise RuntimeError("SegyInlineReader must be used as a context manager.")
        inline_number = int(inline_number)
        if inline_number not in set(map(int, self.ilines)):
            raise ValueError(f"Inline {inline_number} is not present in {self.file_path}.")
        native_section = np.asarray(self._file.iline[inline_number], dtype=np.float32).copy()
        expected_shape = (len(self.xlines), len(self.samples))
        if native_section.shape != expected_shape:
            raise ValueError(
                f"Inline {inline_number} has shape {native_section.shape}; expected {expected_shape}."
            )
        section = native_section[
            :: self.crossline_stride,
            :: self.sample_stride,
        ]
        return SegyInline(
            inline_number=inline_number,
            section=np.ascontiguousarray(section, dtype=np.float32),
            crossline_values=self.processed_xlines,
            sample_values=self.processed_samples,
            native_shape=expected_shape,
            crossline_stride=self.crossline_stride,
            sample_stride=self.sample_stride,
        )


def load_inline_section(
    file_path: str | Path,
    inline_number: int,
    crossline_stride: int = 2,
    sample_stride: int = 2,
) -> np.ndarray:
    """Load one inline using the memory-efficient SEG-Y reader."""

    with SegyInlineReader(file_path, crossline_stride, sample_stride) as reader:
        return reader.read(inline_number).section


def minmax_normalize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("The seismic section must be a finite 2D array.")
    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if not image_max > image_min:
        raise ValueError("The seismic section has a constant amplitude.")
    return np.asarray((image - image_min) / (image_max - image_min), dtype=np.float32)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.layers(inputs)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.b1(inputs), self.b2(inputs), self.b3(inputs), self.b4(inputs)], dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, up_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.se = SEBlock(skip_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv = ConvBlock(up_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Nearest interpolation is part of the checkpoint's reference forward definition.
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="nearest")
        inputs = torch.cat([self.se(skip), inputs], dim=1)
        return self.conv(self.dropout(inputs))


class UASSNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 32,
        )
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.enc5 = ConvBlock(c4, c5)
        self.pool = nn.MaxPool2d(2)
        self.aspp = ASPP(c5, c4)
        self.dec4 = DecoderBlock(c4, c4, c4, dropout)
        self.dec3 = DecoderBlock(c4, c3, c3, dropout)
        self.dec2 = DecoderBlock(c3, c2, c2, dropout)
        self.dec1 = DecoderBlock(c2, c1, c1, dropout)
        self.out_conv = nn.Conv2d(c1, out_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.enc1(inputs)
        encoder2 = self.enc2(self.pool(encoder1))
        encoder3 = self.enc3(self.pool(encoder2))
        encoder4 = self.enc4(self.pool(encoder3))
        encoder5 = self.enc5(self.pool(encoder4))
        decoded = self.aspp(encoder5)
        decoded = self.dec4(decoded, encoder4)
        decoded = self.dec3(decoded, encoder3)
        decoded = self.dec2(decoded, encoder2)
        decoded = self.dec1(decoded, encoder1)
        return self.out_conv(decoded)


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise TypeError("Checkpoint must be a state_dict or contain model/model_state_dict.")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in checkpoint.items()):
        raise TypeError("The selected checkpoint object is not a PyTorch state_dict.")
    if all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def load_uassnet_model(
    weights_path: str | Path,
    device: str | torch.device = "auto",
    base_channels: int = 16,
    dropout: float = 0.2,
    expected_sha256: str | None = None,
) -> tuple[UASSNet, torch.device]:
    """Load a raw or wrapped UASS-Net state dictionary in evaluation mode."""

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path.resolve()}")
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(weights_path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise ValueError(
                f"Model SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}."
            )
    resolved_device = resolve_device(device)
    model = UASSNet(
        in_channels=1,
        out_channels=1,
        base_channels=int(base_channels),
        dropout=float(dropout),
    ).to(resolved_device)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
        try:
            checkpoint = torch.load(weights_path, map_location=resolved_device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location=resolved_device)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model.eval()
    return model, resolved_device


def build_gaussian_overlap_weights(
    overlap: int,
    patch_shape: tuple[int, int],
) -> np.ndarray:
    patch_rows, patch_columns = patch_shape
    if overlap <= 0 or overlap >= min(patch_shape):
        raise ValueError("Overlap must be positive and smaller than each patch dimension.")
    weights = np.ones(patch_shape, dtype=np.float32)
    sigma_factor = 0.5 / ((overlap / 4.0) ** 2)
    edge_weights = np.asarray(
        [np.exp(-((index - overlap + 1) ** 2) * sigma_factor) for index in range(overlap)],
        dtype=np.float32,
    )
    # Preserve the source assignment order: complete rows first, then columns.
    for index in range(overlap):
        weights[index, :] = edge_weights[index]
        weights[patch_rows - index - 1, :] = edge_weights[index]
    for index in range(overlap):
        weights[:, index] = edge_weights[index]
        weights[:, patch_columns - index - 1] = edge_weights[index]
    return weights


def predict_probability_map(
    model: nn.Module,
    seismic_image: np.ndarray,
    patch_shape: tuple[int, int] = (256, 256),
    overlap: int = 12,
    device: str | torch.device = "auto",
) -> np.ndarray:
    """Predict a probability image using patch z-score and Gaussian blending."""

    seismic_image = np.asarray(seismic_image, dtype=np.float32)
    if seismic_image.ndim != 2 or not np.isfinite(seismic_image).all():
        raise ValueError(f"Expected a finite 2D seismic image, received {seismic_image.shape}.")
    resolved_device = resolve_device(device)
    patch_rows, patch_columns = map(int, patch_shape)
    if min(patch_rows, patch_columns) < 16 or patch_rows % 16 or patch_columns % 16:
        raise ValueError("Patch dimensions must be multiples of 16 and at least 16.")
    if not 0 < overlap < min(patch_shape):
        raise ValueError("Overlap must be positive and smaller than each patch dimension.")

    stride_rows = patch_rows - overlap
    stride_columns = patch_columns - overlap
    image_rows, image_columns = seismic_image.shape
    tile_rows = int(np.round((image_rows + overlap) / stride_rows + 0.5))
    tile_columns = int(np.round((image_columns + overlap) / stride_columns + 0.5))
    padded_rows = stride_rows * tile_rows + overlap
    padded_columns = stride_columns * tile_columns + overlap
    padded_image = np.zeros((padded_rows, padded_columns), dtype=np.float32)
    probability_sum = np.zeros_like(padded_image)
    weight_sum = np.zeros_like(padded_image)
    padded_image[:image_rows, :image_columns] = seismic_image
    overlap_weights = build_gaussian_overlap_weights(overlap, patch_shape)

    model.eval()
    with torch.inference_mode():
        for tile_row in range(tile_rows):
            for tile_column in range(tile_columns):
                row_start = tile_row * stride_rows
                column_start = tile_column * stride_columns
                patch = padded_image[
                    row_start : row_start + patch_rows,
                    column_start : column_start + patch_columns,
                ]
                patch_mean = float(np.mean(patch))
                patch_std = float(np.std(patch))
                if not patch_std > 0.0:
                    raise ValueError(f"Constant seismic patch at tile ({tile_row}, {tile_column}).")
                standardized_patch = (patch - patch_mean) / patch_std
                model_input = torch.from_numpy(standardized_patch[None, None]).to(
                    device=resolved_device,
                    dtype=torch.float32,
                )
                patch_probability = torch.sigmoid(model(model_input))[0, 0].cpu().numpy()
                row_slice = slice(row_start, row_start + patch_rows)
                column_slice = slice(column_start, column_start + patch_columns)
                probability_sum[row_slice, column_slice] += patch_probability * overlap_weights
                weight_sum[row_slice, column_slice] += overlap_weights

    blended_probability = np.divide(
        probability_sum,
        weight_sum,
        out=np.zeros_like(probability_sum),
        where=weight_sum > 0.0,
    )
    result = np.asarray(blended_probability[:image_rows, :image_columns], dtype=np.float32)
    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > 1.0):
        raise RuntimeError("Model inference produced invalid probability values.")
    return result


def _disk_structure(radius: int) -> np.ndarray:
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx * xx + yy * yy <= radius * radius


def extract_paper_ridge_core(
    probability_map: np.ndarray,
    candidate_mask: np.ndarray,
    gradient_ratio: float = 0.1,
    normal_step: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract strict gradient ridges using the manuscript normal test."""

    probability_map = np.asarray(probability_map, dtype=np.float32)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if probability_map.ndim != 2 or min(probability_map.shape) < 2:
        raise ValueError("probability_map must be a 2D array with both dimensions at least 2.")
    if candidate_mask.shape != probability_map.shape:
        raise ValueError("candidate_mask and probability_map must have the same shape.")
    grad_axis0, grad_axis1 = np.gradient(probability_map)
    gradient_magnitude = np.hypot(grad_axis0, grad_axis1)
    max_gradient = float(gradient_magnitude.max())
    gradient_cutoff = float(gradient_ratio) * max_gradient
    eps = np.finfo(np.float32).eps
    normal_axis0 = grad_axis0 / (gradient_magnitude + eps)
    normal_axis1 = grad_axis1 / (gradient_magnitude + eps)
    axis0, axis1 = np.indices(probability_map.shape, dtype=np.float32)
    forward_values = map_coordinates(
        probability_map,
        [axis0 + normal_step * normal_axis0, axis1 + normal_step * normal_axis1],
        order=1,
        mode="nearest",
    )
    backward_values = map_coordinates(
        probability_map,
        [axis0 - normal_step * normal_axis0, axis1 - normal_step * normal_axis1],
        order=1,
        mode="nearest",
    )
    local_maximum = (probability_map >= forward_values) & (probability_map >= backward_values)
    if max_gradient > 0.0:
        gradient_supported = gradient_magnitude >= gradient_cutoff
    else:
        gradient_supported = np.zeros_like(candidate_mask, dtype=bool)
    ridge_core_mask = candidate_mask & local_maximum & gradient_supported
    diagnostics = {
        "gradient_axis0": grad_axis0,
        "gradient_axis1": grad_axis1,
        "gradient_magnitude": gradient_magnitude,
        "gradient_cutoff": gradient_cutoff,
        "local_maximum_mask": local_maximum,
        "ridge_core_mask": ridge_core_mask,
    }
    return ridge_core_mask, diagnostics


def preserve_most_points_around_ridges(
    candidate_mask: np.ndarray,
    ridge_core_mask: np.ndarray,
    support_radius: int = 8,
    min_keep_ratio: float = 0.999,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the notebook's deliberately weak ridge-support constraint."""

    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    ridge_core_mask = np.asarray(ridge_core_mask, dtype=bool)
    if candidate_mask.shape != ridge_core_mask.shape:
        raise ValueError("candidate_mask and ridge_core_mask must have the same shape.")
    if not 0.0 <= min_keep_ratio <= 1.0:
        raise ValueError("min_keep_ratio must be in [0, 1].")
    candidate_count = int(candidate_mask.sum())
    if candidate_count == 0:
        return candidate_mask.copy(), {
            "candidate_count": 0,
            "ridge_core_count": 0,
            "retained_count": 0,
            "retention_ratio": 1.0,
            "added_back_count": 0,
        }
    if not ridge_core_mask.any() or min_keep_ratio == 1.0:
        return candidate_mask.copy(), {
            "candidate_count": candidate_count,
            "ridge_core_count": int(ridge_core_mask.sum()),
            "retained_count": candidate_count,
            "retention_ratio": 1.0,
            "added_back_count": 0,
        }
    support_mask = binary_dilation(ridge_core_mask, structure=_disk_structure(support_radius))
    retained_mask = candidate_mask & support_mask
    target_count = int(np.ceil(min_keep_ratio * candidate_count))
    current_count = int(retained_mask.sum())
    added_back_count = 0
    if current_count < target_count:
        missing_mask = candidate_mask & ~retained_mask
        missing_coordinates = np.argwhere(missing_mask)
        distance_to_ridge = distance_transform_edt(~ridge_core_mask)
        missing_distances = distance_to_ridge[missing_mask]
        number_to_add = min(target_count - current_count, len(missing_coordinates))
        nearest_order = np.argsort(missing_distances, kind="stable")[:number_to_add]
        coordinates_to_add = missing_coordinates[nearest_order]
        retained_mask[coordinates_to_add[:, 0], coordinates_to_add[:, 1]] = True
        added_back_count = int(number_to_add)
    retained_count = int(retained_mask.sum())
    return retained_mask, {
        "candidate_count": candidate_count,
        "ridge_core_count": int(ridge_core_mask.sum()),
        "retained_count": retained_count,
        "retention_ratio": retained_count / candidate_count,
        "added_back_count": added_back_count,
    }


def group_by_gap_1d(values: np.ndarray, max_gap: int = 10) -> list[list[int]]:
    values = np.asarray(values)
    if values.size == 0:
        return []
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    groups: list[list[int]] = []
    current_group = [int(sorted_idx[0])]
    for index in range(1, len(sorted_vals)):
        if sorted_vals[index] - sorted_vals[index - 1] <= max_gap:
            current_group.append(int(sorted_idx[index]))
        else:
            groups.append(current_group)
            current_group = [int(sorted_idx[index])]
    groups.append(current_group)
    return groups


def process_all_data(DBS_data: np.ndarray, max_gap: int = 10, mode: str = "x") -> np.ndarray:
    """Reduce each 1D gap group to its middle point (legacy ``Group`` step)."""

    DBS_data = np.asarray(DBS_data)
    if DBS_data.size == 0:
        return _empty_points(np.int32)
    if DBS_data.ndim != 2 or DBS_data.shape[1] != 2:
        raise ValueError("DBS_data must be an N x 2 array.")
    group_f_points: list[list[float]] = []
    if mode == "x":
        x_unique = np.unique(DBS_data[:, 0])
        for x_value in x_unique:
            ys = DBS_data[DBS_data[:, 0] == x_value, 1]
            for group in group_by_gap_1d(ys, max_gap=max_gap):
                y_representative = ys[group[len(group) // 2]]
                group_f_points.append([x_value, y_representative])
    elif mode == "y":
        y_unique = np.unique(DBS_data[:, 1])
        for y_value in y_unique:
            xs = DBS_data[DBS_data[:, 1] == y_value, 0]
            for group in group_by_gap_1d(xs, max_gap=max_gap):
                x_representative = xs[group[len(group) // 2]]
                group_f_points.append([x_representative, y_value])
    else:
        raise ValueError("mode must be 'x' or 'y'.")
    return np.unique(np.asarray(group_f_points, dtype=np.int32), axis=0)


def _ensure_odd_window(window_size: int, n_points: int, minimum: int = 3) -> int | None:
    if n_points < minimum:
        return None
    window_size = max(minimum, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    max_window = n_points if n_points % 2 == 1 else n_points - 1
    window_size = min(window_size, max_window)
    return window_size if window_size >= minimum else None


def _order_points_by_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an N x 2 array.")
    if len(points) < 2:
        order = np.arange(len(points))
        return order, points.copy(), np.array([1.0, 0.0])
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    principal_direction = vh[0]
    projection = centered @ principal_direction
    order = np.argsort(projection, kind="stable")
    return order, points[order], principal_direction


def _arc_length_parameter(ordered_points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    ordered_points = np.asarray(ordered_points, dtype=np.float64)
    if len(ordered_points) == 0:
        return np.empty(0, dtype=np.float64)
    if len(ordered_points) == 1:
        return np.array([0.0], dtype=np.float64)
    segment_length = np.linalg.norm(np.diff(ordered_points, axis=0), axis=1)
    segment_length = np.maximum(segment_length, eps)
    return np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(segment_length)])


def compute_paper_curvature_diagnostics(
    points: np.ndarray,
    local_window: int = 7,
    tau: float = 0.5,
    delta: float = 0.001,
    edge_margin: int = 2,
) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    n_points = len(points)
    empty_result = {
        "order": np.arange(n_points),
        "ordered_points": points.copy(),
        "principal_direction": np.array([1.0, 0.0]),
        "arc_length": np.zeros(n_points, dtype=np.float64),
        "curvature": np.zeros(n_points, dtype=np.float64),
        "local_mean_curvature": np.zeros(n_points, dtype=np.float64),
        "eta": np.zeros(n_points, dtype=np.float64),
        "strict_anomaly_ordered": np.zeros(n_points, dtype=bool),
        "strict_anomaly_original": np.zeros(n_points, dtype=bool),
    }
    if n_points < 5:
        return empty_result
    order, ordered_points, principal_direction = _order_points_by_pca(points)
    s = _arc_length_parameter(ordered_points)
    x = ordered_points[:, 0]
    z = ordered_points[:, 1]
    dx = np.gradient(x, s, edge_order=2)
    dz = np.gradient(z, s, edge_order=2)
    d2x = np.gradient(dx, s, edge_order=2)
    d2z = np.gradient(dz, s, edge_order=2)
    numerator = np.abs(dx * d2z - dz * d2x)
    denominator = np.power(dx * dx + dz * dz, 1.5)
    curvature = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    window = _ensure_odd_window(local_window, n_points)
    local_mean_curvature = (
        curvature.copy()
        if window is None
        else uniform_filter1d(curvature, size=window, mode="nearest")
    )
    eta = np.abs(curvature - local_mean_curvature) / (local_mean_curvature + float(delta))
    strict_anomaly_ordered = eta > float(tau)
    margin = min(max(int(edge_margin), 0), n_points // 2)
    if margin > 0:
        strict_anomaly_ordered[:margin] = False
        strict_anomaly_ordered[-margin:] = False
    strict_anomaly_original = np.zeros(n_points, dtype=bool)
    strict_anomaly_original[order] = strict_anomaly_ordered
    return {
        "order": order,
        "ordered_points": ordered_points,
        "principal_direction": principal_direction,
        "arc_length": s,
        "curvature": curvature,
        "local_mean_curvature": local_mean_curvature,
        "eta": eta,
        "strict_anomaly_ordered": strict_anomaly_ordered,
        "strict_anomaly_original": strict_anomaly_original,
    }


def weak_paper_curvature_smoothing(
    points: np.ndarray,
    local_window: int = 7,
    smoothing_window: int = 5,
    tau: float = 0.5,
    delta: float = 0.001,
    blend_strength: float = 0.01,
    max_shift: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(points, dtype=np.float64)
    if not 0.0 <= blend_strength <= 1.0:
        raise ValueError("blend_strength must be in [0, 1].")
    if max_shift < 0.0:
        raise ValueError("max_shift must be non-negative.")
    diagnostics = compute_paper_curvature_diagnostics(
        points,
        local_window=local_window,
        tau=tau,
        delta=delta,
    )
    order = diagnostics["order"]
    ordered_points = diagnostics["ordered_points"]
    strict_mask = diagnostics["strict_anomaly_ordered"]
    adjusted_ordered = ordered_points.copy()
    n_points = len(points)
    if n_points >= 5 and np.any(strict_mask) and blend_strength > 0.0:
        window = _ensure_odd_window(smoothing_window, n_points)
        if window is not None:
            target = np.column_stack(
                [
                    uniform_filter1d(ordered_points[:, 0], size=window, mode="nearest"),
                    uniform_filter1d(ordered_points[:, 1], size=window, mode="nearest"),
                ]
            )
            displacement = (target - ordered_points) * float(blend_strength)
            displacement_norm = np.linalg.norm(displacement, axis=1)
            scale = np.ones(n_points, dtype=np.float64)
            too_large = displacement_norm > float(max_shift)
            scale[too_large] = float(max_shift) / displacement_norm[too_large]
            displacement *= scale[:, None]
            adjusted_ordered[strict_mask] += displacement[strict_mask]
    adjusted_points = points.copy()
    adjusted_points[order] = adjusted_ordered
    displacement_all = np.linalg.norm(adjusted_points - points, axis=1)
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "input_count": int(n_points),
            "output_count": int(n_points),
            "strict_anomaly_count": int(strict_mask.sum()),
            "adjusted_count": int(np.sum(displacement_all > 0.0)),
            "retention_ratio": 1.0,
            "mean_displacement": float(displacement_all.mean()) if n_points else 0.0,
            "max_displacement": float(displacement_all.max()) if n_points else 0.0,
        }
    )
    return adjusted_points, diagnostics


def apply_weak_curvature_by_cluster(
    data: np.ndarray,
    labels: np.ndarray,
    local_window: int = 7,
    smoothing_window: int = 5,
    tau: float = 0.5,
    delta: float = 0.001,
    blend_strength: float = 0.01,
    max_shift: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) != len(labels):
        raise ValueError("data must be N x 2 and align with labels.")
    adjusted_data = data.copy()
    cluster_diagnostics: dict[Any, dict[str, Any]] = {}
    for cluster_label in np.unique(labels):
        cluster_indices = np.flatnonzero(labels == cluster_label)
        adjusted_points, diagnostics = weak_paper_curvature_smoothing(
            data[cluster_indices],
            local_window=local_window,
            smoothing_window=smoothing_window,
            tau=tau,
            delta=delta,
            blend_strength=blend_strength,
            max_shift=max_shift,
        )
        adjusted_data[cluster_indices] = adjusted_points
        cluster_diagnostics[cluster_label] = diagnostics
    total_strict = sum(item["strict_anomaly_count"] for item in cluster_diagnostics.values())
    total_adjusted = sum(item["adjusted_count"] for item in cluster_diagnostics.values())
    all_displacements = np.linalg.norm(adjusted_data - data, axis=1)
    summary = {
        "input_count": int(len(data)),
        "output_count": int(len(adjusted_data)),
        "retention_ratio": 1.0,
        "strict_anomaly_count": int(total_strict),
        "adjusted_count": int(total_adjusted),
        "mean_displacement": float(all_displacements.mean()) if len(data) else 0.0,
        "max_displacement": float(all_displacements.max()) if len(data) else 0.0,
        "cluster_diagnostics": cluster_diagnostics,
    }
    return adjusted_data, labels.copy(), summary


def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    v1 = np.asarray(p1, dtype=np.float64) - np.asarray(p2, dtype=np.float64)
    v2 = np.asarray(p3, dtype=np.float64) - np.asarray(p2, dtype=np.float64)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denominator == 0.0:
        return 180.0
    cosine = np.clip(np.dot(v1, v2) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def find_bad_indices_by_sorted_angle(
    data: np.ndarray,
    axis: int,
    reverse: bool = False,
    angle_threshold: float = 90.0,
) -> set[int]:
    data = np.asarray(data, dtype=np.float64)
    if len(data) < 3:
        return set()
    sorted_idx = (
        np.argsort(data[:, axis], kind="stable")[::-1]
        if reverse
        else np.argsort(data[:, axis], kind="stable")
    )
    sorted_data = data[sorted_idx]
    bad_indices: set[int] = set()
    for index in range(1, len(sorted_data) - 1):
        if compute_angle(sorted_data[index - 1], sorted_data[index], sorted_data[index + 1]) < float(angle_threshold):
            bad_indices.add(int(sorted_idx[index]))
    return bad_indices


def apply_original_angle_filter(
    data: np.ndarray,
    labels: np.ndarray,
    angle_threshold: float = 90.0,
    max_rounds: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) != len(labels):
        raise ValueError("data must be N x 2 and align with labels.")
    cleaned_data_list: list[np.ndarray] = []
    cleaned_label_list: list[np.ndarray] = []
    removed_by_cluster: dict[Any, int] = {}
    for cluster_label in np.unique(labels):
        cluster_mask = labels == cluster_label
        cleaned = data[cluster_mask].copy()
        cleaned_labels = labels[cluster_mask].copy()
        original_count = len(cleaned)
        for _ in range(int(max_rounds)):
            bad_idx_set: set[int] = set()
            bad_idx_set |= find_bad_indices_by_sorted_angle(cleaned, 0, False, angle_threshold)
            bad_idx_set |= find_bad_indices_by_sorted_angle(cleaned, 0, True, angle_threshold)
            bad_idx_set |= find_bad_indices_by_sorted_angle(cleaned, 1, False, angle_threshold)
            bad_idx_set |= find_bad_indices_by_sorted_angle(cleaned, 1, True, angle_threshold)
            if not bad_idx_set:
                break
            keep_mask = np.array(
                [index not in bad_idx_set for index in range(len(cleaned))],
                dtype=bool,
            )
            cleaned = cleaned[keep_mask]
            cleaned_labels = cleaned_labels[keep_mask]
            if len(cleaned) < 3:
                break
        removed_by_cluster[cluster_label] = int(original_count - len(cleaned))
        if len(cleaned):
            cleaned_data_list.append(cleaned)
            cleaned_label_list.append(cleaned_labels)
    if cleaned_data_list:
        cleaned_data = np.vstack(cleaned_data_list)
        cleaned_labels = np.concatenate(cleaned_label_list)
    else:
        cleaned_data = _empty_points(np.float64)
        cleaned_labels = _empty_labels(labels.dtype)
    summary = {
        "input_count": int(len(data)),
        "output_count": int(len(cleaned_data)),
        "removed_count": int(len(data) - len(cleaned_data)),
        "retention_ratio": len(cleaned_data) / len(data) if len(data) else 1.0,
        "removed_by_cluster": removed_by_cluster,
    }
    return cleaned_data, cleaned_labels, summary


def _pca_cluster_geometry(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("Each cluster must contain at least one 2D point.")
    centroid = points.mean(axis=0)
    centered = points - centroid
    if len(points) == 1 or np.allclose(centered, 0.0):
        direction = np.array([1.0, 0.0])
    else:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        direction = direction / np.linalg.norm(direction)
    return {"centroid": centroid, "direction": direction, "tree": cKDTree(points)}


def _mean_distance_to_pca_line(
    points: np.ndarray,
    centroid: np.ndarray,
    direction: np.ndarray,
) -> float:
    offsets = np.asarray(points, dtype=float) - centroid
    along_line = np.outer(offsets @ direction, direction)
    orthogonal = offsets - along_line
    return float(np.mean(np.linalg.norm(orthogonal, axis=1)))


def _paper_pair_metrics(
    points_i: np.ndarray,
    geometry_i: dict[str, Any],
    points_j: np.ndarray,
    geometry_j: dict[str, Any],
) -> dict[str, float]:
    cosine = np.clip(abs(np.dot(geometry_i["direction"], geometry_j["direction"])), 0.0, 1.0)
    theta_ij = float(np.degrees(np.arccos(cosine)))
    d_ij = float(np.min(geometry_i["tree"].query(points_j, k=1)[0]))
    e_ij = _mean_distance_to_pca_line(
        points_j,
        geometry_i["centroid"],
        geometry_i["direction"],
    )
    e_ji = _mean_distance_to_pca_line(
        points_i,
        geometry_j["centroid"],
        geometry_j["direction"],
    )
    return {
        "theta": theta_ij,
        "distance": d_ij,
        "extrapolation": min(e_ij, e_ji),
        "extrapolation_i_to_j": e_ij,
        "extrapolation_j_to_i": e_ji,
    }


def merge_clusters_by_paper_geometry(
    data: np.ndarray,
    labels: np.ndarray,
    theta_max: float,
    distance_max: float,
    extrapolation_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Iteratively merge clusters and recompute their PCA geometry."""

    data = np.asarray(data, dtype=float)
    labels = np.asarray(labels)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must be an N x 2 array.")
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length.")
    if theta_max <= 0 or distance_max <= 0 or extrapolation_max <= 0:
        raise ValueError("All merge thresholds must be positive.")
    original_labels = np.unique(labels)
    parameters = {
        "theta_max": float(theta_max),
        "distance_max": float(distance_max),
        "extrapolation_max": float(extrapolation_max),
    }
    if len(original_labels) == 0:
        return _empty_labels(), {
            "input_clusters": 0,
            "final_clusters": 0,
            "parameters": parameters,
            "iterations": [],
        }
    clusters = {
        cluster_label: {
            "points": data[labels == cluster_label],
            "members": {cluster_label},
        }
        for cluster_label in original_labels
    }
    history: list[dict[str, int]] = []
    for iteration in range(1, len(original_labels) + 1):
        cluster_ids = sorted(clusters)
        geometries = {
            cluster_id: _pca_cluster_geometry(clusters[cluster_id]["points"])
            for cluster_id in cluster_ids
        }
        parent = {cluster_id: cluster_id for cluster_id in cluster_ids}

        def find(cluster_id: Any) -> Any:
            while parent[cluster_id] != cluster_id:
                parent[cluster_id] = parent[parent[cluster_id]]
                cluster_id = parent[cluster_id]
            return cluster_id

        def union(cluster_i: Any, cluster_j: Any) -> None:
            root_i = find(cluster_i)
            root_j = find(cluster_j)
            if root_i != root_j:
                parent[root_j] = root_i

        qualifying_pairs = 0
        for position, cluster_i in enumerate(cluster_ids):
            points_i = clusters[cluster_i]["points"]
            for cluster_j in cluster_ids[position + 1 :]:
                points_j = clusters[cluster_j]["points"]
                metrics = _paper_pair_metrics(
                    points_i,
                    geometries[cluster_i],
                    points_j,
                    geometries[cluster_j],
                )
                if (
                    metrics["theta"] < theta_max
                    and metrics["distance"] < distance_max
                    and metrics["extrapolation"] < extrapolation_max
                ):
                    union(cluster_i, cluster_j)
                    qualifying_pairs += 1
        components: defaultdict[Any, list[Any]] = defaultdict(list)
        for cluster_id in cluster_ids:
            components[find(cluster_id)].append(cluster_id)
        output_cluster_count = len(components)
        history.append(
            {
                "iteration": iteration,
                "input_clusters": len(cluster_ids),
                "candidate_pairs": len(cluster_ids) * (len(cluster_ids) - 1) // 2,
                "qualifying_pairs": qualifying_pairs,
                "output_clusters": output_cluster_count,
            }
        )
        if output_cluster_count == len(cluster_ids):
            break
        recomputed_clusters: dict[Any, dict[str, Any]] = {}
        for component_ids in components.values():
            component_id = min(component_ids)
            recomputed_clusters[component_id] = {
                "points": np.vstack(
                    [clusters[cluster_id]["points"] for cluster_id in component_ids]
                ),
                "members": set().union(
                    *[clusters[cluster_id]["members"] for cluster_id in component_ids]
                ),
            }
        clusters = recomputed_clusters
    else:
        raise RuntimeError("Iterative cluster merging did not converge.")
    original_to_merged: dict[Any, int] = {}
    ordered_clusters = sorted(clusters.values(), key=lambda cluster: min(cluster["members"]))
    for merged_label, cluster in enumerate(ordered_clusters):
        for original_label in cluster["members"]:
            original_to_merged[original_label] = merged_label
    merged_labels = np.asarray([original_to_merged[label] for label in labels], dtype=int)
    return merged_labels, {
        "input_clusters": int(len(original_labels)),
        "final_clusters": int(len(ordered_clusters)),
        "parameters": parameters,
        "iterations": history,
    }


def _run_dbscan(
    points: np.ndarray,
    eps: float,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw labels, non-noise points, and non-noise labels."""

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an N x 2 array.")
    if len(points) == 0:
        empty = _empty_labels()
        return empty, points.copy(), empty.copy()
    raw_labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    keep_mask = raw_labels != -1
    return raw_labels, points[keep_mask], raw_labels[keep_mask]


def _select_large_clusters(
    data: np.ndarray,
    merged_labels: np.ndarray,
    min_cluster_points: int,
    min_vertical_range: float,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Apply the notebook's 30-point and vertical-range filters."""

    merged_clusters: defaultdict[int, list[np.ndarray]] = defaultdict(list)
    for point, cluster_label in zip(data, merged_labels):
        merged_clusters[int(cluster_label)].append(point)
    array_clusters = {
        cluster_label: np.asarray(cluster_points)
        for cluster_label, cluster_points in merged_clusters.items()
    }
    big_clusters: dict[int, np.ndarray] = {}
    for cluster_points in array_clusters.values():
        vertical_range = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
        if len(cluster_points) >= min_cluster_points and vertical_range > min_vertical_range:
            big_clusters[len(big_clusters)] = cluster_points
    if not big_clusters:
        return {}, _empty_points(np.float64), _empty_labels()
    big_data = np.vstack(list(big_clusters.values()))
    big_labels = np.concatenate(
        [
            np.full(len(cluster_points), cluster_label)
            for cluster_label, cluster_points in big_clusters.items()
        ]
    )
    return big_clusters, big_data, big_labels


def _interpolate_clusters_legacy(
    data: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, list[str]]:
    """Reproduce the source notebook's x-to-y linear interpolation.

    Degenerate clusters that cannot define a function ``y(x)`` are skipped
    with a warning.  This safeguard is inactive for the inline-600 reference.
    """

    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels)
    interpolated_clusters: dict[int, np.ndarray] = {}
    messages: list[str] = []
    for cluster_label in np.unique(labels):
        cluster_points = data[labels == cluster_label]
        cluster_points = cluster_points[np.argsort(cluster_points[:, 0])]
        if len(cluster_points) < 2 or np.ptp(cluster_points[:, 0]) <= 0:
            messages.append(
                f"Skipped cluster {int(cluster_label)}: fewer than two distinct x coordinates."
            )
            continue
        x_coordinates = cluster_points[:, 0]
        y_coordinates = cluster_points[:, 1]
        interpolation = interp1d(
            x_coordinates,
            y_coordinates,
            kind="linear",
            fill_value="extrapolate",
        )
        x_new = np.arange(np.min(x_coordinates), np.max(x_coordinates) + 1).astype(int)
        y_new = np.round(interpolation(x_new)).astype(int)
        interpolated_clusters[int(cluster_label)] = np.stack((x_new, y_new), axis=1)
    if not interpolated_clusters:
        return {}, _empty_points(np.int64), _empty_labels(), messages
    interpolated_points = np.concatenate(list(interpolated_clusters.values()), axis=0)
    interpolated_labels = np.concatenate(
        [
            np.full(len(cluster_points), cluster_label)
            for cluster_label, cluster_points in interpolated_clusters.items()
        ]
    )
    return interpolated_clusters, interpolated_points, interpolated_labels, messages


def process_probability_map_2d(
    probability_map: np.ndarray,
    config: PostprocessConfig | None = None,
) -> InlinePostprocessResult:
    """Run the complete notebook 2D post-processing pipeline."""

    config = config or PostprocessConfig()
    probability_map = np.asarray(probability_map, dtype=np.float32)
    if probability_map.ndim != 2 or min(probability_map.shape) < 2:
        raise ValueError("probability_map must be a 2D array with both dimensions at least 2.")
    if not np.isfinite(probability_map).all():
        raise ValueError("probability_map contains non-finite values.")
    if np.any(probability_map < 0.0) or np.any(probability_map > 1.0):
        raise ValueError("probability_map must contain probabilities in [0, 1].")

    candidate_mask = probability_map > config.probability_threshold
    candidate_points = np.column_stack(np.where(candidate_mask)).astype(np.int32)
    ridge_core_mask, ridge_diagnostics = extract_paper_ridge_core(
        probability_map,
        candidate_mask,
        gradient_ratio=config.gradient_ratio,
        normal_step=config.normal_step,
    )
    ridge_preserved_mask, ridge_preserved_diagnostics = preserve_most_points_around_ridges(
        candidate_mask,
        ridge_core_mask,
        support_radius=config.ridge_support_radius,
        min_keep_ratio=config.min_keep_ratio,
    )
    ridge_core_points = np.argwhere(ridge_core_mask).astype(np.int32)
    ridge_preserved_points = np.argwhere(ridge_preserved_mask).astype(np.int32)
    grouped_points = process_all_data(
        ridge_preserved_points,
        max_gap=config.group_max_gap,
        mode="y",
    )
    grouped_points = process_all_data(
        grouped_points,
        max_gap=config.group_max_gap,
        mode="x",
    )

    first_raw_labels, first_points, first_labels = _run_dbscan(
        grouped_points,
        eps=config.first_dbscan_eps,
        min_samples=config.first_dbscan_min_samples,
    )
    first_curvature_points, first_curvature_labels, first_curvature_summary = (
        apply_weak_curvature_by_cluster(
            first_points,
            first_labels,
            local_window=config.curvature_local_window,
            smoothing_window=config.curvature_smooth_window,
            tau=config.curvature_tau,
            delta=config.curvature_delta,
            blend_strength=config.curvature_blend_strength,
            max_shift=config.curvature_max_shift,
        )
    )
    first_angle_points, first_angle_labels, first_angle_summary = apply_original_angle_filter(
        first_curvature_points,
        first_curvature_labels,
        angle_threshold=config.angle_threshold,
        max_rounds=config.angle_max_rounds,
    )

    if len(first_angle_points):
        second_raw_labels = DBSCAN(
            eps=config.second_dbscan_eps,
            min_samples=config.second_dbscan_min_samples,
        ).fit(first_angle_points).labels_
        second_keep_mask = second_raw_labels != -1
    else:
        second_raw_labels = _empty_labels()
        second_keep_mask = np.empty(0, dtype=bool)
    second_points = first_angle_points[second_keep_mask]
    # The reference workflow intentionally carries first-DBSCAN labels here.
    second_carried_labels = first_angle_labels[second_keep_mask]

    merged_labels, merge_summary = merge_clusters_by_paper_geometry(
        second_points,
        second_carried_labels,
        theta_max=config.merge_theta_max,
        distance_max=config.merge_distance_max,
        extrapolation_max=config.merge_extrapolation_max,
    )
    large_clusters, large_points, large_labels = _select_large_clusters(
        second_points,
        merged_labels,
        min_cluster_points=config.min_cluster_points,
        min_vertical_range=config.min_cluster_vertical_range,
    )
    second_curvature_points, second_curvature_labels, second_curvature_summary = (
        apply_weak_curvature_by_cluster(
            large_points,
            large_labels,
            local_window=config.curvature_local_window,
            smoothing_window=config.curvature_smooth_window,
            tau=config.curvature_tau,
            delta=config.curvature_delta,
            blend_strength=config.curvature_blend_strength,
            max_shift=config.curvature_max_shift,
        )
    )
    second_angle_points, second_angle_labels, second_angle_summary = apply_original_angle_filter(
        second_curvature_points,
        second_curvature_labels,
        angle_threshold=config.angle_threshold,
        max_rounds=config.angle_max_rounds,
    )
    interpolated_clusters, interpolated_points, interpolated_labels, messages = (
        _interpolate_clusters_legacy(second_angle_points, second_angle_labels)
    )

    return InlinePostprocessResult(
        candidate_mask=candidate_mask,
        candidate_points=candidate_points,
        ridge_core_mask=ridge_core_mask,
        ridge_core_points=ridge_core_points,
        ridge_preserved_mask=ridge_preserved_mask,
        ridge_preserved_points=ridge_preserved_points,
        grouped_points=grouped_points,
        first_dbscan_raw_labels=first_raw_labels,
        first_dbscan_points=first_points,
        first_dbscan_labels=first_labels,
        first_curvature_points=first_curvature_points,
        first_curvature_labels=first_curvature_labels,
        first_angle_points=first_angle_points,
        first_angle_labels=first_angle_labels,
        second_dbscan_raw_labels=second_raw_labels,
        second_dbscan_keep_mask=second_keep_mask,
        second_dbscan_points=second_points,
        second_dbscan_carried_labels=second_carried_labels,
        merged_labels=merged_labels,
        large_clusters=large_clusters,
        large_cluster_points=large_points,
        large_cluster_labels=large_labels,
        second_curvature_points=second_curvature_points,
        second_curvature_labels=second_curvature_labels,
        second_angle_points=second_angle_points,
        second_angle_labels=second_angle_labels,
        interpolated_clusters=interpolated_clusters,
        interpolated_points=interpolated_points,
        interpolated_labels=interpolated_labels,
        diagnostics={
            "ridge": ridge_diagnostics,
            "ridge_preservation": ridge_preserved_diagnostics,
            "first_curvature": first_curvature_summary,
            "first_angle": first_angle_summary,
            "merge": merge_summary,
            "second_curvature": second_curvature_summary,
            "second_angle": second_angle_summary,
        },
        warnings=messages,
    )


def process_inline_section(
    model: nn.Module,
    section: np.ndarray,
    inline_number: int,
    inference_config: InferenceConfig | None = None,
    postprocess_config: PostprocessConfig | None = None,
    device: str | torch.device = "auto",
) -> SingleInlineResult:
    """Normalize, infer, and post-process one already loaded seismic section."""

    inference_config = inference_config or InferenceConfig()
    postprocess_config = postprocess_config or PostprocessConfig()
    section = np.asarray(section, dtype=np.float32)
    seismic_normalized = minmax_normalize(section)
    probability_map = predict_probability_map(
        model,
        seismic_normalized,
        patch_shape=inference_config.patch_shape,
        overlap=inference_config.overlap,
        device=device,
    )
    postprocess = process_probability_map_2d(probability_map, postprocess_config)
    return SingleInlineResult(
        inline_number=int(inline_number),
        seismic_section=section,
        seismic_normalized=seismic_normalized,
        probability_map=probability_map,
        postprocess=postprocess,
        inference_config=inference_config,
        postprocess_config=postprocess_config,
    )


def process_single_inline(
    segy_path: str | Path,
    weights_path: str | Path,
    inline_number: int = REFERENCE_INLINE,
    crossline_stride: int = 2,
    sample_stride: int = 2,
    inference_config: InferenceConfig | None = None,
    postprocess_config: PostprocessConfig | None = None,
    device: str | torch.device = "auto",
    model_base_channels: int = 16,
    model_dropout: float = 0.2,
    expected_model_sha256: str | None = None,
) -> SingleInlineResult:
    """Run the end-to-end workflow from SEG-Y and model weights."""

    segy_path = Path(segy_path)
    weights_path = Path(weights_path)
    model, resolved_device = load_uassnet_model(
        weights_path,
        device=device,
        base_channels=model_base_channels,
        dropout=model_dropout,
        expected_sha256=expected_model_sha256,
    )
    with SegyInlineReader(
        segy_path,
        crossline_stride=crossline_stride,
        sample_stride=sample_stride,
    ) as reader:
        seismic_inline = reader.read(inline_number)
    result = process_inline_section(
        model,
        seismic_inline.section,
        inline_number=inline_number,
        inference_config=inference_config,
        postprocess_config=postprocess_config,
        device=resolved_device,
    )
    result.crossline_values = seismic_inline.crossline_values
    result.sample_values = seismic_inline.sample_values
    result.seismic_path = segy_path
    result.weights_path = weights_path
    result.weights_sha256 = sha256_file(weights_path)
    return result


def validate_inline600_reference(
    result: SingleInlineResult,
    verify_file_hashes: bool = True,
) -> dict[str, Any]:
    """Validate the exact reference input identity, shape, and stage counts."""

    errors: list[str] = []
    if result.inline_number != REFERENCE_INLINE:
        errors.append(f"inline is {result.inline_number}, expected {REFERENCE_INLINE}")
    if result.probability_map.shape != REFERENCE_SHAPE:
        errors.append(f"probability shape is {result.probability_map.shape}, expected {REFERENCE_SHAPE}")
    if result.seismic_normalized.shape != result.probability_map.shape:
        errors.append("seismic and probability shapes differ")
    if result.probability_map.dtype != np.float32:
        errors.append(f"probability dtype is {result.probability_map.dtype}, expected float32")
    if not np.isfinite(result.probability_map).all():
        errors.append("probability map contains non-finite values")
    if verify_file_hashes:
        if result.weights_path is None or result.seismic_path is None:
            errors.append("source paths are unavailable for hash validation")
        else:
            model_hash = result.weights_sha256 or sha256_file(result.weights_path)
            if model_hash != REFERENCE_MODEL_SHA256:
                errors.append(f"model SHA-256 is {model_hash}, expected {REFERENCE_MODEL_SHA256}")
            seismic_hash = sha256_file(result.seismic_path)
            if seismic_hash != REFERENCE_SEISMIC_SHA256:
                errors.append(
                    f"SEG-Y SHA-256 is {seismic_hash}, expected {REFERENCE_SEISMIC_SHA256}"
                )
    actual_counts = result.stage_counts
    for name, expected in REFERENCE_COUNTS.items():
        if actual_counts.get(name) != expected:
            errors.append(f"{name} is {actual_counts.get(name)}, expected {expected}")
    if errors:
        raise AssertionError("Inline-600 reference validation failed:\n- " + "\n- ".join(errors))
    return {
        "validated": True,
        "inline": result.inline_number,
        "shape": list(result.probability_map.shape),
        "counts": actual_counts,
    }


def save_clusters_npz(
    clusters: Mapping[int, np.ndarray],
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Save final clusters without pickle, appending ``.npz`` when omitted."""

    output_path = Path(output_path)
    if not output_path.name.endswith(".npz"):
        # Match NumPy's append-rather-than-replace filename behavior, while
        # resolving the actual target before the overwrite check and return.
        output_path = Path(f"{output_path}.npz")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [(int(label), np.asarray(points, dtype=np.int32)) for label, points in sorted(clusters.items())]
    cluster_ids = np.asarray([label for label, _ in ordered], dtype=np.int32)
    offsets = np.zeros(len(ordered) + 1, dtype=np.int64)
    if ordered:
        offsets[1:] = np.cumsum([len(points) for _, points in ordered])
        points = np.ascontiguousarray(np.vstack([points for _, points in ordered]), dtype=np.int32)
    else:
        points = _empty_points(np.int32)
    np.savez_compressed(
        output_path,
        points=points,
        cluster_ids=cluster_ids,
        point_offsets=offsets,
        point_columns=np.asarray(["profile_index", "sample_index"]),
    )
    return output_path


def save_clusters_pickle(
    clusters: Mapping[int, np.ndarray],
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Save a single-inline cluster dictionary for legacy interoperability.

    Pickle must only be loaded from a trusted source.  Prefer
    :func:`save_clusters_npz` unless a downstream legacy tool requires pickle.
    """

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        int(label): np.ascontiguousarray(points, dtype=np.int32)
        for label, points in sorted(clusters.items())
    }
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("wb") as output_file:
        pickle.dump(serializable, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(output_path)
    return output_path


def _config_to_json(config: Any) -> dict[str, Any]:
    values = asdict(config)
    for key, value in list(values.items()):
        if isinstance(value, tuple):
            values[key] = list(value)
    return values


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run UASS-Net inference and complete 2D fault post-processing on one SEG-Y inline."
    )
    parser.add_argument(
        "input_segy",
        type=Path,
        help="Input regular-grid SEG-Y file, normally under data/segy/.",
    )
    parser.add_argument(
        "model_weights",
        type=Path,
        help=(
            "Path to a user-supplied compatible UASS-Net model state dictionary; "
            "the historical reference filename is model_real.pth, which is not "
            "distributed with this repository."
        ),
    )
    parser.add_argument("--inline", type=int, default=REFERENCE_INLINE, dest="inline_number")
    parser.add_argument("--crossline-stride", type=int, default=2)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device.")
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help=(
            "Optional NPZ output; a missing .npz suffix is appended. "
            "Place generated results under outputs/fault2d/."
        ),
    )
    parser.add_argument(
        "--output-pickle",
        type=Path,
        default=None,
        help="Optional trusted pickle output; place generated results under outputs/fault2d/.",
    )
    parser.add_argument("--validate-inline600-reference", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = process_single_inline(
        segy_path=args.input_segy,
        weights_path=args.model_weights,
        inline_number=args.inline_number,
        crossline_stride=args.crossline_stride,
        sample_stride=args.sample_stride,
        device=args.device,
        expected_model_sha256=(
            REFERENCE_MODEL_SHA256 if args.validate_inline600_reference else None
        ),
    )
    validation = None
    if args.validate_inline600_reference:
        validation = validate_inline600_reference(result, verify_file_hashes=True)
    clusters = result.clusters_for_3d()
    if args.output_npz is not None:
        save_clusters_npz(clusters, args.output_npz, overwrite=args.overwrite)
    if args.output_pickle is not None:
        save_clusters_pickle(clusters, args.output_pickle, overwrite=args.overwrite)
    summary = {
        "inline": result.inline_number,
        "seismic_shape": list(result.seismic_section.shape),
        "probability_shape": list(result.probability_map.shape),
        "stage_counts": result.stage_counts,
        "warnings": result.postprocess.warnings,
        "inference_config": _config_to_json(result.inference_config),
        "postprocess_config": _config_to_json(result.postprocess_config),
        "reference_validation": validation,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
