#!/usr/bin/env python3
"""
Real seismic thrust-fault label construction for transfer learning.

This standalone script consolidates the real-data label-construction workflow
developed in internal research notebooks. Those development notebooks are not
part of this source release; this file is the maintained public implementation.

Purpose
-------
Convert manual thrust-fault interpretations on real seismic sections into
paired seismic-label patches for transfer learning.

Workflow
--------
1. Read a SEGY seismic section.
2. Read manual interpretation points from CSV.
3. Convert CSV coordinates to image indices.
4. Construct line-based fault-core labels.
5. Apply anisotropic elliptical dilation:
       rx^2 / a^2 + rz^2 / c^2 <= 1
   with default a=8 and c=4 pixels.
6. Normalize seismic images.
7. Crop seismic and label maps into 256 x 256 patches.
8. Split patches safely by complete inline, or by buffered spatial blocks when
   only one inline is available. Patch-level random splitting is not used.
9. Assert that train/validation patch footprints do not overlap.
10. Save seismic patches, label patches, metadata, manifests, and previews.

Run the commands below from the repository root. Relative paths embedded in a
configuration JSON are also resolved from that working directory.

Quick example for one interpreted line
--------------------------------------
python real_seismic_label_construction.py \
    --segy data/segy/774.segy \
    --csv data/interpretations/774.csv \
    --inline-id 774 \
    --x-offset 800 \
    --z-offset 664 \
    --x-scale 1 \
    --z-scale 4 \
    --label-shape 3201 1410 \
    --downsample-x 2 \
    --downsample-z 2 \
    --out outputs/datasets/real_labels \
    --preview

Multiple-line example
---------------------
Create a JSON file:

[
  {
    "inline_id": 30,
    "segy_path": "data/segy/30.segy",
    "csv_path": "data/interpretations/30.csv",
    "split": "train",
    "x_offset": 800,
    "z_offset": 664,
    "x_scale": 1,
    "z_scale": 4,
    "label_shape_before_downsample": [3201, 1410],
    "downsample_x": 2,
    "downsample_z": 2
  },
  {
    "inline_id": 230,
    "segy_path": "data/segy/230.segy",
    "csv_path": "data/interpretations/230.csv",
    "split": "val",
    "x_offset": 800,
    "z_offset": 664,
    "x_scale": 1,
    "z_scale": 4,
    "label_shape_before_downsample": [3201, 1410],
    "downsample_x": 2,
    "downsample_z": 2
  }
]

Then run:

python real_seismic_label_construction.py \
    --config-json configs/real_line_inputs.json \
    --out outputs/datasets/real_labels \
    --preview
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from file_utils import sha256_file

try:
    import segyio
    HAS_SEGYIO = True
except ImportError:
    HAS_SEGYIO = False


@dataclass
class LineInputConfig:
    """
    Configuration for one interpreted seismic line.

    Coordinate transform:
        x_index = (csv_x - x_offset) / x_scale
        z_index = (csv_z - z_offset) / z_scale

    Shape convention:
        All seismic and label arrays are stored internally in (x, z) order.
    """

    inline_id: int
    segy_path: str
    csv_path: str

    # Optional explicit whole-line assignment. For publication runs, declare
    # train/val for every line. If omitted for every line, the inline strategy
    # selects validation inline(s) deterministically from the seed.
    split: Optional[str] = None

    section_index: int = 0

    x_col: int = 0
    z_col: int = 1
    fault_id_col: Optional[int] = None

    x_offset: float = 0.0
    z_offset: float = 0.0
    x_scale: float = 1.0
    z_scale: float = 1.0

    label_shape_before_downsample: Optional[Tuple[int, int]] = None

    downsample_x: int = 1
    downsample_z: int = 1


@dataclass
class LabelPatchConfig:
    """Parameters for label construction and patch extraction."""

    dilation_a: int = 8
    dilation_c: int = 4

    patch_size: int = 256
    stride: int = 128

    positive_only: bool = True
    min_label_pixels: int = 1

    target_train: int = 200
    target_val: int = 40

    # Safe train/validation splitting. "inline" keeps complete inline groups
    # separate. With only one retained inline it falls back to a buffered
    # spatial tail split; it never falls back to random patch-level splitting.
    split_strategy: str = "inline"
    val_inline_ids: Tuple[int, ...] = ()
    spatial_axis: str = "x"
    spatial_val_fraction: float = 0.2
    spatial_guard: Optional[int] = None

    save_npy: bool = True
    save_dat: bool = False

    seed: int = 2026


def normalize_minmax(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize an image to [0, 1]."""
    image = image.astype(np.float32, copy=False)
    mn = float(np.nanmin(image))
    mx = float(np.nanmax(image))
    return ((image - mn) / (mx - mn + eps)).astype(np.float32)


def crop_to_common_extent(image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop image and label to their common shape.

    Arrays are assumed to be in (x, z) order.
    """
    nx = min(image.shape[0], label.shape[0])
    nz = min(image.shape[1], label.shape[1])
    return image[:nx, :nz], label[:nx, :nz]


def read_segy_section(
    segy_path: str | Path,
    section_index: int = 0,
    downsample_x: int = 1,
    downsample_z: int = 1,
) -> np.ndarray:
    """
    Read a 2D seismic section from a SEGY file.

    Returns:
        section in (x, z) order.

    ``section_index`` is strict: a 3D index must be in range, and a 2D SEGY
    line accepts only index 0. Invalid indices are never silently clipped or
    ignored because the resolved section is part of the split identity.
    """
    if not HAS_SEGYIO:
        raise ImportError(
            "segyio is required to read SEGY files. Install it with: pip install segyio"
        )

    segy_path = Path(segy_path)
    if not segy_path.exists():
        raise FileNotFoundError(f"SEGY file not found: {segy_path}")

    try:
        segy_file = segyio.open(str(segy_path), "r", ignore_geometry=False)
        geometry_opened = True
    except Exception as geometry_error:
        try:
            segy_file = segyio.open(str(segy_path), "r", ignore_geometry=True)
            geometry_opened = False
            print(
                "Structured SEG-Y geometry was unavailable; "
                "reading the file as one unstructured 2D line."
            )
        except Exception as unstructured_error:
            raise RuntimeError(
                f"Could not open SEG-Y {segy_path} with or without geometry. "
                f"Structured-open error: {geometry_error}"
            ) from unstructured_error

    with segy_file as f:
        # segyio may reuse its trace buffer during iteration, so every trace
        # must be copied before advancing to the next one.
        trace_rows = [
            np.array(trace, dtype=np.float32, copy=True) for trace in f.trace
        ]
        if not trace_rows:
            raise ValueError(f"SEGY file contains no traces: {segy_path}")
        traces = np.stack(trace_rows, axis=0)
        sample_count = traces.shape[1]
        trace_count = traces.shape[0]

        if geometry_opened:
            try:
                inlines = np.asarray(segyio.tools.collect(f.ilines))
                xlines = np.asarray(segyio.tools.collect(f.xlines))
                inline_count = len(inlines)
                xline_count = len(xlines)
            except Exception:
                inline_count = 0
                xline_count = 0
        else:
            inline_count = 0
            xline_count = 0

        if (
            inline_count > 0
            and xline_count > 0
            and trace_count == inline_count * xline_count
        ):
            data_3d = traces.reshape((inline_count, xline_count, sample_count))
            section_index = int(section_index)
            if not 0 <= section_index < inline_count:
                raise IndexError(
                    f"section_index={section_index} is outside the valid 3D range "
                    f"[0, {inline_count - 1}] for {segy_path}."
                )
            section = data_3d[section_index, :, :]
            print(f"Read as 3D SEGY: {data_3d.shape}; selected section index {section_index}")
        else:
            if int(section_index) != 0:
                raise ValueError(
                    f"SEGY {segy_path} was read as one 2D line; section_index must "
                    f"be 0, received {section_index}."
                )
            section = traces
            print(f"Read as 2D SEGY line: {section.shape}")

    section = section[::downsample_x, ::downsample_z]
    return section.astype(np.float32)


def read_interpretation_csv(line_cfg: LineInputConfig) -> pd.DataFrame:
    """
    Read manual interpretation CSV and convert coordinates to image indices.

    Returns:
        DataFrame with x_idx, z_idx, and optionally fault_id.
    """
    csv_path = Path(line_cfg.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    arr = df.to_numpy()

    if arr.shape[1] <= max(line_cfg.x_col, line_cfg.z_col):
        raise ValueError(
            f"CSV has {arr.shape[1]} columns, but x_col={line_cfg.x_col}, z_col={line_cfg.z_col}"
        )

    x = (arr[:, line_cfg.x_col].astype(float) - line_cfg.x_offset) / line_cfg.x_scale
    z = (arr[:, line_cfg.z_col].astype(float) - line_cfg.z_offset) / line_cfg.z_scale

    out = pd.DataFrame(
        {
            "x_idx": np.rint(x).astype(np.int32),
            "z_idx": np.rint(z).astype(np.int32),
        }
    )

    if line_cfg.fault_id_col is not None:
        if arr.shape[1] <= line_cfg.fault_id_col:
            raise ValueError(
                f"CSV has {arr.shape[1]} columns, but fault_id_col={line_cfg.fault_id_col}"
            )
        out["fault_id"] = arr[:, line_cfg.fault_id_col]

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out["x_idx"] = out["x_idx"].astype(np.int32)
    out["z_idx"] = out["z_idx"].astype(np.int32)

    return out


def filter_points_to_shape(points: pd.DataFrame, shape: Tuple[int, int]) -> pd.DataFrame:
    """Keep points inside an array shape in (x, z) order."""
    nx, nz = shape
    mask = (
        (points["x_idx"] >= 0)
        & (points["x_idx"] < nx)
        & (points["z_idx"] >= 0)
        & (points["z_idx"] < nz)
    )
    return points.loc[mask].copy()


def bresenham_line(x0: int, z0: int, x1: int, z1: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw a discrete line between two points in (x, z) coordinates.

    Returns:
        xs, zs arrays containing all pixel coordinates on the line.
    """
    dx = abs(x1 - x0)
    dz = abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sz = 1 if z0 < z1 else -1

    x, z = x0, z0
    xs, zs = [], []

    if dx > dz:
        err = dx / 2
        while x != x1:
            xs.append(x)
            zs.append(z)
            err -= dz
            if err < 0:
                z += sz
                err += dx
            x += sx
    else:
        err = dz / 2
        while z != z1:
            xs.append(x)
            zs.append(z)
            err -= dx
            if err < 0:
                x += sx
                err += dz
            z += sz

    xs.append(x1)
    zs.append(z1)
    return np.asarray(xs, dtype=np.int32), np.asarray(zs, dtype=np.int32)


def draw_points_or_polylines(
    points: pd.DataFrame,
    shape: Tuple[int, int],
    connect_by_fault_id: bool = True,
    connect_without_fault_id: bool = False,
) -> np.ndarray:
    """
    Convert manual interpretation points into a line-based fault-core label.

    If a fault_id column exists, points with the same fault_id are connected as
    polylines. If no fault_id exists, the default behavior is to mark only the
    interpreted points, matching a dense point-picking workflow.
    """
    core = np.zeros(shape, dtype=np.uint8)

    if "fault_id" in points.columns and connect_by_fault_id:
        grouped = points.groupby("fault_id")
        for _, group in grouped:
            group = group.sort_values(["x_idx", "z_idx"])
            coords = group[["x_idx", "z_idx"]].to_numpy(dtype=np.int32)

            if len(coords) == 1:
                x, z = coords[0]
                if 0 <= x < shape[0] and 0 <= z < shape[1]:
                    core[x, z] = 1
            else:
                for p0, p1 in zip(coords[:-1], coords[1:]):
                    xs, zs = bresenham_line(
                        int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1])
                    )
                    valid = (
                        (xs >= 0)
                        & (xs < shape[0])
                        & (zs >= 0)
                        & (zs < shape[1])
                    )
                    core[xs[valid], zs[valid]] = 1

    elif connect_without_fault_id:
        coords = points.sort_values(["x_idx", "z_idx"])[["x_idx", "z_idx"]].to_numpy(
            dtype=np.int32
        )
        for p0, p1 in zip(coords[:-1], coords[1:]):
            xs, zs = bresenham_line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
            valid = (
                (xs >= 0)
                & (xs < shape[0])
                & (zs >= 0)
                & (zs < shape[1])
            )
            core[xs[valid], zs[valid]] = 1
    else:
        xs = points["x_idx"].to_numpy(dtype=np.int32)
        zs = points["z_idx"].to_numpy(dtype=np.int32)
        valid = (
            (xs >= 0)
            & (xs < shape[0])
            & (zs >= 0)
            & (zs < shape[1])
        )
        core[xs[valid], zs[valid]] = 1

    return core


def elliptical_structure(a: int = 8, c: int = 4) -> np.ndarray:
    """
    Create an anisotropic elliptical structuring element in (x, z) order.

    Definition:
        rx^2 / a^2 + rz^2 / c^2 <= 1
    """
    rx = np.arange(-a, a + 1)
    rz = np.arange(-c, c + 1)
    RX, RZ = np.meshgrid(rx, rz, indexing="ij")
    structure = (RX**2 / float(a**2) + RZ**2 / float(c**2)) <= 1.0
    return structure.astype(bool)


def anisotropic_elliptical_dilation(
    core_label: np.ndarray,
    a: int = 8,
    c: int = 4,
) -> np.ndarray:
    """
    Expand a line-based fault-core label into a finite-width fault-zone label.
    """
    structure = elliptical_structure(a=a, c=c)
    dilated = binary_dilation(core_label.astype(bool), structure=structure)
    return dilated.astype(np.uint8)


def build_label_for_line(
    line_cfg: LineInputConfig,
    seismic_shape_after_downsample: Tuple[int, int],
    label_cfg: LabelPatchConfig,
    connect_by_fault_id: bool = True,
    connect_without_fault_id: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Build a final downsampled label map for one interpreted line.

    Steps:
        CSV points -> image indices -> core label -> anisotropic dilation -> downsample.
    """
    if line_cfg.label_shape_before_downsample is not None:
        high_shape = tuple(line_cfg.label_shape_before_downsample)
    else:
        high_shape = (
            seismic_shape_after_downsample[0] * line_cfg.downsample_x,
            seismic_shape_after_downsample[1] * line_cfg.downsample_z,
        )

    points = read_interpretation_csv(line_cfg)
    points = filter_points_to_shape(points, high_shape)

    core = draw_points_or_polylines(
        points,
        high_shape,
        connect_by_fault_id=connect_by_fault_id,
        connect_without_fault_id=connect_without_fault_id,
    )

    dilated = anisotropic_elliptical_dilation(
        core,
        a=label_cfg.dilation_a,
        c=label_cfg.dilation_c,
    )

    label_final = dilated[:: line_cfg.downsample_x, :: line_cfg.downsample_z]
    label_final = label_final[
        : seismic_shape_after_downsample[0], : seismic_shape_after_downsample[1]
    ]

    metadata = {
        "inline_id": line_cfg.inline_id,
        "num_csv_points_after_filter": int(len(points)),
        "high_resolution_label_shape": list(high_shape),
        "downsample": [line_cfg.downsample_x, line_cfg.downsample_z],
        "final_label_shape": list(label_final.shape),
        "dilation": {
            "type": "anisotropic_elliptical",
            "a": label_cfg.dilation_a,
            "c": label_cfg.dilation_c,
        },
        "coordinate_transform": {
            "x_index": f"(csv_x - {line_cfg.x_offset}) / {line_cfg.x_scale}",
            "z_index": f"(csv_z - {line_cfg.z_offset}) / {line_cfg.z_scale}",
        },
    }

    return label_final.astype(np.uint8), metadata


def crop_patches(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
    positive_only: bool = True,
    min_label_pixels: int = 1,
) -> List[Dict]:
    """
    Crop paired seismic and label maps into fixed-size patches.

    Arrays are expected in (x, z) order.
    """
    image, label = crop_to_common_extent(image, label)
    nx, nz = image.shape

    records: List[Dict] = []
    for x0 in range(0, max(nx - patch_size + 1, 1), stride):
        for z0 in range(0, max(nz - patch_size + 1, 1), stride):
            img_patch = image[x0 : x0 + patch_size, z0 : z0 + patch_size]
            lab_patch = label[x0 : x0 + patch_size, z0 : z0 + patch_size]

            if img_patch.shape != (patch_size, patch_size):
                continue
            if lab_patch.shape != (patch_size, patch_size):
                continue

            n_label = int(lab_patch.sum())
            if positive_only and n_label < min_label_pixels:
                continue

            records.append(
                {
                    "x0": int(x0),
                    "z0": int(z0),
                    "image": normalize_minmax(img_patch),
                    "label": lab_patch.astype(np.uint8),
                    "label_pixels": n_label,
                }
            )

    return records


def split_records(
    records: List[Dict],
    label_cfg: LabelPatchConfig,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split records without allowing patch-level train/validation leakage.

    This compatibility wrapper returns only the two record lists. Use
    :func:`split_records_with_summary` when the audit summary is also needed.
    """
    train, val, _ = split_records_with_summary(records, label_cfg)
    return train, val


def _validate_split_config(label_cfg: LabelPatchConfig) -> None:
    """Validate parameters that affect train/validation independence."""
    if label_cfg.patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if label_cfg.stride <= 0:
        raise ValueError("stride must be positive.")
    if label_cfg.target_train <= 0 or label_cfg.target_val <= 0:
        raise ValueError("target_train and target_val must both be positive.")
    if label_cfg.split_strategy not in {"inline", "spatial"}:
        raise ValueError("split_strategy must be 'inline' or 'spatial'.")
    if label_cfg.spatial_axis not in {"x", "z"}:
        raise ValueError("spatial_axis must be 'x' or 'z'.")
    if not 0.0 < label_cfg.spatial_val_fraction < 1.0:
        raise ValueError("spatial_val_fraction must be strictly between 0 and 1.")
    if label_cfg.spatial_guard is not None and label_cfg.spatial_guard < 0:
        raise ValueError("spatial_guard must be non-negative.")


def _source_key(record: Dict) -> str:
    """Return the public logical source ID stored in manifests."""
    if record.get("source_id") is not None:
        return str(record["source_id"])
    return (
        f"inline_{int(record.get('inline_id', -1))}"
        f"_section_{int(record.get('section_index', 0))}"
    )


def _physical_source_key(record: Dict) -> str:
    """Return a content-based section identity for leakage checks when available."""
    checksum = record.get("source_segy_sha256")
    if checksum:
        return f"sha256:{checksum}:section:{int(record.get('section_index', 0))}"
    return _source_key(record)


def _patch_bounds(record: Dict, patch_size: int) -> Tuple[int, int, int, int]:
    """Return a patch footprint as half-open ``(x0, z0, x1, z1)`` bounds."""
    x0 = int(record["x0"])
    z0 = int(record["z0"])
    return x0, z0, x0 + patch_size, z0 + patch_size


def validate_no_train_val_overlap(
    train_records: List[Dict],
    val_records: List[Dict],
    patch_size: int,
) -> None:
    """
    Raise if any train and validation patches overlap on the same source.

    Touching half-open boundaries are allowed; sharing even one source pixel is
    not. Patches from different physical seismic sections are independent.
    """
    train_by_source: Dict[str, List[Tuple[int, int, int, int]]] = {}
    val_by_source: Dict[str, List[Tuple[int, int, int, int]]] = {}
    for record in train_records:
        train_by_source.setdefault(_physical_source_key(record), []).append(
            _patch_bounds(record, patch_size)
        )
    for record in val_records:
        val_by_source.setdefault(_physical_source_key(record), []).append(
            _patch_bounds(record, patch_size)
        )

    for source_id in sorted(set(train_by_source) & set(val_by_source)):
        for train_box in train_by_source[source_id]:
            tx0, tz0, tx1, tz1 = train_box
            for val_box in val_by_source[source_id]:
                vx0, vz0, vx1, vz1 = val_box
                overlaps_x = tx0 < vx1 and vx0 < tx1
                overlaps_z = tz0 < vz1 and vz0 < tz1
                if overlaps_x and overlaps_z:
                    raise RuntimeError(
                        "Train/validation leakage detected for source "
                        f"{source_id!r}: train footprint {train_box} overlaps "
                        f"validation footprint {val_box}."
                    )


def _shuffle_and_cap(records: List[Dict], target: int, seed: int) -> List[Dict]:
    """Deterministically shuffle one already-independent pool and cap its size."""
    indices = np.arange(len(records))
    np.random.default_rng(seed).shuffle(indices)
    return [records[int(i)] for i in indices[:target]]


def _split_by_explicit_assignment(
    records: List[Dict],
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Split when every input line explicitly declares ``train`` or ``val``."""
    values = [record.get("requested_split") for record in records]
    declared = [value is not None for value in values]
    if any(declared) and not all(declared):
        raise ValueError(
            "Explicit line splits are incomplete: set split='train' or split='val' "
            "for every line, or omit split from every line."
        )

    normalized = [str(value).lower() for value in values]
    invalid = sorted(set(normalized) - {"train", "val"})
    if invalid:
        raise ValueError(f"Unsupported explicit split value(s): {invalid}.")

    train = [record for record, value in zip(records, normalized) if value == "train"]
    val = [record for record, value in zip(records, normalized) if value == "val"]
    if not train or not val:
        raise ValueError(
            "Explicit line assignments must retain at least one patch in both "
            "the train and validation pools."
        )

    train_inline_ids = sorted({int(record["inline_id"]) for record in train})
    val_inline_ids = sorted({int(record["inline_id"]) for record in val})
    overlap = sorted(set(train_inline_ids) & set(val_inline_ids))
    if overlap:
        raise ValueError(
            "Each inline must belong to only one split; duplicated train/val "
            f"inline IDs: {overlap}."
        )

    train_sources = {_physical_source_key(record) for record in train}
    val_sources = {_physical_source_key(record) for record in val}
    source_overlap = sorted(train_sources & val_sources)
    if source_overlap:
        raise ValueError(
            "Each physical seismic section must belong to only one split; "
            f"duplicated source sections: {source_overlap}."
        )

    return train, val, {
        "assignment": "explicit_line_config",
        "train_inline_ids": train_inline_ids,
        "val_inline_ids": val_inline_ids,
    }


def _choose_validation_inlines(
    inline_ids: List[int],
    label_cfg: LabelPatchConfig,
) -> List[int]:
    """Choose complete validation inline groups deterministically."""
    known = sorted(set(inline_ids))
    requested = sorted(set(int(value) for value in label_cfg.val_inline_ids))
    if requested:
        unknown = sorted(set(requested) - set(known))
        if unknown:
            raise ValueError(
                f"--val-inline-ids contains IDs absent from retained patches: {unknown}."
            )
        if len(requested) == len(known):
            raise ValueError("Validation inline IDs cannot include every available inline.")
        return requested

    desired_fraction = label_cfg.target_val / float(
        label_cfg.target_train + label_cfg.target_val
    )
    n_val = int(round(len(known) * desired_fraction))
    n_val = max(1, min(len(known) - 1, n_val))
    shuffled = np.asarray(known, dtype=np.int64)
    np.random.default_rng(label_cfg.seed).shuffle(shuffled)
    return sorted(int(value) for value in shuffled[:n_val])


def _split_by_inline(
    records: List[Dict],
    label_cfg: LabelPatchConfig,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Keep all patches from each inline in exactly one split."""
    inline_ids = sorted({int(record["inline_id"]) for record in records})
    if len(inline_ids) < 2:
        if label_cfg.val_inline_ids:
            raise ValueError(
                "--val-inline-ids requires at least two inline groups with retained patches."
            )
        return _split_by_spatial_blocks(
            records,
            label_cfg,
            requested_strategy="inline",
            fallback_reason="only_one_retained_inline",
        )

    val_inline_ids = _choose_validation_inlines(inline_ids, label_cfg)
    val_set = set(val_inline_ids)
    train = [record for record in records if int(record["inline_id"]) not in val_set]
    val = [record for record in records if int(record["inline_id"]) in val_set]
    if not train or not val:
        raise RuntimeError(
            "Inline-group split produced an empty pool. Check positive-patch "
            "filtering or choose different --val-inline-ids."
        )

    train_inline_ids = sorted(set(inline_ids) - val_set)
    return train, val, {
        "assignment": (
            "configured_validation_inlines"
            if label_cfg.val_inline_ids
            else "seeded_complete_inline_groups"
        ),
        "train_inline_ids": train_inline_ids,
        "val_inline_ids": val_inline_ids,
    }


def _split_by_spatial_blocks(
    records: List[Dict],
    label_cfg: LabelPatchConfig,
    requested_strategy: str = "spatial",
    fallback_reason: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Split each source along one axis, discarding a buffered boundary band."""
    axis = label_cfg.spatial_axis
    origin_key = f"{axis}0"
    size_key = "source_nx" if axis == "x" else "source_nz"
    guard = (
        label_cfg.patch_size
        if label_cfg.spatial_guard is None
        else int(label_cfg.spatial_guard)
    )

    grouped: Dict[str, List[Dict]] = {}
    for record in records:
        grouped.setdefault(_physical_source_key(record), []).append(record)

    train: List[Dict] = []
    val: List[Dict] = []
    discarded: List[Dict] = []
    boundaries: List[Dict] = []

    for source_id in sorted(grouped):
        source_records = grouped[source_id]
        if any(size_key not in record or origin_key not in record for record in source_records):
            raise ValueError(
                f"Spatial split requires {size_key} and {origin_key} on every record."
            )
        source_lengths = {int(record[size_key]) for record in source_records}
        if len(source_lengths) != 1:
            raise ValueError(
                f"Spatial split requires one consistent {size_key} value for "
                f"source {source_id!r}."
            )
        source_length = next(iter(source_lengths))
        last_start = source_length - label_cfg.patch_size
        legal_starts = list(range(0, last_start + 1, label_cfg.stride))
        if not legal_starts:
            raise ValueError(
                f"Source {source_id!r} is shorter than patch_size along {axis}."
            )
        legal_start_set = set(legal_starts)
        invalid_starts = sorted(
            {
                int(record[origin_key])
                for record in source_records
                if int(record[origin_key]) not in legal_start_set
            }
        )
        if invalid_starts:
            raise ValueError(
                f"Source {source_id!r} contains non-grid {origin_key} values: "
                f"{invalid_starts[:5]}."
            )

        n_val_origins = max(
            1,
            int(math.ceil(len(legal_starts) * label_cfg.spatial_val_fraction)),
        )
        boundary = int(legal_starts[-n_val_origins])
        train_limit = boundary - guard

        source_train = 0
        source_val = 0
        source_discarded = 0
        for record in source_records:
            start = int(record[origin_key])
            end = start + label_cfg.patch_size
            if start >= boundary:
                val.append(record)
                source_val += 1
            elif end <= train_limit:
                train.append(record)
                source_train += 1
            else:
                discarded.append(record)
                source_discarded += 1

        boundaries.append(
            {
                "source_id": _source_key(source_records[0]),
                "physical_source_id": source_id,
                "axis": axis,
                "source_length": source_length,
                "validation_boundary": boundary,
                "guard_pixels": guard,
                "train_patch_end_limit": train_limit,
                "train_pool_count": source_train,
                "validation_pool_count": source_val,
                "discarded_boundary_count": source_discarded,
            }
        )

    if not train or not val:
        raise RuntimeError(
            "Buffered spatial split could not retain patches in both pools. "
            "Use more spatial coverage, choose the other --spatial-axis, reduce "
            "--spatial-guard explicitly, or add another independently interpreted inline."
        )

    return train, val, {
        "assignment": "buffered_spatial_tail",
        "requested_strategy": requested_strategy,
        "fallback_reason": fallback_reason,
        "spatial_axis": axis,
        "spatial_val_fraction": label_cfg.spatial_val_fraction,
        "spatial_guard": guard,
        "boundaries": boundaries,
        "discarded_boundary_count": len(discarded),
        "train_inline_ids": sorted({int(record["inline_id"]) for record in train}),
        "val_inline_ids": sorted({int(record["inline_id"]) for record in val}),
    }


def split_records_with_summary(
    records: List[Dict],
    label_cfg: LabelPatchConfig,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Create independent train/validation pools and return an audit summary."""
    _validate_split_config(label_cfg)
    if not records:
        raise ValueError("Cannot split an empty record list.")

    source_inline_ids: Dict[str, set[int]] = {}
    for record in records:
        if "inline_id" not in record:
            raise ValueError("Every patch record must contain inline_id.")
        source_inline_ids.setdefault(_physical_source_key(record), set()).add(
            int(record["inline_id"])
        )
    inconsistent_sources = sorted(
        source for source, ids in source_inline_ids.items() if len(ids) > 1
    )
    if inconsistent_sources:
        raise ValueError(
            "One physical source section was assigned multiple inline IDs: "
            f"{inconsistent_sources}."
        )

    seen_candidates: set[Tuple[str, int, int, int]] = set()
    duplicate_candidates: List[Tuple[str, int, int, int]] = []
    for record in records:
        if "x0" not in record or "z0" not in record:
            raise ValueError("Every patch record must contain x0 and z0.")
        key = (
            _physical_source_key(record),
            int(record["x0"]),
            int(record["z0"]),
            label_cfg.patch_size,
        )
        if key in seen_candidates:
            duplicate_candidates.append(key)
        seen_candidates.add(key)
    if duplicate_candidates:
        raise ValueError(
            "Duplicate candidate patches were generated before splitting; check "
            "for repeated physical line configurations. First duplicate: "
            f"{duplicate_candidates[0]}."
        )

    declarations = [record.get("requested_split") for record in records]
    has_any_explicit = any(value is not None for value in declarations)
    if has_any_explicit:
        if label_cfg.split_strategy != "inline":
            raise ValueError(
                "Explicit per-line split assignments cannot be combined with "
                "--split-strategy spatial."
            )
        if label_cfg.val_inline_ids:
            raise ValueError(
                "Explicit per-line split assignments cannot be combined with "
                "--val-inline-ids."
            )
        train_pool, val_pool, details = _split_by_explicit_assignment(records)
        resolved_strategy = "inline"
    elif label_cfg.split_strategy == "inline":
        train_pool, val_pool, details = _split_by_inline(records, label_cfg)
        resolved_strategy = (
            "spatial" if details.get("assignment") == "buffered_spatial_tail" else "inline"
        )
    else:
        if label_cfg.val_inline_ids:
            raise ValueError("--val-inline-ids requires --split-strategy inline.")
        train_pool, val_pool, details = _split_by_spatial_blocks(records, label_cfg)
        resolved_strategy = "spatial"

    validate_no_train_val_overlap(train_pool, val_pool, label_cfg.patch_size)

    train = _shuffle_and_cap(train_pool, label_cfg.target_train, label_cfg.seed + 1)
    val = _shuffle_and_cap(val_pool, label_cfg.target_val, label_cfg.seed + 2)
    if not train or not val:
        raise RuntimeError("Safe splitting must retain at least one patch in each split.")

    validate_no_train_val_overlap(train, val, label_cfg.patch_size)

    summary = {
        "requested_strategy": label_cfg.split_strategy,
        "resolved_strategy": resolved_strategy,
        "seed": label_cfg.seed,
        "target_train_upper_bound": label_cfg.target_train,
        "target_val_upper_bound": label_cfg.target_val,
        "train_pool_before_cap": len(train_pool),
        "val_pool_before_cap": len(val_pool),
        "train_count": len(train),
        "val_count": len(val),
        "train_dropped_by_cap": len(train_pool) - len(train),
        "val_dropped_by_cap": len(val_pool) - len(val),
        "overlap_assertion": "passed",
        **details,
    }
    return train, val, summary


def save_preview_png(
    path: Path,
    image: np.ndarray,
    label: np.ndarray,
) -> None:
    """Save a seismic-label overlay preview figure."""
    import matplotlib.pyplot as plt

    overlay = np.zeros((image.shape[1], image.shape[0], 4), dtype=np.float32)
    overlay[label.T > 0] = [1, 0, 0, 1]

    plt.figure(figsize=(4, 4))
    plt.imshow(image.T, cmap="gray", aspect="auto")
    plt.imshow(overlay, aspect="auto")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150)
    plt.close()


def cached_sha256_file(
    path: str | Path,
    cache: Optional[Dict[str, str]] = None,
) -> str:
    """Hash an input once per resolved path during a multi-line run."""
    cache_key = str(Path(path).resolve())
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    checksum = sha256_file(path)
    if cache is not None:
        cache[cache_key] = checksum
    return checksum


def assert_output_directory_ready(out_dir: str | Path) -> None:
    """Reject output directories containing payload from an earlier run."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {out_dir}")

    payload = sorted(
        (
            path
            for path in out_dir.iterdir()
            if not (
                path.name == ".gitignore"
                and path.is_file()
                and not path.is_symlink()
            )
        ),
        key=lambda path: path.name,
    )
    if payload:
        preview = ", ".join(path.name for path in payload[:5])
        suffix = " ..." if len(payload) > 5 else ""
        raise FileExistsError(
            f"Output directory {out_dir} is not empty ({preview}{suffix}). "
            "Use a new/empty directory so stale patches cannot enter the dataset. "
            "Only a regular repository .gitignore file is allowed."
        )


def _sample_id(record: Dict, patch_size: int) -> str:
    """Build a stable logical ID from the physical source and patch footprint."""
    return (
        f"{_source_key(record)}"
        f"_x{int(record['x0']):06d}"
        f"_z{int(record['z0']):06d}"
        f"_p{patch_size}"
    )


def save_records(
    records: List[Dict],
    out_dir: str | Path,
    split: str,
    save_npy: bool = True,
    save_dat: bool = False,
    preview: bool = False,
) -> List[Dict]:
    """
    Save patches and return one auditable manifest row per sample.
    """
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'.")
    if not save_npy and not save_dat:
        raise ValueError("At least one of save_npy or save_dat must be enabled.")

    out_dir = Path(out_dir)
    seis_dir = out_dir / split / "seismic"
    label_dir = out_dir / split / "labels"
    meta_dir = out_dir / split / "metadata"
    fig_dir = out_dir / split / "preview"

    for d in [seis_dir, label_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if preview:
        fig_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict] = []
    seen_sample_ids: set[str] = set()
    for i, rec in enumerate(records):
        stem = f"{split}_{i:06d}"

        image = rec["image"].astype(np.float32)
        label = rec["label"].astype(np.uint8)
        if image.ndim != 2 or image.shape != label.shape or image.shape[0] != image.shape[1]:
            raise ValueError(
                f"Sample {stem} has incompatible image/label shapes: "
                f"{image.shape} and {label.shape}."
            )
        patch_size = int(image.shape[0])
        sample_id = _sample_id(rec, patch_size)
        if sample_id in seen_sample_ids:
            raise RuntimeError(f"Duplicate logical sample ID: {sample_id}")
        seen_sample_ids.add(sample_id)

        seismic_npy: Optional[Path] = None
        label_npy: Optional[Path] = None
        seismic_dat: Optional[Path] = None
        label_dat: Optional[Path] = None

        if save_npy:
            seismic_npy = seis_dir / f"{stem}.npy"
            label_npy = label_dir / f"{stem}.npy"
            np.save(seismic_npy, image)
            np.save(label_npy, label)

        if save_dat:
            seismic_dat = seis_dir / f"{stem}.dat"
            label_dat = label_dir / f"{stem}.dat"
            image.tofile(seismic_dat)
            label.tofile(label_dat)

        metadata = {k: v for k, v in rec.items() if k not in ["image", "label"]}
        metadata.update(
            {
                "sample_id": sample_id,
                "split": split,
                "array_axis_order": ["x", "z"],
                "image_shape": list(image.shape),
                "patch_size": patch_size,
            }
        )
        metadata_path = meta_dir / f"{stem}.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)

        if preview:
            save_preview_png(fig_dir / f"{stem}.png", image, label)

        x0, z0, x1, z1 = _patch_bounds(rec, patch_size)
        canonical_seismic = seismic_npy if seismic_npy is not None else seismic_dat
        canonical_label = label_npy if label_npy is not None else label_dat
        assert canonical_seismic is not None and canonical_label is not None

        def relative(path: Optional[Path]) -> str:
            return "" if path is None else path.relative_to(out_dir).as_posix()

        def optional_hash(path: Optional[Path]) -> str:
            return "" if path is None else sha256_file(path)

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "source_id": _source_key(rec),
                "inline_id": int(rec["inline_id"]),
                "section_index": int(rec.get("section_index", 0)),
                "x0": x0,
                "z0": z0,
                "x1": x1,
                "z1": z1,
                "patch_size": patch_size,
                "label_pixels": int(rec["label_pixels"]),
                "array_axis_order": "x,z",
                "source_segy": str(rec.get("source_segy", "")),
                "source_csv": str(rec.get("source_csv", "")),
                "source_segy_sha256": str(rec.get("source_segy_sha256", "")),
                "source_csv_sha256": str(rec.get("source_csv_sha256", "")),
                "seismic_file": relative(canonical_seismic),
                "label_file": relative(canonical_label),
                "metadata_file": relative(metadata_path),
                "seismic_sha256": sha256_file(canonical_seismic),
                "label_sha256": sha256_file(canonical_label),
                "metadata_sha256": sha256_file(metadata_path),
                "seismic_dat_file": relative(seismic_dat),
                "label_dat_file": relative(label_dat),
                "seismic_dat_sha256": optional_hash(seismic_dat),
                "label_dat_sha256": optional_hash(label_dat),
            }
        )

    return manifest_rows


PATCH_MANIFEST_FIELDS = [
    "sample_id",
    "split",
    "source_id",
    "inline_id",
    "section_index",
    "x0",
    "z0",
    "x1",
    "z1",
    "patch_size",
    "label_pixels",
    "array_axis_order",
    "source_segy",
    "source_csv",
    "source_segy_sha256",
    "source_csv_sha256",
    "seismic_file",
    "label_file",
    "metadata_file",
    "seismic_sha256",
    "label_sha256",
    "metadata_sha256",
    "seismic_dat_file",
    "label_dat_file",
    "seismic_dat_sha256",
    "label_dat_sha256",
]


def write_patch_manifest(rows: List[Dict], path: str | Path) -> None:
    """Write the complete per-sample split and checksum manifest."""
    path = Path(path)
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("patch_manifest.csv would contain duplicate sample IDs.")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PATCH_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_source_id(line_cfg: LineInputConfig, source_segy_sha256: str) -> str:
    """Identify a section reproducibly from its SEGY content checksum."""
    suffix = source_segy_sha256[:12]
    return (
        f"inline_{int(line_cfg.inline_id)}"
        f"_section_{int(line_cfg.section_index)}_{suffix}"
    )


def process_one_line(
    line_cfg: LineInputConfig,
    label_cfg: LabelPatchConfig,
    connect_by_fault_id: bool = True,
    connect_without_fault_id: bool = False,
    source_checksum_cache: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Process one interpreted seismic line into cropped 256 x 256 patches.
    """
    seismic = read_segy_section(
        line_cfg.segy_path,
        section_index=line_cfg.section_index,
        downsample_x=line_cfg.downsample_x,
        downsample_z=line_cfg.downsample_z,
    )
    seismic = normalize_minmax(seismic)

    label, label_meta = build_label_for_line(
        line_cfg,
        seismic_shape_after_downsample=seismic.shape,
        label_cfg=label_cfg,
        connect_by_fault_id=connect_by_fault_id,
        connect_without_fault_id=connect_without_fault_id,
    )

    seismic, label = crop_to_common_extent(seismic, label)

    patches = crop_patches(
        seismic,
        label,
        patch_size=label_cfg.patch_size,
        stride=label_cfg.stride,
        positive_only=label_cfg.positive_only,
        min_label_pixels=label_cfg.min_label_pixels,
    )

    source_segy_sha256 = cached_sha256_file(
        line_cfg.segy_path, source_checksum_cache
    )
    source_csv_sha256 = cached_sha256_file(
        line_cfg.csv_path, source_checksum_cache
    )
    source_id = make_source_id(line_cfg, source_segy_sha256)
    for p in patches:
        p.update(
            {
                "source_id": source_id,
                "inline_id": int(line_cfg.inline_id),
                "section_index": int(line_cfg.section_index),
                "source_nx": int(seismic.shape[0]),
                "source_nz": int(seismic.shape[1]),
                "source_segy": str(line_cfg.segy_path),
                "source_csv": str(line_cfg.csv_path),
                "source_segy_sha256": source_segy_sha256,
                "source_csv_sha256": source_csv_sha256,
                "requested_split": line_cfg.split,
            }
        )

    metadata = {
        "source_id": source_id,
        "line_config": asdict(line_cfg),
        "label_metadata": label_meta,
        "seismic_shape": list(seismic.shape),
        "array_axis_order": ["x", "z"],
        "input_checksums": {
            "segy_sha256": source_segy_sha256,
            "interpretation_csv_sha256": source_csv_sha256,
        },
        "num_patches": int(len(patches)),
    }

    return patches, metadata


def load_line_inputs_from_json(path: str | Path) -> List[LineInputConfig]:
    """
    Load multiple LineInputConfig objects from a JSON file.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = [raw]

    if not isinstance(raw, list) or not raw:
        raise ValueError("Line input JSON must contain one object or a non-empty list.")

    configs: List[LineInputConfig] = []
    for raw_item in raw:
        if not isinstance(raw_item, dict):
            raise TypeError("Every line input JSON entry must be an object.")
        item = dict(raw_item)
        if (
            "label_shape_before_downsample" in item
            and item["label_shape_before_downsample"] is not None
        ):
            item["label_shape_before_downsample"] = tuple(item["label_shape_before_downsample"])
        if item.get("split") is not None:
            item["split"] = str(item["split"]).lower()
        configs.append(LineInputConfig(**item))

    return configs


def validate_line_inputs(
    line_inputs: List[LineInputConfig],
    label_cfg: LabelPatchConfig,
) -> None:
    """Validate whole-line assignments before expensive SEGY processing."""
    if not line_inputs:
        raise ValueError("At least one line input is required.")

    configured_inline_ids = {int(line.inline_id) for line in line_inputs}
    if label_cfg.val_inline_ids:
        requested_val_ids = set(int(value) for value in label_cfg.val_inline_ids)
        unknown = sorted(requested_val_ids - configured_inline_ids)
        if unknown:
            raise ValueError(
                f"val_inline_ids contains IDs absent from the configuration: {unknown}."
            )
        if requested_val_ids == configured_inline_ids:
            raise ValueError(
                "val_inline_ids cannot include every configured inline; training "
                "would have no complete inline group."
            )

    splits = [line.split.lower() if line.split is not None else None for line in line_inputs]
    invalid = sorted({value for value in splits if value not in {None, "train", "val"}})
    if invalid:
        raise ValueError(f"Unsupported line split value(s): {invalid}.")
    declared = [value is not None for value in splits]
    if any(declared) and not all(declared):
        raise ValueError(
            "Explicit split declarations are all-or-none: assign every line to "
            "train/val, or omit split from every line."
        )
    if all(declared):
        if set(splits) != {"train", "val"}:
            raise ValueError(
                "Explicit line assignments must include at least one train line "
                "and at least one validation line."
            )
        if label_cfg.split_strategy != "inline" or label_cfg.val_inline_ids:
            raise ValueError(
                "Explicit line assignments require split_strategy='inline' and "
                "cannot be combined with val_inline_ids."
            )

    inline_assignments: Dict[int, set[str]] = {}
    for line, split in zip(line_inputs, splits):
        if split is not None:
            inline_assignments.setdefault(int(line.inline_id), set()).add(split)
    conflicting = sorted(
        inline_id for inline_id, values in inline_assignments.items() if len(values) > 1
    )
    if conflicting:
        raise ValueError(
            "The same inline cannot be assigned to train and val: "
            f"{conflicting}."
        )


def make_single_line_input_from_args(args: argparse.Namespace) -> LineInputConfig:
    """
    Build one LineInputConfig from command-line arguments.
    """
    if args.segy is None or args.csv is None or args.inline_id is None:
        raise ValueError(
            "For single-line mode, provide --segy, --csv, and --inline-id, "
            "or use --config-json for multiple lines."
        )

    label_shape = tuple(args.label_shape) if args.label_shape is not None else None

    return LineInputConfig(
        inline_id=int(args.inline_id),
        segy_path=str(args.segy),
        csv_path=str(args.csv),
        section_index=int(args.section_index),
        x_col=int(args.x_col),
        z_col=int(args.z_col),
        fault_id_col=args.fault_id_col,
        x_offset=float(args.x_offset),
        z_offset=float(args.z_offset),
        x_scale=float(args.x_scale),
        z_scale=float(args.z_scale),
        label_shape_before_downsample=label_shape,
        downsample_x=int(args.downsample_x),
        downsample_z=int(args.downsample_z),
    )


def run_split_self_test() -> None:
    """Exercise the leakage barriers without requiring external seismic data."""
    base = {
        "z0": 0,
        "section_index": 0,
        "source_nx": 20,
        "source_nz": 4,
        "label_pixels": 1,
        "requested_split": None,
    }

    inline_records: List[Dict] = []
    for inline_id in (10, 20, 30):
        for x0 in (0, 2, 4):
            inline_records.append(
                {
                    **base,
                    "x0": x0,
                    "inline_id": inline_id,
                    "source_id": f"inline_{inline_id}",
                }
            )
    inline_cfg = LabelPatchConfig(
        patch_size=4,
        stride=2,
        target_train=100,
        target_val=100,
        seed=7,
    )
    inline_train, inline_val, inline_summary = split_records_with_summary(
        inline_records, inline_cfg
    )
    if inline_summary["resolved_strategy"] != "inline":
        raise AssertionError("Multi-inline self-test did not resolve to inline splitting.")
    if {record["inline_id"] for record in inline_train} & {
        record["inline_id"] for record in inline_val
    }:
        raise AssertionError("An inline appeared in both self-test splits.")

    spatial_records = [
        {
            **base,
            "x0": x0,
            "inline_id": 10,
            "source_id": "single_inline",
        }
        for x0 in range(0, 17, 2)
    ]
    spatial_train, spatial_val, spatial_summary = split_records_with_summary(
        spatial_records, inline_cfg
    )
    if spatial_summary["resolved_strategy"] != "spatial":
        raise AssertionError("Single-inline self-test did not use spatial fallback.")
    validate_no_train_val_overlap(spatial_train, spatial_val, inline_cfg.patch_size)
    train_end = max(record["x0"] + inline_cfg.patch_size for record in spatial_train)
    val_start = min(record["x0"] for record in spatial_val)
    if val_start - train_end < spatial_summary["spatial_guard"]:
        raise AssertionError("Spatial self-test did not preserve the configured guard.")

    try:
        validate_no_train_val_overlap(
            [{**base, "x0": 0, "inline_id": 1, "source_id": "same"}],
            [{**base, "x0": 2, "inline_id": 1, "source_id": "same"}],
            patch_size=4,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Deliberate overlapping footprints were not rejected.")

    validate_no_train_val_overlap(
        [{**base, "x0": 0, "inline_id": 1, "source_id": "same"}],
        [{**base, "x0": 4, "inline_id": 1, "source_id": "same"}],
        patch_size=4,
    )
    print("Split self-test passed: inline isolation, spatial guard, and overlap assertion.")


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser.
    """
    parser = argparse.ArgumentParser(
        description="Construct real seismic thrust-fault labels and crop 256x256 patches."
    )

    # Either provide a config JSON or a single-line set of paths.
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="JSON file with one or more line configs (recommended location: configs/).",
    )

    parser.add_argument(
        "--segy",
        type=str,
        default=None,
        help="Path to a SEG-Y file for single-line mode (recommended location: data/segy/).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a manual interpretation CSV (recommended location: data/interpretations/).",
    )
    parser.add_argument(
        "--inline-id",
        type=int,
        default=None,
        help="Inline ID for single-line mode.",
    )
    parser.add_argument(
        "--section-index",
        type=int,
        default=0,
        help="Strict zero-based 3D section index; a 2D SEGY line requires 0.",
    )

    parser.add_argument("--x-col", type=int, default=0)
    parser.add_argument("--z-col", type=int, default=1)
    parser.add_argument("--fault-id-col", type=int, default=None)

    parser.add_argument("--x-offset", type=float, default=0.0)
    parser.add_argument("--z-offset", type=float, default=0.0)
    parser.add_argument("--x-scale", type=float, default=1.0)
    parser.add_argument("--z-scale", type=float, default=1.0)
    parser.add_argument("--label-shape", nargs=2, type=int, default=None, metavar=("NX", "NZ"))

    parser.add_argument("--downsample-x", type=int, default=1)
    parser.add_argument("--downsample-z", type=int, default=1)

    # Label and patch parameters.
    parser.add_argument("--dilation-a", type=int, default=8)
    parser.add_argument("--dilation-c", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--min-label-pixels", type=int, default=1)
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Keep patches without fault labels.",
    )

    parser.add_argument("--target-train", type=int, default=200)
    parser.add_argument("--target-val", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument(
        "--split-strategy",
        choices=["inline", "spatial"],
        default="inline",
        help=(
            "Safe split strategy. Default 'inline' keeps complete inline IDs "
            "independent and falls back to a buffered spatial split for one inline."
        ),
    )
    parser.add_argument(
        "--val-inline-ids",
        nargs="+",
        type=int,
        default=None,
        metavar="INLINE",
        help="Complete inline ID(s) assigned to validation under the inline strategy.",
    )
    parser.add_argument(
        "--spatial-axis",
        choices=["x", "z"],
        default="x",
        help="Axis used by the explicit or single-inline spatial split (default: x).",
    )
    parser.add_argument(
        "--spatial-val-fraction",
        type=float,
        default=0.2,
        help="Fraction of legal patch origins reserved at the spatial tail (default: 0.2).",
    )
    parser.add_argument(
        "--spatial-guard",
        type=int,
        default=None,
        help="Gap in pixels between train and validation footprints (default: patch size).",
    )

    parser.add_argument(
        "--save-dat",
        action="store_true",
        help="Also save patches as raw .dat files.",
    )
    parser.add_argument("--preview", action="store_true", help="Save seismic-label overlay PNGs.")

    parser.add_argument(
        "--connect-without-fault-id",
        action="store_true",
        help="Connect all points even without a fault_id column.",
    )
    parser.add_argument(
        "--no-connect-by-fault-id",
        action="store_true",
        help="Do not connect points even when a fault_id column exists.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="outputs/datasets/real_labels",
        help="Output dataset directory (default: outputs/datasets/real_labels).",
    )
    parser.add_argument(
        "--self-test-split",
        action="store_true",
        help="Run synthetic leakage-barrier tests without reading SEG-Y/CSV inputs.",
    )

    return parser


def main() -> None:
    """
    Command-line entry point.
    """
    args = build_arg_parser().parse_args()

    if args.self_test_split:
        run_split_self_test()
        return

    label_cfg = LabelPatchConfig(
        dilation_a=args.dilation_a,
        dilation_c=args.dilation_c,
        patch_size=args.patch_size,
        stride=args.stride,
        positive_only=not args.include_background,
        min_label_pixels=args.min_label_pixels,
        target_train=args.target_train,
        target_val=args.target_val,
        split_strategy=args.split_strategy,
        val_inline_ids=tuple(args.val_inline_ids or ()),
        spatial_axis=args.spatial_axis,
        spatial_val_fraction=args.spatial_val_fraction,
        spatial_guard=args.spatial_guard,
        save_npy=True,
        save_dat=args.save_dat,
        seed=args.seed,
    )
    _validate_split_config(label_cfg)

    if args.config_json is not None:
        line_inputs = load_line_inputs_from_json(args.config_json)
    else:
        line_inputs = [make_single_line_input_from_args(args)]
    validate_line_inputs(line_inputs, label_cfg)

    out_dir = Path(args.out)
    assert_output_directory_ready(out_dir)

    effective_config = {
        "line_inputs": [asdict(line) for line in line_inputs],
        "label_patch_config": asdict(label_cfg),
        "connect_by_fault_id": not args.no_connect_by_fault_id,
        "connect_without_fault_id": args.connect_without_fault_id,
    }
    canonical_config = json.dumps(
        effective_config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    effective_config_sha256 = hashlib.sha256(canonical_config).hexdigest()

    all_patches: List[Dict] = []
    all_line_metadata: List[Dict] = []
    source_checksum_cache: Dict[str, str] = {}

    configured_sections: Dict[Tuple[str, int], int] = {}
    for position, line_cfg in enumerate(line_inputs):
        segy_checksum = cached_sha256_file(
            line_cfg.segy_path, source_checksum_cache
        )
        physical_section = (segy_checksum, int(line_cfg.section_index))
        if physical_section in configured_sections:
            first_position = configured_sections[physical_section]
            raise ValueError(
                "The same physical SEGY section is configured more than once "
                f"(line config positions {first_position} and {position}). Merge "
                "its interpretation inputs before patch construction."
            )
        configured_sections[physical_section] = position

    out_dir.mkdir(parents=True, exist_ok=True)

    for line_cfg in line_inputs:
        patches, metadata = process_one_line(
            line_cfg,
            label_cfg,
            connect_by_fault_id=not args.no_connect_by_fault_id,
            connect_without_fault_id=args.connect_without_fault_id,
            source_checksum_cache=source_checksum_cache,
        )
        all_patches.extend(patches)
        all_line_metadata.append(metadata)
        print(f"Inline {line_cfg.inline_id}: {len(patches)} patches")

    if len(all_patches) == 0:
        raise RuntimeError(
            "No patches were generated. Check coordinate transforms, label shape, "
            "patch size, and whether positive_only filtering removed all patches."
        )

    train_records, val_records, split_summary = split_records_with_summary(
        all_patches, label_cfg
    )
    print(f"Total patches: {len(all_patches)}")
    print(f"Resolved split strategy: {split_summary['resolved_strategy']}")
    print(f"Train patches: {len(train_records)}")
    print(f"Validation patches: {len(val_records)}")

    train_manifest_rows = save_records(
        train_records,
        out_dir,
        split="train",
        save_npy=label_cfg.save_npy,
        save_dat=label_cfg.save_dat,
        preview=args.preview,
    )
    val_manifest_rows = save_records(
        val_records,
        out_dir,
        split="val",
        save_npy=label_cfg.save_npy,
        save_dat=label_cfg.save_dat,
        preview=args.preview,
    )

    patch_manifest_path = out_dir / "patch_manifest.csv"
    write_patch_manifest(train_manifest_rows + val_manifest_rows, patch_manifest_path)

    line_metadata_path = out_dir / "line_metadata.json"
    with line_metadata_path.open("w", encoding="utf-8") as file:
        json.dump(all_line_metadata, file, indent=2, sort_keys=True)

    label_config_path = out_dir / "label_patch_config.json"
    with label_config_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(label_cfg), file, indent=2, sort_keys=True)

    dataset_manifest = {
        "schema_version": "real-label-dataset/2.0",
        "artifact_type": "paired-seismic-fault-label-patches",
        "array_axis_order": ["x", "z"],
        "effective_config_sha256": effective_config_sha256,
        "effective_config": effective_config,
        "sources": [
            {
                "source_id": metadata["source_id"],
                "inline_id": metadata["line_config"]["inline_id"],
                "section_index": metadata["line_config"]["section_index"],
                "segy_path": metadata["line_config"]["segy_path"],
                "interpretation_csv_path": metadata["line_config"]["csv_path"],
                "seismic_shape": metadata["seismic_shape"],
                "candidate_patch_count": metadata["num_patches"],
                **metadata["input_checksums"],
            }
            for metadata in all_line_metadata
        ],
        "split": split_summary,
        "counts": {
            "candidate_patches": len(all_patches),
            "positive_only": label_cfg.positive_only,
            "train_patches": len(train_records),
            "validation_patches": len(val_records),
        },
        "files": {
            "patch_manifest": patch_manifest_path.name,
            "patch_manifest_sha256": sha256_file(patch_manifest_path),
            "line_metadata": line_metadata_path.name,
            "line_metadata_sha256": sha256_file(line_metadata_path),
            "label_patch_config": label_config_path.name,
            "label_patch_config_sha256": sha256_file(label_config_path),
        },
    }
    dataset_manifest_path = out_dir / "dataset_manifest.json"
    with dataset_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(dataset_manifest, file, indent=2, sort_keys=True)

    print("Train/validation footprint overlap assertion: passed")
    print(f"Patch manifest: {patch_manifest_path.resolve()}")
    print(f"Saved dataset to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
