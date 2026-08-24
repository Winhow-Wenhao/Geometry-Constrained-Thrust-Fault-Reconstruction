#!/usr/bin/env python3
"""Forward-progressive 3D fault-line linking and surface reconstruction.

The cross-section linking stage is an operationalization based on Equations
(26)--(34) of the manuscript:

* symmetric 3D Chamfer distance (Equation 27; directional means are summed),
* PCA orientation difference (Equation 28),
* exponential spatial-orientation similarity (Equation 26),
* a +M future-section search for forward progressive links, a strict
  similarity threshold, one neighborhood-wide best match per source line,
  and K-section consecutive validation (Equations 29--34).

Equation (29) describes a +/- M neighborhood. To obtain deterministic acyclic
trajectories, this implementation sorts sections and evaluates each symmetric
candidate pair once, from the earlier section toward the later section. This
forward-only best-match scope is an explicit implementation choice, not a
claim of algebraic equivalence to a bidirectional argmax.

The manuscript states that validated trajectories are interpolated and
smoothed, but it does not prescribe a unique numerical method. This module
therefore makes that implementation choice explicit: each 2D line is ordered
and resampled by normalized arc length, line coordinates are interpolated
linearly across inlines, and the resulting parameter surface is smoothed with
a configurable Gaussian filter.

The canonical 2D-to-3D interface is a trusted ``line_n`` pickle accompanied by
a versioned ``.metadata.json`` sidecar. The sidecar records the authoritative
inline mapping and coordinate convention; this module discovers it
automatically and verifies the data SHA-256 before unpickling. A pickle without
a sidecar remains supported as a legacy input with an explicit or historical
mapping. Public 3D outputs use JSON, CSV, and non-object NPZ files.

Expected legacy input schema::

    {"line_0": {cluster_id: ndarray(shape=(N, 2))}, ...}

The two point columns are profile position and depth/time sample. With the
default mapping, ``line_0`` is inline 300 and ``line_30`` is inline 330.

Example, from the repository root::

    python fault_surface_reconstruction_3d.py \
        outputs/fault2d/fault_lines_2d_300_330.pkl \
        --output-dir outputs/fault3d/fault3d_output_300_330 \
        --allow-unsafe-pickle
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import pickle
import platform
import re
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


LOGGER = logging.getLogger("fault3d")
LINE_KEY_PATTERN = re.compile(r"^line_(\d+)$")
SCHEMA_VERSION = "1.0"
ALGORITHM_NAME = "forward-progressive-spatial-correlation-3d-fault-reconstruction"
FAULT_LINES_2D_SCHEMA_VERSION = "fault-lines-2d/1.0"
FAULT_LINES_2D_ARTIFACT_TYPE = "fault-lines-2d-interface"
FAULT_LINES_2D_POINT_COLUMNS = ("profile_index", "sample_index")
LEGACY_ARTIFACT_TYPES = {
    "3D-loader-compatible legacy fault-line pickle",
    FAULT_LINES_2D_ARTIFACT_TYPE,
}


@dataclass(frozen=True)
class ReconstructionConfig:
    """Configuration for paper-consistent linking and explicit surface generation."""

    inline_origin: int = 300
    inline_step: int = 1
    inline_start: int = 300
    inline_end: int = 330

    distance_weight: float = 0.03
    orientation_weight: float = 1.0
    max_neighbor_sections: int = 5
    similarity_threshold: float = 0.7
    min_consecutive_sections: int = 3

    section_axis_scale: float = 1.0
    profile_axis_scale: float = 1.0
    depth_axis_scale: float = 1.0

    surface_samples: int = 128
    smooth_sigma_inline: float = 1.0
    smooth_sigma_along: float = 1.0
    observed_weight: float = 0.25

    def validate(self) -> None:
        if self.inline_step <= 0:
            raise ValueError("inline_step must be positive.")
        if self.inline_end < self.inline_start:
            raise ValueError("inline_end must be greater than or equal to inline_start.")
        if self.distance_weight < 0.0 or self.orientation_weight < 0.0:
            raise ValueError("Similarity weights must be non-negative.")
        if not 0.0 < self.similarity_threshold < 1.0:
            raise ValueError("similarity_threshold must lie strictly between 0 and 1.")
        if self.max_neighbor_sections < 1:
            raise ValueError("max_neighbor_sections must be at least 1.")
        if self.min_consecutive_sections < 2:
            raise ValueError("min_consecutive_sections must be at least 2.")
        if min(
            self.section_axis_scale,
            self.profile_axis_scale,
            self.depth_axis_scale,
        ) <= 0.0:
            raise ValueError("All coordinate-axis scales must be positive.")
        if self.surface_samples < 2:
            raise ValueError("surface_samples must be at least 2.")
        if self.smooth_sigma_inline < 0.0 or self.smooth_sigma_along < 0.0:
            raise ValueError("Surface smoothing sigmas must be non-negative.")
        if not 0.0 <= self.observed_weight <= 1.0:
            raise ValueError("observed_weight must lie in [0, 1].")

    @property
    def axis_scales(self) -> tuple[float, float, float]:
        return (
            self.section_axis_scale,
            self.profile_axis_scale,
            self.depth_axis_scale,
        )


@dataclass
class FaultLine:
    """One 2D fault line embedded in the 3D processing-pixel coordinate system."""

    line_id: str
    source_key: str
    section_index: int
    inline_number: int
    cluster_id: str
    points_2d: np.ndarray = field(repr=False)
    points_3d: np.ndarray = field(repr=False)
    tangent: np.ndarray = field(repr=False)
    bbox_min: np.ndarray = field(repr=False)
    bbox_max: np.ndarray = field(repr=False)
    tree: cKDTree = field(repr=False)


@dataclass
class CandidateMatch:
    """An exactly scored candidate match between two different sections."""

    source_id: str
    target_id: str
    source_section_index: int
    target_section_index: int
    source_inline: int
    target_inline: int
    section_gap: int
    chamfer_ab: float
    chamfer_ba: float
    chamfer_distance: float
    orientation_difference: float
    similarity: float
    above_threshold: bool
    source_best: bool = False
    selected: bool = False
    track_edge: bool = False


@dataclass
class FaultTrack:
    """A validated, section-consecutive sequence of 2D fault lines."""

    track_id: str
    line_ids: list[str]
    edge_indices: list[int]
    inline_numbers: list[int]
    section_indices: list[int]
    min_similarity: float
    mean_similarity: float


@dataclass
class FaultSurface:
    """A parameterized surface generated from one validated fault track."""

    track_id: str
    inline_numbers: np.ndarray = field(repr=False)
    normalized_arc_length: np.ndarray = field(repr=False)
    profile_grid: np.ndarray = field(repr=False)
    depth_grid: np.ndarray = field(repr=False)
    vertices: np.ndarray = field(repr=False)
    faces: np.ndarray = field(repr=False)
    observation_rmse: float


@dataclass(frozen=True)
class LoadedFaultLines2D:
    """Validated 2D input together with its effective 3D mapping."""

    sections: dict[int, list[FaultLine]]
    input_metadata: dict[str, Any]
    config: ReconstructionConfig


def sha256_file(file_path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(file_path).open("rb") as input_file:
        while block := input_file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def metadata_path_for_fault_lines(input_pickle: str | Path) -> Path:
    """Return the canonical metadata sidecar path for a 2D result pickle."""

    return Path(input_pickle).with_suffix(".metadata.json")


def load_fault_lines_2d_metadata(
    input_pickle: str | Path,
    *,
    metadata_json: str | Path | None = None,
    ignore_artifact_metadata: bool = False,
    verify_data_hash: bool = True,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Load and validate the versioned 2D-to-3D interface sidecar.

    An explicitly supplied sidecar takes precedence over the canonical
    ``INPUT.metadata.json`` path.  If neither exists, ``(None, None)`` is
    returned for backward compatibility with legacy pickle files.  When a
    sidecar exists, its data hash is checked before any pickle deserialization.
    """

    input_pickle = Path(input_pickle)
    if ignore_artifact_metadata:
        if metadata_json is not None:
            raise ValueError(
                "metadata_json and ignore_artifact_metadata cannot be used together."
            )
        return None, None

    metadata_path = (
        Path(metadata_json)
        if metadata_json is not None
        else metadata_path_for_fault_lines(input_pickle)
    )
    if not metadata_path.is_file():
        if metadata_json is not None:
            raise FileNotFoundError(
                f"2D interface metadata not found: {metadata_path.resolve()}"
            )
        return None, None

    with metadata_path.open("r", encoding="utf-8") as input_file:
        metadata = json.load(input_file)
    if not isinstance(metadata, dict):
        raise TypeError("The 2D interface metadata root must be a JSON object.")
    if metadata.get("schema_version") != FAULT_LINES_2D_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported 2D interface schema_version: "
            f"{metadata.get('schema_version')!r}; expected "
            f"{FAULT_LINES_2D_SCHEMA_VERSION!r}."
        )
    if metadata.get("artifact_type") not in LEGACY_ARTIFACT_TYPES:
        raise ValueError(
            f"Unsupported 2D artifact_type: {metadata.get('artifact_type')!r}."
        )

    line_mapping = metadata.get("line_mapping")
    if not isinstance(line_mapping, Mapping):
        raise TypeError("2D metadata.line_mapping must be an object.")
    try:
        inline_origin = int(line_mapping["inline_origin"])
        inline_step = int(line_mapping["inline_step"])
        inline_start = int(line_mapping["inline_start"])
        inline_end = int(line_mapping["inline_end"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "2D metadata.line_mapping must define integer inline_origin, "
            "inline_step, inline_start, and inline_end."
        ) from error
    if inline_step <= 0 or inline_end < inline_start:
        raise ValueError("2D metadata contains an invalid inline mapping.")
    selected_inline_numbers = line_mapping.get("selected_inline_numbers")
    if selected_inline_numbers is not None:
        if not isinstance(selected_inline_numbers, list):
            raise TypeError("line_mapping.selected_inline_numbers must be an array.")
        try:
            selected_inline_numbers = [int(value) for value in selected_inline_numbers]
        except (TypeError, ValueError) as error:
            raise ValueError("selected_inline_numbers must contain integers.") from error
        if selected_inline_numbers != sorted(set(selected_inline_numbers)):
            raise ValueError("selected_inline_numbers must be strictly increasing and unique.")
        if any(
            value < inline_start
            or value > inline_end
            or (value - inline_origin) % inline_step
            for value in selected_inline_numbers
        ):
            raise ValueError("selected_inline_numbers conflicts with line_mapping.")
        if int(line_mapping.get("selected_inline_count", -1)) != len(selected_inline_numbers):
            raise ValueError("selected_inline_count does not match selected_inline_numbers.")

    coordinates = metadata.get("coordinates")
    if not isinstance(coordinates, Mapping):
        raise TypeError("2D metadata.coordinates must be an object.")
    if tuple(coordinates.get("point_columns", ())) != FAULT_LINES_2D_POINT_COLUMNS:
        raise ValueError(
            "2D metadata point columns must be "
            f"{list(FAULT_LINES_2D_POINT_COLUMNS)!r}."
        )
    if coordinates.get("coordinate_mode") != "downsampled_array_index":
        raise ValueError(
            "Only the downsampled_array_index coordinate mode is supported."
        )

    output = metadata.get("output")
    if not isinstance(output, Mapping):
        raise TypeError("2D metadata.output must be an object.")
    expected_hash = output.get("data_sha256") or output.get("pickle_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("2D metadata.output does not contain a valid data SHA-256.")
    if verify_data_hash:
        if not input_pickle.is_file():
            raise FileNotFoundError(f"Input pickle not found: {input_pickle.resolve()}")
        actual_hash = sha256_file(input_pickle)
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(
                "2D result hash does not match its metadata sidecar: "
                f"expected {expected_hash}, got {actual_hash}."
            )

    return metadata, metadata_path


def _artifact_line_mapping(metadata: Mapping[str, Any] | None) -> dict[str, int] | None:
    if metadata is None:
        return None
    line_mapping = metadata["line_mapping"]
    return {
        "inline_origin": int(line_mapping["inline_origin"]),
        "inline_step": int(line_mapping["inline_step"]),
        "inline_start": int(line_mapping["inline_start"]),
        "inline_end": int(line_mapping["inline_end"]),
    }


def resolve_fault_lines_2d_config(
    metadata: Mapping[str, Any] | None,
    config: ReconstructionConfig | None = None,
    *,
    inline_origin: int | None = None,
    inline_step: int | None = None,
    inline_start: int | None = None,
    inline_end: int | None = None,
) -> ReconstructionConfig:
    """Resolve one authoritative line mapping for loading and reconstruction.

    Metadata supplies defaults for a unified artifact.  Explicit origin/step
    values are assertions and must agree with it; start/end may select a subset
    of the exported range.  Without metadata, the historical 300/1/300/330
    defaults remain available.
    """

    if config is not None and any(
        value is not None
        for value in (inline_origin, inline_step, inline_start, inline_end)
    ):
        raise ValueError("Pass either config or inline mapping overrides, not both.")

    artifact_mapping = _artifact_line_mapping(metadata)
    if config is not None:
        effective = config
        if artifact_mapping is not None:
            if effective.inline_origin != artifact_mapping["inline_origin"]:
                raise ValueError(
                    "ReconstructionConfig.inline_origin conflicts with 2D metadata."
                )
            if effective.inline_step != artifact_mapping["inline_step"]:
                raise ValueError(
                    "ReconstructionConfig.inline_step conflicts with 2D metadata."
                )
            if (
                effective.inline_start < artifact_mapping["inline_start"]
                or effective.inline_end > artifact_mapping["inline_end"]
            ):
                raise ValueError(
                    "ReconstructionConfig inline range extends beyond the 2D artifact."
                )
        effective.validate()
        return effective

    defaults = ReconstructionConfig()
    if artifact_mapping is None:
        mapping = {
            "inline_origin": defaults.inline_origin,
            "inline_step": defaults.inline_step,
            "inline_start": defaults.inline_start,
            "inline_end": defaults.inline_end,
        }
    else:
        mapping = dict(artifact_mapping)

    if inline_origin is not None:
        if artifact_mapping is not None and int(inline_origin) != artifact_mapping["inline_origin"]:
            raise ValueError("--inline-origin conflicts with the 2D artifact metadata.")
        mapping["inline_origin"] = int(inline_origin)
    if inline_step is not None:
        if artifact_mapping is not None and int(inline_step) != artifact_mapping["inline_step"]:
            raise ValueError("--inline-step conflicts with the 2D artifact metadata.")
        mapping["inline_step"] = int(inline_step)
    if inline_start is not None:
        mapping["inline_start"] = int(inline_start)
    if inline_end is not None:
        mapping["inline_end"] = int(inline_end)

    if artifact_mapping is not None and (
        mapping["inline_start"] < artifact_mapping["inline_start"]
        or mapping["inline_end"] > artifact_mapping["inline_end"]
    ):
        raise ValueError("Requested inline range extends beyond the 2D artifact metadata.")
    effective = replace(defaults, **mapping)
    effective.validate()
    return effective


def line_key_to_inline(key: str, origin: int = 300, step: int = 1) -> int:
    """Map ``line_<index>`` to its explicit inline number."""

    match = LINE_KEY_PATTERN.fullmatch(key)
    if match is None:
        raise ValueError(f"Invalid section key {key!r}; expected 'line_<integer>'.")
    return origin + int(match.group(1)) * step


def neighboring_sections(
    section: int,
    available_sections: Iterable[int],
    max_offset: int = 5,
) -> list[int]:
    """Return sorted section numbers within +/- ``max_offset``, excluding self."""

    if max_offset < 1:
        raise ValueError("max_offset must be at least 1.")
    return sorted(
        candidate
        for candidate in set(int(value) for value in available_sections)
        if candidate != section and abs(candidate - section) <= max_offset
    )


def _as_point_array(points: np.ndarray, dimensions: int = 3) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != dimensions:
        raise ValueError(f"Expected an (N, {dimensions}) point array, received {points.shape}.")
    if len(points) < 1:
        raise ValueError("A point set cannot be empty.")
    if not np.isfinite(points).all():
        raise ValueError("Point coordinates must all be finite.")
    return points


def pca_tangent(points: np.ndarray) -> np.ndarray:
    """Return the dominant unit PCA direction of a 2D or 3D point set."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) < 2:
        raise ValueError("PCA requires an (N, D) array with at least two points.")
    if not np.isfinite(points).all():
        raise ValueError("PCA points must all be finite.")
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) == 0 or singular_values[0] <= np.finfo(float).eps:
        raise ValueError("Cannot estimate a direction from coincident points.")
    tangent = right_vectors[0]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= np.finfo(float).eps:
        raise ValueError("The PCA tangent has zero length.")
    return tangent / tangent_norm


def pca_orientation_difference(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Implement Equation (28): ``1 - abs(t_a dot t_b)``."""

    tangent_a = pca_tangent(points_a)
    tangent_b = pca_tangent(points_b)
    dot_product = float(np.clip(np.dot(tangent_a, tangent_b), -1.0, 1.0))
    return float(np.clip(1.0 - abs(dot_product), 0.0, 1.0))


def _directional_mean_min_distance(
    points: np.ndarray,
    target_tree: cKDTree,
) -> float:
    distances, _ = target_tree.query(points, k=1)
    return float(np.mean(distances))


def symmetric_chamfer_components(
    points_a: np.ndarray,
    points_b: np.ndarray,
    tree_a: cKDTree | None = None,
    tree_b: cKDTree | None = None,
) -> tuple[float, float, float]:
    """Return both directional means and their sum from Equation (27)."""

    points_a = _as_point_array(points_a, dimensions=3)
    points_b = _as_point_array(points_b, dimensions=3)
    tree_a = cKDTree(points_a) if tree_a is None else tree_a
    tree_b = cKDTree(points_b) if tree_b is None else tree_b
    distance_ab = _directional_mean_min_distance(points_a, tree_b)
    distance_ba = _directional_mean_min_distance(points_b, tree_a)
    return distance_ab, distance_ba, distance_ab + distance_ba


def symmetric_chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Implement Equation (27), without an additional factor of one half."""

    return symmetric_chamfer_components(points_a, points_b)[2]


def fault_line_similarity(
    points_a: np.ndarray,
    points_b: np.ndarray,
    distance_weight: float = 0.03,
    orientation_weight: float = 1.0,
) -> tuple[float, float, float]:
    """Return ``(similarity, Chamfer distance, orientation difference)``.

    This function implements Equations (26)--(28).
    """

    distance = symmetric_chamfer_distance(points_a, points_b)
    orientation = pca_orientation_difference(points_a, points_b)
    similarity = math.exp(
        -(distance_weight * distance + orientation_weight * orientation)
    )
    return similarity, distance, orientation


def _bbox_distance(line_a: FaultLine, line_b: FaultLine) -> float:
    separation = np.maximum(
        0.0,
        np.maximum(
            line_a.bbox_min - line_b.bbox_max,
            line_b.bbox_min - line_a.bbox_max,
        ),
    )
    return float(np.linalg.norm(separation))


def _orientation_from_tangents(line_a: FaultLine, line_b: FaultLine) -> float:
    dot_product = float(np.clip(np.dot(line_a.tangent, line_b.tangent), -1.0, 1.0))
    return float(np.clip(1.0 - abs(dot_product), 0.0, 1.0))


def _cluster_sort_key(cluster_id: Any) -> tuple[int, Any]:
    if isinstance(cluster_id, (int, np.integer)):
        return (0, int(cluster_id))
    try:
        return (0, int(str(cluster_id)))
    except ValueError:
        return (1, str(cluster_id))


def _make_fault_line(
    source_key: str,
    section_index: int,
    inline_number: int,
    cluster_id: Any,
    points_2d: np.ndarray,
    config: ReconstructionConfig,
) -> FaultLine:
    points_2d = np.asarray(points_2d, dtype=np.float64)
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError(
            f"{source_key} cluster {cluster_id!r} must be (N, 2), "
            f"received {points_2d.shape}."
        )
    if len(points_2d) < 2:
        raise ValueError(f"{source_key} cluster {cluster_id!r} has fewer than two points.")
    if not np.isfinite(points_2d).all():
        raise ValueError(f"{source_key} cluster {cluster_id!r} contains non-finite values.")
    if np.unique(points_2d, axis=0).shape[0] < 2:
        raise ValueError(f"{source_key} cluster {cluster_id!r} has no geometric extent.")

    # Work in relative inline coordinates. Translation does not change a
    # distance, while inline_step must change the cross-section separation.
    section_coordinate = float(inline_number - config.inline_origin)
    points_3d = np.column_stack(
        (
            np.full(len(points_2d), section_coordinate),
            points_2d[:, 0],
            points_2d[:, 1],
        )
    )
    points_3d *= np.asarray(config.axis_scales, dtype=np.float64)
    tangent = pca_tangent(points_3d)
    cluster_text = str(cluster_id)
    line_id = f"{source_key}_c{cluster_text}"
    return FaultLine(
        line_id=line_id,
        source_key=source_key,
        section_index=section_index,
        inline_number=inline_number,
        cluster_id=cluster_text,
        points_2d=points_2d,
        points_3d=points_3d,
        tangent=tangent,
        bbox_min=np.min(points_3d, axis=0),
        bbox_max=np.max(points_3d, axis=0),
        tree=cKDTree(points_3d),
    )


def _validate_results_against_interface_metadata(
    raw_results: Mapping[str, Mapping[Any, Any]],
    metadata: Mapping[str, Any],
) -> None:
    """Check line keys, explicit inline mapping, and counts before 3D use."""

    section_records = metadata.get("sections")
    if not isinstance(section_records, Mapping):
        raise TypeError("2D metadata.sections must be an object.")
    if set(section_records) != set(raw_results):
        raise ValueError("2D metadata section keys do not match the pickle root keys.")

    mapping = _artifact_line_mapping(metadata)
    assert mapping is not None
    total_fault_lines = 0
    total_points = 0
    for source_key, clusters in raw_results.items():
        section_index = int(LINE_KEY_PATTERN.fullmatch(source_key).group(1))
        expected_inline = mapping["inline_origin"] + section_index * mapping["inline_step"]
        record = section_records[source_key]
        if not isinstance(record, Mapping):
            raise TypeError(f"2D metadata.sections.{source_key} must be an object.")
        if int(record.get("inline_number", -1)) != expected_inline:
            raise ValueError(
                f"2D metadata inline_number for {source_key} conflicts with line_mapping."
            )
        if not mapping["inline_start"] <= expected_inline <= mapping["inline_end"]:
            raise ValueError(f"{source_key} maps outside the declared artifact range.")

        section_points = 0
        for points in clusters.values():
            point_array = np.asarray(points)
            if point_array.ndim != 2 or point_array.shape[1] != 2:
                raise ValueError(f"A cluster in {source_key} is not an (N, 2) array.")
            section_points += len(point_array)
        if int(record.get("fault_line_count", -1)) != len(clusters):
            raise ValueError(f"Incorrect metadata fault_line_count for {source_key}.")
        if int(record.get("point_count", -1)) != section_points:
            raise ValueError(f"Incorrect metadata point_count for {source_key}.")
        total_fault_lines += len(clusters)
        total_points += section_points

    declared_inlines = metadata["line_mapping"].get("selected_inline_numbers")
    if declared_inlines is not None:
        actual_inlines = sorted(
            mapping["inline_origin"]
            + int(LINE_KEY_PATTERN.fullmatch(source_key).group(1)) * mapping["inline_step"]
            for source_key in raw_results
        )
        if [int(value) for value in declared_inlines] != actual_inlines:
            raise ValueError(
                "line_mapping.selected_inline_numbers does not match the pickle line keys."
            )

    counts = metadata.get("counts")
    if not isinstance(counts, Mapping):
        raise TypeError("2D metadata.counts must be an object.")
    expected_counts = {
        "sections": len(raw_results),
        "nonempty_sections": sum(bool(clusters) for clusters in raw_results.values()),
        "fault_lines": total_fault_lines,
        "points": total_points,
    }
    for name, expected_value in expected_counts.items():
        if int(counts.get(name, -1)) != expected_value:
            raise ValueError(f"2D metadata.counts.{name} is inconsistent with the pickle.")


def load_fault_lines_2d(
    input_pickle: str | Path,
    config: ReconstructionConfig | None = None,
    *,
    metadata_json: str | Path | None = None,
    allow_unsafe_pickle: bool = False,
    ignore_artifact_metadata: bool = False,
    verify_data_hash: bool = True,
) -> LoadedFaultLines2D:
    """Load the unified 2D artifact and resolve its authoritative mapping.

    New artifacts use a ``line_n`` pickle plus a versioned JSON sidecar.  The
    sidecar is auto-discovered and supplies the inline mapping.  A legacy
    pickle without a sidecar remains supported through the supplied config or
    the historical defaults.
    """

    input_pickle = Path(input_pickle)
    if not input_pickle.is_file():
        raise FileNotFoundError(f"Input pickle not found: {input_pickle.resolve()}")
    interface_metadata, interface_metadata_path = load_fault_lines_2d_metadata(
        input_pickle,
        metadata_json=metadata_json,
        ignore_artifact_metadata=ignore_artifact_metadata,
        verify_data_hash=verify_data_hash,
    )
    effective_config = resolve_fault_lines_2d_config(interface_metadata, config)

    if not allow_unsafe_pickle:
        raise PermissionError(
            "Loading pickle can execute arbitrary code. Pass allow_unsafe_pickle=True "
            "only for a trusted input."
        )
    with input_pickle.open("rb") as input_file:
        raw_results = pickle.load(input_file)
    if not isinstance(raw_results, Mapping):
        raise TypeError("The pickle root must be a mapping of section keys to clusters.")

    parsed_sections: list[tuple[int, str, Mapping[Any, Any]]] = []
    for source_key, clusters in raw_results.items():
        if not isinstance(source_key, str):
            raise TypeError(f"Section keys must be strings, received {type(source_key)!r}.")
        match = LINE_KEY_PATTERN.fullmatch(source_key)
        if match is None:
            raise ValueError(f"Invalid section key {source_key!r}; expected 'line_<integer>'.")
        if not isinstance(clusters, Mapping):
            raise TypeError(f"Clusters for {source_key} must be a mapping.")
        parsed_sections.append((int(match.group(1)), source_key, clusters))
    parsed_sections.sort(key=lambda item: item[0])

    section_indices = [item[0] for item in parsed_sections]
    if len(section_indices) != len(set(section_indices)):
        raise ValueError("Duplicate numerical section indices were found.")
    if interface_metadata is not None:
        _validate_results_against_interface_metadata(raw_results, interface_metadata)
    else:
        LOGGER.warning(
            "No 2D interface metadata sidecar was found; using the supplied or "
            "historical inline mapping without artifact-level verification."
        )

    sections: dict[int, list[FaultLine]] = {}
    total_points = 0
    for section_index, source_key, clusters in parsed_sections:
        inline_number = (
            effective_config.inline_origin + section_index * effective_config.inline_step
        )
        if inline_number < effective_config.inline_start or inline_number > effective_config.inline_end:
            continue
        section_lines: list[FaultLine] = []
        for cluster_id in sorted(clusters, key=_cluster_sort_key):
            fault_line = _make_fault_line(
                source_key=source_key,
                section_index=section_index,
                inline_number=inline_number,
                cluster_id=cluster_id,
                points_2d=clusters[cluster_id],
                config=effective_config,
            )
            section_lines.append(fault_line)
            total_points += len(fault_line.points_2d)
        if section_lines:
            sections[section_index] = section_lines

    if not sections:
        raise ValueError("No non-empty sections fall within the requested inline range.")
    selected_inlines = sorted(
        line.inline_number for lines in sections.values() for line in lines
    )
    input_metadata: dict[str, Any] = {
        "input_file": input_pickle.name,
        "input_sha256": sha256_file(input_pickle),
        "section_count": len(sections),
        "fault_line_count": sum(len(lines) for lines in sections.values()),
        "point_count": total_points,
        "inline_min": min(selected_inlines),
        "inline_max": max(selected_inlines),
        "interface_mode": (
            "fault-lines-2d/1.0" if interface_metadata is not None else "legacy-pickle"
        ),
        "effective_line_mapping": {
            "inline_origin": effective_config.inline_origin,
            "inline_step": effective_config.inline_step,
            "inline_start": effective_config.inline_start,
            "inline_end": effective_config.inline_end,
        },
    }
    if interface_metadata is not None and interface_metadata_path is not None:
        input_metadata.update(
            {
                "interface_metadata_file": interface_metadata_path.name,
                "interface_metadata_sha256": sha256_file(interface_metadata_path),
                "artifact_schema_version": interface_metadata["schema_version"],
                "artifact_line_mapping": dict(interface_metadata["line_mapping"]),
                "artifact_coordinates": dict(interface_metadata["coordinates"]),
            }
        )
    return LoadedFaultLines2D(
        sections=sections,
        input_metadata=input_metadata,
        config=effective_config,
    )


def load_fault_lines_pickle(
    pickle_path: str | Path,
    config: ReconstructionConfig,
    *,
    allow_unsafe_pickle: bool = False,
    metadata_json: str | Path | None = None,
    ignore_artifact_metadata: bool = False,
) -> tuple[dict[int, list[FaultLine]], dict[str, Any]]:
    """Backward-compatible wrapper around :func:`load_fault_lines_2d`."""

    loaded = load_fault_lines_2d(
        pickle_path,
        config,
        metadata_json=metadata_json,
        allow_unsafe_pickle=allow_unsafe_pickle,
        ignore_artifact_metadata=ignore_artifact_metadata,
    )
    return loaded.sections, loaded.input_metadata


def flatten_fault_lines(sections: Mapping[int, Sequence[FaultLine]]) -> dict[str, FaultLine]:
    """Create a stable line-ID lookup."""

    lookup: dict[str, FaultLine] = {}
    for section_index in sorted(sections):
        for line in sections[section_index]:
            if line.line_id in lookup:
                raise ValueError(f"Duplicate fault-line ID: {line.line_id}")
            lookup[line.line_id] = line
    return lookup


def find_candidate_matches(
    sections: Mapping[int, Sequence[FaultLine]],
    config: ReconstructionConfig,
) -> tuple[list[CandidateMatch], dict[str, int]]:
    """Score candidate pairs within M sections, with safe threshold pruning."""

    section_indices = sorted(sections)
    threshold_energy = -math.log(config.similarity_threshold)
    matches: list[CandidateMatch] = []
    statistics = {
        "candidate_pairs_considered": 0,
        "pairs_pruned_by_orientation_or_bbox": 0,
        "pairs_scored_exactly": 0,
        "pairs_above_threshold": 0,
    }

    for source_position, source_section in enumerate(section_indices):
        for target_section in section_indices[source_position + 1 :]:
            section_gap = target_section - source_section
            if section_gap > config.max_neighbor_sections:
                break
            if section_gap < 1:
                continue

            for source_line in sections[source_section]:
                for target_line in sections[target_section]:
                    statistics["candidate_pairs_considered"] += 1
                    orientation = _orientation_from_tangents(source_line, target_line)
                    bbox_lower_bound = _bbox_distance(source_line, target_line)
                    # Each directional Chamfer mean is at least the AABB distance.
                    energy_lower_bound = (
                        config.distance_weight * 2.0 * bbox_lower_bound
                        + config.orientation_weight * orientation
                    )
                    if energy_lower_bound >= threshold_energy:
                        statistics["pairs_pruned_by_orientation_or_bbox"] += 1
                        continue

                    chamfer_ab, chamfer_ba, chamfer = symmetric_chamfer_components(
                        source_line.points_3d,
                        target_line.points_3d,
                        tree_a=source_line.tree,
                        tree_b=target_line.tree,
                    )
                    similarity = math.exp(
                        -(
                            config.distance_weight * chamfer
                            + config.orientation_weight * orientation
                        )
                    )
                    above_threshold = similarity > config.similarity_threshold
                    statistics["pairs_scored_exactly"] += 1
                    statistics["pairs_above_threshold"] += int(above_threshold)
                    matches.append(
                        CandidateMatch(
                            source_id=source_line.line_id,
                            target_id=target_line.line_id,
                            source_section_index=source_section,
                            target_section_index=target_section,
                            source_inline=source_line.inline_number,
                            target_inline=target_line.inline_number,
                            section_gap=section_gap,
                            chamfer_ab=chamfer_ab,
                            chamfer_ba=chamfer_ba,
                            chamfer_distance=chamfer,
                            orientation_difference=orientation,
                            similarity=similarity,
                            above_threshold=above_threshold,
                        )
                    )

        LOGGER.info(
            "Scored candidate neighborhoods for inline %s (%d/%d sections).",
            sections[source_section][0].inline_number,
            source_position + 1,
            len(section_indices),
        )

    return matches, statistics


def select_best_match(
    candidates: Sequence[tuple[str, float]],
    threshold: float = 0.7,
) -> tuple[str, float] | None:
    """Select a deterministic maximum-similarity candidate above a strict threshold."""

    eligible = [candidate for candidate in candidates if candidate[1] > threshold]
    if not eligible:
        return None
    return sorted(eligible, key=lambda item: (-item[1], item[0]))[0]


def select_best_matches(
    matches: Sequence[CandidateMatch],
    config: ReconstructionConfig,
) -> list[int]:
    """Apply Equation (31) and deterministic target-collision resolution.

    Each source line retains one maximum-similarity candidate over its complete
    forward M-section neighborhood. Candidate pairs are scored once and oriented
    from the earlier to the later section because Equations (26)--(28) are
    symmetric and trajectories are built progressively in section order. If
    several sources select the same target, only the highest-similarity link is
    retained. That target-collision rule is an implementation choice because the
    manuscript does not specify it.
    """

    for match in matches:
        match.source_best = False
        match.selected = False
        match.track_edge = False

    source_groups: dict[str, list[int]] = {}
    for index, match in enumerate(matches):
        if match.similarity > config.similarity_threshold:
            source_groups.setdefault(match.source_id, []).append(index)

    source_winners: list[int] = []
    for indices in source_groups.values():
        winner = sorted(
            indices,
            key=lambda idx: (
                -matches[idx].similarity,
                matches[idx].section_gap,
                matches[idx].target_id,
                matches[idx].source_id,
            ),
        )[0]
        matches[winner].source_best = True
        source_winners.append(winner)

    target_groups: dict[str, list[int]] = {}
    for index in source_winners:
        match = matches[index]
        target_groups.setdefault(match.target_id, []).append(index)

    selected_indices: list[int] = []
    for indices in target_groups.values():
        winner = sorted(
            indices,
            key=lambda idx: (
                -matches[idx].similarity,
                matches[idx].section_gap,
                matches[idx].source_id,
                matches[idx].target_id,
            ),
        )[0]
        matches[winner].selected = True
        selected_indices.append(winner)

    return sorted(
        selected_indices,
        key=lambda idx: (
            matches[idx].source_section_index,
            matches[idx].target_section_index,
            matches[idx].source_id,
            matches[idx].target_id,
        ),
    )


def validate_consecutive_tracks(section_numbers: Sequence[int], k: int = 3) -> bool:
    """Return True when at least ``k`` strictly consecutive sections are present."""

    if k < 2:
        raise ValueError("k must be at least 2.")
    ordered = sorted(set(int(value) for value in section_numbers))
    longest_run = 0
    current_run = 0
    previous: int | None = None
    for section in ordered:
        current_run = current_run + 1 if previous is not None and section == previous + 1 else 1
        longest_run = max(longest_run, current_run)
        previous = section
    return longest_run >= k


def _split_consecutive_path(
    line_ids: Sequence[str],
    edge_indices: Sequence[int],
    section_gaps: Sequence[int],
    minimum_lines: int,
) -> list[tuple[list[str], list[int]]]:
    """Split a directed path at non-adjacent links and discard short runs."""

    if len(line_ids) != len(edge_indices) + 1:
        raise ValueError("A path must contain exactly one more line than edges.")
    if len(edge_indices) != len(section_gaps):
        raise ValueError("Each path edge must have one section gap.")
    if minimum_lines < 2:
        raise ValueError("minimum_lines must be at least 2.")

    runs: list[tuple[list[str], list[int]]] = []
    run_line_ids = [line_ids[0]]
    run_edge_indices: list[int] = []
    for edge_index, section_gap, target_id in zip(
        edge_indices,
        section_gaps,
        line_ids[1:],
    ):
        if section_gap == 1:
            run_edge_indices.append(edge_index)
            run_line_ids.append(target_id)
            continue
        if len(run_line_ids) >= minimum_lines:
            runs.append((run_line_ids, run_edge_indices))
        run_line_ids = [target_id]
        run_edge_indices = []
    if len(run_line_ids) >= minimum_lines:
        runs.append((run_line_ids, run_edge_indices))
    return runs


def build_fault_tracks(
    lines_by_id: Mapping[str, FaultLine],
    matches: Sequence[CandidateMatch],
    selected_indices: Sequence[int],
    config: ReconstructionConfig,
) -> list[FaultTrack]:
    """Build one-to-one paths and retain only K-consecutive path segments.

    Matches spanning two to M sections participate in neighborhood-wide best
    matching. Equation (33), however, validates adjacent-section links. A path is
    therefore split at every non-adjacent link, and only resulting runs with at
    least K lines are retained. This prevents a valid K-window from carrying an
    isolated non-consecutive tail into a reconstructed surface.
    """

    outgoing: dict[str, int] = {}
    incoming: dict[str, int] = {}
    for index in selected_indices:
        match = matches[index]
        if match.source_id in outgoing:
            raise RuntimeError(f"Multiple outgoing links for {match.source_id}.")
        if match.target_id in incoming:
            raise RuntimeError(f"Multiple incoming links for {match.target_id}.")
        outgoing[match.source_id] = index
        incoming[match.target_id] = index

    graph_nodes = set(outgoing) | set(incoming)
    starts = sorted(
        (node for node in graph_nodes if node not in incoming),
        key=lambda line_id: (
            lines_by_id[line_id].section_index,
            line_id,
        ),
    )
    raw_paths: list[tuple[list[str], list[int]]] = []
    visited: set[str] = set()
    for start in starts:
        line_ids = [start]
        edge_indices: list[int] = []
        current = start
        while current in outgoing:
            edge_index = outgoing[current]
            target = matches[edge_index].target_id
            if target in line_ids:
                raise RuntimeError("A cycle was found in forward-only section links.")
            edge_indices.append(edge_index)
            line_ids.append(target)
            current = target
        visited.update(line_ids)
        raw_paths.append((line_ids, edge_indices))

    if visited != graph_nodes:
        missing = sorted(graph_nodes - visited)
        raise RuntimeError(f"Unvisited linked nodes remain: {missing[:5]}")

    raw_tracks: list[tuple[list[str], list[int]]] = []
    for path_line_ids, path_edge_indices in raw_paths:
        raw_tracks.extend(
            _split_consecutive_path(
                path_line_ids,
                path_edge_indices,
                [matches[index].section_gap for index in path_edge_indices],
                config.min_consecutive_sections,
            )
        )

    def track_sort_key(item: tuple[list[str], list[int]]) -> tuple[Any, ...]:
        line_ids, _ = item
        track_lines = [lines_by_id[line_id] for line_id in line_ids]
        return (
            track_lines[0].inline_number,
            float(np.mean([np.mean(line.points_2d[:, 0]) for line in track_lines])),
            float(np.mean([np.mean(line.points_2d[:, 1]) for line in track_lines])),
            tuple(line_ids),
        )

    raw_tracks.sort(key=track_sort_key)
    tracks: list[FaultTrack] = []
    assigned_lines: set[str] = set()
    for track_number, (line_ids, edge_indices) in enumerate(raw_tracks, start=1):
        track_lines = [lines_by_id[line_id] for line_id in line_ids]
        section_indices = [line.section_index for line in track_lines]
        inline_numbers = [line.inline_number for line in track_lines]
        if not validate_consecutive_tracks(
            section_indices,
            k=config.min_consecutive_sections,
        ):
            continue
        if len(section_indices) != len(set(section_indices)):
            raise RuntimeError("A track contains more than one line from the same section.")
        overlap = assigned_lines.intersection(line_ids)
        if overlap:
            raise RuntimeError(f"Fault lines assigned to multiple tracks: {sorted(overlap)}")
        assigned_lines.update(line_ids)
        similarities = [matches[index].similarity for index in edge_indices]
        track_id = f"fault_{track_number:04d}"
        tracks.append(
            FaultTrack(
                track_id=track_id,
                line_ids=line_ids,
                edge_indices=edge_indices,
                inline_numbers=inline_numbers,
                section_indices=section_indices,
                min_similarity=float(min(similarities)),
                mean_similarity=float(np.mean(similarities)),
            )
        )
        for index in edge_indices:
            matches[index].track_edge = True

    return tracks


def _order_curve_points(points_2d: np.ndarray) -> np.ndarray:
    points_2d = np.unique(np.asarray(points_2d, dtype=np.float64), axis=0)
    if len(points_2d) < 2:
        raise ValueError("A curve requires at least two distinct points.")
    tangent = pca_tangent(points_2d)
    projections = (points_2d - np.mean(points_2d, axis=0)) @ tangent
    order = np.argsort(projections, kind="stable")
    return points_2d[order]


def _resample_curve(points_2d: np.ndarray, samples: int) -> np.ndarray:
    ordered = _order_curve_points(points_2d)
    segment_lengths = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    keep = np.concatenate(([True], np.diff(cumulative) > np.finfo(float).eps))
    ordered = ordered[keep]
    cumulative = cumulative[keep]
    if len(ordered) < 2 or cumulative[-1] <= np.finfo(float).eps:
        raise ValueError("A curve has zero arc length.")
    cumulative /= cumulative[-1]
    target = np.linspace(0.0, 1.0, samples)
    return np.column_stack(
        (
            np.interp(target, cumulative, ordered[:, 0]),
            np.interp(target, cumulative, ordered[:, 1]),
        )
    )


def _surface_faces(rows: int, columns: int) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            upper_left = row * columns + column
            upper_right = upper_left + 1
            lower_left = (row + 1) * columns + column
            lower_right = lower_left + 1
            faces.append((upper_left, lower_left, upper_right))
            faces.append((upper_right, lower_left, lower_right))
    return np.asarray(faces, dtype=np.int64)


def reconstruct_surface(
    track: FaultTrack,
    lines_by_id: Mapping[str, FaultLine],
    config: ReconstructionConfig,
) -> FaultSurface:
    """Generate one explicit parameter surface from a validated track."""

    track_lines = [lines_by_id[line_id] for line_id in track.line_ids]
    observed_inline = np.asarray([line.inline_number for line in track_lines], dtype=float)
    resampled = [
        _resample_curve(line.points_2d, config.surface_samples)
        for line in track_lines
    ]

    for index in range(1, len(resampled)):
        previous = resampled[index - 1]
        current = resampled[index]
        same_direction = np.linalg.norm(previous[0] - current[0]) + np.linalg.norm(
            previous[-1] - current[-1]
        )
        reverse_direction = np.linalg.norm(previous[0] - current[-1]) + np.linalg.norm(
            previous[-1] - current[0]
        )
        if reverse_direction < same_direction:
            resampled[index] = current[::-1]

    observed_curves = np.stack(resampled, axis=0)
    inline_axis = np.arange(
        int(observed_inline[0]),
        int(observed_inline[-1]) + config.inline_step,
        config.inline_step,
        dtype=float,
    )
    profile_grid = np.empty((len(inline_axis), config.surface_samples), dtype=float)
    depth_grid = np.empty_like(profile_grid)
    for sample_index in range(config.surface_samples):
        profile_grid[:, sample_index] = np.interp(
            inline_axis,
            observed_inline,
            observed_curves[:, sample_index, 0],
        )
        depth_grid[:, sample_index] = np.interp(
            inline_axis,
            observed_inline,
            observed_curves[:, sample_index, 1],
        )

    sigma = (config.smooth_sigma_inline, config.smooth_sigma_along)
    if sigma != (0.0, 0.0):
        smoothed_profile = gaussian_filter(profile_grid, sigma=sigma, mode="nearest")
        smoothed_depth = gaussian_filter(depth_grid, sigma=sigma, mode="nearest")
    else:
        smoothed_profile = profile_grid.copy()
        smoothed_depth = depth_grid.copy()

    for observed_index, inline_number in enumerate(observed_inline.astype(int)):
        row = int((inline_number - int(inline_axis[0])) // config.inline_step)
        smoothed_profile[row] = (
            config.observed_weight * observed_curves[observed_index, :, 0]
            + (1.0 - config.observed_weight) * smoothed_profile[row]
        )
        smoothed_depth[row] = (
            config.observed_weight * observed_curves[observed_index, :, 1]
            + (1.0 - config.observed_weight) * smoothed_depth[row]
        )

    inline_grid = np.repeat(inline_axis[:, None], config.surface_samples, axis=1)
    vertices = np.column_stack(
        (
            inline_grid.ravel(),
            smoothed_profile.ravel(),
            smoothed_depth.ravel(),
        )
    )
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"Non-finite surface vertices were generated for {track.track_id}.")
    faces = _surface_faces(len(inline_axis), config.surface_samples)
    observed_rows = [
        int((int(value) - int(inline_axis[0])) // config.inline_step)
        for value in observed_inline
    ]
    final_observed = np.stack(
        (
            smoothed_profile[observed_rows],
            smoothed_depth[observed_rows],
        ),
        axis=-1,
    )
    observation_rmse = float(
        np.sqrt(np.mean(np.sum((final_observed - observed_curves) ** 2, axis=2)))
    )
    return FaultSurface(
        track_id=track.track_id,
        inline_numbers=inline_axis.astype(np.int64),
        normalized_arc_length=np.linspace(0.0, 1.0, config.surface_samples),
        profile_grid=smoothed_profile,
        depth_grid=smoothed_depth,
        vertices=vertices,
        faces=faces,
        observation_rmse=observation_rmse,
    )


def reconstruct_surfaces(
    tracks: Sequence[FaultTrack],
    lines_by_id: Mapping[str, FaultLine],
    config: ReconstructionConfig,
) -> list[FaultSurface]:
    """Generate surfaces in stable track order."""

    return [reconstruct_surface(track, lines_by_id, config) for track in tracks]


def _write_candidate_csv(
    output_path: Path,
    matches: Sequence[CandidateMatch],
    indices: Sequence[int] | None = None,
) -> None:
    field_names = [
        "source_id",
        "target_id",
        "source_section_index",
        "target_section_index",
        "source_inline",
        "target_inline",
        "section_gap",
        "chamfer_ab",
        "chamfer_ba",
        "chamfer_distance",
        "orientation_difference",
        "similarity",
        "above_threshold",
        "source_best",
        "selected",
        "track_edge",
    ]
    selected_indices = range(len(matches)) if indices is None else indices
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        for index in selected_indices:
            match = matches[index]
            writer.writerow({name: getattr(match, name) for name in field_names})


def _write_track_membership_csv(
    output_path: Path,
    tracks: Sequence[FaultTrack],
    lines_by_id: Mapping[str, FaultLine],
) -> None:
    field_names = [
        "track_id",
        "member_order",
        "line_id",
        "source_key",
        "inline_number",
        "section_index",
        "cluster_id",
        "point_count",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        for track in tracks:
            for member_order, line_id in enumerate(track.line_ids):
                line = lines_by_id[line_id]
                writer.writerow(
                    {
                        "track_id": track.track_id,
                        "member_order": member_order,
                        "line_id": line_id,
                        "source_key": line.source_key,
                        "inline_number": line.inline_number,
                        "section_index": line.section_index,
                        "cluster_id": line.cluster_id,
                        "point_count": len(line.points_2d),
                    }
                )


def _save_flat_surfaces_npz(output_path: Path, surfaces: Sequence[FaultSurface]) -> None:
    if not surfaces:
        np.savez_compressed(
            output_path,
            surface_ids=np.asarray([], dtype="U1"),
            vertices=np.empty((0, 3), dtype=float),
            vertex_offsets=np.asarray([0], dtype=np.int64),
            faces=np.empty((0, 3), dtype=np.int64),
            face_offsets=np.asarray([0], dtype=np.int64),
            grid_shapes=np.empty((0, 2), dtype=np.int64),
        )
        return

    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    vertex_offsets = [0]
    face_offsets = [0]
    grid_shapes: list[tuple[int, int]] = []
    for surface in surfaces:
        vertex_offset = vertex_offsets[-1]
        vertices_parts.append(surface.vertices)
        faces_parts.append(surface.faces + vertex_offset)
        vertex_offsets.append(vertex_offset + len(surface.vertices))
        face_offsets.append(face_offsets[-1] + len(surface.faces))
        grid_shapes.append(surface.profile_grid.shape)

    np.savez_compressed(
        output_path,
        surface_ids=np.asarray([surface.track_id for surface in surfaces]),
        vertices=np.concatenate(vertices_parts, axis=0),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        faces=np.concatenate(faces_parts, axis=0),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        grid_shapes=np.asarray(grid_shapes, dtype=np.int64),
        observation_rmse=np.asarray(
            [surface.observation_rmse for surface in surfaces], dtype=float
        ),
    )


def save_results(
    output_dir: str | Path,
    *,
    config: ReconstructionConfig,
    input_metadata: Mapping[str, Any],
    score_statistics: Mapping[str, int],
    lines_by_id: Mapping[str, FaultLine],
    matches: Sequence[CandidateMatch],
    selected_indices: Sequence[int],
    tracks: Sequence[FaultTrack],
    surfaces: Sequence[FaultSurface],
) -> None:
    """Write deterministic, non-pickle reconstruction products."""

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir.resolve()}. "
            "Use a new directory to avoid mixing reconstruction runs."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_candidate_csv(output_dir / "candidate_scores.csv", matches)
    _write_candidate_csv(
        output_dir / "selected_links.csv",
        matches,
        indices=selected_indices,
    )
    _write_track_membership_csv(
        output_dir / "track_membership.csv",
        tracks,
        lines_by_id,
    )
    _save_flat_surfaces_npz(output_dir / "fault_surfaces.npz", surfaces)

    track_records = [
        {
            "track_id": track.track_id,
            "line_ids": track.line_ids,
            "inline_numbers": track.inline_numbers,
            "section_indices": track.section_indices,
            "edge_indices": track.edge_indices,
            "min_similarity": track.min_similarity,
            "mean_similarity": track.mean_similarity,
        }
        for track in tracks
    ]
    with (output_dir / "tracks.json").open("w", encoding="utf-8") as output_file:
        json.dump(track_records, output_file, indent=2, sort_keys=True)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM_NAME,
        "paper_equations": [26, 27, 28, 29, 30, 31, 32, 33, 34],
        "conformance_note": (
            "Equations 26--28 and the reported M, tau, and K values are used "
            "directly. The manuscript's +/-M search is operationalized as an "
            "acyclic +M forward progression; this directional argmax choice is "
            "recorded below and is not asserted to be algebraically equivalent."
        ),
        "input": dict(input_metadata),
        "coordinates": {
            "matching_axis_order": [
                "section_pixel",
                "profile_pixel",
                "depth_pixel",
            ],
            "surface_axis_order": [
                "inline_number",
                "profile_pixel",
                "depth_pixel",
            ],
            "axis_scales": list(config.axis_scales),
            "line_key_mapping": (
                "inline_number = inline_origin + line_index * inline_step"
            ),
        },
        "parameters": asdict(config),
        "implementation_choices_not_fixed_by_the_manuscript": {
            "directed_neighborhood_representation": (
                "score each symmetric +/-M pair once and orient it from the "
                "earlier section to the later section for progressive linking"
            ),
            "best_match_scope": (
                "one maximum-similarity target over each source line's complete "
                "forward M-section neighborhood"
            ),
            "target_collision_policy": (
                "retain the highest-similarity source-best link for each target"
            ),
            "non_adjacent_link_policy": (
                "gap 2--M winners participate in best matching, then split paths; "
                "only K-or-longer adjacent-section runs become surfaces"
            ),
            "surface_parameterization": "normalized arc length",
            "cross_inline_interpolation": "linear",
            "surface_smoothing": "2D Gaussian filter",
        },
        "counts": {
            **dict(score_statistics),
            "fault_lines_loaded": len(lines_by_id),
            "source_best_matches": sum(match.source_best for match in matches),
            "selected_matches": len(selected_indices),
            "adjacent_selected_matches": sum(
                matches[index].section_gap == 1 for index in selected_indices
            ),
            "validated_track_edges": sum(match.track_edge for match in matches),
            "validated_tracks": len(tracks),
            "surfaces": len(surfaces),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "output_schema": {
            "candidate_scores.csv": "all exactly evaluated, non-pruned pairs",
            "selected_links.csv": "neighborhood-best links after target collision",
            "track_membership.csv": "2D line membership of validated tracks",
            "tracks.json": "validated track topology and link statistics",
            "fault_surfaces.npz": {
                "surface_ids": "surface ID for each offset interval",
                "vertices": "flat [inline, profile, depth] float array",
                "vertex_offsets": "surface boundaries in vertices",
                "faces": "flat global triangle-index array",
                "face_offsets": "surface boundaries in faces",
                "grid_shapes": "[inline_count, along_line_samples] per surface",
                "observation_rmse": "smoothed-to-resampled observation RMSE",
            },
        },
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2, sort_keys=True)


def run_reconstruction(
    input_pickle: str | Path,
    output_dir: str | Path,
    config: ReconstructionConfig | None = None,
    *,
    metadata_json: str | Path | None = None,
    allow_unsafe_pickle: bool = False,
    ignore_artifact_metadata: bool = False,
    generate_surfaces: bool = True,
) -> dict[str, Any]:
    """Run 3D reconstruction from the unified 2D interface or a legacy pickle."""

    loaded = load_fault_lines_2d(
        input_pickle,
        config,
        metadata_json=metadata_json,
        allow_unsafe_pickle=allow_unsafe_pickle,
        ignore_artifact_metadata=ignore_artifact_metadata,
    )
    sections = loaded.sections
    input_metadata = loaded.input_metadata
    config = loaded.config
    lines_by_id = flatten_fault_lines(sections)
    LOGGER.info(
        "Loaded %d fault lines (%d points) from %d sections, inline %d--%d.",
        input_metadata["fault_line_count"],
        input_metadata["point_count"],
        input_metadata["section_count"],
        input_metadata["inline_min"],
        input_metadata["inline_max"],
    )
    matches, score_statistics = find_candidate_matches(sections, config)
    selected_indices = select_best_matches(matches, config)
    tracks = build_fault_tracks(
        lines_by_id,
        matches,
        selected_indices,
        config,
    )
    surfaces = (
        reconstruct_surfaces(tracks, lines_by_id, config)
        if generate_surfaces
        else []
    )
    save_results(
        output_dir,
        config=config,
        input_metadata=input_metadata,
        score_statistics=score_statistics,
        lines_by_id=lines_by_id,
        matches=matches,
        selected_indices=selected_indices,
        tracks=tracks,
        surfaces=surfaces,
    )
    summary = {
        "sections": input_metadata["section_count"],
        "fault_lines": len(lines_by_id),
        "candidate_pairs": score_statistics["candidate_pairs_considered"],
        "exact_scores": score_statistics["pairs_scored_exactly"],
        "above_threshold": score_statistics["pairs_above_threshold"],
        "selected_links": len(selected_indices),
        "tracks": len(tracks),
        "surfaces": len(surfaces),
    }
    LOGGER.info("Reconstruction summary: %s", json.dumps(summary, sort_keys=True))
    return summary


def _synthetic_fault(offset: float) -> np.ndarray:
    profile = np.linspace(100.0, 120.0, 21)
    depth = 500.0 + 0.4 * profile + offset
    return np.column_stack((profile, depth))


def run_self_tests() -> None:
    """Run fast formula and trajectory invariance checks without external data."""

    line_a = np.asarray(
        [[300.0, 0.0, 0.0], [300.0, 1.0, 0.0], [300.0, 2.0, 0.0]]
    )
    line_b = line_a.copy()
    line_b[:, 0] = 301.0
    assert np.isclose(symmetric_chamfer_distance(line_a, line_a), 0.0)
    assert np.isclose(symmetric_chamfer_distance(line_a, line_b), 2.0)
    assert np.isclose(
        symmetric_chamfer_distance(line_a, line_b),
        symmetric_chamfer_distance(line_b, line_a),
    )
    similarity, distance, orientation = fault_line_similarity(line_a, line_b)
    assert np.isclose(distance, 2.0)
    assert np.isclose(orientation, 0.0)
    assert np.isclose(similarity, math.exp(-0.06))

    horizontal = np.asarray([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vertical = np.asarray([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.isclose(pca_orientation_difference(horizontal, horizontal[::-1]), 0.0)
    assert np.isclose(pca_orientation_difference(horizontal, vertical), 1.0)

    assert line_key_to_inline("line_0", origin=300) == 300
    assert line_key_to_inline("line_30", origin=300) == 330
    assert neighboring_sections(300, range(300, 331), 5) == [301, 302, 303, 304, 305]
    assert neighboring_sections(315, range(300, 331), 5) == list(range(310, 315)) + list(
        range(316, 321)
    )
    assert validate_consecutive_tracks([300, 301, 302], 3)
    assert not validate_consecutive_tracks([300, 301], 3)
    assert not validate_consecutive_tracks([300, 302, 304], 3)
    assert select_best_match([("a", 0.700000)], 0.7) is None
    assert select_best_match([("a", 0.700001)], 0.7) == ("a", 0.700001)
    assert _split_consecutive_path(
        ["line_0", "line_1", "line_2", "line_4", "line_5", "line_6"],
        [10, 11, 12, 13, 14],
        [1, 1, 2, 1, 1],
        3,
    ) == [
        (["line_0", "line_1", "line_2"], [10, 11]),
        (["line_4", "line_5", "line_6"], [13, 14]),
    ]

    selection_config = ReconstructionConfig(inline_end=310)
    selection_matches = [
        CandidateMatch(
            source_id="line_0_c0",
            target_id="line_1_c0",
            source_section_index=0,
            target_section_index=1,
            source_inline=300,
            target_inline=301,
            section_gap=1,
            chamfer_ab=1.0,
            chamfer_ba=1.0,
            chamfer_distance=2.0,
            orientation_difference=0.0,
            similarity=0.80,
            above_threshold=True,
        ),
        CandidateMatch(
            source_id="line_0_c0",
            target_id="line_2_c0",
            source_section_index=0,
            target_section_index=2,
            source_inline=300,
            target_inline=302,
            section_gap=2,
            chamfer_ab=1.0,
            chamfer_ba=1.0,
            chamfer_distance=2.0,
            orientation_difference=0.0,
            similarity=0.90,
            above_threshold=True,
        ),
        CandidateMatch(
            source_id="line_1_c1",
            target_id="line_2_c0",
            source_section_index=1,
            target_section_index=2,
            source_inline=301,
            target_inline=302,
            section_gap=1,
            chamfer_ab=1.0,
            chamfer_ba=1.0,
            chamfer_distance=2.0,
            orientation_difference=0.0,
            similarity=0.85,
            above_threshold=True,
        ),
    ]
    selection_winners = select_best_matches(selection_matches, selection_config)
    assert selection_winners == [1]
    assert not selection_matches[0].source_best
    assert selection_matches[1].source_best and selection_matches[1].selected
    assert selection_matches[2].source_best and not selection_matches[2].selected

    step_config = ReconstructionConfig(inline_step=2, inline_end=304)
    step_a = _make_fault_line("line_0", 0, 300, 0, _synthetic_fault(0.0), step_config)
    step_b = _make_fault_line("line_1", 1, 302, 0, _synthetic_fault(0.0), step_config)
    assert np.isclose(
        symmetric_chamfer_distance(step_a.points_3d, step_b.points_3d),
        4.0,
    )

    config = ReconstructionConfig(
        inline_start=300,
        inline_end=330,
        surface_samples=32,
        smooth_sigma_inline=0.5,
        smooth_sigma_along=0.5,
    )
    sections: dict[int, list[FaultLine]] = {}
    for section_index in range(31):
        source_key = f"line_{section_index}"
        sections[section_index] = [
            _make_fault_line(
                source_key,
                section_index,
                300 + section_index,
                0,
                _synthetic_fault(0.05 * section_index),
                config,
            )
        ]
    for section_index in (10, 11):
        sections[section_index].append(
            _make_fault_line(
                f"line_{section_index}",
                section_index,
                300 + section_index,
                99,
                _synthetic_fault(100.0 + 0.05 * section_index),
                config,
            )
        )
    lines_by_id = flatten_fault_lines(sections)
    matches, _ = find_candidate_matches(sections, config)
    selected = select_best_matches(matches, config)
    tracks = build_fault_tracks(lines_by_id, matches, selected, config)
    assert len(tracks) == 1
    assert len(tracks[0].line_ids) == 31
    assert all(not line_id.endswith("_c99") for line_id in tracks[0].line_ids)
    surface = reconstruct_surface(tracks[0], lines_by_id, config)
    assert np.isfinite(surface.vertices).all()
    assert surface.vertices[:, 0].min() == 300
    assert surface.vertices[:, 0].max() == 330
    LOGGER.info("All synthetic self-tests passed.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Link 2D fault lines and reconstruct 3D surfaces with a forward "
            "progressive operationalization based on manuscript Equations "
            "(26)--(34)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_pickle",
        nargs="?",
        help=(
            "2D fault-line pickle. Its canonical .metadata.json sidecar is "
            "auto-discovered when present; batch outputs are normally under "
            "outputs/fault2d/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/fault3d/fault3d_output_300_330",
        help="Directory for generated 3D reconstruction artifacts.",
    )
    parser.add_argument(
        "--metadata-json",
        help="Explicit 2D interface sidecar; defaults to INPUT.metadata.json.",
    )
    parser.add_argument(
        "--ignore-artifact-metadata",
        action="store_true",
        help="Ignore a sidecar and use explicit or historical inline mapping values.",
    )
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="Acknowledge that pickle loading can execute arbitrary code.",
    )
    parser.add_argument(
        "--inline-origin",
        type=int,
        default=None,
        help="Assert the artifact origin; legacy fallback is 300.",
    )
    parser.add_argument(
        "--inline-step",
        type=int,
        default=None,
        help="Assert the artifact step; legacy fallback is 1.",
    )
    parser.add_argument(
        "--inline-start",
        type=int,
        default=None,
        help="Optional inclusive subset start; artifact/legacy default otherwise.",
    )
    parser.add_argument(
        "--inline-end",
        type=int,
        default=None,
        help="Optional inclusive subset end; artifact/legacy default otherwise.",
    )
    parser.add_argument("--distance-weight", type=float, default=0.03)
    parser.add_argument("--orientation-weight", type=float, default=1.0)
    parser.add_argument("--max-neighbor-sections", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.7)
    parser.add_argument("--min-consecutive-sections", type=int, default=3)
    parser.add_argument(
        "--axis-scales",
        type=float,
        nargs=3,
        metavar=("SECTION", "PROFILE", "DEPTH"),
        default=(1.0, 1.0, 1.0),
    )
    parser.add_argument("--surface-samples", type=int, default=128)
    parser.add_argument("--smooth-sigma-inline", type=float, default=1.0)
    parser.add_argument("--smooth-sigma-along", type=float, default=1.0)
    parser.add_argument("--observed-weight", type=float, default=0.25)
    parser.add_argument("--no-surfaces", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(levelname)s %(message)s",
    )

    if arguments.self_test:
        run_self_tests()
        if arguments.input_pickle is None:
            return 0
    if arguments.input_pickle is None:
        parser.error("input_pickle is required unless --self-test is used alone.")

    try:
        interface_metadata, _ = load_fault_lines_2d_metadata(
            arguments.input_pickle,
            metadata_json=arguments.metadata_json,
            ignore_artifact_metadata=arguments.ignore_artifact_metadata,
            verify_data_hash=True,
        )
        mapping_config = resolve_fault_lines_2d_config(
            interface_metadata,
            inline_origin=arguments.inline_origin,
            inline_step=arguments.inline_step,
            inline_start=arguments.inline_start,
            inline_end=arguments.inline_end,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    axis_scales = tuple(float(value) for value in arguments.axis_scales)
    config = replace(
        mapping_config,
        distance_weight=arguments.distance_weight,
        orientation_weight=arguments.orientation_weight,
        max_neighbor_sections=arguments.max_neighbor_sections,
        similarity_threshold=arguments.similarity_threshold,
        min_consecutive_sections=arguments.min_consecutive_sections,
        section_axis_scale=axis_scales[0],
        profile_axis_scale=axis_scales[1],
        depth_axis_scale=axis_scales[2],
        surface_samples=arguments.surface_samples,
        smooth_sigma_inline=arguments.smooth_sigma_inline,
        smooth_sigma_along=arguments.smooth_sigma_along,
        observed_weight=arguments.observed_weight,
    )
    run_reconstruction(
        input_pickle=arguments.input_pickle,
        output_dir=arguments.output_dir,
        config=config,
        metadata_json=arguments.metadata_json,
        allow_unsafe_pickle=arguments.allow_unsafe_pickle,
        ignore_artifact_metadata=arguments.ignore_artifact_metadata,
        generate_surfaces=not arguments.no_surfaces,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
