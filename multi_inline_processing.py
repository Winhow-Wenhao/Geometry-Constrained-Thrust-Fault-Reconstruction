#!/usr/bin/env python3
"""Batch UASS-Net inference and complete 2D fault processing for SEG-Y inlines.

The output is the versioned 2D-to-3D interface consumed by
``fault_surface_reconstruction_3d.py``. It consists of a strict data pickle::

    {
        "line_0": {0: ndarray(shape=(N, 2)), ...},
        "line_1": {0: ndarray(shape=(M, 2)), ...},
        ...,
    }

and a canonical ``.metadata.json`` sidecar. The point columns are
*downsampled array indices* in the order ``(profile_index, sample_index)``.
The 3D loader auto-discovers the sidecar, verifies the data hash, and uses its
inline mapping. Metadata remains separate because adding a metadata key to the
pickle root would violate the strict ``line_<integer>`` data contract.

Example, from the project root::

    python github/Best/fault_process_multi_inline.py \
        segy/test_300_400.segy \
        --model-weights github/Best/model_real.pth \
        --inline-start 300 --inline-end 330 \
        --output-pickle github/Best/fault_lines_2d_300_330.pkl

The pickle is a trusted local interchange artifact.  Loading pickle files from
an untrusted source can execute arbitrary code.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import platform
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import segyio
import torch

from fault_postprocessing_2d import (
    InferenceConfig,
    PostprocessConfig,
    SegyInlineReader,
    load_uassnet_model,
    process_inline_section,
    sha256_file,
)


LOGGER = logging.getLogger("fault2d-batch")
SCHEMA_VERSION = "fault-lines-2d/1.0"
ARTIFACT_TYPE = "fault-lines-2d-interface"
POINT_COLUMNS = ("profile_index", "sample_index")
REFERENCE_MODEL_SHA256 = (
    "d154ab68869cfc9c789b94835cb2614c182c7dd0fa3fd56cf2af4bb9b2b638aa"
)


def line_key_for_inline(
    inline_number: int,
    *,
    inline_origin: int,
    inline_step: int,
) -> str:
    """Return the 3D-loader-compatible key for an absolute inline number."""

    if inline_step <= 0:
        raise ValueError("inline_step must be positive.")
    offset = int(inline_number) - int(inline_origin)
    if offset < 0:
        raise ValueError(
            f"Inline {inline_number} precedes inline_origin {inline_origin}; "
            "the 3D loader does not accept negative line indices."
        )
    line_index, remainder = divmod(offset, inline_step)
    if remainder:
        raise ValueError(
            f"Inline {inline_number} is not aligned with origin {inline_origin} "
            f"and step {inline_step}."
        )
    return f"line_{line_index}"


def select_inline_numbers(
    available_inlines: Sequence[int] | np.ndarray,
    *,
    inline_origin: int,
    inline_start: int,
    inline_end: int,
    inline_step: int,
    allow_missing: bool,
) -> list[int]:
    """Select and validate the absolute SEG-Y inline headers to process."""

    if inline_end < inline_start:
        raise ValueError("inline_end must be greater than or equal to inline_start.")
    if inline_step <= 0:
        raise ValueError("inline_step must be positive.")
    if inline_start < inline_origin:
        raise ValueError("inline_start cannot precede inline_origin.")
    if (inline_start - inline_origin) % inline_step:
        raise ValueError("inline_start is not aligned with inline_origin/inline_step.")
    if (inline_end - inline_origin) % inline_step:
        raise ValueError("inline_end is not aligned with inline_origin/inline_step.")

    available = [int(value) for value in np.asarray(available_inlines).ravel()]
    if len(available) != len(set(available)):
        raise ValueError("SEG-Y inline headers are not unique.")
    available_set = set(available)
    requested = list(range(inline_start, inline_end + 1, inline_step))
    missing = [inline for inline in requested if inline not in available_set]
    if missing and not allow_missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise ValueError(f"Requested SEG-Y inlines are missing: {preview}")
    selected = [inline for inline in requested if inline in available_set]
    if not selected:
        raise ValueError("No requested inlines are present in the SEG-Y file.")
    return selected


def _cluster_sort_key(cluster_id: Any) -> tuple[int, Any]:
    if isinstance(cluster_id, (int, np.integer)):
        return (0, int(cluster_id))
    try:
        return (0, int(str(cluster_id)))
    except ValueError:
        return (1, str(cluster_id))


def _normalise_clusters(
    clusters: Mapping[Any, Any],
    *,
    section_shape: tuple[int, int],
    line_key: str,
) -> dict[int, np.ndarray]:
    """Validate clusters and return deterministic integer IDs and numeric arrays."""

    if not isinstance(clusters, Mapping):
        raise TypeError(f"{line_key} post-processing output must be a mapping.")

    normalised: dict[int, np.ndarray] = {}
    for new_cluster_id, original_cluster_id in enumerate(
        sorted(clusters, key=_cluster_sort_key)
    ):
        points = np.asarray(clusters[original_cluster_id], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"{line_key} cluster {original_cluster_id!r} must have shape (N, 2); "
                f"received {points.shape}."
            )
        if len(points) < 2:
            raise ValueError(
                f"{line_key} cluster {original_cluster_id!r} has fewer than two points."
            )
        if not np.isfinite(points).all():
            raise ValueError(
                f"{line_key} cluster {original_cluster_id!r} contains non-finite values."
            )
        if np.unique(points, axis=0).shape[0] < 2:
            raise ValueError(
                f"{line_key} cluster {original_cluster_id!r} has no geometric extent."
            )

        profile_limit, sample_limit = section_shape
        if (
            np.min(points[:, 0]) < 0.0
            or np.max(points[:, 0]) > profile_limit - 1
            or np.min(points[:, 1]) < 0.0
            or np.max(points[:, 1]) > sample_limit - 1
        ):
            raise ValueError(
                f"{line_key} cluster {original_cluster_id!r} lies outside the "
                f"processed section bounds {section_shape}."
            )

        rounded = np.rint(points)
        if np.allclose(points, rounded, rtol=0.0, atol=1e-9):
            if rounded.min() < np.iinfo(np.int32).min or rounded.max() > np.iinfo(np.int32).max:
                raise ValueError(f"{line_key} cluster coordinates exceed int32 bounds.")
            output_points = rounded.astype(np.int32)
        else:
            output_points = points.astype(np.float64, copy=False)
        normalised[new_cluster_id] = np.ascontiguousarray(output_points)

    return normalised


def validate_legacy_results(results: Mapping[str, Mapping[int, np.ndarray]]) -> None:
    """Validate the complete root schema expected by the 3D pickle adapter."""

    if not isinstance(results, Mapping) or not results:
        raise ValueError("The exported results mapping cannot be empty.")
    seen_indices: set[int] = set()
    total_clusters = 0
    for line_key, clusters in results.items():
        if not isinstance(line_key, str) or not line_key.startswith("line_"):
            raise ValueError(f"Invalid outer key {line_key!r}; expected line_<integer>.")
        try:
            line_index = int(line_key[5:])
        except ValueError as error:
            raise ValueError(f"Invalid outer key {line_key!r}; expected line_<integer>.") from error
        if line_index < 0 or line_key != f"line_{line_index}":
            raise ValueError(f"Non-canonical outer key: {line_key!r}.")
        if line_index in seen_indices:
            raise ValueError(f"Duplicate numerical line index: {line_index}.")
        seen_indices.add(line_index)
        if not isinstance(clusters, Mapping):
            raise TypeError(f"Clusters for {line_key} must be a mapping.")
        total_clusters += len(clusters)
        for cluster_id, points in clusters.items():
            if not isinstance(cluster_id, int):
                raise TypeError(f"Cluster IDs for {line_key} must be Python integers.")
            point_array = np.asarray(points)
            if point_array.ndim != 2 or point_array.shape[1] != 2:
                raise ValueError(f"{line_key} cluster {cluster_id} is not an (N, 2) array.")
            if len(point_array) < 2 or not np.isfinite(point_array).all():
                raise ValueError(f"{line_key} cluster {cluster_id} is invalid.")
    if total_clusters == 0:
        raise ValueError(
            "Every processed section is empty; the 3D loader requires at least "
            "one fault line."
        )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _axis_metadata(values: Sequence[Any] | np.ndarray, *, stride: int) -> dict[str, Any]:
    array = np.asarray(values).ravel()
    if len(array) == 0:
        raise ValueError("A SEG-Y coordinate axis cannot be empty.")
    record: dict[str, Any] = {
        "count": int(len(array)),
        "start": _jsonable(array[0]),
        "end": _jsonable(array[-1]),
        "stride_from_native_axis": int(stride),
    }
    if len(array) == 1:
        record.update({"regular": True, "step": None})
        return record
    differences = np.diff(array.astype(np.float64))
    regular = bool(np.allclose(differences, differences[0], rtol=0.0, atol=1e-9))
    record["regular"] = regular
    if regular:
        record["step"] = float(differences[0])
    else:
        record["values"] = _jsonable(array)
    return record


def _processed_axis(
    reader_axis: Sequence[Any] | np.ndarray,
    *,
    stride: int,
    expected_length: int,
    axis_name: str,
) -> np.ndarray:
    """Accept reader properties that expose either native or processed axes."""

    axis = np.asarray(reader_axis).ravel()
    sampled = axis[::stride]
    if len(sampled) == expected_length:
        return sampled
    if len(axis) == expected_length:
        return axis
    raise ValueError(
        f"{axis_name} axis length {len(axis)} is incompatible with processed "
        f"section length {expected_length} and stride {stride}."
    )


def _atomic_pickle_dump(
    value: Any,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output pickle already exists: {output_path.resolve()}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            pickle.dump(value, output_file, protocol=4)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_json_dump(
    value: Mapping[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Metadata file already exists: {output_path.resolve()}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(_jsonable(value), output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_json_object(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration JSON must contain an object: {input_path}")
    return value


def metadata_path_for_fault_lines(output_pickle: str | Path) -> Path:
    """Return the canonical sidecar path for a 2D fault-line pickle."""

    output_pickle = Path(output_pickle)
    return output_pickle.with_suffix(".metadata.json")


def _default_metadata_path(output_pickle: Path) -> Path:
    """Backward-compatible private alias for the public path helper."""

    return metadata_path_for_fault_lines(output_pickle)


def _validate_interface_metadata(
    metadata: Mapping[str, Any],
    results: Mapping[str, Mapping[int, np.ndarray]],
) -> None:
    """Validate the metadata fields shared with the 3D consumer."""

    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"metadata.schema_version must be {SCHEMA_VERSION!r}."
        )
    if metadata.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"metadata.artifact_type must be {ARTIFACT_TYPE!r}.")

    line_mapping = metadata.get("line_mapping")
    if not isinstance(line_mapping, Mapping):
        raise TypeError("metadata.line_mapping must be an object.")
    try:
        inline_origin = int(line_mapping["inline_origin"])
        inline_step = int(line_mapping["inline_step"])
        inline_start = int(line_mapping["inline_start"])
        inline_end = int(line_mapping["inline_end"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "metadata.line_mapping must define integer inline_origin, "
            "inline_step, inline_start, and inline_end."
        ) from error
    if inline_step <= 0 or inline_end < inline_start:
        raise ValueError("metadata.line_mapping contains an invalid range or step.")

    coordinates = metadata.get("coordinates")
    if not isinstance(coordinates, Mapping):
        raise TypeError("metadata.coordinates must be an object.")
    if tuple(coordinates.get("point_columns", ())) != POINT_COLUMNS:
        raise ValueError(
            f"metadata.coordinates.point_columns must be {list(POINT_COLUMNS)!r}."
        )
    if coordinates.get("coordinate_mode") != "downsampled_array_index":
        raise ValueError(
            "metadata.coordinates.coordinate_mode must be "
            "'downsampled_array_index'."
        )

    section_records = metadata.get("sections")
    if not isinstance(section_records, Mapping):
        raise TypeError("metadata.sections must be an object.")
    if set(section_records) != set(results):
        raise ValueError("metadata.sections keys do not match the exported line keys.")
    for line_key, clusters in results.items():
        line_index = int(line_key[5:])
        expected_inline = inline_origin + line_index * inline_step
        record = section_records[line_key]
        if not isinstance(record, Mapping):
            raise TypeError(f"metadata.sections.{line_key} must be an object.")
        if int(record.get("inline_number", -1)) != expected_inline:
            raise ValueError(
                f"metadata.sections.{line_key}.inline_number does not agree "
                "with line_mapping."
            )
        if not inline_start <= expected_inline <= inline_end:
            raise ValueError(f"{line_key} maps outside the declared inline range.")
        expected_points = sum(len(points) for points in clusters.values())
        if int(record.get("fault_line_count", -1)) != len(clusters):
            raise ValueError(f"Incorrect fault_line_count for {line_key}.")
        if int(record.get("point_count", -1)) != expected_points:
            raise ValueError(f"Incorrect point_count for {line_key}.")

    counts = metadata.get("counts")
    if not isinstance(counts, Mapping):
        raise TypeError("metadata.counts must be an object.")
    expected_counts = {
        "sections": len(results),
        "nonempty_sections": sum(bool(clusters) for clusters in results.values()),
        "fault_lines": sum(len(clusters) for clusters in results.values()),
        "points": sum(
            len(points)
            for clusters in results.values()
            for points in clusters.values()
        ),
    }
    for name, expected_value in expected_counts.items():
        if int(counts.get(name, -1)) != expected_value:
            raise ValueError(f"metadata.counts.{name} is inconsistent with the data.")


def export_fault_lines_2d(
    results: Mapping[str, Mapping[int, np.ndarray]],
    output_pickle: str | Path,
    metadata: Mapping[str, Any],
    *,
    metadata_json: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the versioned 2D-to-3D interface artifact.

    The data pickle contains only ``line_<index> -> cluster -> (N, 2)`` arrays,
    while the canonical JSON sidecar contains the authoritative inline mapping,
    coordinate convention, parameters, provenance, counts, and data hash.
    ``fault_surface_reconstruction_3d.py`` auto-discovers this sidecar.
    """

    output_pickle = Path(output_pickle)
    metadata_path = (
        Path(metadata_json)
        if metadata_json is not None
        else metadata_path_for_fault_lines(output_pickle)
    )
    if output_pickle.resolve() == metadata_path.resolve():
        raise ValueError("The pickle and metadata paths must be different.")
    for output_path in (output_pickle, metadata_path):
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path.resolve()}")

    validate_legacy_results(results)
    _validate_interface_metadata(metadata, results)
    _atomic_pickle_dump(results, output_pickle, overwrite=overwrite)

    finalized_metadata = dict(metadata)
    finalized_metadata["output"] = {
        "data_file": output_pickle.name,
        "data_format": "trusted-pickle",
        "data_sha256": sha256_file(output_pickle),
        # Legacy aliases remain for readers created before the unified interface.
        "pickle_file": output_pickle.name,
        "pickle_sha256": sha256_file(output_pickle),
        "metadata_file": metadata_path.name,
        "trusted_pickle_warning": (
            "Only load this pickle when it was produced locally or obtained "
            "from a trusted source."
        ),
    }
    _atomic_json_dump(finalized_metadata, metadata_path, overwrite=overwrite)
    return finalized_metadata


def configure_determinism() -> None:
    """Configure deterministic inference where supported by the Torch backend."""

    # CUDA requires this workspace setting for deterministic CuBLAS operations.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_batch(
    *,
    input_segy: str | Path,
    model_weights: str | Path,
    output_pickle: str | Path,
    metadata_json: str | Path | None,
    inline_origin: int,
    inline_start: int,
    inline_end: int,
    inline_step: int,
    crossline_stride: int,
    sample_stride: int,
    inference_config: InferenceConfig,
    postprocess_config: PostprocessConfig,
    device: str,
    base_channels: int,
    dropout: float,
    expected_model_sha256: str | None,
    allow_missing_inlines: bool,
    fail_on_empty_section: bool,
    hash_input_segy: bool,
    overwrite: bool,
    deterministic: bool,
) -> dict[str, Any]:
    """Process selected SEG-Y inlines and write a 3D-compatible pickle."""

    input_segy = Path(input_segy)
    model_weights = Path(model_weights)
    output_pickle = Path(output_pickle)
    metadata_path = (
        Path(metadata_json)
        if metadata_json is not None
        else _default_metadata_path(output_pickle)
    )
    if crossline_stride <= 0 or sample_stride <= 0:
        raise ValueError("crossline_stride and sample_stride must be positive.")
    if output_pickle.resolve() == metadata_path.resolve():
        raise ValueError("The pickle and metadata paths must be different.")
    for output_path in (output_pickle, metadata_path):
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path.resolve()}")

    if deterministic:
        configure_determinism()

    model, torch_device = load_uassnet_model(
        weights_path=model_weights,
        device=device,
        base_channels=base_channels,
        dropout=dropout,
        expected_sha256=expected_model_sha256,
    )
    model.eval()

    started = time.perf_counter()
    results: dict[str, dict[int, np.ndarray]] = {}
    section_records: dict[str, dict[str, Any]] = {}
    processed_profile_axis: np.ndarray | None = None
    processed_sample_axis: np.ndarray | None = None
    processed_shape: tuple[int, int] | None = None

    with SegyInlineReader(
        input_segy,
        crossline_stride=crossline_stride,
        sample_stride=sample_stride,
    ) as reader:
        available_inlines = np.asarray(reader.ilines, dtype=np.int64)
        selected_inlines = select_inline_numbers(
            available_inlines,
            inline_origin=inline_origin,
            inline_start=inline_start,
            inline_end=inline_end,
            inline_step=inline_step,
            allow_missing=allow_missing_inlines,
        )

        for position, inline_number in enumerate(selected_inlines, start=1):
            section_started = time.perf_counter()
            seismic_inline = reader.read(inline_number)
            section_source = getattr(seismic_inline, "section", seismic_inline)
            section = np.asarray(section_source, dtype=np.float32)
            if section.ndim != 2 or not np.isfinite(section).all():
                raise ValueError(
                    f"Inline {inline_number} did not produce a finite 2D section."
                )
            returned_inline_number = getattr(seismic_inline, "inline_number", inline_number)
            if int(returned_inline_number) != int(inline_number):
                raise ValueError(
                    f"SEG-Y reader returned inline {returned_inline_number} while "
                    f"inline {inline_number} was requested."
                )
            section_shape = (int(section.shape[0]), int(section.shape[1]))
            if processed_shape is None:
                processed_shape = section_shape
                processed_profile_axis = _processed_axis(
                    getattr(seismic_inline, "crossline_values", reader.xlines),
                    stride=crossline_stride,
                    expected_length=section_shape[0],
                    axis_name="Crossline",
                )
                processed_sample_axis = _processed_axis(
                    getattr(seismic_inline, "sample_values", reader.samples),
                    stride=sample_stride,
                    expected_length=section_shape[1],
                    axis_name="Sample",
                )
            elif section_shape != processed_shape:
                raise ValueError(
                    f"Inline {inline_number} shape {section_shape} differs from "
                    f"the first processed shape {processed_shape}."
                )

            line_key = line_key_for_inline(
                inline_number,
                inline_origin=inline_origin,
                inline_step=inline_step,
            )
            if line_key in results:
                raise RuntimeError(f"Duplicate line key generated: {line_key}")

            inline_result = process_inline_section(
                model=model,
                section=section,
                inline_number=inline_number,
                inference_config=inference_config,
                postprocess_config=postprocess_config,
                device=torch_device,
            )
            clusters = _normalise_clusters(
                inline_result.postprocess.interpolated_clusters,
                section_shape=section_shape,
                line_key=line_key,
            )
            if not clusters and fail_on_empty_section:
                raise RuntimeError(f"No final fault-line clusters for inline {inline_number}.")
            results[line_key] = clusters
            point_count = int(sum(len(points) for points in clusters.values()))
            section_records[line_key] = {
                "inline_number": int(inline_number),
                "processed_shape": list(section_shape),
                "fault_line_count": len(clusters),
                "point_count": point_count,
                "stage_counts": _jsonable(inline_result.stage_counts),
                "elapsed_seconds": time.perf_counter() - section_started,
            }
            LOGGER.info(
                "Processed inline %d as %s (%d/%d): %d lines, %d points.",
                inline_number,
                line_key,
                position,
                len(selected_inlines),
                len(clusters),
                point_count,
            )

        native_xlines = np.asarray(reader.xlines).ravel()
        native_samples = np.asarray(reader.samples).ravel()

    assert processed_shape is not None
    assert processed_profile_axis is not None
    assert processed_sample_axis is not None
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "input": {
            "segy_file": input_segy.name,
            "segy_sha256": sha256_file(input_segy) if hash_input_segy else None,
            "model_file": model_weights.name,
            "model_sha256": sha256_file(model_weights),
        },
        "line_mapping": {
            "formula": "inline_number = inline_origin + line_index * inline_step",
            "inline_origin": int(inline_origin),
            "inline_step": int(inline_step),
            "inline_start": int(inline_start),
            "inline_end": int(inline_end),
            "selected_inline_count": len(results),
            "selected_inline_numbers": [int(value) for value in selected_inlines],
        },
        "coordinates": {
            "point_columns": list(POINT_COLUMNS),
            "coordinate_mode": "downsampled_array_index",
            "coordinate_units": ["processed_pixel", "processed_pixel"],
            "processed_section_shape": list(processed_shape),
            "crossline_stride": int(crossline_stride),
            "sample_stride": int(sample_stride),
            "native_crossline_axis": _axis_metadata(native_xlines, stride=1),
            "processed_crossline_axis": _axis_metadata(
                processed_profile_axis,
                stride=crossline_stride,
            ),
            "native_sample_axis": _axis_metadata(native_samples, stride=1),
            "processed_sample_axis": _axis_metadata(
                processed_sample_axis,
                stride=sample_stride,
            ),
        },
        "parameters": {
            "inference": _jsonable(inference_config),
            "postprocess": _jsonable(postprocess_config),
            "model": {
                "base_channels": int(base_channels),
                "dropout": float(dropout),
                "device": str(torch_device),
                "deterministic": bool(deterministic),
            },
        },
        "counts": {
            "sections": len(results),
            "nonempty_sections": sum(bool(clusters) for clusters in results.values()),
            "fault_lines": sum(len(clusters) for clusters in results.values()),
            "points": sum(
                len(points)
                for clusters in results.values()
                for points in clusters.values()
            ),
        },
        "sections": section_records,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "segyio": getattr(segyio, "__version__", "unknown"),
            "batch_script_sha256": sha256_file(Path(__file__)),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    metadata = export_fault_lines_2d(
        results,
        output_pickle,
        metadata,
        metadata_json=metadata_path,
        overwrite=overwrite,
    )
    LOGGER.info("Wrote trusted pickle: %s", output_pickle.resolve())
    LOGGER.info("Wrote metadata: %s", metadata_path.resolve())
    return metadata


def build_argument_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Run UASS-Net and complete 2D fault post-processing over selected "
            "SEG-Y inlines, then export the line_n pickle consumed by the 3D module."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_segy", help="Regular-grid SEG-Y volume.")
    parser.add_argument(
        "--model-weights",
        default=str(script_dir / "model_real.pth"),
    )
    parser.add_argument("--output-pickle", default="fault_lines_2d_300_330.pkl")
    parser.add_argument(
        "--metadata-json",
        help="Sidecar path; defaults to OUTPUT_PICKLE with .metadata.json suffix.",
    )
    parser.add_argument("--inline-origin", type=int, default=300)
    parser.add_argument("--inline-start", type=int, default=300)
    parser.add_argument("--inline-end", type=int, default=330)
    parser.add_argument("--inline-step", type=int, default=1)
    parser.add_argument("--crossline-stride", type=int, default=2)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument(
        "--patch-shape",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLUMNS"),
        default=(256, 256),
    )
    parser.add_argument("--overlap", type=int, default=12)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a Torch device.")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--expected-model-sha256",
        default=REFERENCE_MODEL_SHA256,
        help="Set to an empty string to disable the checkpoint hash check.",
    )
    parser.add_argument(
        "--inference-config-json",
        help="Optional JSON object with additional InferenceConfig keyword values.",
    )
    parser.add_argument(
        "--postprocess-config-json",
        help="Optional JSON object passed as PostprocessConfig keyword values.",
    )
    parser.add_argument("--allow-missing-inlines", action="store_true")
    parser.add_argument("--fail-on-empty-section", action="store_true")
    parser.add_argument("--skip-input-sha256", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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

    inference_keywords = _load_json_object(arguments.inference_config_json)
    inference_keywords.update(
        {
            "patch_shape": tuple(int(value) for value in arguments.patch_shape),
            "overlap": int(arguments.overlap),
        }
    )
    postprocess_keywords = _load_json_object(arguments.postprocess_config_json)
    inference_config = InferenceConfig(**inference_keywords)
    postprocess_config = PostprocessConfig(**postprocess_keywords)

    metadata = run_batch(
        input_segy=arguments.input_segy,
        model_weights=arguments.model_weights,
        output_pickle=arguments.output_pickle,
        metadata_json=arguments.metadata_json,
        inline_origin=arguments.inline_origin,
        inline_start=arguments.inline_start,
        inline_end=arguments.inline_end,
        inline_step=arguments.inline_step,
        crossline_stride=arguments.crossline_stride,
        sample_stride=arguments.sample_stride,
        inference_config=inference_config,
        postprocess_config=postprocess_config,
        device=arguments.device,
        base_channels=arguments.base_channels,
        dropout=arguments.dropout,
        expected_model_sha256=arguments.expected_model_sha256 or None,
        allow_missing_inlines=arguments.allow_missing_inlines,
        fail_on_empty_section=arguments.fail_on_empty_section,
        hash_input_segy=not arguments.skip_input_sha256,
        overwrite=arguments.overwrite,
        deterministic=arguments.deterministic,
    )
    LOGGER.info("Batch summary: %s", json.dumps(metadata["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
