#!/usr/bin/env python3
"""Export and visualize reconstructed 3D fault surfaces.

The viewer accepts either of the following artifacts:

1. A current ``fault_surfaces.npz`` file produced by
   ``fault_surface_reconstruction_3d.py``.
2. A legacy pickle whose root keys identify already gridded surfaces and whose
   values contain ``grid_x``, ``grid_y``, and ``grid_z``.  The optional
   ``grid_z_org`` field is retained when it is present.

The legacy pickle is an already gridded surface product.  It is therefore
visualized directly and is not sent through the current 3D linking and
reconstruction algorithm again.  NaN values outside an interpolated surface's
support are preserved as holes rather than filled with invented values.

For a SEG-Y volume, only requested inline sections are read.  The complete
volume is never materialized in memory.  Public outputs use PNG, CSV, JSON,
and non-object NumPy NPZ files; the input pickle is not copied into the output.

Example from the repository root after placing the external SEG-Y input under
``data/segy``::

    python demo/visualize_fault_surfaces_3d.py \
        demo/outputs_400_500.pkl \
        --input-format legacy-grid-pickle \
        --allow-legacy-pickle \
        --segy data/segy/400_500.segy \
        --inline-origin 400 \
        --crossline-stride 2 \
        --sample-stride 2 \
        --z-field grid_z \
        --vertical-unit m \
        --inline-slices 400 450 500 \
        --output-dir outputs/visualization/fault3d_demo_400_500

Important coordinate convention
-------------------------------
Legacy ``grid_x``, ``grid_y``, and ``grid_z`` values are processed-grid
indices, not survey coordinates.  With the defaults above they are mapped to
the actual axes read from the SEG-Y file as follows:

* ``grid_x`` -> inline axis, using ``inline_origin`` for the legacy offset;
* ``grid_y`` -> ``segy.xlines[grid_y * crossline_stride]``;
* ``grid_z`` -> ``segy.samples[grid_z * sample_stride]``.

Fractional values such as ``grid_z_org`` are mapped by linear interpolation.
No time/depth unit is inferred from the SEG-Y sample coordinates.  Specify
``--vertical-unit`` only when that unit is known independently.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.patches import Patch
import numpy as np
import segyio


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from file_utils import sha256_file


LOGGER = logging.getLogger("fault3d-viewer")
SCHEMA_VERSION = "fault-surface-visualization/1.1"
LEGACY_REQUIRED_FIELDS = ("grid_x", "grid_y", "grid_z")
LEGACY_OPTIONAL_FIELDS = ("grid_z_org",)
OUTPUT_BASENAMES = {
    "fault_surfaces_oblique.png",
    "fault_surfaces_map_view.png",
    "surfaces_normalized.npz",
    "surface_inventory.csv",
    "inline_intersections.csv",
    "visualization_metadata.json",
    "seismic_slices.npz",
}


@dataclass(frozen=True)
class SegyGeometry:
    """Geometry arrays and lightweight SEG-Y header information."""

    inline_headers: np.ndarray
    crossline_headers: np.ndarray
    sample_coordinates: np.ndarray
    trace_count: int
    sorting: int
    format_code: int


@dataclass
class SurfaceGrid:
    """One structured surface in processed-index and SEG-Y-axis coordinates."""

    surface_id: str
    inline_index: np.ndarray
    profile_index: np.ndarray
    sample_index: np.ndarray
    inline_header: np.ndarray
    crossline_header: np.ndarray
    sample_coordinate: np.ndarray
    original_sample_index: np.ndarray | None = None
    original_sample_coordinate: np.ndarray | None = None
    faces_local: np.ndarray | None = None
    observation_rmse: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.inline_index.shape)

    @property
    def vertex_count(self) -> int:
        return int(self.inline_index.size)

    @property
    def valid_vertex_count(self) -> int:
        return int(np.isfinite(self.sample_index).sum())

    @property
    def invalid_vertex_count(self) -> int:
        return self.vertex_count - self.valid_vertex_count

    @property
    def face_count(self) -> int:
        if self.faces_local is not None:
            return int(len(self.faces_local))
        rows, columns = self.shape
        return max(0, 2 * (rows - 1) * (columns - 1))


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Unpickle plain NumPy arrays without permitting arbitrary imports."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy" and name == "ndarray":
            return np.ndarray
        if module == "numpy" and name == "dtype":
            return np.dtype
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"}:
            if name == "_reconstruct":
                return np.core.multiarray._reconstruct
            if name == "scalar":
                return np.core.multiarray.scalar
        raise pickle.UnpicklingError(
            f"Blocked pickle global: {module}.{name}. "
            "Only plain NumPy-array artifacts are accepted."
        )


def natural_key(text: str) -> tuple[Any, ...]:
    """Sort identifiers containing numbers in human-readable order."""

    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    )


def read_segy_geometry(segy_path: Path) -> SegyGeometry:
    """Read survey axes and headers without reading the seismic volume."""

    with segyio.open(str(segy_path), "r", strict=True) as segy_file:
        geometry = SegyGeometry(
            inline_headers=np.asarray(segy_file.ilines, dtype=np.int64),
            crossline_headers=np.asarray(segy_file.xlines, dtype=np.int64),
            sample_coordinates=np.asarray(segy_file.samples, dtype=np.float64),
            trace_count=int(segy_file.tracecount),
            sorting=int(segy_file.sorting),
            format_code=int(segy_file.format),
        )
    if min(
        len(geometry.inline_headers),
        len(geometry.crossline_headers),
        len(geometry.sample_coordinates),
    ) == 0:
        raise ValueError("The SEG-Y geometry contains an empty coordinate axis.")
    for name, axis in (
        ("inline", geometry.inline_headers),
        ("crossline", geometry.crossline_headers),
        ("sample", geometry.sample_coordinates),
    ):
        if not np.isfinite(axis).all():
            raise ValueError(f"The SEG-Y {name} axis contains non-finite values.")
        if len(axis) > 1 and not np.all(np.diff(axis) > 0):
            raise ValueError(f"The SEG-Y {name} axis must be strictly increasing.")
    return geometry


def axis_step(axis: np.ndarray) -> float | None:
    """Return the regular step, or ``None`` when an axis is irregular."""

    if len(axis) < 2:
        return None
    differences = np.diff(axis.astype(float))
    if np.allclose(differences, differences[0], rtol=1e-7, atol=1e-9):
        return float(differences[0])
    return None


def index_to_axis_values(
    indices: np.ndarray,
    axis: np.ndarray,
    name: str,
    *,
    allow_nan: bool = False,
) -> np.ndarray:
    """Map fractional indices to an axis, optionally preserving no-data NaNs."""

    values = np.asarray(indices, dtype=np.float64)
    if np.isinf(values).any() or (not allow_nan and np.isnan(values).any()):
        raise ValueError(f"{name} indices contain unsupported NaN or infinity values.")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"{name} indices do not contain any finite values.")
    tolerance = 1e-7
    minimum = float(values[finite].min())
    maximum = float(values[finite].max())
    if minimum < -tolerance or maximum > (len(axis) - 1) + tolerance:
        raise ValueError(
            f"{name} index range [{minimum:g}, {maximum:g}] is outside "
            f"[0, {len(axis) - 1}]. Check the declared processing stride."
        )
    clipped = np.clip(values[finite], 0.0, float(len(axis) - 1))
    mapped_finite = np.interp(
        clipped,
        np.arange(len(axis), dtype=np.float64),
        axis.astype(np.float64),
    )
    mapped = np.full(values.shape, np.nan, dtype=np.float64)
    mapped[finite] = mapped_finite
    return mapped


def axis_values_to_indices(values: np.ndarray, axis: np.ndarray, name: str) -> np.ndarray:
    """Map values on an increasing axis to fractional zero-based indices."""

    coordinates = np.asarray(values, dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{name} coordinates contain NaN or infinity.")
    tolerance = max(1e-7, abs(float(axis[-1] - axis[0])) * 1e-10)
    if coordinates.min() < axis[0] - tolerance or coordinates.max() > axis[-1] + tolerance:
        raise ValueError(
            f"{name} coordinate range [{coordinates.min():g}, {coordinates.max():g}] "
            f"is outside the SEG-Y range [{axis[0]:g}, {axis[-1]:g}]."
        )
    mapped = np.interp(
        coordinates.ravel(),
        axis.astype(np.float64),
        np.arange(len(axis), dtype=np.float64),
    )
    return mapped.reshape(coordinates.shape)


def processed_axes(
    geometry: SegyGeometry,
    crossline_stride: int,
    sample_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the SEG-Y axes retained by the 2D processing strides."""

    if crossline_stride < 1 or sample_stride < 1:
        raise ValueError("crossline_stride and sample_stride must be positive.")
    return (
        geometry.crossline_headers[::crossline_stride],
        geometry.sample_coordinates[::sample_stride],
    )


def _validate_grid_arrays(
    surface_id: str,
    fields: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Validate required and optional arrays in one legacy surface grid."""

    missing = [name for name in LEGACY_REQUIRED_FIELDS if name not in fields]
    if missing:
        raise ValueError(f"Surface {surface_id!r} is missing fields: {missing}.")
    field_names = LEGACY_REQUIRED_FIELDS + tuple(
        name for name in LEGACY_OPTIONAL_FIELDS if name in fields
    )
    arrays = {name: np.asarray(fields[name]) for name in field_names}
    reference_shape = arrays["grid_x"].shape
    if len(reference_shape) != 2 or min(reference_shape) < 2:
        raise ValueError(
            f"Surface {surface_id!r} must be a 2D grid with at least 2 x 2 nodes."
        )
    for name, array in arrays.items():
        if array.shape != reference_shape:
            raise ValueError(
                f"Surface {surface_id!r} field {name!r} has shape {array.shape}; "
                f"expected {reference_shape}."
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"Surface {surface_id!r} field {name!r} is not numeric.")
        if name in {"grid_x", "grid_y"} and not np.isfinite(array).all():
            raise ValueError(
                f"Surface {surface_id!r} field {name!r} contains NaN or infinity."
            )
        if name in {"grid_z", "grid_z_org"}:
            if np.isinf(array).any():
                raise ValueError(
                    f"Surface {surface_id!r} field {name!r} contains infinity."
                )
            if not np.isfinite(array).any():
                raise ValueError(
                    f"Surface {surface_id!r} field {name!r} has no finite values."
                )
    return arrays


def load_legacy_surface_pickle(
    input_path: Path,
    geometry: SegyGeometry,
    *,
    inline_origin: int,
    legacy_grid_x: str,
    crossline_stride: int,
    sample_stride: int,
    z_field: str,
) -> list[SurfaceGrid]:
    """Load the already-gridded legacy artifact with a restricted unpickler."""

    with input_path.open("rb") as input_file:
        raw = RestrictedNumpyUnpickler(input_file).load()
    if not isinstance(raw, Mapping) or not raw:
        raise TypeError("The legacy pickle root must be a non-empty mapping.")

    processed_crosslines, processed_samples = processed_axes(
        geometry, crossline_stride, sample_stride
    )
    inline_lookup = {
        int(header): index for index, header in enumerate(geometry.inline_headers)
    }
    if legacy_grid_x == "offset" and inline_origin not in inline_lookup:
        raise ValueError(
            f"inline_origin {inline_origin} is not present in the SEG-Y inline axis."
        )
    origin_index = inline_lookup.get(inline_origin, 0)

    surfaces: list[SurfaceGrid] = []
    for raw_id in sorted(raw, key=lambda item: natural_key(str(item))):
        surface_id = str(raw_id)
        fields = raw[raw_id]
        if not isinstance(fields, Mapping):
            raise TypeError(f"Surface {surface_id!r} must map field names to arrays.")
        arrays = _validate_grid_arrays(surface_id, fields)
        if z_field not in arrays:
            raise ValueError(
                f"Surface {surface_id!r} does not contain requested --z-field "
                f"{z_field!r}. Available fields: {sorted(arrays)}."
            )
        grid_x = arrays["grid_x"].astype(np.float64, copy=False)
        if legacy_grid_x == "offset":
            inline_index = grid_x + float(origin_index)
        else:
            inline_index = axis_values_to_indices(
                grid_x, geometry.inline_headers, "inline"
            )
        profile_index = arrays["grid_y"].astype(np.float64, copy=False)
        sample_index = arrays[z_field].astype(np.float64, copy=False)
        original_sample_index = (
            arrays["grid_z_org"].astype(np.float64, copy=False)
            if "grid_z_org" in arrays
            else None
        )
        original_sample_coordinate = (
            index_to_axis_values(
                original_sample_index,
                processed_samples,
                "original processed sample",
                allow_nan=True,
            )
            if original_sample_index is not None
            else None
        )

        surfaces.append(
            SurfaceGrid(
                surface_id=surface_id,
                inline_index=inline_index,
                profile_index=profile_index,
                sample_index=sample_index,
                inline_header=index_to_axis_values(
                    inline_index, geometry.inline_headers, "inline"
                ),
                crossline_header=index_to_axis_values(
                    profile_index, processed_crosslines, "processed profile"
                ),
                sample_coordinate=index_to_axis_values(
                    sample_index,
                    processed_samples,
                    "processed sample",
                    allow_nan=True,
                ),
                original_sample_index=original_sample_index,
                original_sample_coordinate=original_sample_coordinate,
                faces_local=finite_structured_faces(sample_index),
            )
        )
    return surfaces


def _require_npz_array(archive: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in archive:
        raise ValueError(f"Canonical NPZ is missing array {name!r}.")
    return np.asarray(archive[name])


def load_canonical_npz(
    input_path: Path,
    geometry: SegyGeometry,
    *,
    crossline_stride: int,
    sample_stride: int,
) -> list[SurfaceGrid]:
    """Load the flattened mesh NPZ emitted by the current reconstruction code."""

    processed_crosslines, processed_samples = processed_axes(
        geometry, crossline_stride, sample_stride
    )
    with np.load(input_path, allow_pickle=False) as archive:
        surface_ids = _require_npz_array(archive, "surface_ids")
        vertices = _require_npz_array(archive, "vertices")
        vertex_offsets = _require_npz_array(archive, "vertex_offsets")
        grid_shapes = _require_npz_array(archive, "grid_shapes")
        faces = _require_npz_array(archive, "faces")
        face_offsets = _require_npz_array(archive, "face_offsets")
        observation_rmse = (
            np.asarray(archive["observation_rmse"])
            if "observation_rmse" in archive
            else None
        )

    if surface_ids.ndim != 1 or surface_ids.dtype.kind not in "US":
        raise TypeError("surface_ids must be a one-dimensional string array.")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3).")
    if not np.issubdtype(vertices.dtype, np.number) or not np.isfinite(vertices).all():
        raise ValueError("vertices must be a finite numeric array.")
    surface_count = len(surface_ids)
    if grid_shapes.shape != (surface_count, 2):
        raise ValueError("grid_shapes must have shape (surface_count, 2).")
    if not np.issubdtype(grid_shapes.dtype, np.integer):
        raise TypeError("grid_shapes must use an integer dtype.")
    if vertex_offsets.shape != (surface_count + 1,):
        raise ValueError("vertex_offsets must have surface_count + 1 entries.")
    if not np.issubdtype(vertex_offsets.dtype, np.integer):
        raise TypeError("vertex_offsets must use an integer dtype.")
    if face_offsets.shape != (surface_count + 1,):
        raise ValueError("face_offsets must have surface_count + 1 entries.")
    if not np.issubdtype(face_offsets.dtype, np.integer):
        raise TypeError("face_offsets must use an integer dtype.")
    if np.any(np.diff(vertex_offsets) < 0) or np.any(np.diff(face_offsets) < 0):
        raise ValueError("Canonical NPZ offsets must be monotonically non-decreasing.")
    if int(vertex_offsets[0]) != 0 or int(vertex_offsets[-1]) != len(vertices):
        raise ValueError("vertex_offsets do not span the vertices array.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3).")
    if not np.issubdtype(faces.dtype, np.integer):
        raise TypeError("faces must use an integer dtype.")
    if int(face_offsets[0]) != 0 or int(face_offsets[-1]) != len(faces):
        raise ValueError("face_offsets do not span the faces array.")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("faces contain an out-of-range vertex index.")
    if observation_rmse is not None:
        if observation_rmse.shape != (surface_count,):
            raise ValueError("observation_rmse must have one value per surface.")
        if not np.issubdtype(observation_rmse.dtype, np.number):
            raise TypeError("observation_rmse must be numeric.")
        if not np.isfinite(observation_rmse).all():
            raise ValueError("observation_rmse contains NaN or infinity.")

    surfaces: list[SurfaceGrid] = []
    for surface_index, raw_id in enumerate(surface_ids):
        start = int(vertex_offsets[surface_index])
        stop = int(vertex_offsets[surface_index + 1])
        rows, columns = (int(value) for value in grid_shapes[surface_index])
        if rows < 2 or columns < 2 or rows * columns != stop - start:
            raise ValueError(
                f"Invalid grid shape {(rows, columns)} for surface {raw_id!s}."
            )
        face_start = int(face_offsets[surface_index])
        face_stop = int(face_offsets[surface_index + 1])
        surface_faces_global = faces[face_start:face_stop]
        if len(surface_faces_global) and (
            surface_faces_global.min() < start or surface_faces_global.max() >= stop
        ):
            raise ValueError(
                f"Face interval for surface {raw_id!s} references another surface."
            )
        faces_local = surface_faces_global.astype(np.int64, copy=True) - start
        expected_faces = structured_faces(rows, columns)
        if len(faces_local) != len(expected_faces):
            raise ValueError(
                f"Surface {raw_id!s} has {len(faces_local)} faces; the declared "
                f"structured grid requires {len(expected_faces)}."
            )
        canonical_actual = np.sort(faces_local, axis=1)
        canonical_expected = np.sort(expected_faces, axis=1)
        actual_order = np.lexsort(canonical_actual.T[::-1])
        expected_order = np.lexsort(canonical_expected.T[::-1])
        if not np.array_equal(
            canonical_actual[actual_order], canonical_expected[expected_order]
        ):
            raise ValueError(
                f"Surface {raw_id!s} faces do not match its structured grid topology."
            )
        grid = vertices[start:stop].reshape(rows, columns, 3)
        inline_header = grid[..., 0].astype(np.float64, copy=False)
        inline_index = axis_values_to_indices(
            inline_header, geometry.inline_headers, "inline"
        )
        profile_index = grid[..., 1].astype(np.float64, copy=False)
        sample_index = grid[..., 2].astype(np.float64, copy=False)
        surfaces.append(
            SurfaceGrid(
                surface_id=str(raw_id),
                inline_index=inline_index,
                profile_index=profile_index,
                sample_index=sample_index,
                inline_header=inline_header,
                crossline_header=index_to_axis_values(
                    profile_index, processed_crosslines, "processed profile"
                ),
                sample_coordinate=index_to_axis_values(
                    sample_index, processed_samples, "processed sample"
                ),
                faces_local=faces_local,
                observation_rmse=(
                    float(observation_rmse[surface_index])
                    if observation_rmse is not None
                    else None
                ),
            )
        )
    return surfaces


def detect_input_format(input_path: Path) -> str:
    """Detect only the two explicitly supported artifact formats by suffix."""

    suffix = input_path.suffix.lower()
    if suffix == ".npz":
        return "canonical-npz"
    if suffix in {".pkl", ".pickle"}:
        return "legacy-grid-pickle"
    raise ValueError(
        f"Cannot infer an input format from {input_path.name!r}; use --input-format."
    )


def load_run_metadata(
    input_path: Path,
    explicit_path: str | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Load reconstruction metadata explicitly or from the NPZ directory."""

    metadata_path = (
        Path(explicit_path).expanduser().resolve()
        if explicit_path is not None
        else input_path.parent / "run_metadata.json"
    )
    if not metadata_path.is_file():
        if explicit_path is not None:
            raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
        return None, None
    try:
        with metadata_path.open("r", encoding="utf-8") as input_file:
            metadata = json.load(input_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid reconstruction metadata JSON: {error}") from error
    if not isinstance(metadata, dict):
        raise TypeError("Reconstruction run metadata must contain a JSON object.")
    return metadata, metadata_path


def metadata_processing_strides(metadata: Mapping[str, Any]) -> tuple[int, int]:
    """Extract the 2D processing strides propagated into a 3D run record."""

    input_record = metadata.get("input")
    if not isinstance(input_record, Mapping):
        raise ValueError("run_metadata.input is missing or is not an object.")
    coordinates = input_record.get("artifact_coordinates")
    if not isinstance(coordinates, Mapping):
        raise ValueError(
            "run_metadata.input.artifact_coordinates is missing. Supply both "
            "--crossline-stride and --sample-stride explicitly."
        )
    if coordinates.get("coordinate_mode") != "downsampled_array_index":
        raise ValueError(
            "Canonical surface coordinates are not declared as "
            "downsampled_array_index in run_metadata."
        )
    try:
        crossline_stride = int(coordinates["crossline_stride"])
        sample_stride = int(coordinates["sample_stride"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "run_metadata artifact coordinates do not contain valid processing strides."
        ) from error
    if min(crossline_stride, sample_stride) < 1:
        raise ValueError("Processing strides in run_metadata must be positive.")
    return crossline_stride, sample_stride


def resolve_processing_strides(
    *,
    input_format: str,
    input_path: Path,
    run_metadata_path: str | None,
    crossline_stride: int | None,
    sample_stride: int | None,
) -> tuple[int, int, dict[str, Any] | None, Path | None]:
    """Resolve explicit legacy strides or verify canonical metadata strides."""

    if input_format == "legacy-grid-pickle":
        if run_metadata_path is not None:
            raise ValueError("--run-metadata applies only to canonical NPZ input.")
        if crossline_stride is None or sample_stride is None:
            raise ValueError(
                "Legacy grid mapping requires explicit --crossline-stride and "
                "--sample-stride values because the pickle has no metadata."
            )
        return int(crossline_stride), int(sample_stride), None, None

    metadata, metadata_path = load_run_metadata(input_path, run_metadata_path)
    metadata_strides: tuple[int, int] | None = None
    if metadata is not None:
        try:
            metadata_strides = metadata_processing_strides(metadata)
        except ValueError:
            if crossline_stride is None or sample_stride is None:
                raise
            LOGGER.warning(
                "The available run metadata does not contain usable artifact "
                "strides; using both explicit CLI stride values instead."
            )
            return (
                int(crossline_stride),
                int(sample_stride),
                None,
                metadata_path,
            )
    if metadata_strides is None:
        if crossline_stride is None or sample_stride is None:
            raise ValueError(
                "Canonical NPZ mapping requires its run_metadata.json or explicit "
                "--crossline-stride and --sample-stride values."
            )
        return int(crossline_stride), int(sample_stride), None, None

    metadata_crossline, metadata_sample = metadata_strides
    if crossline_stride is not None and int(crossline_stride) != metadata_crossline:
        raise ValueError(
            f"--crossline-stride {crossline_stride} conflicts with run_metadata "
            f"value {metadata_crossline}."
        )
    if sample_stride is not None and int(sample_stride) != metadata_sample:
        raise ValueError(
            f"--sample-stride {sample_stride} conflicts with run_metadata value "
            f"{metadata_sample}."
        )
    return metadata_crossline, metadata_sample, metadata, metadata_path


def structured_faces(rows: int, columns: int, vertex_offset: int = 0) -> np.ndarray:
    """Triangulate a row-major structured grid with two faces per cell."""

    if rows < 2 or columns < 2:
        return np.empty((0, 3), dtype=np.int64)
    row, column = np.meshgrid(
        np.arange(rows - 1, dtype=np.int64),
        np.arange(columns - 1, dtype=np.int64),
        indexing="ij",
    )
    top_left = (row * columns + column).ravel() + int(vertex_offset)
    bottom_left = top_left + columns
    top_right = top_left + 1
    bottom_right = bottom_left + 1
    first = np.column_stack((top_left, bottom_left, top_right))
    second = np.column_stack((top_right, bottom_left, bottom_right))
    return np.concatenate((first, second), axis=0)


def finite_structured_faces(sample_grid: np.ndarray) -> np.ndarray:
    """Return only triangles whose three vertical coordinates are finite."""

    rows, columns = sample_grid.shape
    faces = structured_faces(rows, columns)
    valid_vertices = np.isfinite(sample_grid).ravel()
    return faces[np.all(valid_vertices[faces], axis=1)]


def stable_colors(surface_count: int) -> list[tuple[float, float, float, float]]:
    """Return stable, visually distinct colors for surface identifiers."""

    color_map = plt.get_cmap("tab20")
    return [color_map(index % 20) for index in range(surface_count)]


def inclusive_subsample_indices(length: int, stride: int) -> np.ndarray:
    """Subsample a structured axis while retaining both boundary nodes."""

    if stride <= 1 or length <= 2:
        return np.arange(length, dtype=np.int64)
    indices = np.arange(0, length, stride, dtype=np.int64)
    if indices[-1] != length - 1:
        indices = np.append(indices, length - 1)
    return indices


def plot_stride(surfaces: Sequence[SurfaceGrid], max_vertices: int) -> int:
    """Choose one topology-preserving stride for the display meshes."""

    if max_vertices < 100:
        raise ValueError("max_plot_vertices must be at least 100.")
    total = sum(surface.vertex_count for surface in surfaces)
    if total <= max_vertices:
        return 1
    return max(2, int(math.ceil(math.sqrt(total / max_vertices))))


def decimated_surface(
    surface: SurfaceGrid,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return six structured coordinate grids with shared boundary-safe sampling."""

    rows, columns = surface.shape
    row_index = inclusive_subsample_indices(rows, stride)
    column_index = inclusive_subsample_indices(columns, stride)
    selection = np.ix_(row_index, column_index)
    return (
        surface.inline_header[selection],
        surface.crossline_header[selection],
        surface.sample_coordinate[selection],
        surface.inline_index[selection],
        surface.profile_index[selection],
        surface.sample_index[selection],
    )


def vertical_label(vertical_unit: str) -> str:
    if vertical_unit == "unspecified":
        return "SEG-Y sample coordinate"
    return f"SEG-Y sample coordinate ({vertical_unit})"


def finite_min_max(array: np.ndarray, name: str) -> tuple[float, float]:
    """Return finite extrema while treating NaNs as an explicit no-data mask."""

    finite = np.asarray(array)[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"{name} does not contain any finite values.")
    return float(finite.min()), float(finite.max())


def save_oblique_plot(
    output_path: Path,
    surfaces: Sequence[SurfaceGrid],
    colors: Sequence[tuple[float, float, float, float]],
    *,
    stride: int,
    vertical_unit: str,
    dpi: int,
) -> None:
    """Save an oblique view with one stable color per reconstructed surface."""

    figure = plt.figure(figsize=(13.5, 9.0))
    axis = figure.add_subplot(111, projection="3d")
    for surface, color in zip(surfaces, colors):
        inline, crossline, sample, _, _, _ = decimated_surface(surface, stride)
        axis.plot_surface(
            inline,
            crossline,
            sample,
            color=color,
            alpha=0.78,
            linewidth=0.0,
            antialiased=True,
            shade=True,
        )
    axis.set_xlabel("Inline header", labelpad=10)
    axis.set_ylabel("Crossline header", labelpad=12)
    axis.set_zlabel(vertical_label(vertical_unit), labelpad=10)
    axis.set_title("Reconstructed 3D fault surfaces", pad=18)
    axis.invert_zaxis()
    axis.view_init(elev=24, azim=-122)
    axis.set_box_aspect((1.2, 2.5, 2.0))
    handles = [
        Patch(facecolor=color, edgecolor="none", label=surface.surface_id)
        for surface, color in zip(surfaces, colors)
    ]
    axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        frameon=False,
    )
    figure.text(
        0.01,
        0.01,
        "Display aspect is optimized for inspection and is not a metric scale.",
        fontsize=8,
        color="0.35",
    )
    figure.subplots_adjust(left=0.02, right=0.80, bottom=0.06, top=0.94)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _surface_boundary(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the outer boundary of a structured grid without duplicate corners."""

    x = np.concatenate(
        (x_grid[0, :], x_grid[1:, -1], x_grid[-1, -2::-1], x_grid[-2:0:-1, 0])
    )
    y = np.concatenate(
        (y_grid[0, :], y_grid[1:, -1], y_grid[-1, -2::-1], y_grid[-2:0:-1, 0])
    )
    return x, y


def save_map_plot(
    output_path: Path,
    surfaces: Sequence[SurfaceGrid],
    colors: Sequence[tuple[float, float, float, float]],
    *,
    stride: int,
    vertical_unit: str,
    dpi: int,
) -> None:
    """Save a top view colored by the mapped SEG-Y sample coordinate."""

    figure, axis = plt.subplots(figsize=(13.0, 6.5))
    z_ranges = [
        finite_min_max(surface.sample_coordinate, surface.surface_id)
        for surface in surfaces
    ]
    z_min = min(values[0] for values in z_ranges)
    z_max = max(values[1] for values in z_ranges)
    normalization = mpl_colors.Normalize(vmin=z_min, vmax=z_max)
    color_map = plt.get_cmap("turbo")
    for surface, outline_color in zip(surfaces, colors):
        inline, crossline, sample, _, _, _ = decimated_surface(surface, stride)
        axis.pcolormesh(
            crossline,
            inline,
            sample,
            cmap=color_map,
            norm=normalization,
            shading="auto",
            alpha=0.62,
            rasterized=True,
        )
        boundary_x, boundary_y = _surface_boundary(crossline, inline)
        axis.plot(boundary_x, boundary_y, color=outline_color, linewidth=1.1)
    scalar_mappable = plt.cm.ScalarMappable(norm=normalization, cmap=color_map)
    colorbar = figure.colorbar(scalar_mappable, ax=axis, pad=0.02)
    colorbar.set_label(vertical_label(vertical_unit))
    axis.set_xlabel("Crossline header")
    axis.set_ylabel("Inline header")
    axis.set_title("Fault-surface map view colored by vertical coordinate")
    axis.grid(color="0.88", linewidth=0.6)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def inline_header_to_index(header: int, geometry: SegyGeometry) -> int:
    matches = np.flatnonzero(geometry.inline_headers == int(header))
    if len(matches) != 1:
        raise ValueError(f"Requested inline {header} is not present exactly once in SEG-Y.")
    return int(matches[0])


def surface_inline_intersection(
    surface: SurfaceGrid,
    inline_index: int,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return ordered fault points exactly represented on one inline grid row."""

    mask = np.isclose(surface.inline_index, float(inline_index), atol=tolerance, rtol=0.0)
    if not mask.any():
        return None
    profile_index = surface.profile_index[mask]
    crossline_header = surface.crossline_header[mask]
    sample_index = surface.sample_index[mask]
    sample_coordinate = surface.sample_coordinate[mask]
    order = np.argsort(profile_index, kind="stable")
    return (
        profile_index[order],
        crossline_header[order],
        sample_index[order],
        sample_coordinate[order],
    )


def write_intersection_csv(
    output_path: Path,
    surfaces: Sequence[SurfaceGrid],
    inline_slices: Sequence[int],
    geometry: SegyGeometry,
) -> dict[int, list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Write all plotted inline intersections and return them for rendering."""

    records: dict[
        int, list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ] = {}
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "inline_header",
                "surface_id",
                "point_order",
                "processed_profile_index",
                "crossline_header",
                "processed_sample_index",
                "sample_coordinate",
            ),
        )
        writer.writeheader()
        for inline_header in inline_slices:
            inline_index = inline_header_to_index(inline_header, geometry)
            inline_records = []
            for surface in surfaces:
                intersection = surface_inline_intersection(surface, inline_index)
                if intersection is None:
                    continue
                profile_index, crossline, sample_index, sample_coordinate = intersection
                inline_records.append(
                    (
                        surface.surface_id,
                        profile_index,
                        crossline,
                        sample_index,
                        sample_coordinate,
                    )
                )
                for point_order in range(len(profile_index)):
                    if not (
                        np.isfinite(sample_index[point_order])
                        and np.isfinite(sample_coordinate[point_order])
                    ):
                        continue
                    writer.writerow(
                        {
                            "inline_header": inline_header,
                            "surface_id": surface.surface_id,
                            "point_order": point_order,
                            "processed_profile_index": f"{profile_index[point_order]:.9g}",
                            "crossline_header": f"{crossline[point_order]:.9g}",
                            "processed_sample_index": f"{sample_index[point_order]:.9g}",
                            "sample_coordinate": f"{sample_coordinate[point_order]:.9g}",
                        }
                    )
            records[int(inline_header)] = inline_records
    return records


def robust_amplitude_limit(section: np.ndarray, percentile: float) -> float:
    finite = np.abs(section[np.isfinite(section)])
    if finite.size == 0:
        raise ValueError("The requested SEG-Y inline contains no finite amplitudes.")
    limit = float(np.percentile(finite, percentile))
    return limit if limit > 0.0 else 1.0


def save_inline_overlays(
    segy_path: Path,
    output_dir: Path,
    inline_slices: Sequence[int],
    intersections: Mapping[
        int, Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ],
    geometry: SegyGeometry,
    color_lookup: Mapping[str, tuple[float, float, float, float]],
    *,
    amplitude_percentile: float,
    display_xline_stride: int,
    display_sample_stride: int,
    vertical_unit: str,
    dpi: int,
    export_seismic_slices: bool,
) -> tuple[list[Path], tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    """Stream requested SEG-Y inlines and overlay fault-surface intersections."""

    if not 50.0 <= amplitude_percentile <= 100.0:
        raise ValueError("amplitude_clip_percentile must lie in [50, 100].")
    if min(display_xline_stride, display_sample_stride) < 1:
        raise ValueError("Seismic display strides must be positive.")

    output_paths: list[Path] = []
    exported_sections: list[np.ndarray] = []
    display_xlines = geometry.crossline_headers[::display_xline_stride]
    display_samples = geometry.sample_coordinates[::display_sample_stride]
    with segyio.open(str(segy_path), "r", strict=True) as segy_file:
        for inline_header in inline_slices:
            section = np.asarray(segy_file.iline[int(inline_header)], dtype=np.float32)
            expected = (
                len(geometry.crossline_headers),
                len(geometry.sample_coordinates),
            )
            if section.shape != expected:
                raise ValueError(
                    f"Inline {inline_header} has shape {section.shape}; expected {expected}."
                )
            displayed = np.ascontiguousarray(
                section[::display_xline_stride, ::display_sample_stride]
            )
            amplitude_limit = robust_amplitude_limit(displayed, amplitude_percentile)

            figure, axis = plt.subplots(figsize=(13.5, 7.5))
            image = axis.imshow(
                displayed.T,
                cmap="gray",
                vmin=-amplitude_limit,
                vmax=amplitude_limit,
                aspect="auto",
                interpolation="nearest",
                origin="upper",
                extent=(
                    float(display_xlines[0]),
                    float(display_xlines[-1]),
                    float(display_samples[-1]),
                    float(display_samples[0]),
                ),
            )
            plotted_ids: list[str] = []
            for (
                surface_id,
                _profile_index,
                crossline_header,
                _sample_index,
                sample_coordinate,
            ) in intersections[int(inline_header)]:
                if not np.isfinite(sample_coordinate).any():
                    continue
                axis.plot(
                    crossline_header,
                    sample_coordinate,
                    color=color_lookup[surface_id],
                    linewidth=1.8,
                    alpha=0.95,
                    label=surface_id,
                )
                plotted_ids.append(surface_id)
            axis.set_xlim(display_xlines[0], display_xlines[-1])
            axis.set_ylim(display_samples[-1], display_samples[0])
            axis.set_xlabel("Crossline header")
            axis.set_ylabel(vertical_label(vertical_unit))
            axis.set_title(
                f"SEG-Y inline {inline_header} with reconstructed fault intersections"
            )
            colorbar = figure.colorbar(image, ax=axis, pad=0.015)
            colorbar.set_label("Seismic amplitude")
            if plotted_ids:
                handles, labels = axis.get_legend_handles_labels()
                unique = dict(zip(labels, handles))
                axis.legend(
                    unique.values(),
                    unique.keys(),
                    loc="upper left",
                    bbox_to_anchor=(1.09, 1.0),
                    fontsize=8,
                    frameon=False,
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No surface has a node on this inline",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )
            output_path = output_dir / f"inline_{int(inline_header):04d}_overlay.png"
            figure.subplots_adjust(left=0.08, right=0.78, bottom=0.10, top=0.92)
            figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(figure)
            output_paths.append(output_path)
            if export_seismic_slices:
                exported_sections.append(displayed)
            del section, displayed

    if not export_seismic_slices:
        return output_paths, None
    return (
        output_paths,
        (
            np.asarray(inline_slices, dtype=np.int64),
            np.stack(exported_sections).astype(np.float32, copy=False),
            np.asarray((display_xline_stride, display_sample_stride), dtype=np.int64),
        ),
    )


def save_normalized_npz(
    output_path: Path,
    surfaces: Sequence[SurfaceGrid],
    *,
    crossline_stride: int,
    sample_stride: int,
    vertical_unit: str,
    z_field: str,
) -> None:
    """Save full-resolution surfaces in a safe, flattened, non-object schema."""

    surface_ids = np.asarray([surface.surface_id for surface in surfaces])
    processed_parts: list[np.ndarray] = []
    display_parts: list[np.ndarray] = []
    original_sample_parts: list[np.ndarray] = []
    original_coordinate_parts: list[np.ndarray] = []
    vertex_valid_parts: list[np.ndarray] = []
    face_parts: list[np.ndarray] = []
    observation_rmse: list[float] = []
    vertex_offsets = [0]
    face_offsets = [0]
    grid_shapes: list[tuple[int, int]] = []
    for surface in surfaces:
        vertex_offset = vertex_offsets[-1]
        processed_parts.append(
            np.column_stack(
                (
                    surface.inline_index.ravel(),
                    surface.profile_index.ravel(),
                    surface.sample_index.ravel(),
                )
            ).astype(np.float32)
        )
        display_parts.append(
            np.column_stack(
                (
                    surface.inline_header.ravel(),
                    surface.crossline_header.ravel(),
                    surface.sample_coordinate.ravel(),
                )
            ).astype(np.float32)
        )
        vertex_valid_parts.append(np.isfinite(surface.sample_index).ravel())
        if surface.original_sample_index is None:
            original_sample_parts.append(
                np.full(surface.vertex_count, np.nan, dtype=np.float32)
            )
            original_coordinate_parts.append(
                np.full(surface.vertex_count, np.nan, dtype=np.float32)
            )
        else:
            original_sample_parts.append(
                surface.original_sample_index.ravel().astype(np.float32)
            )
            original_coordinate_parts.append(
                surface.original_sample_coordinate.ravel().astype(np.float32)
            )
        if surface.faces_local is None:
            faces = structured_faces(*surface.shape, vertex_offset=vertex_offset)
        else:
            faces = surface.faces_local.astype(np.int64, copy=False) + vertex_offset
        face_parts.append(faces)
        observation_rmse.append(
            float(surface.observation_rmse)
            if surface.observation_rmse is not None
            else float("nan")
        )
        vertex_offsets.append(vertex_offset + surface.vertex_count)
        face_offsets.append(face_offsets[-1] + len(faces))
        grid_shapes.append(surface.shape)

    np.savez_compressed(
        output_path,
        schema_version=np.asarray([SCHEMA_VERSION]),
        processed_columns=np.asarray(
            ["inline_index", "processed_profile_index", "processed_sample_index"]
        ),
        display_columns=np.asarray(
            ["inline_header", "crossline_header", "sample_coordinate"]
        ),
        crossline_stride=np.asarray([crossline_stride], dtype=np.int64),
        sample_stride=np.asarray([sample_stride], dtype=np.int64),
        vertical_unit=np.asarray([vertical_unit]),
        source_z_field=np.asarray([z_field]),
        surface_ids=surface_ids,
        vertices_processed=np.concatenate(processed_parts, axis=0),
        vertices_display=np.concatenate(display_parts, axis=0),
        original_sample_index=np.concatenate(original_sample_parts),
        original_sample_coordinate=np.concatenate(original_coordinate_parts),
        vertex_valid=np.concatenate(vertex_valid_parts).astype(np.bool_),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        faces=np.concatenate(face_parts, axis=0),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        grid_shapes=np.asarray(grid_shapes, dtype=np.int64),
        observation_rmse=np.asarray(observation_rmse, dtype=np.float64),
    )


def optional_range(array: np.ndarray | None) -> tuple[str, str]:
    if array is None:
        return "", ""
    minimum, maximum = finite_min_max(array, "optional surface field")
    return f"{minimum:.9g}", f"{maximum:.9g}"


def write_inventory_csv(
    output_path: Path,
    surfaces: Sequence[SurfaceGrid],
    *,
    input_format: str,
    z_field: str,
) -> None:
    """Write one concise, full-resolution record per surface."""

    fieldnames = (
        "surface_id",
        "input_format",
        "z_field",
        "grid_rows",
        "grid_columns",
        "vertex_count",
        "valid_vertex_count",
        "no_data_vertex_count",
        "triangle_count",
        "grid_z_org_available",
        "inline_index_min",
        "inline_index_max",
        "profile_index_min",
        "profile_index_max",
        "sample_index_min",
        "sample_index_max",
        "inline_header_min",
        "inline_header_max",
        "crossline_header_min",
        "crossline_header_max",
        "sample_coordinate_min",
        "sample_coordinate_max",
        "original_sample_index_min",
        "original_sample_index_max",
    )
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for surface in surfaces:
            original_min, original_max = optional_range(surface.original_sample_index)
            sample_min, sample_max = finite_min_max(
                surface.sample_index, f"{surface.surface_id} sample index"
            )
            coordinate_min, coordinate_max = finite_min_max(
                surface.sample_coordinate,
                f"{surface.surface_id} sample coordinate",
            )
            rows, columns = surface.shape
            writer.writerow(
                {
                    "surface_id": surface.surface_id,
                    "input_format": input_format,
                    "z_field": z_field,
                    "grid_rows": rows,
                    "grid_columns": columns,
                    "vertex_count": surface.vertex_count,
                    "valid_vertex_count": surface.valid_vertex_count,
                    "no_data_vertex_count": surface.invalid_vertex_count,
                    "triangle_count": surface.face_count,
                    "grid_z_org_available": surface.original_sample_index is not None,
                    "inline_index_min": f"{surface.inline_index.min():.9g}",
                    "inline_index_max": f"{surface.inline_index.max():.9g}",
                    "profile_index_min": f"{surface.profile_index.min():.9g}",
                    "profile_index_max": f"{surface.profile_index.max():.9g}",
                    "sample_index_min": f"{sample_min:.9g}",
                    "sample_index_max": f"{sample_max:.9g}",
                    "inline_header_min": f"{surface.inline_header.min():.9g}",
                    "inline_header_max": f"{surface.inline_header.max():.9g}",
                    "crossline_header_min": f"{surface.crossline_header.min():.9g}",
                    "crossline_header_max": f"{surface.crossline_header.max():.9g}",
                    "sample_coordinate_min": f"{coordinate_min:.9g}",
                    "sample_coordinate_max": f"{coordinate_max:.9g}",
                    "original_sample_index_min": original_min,
                    "original_sample_index_max": original_max,
                }
            )


def prepare_output_dir(
    output_dir: Path,
    inline_slices: Sequence[int],
    *,
    export_seismic_slices: bool,
    overwrite: bool,
) -> None:
    """Refuse accidental replacement unless the user explicitly allows it."""

    expected = set(OUTPUT_BASENAMES)
    if not export_seismic_slices:
        expected.discard("seismic_slices.npz")
    expected.update(f"inline_{int(value):04d}_overlay.png" for value in inline_slices)
    if output_dir.is_dir():
        unexpected = [path for path in output_dir.iterdir() if path.name not in expected]
        if unexpected:
            names = ", ".join(path.name for path in sorted(unexpected)[:5])
            raise FileExistsError(
                f"Output directory contains unrelated files: {names}. "
                "Choose a dedicated output directory."
            )
    existing = [output_dir / name for name in sorted(expected) if (output_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing[:5])
        raise FileExistsError(
            f"Output files already exist in {output_dir}: {names}. Use --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def file_record(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
    }
    record["sha256"] = sha256_file(path) if include_hash else None
    return record


def geometry_record(geometry: SegyGeometry) -> dict[str, Any]:
    return {
        "trace_count": geometry.trace_count,
        "sorting": geometry.sorting,
        "format_code": geometry.format_code,
        "inline": {
            "count": len(geometry.inline_headers),
            "minimum": int(geometry.inline_headers[0]),
            "maximum": int(geometry.inline_headers[-1]),
            "regular_step": axis_step(geometry.inline_headers),
        },
        "crossline": {
            "count": len(geometry.crossline_headers),
            "minimum": int(geometry.crossline_headers[0]),
            "maximum": int(geometry.crossline_headers[-1]),
            "regular_step": axis_step(geometry.crossline_headers),
        },
        "sample_coordinate": {
            "count": len(geometry.sample_coordinates),
            "minimum": float(geometry.sample_coordinates[0]),
            "maximum": float(geometry.sample_coordinates[-1]),
            "regular_step": axis_step(geometry.sample_coordinates),
        },
    }


def write_metadata(
    output_path: Path,
    *,
    input_path: Path,
    segy_path: Path,
    input_format: str,
    geometry: SegyGeometry,
    surfaces: Sequence[SurfaceGrid],
    output_files: Sequence[Path],
    arguments: argparse.Namespace,
    effective_plot_stride: int,
    reconstruction_metadata: Mapping[str, Any] | None,
    reconstruction_metadata_path: Path | None,
) -> None:
    """Write provenance, coordinate mapping, warnings, and output hashes."""

    warnings: list[str] = []
    if input_format == "legacy-grid-pickle":
        warnings.append(
            "The legacy pickle has no sidecar metadata. Its grid-to-SEG-Y mapping "
            "is an explicit command-line assertion recorded in this file."
        )
        warnings.append(
            "This artifact contains already gridded surfaces and was not rerun "
            "through the current 3D linking/reconstruction module."
        )
        if any(surface.original_sample_index is None for surface in surfaces):
            warnings.append(
                "At least one legacy surface does not contain grid_z_org; its "
                "original_sample arrays are exported as NaN."
            )
    elif reconstruction_metadata is None:
        warnings.append(
            "Canonical processing strides were supplied explicitly because no "
            "usable artifact-coordinate mapping was available in run metadata."
        )
    if arguments.vertical_unit == "unspecified":
        warnings.append(
            "No time/depth unit was asserted; plots label the vertical axis as "
            "SEG-Y sample coordinate."
        )
    no_data_vertices = sum(surface.invalid_vertex_count for surface in surfaces)
    if no_data_vertices:
        warnings.append(
            f"The input contains {no_data_vertices} NaN no-data surface nodes. "
            "They were preserved as holes and triangles touching them were omitted."
        )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "3D fault-surface visualization and normalized export",
        "input": {
            "fault_surfaces": file_record(input_path),
            "segy": file_record(
                segy_path, include_hash=not arguments.skip_segy_hash
            ),
            "reconstruction_run_metadata": (
                file_record(reconstruction_metadata_path)
                if reconstruction_metadata_path is not None
                else None
            ),
            "detected_format": input_format,
            "legacy_pickle_acknowledged": bool(arguments.allow_legacy_pickle),
            "canonical_metadata_verified": reconstruction_metadata is not None,
        },
        "segy_geometry": geometry_record(geometry),
        "coordinate_mapping": {
            "normalized_processed_axis_order": [
                "inline_index",
                "processed_profile_index",
                "processed_sample_index",
            ],
            "display_axis_order": [
                "inline_header",
                "crossline_header",
                "sample_coordinate",
            ],
            "legacy_grid_x": arguments.legacy_grid_x,
            "inline_origin": arguments.inline_origin,
            "crossline_stride": arguments.crossline_stride,
            "sample_stride": arguments.sample_stride,
            "z_field": arguments.z_field if input_format == "legacy-grid-pickle" else None,
            "vertical_unit": arguments.vertical_unit,
            "fractional_index_mapping": "linear interpolation on each SEG-Y axis",
            "no_data_policy": (
                "preserve NaN grid_z nodes, export vertex_valid, and omit triangles "
                "touching invalid nodes"
            ),
        },
        "visualization": {
            "inline_slices": list(arguments.inline_slices),
            "amplitude_clip_percentile": arguments.amplitude_clip_percentile,
            "seismic_xline_display_stride": arguments.seismic_xline_display_stride,
            "seismic_sample_display_stride": arguments.seismic_sample_display_stride,
            "max_plot_vertices": arguments.max_plot_vertices,
            "effective_structured_plot_stride": effective_plot_stride,
            "oblique_box_aspect": [1.2, 2.5, 2.0],
            "oblique_box_aspect_is_metric": False,
            "dpi": arguments.dpi,
        },
        "counts": {
            "surfaces": len(surfaces),
            "vertices": sum(surface.vertex_count for surface in surfaces),
            "valid_vertices": sum(
                surface.valid_vertex_count for surface in surfaces
            ),
            "no_data_vertices": no_data_vertices,
            "triangles": sum(surface.face_count for surface in surfaces),
            "inline_intersection_points": sum(
                int(np.isfinite(intersection[3]).sum())
                for surface in surfaces
                for inline_header in arguments.inline_slices
                if (
                    intersection := surface_inline_intersection(
                        surface, inline_header_to_index(inline_header, geometry)
                    )
                )
                is not None
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "segyio": getattr(segyio, "__version__", "unknown"),
        },
        "warnings": warnings,
        "outputs": {
            path.name: file_record(path) for path in sorted(output_files)
        },
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2, sort_keys=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize and visualize current or legacy structured 3D fault "
            "surfaces against selected SEG-Y inline sections."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fault_surfaces", help="Legacy grid pickle or canonical NPZ.")
    parser.add_argument("--segy", required=True, help="3D SEG-Y volume used as context.")
    parser.add_argument(
        "--input-format",
        choices=("auto", "legacy-grid-pickle", "canonical-npz"),
        default="auto",
    )
    parser.add_argument(
        "--run-metadata",
        help=(
            "run_metadata.json paired with canonical NPZ input; when omitted, "
            "the NPZ directory is searched automatically."
        ),
    )
    parser.add_argument(
        "--allow-legacy-pickle",
        action="store_true",
        help=(
            "Acknowledge loading the legacy pickle. A restricted NumPy-only "
            "unpickler is still used."
        ),
    )
    parser.add_argument(
        "--legacy-grid-x",
        choices=("offset", "absolute"),
        default="offset",
        help="Interpret legacy grid_x as an inline offset or an absolute header.",
    )
    parser.add_argument(
        "--inline-origin",
        type=int,
        default=None,
        help="Required SEG-Y inline corresponding to legacy grid_x == 0.",
    )
    parser.add_argument(
        "--crossline-stride",
        type=int,
        default=None,
        help="Required for legacy input; verified against canonical run metadata.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=None,
        help="Required for legacy input; verified against canonical run metadata.",
    )
    parser.add_argument(
        "--z-field",
        choices=("grid_z", "grid_z_org"),
        default="grid_z",
        help="Legacy vertical surface field; grid_z is the final smoothed result.",
    )
    parser.add_argument(
        "--vertical-unit",
        choices=("unspecified", "ms", "s", "m"),
        default="unspecified",
        help="Known unit of SEG-Y sample coordinates; no conversion is performed.",
    )
    parser.add_argument(
        "--inline-slices",
        type=int,
        nargs="+",
        default=None,
        help="SEG-Y inline headers for seismic overlays; defaults to first/middle/last.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/visualization/fault3d_display",
    )
    parser.add_argument("--max-plot-vertices", type=int, default=100_000)
    parser.add_argument("--amplitude-clip-percentile", type=float, default=99.0)
    parser.add_argument("--seismic-xline-display-stride", type=int, default=2)
    parser.add_argument("--seismic-sample-display-stride", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--export-seismic-slices",
        action="store_true",
        help="Also save the downsampled inline amplitudes in a non-object NPZ.",
    )
    parser.add_argument(
        "--skip-segy-hash",
        action="store_true",
        help="Skip the potentially slow SHA-256 calculation for the large SEG-Y file.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def run(arguments: argparse.Namespace) -> list[Path]:
    input_path = Path(arguments.fault_surfaces).expanduser().resolve()
    segy_path = Path(arguments.segy).expanduser().resolve()
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Fault-surface input not found: {input_path}")
    if not segy_path.is_file():
        raise FileNotFoundError(f"SEG-Y input not found: {segy_path}")
    input_format = (
        detect_input_format(input_path)
        if arguments.input_format == "auto"
        else arguments.input_format
    )
    if input_format == "legacy-grid-pickle" and not arguments.allow_legacy_pickle:
        raise ValueError(
            "Legacy pickle loading requires --allow-legacy-pickle. "
            "Only use it for a trusted local artifact."
        )
    (
        arguments.crossline_stride,
        arguments.sample_stride,
        reconstruction_metadata,
        reconstruction_metadata_path,
    ) = resolve_processing_strides(
        input_format=input_format,
        input_path=input_path,
        run_metadata_path=arguments.run_metadata,
        crossline_stride=arguments.crossline_stride,
        sample_stride=arguments.sample_stride,
    )
    if (
        input_format == "legacy-grid-pickle"
        and arguments.legacy_grid_x == "offset"
        and arguments.inline_origin is None
    ):
        raise ValueError(
            "Legacy offset grid_x mapping requires an explicit --inline-origin."
        )
    if min(arguments.crossline_stride, arguments.sample_stride) < 1:
        raise ValueError("Processing strides must be positive.")
    if arguments.dpi < 50:
        raise ValueError("dpi must be at least 50.")
    if arguments.max_plot_vertices < 100:
        raise ValueError("max_plot_vertices must be at least 100.")
    if min(
        arguments.seismic_xline_display_stride,
        arguments.seismic_sample_display_stride,
    ) < 1:
        raise ValueError("Seismic display strides must be positive.")
    if not 50.0 <= arguments.amplitude_clip_percentile <= 100.0:
        raise ValueError("amplitude_clip_percentile must lie in [50, 100].")

    LOGGER.info("Reading SEG-Y geometry: %s", segy_path)
    geometry = read_segy_geometry(segy_path)
    if arguments.inline_slices is None:
        arguments.inline_slices = [
            int(geometry.inline_headers[0]),
            int(geometry.inline_headers[len(geometry.inline_headers) // 2]),
            int(geometry.inline_headers[-1]),
        ]
    arguments.inline_slices = list(dict.fromkeys(arguments.inline_slices))
    for inline_header in arguments.inline_slices:
        inline_header_to_index(inline_header, geometry)
    prepare_output_dir(
        output_dir,
        arguments.inline_slices,
        export_seismic_slices=arguments.export_seismic_slices,
        overwrite=arguments.overwrite,
    )

    LOGGER.info("Loading %s fault surfaces: %s", input_format, input_path)
    if input_format == "legacy-grid-pickle":
        surfaces = load_legacy_surface_pickle(
            input_path,
            geometry,
            inline_origin=int(arguments.inline_origin or 0),
            legacy_grid_x=arguments.legacy_grid_x,
            crossline_stride=arguments.crossline_stride,
            sample_stride=arguments.sample_stride,
            z_field=arguments.z_field,
        )
    else:
        surfaces = load_canonical_npz(
            input_path,
            geometry,
            crossline_stride=arguments.crossline_stride,
            sample_stride=arguments.sample_stride,
        )
    if not surfaces:
        raise ValueError("No fault surfaces were loaded.")
    LOGGER.info(
        "Validated %d surfaces, %d/%d finite vertices, and %d triangles.",
        len(surfaces),
        sum(surface.valid_vertex_count for surface in surfaces),
        sum(surface.vertex_count for surface in surfaces),
        sum(surface.face_count for surface in surfaces),
    )

    colors = stable_colors(len(surfaces))
    color_lookup = {
        surface.surface_id: color for surface, color in zip(surfaces, colors)
    }
    effective_plot_stride = plot_stride(surfaces, arguments.max_plot_vertices)
    outputs: list[Path] = []

    normalized_path = output_dir / "surfaces_normalized.npz"
    save_normalized_npz(
        normalized_path,
        surfaces,
        crossline_stride=arguments.crossline_stride,
        sample_stride=arguments.sample_stride,
        vertical_unit=arguments.vertical_unit,
        z_field=(
            arguments.z_field
            if input_format == "legacy-grid-pickle"
            else "canonical vertices[:,2]"
        ),
    )
    outputs.append(normalized_path)

    inventory_path = output_dir / "surface_inventory.csv"
    write_inventory_csv(
        inventory_path,
        surfaces,
        input_format=input_format,
        z_field=arguments.z_field if input_format == "legacy-grid-pickle" else "vertices[:,2]",
    )
    outputs.append(inventory_path)

    intersections_path = output_dir / "inline_intersections.csv"
    intersections = write_intersection_csv(
        intersections_path, surfaces, arguments.inline_slices, geometry
    )
    outputs.append(intersections_path)

    oblique_path = output_dir / "fault_surfaces_oblique.png"
    LOGGER.info("Rendering oblique surface view.")
    save_oblique_plot(
        oblique_path,
        surfaces,
        colors,
        stride=effective_plot_stride,
        vertical_unit=arguments.vertical_unit,
        dpi=arguments.dpi,
    )
    outputs.append(oblique_path)

    map_path = output_dir / "fault_surfaces_map_view.png"
    LOGGER.info("Rendering fault-surface map view.")
    save_map_plot(
        map_path,
        surfaces,
        colors,
        stride=effective_plot_stride,
        vertical_unit=arguments.vertical_unit,
        dpi=arguments.dpi,
    )
    outputs.append(map_path)

    LOGGER.info("Streaming %d SEG-Y inline overlays.", len(arguments.inline_slices))
    overlay_paths, exported = save_inline_overlays(
        segy_path,
        output_dir,
        arguments.inline_slices,
        intersections,
        geometry,
        color_lookup,
        amplitude_percentile=arguments.amplitude_clip_percentile,
        display_xline_stride=arguments.seismic_xline_display_stride,
        display_sample_stride=arguments.seismic_sample_display_stride,
        vertical_unit=arguments.vertical_unit,
        dpi=arguments.dpi,
        export_seismic_slices=arguments.export_seismic_slices,
    )
    outputs.extend(overlay_paths)
    if exported is not None:
        inline_headers, amplitudes, display_strides = exported
        seismic_slices_path = output_dir / "seismic_slices.npz"
        np.savez_compressed(
            seismic_slices_path,
            inline_headers=inline_headers,
            crossline_headers=geometry.crossline_headers[
                :: int(display_strides[0])
            ],
            sample_coordinates=geometry.sample_coordinates[
                :: int(display_strides[1])
            ],
            amplitudes=amplitudes,
            display_strides=display_strides,
        )
        outputs.append(seismic_slices_path)

    metadata_path = output_dir / "visualization_metadata.json"
    LOGGER.info("Writing hashes and provenance metadata.")
    write_metadata(
        metadata_path,
        input_path=input_path,
        segy_path=segy_path,
        input_format=input_format,
        geometry=geometry,
        surfaces=surfaces,
        output_files=outputs,
        arguments=arguments,
        effective_plot_stride=effective_plot_stride,
        reconstruction_metadata=reconstruction_metadata,
        reconstruction_metadata_path=reconstruction_metadata_path,
    )
    outputs.append(metadata_path)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(levelname)s %(message)s",
    )
    try:
        outputs = run(arguments)
    except (FileNotFoundError, OSError, TypeError, ValueError, pickle.UnpicklingError) as error:
        parser.error(str(error))
    LOGGER.info("Created %d output files in %s", len(outputs), Path(arguments.output_dir).resolve())
    for output in outputs:
        LOGGER.info("  %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
