#!/usr/bin/env python3
"""
Real seismic thrust-fault label construction for transfer learning.

This script corresponds to the notebook:

    real_seismic_label_construction_paper_style.ipynb

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
8. Split patches into train/validation.
9. Save seismic patches, label patches, metadata, and preview figures.

Quick example for one interpreted line
--------------------------------------
python construct_real_seismic_labels.py \
    --segy /path/to/774.segy \
    --csv /path/to/774.csv \
    --inline-id 774 \
    --x-offset 800 \
    --z-offset 664 \
    --x-scale 1 \
    --z-scale 4 \
    --label-shape 3201 1410 \
    --downsample-x 2 \
    --downsample-z 2 \
    --out ./real_thrust_fault_labels_paper_style \
    --preview

Multiple-line example
---------------------
Create a JSON file:

[
  {
    "inline_id": 30,
    "segy_path": "/path/to/30.segy",
    "csv_path": "/path/to/30.csv",
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
    "segy_path": "/path/to/230.segy",
    "csv_path": "/path/to/230.csv",
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

python construct_real_seismic_labels.py \
    --config-json line_inputs.json \
    --out ./real_thrust_fault_labels_paper_style \
    --preview
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

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
    val_fraction_if_small: float = 0.2

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
    """
    if not HAS_SEGYIO:
        raise ImportError(
            "segyio is required to read SEGY files. Install it with: pip install segyio"
        )

    segy_path = Path(segy_path)
    if not segy_path.exists():
        raise FileNotFoundError(f"SEGY file not found: {segy_path}")

    with segyio.open(str(segy_path), "r", ignore_geometry=False) as f:
        traces = np.asarray([np.asarray(trace, dtype=np.float32) for trace in f.trace])
        sample_count = traces.shape[1]
        trace_count = traces.shape[0]

        try:
            inlines = np.asarray(segyio.tools.collect(f.ilines))
            xlines = np.asarray(segyio.tools.collect(f.xlines))
            inline_count = len(inlines)
            xline_count = len(xlines)
        except Exception:
            inline_count = 0
            xline_count = 0

        if inline_count > 0 and xline_count > 0 and trace_count == inline_count * xline_count:
            data_3d = traces.reshape((inline_count, xline_count, sample_count))
            section_index = int(np.clip(section_index, 0, inline_count - 1))
            section = data_3d[section_index, :, :]
            print(f"Read as 3D SEGY: {data_3d.shape}; selected section index {section_index}")
        else:
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
    Shuffle and split patch records into train and validation sets.

    If enough patches exist, use target_train and target_val. Otherwise use a
    fallback ratio split.
    """
    rng = np.random.default_rng(label_cfg.seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    records = [records[i] for i in indices]

    total_target = label_cfg.target_train + label_cfg.target_val
    if len(records) >= total_target:
        train = records[: label_cfg.target_train]
        val = records[
            label_cfg.target_train : label_cfg.target_train + label_cfg.target_val
        ]
    else:
        n_val = (
            max(1, int(round(len(records) * label_cfg.val_fraction_if_small)))
            if len(records) > 1
            else 0
        )
        val = records[:n_val]
        train = records[n_val:]

    return train, val


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


def save_records(
    records: List[Dict],
    out_dir: str | Path,
    split: str,
    save_npy: bool = True,
    save_dat: bool = False,
    preview: bool = False,
) -> None:
    """
    Save seismic patches, labels, metadata, and optional preview figures.
    """
    out_dir = Path(out_dir)
    seis_dir = out_dir / split / "seismic"
    label_dir = out_dir / split / "labels"
    meta_dir = out_dir / split / "metadata"
    fig_dir = out_dir / split / "preview"

    for d in [seis_dir, label_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if preview:
        fig_dir.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(records):
        stem = f"{split}_{i:06d}"

        image = rec["image"].astype(np.float32)
        label = rec["label"].astype(np.uint8)

        if save_npy:
            np.save(seis_dir / f"{stem}.npy", image)
            np.save(label_dir / f"{stem}.npy", label)

        if save_dat:
            image.tofile(seis_dir / f"{stem}.dat")
            label.tofile(label_dir / f"{stem}.dat")

        metadata = {k: v for k, v in rec.items() if k not in ["image", "label"]}
        with open(meta_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if preview:
            save_preview_png(fig_dir / f"{stem}.png", image, label)


def process_one_line(
    line_cfg: LineInputConfig,
    label_cfg: LabelPatchConfig,
    connect_by_fault_id: bool = True,
    connect_without_fault_id: bool = False,
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

    for p in patches:
        p["inline_id"] = int(line_cfg.inline_id)

    metadata = {
        "line_config": asdict(line_cfg),
        "label_metadata": label_meta,
        "seismic_shape": list(seismic.shape),
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

    configs: List[LineInputConfig] = []
    for item in raw:
        if "label_shape_before_downsample" in item and item["label_shape_before_downsample"] is not None:
            item["label_shape_before_downsample"] = tuple(item["label_shape_before_downsample"])
        configs.append(LineInputConfig(**item))

    return configs


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


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser.
    """
    parser = argparse.ArgumentParser(
        description="Construct real seismic thrust-fault labels and crop 256x256 patches."
    )

    # Either provide a config JSON or a single-line set of paths.
    parser.add_argument("--config-json", type=str, default=None, help="JSON file with one or more line configs.")

    parser.add_argument("--segy", type=str, default=None, help="Path to SEGY file for single-line mode.")
    parser.add_argument("--csv", type=str, default=None, help="Path to manual interpretation CSV for single-line mode.")
    parser.add_argument("--inline-id", type=int, default=None, help="Inline ID for single-line mode.")
    parser.add_argument("--section-index", type=int, default=0)

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
    parser.add_argument("--include-background", action="store_true", help="Keep patches without fault labels.")

    parser.add_argument("--target-train", type=int, default=200)
    parser.add_argument("--target-val", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--save-dat", action="store_true", help="Also save patches as raw .dat files.")
    parser.add_argument("--preview", action="store_true", help="Save seismic-label overlay PNGs.")

    parser.add_argument("--connect-without-fault-id", action="store_true", help="Connect all points even without a fault_id column.")
    parser.add_argument("--no-connect-by-fault-id", action="store_true", help="Do not connect points even when a fault_id column exists.")

    parser.add_argument("--out", type=str, default="./real_thrust_fault_labels_paper_style")

    return parser


def main() -> None:
    """
    Command-line entry point.
    """
    args = build_arg_parser().parse_args()

    if args.config_json is not None:
        line_inputs = load_line_inputs_from_json(args.config_json)
    else:
        line_inputs = [make_single_line_input_from_args(args)]

    label_cfg = LabelPatchConfig(
        dilation_a=args.dilation_a,
        dilation_c=args.dilation_c,
        patch_size=args.patch_size,
        stride=args.stride,
        positive_only=not args.include_background,
        min_label_pixels=args.min_label_pixels,
        target_train=args.target_train,
        target_val=args.target_val,
        save_npy=True,
        save_dat=args.save_dat,
        seed=args.seed,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_patches: List[Dict] = []
    all_line_metadata: List[Dict] = []

    for line_cfg in line_inputs:
        patches, metadata = process_one_line(
            line_cfg,
            label_cfg,
            connect_by_fault_id=not args.no_connect_by_fault_id,
            connect_without_fault_id=args.connect_without_fault_id,
        )
        all_patches.extend(patches)
        all_line_metadata.append(metadata)
        print(f"Inline {line_cfg.inline_id}: {len(patches)} patches")

    if len(all_patches) == 0:
        raise RuntimeError(
            "No patches were generated. Check coordinate transforms, label shape, "
            "patch size, and whether positive_only filtering removed all patches."
        )

    train_records, val_records = split_records(all_patches, label_cfg)
    print(f"Total patches: {len(all_patches)}")
    print(f"Train patches: {len(train_records)}")
    print(f"Validation patches: {len(val_records)}")

    save_records(
        train_records,
        out_dir,
        split="train",
        save_npy=label_cfg.save_npy,
        save_dat=label_cfg.save_dat,
        preview=args.preview,
    )
    save_records(
        val_records,
        out_dir,
        split="val",
        save_npy=label_cfg.save_npy,
        save_dat=label_cfg.save_dat,
        preview=args.preview,
    )

    with open(out_dir / "line_metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_line_metadata, f, indent=2)

    with open(out_dir / "label_patch_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(label_cfg), f, indent=2)

    print(f"Saved dataset to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
