# Configurations

Store machine-readable JSON configurations for data construction, inference,
2D post-processing, and 3D reconstruction here.

Relative paths inside a real-line configuration are interpreted from the
directory in which the command is launched. Run commands from the repository
root and keep embedded paths relative to the repository root.

`configs/real_line_inputs.json` is a JSON array with one object per interpreted
line. A reproducible configuration may assign each complete line explicitly to
the training or validation pool:

```json
[
  {
    "inline_id": 30,
    "segy_path": "data/segy/30.segy",
    "csv_path": "data/interpretations/30.csv",
    "split": "train",
    "section_index": 0,
    "x_col": 0,
    "z_col": 1,
    "fault_id_col": 2,
    "x_offset": 800,
    "z_offset": 664,
    "x_scale": 1,
    "z_scale": 4,
    "label_shape_before_downsample": [3201, 1410],
    "downsample_x": 2,
    "downsample_z": 2
  },
  {
    "inline_id": 830,
    "segy_path": "data/segy/830.segy",
    "csv_path": "data/interpretations/830.csv",
    "split": "val",
    "section_index": 0,
    "x_col": 0,
    "z_col": 1,
    "fault_id_col": 2,
    "x_offset": 800,
    "z_offset": 664,
    "x_scale": 1,
    "z_scale": 4,
    "label_shape_before_downsample": [3201, 1410],
    "downsample_x": 2,
    "downsample_z": 2
  }
]
```

The numeric coordinate values above only demonstrate the schema; replace them
with the verified values for each line. `section_index` is a strict zero-based
index for a 3D SEG-Y volume and must be `0` for a 2D SEG-Y line; out-of-range
values are rejected rather than clipped. `split` accepts only `train` or `val`.
When explicit assignments are used, assign every configured line so that the
dataset has no ambiguous partial split. Do not list the same physical SEG-Y
section more than once; content checksums and section indices are used to reject
duplicate source configurations before patch construction.

If `split` is omitted, the default remains group-safe: complete inlines, rather
than individual overlapping patches, are assigned to the two pools using the
configured seed. If only one inline produces usable patches, the builder
automatically falls back to a contiguous spatial validation region separated
from training by a guard. Boundary-crossing and guard-region patches are
discarded. `target_train` and `target_val` are upper limits within the already
separated pools, not instructions to move patches across the boundary.

Run it from the repository root with:

```bash
python real_seismic_label_construction.py \
  --config-json configs/real_line_inputs.json \
  --out outputs/datasets/real_labels
```

The output directory must be new or contain only its placeholder `.gitignore`.
Successful runs create `patch_manifest.csv` for per-patch split, coordinate,
path, and checksum records, plus `dataset_manifest.json` for source checksums,
the resolved split strategy, parameters, and actual counts.

Absolute local paths remain supported, but do not commit machine-specific
paths. Commit a publication configuration only when its paths are portable and
its source files, coordinate transforms, split assignments, and checksums have
been verified.
