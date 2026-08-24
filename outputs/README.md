# Generated outputs

This directory is reserved for new local runs. Generated files are ignored by
Git; the committed reference products remain under `demo/`.

```text
outputs/
├── datasets/
│   ├── synthetic/
│   └── real_labels/
├── training/
├── fault2d/
├── fault3d/
└── visualization/
```

Both `datasets/synthetic/` and `datasets/real_labels/` use the canonical
`(x, z)` seismic/label array order. Synthetic datasets generated before this
contract was introduced used `(z, x)` and must be regenerated rather than
mixed with current files.

Create every generated dataset in a new directory. A dataset target may contain
its committed `.gitignore`, but it must not contain an earlier generated
payload. The generators refuse non-empty payload directories; this prevents
stale samples from an earlier axis convention or train/validation split from
remaining visible to the training loader.

A real-label dataset has this additional provenance contract:

```text
outputs/datasets/real_labels/
├── dataset_manifest.json
├── patch_manifest.csv
├── line_metadata.json
├── label_patch_config.json
├── train/
│   ├── seismic/
│   ├── labels/
│   └── metadata/
└── val/
    ├── seismic/
    ├── labels/
    └── metadata/
```

By default, the real-label builder assigns whole inlines to training or
validation. A configuration can make those assignments explicit with
`"split": "train"` or `"split": "val"` on every line. With only one usable
inline, it instead uses a contiguous spatial validation region and removes
patches crossing the configured guard. The `target_train` and `target_val`
values cap the two safe pools independently; they do not guarantee exact output
counts.

`patch_manifest.csv` records each saved patch's stable identity, split, source
inline, spatial footprint, relative output files, and checksums.
`dataset_manifest.json` records source identities and checksums, the requested
and resolved split strategy, parameters, target upper limits, actual counts,
and the number of patches excluded at spatial boundaries or by count caps.
