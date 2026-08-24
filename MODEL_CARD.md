# Model card: UASS-Net thrust-fault segmentation reference checkpoint

Model-card status: **Reference-only -- checkpoint identity verified; checkpoint
not distributed; training and license metadata incomplete**  
Last reviewed: 2026-08-24

This card documents the external `model_real.pth` reference checkpoint. The
checkpoint is not distributed with this source-code repository. Its
architecture and recorded file identity were verified against the current code,
but the raw state dictionary does not contain a complete training manifest,
evaluation record, rightsholder statement, or model-weight license.

Users must either lawfully obtain a compatible checkpoint or train their own
with [`uassnet_training.py`](uassnet_training.py). The
[`source notebook`](demo/fault_process_2d_end_to_end_github.ipynb) documents the
2D workflow, while the
[`executed notebook`](demo/fault_process_2d_end_to_end_github_executed.ipynb)
is included as a display record of the reference run. Its stored outputs are not
covered by the repository's MIT software license; see
[`ASSET_LICENSES.md`](ASSET_LICENSES.md).

## Model summary

| Field | Value |
|---|---|
| Model family | UASS-Net, a 2D binary semantic-segmentation network |
| Task | Thrust-fault probability prediction from 2D seismic sections |
| Availability | **External / not distributed** |
| Input tensor | One-channel `1 x 256 x 256` seismic patch per sample |
| Spatial axis order | `(x, z)`; batched tensors use `(batch, channel, x, z)` |
| Input preprocessing | Min-max normalization followed by per-patch z-score standardization |
| Output tensor | One-channel `1 x 256 x 256` logits; apply sigmoid for probabilities |
| Base channels | 16 |
| Decoder dropout | 0.2 |
| Decoder interpolation | Nearest-neighbor |
| Trainable parameters | 5,789,423 |
| Framework | PyTorch 2.1.0; locally verified with `2.1.0+cu121` |
| Checkpoint format | Raw PyTorch `state_dict`, 156 tensor entries |
| Checkpoint size | 23,235,267 bytes |
| SHA-256 | `d154ab68869cfc9c789b94835cb2614c182c7dd0fa3fd56cf2af4bb9b2b638aa` |
| Model version | **Pending** |
| Release date | **Not released** |
| Weight rightsholder | **Pending** |
| Weight license | **Pending; no redistribution or reuse license asserted** |

The recorded reference checkpoint strictly loaded with `base_channels=16` and
`dropout=0.2` into the current definitions in
[`uassnet_training.py`](uassnet_training.py) and
[`fault_post_processing_2d.py`](fault_post_processing_2d.py). The source and
executed notebooks use the same model configuration.

## Architecture

The implementation contains:

- five encoder feature levels with four `2 x 2` max-pooling operations;
- two-convolution blocks with batch normalization and ReLU activation;
- an ASPP bottleneck with a `1 x 1` branch and `3 x 3` branches using dilation
  rates 6, 12, and 18;
- squeeze-and-excitation attention on decoder skip features;
- four nearest-neighbor decoder upsampling blocks;
- decoder dropout of 0.2; and
- a final `3 x 3` one-channel convolution producing logits.

## Intended use

The architecture and workflow are intended for research on thrust-fault
probability prediction in seismic reflection sections. In the complete
workflow, a model probability map is followed by geometry-constrained 2D
processing, cross-inline linking, and 3D surface reconstruction.

Suitable uses include:

- inspecting the included precomputed notebook and PNG display assets;
- running the workflow with a lawfully obtained compatible checkpoint and
  appropriately licensed seismic data;
- training a new compatible checkpoint with the provided code; and
- research comparison or method development after confirming all model-weight
  and data permissions described in
  [`ASSET_LICENSES.md`](ASSET_LICENSES.md).

## Out-of-scope use

The reference checkpoint has not been validated for:

- operational or safety-critical subsurface decisions without expert review;
- geological or processing domains substantially different from the training
  domain;
- treating unprocessed probability pixels as complete fault surfaces;
- calibrated uncertainty estimates; or
- guaranteed fault-detection completeness.

## Training data and provenance

The current code implements an intended two-stage workflow:

1. synthetic pretraining using data produced by
   `synthetic_thrust_data_generation.py`; and
2. transfer learning using manually interpreted real seismic patches produced
   by `real_seismic_label_construction.py`.

The generator defaults to 12,000 synthetic training patches and 2,000
validation patches. The real-label builder defaults to `target_train=200` and
`target_val=40`, but these values are upper limits applied independently after
safe training and validation pools have been formed. They do not guarantee
exact counts and do not permit moving a patch across the split boundary. These
are current code defaults, not proof that the external reference checkpoint was
trained from those exact files or splits.

For newly constructed real datasets, the safe default assigns complete inlines
to training or validation. Each line object in `configs/real_line_inputs.json`
may instead declare an explicit `split` of `train` or `val`. If only one inline
produces usable patches, the builder automatically uses a contiguous spatial
holdout with a guard and discards patches that cross the boundary or guard.
This prevents directly overlapping source pixels from entering both pools.
Every successful run writes `patch_manifest.csv` and `dataset_manifest.json`
with the resolved split, spatial footprints, input and output checksums,
parameters, actual counts, and exclusions.

The recorded reference checkpoint is a model-parameter file only. It does not
identify:

- the exact synthetic configuration and generated-sample checksums;
- the exact real SEG-Y and interpretation files;
- the final train/validation sample manifest;
- the source-code commit used for training; or
- the training hardware, elapsed time, best epoch, and validation history.

The source and license of the NZ3D seismic assets used by the reference workflow
are recorded separately in [`ASSET_LICENSES.md`](ASSET_LICENSES.md). Wenhao
Zheng has confirmed that he independently created the interpretation geometry
and has not assigned or transferred those rights. The manual interpretation
files and training patches nevertheless remain external and are not distributed
in this release.

## Reference training protocol

The current `uassnet_training.py` defaults are:

| Stage or setting | Configuration |
|---|---|
| Synthetic pretraining | 66 epochs, Adam, initial learning rate `1e-3` |
| Transfer learning | 50 epochs, initial learning rate `1e-5` |
| Transfer data | Pass `--data-root outputs/datasets/real_labels` explicitly |
| Transfer initialization | Required explicitly through `--pretrained`; the CLI rejects transfer mode without it |
| Encoder freezing | Disabled by default; `--freeze-encoder` is valid only for transfer learning with a pretrained checkpoint |
| Scheduler | StepLR, step size 20 epochs, factor 0.1 |
| Batch size | 16 |
| Patch size | `256 x 256` |
| Real-data split | Complete-inline grouping by default; explicit per-line `train`/`val` assignments supported |
| Single-inline fallback | Contiguous spatial holdout with a guard; crossing patches are excluded |
| Real-data target counts | `200` train and `40` validation, each used only as an upper limit for its safe pool |
| Real-data manifests | `patch_manifest.csv` and `dataset_manifest.json` |
| Loss | `0.9` frequency-weighted binary cross-entropy plus `0.1` Dice loss |
| Augmentation | Random reversal of physical `x` with probability 0.5 using `np.flip(..., axis=0)` unless disabled |
| Random seed | 2026 |
| Synthetic checkpoint | `outputs/training/uassnet_synthetic_best.pth` |
| Transfer checkpoint | `outputs/training/uassnet_transfer_best.pth` |

Training-generated checkpoints include optimizer, scheduler, configuration,
history, class weights, epoch, and best validation loss. The external
`model_real.pth` reference checkpoint is different: it is a raw state
dictionary and does not carry those records.

## Preprocessing and inference

### Training and single-patch prediction

For each non-label patch, `uassnet_training.py`:

1. converts the array to `float32`;
2. min-max normalizes it to `[0, 1]`;
3. z-score standardizes it using that patch's mean and population standard
   deviation; and
4. passes the standardized patch to the network.

Constant seismic patches are rejected with a clear error. Labels are only
binarized and are not standardized. NumPy inputs use `.npy`; raw `.dat` inputs
must be headerless C-order `(x, z)` square arrays with `float32` seismic values
and `uint8` labels. The configured `--patch-size` must match the writer.

### Full-inline inference

The notebooks and `fault_post_processing_2d.py` first min-max normalize the
processed seismic section, then z-score each overlapping inference tile before
model evaluation. The default tile size is `256 x 256`, overlap is 12 pixels,
and overlapping probabilities are combined with Gaussian weights.

For non-padded interior tiles, min-max followed by z-score is invariant to the
preceding positive affine scaling, so the standardized values agree with
per-tile min-max followed by z-score up to floating-point rounding. Boundary
tiles include zero padding after section normalization and therefore have a
different edge context from isolated training patches.

### Array-axis convention

The canonical stored-array convention is `(x, z)` for both synthetic and real
training patches. The synthetic generator performs its physical simulation
internally in `(z, x)` order, then transposes both seismic and label arrays at
the public `generate_one_sample` boundary. Its root configuration and
per-sample metadata explicitly declare `array_axis_order: ["x", "z"]`. The
real-label and SEG-Y paths already use `(x, z)` or equivalently
`(profile, sample)`.

Training tensors therefore use `(batch, channel, x, z)`. Random augmentation
reverses physical `x` along array axis 0 for both seismic and label. Legacy
synthetic datasets without the explicit `(x, z)` manifest were stored in
`(z, x)` order and are rejected by the current training loader. They must be
regenerated in a new or empty directory. Headerless square DAT patches cannot
self-describe their axes, so callers are responsible for supplying them in
`(x, z)` order.

## Evaluation and reproducibility status

No checkpoint-specific held-out metrics, confusion matrix, calibration result,
or exact test manifest are distributed. The inline-600 assertions in the
notebooks and inference code are implementation-regression checks for one
external SEG-Y input; they are not general performance metrics.

The committed 400--500 3D PNG demonstration starts from an external legacy,
already-gridded surface pickle. It demonstrates visualization, not an
end-to-end evaluation of the reference checkpoint through the current
multi-inline and 3D reconstruction modules.

The complete reference figures cannot be reproduced from this repository
alone. Exact regeneration requires the external reference checkpoint, the
applicable external SEG-Y input, and, for the 3D figures, the external raw or
numeric surface artifacts. The included executed notebook and PNG files are
display records of the reference outputs, not proof that those external inputs
are available or licensed for redistribution.

## Known limitations and risks

- Performance outside the undocumented training survey and processing domain
  is unknown.
- Sigmoid outputs are not documented as calibrated probabilities.
- Legacy synthetic datasets and checkpoints may encode the former `(z, x)`
  convention; regenerate the data and retrain before claiming unified-axis
  reproducibility. The external reference checkpoint does not contain enough
  training provenance to establish its historical array orientation.
- Transfer mode requires a pretrained checkpoint but does not automatically
  select the real-label dataset; `--pretrained` and the appropriate
  `--data-root` must both be passed explicitly.
- Overlapping patches can still occur within a training pool or within a
  validation pool. Across pools, the real-label builder uses complete-inline
  grouping or, for a single inline, a guarded spatial split and validates that
  source footprints do not overlap. A single-inline spatial holdout is still a
  weaker generalization test than validation on independent inlines or surveys.
- Real-label output directories must be new or contain only the placeholder
  `.gitignore`; mixing files from different runs would invalidate the generated
  manifests and split guarantees.
- The final surfaces depend on non-neural 2D and 3D geometric thresholds.
- Exact training provenance and model-weight permission remain incomplete; the
  interpretation-geometry rightsholder has been confirmed separately.

## Basic use

Run commands from the repository root. Verify the architecture on CPU without a
checkpoint:

```bash
python uassnet_training.py --mode sanity --device cpu
```

To predict one `256 x 256` NumPy or float32 DAT seismic patch, first lawfully
obtain a compatible checkpoint or train one with this repository, then run:

```bash
python uassnet_training.py \
  --mode predict \
  --checkpoint /path/to/compatible_checkpoint.pth \
  --patch /path/to/seismic_patch.npy \
  --out-dir outputs/training/predictions \
  --base-channels 16 \
  --dropout 0.2 \
  --patch-size 256 \
  --device cpu
```

The result is written as
`outputs/training/predictions/<patch_stem>_probability.npy`. The
[`source notebook`](demo/fault_process_2d_end_to_end_github.ipynb) gives the
editable single-inline workflow, and the
[`executed notebook`](demo/fault_process_2d_end_to_end_github_executed.ipynb)
shows its precomputed reference outputs. See [`README.md`](README.md) for the
single-inline, multi-inline, and 3D workflow descriptions.

## Conditional integrity check

If you have legally obtained the exact external reference checkpoint, verify
its identity from the repository root with:

```bash
sha256sum /path/to/model_real.pth
```

Expected SHA-256:

```text
d154ab68869cfc9c789b94835cb2614c182c7dd0fa3fd56cf2af4bb9b2b638aa
```

The hash identifies the recorded reference file only. It neither grants access
nor establishes a license. A newly trained compatible checkpoint will normally
have a different hash.

## License and redistribution

The repository's MIT License applies to source code and software documentation;
it does not grant access to or permission to redistribute `model_real.pth`.
The reference checkpoint is external and not distributed. No redistribution or
reuse permission for it should be inferred until the weight rightsholder,
training-data terms, and an explicit model-weight license are confirmed. See
[`ASSET_LICENSES.md`](ASSET_LICENSES.md) for the current asset inventory.

## Citation

Use the manuscript citation in the [README citation section](README.md#citation).
When using the NZ3D seismic data or its visualized derivatives, also cite the
source dataset and follow the terms recorded in
[`ASSET_LICENSES.md`](ASSET_LICENSES.md).
