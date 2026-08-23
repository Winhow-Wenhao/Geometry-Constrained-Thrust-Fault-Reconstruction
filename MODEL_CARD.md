# Model card: UASS-Net thrust-fault segmentation checkpoint

Model-card status: **Draft -- release metadata incomplete**  
Last reviewed: 2026-08-22

This card describes the bundled `model_real.pth` file. Checkpoint identity and
architecture compatibility have been verified, but its complete training
provenance and redistribution license have not. Do not interpret this card as
permission to redistribute the weights.

## Model summary

| Field | Value |
|---|---|
| Model family | UASS-Net, a 2D binary semantic-segmentation network |
| Task | Thrust-fault probability prediction from 2D seismic sections |
| Input | One-channel `256 x 256` seismic patch |
| Output | One-channel `256 x 256` logits; apply sigmoid for probabilities |
| Base channels | 16 |
| Decoder dropout | 0.2 |
| Decoder interpolation | Nearest-neighbor |
| Trainable parameters | 5,789,423 |
| Framework | PyTorch 2.1.0; locally verified with `2.1.0+cu121` |
| Checkpoint format | Raw PyTorch `state_dict` |
| Checkpoint size | 23,235,267 bytes |
| SHA-256 | `d154ab68869cfc9c789b94835cb2614c182c7dd0fa3fd56cf2af4bb9b2b638aa` |
| Model version | **TO BE CONFIRMED** |
| Release date | **TO BE CONFIRMED** |
| Weight rightsholder | **TO BE CONFIRMED** |
| Weight license | **TO BE CONFIRMED** |

The checkpoint strictly loads into the definitions in
`train_uassnet_article.py` and `fault_process_2d_end_to_end.py` with the
configuration above.

## Architecture

The current implementation contains:

- five encoder feature levels with four max-pooling operations;
- an ASPP bottleneck with a `1 x 1` branch and dilation rates 6, 12, and 18;
- squeeze-and-excitation attention on decoder skip features;
- four nearest-neighbor decoder upsampling blocks;
- decoder dropout of 0.2; and
- a final `3 x 3` one-channel convolution producing logits.

The manuscript wording about the number of down/up-sampling stages should be
read together with the implementation notes in the main README.

## Intended use

The model is intended for research on detecting thrust-fault probability
ridges in seismic reflection sections. The complete reference workflow uses
its probability map as input to geometric 2D post-processing, multi-inline
linking, and 3D surface reconstruction.

Suitable uses include:

- reproducing or evaluating the accompanying research workflow;
- research comparison on appropriately licensed seismic data; and
- fine-tuning or methodological development when permitted by the eventual
  model-weight and training-data licenses.

## Out-of-scope use

The checkpoint is not validated for:

- operational or safety-critical subsurface decisions without expert review;
- geological settings substantially different from the training domain;
- direct interpretation of unprocessed model probabilities as complete fault
  surfaces; or
- claims of calibrated uncertainty or guaranteed detection completeness.

## Training data

The intended manuscript workflow combines synthetic pretraining and transfer
learning using manually interpreted real seismic sections. The repository
describes real interpreted inlines 30, 230, 430, 630, and 830, with a target
split of 200 training and 40 validation patches.

However, the raw checkpoint contains parameters only. It does not identify the
exact files, versions, sample manifest, or split used to produce these weights.
The following must therefore be completed before archival release:

| Item | Status |
|---|---|
| Synthetic-data configuration and checksum manifest | **TO BE CONFIRMED** |
| Real SEG-Y provider, dataset identifier, and license | **TO BE CONFIRMED** |
| Manual-interpretation creator and permission | **TO BE CONFIRMED** |
| Exact five-inline coordinate configuration | **TO BE CONFIRMED** |
| Fixed 200/40 patch split manifest | **TO BE CONFIRMED** |

See [ASSET_LICENSES.md](ASSET_LICENSES.md) for release restrictions on these
assets.

## Reference training protocol

The current scripts and manuscript describe the following intended protocol;
these values are not embedded in the raw checkpoint and are not, by
themselves, proof of its training history.

| Stage | Intended configuration |
|---|---|
| Synthetic pretraining | 66 epochs, Adam, initial learning rate `1e-3` |
| Transfer learning | 50 epochs, initial learning rate `1e-5`, all components trainable |
| Scheduler | Step size 20 epochs, factor 0.1 |
| Batch size | 16 |
| Patch size | `256 x 256` |
| Loss | `0.9` weighted BCE plus `0.1` Dice loss |

Checkpoint-specific best epoch, validation loss, training history, random
seed, hardware, elapsed time, and source commit are **TO BE CONFIRMED**.

## Preprocessing and inference

Training and single-patch prediction min-max normalize individual patches to
`[0, 1]`. The current end-to-end SEG-Y path first min-max normalizes a complete
processed section and then z-score standardizes each overlapping inference
patch. These inputs are not numerically equivalent and this difference should
be considered when comparing entry points.

The current data-generation scripts also need an explicit archival decision on
array orientation: the synthetic generator represents arrays as `(z, x)`,
whereas real-label construction and SEG-Y inference use `(profile, sample)` or
`(x, z)`. The square patch shape hides this distinction at the tensor-shape
level. The intended orientation and any required transpose must be confirmed
before claiming exact retraining reproducibility.

## Evaluation

No checkpoint-specific validation metrics, test split, confusion matrix, or
best-epoch record are bundled. The inline-600 stage counts in the inference
code are implementation-regression checks for a particular external SEG-Y,
not general model-performance metrics.

The supplied 400--500 3D demo starts from a legacy already-gridded surface
pickle. It demonstrates visualization but is not an evaluation of this
checkpoint through the current multi-inline and 3D reconstruction pipeline.

## Known limitations

- Performance outside the training survey and acquisition/processing domain
  has not been documented.
- Probability values are not documented as calibrated probabilities.
- Training and full-inline inference currently use different normalization
  sequences.
- Synthetic and real-data array-axis conventions require resolution.
- The final interpretation depends strongly on non-neural 2D and 3D geometric
  thresholds.
- The distributed checkpoint does not contain optimizer state, scheduler
  state, training history, dataset identifiers, or provenance metadata.

## Basic use

After installing the dependencies, verify the architecture on CPU:

```bash
python train_uassnet_article.py --mode sanity --device cpu
```

Predict one `256 x 256` NumPy or float32 DAT patch:

```bash
python train_uassnet_article.py \
  --mode predict \
  --checkpoint model_real.pth \
  --patch /path/to/seismic_patch.npy \
  --out-dir patch_predictions \
  --base-channels 16 \
  --dropout 0.2 \
  --patch-size 256
```

See the main [README.md](README.md) for single-inline and multi-inline SEG-Y
workflows.

## License and redistribution

**TO BE CONFIRMED.** The repository's MIT source-code license does not
automatically establish a license for `model_real.pth`. Before public release,
the weight rightsholder must explicitly state the applicable license and
confirm that the training-data terms permit distribution of the trained
weights. Until then, no permission to redistribute or reuse the checkpoint
should be inferred.

## Citation

Please use the repository's [CITATION.cff](CITATION.cff). Add the final article
year and DOI when they become available.

## Contact and maintenance

Model maintainer and licensing contact: **TO BE CONFIRMED**.
