# Asset licensing and provenance

Last reviewed: 2026-08-26

This document records the provenance and release status of non-code assets
associated with this repository. This source-code release with display-only
material contains code, the source and executed 2D notebooks, and precomputed
PNG figures. It does not contain model weights, SEG-Y data, or raw/numeric demo
products. The two reference SEG-Y subsets are distributed separately through
the versioned [Zenodo data record](https://doi.org/10.5281/zenodo.22070539).

This inventory is not legal advice and does not replace the terms published by
the original data provider. "Display asset" describes an asset's role in this
repository; it is not a substitute for permission to distribute that asset.

## Scope of the MIT software license

The root [MIT License](LICENSE) covers the source code, notebook source cells,
and software documentation authored for this repository, copyright (c) 2026
Wenhao Zheng.

The MIT License does not automatically apply to:

- the stored outputs in the executed notebook;
- the precomputed 2D or 3D PNG figures;
- `model_real.pth`;
- external SEG-Y files;
- manual fault interpretations or real training labels; or
- fault-surface files and other scientific data products derived from those
  inputs.

## Status definitions

- **Included**: the file is part of this source-code release.
- **Included / display asset**: the file is included to show a precomputed
  reference result. This status does not grant reuse rights beyond the stated
  license and permissions.
- **External / not distributed**: the asset is referenced by the workflow or
  recorded for identity, but is intentionally excluded from this repository.
- **External to repository / distributed separately**: the asset is excluded
  from the GitHub repository but is publicly archived at the cited location.
- **Confirmed source license**: the upstream source and its published license
  have been documented. This does not confirm rights in independently authored
  interpretation geometry.
- **Confirmed geometry rightsholder**: Wenhao Zheng has confirmed that the
  interpretation geometry was independently created by him, that its rights
  have not been assigned or transferred, and that he authorizes the release
  described below.

## Interpretation-geometry rightsholder declaration

On 2026-08-24, Wenhao Zheng confirmed that all fault-interpretation geometry
used in this repository's included notebook outputs and PNG display assets was
independently created by him and that the associated rights have not been
assigned or transferred.

Copyright in those interpretation-geometry and repository-authored
visualization contributions is held by Wenhao Zheng. He authorizes their public
distribution under
[CC BY-NC-SA 3.0 US](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
This declaration does not claim ownership of the underlying NZ3D seismic data,
the external model checkpoint, or any other third-party material. Those
components remain subject to their own terms.

## Asset inventory

| Asset | Repository status | Provenance | License and release status |
|---|---|---|---|
| Source code, software documentation, and source cells in [`demo/fault_process_2d_end_to_end_github.ipynb`](demo/fault_process_2d_end_to_end_github.ipynb) | Included | Authored for the accompanying manuscript | **Confirmed:** MIT; see [LICENSE](LICENSE). The source notebook has no stored execution outputs. |
| [`demo/fault_process_2d_end_to_end_github_executed.ipynb`](demo/fault_process_2d_end_to_end_github_executed.ipynb) | Included / display asset | Same authored source cells plus stored reference outputs from the external checkpoint and NZ3D inline | MIT covers the authored source cells. The stored display outputs, including Wenhao Zheng's confirmed interpretation-geometry contributions, are released under CC BY-NC-SA 3.0 US. Outputs containing NZ3D amplitudes must retain source attribution and identify the visualization and overlay changes. |
| `model_real.pth` | External / not distributed | Reference UASS-Net raw PyTorch state dictionary | No public model-weight license is asserted. Confirm the weight rightsholder, training-data terms, and an explicit weight license before any later distribution. |
| `data/segy/inline600.segy` | External to repository / distributed separately | Cropped derivative of the NZ3D/MGL1801 PSDM volume described below | **Published:** [Zenodo v1.0.0](https://doi.org/10.5281/zenodo.22070539), under CC BY-NC-SA 3.0 US. Reuse must retain attribution, identify the crop or other changes, remain non-commercial, and use compatible ShareAlike terms. |
| `data/segy/400_500.segy` | External to repository / distributed separately | Cropped derivative of the same NZ3D/MGL1801 PSDM volume | **Published:** [Zenodo v1.0.0](https://doi.org/10.5281/zenodo.22070539), under CC BY-NC-SA 3.0 US and subject to the same conditions. |
| `demo/outputs_400_500.pkl` | External / not distributed | Legacy gridded fault-surface product derived from NZ3D seismic data and Wenhao Zheng's interpretation | Interpretation-geometry rightsholder confirmed. The file remains excluded by release choice. If it is released later, apply CC BY-NC-SA 3.0 US and retain all applicable NZ3D attribution and change notices. |
| `outputs/visualization/fault3d_demo_400_500/surfaces_normalized.npz`, all CSV files in that directory, and `outputs/visualization/fault3d_demo_400_500/visualization_metadata.json` | External / not distributed | Numeric and metadata derivatives of `demo/outputs_400_500.pkl` generated at the current viewer output path | Interpretation-geometry rightsholder confirmed. These files remain excluded by release choice; any later release must retain provenance and use CC BY-NC-SA 3.0 US. |
| `demo/figures/fault2d_demo_600/01_*.png` through `08_*.png` | Included / display assets | Model and geometry-processing results derived from the external reference checkpoint and NZ3D inline | Copyright in Wenhao Zheng's interpretation-geometry and visualization contributions is confirmed. Included display assets are released under CC BY-NC-SA 3.0 US and are not covered by MIT. |
| [`demo/figures/fault2d_demo_600/09_final_fault_traces_on_seismic.png`](demo/figures/fault2d_demo_600/09_final_fault_traces_on_seismic.png) | Included / display asset | NZ3D inline amplitude image with Wenhao Zheng's derived fault traces overlaid | Released under CC BY-NC-SA 3.0 US. Attribution, NonCommercial, ShareAlike, and identification of changes are required. Changes include inline extraction/cropping, display scaling, and fault-trace overlay. Interpretation-geometry rightsholder confirmed. |
| [`demo/figures/fault3d_demo_400_500/fault_surfaces_map_view.png`](demo/figures/fault3d_demo_400_500/fault_surfaces_map_view.png) and [`fault_surfaces_oblique.png`](demo/figures/fault3d_demo_400_500/fault_surfaces_oblique.png) | Included / display assets | Rendered geometry derived from Wenhao Zheng's external legacy fault-surface product | Copyright in the interpretation-geometry and visualization contributions is confirmed. Included display assets are released under CC BY-NC-SA 3.0 US and are not covered by MIT. |
| `demo/figures/fault3d_demo_400_500/inline_0*_overlay.png` | Included / display assets | Rendered overlays containing cropped NZ3D seismic amplitudes and Wenhao Zheng's derived fault surfaces | Released under CC BY-NC-SA 3.0 US. Attribution, NonCommercial, ShareAlike, and identification of changes are required. Changes include spatial subsetting, display scaling/rendering, and fault-surface overlay. Interpretation-geometry rightsholder confirmed. |
| Synthetic datasets generated by `synthetic_thrust_data_generation.py` | External / not distributed | Generated by repository code | No separate generated-data license is declared. Assign one and record the configuration and checksum manifest before distributing a generated dataset. |
| Manual interpretation CSV files and real training patches | External / not distributed | Manuscript inputs containing Wenhao Zheng's independently authored interpretations and seismic-derived patches | Interpretation-geometry rightsholder confirmed. These files remain excluded by release choice; any later release must also comply with the applicable seismic-source terms and document its provenance. |

## Zenodo derived-data record

The two reference SEG-Y subsets are archived independently from this GitHub
source-code repository in:

> Zheng, Wenhao, et al. (2026). *Reproducibility Seismic Data for
> Geometry-Constrained Thrust Fault Reconstruction* (Version 1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.22070539

The record contains `inline600.segy`, `400_500.segy`, their checksum manifest,
and standalone data-license and data-README files. It identifies the MGDS source
record below as `IsDerivedFrom` and distributes the cropped subsets under
CC BY-NC-SA 3.0 US. The all-versions concept DOI is
https://doi.org/10.5281/zenodo.22070538; reproducibility instructions and fixed
checksums should cite the version DOI above.

## NZ3D seismic source record

The source for the repository's cropped seismic inputs and amplitude-bearing
display assets is:

> Bangs, Nathan, et al. (2022). *NZ3D seismic reflection data volume -
> Prestack Depth Migration (PSDM).* Marine Geoscience Data System (MGDS).
> https://doi.org/10.26022/IEDA/331022

The authoritative MGDS record identifies the dataset as an MGL1801/NZ3D
prestack depth-migrated SEG-Y volume and lists the license as
[Creative Commons Attribution-NonCommercial-ShareAlike 3.0 United States
(CC BY-NC-SA 3.0 US)](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).

The official dataset record is:
https://www.marine-geo.org/doi/10.26022/IEDA/331022

The referenced SEG-Y files are cropped derivatives, not the complete MGDS
distribution. The included amplitude-bearing figures and executed-notebook
outputs are visualized adaptations. They must state that changes were made and
must not imply endorsement by MGDS or the dataset creators.

### Suggested attribution for amplitude-bearing display assets

```text
Contains cropped and visualized derivatives of:
Bangs, Nathan, et al. (2022), NZ3D seismic reflection data volume -
Prestack Depth Migration (PSDM), Marine Geoscience Data System (MGDS),
doi:10.26022/IEDA/331022, licensed under CC BY-NC-SA 3.0 US.
Changes include spatial subsetting, display scaling/visualization, and
overlaying derived fault traces or surfaces, as applicable.
```

The attribution above covers the NZ3D source material. Wenhao Zheng's separate
rightsholder declaration above covers only his interpretation-geometry and
repository-authored visualization contributions.

## Recorded external asset identities

The following hashes identify the external files used for the reference
workflow. For the unpublished checkpoint and legacy pickle, a recorded hash does
not mean that redistribution permission has been granted. Distribution and reuse
of the two SEG-Y subsets are governed by the Zenodo record and the terms stated
above.

| External asset | Size (bytes) | SHA-256 | Distribution |
|---|---:|---|---|
| `model_real.pth` | 23,235,267 | `d154ab68869cfc9c789b94835cb2614c182c7dd0fa3fd56cf2af4bb9b2b638aa` | External / not distributed |
| `demo/outputs_400_500.pkl` | 9,625,283 | `bbac865f3eb57a1c27055bcdf2d2d52fd36f57fb89abe6a109cc2fed006fd1ca` | External / not distributed |
| `data/segy/inline600.segy` | 102,888,720 | `87ed66c6c91839661bf4a7a765175cbd332e0922ce85d63c09b44b91539ff785` | [Zenodo v1.0.0](https://doi.org/10.5281/zenodo.22070539) |
| `data/segy/400_500.segy` | 3,463,802,640 | `ca209ae7aec887a019b8aafc7f0f9ba240966e36f7e82b0876fb88c81e7e68cb` | [Zenodo v1.0.0](https://doi.org/10.5281/zenodo.22070539) |

## Requirements for any later model-weight release

Before publicly distributing `model_real.pth`, record:

- the weight rightsholder and their authorization to release it;
- the exact training-data sources and applicable terms;
- an explicit model-weight license;
- the model version and release date; and
- the required model and manuscript citation.

Any later license must be stated explicitly here and in
[MODEL_CARD.md](MODEL_CARD.md); it must not be inferred from the source-code
license.

## Current release status

This source-only release includes the repository-authored code, software
documentation, source and display notebooks, and the identified precomputed
PNG display assets. The MIT and CC BY-NC-SA 3.0 US licensing boundaries,
NZ3D provenance and attribution, modification notices, and external reference
hashes are recorded above. Wenhao Zheng's interpretation-geometry and
visualization contributions are released under the terms stated in this file.

The two reference SEG-Y inputs are not bundled in this GitHub repository but are
distributed separately through Zenodo. Model weights, manual interpretation
files, legacy pickles, and raw or numeric demo products remain intentionally
undistributed. Their absence is part of the defined scope of this release and
does not represent an unfinished release task.

If a model checkpoint is distributed in the future, its rightsholder,
training-data permissions, explicit model-weight license, version, release
date, and required citations must first be recorded as specified in
"Requirements for any later model-weight release" above.
