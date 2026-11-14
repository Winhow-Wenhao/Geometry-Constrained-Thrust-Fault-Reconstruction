# Demonstrations

This directory contains demonstration source code together with selected
display-only PNG figures. Large inputs, model weights, pickles, NumPy data
products, CSV tables, and run-metadata files are intentionally not stored here.
The reference SEG-Y subsets, checkpoint, and legacy fault-surface pickle are
distributed through [Zenodo v1.1.0](https://doi.org/10.5281/zenodo.22641723).

## 2D end-to-end notebooks

- `fault_process_2d_end_to_end_github.ipynb` is the clean source notebook. Its
  code and Markdown cells are retained, while all saved cell outputs and
  execution counts are cleared. It imports the maintained UASS-Net, SEG-Y,
  inference, and post-processing implementations from the repository root.
  Use this file when rerunning the workflow.
- `fault_process_2d_end_to_end_github_executed.ipynb` contains the same source
  cells and notebook metadata, plus the nine precomputed reference figures and
  execution counts from the validated inline-600 run. Use it to inspect the
  expected result without executing the notebook.

The clean notebook discovers the repository root when its kernel starts from
either the repository root or this `demo/` directory. Re-execution requires
the following external assets at the paths expected by the notebook:

```text
model_real.pth
data/segy/inline600.segy
```

Both files are available from [Zenodo v1.1.0](https://doi.org/10.5281/zenodo.22641723); the
reference notebook verifies their SHA-256 checksums. A newly trained compatible
model can be used for non-reference runs through the maintained workflow, but
it will not reproduce the stored reference outputs. See the root `README.md`,
`MODEL_CARD.md`, and `ASSET_LICENSES.md` for acquisition, provenance, and
release-status information.

During the shared-module refactor, the source cells of the clean and executed
notebooks were synchronized and the nine embedded PNG payloads were retained
byte-for-byte. The stored outputs are a display record of the previously
validated run rather than a newly executed run of this repository snapshot.

## 3D visualization source

`visualize_fault_surfaces_3d.py` visualizes either a canonical
`fault_surfaces.npz` produced by `fault_surface_reconstruction_3d.py` or a
trusted legacy gridded-surface pickle. The exact precomputed 400--500 figures
also require the external seismic context and legacy surface input:

```text
data/segy/400_500.segy
demo/outputs_400_500.pkl
```

Neither input is stored in this demonstration directory; both are available
from [Zenodo v1.1.0](https://doi.org/10.5281/zenodo.22641723). Never load an untrusted
pickle: verify the SHA-256 of `outputs_400_500.pkl` against the root README
before passing `--allow-legacy-pickle`. Run
`python demo/visualize_fault_surfaces_3d.py --help` from the repository root for
the supported formats and required coordinate options.

## Precomputed figures

- `figures/fault2d_demo_600/` contains the nine PNG figures preserved in the
  executed 2D notebook.
- `figures/fault3d_demo_400_500/` contains five PNG views and seismic overlays
  from the 400--500 visualization example.

These PNG files are illustrative precomputed results, not inputs to the code.
No pickle, NPZ, CSV, JSON, SEG-Y, or checkpoint file is stored under
`figures/`.

## Licensing

The root MIT License covers repository-authored source code, notebook source
cells, and software documentation. Wenhao Zheng has confirmed that he
independently created the included interpretation geometry and has not assigned
or transferred those rights. The stored notebook outputs and PNG display assets
are separately released under CC BY-NC-SA 3.0 US, subject to the NZ3D source
attribution and modification notices in the root `ASSET_LICENSES.md`. The
reference checkpoint and legacy pickle are distributed through Zenodo under
CC BY-NC-SA 3.0 US and are not covered by the MIT License.
