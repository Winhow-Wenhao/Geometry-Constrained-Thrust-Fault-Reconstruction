# Local input data

Place externally obtained input data under this directory after cloning the
repository. Run documented commands from the repository root.

Download the two reference seismic inputs from the citable
[Zenodo v1.1.0 data record](https://doi.org/10.5281/zenodo.22641723). The same record
also provides the reference checkpoint `model_real.pth` (place it in the
repository root) and the legacy fault-surface pickle `outputs_400_500.pkl`
(place it under `demo/`). Use this versioned record for download, citation,
and reference-file identity. No external data are downloaded automatically
when this repository is cloned.

Use these locations for real seismic inputs and manual interpretations:

```text
data/segy/<inline>.segy
data/interpretations/<inline>.csv
```

For example, the current demonstrations expect:

```text
data/segy/inline600.segy
data/segy/400_500.segy
```

Verify the four reference files after downloading:

| Local path | Size (bytes) | SHA-256 |
|---|---:|---|
| `data/segy/inline600.segy` | 102,888,720 | `87ed66c6c91839661bf4a7a765175cbd332e0922ce85d63c09b44b91539ff785` |
| `data/segy/400_500.segy` | 3,463,802,640 | `ca209ae7aec887a019b8aafc7f0f9ba240966e36f7e82b0876fb88c81e7e68cb` |
| `model_real.pth` | 23,235,267 | `d154ab68869cfc9c789b94835cb2614c182c7dd0fa3fd56cf2af4bb9b2b638aa` |
| `demo/outputs_400_500.pkl` | 9,625,283 | `bbac865f3eb57a1c27055bcdf2d2d52fd36f57fb89abe6a109cc2fed006fd1ca` |

From the repository root, checksums can be checked with:

```bash
sha256sum data/segy/inline600.segy data/segy/400_500.segy \
  model_real.pth demo/outputs_400_500.pkl
```

The multi-inline command example uses `data/segy/test_300_400.segy`; replace
that filename with the name of the volume you place in `data/segy/`.

SEG-Y and interpretation CSV files in these input directories, and the
checkpoint and pickle files elsewhere in the repository, are intentionally
ignored by Git. This directory should contain download instructions and
checksums, not externally obtained survey volumes or interpretations.

The download location does not change the provenance or license of the external
data. The two Zenodo SEG-Y subsets are cropped derivatives of the NZ3D/MGL1801
PSDM volume
([MGDS DOI 10.26022/IEDA/331022](https://doi.org/10.26022/IEDA/331022)); the
checkpoint and the fault-surface pickle are derived products of the same
volume. All four are distributed under CC BY-NC-SA 3.0 US. See
[`../ASSET_LICENSES.md`](../ASSET_LICENSES.md) before using or redistributing
them.
