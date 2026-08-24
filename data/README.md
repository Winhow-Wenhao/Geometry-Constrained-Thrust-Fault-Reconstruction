# Local input data

Place externally obtained input data under this directory after cloning the
repository. Run documented commands from the repository root.

Download the seismic inputs from the
[project Google Drive data folder](https://drive.google.com/drive/folders/1MtPpidmfl3yWqn-X-P5Cn3K74hVw8g6S).
Google Drive is an external download location: the files are not downloaded
automatically when this repository is cloned.

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

Verify the two reference SEG-Y files after downloading:

| Local path | Size (bytes) | SHA-256 |
|---|---:|---|
| `data/segy/inline600.segy` | 102,888,720 | `87ed66c6c91839661bf4a7a765175cbd332e0922ce85d63c09b44b91539ff785` |
| `data/segy/400_500.segy` | 3,463,802,640 | `ca209ae7aec887a019b8aafc7f0f9ba240966e36f7e82b0876fb88c81e7e68cb` |

From the repository root, checksums can be checked with:

```bash
sha256sum data/segy/inline600.segy data/segy/400_500.segy
```

The multi-inline command example uses `data/segy/test_300_400.segy`; replace
that filename with the name of the volume you place in `data/segy/`.

SEG-Y and interpretation CSV files in these input directories are intentionally
ignored by Git. This directory should contain download instructions and
checksums, not externally obtained survey volumes or interpretations.

The Google Drive location does not change the provenance or license of the
external data. See [`../ASSET_LICENSES.md`](../ASSET_LICENSES.md) before using
or redistributing the seismic files.
