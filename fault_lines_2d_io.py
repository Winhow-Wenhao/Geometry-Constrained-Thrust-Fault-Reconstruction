"""Shared contract helpers for the versioned 2D-to-3D fault-line artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


FAULT_LINES_2D_SCHEMA_VERSION = "fault-lines-2d/1.0"
FAULT_LINES_2D_ARTIFACT_TYPE = "fault-lines-2d-interface"
FAULT_LINES_2D_POINT_COLUMNS = ("profile_index", "sample_index")


def metadata_path_for_fault_lines(fault_lines_pickle: str | Path) -> Path:
    """Return the canonical metadata sidecar path for a 2D result pickle."""

    return Path(fault_lines_pickle).with_suffix(".metadata.json")


def cluster_sort_key(cluster_id: Any) -> tuple[int, Any]:
    """Return a deterministic key that orders numeric cluster IDs first."""

    if isinstance(cluster_id, (int, np.integer)):
        return (0, int(cluster_id))
    try:
        return (0, int(str(cluster_id)))
    except ValueError:
        return (1, str(cluster_id))
