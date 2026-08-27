"""Small, dependency-free helpers shared by the command-line programs."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(
    file_path: str | Path,
    block_size: int = 1024 * 1024,
) -> str:
    """Return a streaming SHA-256 digest without loading a file into memory."""

    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    digest = hashlib.sha256()
    with Path(file_path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()
