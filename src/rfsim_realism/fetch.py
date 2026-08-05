from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

from .ucc_static import SOURCE_ARCHIVE_SHA256

SOURCE_URL = (
    "https://raw.githubusercontent.com/uccmisl/5Gdataset/"
    "v1.0.0/5G-production-dataset.zip"
)


def fetch_archive(destination: str | Path, source_url: str = SOURCE_URL) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != SOURCE_ARCHIVE_SHA256:
            raise ValueError(
                f"existing archive has SHA-256 {digest}; expected {SOURCE_ARCHIVE_SHA256}"
            )
        return destination

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    try:
        with urlopen(source_url) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                digest.update(block)
                output.write(block)
        actual = digest.hexdigest()
        if actual != SOURCE_ARCHIVE_SHA256:
            raise ValueError(
                f"downloaded archive has SHA-256 {actual}; expected {SOURCE_ARCHIVE_SHA256}"
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
