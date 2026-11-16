"""Archive only the portable report, publishing atomically after successful compression."""

from __future__ import annotations

import os
from pathlib import Path
import tarfile
import tempfile
import zipfile


def archive_report(
    html_path: Path, mode: str = "auto", threshold_mb: float = 50
) -> Path | None:
    if mode not in {"auto", "none", "zip", "tar.gz"} or threshold_mb < 0:
        raise ValueError("Invalid archive mode or negative size threshold")
    if mode == "none" or (
        mode == "auto" and html_path.stat().st_size <= threshold_mb * 1024 * 1024
    ):
        return None
    fmt = "zip" if mode == "auto" else mode
    destination = html_path.with_suffix("." + fmt)
    fd, name = tempfile.mkstemp(prefix=".report-archive-", dir=html_path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        if fmt == "zip":
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.write(html_path, arcname=html_path.name)
        else:
            with tarfile.open(temporary, "w:gz") as archive:
                archive.add(html_path, arcname=html_path.name, recursive=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
