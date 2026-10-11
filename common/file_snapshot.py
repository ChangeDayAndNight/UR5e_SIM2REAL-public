"""Single-open snapshots for authority-bearing regular files.

Callers must make every decision about a mutable file from one snapshot.  In
particular, the digest and decoded YAML/JSON must never come from separate
pathname opens: an attacker could otherwise present trusted bytes for the
hash and different bytes for parsing before restoring the trusted file.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat


class FileSnapshotError(RuntimeError):
    """A regular file could not be captured as one stable byte snapshot."""


@dataclass(frozen=True)
class RegularFileSnapshot:
    """Bytes and metadata derived from the same open file description."""

    payload: bytes
    sha256: str
    size: int


def read_regular_file_snapshot(
    path: Path | str,
    *,
    label: str = "file",
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> RegularFileSnapshot:
    """Read and hash a stable regular file through exactly one open handle."""
    if expected_sha256 is not None and re.fullmatch(
        r"[0-9a-f]{64}", str(expected_sha256)
    ) is None:
        raise FileSnapshotError(f"{label} expected sha256 is malformed")
    if expected_size is not None and (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise FileSnapshotError(f"{label} expected size is malformed")

    source = Path(path)
    try:
        with source.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise FileSnapshotError(f"{label} is not a regular file")
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except FileSnapshotError:
        raise
    except OSError as exc:
        raise FileSnapshotError(f"cannot read {label}: {exc}") from exc

    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(payload) != before.st_size:
        raise FileSnapshotError(f"{label} changed while it was being read")

    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise FileSnapshotError(f"{label} sha256 mismatch")
    if expected_size is not None and len(payload) != expected_size:
        raise FileSnapshotError(f"{label} size mismatch")
    return RegularFileSnapshot(payload=payload, sha256=digest, size=len(payload))
