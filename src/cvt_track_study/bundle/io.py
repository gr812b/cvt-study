"""Canonical JSON I/O and integrity checks for track bundles."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json_bytes, sha256_bytes
from .model import TrackBundle, TrackBundleError
from .validation import validate_track_bundle


def write_track_bundle(path: Path, data: Mapping[str, Any]) -> TrackBundle:
    validate_track_bundle(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(data)
    digest = sha256_bytes(payload)
    path.write_bytes(payload)
    path.with_name("track_bundle.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return TrackBundle(data=dict(data), path=path.resolve(), sha256=digest)


def load_track_bundle(path: str | Path, *, verify_checksum: bool = True) -> TrackBundle:
    resolved = Path(path).resolve()
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrackBundleError(f"Unable to read track bundle {resolved}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise TrackBundleError("Track bundle root must be a JSON object.")
    validate_track_bundle(raw)
    payload = resolved.read_bytes()
    digest = sha256_bytes(payload)
    checksum_path = resolved.with_name("track_bundle.sha256")
    if verify_checksum and checksum_path.exists():
        parts = checksum_path.read_text(encoding="utf-8").strip().split()
        if not parts:
            raise TrackBundleError(f"Track bundle checksum file is empty: {checksum_path}")
        expected = parts[0]
        if expected != digest:
            raise TrackBundleError(
                f"Track bundle checksum mismatch: expected {expected}, computed {digest}."
            )
    return TrackBundle(data=dict(raw), path=resolved, sha256=digest)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
