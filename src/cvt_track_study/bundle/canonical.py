"""Canonical serialization helpers shared by bundle building and validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


FINGERPRINT_FIELDS = (
    "format",
    "schema_version",
    "identity",
    "coordinate_contract",
    "simulation_contract",
    "evidence",
    "uncertainty_contract",
    "provenance",
)


def canonical_json_bytes(data: Mapping[str, Any], *, pretty: bool = True) -> bytes:
    """Return deterministic, standards-compliant JSON bytes."""

    text = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    return (text + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def content_fingerprint(data: Mapping[str, Any]) -> str:
    payload = {key: data[key] for key in FINGERPRINT_FIELDS if key in data}
    return sha256_bytes(canonical_json_bytes(payload, pretty=False))
