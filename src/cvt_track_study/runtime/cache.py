"""Content-addressed JSON cache for deterministic simulation summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4


class SimulationCache:
    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = root.resolve()
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @staticmethod
    def key(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            self.misses += 1
            return None
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: str, value: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
        self.writes += 1

    def status(self) -> dict[str, int | str | bool]:
        files = list(self.root.glob("*/*.json")) if self.root.exists() else []
        return {
            "enabled": self.enabled,
            "root": str(self.root),
            "entry_count": len(files),
            "size_bytes": sum(path.stat().st_size for path in files),
            "session_hits": self.hits,
            "session_misses": self.misses,
            "session_writes": self.writes,
        }

    def clear(self) -> int:
        count = int(self.status()["entry_count"])
        if self.root.exists():
            shutil.rmtree(self.root)
        return count

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Cache keys must be lower-case SHA-256 hex strings.")
        return self.root / key[:2] / f"{key}.json"
