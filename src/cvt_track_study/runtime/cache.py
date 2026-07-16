"""Content-addressed JSON cache for deterministic simulation summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from threading import Lock
from typing import Any, Mapping
from uuid import uuid4


class SimulationCache:
    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = root.resolve()
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._counter_lock = Lock()

    @staticmethod
    def key(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            self._increment("misses")
            return None
        path = self._path(key)
        if not path.is_file():
            self._increment("misses")
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._increment("misses")
            return None
        self._increment("hits")
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
        self._increment("writes")

    def status(self) -> dict[str, int | str | bool]:
        files = list(self.root.glob("*/*.json")) if self.root.exists() else []
        with self._counter_lock:
            hits, misses, writes = self.hits, self.misses, self.writes
        return {
            "enabled": self.enabled,
            "root": str(self.root),
            "entry_count": len(files),
            "size_bytes": sum(path.stat().st_size for path in files),
            "session_hits": hits,
            "session_misses": misses,
            "session_writes": writes,
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

    def _increment(self, field: str) -> None:
        with self._counter_lock:
            setattr(self, field, int(getattr(self, field)) + 1)
