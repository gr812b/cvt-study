"""Atomic result ownership and resumable per-scenario checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping


class WorkspaceError(RuntimeError):
    pass


class ResultWorkspace:
    def __init__(
        self,
        output: Path,
        *,
        fingerprint: str,
        resume: bool = False,
        restart: bool = False,
    ) -> None:
        self.output = output.resolve()
        self.incomplete = self.output.with_name(f".{self.output.name}.incomplete")
        self.fingerprint = fingerprint
        if resume and restart:
            raise WorkspaceError("--resume and --restart are mutually exclusive.")
        if self.output.exists():
            raise WorkspaceError(f"Completed result already exists: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if restart and self.incomplete.exists():
            shutil.rmtree(self.incomplete)
        if self.incomplete.exists():
            if not resume:
                raise WorkspaceError(
                    f"Incomplete result exists: {self.incomplete}. Use --resume or --restart."
                )
            metadata = self._read_metadata()
            if metadata.get("fingerprint") != fingerprint:
                raise WorkspaceError(
                    "The incomplete workspace belongs to different resolved inputs; use --restart."
                )
        else:
            self.incomplete.mkdir(parents=False)
            self._write_metadata({"fingerprint": fingerprint, "state": "running"})
        self.checkpoints.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.incomplete

    @property
    def checkpoints(self) -> Path:
        return self.incomplete / "checkpoints"

    def load_checkpoint(self, replicate: int) -> dict[str, Any] | None:
        path = self.checkpoints / f"scenario_{replicate:06d}.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("fingerprint") != self.fingerprint:
            raise WorkspaceError(f"Checkpoint fingerprint mismatch: {path}")
        return data

    def write_checkpoint(self, replicate: int, payload: Mapping[str, Any]) -> None:
        path = self.checkpoints / f"scenario_{replicate:06d}.json"
        temporary = path.with_suffix(".json.tmp")
        data = {"fingerprint": self.fingerprint, **payload}
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def commit(self) -> Path:
        self._write_metadata({"fingerprint": self.fingerprint, "state": "complete"})
        shutil.rmtree(self.checkpoints, ignore_errors=True)
        self.incomplete.replace(self.output)
        return self.output

    def _read_metadata(self) -> dict[str, Any]:
        path = self.incomplete / "workspace.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkspaceError(
                f"Incomplete workspace metadata is unreadable: {path}"
            ) from exc

    def _write_metadata(self, data: Mapping[str, Any]) -> None:
        (self.incomplete / "workspace.json").write_text(
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
