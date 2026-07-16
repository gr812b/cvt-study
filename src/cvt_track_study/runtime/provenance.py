"""Portable run provenance and a small human-readable provenance graph."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from cvt_track_study import __version__


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_provenance(
    *,
    command: Sequence[str],
    project: Path,
    bundle_path: Path,
    study_name: str,
    study_fingerprint: str,
    resolved_configuration_fingerprint: str,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "framework_version": __version__,
        "git_commit": _git_commit(project),
        "command": list(command),
        "project": str(project.resolve()),
        "study_name": study_name,
        "study_fingerprint_sha256": study_fingerprint,
        "resolved_configuration_fingerprint_sha256": resolved_configuration_fingerprint,
        "track_bundle_file": bundle_path.name,
        "track_bundle_sha256": file_sha256(bundle_path),
    }


def write_provenance(output: Path, provenance: Mapping[str, Any]) -> None:
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    labels = [
        ("Measured GPX", "Input evidence"),
        ("Track build", str(provenance.get("track_bundle_sha256", ""))[:12]),
        ("Evidence bundle", str(provenance.get("track_bundle_file", ""))),
        ("Study", str(provenance.get("study_name", ""))),
        ("Report", str(provenance.get("study_fingerprint_sha256", ""))[:12]),
    ]
    width, height = 760, 170
    boxes: list[str] = []
    for index, (title, detail) in enumerate(labels):
        x = 15 + index * 150
        if index:
            boxes.append(
                f'<path d="M {x-28} 78 L {x-5} 78" stroke="#4b5563" stroke-width="2" marker-end="url(#a)"/>'
            )
        boxes.append(
            f'<rect x="{x}" y="42" width="125" height="72" rx="8" fill="#f8fafc" stroke="#334155"/>'
            f'<text x="{x+62.5}" y="70" text-anchor="middle" font-size="13" font-weight="600">{escape(title)}</text>'
            f'<text x="{x+62.5}" y="94" text-anchor="middle" font-size="10" fill="#475569">{escape(detail)}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4b5563"/></marker></defs>'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="15" y="24" font-size="14" font-weight="600">Run provenance</text>'
        + "".join(boxes)
        + '</svg>'
    )
    (output / "provenance_graph.svg").write_text(svg, encoding="utf-8")


def _git_commit(project: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None
