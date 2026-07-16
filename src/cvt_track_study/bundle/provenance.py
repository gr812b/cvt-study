"""Portable, track-scoped provenance for versioned bundles."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cvt_track_study.config.merge import ProvenanceStep
from cvt_track_study.track.model import TrackBuildResult

from .serialization import json_safe


def track_provenance(result: TrackBuildResult) -> dict[str, Any]:
    root = result.resolution.paths.root
    relevant = {
        path: steps
        for path, steps in result.resolution.provenance.items()
        if _is_track_provenance_path(path)
    }
    source_paths: dict[Path, str] = {
        result.resolution.paths.project_file.resolve(): "project",
        result.resolution.paths.track_file.resolve(): "project",
        result.resolution.paths.runs_file.resolve(): "project",
        result.resolution.paths.events_file.resolve(): "project",
    }
    for steps in relevant.values():
        for step in steps:
            source_path = _extract_source_path(step.source)
            if source_path is not None and source_path.exists():
                source_paths.setdefault(
                    source_path.resolve(),
                    "project" if _is_within(source_path, root) else "external_profile",
                )
    source_files = sorted(
        (
            {
                "scope": scope,
                "path": (
                    str(path.relative_to(root))
                    if _is_within(path, root)
                    else path.name
                ),
                "sha256": _sha256_file(path),
            }
            for path, scope in source_paths.items()
        ),
        key=lambda item: (item["scope"], item["path"], item["sha256"]),
    )
    gpx_sources = sorted(
        (
            {
                "run_id": item.metadata.run_id,
                "vehicle_id": item.metadata.vehicle_id,
                "driver_id": item.metadata.driver_id,
                "path": str(item.metadata.source_file.relative_to(root)),
                "sha256": str(item.summary["source_sha256"]),
            }
            for item in result.ingestion_results
        ),
        key=lambda item: (
            item["run_id"],
            item["vehicle_id"],
            item["driver_id"],
            item["path"],
        ),
    )
    return {
        "scope": (
            "Track reconstruction inputs only. Vehicle physical parameters and study "
            "definitions are deliberately excluded so one bundle can be reused across vehicles."
        ),
        "source_files": source_files,
        "source_gpx": gpx_sources,
        "resolved_configuration_provenance": {
            path: [_portable_step(step, root) for step in steps]
            for path, steps in sorted(relevant.items())
        },
        "track_build": json_safe(
            {
                key: value
                for key, value in result.metadata.items()
                if key
                not in {
                    "created_utc",
                    "project_root",
                    "track_bundle_schema_version",
                    "track_bundle_sha256",
                    "track_bundle_content_fingerprint_sha256",
                }
            }
        ),
    }


def _is_track_provenance_path(path: str) -> bool:
    return (
        path == "runs"
        or path == "events"
        or path.startswith("track.")
        or path
        in {
            "project.name",
            "project.schema_version",
            "project.track",
            "project.runs",
            "project.events",
        }
    )


def _extract_source_path(source: str) -> Path | None:
    if source.endswith(")") and " (" in source:
        candidate = Path(source.rsplit(" (", 1)[1][:-1])
        return candidate if candidate.is_absolute() else None
    candidate = Path(source)
    return candidate if candidate.is_absolute() else None


def _portable_step(step: ProvenanceStep, root: Path) -> dict[str, Any]:
    return {
        "layer": step.layer,
        "source": _portable_source(step.source, root),
        "action": step.action,
        "value": _portable_value(step.value, root),
    }


def _portable_source(source: str, root: Path) -> str:
    if source.endswith(")") and " (" in source:
        prefix, raw_path = source.rsplit(" (", 1)
        candidate = Path(raw_path[:-1])
        if candidate.is_absolute():
            portable = str(candidate.relative_to(root)) if _is_within(candidate, root) else candidate.name
            return f"{prefix} ({portable})"
    candidate = Path(source)
    if candidate.is_absolute():
        return str(candidate.relative_to(root)) if _is_within(candidate, root) else candidate.name
    return source


def _portable_value(value: Any, root: Path) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _portable_value(child, root) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_value(child, root) for child in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            return str(candidate.relative_to(root)) if _is_within(candidate, root) else candidate.name
    return json_safe(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
