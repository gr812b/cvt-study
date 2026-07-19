"""Project-level GPX/FIT telemetry ingestion orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from cvt_track_study.config import ProjectError, ProjectLoader, ResolutionResult

from .cleanup import apply_telemetry_cleanup
from .export import export_ingestion_results
from .model import GPXIngestionResult, GPXRunMetadata
from .ingestion import TelemetryParseError, ingest_telemetry_run


def ingest_project(
    project: str | Path,
    *,
    run_ids: Iterable[str] = (),
    output_directory: Path | None = None,
) -> tuple[ResolutionResult, tuple[GPXIngestionResult, ...], Path]:
    resolution = ProjectLoader().resolve(project)
    if resolution.error_count:
        raise ProjectError(
            f"Project validation failed with {resolution.error_count} error(s); "
            "run cvt-study validate first."
        )
    selected = set(run_ids)
    raw_runs = resolution.data.get("runs", [])
    track_config = resolution.data.get("track", {})
    results: list[GPXIngestionResult] = []
    for raw in raw_runs:
        if not isinstance(raw, Mapping):
            continue
        run_id = str(raw.get("run_id", ""))
        if selected and run_id not in selected:
            continue
        source = (resolution.paths.runs_file.parent / str(raw["file"])).resolve()
        metadata = GPXRunMetadata(
            run_id=run_id,
            vehicle_id=str(raw["vehicle_id"]),
            driver_id=str(raw["driver_id"]),
            source_file=source,
            use_for_centreline=bool(raw["use_for_centreline"]),
            use_for_gate_evidence=bool(raw["use_for_gate_evidence"]),
        )
        try:
            parsed = ingest_telemetry_run(metadata)
            results.append(apply_telemetry_cleanup(parsed, track_config))
        except (TelemetryParseError, ValueError) as exc:
            raise ProjectError(str(exc)) from exc
    if selected:
        found = {item.metadata.run_id for item in results}
        missing = sorted(selected - found)
        if missing:
            raise ProjectError("Unknown run id(s): " + ", ".join(missing))
    if not results:
        raise ProjectError("No telemetry runs were selected for ingestion.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_directory or resolution.paths.results_directory / "ingestion" / stamp
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(resolution.paths.root),
        "selected_run_ids": [item.metadata.run_id for item in results],
        "run_count": len(results),
        "source_telemetry_sha256": {
            item.metadata.run_id: item.summary["source_sha256"] for item in results
        },
        "source_formats": {
            item.metadata.run_id: item.summary["source_format"] for item in results
        },
        "cleaned_point_counts": {
            item.metadata.run_id: int(item.summary["clean_positioned_point_count"])
            for item in results
        },
        "rejected_excursion_point_counts": {
            item.metadata.run_id: int(item.summary["isolated_excursion_point_count"])
            for item in results
        },
    }
    export_ingestion_results(
        output,
        results,
        manifest=manifest,
        export_configuration=resolution.export,
    )
    return resolution, tuple(results), output
