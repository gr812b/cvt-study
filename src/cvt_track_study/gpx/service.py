"""Project-level GPX/FIT telemetry ingestion orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from cvt_track_study.config import ProjectError, ProjectLoader, ResolutionResult

from .export import export_ingestion_results
from .model import GPXIngestionResult
from .ingestion import TelemetryParseError
from .pipeline import ingest_configured_run


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
    raw_events = tuple(
        item
        for item in resolution.data.get("events", [])
        if isinstance(item, Mapping)
    )
    results: list[GPXIngestionResult] = []
    for raw in raw_runs:
        if not isinstance(raw, Mapping):
            continue
        run_id = str(raw.get("run_id", ""))
        if selected and run_id not in selected:
            continue
        try:
            results.append(
                ingest_configured_run(
                    raw,
                    runs_directory=resolution.paths.runs_file.parent,
                    track_config=track_config,
                    events=raw_events,
                )
            )
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
    output = (
        output_directory
        or resolution.paths.results_directory / "ingestion" / stamp
    )
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(resolution.paths.root),
        "selected_run_ids": [item.metadata.run_id for item in results],
        "run_count": len(results),
        "source_telemetry_sha256": {
            item.metadata.run_id: item.summary["source_sha256"]
            for item in results
        },
        "source_formats": {
            item.metadata.run_id: item.summary["source_format"]
            for item in results
        },
        "cleaned_point_counts": {
            item.metadata.run_id: int(item.summary["clean_positioned_point_count"])
            for item in results
        },
        "rejected_excursion_point_counts": {
            item.metadata.run_id: int(item.summary["isolated_excursion_point_count"])
            for item in results
        },
        "lap_time_reconstructed_runs": [
            item.metadata.run_id
            for item in results
            if bool(item.summary.get("lap_time_reconstruction_applied"))
        ],
        "reconstructed_lap_counts": {
            item.metadata.run_id: int(
                item.summary.get("reconstructed_lap_count", 0)
            )
            for item in results
        },
        "reconstructed_speed_counts": {
            item.metadata.run_id: int(
                item.summary.get("reconstructed_speed_count", 0)
            )
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
