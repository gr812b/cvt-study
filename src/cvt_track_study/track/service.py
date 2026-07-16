"""Project-level Phase 3 track reconstruction service."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping

from cvt_track_study.config.diagnostics import DiagnosticBag
from cvt_track_study.config.project import ProjectError, ProjectLoader
from cvt_track_study.gpx.model import GPXRunMetadata
from cvt_track_study.gpx.ingestion import TelemetryParseError, ingest_telemetry_run

from .export import export_track_build
from .model import TrackBuildResult
from .reconstruction import build_track_evidence


def build_project_track(
    project: str | Path,
    *,
    output_directory: Path | None = None,
) -> TrackBuildResult:
    resolution = ProjectLoader().resolve(project)
    if resolution.error_count:
        raise ProjectError(
            f"Project validation failed with {resolution.error_count} error(s); run cvt-study validate first."
        )
    ingestion_results = []
    for raw in resolution.data.get("runs", []):
        if not isinstance(raw, Mapping):
            continue
        metadata = GPXRunMetadata(
            run_id=str(raw["run_id"]),
            vehicle_id=str(raw["vehicle_id"]),
            driver_id=str(raw["driver_id"]),
            source_file=(resolution.paths.runs_file.parent / str(raw["file"])).resolve(),
            use_for_centreline=bool(raw["use_for_centreline"]),
            use_for_gate_evidence=bool(raw["use_for_gate_evidence"]),
        )
        try:
            ingestion_results.append(ingest_telemetry_run(metadata))
        except TelemetryParseError as exc:
            raise ProjectError(str(exc)) from exc
    if not ingestion_results:
        raise ProjectError("Track reconstruction requires at least one telemetry run.")
    ingestion_errors = [
        item.metadata.run_id for item in ingestion_results if item.error_count
    ]
    if ingestion_errors:
        raise ProjectError(
            "Telemetry ingestion reported fatal timing/data errors for run(s): "
            + ", ".join(ingestion_errors)
            + ". Run cvt-study ingest and resolve those errors before build-track."
        )

    diagnostics = DiagnosticBag(resolution.diagnostics)
    for ingestion in ingestion_results:
        diagnostics.extend(ingestion.diagnostics)
    track_config = resolution.data.get("track", {})
    raw_events = resolution.data.get("events", [])
    try:
        (
            centreline,
            laps,
            matched,
            profile,
            event_projection,
            response_features,
            event_passes,
            gate_evidence,
            gate_review,
        ) = build_track_evidence(
            tuple(ingestion_results), track_config, raw_events, diagnostics
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectError(f"Track reconstruction failed: {exc}") from exc

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_directory or resolution.paths.results_directory / "track_build" / stamp
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    metadata = {
        "schema_version": 1,
        "phase": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(resolution.paths.root),
        "run_ids": [item.metadata.run_id for item in ingestion_results],
        "source_telemetry_sha256": {
            item.metadata.run_id: item.summary["source_sha256"]
            for item in ingestion_results
        },
        "source_formats": {
            item.metadata.run_id: item.summary["source_format"]
            for item in ingestion_results
        },
        "reference_lap_id": int(laps.loc[laps["reference_lap"], "lap_id"].iloc[0]),
        "track_length_m": centreline.length_m,
        "complete_lap_count": len(laps),
        "valid_lap_count": int(laps["analysis_valid"].sum()),
        "accepted_gate_count": int((gate_review["recommendation"] == "accepted").sum()),
        "grade_force_enabled": False,
    }
    result = TrackBuildResult(
        resolution=resolution,
        ingestion_results=tuple(ingestion_results),
        centreline=centreline,
        laps=laps,
        matched_points=matched,
        track_profile=profile,
        event_projection=event_projection,
        response_features=response_features,
        event_passes=event_passes,
        gate_evidence=gate_evidence,
        gate_review=gate_review,
        diagnostics=diagnostics.items,
        metadata=metadata,
    )
    export_track_build(output, result)
    return replace(result, output_directory=output)
