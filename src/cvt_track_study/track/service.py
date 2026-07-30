"""Project-level Phase 3 track reconstruction service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cvt_track_study.config import ProjectError, ProjectLoader
from cvt_track_study.config.diagnostics import DiagnosticBag
from cvt_track_study.gpx.ingestion import TelemetryParseError
from cvt_track_study.gpx.pipeline import ingest_configured_run

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
            f"Project validation failed with {resolution.error_count} error(s); "
            "run cvt-study validate first."
        )
    track_config = resolution.data.get("track", {})
    raw_events = tuple(
        item
        for item in resolution.data.get("events", [])
        if isinstance(item, Mapping)
    )
    ingestion_results = []
    for raw in resolution.data.get("runs", []):
        if not isinstance(raw, Mapping):
            continue
        try:
            ingestion_results.append(
                ingest_configured_run(
                    raw,
                    runs_directory=resolution.paths.runs_file.parent,
                    track_config=track_config,
                    events=raw_events,
                )
            )
        except (TelemetryParseError, ValueError) as exc:
            raise ProjectError(str(exc)) from exc
    if not ingestion_results:
        raise ProjectError(
            "Track reconstruction requires at least one telemetry run."
        )
    ingestion_errors = [
        item.metadata.run_id
        for item in ingestion_results
        if item.error_count
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
    try:
        evidence = build_track_evidence(
            tuple(ingestion_results),
            track_config,
            raw_events,
            diagnostics,
        )
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
            rejected_map_points,
        ) = evidence
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectError(f"Track reconstruction failed: {exc}") from exc

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        output_directory
        or resolution.paths.results_directory / "track_build" / stamp
    )
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()

    route_summary = evidence.route_variant_summary
    route_pairwise = evidence.route_variant_pairwise
    route_settings = evidence.route_variant_settings
    route_variant_metadata = {
        "enabled": bool(route_settings.enabled),
        "selection_policy": route_settings.selection,
        "reference_run_id": route_settings.reference_run_id,
        "selected_variant_id": evidence.selected_route_variant_id,
        "supported_variant_count": int(
            route_summary.get("supported", pd.Series(dtype=bool)).astype(bool).sum()
        ),
        "detected_variant_count": int(len(route_summary)),
        "pairwise_comparison_count": int(len(route_pairwise)),
        "thresholds": {
            "minimum_supported_laps": route_settings.minimum_supported_laps,
            "sample_spacing_m": route_settings.sample_spacing_m,
            "same_variant_p95_distance_m": (
                route_settings.same_variant_p95_distance_m
            ),
            "divergence_distance_m": route_settings.divergence_distance_m,
            "maximum_divergent_fraction": (
                route_settings.maximum_divergent_fraction
            ),
            "maximum_length_relative_difference": (
                route_settings.maximum_length_relative_difference
            ),
            "maximum_within_variant_length_deviation_fraction": (
                route_settings.maximum_within_variant_length_deviation_fraction
            ),
        },
        "variants": _json_records(route_summary),
        "pairwise_comparisons": _json_records(route_pairwise),
        "interpretation": (
            "Supported alternate route variants are retained as valid route evidence "
            "but excluded from the selected route's centreline and speed-gate evidence. "
            "They are not averaged together and are not called telemetry errors."
        ),
    }

    metadata = {
        "schema_version": 2,
        "phase": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(resolution.paths.root),
        "run_ids": [
            item.metadata.run_id for item in ingestion_results
        ],
        "source_telemetry_sha256": {
            item.metadata.run_id: item.summary["source_sha256"]
            for item in ingestion_results
        },
        "source_formats": {
            item.metadata.run_id: item.summary["source_format"]
            for item in ingestion_results
        },
        "lap_time_reconstructed_runs": [
            item.metadata.run_id
            for item in ingestion_results
            if bool(item.summary.get("lap_time_reconstruction_applied"))
        ],
        "reconstructed_speed_evidence_is_supplemental": True,
        "reference_lap_id": int(
            laps.loc[laps["reference_lap"], "lap_id"].iloc[0]
        ),
        "centreline_method": "iterative_robust_multi_lap_consensus",
        "centreline_input_lap_count": int(
            laps["centreline_included"].sum()
        ),
        "centreline_excluded_lap_count": int(
            laps["consensus_excluded"].sum()
        ),
        "centreline_consensus_iteration_count": int(
            laps["consensus_iteration_count"].max()
        ),
        "track_length_m": centreline.length_m,
        "complete_lap_count": len(laps),
        "valid_lap_count": int(laps["analysis_valid"].sum()),
        "accepted_gate_count": int(
            (gate_review["recommendation"] == "accepted").sum()
        ),
        "pre_lap_rejected_point_count": sum(
            len(item.rejected_points) for item in ingestion_results
        ),
        "post_map_rejected_point_count": len(rejected_map_points),
        "grade_force_enabled": False,
        "route_variant_detection": route_variant_metadata,
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
        rejected_map_points=rejected_map_points,
        route_variant_summary=route_summary,
        route_variant_pairwise=route_pairwise,
        diagnostics=diagnostics.items,
        metadata=metadata,
    )
    export_track_build(output, result)
    return replace(result, output_directory=output)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            row[str(key)] = value
        records.append(row)
    return records
