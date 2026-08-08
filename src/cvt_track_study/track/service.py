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
from .family import (
    RouteFamilyEvidenceBuild,
    RouteFamilySettings,
    build_track_family_evidence,
)
from .model import TrackBuildResult


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
    try:
        family = build_track_family_evidence(
            tuple(ingestion_results),
            track_config,
            raw_events,
            diagnostics,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectError(f"Track reconstruction failed: {exc}") from exc

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        output_directory
        or resolution.paths.results_directory / "track_build" / stamp
    )
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()

    member_results: dict[str, TrackBuildResult] = {}
    for route_id, evidence in family.routes.items():
        metadata = _route_metadata(
            resolution=resolution,
            ingestion_results=tuple(ingestion_results),
            evidence=evidence,
            family=family,
            route_id=route_id,
        )
        member_results[route_id] = TrackBuildResult(
            resolution=resolution,
            ingestion_results=tuple(ingestion_results),
            centreline=evidence.centreline,
            laps=evidence.laps,
            matched_points=evidence.matched_points,
            track_profile=evidence.track_profile,
            event_projection=evidence.event_projection,
            response_features=evidence.response_features,
            event_passes=evidence.event_passes,
            gate_evidence=evidence.gate_evidence,
            gate_review=evidence.gate_review,
            rejected_map_points=evidence.rejected_map_points,
            route_variant_summary=family.route_variant_summary,
            route_variant_pairwise=family.route_variant_pairwise,
            route_variant_id=route_id,
            nominal_route_variant_id=family.nominal_route_variant_id,
            shared_gate_evidence=family.shared_gate_evidence,
            event_route_applicability=family.event_route_applicability,
            course_cases=family.course_cases,
            branch_summary=family.branch_summary,
            event_route_sections=family.event_route_sections,
            diagnostics=diagnostics.items,
            metadata=metadata,
        )

    nominal = member_results[family.nominal_route_variant_id]
    result = replace(
        nominal,
        route_family_members=member_results,
        output_directory=output,
    )
    export_track_build(output, result)
    return result


def _route_metadata(
    *,
    resolution: Any,
    ingestion_results: tuple[Any, ...],
    evidence: Any,
    family: RouteFamilyEvidenceBuild,
    route_id: str,
) -> dict[str, Any]:
    laps = evidence.laps
    gate_review = evidence.gate_review
    route_summary = family.route_variant_summary
    route_settings = evidence.route_variant_settings
    family_settings = RouteFamilySettings.from_mapping(
        resolution.data.get("track", {})
    )
    topology_p95_limit = (
        family_settings.topology_same_route_p95_distance_m
        if family_settings.topology_same_route_p95_distance_m is not None
        else 1.5 * route_settings.same_variant_p95_distance_m
    )
    topology_divergent_limit = (
        family_settings.topology_maximum_divergent_fraction
        if family_settings.topology_maximum_divergent_fraction is not None
        else route_settings.maximum_divergent_fraction
    )
    topology_length_limit = (
        family_settings.topology_maximum_length_relative_difference
        if family_settings.topology_maximum_length_relative_difference is not None
        else route_settings.maximum_length_relative_difference
    )
    reference = laps.loc[
        laps["reference_lap"].astype(bool), "lap_id"
    ]
    route_variant_metadata = {
        "enabled": bool(route_settings.enabled),
        "family_enabled": len(family.routes) > 1,
        "selection_policy": route_settings.selection,
        "reference_run_id": route_settings.reference_run_id,
        "selected_variant_id": route_id,
        "nominal_variant_id": family.nominal_route_variant_id,
        "supported_variant_count": int(
            route_summary.get("supported", pd.Series(dtype=bool)).astype(bool).sum()
        ),
        "exported_variant_count": int(len(family.routes)),
        "detected_variant_count": int(len(route_summary)),
        "pairwise_comparison_count": int(len(family.route_variant_pairwise)),
        "thresholds": {
            "minimum_supported_laps": route_settings.minimum_supported_laps,
            "sample_spacing_m": route_settings.sample_spacing_m,
            "same_variant_p95_distance_m": route_settings.same_variant_p95_distance_m,
            "divergence_distance_m": route_settings.divergence_distance_m,
            "maximum_divergent_fraction": route_settings.maximum_divergent_fraction,
            "maximum_length_relative_difference": route_settings.maximum_length_relative_difference,
            "maximum_within_variant_length_deviation_fraction": (
                route_settings.maximum_within_variant_length_deviation_fraction
            ),
        },
        "topology_merge": {
            "enabled": bool(family_settings.merge_line_clusters),
            "same_route_p95_distance_m": topology_p95_limit,
            "maximum_divergent_fraction": topology_divergent_limit,
            "maximum_length_relative_difference": topology_length_limit,
            "attachment_ambiguity_ratio": (
                family_settings.topology_attachment_ambiguity_ratio
            ),
            "method": (
                "supported strict-cluster components with conservative "
                "attachment of isolated line clusters"
            ),
        },
        "variants": _json_records(route_summary),
        "pairwise_comparisons": _json_records(family.route_variant_pairwise),
        "course_cases": _json_records(family.course_cases),
        "branch_network": {
            "enabled": bool(not family.branch_summary.empty),
            "branch_count": int(len(family.branch_summary)),
            "branches": _json_records(family.branch_summary),
            "event_sections": _json_records(family.event_route_sections),
        },
        "interpretation": (
            "Strict whole-lap clusters classify which branch a traversal used. "
            "Supported variants are then represented as one shared closed course "
            "with local divergence/re-merge corridors. Shared-section gate evidence "
            "pools compatible laps from every supported branch; only branch-local "
            "events are restricted to laps that used that branch."
        ),
    }
    return {
        "schema_version": 3,
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
        "lap_time_reconstructed_runs": [
            item.metadata.run_id
            for item in ingestion_results
            if bool(item.summary.get("lap_time_reconstruction_applied"))
        ],
        "reconstructed_speed_evidence_is_supplemental": True,
        "reference_lap_id": int(reference.iloc[0]) if not reference.empty else None,
        "centreline_method": "iterative_robust_multi_lap_consensus_per_route",
        "centreline_input_lap_count": int(laps["centreline_included"].sum()),
        "centreline_excluded_lap_count": int(laps["consensus_excluded"].sum()),
        "centreline_consensus_iteration_count": int(
            laps["consensus_iteration_count"].max()
        ),
        "track_length_m": evidence.centreline.length_m,
        "complete_lap_count": len(laps),
        "valid_lap_count": int(laps["analysis_valid"].sum()),
        "gate_evidence_unique_lap_count": int(
            evidence.event_passes.loc[
                evidence.event_passes["eligible"].astype(bool), "lap_id"
            ].nunique()
        ) if not evidence.event_passes.empty else 0,
        "accepted_gate_count": int(
            (gate_review["recommendation"] == "accepted").sum()
        ),
        "pre_lap_rejected_point_count": sum(
            len(item.rejected_points) for item in ingestion_results
        ),
        "post_map_rejected_point_count": len(evidence.rejected_map_points),
        "grade_force_enabled": False,
        "route_variant_id": route_id,
        "nominal_route_variant_id": family.nominal_route_variant_id,
        "route_family_manifest": "route_family_manifest.json",
        "route_variant_detection": route_variant_metadata,
    }


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
