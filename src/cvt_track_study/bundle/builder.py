"""Orchestrate creation of the self-contained Phase 4 track bundle."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cvt_track_study import __version__
from cvt_track_study.track.model import TrackBuildResult
from cvt_track_study.track.settings import ReconstructionSettings

from .canonical import content_fingerprint
from .gates import speed_gate_contracts
from .geometry import (
    centreline_contract,
    observed_profile_contract,
    physical_feature_contracts,
    response_group_contracts,
)
from .io import write_track_bundle
from .model import CURRENT_TRACK_BUNDLE_SCHEMA, TRACK_BUNDLE_FORMAT, TrackBundle
from .provenance import track_provenance
from .serialization import json_safe, records


def build_track_bundle(result: TrackBuildResult) -> dict[str, Any]:
    """Convert a Phase 3 result into the stable simulator/evidence boundary."""

    length = float(result.centreline.length_m)
    track_cfg = result.resolution.data.get("track", {})
    settings = ReconstructionSettings.from_mapping(track_cfg)
    physical_features = physical_feature_contracts(result.event_projection, length)
    response_groups = response_group_contracts(result.response_features, length)
    obstacle_models_ready = all(
        feature.get("obstacle_model", {}).get("status") == "declared"
        for feature in physical_features
    )
    simulation_contract = {
        "track_length_m": length,
        "grade_force_enabled": False,
        "capabilities": {
            "speed_gates_ready": bool(
                (result.gate_review["recommendation"] == "accepted").any()
            ),
            "obstacle_models_ready": obstacle_models_ready,
            "uncertainty_roles_ready": obstacle_models_ready,
            "grade_force_ready": False,
        },
        "centreline": centreline_contract(result),
        "observed_profile": observed_profile_contract(result.track_profile),
        "physical_features": physical_features,
        "response_groups": response_groups,
        "speed_gates": speed_gate_contracts(result, length),
    }
    bundle: dict[str, Any] = {
        "format": TRACK_BUNDLE_FORMAT,
        "schema_version": CURRENT_TRACK_BUNDLE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {"package": "cvt-track-study", "version": __version__},
        "identity": {
            "project_name": str(result.resolution.data.get("project", {}).get("name", "")),
            "track_name": str(track_cfg.get("name", "")),
            "closed_course": bool(track_cfg.get("closed_course", True)),
            "surface_class": str(track_cfg.get("surface_class", "unspecified")),
        },
        "coordinate_contract": {
            "coordinate": "s",
            "unit": "m",
            "origin": "lap_gate_projected_to_reference_centreline",
            "direction": "recorded_driving_direction",
            "domain": {"minimum": 0.0, "maximum": length},
            "interval_convention": (
                "start inclusive; end exclusive; intervals follow driving direction; "
                "wraps_start_finish marks end_s_m < start_s_m"
            ),
        },
        "simulation_contract": simulation_contract,
        "evidence": {
            "lap_summary": {
                "complete_lap_count": int(len(result.laps)),
                "valid_lap_count": int(result.laps["analysis_valid"].sum()),
                "reference_lap_id": int(
                    result.laps.loc[result.laps["reference_lap"], "lap_id"].iloc[0]
                ),
                "records": records(result.laps),
            },
            "gate_confidence_method": {
                "method_version": "1.0.0",
                "component_scale": "0_to_100",
                "overall_scale": "0_to_100",
                "weights": dict(settings.weights),
                "thresholds": {
                    "minimum_valid_passes": settings.minimum_valid_passes,
                    "target_pass_count": settings.target_pass_count,
                    "braking_threshold_mps": settings.braking_threshold_mps,
                    "repeatability_scale_mps": settings.repeatability_scale_mps,
                    "vehicle_agreement_scale_mps": settings.vehicle_agreement_scale_mps,
                    "accept_score": settings.accept_score,
                    "review_score": settings.review_score,
                },
                "records": records(result.gate_evidence),
            },
            "event_passes": records(result.event_passes),
            "review_records": records(result.gate_review),
        },
        "uncertainty_contract": {
            "geometry": {
                "representation": "declared horizontal and extent uncertainty per physical feature",
                "propagation_status": "carried_as_evidence_not_yet_sampled",
            },
            "gate_speed": {
                "representation": "empirical eligible-lap entry-speed samples",
                "propagation_status": "ready_for_paired_sampling",
            },
            "obstacle_models": {
                "representation": "uncertainty-aware model choice and parameters per physical feature",
                "propagation_status": (
                    "ready_for_role_separated_sampling"
                    if obstacle_models_ready
                    else "not_ready"
                ),
            },
            "elevation": {
                "representation": "empirical p10/median/p90 profile from valid laps",
                "propagation_status": "stored_only_grade_force_disabled",
            },
        },
        "provenance": track_provenance(result),
    }
    bundle["content_fingerprint_sha256"] = content_fingerprint(bundle)
    return json_safe(bundle)


def export_bundle_for_track_build(directory: Path, result: TrackBuildResult) -> TrackBundle:
    return write_track_bundle(directory / "track_bundle.json", build_track_bundle(result))
