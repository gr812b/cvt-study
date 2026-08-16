"""Orchestrate creation of the self-contained Phase 4 track bundle."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cvt_track_study import __version__
from cvt_track_study.track.model import TrackBuildResult
from cvt_track_study.track.settings import ReconstructionSettings
from cvt_track_study.track.grade import screen_grade_materiality

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
    """Convert one route member into the stable simulator/evidence boundary."""

    length = float(result.centreline.length_m)
    track_cfg = result.resolution.data.get("track", {})
    settings = ReconstructionSettings.from_mapping(track_cfg)
    physical_features = physical_feature_contracts(result.event_projection, length)
    response_groups = response_group_contracts(result.response_features, length)
    obstacle_models_ready = all(
        feature.get("obstacle_model", {}).get("status") == "declared"
        for feature in physical_features
    )
    grade_screen = screen_grade_materiality(result.track_profile)
    speed_gates = _simulatable_speed_gates(speed_gate_contracts(result, length))
    simulation_contract = {
        "track_length_m": length,
        "global_speed_guardrail_mps": settings.maximum_reasonable_speed_mps,
        "global_speed_guardrail_semantics": (
            "very permissive non-fitted fallback ceiling; reused from the declared "
            "maximum reasonable telemetry speed so no simulated world is globally uncapped"
        ),
        "grade_force_enabled": False,
        "grade_screen": grade_screen,
        "capabilities": {
            "speed_gates_ready": bool(
                speed_gates or np.isfinite(settings.maximum_reasonable_speed_mps)
            ),
            "obstacle_models_ready": obstacle_models_ready,
            "uncertainty_roles_ready": obstacle_models_ready,
            "grade_force_ready": False,
        },
        "centreline": centreline_contract(result),
        "observed_profile": observed_profile_contract(result.track_profile),
        "physical_features": physical_features,
        "response_groups": response_groups,
        "speed_gates": speed_gates,
    }
    route_family = {
        "route_variant_id": result.route_variant_id,
        "nominal_route_variant_id": result.nominal_route_variant_id,
        "is_nominal": result.route_variant_id == result.nominal_route_variant_id,
        "aggregation_policy": "unweighted_equal_course_cases",
        "course_cases": records(result.course_cases),
        "shared_gate_evidence": records(result.shared_gate_evidence),
        "event_route_applicability": records(result.event_route_applicability),
    }
    bundle: dict[str, Any] = {
        "format": TRACK_BUNDLE_FORMAT,
        "schema_version": CURRENT_TRACK_BUNDLE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {"package": "cvt-track-study", "version": __version__},
        "identity": {
            "project_name": str(
                result.resolution.data.get("project", {}).get("name", "")
            ),
            "track_name": str(track_cfg.get("name", "")),
            "route_variant_id": result.route_variant_id,
            "nominal_route_variant_id": result.nominal_route_variant_id,
            "closed_course": bool(track_cfg.get("closed_course", True)),
            "surface_class": str(track_cfg.get("surface_class", "unspecified")),
        },
        "coordinate_contract": {
            "coordinate": "s",
            "unit": "m",
            "origin": "lap_gate_projected_to_this_route_centreline",
            "direction": "recorded_driving_direction",
            "domain": {"minimum": 0.0, "maximum": length},
            "interval_convention": (
                "start inclusive; end exclusive; intervals follow driving direction; "
                "wraps_start_finish marks end_s_m < start_s_m"
            ),
        },
        "route_family_contract": route_family,
        "simulation_contract": simulation_contract,
        "evidence": {
            "lap_summary": {
                "complete_lap_count": int(len(result.laps)),
                "valid_lap_count": int(result.laps["analysis_valid"].sum()),
                "reference_lap_id": int(
                    result.laps.loc[
                        result.laps["reference_lap"], "lap_id"
                    ].iloc[0]
                ),
                "records": records(result.laps),
            },
            "gate_confidence_method": {
                "method_version": "1.4.0-source-balanced-hierarchical-gates",
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
            "gate_pooling_contract": {
                "unit": "physical_event_id",
                "deduplication": "one contribution per lap per physical event",
                "shared_course_policy": (
                    "one deduplicated empirical gate contract is scored once and "
                    "reused across every supported traversal"
                ),
                "branch_policy": (
                    "events whose analysis window overlaps a sustained divergence "
                    "corridor retain route-specific evidence"
                ),
                "compatibility": (
                    "local route projection must satisfy bracketed event-window "
                    "coverage, projection error, and local direction checks"
                ),
            },
            "grade_materiality_screen": grade_screen,
            "event_passes": records(result.event_passes),
            "review_records": records(result.gate_review),
        },
        "uncertainty_contract": {
            "geometry": {
                "representation": (
                    "separate route bundles; declared horizontal and extent "
                    "uncertainty per physical feature"
                ),
                "propagation_status": "route_cases_ready_geometry_uncertainty_still_stored",
            },
            "gate_speed": {
                "representation": (
                    "target-vehicle empirical hard/response samples plus deterministic "
                    "conservative guardrails and a global fallback ceiling"
                ),
                "propagation_status": "ready_for_target_vehicle_paired_sampling",
            },
            "obstacle_models": {
                "representation": (
                    "uncertainty-aware model choice and parameters per physical feature"
                ),
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
    return _finalize_bundle(bundle)


def _simulatable_speed_gates(
    gates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep evidence-backed gates and deterministic conservative guardrails.

    Review-only rows with no samples remain evidence, not simulator constraints. A
    conservative guardrail is different: it is explicitly a policy ceiling rather
    than a fitted observation, so it remains simulatable even if a future sparse
    reconstruction has no empirical identity to attach to it.
    """

    output: list[dict[str, Any]] = []
    for gate in gates:
        distribution = gate.get("target_speed_distribution", {})
        samples = (
            distribution.get("samples", [])
            if isinstance(distribution, dict)
            else []
        )
        deterministic_guardrail = (
            str(gate.get("enforcement_class", "")) == "conservative_guardrail"
            and bool(gate.get("active_by_default", False))
            and isinstance(distribution, dict)
            and isinstance(distribution.get("summary"), dict)
            and distribution["summary"].get("median_mps") is not None
        )
        if samples or deterministic_guardrail:
            output.append(gate)
    return output


def _finalize_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Sanitize non-finite evidence values before strict JSON fingerprinting."""

    safe_bundle = json_safe(bundle)
    safe_bundle["content_fingerprint_sha256"] = content_fingerprint(safe_bundle)
    return safe_bundle


def export_bundle_for_track_build(
    directory: Path, result: TrackBuildResult
) -> TrackBundle:
    return write_track_bundle(
        directory / "track_bundle.json", build_track_bundle(result)
    )
