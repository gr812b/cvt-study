"""Decomposed speed-gate confidence scoring and review recommendations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cvt_track_study.config.diagnostics import DiagnosticBag

from .settings import ReconstructionSettings


def score_speed_gates(
    passes: pd.DataFrame,
    events: pd.DataFrame,
    settings: ReconstructionSettings,
    diagnostics: DiagnosticBag,
) -> pd.DataFrame:
    weights = settings.weights
    total_weight = sum(weights.values())
    if not math.isclose(total_weight, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        diagnostics.warning(
            "GATE_CONFIDENCE_WEIGHTS_NORMALIZED",
            f"Gate-confidence weights sum to {total_weight:.6g}; they were normalized to one.",
        )
        weights = {key: value / total_weight for key, value in weights.items()}
    event_lookup = events.set_index("id")
    rows: list[dict[str, Any]] = []
    for event_id, all_passes in passes.groupby("event_id", sort=False):
        event = event_lookup.loc[event_id]
        valid = all_passes[all_passes["eligible"]].copy()
        count = len(valid)
        speeds = valid["entry_speed_mps"].dropna()
        pass_score = min(1.0, count / max(settings.target_pass_count, 1))
        if count:
            iqr = float(speeds.quantile(0.75) - speeds.quantile(0.25))
            repeatability = max(0.0, 1.0 - iqr / settings.repeatability_scale_mps)
            braking = float((valid["braking_drop_mps"] >= settings.braking_threshold_mps).mean())
            correlation = _safe_correlation(valid["entry_speed_mps"], valid["lap_median_speed_mps"])
            pace_independence = 1.0 - abs(correlation) if np.isfinite(correlation) else 0.5
        else:
            iqr = math.nan
            repeatability = 0.0
            braking = 0.0
            correlation = math.nan
            pace_independence = 0.0
        effective_coordinate_error = float(event["feature_start_effective_error_m"])
        coordinate_quality = max(
            0.0,
            1.0 - effective_coordinate_error / settings.maximum_map_error_m,
        )
        total_slowdown = (
            valid["approach_speed_mps"] - valid["event_min_speed_mps"]
            if count
            else pd.Series(dtype=float)
        )
        median_total_slowdown = float(total_slowdown.median()) if count else math.nan
        slowdown_fraction = (
            float((total_slowdown >= settings.braking_threshold_mps).mean())
            if count
            else math.nan
        )
        if count and slowdown_fraction >= 0.75 and median_total_slowdown >= 1.5:
            slowdown_signature = "strong"
        elif count and (slowdown_fraction >= 0.5 or median_total_slowdown >= 1.0):
            slowdown_signature = "moderate"
        elif count:
            slowdown_signature = "weak"
        else:
            slowdown_signature = "insufficient_passes"
        median_min_rel = float(valid["event_min_rel_m"].median()) if count else math.nan
        median_recovery_distance = (
            float(valid["recovery_distance_m"].median())
            if count and valid["recovery_distance_m"].notna().any()
            else math.nan
        )
        vehicle_medians = valid.groupby("vehicle_id")["entry_speed_mps"].median()
        if len(vehicle_medians) >= 2:
            spread = float(vehicle_medians.max() - vehicle_medians.min())
            cross_vehicle = max(0.0, 1.0 - spread / settings.vehicle_agreement_scale_mps)
            cross_vehicle_status = "measured"
        else:
            spread = math.nan
            cross_vehicle = 0.5
            cross_vehicle_status = "single_vehicle_neutral"
        components = {
            "pass_count": pass_score,
            "speed_repeatability": repeatability,
            "braking_evidence": braking,
            "pace_independence": pace_independence,
            "coordinate_quality": coordinate_quality,
            "cross_vehicle_agreement": cross_vehicle,
        }
        score = 100.0 * sum(weights[key] * components[key] for key in weights)
        candidate = bool(event["gate_candidate"])
        anchor_projection_error = float(event["anchor_projection_error_m"])
        if (
            anchor_projection_error > settings.maximum_map_error_m
            or effective_coordinate_error > settings.maximum_map_error_m
        ):
            recommendation = "must_fix"
        elif not candidate:
            recommendation = "not_a_candidate"
        elif count < settings.minimum_valid_passes:
            recommendation = "recommended_review"
        elif score >= settings.accept_score:
            recommendation = "accepted"
        elif score >= settings.review_score:
            recommendation = "recommended_review"
        else:
            recommendation = "rejected"
        reasons = _gate_reasons(components, count, event, settings, cross_vehicle_status)
        rows.append(
            {
                "event_id": event_id,
                "event_name": event["name"],
                "sequence": int(event["sequence"]),
                "response_group_id": event["response_group_id"],
                "gate_candidate": bool(event["gate_candidate"]),
                "valid_pass_count": count,
                "vehicle_count": int(valid["vehicle_id"].nunique()) if count else 0,
                "entry_speed_median_mps": float(speeds.median()) if count else math.nan,
                "entry_speed_mean_mps": float(speeds.mean()) if count else math.nan,
                "entry_speed_standard_deviation_mps": float(speeds.std(ddof=1)) if count > 1 else math.nan,
                "entry_speed_p10_mps": float(speeds.quantile(0.10)) if count else math.nan,
                "entry_speed_p90_mps": float(speeds.quantile(0.90)) if count else math.nan,
                "entry_speed_iqr_mps": iqr,
                "vehicle_median_spread_mps": spread,
                "pace_correlation": correlation,
                "median_approach_to_min_slowdown_mps": median_total_slowdown,
                "slowdown_lap_fraction": slowdown_fraction,
                "slowdown_signature": slowdown_signature,
                "median_event_min_rel_m": median_min_rel,
                "median_recovery_distance_m": median_recovery_distance,
                "pass_count_score": 100.0 * pass_score,
                "speed_repeatability_score": 100.0 * repeatability,
                "braking_evidence_score": 100.0 * braking,
                "pace_independence_score": 100.0 * pace_independence,
                "coordinate_effective_error_m": effective_coordinate_error,
                "coordinate_quality_score": 100.0 * coordinate_quality,
                "cross_vehicle_agreement_score": 100.0 * cross_vehicle,
                "overall_confidence_score": score,
                "recommendation": recommendation,
                "reasons": "; ".join(reasons),
                "cross_vehicle_status": cross_vehicle_status,
            }
        )
    return pd.DataFrame(rows).sort_values("sequence").reset_index(drop=True)

def _gate_reasons(
    components: Mapping[str, float],
    count: int,
    event: pd.Series,
    settings: ReconstructionSettings,
    cross_vehicle_status: str,
) -> list[str]:
    reasons: list[str] = []
    if count < settings.minimum_valid_passes:
        reasons.append(f"only {count} valid passes")
    if components["speed_repeatability"] < 0.5:
        reasons.append("entry speed varies strongly between laps")
    if components["braking_evidence"] < 0.5:
        reasons.append("braking before the event is not repeatable")
    if components["pace_independence"] < 0.5:
        reasons.append("entry speed follows overall lap pace")
    if components["coordinate_quality"] < 0.5:
        reasons.append("physical entry location is poorly constrained")
    if cross_vehicle_status != "measured":
        reasons.append("cross-vehicle agreement is not yet measured")
    if str(event["review_flags"]):
        reasons.extend(item.replace("_", " ") for item in str(event["review_flags"]).split(";") if item)
    if not reasons:
        reasons.append("repeatable entry state with supporting braking and map evidence")
    return reasons

def build_gate_review(
    evidence: pd.DataFrame,
    events: pd.DataFrame,
    settings: ReconstructionSettings,
) -> pd.DataFrame:
    projection = events[
        [
            "id",
            "anchor_s_m",
            "anchor_projection_error_m",
            "anchor_horizontal_uncertainty_m",
            "anchor_source",
            "review_flags",
            "feature_start_rel_m",
            "feature_start_source",
            "feature_start_provenance",
            "feature_start_projection_error_m",
            "feature_start_horizontal_uncertainty_m",
            "feature_start_effective_error_m",
            "feature_end_rel_m",
            "feature_end_source",
            "feature_end_provenance",
            "feature_end_projection_error_m",
            "feature_end_horizontal_uncertainty_m",
            "feature_end_effective_error_m",
            "entry_start_rel_m",
            "entry_end_rel_m",
            "recovery_limit_m",
            "source_event_ids",
            "source_event_names",
            "analysis_feature_type",
            "analysis_role",
        ]
    ].rename(columns={"id": "event_id"})
    review = evidence.merge(projection, on="event_id", how="left")
    priority_order = {
        "must_fix": 0,
        "recommended_review": 1,
        "rejected": 2,
        "accepted": 3,
        "not_a_candidate": 4,
    }
    review["review_priority"] = review["recommendation"].map(priority_order).fillna(9)
    review["suggested_action"] = review.apply(
        lambda row: _suggested_action(row, settings), axis=1
    )
    return review.sort_values(["review_priority", "sequence"]).reset_index(drop=True)

def _suggested_action(row: pd.Series, settings: ReconstructionSettings) -> str:
    if row["recommendation"] == "not_a_candidate":
        if row.get("analysis_role") == "lap_gate":
            return "No speed-gate action; this event only separates complete laps."
        return "No speed-gate action; this response was deliberately not nominated."
    if row["recommendation"] == "accepted":
        flags = [
            item.replace("_", " ")
            for item in str(row.get("review_flags", "")).split(";")
            if item
        ]
        if flags:
            return (
                "Gate evidence is accepted, but retain/review the declared assumptions: "
                + ", ".join(flags)
                + "."
            )
        return "No required action; retain the empirical entry-speed distribution."
    if row["recommendation"] == "must_fix":
        return "Verify the anchor coordinate against video/map evidence before using this event."
    if row["valid_pass_count"] < settings.minimum_valid_passes:
        return "Add or retain more complete laps before accepting this gate."
    if row["speed_repeatability_score"] < 50:
        return "Review whether the event belongs in a compound response group or has inconsistent entry placement."
    if row["braking_evidence_score"] < 50:
        return "Move the entry measurement window or verify that this event genuinely constrains speed."
    if row["cross_vehicle_status"] != "measured":
        return "Add a second vehicle/run when available; current score uses a neutral single-vehicle term."
    return "Review the evidence components and event geometry."

def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return math.nan
    left_rank = pair.iloc[:, 0].rank(method="average")
    right_rank = pair.iloc[:, 1].rank(method="average")
    return float(left_rank.corr(right_rank))
