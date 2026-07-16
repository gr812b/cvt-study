"""Decomposed speed-gate confidence scoring and review recommendations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

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
        sustained = _sustained_response_statistics(valid, settings)
        sustained_qualified = bool(
            recommendation == "accepted" and sustained["evidence_pass"]
        )
        sustained_status = (
            "accepted"
            if sustained_qualified
            else "entry_only_fallback"
            if recommendation == "accepted"
            else "not_evaluated"
        )
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
                "event_min_speed_median_mps": sustained["speed_median_mps"],
                "event_min_speed_mean_mps": sustained["speed_mean_mps"],
                "event_min_speed_standard_deviation_mps": sustained[
                    "speed_standard_deviation_mps"
                ],
                "event_min_speed_p10_mps": sustained["speed_p10_mps"],
                "event_min_speed_p90_mps": sustained["speed_p90_mps"],
                "event_min_speed_iqr_mps": sustained["speed_iqr_mps"],
                "event_min_location_iqr_m": sustained["location_iqr_m"],
                "sustained_slowdown_success_fraction": sustained[
                    "slowdown_success_fraction"
                ],
                "sustained_slowdown_p_value": sustained["slowdown_p_value"],
                "sustained_leave_one_out_max_speed_shift_mps": sustained[
                    "leave_one_out_max_speed_shift_mps"
                ],
                "sustained_leave_one_out_max_location_shift_m": sustained[
                    "leave_one_out_max_location_shift_m"
                ],
                "sustained_confidence_score": sustained["confidence_score"],
                "sustained_gate_qualified": sustained_qualified,
                "sustained_gate_status": sustained_status,
                "sustained_gate_reason": sustained["reason"],
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


def _sustained_response_statistics(
    valid: pd.DataFrame, settings: ReconstructionSettings
) -> dict[str, Any]:
    """Qualify a response-minimum gate using repeatability and leave-one-out stability."""

    required = valid[
        ["entry_speed_mps", "event_min_speed_mps", "event_min_rel_m"]
    ].dropna()
    count = len(required)
    empty = {
        "speed_median_mps": math.nan,
        "speed_mean_mps": math.nan,
        "speed_standard_deviation_mps": math.nan,
        "speed_p10_mps": math.nan,
        "speed_p90_mps": math.nan,
        "speed_iqr_mps": math.nan,
        "location_iqr_m": math.nan,
        "slowdown_success_fraction": math.nan,
        "slowdown_p_value": math.nan,
        "leave_one_out_max_speed_shift_mps": math.nan,
        "leave_one_out_max_location_shift_m": math.nan,
        "confidence_score": 0.0,
        "evidence_pass": False,
        "reason": "insufficient complete event-response passes",
    }
    if count < settings.minimum_valid_passes:
        return empty

    speeds = required["event_min_speed_mps"].astype(float)
    locations = required["event_min_rel_m"].astype(float)
    slowdown = required["entry_speed_mps"].astype(float) - speeds
    successes = int((slowdown >= settings.braking_threshold_mps).sum())
    success_fraction = successes / count
    p_value = float(binomtest(successes, count, 0.5, alternative="greater").pvalue)
    speed_iqr = float(speeds.quantile(0.75) - speeds.quantile(0.25))
    location_iqr = float(locations.quantile(0.75) - locations.quantile(0.25))
    loo_speed = _leave_one_out_median_shift(speeds.to_numpy(float))
    loo_location = _leave_one_out_median_shift(locations.to_numpy(float))
    maximum_location_iqr = max(5.0, 2.0 * settings.profile_spacing_m)
    maximum_loo_speed_shift = max(0.25, 0.25 * settings.repeatability_scale_mps)
    maximum_loo_location_shift = max(2.5, settings.profile_spacing_m)
    checks = {
        "repeatable_slowdown": success_fraction >= 0.80 and p_value <= 0.05,
        "speed_repeatability": speed_iqr <= settings.repeatability_scale_mps,
        "location_repeatability": location_iqr <= maximum_location_iqr,
        "leave_one_out_speed_stability": loo_speed <= maximum_loo_speed_shift,
        "leave_one_out_location_stability": loo_location <= maximum_loo_location_shift,
    }
    failed = [name.replace("_", " ") for name, passed in checks.items() if not passed]
    confidence = 100.0 * sum(checks.values()) / len(checks)
    return {
        "speed_median_mps": float(speeds.median()),
        "speed_mean_mps": float(speeds.mean()),
        "speed_standard_deviation_mps": (
            float(speeds.std(ddof=1)) if count > 1 else math.nan
        ),
        "speed_p10_mps": float(speeds.quantile(0.10)),
        "speed_p90_mps": float(speeds.quantile(0.90)),
        "speed_iqr_mps": speed_iqr,
        "location_iqr_m": location_iqr,
        "slowdown_success_fraction": success_fraction,
        "slowdown_p_value": p_value,
        "leave_one_out_max_speed_shift_mps": loo_speed,
        "leave_one_out_max_location_shift_m": loo_location,
        "confidence_score": confidence,
        "evidence_pass": all(checks.values()),
        "reason": (
            "repeatable response minimum with significant slowdown and stable leave-one-out medians"
            if all(checks.values())
            else "entry-only gate retained; failed " + ", ".join(failed)
        ),
    }


def _leave_one_out_median_shift(values: np.ndarray) -> float:
    full = float(np.median(values))
    shifts = [
        abs(float(np.median(np.delete(values, index))) - full)
        for index in range(len(values))
    ]
    return max(shifts, default=math.inf)

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
        if row.get("sustained_gate_status") == "entry_only_fallback":
            return (
                "Retain the accepted entry gate; do not add a response-minimum gate: "
                + str(row.get("sustained_gate_reason", "insufficient sustained evidence"))
                + "."
            )
        return "Retain both the empirical entry gate and qualified response-minimum gate."
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
