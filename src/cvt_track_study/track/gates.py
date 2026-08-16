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
    """Score hard absolute gates and independent relative-response evidence.

    Three evidence roles are deliberately separated:

    * ``hard_absolute``: strong evidence that different measured vehicles converge
      on approximately the same absolute entry speed;
    * ``relative_response``: strong evidence that each vehicle repeatedly loses a
      similar *fraction* of its approach speed at the same feature, even when the
      vehicles have different absolute pace;
    * ``conservative_guardrail``: a permissive source-balanced upper envelope used
      only so an otherwise unsupported feature is never completely uncapped.

    The guardrail is not an accepted fitted feature speed and is never allowed to
    promote a review/rejected gate into a hard absolute gate.
    """

    weights = settings.weights
    total_weight = sum(weights.values())
    if not math.isclose(total_weight, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        diagnostics.warning(
            "GATE_CONFIDENCE_WEIGHTS_NORMALIZED",
            f"Gate-confidence weights sum to {total_weight:.6g}; they were normalized to one.",
        )
        weights = {key: value / total_weight for key, value in weights.items()}

    event_lookup = events.set_index("id")
    eligible_global = passes[passes["eligible"].astype(bool)]
    global_vehicle_count = int(eligible_global["vehicle_id"].nunique())
    rows: list[dict[str, Any]] = []

    for event_id, all_passes in passes.groupby("event_id", sort=False):
        event = event_lookup.loc[event_id]
        valid = all_passes[all_passes["eligible"].astype(bool)].copy()
        count = len(valid)
        speeds = valid["entry_speed_mps"].dropna()
        # Pooled values remain useful descriptive statistics, but they are not
        # allowed to dominate confidence when one vehicle contributes many more laps.
        # Scoring below uses the *worst supported measured source* for the dynamic
        # evidence components. This deliberately makes confidence conservative.
        if count:
            iqr = float(speeds.quantile(0.75) - speeds.quantile(0.25))
            correlation = _safe_correlation(
                valid["entry_speed_mps"], valid["lap_median_speed_mps"]
            )
        else:
            iqr = math.nan
            correlation = math.nan

        source_entry = _source_entry_evidence(valid, settings)
        if source_entry:
            pass_score = min(row["pass_score"] for row in source_entry)
            repeatability = min(row["repeatability"] for row in source_entry)
            braking = min(row["braking"] for row in source_entry)
            pace_independence = min(row["pace_independence"] for row in source_entry)
            minimum_source_pass_count = int(
                min(row["pass_count"] for row in source_entry)
            )
            worst_source_entry_iqr = float(
                max(row["entry_iqr_mps"] for row in source_entry)
            )
        else:
            pass_score = 0.0
            repeatability = 0.0
            braking = 0.0
            pace_independence = 0.0
            minimum_source_pass_count = 0
            worst_source_entry_iqr = math.nan

        pace_ratio = _positive_ratio(
            valid.get("entry_speed_mps", pd.Series(dtype=float)),
            valid.get("lap_median_speed_mps", pd.Series(dtype=float)),
        )
        pace_ratio_iqr = (
            float(pace_ratio.quantile(0.75) - pace_ratio.quantile(0.25))
            if len(pace_ratio)
            else math.nan
        )
        vehicle_pace_ratios = _vehicle_ratio_medians(
            valid, numerator="entry_speed_mps", denominator="lap_median_speed_mps"
        )
        vehicle_pace_ratio_spread = (
            float(vehicle_pace_ratios.max() - vehicle_pace_ratios.min())
            if len(vehicle_pace_ratios) >= 2
            else math.nan
        )

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
            cross_vehicle = max(
                0.0, 1.0 - spread / settings.vehicle_agreement_scale_mps
            )
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
        geometry_usable = bool(
            anchor_projection_error <= settings.maximum_map_error_m
            and effective_coordinate_error <= settings.maximum_map_error_m
        )

        # When multiple measured vehicles are available, a hard *absolute* cap
        # requires direct cross-vehicle agreement. Other evidence may still support
        # a relative response, but cannot overpower disagreement in absolute speed.
        hard_cross_vehicle_supported = bool(
            global_vehicle_count < 2
            or (
                cross_vehicle_status == "measured"
                and cross_vehicle >= 0.50
                and minimum_source_pass_count >= settings.minimum_valid_passes
                and np.isfinite(worst_source_entry_iqr)
                and worst_source_entry_iqr <= settings.repeatability_scale_mps
            )
        )

        if not geometry_usable:
            recommendation = "must_fix"
        elif not candidate:
            recommendation = "not_a_candidate"
        elif count < settings.minimum_valid_passes:
            recommendation = "recommended_review"
        elif score >= settings.accept_score and hard_cross_vehicle_supported:
            recommendation = "accepted"
        elif score >= settings.review_score or score >= settings.accept_score:
            recommendation = "recommended_review"
        else:
            recommendation = "rejected"

        reasons = _gate_reasons(
            components, count, event, settings, cross_vehicle_status
        )
        if (
            candidate
            and score >= settings.accept_score
            and not hard_cross_vehicle_supported
        ):
            if minimum_source_pass_count < settings.minimum_valid_passes:
                reasons.append(
                    "at least one measured vehicle has too few passes for a hard absolute cap"
                )
            if (
                np.isfinite(worst_source_entry_iqr)
                and worst_source_entry_iqr > settings.repeatability_scale_mps
            ):
                reasons.append(
                    "at least one measured vehicle has too much within-source entry-speed spread for a hard absolute cap"
                )
            if cross_vehicle_status == "measured" and cross_vehicle < 0.50:
                reasons.append(
                    "absolute entry speed is not sufficiently consistent across measured vehicles"
                )

        sustained = _sustained_response_statistics(
            valid,
            settings,
            require_cross_vehicle=(global_vehicle_count >= 2),
        )
        sustained_qualified = bool(
            candidate
            and geometry_usable
            and count >= settings.minimum_valid_passes
            and sustained["evidence_pass"]
        )
        sustained_status = (
            "accepted_relative_response" if sustained_qualified else "not_qualified"
        )

        guardrail_cap = _conservative_guardrail_cap(valid, settings)
        guardrail_active = bool(
            candidate
            and geometry_usable
            and np.isfinite(guardrail_cap)
            and recommendation != "accepted"
        )
        if recommendation == "accepted":
            enforcement_class = (
                "hard_absolute_plus_relative_response"
                if sustained_qualified
                else "hard_absolute"
            )
        elif sustained_qualified:
            enforcement_class = "guardrail_plus_relative_response"
        elif guardrail_active:
            enforcement_class = "conservative_guardrail_only"
        else:
            enforcement_class = "none"

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
                "entry_speed_standard_deviation_mps": (
                    float(speeds.std(ddof=1)) if count > 1 else math.nan
                ),
                "entry_speed_p10_mps": float(speeds.quantile(0.10)) if count else math.nan,
                "entry_speed_p90_mps": float(speeds.quantile(0.90)) if count else math.nan,
                "entry_speed_iqr_mps": iqr,
                "minimum_source_pass_count": minimum_source_pass_count,
                "worst_source_entry_iqr_mps": worst_source_entry_iqr,
                "entry_pace_ratio_median": float(pace_ratio.median()) if len(pace_ratio) else math.nan,
                "entry_pace_ratio_iqr": pace_ratio_iqr,
                "vehicle_pace_ratio_spread": vehicle_pace_ratio_spread,
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
                "response_ratio_median": sustained["ratio_median"],
                "response_ratio_p10": sustained["ratio_p10"],
                "response_ratio_p90": sustained["ratio_p90"],
                "response_ratio_worst_source_iqr": sustained["worst_source_ratio_iqr"],
                "response_ratio_cross_vehicle_spread": sustained[
                    "cross_vehicle_ratio_spread"
                ],
                "sustained_slowdown_success_fraction": sustained[
                    "slowdown_success_fraction"
                ],
                "sustained_worst_source_slowdown_fraction": sustained[
                    "worst_source_slowdown_fraction"
                ],
                "sustained_slowdown_p_value": sustained["slowdown_p_value"],
                "sustained_worst_source_p_value": sustained["worst_source_p_value"],
                "sustained_leave_one_out_max_speed_shift_mps": sustained[
                    "leave_one_out_max_speed_shift_mps"
                ],
                "sustained_leave_one_out_max_ratio_shift": sustained[
                    "worst_source_leave_one_out_ratio_shift"
                ],
                "sustained_leave_one_out_max_location_shift_m": sustained[
                    "leave_one_out_max_location_shift_m"
                ],
                "sustained_confidence_score": sustained["confidence_score"],
                "sustained_gate_qualified": sustained_qualified,
                "sustained_gate_status": sustained_status,
                "sustained_gate_reason": sustained["reason"],
                "hard_absolute_gate_qualified": recommendation == "accepted",
                "guardrail_cap_mps": guardrail_cap,
                "guardrail_active_fallback": guardrail_active,
                "enforcement_class": enforcement_class,
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
    valid: pd.DataFrame,
    settings: ReconstructionSettings,
    *,
    require_cross_vehicle: bool,
) -> dict[str, Any]:
    """Qualify a feature response without requiring an accepted entry-speed gate.

    Confidence uses the *worst measured source* for slowdown rate, p-value,
    response-ratio IQR, location IQR, and leave-one-out stability. This prevents
    the larger Cornell sample from overwhelming the smaller McMaster sample.
    Cross-vehicle agreement is evaluated on the dimensionless minimum/approach
    speed ratio rather than raw absolute speed.
    """

    required = valid[
        [
            "approach_speed_mps",
            "entry_speed_mps",
            "event_min_speed_mps",
            "event_min_rel_m",
            "vehicle_id",
        ]
    ].dropna()
    required = required[required["approach_speed_mps"].astype(float) > 0.5].copy()
    count = len(required)
    empty = {
        "speed_median_mps": math.nan,
        "speed_mean_mps": math.nan,
        "speed_standard_deviation_mps": math.nan,
        "speed_p10_mps": math.nan,
        "speed_p90_mps": math.nan,
        "speed_iqr_mps": math.nan,
        "location_iqr_m": math.nan,
        "ratio_median": math.nan,
        "ratio_p10": math.nan,
        "ratio_p90": math.nan,
        "worst_source_ratio_iqr": math.nan,
        "cross_vehicle_ratio_spread": math.nan,
        "slowdown_success_fraction": math.nan,
        "worst_source_slowdown_fraction": math.nan,
        "slowdown_p_value": math.nan,
        "worst_source_p_value": math.nan,
        "leave_one_out_max_speed_shift_mps": math.nan,
        "worst_source_leave_one_out_ratio_shift": math.nan,
        "leave_one_out_max_location_shift_m": math.nan,
        "confidence_score": 0.0,
        "evidence_pass": False,
        "reason": "insufficient complete event-response passes",
    }
    if count < settings.minimum_valid_passes:
        return empty

    required["response_ratio"] = (
        required["event_min_speed_mps"].astype(float)
        / required["approach_speed_mps"].astype(float)
    )
    required = required.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["response_ratio"]
    )
    required = required[required["response_ratio"] >= 0.0]
    if len(required) < settings.minimum_valid_passes:
        return empty

    speeds = required["event_min_speed_mps"].astype(float)
    locations = required["event_min_rel_m"].astype(float)
    ratios = required["response_ratio"].astype(float)
    slowdown = required["approach_speed_mps"].astype(float) - speeds
    successes = int((slowdown >= settings.braking_threshold_mps).sum())
    success_fraction = successes / len(required)
    p_value = float(
        binomtest(successes, len(required), 0.5, alternative="greater").pvalue
    )

    per_source: list[dict[str, float]] = []
    for _, source in required.groupby("vehicle_id", sort=False):
        source_ratio = source["response_ratio"].astype(float)
        source_location = source["event_min_rel_m"].astype(float)
        source_slowdown = (
            source["approach_speed_mps"].astype(float)
            - source["event_min_speed_mps"].astype(float)
        )
        source_successes = int(
            (source_slowdown >= settings.braking_threshold_mps).sum()
        )
        source_count = len(source)
        per_source.append(
            {
                "count": float(source_count),
                "slowdown_fraction": source_successes / source_count,
                "p_value": float(
                    binomtest(
                        source_successes,
                        source_count,
                        0.5,
                        alternative="greater",
                    ).pvalue
                ),
                "ratio_median": float(source_ratio.median()),
                "ratio_iqr": float(
                    source_ratio.quantile(0.75) - source_ratio.quantile(0.25)
                ),
                "location_iqr": float(
                    source_location.quantile(0.75) - source_location.quantile(0.25)
                ),
                "loo_ratio": _leave_one_out_median_shift(
                    source_ratio.to_numpy(float)
                ),
                "loo_location": _leave_one_out_median_shift(
                    source_location.to_numpy(float)
                ),
            }
        )

    source_medians = [row["ratio_median"] for row in per_source]
    cross_vehicle_ratio_spread = (
        max(source_medians) - min(source_medians)
        if len(source_medians) >= 2
        else math.nan
    )
    worst_source_slowdown = min(row["slowdown_fraction"] for row in per_source)
    worst_source_p = max(row["p_value"] for row in per_source)
    worst_source_ratio_iqr = max(row["ratio_iqr"] for row in per_source)
    worst_source_location_iqr = max(row["location_iqr"] for row in per_source)
    worst_source_loo_ratio = max(row["loo_ratio"] for row in per_source)
    worst_source_loo_location = max(row["loo_location"] for row in per_source)
    minimum_source_count = min(row["count"] for row in per_source)

    # Conservative policy: every measured source must independently support the
    # response. The thresholds are intentionally stricter than the old pooled
    # entry-gate score, and no confidence component can be rescued by sample-size
    # dominance from another vehicle.
    checks = {
        "source_sample_support": minimum_source_count >= 3,
        "repeatable_slowdown_each_source": worst_source_slowdown >= 0.80,
        "significant_slowdown_each_source": worst_source_p <= 0.05,
        "response_ratio_repeatability_each_source": worst_source_ratio_iqr <= 0.25,
        "location_repeatability_each_source": worst_source_location_iqr <= max(
            10.0, 2.0 * settings.profile_spacing_m
        ),
        "leave_one_out_ratio_stability_each_source": worst_source_loo_ratio <= 0.08,
        "leave_one_out_location_stability_each_source": worst_source_loo_location <= max(
            5.0, settings.profile_spacing_m
        ),
        "cross_vehicle_relative_response_agreement": (
            not require_cross_vehicle
            or (
                len(source_medians) >= 2
                and cross_vehicle_ratio_spread <= 0.15
            )
        ),
    }
    failed = [name.replace("_", " ") for name, passed in checks.items() if not passed]
    confidence = 100.0 * sum(checks.values()) / len(checks)
    all_pass = all(checks.values())

    return {
        "speed_median_mps": float(speeds.median()),
        "speed_mean_mps": float(speeds.mean()),
        "speed_standard_deviation_mps": (
            float(speeds.std(ddof=1)) if len(speeds) > 1 else math.nan
        ),
        "speed_p10_mps": float(speeds.quantile(0.10)),
        "speed_p90_mps": float(speeds.quantile(0.90)),
        "speed_iqr_mps": float(speeds.quantile(0.75) - speeds.quantile(0.25)),
        "location_iqr_m": worst_source_location_iqr,
        "ratio_median": float(np.median(source_medians)),
        "ratio_p10": float(ratios.quantile(0.10)),
        "ratio_p90": float(ratios.quantile(0.90)),
        "worst_source_ratio_iqr": worst_source_ratio_iqr,
        "cross_vehicle_ratio_spread": cross_vehicle_ratio_spread,
        "slowdown_success_fraction": success_fraction,
        "worst_source_slowdown_fraction": worst_source_slowdown,
        "slowdown_p_value": p_value,
        "worst_source_p_value": worst_source_p,
        "leave_one_out_max_speed_shift_mps": _leave_one_out_median_shift(
            speeds.to_numpy(float)
        ),
        "worst_source_leave_one_out_ratio_shift": worst_source_loo_ratio,
        "leave_one_out_max_location_shift_m": worst_source_loo_location,
        "confidence_score": confidence,
        "evidence_pass": all_pass,
        "reason": (
            "every measured vehicle independently shows a repeatable, stable relative response"
            if all_pass and require_cross_vehicle
            else "repeatable, stable relative response in the available measured source"
            if all_pass
            else "relative-response evidence failed: " + ", ".join(failed)
        ),
    }


def _source_entry_evidence(
    valid: pd.DataFrame, settings: ReconstructionSettings
) -> list[dict[str, float]]:
    """Return equally important per-vehicle entry evidence components.

    Raw lap counts are intentionally not pooled here. A source with 32 laps gets
    a more precise estimate of its own behaviour, but it cannot numerically drown
    out a second source with 14 laps when deciding whether an absolute cap is
    genuinely transferable across measured vehicles.
    """

    rows: list[dict[str, float]] = []
    for _, source in valid.groupby("vehicle_id", sort=False):
        speeds = pd.to_numeric(source["entry_speed_mps"], errors="coerce").dropna()
        if speeds.empty:
            continue
        iqr = float(speeds.quantile(0.75) - speeds.quantile(0.25))
        correlation = _safe_correlation(
            source["entry_speed_mps"], source["lap_median_speed_mps"]
        )
        rows.append(
            {
                "pass_count": float(len(source)),
                "pass_score": min(
                    1.0, len(source) / max(settings.target_pass_count, 1)
                ),
                "entry_iqr_mps": iqr,
                "repeatability": max(
                    0.0, 1.0 - iqr / settings.repeatability_scale_mps
                ),
                "braking": float(
                    (
                        source["braking_drop_mps"]
                        >= settings.braking_threshold_mps
                    ).mean()
                ),
                "pace_independence": (
                    1.0 - abs(correlation) if np.isfinite(correlation) else 0.5
                ),
            }
        )
    return rows


def _positive_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    pair = pd.concat(
        [pd.to_numeric(numerator, errors="coerce"), pd.to_numeric(denominator, errors="coerce")],
        axis=1,
    ).dropna()
    if pair.empty:
        return pd.Series(dtype=float)
    pair = pair[pair.iloc[:, 1] > 0.5]
    ratio = pair.iloc[:, 0] / pair.iloc[:, 1]
    return ratio.replace([np.inf, -np.inf], np.nan).dropna()


def _vehicle_ratio_medians(
    valid: pd.DataFrame, *, numerator: str, denominator: str
) -> pd.Series:
    rows: dict[str, float] = {}
    for vehicle_id, group in valid.groupby("vehicle_id", sort=False):
        ratio = _positive_ratio(group[numerator], group[denominator])
        if len(ratio):
            rows[str(vehicle_id)] = float(ratio.median())
    return pd.Series(rows, dtype=float)


def _conservative_guardrail_cap(
    valid: pd.DataFrame, settings: ReconstructionSettings
) -> float:
    """Permissive upper speed envelope, not a fitted feature target.

    Each vehicle gets its own 95th-percentile entry estimate. The largest source
    value is used, then a 1.5 m/s margin is added. Thus more laps from one vehicle
    cannot pull the cap downward, and disagreement makes the guardrail looser,
    not more restrictive. With no usable speed evidence the global reasonable
    speed ceiling is used instead.
    """

    usable = valid.dropna(subset=["entry_speed_mps"])
    if usable.empty:
        return float(settings.maximum_reasonable_speed_mps)
    source_upper = usable.groupby("vehicle_id")["entry_speed_mps"].quantile(0.95)
    observed_upper = float(source_upper.max()) if len(source_upper) else float(
        usable["entry_speed_mps"].max()
    )
    cap = observed_upper + 1.5
    return float(min(settings.maximum_reasonable_speed_mps, max(0.0, cap)))


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
        if bool(row.get("sustained_gate_qualified", False)):
            return (
                "Retain the hard absolute entry gate and the independently qualified "
                "relative-response gate."
            )
        return "Retain the hard absolute entry gate; no relative-response gate qualified."
    if bool(row.get("sustained_gate_qualified", False)):
        return (
            "Keep the conservative entry guardrail and the independently qualified "
            "relative-response gate; do not promote the entry speed to a hard absolute cap."
        )
    if bool(row.get("guardrail_active_fallback", False)):
        return (
            "Keep only the conservative upper-envelope guardrail here; current evidence "
            "does not justify fitting a hard absolute or relative-response cap."
        )
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
