"""Statistical summaries, numerical quality gates, and input-contract export."""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from cvt_track_study.simulation.service import SimulationError
from cvt_track_study.uncertainty import (
    convergence_diagnostics,
    paired_design_statistics,
    summarize_samples,
)

METRICS = (
    "bounded_lap_time_s",
    "infinite_reference_lap_time_s",
    "lap_time_penalty_vs_infinite_s",
    "finite_ratio_opportunity_loss_energy_kj",
    "bounded_total_opportunity_loss_energy_kj",
    "reference_shared_launch_loss_energy_kj",
)


def summarize_study(
    study_type: str,
    rows: Sequence[Mapping[str, Any]],
    study_raw: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    if study_type == "structural_sensitivity":
        return _summarize_structural(rows)
    bootstrap = int(study_raw.get("reporting", {}).get("bootstrap_resamples", 1000))
    thresholds = study_raw.get("decision_thresholds", {})
    designs: dict[str, Any] = {}
    for index, design_id in enumerate(sorted({str(row["design_id"]) for row in rows})):
        subset = [row for row in rows if row["design_id"] == design_id]
        record: dict[str, Any] = {
            "design_path": subset[0]["design_path"],
            "design_value": subset[0]["design_value"],
            "design_value_si": subset[0]["design_value_si"],
        }
        for metric in METRICS:
            interval = summarize_samples(
                [float(row[metric]) for row in subset],
                bootstrap_seed=seed + 1009 * (index + 1) + sum(ord(c) for c in metric),
                bootstrap_resamples=bootstrap,
            )
            record[metric] = interval.as_dict()
        if "maximum_time_penalty_s" in thresholds:
            limit = float(thresholds["maximum_time_penalty_s"])
            record["time_penalty_threshold"] = _threshold_summary(
                [
                    float(row["lap_time_penalty_vs_infinite_s"]) <= limit
                    for row in subset
                ],
                limit=limit,
                unit="s",
            )
        if "maximum_opportunity_loss_kj" in thresholds:
            limit = float(thresholds["maximum_opportunity_loss_kj"])
            record["opportunity_loss_threshold"] = _threshold_summary(
                [
                    float(row["finite_ratio_opportunity_loss_energy_kj"]) <= limit
                    for row in subset
                ],
                limit=limit,
                unit="kJ",
            )
        designs[design_id] = record
    if study_type == "design_sweep":
        for metric in (
            "bounded_lap_time_s",
            "lap_time_penalty_vs_infinite_s",
            "finite_ratio_opportunity_loss_energy_kj",
        ):
            ranking = paired_design_statistics(
                rows,
                design_key="design_id",
                replicate_key="replicate",
                metric=metric,
                bootstrap_seed=seed + 7919 + sum(ord(c) for c in metric),
                bootstrap_resamples=bootstrap,
            )
            for design_id, values in ranking.items():
                designs[design_id][f"paired_ranking.{metric}"] = values
    return {"study_type": study_type, "designs": designs}


def quality_summary(
    rows: Sequence[Mapping[str, Any]], study_raw: Mapping[str, Any]
) -> dict[str, Any]:
    if not rows:
        raise SimulationError("A study cannot be quality-checked without completed cases.")
    settings = study_raw.get("quality", {})
    maximum_residual = float(
        settings.get("maximum_abs_energy_balance_relative_error", 0.01)
    )
    maximum_powertrain_residual = float(
        settings.get("maximum_abs_powertrain_balance_relative_error", 0.01)
    )
    max_vehicle = max(
        max(
            abs(float(row["bounded_energy_balance_relative_error"])),
            abs(float(row["reference_energy_balance_relative_error"])),
        )
        for row in rows
    )
    max_powertrain = max(
        max(
            abs(float(row["bounded_powertrain_energy_balance_relative_error"])),
            abs(float(row["reference_powertrain_energy_balance_relative_error"])),
        )
        for row in rows
    )
    checks = {
        "all_cases_completed": all(
            bool(row["bounded_completed"]) and bool(row["reference_completed"])
            for row in rows
        ),
        "all_reference_dominance_checks_pass": all(
            bool(row["reference_dominance_pass"]) for row in rows
        ),
        "all_gate_compliance_checks_pass": all(
            bool(row["bounded_gates_compliant_0p5_kmh"])
            and bool(row["reference_gates_compliant_0p5_kmh"])
            for row in rows
        ),
        "vehicle_energy_balance_pass": max_vehicle <= maximum_residual,
        "powertrain_energy_balance_pass": max_powertrain <= maximum_powertrain_residual,
    }
    return {
        **checks,
        "maximum_abs_vehicle_energy_balance_relative_error": max_vehicle,
        "maximum_allowed_vehicle_energy_balance_relative_error": maximum_residual,
        "maximum_abs_powertrain_energy_balance_relative_error": max_powertrain,
        "maximum_allowed_powertrain_energy_balance_relative_error": maximum_powertrain_residual,
        "valid_for_decision": all(checks.values()),
    }


def input_contracts(registry: Any) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for item in registry.inputs:
        records[item.path] = {
            "category": item.category,
            "feature_id": item.feature_id,
            "alternative_model_type": item.alternative_model_type,
            "contract": _json_safe(asdict(item.value)),
        }
    return records


def convergence_summary(
    study_type: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if study_type == "structural_sensitivity":
        return {
            "status": "not_applicable",
            "reason": (
                "One-at-a-time quantile levels are deterministic sensitivity cases, "
                "not Monte Carlo replicates."
            ),
        }
    result: dict[str, Any] = {}
    for design_id in sorted({str(row["design_id"]) for row in rows}):
        subset = [row for row in rows if row["design_id"] == design_id]
        result[design_id] = {
            metric: convergence_diagnostics([float(row[metric]) for row in subset])
            for metric in (
                "lap_time_penalty_vs_infinite_s",
                "finite_ratio_opportunity_loss_energy_kj",
            )
        }
    return result



def _threshold_summary(
    outcomes: Sequence[bool], *, limit: float, unit: str
) -> dict[str, float | int | str]:
    count = len(outcomes)
    if count == 0:
        raise SimulationError("Threshold summaries require at least one scenario.")
    successes = int(sum(bool(value) for value in outcomes))
    fraction = successes / count
    # Wilson score interval remains finite and informative at 0/n and n/n.
    z = 1.959963984540054
    denominator = 1.0 + z * z / count
    center = (fraction + z * z / (2.0 * count)) / denominator
    half = (
        z
        * np.sqrt(
            fraction * (1.0 - fraction) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return {
        "limit": float(limit),
        "unit": unit,
        "scenario_count": count,
        "scenarios_within": successes,
        "estimated_probability_within": float(fraction),
        "wilson_95_low": float(max(0.0, center - half)),
        "wilson_95_high": float(min(1.0, center + half)),
    }

def _summarize_structural(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for parameter in sorted({str(row["parameter_path"]) for row in rows}):
        subset = [row for row in rows if row["parameter_path"] == parameter]
        nominal_rows = [row for row in subset if row.get("level_kind") == "nominal"]
        if len(nominal_rows) != 1:
            raise SimulationError(
                f"Structural parameter {parameter!r} must have exactly one nominal case."
            )
        nominal = nominal_rows[0]
        levels = []
        for row in sorted(
            subset,
            key=lambda item: (
                0 if item.get("level_kind") == "nominal" else 1,
                (
                    float(item["level_probability"])
                    if item.get("level_probability") not in (None, "")
                    else str(item.get("design_value", ""))
                ),
            ),
        ):
            levels.append(
                {
                    "design_id": row["design_id"],
                    "level_kind": row["level_kind"],
                    "level_probability": row["level_probability"],
                    "value": row["design_value"],
                    "value_si": row["design_value_si"],
                    "choice_value": row.get("design_choice_value"),
                    "lap_time_penalty_vs_infinite_s": row[
                        "lap_time_penalty_vs_infinite_s"
                    ],
                    "finite_ratio_opportunity_loss_energy_kj": row[
                        "finite_ratio_opportunity_loss_energy_kj"
                    ],
                    "time_penalty_change_from_nominal_s": float(
                        row["lap_time_penalty_vs_infinite_s"]
                    )
                    - float(nominal["lap_time_penalty_vs_infinite_s"]),
                    "opportunity_loss_change_from_nominal_kj": float(
                        row["finite_ratio_opportunity_loss_energy_kj"]
                    )
                    - float(nominal["finite_ratio_opportunity_loss_energy_kj"]),
                }
            )
        parameters[parameter] = {
            "nominal_value": nominal["design_value"],
            "nominal_value_si": nominal["design_value_si"],
            "nominal_choice_value": nominal.get("design_choice_value"),
            "levels": levels,
            "time_penalty_span_s": max(
                float(row["lap_time_penalty_vs_infinite_s"]) for row in subset
            )
            - min(float(row["lap_time_penalty_vs_infinite_s"]) for row in subset),
            "opportunity_loss_span_kj": max(
                float(row["finite_ratio_opportunity_loss_energy_kj"]) for row in subset
            )
            - min(
                float(row["finite_ratio_opportunity_loss_energy_kj"]) for row in subset
            ),
        }
    return {"study_type": "structural_sensitivity", "parameters": parameters}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
