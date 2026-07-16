"""Screening attribution for sampled and one-at-a-time studies."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


ATTRIBUTION_METRICS = (
    "lap_time_penalty_vs_infinite_s",
    "finite_ratio_opportunity_loss_energy_kj",
)


def build_uncertainty_attribution(
    *,
    study_type: str,
    rows: Sequence[Mapping[str, Any]],
    scenario_draws: Sequence[Mapping[str, Any]],
    input_contracts: Mapping[str, Any],
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    if study_type == "structural_sensitivity":
        return _structural_attribution(rows, input_contracts)
    replicate_count = len({int(row["replicate"]) for row in rows})
    if replicate_count < 8:
        return {
            "status": "suppressed",
            "scenario_count": replicate_count,
            "reason": "At least eight paired scenarios are required for exploratory attribution.",
            "designs": {},
            "warnings": ["Attribution was suppressed because the sample is too small."],
        }
    status = "screening" if replicate_count >= 20 else "exploratory"
    draw_by_replicate = {int(draw["replicate"]): draw for draw in scenario_draws}
    warnings: list[str] = []
    designs: dict[str, Any] = {}
    for design_index, design_id in enumerate(
        sorted({str(row["design_id"]) for row in rows})
    ):
        subset = sorted(
            (row for row in rows if str(row["design_id"]) == design_id),
            key=lambda row: int(row["replicate"]),
        )
        numeric, categorical = _predictors(subset, draw_by_replicate)
        design_record: dict[str, Any] = {"metrics": {}}
        for metric_index, metric in enumerate(ATTRIBUTION_METRICS):
            y = np.asarray([float(row[metric]) for row in subset], dtype=float)
            numeric_rows: list[dict[str, Any]] = []
            for input_index, (path, x) in enumerate(sorted(numeric.items())):
                if len(x) != len(y) or np.std(x) <= 1e-14:
                    continue
                slope = _slope(x, y)
                pearson = float(np.corrcoef(x, y)[0, 1])
                rank = float(spearmanr(x, y).statistic)
                low, high = _bootstrap_slope_interval(
                    x,
                    y,
                    seed=seed + 100003 * (design_index + 1) + 1009 * (metric_index + 1) + input_index,
                    resamples=max(100, min(bootstrap_resamples, 500)),
                )
                x_sigma = float(np.std(x, ddof=1))
                weighted = abs(slope) * x_sigma
                median_y = float(np.median(y))
                elasticity = (
                    slope * float(np.median(x)) / median_y
                    if abs(median_y) > 1e-12
                    else None
                )
                numeric_rows.append(
                    {
                        "path": path,
                        "response_slope": slope,
                        "slope_bootstrap_95_low": low,
                        "slope_bootstrap_95_high": high,
                        "pearson_correlation": pearson,
                        "spearman_rank_correlation": rank,
                        "normalized_elasticity": elasticity,
                        "observed_input_standard_deviation": x_sigma,
                        "uncertainty_weighted_effect": weighted,
                    }
                )
            denominator = sum(
                float(item["uncertainty_weighted_effect"]) ** 2 for item in numeric_rows
            )
            for item in numeric_rows:
                item["relative_screening_importance"] = (
                    float(item["uncertainty_weighted_effect"]) ** 2 / denominator
                    if denominator > 0.0
                    else 0.0
                )
            numeric_rows.sort(
                key=lambda item: float(item["relative_screening_importance"]), reverse=True
            )
            categorical_rows = [
                _categorical_effect(path, values, y)
                for path, values in sorted(categorical.items())
                if len(set(values)) >= 2
            ]
            design_record["metrics"][metric] = {
                "numeric": numeric_rows,
                "categorical": categorical_rows,
            }
        designs[design_id] = design_record

        paths = sorted(numeric)
        for i, left in enumerate(paths):
            for right in paths[i + 1 :]:
                if len(numeric[left]) != len(numeric[right]):
                    continue
                corr = float(np.corrcoef(numeric[left], numeric[right])[0, 1])
                if np.isfinite(corr) and abs(corr) >= 0.8:
                    warnings.append(
                        f"Strong sampled-input correlation ({corr:+.2f}) between {left} and {right}; marginal screening slopes are not causal partitions."
                    )
    if status == "exploratory":
        warnings.insert(
            0,
            "Eight to nineteen scenarios support exploratory direction checks only; use at least twenty for screening attribution.",
        )
    return {
        "status": status,
        "scenario_count": replicate_count,
        "method": "marginal slope, rank association, and uncertainty-weighted screening",
        "causal_variance_partition": False,
        "designs": designs,
        "warnings": list(dict.fromkeys(warnings)),
    }


def flatten_attribution(attribution: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for design_id, design in attribution.get("designs", {}).items():
        for metric, record in design.get("metrics", {}).items():
            for item in record.get("numeric", []):
                rows.append(
                    {"design_id": design_id, "metric": metric, "input_type": "numeric", **item}
                )
            for item in record.get("categorical", []):
                rows.append(
                    {"design_id": design_id, "metric": metric, "input_type": "categorical", **item}
                )
    for item in attribution.get("parameters", []):
        rows.append({"design_id": "structural", **item})
    return rows


def _predictors(
    rows: Sequence[Mapping[str, Any]], draw_by_replicate: Mapping[int, Mapping[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    numeric: dict[str, list[float]] = defaultdict(list)
    categorical: dict[str, list[str]] = defaultdict(list)
    all_numeric: set[str] = set()
    all_categorical: set[str] = set()
    draws: list[Mapping[str, Any]] = []
    for row in rows:
        draw = draw_by_replicate.get(int(row["replicate"]), {})
        draws.append(draw)
        all_numeric.update(str(path) for path in draw.get("quantity_values_si", {}))
        all_numeric.update(
            f"gate.{gate_id}.target_speed_mps"
            for gate_id in draw.get("gate_target_speeds_mps", {})
        )
        all_categorical.update(str(path) for path in draw.get("choice_values", {}))
    for draw in draws:
        quantities = draw.get("quantity_values_si", {})
        gates = draw.get("gate_target_speeds_mps", {})
        choices = draw.get("choice_values", {})
        for path in all_numeric:
            if path.startswith("gate.") and path.endswith(".target_speed_mps"):
                gate_id = path[len("gate.") : -len(".target_speed_mps")]
                value = gates.get(gate_id)
            else:
                value = quantities.get(path)
            if value is None:
                numeric[path].append(float("nan"))
            else:
                numeric[path].append(float(value))
        for path in all_categorical:
            categorical[path].append(str(choices.get(path, "<nominal>")))
    complete_numeric = {
        path: np.asarray(values, dtype=float)
        for path, values in numeric.items()
        if values and np.all(np.isfinite(values))
    }
    return complete_numeric, dict(categorical)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - np.mean(x)
    denominator = float(np.dot(centered, centered))
    return float(np.dot(centered, y - np.mean(y)) / denominator)


def _bootstrap_slope_interval(
    x: np.ndarray, y: np.ndarray, *, seed: int, resamples: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(resamples, len(x)))
    xb, yb = x[indices], y[indices]
    xc = xb - np.mean(xb, axis=1, keepdims=True)
    yc = yb - np.mean(yb, axis=1, keepdims=True)
    denominator = np.sum(xc * xc, axis=1)
    slopes = np.divide(
        np.sum(xc * yc, axis=1),
        denominator,
        out=np.full(resamples, np.nan),
        where=denominator > 1e-20,
    )
    finite = slopes[np.isfinite(slopes)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, (0.025, 0.975), method="linear")
    return float(low), float(high)


def _categorical_effect(path: str, values: Sequence[str], y: np.ndarray) -> dict[str, Any]:
    levels: dict[str, dict[str, float | int]] = {}
    grand = float(np.mean(y))
    total = float(np.sum((y - grand) ** 2))
    between = 0.0
    for level in sorted(set(values)):
        mask = np.asarray([value == level for value in values], dtype=bool)
        mean = float(np.mean(y[mask]))
        count = int(np.sum(mask))
        between += count * (mean - grand) ** 2
        levels[level] = {"count": count, "mean_output": mean}
    means = [float(record["mean_output"]) for record in levels.values()]
    return {
        "path": path,
        "levels": levels,
        "output_span": max(means) - min(means),
        "eta_squared_screening": between / total if total > 0.0 else 0.0,
    }


def _structural_attribution(
    rows: Sequence[Mapping[str, Any]], input_contracts: Mapping[str, Any]
) -> dict[str, Any]:
    parameters: list[dict[str, Any]] = []
    for path in sorted({str(row["parameter_path"]) for row in rows}):
        subset = [row for row in rows if str(row["parameter_path"]) == path]
        nominal = next(row for row in subset if row.get("level_kind") == "nominal")
        numeric = all(row.get("design_value_si") is not None for row in subset)
        for metric in ATTRIBUTION_METRICS:
            if numeric and len(subset) >= 2:
                ordered = sorted(subset, key=lambda row: float(row["design_value_si"]))
                x = np.asarray([float(row["design_value_si"]) for row in ordered])
                y = np.asarray([float(row[metric]) for row in ordered])
                global_slope = _slope(x, y)
                nominal_x = float(nominal["design_value_si"])
                lower = [row for row in ordered if float(row["design_value_si"]) < nominal_x]
                upper = [row for row in ordered if float(row["design_value_si"]) > nominal_x]
                if lower and upper:
                    left, right = lower[-1], upper[0]
                    local_slope = (
                        float(right[metric]) - float(left[metric])
                    ) / (
                        float(right["design_value_si"]) - float(left["design_value_si"])
                    )
                else:
                    local_slope = global_slope
                nominal_y = float(nominal[metric])
                parameters.append(
                    {
                        "metric": metric,
                        "path": path,
                        "input_type": "numeric",
                        "local_slope_at_nominal": local_slope,
                        "global_response_slope": global_slope,
                        "normalized_elasticity": (
                            local_slope * nominal_x / nominal_y
                            if abs(nominal_y) > 1e-12
                            else None
                        ),
                        "response_span": float(np.max(y) - np.min(y)),
                        "declared_contract": input_contracts.get(path),
                    }
                )
            else:
                values = {str(row.get("design_choice_value")): float(row[metric]) for row in subset}
                parameters.append(
                    {
                        "metric": metric,
                        "path": path,
                        "input_type": "categorical",
                        "levels": values,
                        "response_span": max(values.values()) - min(values.values()),
                        "declared_contract": input_contracts.get(path),
                    }
                )
    return {
        "status": "one_at_a_time",
        "scenario_count": 1,
        "method": "exact nominal plus declared structural levels",
        "parameters": parameters,
        "designs": {},
        "warnings": [
            "One-at-a-time slopes do not include interactions between uncertain inputs."
        ],
    }
