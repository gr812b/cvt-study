"""Regenerate the full-uncertainty report from saved machine artifacts.

The report deliberately separates three distinct ideas:

* probabilistic/parametric structural draws;
* coherent measured-traversal draws;
* unweighted epistemic track reconstructions.

It can rebuild every table and plot without rerunning a simulation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .catalog import REPORTS
from .html import dataframe_table, figure, metric_cards, read_json, render_page, write_json


METRICS: dict[str, tuple[str, str]] = {
    "bounded_lap_time_s": ("Absolute bounded lap time", "s"),
    "lap_time_penalty_vs_infinite_s": ("Finite-ratio lap-time penalty", "s"),
    "finite_ratio_opportunity_loss_energy_kj": (
        "Finite-ratio opportunity loss",
        "kJ",
    ),
}

LOSS_COLUMNS = (
    "bounded_drivetrain_loss_energy_kj",
    "bounded_clutch_loss_energy_kj",
    "bounded_tire_slip_loss_energy_kj",
    "bounded_brake_loss_energy_kj",
    "bounded_rolling_loss_energy_kj",
    "bounded_aerodynamic_loss_energy_kj",
    "bounded_obstacle_loss_energy_kj",
)

RATIO_COLUMNS = (
    "bounded_time_maximum_ratio_s",
    "bounded_time_variable_ratio_s",
    "bounded_time_minimum_ratio_s",
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def regenerate_full_uncertainty_report(output: Path) -> Path:
    """Rebuild report artifacts from CSV/JSON files already in ``output``."""

    output = output.resolve()
    rows = _read_csv(output / "replicate_results.csv")
    if rows.empty:
        raise FileNotFoundError(
            f"Cannot regenerate full uncertainty report; no replicate rows in {output}."
        )
    manifest = read_json(output / "run_manifest.json", {}) or {}
    summary = read_json(output / "summary.json", {}) or {}
    legacy_convergence = read_json(output / "convergence.json", {}) or {}
    contracts = read_json(output / "input_contracts.json", {}) or {}
    scenario_records = _read_jsonl(output / "scenario_draws.jsonl")
    scenario_frame, input_long = _scenario_frames(scenario_records)

    plots = output / "report_plots"
    plots.mkdir(parents=True, exist_ok=True)

    parameter_inventory = _parameter_inventory(contracts)
    gate_inventory = _gate_inventory(input_long)
    track_inventory = _track_inventory(output, rows)
    driver_table = _driver_explorer(rows, input_long)
    family_table = _family_screening(driver_table)
    derived = _derived_inputs(scenario_frame, contracts)
    convergence = _convergence_table(rows, manifest)
    adequacy = _adequacy_table(rows, manifest, convergence)
    track_case_summary, paired_track_effects = _track_case_analysis(rows, manifest)
    scenario_explorer = _scenario_explorer(rows, scenario_frame, derived)
    loss_summary = _loss_summary(rows)
    ratio_summary = _ratio_summary(rows)
    threshold_summary = _threshold_summary(rows)

    _write_csv(output / "full_uncertainty_parameter_inventory.csv", parameter_inventory)
    _write_csv(output / "full_uncertainty_gate_target_inventory.csv", gate_inventory)
    _write_csv(output / "full_uncertainty_track_case_inventory.csv", track_inventory)
    _write_csv(output / "full_uncertainty_driver_explorer.csv", driver_table)
    _write_csv(output / "full_uncertainty_family_screening.csv", family_table)
    _write_csv(output / "full_uncertainty_derived_inputs.csv", derived)
    _write_csv(output / "full_uncertainty_convergence.csv", convergence)
    _write_csv(output / "full_uncertainty_adequacy.csv", adequacy)
    _write_csv(output / "full_uncertainty_track_case_summary.csv", track_case_summary)
    _write_csv(output / "full_uncertainty_paired_track_effects.csv", paired_track_effects)
    _write_csv(output / "full_uncertainty_scenario_explorer.csv", scenario_explorer)

    _distribution_plots(rows, plots)
    _driver_plots(driver_table, plots)
    _family_plot(family_table, plots)
    _derived_plot(derived, plots)
    _convergence_plot(rows, manifest, plots)
    _track_case_plots(track_case_summary, paired_track_effects, manifest, plots)

    nominal_reference = _nominal_reference(output)
    target = output / REPORTS["full_uncertainty"].html_filename
    target.write_text(
        _build_html(
            rows=rows,
            manifest=manifest,
            summary=summary,
            legacy_convergence=legacy_convergence,
            parameter_inventory=parameter_inventory,
            gate_inventory=gate_inventory,
            track_inventory=track_inventory,
            driver_table=driver_table,
            family_table=family_table,
            derived=derived,
            convergence=convergence,
            adequacy=adequacy,
            track_case_summary=track_case_summary,
            paired_track_effects=paired_track_effects,
            scenario_explorer=scenario_explorer,
            loss_summary=loss_summary,
            ratio_summary=ratio_summary,
            threshold_summary=threshold_summary,
            nominal_reference=nominal_reference,
            plots=plots,
        ),
        encoding="utf-8",
    )
    return target


# ---------------------------------------------------------------------------
# Reading and normalization
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
        if isinstance(value, Mapping):
            records.append(dict(value))
    return records


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _scenario_frames(
    records: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for record in records:
        replicate = int(record.get("replicate", len(wide_rows)))
        identity = record.get("gate_sample_identity")
        identity = identity if isinstance(identity, Mapping) else {}
        base = {
            "replicate": replicate,
            "base_draw_id": int(record.get("base_draw_id", replicate)),
            "track_case_id": str(record.get("track_case_id", "nominal")),
            "track_case_category": str(record.get("track_case_category", "unknown")),
            "run_id": str(identity.get("run_id", "")),
            "lap_id": identity.get("lap_id", ""),
            "vehicle_id": str(identity.get("vehicle_id", "")),
            "driver_id": str(identity.get("driver_id", "")),
        }
        wide = dict(base)
        for input_type, key in (
            ("structural", "quantity_values_si"),
            ("measured_traversal", "gate_target_speeds_mps"),
            ("categorical", "choice_values"),
        ):
            values = record.get(key, {})
            if not isinstance(values, Mapping):
                continue
            for path, raw_value in values.items():
                value = raw_value
                wide[str(path)] = value
                long_rows.append(
                    {
                        **base,
                        "input_type": input_type,
                        "family": _family_for_path(str(path), input_type=input_type),
                        "path": str(path),
                        "value": value,
                    }
                )
        wide_rows.append(wide)
    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows)


# ---------------------------------------------------------------------------
# What was varied
# ---------------------------------------------------------------------------


def _parameter_inventory(contracts: Mapping[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path, outer in contracts.items():
        if not isinstance(outer, Mapping):
            continue
        contract = outer.get("contract", {})
        if not isinstance(contract, Mapping):
            continue
        uncertainty = contract.get("uncertainty", {})
        uncertainty = uncertainty if isinstance(uncertainty, Mapping) else {}
        distribution = str(uncertainty.get("distribution", "fixed"))
        if distribution == "fixed":
            continue
        source = contract.get("source", {})
        source = source if isinstance(source, Mapping) else {}
        records.append(
            {
                "family": _family_for_path(str(path), input_type="structural"),
                "parameter_path": str(path),
                "nominal": contract.get("nominal", ""),
                "unit": contract.get("unit", ""),
                "distribution": distribution,
                "declared_range": _range_text(uncertainty),
                "lower": uncertainty.get("lower", ""),
                "mode": uncertainty.get("mode", ""),
                "upper": uncertainty.get("upper", ""),
                "confidence_level": uncertainty.get("confidence_level", ""),
                "confidence_half_width": uncertainty.get("confidence_half_width", ""),
                "correlation_group": contract.get("correlation_group", ""),
                "source_kind": source.get("kind", ""),
                "source_reference": source.get("reference", ""),
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["family", "parameter_path"])


def _range_text(uncertainty: Mapping[str, Any]) -> str:
    distribution = str(uncertainty.get("distribution", "fixed"))
    lower = uncertainty.get("lower")
    upper = uncertainty.get("upper")
    mode = uncertainty.get("mode")
    choices = uncertainty.get("choices", [])
    samples = uncertainty.get("samples", [])
    if distribution == "triangular" and lower is not None and upper is not None:
        return f"{lower:g} to {upper:g}; mode {mode:g}" if mode is not None else f"{lower:g} to {upper:g}"
    if lower is not None and upper is not None:
        return f"{lower:g} to {upper:g}"
    if uncertainty.get("standard_deviation") is not None:
        return f"σ={uncertainty['standard_deviation']:g}"
    if uncertainty.get("confidence_half_width") is not None:
        level = uncertainty.get("confidence_level")
        suffix = f" at {100 * float(level):g}%" if level is not None else ""
        return f"±{float(uncertainty['confidence_half_width']):g}{suffix}"
    if choices:
        return ", ".join(str(value) for value in choices)
    if samples:
        numeric = [float(value) for value in samples]
        return f"empirical {min(numeric):g} to {max(numeric):g} ({len(numeric)} samples)"
    return distribution


def _gate_inventory(input_long: pd.DataFrame) -> pd.DataFrame:
    if input_long.empty:
        return pd.DataFrame()
    gate = input_long[input_long["input_type"] == "measured_traversal"].copy()
    if gate.empty:
        return pd.DataFrame()
    gate["value"] = pd.to_numeric(gate["value"], errors="coerce")
    records = []
    for path, group in gate.groupby("path", sort=True):
        values = group["value"].dropna()
        if values.empty:
            continue
        records.append(
            {
                "gate_target_id": path,
                "kind": "response minimum" if str(path).endswith(":response_minimum") else "entry target",
                "sample_count": int(len(values)),
                "unique_speed_count": int(values.nunique()),
                "minimum_kmh": float(3.6 * values.min()),
                "median_kmh": float(3.6 * values.median()),
                "maximum_kmh": float(3.6 * values.max()),
                "p10_kmh": float(3.6 * values.quantile(0.1)),
                "p90_kmh": float(3.6 * values.quantile(0.9)),
            }
        )
    return pd.DataFrame(records)


def _track_inventory(output: Path, rows: pd.DataFrame) -> pd.DataFrame:
    manifest_path = output / "track_ensemble" / "manifest.json"
    manifest = read_json(manifest_path, {}) or {}
    records: list[dict[str, Any]] = []
    for record in manifest.get("cases", []):
        if not isinstance(record, Mapping):
            continue
        case_id = str(record.get("case_id", ""))
        raw_path = str(record.get("file", "")).replace("\\", "/")
        path = output / raw_path
        bundle = read_json(path, {}) or {}
        contract = bundle.get("simulation_contract", {}) if isinstance(bundle, Mapping) else {}
        gates = contract.get("speed_gates", []) if isinstance(contract, Mapping) else []
        active = [gate for gate in gates if isinstance(gate, Mapping) and bool(gate.get("active_by_default"))]
        case_rows = rows[rows.get("track_case_id", pd.Series(dtype=str)).astype(str) == case_id] if "track_case_id" in rows else pd.DataFrame()
        records.append(
            {
                "track_case_id": case_id,
                "category": record.get("category", ""),
                "label": record.get("label", case_id),
                "scenario_count": int(len(case_rows)),
                "base_draw_count": int(case_rows["base_draw_id"].nunique()) if "base_draw_id" in case_rows else int(len(case_rows)),
                "track_length_m": contract.get("track_length_m", "") if isinstance(contract, Mapping) else "",
                "active_gate_count": len(active),
                "bundle_fingerprint": record.get("fingerprint", ""),
            }
        )
    if not records and "track_case_id" in rows:
        for case_id, group in rows.groupby("track_case_id", sort=False):
            records.append(
                {
                    "track_case_id": case_id,
                    "category": group.get("track_case_category", pd.Series([""])).iloc[0],
                    "label": case_id,
                    "scenario_count": len(group),
                    "base_draw_count": group["base_draw_id"].nunique() if "base_draw_id" in group else len(group),
                    "track_length_m": "",
                    "active_gate_count": "",
                    "bundle_fingerprint": group.get("track_bundle_fingerprint", pd.Series([""])).iloc[0],
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Driver screening
# ---------------------------------------------------------------------------


def _driver_explorer(rows: pd.DataFrame, input_long: pd.DataFrame) -> pd.DataFrame:
    if input_long.empty:
        return pd.DataFrame()
    output = rows.copy()
    if "design_id" in output and output["design_id"].nunique() > 1:
        nominal = output[output["design_id"].astype(str) == "nominal"]
        if not nominal.empty:
            output = nominal
    output_columns = ["replicate", *[metric for metric in METRICS if metric in output]]
    merged = input_long.merge(output[output_columns], on="replicate", how="inner")
    records: list[dict[str, Any]] = []
    for metric in METRICS:
        if metric not in merged:
            continue
        for path, group in merged.groupby("path", sort=False):
            x = pd.to_numeric(group["value"], errors="coerce")
            y = pd.to_numeric(group[metric], errors="coerce")
            valid = x.notna() & y.notna()
            x = x[valid]
            y = y[valid]
            if len(x) < 5 or x.nunique() < 2 or y.nunique() < 2:
                continue
            pearson = float(x.corr(y, method="pearson"))
            spearman = float(x.corr(y, method="spearman"))
            slope = float(np.polyfit(x.to_numpy(float), y.to_numpy(float), 1)[0])
            first = group.loc[valid].iloc[0]
            records.append(
                {
                    "metric": metric,
                    "metric_name": METRICS[metric][0],
                    "input_type": first["input_type"],
                    "family": first["family"],
                    "path": path,
                    "sample_count": int(len(x)),
                    "observed_minimum": float(x.min()),
                    "observed_median": float(x.median()),
                    "observed_maximum": float(x.max()),
                    "pearson_correlation": pearson,
                    "spearman_rank_correlation": spearman,
                    "response_slope": slope,
                    "direction": _direction_text(metric, spearman),
                    "absolute_rank_association": abs(spearman),
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["relative_screening_importance"] = 0.0
    for metric, index in frame.groupby("metric").groups.items():
        maximum = float(frame.loc[index, "absolute_rank_association"].max())
        if maximum > 0:
            frame.loc[index, "relative_screening_importance"] = (
                frame.loc[index, "absolute_rank_association"] / maximum
            )
    frame["rank"] = frame.groupby("metric")["absolute_rank_association"].rank(
        method="first", ascending=False
    ).astype(int)
    return frame.sort_values(["metric", "rank", "path"])


def _direction_text(metric: str, correlation: float) -> str:
    if abs(correlation) < 0.05:
        return "little monotonic association"
    higher = correlation > 0
    if metric == "bounded_lap_time_s":
        return "higher input → slower lap" if higher else "higher input → faster lap"
    if metric == "lap_time_penalty_vs_infinite_s":
        return "higher input → larger finite-range penalty" if higher else "higher input → smaller finite-range penalty"
    return "higher input → larger opportunity loss" if higher else "higher input → smaller opportunity loss"


def _family_screening(driver_table: pd.DataFrame) -> pd.DataFrame:
    if driver_table.empty:
        return pd.DataFrame()
    records = []
    for (metric, family), group in driver_table.groupby(["metric", "family"]):
        strongest = group.sort_values("absolute_rank_association", ascending=False).iloc[0]
        records.append(
            {
                "metric": metric,
                "metric_name": METRICS.get(metric, (metric, ""))[0],
                "family": family,
                "input_count": int(group["path"].nunique()),
                "strongest_path": strongest["path"],
                "strongest_absolute_rank_association": float(strongest["absolute_rank_association"]),
                "median_absolute_rank_association": float(group["absolute_rank_association"].median()),
                "interpretation": "screening association; overlapping families are not additive variance shares",
            }
        )
    return pd.DataFrame(records).sort_values(
        ["metric", "strongest_absolute_rank_association"], ascending=[True, False]
    )


def _family_for_path(path: str, *, input_type: str) -> str:
    if input_type == "measured_traversal" or path.startswith("gate:"):
        return "measured traversal"
    if path in {"drivetrain.efficiency", "drivetrain.engine.power_scale"}:
        return "delivered wheel power"
    if path.startswith("obstacle."):
        return "discrete obstacles"
    if path in {
        "track.surface.friction_coefficient",
        "vehicle.tire.peak_traction_scale",
        "vehicle.tire.slip_stiffness",
        "vehicle.driven_normal_load_fraction",
    }:
        return "traction capacity"
    if path.startswith("vehicle.aero.") or path == "vehicle.rolling_resistance_coefficient":
        return "continuous resistance"
    if path.startswith("driver."):
        return "driver and braking"
    if path == "drivetrain.engine.target_speed":
        return "powertrain control"
    if path.startswith("vehicle."):
        return "vehicle geometry and inertia"
    return "other structural"


# ---------------------------------------------------------------------------
# Derived combinations and mechanisms
# ---------------------------------------------------------------------------


def _derived_inputs(
    scenario_frame: pd.DataFrame,
    contracts: Mapping[str, Any],
) -> pd.DataFrame:
    if scenario_frame.empty:
        return pd.DataFrame()
    frame = scenario_frame[[column for column in (
        "replicate", "base_draw_id", "track_case_id", "run_id", "lap_id", "vehicle_id", "driver_id"
    ) if column in scenario_frame]].copy()

    efficiency = _numeric_column(scenario_frame, "drivetrain.efficiency")
    power = _numeric_column(scenario_frame, "drivetrain.engine.power_scale")
    if efficiency is not None and power is not None:
        frame["effective_wheel_power_multiplier"] = efficiency * power

    traction_paths = (
        "track.surface.friction_coefficient",
        "vehicle.tire.peak_traction_scale",
        "vehicle.driven_normal_load_fraction",
    )
    traction = []
    nominal_product = 1.0
    for path in traction_paths:
        values = _numeric_column(scenario_frame, path)
        nominal = _contract_nominal(contracts, path)
        if values is None or nominal is None or nominal == 0:
            traction = []
            break
        traction.append(values)
        nominal_product *= nominal
    if traction:
        product = traction[0].copy()
        for values in traction[1:]:
            product *= values
        frame["effective_traction_multiplier_vs_nominal"] = product / nominal_product

    gate_columns = [column for column in scenario_frame if str(column).startswith("gate:")]
    if gate_columns:
        gate_values = scenario_frame[gate_columns].apply(pd.to_numeric, errors="coerce")
        pace = gate_values.median(axis=1, skipna=True)
        median = float(pace.median())
        frame["measured_traversal_pace_index"] = pace / median if median > 0 else np.nan

    obstacle_columns = [column for column in scenario_frame if str(column).startswith("obstacle.")]
    if obstacle_columns:
        ranked = scenario_frame[obstacle_columns].apply(pd.to_numeric, errors="coerce").rank(pct=True)
        frame["obstacle_severity_percentile_index"] = ranked.mean(axis=1, skipna=True)

    return frame


def _numeric_column(frame: pd.DataFrame, path: str) -> pd.Series | None:
    if path not in frame:
        return None
    return pd.to_numeric(frame[path], errors="coerce")


def _contract_nominal(contracts: Mapping[str, Any], path: str) -> float | None:
    outer = contracts.get(path)
    if not isinstance(outer, Mapping):
        return None
    contract = outer.get("contract")
    if not isinstance(contract, Mapping):
        return None
    try:
        return float(contract.get("nominal"))
    except (TypeError, ValueError):
        return None


def _loss_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in LOSS_COLUMNS:
        if column not in rows:
            continue
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        if values.empty:
            continue
        records.append(
            {
                "loss_mechanism": column.replace("bounded_", "").replace("_loss_energy_kj", "").replace("_", " "),
                "p10_kj": float(values.quantile(0.1)),
                "median_kj": float(values.median()),
                "p90_kj": float(values.quantile(0.9)),
            }
        )
    frame = pd.DataFrame(records)
    if not frame.empty:
        total = float(frame["median_kj"].sum())
        frame["median_share_percent"] = 100.0 * frame["median_kj"] / total if total > 0 else 0.0
        frame = frame.sort_values("median_kj", ascending=False)
    return frame


def _ratio_summary(rows: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "bounded_time_maximum_ratio_s": "maximum reduction ratio",
        "bounded_time_variable_ratio_s": "variable ratio region",
        "bounded_time_minimum_ratio_s": "minimum reduction ratio / overdrive bound",
    }
    records = []
    lap = pd.to_numeric(rows.get("bounded_lap_time_s"), errors="coerce")
    for column in RATIO_COLUMNS:
        if column not in rows:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        fractions = values / lap
        records.append(
            {
                "region": labels[column],
                "p10_time_s": float(values.quantile(0.1)),
                "median_time_s": float(values.median()),
                "p90_time_s": float(values.quantile(0.9)),
                "median_lap_fraction_percent": float(100.0 * fractions.median()),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Convergence, adequacy, and track-case isolation
# ---------------------------------------------------------------------------


def _convergence_table(rows: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    records = []
    for metric, (name, unit) in METRICS.items():
        if metric not in rows:
            continue
        values = pd.to_numeric(rows[metric], errors="coerce").dropna().reset_index(drop=True)
        if len(values) < 4:
            continue
        middle = len(values) // 2
        first = values.iloc[:middle]
        second = values.iloc[middle:]
        median = float(values.median())
        median_difference = abs(float(first.median()) - float(second.median()))
        relative = median_difference / max(abs(median), 1e-12)
        p10_difference = abs(float(first.quantile(0.1)) - float(second.quantile(0.1)))
        p90_difference = abs(float(first.quantile(0.9)) - float(second.quantile(0.9)))
        if metric == "bounded_lap_time_s":
            stable = relative <= 0.03 and max(p10_difference, p90_difference) <= 2.0
        elif metric == "lap_time_penalty_vs_infinite_s":
            stable = relative <= 0.05 and max(p10_difference, p90_difference) <= 0.75
        else:
            stable = relative <= 0.05 and max(p10_difference, p90_difference) <= 15.0
        records.append(
            {
                "metric": metric,
                "metric_name": name,
                "unit": unit,
                "scenario_count": int(len(values)),
                "full_median": median,
                "split_half_median_difference": median_difference,
                "split_half_relative_difference_percent": 100.0 * relative,
                "split_half_p10_difference": p10_difference,
                "split_half_p90_difference": p90_difference,
                "status": "exploratory convergence passed" if stable else "more common draws recommended",
            }
        )
    return pd.DataFrame(records)


def _adequacy_table(
    rows: pd.DataFrame,
    manifest: Mapping[str, Any],
    convergence: pd.DataFrame,
) -> pd.DataFrame:
    scenario_count = int(rows["replicate"].nunique()) if "replicate" in rows else len(rows)
    track_count = int(rows["track_case_id"].nunique()) if "track_case_id" in rows else 1
    per_case = rows.groupby("track_case_id").size() if "track_case_id" in rows else pd.Series([len(rows)])
    penalty = pd.to_numeric(rows.get("lap_time_penalty_vs_infinite_s"), errors="coerce").dropna()
    positive = float((penalty > 0).mean()) if len(penalty) else math.nan
    paired = bool(manifest.get("track_case_pairing_complete", False)) or (
        str(manifest.get("track_case_assignment", "")) == "fully_crossed_common_draws"
    )
    numerical = manifest.get("numerical_quality", {})
    numerical_valid = bool(numerical.get("numerically_valid", False)) if isinstance(numerical, Mapping) else False

    convergence_lookup = {
        row["metric"]: row["status"] for _, row in convergence.iterrows()
    }
    return pd.DataFrame(
        [
            {
                "area": "Numerical execution",
                "status": "strong" if numerical_valid else "review required",
                "basis": f"{scenario_count} joint scenarios; numerical quality flag={numerical_valid}",
            },
            {
                "area": "Direction of finite-range comparison",
                "status": "strong" if np.isfinite(positive) and positive == 1.0 else "mixed",
                "basis": f"positive bounded-minus-infinite penalty in {int((penalty > 0).sum())}/{len(penalty)} evaluated scenarios",
            },
            {
                "area": "Absolute lap-time centre and tails",
                "status": convergence_lookup.get("bounded_lap_time_s", "not assessed"),
                "basis": "split-half median and p10/p90 stability",
            },
            {
                "area": "Finite-ratio penalty centre and tails",
                "status": convergence_lookup.get("lap_time_penalty_vs_infinite_s", "not assessed"),
                "basis": "split-half median and p10/p90 stability",
            },
            {
                "area": "Opportunity-loss centre and tails",
                "status": convergence_lookup.get("finite_ratio_opportunity_loss_energy_kj", "not assessed"),
                "basis": "split-half median and p10/p90 stability",
            },
            {
                "area": "Track-case effects",
                "status": "paired and isolated" if paired else "not isolated in this legacy run",
                "basis": (
                    f"{track_count} track cases × {int(per_case.min()) if len(per_case) else 0}–{int(per_case.max()) if len(per_case) else 0} scenarios; "
                    + ("same common draws replayed on every case" if paired else "different random draws assigned to different cases")
                ),
            },
            {
                "area": "Real-world probability interpretation",
                "status": "not calibrated",
                "basis": "track reconstructions are unweighted epistemic alternatives, so pooled percentiles are joint-scenario percentiles rather than credible probabilities",
            },
        ]
    )


def _track_case_analysis(
    rows: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "track_case_id" not in rows:
        return pd.DataFrame(), pd.DataFrame()
    records = []
    for case_id, group in rows.groupby("track_case_id", sort=False):
        lap = pd.to_numeric(group.get("bounded_lap_time_s"), errors="coerce").dropna()
        penalty = pd.to_numeric(group.get("lap_time_penalty_vs_infinite_s"), errors="coerce").dropna()
        records.append(
            {
                "track_case_id": case_id,
                "category": group.get("track_case_category", pd.Series([""])).iloc[0],
                "scenario_count": len(group),
                "base_draw_count": group["base_draw_id"].nunique() if "base_draw_id" in group else len(group),
                "bounded_lap_p10_s": float(lap.quantile(0.1)) if len(lap) else math.nan,
                "bounded_lap_median_s": float(lap.median()) if len(lap) else math.nan,
                "bounded_lap_p90_s": float(lap.quantile(0.9)) if len(lap) else math.nan,
                "penalty_p10_s": float(penalty.quantile(0.1)) if len(penalty) else math.nan,
                "penalty_median_s": float(penalty.median()) if len(penalty) else math.nan,
                "penalty_p90_s": float(penalty.quantile(0.9)) if len(penalty) else math.nan,
            }
        )
    summary = pd.DataFrame(records)

    paired = bool(manifest.get("track_case_pairing_complete", False))
    if not paired or "base_draw_id" not in rows or "nominal" not in set(rows["track_case_id"].astype(str)):
        return summary, pd.DataFrame()

    effect_records = []
    for metric, label in (
        ("bounded_lap_time_s", "bounded lap time"),
        ("lap_time_penalty_vs_infinite_s", "finite-range penalty"),
        ("finite_ratio_opportunity_loss_energy_kj", "opportunity loss"),
    ):
        if metric not in rows:
            continue
        pivot = rows.pivot_table(index="base_draw_id", columns="track_case_id", values=metric, aggfunc="first")
        if "nominal" not in pivot:
            continue
        for case_id in pivot.columns:
            if case_id == "nominal":
                continue
            difference = (pivot[case_id] - pivot["nominal"]).dropna()
            if difference.empty:
                continue
            effect_records.append(
                {
                    "metric": metric,
                    "metric_name": label,
                    "track_case_id": case_id,
                    "paired_draw_count": int(len(difference)),
                    "delta_p10": float(difference.quantile(0.1)),
                    "delta_median": float(difference.median()),
                    "delta_p90": float(difference.quantile(0.9)),
                }
            )
    return summary, pd.DataFrame(effect_records)


def _threshold_summary(rows: pd.DataFrame) -> pd.DataFrame:
    penalty = pd.to_numeric(rows.get("lap_time_penalty_vs_infinite_s"), errors="coerce").dropna()
    records = []
    for threshold in (0.0, 1.0, 3.0, 5.0, 7.5):
        records.append(
            {
                "threshold_s": threshold,
                "scenario_count_at_or_above": int((penalty >= threshold).sum()),
                "scenario_fraction_at_or_above_percent": float(100.0 * (penalty >= threshold).mean()) if len(penalty) else math.nan,
                "interpretation": "frequency within the constructed joint scenario set; not a calibrated real-world probability",
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Scenario explorer
# ---------------------------------------------------------------------------


def _scenario_explorer(
    rows: pd.DataFrame,
    scenario_frame: pd.DataFrame,
    derived: pd.DataFrame,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    selected: list[tuple[str, int]] = []
    completed = rows.copy()
    if "bounded_completed" in completed:
        completed = completed[completed["bounded_completed"].astype(bool)]
    lap = pd.to_numeric(completed.get("bounded_lap_time_s"), errors="coerce")
    if len(completed) and lap.notna().any():
        selected.append(("fastest bounded", int(lap.idxmin())))
        ordered = lap.dropna().sort_values()
        selected.append(("median bounded", int(ordered.index[len(ordered) // 2])))
        selected.append(("slowest bounded", int(lap.idxmax())))
    penalty = pd.to_numeric(completed.get("lap_time_penalty_vs_infinite_s"), errors="coerce")
    if penalty.notna().any():
        selected.append(("largest finite-range penalty", int(penalty.idxmax())))
    opportunity = pd.to_numeric(completed.get("finite_ratio_opportunity_loss_energy_kj"), errors="coerce")
    if opportunity.notna().any():
        selected.append(("largest opportunity loss", int(opportunity.idxmax())))

    records = []
    scenario_lookup = scenario_frame.set_index("replicate", drop=False) if not scenario_frame.empty and "replicate" in scenario_frame else pd.DataFrame()
    derived_lookup = derived.set_index("replicate", drop=False) if not derived.empty and "replicate" in derived else pd.DataFrame()
    for role, index in selected:
        row = completed.loc[index]
        replicate = int(row.get("replicate", -1))
        scenario = scenario_lookup.loc[replicate] if not scenario_lookup.empty and replicate in scenario_lookup.index else pd.Series(dtype=object)
        combo = derived_lookup.loc[replicate] if not derived_lookup.empty and replicate in derived_lookup.index else pd.Series(dtype=object)
        records.append(
            {
                "scenario_role": role,
                "replicate": replicate,
                "base_draw_id": row.get("base_draw_id", scenario.get("base_draw_id", "")),
                "track_case_id": row.get("track_case_id", ""),
                "measured_run_id": scenario.get("run_id", ""),
                "measured_lap_id": scenario.get("lap_id", ""),
                "drivetrain_efficiency": scenario.get("drivetrain.efficiency", ""),
                "engine_power_scale": scenario.get("drivetrain.engine.power_scale", ""),
                "effective_wheel_power_multiplier": combo.get("effective_wheel_power_multiplier", ""),
                "rolling_resistance_coefficient": scenario.get("vehicle.rolling_resistance_coefficient", ""),
                "surface_friction_coefficient": scenario.get("track.surface.friction_coefficient", ""),
                "tire_traction_scale": scenario.get("vehicle.tire.peak_traction_scale", ""),
                "effective_traction_multiplier_vs_nominal": combo.get("effective_traction_multiplier_vs_nominal", ""),
                "bounded_lap_time_s": row.get("bounded_lap_time_s", ""),
                "infinite_reference_lap_time_s": row.get("infinite_reference_lap_time_s", ""),
                "finite_ratio_penalty_s": row.get("lap_time_penalty_vs_infinite_s", ""),
                "opportunity_loss_kj": row.get("finite_ratio_opportunity_loss_energy_kj", ""),
                "minimum_ratio_time_s": row.get("bounded_time_minimum_ratio_s", ""),
                "brake_loss_kj": row.get("bounded_brake_loss_energy_kj", ""),
                "drivetrain_loss_kj": row.get("bounded_drivetrain_loss_energy_kj", ""),
                "rolling_loss_kj": row.get("bounded_rolling_loss_energy_kj", ""),
                "aerodynamic_loss_kj": row.get("bounded_aerodynamic_loss_energy_kj", ""),
                "obstacle_loss_kj": row.get("bounded_obstacle_loss_energy_kj", ""),
            }
        )
    return pd.DataFrame(records).drop_duplicates(subset=["scenario_role", "replicate"])


def _nominal_reference(output: Path) -> Mapping[str, Any] | None:
    value = read_json(output / "nominal_reference.json", None)
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        return value[0]
    if isinstance(value, Mapping):
        return value
    return None


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def _distribution_plots(rows: pd.DataFrame, plots: Path) -> None:
    if "bounded_lap_time_s" in rows and "infinite_reference_lap_time_s" in rows:
        figure_obj, axis = plt.subplots(figsize=(10.5, 5.6))
        for column, label in (
            ("bounded_lap_time_s", "bounded CVT"),
            ("infinite_reference_lap_time_s", "infinite-ratio reference"),
        ):
            values = pd.to_numeric(rows[column], errors="coerce").dropna()
            axis.hist(values, bins=min(24, max(8, int(np.sqrt(len(values))))), alpha=0.55, label=label)
        axis.set_xlabel("Lap time [s]")
        axis.set_ylabel("Joint scenario count")
        axis.set_title("Absolute lap-time distribution across joint scenarios")
        axis.legend()
        axis.grid(True, axis="y", alpha=0.25)
        figure_obj.tight_layout()
        figure_obj.savefig(plots / "absolute_lap_time_distribution.png", dpi=180)
        plt.close(figure_obj)

    if "design_id" in rows and "bounded_completed" in rows:
        grouped = rows.groupby("design_id")["bounded_completed"].apply(lambda series: series.astype(bool).mean()).sort_values()
        figure_obj, axis = plt.subplots(figsize=(10, max(4.5, 0.42 * len(grouped))))
        axis.barh(grouped.index.astype(str), grouped.values)
        axis.set_xlim(0, 1)
        axis.set_xlabel("Completion fraction")
        axis.set_title("Completion across joint scenarios")
        axis.grid(True, axis="x", alpha=0.25)
        figure_obj.tight_layout()
        figure_obj.savefig(plots / "completion_by_design.png", dpi=180)
        plt.close(figure_obj)

    if "lap_time_penalty_vs_infinite_s" in rows:
        values = pd.to_numeric(rows["lap_time_penalty_vs_infinite_s"], errors="coerce").dropna()
        figure_obj, axis = plt.subplots(figsize=(10.5, 5.4))
        axis.hist(values, bins=min(24, max(8, int(np.sqrt(len(values))))))
        axis.axvline(0, linewidth=1)
        axis.set_xlabel("Bounded minus infinite-reference lap time [s]")
        axis.set_ylabel("Joint scenario count")
        axis.set_title("Paired finite-ratio time penalty")
        axis.grid(True, axis="y", alpha=0.25)
        figure_obj.tight_layout()
        figure_obj.savefig(plots / "paired_penalty_distribution.png", dpi=180)
        plt.close(figure_obj)

    ratio_columns = [column for column in RATIO_COLUMNS if column in rows]
    if ratio_columns:
        labels = [
            "maximum ratio",
            "variable region",
            "minimum ratio / overdrive bound",
        ][: len(ratio_columns)]
        figure_obj, axis = plt.subplots(figsize=(10.5, 5.6))
        axis.boxplot(
            [pd.to_numeric(rows[column], errors="coerce").dropna() for column in ratio_columns],
            tick_labels=labels,
            showfliers=False,
        )
        axis.set_ylabel("Time [s]")
        axis.set_title("CVT ratio-region occupancy")
        axis.grid(True, axis="y", alpha=0.25)
        figure_obj.tight_layout()
        figure_obj.savefig(plots / "ratio_occupancy_distribution.png", dpi=180)
        plt.close(figure_obj)

    loss_columns = [column for column in LOSS_COLUMNS if column in rows]
    if loss_columns:
        med = np.array([pd.to_numeric(rows[column], errors="coerce").median() for column in loss_columns])
        low = np.array([pd.to_numeric(rows[column], errors="coerce").quantile(0.1) for column in loss_columns])
        high = np.array([pd.to_numeric(rows[column], errors="coerce").quantile(0.9) for column in loss_columns])
        x = np.arange(len(loss_columns))
        labels = [column.replace("bounded_", "").replace("_loss_energy_kj", "").replace("_", " ") for column in loss_columns]
        figure_obj, axis = plt.subplots(figsize=(11, 5.6))
        axis.bar(x, med)
        axis.errorbar(x, med, yerr=np.vstack((med - low, high - med)), fmt="none", capsize=3)
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_ylabel("Energy [kJ]")
        axis.set_title("Physical loss distribution: median and p10–p90")
        axis.grid(True, axis="y", alpha=0.25)
        figure_obj.tight_layout()
        figure_obj.savefig(plots / "physical_loss_distribution.png", dpi=180)
        plt.close(figure_obj)


def _driver_plots(driver_table: pd.DataFrame, plots: Path) -> None:
    if driver_table.empty:
        return
    filenames = {
        "bounded_lap_time_s": "absolute_lap_time_uncertainty_drivers.png",
        "lap_time_penalty_vs_infinite_s": "finite_ratio_penalty_uncertainty_drivers.png",
        "finite_ratio_opportunity_loss_energy_kj": "opportunity_loss_uncertainty_drivers.png",
    }
    for metric, filename in filenames.items():
        data = driver_table[driver_table["metric"] == metric].sort_values(
            "absolute_rank_association", ascending=False
        )
        if data.empty:
            continue
        _driver_bar(data.head(15), plots / filename, METRICS[metric][0], top=True)
        _driver_bar(data, plots / filename.replace(".png", "_all.png"), METRICS[metric][0], top=False)
    source = plots / "absolute_lap_time_uncertainty_drivers.png"
    if source.is_file():
        (plots / "top_uncertainty_drivers.png").write_bytes(source.read_bytes())


def _driver_bar(data: pd.DataFrame, path: Path, title: str, *, top: bool) -> None:
    ordered = data.sort_values("absolute_rank_association").copy()
    labels = [f"[{family}] {path_text}" for family, path_text in zip(ordered["family"], ordered["path"])]
    values = ordered["absolute_rank_association"].to_numpy(float)
    signs = np.sign(ordered["spearman_rank_correlation"].to_numpy(float))
    figure_obj, axis = plt.subplots(figsize=(12, max(5.8, 0.36 * len(ordered) + 1.8)))
    axis.barh(np.arange(len(ordered)), values * signs)
    axis.axvline(0, linewidth=1)
    axis.set_yticks(np.arange(len(ordered)), labels)
    axis.set_xlabel("Signed marginal Spearman rank association")
    axis.set_title(
        ("Leading" if top else "All")
        + f" screening associations with {title.lower()}\n"
        + "not additive variance shares; correlated inputs must be read as families"
    )
    axis.grid(True, axis="x", alpha=0.25)
    figure_obj.tight_layout()
    figure_obj.savefig(path, dpi=180)
    plt.close(figure_obj)


def _family_plot(family_table: pd.DataFrame, plots: Path) -> None:
    if family_table.empty:
        return
    metrics = list(METRICS)
    families = sorted(family_table["family"].unique())
    matrix = np.zeros((len(families), len(metrics)))
    for i, family in enumerate(families):
        for j, metric in enumerate(metrics):
            match = family_table[(family_table["family"] == family) & (family_table["metric"] == metric)]
            if not match.empty:
                matrix[i, j] = float(match.iloc[0]["strongest_absolute_rank_association"])
    figure_obj, axis = plt.subplots(figsize=(11, max(5.5, 0.55 * len(families) + 2)))
    image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=max(0.01, float(matrix.max())))
    axis.set_xticks(np.arange(len(metrics)), [METRICS[metric][0] for metric in metrics], rotation=25, ha="right")
    axis.set_yticks(np.arange(len(families)), families)
    axis.set_title("Strongest marginal association within each uncertainty family")
    figure_obj.colorbar(image, ax=axis, label="Absolute Spearman rank association")
    figure_obj.tight_layout()
    figure_obj.savefig(plots / "uncertainty_family_screening.png", dpi=180)
    plt.close(figure_obj)


def _derived_plot(derived: pd.DataFrame, plots: Path) -> None:
    columns = [
        ("effective_wheel_power_multiplier", "Effective wheel-power multiplier"),
        ("effective_traction_multiplier_vs_nominal", "Effective traction multiplier vs nominal"),
        ("measured_traversal_pace_index", "Measured traversal pace index"),
        ("obstacle_severity_percentile_index", "Aggregate obstacle-severity percentile index"),
    ]
    available = [(column, label) for column, label in columns if column in derived]
    if not available:
        return
    figure_obj, axes = plt.subplots(len(available), 1, figsize=(10.5, 3.2 * len(available)))
    axes = np.atleast_1d(axes)
    for axis, (column, label) in zip(axes, available):
        values = pd.to_numeric(derived[column], errors="coerce").dropna()
        axis.hist(values, bins=min(24, max(8, int(np.sqrt(len(values))))))
        axis.axvline(values.median(), linewidth=1.2, label="median")
        axis.set_xlabel(label)
        axis.set_ylabel("Scenario count")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
    figure_obj.suptitle("Compounded uncertainty combinations")
    figure_obj.tight_layout()
    figure_obj.savefig(plots / "derived_uncertainty_combinations.png", dpi=180)
    plt.close(figure_obj)


def _convergence_plot(rows: pd.DataFrame, manifest: Mapping[str, Any], plots: Path) -> None:
    metrics = [metric for metric in METRICS if metric in rows]
    if not metrics:
        return
    paired = bool(manifest.get("track_case_pairing_complete", False)) and "base_draw_id" in rows
    if paired:
        draw_order = sorted(pd.to_numeric(rows["base_draw_id"], errors="coerce").dropna().unique())
        checkpoints = range(3, len(draw_order) + 1)
        x_label = "Common base draws included"
    else:
        draw_order = list(range(len(rows)))
        checkpoints = range(max(10, len(rows) // 20), len(rows) + 1, max(1, len(rows) // 30))
        if checkpoints and (len(rows) not in checkpoints):
            checkpoints = list(checkpoints) + [len(rows)]
        x_label = "Joint scenarios included"

    figure_obj, axes = plt.subplots(len(metrics), 1, figsize=(11, 4.0 * len(metrics)))
    axes = np.atleast_1d(axes)
    for axis, metric in zip(axes, metrics):
        x_values = []
        p10_values = []
        median_values = []
        p90_values = []
        for count in checkpoints:
            if paired:
                subset = rows[pd.to_numeric(rows["base_draw_id"], errors="coerce").isin(draw_order[:count])]
            else:
                subset = rows.iloc[:count]
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            if len(values) < 3:
                continue
            x_values.append(count)
            p10_values.append(values.quantile(0.1))
            median_values.append(values.median())
            p90_values.append(values.quantile(0.9))
        axis.plot(x_values, median_values, label="median")
        axis.plot(x_values, p10_values, label="p10", alpha=0.8)
        axis.plot(x_values, p90_values, label="p90", alpha=0.8)
        axis.set_xlabel(x_label)
        axis.set_ylabel(f"{METRICS[metric][0]} [{METRICS[metric][1]}]")
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure_obj.suptitle("Convergence of central and tail summaries")
    figure_obj.tight_layout()
    figure_obj.savefig(plots / "full_uncertainty_convergence.png", dpi=180)
    plt.close(figure_obj)


def _track_case_plots(
    summary: pd.DataFrame,
    paired_effects: pd.DataFrame,
    manifest: Mapping[str, Any],
    plots: Path,
) -> None:
    if summary.empty:
        return
    for metric_prefix, filename, xlabel, title in (
        ("bounded_lap", "track_case_lap_time.png", "Bounded lap time [s]", "Conditional outcomes by track reconstruction"),
        ("penalty", "track_case_penalty.png", "Bounded minus infinite lap time [s]", "Conditional finite-range penalty by track reconstruction"),
    ):
        median_col = f"{metric_prefix}_median_s"
        p10_col = f"{metric_prefix}_p10_s"
        p90_col = f"{metric_prefix}_p90_s"
        if median_col not in summary:
            continue
        data = summary.dropna(subset=[median_col]).sort_values(median_col)
        y = np.arange(len(data))
        med = data[median_col].to_numpy(float)
        low = data[p10_col].to_numpy(float)
        high = data[p90_col].to_numpy(float)
        figure_obj, axis = plt.subplots(figsize=(11, max(5.5, 0.4 * len(data))))
        axis.errorbar(med, y, xerr=np.vstack((med - low, high - med)), fmt="o", capsize=3)
        axis.set_yticks(y, data["track_case_id"])
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(True, axis="x", alpha=0.25)
        figure_obj.tight_layout()
        figure_obj.savefig(plots / filename, dpi=180)
        plt.close(figure_obj)

    if paired_effects.empty:
        return
    data = paired_effects[paired_effects["metric"] == "bounded_lap_time_s"].sort_values("delta_median")
    if data.empty:
        return
    y = np.arange(len(data))
    med = data["delta_median"].to_numpy(float)
    low = data["delta_p10"].to_numpy(float)
    high = data["delta_p90"].to_numpy(float)
    figure_obj, axis = plt.subplots(figsize=(11, max(5.5, 0.42 * len(data))))
    axis.errorbar(med, y, xerr=np.vstack((med - low, high - med)), fmt="o", capsize=3)
    axis.axvline(0, linewidth=1)
    axis.set_yticks(y, data["track_case_id"])
    axis.set_xlabel("Paired change from nominal-track lap time [s]")
    axis.set_title("Isolated track-reconstruction effect under common draws")
    axis.grid(True, axis="x", alpha=0.25)
    figure_obj.tight_layout()
    figure_obj.savefig(plots / "paired_track_case_effects.png", dpi=180)
    plt.close(figure_obj)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------


def _build_html(
    *,
    rows: pd.DataFrame,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    legacy_convergence: Mapping[str, Any],
    parameter_inventory: pd.DataFrame,
    gate_inventory: pd.DataFrame,
    track_inventory: pd.DataFrame,
    driver_table: pd.DataFrame,
    family_table: pd.DataFrame,
    derived: pd.DataFrame,
    convergence: pd.DataFrame,
    adequacy: pd.DataFrame,
    track_case_summary: pd.DataFrame,
    paired_track_effects: pd.DataFrame,
    scenario_explorer: pd.DataFrame,
    loss_summary: pd.DataFrame,
    ratio_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    nominal_reference: Mapping[str, Any] | None,
    plots: Path,
) -> str:
    scenario_count = int(rows["replicate"].nunique()) if "replicate" in rows else len(rows)
    track_count = int(rows["track_case_id"].nunique()) if "track_case_id" in rows else 1
    base_draw_count = int(manifest.get("base_draw_count", rows["base_draw_id"].nunique() if "base_draw_id" in rows else scenario_count))
    structural_count = int(len(parameter_inventory))
    gate_count = int(len(gate_inventory))
    identity_count = int(
        pd.DataFrame(
            [
                {
                    "run_id": record,
                }
                for record in []
            ]
        ).shape[0]
    )
    # identity count is more directly available from the saved scenario records via
    # the derived table's run/lap columns.
    if not derived.empty and {"run_id", "lap_id"}.issubset(derived.columns):
        identity_count = int(derived[["run_id", "lap_id", "vehicle_id", "driver_id"]].drop_duplicates().shape[0])
    else:
        identity_count = int(manifest.get("paired_gate_identity_count", 0))

    bounded = _distribution_stats(rows, "bounded_lap_time_s")
    reference = _distribution_stats(rows, "infinite_reference_lap_time_s")
    penalty = _distribution_stats(rows, "lap_time_penalty_vs_infinite_s")
    opportunity = _distribution_stats(rows, "finite_ratio_opportunity_loss_energy_kj")
    positive_count = int((pd.to_numeric(rows.get("lap_time_penalty_vs_infinite_s"), errors="coerce") > 0).sum())
    paired_track = bool(manifest.get("track_case_pairing_complete", False))

    cards = metric_cards(
        [
            ("Joint scenarios", str(scenario_count), "note"),
            ("Common stochastic draws", str(base_draw_count), "note"),
            ("Track interpretations", str(track_count), "note"),
            ("Structural parameters", str(structural_count), "note"),
            ("Gate-related targets", str(gate_count), "note"),
            ("Measured lap identities", str(identity_count), "note"),
            ("Bounded lap median", _format_stat(bounded, "median", " s"), "warning"),
            ("Finite-range penalty median", _format_stat(penalty, "median", " s"), "warning"),
            ("Positive penalty scenarios", f"{positive_count}/{scenario_count}", "good" if positive_count == scenario_count else "warning"),
        ]
    )

    body = cards
    body += (
        '<div class="section-intro"><strong>Executive answer.</strong>'
        "This is a joint scenario study, not a calibrated probability forecast. "
        "Structural inputs and coherent measured traversals are sampled, while track reconstructions are "
        "unweighted epistemic alternatives. The most defensible result is the direction and broad scale of "
        "the bounded-versus-infinite comparison; exact tails require convergence review.</div>"
    )
    body += "<h2>Study adequacy</h2>"
    body += '<p class="table-note">This panel states what the run can support before presenting detailed results.</p>'
    body += dataframe_table(
        adequacy,
        sticky_columns=("area",),
        max_rows=100,
        sortable=True,
        searchable=False,
        compact=True,
    )

    body += "<h2>What was varied?</h2>"
    body += (
        '<div class="section-intro"><strong>Scope of the joint study.</strong>'
        "The tables below are the complete input contract. They appear before the result plots so the size of "
        "each response is always interpreted relative to the range that was actually tested.</div>"
    )
    body += "<h3>Declared structural parameters and ranges</h3>"
    body += dataframe_table(
        parameter_inventory,
        columns=(
            "family", "parameter_path", "nominal", "unit", "distribution", "declared_range",
            "correlation_group", "source_kind", "source_reference",
        ),
        sticky_columns=("family", "parameter_path"),
        max_rows=1000,
        searchable=True,
        search_placeholder="Search all structural parameters…",
        compact=True,
    )
    body += "<h3>Measured traversal targets replayed from real laps</h3>"
    body += '<p class="table-note">These are gate-related target quantities, not the number of independent physical obstacles. Entry targets and response minima can both appear.</p>'
    body += dataframe_table(
        gate_inventory,
        sticky_columns=("gate_target_id",),
        max_rows=1000,
        searchable=True,
        search_placeholder="Search gate targets…",
        compact=True,
    )
    body += "<h3>Epistemic track interpretations</h3>"
    body += '<p class="table-note">Every listed track case is an admitted reconstruction alternative. They are not probability weighted.</p>'
    body += dataframe_table(
        track_inventory,
        sticky_columns=("track_case_id", "label"),
        max_rows=100,
        searchable=True,
        search_placeholder="Search track interpretations…",
        compact=True,
    )

    body += "<h2>Absolute vehicle performance</h2>"
    body += (
        '<div class="section-intro"><strong>What this section shows.</strong>'
        "Read absolute lap time before the paired CVT penalty. A worse engine or less efficient drivetrain can "
        "make the whole vehicle slower while also shrinking the apparent advantage of the infinite reference.</div>"
    )
    performance_cards = [
        ("Bounded p10 / median / p90", _triplet(bounded, " s"), "warning"),
        ("Infinite p10 / median / p90", _triplet(reference, " s"), "note"),
    ]
    if nominal_reference:
        performance_cards.extend(
            [
                ("Exact nominal bounded", _fmt(nominal_reference.get("bounded_lap_time_s"), " s"), "note"),
                ("Exact nominal penalty", _fmt(nominal_reference.get("lap_time_penalty_vs_infinite_s"), " s"), "note"),
            ]
        )
    body += metric_cards(performance_cards)
    if not nominal_reference:
        body += '<div class="card warning"><strong>Legacy artifact note.</strong>This completed run predates the saved exact-nominal reference. Future runs save <code>nominal_reference.json</code> automatically; this report therefore does not invent a nominal marker.</div>'
    body += figure(plots / "absolute_lap_time_distribution.png", "Absolute bounded and infinite-reference lap times across the constructed joint scenario set.")
    body += figure(plots / "completion_by_design.png", "Completion fraction. Incomplete cases must remain visible rather than being silently removed.")

    body += "<h2>CVT finite-range limitation</h2>"
    body += (
        '<div class="section-intro"><strong>What this section shows.</strong>'
        "The paired comparison holds every scenario realization fixed and changes only the bounded versus infinite-ratio CVT. "
        "A smaller penalty is not automatically a better vehicle: low wheel power can reduce the opportunity available to the reference.</div>"
    )
    body += metric_cards(
        [
            ("Penalty p10 / median / p90", _triplet(penalty, " s"), "warning"),
            ("Opportunity p10 / median / p90", _triplet(opportunity, " kJ"), "warning"),
            ("Positive penalty", f"{positive_count}/{scenario_count} scenarios", "good" if positive_count == scenario_count else "warning"),
        ]
    )
    body += figure(plots / "paired_penalty_distribution.png", "Paired bounded-minus-infinite lap-time penalty for every joint scenario.")
    body += dataframe_table(
        threshold_summary,
        sticky_columns=("threshold_s",),
        max_rows=50,
        sortable=True,
        compact=True,
    )
    body += figure(plots / "ratio_occupancy_distribution.png", "Time spent at the maximum ratio, within the variable region, and pinned at the minimum-ratio/overdrive boundary.")
    body += dataframe_table(
        ratio_summary,
        sticky_columns=("region",),
        max_rows=20,
        sortable=True,
        compact=True,
    )

    body += "<h2>Sources of uncertainty</h2>"
    body += (
        '<div class="section-intro"><strong>How to read these rankings.</strong>'
        "Plots show the leading signed marginal rank associations for readability. The complete sortable explorer underneath contains every sampled structural and traversal input for all three headline outputs. "
        "Associations are screening signals, not additive causal shares.</div>"
    )
    body += figure(plots / "uncertainty_family_screening.png", "Strongest marginal association within each uncertainty family. Use this family view before interpreting tiny differences between individual ranks.")
    body += figure(plots / "absolute_lap_time_uncertainty_drivers.png", "Leading signed associations with absolute bounded lap time.")
    body += figure(plots / "finite_ratio_penalty_uncertainty_drivers.png", "Leading signed associations with the paired finite-range time penalty.")
    body += figure(plots / "opportunity_loss_uncertainty_drivers.png", "Leading signed associations with finite-ratio opportunity loss.")
    body += "<h3>Complete driver explorer</h3>"
    body += '<p class="table-note">All inputs are present—not only the top ten. Click any header to sort ascending, descending, then return to original rank order.</p>'
    body += dataframe_table(
        driver_table,
        columns=(
            "metric_name", "rank", "family", "path", "input_type", "sample_count",
            "observed_minimum", "observed_median", "observed_maximum",
            "spearman_rank_correlation", "pearson_correlation", "direction",
            "relative_screening_importance",
        ),
        sticky_columns=("metric_name", "path"),
        max_rows=5000,
        searchable=True,
        search_placeholder="Search every uncertainty driver…",
    )
    body += "<details><summary>All-driver plots</summary>"
    body += figure(plots / "absolute_lap_time_uncertainty_drivers_all.png", "Every marginal driver of absolute bounded lap time.")
    body += figure(plots / "finite_ratio_penalty_uncertainty_drivers_all.png", "Every marginal driver of the finite-range penalty.")
    body += figure(plots / "opportunity_loss_uncertainty_drivers_all.png", "Every marginal driver of opportunity loss.")
    body += "</details>"

    body += "<h2>Track interpretation</h2>"
    if paired_track:
        body += (
            '<div class="card good"><strong>Paired track-case design.</strong>'
            "The same structural draw and measured lap identity were replayed on every track reconstruction, so changes from nominal isolate the reconstruction choice.</div>"
        )
        body += figure(plots / "paired_track_case_effects.png", "Paired change in bounded lap time from the nominal track for every reconstruction alternative.")
        body += dataframe_table(
            paired_track_effects,
            sticky_columns=("metric_name", "track_case_id"),
            max_rows=1000,
            searchable=True,
            search_placeholder="Search paired track-case effects…",
        )
    else:
        body += (
            '<div class="card warning"><strong>Legacy unpaired track-case layout.</strong>'
            "This 120-scenario run assigned different structural and measured-traversal draws to different track cases. The plots below describe outcomes observed within each case, but do not isolate causal track-case effects. "
            "The updated runner fixes this by crossing common draws with every track interpretation.</div>"
        )
    body += figure(plots / "track_case_lap_time.png", "Conditional bounded lap-time distributions by track case. Interpret causally only when the report states that track cases are paired.")
    body += figure(plots / "track_case_penalty.png", "Conditional finite-range penalty distributions by track case.")
    body += dataframe_table(
        track_case_summary,
        sticky_columns=("track_case_id",),
        max_rows=1000,
        searchable=True,
        search_placeholder="Search track-case outcomes…",
        compact=True,
    )

    body += "<h2>Physical mechanisms and compounded inputs</h2>"
    body += (
        '<div class="section-intro"><strong>Why derived combinations matter.</strong>'
        "Engine power and efficiency multiply to determine delivered wheel power; traction inputs multiply to determine force capacity. Showing the combined distributions helps reveal accidental double counting or implausibly broad tails.</div>"
    )
    body += figure(plots / "derived_uncertainty_combinations.png", "Derived wheel-power, traction, measured-pace, and obstacle-severity combinations.")
    body += figure(plots / "physical_loss_distribution.png", "Median physical losses with p10–p90 joint-scenario ranges. Opportunity loss is excluded because it is counterfactual rather than dissipative.")
    body += dataframe_table(
        loss_summary,
        sticky_columns=("loss_mechanism",),
        max_rows=50,
        sortable=True,
        compact=True,
    )

    body += "<h2>Convergence and effective evidence</h2>"
    body += (
        '<div class="section-intro"><strong>What this section shows.</strong>'
        "Central estimates can settle before tails. For the new fully crossed layout, the effective stochastic sample size is the number of common base draws, not the total number of track-case combinations.</div>"
    )
    body += figure(plots / "full_uncertainty_convergence.png", "Running p10, median, and p90 estimates as scenarios or common base draws accumulate.")
    body += dataframe_table(
        convergence,
        sticky_columns=("metric_name",),
        max_rows=50,
        sortable=True,
        compact=True,
    )
    body += (
        '<div class="card note"><strong>Recommended final layout.</strong>'
        "Use at least 30 common structural/traversal draws crossed with every admitted track case for a serious engineering report; 50 per track case is preferable for more stable p10–p90 statements. Stop based on convergence, not only a preset count.</div>"
    )

    body += "<h2>Scenario explorer</h2>"
    body += '<p class="table-note">Representative fastest, median, slowest, and largest-penalty cases with the key physical inputs that created them.</p>'
    body += dataframe_table(
        scenario_explorer,
        sticky_columns=("scenario_role", "replicate"),
        max_rows=100,
        sortable=True,
    )

    body += "<h2>Detailed supporting appendices</h2>"
    body += "<details><summary>Complete family-screening table</summary>"
    body += dataframe_table(
        family_table,
        sticky_columns=("metric_name", "family"),
        max_rows=1000,
        searchable=True,
        search_placeholder="Search uncertainty families…",
    )
    body += "</details>"
    body += "<details><summary>Complete derived-input scenario table</summary>"
    body += dataframe_table(
        derived,
        sticky_columns=("replicate", "track_case_id"),
        max_rows=10000,
        searchable=True,
        search_placeholder="Search derived scenario inputs…",
    )
    body += "</details>"
    body += "<details><summary>Legacy convergence JSON</summary><pre>"
    body += json.dumps(legacy_convergence, indent=2, sort_keys=True)
    body += "</pre></details>"
    body += "<details><summary>Sampling and evidence manifest</summary><pre>"
    body += json.dumps(_manifest_subset(manifest), indent=2, sort_keys=True)
    body += "</pre></details>"

    body += "<h2>Interpretation limits</h2><ul>"
    body += "<li>Track reconstructions are unweighted epistemic alternatives, so pooled percentiles are joint-scenario percentiles rather than calibrated credible intervals.</li>"
    body += "<li>Marginal driver rankings are not additive variance shares and can overlap strongly for correlated gate speeds or multiplicative physical inputs.</li>"
    body += "<li>Full uncertainty quantifies the answer range under declared contracts; it cannot compensate for missing physics or an unjustified input range.</li>"
    body += "<li>Telemetry elevation uncertainty remains outside the force model while grade force is disabled.</li>"
    body += "</ul>"

    return render_page(
        title=REPORTS["full_uncertainty"].title,
        subtitle=REPORTS["full_uncertainty"].question,
        body=body,
        report_key="full_uncertainty",
        source_note="Regenerated entirely from saved scenario, input-contract, and result artifacts; no simulation was rerun.",
    )


def _distribution_stats(rows: pd.DataFrame, column: str) -> dict[str, float]:
    if column not in rows:
        return {}
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        "count": float(len(values)),
        "minimum": float(values.min()),
        "p10": float(values.quantile(0.1)),
        "median": float(values.median()),
        "p90": float(values.quantile(0.9)),
        "maximum": float(values.max()),
    }


def _fmt(value: Any, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(number) else f"{number:.3f}{suffix}"


def _format_stat(stats: Mapping[str, float], key: str, suffix: str) -> str:
    return _fmt(stats.get(key), suffix)


def _triplet(stats: Mapping[str, float], suffix: str) -> str:
    if not stats:
        return "n/a"
    return f"{stats['p10']:.2f} / {stats['median']:.2f} / {stats['p90']:.2f}{suffix}"


def _manifest_subset(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "study_name",
        "study_type",
        "vehicle_id",
        "sampling_mode",
        "sampling_layout",
        "scenario_count",
        "base_draw_count",
        "scenarios_per_track_case",
        "track_case_pairing_complete",
        "track_case_assignment",
        "sampling_replicates_interpretation",
        "sampled_input_paths",
        "sampled_gate_ids",
        "gate_sampling_policy",
        "paired_gate_identity_count",
        "common_gate_identity_count_across_track_ensemble",
        "track_ensemble_case_count",
        "track_ensemble_case_ids",
        "track_ensemble_policy",
        "uncertainty_not_propagated",
        "numerical_quality",
        "evidence_assessment",
    )
    return {key: manifest.get(key) for key in keys if key in manifest}
