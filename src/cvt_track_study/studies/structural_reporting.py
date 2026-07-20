"""Defensible, regenerable HTML reporting for structural sensitivity.

The report is generated entirely from completed machine artifacts. No vehicle
or drivetrain simulation is required to regenerate it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import html
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvt_track_study.reports.html import dataframe_table, figure, metric_cards, render_page


_DEFAULT_HEADLINE_METRICS = (
    "bounded_lap_time_s",
    "bounded_maximum_speed_kmh",
    "bounded_engine_energy_kj",
    "bounded_obstacle_loss_energy_kj",
    "lap_time_penalty_vs_infinite_s",
    "finite_ratio_opportunity_loss_energy_kj",
)

_LEGACY_PLOT_ALIASES = {
    "bounded_maximum_speed_kmh": "structural_maximumpeed_tornado.png",
    "bounded_engine_energy_kj": "structural_engine_tornado.png",
}

_LEVEL_COLUMNS = (
    "parameter_path", "design_id", "level_kind", "level_probability",
    "design_value", "design_value_si", "design_choice_value",
    "bounded_completed", "bounded_termination_reason", "bounded_lap_time_s",
    "bounded_maximum_speed_kmh", "bounded_average_speed_kmh", "bounded_distance_m",
    "bounded_engine_energy_kj", "bounded_transmitted_energy_kj",
    "bounded_drivetrain_loss_energy_kj", "bounded_clutch_loss_energy_kj",
    "bounded_engine_operating_shortfall_energy_kj", "bounded_tire_slip_loss_energy_kj",
    "bounded_brake_loss_energy_kj", "bounded_rolling_loss_energy_kj",
    "bounded_aerodynamic_loss_energy_kj", "bounded_obstacle_loss_energy_kj",
    "bounded_time_maximum_ratio_s", "bounded_time_variable_ratio_s",
    "bounded_time_minimum_ratio_s", "bounded_time_braking_s",
    "bounded_time_traction_limited_s", "lap_time_penalty_vs_infinite_s",
    "finite_ratio_opportunity_loss_energy_kj",
)


def write_structural_outputs(
    *, output: Path, rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any], manifest: Mapping[str, Any],
    input_contracts: Mapping[str, Any],
) -> None:
    """Write all structural artifacts and the complete HTML report."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Structural reporting requires at least one parameter level.")

    range_frame = _metric_range_frame(summary)
    level_frame = _parameter_level_frame(frame)
    input_ranges = _input_range_frame(frame, input_contracts)
    priorities = _measurement_priority_frame(summary, input_ranges)
    families = _uncertainty_family_frame(input_ranges)
    nominal = _nominal_summary_frame(frame)

    _write_frame(output / "structural_metric_ranges.csv", range_frame)
    _write_frame(output / "structural_parameter_levels.csv", level_frame)
    _write_frame(output / "structural_input_ranges.csv", input_ranges)
    _write_frame(output / "structural_measurement_priorities.csv", priorities)
    _write_frame(output / "structural_uncertainty_families.csv", families)
    _write_frame(output / "structural_nominal_summary.csv", nominal)

    plots = _write_plots(
        output=output,
        frame=frame,
        summary=summary,
        input_ranges=input_ranges,
        top_count=int(manifest.get("structural_report_top_parameter_count", 15)),
        response_count=int(manifest.get("structural_response_curve_count", 6)),
    )
    _write_html(
        output / "structural_sensitivity_report.html",
        output=output,
        frame=frame,
        summary=summary,
        manifest=manifest,
        input_contracts=input_contracts,
        range_frame=range_frame,
        level_frame=level_frame,
        input_ranges=input_ranges,
        priorities=priorities,
        families=families,
        nominal=nominal,
        plots=plots,
    )
    _append_report_links(output)


def regenerate_structural_outputs(output: Path) -> Path:
    """Regenerate plots and HTML from completed artifacts without simulation."""
    output = output.resolve()
    required = ("replicate_results.csv", "summary.json", "run_manifest.json", "input_contracts.json")
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError("Cannot regenerate structural report; missing " + ", ".join(missing))
    write_structural_outputs(
        output=output,
        rows=pd.read_csv(output / "replicate_results.csv").to_dict(orient="records"),
        summary=_read_json(output / "summary.json"),
        manifest=_read_json(output / "run_manifest.json"),
        input_contracts=_read_json(output / "input_contracts.json"),
    )
    return output / "structural_sensitivity_report.html"


def _metric_range_frame(summary: Mapping[str, Any]) -> pd.DataFrame:
    definitions = _metric_definitions(summary)
    records: list[dict[str, Any]] = []
    parameters = summary.get("parameters", {})
    if not isinstance(parameters, Mapping):
        return pd.DataFrame()
    for parameter_path, parameter in parameters.items():
        if not isinstance(parameter, Mapping):
            continue
        metrics = parameter.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        for metric, row in metrics.items():
            if not isinstance(row, Mapping):
                continue
            definition = definitions.get(str(metric), {})
            records.append({
                "parameter_path": str(parameter_path),
                "category": parameter.get("category", ""),
                "metric": str(metric),
                "label": row.get("label", definition.get("label", metric)),
                "unit": row.get("unit", definition.get("unit", "")),
                "nominal": row.get("nominal"),
                "minimum": row.get("minimum"),
                "maximum": row.get("maximum"),
                "span": row.get("span"),
                "minimum_change_from_nominal": row.get("minimum_change_from_nominal"),
                "maximum_change_from_nominal": row.get("maximum_change_from_nominal"),
                "maximum_abs_change_from_nominal": row.get("maximum_abs_change_from_nominal"),
                "maximum_abs_percent_change_from_nominal": row.get("maximum_abs_percent_change_from_nominal"),
                "minimum_design_id": row.get("minimum_design_id", ""),
                "maximum_design_id": row.get("maximum_design_id", ""),
            })
    return pd.DataFrame(records)


def _parameter_level_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[[name for name in _LEVEL_COLUMNS if name in frame.columns]].copy()


def _input_range_frame(frame: pd.DataFrame, input_contracts: Mapping[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for parameter_path, levels in frame.groupby("parameter_path", sort=True):
        path = str(parameter_path)
        contract_row = input_contracts.get(path, {})
        contract = contract_row.get("contract", {}) if isinstance(contract_row, Mapping) else {}
        uncertainty = contract.get("uncertainty", {}) if isinstance(contract, Mapping) else {}
        source = contract.get("source", {}) if isinstance(contract, Mapping) else {}
        unit = str(contract.get("unit", "")) if isinstance(contract, Mapping) else ""
        raw_values = levels.get("design_value", pd.Series(dtype=object)).dropna().tolist()
        numeric_values = pd.to_numeric(pd.Series(raw_values), errors="coerce")
        all_numeric = bool(raw_values) and int(numeric_values.notna().sum()) == len(raw_values)
        if all_numeric:
            minimum: Any = float(numeric_values.min())
            maximum: Any = float(numeric_values.max())
            range_text = _range_text(minimum, maximum, unit)
        else:
            minimum, maximum = "", ""
            range_text = ", ".join(str(value) for value in dict.fromkeys(raw_values))
        records.append({
            "parameter_path": path,
            "category": _path_category(path),
            "nominal": contract.get("nominal", "") if isinstance(contract, Mapping) else "",
            "unit": unit,
            "tested_minimum": minimum,
            "tested_maximum": maximum,
            "tested_range": range_text,
            "tested_level_count": int(len(levels)),
            "distribution": uncertainty.get("distribution", "") if isinstance(uncertainty, Mapping) else "",
            "uncertainty_role": uncertainty.get("role", "") if isinstance(uncertainty, Mapping) else "",
            "source_kind": source.get("kind", "") if isinstance(source, Mapping) else "",
            "source_reference": source.get("reference", "") if isinstance(source, Mapping) else "",
        })
    return pd.DataFrame(records)


def _measurement_priority_frame(summary: Mapping[str, Any], input_ranges: pd.DataFrame) -> pd.DataFrame:
    range_lookup = input_ranges.set_index("parameter_path").to_dict(orient="index") if not input_ranges.empty else {}
    energy = _ranking_lookup(summary, "bounded_engine_energy_kj")
    penalty = _ranking_lookup(summary, "lap_time_penalty_vs_infinite_s")
    speed = _ranking_lookup(summary, "bounded_maximum_speed_kmh")
    records = []
    for index, row in enumerate(_ranking(summary, "bounded_lap_time_s")[:12], start=1):
        path = str(row.get("path", ""))
        records.append({
            "priority": index,
            "parameter_path": path,
            "category": row.get("category", ""),
            "tested_range": range_lookup.get(path, {}).get("tested_range", ""),
            "maximum_abs_lap_time_change_s": row.get("maximum_abs_change_from_nominal"),
            "maximum_abs_engine_energy_change_kj": _ranking_change(energy.get(path)),
            "maximum_abs_finite_ratio_penalty_change_s": _ranking_change(penalty.get(path)),
            "maximum_abs_speed_change_kmh": _ranking_change(speed.get(path)),
            "why_it_matters": _priority_reason(path),
            "measurement_focus": _measurement_focus(path),
        })
    return pd.DataFrame(records)


def _uncertainty_family_frame(input_ranges: pd.DataFrame) -> pd.DataFrame:
    available = set(input_ranges.get("parameter_path", pd.Series(dtype=str)).astype(str))
    definitions = (
        ("delivered_wheel_power", ("drivetrain.engine.power_scale", "drivetrain.efficiency"),
         "available wheel power is approximately engine power scale multiplied by drivetrain efficiency",
         "Broad independent ranges can compound into a much wider wheel-power range than either one-at-a-time result suggests.",
         "Confirm that engine-map uncertainty and transmission-efficiency uncertainty represent separate measurements before joint sampling."),
        ("traction_capacity", ("track.surface.friction_coefficient", "vehicle.tire.peak_traction_scale", "vehicle.driven_normal_load_fraction"),
         "available longitudinal force is approximately surface friction multiplied by tire scale and driven normal load",
         "Several broad factors may describe overlapping uncertainty in the same unmeasured traction capacity.",
         "Calibrate or correlate these factors before treating all of them as independent in full uncertainty."),
    )
    records = []
    for family, members, relation, risk, action in definitions:
        present = [member for member in members if member in available]
        if len(present) >= 2:
            records.append({
                "family": family,
                "members": "; ".join(present),
                "physical_relationship": relation,
                "joint_sampling_risk": risk,
                "recommended_action": action,
            })
    return pd.DataFrame(records)


def _nominal_summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    nominal = _nominal_row(frame)
    definitions = {
        "bounded_lap_time_s": ("Bounded lap time", "s"),
        "infinite_reference_lap_time_s": ("Infinite-reference lap time", "s"),
        "bounded_maximum_speed_kmh": ("Maximum speed", "km/h"),
        "bounded_engine_energy_kj": ("Engine energy", "kJ"),
        "bounded_obstacle_loss_energy_kj": ("Obstacle loss", "kJ"),
        "lap_time_penalty_vs_infinite_s": ("Finite-ratio lap-time penalty", "s"),
        "finite_ratio_opportunity_loss_energy_kj": ("Finite-ratio opportunity loss", "kJ"),
        "bounded_time_maximum_ratio_s": ("Time at maximum ratio", "s"),
        "bounded_time_variable_ratio_s": ("Time in variable ratio", "s"),
        "bounded_time_minimum_ratio_s": ("Time at minimum ratio", "s"),
    }
    return pd.DataFrame([
        {"metric": metric, "label": label, "value": nominal.get(metric), "unit": unit}
        for metric, (label, unit) in definitions.items() if metric in nominal
    ])


def _write_plots(*, output: Path, frame: pd.DataFrame, summary: Mapping[str, Any],
                 input_ranges: pd.DataFrame, top_count: int, response_count: int) -> dict[str, Any]:
    paths: dict[str, Any] = {"tornado": {}, "response": []}
    for metric in _headline_metrics(summary):
        if not _ranking(summary, metric):
            continue
        path = output / _tornado_filename(metric)
        _tornado_plot(path, summary, input_ranges, metric, top_count=top_count)
        if path.is_file():
            paths["tornado"][metric] = path
            legacy = _LEGACY_PLOT_ALIASES.get(metric)
            if legacy:
                shutil.copyfile(path, output / legacy)
    heatmap = output / "structural_loss_mechanism_heatmap.png"
    _mechanism_heatmap(heatmap, summary, top_count=top_count)
    if heatmap.is_file(): paths["heatmap"] = heatmap
    comparison = output / "structural_absolute_vs_ratio_drivers.png"
    _absolute_vs_ratio_plot(comparison, summary, top_count=min(12, top_count))
    if comparison.is_file(): paths["absolute_vs_ratio"] = comparison
    ratio = output / "structural_ratio_occupancy_sensitivity.png"
    _ratio_occupancy_plot(ratio, summary, top_count=min(12, top_count))
    if ratio.is_file(): paths["ratio_occupancy"] = ratio

    response_manifest = []
    for row in _ranking(summary, "bounded_lap_time_s")[:response_count]:
        parameter = str(row.get("path", ""))
        if not parameter:
            continue
        plot = output / f"structural_response_{_slug(parameter)}.png"
        _response_curve_plot(plot, frame, parameter, input_ranges)
        if plot.is_file():
            paths["response"].append({"parameter_path": parameter, "plot": plot})
            response_manifest.append({"parameter_path": parameter, "plot": plot.name})
    (output / "structural_response_curves_manifest.json").write_text(
        json.dumps(response_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def _tornado_plot(path: Path, summary: Mapping[str, Any], input_ranges: pd.DataFrame,
                  metric: str, *, top_count: int) -> None:
    ranking = list(reversed(_ranking(summary, metric)[:top_count]))
    if not ranking: return
    range_lookup = input_ranges.set_index("parameter_path")["tested_range"].to_dict() if not input_ranges.empty else {}
    labels = []
    for row in ranking:
        parameter = str(row.get("path", ""))
        tested = str(range_lookup.get(parameter, ""))
        labels.append(parameter + (f"\n[{tested}]" if tested else ""))
    low = np.asarray([_float(row.get("minimum_change_from_nominal")) for row in ranking])
    high = np.asarray([_float(row.get("maximum_change_from_nominal")) for row in ranking])
    y = np.arange(len(ranking))
    fig, ax = plt.subplots(figsize=(12, max(6.0, 0.62 * len(ranking) + 1.8)))
    ax.barh(y, high - low, left=low, height=0.62)
    ax.scatter(np.zeros(len(ranking)), y, marker="|", s=90, zorder=4)
    ax.axvline(0.0, linewidth=1.1)
    ax.set_yticks(y, labels)
    definition = _metric_definition(summary, metric)
    ax.set_xlabel(f"Change from nominal [{definition['unit']}]")
    ax.set_title(f"One-at-a-time structural response: {definition['label']}\n"
                 "labels include the exact tested input range; bar endpoints are the minimum and maximum responses")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _mechanism_heatmap(path: Path, summary: Mapping[str, Any], *, top_count: int) -> None:
    metrics = [metric for metric in (
        "bounded_lap_time_s", "bounded_maximum_speed_kmh", "bounded_engine_energy_kj",
        "bounded_drivetrain_loss_energy_kj", "bounded_clutch_loss_energy_kj",
        "bounded_tire_slip_loss_energy_kj", "bounded_brake_loss_energy_kj",
        "bounded_rolling_loss_energy_kj", "bounded_aerodynamic_loss_energy_kj",
        "bounded_obstacle_loss_energy_kj", "bounded_time_maximum_ratio_s",
        "bounded_time_variable_ratio_s", "bounded_time_minimum_ratio_s",
    ) if _ranking(summary, metric)]
    if not metrics: return
    scores: dict[str, float] = {}
    for metric in metrics:
        for row in _ranking(summary, metric):
            p = str(row.get("path", "")); scores[p] = max(scores.get(p, 0.0), _float(row.get("relative_screening_importance")))
    parameters = [item[0] for item in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_count]]
    lookup = {metric: {str(row.get("path", "")): _float(row.get("relative_screening_importance")) for row in _ranking(summary, metric)} for metric in metrics}
    matrix = np.asarray([[lookup[m].get(p, 0.0) for m in metrics] for p in parameters], dtype=float)
    fig, ax = plt.subplots(figsize=(max(11, .75 * len(metrics) + 5), max(6, .48 * len(parameters) + 2)))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metrics)), [_metric_definition(summary, m)["label"] for m in metrics], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(parameters)), parameters)
    ax.set_title("Relative one-at-a-time structural influence by output mechanism")
    fig.colorbar(image, ax=ax, label="Influence relative to the strongest parameter for each metric")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _absolute_vs_ratio_plot(path: Path, summary: Mapping[str, Any], *, top_count: int) -> None:
    absolute = _ranking_lookup(summary, "bounded_lap_time_s")
    penalty = _ranking_lookup(summary, "lap_time_penalty_vs_infinite_s")
    ordered = []
    for row in _ranking(summary, "bounded_lap_time_s") + _ranking(summary, "lap_time_penalty_vs_infinite_s"):
        parameter = str(row.get("path", ""))
        if parameter and parameter not in ordered: ordered.append(parameter)
        if len(ordered) >= top_count: break
    if not ordered: return
    ordered = list(reversed(ordered)); y = np.arange(len(ordered)); width = .36
    fig, ax = plt.subplots(figsize=(12, max(6, .5 * len(ordered) + 2)))
    ax.barh(y - width/2, [_ranking_importance(absolute.get(p)) for p in ordered], height=width, label="absolute bounded lap time")
    ax.barh(y + width/2, [_ranking_importance(penalty.get(p)) for p in ordered], height=width, label="finite-ratio lap-time penalty")
    ax.set_yticks(y, ordered); ax.set_xlim(left=0.0)
    ax.set_xlabel("Relative one-at-a-time screening importance within each metric")
    ax.set_title("Inputs controlling absolute performance versus finite-ratio restriction\neach series is normalized to its own strongest parameter")
    ax.grid(True, axis="x", alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _ratio_occupancy_plot(path: Path, summary: Mapping[str, Any], *, top_count: int) -> None:
    metrics = ("bounded_time_maximum_ratio_s", "bounded_time_variable_ratio_s", "bounded_time_minimum_ratio_s")
    lookups = {m: _ranking_lookup(summary, m) for m in metrics}
    parameters = set().union(*(set(v) for v in lookups.values()))
    scores = {p: max(_ranking_change(lookups[m].get(p)) for m in metrics) for p in parameters}
    ordered = [item[0] for item in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_count]]
    if not ordered: return
    ordered = list(reversed(ordered)); y = np.arange(len(ordered)); width = .24
    fig, ax = plt.subplots(figsize=(12, max(6, .52 * len(ordered) + 2)))
    for index, (metric, label) in enumerate(zip(metrics, ("maximum ratio", "variable ratio", "minimum ratio"))):
        ax.barh(y + (index - 1)*width, [_ranking_change(lookups[metric].get(p)) for p in ordered], height=width, label=label)
    ax.set_yticks(y, ordered); ax.set_xlabel("Largest absolute change from nominal ratio-region time [s]")
    ax.set_title("Which inputs move the bounded CVT between its ratio regions?")
    ax.grid(True, axis="x", alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _response_curve_plot(path: Path, frame: pd.DataFrame, parameter: str, input_ranges: pd.DataFrame) -> None:
    subset = frame.loc[frame["parameter_path"].astype(str) == parameter].copy()
    required = {"bounded_lap_time_s", "reference_lap_time_s", "lap_time_penalty_vs_infinite_s"}
    if subset.empty or not required.issubset(subset.columns): return
    nominal = subset.loc[subset["level_kind"].astype(str) == "nominal"] if "level_kind" in subset else subset.head(1)
    if nominal.empty: nominal = subset.head(1)
    nominal_row = nominal.iloc[0]
    numeric = pd.to_numeric(subset["design_value"], errors="coerce")
    use_numeric = int(numeric.notna().sum()) == len(subset)
    if use_numeric:
        subset = subset.assign(_x=numeric).sort_values("_x"); x = subset["_x"].to_numpy(float); labels = None
    else:
        subset = subset.reset_index(drop=True); x = np.arange(len(subset), dtype=float)
        labels = subset.get("design_choice_value", subset["design_value"]).astype(str).tolist()
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for metric, label in (("bounded_lap_time_s", "bounded lap-time change"),
                          ("reference_lap_time_s", "infinite-reference lap-time change"),
                          ("lap_time_penalty_vs_infinite_s", "finite-ratio penalty change")):
        values = pd.to_numeric(subset[metric], errors="coerce").to_numpy(float)
        ax.plot(x, values - _float(nominal_row.get(metric)), marker="o", label=label)
    ax.axhline(0.0, linewidth=1.0)
    if labels is not None: ax.set_xticks(x, labels, rotation=25, ha="right")
    match = input_ranges.loc[input_ranges["parameter_path"] == parameter]
    unit = str(match.iloc[0].get("unit", "")) if not match.empty else ""
    display_unit = "" if unit in {"1", "dimensionless"} else unit
    ax.set_xlabel(parameter + (f" [{display_unit}]" if display_unit else "")); ax.set_ylabel("Change from nominal [s]")
    ax.set_title(f"Structural response curve: {parameter}\nabsolute performance and finite-ratio restriction are shown separately")
    ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _write_html(path: Path, *, output: Path, frame: pd.DataFrame, summary: Mapping[str, Any],
                manifest: Mapping[str, Any], input_contracts: Mapping[str, Any],
                range_frame: pd.DataFrame, level_frame: pd.DataFrame,
                input_ranges: pd.DataFrame, priorities: pd.DataFrame,
                families: pd.DataFrame, nominal: pd.DataFrame,
                plots: Mapping[str, Any]) -> None:
    nominal_lookup = nominal.set_index("metric")["value"].to_dict() if not nominal.empty else {}
    quality = summary.get("numerical_quality", {})
    completed = int(summary.get("completed_level_count", 0)); total = int(summary.get("level_count", len(frame)))
    all_complete = bool(total and completed == total)
    cards = metric_cards([
        ("Nominal bounded lap", _fmt(nominal_lookup.get("bounded_lap_time_s"), " s"), "good" if all_complete else "warning"),
        ("Nominal maximum speed", _fmt(nominal_lookup.get("bounded_maximum_speed_kmh"), " km/h"), "note"),
        ("Nominal engine energy", _fmt(nominal_lookup.get("bounded_engine_energy_kj"), " kJ"), "note"),
        ("Finite-ratio penalty", _fmt(nominal_lookup.get("lap_time_penalty_vs_infinite_s"), " s"), "warning"),
        ("Finite-ratio opportunity", _fmt(nominal_lookup.get("finite_ratio_opportunity_loss_energy_kj"), " kJ"), "warning"),
        ("Completed levels", f"{completed}/{total}", "good" if all_complete else "bad"),
    ])
    findings = '<div class="finding-grid">' + ''.join([
        _finding("Absolute lap time", _ranking_sentence(_first_ranking(summary, "bounded_lap_time_s"), "largest tested lap-time movement")),
        _finding("Finite-ratio restriction", _ranking_sentence(_first_ranking(summary, "lap_time_penalty_vs_infinite_s"), "largest movement in bounded-versus-infinite penalty")),
        _finding("Maximum speed", _ranking_sentence(_first_ranking(summary, "bounded_maximum_speed_kmh"), "largest tested maximum-speed movement")),
        _finding("Engine energy", _ranking_sentence(_first_ranking(summary, "bounded_engine_energy_kj"), "largest tested engine-energy movement")),
    ]) + '</div>'

    body = '''<nav class="report-nav">
<a href="#overview">Overview</a><a href="#drivers">Primary drivers</a>
<a href="#curves">Response curves</a><a href="#mechanisms">Mechanisms</a>
<a href="#measurement">Measurement priorities</a><a href="#quality">Quality</a>
<a href="#appendices">Supporting tables</a></nav>'''
    body += '<section id="overview"><h2>1. Executive summary and nominal reference</h2>'
    body += _section_intro("What this section shows", "The exact nominal simulation that every one-at-a-time case is compared against, followed by the highest-level engineering conclusions. Structural sensitivity ranks responses across the ranges you declared; it does not claim that the same parameter would dominate under a different range.")
    body += cards + findings
    body += '<div class="card note"><strong>Study contract.</strong> This is deterministic one-at-a-time screening. One structural input changes at a time while the track, measured gate realization, vehicle, and every other input remain nominal. The bars are physical response spans, not Monte Carlo confidence intervals.</div>'
    body += _efficiency_warning(input_ranges) + '</section>'

    body += '<section id="drivers"><h2>2. Primary structural drivers</h2>'
    body += _section_intro("What this section shows", "The tornado plots rank the largest one-at-a-time movement in each headline output. Every parameter label includes the exact tested input range so importance is not separated from the range that produced it.")
    tornado = plots.get("tornado", {})
    for metric in _headline_metrics(summary):
        plot = tornado.get(metric) if isinstance(tornado, Mapping) else None
        if plot:
            body += f'<h3>{html.escape(_metric_definition(summary, metric)["label"])}</h3>'
            body += figure(Path(plot), _tornado_caption(metric))
    if plots.get("absolute_vs_ratio"):
        body += '<h3>Absolute performance versus finite-ratio restriction</h3>'
        body += figure(Path(plots["absolute_vs_ratio"]), "This comparison prevents a smaller bounded-versus-infinite penalty from being mistaken for a faster vehicle. Each series is normalized within its own metric, so bar lengths compare rankings rather than physical units.")
    body += '</section>'

    body += '<section id="curves"><h2>3. Response curves for the leading parameters</h2>'
    body += _section_intro("What this section shows", "Every evaluated level is shown for the leading absolute-lap-time inputs. These plots reveal one-sided ranges, plateaus, diminishing returns, and nonlinear behavior that a tornado endpoint alone cannot show.")
    for response in plots.get("response", []):
        parameter = str(response["parameter_path"])
        plot = Path(response["plot"])
        body += f"<h3>{html.escape(parameter)}</h3>"
        body += figure(plot, f"Level-by-level response for {parameter}. Bounded and infinite-reference lap-time changes are separated from the change in their finite-ratio difference.")
    body += '</section>'

    body += '<section id="mechanisms"><h2>4. Physical mechanisms and CVT operating regions</h2>'
    body += _section_intro("What this section shows", "These plots connect the ranking to mechanism: which physical loss or operating state changes, and whether an input moves the CVT toward its maximum ratio, through the variable region, or against its minimum-ratio limit.")
    if (output / "physical_energy_attribution.png").is_file():
        body += figure(output / "physical_energy_attribution.png", "Nominal physical loss accounting provides context for the sensitivity rankings. Finite-ratio opportunity is counterfactual and is not added to dissipative losses.")
    if plots.get("heatmap"):
        body += figure(Path(plots["heatmap"]), "Relative influence by output mechanism. Each column is normalized to its own strongest parameter and should not be read as a common physical scale across columns.")
    if plots.get("ratio_occupancy"):
        body += figure(Path(plots["ratio_occupancy"]), "Largest change in time spent at maximum ratio, within the variable region, and at minimum ratio. This exposes tuning changes that may be mechanically important even when lap time barely moves.")
    if (output / "structural_sensitivity.png").is_file():
        body += '<details><summary>Legacy structural summary plot</summary>' + figure(output / "structural_sensitivity.png", "Original structural-sensitivity summary retained for continuity with earlier reports.") + '</details>'
    body += '</section>'

    body += '<section id="measurement"><h2>5. Measurement priorities and joint-uncertainty preparation</h2>'
    body += _section_intro("What this section shows", "The priority table turns the screening result into an evidence plan. It ranks parameters by absolute lap-time movement across their declared ranges and keeps energy, speed, and finite-ratio effects visible beside that ranking.")
    body += '<p class="table-note">Click any header to sort ascending, descending, then return to the original priority order. The parameter path remains fixed while scrolling.</p>'
    body += dataframe_table(priorities, max_rows=100, sticky_columns=("parameter_path",), compact=True, column_labels={
        "priority": "Priority", "parameter_path": "Parameter", "tested_range": "Tested range",
        "maximum_abs_lap_time_change_s": "Max |Δ lap time| [s]",
        "maximum_abs_engine_energy_change_kj": "Max |Δ engine energy| [kJ]",
        "maximum_abs_finite_ratio_penalty_change_s": "Max |Δ finite-ratio penalty| [s]",
        "maximum_abs_speed_change_kmh": "Max |Δ speed| [km/h]",
        "why_it_matters": "Why it matters", "measurement_focus": "Measurement focus",
    })
    if not families.empty:
        body += '<h3>Potentially compounding uncertainty families</h3><div class="card warning"><strong>Before full uncertainty.</strong> One-at-a-time results cannot reveal interactions. Where several broad inputs multiply into one physical capacity, confirm that they represent distinct unknowns or declare their correlation before sampling them together.</div>'
        body += dataframe_table(families, max_rows=30, sticky_columns=("family",), compact=True)
    body += '</section>'

    body += '<section id="quality"><h2>6. Numerical health and interpretation limits</h2>'
    body += _section_intro("What this section shows", "A sensitivity ranking is useful only when the cases completed and the physical balances stayed within tolerance. The limits below also state what this report intentionally does not answer.")
    body += dataframe_table(pd.DataFrame([{"check": key, "result": value} for key, value in quality.items()]), max_rows=100, sticky_columns=("check",), compact=True)
    body += '<ul><li>One-at-a-time screening does not measure interactions between uncertain inputs.</li><li>A parameter ranks highly only across the tested declared range; changing that range can change the ranking.</li><li>A smaller finite-ratio penalty can accompany a slower vehicle because the infinite-ratio reference also loses opportunity.</li><li>A zero response at tested levels does not prove that the mechanism is irrelevant in every combined scenario.</li><li>Full uncertainty remains the correct report for simultaneous variation and answer distributions.</li></ul></section>'

    body += '<section id="appendices"><h2>7. Detailed supporting tables</h2>'
    body += _section_intro("What this section shows", "These are the underlying machine-derived tables supporting the plots above. They are placed last so the report reads from conclusion to evidence rather than opening with wide data inventories.")
    body += '<details><summary>Declared structural input inventory and tested ranges</summary><p class="table-note">The parameter path remains fixed while scrolling. Click a header to cycle ascending, descending, and original order.</p>'
    body += dataframe_table(input_ranges, max_rows=500, sticky_columns=("parameter_path",), column_labels={"parameter_path": "Parameter", "tested_range": "Tested range"}) + '</details>'
    for metric in _headline_metrics(summary):
        ranking = pd.DataFrame(_ranking(summary, metric))
        if not ranking.empty:
            body += f'<details><summary>Complete ranking — {html.escape(_metric_definition(summary, metric)["label"])}</summary>'
            body += dataframe_table(ranking, max_rows=500, sticky_columns=("path",), column_labels={"path": "Parameter"}) + '</details>'
    body += '<details><summary>Complete metric ranges</summary><p class="table-note">One row per parameter and output metric. Parameter and metric label remain visible during horizontal scrolling.</p>'
    body += dataframe_table(range_frame, max_rows=max(1000, len(range_frame)), sticky_columns=("parameter_path", "label"), column_labels={"parameter_path": "Parameter", "label": "Output"}) + '</details>'
    body += '<details><summary>Every evaluated parameter level</summary><p class="table-note">The complete deterministic case table. Parameter and design level remain fixed while scrolling.</p>'
    body += dataframe_table(level_frame, max_rows=max(1000, len(level_frame)), sticky_columns=("parameter_path", "design_id"), column_labels={"parameter_path": "Parameter", "design_id": "Evaluated level"}) + '</details>'
    body += '<details><summary>Complete input contracts</summary><pre>' + html.escape(json.dumps(input_contracts, indent=2, sort_keys=True)) + '</pre></details>'
    body += '<h3>Machine-readable artifacts</h3><ul>' + ''.join(f'<li><code>{name}</code></li>' for name in (
        "structural_metric_ranges.csv", "structural_parameter_levels.csv", "structural_input_ranges.csv",
        "structural_measurement_priorities.csv", "structural_uncertainty_families.csv",
        "structural_nominal_summary.csv", "replicate_results.csv", "summary.json",
        "input_contracts.json", "run_manifest.json")) + '</ul></section>'

    path.write_text(render_page(
        title="Structural sensitivity report",
        subtitle="Which physical and modelling assumptions move the fixed nominal result, by how much, and through which mechanism?",
        body=body, report_key="structural_sensitivity",
        source_note="Regenerable from completed CSV and JSON artifacts; no simulation is required.",
    ), encoding="utf-8")


def _headline_metrics(summary: Mapping[str, Any]) -> tuple[str, ...]:
    raw = summary.get("headline_metrics")
    return tuple(str(metric) for metric in raw) if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else _DEFAULT_HEADLINE_METRICS


def _metric_definitions(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = summary.get("metric_definitions", {}); return raw if isinstance(raw, Mapping) else {}


def _metric_definition(summary: Mapping[str, Any], metric: str) -> dict[str, str]:
    raw = _metric_definitions(summary).get(metric, {})
    return {"label": str(raw.get("label", _humanize_metric(metric))), "unit": str(raw.get("unit", ""))} if isinstance(raw, Mapping) else {"label": _humanize_metric(metric), "unit": ""}


def _ranking(summary: Mapping[str, Any], metric: str) -> list[Mapping[str, Any]]:
    rankings = summary.get("rankings", {}); raw = rankings.get(metric, []) if isinstance(rankings, Mapping) else []
    return [item for item in raw if isinstance(item, Mapping)]


def _ranking_lookup(summary: Mapping[str, Any], metric: str) -> dict[str, Mapping[str, Any]]:
    return {str(item.get("path", "")): item for item in _ranking(summary, metric)}


def _first_ranking(summary: Mapping[str, Any], metric: str) -> Mapping[str, Any] | None:
    rows = _ranking(summary, metric); return rows[0] if rows else None


def _ranking_change(record: Mapping[str, Any] | None) -> float:
    return _float(record.get("maximum_abs_change_from_nominal")) if record else math.nan


def _ranking_importance(record: Mapping[str, Any] | None) -> float:
    return _float(record.get("relative_screening_importance")) if record else 0.0


def _nominal_row(frame: pd.DataFrame) -> Mapping[str, Any]:
    if "level_kind" in frame:
        rows = frame.loc[frame["level_kind"].astype(str) == "nominal"]
        if not rows.empty: return rows.iloc[0].to_dict()
    return frame.iloc[0].to_dict()


def _range_text(minimum: float, maximum: float, unit: str) -> str:
    suffix = f" {unit}" if unit and unit not in {"1", "dimensionless"} else ""
    return f"{minimum:.6g}{suffix}" if np.isclose(minimum, maximum) else f"{minimum:.6g}–{maximum:.6g}{suffix}"


def _fmt(value: Any, suffix: str = "") -> str:
    value = _float(value); return "n/a" if not np.isfinite(value) else f"{value:.3f}{suffix}"


def _float(value: Any) -> float:
    try: number = float(value)
    except (TypeError, ValueError): return math.nan
    return number if np.isfinite(number) else math.nan


def _humanize_metric(metric: str) -> str:
    return metric.replace("bounded_", "").replace("_kmh", "").replace("_kj", "").replace("_s", "").replace("_", " ").strip().title()


def _tornado_filename(metric: str) -> str:
    names = {
        "bounded_lap_time_s": "structural_lap_time_tornado.png",
        "bounded_maximum_speed_kmh": "structural_maximum_speed_tornado.png",
        "bounded_engine_energy_kj": "structural_engine_energy_tornado.png",
        "bounded_obstacle_loss_energy_kj": "structural_obstacle_loss_tornado.png",
        "lap_time_penalty_vs_infinite_s": "structural_lap_time_penalty_vs_infinite_tornado.png",
        "finite_ratio_opportunity_loss_energy_kj": "structural_finite_ratio_opportunity_loss_tornado.png",
    }
    return names.get(metric, f"structural_{_slug(metric)}_tornado.png")


def _tornado_caption(metric: str) -> str:
    return {
        "bounded_lap_time_s": "Absolute lap-time response across each declared input range. This is the primary ranking for improving the real-vehicle prediction.",
        "lap_time_penalty_vs_infinite_s": "Change in bounded-versus-infinite lap-time penalty. A smaller penalty is not automatically better because both vehicles may have become slower.",
        "finite_ratio_opportunity_loss_energy_kj": "Change in counterfactual finite-ratio opportunity. This is not a physical heat-loss component.",
        "bounded_engine_energy_kj": "Engine-energy response across the tested structural ranges. Energy sensitivity can remain material where gate constraints limit lap-time gains.",
        "bounded_maximum_speed_kmh": "Maximum-speed response. This highlights effective gearing and high-speed road-load uncertainty.",
        "bounded_obstacle_loss_energy_kj": "Total modeled obstacle-energy response. Large energy movement does not necessarily produce equally large lap-time movement.",
    }.get(metric, "One-at-a-time structural response across the exact declared input ranges.")


def _section_intro(title: str, text: str) -> str:
    return f'<div class="section-intro"><strong>{html.escape(title)}</strong>{html.escape(text)}</div>'


def _finding(title: str, text: str) -> str:
    return f'<div class="finding"><strong>{html.escape(title)}</strong>{html.escape(text)}</div>'


def _ranking_sentence(record: Mapping[str, Any] | None, phrase: str) -> str:
    if not record: return "No ranking was available."
    return f"{record.get('path', '')} produced the {phrase}: {_float(record.get('maximum_abs_change_from_nominal')):.3g} {record.get('unit', '')} across its declared range."


def _efficiency_warning(input_ranges: pd.DataFrame) -> str:
    row = input_ranges.loc[input_ranges["parameter_path"] == "drivetrain.efficiency"] if not input_ranges.empty else pd.DataFrame()
    if row.empty: return ""
    nominal, minimum = _float(row.iloc[0].get("nominal")), _float(row.iloc[0].get("tested_minimum"))
    if np.isfinite(nominal) and nominal >= .999 and np.isfinite(minimum) and minimum < .95:
        return '<div class="card warning"><strong>Optimistic nominal efficiency.</strong> The nominal drivetrain efficiency is 1.0 while the declared range extends substantially lower. Efficiency therefore has a one-sided response and the nominal physical-loss breakdown contains zero drivetrain loss. Treat the nominal case as idealized unless 100% efficiency is deliberate.</div>'
    return ""


def _priority_reason(path: str) -> str:
    return {
        "drivetrain.efficiency": "Directly controls delivered wheel power and modeled drivetrain loss.",
        "drivetrain.engine.power_scale": "Directly controls available engine power and exposes finite-ratio limitations.",
        "vehicle.rolling_resistance_coefficient": "Acts throughout the lap and changes both time and energy demand.",
        "track.surface.friction_coefficient": "Controls usable longitudinal force and traction-limited time.",
        "vehicle.tire.peak_traction_scale": "Controls tire force capacity and tire-slip behavior.",
        "vehicle.driven_normal_load_fraction": "Scales driven-tire normal load and therefore traction capacity.",
        "vehicle.tire_diameter": "Changes loaded rolling radius, effective gearing, and attainable speed.",
        "vehicle.aero.drag_area": "Changes high-speed road load and aerodynamic energy consumption.",
    }.get(path, "Large one-at-a-time movement in absolute lap time across the declared range.")


def _measurement_focus(path: str) -> str:
    return {
        "drivetrain.efficiency": "Measure input/output shaft power or adopt a defensible loaded efficiency map.",
        "drivetrain.engine.power_scale": "Validate the engine power curve under actual intake, exhaust, governor, and test conditions.",
        "vehicle.rolling_resistance_coefficient": "Use coast-down, tow-force, or controlled surface testing.",
        "track.surface.friction_coefficient": "Use controlled acceleration/braking data or surface-specific traction tests.",
        "vehicle.tire.peak_traction_scale": "Calibrate against measured wheel slip and longitudinal acceleration.",
        "vehicle.driven_normal_load_fraction": "Measure axle loads or estimate dynamic transfer with validated geometry.",
        "vehicle.tire_diameter": "Measure loaded rolling circumference at race pressure and vehicle load.",
        "vehicle.aero.drag_area": "Use coast-down data or a geometry-supported CdA estimate.",
    }.get(path, "Tighten the declared range using direct measurement or source-backed calibration.")



def _path_category(path: str) -> str:
    if path.startswith("drivetrain."):
        return "drivetrain"
    if path.startswith("vehicle."):
        return "vehicle"
    if path.startswith("driver."):
        return "driver"
    if path.startswith("track.surface."):
        return "surface"
    if path.startswith("track.features."):
        return "obstacle"
    return path.split(".", 1)[0] if "." in path else "other"

def _response_parameter_from_path(path: Path) -> str:
    return path.stem.removeprefix("structural_response_").replace("-", ".")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "parameter"


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, index=False)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_report_links(output: Path) -> None:
    addition = "\n## Structural review\n\n- [Open the complete HTML structural-sensitivity report](structural_sensitivity_report.html)\n- [Measurement priorities](structural_measurement_priorities.csv)\n- [Complete parameter-level results](structural_parameter_levels.csv)\n- [Complete metric ranges](structural_metric_ranges.csv)\n"
    for name in ("SUMMARY.md", "REPORT.md"):
        path = output / name
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if "structural_sensitivity_report.html" not in text:
                path.write_text(text.rstrip() + "\n" + addition, encoding="utf-8")
