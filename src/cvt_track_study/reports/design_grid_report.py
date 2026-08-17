"""Post-process a paired design result into richer two-dimensional grid views."""

from __future__ import annotations

import base64
import html
import json
import math
import os
import tomllib
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd

from .postprocess import _design_ranking, write_design_comparison_report


def write_multivariable_design_comparison_report(output: Path) -> Path:
    """Regenerate the normal report, then add Cartesian-grid decision views."""

    output = output.resolve()
    target = write_design_comparison_report(output)
    rows = _read_csv(output / "replicate_results.csv")
    manifest = _read_json(output / "run_manifest.json")
    paths = tuple(str(value) for value in manifest.get("design_variable_paths", ()))
    if len(paths) != 2 or rows.empty:
        return target

    ranking = _design_ranking(rows)
    if ranking.empty:
        return target

    summary = _attach_design_values(ranking, rows, paths)
    if summary.empty or any(path not in summary for path in paths):
        return target

    summary.to_csv(output / "design_grid_summary.csv", index=False)
    plots = output / "report_plots"
    plots.mkdir(exist_ok=True)

    grid_figures = _write_heatmaps(summary, paths, plots)
    extra_figures = _write_interpretability_views(summary, rows, ranking, paths, plots)
    speed_sanity_html = _speed_sanity_section(output, rows, ranking, plots)
    if not grid_figures and not extra_figures and not speed_sanity_html:
        return target

    document = target.read_text(encoding="utf-8")
    section = _grid_section(
        grid_figures=grid_figures,
        extra_figures=extra_figures,
        speed_sanity_html=speed_sanity_html,
        paths=paths,
        ranking=ranking,
    )
    anchor = "<h2>Absolute performance</h2>"
    if anchor in document:
        document = document.replace(anchor, section + anchor, 1)
    else:
        document = document.replace("</main>", section + "</main>", 1)
    target.write_text(document, encoding="utf-8")
    return target


def _attach_design_values(
    ranking: pd.DataFrame,
    rows: pd.DataFrame,
    paths: tuple[str, str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for design_id, group in rows.groupby("design_id", sort=False):
        values: dict[str, Any] = {}
        if "design_values_json" in group:
            try:
                parsed = json.loads(str(group["design_values_json"].iloc[0]))
                if isinstance(parsed, Mapping):
                    values = {str(key): value for key, value in parsed.items()}
            except (TypeError, ValueError, json.JSONDecodeError):
                values = {}
        for path in paths:
            column = f"design::{path}"
            if path not in values and column in group:
                values[path] = group[column].iloc[0]
        if all(path in values for path in paths):
            records.append(
                {
                    "design_id": design_id,
                    paths[0]: float(values[paths[0]]),
                    paths[1]: float(values[paths[1]]),
                }
            )
    axes = pd.DataFrame(records)
    if axes.empty:
        return pd.DataFrame()
    return ranking.merge(axes, on="design_id", how="inner")


def _write_heatmaps(
    summary: pd.DataFrame,
    paths: tuple[str, str],
    plots: Path,
) -> list[tuple[Path, str]]:
    x_path, y_path = paths
    metrics = (
        (
            "lap_time_median_s",
            "design_grid_lap_time.png",
            "Median bounded lap time",
            "Lap time [s]",
            2,
        ),
        (
            "paired_regret_median_s",
            "design_grid_regret.png",
            "Median paired regret",
            "Regret [s]",
            2,
        ),
        (
            "paired_win_fraction",
            "design_grid_win_fraction.png",
            "Paired win fraction in selected worlds",
            "Win fraction",
            3,
        ),
        (
            "opportunity_loss_median_kj",
            "design_grid_opportunity_loss.png",
            "Median finite-ratio opportunity loss",
            "Opportunity loss [kJ]",
            1,
        ),
        (
            "minimum_ratio_time_median_s",
            "design_grid_min_ratio_time.png",
            "Median time at minimum CVT reduction",
            "Time [s]",
            1,
        ),
        (
            "completion_fraction",
            "design_grid_completion.png",
            "Completion fraction",
            "Completion fraction",
            3,
        ),
    )
    generated: list[tuple[Path, str]] = []
    for metric, filename, title, colourbar, digits in metrics:
        if metric not in summary:
            continue
        matrix = summary.pivot_table(
            index=y_path,
            columns=x_path,
            values=metric,
            aggfunc="median",
        ).sort_index().sort_index(axis=1)
        if matrix.empty:
            continue
        figure_obj, axis = plt.subplots(
            figsize=(
                max(8.5, 1.2 * len(matrix.columns)),
                max(5.2, 0.8 * len(matrix.index)),
            )
        )
        data = matrix.to_numpy(float)
        image = axis.imshow(data, aspect="auto", interpolation="nearest")
        axis.set_xticks(np.arange(len(matrix.columns)), [f"{value:g}" for value in matrix.columns])
        axis.set_yticks(np.arange(len(matrix.index)), [f"{value:g}" for value in matrix.index])
        axis.set_xlabel(_axis_label(x_path))
        axis.set_ylabel(_axis_label(y_path))
        axis.set_title(title)
        for row_index in range(data.shape[0]):
            for column_index in range(data.shape[1]):
                value = data[row_index, column_index]
                if not np.isfinite(value):
                    continue
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.{digits}f}",
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.75,
                    },
                )
        figure_obj.colorbar(image, ax=axis, label=colourbar)
        figure_obj.tight_layout()
        path = plots / filename
        figure_obj.savefig(path, dpi=180)
        plt.close(figure_obj)
        generated.append((path, title))
    return generated


def _write_interpretability_views(
    summary: pd.DataFrame,
    rows: pd.DataFrame,
    ranking: pd.DataFrame,
    paths: tuple[str, str],
    plots: Path,
) -> list[tuple[Path, str]]:
    generated: list[tuple[Path, str]] = []
    contrast = _write_best_vs_worst_view(rows, ranking, plots)
    if contrast is not None:
        generated.append(contrast)
    generated.extend(_write_one_axis_slices(summary, paths, plots))
    plateau = _write_plateau_view(summary, paths, plots)
    if plateau is not None:
        generated.append(plateau)
    return generated


def _write_best_vs_worst_view(
    rows: pd.DataFrame,
    ranking: pd.DataFrame,
    plots: Path,
) -> tuple[Path, str] | None:
    if ranking.empty or len(ranking) < 2:
        return None
    best_id = str(ranking.iloc[0]["design_id"])
    worst_id = str(ranking.iloc[-1]["design_id"])
    keys = [
        column
        for column in ("replicate", "scenario_seed", "base_draw_id", "track_case_id")
        if column in rows.columns
    ]
    if not keys:
        return None
    columns = keys + ["bounded_lap_time_s", "bounded_completed"]
    left = rows.loc[rows["design_id"] == best_id, columns].copy()
    right = rows.loc[rows["design_id"] == worst_id, columns].copy()
    merged = left.merge(right, on=keys, suffixes=("_best", "_worst"))
    if merged.empty:
        return None
    best = pd.to_numeric(merged["bounded_lap_time_s_best"], errors="coerce")
    worst = pd.to_numeric(merged["bounded_lap_time_s_worst"], errors="coerce")
    if "bounded_completed_best" in merged:
        best.loc[~merged["bounded_completed_best"].astype(bool)] = np.nan
    if "bounded_completed_worst" in merged:
        worst.loc[~merged["bounded_completed_worst"].astype(bool)] = np.nan
    delta = (worst - best).replace([np.inf, -np.inf], np.nan).dropna()
    if delta.empty:
        return None

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bins = min(20, max(6, int(np.sqrt(len(delta)))))
    ax.hist(delta.to_numpy(float), bins=bins)
    q10 = float(delta.quantile(0.10))
    q50 = float(delta.median())
    q90 = float(delta.quantile(0.90))
    for value, label in ((q10, "p10"), (q50, "median"), (q90, "p90")):
        ax.axvline(value, linestyle="--", linewidth=1)
        ax.text(value, ax.get_ylim()[1] * 0.94, label, rotation=90, va="top", ha="right")
    ax.set_xlabel("Worst-design lap time minus best-design lap time [s]")
    ax.set_ylabel("Paired worlds")
    ax.set_title("Best-to-worst spread across the same worlds")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = plots / "design_best_vs_worst_delta.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    caption = (
        f"World-by-world gap between the current best candidate and the current worst candidate. "
        f"Positive values mean the best design is faster; the median gap is {q50:.2f} s "
        f"(p10 {q10:.2f} s, p90 {q90:.2f} s)."
    )
    return path, caption


def _write_one_axis_slices(
    summary: pd.DataFrame,
    paths: tuple[str, str],
    plots: Path,
) -> list[tuple[Path, str]]:
    x_path, y_path = paths
    generated: list[tuple[Path, str]] = []
    metric = "paired_regret_median_s"
    if metric not in summary:
        return generated

    generated.append(
        _write_slice_plot(
            summary=summary,
            x_path=x_path,
            group_path=y_path,
            metric=metric,
            filename="design_slice_primary_axis.png",
            title=f"One variable at a time — {_axis_label(x_path)}",
            xlabel=_axis_label(x_path),
            group_label=_axis_label(y_path),
            plots=plots,
        )
    )
    generated.append(
        _write_slice_plot(
            summary=summary,
            x_path=y_path,
            group_path=x_path,
            metric=metric,
            filename="design_slice_secondary_axis.png",
            title=f"One variable at a time — {_axis_label(y_path)}",
            xlabel=_axis_label(y_path),
            group_label=_axis_label(x_path),
            plots=plots,
        )
    )
    return [item for item in generated if item is not None]


def _write_slice_plot(
    *,
    summary: pd.DataFrame,
    x_path: str,
    group_path: str,
    metric: str,
    filename: str,
    title: str,
    xlabel: str,
    group_label: str,
    plots: Path,
) -> tuple[Path, str] | None:
    if x_path not in summary or group_path not in summary or metric not in summary:
        return None
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    plotted = False
    for group_value, group in summary.groupby(group_path, sort=True):
        ordered = group.sort_values(x_path)
        x = pd.to_numeric(ordered[x_path], errors="coerce")
        y = pd.to_numeric(ordered[metric], errors="coerce")
        mask = x.notna() & y.notna()
        if not mask.any():
            continue
        plotted = True
        ax.plot(x[mask].to_numpy(float), y[mask].to_numpy(float), marker="o", label=f"{group_label} = {group_value:g}")
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median paired regret [s]")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Held fixed", loc="best")
    fig.tight_layout()
    path = plots / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    caption = (
        f"Median paired regret when {xlabel.lower()} changes along the x-axis and "
        f"{group_label.lower()} is held fixed line-by-line. This is the clearest one-variable-at-a-time view."
    )
    return path, caption


def _write_plateau_view(
    summary: pd.DataFrame,
    paths: tuple[str, str],
    plots: Path,
) -> tuple[Path, str] | None:
    x_path, y_path = paths
    metric = "paired_regret_median_s"
    if metric not in summary:
        return None
    matrix = summary.pivot_table(
        index=y_path,
        columns=x_path,
        values=metric,
        aggfunc="median",
    ).sort_index().sort_index(axis=1)
    if matrix.empty:
        return None
    data = matrix.to_numpy(float)
    categories = np.full_like(data, np.nan)
    thresholds = (0.03, 0.10, 0.50)
    finite = np.isfinite(data)
    categories[finite] = np.select(
        [data[finite] <= thresholds[0], data[finite] <= thresholds[1], data[finite] <= thresholds[2]],
        [0, 1, 2],
        default=3,
    )

    cmap = ListedColormap(["#2a9d8f", "#8ec07c", "#f4a261", "#e76f51"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(
        figsize=(max(8.5, 1.2 * len(matrix.columns)), max(5.2, 0.8 * len(matrix.index)))
    )
    image = ax.imshow(categories, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(matrix.columns)), [f"{value:g}" for value in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)), [f"{value:g}" for value in matrix.index])
    ax.set_xlabel(_axis_label(x_path))
    ax.set_ylabel(_axis_label(y_path))
    ax.set_title("Decision plateau map from median paired regret")
    for row_index in range(data.shape[0]):
        for column_index in range(data.shape[1]):
            value = data[row_index, column_index]
            if not np.isfinite(value):
                continue
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )
    colourbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3])
    colourbar.ax.set_yticklabels([
        f"≤ {thresholds[0]:.2f} s",
        f"≤ {thresholds[1]:.2f} s",
        f"≤ {thresholds[2]:.2f} s",
        f"> {thresholds[2]:.2f} s",
    ])
    colourbar.set_label("Median paired regret class")
    fig.tight_layout()
    path = plots / "design_plateau_map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    caption = (
        "A simple plateau classification for the same regret numbers: cells within 0.03 s of the world-by-world best behave like a practical tied optimum, "
        "while the orange and red cells are materially worse."
    )
    return path, caption


def _speed_sanity_section(
    output: Path,
    rows: pd.DataFrame,
    ranking: pd.DataFrame,
    plots: Path,
) -> str:
    try:
        project_root = _resolve_project_root(output)
        if project_root is None:
            return ""
        sources = _load_speed_sources(project_root)
        if not sources:
            return ""
        case_views = _build_speed_sanity_views(output, project_root, rows, ranking, plots, sources)
        if not case_views:
            return ""
    except Exception as exc:
        return (
            '<h3>Speed-distribution sanity check</h3>'
            '<div class="card warning"><strong>Sanity check skipped.</strong> '
            + html.escape(str(exc))
            + '</div>'
        )

    options = []
    views_html = []
    for index, view in enumerate(case_views):
        key = f"speed-sanity-{index}"
        options.append(f'<option value="{key}">{html.escape(view["label"])}</option>')
        hidden = "" if index == 0 else ' hidden="hidden"'
        views_html.append(
            f'<div data-speed-sanity-view="{key}"{hidden}>'
            + _inline_figure(view["path"], view["caption"])
            + f'<p class="caption">{html.escape(view["note"])}</p>'
            + '</div>'
        )

    return (
        '<h3>Speed-distribution sanity check</h3>'
        '<div class="section-intro"><strong>What this section shows.</strong> '
        'This is only a sanity check. For one strong candidate, a few representative traffic worlds are rerun and the resulting time-weighted speed histogram is compared against the raw McMaster and Cornell source traces. '
        'It is not a validation proof; it is just a quick check that the simulated speed occupancy does not look wildly alien relative to the evidence.</div>'
        '<div class="design-order-control">'
        '<label for="speed-sanity-select">Representative case:</label>'
        '<select id="speed-sanity-select" data-speed-sanity-select>'
        + ''.join(options)
        + '</select>'
        '<div class="order-help">This selector is here so the section can grow later; right now it shows one representative median-traffic world on the nominal track reconstruction.</div>'
        '</div>'
        + ''.join(views_html)
        + """
<script>
(function () {
  const select = document.querySelector('[data-speed-sanity-select]');
  if (!select) return;
  function apply() {
    const value = select.value;
    document.querySelectorAll('[data-speed-sanity-view]').forEach((node) => {
      node.hidden = node.getAttribute('data-speed-sanity-view') !== value;
    });
  }
  select.addEventListener('change', apply);
  apply();
})();
</script>
"""
    )


def _build_speed_sanity_views(
    output: Path,
    project_root: Path,
    rows: pd.DataFrame,
    ranking: pd.DataFrame,
    plots: Path,
    sources: Mapping[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    if ranking.empty:
        return []
    top_design = str(ranking.iloc[0]["design_id"])
    subset = rows[rows["design_id"] == top_design].copy()
    if subset.empty:
        return []
    if "track_case_id" in subset.columns and (subset["track_case_id"].astype(str) == "nominal").any():
        subset = subset[subset["track_case_id"].astype(str) == "nominal"].copy()
    if subset.empty:
        return []

    representative_rows = _pick_representative_rows(subset)
    if not representative_rows:
        return []

    scenarios = _load_scenarios(output)
    if not scenarios:
        return []

    design_values_si = _design_values_si_for_design(subset)
    if not design_values_si:
        return []

    study_name, vehicle_id, bundle, study_raw, vehicle_raw, track_raw = _load_project_context(output)
    if not isinstance(study_raw, Mapping) or not isinstance(vehicle_raw, Mapping) or not isinstance(track_raw, Mapping):
        return []

    views: list[dict[str, Any]] = []
    for label, row in representative_rows:
        replicate = int(row["replicate"])
        scenario = scenarios.get(replicate)
        if scenario is None:
            continue
        trace = _simulate_bounded_case(
            scenario=scenario,
            design_values_si=design_values_si,
            vehicle_id=vehicle_id,
            vehicle_raw=vehicle_raw,
            study_raw=study_raw,
            track_raw=track_raw,
            bundle=bundle,
        )
        path = plots / f"speed_sanity_{_slug(label)}.png"
        caption, note = _plot_speed_histogram(path, trace, sources, row, top_design, label)
        views.append({"label": label, "path": path, "caption": caption, "note": note})
    return views


def _pick_representative_rows(subset: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    rows = subset.copy()
    rows["reference_traffic_penalty_s"] = pd.to_numeric(
        rows.get("reference_traffic_penalty_s"), errors="coerce"
    )
    rows["bounded_lap_time_s"] = pd.to_numeric(rows.get("bounded_lap_time_s"), errors="coerce")
    metric = rows["reference_traffic_penalty_s"]
    if metric.isna().all():
        ordered = rows.sort_values("bounded_lap_time_s").reset_index(drop=True)
        if ordered.empty:
            return []
        return [("representative", ordered.iloc[len(ordered) // 2])]

    ordered = rows.sort_values("reference_traffic_penalty_s").reset_index(drop=True)
    target = float(metric.quantile(0.50))
    distances = (ordered["reference_traffic_penalty_s"] - target).abs()
    index = int(distances.idxmin())
    return [("median traffic", ordered.loc[index])]


def _load_project_context(output: Path):
    from cvt_track_study.bundle import load_track_bundle

    manifest = _read_json(output / "run_manifest.json")
    provenance = _read_json(output / "provenance.json")
    study_name = str(manifest.get("study_name") or provenance.get("study_name") or "baseline")
    resolved_path = output / "resolved_inputs" / "resolved_inputs.toml"
    if not resolved_path.is_file():
        resolved_path = output / "resolved_inputs.toml"
    data = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
    studies = data.get("studies", {})
    current_study_raw = studies.get(study_name, {}) if isinstance(studies, Mapping) else {}
    base_study_name = str(current_study_raw.get("base_case", {}).get("study") or study_name)
    study_raw = studies.get(base_study_name, current_study_raw) if isinstance(studies, Mapping) else current_study_raw
    vehicle_id = str(
        manifest.get("vehicle_id")
        or current_study_raw.get("study", {}).get("vehicle_id")
        or study_raw.get("study", {}).get("vehicle_id")
        or ""
    )
    vehicles = data.get("vehicles", {})
    vehicle_raw = vehicles.get(vehicle_id, {}) if isinstance(vehicles, Mapping) else {}
    track_raw = data.get("track", {}) if isinstance(data.get("track", {}), Mapping) else {}
    bundle = load_track_bundle(output / "track_bundle.json")
    return study_name, vehicle_id, bundle, study_raw, vehicle_raw, track_raw


def _simulate_bounded_case(
    *,
    scenario: Any,
    design_values_si: Mapping[str, float],
    vehicle_id: str,
    vehicle_raw: Mapping[str, Any],
    study_raw: Mapping[str, Any],
    track_raw: Mapping[str, Any],
    bundle: Any,
):
    from cvt_track_study.simulation.integrator import run_simulation
    from cvt_track_study.simulation.service import resolve_simulation_cases
    from cvt_track_study.simulation.traffic import (
        TrafficReferenceProfile,
        traffic_realization_from_mapping,
    )

    bounded_case, reference_case, settings, runtime_track = resolve_simulation_cases(
        vehicle_id=vehicle_id,
        vehicle_raw=vehicle_raw,
        study_raw=study_raw,
        track_raw=track_raw,
        bundle=bundle,
        quantity_values_si=scenario.quantity_values_si,
        choice_values=scenario.choice_values,
        gate_target_speeds_mps=scenario.gate_target_speeds_mps,
        design_values_si=design_values_si,
        shared_reference=True,
    )
    traffic = traffic_realization_from_mapping(scenario.traffic_realization)
    reference_free = run_simulation(case=reference_case, track=runtime_track, settings=settings)
    traffic_reference = TrafficReferenceProfile.from_trace(reference_free, spacing_m=1.0)
    return run_simulation(
        case=bounded_case,
        track=runtime_track,
        settings=settings,
        traffic=traffic,
        traffic_reference=(traffic_reference if traffic is not None else None),
    )


def _plot_speed_histogram(
    path: Path,
    trace: Any,
    sources: Mapping[str, pd.DataFrame],
    row: pd.Series,
    design_id: str,
    label: str,
) -> tuple[str, str]:
    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    speed_limit_kmh = 90.0
    sim_speed = np.asarray(trace.numeric["vehicle_speed_kmh"], dtype=float)
    sim_weight = _weights_from_time(np.asarray(trace.numeric["time_s"], dtype=float))
    sim_mask = np.isfinite(sim_speed) & np.isfinite(sim_weight) & (sim_speed >= 0.0) & (sim_speed <= speed_limit_kmh)
    sim_speed = sim_speed[sim_mask]
    sim_weight = sim_weight[sim_mask]
    maximum = float(np.nanmax(sim_speed)) if len(sim_speed) else 0.0
    for frame in sources.values():
        if not frame.empty:
            clipped = frame[(frame["speed_kmh"] >= 0.0) & (frame["speed_kmh"] <= speed_limit_kmh)]
            if not clipped.empty:
                maximum = max(maximum, float(np.nanmax(clipped["speed_kmh"].to_numpy(float))))
    vmax = max(20.0, math.ceil(maximum / 5.0) * 5.0)
    bins = np.linspace(0.0, vmax, 24)

    ax.hist(
        sim_speed,
        bins=bins,
        weights=sim_weight / max(sim_weight.sum(), 1.0),
        alpha=0.45,
        label="Simulated bounded case",
    )
    for source_label, frame in sources.items():
        clipped = frame[(frame["speed_kmh"] >= 0.0) & (frame["speed_kmh"] <= speed_limit_kmh)]
        speed = clipped["speed_kmh"].to_numpy(float)
        weight = clipped["weight_s"].to_numpy(float)
        if len(speed) == 0 or np.nansum(weight) <= 0.0:
            continue
        ax.hist(
            speed,
            bins=bins,
            weights=weight / np.nansum(weight),
            histtype="step",
            linewidth=2,
            label=source_label,
        )
    ax.set_xlabel("Vehicle speed [km/h]")
    ax.set_ylabel("Fraction of observed time")
    ax.set_title(f"Speed occupancy sanity check — {label}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    traffic_penalty = pd.to_numeric(pd.Series([row.get("reference_traffic_penalty_s")]), errors="coerce").iloc[0]
    bounded_lap = pd.to_numeric(pd.Series([row.get("bounded_lap_time_s")]), errors="coerce").iloc[0]
    caption = (
        f"Time-weighted speed histogram for the top-ranked design ({design_id}) in the {label} representative world, "
        f"overlaid with the raw McMaster and Cornell source traces."
    )
    note = (
        f"Representative world details: traffic penalty ≈ {traffic_penalty:.2f} s, bounded lap time ≈ {bounded_lap:.2f} s."
    )
    return caption, note


def _weights_from_time(time_s: np.ndarray) -> np.ndarray:
    if time_s.size == 0:
        return np.array([], dtype=float)
    delta = np.diff(time_s)
    if delta.size == 0:
        return np.ones_like(time_s, dtype=float)
    positive = delta[delta > 0.0]
    tail = float(np.median(positive)) if positive.size else 1.0
    weights = np.diff(time_s, append=time_s[-1] + tail)
    weights = np.where(weights > 0.0, weights, tail)
    return weights.astype(float)


def _resolve_project_root(output: Path) -> Path | None:
    env = os.environ.get("CVT_TRACK_STUDY_PROJECT_ROOT")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    provenance = _read_json(output / "provenance.json")
    project = provenance.get("project")
    if isinstance(project, str) and project:
        candidates.append(Path(project))
        normalized = Path(project.replace("\\", "/"))
        candidates.append(normalized)
        stem = normalized.name or Path(project).name
        if stem:
            candidates.append(Path("/mnt/data/projects_unz") / stem)
    for root in candidates:
        if (root / "track" / "track.toml").is_file() and (root / "studies").is_dir():
            return root.resolve()
    return None


def _load_speed_sources(project_root: Path) -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    ingestion_dir = project_root / "results" / "ingestion"
    canonical_files = sorted(ingestion_dir.glob("*/canonical_points.csv"))
    if canonical_files:
        frame = _read_csv(canonical_files[-1])
        if not frame.empty:
            mcmaster = frame[frame.get("vehicle_id", pd.Series(dtype=str)).astype(str) == "mcmaster"].copy()
            mcmaster_profile = _speed_profile_from_canonical(mcmaster)
            if not mcmaster_profile.empty:
                sources["McMaster raw telemetry"] = mcmaster_profile
    cornell_csv = project_root / "track" / "csv" / "CORNELL.csv"
    if cornell_csv.is_file():
        cornell = _speed_profile_from_simple_csv(_read_csv(cornell_csv), speed_col="speed_kmh", timestamp_col="timestamp")
        if not cornell.empty:
            sources["Cornell raw telemetry"] = cornell
    return sources


def _speed_profile_from_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["speed_kmh", "weight_s"])
    speed_col = _first_existing(
        frame,
        ("analysis_speed_mps", "reported_speed_mps", "device_speed_mps", "derived_speed_mps"),
    )
    if speed_col is None:
        return pd.DataFrame(columns=["speed_kmh", "weight_s"])
    speed = pd.to_numeric(frame[speed_col], errors="coerce") * 3.6
    weight = pd.to_numeric(frame.get("time_step_s"), errors="coerce")
    if "timestamp_utc" in frame:
        timestamp = pd.to_datetime(frame["timestamp_utc"], errors="coerce")
        dt = timestamp.shift(-1).sub(timestamp).dt.total_seconds()
        weight = weight.where(weight.notna() & (weight > 0.0), dt)
    positive = weight[weight > 0.0]
    fill = float(positive.median()) if len(positive) else 1.0
    weight = weight.where(weight > 0.0, fill).fillna(fill)
    result = pd.DataFrame({"speed_kmh": speed, "weight_s": weight})
    return result.replace([np.inf, -np.inf], np.nan).dropna()


def _speed_profile_from_simple_csv(frame: pd.DataFrame, *, speed_col: str, timestamp_col: str) -> pd.DataFrame:
    if frame.empty or speed_col not in frame:
        return pd.DataFrame(columns=["speed_kmh", "weight_s"])
    speed = pd.to_numeric(frame[speed_col], errors="coerce")
    if timestamp_col in frame:
        timestamp = pd.to_datetime(frame[timestamp_col], errors="coerce")
        dt = timestamp.shift(-1).sub(timestamp).dt.total_seconds()
        positive = dt[dt > 0.0]
        fill = float(positive.median()) if len(positive) else 1.0
        weight = dt.where(dt > 0.0, fill).fillna(fill)
    else:
        weight = pd.Series(1.0, index=frame.index)
    result = pd.DataFrame({"speed_kmh": speed, "weight_s": weight})
    return result.replace([np.inf, -np.inf], np.nan).dropna()


def _load_scenarios(output: Path) -> dict[int, Any]:
    from cvt_track_study.uncertainty.model import ScenarioDraw

    path = output / "scenario_draws.jsonl"
    if not path.is_file():
        return {}
    scenarios: dict[int, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            replicate = int(payload["replicate"])
            scenarios[replicate] = ScenarioDraw(
                replicate=replicate,
                seed=int(payload["seed"]),
                sampling_mode=str(payload.get("sampling_mode", "uncertainty")),
                quantity_values_si=dict(payload.get("quantity_values_si", {})),
                choice_values=dict(payload.get("choice_values", {})),
                gate_target_speeds_mps=dict(payload.get("gate_target_speeds_mps", {})),
                independently_sampled_gate_ids=tuple(payload.get("independently_sampled_gate_ids", ())),
                sampling_design=str(payload.get("sampling_design", "latin_hypercube_rank_correlated")),
                traffic_realization=payload.get("traffic_realization"),
            )
    return scenarios


def _design_values_si_for_design(rows: pd.DataFrame) -> dict[str, float]:
    if rows.empty or "design_values_si_json" not in rows:
        return {}
    try:
        payload = json.loads(str(rows["design_values_si_json"].iloc[0]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    values: dict[str, float] = {}
    for key, value in payload.items():
        try:
            values[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _grid_section(
    *,
    grid_figures: list[tuple[Path, str]],
    extra_figures: list[tuple[Path, str]],
    speed_sanity_html: str,
    paths: tuple[str, str],
    ranking: pd.DataFrame,
) -> str:
    best = ranking.iloc[0] if not ranking.empty else None
    worst = ranking.iloc[-1] if not ranking.empty else None
    spread = ""
    if best is not None and worst is not None:
        spread = (
            '<div class="finding-grid">'
            + _finding(
                "Current best median regret",
                f"{float(best['paired_regret_median_s']):.3f} s at {html.escape(str(best['design_id']))}",
            )
            + _finding(
                "Current worst median regret",
                f"{float(worst['paired_regret_median_s']):.3f} s at {html.escape(str(worst['design_id']))}",
            )
            + _finding(
                "What to look for",
                "The heatmaps show the whole surface, the one-axis plots show whether either variable moves the answer monotonically, and the plateau map shows where several cells are effectively tied.",
            )
            + '</div>'
        )
    grid_html = ''.join(_inline_figure(path, caption) for path, caption in grid_figures)
    extra_html = ''.join(_inline_figure(path, caption) for path, caption in extra_figures)
    return (
        '<h2>Two-dimensional design grid</h2>'
        '<div class="section-intro"><strong>What this section shows.</strong> '
        'Every cell is one simultaneous combination of '
        f'<code>{html.escape(paths[0])}</code> and <code>{html.escape(paths[1])}</code>, '
        'evaluated on the same paired uncertainty-informed worlds. Win fractions describe this reduced screening set and are not calibrated probabilities.</div>'
        + spread
        + grid_html
        + '<h3>Interpreting worst-to-best spread and one-variable slices</h3>'
        + extra_html
        + speed_sanity_html
    )


def _finding(title: str, text: str) -> str:
    return (
        '<div class="finding"><strong>'
        + html.escape(title)
        + '</strong>'
        + html.escape(text)
        + '</div>'
    )


def _inline_figure(path: Path, caption: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        '<figure class="figure">'
        f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def _axis_label(path: str) -> str:
    return {
        "drivetrain.final_drive_ratio": "Final-drive ratio",
        "drivetrain.cvt.maximum_reduction_ratio": "Maximum CVT reduction",
        "drivetrain.cvt.minimum_reduction_ratio": "Minimum CVT reduction",
    }.get(path, path)


def _slug(text: str) -> str:
    token = ''.join(char.lower() if char.isalnum() else '-' for char in text)
    return '-'.join(part for part in token.split('-') if part) or 'case'


def _first_existing(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}
