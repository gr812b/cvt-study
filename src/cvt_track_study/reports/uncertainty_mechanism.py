"""Mechanism-first additions to the canonical full-uncertainty report."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .html import dataframe_table, figure


LOSS_COLUMNS = (
    "bounded_drivetrain_loss_energy_kj",
    "bounded_clutch_loss_energy_kj",
    "bounded_tire_slip_loss_energy_kj",
    "bounded_brake_loss_energy_kj",
    "bounded_rolling_loss_energy_kj",
    "bounded_aerodynamic_loss_energy_kj",
    "bounded_obstacle_loss_energy_kj",
)
_MARKER_START = "<!-- cvt-mechanism-overlay:start -->"
_MARKER_END = "<!-- cvt-mechanism-overlay:end -->"


def enhance_full_uncertainty_report(output: Path, target: Path) -> Path:
    output = output.resolve()
    target = target.resolve()
    rows = pd.read_csv(output / "replicate_results.csv")
    nominal = _nominal_row(output)
    summary = _mechanism_summary(rows, nominal)
    summary.to_csv(output / "full_uncertainty_mechanism_summary.csv", index=False)

    plots = output / "report_plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_path = plots / "physical_losses_opportunity_and_time.png"
    _mechanism_plot(rows, nominal, plot_path)

    nominal_text = (
        "The diamond is the exact nominal bounded-versus-infinite comparison "
        "saved by the runner; it is not estimated from the uncertainty draws."
        if nominal
        else "No exact nominal reference artifact was available for this legacy run."
    )
    snippet = (
        _MARKER_START
        + '<h2 id="physical-mechanism-uncertainty">Physical mechanism uncertainty</h2>'
        + '<div class="section-intro"><strong>What this section shows.</strong> '
        + "Physical losses are shown with p10–p90 error bars. Finite-ratio opportunity "
        + "loss and lap-time penalty are kept separate because they are counterfactual "
        + "limitations, not dissipative losses. "
        + html.escape(nominal_text)
        + "</div>"
        + figure(
            plot_path,
            "Top: physical loss medians and p10–p90 ranges. Middle: finite-ratio "
            "opportunity loss with the high-minus-low span. Bottom: finite-ratio "
            "lap-time penalty with the same span. Diamonds mark the exact nominal case.",
        )
        + dataframe_table(
            summary,
            searchable=False,
            max_rows=100,
            table_id="mechanism-uncertainty-summary",
        )
        + _MARKER_END
    )

    document = target.read_text(encoding="utf-8")
    if _MARKER_START in document and _MARKER_END in document:
        before = document.split(_MARKER_START, 1)[0]
        after = document.split(_MARKER_END, 1)[1]
        document = before + after
    insertion = document.find("<h2")
    if insertion < 0:
        insertion = document.lower().find("</main>")
    if insertion < 0:
        insertion = document.lower().find("</body>")
    if insertion < 0:
        insertion = len(document)
    document = document[:insertion] + snippet + document[insertion:]
    target.write_text(document, encoding="utf-8")
    return target


def _nominal_row(output: Path) -> dict[str, Any]:
    path = output / "nominal_reference.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return dict(raw[0])
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _mechanism_summary(rows: pd.DataFrame, nominal: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for column in LOSS_COLUMNS:
        if column not in rows:
            continue
        records.append(_summary_record(column, rows[column], nominal.get(column)))
    for column in (
        "finite_ratio_opportunity_loss_energy_kj",
        "lap_time_penalty_vs_infinite_s",
    ):
        if column in rows:
            records.append(_summary_record(column, rows[column], nominal.get(column)))
    return pd.DataFrame(records)


def _summary_record(name: str, values: pd.Series, nominal: Any) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    p10 = float(numeric.quantile(0.10)) if len(numeric) else np.nan
    median = float(numeric.median()) if len(numeric) else np.nan
    p90 = float(numeric.quantile(0.90)) if len(numeric) else np.nan
    try:
        nominal_value = float(nominal)
    except (TypeError, ValueError):
        nominal_value = np.nan
    return {
        "metric": name,
        "sample_count": int(len(numeric)),
        "p10": p10,
        "median": median,
        "p90": p90,
        "high_minus_low": p90 - p10 if np.isfinite(p10) and np.isfinite(p90) else np.nan,
        "exact_nominal": nominal_value,
    }


def _mechanism_plot(
    rows: pd.DataFrame,
    nominal: dict[str, Any],
    path: Path,
) -> None:
    figure_obj, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)

    loss_records = []
    for column in LOSS_COLUMNS:
        if column not in rows:
            continue
        record = _summary_record(column, rows[column], nominal.get(column))
        if record["sample_count"]:
            loss_records.append(record)
    if loss_records:
        x = np.arange(len(loss_records))
        med = np.asarray([row["median"] for row in loss_records], dtype=float)
        low = np.asarray([row["p10"] for row in loss_records], dtype=float)
        high = np.asarray([row["p90"] for row in loss_records], dtype=float)
        axes[0].errorbar(
            x,
            med,
            yerr=np.vstack((med - low, high - med)),
            fmt="o",
            capsize=4,
            label="uncertainty median and p10–p90",
        )
        nominal_values = np.asarray(
            [row["exact_nominal"] for row in loss_records], dtype=float
        )
        finite = np.isfinite(nominal_values)
        if finite.any():
            axes[0].scatter(
                x[finite], nominal_values[finite], marker="D", label="exact nominal"
            )
        axes[0].set_xticks(
            x,
            [
                row["metric"]
                .replace("bounded_", "")
                .replace("_loss_energy_kj", "")
                .replace("_", " ")
                for row in loss_records
            ],
            rotation=25,
            ha="right",
        )
        axes[0].legend()
    axes[0].set_ylabel("Energy [kJ]")
    axes[0].set_title("Physical losses")
    axes[0].grid(True, axis="y", alpha=0.25)

    _interval_panel(
        axes[1],
        rows,
        nominal,
        "finite_ratio_opportunity_loss_energy_kj",
        "Finite-ratio opportunity loss",
        "Energy [kJ]",
    )
    _interval_panel(
        axes[2],
        rows,
        nominal,
        "lap_time_penalty_vs_infinite_s",
        "Finite-ratio lap-time penalty",
        "Time [s]",
    )
    figure_obj.savefig(path, dpi=180)
    plt.close(figure_obj)


def _interval_panel(
    axis: Any,
    rows: pd.DataFrame,
    nominal: dict[str, Any],
    column: str,
    title: str,
    ylabel: str,
) -> None:
    if column not in rows:
        axis.text(0.5, 0.5, "metric unavailable", ha="center", va="center")
        axis.set_title(title)
        return
    record = _summary_record(column, rows[column], nominal.get(column))
    p10, median, p90 = record["p10"], record["median"], record["p90"]
    axis.errorbar(
        [0.0],
        [median],
        yerr=[[median - p10], [p90 - median]],
        fmt="o",
        capsize=6,
        label="median and p10–p90",
    )
    if np.isfinite(record["exact_nominal"]):
        axis.scatter([0.12], [record["exact_nominal"]], marker="D", label="exact nominal")
    axis.annotate(
        f"p90 − p10 = {record['high_minus_low']:.3g}",
        xy=(0.0, p90),
        xytext=(0.18, p90),
        arrowprops={"arrowstyle": "-"},
        va="center",
    )
    axis.set_xlim(-0.35, 0.75)
    axis.set_xticks([])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
