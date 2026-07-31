"""Post-process a paired design result into two-dimensional grid heatmaps."""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    generated = _write_heatmaps(summary, paths, plots)
    if not generated:
        return target

    document = target.read_text(encoding="utf-8")
    section = _grid_section(generated, paths)
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
            "maximum_ratio_time_median_s",
            "design_grid_max_ratio_time.png",
            "Median time at maximum CVT reduction",
            "Time [s]",
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


def _grid_section(
    generated: list[tuple[Path, str]],
    paths: tuple[str, str],
) -> str:
    figures = "".join(
        _inline_figure(path, caption) for path, caption in generated
    )
    return (
        "<h2>Two-dimensional design grid</h2>"
        '<p class="subtitle">Every cell is one simultaneous combination of '
        f"<code>{html.escape(paths[0])}</code> and "
        f"<code>{html.escape(paths[1])}</code>, evaluated on the same paired "
        "uncertainty-informed worlds. Win fractions describe this selected "
        "screening set and are not calibrated probabilities.</p>"
        + figures
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
