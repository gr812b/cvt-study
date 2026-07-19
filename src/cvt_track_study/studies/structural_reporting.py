"""Complete HTML review package for structural sensitivity."""

from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .structural_analysis import (
    HEADLINE_METRICS,
    METRIC_DEFINITIONS,
    metric_range_rows,
    parameter_level_rows,
)


def write_structural_outputs(
    *,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    input_contracts: Mapping[str, Any],
) -> None:
    range_rows = metric_range_rows(
        summary
    )
    level_rows = parameter_level_rows(
        rows
    )
    _write_rows(
        output
        / "structural_metric_ranges.csv",
        range_rows,
    )
    _write_rows(
        output
        / "structural_parameter_levels.csv",
        level_rows,
    )

    plot_paths: list[Path] = []
    for metric in HEADLINE_METRICS:
        if metric not in summary.get(
            "rankings", {}
        ):
            continue
        path = output / (
            "structural_"
            + metric.replace(
                "bounded_", ""
            )
            .replace(
                "_energy_kj", ""
            )
            .replace("_s", "")
            .replace("_kmh", "")
            + "_tornado.png"
        )
        _tornado_plot(
            path,
            summary,
            metric,
            top_count=int(
                manifest.get(
                    "structural_report_top_parameter_count",
                    15,
                )
            ),
        )
        if path.is_file():
            plot_paths.append(path)

    heatmap_path = (
        output
        / "structural_loss_mechanism_heatmap.png"
    )
    _mechanism_heatmap(
        heatmap_path,
        summary,
        top_count=int(
            manifest.get(
                "structural_report_top_parameter_count",
                15,
            )
        ),
    )
    if heatmap_path.is_file():
        plot_paths.append(heatmap_path)

    html_path = (
        output
        / "structural_sensitivity_report.html"
    )
    _write_html(
        html_path,
        summary=summary,
        manifest=manifest,
        input_contracts=input_contracts,
        range_rows=range_rows,
        level_rows=level_rows,
        plot_paths=plot_paths,
    )
    _append_report_links(output)


def _tornado_plot(
    path: Path,
    summary: Mapping[str, Any],
    metric: str,
    *,
    top_count: int,
) -> None:
    ranking = list(
        summary.get("rankings", {}).get(
            metric, ()
        )
    )[:top_count]
    if not ranking:
        return

    ranking = list(reversed(ranking))
    labels = [
        str(item["path"])
        for item in ranking
    ]
    low = np.asarray(
        [
            float(
                item[
                    "minimum_change_from_nominal"
                ]
            )
            for item in ranking
        ]
    )
    high = np.asarray(
        [
            float(
                item[
                    "maximum_change_from_nominal"
                ]
            )
            for item in ranking
        ]
    )
    y = np.arange(len(ranking))
    figure, axis = plt.subplots(
        figsize=(
            12,
            max(
                5.5,
                0.46 * len(ranking)
                + 1.5,
            ),
        )
    )
    axis.barh(
        y,
        high - low,
        left=low,
        height=0.62,
    )
    axis.scatter(
        np.zeros(len(ranking)),
        y,
        marker="|",
        s=90,
        zorder=4,
    )
    axis.axvline(
        0.0, linewidth=1.1
    )
    axis.set_yticks(y, labels)
    definition = METRIC_DEFINITIONS[
        metric
    ]
    axis.set_xlabel(
        f"Change from nominal [{definition[1]}]"
    )
    axis.set_title(
        f"One-at-a-time structural response: {definition[0]}\n"
        "bar endpoints are the minimum and maximum declared levels"
    )
    axis.grid(
        True, axis="x", alpha=0.25
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _mechanism_heatmap(
    path: Path,
    summary: Mapping[str, Any],
    *,
    top_count: int,
) -> None:
    metrics = [
        metric
        for metric in (
            "bounded_lap_time_s",
            "bounded_maximum_speed_kmh",
            "bounded_engine_energy_kj",
            "bounded_drivetrain_loss_energy_kj",
            "bounded_clutch_loss_energy_kj",
            "bounded_tire_slip_loss_energy_kj",
            "bounded_brake_loss_energy_kj",
            "bounded_rolling_loss_energy_kj",
            "bounded_aerodynamic_loss_energy_kj",
            "bounded_obstacle_loss_energy_kj",
            "bounded_time_maximum_ratio_s",
            "bounded_time_variable_ratio_s",
            "bounded_time_minimum_ratio_s",
        )
        if metric
        in summary.get("rankings", {})
    ]
    if not metrics:
        return

    path_scores: dict[str, float] = {}
    for metric in metrics:
        for item in summary[
            "rankings"
        ][metric]:
            path_scores[item["path"]] = max(
                path_scores.get(
                    item["path"], 0.0
                ),
                float(
                    item[
                        "relative_screening_importance"
                    ]
                ),
            )
    paths = [
        item[0]
        for item in sorted(
            path_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_count]
    ]
    if not paths:
        return

    lookup = {
        metric: {
            item["path"]: float(
                item[
                    "relative_screening_importance"
                ]
            )
            for item in summary[
                "rankings"
            ][metric]
        }
        for metric in metrics
    }
    matrix = np.asarray(
        [
            [
                lookup[metric].get(
                    parameter, 0.0
                )
                for metric in metrics
            ]
            for parameter in paths
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(
        figsize=(
            max(
                11,
                0.75 * len(metrics)
                + 5,
            ),
            max(
                6,
                0.48 * len(paths)
                + 2,
            ),
        )
    )
    image = axis.imshow(
        matrix,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_xticks(
        np.arange(len(metrics)),
        [
            METRIC_DEFINITIONS[
                metric
            ][0]
            for metric in metrics
        ],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(paths)),
        paths,
    )
    axis.set_title(
        "Relative one-at-a-time structural influence by output mechanism"
    )
    figure.colorbar(
        image,
        ax=axis,
        label=(
            "Influence relative to the strongest "
            "parameter for each metric"
        ),
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_html(
    path: Path,
    *,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    input_contracts: Mapping[str, Any],
    range_rows: Sequence[Mapping[str, Any]],
    level_rows: Sequence[Mapping[str, Any]],
    plot_paths: Sequence[Path],
) -> None:
    quality = summary.get(
        "numerical_quality", {}
    )
    plot_html = "\n".join(
        (
            "<section><h3>"
            + html.escape(
                _plot_title(plot.name)
            )
            + "</h3><img alt=\""
            + html.escape(
                _plot_title(plot.name)
            )
            + "\" src=\"data:image/png;base64,"
            + base64.b64encode(
                plot.read_bytes()
            ).decode("ascii")
            + "\"></section>"
        )
        for plot in plot_paths
    )

    top_tables = []
    for metric in HEADLINE_METRICS:
        ranking = summary.get(
            "rankings", {}
        ).get(metric, [])
        if not ranking:
            continue
        frame = pd.DataFrame(
            ranking[:15]
        )[
            [
                "rank",
                "path",
                "category",
                "nominal",
                "minimum_change_from_nominal",
                "maximum_change_from_nominal",
                "maximum_abs_change_from_nominal",
                "maximum_abs_percent_change_from_nominal",
                "unit",
            ]
        ]
        top_tables.append(
            "<h3>"
            + html.escape(
                METRIC_DEFINITIONS[
                    metric
                ][0]
            )
            + "</h3>"
            + frame.to_html(
                index=False,
                escape=True,
                float_format=lambda value: (
                    f"{value:.5g}"
                ),
            )
        )

    inventory_rows = []
    selected = set(
        manifest.get(
            "structural_parameter_paths",
            (),
        )
    )
    for parameter in sorted(selected):
        contract = input_contracts.get(
            parameter, {}
        )
        raw_contract = contract.get(
            "contract", {}
        )
        uncertainty = raw_contract.get(
            "uncertainty", {}
        )
        source = raw_contract.get(
            "source", {}
        )
        inventory_rows.append(
            {
                "parameter_path": parameter,
                "category": contract.get(
                    "category", ""
                ),
                "nominal": raw_contract.get(
                    "nominal", ""
                ),
                "unit": raw_contract.get(
                    "unit", ""
                ),
                "distribution": uncertainty.get(
                    "distribution", ""
                ),
                "role": uncertainty.get(
                    "role", ""
                ),
                "source_kind": source.get(
                    "kind", ""
                ),
                "source_reference": source.get(
                    "reference", ""
                ),
            }
        )

    range_table = pd.DataFrame(
        range_rows
    ).to_html(
        index=False,
        escape=True,
        float_format=lambda value: (
            f"{value:.5g}"
        ),
    )
    level_table = pd.DataFrame(
        level_rows
    ).to_html(
        index=False,
        escape=True,
        float_format=lambda value: (
            f"{value:.5g}"
        ),
    )
    inventory_table = pd.DataFrame(
        inventory_rows
    ).to_html(
        index=False,
        escape=True,
    )

    quality_rows = [
        {
            "check": key,
            "result": value,
        }
        for key, value in quality.items()
        if isinstance(value, bool)
    ]
    quality_table = pd.DataFrame(
        quality_rows
    ).to_html(
        index=False, escape=True
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Structural sensitivity review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1450px; line-height: 1.45; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; display: block; overflow-x: auto; }}
th, td {{ border: 1px solid #ddd; padding: 0.35rem; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #f4f4f4; }}
.note {{ padding: 0.85rem; background: #f6f6f6; border-left: 4px solid #777; }}
.warning {{ padding: 0.85rem; background: #fff7e6; border-left: 4px solid #a66a00; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 0.8rem; }}
.card {{ border: 1px solid #ddd; padding: 0.8rem; }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body>
<h1>All-declared structural sensitivity review</h1>
<p class="note">
This is deterministic one-at-a-time screening. Every bar and range is the response across declared input levels while all other inputs remain nominal. These are physical response ranges, not Monte Carlo confidence intervals or statistical error bars.
</p>

<h2>Run overview</h2>
<div class="grid">
<div class="card"><strong>Parameters screened</strong><br>{int(summary.get("parameter_count", 0))}</div>
<div class="card"><strong>Parameter levels</strong><br>{int(summary.get("level_count", 0))}</div>
<div class="card"><strong>Completed bounded levels</strong><br>{int(summary.get("completed_level_count", 0))}</div>
<div class="card"><strong>Bounded simulations</strong><br>{int(manifest.get("bounded_simulation_count", 0))}</div>
<div class="card"><strong>Reference simulations</strong><br>{int(manifest.get("reference_simulation_count", 0))}</div>
<div class="card"><strong>Persistent cache hits</strong><br>{int(manifest.get("simulation_cache_hits", 0))}</div>
<div class="card"><strong>Workers</strong><br>{int(manifest.get("parallel_workers", 1))}</div>
<div class="card"><strong>Selection</strong><br>{html.escape(str(manifest.get("structural_selection_mode", "")))}</div>
</div>

<h2>Numerical health</h2>
{quality_table}

<h2>What controls the nominal result?</h2>
<p>
The tables below rank the maximum absolute response from nominal for each output. A parameter can rank highly for absolute lap time or energy while having almost no influence on the bounded-versus-infinite CVT penalty; those are different engineering questions.
</p>
{"".join(top_tables)}

<h2>Response plots</h2>
{plot_html}

<h2>Declared structural input inventory</h2>
<p>
This is the exact set discovered from the resolved vehicle, drivetrain, driver, track surface, and obstacle contracts after excluding fixed and explicitly excluded inputs.
</p>
{inventory_table}

<h2>Complete metric ranges</h2>
<p>
For every parameter and output, this table reports the nominal value, extrema, total span, and largest change from nominal.
</p>
{range_table}

<h2>Every evaluated level</h2>
<p>
This is the full deterministic case table, including completion and termination status. It is intentionally not reduced to only the bounded-versus-infinite energy diagnostic.
</p>
{level_table}

<h2>Interpretation limits</h2>
<ul>
<li>One-at-a-time screening does not measure interactions between uncertain inputs.</li>
<li>A zero response means the selected output did not change at the tested declared levels; it does not prove the mechanism is physically irrelevant in every combined scenario.</li>
<li>Completion changes and maximum-time terminations must be interpreted before ranking lap-time spans.</li>
<li>The joint full-uncertainty study remains the correct tool for combined scenario distributions and interaction-sensitive behavior.</li>
</ul>

<h2>Machine-readable artifacts</h2>
<ul>
<li><code>structural_metric_ranges.csv</code></li>
<li><code>structural_parameter_levels.csv</code></li>
<li><code>replicate_results.csv</code></li>
<li><code>summary.json</code></li>
<li><code>input_contracts.json</code></li>
<li><code>run_manifest.json</code></li>
</ul>
</body>
</html>
"""
    path.write_text(
        document, encoding="utf-8"
    )


def _append_report_links(
    output: Path,
) -> None:
    addition = (
        "\n## Structural review\n\n"
        "- [Open the complete HTML structural-sensitivity report]"
        "(structural_sensitivity_report.html)\n"
        "- [Complete parameter-level results]"
        "(structural_parameter_levels.csv)\n"
        "- [Complete metric ranges]"
        "(structural_metric_ranges.csv)\n"
    )
    for name in (
        "SUMMARY.md",
        "REPORT.md",
    ):
        path = output / name
        if not path.is_file():
            continue
        text = path.read_text(
            encoding="utf-8"
        )
        if (
            "structural_sensitivity_report.html"
            not in text
        ):
            path.write_text(
                text.rstrip()
                + "\n"
                + addition,
                encoding="utf-8",
            )


def _write_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        path.write_text(
            "", encoding="utf-8"
        )
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_title(filename: str) -> str:
    return (
        filename.removesuffix(".png")
        .replace("structural_", "")
        .replace("_tornado", "")
        .replace("_", " ")
        .title()
    )
