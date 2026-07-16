"""Artifacts for Phase 6 uncertainty and design studies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def write_study_outputs(
    *,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    scenario_draws: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    convergence: Mapping[str, Any],
    manifest: Mapping[str, Any],
    input_contracts: Mapping[str, Any],
    study_type: str,
) -> None:
    _write_rows(output / "replicate_results.csv", rows)
    _write_json_lines(output / "scenario_draws.jsonl", scenario_draws)
    _write_json(output / "summary.json", summary)
    _write_json(output / "convergence.json", convergence)
    _write_json(output / "run_manifest.json", manifest)
    _write_json(output / "input_contracts.json", input_contracts)
    _write_summary_csv(output / "summary.csv", summary)
    _write_plots(output, rows, summary, study_type)
    _write_report(output / "REPORT.md", summary, convergence, manifest)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    designs = summary.get("designs", {})
    if not isinstance(designs, Mapping) or not designs:
        parameters = summary.get("parameters", {})
        rows = []
        if isinstance(parameters, Mapping):
            for parameter, record in parameters.items():
                if not isinstance(record, Mapping):
                    continue
                for level in record.get("levels", []):
                    rows.append({"parameter_path": parameter, **level})
        _write_rows(path, rows)
        return
    rows: list[dict[str, Any]] = []
    for design_id, record in designs.items():
        if not isinstance(record, Mapping):
            continue
        flat: dict[str, Any] = {"design_id": design_id}
        flat.update({k: v for k, v in record.items() if not isinstance(v, Mapping)})
        for metric, values in record.items():
            if isinstance(values, Mapping):
                for name, value in values.items():
                    flat[f"{metric}.{name}"] = value
        rows.append(flat)
    _write_rows(path, rows)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_json_lines(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _write_plots(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    study_type: str,
) -> None:
    if not rows:
        return
    if study_type == "structural_sensitivity":
        _structural_plot(output / "structural_sensitivity.png", rows)
        return
    designs = summary.get("designs", {})
    if not isinstance(designs, Mapping) or not designs:
        return
    labels = list(designs)
    x = np.arange(len(labels))
    for metric, title, filename, ylabel in (
        (
            "lap_time_penalty_vs_infinite_s",
            "Finite-ratio lap-time penalty",
            "time_penalty_with_uncertainty.png",
            "Time penalty [s]",
        ),
        (
            "finite_ratio_opportunity_loss_energy_kj",
            "Finite-ratio opportunity loss",
            "opportunity_loss_with_uncertainty.png",
            "Opportunity loss [kJ]",
        ),
    ):
        medians = np.asarray([designs[label][metric]["median"] for label in labels], dtype=float)
        low = np.asarray([designs[label][metric]["p10"] for label in labels], dtype=float)
        high = np.asarray([designs[label][metric]["p90"] for label in labels], dtype=float)
        figure, axis = plt.subplots(figsize=(10.5, 5.5))
        axis.errorbar(x, medians, yerr=np.vstack((medians - low, high - medians)), fmt="o", capsize=4)
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title + " (median and p10–p90)")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=180)
        plt.close(figure)


def _structural_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    parameters = sorted({str(row["parameter_path"]) for row in rows})
    figure, axis = plt.subplots(figsize=(11, max(5.5, 0.7 * len(parameters))))
    for index, parameter in enumerate(parameters):
        subset = sorted(
            (row for row in rows if row["parameter_path"] == parameter),
            key=lambda row: (
                0.5 if row.get("level_probability") in (None, "")
                else float(row["level_probability"])
            ),
        )
        x = [float(row["lap_time_penalty_vs_infinite_s"]) for row in subset]
        if all(row.get("level_probability") in (None, "") for row in subset):
            offsets = np.linspace(-0.18, 0.18, len(subset)) if len(subset) > 1 else np.asarray([0.0])
            y = [index + float(offset) for offset in offsets]
        else:
            y = [
                index
                + 0.12
                * (
                    (
                        0.5
                        if row.get("level_probability") in (None, "")
                        else float(row["level_probability"])
                    )
                    - 0.5
                )
                for row in subset
            ]
        axis.plot(x, y, marker="o")
    axis.set_yticks(range(len(parameters)), parameters)
    axis.set_xlabel("Lap-time penalty vs infinite reference [s]")
    axis.set_title("One-at-a-time structural sensitivity over declared uncertainty quantiles")
    axis.grid(True, axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(
    path: Path,
    summary: Mapping[str, Any],
    convergence: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    lines = [
        f"# {manifest.get('study_name', 'Study')} results",
        "",
        f"Study type: `{manifest.get('study_type')}`  ",
        f"Sampling mode: `{manifest.get('sampling_mode')}`  ",
        f"Scenario count: {manifest.get('scenario_count')}  ",
        f"Bounded simulations: {manifest.get('bounded_simulation_count')}  ",
        f"Reference simulations: {manifest.get('reference_simulation_count')}  ",
        f"Reference cache hits: {manifest.get('reference_cache_hits')}  ",
        "",
        "## Interpretation",
        "",
        "Output p10–p90 intervals describe variation across the declared physical scenarios. "
        "Bootstrap intervals describe finite-sample error in the estimated percentiles; they are not additional physical variability.",
        "",
        "## Convergence",
        "",
        "```json",
        json.dumps(convergence, indent=2, sort_keys=True, allow_nan=False),
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
