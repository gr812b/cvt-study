"""Phase 7 attribution plus Phase 8 hierarchical report orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from . import reporting as phase6_reporting
from .attribution import build_uncertainty_attribution, flatten_attribution
from .decision import synthesize_decision
from .energy import build_energy_accounting
from .report_writer import write_hierarchical_reports


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
    seed = int(manifest.get("random_seed", 0))
    bootstrap = int(manifest.get("bootstrap_resamples", 1000))
    energy, energy_shares, feature_energy = build_energy_accounting(
        rows, seed=seed, bootstrap_resamples=bootstrap
    )
    attribution = build_uncertainty_attribution(
        study_type=study_type,
        rows=rows,
        scenario_draws=scenario_draws,
        input_contracts=input_contracts,
        seed=seed,
        bootstrap_resamples=bootstrap,
    )
    decision = synthesize_decision(
        summary=summary,
        convergence=convergence,
        attribution=attribution,
        manifest=manifest,
    )
    manifest_output = {
        **manifest,
        "decision_readiness": {
            key: bool(decision.get(key, False))
            for key in (
                "numerically_valid",
                "evidence_ready",
                "statistically_ready",
                "directionally_robust",
                "decision_ready",
            )
        },
    }

    phase6_reporting._write_rows(output / "replicate_results.csv", rows)
    phase6_reporting._write_json_lines(output / "scenario_draws.jsonl", scenario_draws)
    phase6_reporting._write_json(output / "summary.json", summary)
    phase6_reporting._write_json(output / "convergence.json", convergence)
    phase6_reporting._write_json(output / "run_manifest.json", manifest_output)
    phase6_reporting._write_json(output / "input_contracts.json", input_contracts)
    phase6_reporting._write_json(output / "energy_accounting.json", energy)
    phase6_reporting._write_json(output / "uncertainty_attribution.json", attribution)
    phase6_reporting._write_json(output / "decision_summary.json", decision)
    phase6_reporting._write_summary_csv(output / "summary.csv", summary)
    phase6_reporting._write_rows(output / "energy_accounting.csv", _flatten_energy(energy))
    phase6_reporting._write_rows(output / "physical_loss_shares.csv", energy_shares)
    phase6_reporting._write_rows(output / "feature_energy_results.csv", feature_energy)
    phase6_reporting._write_rows(
        output / "uncertainty_attribution.csv", flatten_attribution(attribution)
    )
    phase6_reporting._write_plots(output, rows, summary, study_type)
    _energy_plot(output / "physical_energy_attribution.png", energy)
    _attribution_plot(output / "uncertainty_attribution.png", attribution)
    write_hierarchical_reports(
        output=output,
        decision=decision,
        summary=summary,
        convergence=convergence,
        energy=energy,
        attribution=attribution,
        manifest=manifest_output,
    )


def regenerate_study_reports(output: Path) -> None:
    names = (
        "decision_summary",
        "summary",
        "convergence",
        "energy_accounting",
        "uncertainty_attribution",
        "run_manifest",
    )
    data: dict[str, Any] = {}
    for name in names:
        path = output / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Cannot regenerate report; missing {path}")
        data[name] = json.loads(path.read_text(encoding="utf-8"))
    write_hierarchical_reports(
        output=output,
        decision=data["decision_summary"],
        summary=data["summary"],
        convergence=data["convergence"],
        energy=data["energy_accounting"],
        attribution=data["uncertainty_attribution"],
        manifest=data["run_manifest"],
    )


def _flatten_energy(energy: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for design_id, design in energy.get("designs", {}).items():
        for side, components in design.items():
            for component, record in components.items():
                rows.append(
                    {
                        "design_id": design_id,
                        "side": side,
                        "component": component,
                        **record,
                    }
                )
    return rows


def _energy_plot(path: Path, energy: Mapping[str, Any]) -> None:
    designs = energy.get("designs", {})
    if not designs:
        return
    design_id = next(iter(designs))
    bounded = designs[design_id].get("bounded", {})
    components = [
        name
        for name in (
            "drivetrain_loss_energy_kj",
            "clutch_loss_energy_kj",
            "tire_slip_loss_energy_kj",
            "brake_loss_energy_kj",
            "rolling_loss_energy_kj",
            "aerodynamic_loss_energy_kj",
            "obstacle_loss_energy_kj",
        )
        if name in bounded
    ]
    if not components:
        return
    medians = np.asarray([bounded[name]["median"] for name in components], dtype=float)
    low = np.asarray([bounded[name]["p10"] for name in components], dtype=float)
    high = np.asarray([bounded[name]["p90"] for name in components], dtype=float)
    x = np.arange(len(components))
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(x, medians, color="#4f81bd")
    axis.errorbar(
        x,
        medians,
        yerr=np.vstack((medians - low, high - medians)),
        fmt="none",
        color="black",
        capsize=3,
    )
    axis.set_xticks(
        x,
        [name.replace("_energy_kj", "").replace("_", " ") for name in components],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Energy [kJ]")
    axis.set_title(f"Physical loss accounting — {design_id} (median, p10–p90)")
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _attribution_plot(path: Path, attribution: Mapping[str, Any]) -> None:
    designs = attribution.get("designs", {})
    if not designs:
        return
    design_id = next(iter(designs))
    rows = (
        designs[design_id]
        .get("metrics", {})
        .get("lap_time_penalty_vs_infinite_s", {})
        .get("numeric", [])[:12]
    )
    if not rows:
        return
    rows = list(reversed(rows))
    figure, axis = plt.subplots(figsize=(11, max(5.5, 0.45 * len(rows))))
    axis.barh(
        [str(row["path"]) for row in rows],
        [float(row["relative_screening_importance"]) for row in rows],
        color="#c0504d",
    )
    axis.set_xlabel("Relative screening importance")
    axis.set_title(f"Lap-time uncertainty screening — {design_id}")
    axis.grid(True, axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
