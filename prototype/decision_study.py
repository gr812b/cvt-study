from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from measured_track import build_track_from_bundle, load_study_bundle
from metrics import compare_to_reference, summarize_trace
from models import (
    DriverModel,
    EngineModel,
    INCH_TO_METRE,
    IdealCVTModel,
    SimulationSettings,
    TireModel,
    VehicleModel,
)
from simulation import SimulationTrace, StudyCase, run_simulation
from track_builder import Track

WATTS_PER_HORSEPOWER = 745.6998715822702


def _summary(trace: SimulationTrace, engine: EngineModel) -> dict[str, object]:
    return summarize_trace(
        trace,
        target_engine_rpm=engine.peak_power_rpm,
        ideal_peak_power_w=engine.peak_power_w,
    )


def run_pair(
    *,
    track: Track,
    minimum_speed_ratio: float,
    maximum_speed_ratio: float,
    final_drive_ratio: float,
    wheel_radius_in: float,
    vehicle_mass_kg: float,
    tire_slip_stiffness: float,
    tire_peak_traction_scale: float = 1.0,
    integration_step_s: float,
    report_step_s: float,
    maximum_time_s: float | None = None,
) -> tuple[SimulationTrace, SimulationTrace, dict[str, object], dict[str, object], dict[str, object]]:
    engine = EngineModel.baja_br10()
    vehicle = VehicleModel(mass_kg=vehicle_mass_kg)
    tire = TireModel(
        wheel_radius_m=wheel_radius_in * INCH_TO_METRE,
        peak_traction_scale=tire_peak_traction_scale,
        slip_stiffness_n_per_mps=tire_slip_stiffness,
    )
    cvt = IdealCVTModel(
        minimum_speed_ratio=minimum_speed_ratio,
        maximum_speed_ratio=maximum_speed_ratio,
        final_drive_ratio=final_drive_ratio,
    )
    driver = DriverModel()
    settings = SimulationSettings(
        maximum_time_s=(
            maximum_time_s
            if maximum_time_s is not None
            else max(300.0, track.length_m / 2.5)
        ),
        integration_step_s=integration_step_s,
        report_step_s=report_step_s,
    )
    bounded_case = StudyCase(
        name="Measured-track bounded ideal CVT",
        engine=engine,
        vehicle=vehicle,
        tire=tire,
        cvt=cvt,
        driver=driver,
        infinite_cvt=False,
    )
    reference_case = StudyCase(
        name="Measured-track unbounded ideal CVT",
        engine=engine,
        vehicle=vehicle,
        tire=tire,
        cvt=cvt,
        driver=driver,
        infinite_cvt=True,
    )
    reference = run_simulation(case=reference_case, track=track, settings=settings)
    bounded = run_simulation(case=bounded_case, track=track, settings=settings)
    reference_summary = _summary(reference, engine)
    bounded_summary = compare_to_reference(_summary(bounded, engine), reference_summary)
    setup = {
        "engine": {
            "peak_power_rpm": engine.peak_power_rpm,
            "peak_power_w": engine.peak_power_w,
        },
        "vehicle": asdict(vehicle),
        "tire": asdict(tire),
        "cvt": asdict(cvt),
        "driver": asdict(driver),
        "settings": asdict(settings),
    }
    return bounded, reference, bounded_summary, reference_summary, setup


def _cumulative_integral(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(time_s)
    return np.r_[0.0, np.cumsum(increments)]


def _gate_table(track: Track, bounded: SimulationTrace, reference: SimulationTrace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    bx = bounded.numeric["distance_m"]
    rx = reference.numeric["distance_m"]
    for gate in track.speed_gates:
        target_kmh = gate.target_speed_mps * 3.6
        bounded_kmh = float(
            np.interp(gate.position_m, bx, bounded.numeric["vehicle_speed_kmh"])
        )
        reference_kmh = float(
            np.interp(gate.position_m, rx, reference.numeric["vehicle_speed_kmh"])
        )
        tolerance_kmh = 0.5
        rows.append(
            {
                "source_group_id": gate.source_group_id,
                "gate_name": gate.name,
                "position_m": gate.position_m,
                "target_speed_kmh": target_kmh,
                "confidence_score": gate.confidence_score,
                "confidence_class": gate.confidence_class,
                "bounded_speed_kmh": bounded_kmh,
                "infinite_speed_kmh": reference_kmh,
                "bounded_excess_over_ceiling_kmh": max(0.0, bounded_kmh - target_kmh),
                "infinite_excess_over_ceiling_kmh": max(0.0, reference_kmh - target_kmh),
                "bounded_ceiling_compliant_0p5kmh": bounded_kmh <= target_kmh + tolerance_kmh,
                "infinite_ceiling_compliant_0p5kmh": reference_kmh <= target_kmh + tolerance_kmh,
            }
        )
    return pd.DataFrame(rows)


def _plot_pair(
    *,
    track: Track,
    bounded: SimulationTrace,
    reference: SimulationTrace,
    bounded_summary: Mapping[str, object],
    minimum_ratio: float,
    maximum_ratio: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    x = bounded.numeric["distance_m"]
    xr = reference.numeric["distance_m"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(x, bounded.numeric["vehicle_speed_kmh"], label="Bounded CVT", color="#1f77b4")
    axes[0].plot(xr, reference.numeric["vehicle_speed_kmh"], label="Unbounded reference", color="#333333", linestyle="--")
    for index, gate in enumerate(track.speed_gates):
        axes[0].scatter(
            gate.position_m,
            gate.target_speed_mps * 3.6,
            marker="*",
            s=90,
            color="#d95f02",
            label="Accepted measured gate" if index == 0 else None,
            zorder=5,
        )
    axes[0].set_ylabel("Speed [km/h]")
    axes[0].set_title("Measured-gate track: bounded versus unbounded CVT")
    axes[0].legend(ncol=3)

    time = bounded.numeric["time_s"]
    clutch = _cumulative_integral(bounded.numeric["clutch_loss_power_w"], time) / 1000.0
    off_peak = _cumulative_integral(
        bounded.numeric["operating_point_shortfall_power_w"], time
    ) / 1000.0
    axes[1].plot(x, clutch, label="Clutch contribution", color="#7570b3")
    axes[1].plot(x, off_peak, label="Off-peak contribution", color="#e7298a")
    axes[1].plot(x, clutch + off_peak, label="Total ratio opportunity loss", color="#d95f02", linewidth=2.0)
    axes[1].set_ylabel("Cumulative opportunity loss [kJ]")
    axes[1].legend(ncol=3)

    axes[2].plot(x, bounded.numeric["cvt_ratio"], color="#1b9e77", label="Selected ratio")
    axes[2].axhline(maximum_ratio, linestyle=":", color="#555555", label="Low-speed bound")
    axes[2].axhline(minimum_ratio, linestyle="--", color="#555555", label="High-speed bound")
    axes[2].set_ylabel(r"CVT ratio $\omega_e/\omega_s$")
    axes[2].set_xlabel("Track distance, s [m]")
    axes[2].legend(ncol=3)
    for ax in axes:
        ax.grid(True, alpha=0.22)
        for gate in track.speed_gates:
            ax.axvline(gate.position_m, color="#d95f02", alpha=0.10, linewidth=0.8)
    fig.suptitle(
        f"Time penalty {float(bounded_summary['lap_time_penalty_vs_infinite_s']):.2f} s | "
        f"opportunity loss {float(bounded_summary['finite_ratio_opportunity_loss_energy_kj']):.1f} kJ"
    )
    fig.tight_layout()
    path_main = output_dir / "01_bounded_vs_unbounded_decision_view.png"
    fig.savefig(path_main, dpi=190, bbox_inches="tight")
    plt.close(fig)

    labels = ["Clutch", "Off-peak", "Tire slip", "Braking", "Effective events", "Rolling + aero"]
    values = [
        float(bounded_summary["clutch_loss_energy_kj"]),
        float(bounded_summary["engine_operating_shortfall_energy_kj"]),
        float(bounded_summary["tire_slip_loss_energy_kj"]),
        float(bounded_summary["brake_loss_energy_kj"]),
        float(bounded_summary["obstacle_loss_energy_kj"]),
        float(bounded_summary["rolling_loss_energy_kj"]) + float(bounded_summary["aerodynamic_loss_energy_kj"]),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(labels, values, color=["#7570b3", "#e7298a", "#66a61e", "#a6761d", "#e6ab02", "#1f78b4"])
    ax.bar_label(bars, fmt="%.1f")
    ax.set_ylabel("Energy [kJ]")
    ax.set_title("Bounded-CVT energy accounting (effective-event loss remains scenario-based)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path_energy = output_dir / "02_energy_accounting.png"
    fig.savefig(path_energy, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path_main, path_energy


def run_measured_comparison(
    *,
    bundle_path: Path,
    output_dir: Path,
    minimum_speed_ratio: float = 0.9,
    maximum_speed_ratio: float = 3.5,
    final_drive_ratio: float = 7.556,
    wheel_radius_in: float = 11.0,
    vehicle_mass_kg: float = 300.0,
    tire_slip_stiffness: float = 2500.0,
    tire_peak_traction_scale: float = 1.0,
    integration_step_s: float = 0.001,
    report_step_s: float = 0.02,
    minimum_gate_confidence: float = 60.0,
    gate_speed_quantile: str = "median",
    loss_quantile: str = "nominal",
    loss_scale: float = 1.0,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_study_bundle(bundle_path)
    track = build_track_from_bundle(
        bundle,
        gate_speed_quantile=gate_speed_quantile,
        minimum_gate_confidence=minimum_gate_confidence,
        loss_quantile=loss_quantile,
        loss_scale=loss_scale,
    )
    bounded, reference, bounded_summary, reference_summary, setup = run_pair(
        track=track,
        minimum_speed_ratio=minimum_speed_ratio,
        maximum_speed_ratio=maximum_speed_ratio,
        final_drive_ratio=final_drive_ratio,
        wheel_radius_in=wheel_radius_in,
        vehicle_mass_kg=vehicle_mass_kg,
        tire_slip_stiffness=tire_slip_stiffness,
        tire_peak_traction_scale=tire_peak_traction_scale,
        integration_step_s=integration_step_s,
        report_step_s=report_step_s,
    )
    bounded.write_csv(output_dir / "bounded_trace.csv")
    reference.write_csv(output_dir / "unbounded_reference_trace.csv")
    (output_dir / "bounded_summary.json").write_text(json.dumps(bounded_summary, indent=2), encoding="utf-8")
    (output_dir / "unbounded_reference_summary.json").write_text(json.dumps(reference_summary, indent=2), encoding="utf-8")
    (output_dir / "resolved_measured_case.json").write_text(
        json.dumps({"track": track.to_dict(), "setup": setup}, indent=2), encoding="utf-8"
    )
    gate_table = _gate_table(track, bounded, reference)
    gate_table.to_csv(output_dir / "simulated_gate_check.csv", index=False)
    _plot_pair(
        track=track,
        bounded=bounded,
        reference=reference,
        bounded_summary=bounded_summary,
        minimum_ratio=minimum_speed_ratio,
        maximum_ratio=maximum_speed_ratio,
        output_dir=output_dir,
    )
    maximum_gate_excess = max(
        float(gate_table["bounded_excess_over_ceiling_kmh"].max()),
        float(gate_table["infinite_excess_over_ceiling_kmh"].max()),
    ) if not gate_table.empty else 0.0
    all_gates_compliant = bool(
        gate_table["bounded_ceiling_compliant_0p5kmh"].all()
        and gate_table["infinite_ceiling_compliant_0p5kmh"].all()
    ) if not gate_table.empty else True
    report = (
        "# Measured-track bounded versus unbounded CVT\n\n"
        f"- Accepted gates used: {len(track.speed_gates)}\n"
        f"- Effective event features used: {len(track.features)}\n"
        f"- Bounded lap time: {float(bounded_summary['lap_time_s']):.3f} s\n"
        f"- Unbounded lap time: {float(reference_summary['lap_time_s']):.3f} s\n"
        f"- Time penalty: {float(bounded_summary['lap_time_penalty_vs_infinite_s']):.3f} s\n"
        f"- Ratio opportunity loss: {float(bounded_summary['finite_ratio_opportunity_loss_energy_kj']):.3f} kJ\n"
        f"- Reference dominance check: {bounded_summary['reference_dominance_pass']}\n"
        f"- Bounded energy-balance relative error: {float(bounded_summary['energy_balance_relative_error']):.5f}\n"
        f"- Unbounded energy-balance relative error: {float(reference_summary['energy_balance_relative_error']):.5f}\n"
        f"- All gates within 0.5 km/h ceiling tolerance: {all_gates_compliant}\n"
        f"- Maximum simulated gate-ceiling excess: {maximum_gate_excess:.3f} km/h\n\n"
        "The effective-event loss channel is an uncertainty seed derived from observed net kinetic-state change. It is not calibrated terrain dissipation.\n"
    )
    (output_dir / "DECISION_REPORT.md").write_text(report, encoding="utf-8")
    return {
        "track": track,
        "bounded": bounded,
        "reference": reference,
        "bounded_summary": bounded_summary,
        "reference_summary": reference_summary,
        "gate_table": gate_table,
    }


def _parameterized_cvt(
    parameter: str,
    value: float,
    *,
    minimum_speed_ratio: float,
    maximum_speed_ratio: float,
    final_drive_ratio: float,
) -> tuple[float, float, float]:
    values = {
        "minimum_speed_ratio": minimum_speed_ratio,
        "maximum_speed_ratio": maximum_speed_ratio,
        "final_drive_ratio": final_drive_ratio,
    }
    if parameter not in values:
        raise ValueError(f"Unsupported sweep parameter: {parameter}")
    values[parameter] = float(value)
    if values["minimum_speed_ratio"] >= values["maximum_speed_ratio"]:
        raise ValueError("Every sweep case requires minimum_speed_ratio < maximum_speed_ratio")
    return (
        values["minimum_speed_ratio"],
        values["maximum_speed_ratio"],
        values["final_drive_ratio"],
    )


def _scenario_draws(bundle: Mapping[str, object], rng: np.random.Generator) -> tuple[dict[str, float], dict[str, float]]:
    gates: dict[str, float] = {}
    for raw in bundle.get("speed_gates", []):
        gate = dict(raw)
        gates[str(gate["analysis_group_id"])] = float(
            rng.triangular(
                float(gate["target_speed_p10_kmh"]),
                float(gate["target_speed_median_kmh"]),
                float(gate["target_speed_p90_kmh"]),
            )
        )
    losses: dict[str, float] = {}
    for raw in bundle.get("event_groups", []):
        event = dict(raw)
        if str(event.get("analysis_role")) != "track_event":
            continue
        raw_values = sorted(
            [
                float(event["effective_specific_loss_low_j_per_kg"]),
                float(event["effective_specific_loss_nominal_j_per_kg"]),
                float(event["effective_specific_loss_high_j_per_kg"]),
            ]
        )
        low, nominal, high = raw_values
        losses[str(event["analysis_group_id"])] = float(rng.triangular(low, nominal, high)) if high > low else nominal
    return gates, losses


def run_measured_sweep(
    *,
    bundle_path: Path,
    output_dir: Path,
    parameter: str,
    values: Sequence[float],
    replicates: int = 12,
    random_seed: int = 20260714,
    minimum_speed_ratio: float = 0.9,
    maximum_speed_ratio: float = 3.5,
    final_drive_ratio: float = 7.556,
    wheel_radius_in: float = 11.0,
    vehicle_mass_kg: float = 300.0,
    tire_slip_stiffness: float = 2500.0,
    tire_peak_traction_scale: float = 1.0,
    integration_step_s: float = 0.001,
    report_step_s: float = 0.05,
    minimum_gate_confidence: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_study_bundle(bundle_path)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    gate_draw_rows: list[dict[str, object]] = []
    loss_draw_rows: list[dict[str, object]] = []
    for replicate in range(1, replicates + 1):
        gate_overrides, loss_overrides = _scenario_draws(bundle, rng)
        gate_draw_rows.extend(
            {"replicate": replicate, "analysis_group_id": key, "target_speed_kmh": value}
            for key, value in gate_overrides.items()
        )
        loss_draw_rows.extend(
            {
                "replicate": replicate,
                "analysis_group_id": key,
                "effective_specific_loss_j_per_kg": value,
            }
            for key, value in loss_overrides.items()
        )
        track = build_track_from_bundle(
            bundle,
            minimum_gate_confidence=minimum_gate_confidence,
            gate_speed_overrides_kmh=gate_overrides,
            event_loss_overrides_j_per_kg=loss_overrides,
        )

        # The unbounded reference has the same wheel-power behavior regardless of
        # the finite CVT ratio bounds or final-drive value being swept. Compute it
        # once for this paired uncertainty realization, then reuse its summary for
        # every bounded design in the replicate.
        engine = EngineModel.baja_br10()
        vehicle = VehicleModel(mass_kg=vehicle_mass_kg)
        tire = TireModel(
            wheel_radius_m=wheel_radius_in * INCH_TO_METRE,
            peak_traction_scale=tire_peak_traction_scale,
            slip_stiffness_n_per_mps=tire_slip_stiffness,
        )
        reference_cvt = IdealCVTModel(
            minimum_speed_ratio=minimum_speed_ratio,
            maximum_speed_ratio=maximum_speed_ratio,
            final_drive_ratio=final_drive_ratio,
        )
        driver = DriverModel()
        settings = SimulationSettings(
            maximum_time_s=max(300.0, track.length_m / 2.5),
            integration_step_s=integration_step_s,
            report_step_s=report_step_s,
        )
        reference = run_simulation(
            case=StudyCase(
                name=f"Measured-track unbounded ideal CVT, replicate {replicate}",
                engine=engine,
                vehicle=vehicle,
                tire=tire,
                cvt=reference_cvt,
                driver=driver,
                infinite_cvt=True,
            ),
            track=track,
            settings=settings,
        )
        reference_summary = _summary(reference, engine)

        for value in values:
            minimum, maximum, final_drive = _parameterized_cvt(
                parameter,
                float(value),
                minimum_speed_ratio=minimum_speed_ratio,
                maximum_speed_ratio=maximum_speed_ratio,
                final_drive_ratio=final_drive_ratio,
            )
            bounded = run_simulation(
                case=StudyCase(
                    name=f"Measured-track bounded ideal CVT, {parameter}={float(value):g}",
                    engine=engine,
                    vehicle=vehicle,
                    tire=tire,
                    cvt=IdealCVTModel(
                        minimum_speed_ratio=minimum,
                        maximum_speed_ratio=maximum,
                        final_drive_ratio=final_drive,
                    ),
                    driver=driver,
                    infinite_cvt=False,
                ),
                track=track,
                settings=settings,
            )
            bounded_summary = compare_to_reference(_summary(bounded, engine), reference_summary)
            rows.append(
                {
                    "replicate": replicate,
                    "parameter": parameter,
                    "parameter_value": float(value),
                    "minimum_speed_ratio": minimum,
                    "maximum_speed_ratio": maximum,
                    "final_drive_ratio": final_drive,
                    "lap_time_s": bounded_summary["lap_time_s"],
                    "reference_lap_time_s": reference_summary["lap_time_s"],
                    "time_penalty_vs_unbounded_s": bounded_summary["lap_time_penalty_vs_infinite_s"],
                    "opportunity_loss_energy_kj": bounded_summary["finite_ratio_opportunity_loss_energy_kj"],
                    "clutch_loss_energy_kj": bounded_summary["clutch_loss_energy_kj"],
                    "off_peak_loss_energy_kj": bounded_summary["engine_operating_shortfall_energy_kj"],
                    "positive_demand_variable_ratio_time_s": bounded_summary["positive_demand_time_variable_ratio_s"],
                    "positive_demand_low_ratio_time_s": bounded_summary["positive_demand_time_low_ratio_s"],
                    "positive_demand_high_ratio_time_s": bounded_summary["positive_demand_time_high_ratio_s"],
                    "energy_balance_relative_error": bounded_summary["energy_balance_relative_error"],
                    "reference_dominance_pass": bounded_summary["reference_dominance_pass"],
                }
            )
    raw = pd.DataFrame(rows)
    raw["opportunity_loss_regret_kj"] = raw["opportunity_loss_energy_kj"] - raw.groupby(
        "replicate"
    )["opportunity_loss_energy_kj"].transform("min")
    metrics = [
        "opportunity_loss_energy_kj",
        "time_penalty_vs_unbounded_s",
        "positive_demand_variable_ratio_time_s",
        "clutch_loss_energy_kj",
        "off_peak_loss_energy_kj",
        "opportunity_loss_regret_kj",
    ]
    summaries: list[dict[str, object]] = []
    for value, group in raw.groupby("parameter_value", sort=True):
        row: dict[str, object] = {"parameter": parameter, "parameter_value": value, "replicates": len(group)}
        for metric in metrics:
            row[f"{metric}_p10"] = float(group[metric].quantile(0.10))
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_p90"] = float(group[metric].quantile(0.90))
        row["reference_dominance_fraction"] = float(group["reference_dominance_pass"].mean())
        row["best_loss_win_fraction"] = float(
            (group["opportunity_loss_regret_kj"] <= 1.0e-9).mean()
        )
        row["maximum_abs_energy_balance_relative_error"] = float(group["energy_balance_relative_error"].abs().max())
        summaries.append(row)
    summary = pd.DataFrame(summaries).sort_values("parameter_value").reset_index(drop=True)
    raw.to_csv(output_dir / "sweep_replicates.csv", index=False)
    pd.DataFrame(gate_draw_rows).to_csv(output_dir / "scenario_gate_draws.csv", index=False)
    pd.DataFrame(loss_draw_rows).to_csv(output_dir / "scenario_loss_draws.csv", index=False)
    summary.to_csv(output_dir / "sweep_confidence_summary.csv", index=False)
    _plot_sweep_confidence(summary, parameter, output_dir)
    ranked = summary.sort_values(
        ["opportunity_loss_energy_kj_median", "time_penalty_vs_unbounded_s_median"]
    )
    ranked.to_csv(output_dir / "sweep_ranking.csv", index=False)
    (output_dir / "SWEEP_REPORT.md").write_text(
        "# Measured-track CVT sweep\n\n"
        f"- Parameter: {parameter}\n"
        f"- Values: {', '.join(str(value) for value in values)}\n"
        f"- Paired uncertainty replicates: {replicates}\n"
        f"- Random seed: {random_seed}\n"
        f"- Best median opportunity-loss value: {float(ranked.iloc[0]['parameter_value']):g}\n"
        f"- Its paired best-loss win fraction: {float(ranked.iloc[0]['best_loss_win_fraction']):.3f}\n"
        f"- Maximum absolute energy-balance residual: {float(summary['maximum_abs_energy_balance_relative_error'].max()):.5f}\n"
        f"- All unbounded-reference dominance checks passed: {bool((summary['reference_dominance_fraction'] == 1.0).all())}\n\n"
        "Bars show the 10th–90th percentile across paired measured gate-speed and effective-event-loss scenarios. Effective-event losses remain uncalibrated GPS-derived scenario seeds.\n",
        encoding="utf-8",
    )
    return raw, summary


def _plot_sweep_confidence(summary: pd.DataFrame, parameter: str, output_dir: Path) -> Path:
    labels = {
        "final_drive_ratio": "Final-drive reduction ratio",
        "minimum_speed_ratio": "Minimum CVT ratio (high-speed end)",
        "maximum_speed_ratio": "Maximum CVT ratio (low-speed end)",
    }
    x = summary["parameter_value"].to_numpy(dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11), sharex=True)
    specs = [
        ("opportunity_loss_energy_kj", "Ratio opportunity loss [kJ]", "#d95f02"),
        ("time_penalty_vs_unbounded_s", "Time penalty vs unbounded [s]", "#1f78b4"),
        ("positive_demand_variable_ratio_time_s", "Positive-demand time in variable range [s]", "#1b9e77"),
    ]
    for ax, (metric, ylabel, color) in zip(axes, specs):
        median = summary[f"{metric}_median"].to_numpy(dtype=float)
        low = median - summary[f"{metric}_p10"].to_numpy(dtype=float)
        high = summary[f"{metric}_p90"].to_numpy(dtype=float) - median
        ax.errorbar(x, median, yerr=np.vstack([low, high]), marker="o", capsize=4, color=color, linewidth=1.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.23)
    axes[0].set_title("CVT design sweep with measured-input uncertainty")
    axes[-1].set_xlabel(labels.get(parameter, parameter))
    fig.tight_layout()
    path = output_dir / "01_sweep_decision_with_confidence.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path
