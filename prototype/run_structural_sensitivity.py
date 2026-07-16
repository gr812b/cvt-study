from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from metrics import compare_to_reference, summarize_trace
from models import (
    DriverModel,
    EngineModel,
    EngineTorquePoint,
    INCH_TO_METRE,
    IdealCVTModel,
    SimulationSettings,
    TireModel,
    VehicleModel,
)
from simulation import StudyCase, run_simulation
from track_builder import Track
from standard_case import (
    STANDARD_FINAL_DRIVE_RATIO,
    STANDARD_INTEGRATION_STEP_S,
    STANDARD_MAXIMUM_CVT_RATIO,
    STANDARD_MINIMUM_CVT_RATIO,
    STANDARD_WHEEL_RADIUS_IN,
)

HERE = Path(__file__).resolve().parent
DEFAULT_TRACK = HERE / "tracks" / "lot_m.json"

# Seven levels give readable curves without implying Monte-Carlo precision.
DEFAULT_LEVELS: dict[str, tuple[float, ...]] = {
    "drag_area_scale": (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30),
    "engine_power_scale": (0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05),
    "transmission_efficiency": (0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 1.00),
    "rolling_resistance_scale": (0.70, 0.85, 1.00, 1.15, 1.30, 1.40, 1.50),
    "peak_traction_scale": (0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20),
    "slip_stiffness_scale": (0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.45),
    "obstacle_resistance_scale": (0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.40),
}

LABELS = {
    "drag_area_scale": "Aerodynamic $C_DA$ scale",
    "engine_power_scale": "Engine power scale",
    "transmission_efficiency": "Drivetrain efficiency",
    "rolling_resistance_scale": "Rolling-resistance scale",
    "peak_traction_scale": "Peak tire-traction scale",
    "slip_stiffness_scale": "Tire slip-build-up stiffness scale",
    "obstacle_resistance_scale": "Obstacle-resistance scale",
}

DIRECTION_NOTES = {
    "drag_area_scale": "Low assumed drag tends to overvalue taller gearing.",
    "engine_power_scale": "High assumed engine power tends to create more high-speed opportunity.",
    "transmission_efficiency": "High assumed efficiency tends to create more high-speed opportunity.",
    "rolling_resistance_scale": "Low assumed rolling resistance tends to overvalue taller gearing.",
    "peak_traction_scale": "Higher peak traction raises the maximum force the ground can provide.",
    "slip_stiffness_scale": "Higher slip stiffness develops traction with less wheelspin.",
    "obstacle_resistance_scale": "High obstacle resistance can favor shorter gearing and stronger recovery.",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired structural sensitivities against the infinite-CVT reference.")
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ideal_cvt_track_study/structural_sensitivity"))
    parser.add_argument("--minimum-cvt-ratio", type=float, default=STANDARD_MINIMUM_CVT_RATIO)
    parser.add_argument("--maximum-cvt-ratio", type=float, default=STANDARD_MAXIMUM_CVT_RATIO)
    parser.add_argument("--final-drive-ratio", type=float, default=STANDARD_FINAL_DRIVE_RATIO)
    parser.add_argument("--wheel-radius-in", type=float, default=STANDARD_WHEEL_RADIUS_IN)
    parser.add_argument("--vehicle-mass-kg", type=float, default=300.0)
    parser.add_argument("--integration-step-s", type=float, default=STANDARD_INTEGRATION_STEP_S)
    parser.add_argument("--report-step-s", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parameters", nargs="+", choices=tuple(DEFAULT_LEVELS), default=list(DEFAULT_LEVELS))
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args(argv)


def _scaled_engine(base: EngineModel, scale: float) -> EngineModel:
    return EngineModel(
        points=tuple(EngineTorquePoint(p.rpm, p.torque_nm * scale) for p in base.points),
        target_rpm=base.target_rpm,
    )


def _run(payload: tuple[Any, ...]) -> dict[str, Any]:
    parameter, value, track, base_engine, base_vehicle, base_cvt, base_tire, driver, settings = payload
    engine = base_engine
    vehicle = base_vehicle
    cvt = base_cvt
    case_kwargs = {
        "traction_scale": 1.0,
        "rolling_resistance_scale": 1.0,
        "obstacle_resistance_scale": 1.0,
    }
    if parameter == "drag_area_scale":
        vehicle = replace(vehicle, drag_coefficient=vehicle.drag_coefficient * value)
    elif parameter == "engine_power_scale":
        engine = _scaled_engine(engine, value)
    elif parameter == "transmission_efficiency":
        cvt = replace(cvt, transmission_efficiency=value)
    elif parameter == "peak_traction_scale":
        base_tire = replace(base_tire, peak_traction_scale=value)
    elif parameter == "slip_stiffness_scale":
        base_tire = replace(base_tire, slip_stiffness_n_per_mps=base_tire.slip_stiffness_n_per_mps * value)
    elif parameter in case_kwargs:
        case_kwargs[parameter] = value
    else:
        raise ValueError(parameter)

    bounded_case = StudyCase("bounded", engine, vehicle, base_tire, cvt, driver, False, **case_kwargs)
    reference_case = StudyCase("infinite", engine, vehicle, base_tire, cvt, driver, True, **case_kwargs)
    reference = run_simulation(case=reference_case, track=track, settings=settings)
    bounded = run_simulation(case=bounded_case, track=track, settings=settings)
    ref_summary = summarize_trace(reference, target_engine_rpm=engine.peak_power_rpm, ideal_peak_power_w=engine.peak_power_w)
    summary = compare_to_reference(
        summarize_trace(bounded, target_engine_rpm=engine.peak_power_rpm, ideal_peak_power_w=engine.peak_power_w),
        ref_summary,
    )
    return {
        "parameter": parameter,
        "value": value,
        "bounded_lap_time_s": summary["lap_time_s"],
        "infinite_lap_time_s": ref_summary["lap_time_s"],
        "time_penalty_s": summary["lap_time_penalty_vs_infinite_s"],
        "opportunity_loss_energy_kj": summary["finite_ratio_opportunity_loss_energy_kj"],
        "opportunity_loss_average_hp": summary["finite_ratio_opportunity_loss_average_power_hp"],
        "time_low_ratio_s": summary["time_low_ratio_s"],
        "time_variable_ratio_s": summary["time_variable_ratio_s"],
        "time_high_ratio_s": summary["time_high_ratio_s"],
        "aero_loss_energy_kj": summary["aerodynamic_loss_energy_kj"],
        "rolling_loss_energy_kj": summary["rolling_loss_energy_kj"],
        "obstacle_loss_energy_kj": summary["obstacle_loss_energy_kj"],
        "tire_slip_loss_energy_kj": summary["tire_slip_loss_energy_kj"],
        "completed": bool(summary["completed"] and ref_summary["completed"]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], output_dir: Path, show: bool) -> list[Path]:
    paths: list[Path] = []
    for metric, ylabel, filename in (
        ("time_penalty_s", "Time penalty vs infinite CVT [s]", "01_time_penalty_sensitivity.png"),
        ("opportunity_loss_energy_kj", "Finite-ratio opportunity loss [kJ]", "02_opportunity_loss_sensitivity.png"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for parameter in DEFAULT_LEVELS:
            subset = sorted((r for r in rows if r["parameter"] == parameter), key=lambda r: r["value"])
            if not subset:
                continue
            x = np.asarray([r["value"] for r in subset], dtype=float)
            y = np.asarray([r[metric] for r in subset], dtype=float)
            ax.plot(x, y, marker="o", label=LABELS[parameter])
        ax.axvline(1.0, linestyle=":", linewidth=1.0, label="Nominal scale / 100%")
        ax.set_xlabel("Parameter value or scale")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.replace(" [s]", "").replace(" [kJ]", ""))
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8, ncol=2)
        fig.tight_layout(); path=output_dir/filename; fig.savefig(path, dpi=180); paths.append(path)

    # Tornado-style normalized endpoint effect, centered on each parameter's nominal value.
    effects=[]
    for parameter in DEFAULT_LEVELS:
        subset=sorted((r for r in rows if r["parameter"]==parameter), key=lambda r:r["value"])
        if not subset:
            continue
        nominal=min(subset, key=lambda r:abs(r["value"]-1.0))
        effects.append((parameter, subset[0]["time_penalty_s"]-nominal["time_penalty_s"], subset[-1]["time_penalty_s"]-nominal["time_penalty_s"]))
    effects.sort(key=lambda item:max(abs(item[1]),abs(item[2])))
    fig, ax=plt.subplots(figsize=(10,6))
    y=np.arange(len(effects)); lows=np.asarray([e[1] for e in effects]); highs=np.asarray([e[2] for e in effects])
    ax.barh(y, lows, label="Low endpoint minus nominal")
    ax.barh(y, highs, label="High endpoint minus nominal")
    ax.set_yticks(y, [LABELS[e[0]] for e in effects]); ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel("Change in bounded-vs-infinite time penalty [s]")
    ax.set_title("Structural sensitivity: directional endpoint effect")
    ax.grid(True, axis="x", alpha=0.25); ax.legend()
    fig.tight_layout(); path=output_dir/"03_directional_tornado_time_penalty.png"; fig.savefig(path,dpi=180); paths.append(path)
    if show: plt.show()
    else: plt.close("all")
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    args=parse_args(argv); args.output_dir.mkdir(parents=True, exist_ok=True)
    track=Track.from_json(args.track)
    engine=EngineModel.baja_br10(); vehicle=VehicleModel(mass_kg=args.vehicle_mass_kg)
    cvt=IdealCVTModel(args.minimum_cvt_ratio,args.maximum_cvt_ratio,args.final_drive_ratio,1.0,True)
    tire=TireModel(args.wheel_radius_in*INCH_TO_METRE)
    driver=DriverModel(); settings=SimulationSettings(maximum_time_s=max(180.0,track.length_m/4.0),integration_step_s=args.integration_step_s,report_step_s=args.report_step_s)
    payloads=[(p,v,track,engine,vehicle,cvt,tire,driver,settings) for p in args.parameters for v in DEFAULT_LEVELS[p]]
    rows=[]
    if args.workers==1:
        for i,payload in enumerate(payloads,1):
            rows.append(_run(payload)); print(f"[{i:02d}/{len(payloads):02d}] {payload[0]}={payload[1]:.3f}",flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(_run,p) for p in payloads]
            for i,future in enumerate(as_completed(futures),1):
                rows.append(future.result()); print(f"[{i:02d}/{len(payloads):02d}] complete",flush=True)
    rows.sort(key=lambda r:(r["parameter"],r["value"]))
    _write_csv(args.output_dir/"structural_sensitivity.csv",rows)
    plots=_plot(rows,args.output_dir,not args.no_show)
    manifest={"track":track.name,"base_case":{"cvt_minimum_ratio":0.9,"cvt_maximum_ratio":3.5,"final_drive_ratio":7.556,"wheel_radius_in":STANDARD_WHEEL_RADIUS_IN},"levels":{k:list(v) for k,v in DEFAULT_LEVELS.items()},"directional_interpretation":DIRECTION_NOTES,"rows":rows}
    (args.output_dir/"structural_sensitivity.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    report=["# Structural sensitivity report","",f"Base case: CVT 3.5–0.9, final drive 7.556, 22 in tire diameter (11 in wheel radius).","", "Each line is a paired bounded/infinite-CVT comparison. Only one structural assumption changes at a time.",""]
    report += [f"- **{LABELS[k]}:** {v}" for k,v in DIRECTION_NOTES.items()]
    (args.output_dir/"README.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(f"Wrote {len(rows)} cases and {len(plots)} plots to {args.output_dir}")

if __name__ == "__main__": main()
