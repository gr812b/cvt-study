"""Energy accounting and bounded-versus-infinite decision metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .integrator import SimulationTrace


def _integral(values: np.ndarray, time_s: np.ndarray) -> float:
    if len(time_s) < 2:
        return 0.0
    return float(np.trapezoid(values, time_s))


def _time(mask: np.ndarray, time_s: np.ndarray) -> float:
    return _integral(mask.astype(float), time_s)


def summarize_trace(
    trace: SimulationTrace, *, target_engine_rpm: float, target_power_w: float
) -> dict[str, Any]:
    n = trace.numeric
    t = n["time_s"]
    speed = n["vehicle_speed_mps"]
    modes = np.asarray(trace.text["cvt_mode"], dtype=object)
    throttle = n["throttle"]
    braking = n["brake_force_command_n"] > 1.0
    positive_demand = (throttle > 0.05) & ~braking
    maximum_ratio = np.asarray(
        [str(mode).startswith("maximum_ratio") for mode in modes], dtype=bool
    )
    variable = modes == "variable_target_rpm"
    minimum_ratio = modes == "minimum_ratio_synchronous"

    integrated = trace.integrals
    energies = {
        "engine_energy_kj": integrated["engine_energy_j"] / 1000.0,
        "transmitted_energy_kj": integrated["transmitted_energy_j"] / 1000.0,
        "drivetrain_loss_energy_kj": integrated["drivetrain_loss_energy_j"] / 1000.0,
        "clutch_loss_energy_kj": integrated["clutch_loss_energy_j"] / 1000.0,
        "engine_operating_shortfall_energy_kj": integrated[
            "operating_shortfall_energy_j"
        ] / 1000.0,
        "tire_slip_loss_energy_kj": integrated["tire_slip_loss_energy_j"] / 1000.0,
        "brake_loss_energy_kj": integrated["brake_loss_energy_j"] / 1000.0,
        "rolling_loss_energy_kj": integrated["rolling_loss_energy_j"] / 1000.0,
        "aerodynamic_loss_energy_kj": integrated[
            "aerodynamic_loss_energy_j"
        ] / 1000.0,
        "obstacle_loss_energy_kj": integrated["obstacle_loss_energy_j"] / 1000.0,
        "net_grade_work_kj": integrated["grade_work_j"] / 1000.0,
    }
    energies["finite_ratio_opportunity_loss_energy_kj"] = (
        energies["clutch_loss_energy_kj"]
        + energies["engine_operating_shortfall_energy_kj"]
    )
    powertrain_residual_j = 1000.0 * (
        energies["engine_energy_kj"]
        - energies["transmitted_energy_kj"]
        - energies["drivetrain_loss_energy_kj"]
        - energies["clutch_loss_energy_kj"]
    )
    initial_ke = float(n["total_kinetic_energy_j"][0])
    final_ke = float(n["total_kinetic_energy_j"][-1])
    transmitted = energies["transmitted_energy_kj"] * 1000.0
    residual = (
        transmitted
        + initial_ke
        - final_ke
        - 1000.0
        * (
            energies["tire_slip_loss_energy_kj"]
            + energies["brake_loss_energy_kj"]
            + energies["rolling_loss_energy_kj"]
            + energies["aerodynamic_loss_energy_kj"]
            + energies["obstacle_loss_energy_kj"]
            + energies["net_grade_work_kj"]
        )
    )
    scale = max(abs(transmitted) + initial_ke + abs(energies["net_grade_work_kj"] * 1000.0), 1.0)
    summary: dict[str, Any] = {
        "case": trace.case_name,
        "track": trace.track_name,
        "completed": trace.completed,
        "termination_reason": trace.termination_reason,
        "lap_time_s": trace.final_time_s,
        "distance_m": trace.final_distance_m,
        "average_speed_kmh": 3.6 * trace.final_distance_m / max(trace.final_time_s, 1e-12),
        "maximum_speed_kmh": 3.6 * float(np.max(speed)),
        "minimum_engine_rpm": float(np.min(n["engine_speed_rpm"])),
        "maximum_engine_rpm": float(np.max(n["engine_speed_rpm"])),
        "time_maximum_ratio_s": _time(maximum_ratio, t),
        "time_variable_ratio_s": _time(variable, t),
        "time_minimum_ratio_s": _time(minimum_ratio, t),
        "positive_demand_time_maximum_ratio_s": _time(positive_demand & maximum_ratio, t),
        "positive_demand_time_variable_ratio_s": _time(positive_demand & variable, t),
        "positive_demand_time_minimum_ratio_s": _time(positive_demand & minimum_ratio, t),
        "time_braking_s": _time(braking, t),
        "time_traction_limited_s": _time(n["tire_utilization"] >= 0.95, t),
        "maximum_abs_tire_slip_speed_mps": float(np.max(np.abs(n["tire_slip_speed_mps"]))),
        "initial_total_kinetic_energy_kj": initial_ke / 1000.0,
        "final_total_kinetic_energy_kj": final_ke / 1000.0,
        "energy_balance_residual_kj": residual / 1000.0,
        "energy_balance_relative_error": residual / scale,
        "powertrain_energy_balance_residual_kj": powertrain_residual_j / 1000.0,
        "powertrain_energy_balance_relative_error": powertrain_residual_j / max(
            integrated["engine_energy_j"], 1.0
        ),
        "target_engine_rpm": target_engine_rpm,
        "target_power_w": target_power_w,
        "mode_sample_counts": dict(Counter(str(mode) for mode in modes)),
        "obstacle_energy_by_feature_kj": {
            key: value / 1000.0
            for key, value in trace.feature_obstacle_energy_j.items()
        },
        "feature_entry_speeds_mps": dict(trace.feature_entry_speeds_mps),
        **energies,
    }
    return summary


def compare_summaries(
    bounded: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    time_penalty = float(bounded["lap_time_s"]) - float(reference["lap_time_s"])
    dominance = (
        bool(reference["completed"])
        and (
            not bool(bounded["completed"])
            or float(reference["lap_time_s"]) <= float(bounded["lap_time_s"]) + 1e-9
        )
    )
    return {
        "bounded_lap_time_s": bounded["lap_time_s"],
        "infinite_reference_lap_time_s": reference["lap_time_s"],
        "lap_time_penalty_vs_infinite_s": time_penalty,
        "finite_ratio_opportunity_loss_energy_kj": max(
            0.0,
            float(bounded["finite_ratio_opportunity_loss_energy_kj"])
            - float(reference["finite_ratio_opportunity_loss_energy_kj"]),
        ),
        "bounded_total_opportunity_loss_energy_kj": bounded[
            "finite_ratio_opportunity_loss_energy_kj"
        ],
        "reference_shared_launch_loss_energy_kj": reference[
            "finite_ratio_opportunity_loss_energy_kj"
        ],
        "reference_dominance_pass": dominance,
        "bounded_energy_balance_relative_error": bounded[
            "energy_balance_relative_error"
        ],
        "reference_energy_balance_relative_error": reference[
            "energy_balance_relative_error"
        ],
    }


def gate_compliance_rows(
    trace: SimulationTrace, track: Any
) -> list[dict[str, Any]]:
    distance = trace.numeric["distance_m"]
    speed = trace.numeric["vehicle_speed_mps"]
    rows: list[dict[str, Any]] = []
    for gate in track.speed_gates:
        simulated = float(np.interp(gate.position_s_m, distance, speed))
        excess_kmh = 3.6 * (simulated - gate.target_speed_mps)
        rows.append(
            {
                "gate_id": gate.identifier,
                "response_group_id": gate.response_group_id,
                "name": gate.name,
                "position_s_m": gate.position_s_m,
                "target_speed_mps": gate.target_speed_mps,
                "simulated_speed_mps": simulated,
                "excess_over_ceiling_kmh": excess_kmh,
                "compliant_within_0p5_kmh": excess_kmh <= 0.5,
            }
        )
    return rows
