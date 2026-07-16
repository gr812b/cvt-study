"""Fixed-step endurance-lap integrator with an implicit tire-slip coordinate."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from math import tanh
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .dynamics import evaluate_dynamics
from .models import RAD_PER_SECOND_TO_RPM, SimulationSettings, StudyCase
from .track import RuntimeTrack

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SimulationTrace:
    case_name: str
    track_name: str
    completed: bool
    termination_reason: str
    numeric: Mapping[str, FloatArray]
    text: Mapping[str, tuple[str, ...]]
    integrals: Mapping[str, float]
    feature_entry_speeds_mps: Mapping[str, float]
    feature_obstacle_energy_j: Mapping[str, float]

    @property
    def final_time_s(self) -> float:
        return float(self.numeric["time_s"][-1])

    @property
    def final_distance_m(self) -> float:
        return float(self.numeric["distance_m"][-1])

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        numeric_keys = list(self.numeric)
        text_keys = list(self.text)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([*numeric_keys, *text_keys])
            for index in range(len(self.numeric["time_s"])):
                writer.writerow(
                    [
                        *(float(self.numeric[key][index]) for key in numeric_keys),
                        *(self.text[key][index] for key in text_keys),
                    ]
                )


def implicit_tire_slip_step(
    *,
    previous_slip_speed_mps: float,
    step_s: float,
    free_slip_acceleration_mps2: float,
    tire_force_coefficient: float,
    tire_limit_n: float,
    tire_stiffness_n_per_mps: float,
) -> tuple[float, float]:
    """Backward-Euler update for the stiff wheel/vehicle slip coordinate.

    The tanh law is monotone, so the residual is strictly increasing.  The force
    bound gives an exact bracket; deterministic bisection avoids Newton jumps
    between saturated branches and requires no solver-specific tolerances.
    """

    if tire_limit_n <= 0.0:
        return (
            previous_slip_speed_mps + step_s * free_slip_acceleration_mps2,
            0.0,
        )
    free_update = previous_slip_speed_mps + step_s * free_slip_acceleration_mps2
    span = step_s * tire_force_coefficient * tire_limit_n
    low = free_update - span
    high = free_update + span

    def residual(value: float) -> float:
        force = tire_limit_n * tanh(
            tire_stiffness_n_per_mps * value / tire_limit_n
        )
        return value - previous_slip_speed_mps - step_s * (
            free_slip_acceleration_mps2 - tire_force_coefficient * force
        )

    for _ in range(24):
        middle = 0.5 * (low + high)
        if residual(middle) > 0.0:
            high = middle
        else:
            low = middle
    slip = 0.5 * (low + high)
    force = tire_limit_n * tanh(
        tire_stiffness_n_per_mps * slip / tire_limit_n
    )
    return float(slip), float(force)


def run_simulation(
    *, case: StudyCase, track: RuntimeTrack, settings: SimulationSettings
) -> SimulationTrace:
    step = settings.integration_step_s
    time_s = 0.0
    distance = 0.0
    speed = settings.initial_vehicle_speed_mps
    wheel_speed = settings.initial_wheel_speed_rad_s
    slip = case.vehicle.wheel_radius_m * wheel_speed - speed
    internal: list[tuple[float, float, float, float]] = [(time_s, distance, speed, wheel_speed)]
    integrals = {
        "engine_energy_j": 0.0,
        "transmitted_energy_j": 0.0,
        "drivetrain_loss_energy_j": 0.0,
        "clutch_loss_energy_j": 0.0,
        "operating_shortfall_energy_j": 0.0,
        "tire_slip_loss_energy_j": 0.0,
        "brake_loss_energy_j": 0.0,
        "rolling_loss_energy_j": 0.0,
        "aerodynamic_loss_energy_j": 0.0,
        "obstacle_loss_energy_j": 0.0,
        "grade_work_j": 0.0,
    }
    feature_entry_speeds: dict[str, float] = {}
    feature_obstacle_energy = {feature.identifier: 0.0 for feature in track.features}
    completed = False
    reason = "maximum_time_reached"

    for _ in range(int(np.ceil(settings.maximum_time_s / step))):
        for feature in track.features:
            if (
                feature.identifier not in feature_entry_speeds
                and feature.interval.local_distance(distance, track.length_m) is not None
            ):
                feature_entry_speeds[feature.identifier] = speed
        dynamics = evaluate_dynamics(
            distance_m=distance,
            vehicle_speed_mps=speed,
            wheel_speed_rad_s=wheel_speed,
            case=case,
            track=track,
            feature_entry_speeds_mps=feature_entry_speeds,
        )
        road_resistance = (
            dynamics.grade_force_n
            + dynamics.rolling_force_n
            + dynamics.aerodynamic_force_n
            + dynamics.obstacle_force_n
        )
        free_slip_acceleration = (
            case.vehicle.wheel_radius_m
            * (dynamics.powertrain.wheel_drive_torque_nm - dynamics.brake_torque_nm)
            / case.vehicle.wheel_rotational_inertia_kg_m2
            + road_resistance / case.vehicle.mass_kg
        )
        coefficient = (
            case.vehicle.wheel_radius_m**2
            / case.vehicle.wheel_rotational_inertia_kg_m2
            + 1.0 / case.vehicle.mass_kg
        )
        new_slip, new_tire_force = implicit_tire_slip_step(
            previous_slip_speed_mps=slip,
            step_s=step,
            free_slip_acceleration_mps2=free_slip_acceleration,
            tire_force_coefficient=coefficient,
            tire_limit_n=dynamics.tire_limit_n,
            tire_stiffness_n_per_mps=case.tire.slip_stiffness_n_per_mps,
        )
        new_speed = max(
            0.0,
            speed + step * (new_tire_force - road_resistance) / case.vehicle.mass_kg,
        )
        new_wheel_speed = max(
            0.0, (new_slip + new_speed) / case.vehicle.wheel_radius_m
        )
        # Enforcing the non-negative wheel-speed constraint changes the solved
        # state.  Reconstruct slip from the constrained physical speeds rather
        # than carrying the unconstrained root into the next step.  Without
        # this projection, a locked wheel can accumulate an impossible
        # hundreds-of-metres-per-second slip state and remain numerically stuck.
        new_slip = case.vehicle.wheel_radius_m * new_wheel_speed - new_speed
        new_distance = distance + 0.5 * step * (speed + new_speed)
        new_time = time_s + step
        if new_distance >= track.length_m:
            fraction = (track.length_m - distance) / max(new_distance - distance, 1.0e-12)
            new_time = time_s + fraction * step
            new_distance = track.length_m
            new_speed = speed + fraction * (new_speed - speed)
            new_slip = slip + fraction * (new_slip - slip)
            new_wheel_speed = max(
                0.0, (new_slip + new_speed) / case.vehicle.wheel_radius_m
            )
            new_slip = (
                case.vehicle.wheel_radius_m * new_wheel_speed - new_speed
            )
            completed = True
            reason = "track_complete"

        _record_feature_entry_crossings(
            features=track.features,
            recorded=feature_entry_speeds,
            start_distance_m=distance,
            end_distance_m=new_distance,
            start_speed_mps=speed,
            end_speed_mps=new_speed,
        )

        actual_step = new_time - time_s
        average_speed = 0.5 * (speed + new_speed)
        average_wheel_speed = 0.5 * (wheel_speed + new_wheel_speed)
        drive_work = (
            dynamics.powertrain.wheel_drive_torque_nm
            * average_wheel_speed
            * actual_step
        )
        tire_slip_work = (
            new_tire_force
            * (case.vehicle.wheel_radius_m * average_wheel_speed - average_speed)
            * actual_step
        )
        integrals["engine_energy_j"] += dynamics.powertrain.engine_power_w * actual_step
        integrals["transmitted_energy_j"] += max(0.0, drive_work)
        integrals["drivetrain_loss_energy_j"] += (
            dynamics.powertrain.drivetrain_loss_power_w * actual_step
        )
        if dynamics.powertrain.clutch_loss_power_w > 0.0:
            integrals["clutch_loss_energy_j"] += max(
                0.0,
                dynamics.powertrain.engine_power_w * case.cvt.efficiency * actual_step
                - drive_work,
            )
        integrals["operating_shortfall_energy_j"] += (
            dynamics.powertrain.operating_shortfall_power_w * actual_step
        )
        integrals["tire_slip_loss_energy_j"] += max(0.0, tire_slip_work)
        integrals["brake_loss_energy_j"] += max(
            0.0, dynamics.brake_torque_nm * average_wheel_speed * actual_step
        )
        distance_step = average_speed * actual_step
        integrals["rolling_loss_energy_j"] += max(
            0.0, dynamics.rolling_force_n * distance_step
        )
        integrals["aerodynamic_loss_energy_j"] += max(
            0.0, dynamics.aerodynamic_force_n * distance_step
        )
        integrals["obstacle_loss_energy_j"] += max(
            0.0, dynamics.obstacle_force_n * distance_step
        )
        for feature_id, force_n in dynamics.obstacle_feature_forces_n:
            feature_obstacle_energy[feature_id] += max(0.0, force_n * distance_step)
        integrals["grade_work_j"] += dynamics.grade_force_n * distance_step

        time_s, distance, speed, wheel_speed, slip = (
            new_time,
            new_distance,
            new_speed,
            new_wheel_speed,
            new_slip,
        )
        internal.append((time_s, distance, speed, wheel_speed))
        if completed:
            break

    array = np.asarray(internal, dtype=float)
    report_times = np.arange(0.0, array[-1, 0], settings.report_step_s, dtype=float)
    if report_times.size == 0 or report_times[-1] < array[-1, 0]:
        report_times = np.append(report_times, array[-1, 0])
    # Retain exact gate crossings in the public trace.  Gate compliance must be
    # evaluated from the integration path, not inferred across a coarser report
    # interval immediately after a discontinuous gate-ceiling release.
    gate_positions = np.asarray(
        [gate.position_s_m for gate in track.speed_gates], dtype=float
    )
    if gate_positions.size:
        valid_positions = gate_positions[
            (gate_positions >= array[0, 1]) & (gate_positions <= array[-1, 1])
        ]
        gate_times = np.interp(valid_positions, array[:, 1], array[:, 0])
        report_times = np.unique(np.concatenate((report_times, gate_times)))
    distances = np.interp(report_times, array[:, 0], array[:, 1])
    speeds = np.interp(report_times, array[:, 0], array[:, 2])
    wheel_speeds = np.interp(report_times, array[:, 0], array[:, 3])
    return _report_trace(
        case=case,
        track=track,
        settings=settings,
        completed=completed,
        reason=reason,
        times=report_times,
        distances=distances,
        speeds=speeds,
        wheel_speeds=wheel_speeds,
        integrals=integrals,
        feature_entry_speeds_mps=feature_entry_speeds,
        feature_obstacle_energy_j=feature_obstacle_energy,
    )


def _record_feature_entry_crossings(
    *,
    features: tuple[Any, ...],
    recorded: dict[str, float],
    start_distance_m: float,
    end_distance_m: float,
    start_speed_mps: float,
    end_speed_mps: float,
) -> None:
    """Capture interpolated simulated speed at the physical feature boundary."""

    span = end_distance_m - start_distance_m
    if span <= 0.0:
        return
    for feature in features:
        if feature.identifier in recorded:
            continue
        boundary = float(feature.interval.start_s_m)
        if start_distance_m < boundary <= end_distance_m:
            fraction = (boundary - start_distance_m) / span
            recorded[feature.identifier] = max(
                0.0,
                float(start_speed_mps)
                + fraction * (float(end_speed_mps) - float(start_speed_mps)),
            )


def _report_trace(
    *,
    case: StudyCase,
    track: RuntimeTrack,
    settings: SimulationSettings,
    completed: bool,
    reason: str,
    times: np.ndarray,
    distances: np.ndarray,
    speeds: np.ndarray,
    wheel_speeds: np.ndarray,
    integrals: Mapping[str, float],
    feature_entry_speeds_mps: Mapping[str, float],
    feature_obstacle_energy_j: Mapping[str, float],
) -> SimulationTrace:
    keys = (
        "time_s", "distance_m", "vehicle_speed_mps", "vehicle_speed_kmh",
        "vehicle_acceleration_mps2", "wheel_speed_rad_s", "wheel_speed_rpm",
        "wheel_patch_speed_mps", "tire_slip_speed_mps", "tire_force_n",
        "tire_limit_n", "tire_utilization", "reference_elevation_m",
        "modeled_elevation_offset_m", "modeled_grade_degrees", "curvature_1_per_m",
        "friction_coefficient", "normal_load_scale", "target_speed_mps", "throttle",
        "brake_force_command_n", "brake_torque_nm", "cvt_ratio", "ratio_required",
        "engine_speed_rpm", "engine_torque_nm", "engine_power_w", "transmitted_power_w",
        "wheel_drive_torque_nm", "drivetrain_loss_power_w", "clutch_loss_power_w",
        "operating_shortfall_power_w",
        "tire_slip_loss_power_w", "brake_loss_power_w", "rolling_loss_power_w",
        "aerodynamic_loss_power_w", "obstacle_loss_power_w", "grade_power_w",
        "grade_force_n", "rolling_force_n", "aerodynamic_force_n", "obstacle_force_n",
        "lateral_force_n", "normal_load_n", "vehicle_kinetic_energy_j",
        "wheel_kinetic_energy_j", "total_kinetic_energy_j",
    )
    numeric: dict[str, list[float]] = {key: [] for key in keys}
    text: dict[str, list[str]] = {
        "cvt_mode": [], "active_feature_ids": [], "active_feature_names": []
    }
    for t, s, v, omega in zip(times, distances, speeds, wheel_speeds):
        d = evaluate_dynamics(
            distance_m=float(s), vehicle_speed_mps=float(v), wheel_speed_rad_s=float(omega),
            case=case, track=track,
            feature_entry_speeds_mps=feature_entry_speeds_mps,
        )
        target = d.driver.target_speed_mps
        vehicle_ke = 0.5 * case.vehicle.mass_kg * v**2
        wheel_ke = 0.5 * case.vehicle.wheel_rotational_inertia_kg_m2 * omega**2
        values = {
            "time_s": t, "distance_m": s, "vehicle_speed_mps": v,
            "vehicle_speed_kmh": 3.6 * v,
            "vehicle_acceleration_mps2": d.vehicle_acceleration_mps2,
            "wheel_speed_rad_s": omega, "wheel_speed_rpm": omega * RAD_PER_SECOND_TO_RPM,
            "wheel_patch_speed_mps": case.vehicle.wheel_radius_m * omega,
            "tire_slip_speed_mps": d.tire_slip_speed_mps, "tire_force_n": d.tire_force_n,
            "tire_limit_n": d.tire_limit_n, "tire_utilization": d.tire_utilization,
            "reference_elevation_m": np.nan if d.track.reference_elevation_m is None else d.track.reference_elevation_m,
            "modeled_elevation_offset_m": d.track.modeled_elevation_offset_m,
            "modeled_grade_degrees": d.track.modeled_grade_degrees,
            "curvature_1_per_m": d.track.curvature_1_per_m,
            "friction_coefficient": d.track.friction_coefficient,
            "normal_load_scale": d.track.normal_load_scale,
            "target_speed_mps": np.nan if not np.isfinite(target) else target,
            "throttle": d.driver.throttle, "brake_force_command_n": d.driver.brake_force_n,
            "brake_torque_nm": d.brake_torque_nm, "cvt_ratio": d.powertrain.cvt_ratio,
            "ratio_required": d.powertrain.ratio_required,
            "engine_speed_rpm": d.powertrain.engine_speed_rpm,
            "engine_torque_nm": d.powertrain.engine_torque_nm,
            "engine_power_w": d.powertrain.engine_power_w,
            "transmitted_power_w": d.powertrain.transmitted_power_w,
            "wheel_drive_torque_nm": d.powertrain.wheel_drive_torque_nm,
            "drivetrain_loss_power_w": d.powertrain.drivetrain_loss_power_w,
            "clutch_loss_power_w": d.powertrain.clutch_loss_power_w,
            "operating_shortfall_power_w": d.powertrain.operating_shortfall_power_w,
            "tire_slip_loss_power_w": d.tire_slip_loss_power_w,
            "brake_loss_power_w": d.brake_loss_power_w,
            "rolling_loss_power_w": d.rolling_loss_power_w,
            "aerodynamic_loss_power_w": d.aerodynamic_loss_power_w,
            "obstacle_loss_power_w": d.obstacle_loss_power_w,
            "grade_power_w": d.grade_power_w, "grade_force_n": d.grade_force_n,
            "rolling_force_n": d.rolling_force_n,
            "aerodynamic_force_n": d.aerodynamic_force_n,
            "obstacle_force_n": d.obstacle_force_n,
            "lateral_force_n": d.lateral_force_n, "normal_load_n": d.normal_load_n,
            "vehicle_kinetic_energy_j": vehicle_ke, "wheel_kinetic_energy_j": wheel_ke,
            "total_kinetic_energy_j": vehicle_ke + wheel_ke,
        }
        for key in keys:
            numeric[key].append(float(values[key]))
        text["cvt_mode"].append(d.powertrain.mode)
        text["active_feature_ids"].append(" | ".join(d.track.active_feature_ids))
        text["active_feature_names"].append(" | ".join(d.track.active_feature_names))
    return SimulationTrace(
        case_name=case.name, track_name=track.name, completed=completed,
        termination_reason=reason,
        numeric={key: np.asarray(value, dtype=float) for key, value in numeric.items()},
        text={key: tuple(value) for key, value in text.items()},
        integrals={key: float(value) for key, value in integrals.items()},
        feature_entry_speeds_mps={
            key: float(value) for key, value in feature_entry_speeds_mps.items()
        },
        feature_obstacle_energy_j={
            key: float(value) for key, value in feature_obstacle_energy_j.items()
        },
    )
