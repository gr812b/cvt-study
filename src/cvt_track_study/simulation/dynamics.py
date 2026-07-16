"""Longitudinal vehicle, wheel, tire, driver, and resistance equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from math import cos, isfinite, sin, tanh


from .models import DriverModel, StudyCase
from .powertrain import PowertrainSample, evaluate_powertrain
from .track import RuntimeTrack, TrackSample


@dataclass(frozen=True, slots=True)
class DriverCommand:
    throttle: float
    brake_force_n: float
    target_speed_mps: float


@dataclass(frozen=True, slots=True)
class DynamicsSample:
    driver: DriverCommand
    track: TrackSample
    powertrain: PowertrainSample
    tire_force_n: float
    tire_limit_n: float
    tire_utilization: float
    tire_slip_speed_mps: float
    brake_torque_nm: float
    grade_force_n: float
    rolling_force_n: float
    aerodynamic_force_n: float
    obstacle_force_n: float
    obstacle_feature_forces_n: tuple[tuple[str, float], ...]
    lateral_force_n: float
    normal_load_n: float
    vehicle_acceleration_mps2: float
    wheel_acceleration_rad_s2: float
    tire_slip_loss_power_w: float
    brake_loss_power_w: float
    rolling_loss_power_w: float
    aerodynamic_loss_power_w: float
    obstacle_loss_power_w: float
    grade_power_w: float


def driver_command(
    *,
    track: RuntimeTrack,
    distance_m: float,
    vehicle_speed_mps: float,
    driver: DriverModel,
    braking_deceleration_mps2: float,
) -> DriverCommand:
    target = track.safe_speed_ceiling_mps(
        distance_m,
        braking_deceleration_mps2=braking_deceleration_mps2,
    )
    if not isfinite(target):
        return DriverCommand(throttle=1.0, brake_force_n=0.0, target_speed_mps=target)
    # The kinematic envelope is the speed from which the declared effective
    # braking deceleration must begin immediately.  A proportional brake
    # controller systematically undershoots that required deceleration and
    # therefore misses the gate.  The explicit margin starts the full-brake
    # branch slightly before the mathematical envelope and prevents chatter.
    if vehicle_speed_mps >= target - driver.braking_trigger_margin_mps:
        return DriverCommand(
            throttle=0.0,
            brake_force_n=driver.maximum_brake_force_n,
            target_speed_mps=target,
        )
    return DriverCommand(throttle=1.0, brake_force_n=0.0, target_speed_mps=target)


def evaluate_dynamics(
    *,
    distance_m: float,
    vehicle_speed_mps: float,
    wheel_speed_rad_s: float,
    case: StudyCase,
    track: RuntimeTrack,
    feature_entry_speeds_mps: Mapping[str, float] | None = None,
) -> DynamicsSample:
    vehicle_speed = max(0.0, float(vehicle_speed_mps))
    wheel_speed = max(0.0, float(wheel_speed_rad_s))
    vehicle = case.vehicle
    sample = track.sample(
        distance_m,
        vehicle_speed_mps=vehicle_speed,
        vehicle_mass_kg=vehicle.mass_kg,
        gravity_mps2=vehicle.gravity_mps2,
        feature_entry_speeds_mps=feature_entry_speeds_mps,
    )
    straight_line_tire_brake_limit_n = (
        sample.friction_coefficient
        * case.tire.peak_traction_scale
        * vehicle.driven_normal_load_fraction
        * vehicle.mass_kg
        * vehicle.gravity_mps2
    )
    effective_braking_deceleration = min(
        case.driver.maximum_braking_deceleration_mps2,
        case.driver.maximum_brake_force_n / vehicle.mass_kg,
        straight_line_tire_brake_limit_n / vehicle.mass_kg,
    )
    command = driver_command(
        track=track,
        distance_m=distance_m,
        vehicle_speed_mps=vehicle_speed,
        driver=case.driver,
        braking_deceleration_mps2=max(effective_braking_deceleration, 1.0e-9),
    )
    powertrain = evaluate_powertrain(
        wheel_speed_rad_s=wheel_speed,
        throttle=command.throttle,
        engine=case.engine,
        cvt=case.cvt,
        infinite_cvt=case.infinite_cvt,
    )

    grade = sample.modeled_grade_radians
    total_normal = max(
        0.0,
        vehicle.mass_kg
        * vehicle.gravity_mps2
        * max(cos(grade), 0.0)
        * sample.normal_load_scale,
    )
    lateral_force = abs(vehicle.mass_kg * vehicle_speed**2 * sample.curvature_1_per_m)
    effective_mu = sample.friction_coefficient * case.tire.peak_traction_scale
    driven_limit = effective_mu * vehicle.driven_normal_load_fraction * total_normal
    # Curvature and lateral demand are retained as diagnostics, but the Phase 5
    # longitudinal host does not spend tire capacity on an unvalidated lateral
    # model.  Driver-limited corner entry is represented by measured speed gates.
    tire_limit = driven_limit
    slip_speed = vehicle.wheel_radius_m * wheel_speed - vehicle_speed
    if tire_limit > 0.0:
        tire_force = tire_limit * tanh(
            case.tire.slip_stiffness_n_per_mps * slip_speed / tire_limit
        )
        utilization = abs(tire_force) / tire_limit
    else:
        tire_force = 0.0
        utilization = 0.0

    grade_force = vehicle.mass_kg * vehicle.gravity_mps2 * sin(grade)
    rolling_force = (
        vehicle.rolling_resistance_coefficient
        * total_normal
        * tanh(vehicle_speed / 0.1)
    )
    aerodynamic_force = 0.5 * vehicle.air_density_kg_m3 * vehicle.drag_area_m2 * vehicle_speed**2
    obstacle_force = sample.obstacle_resistance_force_n
    brake_torque = command.brake_force_n * vehicle.wheel_radius_m
    vehicle_acceleration = (
        tire_force - grade_force - rolling_force - aerodynamic_force - obstacle_force
    ) / vehicle.mass_kg
    wheel_acceleration = (
        powertrain.wheel_drive_torque_nm - brake_torque - tire_force * vehicle.wheel_radius_m
    ) / vehicle.wheel_rotational_inertia_kg_m2
    if vehicle_speed <= 0.0 and vehicle_acceleration < 0.0:
        vehicle_acceleration = 0.0
    if wheel_speed <= 0.0 and wheel_acceleration < 0.0:
        wheel_acceleration = 0.0

    return DynamicsSample(
        driver=command,
        track=sample,
        powertrain=powertrain,
        tire_force_n=float(tire_force),
        tire_limit_n=float(tire_limit),
        tire_utilization=float(utilization),
        tire_slip_speed_mps=float(slip_speed),
        brake_torque_nm=float(brake_torque),
        grade_force_n=float(grade_force),
        rolling_force_n=float(rolling_force),
        aerodynamic_force_n=float(aerodynamic_force),
        obstacle_force_n=float(obstacle_force),
        obstacle_feature_forces_n=sample.feature_resistance_forces_n,
        lateral_force_n=float(lateral_force),
        normal_load_n=float(total_normal),
        vehicle_acceleration_mps2=float(vehicle_acceleration),
        wheel_acceleration_rad_s2=float(wheel_acceleration),
        tire_slip_loss_power_w=max(0.0, float(tire_force * slip_speed)),
        brake_loss_power_w=max(0.0, float(brake_torque * wheel_speed)),
        rolling_loss_power_w=max(0.0, float(rolling_force * vehicle_speed)),
        aerodynamic_loss_power_w=max(0.0, float(aerodynamic_force * vehicle_speed)),
        obstacle_loss_power_w=max(0.0, float(obstacle_force * vehicle_speed)),
        grade_power_w=float(grade_force * vehicle_speed),
    )
