from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Iterable

import numpy as np

RPM_TO_RAD_PER_SECOND = 2.0 * pi / 60.0
RAD_PER_SECOND_TO_RPM = 60.0 / (2.0 * pi)
INCH_TO_METRE = 0.0254
FOOT_POUND_TO_NEWTON_METRE = 1.3558179483


@dataclass(frozen=True, slots=True)
class EngineTorquePoint:
    rpm: float
    torque_nm: float

    def __post_init__(self) -> None:
        if not isfinite(self.rpm) or self.rpm < 0.0:
            raise ValueError("rpm must be finite and non-negative")
        if not isfinite(self.torque_nm):
            raise ValueError("torque_nm must be finite")


@dataclass(frozen=True, slots=True)
class EngineModel:
    points: tuple[EngineTorquePoint, ...]
    target_rpm: float | None = None

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("at least two engine torque points are required")
        rpms = [p.rpm for p in self.points]
        if rpms != sorted(rpms) or len(set(rpms)) != len(rpms):
            raise ValueError("engine points must have strictly increasing rpm")
        if self.target_rpm is not None and (not isfinite(self.target_rpm) or self.target_rpm <= 0.0):
            raise ValueError("target_rpm must be positive and finite when provided")

    @property
    def rpm_values(self) -> np.ndarray:
        return np.asarray([point.rpm for point in self.points], dtype=float)

    @property
    def torque_values_nm(self) -> np.ndarray:
        return np.asarray([point.torque_nm for point in self.points], dtype=float)

    def torque_nm(self, rpm: float) -> float:
        return float(
            np.interp(
                float(rpm),
                self.rpm_values,
                self.torque_values_nm,
                left=0.0,
                right=0.0,
            )
        )

    def power_w(self, rpm: float) -> float:
        return self.torque_nm(rpm) * rpm * RPM_TO_RAD_PER_SECOND

    @property
    def peak_power_rpm(self) -> float:
        if self.target_rpm is not None:
            return float(self.target_rpm)
        dense_rpm = np.linspace(self.points[0].rpm, self.points[-1].rpm, 4001)
        powers = np.asarray([self.power_w(float(rpm)) for rpm in dense_rpm])
        return float(dense_rpm[int(np.argmax(powers))])

    @property
    def peak_power_w(self) -> float:
        return self.power_w(self.peak_power_rpm)

    @classmethod
    def baja_br10(cls) -> "EngineModel":
        # Same full-throttle points used by the existing launch tools.
        raw = (
            (1000.0, 0.0),
            (1800.0, 18.0),
            (2400.0, 18.5),
            (2600.0, 18.1),
            (2800.0, 17.4),
            (3000.0, 16.6),
            (3200.0, 15.4),
            (3400.0, 14.5),
            (3600.0, 13.5),
            (4000.0, 0.0),
        )
        return cls(
            points=tuple(
                EngineTorquePoint(rpm=rpm, torque_nm=torque_ft_lb * FOOT_POUND_TO_NEWTON_METRE)
                for rpm, torque_ft_lb in raw
            ),
            target_rpm=3000.0,
        )


@dataclass(frozen=True, slots=True)
class VehicleModel:
    mass_kg: float = 300.0
    wheel_rotational_inertia_kg_m2: float = 0.2
    driven_normal_load_fraction: float = 1.0
    drag_coefficient: float = 0.6
    frontal_area_m2: float = 1.11484
    air_density_kg_m3: float = 1.225
    gravity_mps2: float = 9.80665

    def __post_init__(self) -> None:
        positive = {
            "mass_kg": self.mass_kg,
            "wheel_rotational_inertia_kg_m2": self.wheel_rotational_inertia_kg_m2,
            "frontal_area_m2": self.frontal_area_m2,
            "air_density_kg_m3": self.air_density_kg_m3,
            "gravity_mps2": self.gravity_mps2,
        }
        for name, value in positive.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < self.driven_normal_load_fraction <= 1.0:
            raise ValueError("driven_normal_load_fraction must lie in (0, 1]")
        if not isfinite(self.drag_coefficient) or self.drag_coefficient < 0.0:
            raise ValueError("drag_coefficient must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TireModel:
    """Reduced-order longitudinal tire model.

    ``peak_traction_scale`` multiplies the track friction coefficient and controls
    the maximum available longitudinal force. ``slip_stiffness_n_per_mps``
    controls how quickly force builds with tire-surface slip speed.
    """

    wheel_radius_m: float = 11.0 * INCH_TO_METRE
    peak_traction_scale: float = 1.0
    slip_stiffness_n_per_mps: float = 2500.0

    def __post_init__(self) -> None:
        if not isfinite(self.wheel_radius_m) or self.wheel_radius_m <= 0.0:
            raise ValueError("wheel_radius_m must be positive and finite")
        if not isfinite(self.peak_traction_scale) or self.peak_traction_scale <= 0.0:
            raise ValueError("peak_traction_scale must be positive and finite")
        if not isfinite(self.slip_stiffness_n_per_mps) or self.slip_stiffness_n_per_mps <= 0.0:
            raise ValueError("slip_stiffness_n_per_mps must be positive and finite")


TIRE_PRESETS: dict[str, tuple[float, float]] = {
    # (peak traction scale, slip stiffness scale)
    "low": (0.75, 0.65),
    "medium": (1.00, 1.00),
    "high": (1.20, 1.35),
}


def tire_model_from_levels(
    *,
    wheel_radius_m: float,
    peak_traction: str = "medium",
    slip_buildup: str = "medium",
    base_slip_stiffness_n_per_mps: float = 2500.0,
) -> TireModel:
    """Build a tire from independently selected low/medium/high axes.

    Peak traction sets the force ceiling. Slip buildup sets the initial slope:
    ``high`` builds force quickly, while ``low`` requires more wheelspin.
    """
    if peak_traction not in TIRE_PRESETS:
        raise ValueError(f"unknown peak_traction level: {peak_traction}")
    if slip_buildup not in TIRE_PRESETS:
        raise ValueError(f"unknown slip_buildup level: {slip_buildup}")
    peak_scale = TIRE_PRESETS[peak_traction][0]
    stiffness_scale = TIRE_PRESETS[slip_buildup][1]
    return TireModel(
        wheel_radius_m=wheel_radius_m,
        peak_traction_scale=peak_scale,
        slip_stiffness_n_per_mps=base_slip_stiffness_n_per_mps * stiffness_scale,
    )


@dataclass(frozen=True, slots=True)
class IdealCVTModel:
    """Perfectly controlled CVT with finite ratio bounds.

    ``speed_ratio`` is defined as engine speed divided by secondary shaft speed.
    Therefore ``maximum_speed_ratio`` is the low-gear/high-reduction end and
    ``minimum_speed_ratio`` is the high-gear end.
    """

    minimum_speed_ratio: float = 0.9
    maximum_speed_ratio: float = 3.5
    final_drive_ratio: float = 7.556
    transmission_efficiency: float = 1.0
    ideal_launch_clutch: bool = True

    def __post_init__(self) -> None:
        values = {
            "minimum_speed_ratio": self.minimum_speed_ratio,
            "maximum_speed_ratio": self.maximum_speed_ratio,
            "final_drive_ratio": self.final_drive_ratio,
            "transmission_efficiency": self.transmission_efficiency,
        }
        for name, value in values.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.minimum_speed_ratio >= self.maximum_speed_ratio:
            raise ValueError("minimum_speed_ratio must be less than maximum_speed_ratio")
        if self.transmission_efficiency > 1.0:
            raise ValueError("transmission_efficiency cannot exceed 1")


@dataclass(frozen=True, slots=True)
class DriverModel:
    maximum_braking_deceleration_mps2: float = 5.0
    maximum_brake_force_n: float = 5000.0
    # This is a numerical speed-envelope tracker, not a human reaction model.
    # A high gain keeps arrival-speed error small at measured gates while the
    # finite braking envelope still determines where braking begins.
    speed_control_gain: float = 25.0
    speed_margin_mps: float = 0.10

    def __post_init__(self) -> None:
        for name, value in (
            ("maximum_braking_deceleration_mps2", self.maximum_braking_deceleration_mps2),
            ("maximum_brake_force_n", self.maximum_brake_force_n),
            ("speed_control_gain", self.speed_control_gain),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not isfinite(self.speed_margin_mps) or self.speed_margin_mps < 0.0:
            raise ValueError("speed_margin_mps must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    maximum_time_s: float = 180.0
    integration_step_s: float = 0.001
    report_step_s: float = 0.02
    initial_vehicle_speed_mps: float = 0.0
    initial_wheel_speed_rad_s: float = 0.0
    ideal_reference_torque_cap_nm: float = 2500.0
    ideal_reference_omega_floor_rad_s: float = 0.5

    def __post_init__(self) -> None:
        for name, value in (
            ("maximum_time_s", self.maximum_time_s),
            ("integration_step_s", self.integration_step_s),
            ("report_step_s", self.report_step_s),
            ("ideal_reference_torque_cap_nm", self.ideal_reference_torque_cap_nm),
            ("ideal_reference_omega_floor_rad_s", self.ideal_reference_omega_floor_rad_s),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.initial_vehicle_speed_mps < 0.0 or self.initial_wheel_speed_rad_s < 0.0:
            raise ValueError("initial speeds must be non-negative")


def parse_float_list(values: Iterable[str | float]) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed:
        raise ValueError("at least one value is required")
    if any(not isfinite(value) for value in parsed):
        raise ValueError("all values must be finite")
    return parsed
