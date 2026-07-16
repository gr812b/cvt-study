"""Validated nominal simulation inputs.

The uncertainty declarations remain in the resolved project and track bundle. Baseline
runs use their nominal values, while Phase 6 study runners draw paired realizations
from the same contracts without changing these mechanism classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Any, Mapping

import numpy as np

from cvt_track_study.config.uncertainty import UncertainChoice, UncertainQuantity
from cvt_track_study.config.units import require_dimension

RPM_TO_RAD_PER_SECOND = 2.0 * pi / 60.0
RAD_PER_SECOND_TO_RPM = 60.0 / (2.0 * pi)


class SimulationInputError(ValueError):
    """Raised when resolved inputs cannot form a physically valid simulation case."""


@dataclass(frozen=True, slots=True)
class EngineTorquePoint:
    rpm: float
    torque_nm: float

    def __post_init__(self) -> None:
        if not isfinite(self.rpm) or self.rpm < 0.0:
            raise SimulationInputError("Engine RPM points must be finite and non-negative.")
        if not isfinite(self.torque_nm):
            raise SimulationInputError("Engine torque points must be finite.")


@dataclass(frozen=True, slots=True)
class EngineModel:
    points: tuple[EngineTorquePoint, ...]
    target_rpm: float
    power_scale: float = 1.0

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise SimulationInputError("At least two engine torque points are required.")
        rpms = tuple(point.rpm for point in self.points)
        if rpms != tuple(sorted(rpms)) or len(set(rpms)) != len(rpms):
            raise SimulationInputError("Engine torque-curve RPM values must be strictly increasing.")
        if not isfinite(self.target_rpm) or self.target_rpm <= 0.0:
            raise SimulationInputError("Engine target RPM must be positive and finite.")
        if not isfinite(self.power_scale) or self.power_scale <= 0.0:
            raise SimulationInputError("Engine power scale must be positive and finite.")

    @property
    def rpm_values(self) -> np.ndarray:
        return np.asarray([point.rpm for point in self.points], dtype=float)

    @property
    def torque_values_nm(self) -> np.ndarray:
        return np.asarray([point.torque_nm for point in self.points], dtype=float)

    def torque_nm(self, rpm: float) -> float:
        return self.power_scale * float(
            np.interp(float(rpm), self.rpm_values, self.torque_values_nm, left=0.0, right=0.0)
        )

    def power_w(self, rpm: float) -> float:
        return self.torque_nm(rpm) * float(rpm) * RPM_TO_RAD_PER_SECOND

    @property
    def target_power_w(self) -> float:
        return self.power_w(self.target_rpm)

    @classmethod
    def baja_br10_reference(cls, *, target_rpm: float, power_scale: float) -> "EngineModel":
        # Reference full-throttle curve used by the prior study.  It is deliberately
        # identified as a model profile rather than a measured curve for every vehicle.
        ft_lb_to_nm = 1.3558179483314004
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
                EngineTorquePoint(rpm=rpm, torque_nm=torque_ft_lb * ft_lb_to_nm)
                for rpm, torque_ft_lb in raw
            ),
            target_rpm=target_rpm,
            power_scale=power_scale,
        )


@dataclass(frozen=True, slots=True)
class VehicleModel:
    mass_kg: float
    wheel_radius_m: float
    wheel_rotational_inertia_kg_m2: float
    driven_normal_load_fraction: float
    drag_area_m2: float
    air_density_kg_m3: float
    rolling_resistance_coefficient: float
    gravity_mps2: float

    def __post_init__(self) -> None:
        positive = {
            "mass_kg": self.mass_kg,
            "wheel_radius_m": self.wheel_radius_m,
            "wheel_rotational_inertia_kg_m2": self.wheel_rotational_inertia_kg_m2,
            "air_density_kg_m3": self.air_density_kg_m3,
            "gravity_mps2": self.gravity_mps2,
        }
        for name, value in positive.items():
            if not isfinite(value) or value <= 0.0:
                raise SimulationInputError(f"{name} must be positive and finite.")
        if not 0.0 < self.driven_normal_load_fraction <= 1.0:
            raise SimulationInputError("driven_normal_load_fraction must lie in (0, 1].")
        for name, value in (
            ("drag_area_m2", self.drag_area_m2),
            ("rolling_resistance_coefficient", self.rolling_resistance_coefficient),
        ):
            if not isfinite(value) or value < 0.0:
                raise SimulationInputError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class TireModel:
    peak_traction_scale: float
    slip_stiffness_n_per_mps: float

    def __post_init__(self) -> None:
        if not isfinite(self.peak_traction_scale) or self.peak_traction_scale <= 0.0:
            raise SimulationInputError("peak_traction_scale must be positive and finite.")
        if not isfinite(self.slip_stiffness_n_per_mps) or self.slip_stiffness_n_per_mps <= 0.0:
            raise SimulationInputError("slip_stiffness_n_per_mps must be positive and finite.")


@dataclass(frozen=True, slots=True)
class CVTModel:
    minimum_reduction_ratio: float
    maximum_reduction_ratio: float
    final_drive_ratio: float
    efficiency: float
    ideal_launch_clutch: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_reduction_ratio", self.minimum_reduction_ratio),
            ("maximum_reduction_ratio", self.maximum_reduction_ratio),
            ("final_drive_ratio", self.final_drive_ratio),
            ("efficiency", self.efficiency),
        ):
            if not isfinite(value) or value <= 0.0:
                raise SimulationInputError(f"{name} must be positive and finite.")
        if self.minimum_reduction_ratio >= self.maximum_reduction_ratio:
            raise SimulationInputError(
                "minimum_reduction_ratio must be less than maximum_reduction_ratio."
            )
        if self.efficiency > 1.0:
            raise SimulationInputError("CVT/drivetrain efficiency cannot exceed one.")


@dataclass(frozen=True, slots=True)
class DriverModel:
    maximum_braking_deceleration_mps2: float
    maximum_brake_force_n: float
    braking_trigger_margin_mps: float

    def __post_init__(self) -> None:
        for name, value in (
            ("maximum_braking_deceleration_mps2", self.maximum_braking_deceleration_mps2),
            ("maximum_brake_force_n", self.maximum_brake_force_n),
        ):
            if not isfinite(value) or value <= 0.0:
                raise SimulationInputError(f"{name} must be positive and finite.")
        if (
            not isfinite(self.braking_trigger_margin_mps)
            or self.braking_trigger_margin_mps < 0.0
        ):
            raise SimulationInputError(
                "braking_trigger_margin_mps must be finite and non-negative."
            )


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    maximum_time_s: float
    integration_step_s: float
    report_step_s: float
    initial_vehicle_speed_mps: float
    initial_wheel_speed_rad_s: float

    def __post_init__(self) -> None:
        for name, value in (
            ("maximum_time_s", self.maximum_time_s),
            ("integration_step_s", self.integration_step_s),
            ("report_step_s", self.report_step_s),
        ):
            if not isfinite(value) or value <= 0.0:
                raise SimulationInputError(f"{name} must be positive and finite.")
        if self.report_step_s < self.integration_step_s:
            raise SimulationInputError("report_step_s must be at least integration_step_s.")
        if self.initial_vehicle_speed_mps < 0.0 or self.initial_wheel_speed_rad_s < 0.0:
            raise SimulationInputError("Initial speeds must be non-negative.")


@dataclass(frozen=True, slots=True)
class StudyCase:
    name: str
    engine: EngineModel
    vehicle: VehicleModel
    tire: TireModel
    cvt: CVTModel
    driver: DriverModel
    infinite_cvt: bool = False


def quantity_si(raw: Mapping[str, Any], expected_dimension: str) -> float:
    """Read one already-validated uncertainty-aware quantity as its nominal SI value."""

    quantity = UncertainQuantity.from_mapping(raw)
    require_dimension(quantity.unit, expected_dimension)
    return quantity.nominal_si()[0]


def quantity_nominal(raw: Mapping[str, Any], expected_dimension: str = "dimensionless") -> float:
    quantity = UncertainQuantity.from_mapping(raw)
    require_dimension(quantity.unit, expected_dimension)
    return quantity.nominal_si()[0]


def choice_nominal(raw: Mapping[str, Any]) -> str:
    return UncertainChoice.from_mapping(raw).nominal
