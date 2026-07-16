"""Bounded ideal-CVT and infinite-ratio reference mechanisms."""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    CVTModel,
    EngineModel,
    RAD_PER_SECOND_TO_RPM,
    RPM_TO_RAD_PER_SECOND,
)


@dataclass(frozen=True, slots=True)
class PowertrainSample:
    cvt_ratio: float
    ratio_required: float
    mode: str
    engine_speed_rpm: float
    engine_torque_nm: float
    engine_power_w: float
    transmitted_power_w: float
    wheel_drive_torque_nm: float
    drivetrain_loss_power_w: float
    clutch_loss_power_w: float
    operating_shortfall_power_w: float


def evaluate_powertrain(
    *,
    wheel_speed_rad_s: float,
    throttle: float,
    engine: EngineModel,
    cvt: CVTModel,
    infinite_cvt: bool,
) -> PowertrainSample:
    if infinite_cvt:
        return _infinite(
            wheel_speed_rad_s=wheel_speed_rad_s,
            throttle=throttle,
            engine=engine,
            cvt=cvt,
        )
    return _bounded(
        wheel_speed_rad_s=wheel_speed_rad_s,
        throttle=throttle,
        engine=engine,
        cvt=cvt,
    )


def _bounded(
    *, wheel_speed_rad_s: float, throttle: float, engine: EngineModel, cvt: CVTModel
) -> PowertrainSample:
    wheel_speed = max(0.0, float(wheel_speed_rad_s))
    secondary_speed = cvt.final_drive_ratio * wheel_speed
    target_omega = engine.target_rpm * RPM_TO_RAD_PER_SECOND
    ratio_required = target_omega / max(secondary_speed, 1.0e-12)

    if ratio_required > cvt.maximum_reduction_ratio:
        ratio = cvt.maximum_reduction_ratio
        synchronous_omega = ratio * secondary_speed
        if cvt.ideal_launch_clutch and synchronous_omega < target_omega:
            mode = "maximum_ratio_clutch"
            engine_omega = target_omega
        else:
            mode = "maximum_ratio_synchronous"
            engine_omega = synchronous_omega
    elif ratio_required < cvt.minimum_reduction_ratio:
        ratio = cvt.minimum_reduction_ratio
        mode = "minimum_ratio_synchronous"
        engine_omega = ratio * secondary_speed
    else:
        ratio = ratio_required
        mode = "variable_target_rpm"
        engine_omega = target_omega

    engine_rpm = engine_omega * RAD_PER_SECOND_TO_RPM
    engine_torque = max(0.0, engine.torque_nm(engine_rpm)) * throttle
    engine_power = engine_torque * engine_omega
    wheel_torque = engine_torque * ratio * cvt.final_drive_ratio * cvt.efficiency
    transmitted_power = max(0.0, wheel_torque * wheel_speed)
    clutch_loss = (
        max(0.0, engine_power * cvt.efficiency - transmitted_power)
        if mode == "maximum_ratio_clutch"
        else 0.0
    )
    operating_shortfall = max(0.0, engine.target_power_w * throttle - engine_power)
    return PowertrainSample(
        cvt_ratio=float(ratio),
        ratio_required=float(ratio_required),
        mode=mode,
        engine_speed_rpm=float(engine_rpm),
        engine_torque_nm=float(engine_torque),
        engine_power_w=float(engine_power),
        transmitted_power_w=float(transmitted_power),
        wheel_drive_torque_nm=float(wheel_torque),
        drivetrain_loss_power_w=float(engine_power * (1.0 - cvt.efficiency)),
        clutch_loss_power_w=float(clutch_loss),
        operating_shortfall_power_w=float(operating_shortfall),
    )


def _infinite(
    *,
    wheel_speed_rad_s: float,
    throttle: float,
    engine: EngineModel,
    cvt: CVTModel,
) -> PowertrainSample:
    wheel_speed = max(0.0, float(wheel_speed_rad_s))
    secondary_speed = cvt.final_drive_ratio * wheel_speed
    target_omega = engine.target_rpm * RPM_TO_RAD_PER_SECOND
    ratio_required = target_omega / max(secondary_speed, 1.0e-12)
    engine_torque = max(0.0, engine.torque_nm(engine.target_rpm)) * throttle
    engine_power = engine_torque * target_omega
    # An unbounded ratio must not imply infinite launch torque.  The reference
    # shares the bounded design's maximum-ratio launch torque capacity and only
    # removes the finite ratio window once wheel speed is established.
    launch_torque_cap = (
        engine_torque
        * cvt.maximum_reduction_ratio
        * cvt.final_drive_ratio
        * cvt.efficiency
    )
    requested_wheel_torque = (
        float("inf")
        if wheel_speed <= 1.0e-12
        else engine_power * cvt.efficiency / wheel_speed
    )
    wheel_torque = min(requested_wheel_torque, launch_torque_cap)
    transmitted_power = wheel_torque * wheel_speed
    launch_limited = requested_wheel_torque > launch_torque_cap
    clutch_loss = (
        max(0.0, engine_power * cvt.efficiency - transmitted_power)
        if launch_limited
        else 0.0
    )
    return PowertrainSample(
        cvt_ratio=float(ratio_required),
        ratio_required=float(ratio_required),
        mode="infinite_launch_clutch" if launch_limited else "infinite_target_rpm",
        engine_speed_rpm=float(engine.target_rpm),
        engine_torque_nm=float(engine_torque),
        engine_power_w=float(engine_power),
        transmitted_power_w=float(transmitted_power),
        wheel_drive_torque_nm=float(wheel_torque),
        drivetrain_loss_power_w=float(engine_power * (1.0 - cvt.efficiency)),
        clutch_loss_power_w=float(clutch_loss),
        operating_shortfall_power_w=max(
            0.0, engine.target_power_w * throttle - engine_power
        ),
    )
