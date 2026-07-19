from __future__ import annotations

from cvt_track_study.studies.structural_analysis import (
    summarize_structural_screening,
)


def _row(level, lap_time, speed, engine):
    return {
        "parameter_path": "vehicle.mass",
        "design_id": f"vehicle.mass@{level}",
        "level_kind": (
            "nominal"
            if level == "nominal"
            else "quantile"
        ),
        "level_probability": (
            None
            if level == "nominal"
            else float(level)
        ),
        "design_value": level,
        "design_value_si": 1.0,
        "design_choice_value": None,
        "bounded_completed": True,
        "bounded_termination_reason": "completed",
        "bounded_lap_time_s": lap_time,
        "bounded_maximum_speed_kmh": speed,
        "bounded_average_speed_kmh": 30.0,
        "bounded_distance_m": 1000.0,
        "bounded_engine_energy_kj": engine,
        "bounded_transmitted_energy_kj": engine - 1.0,
        "bounded_drivetrain_loss_energy_kj": 1.0,
        "bounded_clutch_loss_energy_kj": 0.1,
        "bounded_engine_operating_shortfall_energy_kj": 0.2,
        "bounded_tire_slip_loss_energy_kj": 0.3,
        "bounded_brake_loss_energy_kj": 0.4,
        "bounded_rolling_loss_energy_kj": 0.5,
        "bounded_aerodynamic_loss_energy_kj": 0.6,
        "bounded_obstacle_loss_energy_kj": 0.7,
        "bounded_time_maximum_ratio_s": 10.0,
        "bounded_time_variable_ratio_s": 80.0,
        "bounded_time_minimum_ratio_s": 10.0,
        "bounded_time_braking_s": 5.0,
        "bounded_time_traction_limited_s": 1.0,
        "lap_time_penalty_vs_infinite_s": 0.1,
        "finite_ratio_opportunity_loss_energy_kj": 0.2,
    }


def test_summary_ranks_absolute_metrics_not_only_cvt_penalty():
    summary = summarize_structural_screening(
        [
            _row("0.05", 110.0, 45.0, 50.0),
            _row("nominal", 100.0, 50.0, 45.0),
            _row("0.95", 95.0, 55.0, 42.0),
        ]
    )
    parameter = summary["parameters"]["vehicle.mass"]
    assert parameter["absolute_lap_time_span_s"] == 15.0
    assert parameter["maximum_speed_span_kmh"] == 10.0
    assert (
        parameter["metrics"]["bounded_engine_energy_kj"]["span"]
        == 8.0
    )
    assert (
        summary["rankings"]["bounded_lap_time_s"][0]["path"]
        == "vehicle.mass"
    )
    assert summary["statistical_error_bars_applicable"] is False
