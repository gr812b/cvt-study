from __future__ import annotations

from pathlib import Path

from cvt_track_study.studies.structural_analysis import (
    summarize_structural_screening,
)
from cvt_track_study.studies.structural_reporting import (
    write_structural_outputs,
)


def _row(level, lap_time):
    return {
        "parameter_path": "vehicle.mass",
        "design_id": f"vehicle.mass@{level}",
        "level_kind": (
            "nominal"
            if level == "nominal"
            else "quantile"
        ),
        "level_probability": None,
        "design_value": level,
        "design_value_si": 1.0,
        "design_choice_value": None,
        "bounded_completed": True,
        "bounded_termination_reason": "completed",
        "bounded_lap_time_s": lap_time,
        "bounded_maximum_speed_kmh": 50.0,
        "bounded_average_speed_kmh": 30.0,
        "bounded_distance_m": 1000.0,
        "bounded_engine_energy_kj": 40.0,
        "bounded_transmitted_energy_kj": 39.0,
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


def test_html_contains_absolute_outputs_and_full_tables(tmp_path: Path):
    rows = [_row("nominal", 100.0), _row("high", 110.0)]
    summary = {
        **summarize_structural_screening(rows),
        "numerical_quality": {
            "numerically_valid": True,
            "all_cases_completed": True,
        },
    }
    (tmp_path / "SUMMARY.md").write_text("# Summary\n", encoding="utf-8")
    (tmp_path / "REPORT.md").write_text("# Report\n", encoding="utf-8")
    write_structural_outputs(
        output=tmp_path,
        rows=rows,
        summary=summary,
        manifest={
            "structural_parameter_paths": ["vehicle.mass"],
            "structural_selection_mode": "all_declared_structural",
            "bounded_simulation_count": 2,
            "reference_simulation_count": 2,
            "parallel_workers": 1,
        },
        input_contracts={
            "vehicle.mass": {
                "category": "structural",
                "contract": {
                    "nominal": 280,
                    "unit": "kg",
                    "uncertainty": {
                        "distribution": "uniform",
                        "role": "structural",
                    },
                    "source": {
                        "kind": "measured",
                        "reference": "scale",
                    },
                },
            }
        },
    )
    report = (
        tmp_path / "structural_sensitivity_report.html"
    ).read_text(encoding="utf-8")
    assert "All-declared structural sensitivity review" in report
    assert "Maximum speed" in report
    assert "Engine energy" in report
    assert "Every evaluated level" in report
    assert (tmp_path / "structural_metric_ranges.csv").is_file()
    assert (tmp_path / "structural_parameter_levels.csv").is_file()
