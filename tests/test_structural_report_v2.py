from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cvt_track_study.studies.structural_reporting import regenerate_structural_outputs


METRICS = {
    "bounded_lap_time_s": ("Bounded lap time", "s", 100.0),
    "bounded_maximum_speed_kmh": ("Maximum speed", "km/h", 50.0),
    "bounded_engine_energy_kj": ("Engine energy", "kJ", 500.0),
    "bounded_obstacle_loss_energy_kj": ("Obstacle loss", "kJ", 20.0),
    "lap_time_penalty_vs_infinite_s": ("Finite-ratio lap-time penalty", "s", 5.0),
    "finite_ratio_opportunity_loss_energy_kj": ("Finite-ratio opportunity loss", "kJ", 100.0),
    "bounded_drivetrain_loss_energy_kj": ("Drivetrain loss", "kJ", 0.0),
    "bounded_clutch_loss_energy_kj": ("Clutch loss", "kJ", 2.0),
    "bounded_tire_slip_loss_energy_kj": ("Tire-slip loss", "kJ", 10.0),
    "bounded_brake_loss_energy_kj": ("Brake loss", "kJ", 60.0),
    "bounded_rolling_loss_energy_kj": ("Rolling loss", "kJ", 30.0),
    "bounded_aerodynamic_loss_energy_kj": ("Aerodynamic loss", "kJ", 40.0),
    "bounded_time_maximum_ratio_s": ("Time at maximum ratio", "s", 12.0),
    "bounded_time_variable_ratio_s": ("Time in variable ratio", "s", 38.0),
    "bounded_time_minimum_ratio_s": ("Time at minimum ratio", "s", 50.0),
}


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    definitions = (
        ("drivetrain.efficiency", 1.0, 0.8, 1.0, 8.0),
        ("drivetrain.engine.power_scale", 1.0, 0.85, 1.05, 5.0),
        ("vehicle.rolling_resistance_coefficient", 0.02, 0.015, 0.04, 2.0),
        ("track.surface.friction_coefficient", 0.8, 0.65, 0.95, 3.0),
        ("vehicle.tire.peak_traction_scale", 1.0, 0.8, 1.1, 2.5),
        ("vehicle.driven_normal_load_fraction", 0.6, 0.5, 0.7, 1.5),
    )
    for parameter, nominal, low, high, lap_span in definitions:
        for kind, value, direction in (("nominal", nominal, 0.0), ("quantile", low, lap_span), ("quantile", high, -0.2 * lap_span)):
            row: dict[str, object] = {
                "replicate": 0,
                "parameter_path": parameter,
                "design_id": f"{parameter}@{kind}-{value}",
                "level_kind": kind,
                "level_probability": None if kind == "nominal" else (0.05 if value == low else 0.95),
                "design_value": value,
                "design_value_si": value,
                "design_choice_value": None,
                "bounded_completed": True,
                "bounded_termination_reason": "completed",
                "bounded_lap_time_s": 100.0 + direction,
                "reference_lap_time_s": 95.0 + 1.15 * direction,
                "bounded_maximum_speed_kmh": 50.0 - 0.2 * direction,
                "bounded_average_speed_kmh": 40.0 - 0.1 * direction,
                "bounded_distance_m": 1500.0,
                "bounded_engine_energy_kj": 500.0 + 12.0 * direction,
                "bounded_transmitted_energy_kj": 450.0 + 8.0 * direction,
                "bounded_drivetrain_loss_energy_kj": max(0.0, 8.0 * direction),
                "bounded_clutch_loss_energy_kj": 2.0,
                "bounded_engine_operating_shortfall_energy_kj": 5.0,
                "bounded_tire_slip_loss_energy_kj": 10.0 + direction,
                "bounded_brake_loss_energy_kj": 60.0,
                "bounded_rolling_loss_energy_kj": 30.0 + direction,
                "bounded_aerodynamic_loss_energy_kj": 40.0,
                "bounded_obstacle_loss_energy_kj": 20.0,
                "bounded_time_maximum_ratio_s": 12.0,
                "bounded_time_variable_ratio_s": 38.0 + 0.5 * direction,
                "bounded_time_minimum_ratio_s": 50.0 + direction,
                "bounded_time_braking_s": 30.0,
                "bounded_time_traction_limited_s": 15.0 + direction,
                "lap_time_penalty_vs_infinite_s": 5.0 - 0.15 * direction,
                "finite_ratio_opportunity_loss_energy_kj": 100.0 - 3.0 * direction,
            }
            rows.append(row)
    return rows


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    rankings: dict[str, list[dict[str, object]]] = {}
    parameters: dict[str, dict[str, object]] = {}
    for parameter, group in frame.groupby("parameter_path", sort=True):
        nominal_row = group.loc[group["level_kind"] == "nominal"].iloc[0]
        metric_rows: dict[str, dict[str, object]] = {}
        for metric, (label, unit, _) in METRICS.items():
            values = group[metric].astype(float)
            nominal = float(nominal_row[metric])
            minimum = float(values.min())
            maximum = float(values.max())
            record = {
                "label": label,
                "unit": unit,
                "nominal": nominal,
                "minimum": minimum,
                "maximum": maximum,
                "span": maximum - minimum,
                "minimum_change_from_nominal": minimum - nominal,
                "maximum_change_from_nominal": maximum - nominal,
                "maximum_abs_change_from_nominal": max(abs(minimum - nominal), abs(maximum - nominal)),
                "maximum_abs_percent_change_from_nominal": 0.0 if nominal == 0.0 else 100.0 * max(abs(minimum - nominal), abs(maximum - nominal)) / abs(nominal),
                "minimum_design_id": str(group.loc[values.idxmin(), "design_id"]),
                "maximum_design_id": str(group.loc[values.idxmax(), "design_id"]),
            }
            metric_rows[metric] = record
        parameters[str(parameter)] = {"category": str(parameter).split(".", 1)[0], "metrics": metric_rows}

    for metric, (label, unit, _) in METRICS.items():
        records = []
        for parameter, parameter_data in parameters.items():
            metric_data = parameter_data["metrics"][metric]
            records.append({
                "path": parameter,
                "category": parameter_data["category"],
                "label": label,
                "unit": unit,
                **metric_data,
            })
        records.sort(key=lambda row: float(row["maximum_abs_change_from_nominal"]), reverse=True)
        maximum = max(float(row["maximum_abs_change_from_nominal"]) for row in records) or 1.0
        for index, record in enumerate(records, start=1):
            record["rank"] = index
            record["relative_screening_importance"] = float(record["maximum_abs_change_from_nominal"]) / maximum
        rankings[metric] = records

    return {
        "study_type": "structural_sensitivity",
        "method": "one_at_a_time",
        "parameter_count": len(parameters),
        "level_count": len(rows),
        "completed_level_count": len(rows),
        "headline_metrics": list(METRICS)[:6],
        "metric_definitions": {metric: {"label": label, "unit": unit} for metric, (label, unit, _) in METRICS.items()},
        "parameters": parameters,
        "rankings": rankings,
        "numerical_quality": {"numerically_valid": True, "all_cases_completed": True},
    }


def _contracts(rows: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    for parameter in sorted({str(row["parameter_path"]) for row in rows}):
        result[parameter] = {
            "category": "structural",
            "contract": {
                "nominal": 1.0 if parameter != "vehicle.rolling_resistance_coefficient" else 0.02,
                "unit": "1",
                "uncertainty": {"distribution": "uniform", "role": "structural"},
                "source": {"kind": "assumption", "reference": "test fixture"},
            },
        }
    return result


def test_regenerates_complete_structural_report_without_simulation(tmp_path: Path) -> None:
    rows = _rows()
    pd.DataFrame(rows).to_csv(tmp_path / "replicate_results.csv", index=False)
    (tmp_path / "summary.json").write_text(json.dumps(_summary(rows)), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "study_type": "structural_sensitivity",
        "parallel_workers": 6,
        "bounded_simulation_count": len(rows),
        "reference_simulation_count": len(rows),
        "structural_report_top_parameter_count": 10,
        "structural_response_curve_count": 6,
    }), encoding="utf-8")
    (tmp_path / "input_contracts.json").write_text(json.dumps(_contracts(rows)), encoding="utf-8")
    (tmp_path / "structural_sensitivity.png").write_bytes(b"legacy")

    report = regenerate_structural_outputs(tmp_path)

    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "Executive summary and nominal reference" in text
    assert "Absolute performance versus finite-ratio restriction" in text
    assert "Measurement priorities and joint-uncertainty preparation" in text
    assert "Optimistic nominal efficiency" in text
    assert "data-sort-state=\"none\"" in text
    assert "sticky-col" in text
    assert "ascending, descending, and original order" in text
    assert "Maximumpeed" not in text
    assert (tmp_path / "structural_absolute_vs_ratio_drivers.png").is_file()
    assert (tmp_path / "structural_ratio_occupancy_sensitivity.png").is_file()
    assert (tmp_path / "structural_lap_time_tornado.png").is_file()
    assert (tmp_path / "structural_maximum_speed_tornado.png").is_file()
    assert (tmp_path / "structural_maximumpeed_tornado.png").is_file()  # compatibility alias
    assert (tmp_path / "structural_engine_energy_tornado.png").is_file()
    assert (tmp_path / "structural_engine_tornado.png").is_file()  # compatibility alias
    assert (tmp_path / "structural_measurement_priorities.csv").is_file()
    assert (tmp_path / "structural_uncertainty_families.csv").is_file()
    assert len(list(tmp_path.glob("structural_response_*.png"))) == 6
