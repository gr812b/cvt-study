from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cvt_track_study.reports.full_uncertainty import regenerate_full_uncertainty_report


def _write_fixture(output: Path) -> None:
    rows = []
    scenarios = []
    replicate = 0
    for base_draw in range(6):
        efficiency = 0.76 + 0.04 * base_draw
        power = 0.82 + 0.03 * base_draw
        gate_speed = 7.0 + 0.2 * base_draw
        for case_id, case_delta in (("nominal", 0.0), ("event_windows_narrow", 0.3)):
            bounded = 152.0 - 8.0 * efficiency - 5.0 * power + case_delta + 0.1 * base_draw
            penalty = 1.0 + 4.0 * efficiency + 2.0 * power + 0.1 * case_delta
            rows.append(
                {
                    "replicate": replicate,
                    "base_draw_id": base_draw,
                    "track_pair_id": f"draw-{base_draw:06d}",
                    "track_case_id": case_id,
                    "track_case_category": "nominal" if case_id == "nominal" else "event_windows",
                    "design_id": "nominal",
                    "bounded_completed": True,
                    "reference_completed": True,
                    "bounded_lap_time_s": bounded,
                    "infinite_reference_lap_time_s": bounded - penalty,
                    "lap_time_penalty_vs_infinite_s": penalty,
                    "finite_ratio_opportunity_loss_energy_kj": 80.0 + 20.0 * efficiency,
                    "bounded_time_maximum_ratio_s": 5.0,
                    "bounded_time_variable_ratio_s": 70.0,
                    "bounded_time_minimum_ratio_s": 70.0,
                    "bounded_drivetrain_loss_energy_kj": 25.0,
                    "bounded_clutch_loss_energy_kj": 2.0,
                    "bounded_tire_slip_loss_energy_kj": 20.0,
                    "bounded_brake_loss_energy_kj": 50.0,
                    "bounded_rolling_loss_energy_kj": 30.0,
                    "bounded_aerodynamic_loss_energy_kj": 28.0,
                    "bounded_obstacle_loss_energy_kj": 10.0,
                }
            )
            scenarios.append(
                {
                    "replicate": replicate,
                    "base_draw_id": base_draw,
                    "track_pair_id": f"draw-{base_draw:06d}",
                    "track_case_id": case_id,
                    "track_case_category": "nominal" if case_id == "nominal" else "event_windows",
                    "sampling_mode": "all_declared",
                    "seed": 1000 + base_draw,
                    "quantity_values_si": {
                        "drivetrain.efficiency": efficiency,
                        "drivetrain.engine.power_scale": power,
                        "vehicle.rolling_resistance_coefficient": 0.02 + 0.002 * base_draw,
                    },
                    "choice_values": {},
                    "gate_target_speeds_mps": {"gate:E03": gate_speed},
                    "gate_sample_identity": {
                        "run_id": "run_A",
                        "lap_id": base_draw,
                        "vehicle_id": "vehicle_A",
                        "driver_id": "driver_A",
                    },
                    "independently_sampled_gate_ids": [],
                }
            )
            replicate += 1

    pd.DataFrame(rows).to_csv(output / "replicate_results.csv", index=False)
    (output / "scenario_draws.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in scenarios) + "\n",
        encoding="utf-8",
    )
    contracts = {
        "drivetrain.efficiency": {
            "category": "structural",
            "contract": {
                "nominal": 0.9,
                "unit": "1",
                "uncertainty": {
                    "distribution": "triangular",
                    "lower": 0.75,
                    "mode": 0.9,
                    "upper": 1.0,
                    "role": "structural",
                },
                "source": {"kind": "estimate", "reference": "test fixture"},
            },
        },
        "drivetrain.engine.power_scale": {
            "category": "structural",
            "contract": {
                "nominal": 1.0,
                "unit": "1",
                "uncertainty": {
                    "distribution": "uniform",
                    "lower": 0.8,
                    "upper": 1.05,
                    "role": "structural",
                },
                "source": {"kind": "estimate", "reference": "test fixture"},
            },
        },
        "vehicle.rolling_resistance_coefficient": {
            "category": "structural",
            "contract": {
                "nominal": 0.03,
                "unit": "1",
                "uncertainty": {
                    "distribution": "uniform",
                    "lower": 0.015,
                    "upper": 0.05,
                    "role": "structural",
                },
                "source": {"kind": "estimate", "reference": "test fixture"},
            },
        },
    }
    (output / "input_contracts.json").write_text(json.dumps(contracts), encoding="utf-8")
    manifest = {
        "study_type": "full_uncertainty",
        "study_name": "full_uncertainty",
        "scenario_count": len(rows),
        "sampling_layout": "cross_track_cases",
        "base_draw_count": 6,
        "scenarios_per_track_case": 6,
        "track_case_pairing_complete": True,
        "track_case_assignment": "fully_crossed_common_draws",
        "track_ensemble_case_count": 2,
        "track_ensemble_policy": "unweighted_epistemic_scenarios_not_calibrated_probabilities",
        "paired_gate_identity_count": 6,
        "numerical_quality": {"numerically_valid": True},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output / "summary.json").write_text("{}", encoding="utf-8")
    (output / "convergence.json").write_text("{}", encoding="utf-8")
    (output / "nominal_reference.json").write_text(
        json.dumps([
            {
                "bounded_lap_time_s": 140.0,
                "infinite_reference_lap_time_s": 134.0,
                "lap_time_penalty_vs_infinite_s": 6.0,
                "finite_ratio_opportunity_loss_energy_kj": 200.0,
            }
        ]),
        encoding="utf-8",
    )


def test_full_uncertainty_report_regenerates_without_simulation(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    report = regenerate_full_uncertainty_report(tmp_path)
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "What was varied?" in text
    assert "Study adequacy" in text
    assert "All inputs are present—not only the top ten" in text
    assert "paired and isolated" in text
    assert "data-table-search" in text
    assert "ascending, descending" in text
    assert (tmp_path / "full_uncertainty_driver_explorer.csv").is_file()
    assert (tmp_path / "full_uncertainty_parameter_inventory.csv").is_file()
    assert (tmp_path / "full_uncertainty_paired_track_effects.csv").is_file()
    assert (tmp_path / "report_plots" / "absolute_lap_time_uncertainty_drivers_all.png").is_file()
    assert (tmp_path / "report_plots" / "paired_track_case_effects.png").is_file()

    inventory = pd.read_csv(tmp_path / "full_uncertainty_parameter_inventory.csv")
    assert set(inventory["parameter_path"]) == {
        "drivetrain.efficiency",
        "drivetrain.engine.power_scale",
        "vehicle.rolling_resistance_coefficient",
    }
    drivers = pd.read_csv(tmp_path / "full_uncertainty_driver_explorer.csv")
    assert set(drivers["metric"]) == {
        "bounded_lap_time_s",
        "lap_time_penalty_vs_infinite_s",
        "finite_ratio_opportunity_loss_energy_kj",
    }
    assert len(drivers) > 9  # Every input/metric combination, not a top-N truncation.
