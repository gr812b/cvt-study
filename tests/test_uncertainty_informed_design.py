from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cvt_track_study.studies.scenario_reduction import (
    ReductionSettings,
    load_uncertainty_source,
    reduce_uncertainty_source,
    resolve_source_result,
)


def _write_source(root: Path) -> Path:
    result = root / "results" / "full_uncertainty" / "source--abc"
    ensemble = result / "track_ensemble"
    ensemble.mkdir(parents=True)
    (result / "run_manifest.json").write_text(
        json.dumps(
            {
                "study_type": "full_uncertainty",
                "vehicle_id": "mcmaster",
                "study_fingerprint_sha256": "abc123",
            }
        ),
        encoding="utf-8",
    )
    cases = []
    for case_id, category in (
        ("nominal", "nominal"),
        ("mild", "centreline"),
        ("important", "gate_policy"),
    ):
        path = ensemble / f"{case_id}.json"
        path.write_text("{}", encoding="utf-8")
        cases.append(
            {
                "case_id": case_id,
                "category": category,
                "label": case_id,
                "file": f"track_ensemble/{case_id}.json",
            }
        )
    (ensemble / "manifest.json").write_text(
        json.dumps({"cases": cases}), encoding="utf-8"
    )

    scenario_rows = []
    result_rows = []
    replicate = 0
    for draw in range(10):
        for case_id in ("nominal", "mild", "important"):
            scenario_rows.append(
                {
                    "replicate": replicate,
                    "seed": 1000 + draw,
                    "base_draw_id": draw,
                    "track_case_id": case_id,
                    "track_case_category": "nominal" if case_id == "nominal" else "stress",
                    "sampling_mode": "all_declared",
                    "sampling_design": "latin_hypercube_rank_correlated",
                    "quantity_values_si": {
                        "drivetrain.final_drive_ratio": 7.556,
                        "drivetrain.efficiency": 0.75 + 0.02 * draw,
                        "drivetrain.engine.power_scale": 0.9 + 0.01 * draw,
                    },
                    "choice_values": {},
                    "gate_target_speeds_mps": {"gate:E01": 4.0 + 0.1 * draw},
                    "gate_sample_identity": {
                        "run_id": "run",
                        "lap_id": draw,
                        "vehicle_id": "mcmaster",
                        "driver_id": "driver",
                    },
                    "independently_sampled_gate_ids": [],
                }
            )
            case_shift = {"nominal": 0.0, "mild": 0.2, "important": 6.0}[case_id]
            result_rows.append(
                {
                    "replicate": replicate,
                    "base_draw_id": draw,
                    "track_case_id": case_id,
                    "design_id": "nominal",
                    "bounded_completed": True,
                    "reference_completed": True,
                    "bounded_lap_time_s": 250.0 + 2.0 * draw + case_shift,
                    "lap_time_penalty_vs_infinite_s": 2.0 + 0.1 * draw + 0.2 * case_shift,
                    "finite_ratio_opportunity_loss_energy_kj": 40.0 + draw + 2.0 * case_shift,
                    "bounded_time_maximum_ratio_s": 15.0 + draw,
                    "bounded_time_minimum_ratio_s": 4.0 + 0.25 * draw,
                    "bounded_obstacle_loss_energy_kj": 20.0 + draw,
                    "bounded_tire_slip_loss_energy_kj": 5.0 + 0.5 * draw,
                    "bounded_energy_balance_relative_error": 0.001,
                    "reference_energy_balance_relative_error": 0.001,
                    "bounded_powertrain_energy_balance_relative_error": 0.001,
                    "reference_powertrain_energy_balance_relative_error": 0.001,
                }
            )
            replicate += 1
    (result / "scenario_draws.jsonl").write_text(
        "\n".join(json.dumps(row) for row in scenario_rows) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(result_rows).to_csv(result / "replicate_results.csv", index=False)
    return result


def test_reduction_selects_actual_paired_worlds(tmp_path: Path) -> None:
    result = _write_source(tmp_path)
    source = load_uncertainty_source(result)
    reduced = reduce_uncertainty_source(
        source,
        settings=ReductionSettings(
            enabled=True,
            maximum_base_draws=4,
            maximum_track_cases=2,
        ),
        design_path="drivetrain.final_drive_ratio",
    )
    assert len(reduced.selected_base_draw_ids) == 4
    assert reduced.selected_track_case_ids == ("nominal", "important")
    assert len(reduced.selected_scenarios) == 8
    assert all(
        int(record["base_draw_id"]) in reduced.selected_base_draw_ids
        for record in reduced.selected_scenarios
    )
    assert {
        str(record["track_case_id"]) for record in reduced.selected_scenarios
    } == {"nominal", "important"}


def test_latest_source_resolution(tmp_path: Path) -> None:
    expected = _write_source(tmp_path)
    actual = resolve_source_result(
        project_root=tmp_path,
        results_directory=tmp_path / "results",
        source_value="latest",
    )
    assert actual == expected.resolve()
