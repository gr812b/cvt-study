from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cvt_track_study.reports.postprocess import (
    write_design_comparison_report,
    write_full_uncertainty_report,
    write_nominal_simulation_report,
)


def _png(path: Path) -> None:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    figure.savefig(path)
    plt.close(figure)


def test_nominal_report_has_mechanism_sections(tmp_path: Path) -> None:
    (tmp_path / "bounded_summary.json").write_text(
        json.dumps({"completed": True, "lap_time_s": 140, "maximum_speed_kmh": 50}),
        encoding="utf-8",
    )
    (tmp_path / "infinite_reference_summary.json").write_text(
        json.dumps({"completed": True, "lap_time_s": 134}), encoding="utf-8"
    )
    (tmp_path / "comparison_summary.json").write_text(
        json.dumps(
            {
                "lap_time_penalty_vs_infinite_s": 6,
                "finite_ratio_opportunity_loss_energy_kj": 200,
                "reference_dominance_pass": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    for name in ("01_speed_comparison.png", "02_ratio_trace.png", "03_energy_accounting.png"):
        _png(tmp_path / name)
    path = write_nominal_simulation_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "Primary performance traces" in text
    assert "Obstacle energy by feature" in text
    assert (tmp_path / "report_manifest.json").is_file()


def _study_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "replicate": 0,
                "track_case_id": "nominal",
                "design_id": "d1",
                "bounded_completed": True,
                "reference_completed": True,
                "reference_dominance_pass": True,
                "bounded_gates_compliant_0p5_kmh": True,
                "reference_gates_compliant_0p5_kmh": True,
                "bounded_lap_time_s": 140.0,
                "infinite_reference_lap_time_s": 134.0,
                "lap_time_penalty_vs_infinite_s": 6.0,
                "finite_ratio_opportunity_loss_energy_kj": 200.0,
                "bounded_time_maximum_ratio_s": 10.0,
                "bounded_time_variable_ratio_s": 50.0,
                "bounded_time_minimum_ratio_s": 80.0,
            },
            {
                "replicate": 1,
                "track_case_id": "gate_strict",
                "design_id": "d2",
                "bounded_completed": False,
                "reference_completed": True,
                "reference_dominance_pass": True,
                "bounded_gates_compliant_0p5_kmh": True,
                "reference_gates_compliant_0p5_kmh": True,
                "bounded_lap_time_s": 240.0,
                "infinite_reference_lap_time_s": 180.0,
                "lap_time_penalty_vs_infinite_s": 60.0,
                "finite_ratio_opportunity_loss_energy_kj": 500.0,
                "bounded_time_maximum_ratio_s": 20.0,
                "bounded_time_variable_ratio_s": 100.0,
                "bounded_time_minimum_ratio_s": 120.0,
            },
        ]
    )


def test_full_and_design_reports_surface_track_case_and_failures(tmp_path: Path) -> None:
    rows = _study_rows()
    rows.to_csv(tmp_path / "replicate_results.csv", index=False)
    for name, value in (
        ("summary.json", {}),
        ("convergence.json", {}),
        (
            "run_manifest.json",
            {
                "study_type": "full_uncertainty",
                "track_ensemble_case_count": 2,
                "track_ensemble_case_ids": ["nominal", "gate_strict"],
            },
        ),
    ):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    full = write_full_uncertainty_report(tmp_path)
    text = full.read_text(encoding="utf-8")
    assert "track_case_id" in text
    assert "Run health and censoring" in text

    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"study_type": "design_sweep"}), encoding="utf-8"
    )
    design = write_design_comparison_report(tmp_path)
    assert "Completion probability" in design.read_text(encoding="utf-8")
