from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cvt_track_study.reports.postprocess import (
    _design_configuration_order,
    _design_ranking,
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
    text = design.read_text(encoding="utf-8")
    assert "Completion probability" in text
    assert "Ranked by paired result" in text
    assert "Configuration order" in text
    assert 'data-design-order-view="ranked"' in text
    assert 'data-design-order-view="configured"' in text
    assert (tmp_path / "report_plots" / "design_ratio_time_configured.png").is_file()


def test_configuration_order_uses_declared_cartesian_axis_order(tmp_path: Path) -> None:
    paths = ("first.axis", "second.axis")
    records = []
    for design_id, first, second, lap_time in (
        ("f3-s5", 3.0, 5.0, 101.0),
        ("f4-s6", 4.0, 6.0, 99.0),
        ("f4-s5", 4.0, 5.0, 100.0),
        ("f3-s6", 3.0, 6.0, 102.0),
    ):
        records.append(
            {
                "replicate": 0,
                "design_id": design_id,
                "design_values_json": json.dumps(
                    {paths[0]: first, paths[1]: second}
                ),
                "bounded_completed": True,
                "bounded_lap_time_s": lap_time,
                "lap_time_penalty_vs_infinite_s": lap_time - 95.0,
                "finite_ratio_opportunity_loss_energy_kj": 1.0,
            }
        )
    rows = pd.DataFrame(records)
    resolved = tmp_path / "resolved_inputs"
    resolved.mkdir()
    (resolved / "resolved_inputs.toml").write_text(
        """
[studies.grid.study]
name = "grid"
type = "design_sweep"

[[studies.grid.design_variables]]
path = "first.axis"
values = [4.0, 3.0]

[[studies.grid.design_variables]]
path = "second.axis"
values = [6.0, 5.0]
""".strip(),
        encoding="utf-8",
    )
    configured, note = _design_configuration_order(
        rows,
        _design_ranking(rows),
        output=tmp_path,
        manifest={"study_name": "grid", "design_variable_paths": list(paths)},
    )
    assert configured["design_id"].tolist() == [
        "f4-s6",
        "f4-s5",
        "f3-s6",
        "f3-s5",
    ]
    assert "declared" in note


def test_configuration_order_falls_back_to_numeric_axis_order(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "replicate": 0,
                "design_id": f"d{value:g}",
                "design_value": value,
                "bounded_completed": True,
                "bounded_lap_time_s": 100.0 - value,
                "lap_time_penalty_vs_infinite_s": 1.0,
                "finite_ratio_opportunity_loss_energy_kj": 1.0,
            }
            for value in (7.0, 3.0, 5.0)
        ]
    )
    configured, note = _design_configuration_order(
        rows,
        _design_ranking(rows),
        output=tmp_path,
        manifest={"design_variable_paths": ["ratio"]},
    )
    assert configured["design_id"].tolist() == ["d3", "d5", "d7"]
    assert "minimum to maximum" in note


def test_design_report_switches_table_and_candidate_plots_together(
    tmp_path: Path,
) -> None:
    rows = _study_rows()
    rows.to_csv(tmp_path / "replicate_results.csv", index=False)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "study_type": "design_sweep",
                "design_variable_paths": ["drivetrain.final_drive_ratio"],
            }
        ),
        encoding="utf-8",
    )
    report = write_design_comparison_report(tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "Ranked by paired result" in text
    assert "Configuration order" in text
    assert 'data-design-order-view="ranked"' in text
    assert 'data-design-order-view="configured"' in text
    assert "select[data-design-order-select]" in text
    assert (tmp_path / "report_plots" / "design_lap_time.png").is_file()
    assert (tmp_path / "report_plots" / "design_lap_time_configured.png").is_file()
    assert (tmp_path / "report_plots" / "design_ratio_time_configured.png").is_file()
