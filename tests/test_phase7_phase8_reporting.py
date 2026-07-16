from __future__ import annotations

import csv
import json
from pathlib import Path

from cvt_track_study.simulation.reporting_v8 import (
    regenerate_baseline_reports,
    write_baseline_hierarchy,
)
from cvt_track_study.config.diagnostics import Diagnostic, Severity
from cvt_track_study.runtime.evidence import assess_evidence
from cvt_track_study.studies.decision import synthesize_decision
from cvt_track_study.studies.reporting_v8 import regenerate_study_reports


def test_baseline_hierarchy_is_decision_first_and_regenerable(tmp_path: Path) -> None:
    bounded = _case("bounded", 10.5, 8.0)
    reference = _case("reference", 10.0, 1.0)
    comparison = {
        "lap_time_penalty_vs_infinite_s": 0.5,
        "finite_ratio_opportunity_loss_energy_kj": 7.0,
        "reference_dominance_pass": True,
    }
    with (tmp_path / "gate_compliance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bounded_compliant_0p5_kmh", "reference_compliant_0p5_kmh"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "bounded_compliant_0p5_kmh": True,
                "reference_compliant_0p5_kmh": True,
            }
        )
    for name, value in (
        ("bounded_summary", bounded),
        ("infinite_reference_summary", reference),
        ("comparison_summary", comparison),
        ("run_manifest", {"study": "baseline"}),
    ):
        (tmp_path / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    write_baseline_hierarchy(
        output=tmp_path,
        bounded=bounded,
        reference=reference,
        comparison=comparison,
        manifest={"study": "baseline"},
    )
    assert "Nominal lap-time penalty" in (tmp_path / "SUMMARY.md").read_text()
    assert "counterfactual" in (tmp_path / "REPORT.md").read_text()
    before = (tmp_path / "REPORT.md").read_text()
    regenerate_baseline_reports(tmp_path)
    assert (tmp_path / "REPORT.md").read_text() == before


def test_study_report_regeneration_uses_only_machine_artifacts(tmp_path: Path) -> None:
    data = {
        "decision_summary": {
            "recommendation": "No design recommendation is produced by this study type.",
            "confidence": "exploratory_distribution",
            "numerical_quality_valid": True,
            "directionally_robust": False,
            "warnings": ["small sample"],
            "recommended_next_actions": ["run more scenarios"],
            "metric_winners": {},
        },
        "summary": {"numerical_quality": {"valid_for_decision": True}},
        "convergence": {"status": "not_applicable", "reason": "test"},
        "energy_accounting": {"designs": {}},
        "uncertainty_attribution": {"status": "suppressed", "warnings": []},
        "run_manifest": {
            "study_name": "uncertainty",
            "study_type": "full_uncertainty",
            "sampling_mode": "all",
            "scenario_count": 1,
            "design_point_count": 1,
            "paired_scenarios": True,
            "random_seed": 7,
        },
    }
    for name, value in data.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    regenerate_study_reports(tmp_path)
    assert "Engineering decision summary" in (tmp_path / "SUMMARY.md").read_text()
    assert "Measured track-based" in (tmp_path / "REPORT.md").read_text()
    assert (tmp_path / "appendix" / "README.md").is_file()


def test_numerical_validity_is_not_mislabelled_directional_robustness() -> None:
    decision = synthesize_decision(
        summary={"numerical_quality": {"valid_for_decision": True}},
        convergence={"status": "not_applicable"},
        attribution={"warnings": []},
        manifest={"study_type": "full_uncertainty", "scenario_count": 1},
    )
    assert decision["numerical_quality_valid"] is True
    assert decision["directionally_robust"] is False
    assert decision["evidence_ready"] is False
    assert decision["statistically_ready"] is False
    assert decision["decision_ready"] is False
    assert "valid_for_decision" not in decision


def test_evidence_assessment_preserves_project_and_track_warnings() -> None:
    assessment = assess_evidence(
        diagnostics=(
            Diagnostic(
                severity=Severity.WARNING,
                code="INHERITED_DEFAULTS_ACTIVE",
                message="Two vehicle inputs use broad defaults.",
            ),
        ),
        bundle={
            "evidence": {
                "review_records": [
                    {"recommendation": "accepted"},
                    {"recommendation": "recommended_review"},
                    {"recommendation": "must_fix"},
                ],
                "lap_summary": {
                    "records": [
                        {
                            "run_id": "run_1",
                            "vehicle_id": "vehicle_A",
                            "driver_id": "driver_1",
                        }
                    ]
                },
            },
            "simulation_contract": {
                "speed_gates": [
                    {
                        "active_by_default": True,
                        "confidence": {"cross_vehicle_status": "single_vehicle_neutral"},
                    }
                ]
            },
        },
    )
    assert assessment["ready"] is False
    assert assessment["project_warning_count"] == 1
    assert assessment["track_review_status_counts"]["must_fix"] == 1
    assert assessment["track_review_status_counts"]["recommended_review"] == 1
    assert any("cross-vehicle" in item for item in assessment["warnings"])
    assert len(assessment["blocking_reasons"]) == 3


def test_design_sweep_requires_numerical_evidence_and_statistical_readiness() -> None:
    designs = {}
    for value, identifier, time, energy in (
        (1.0, "low", 2.0, 20.0),
        (2.0, "middle", 1.0, 10.0),
        (3.0, "high", 3.0, 30.0),
    ):
        designs[identifier] = {
            "design_value_si": value,
            "lap_time_penalty_vs_infinite_s": {"median": time},
            "finite_ratio_opportunity_loss_energy_kj": {"median": energy},
            "paired_ranking.lap_time_penalty_vs_infinite_s": {
                "paired_win_fraction_bootstrap_95_low": 0.7
            },
            "paired_ranking.finite_ratio_opportunity_loss_energy_kj": {
                "paired_win_fraction_bootstrap_95_low": 0.7
            },
        }
    convergence = {
        identifier: {
            metric: {"status": "adequate_quick_check"}
            for metric in (
                "lap_time_penalty_vs_infinite_s",
                "finite_ratio_opportunity_loss_energy_kj",
            )
        }
        for identifier in designs
    }
    decision = synthesize_decision(
        summary={"numerical_quality": {"numerically_valid": True}, "designs": designs},
        convergence=convergence,
        attribution={"warnings": []},
        manifest={
            "study_type": "design_sweep",
            "scenario_count": 20,
            "evidence_assessment": {"ready": True, "warnings": []},
        },
    )
    assert decision["numerically_valid"] is True
    assert decision["evidence_ready"] is True
    assert decision["statistically_ready"] is True
    assert decision["directionally_robust"] is True
    assert decision["decision_ready"] is True


def _case(name: str, lap: float, opportunity: float) -> dict[str, object]:
    return {
        "case": name,
        "completed": True,
        "lap_time_s": lap,
        "average_speed_kmh": 40.0,
        "finite_ratio_opportunity_loss_energy_kj": opportunity,
        "energy_balance_relative_error": 0.001,
        "powertrain_energy_balance_relative_error": 0.002,
        "time_minimum_ratio_s": 1.0,
        "time_maximum_ratio_s": 2.0,
        "time_variable_ratio_s": 7.0,
        "drivetrain_loss_energy_kj": 1.0,
        "clutch_loss_energy_kj": 1.0,
        "tire_slip_loss_energy_kj": 1.0,
        "brake_loss_energy_kj": 1.0,
        "rolling_loss_energy_kj": 1.0,
        "aerodynamic_loss_energy_kj": 1.0,
        "obstacle_loss_energy_kj": 1.0,
    }
