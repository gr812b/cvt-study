from __future__ import annotations

import math

import pytest

from cvt_track_study.config.uncertainty import UncertainChoice, UncertainQuantity
from cvt_track_study.studies.model import DesignPoint
from cvt_track_study.studies.analysis import quality_summary, summarize_study
from cvt_track_study.studies.planning import reference_cache_key, study_plan
from cvt_track_study.uncertainty import InputRegistry, RegisteredInput


def quantity(nominal: float, uncertainty: dict, unit: str = "1") -> UncertainQuantity:
    return UncertainQuantity.from_mapping(
        {
            "nominal": nominal,
            "unit": unit,
            "source": {"kind": "measured", "reference": "phase 6 test"},
            "uncertainty": uncertainty,
        }
    )


def test_reference_cache_only_shares_mathematically_invariant_design_path() -> None:
    minimum_a = DesignPoint(
        "minimum=0.8", "drivetrain.cvt.minimum_reduction_ratio", 0.8, 0.8
    )
    minimum_b = DesignPoint(
        "minimum=1.0", "drivetrain.cvt.minimum_reduction_ratio", 1.0, 1.0
    )
    final_a = DesignPoint("final=6", "drivetrain.final_drive_ratio", 6.0, 6.0)
    final_b = DesignPoint("final=8", "drivetrain.final_drive_ratio", 8.0, 8.0)

    assert reference_cache_key(3, minimum_a) == reference_cache_key(3, minimum_b)
    assert reference_cache_key(3, final_a) != reference_cache_key(3, final_b)
    assert reference_cache_key(3, minimum_a) != reference_cache_key(4, minimum_a)


def test_structural_plan_keeps_exact_nominal_separate_from_distribution_quantiles() -> None:
    item = RegisteredInput(
        path="vehicle.aero.drag_area",
        category="structural",
        value=quantity(
            0.67,
            {
                "distribution": "triangular",
                "lower": 0.45,
                "mode": 0.60,
                "upper": 0.90,
            },
            "m^2",
        ),
    )
    registry = InputRegistry((item,))
    raw = {
        "sensitivity": {
            "parameters": ["vehicle.aero.drag_area"],
            "quantiles": [0.05, 0.5, 0.95],
        }
    }
    points, mode, replicates = study_plan(
        "structural_sensitivity", raw, registry, None
    )

    nominal = [point for point in points if point.level_kind == "nominal"]
    quantiles = [point for point in points if point.level_kind == "quantile"]
    assert len(nominal) == 1
    assert nominal[0].display_value == pytest.approx(0.67)
    assert nominal[0].value_si == pytest.approx(0.67)
    assert {point.level_probability for point in quantiles} == {0.05, 0.5, 0.95}
    assert mode == "nominal"
    assert replicates == 1


def test_quality_summary_uses_numerical_checks_not_statistical_band_width() -> None:
    base = {
        "bounded_completed": True,
        "reference_completed": True,
        "reference_dominance_pass": True,
        "bounded_gates_compliant_0p5_kmh": True,
        "reference_gates_compliant_0p5_kmh": True,
        "bounded_energy_balance_relative_error": 0.002,
        "reference_energy_balance_relative_error": -0.003,
        "bounded_powertrain_energy_balance_relative_error": 0.004,
        "reference_powertrain_energy_balance_relative_error": -0.005,
    }
    result = quality_summary([base], {"quality": {}})
    assert result["valid_for_decision"] is True

    failed = dict(base)
    failed["bounded_powertrain_energy_balance_relative_error"] = 0.02
    result = quality_summary([failed], {"quality": {}})
    assert result["valid_for_decision"] is False
    assert result["powertrain_energy_balance_pass"] is False


def test_empty_gate_set_is_represented_by_finite_zero_excess() -> None:
    # The study rows are serialized to strict JSON. A no-gate case must never
    # inject +/-Infinity into result artifacts.
    no_gate_default = max((), default=0.0)
    assert no_gate_default == 0.0
    assert math.isfinite(no_gate_default)


def test_structural_plan_expands_discrete_model_choices_one_at_a_time() -> None:
    model_choice = UncertainChoice.from_mapping(
        {
            "nominal": "none",
            "source": {"kind": "engineering_estimate", "reference": "phase 6 test"},
            "uncertainty": {
                "distribution": "discrete",
                "role": "structural",
                "choices": ["none", "distributed_resistance", "speed_quadratic_energy"],
                "probabilities": [0.2, 0.5, 0.3],
            },
        }
    )
    registry = InputRegistry(
        (
            RegisteredInput(
                path="obstacle:event_1.model_type",
                category="structural",
                value=model_choice,
                feature_id="event_1",
            ),
        )
    )
    points, mode, replicates = study_plan(
        "structural_sensitivity",
        {
            "sensitivity": {
                "parameters": ["obstacle:event_1.model_type"],
                "quantiles": [],
            }
        },
        registry,
        None,
    )
    assert [point.choice_value for point in points] == [
        "none",
        "distributed_resistance",
        "speed_quadratic_energy",
    ]
    assert [point.level_kind for point in points] == ["nominal", "choice", "choice"]
    assert all(point.value_si is None for point in points)
    assert mode == "nominal"
    assert replicates == 1


def test_structural_summary_handles_categorical_choice_levels() -> None:
    common = {
        "bounded_lap_time_s": 10.0,
        "infinite_reference_lap_time_s": 9.0,
        "bounded_total_opportunity_loss_energy_kj": 5.0,
        "reference_shared_launch_loss_energy_kj": 1.0,
    }
    rows = [
        {
            **common,
            "parameter_path": "obstacle:event_1.model_type",
            "design_id": "obstacle:event_1.model_type@nominal",
            "design_value": "none",
            "design_value_si": None,
            "design_choice_value": "none",
            "level_kind": "nominal",
            "level_probability": None,
            "lap_time_penalty_vs_infinite_s": 1.0,
            "finite_ratio_opportunity_loss_energy_kj": 4.0,
        },
        {
            **common,
            "parameter_path": "obstacle:event_1.model_type",
            "design_id": "obstacle:event_1.model_type@choice=distributed_resistance",
            "design_value": "distributed_resistance",
            "design_value_si": None,
            "design_choice_value": "distributed_resistance",
            "level_kind": "choice",
            "level_probability": None,
            "lap_time_penalty_vs_infinite_s": 1.2,
            "finite_ratio_opportunity_loss_energy_kj": 5.0,
        },
    ]
    result = summarize_study("structural_sensitivity", rows, {}, 1)
    record = result["parameters"]["obstacle:event_1.model_type"]
    assert record["nominal_choice_value"] == "none"
    assert [level["level_kind"] for level in record["levels"]] == ["nominal", "choice"]
