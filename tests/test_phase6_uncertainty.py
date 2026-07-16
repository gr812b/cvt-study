from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from cvt_track_study.bundle import TrackBundle
from cvt_track_study.config.uncertainty import UncertainChoice, UncertainQuantity
from cvt_track_study.contracts.obstacles import obstacle_model_alternatives
from cvt_track_study.uncertainty import (
    CorrelationGroup,
    InputRegistry,
    RegisteredInput,
    SamplingError,
    SamplingPlan,
    ScenarioSampler,
    paired_design_statistics,
    quantity_from_uniform,
    summarize_samples,
)


def quantity(
    nominal: float,
    uncertainty: dict,
    *,
    unit: str = "1",
    correlation_group: str | None = None,
) -> UncertainQuantity:
    raw = {
        "nominal": nominal,
        "unit": unit,
        "source": {"kind": "measured", "reference": "test fixture"},
        "uncertainty": uncertainty,
    }
    if correlation_group:
        raw["correlation_group"] = correlation_group
    return UncertainQuantity.from_mapping(raw)


def choice(nominal: str, uncertainty: dict) -> UncertainChoice:
    return UncertainChoice.from_mapping(
        {
            "nominal": nominal,
            "source": {"kind": "engineering_estimate", "reference": "test fixture"},
            "uncertainty": uncertainty,
        }
    )


def bundle_with_gates() -> TrackBundle:
    def gate(identifier: str, values: list[tuple[int, float]]) -> dict:
        return {
            "id": identifier,
            "active_by_default": True,
            "target_speed_distribution": {
                "samples": [
                    {
                        "run_id": "run_A",
                        "lap_id": lap,
                        "vehicle_id": "vehicle_A",
                        "driver_id": "driver_1",
                        "value_mps": value,
                    }
                    for lap, value in values
                ]
            },
        }

    return TrackBundle(
        {
            "schema_version": "1.2.0",
            "simulation_contract": {
                "track_length_m": 100.0,
                "speed_gates": [
                    gate("gate:A", [(1, 1.0), (2, 2.0), (3, 3.0)]),
                    gate("gate:B", [(1, 10.0), (2, 20.0), (3, 30.0)]),
                ],
                "physical_features": [],
                "response_groups": [],
            },
        }
    )


def test_numeric_distribution_transforms_are_deterministic_and_bounded() -> None:
    uniform = quantity(2.0, {"distribution": "uniform", "lower": 1.0, "upper": 3.0})
    triangular = quantity(
        2.0,
        {"distribution": "triangular", "lower": 1.0, "mode": 2.0, "upper": 5.0},
    )
    truncated = quantity(
        2.0,
        {
            "distribution": "truncated_normal",
            "standard_deviation": 1.0,
            "lower": 0.5,
            "upper": 3.5,
        },
    )
    assert quantity_from_uniform(uniform, 0.0) > 1.0
    assert quantity_from_uniform(uniform, 1.0) < 3.0
    assert quantity_from_uniform(triangular, 0.5) == pytest.approx(2.5505102572)
    draws = [quantity_from_uniform(truncated, u) for u in np.linspace(0, 1, 101)]
    assert min(draws) >= 0.5
    assert max(draws) <= 3.5


def test_gate_sampling_preserves_observed_lap_pairing() -> None:
    sampler = ScenarioSampler(
        registry=InputRegistry(()),
        bundle=bundle_with_gates(),
        plan=SamplingPlan(mode="measured_track", replicates=30, random_seed=42),
    )
    scenarios = sampler.draw_all()
    assert sampler.paired_gate_identity_count == 3
    for scenario in scenarios:
        assert scenario.gate_sample_identity is not None
        assert not scenario.independently_sampled_gate_ids
        assert scenario.gate_target_speeds_mps["gate:B"] == pytest.approx(
            10.0 * scenario.gate_target_speeds_mps["gate:A"]
        )


def test_scenario_sampling_is_reproducible_and_seed_changes_draws() -> None:
    registry = InputRegistry(
        (
            RegisteredInput(
                "vehicle.mass",
                "structural",
                quantity(200.0, {"distribution": "normal", "standard_deviation": 5.0}, unit="kg"),
            ),
        )
    )
    first = ScenarioSampler(
        registry=registry,
        bundle=bundle_with_gates(),
        plan=SamplingPlan(mode="all_declared", replicates=4, random_seed=123),
    ).draw_all()
    second = ScenarioSampler(
        registry=registry,
        bundle=bundle_with_gates(),
        plan=SamplingPlan(mode="all_declared", replicates=4, random_seed=123),
    ).draw_all()
    third = ScenarioSampler(
        registry=registry,
        bundle=bundle_with_gates(),
        plan=SamplingPlan(mode="all_declared", replicates=4, random_seed=124),
    ).draw_all()
    assert [row.serializable() for row in first] == [row.serializable() for row in second]
    assert [row.serializable() for row in first] != [row.serializable() for row in third]


def test_gaussian_copula_honours_declared_positive_correlation() -> None:
    registry = InputRegistry(
        (
            RegisteredInput(
                "vehicle.a",
                "structural",
                quantity(
                    0.0,
                    {"distribution": "normal", "standard_deviation": 1.0},
                    correlation_group="pair",
                ),
            ),
            RegisteredInput(
                "vehicle.b",
                "structural",
                quantity(
                    0.0,
                    {"distribution": "normal", "standard_deviation": 1.0},
                    correlation_group="pair",
                ),
            ),
        )
    )
    group = CorrelationGroup(
        "pair",
        ("vehicle.a", "vehicle.b"),
        np.asarray([[1.0, 0.85], [0.85, 1.0]]),
    )
    draws = ScenarioSampler(
        registry=registry,
        bundle=bundle_with_gates(),
        plan=SamplingPlan(
            mode="all_declared",
            replicates=1500,
            random_seed=55,
            correlation_groups=(group,),
        ),
    ).draw_all()
    a = [row.quantity_values_si["vehicle.a"] for row in draws]
    b = [row.quantity_values_si["vehicle.b"] for row in draws]
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(0.85, abs=0.04)


def test_invalid_correlation_matrix_is_rejected() -> None:
    registry = InputRegistry(
        (
            RegisteredInput(
                "vehicle.a",
                "structural",
                quantity(1.0, {"distribution": "uniform", "lower": 0.0, "upper": 2.0}),
            ),
            RegisteredInput(
                "vehicle.b",
                "structural",
                quantity(1.0, {"distribution": "uniform", "lower": 0.0, "upper": 2.0}),
            ),
        )
    )
    with pytest.raises(SamplingError, match="positive semidefinite"):
        ScenarioSampler(
            registry=registry,
            bundle=bundle_with_gates(),
            plan=SamplingPlan(
                mode="all_declared",
                replicates=1,
                random_seed=1,
                correlation_groups=(
                    CorrelationGroup(
                        "bad",
                        ("vehicle.a", "vehicle.b"),
                        np.asarray([[1.0, 2.0], [2.0, 1.0]]),
                    ),
                ),
            ),
        )


def test_paired_rankings_split_ties_and_compute_regret() -> None:
    rows = [
        {"design": "A", "rep": 0, "metric": 1.0},
        {"design": "B", "rep": 0, "metric": 2.0},
        {"design": "A", "rep": 1, "metric": 3.0},
        {"design": "B", "rep": 1, "metric": 2.0},
        {"design": "A", "rep": 2, "metric": 1.0},
        {"design": "B", "rep": 2, "metric": 1.0},
    ]
    result = paired_design_statistics(
        rows,
        design_key="design",
        replicate_key="rep",
        metric="metric",
        bootstrap_seed=9,
        bootstrap_resamples=300,
    )
    assert result["A"]["paired_win_fraction"] == pytest.approx(0.5)
    assert result["B"]["paired_win_fraction"] == pytest.approx(0.5)
    assert result["A"]["paired_win_fraction_bootstrap_95_low"] <= 0.5
    assert result["A"]["paired_win_fraction_bootstrap_95_high"] >= 0.5
    assert result["A"]["mean_paired_regret"] == pytest.approx(1 / 3)
    assert result["B"]["mean_paired_regret"] == pytest.approx(1 / 3)


def test_bootstrap_interval_is_separate_from_physical_percentile_band() -> None:
    result = summarize_samples(np.arange(1.0, 21.0), bootstrap_seed=5, bootstrap_resamples=500)
    assert result.p10 < result.median < result.p90
    assert result.median_bootstrap_low <= result.median <= result.median_bootstrap_high
    assert result.count == 20


def test_discrete_obstacle_model_requires_complete_parameter_contract_per_branch() -> None:
    raw = {
        "status": "declared",
        "model_type": {
            "nominal": "none",
            "source": {"kind": "engineering_estimate", "reference": "video review"},
            "uncertainty": {
                "distribution": "discrete",
                "role": "structural",
                "choices": ["none", "speed_quadratic_energy"],
                "probabilities": [0.25, 0.75],
            },
        },
        "alternatives": {
            "none": {"parameters": {}},
            "speed_quadratic_energy": {
                "parameters": {
                    "specific_fixed_energy": {
                        "nominal": 0.2,
                        "unit": "J/kg",
                        "source": {"kind": "engineering_estimate", "reference": "video"},
                        "uncertainty": {"distribution": "uniform", "role": "structural", "lower": 0.0, "upper": 0.5},
                    },
                    "impact_coefficient": {
                        "nominal": 5.0,
                        "unit": "kg",
                        "source": {"kind": "engineering_estimate", "reference": "video"},
                        "uncertainty": {"distribution": "uniform", "role": "structural", "lower": 1.0, "upper": 10.0},
                    },
                }
            },
        },
    }
    model_choice, alternatives = obstacle_model_alternatives(raw)
    assert model_choice.nominal == "none"
    assert set(alternatives) == {"none", "speed_quadratic_energy"}
    broken = deepcopy(raw)
    del broken["alternatives"]["speed_quadratic_energy"]["parameters"]["impact_coefficient"]
    with pytest.raises(ValueError, match="missing impact_coefficient"):
        obstacle_model_alternatives(broken)


def test_single_sample_convergence_uses_json_safe_null_for_standard_error() -> None:
    from cvt_track_study.uncertainty import convergence_diagnostics

    result = convergence_diagnostics([1.0])
    assert result["mean_monte_carlo_standard_error"] is None
    import json

    json.dumps(result, allow_nan=False)


def test_obstacle_coefficient_uncertainty_must_have_nonnegative_support() -> None:
    raw = {
        "status": "declared",
        "model_type": {
            "nominal": "distributed_resistance",
            "source": {"kind": "engineering_estimate", "reference": "test"},
            "uncertainty": {"distribution": "fixed", "reason": "test model"},
        },
        "parameters": {
            "resistance_force": {
                "nominal": 100.0,
                "unit": "N",
                "source": {"kind": "engineering_estimate", "reference": "test"},
                "uncertainty": {
                    "distribution": "normal",
                    "role": "structural",
                    "standard_deviation": 5.0,
                },
            }
        },
    }
    with pytest.raises(Exception, match="explicit non-negative support"):
        obstacle_model_alternatives(raw)


def test_smooth_profile_scale_supports_must_never_cross() -> None:
    def q(nominal: float, lower: float, upper: float) -> dict:
        return {
            "nominal": nominal,
            "unit": "1",
            "source": {"kind": "engineering_estimate", "reference": "test"},
            "uncertainty": {
                "distribution": "uniform",
                "role": "structural",
                "lower": lower,
                "upper": upper,
            },
        }

    fixed = lambda nominal, unit: {
        "nominal": nominal,
        "unit": unit,
        "source": {"kind": "engineering_estimate", "reference": "test"},
        "uncertainty": {"distribution": "fixed", "reason": "test"},
    }
    raw = {
        "status": "declared",
        "model_type": {
            "nominal": "smooth_profile",
            "source": {"kind": "engineering_estimate", "reference": "test"},
            "uncertainty": {"distribution": "fixed", "reason": "test model"},
        },
        "parameters": {
            "vertical_amplitude": fixed(0.1, "m"),
            "specific_fixed_energy": fixed(0.0, "J/kg"),
            "impact_coefficient": fixed(0.0, "kg"),
            "traction_multiplier": q(1.0, 0.9, 1.1),
            "minimum_normal_load_scale": q(0.6, 0.5, 0.8),
            "maximum_normal_load_scale": q(1.0, 0.7, 1.2),
        },
    }
    with pytest.raises(Exception, match="supports.*overlap"):
        obstacle_model_alternatives(raw)


def test_selected_structural_sampling_only_draws_requested_paths() -> None:
    registry = InputRegistry(
        (
            RegisteredInput(
                "vehicle.mass",
                "structural",
                quantity(200.0, {"distribution": "uniform", "lower": 190.0, "upper": 210.0}, unit="kg"),
            ),
            RegisteredInput(
                "vehicle.aero.drag_area",
                "structural",
                quantity(0.7, {"distribution": "uniform", "lower": 0.5, "upper": 0.9}, unit="m^2"),
            ),
        )
    )
    draws = ScenarioSampler(
        registry=registry,
        bundle=bundle_with_gates(),
        plan=SamplingPlan(
            mode="selected_structural",
            replicates=2,
            random_seed=7,
            selected_paths=("vehicle.mass",),
        ),
    ).draw_all()
    assert all(set(draw.quantity_values_si) == {"vehicle.mass"} for draw in draws)
    assert all(not draw.gate_target_speeds_mps for draw in draws)


def test_stochastic_obstacle_inputs_require_explicit_semantic_role() -> None:
    raw = {
        "status": "declared",
        "model_type": {
            "nominal": "distributed_resistance",
            "source": {"kind": "engineering_estimate", "reference": "test"},
            "uncertainty": {"distribution": "fixed", "reason": "test model"},
        },
        "parameters": {
            "resistance_force": {
                "nominal": 100.0,
                "unit": "N",
                "source": {"kind": "engineering_estimate", "reference": "test"},
                "uncertainty": {
                    "distribution": "uniform",
                    "lower": 50.0,
                    "upper": 150.0,
                },
            }
        },
    }
    with pytest.raises(Exception, match="must declare uncertainty.role"):
        obstacle_model_alternatives(raw)


def test_measured_track_mode_only_samples_obstacles_declared_as_measured_variation() -> None:
    def obstacle(feature_id: str, role: str) -> dict:
        return {
            "id": feature_id,
            "obstacle_model": {
                "status": "declared",
                "model_type": {
                    "nominal": "distributed_resistance",
                    "source": {"kind": "engineering_estimate", "reference": "test"},
                    "uncertainty": {"distribution": "fixed", "reason": "test model"},
                },
                "parameters": {
                    "resistance_force": {
                        "nominal": 100.0,
                        "unit": "N",
                        "source": {"kind": "engineering_estimate", "reference": "test"},
                        "uncertainty": {
                            "distribution": "uniform",
                            "role": role,
                            "lower": 50.0,
                            "upper": 150.0,
                        },
                    }
                },
            },
        }

    from cvt_track_study.uncertainty import build_input_registry

    bundle = TrackBundle(
        {
            "schema_version": "1.2.0",
            "simulation_contract": {
                "track_length_m": 100.0,
                "speed_gates": [],
                "response_groups": [],
                "physical_features": [
                    obstacle("structural_feature", "structural"),
                    obstacle("varying_feature", "measured_track"),
                ],
            },
        }
    )
    registry = build_input_registry(
        vehicle_raw={}, base_study_raw={}, track_raw={}, bundle=bundle
    )
    sampler = ScenarioSampler(
        registry=registry,
        bundle=bundle,
        plan=SamplingPlan(mode="measured_track", replicates=2, random_seed=7),
    )
    assert sampler.sampled_paths == (
        "obstacle.varying_feature.distributed_resistance.resistance_force",
    )
    draws = sampler.draw_all()
    assert all(
        "obstacle.structural_feature.distributed_resistance.resistance_force"
        not in draw.quantity_values_si
        for draw in draws
    )


def test_selected_structural_rejects_non_structural_role() -> None:
    registry = InputRegistry(
        (
            RegisteredInput(
                "obstacle.a.distributed_resistance.resistance_force",
                "measured_track",
                quantity(
                    100.0,
                    {
                        "distribution": "uniform",
                        "role": "measured_track",
                        "lower": 50.0,
                        "upper": 150.0,
                    },
                    unit="N",
                ),
            ),
        )
    )
    with pytest.raises(SamplingError, match="only inputs with uncertainty.role='structural'"):
        ScenarioSampler(
            registry=registry,
            bundle=bundle_with_gates(),
            plan=SamplingPlan(
                mode="selected_structural",
                replicates=1,
                random_seed=7,
                selected_paths=(
                    "obstacle.a.distributed_resistance.resistance_force",
                ),
            ),
        )


def test_fixed_inputs_without_gate_variation_collapse_to_deterministic_scenarios() -> None:
    registry = InputRegistry(
        (
            RegisteredInput(
                "vehicle.mass",
                "structural",
                quantity(
                    200.0,
                    {"distribution": "fixed", "reason": "deterministic contract test"},
                    unit="kg",
                ),
            ),
        )
    )
    bundle = TrackBundle(
        {
            "schema_version": "1.2.0",
            "simulation_contract": {
                "track_length_m": 100.0,
                "speed_gates": [],
                "physical_features": [],
                "response_groups": [],
            },
        }
    )
    scenarios = ScenarioSampler(
        registry=registry,
        bundle=bundle,
        plan=SamplingPlan(mode="all_declared", replicates=3, random_seed=11),
    ).draw_all()
    assert all(not scenario.quantity_values_si for scenario in scenarios)
    assert all(not scenario.choice_values for scenario in scenarios)
    assert all(not scenario.gate_target_speeds_mps for scenario in scenarios)
