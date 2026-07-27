from __future__ import annotations

import numpy as np
import pandas as pd

from cvt_track_study.config.uncertainty import UncertainQuantity
from cvt_track_study.uncertainty.registry import InputRegistry, RegisteredInput
from cvt_track_study.uncertainty.sampling import (
    CorrelationGroup,
    SamplingPlan,
    ScenarioSampler,
)


def _quantity() -> UncertainQuantity:
    return UncertainQuantity.from_mapping(
        {
            "nominal": 0.5,
            "unit": "1",
            "source": {
                "kind": "engineering_estimate",
                "reference": "test",
            },
            "uncertainty": {
                "distribution": "uniform",
                "lower": 0.0,
                "upper": 1.0,
                "role": "structural",
            },
        }
    )


class _Bundle:
    active_speed_gates = [
        {
            "id": "g1",
            "target_speed_distribution": {
                "samples": [
                    {
                        "run_id": "fit",
                        "lap_id": 1,
                        "vehicle_id": "v1",
                        "driver_id": "d1",
                        "value_mps": 5.0,
                        "measurement_standard_deviation_mps": 0.1,
                        "speed_certainty": "native_high",
                    },
                    {
                        "run_id": "reconstructed",
                        "lap_id": 1,
                        "vehicle_id": "v2",
                        "driver_id": "d2",
                        "value_mps": 5.0,
                        "measurement_standard_deviation_mps": 1.0,
                        "speed_certainty": "reconstructed_low",
                    },
                ]
            },
        },
        {
            "id": "g2",
            "target_speed_distribution": {
                "samples": [
                    {
                        "run_id": "fit",
                        "lap_id": 1,
                        "vehicle_id": "v1",
                        "driver_id": "d1",
                        "value_mps": 7.0,
                        "measurement_standard_deviation_mps": 0.1,
                        "speed_certainty": "native_high",
                    },
                    {
                        "run_id": "reconstructed",
                        "lap_id": 1,
                        "vehicle_id": "v2",
                        "driver_id": "d2",
                        "value_mps": 7.0,
                        "measurement_standard_deviation_mps": 1.0,
                        "speed_certainty": "reconstructed_low",
                    },
                ]
            },
        },
    ]


def test_latin_hypercube_preserves_strata_and_correlation():
    quantity_a = _quantity()
    quantity_b = _quantity()
    registry = InputRegistry(
        (
            RegisteredInput("a", "structural", quantity_a),
            RegisteredInput("b", "structural", quantity_b),
        )
    )
    count = 200
    sampler = ScenarioSampler(
        registry=registry,
        bundle=_Bundle(),
        plan=SamplingPlan(
            mode="all_declared",
            replicates=count,
            random_seed=123,
            correlation_groups=(
                CorrelationGroup(
                    "family",
                    ("a", "b"),
                    np.asarray([[1.0, 0.8], [0.8, 1.0]]),
                ),
            ),
        ),
    )
    draws = sampler.draw_all()
    a = np.asarray([draw.quantity_values_si["a"] for draw in draws])
    b = np.asarray([draw.quantity_values_si["b"] for draw in draws])
    assert set(np.floor(a * count).astype(int)) == set(range(count))
    assert set(np.floor(b * count).astype(int)) == set(range(count))
    assert pd.Series(a).corr(pd.Series(b), method="spearman") > 0.65


def test_reconstructed_gate_speed_receives_wider_error_than_fit():
    registry = InputRegistry(())
    draws = ScenarioSampler(
        registry=registry,
        bundle=_Bundle(),
        plan=SamplingPlan(
            mode="measured_track",
            replicates=400,
            random_seed=456,
        ),
    ).draw_all()
    deviations = {"fit": [], "reconstructed": []}
    for draw in draws:
        deviations[draw.gate_sample_identity.run_id].append(
            draw.gate_target_speeds_mps["g1"] - 5.0
        )
    assert len(deviations["fit"]) > 100
    assert len(deviations["reconstructed"]) > 100
    assert np.std(deviations["reconstructed"]) > 5.0 * np.std(deviations["fit"])
