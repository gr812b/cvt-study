from math import isclose

import pytest

from cvt_track_study.config import (
    DistributionKind,
    SourceKind,
    SourceSpec,
    UncertainChoice,
    UncertainQuantity,
    UncertaintySpec,
    UncertaintyValidationError,
)


def test_fixed_requires_explicit_reason() -> None:
    with pytest.raises(UncertaintyValidationError, match="explicit reason"):
        UncertaintySpec(distribution=DistributionKind.FIXED).validate()


def test_fixed_with_reason_is_valid() -> None:
    UncertaintySpec(
        distribution=DistributionKind.FIXED,
        reason="Exact tooth-count ratio.",
    ).validate()


def test_normal_confidence_band_resolves_to_sigma() -> None:
    uncertainty = UncertaintySpec(
        distribution=DistributionKind.NORMAL,
        confidence_half_width=1.96,
        confidence_level=0.95,
    )
    sigma = uncertainty.standard_deviation_for(10.0)
    assert isclose(sigma, 1.0, rel_tol=2e-3)


def test_quantity_rejects_missing_uncertainty() -> None:
    with pytest.raises(UncertaintyValidationError, match="uncertainty"):
        UncertainQuantity.from_mapping(
            {
                "nominal": 245.0,
                "unit": "kg",
                "source": {"kind": "measured", "reference": "scale"},
            }
        )


def test_quantity_from_mapping_preserves_provenance() -> None:
    quantity = UncertainQuantity.from_mapping(
        {
            "nominal": 245.0,
            "unit": "kg",
            "source": {
                "kind": "measured",
                "reference": "four-corner scales",
            },
            "uncertainty": {
                "distribution": "normal",
                "standard_deviation": 0.1,
            },
            "correlation_group": "scale_session_2026_07_10",
        }
    )
    assert quantity.source.kind is SourceKind.MEASURED
    assert quantity.correlation_group == "scale_session_2026_07_10"


def test_nominal_must_lie_inside_bounds() -> None:
    quantity = UncertainQuantity(
        nominal=5.0,
        unit="m",
        source=SourceSpec(SourceKind.MEASURED, "fixture"),
        uncertainty=UncertaintySpec(
            distribution=DistributionKind.UNIFORM,
            lower=6.0,
            upper=7.0,
        ),
    )
    with pytest.raises(UncertaintyValidationError, match="inside"):
        quantity.validate()


def test_discrete_probabilities_must_sum_to_one() -> None:
    with pytest.raises(UncertaintyValidationError, match="sum to one"):
        UncertaintySpec(
            distribution=DistributionKind.DISCRETE,
            choices=("low", "high"),
            probabilities=(0.3, 0.3),
        ).validate()


def test_uniform_rejects_normal_spread_parameters() -> None:
    with pytest.raises(UncertaintyValidationError, match="normal spread"):
        UncertaintySpec(
            distribution=DistributionKind.UNIFORM,
            lower=0.0,
            upper=1.0,
            standard_deviation=0.2,
        ).validate()


def test_numeric_quantity_rejects_discrete_model_choices() -> None:
    with pytest.raises(UncertaintyValidationError, match="UncertainChoice"):
        UncertainQuantity.from_mapping(
            {
                "nominal": 1.0,
                "unit": "1",
                "source": {"kind": "engineering_estimate", "reference": "test"},
                "uncertainty": {
                    "distribution": "discrete",
                    "choices": ["low", "high"],
                },
            }
        )


def test_relative_spread_rejects_zero_nominal() -> None:
    with pytest.raises(UncertaintyValidationError, match="zero nominal"):
        UncertainQuantity.from_mapping(
            {
                "nominal": 0.0,
                "unit": "m",
                "source": {"kind": "measured", "reference": "test"},
                "uncertainty": {
                    "distribution": "normal",
                    "relative_standard_deviation": 0.1,
                },
            }
        )


def test_categorical_choice_requires_nominal_among_choices() -> None:
    choice = UncertainChoice.from_mapping(
        {
            "nominal": "impact",
            "source": {
                "kind": "engineering_estimate",
                "reference": "video review",
            },
            "uncertainty": {
                "distribution": "discrete",
                "choices": ["impact", "roughness"],
                "probabilities": [0.6, 0.4],
            },
        }
    )
    assert choice.nominal == "impact"


def test_categorical_choice_rejects_numeric_distribution() -> None:
    with pytest.raises(UncertaintyValidationError, match="fixed or discrete"):
        UncertainChoice.from_mapping(
            {
                "nominal": "impact",
                "source": {
                    "kind": "engineering_estimate",
                    "reference": "video review",
                },
                "uncertainty": {
                    "distribution": "uniform",
                    "lower": 0.0,
                    "upper": 1.0,
                },
            }
        )


def test_uncertainty_role_is_parsed_and_rejects_unknown_values() -> None:
    from cvt_track_study.config import UncertaintyRole

    quantity = UncertainQuantity.from_mapping(
        {
            "nominal": 2.0,
            "unit": "m",
            "source": {"kind": "measured", "reference": "repeated track survey"},
            "uncertainty": {
                "distribution": "uniform",
                "role": "measured_track",
                "lower": 1.5,
                "upper": 2.5,
            },
        }
    )
    assert quantity.uncertainty.role is UncertaintyRole.MEASURED_TRACK

    with pytest.raises(UncertaintyValidationError, match="uncertainty.role"):
        UncertainQuantity.from_mapping(
            {
                "nominal": 2.0,
                "unit": "m",
                "source": {"kind": "measured", "reference": "test"},
                "uncertainty": {
                    "distribution": "uniform",
                    "role": "mystery",
                    "lower": 1.5,
                    "upper": 2.5,
                },
            }
        )
