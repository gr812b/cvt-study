"""Probability transforms for uncertainty-aware quantities and choices.

All stochastic inputs are sampled from a unit-uniform variate.  This makes the
individual distributions composable with a Gaussian copula for declared input
correlations while retaining deterministic reproducibility from one random seed.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np
from scipy.stats import norm, truncnorm

from cvt_track_study.config.uncertainty import (
    DistributionKind,
    UncertainChoice,
    UncertainQuantity,
    UncertaintyValidationError,
)
from cvt_track_study.config.units import convert_to_si

_EPS = np.finfo(float).eps


def quantity_from_uniform(quantity: UncertainQuantity, u: float) -> float:
    """Transform ``u`` in ``[0, 1]`` into one SI-valued quantity draw."""

    value = quantity_in_declared_units_from_uniform(quantity, u)
    return float(convert_to_si(value, quantity.unit)[0])


def quantity_in_declared_units_from_uniform(
    quantity: UncertainQuantity, u: float
) -> float:
    """Transform ``u`` into a draw expressed in the quantity's declared units."""

    probability = _open_unit_interval(u)
    spec = quantity.uncertainty
    nominal = quantity.nominal
    kind = spec.distribution
    if kind is DistributionKind.FIXED:
        return float(nominal)
    if kind is DistributionKind.NORMAL:
        return float(norm.ppf(probability, loc=nominal, scale=spec.standard_deviation_for(nominal)))
    if kind is DistributionKind.TRUNCATED_NORMAL:
        sigma = spec.standard_deviation_for(nominal)
        assert spec.lower is not None and spec.upper is not None
        a = (spec.lower - nominal) / sigma
        b = (spec.upper - nominal) / sigma
        return float(truncnorm.ppf(probability, a, b, loc=nominal, scale=sigma))
    if kind is DistributionKind.UNIFORM:
        assert spec.lower is not None and spec.upper is not None
        return float(spec.lower + probability * (spec.upper - spec.lower))
    if kind is DistributionKind.TRIANGULAR:
        assert spec.lower is not None and spec.upper is not None and spec.mode is not None
        return _triangular_ppf(probability, spec.lower, spec.mode, spec.upper)
    if kind is DistributionKind.EMPIRICAL:
        return _empirical_ppf(probability, np.asarray(spec.samples, dtype=float))
    raise UncertaintyValidationError(
        f"Numeric quantity cannot be sampled from {kind.value!r}."
    )


def choice_from_uniform(choice: UncertainChoice, u: float) -> str:
    """Transform ``u`` into a categorical model choice."""

    spec = choice.uncertainty
    if spec.distribution is DistributionKind.FIXED:
        return choice.nominal
    if spec.distribution is not DistributionKind.DISCRETE:
        raise UncertaintyValidationError(
            f"Categorical choice cannot be sampled from {spec.distribution.value!r}."
        )
    probability = _open_unit_interval(u)
    weights = (
        np.asarray(spec.probabilities, dtype=float)
        if spec.probabilities
        else np.full(len(spec.choices), 1.0 / len(spec.choices), dtype=float)
    )
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(cumulative, probability, side="right"))
    return spec.choices[min(index, len(spec.choices) - 1)]


def quantity_quantile_si(quantity: UncertainQuantity, probability: float) -> float:
    """Return a deterministic SI-valued quantile for sensitivity level creation."""

    return quantity_from_uniform(quantity, probability)


def is_stochastic(value: UncertainQuantity | UncertainChoice) -> bool:
    return value.uncertainty.distribution is not DistributionKind.FIXED


def _triangular_ppf(u: float, lower: float, mode: float, upper: float) -> float:
    width = upper - lower
    if width <= 0.0:
        raise UncertaintyValidationError("Triangular bounds must have positive width.")
    split = (mode - lower) / width
    if u < split:
        return float(lower + np.sqrt(u * width * (mode - lower)))
    return float(upper - np.sqrt((1.0 - u) * width * (upper - mode)))


def _empirical_ppf(u: float, samples: np.ndarray) -> float:
    if samples.ndim != 1 or samples.size < 2 or not np.all(np.isfinite(samples)):
        raise UncertaintyValidationError("Empirical samples must be a finite 1-D array.")
    ordered = np.sort(samples)
    # Empirical evidence is resampled without inventing values between observed
    # points. A copula can still correlate the selected ranks across inputs.
    index = min(int(np.floor(u * ordered.size)), ordered.size - 1)
    return float(ordered[index])


def _open_unit_interval(value: Any) -> float:
    try:
        u = float(value)
    except (TypeError, ValueError) as exc:
        raise UncertaintyValidationError("Uniform variate must be numeric.") from exc
    if not isfinite(u) or not 0.0 <= u <= 1.0:
        raise UncertaintyValidationError("Uniform variate must lie in [0, 1].")
    return float(np.clip(u, _EPS, 1.0 - _EPS))
