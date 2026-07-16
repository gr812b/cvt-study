"""Uncertainty-first contracts for physical quantities and model choices.

A physical numeric input cannot enter the clean pipeline as a bare float. It must
have a nominal value, units, provenance, and an explicit uncertainty declaration.
Treating a quantity as exact is supported only through a justified ``fixed``
declaration. Discrete model alternatives use :class:`UncertainChoice`, keeping
unit-bearing numeric quantities distinct from categorical model-form uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from statistics import NormalDist
from typing import Any, Mapping

from .units import UnitValidationError, convert_to_si, get_unit


class UncertaintyValidationError(ValueError):
    """Raised when an uncertainty declaration is incomplete or contradictory."""


class DistributionKind(str, Enum):
    FIXED = "fixed"
    NORMAL = "normal"
    TRUNCATED_NORMAL = "truncated_normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    EMPIRICAL = "empirical"
    DISCRETE = "discrete"


class UncertaintyRole(str, Enum):
    """Semantic role controlling which studies sample an uncertain input.

    ``structural`` represents uncertainty in the physical/model contract itself.
    ``measured_track`` represents repeatable variation evidenced by track data.
    ``initial_condition`` represents uncertainty in the state at simulation start.
    """

    STRUCTURAL = "structural"
    MEASURED_TRACK = "measured_track"
    INITIAL_CONDITION = "initial_condition"


class SourceKind(str, Enum):
    MEASURED = "measured"
    MANUFACTURER = "manufacturer"
    DERIVED = "derived"
    CALIBRATED = "calibrated"
    ENGINEERING_ESTIMATE = "engineering_estimate"
    INHERITED_DEFAULT = "inherited_default"


@dataclass(frozen=True)
class SourceSpec:
    kind: SourceKind
    reference: str
    notes: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SourceSpec":
        try:
            kind = SourceKind(str(raw["kind"]))
        except (KeyError, ValueError) as exc:
            valid = ", ".join(item.value for item in SourceKind)
            raise UncertaintyValidationError(
                f"source.kind must be one of: {valid}."
            ) from exc
        source = cls(
            kind=kind,
            reference=str(raw.get("reference", "")),
            notes=str(raw.get("notes", "")),
        )
        source.validate()
        return source

    def validate(self) -> None:
        if not self.reference.strip():
            raise UncertaintyValidationError(
                "Every quantity or choice needs a non-empty source reference."
            )


@dataclass(frozen=True)
class UncertaintySpec:
    """Declaration of uncertainty around a nominal value or model choice."""

    distribution: DistributionKind
    standard_deviation: float | None = None
    relative_standard_deviation: float | None = None
    confidence_half_width: float | None = None
    confidence_level: float | None = None
    lower: float | None = None
    upper: float | None = None
    mode: float | None = None
    samples: tuple[float, ...] = field(default_factory=tuple)
    choices: tuple[str, ...] = field(default_factory=tuple)
    probabilities: tuple[float, ...] = field(default_factory=tuple)
    reason: str = ""
    role: UncertaintyRole | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "UncertaintySpec":
        try:
            kind = DistributionKind(str(raw["distribution"]))
        except KeyError as exc:
            raise UncertaintyValidationError(
                "Uncertainty declaration is missing 'distribution'."
            ) from exc
        except ValueError as exc:
            valid = ", ".join(item.value for item in DistributionKind)
            raise UncertaintyValidationError(
                f"Unknown uncertainty distribution. Expected one of: {valid}."
            ) from exc

        spec = cls(
            distribution=kind,
            standard_deviation=_optional_float(raw.get("standard_deviation")),
            relative_standard_deviation=_optional_float(
                raw.get("relative_standard_deviation")
            ),
            confidence_half_width=_optional_float(raw.get("confidence_half_width")),
            confidence_level=_optional_float(raw.get("confidence_level")),
            lower=_optional_float(raw.get("lower")),
            upper=_optional_float(raw.get("upper")),
            mode=_optional_float(raw.get("mode")),
            samples=_float_sequence(raw.get("samples", ()), "samples"),
            choices=_string_sequence(raw.get("choices", ()), "choices"),
            probabilities=_float_sequence(raw.get("probabilities", ()), "probabilities"),
            reason=str(raw.get("reason", "")),
            role=_optional_role(raw.get("role")),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        kind = self.distribution

        if kind is DistributionKind.FIXED:
            if not self.reason.strip():
                raise UncertaintyValidationError(
                    "A fixed (zero-uncertainty) value requires an explicit reason."
                )
            self._reject_all_spread_parameters("fixed")
            return

        if kind in {DistributionKind.NORMAL, DistributionKind.TRUNCATED_NORMAL}:
            forms = [
                self.standard_deviation is not None,
                self.relative_standard_deviation is not None,
                self.confidence_half_width is not None
                or self.confidence_level is not None,
            ]
            if sum(forms) != 1:
                raise UncertaintyValidationError(
                    "Normal uncertainty requires exactly one of: standard_deviation, "
                    "relative_standard_deviation, or confidence_half_width plus "
                    "confidence_level."
                )
            if self.standard_deviation is not None and self.standard_deviation <= 0:
                raise UncertaintyValidationError("standard_deviation must be positive.")
            if (
                self.relative_standard_deviation is not None
                and self.relative_standard_deviation <= 0
            ):
                raise UncertaintyValidationError(
                    "relative_standard_deviation must be positive."
                )
            if self.confidence_half_width is not None or self.confidence_level is not None:
                if self.confidence_half_width is None or self.confidence_level is None:
                    raise UncertaintyValidationError(
                        "confidence_half_width and confidence_level must be provided together."
                    )
                if self.confidence_half_width <= 0:
                    raise UncertaintyValidationError(
                        "confidence_half_width must be positive."
                    )
                if not 0 < self.confidence_level < 1:
                    raise UncertaintyValidationError(
                        "confidence_level must lie strictly between 0 and 1."
                    )
            self._reject_sequence_parameters("normal")
            if self.mode is not None:
                raise UncertaintyValidationError(
                    "Normal uncertainty cannot define a triangular mode."
                )
            if kind is DistributionKind.TRUNCATED_NORMAL:
                self._validate_bounds(required=True)
            elif self.lower is not None or self.upper is not None:
                raise UncertaintyValidationError(
                    "Use truncated_normal when lower or upper bounds are required."
                )
            return

        if kind is DistributionKind.UNIFORM:
            self._reject_normal_parameters("uniform")
            self._reject_sequence_parameters("uniform")
            if self.mode is not None:
                raise UncertaintyValidationError(
                    "Uniform uncertainty cannot define a mode."
                )
            self._validate_bounds(required=True)
            return

        if kind is DistributionKind.TRIANGULAR:
            self._reject_normal_parameters("triangular")
            self._reject_sequence_parameters("triangular")
            self._validate_bounds(required=True)
            if self.mode is None:
                raise UncertaintyValidationError("triangular uncertainty requires mode.")
            assert self.lower is not None and self.upper is not None
            if not self.lower <= self.mode <= self.upper:
                raise UncertaintyValidationError(
                    "triangular mode must lie between lower and upper."
                )
            return

        if kind is DistributionKind.EMPIRICAL:
            self._reject_normal_parameters("empirical")
            if any(value is not None for value in (self.lower, self.upper, self.mode)):
                raise UncertaintyValidationError(
                    "Empirical uncertainty cannot define bounds or mode."
                )
            if self.choices or self.probabilities:
                raise UncertaintyValidationError(
                    "Empirical uncertainty uses samples, not discrete choices."
                )
            if len(self.samples) < 2:
                raise UncertaintyValidationError(
                    "empirical uncertainty requires at least two samples."
                )
            if not all(isfinite(value) for value in self.samples):
                raise UncertaintyValidationError("empirical samples must be finite.")
            return

        if kind is DistributionKind.DISCRETE:
            self._reject_normal_parameters("discrete")
            if any(value is not None for value in (self.lower, self.upper, self.mode)):
                raise UncertaintyValidationError(
                    "Discrete uncertainty cannot define numeric bounds or mode."
                )
            if self.samples:
                raise UncertaintyValidationError(
                    "Discrete uncertainty uses choices, not empirical samples."
                )
            if len(self.choices) < 2:
                raise UncertaintyValidationError(
                    "discrete uncertainty requires at least two choices."
                )
            if len(set(self.choices)) != len(self.choices):
                raise UncertaintyValidationError("discrete choices must be unique.")
            if self.probabilities:
                if len(self.probabilities) != len(self.choices):
                    raise UncertaintyValidationError(
                        "probabilities must have the same length as choices."
                    )
                if any(value < 0 for value in self.probabilities):
                    raise UncertaintyValidationError(
                        "discrete probabilities cannot be negative."
                    )
                if abs(sum(self.probabilities) - 1.0) > 1e-9:
                    raise UncertaintyValidationError(
                        "discrete probabilities must sum to one."
                    )
            return

        raise AssertionError(f"Unhandled distribution kind: {kind}")

    def standard_deviation_for(self, nominal: float) -> float:
        """Resolve a normal declaration to an absolute standard deviation."""
        if self.distribution not in {
            DistributionKind.NORMAL,
            DistributionKind.TRUNCATED_NORMAL,
        }:
            raise UncertaintyValidationError(
                "standard_deviation_for is only valid for normal distributions."
            )
        self.validate()
        if self.standard_deviation is not None:
            return self.standard_deviation
        if self.relative_standard_deviation is not None:
            if nominal == 0:
                raise UncertaintyValidationError(
                    "relative_standard_deviation cannot be used with a zero nominal value."
                )
            return abs(nominal) * self.relative_standard_deviation
        assert self.confidence_half_width is not None
        assert self.confidence_level is not None
        quantile = NormalDist().inv_cdf((1.0 + self.confidence_level) / 2.0)
        return self.confidence_half_width / quantile

    def _validate_bounds(self, *, required: bool) -> None:
        if required and (self.lower is None or self.upper is None):
            raise UncertaintyValidationError("Both lower and upper bounds are required.")
        if self.lower is not None and self.upper is not None:
            if not isfinite(self.lower) or not isfinite(self.upper):
                raise UncertaintyValidationError("Bounds must be finite.")
            if self.lower >= self.upper:
                raise UncertaintyValidationError("lower must be less than upper.")

    def _reject_all_spread_parameters(self, label: str) -> None:
        numeric = (
            self.standard_deviation,
            self.relative_standard_deviation,
            self.confidence_half_width,
            self.confidence_level,
            self.lower,
            self.upper,
            self.mode,
        )
        if (
            any(value is not None for value in numeric)
            or self.samples
            or self.choices
            or self.probabilities
        ):
            raise UncertaintyValidationError(
                f"{label} uncertainty cannot also define spread parameters."
            )

    def _reject_normal_parameters(self, label: str) -> None:
        values = (
            self.standard_deviation,
            self.relative_standard_deviation,
            self.confidence_half_width,
            self.confidence_level,
        )
        if any(value is not None for value in values):
            raise UncertaintyValidationError(
                f"{label} uncertainty cannot define normal spread parameters."
            )

    def _reject_sequence_parameters(self, label: str) -> None:
        if self.samples or self.choices or self.probabilities:
            raise UncertaintyValidationError(
                f"{label} uncertainty cannot define samples or choices."
            )


@dataclass(frozen=True)
class UncertainQuantity:
    """A nominal numeric quantity with units, provenance, and uncertainty."""

    nominal: float
    unit: str
    source: SourceSpec
    uncertainty: UncertaintySpec
    correlation_group: str | None = None
    notes: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "UncertainQuantity":
        missing = [
            key for key in ("nominal", "unit", "source", "uncertainty") if key not in raw
        ]
        if missing:
            raise UncertaintyValidationError(
                "Quantity is missing required fields: " + ", ".join(missing)
            )
        source_raw = raw["source"]
        uncertainty_raw = raw["uncertainty"]
        if not isinstance(source_raw, Mapping):
            raise UncertaintyValidationError("source must be a table/object.")
        if not isinstance(uncertainty_raw, Mapping):
            raise UncertaintyValidationError("uncertainty must be a table/object.")
        try:
            nominal = float(raw["nominal"])
        except (TypeError, ValueError) as exc:
            raise UncertaintyValidationError("Quantity nominal must be numeric.") from exc
        quantity = cls(
            nominal=nominal,
            unit=str(raw["unit"]),
            source=SourceSpec.from_mapping(source_raw),
            uncertainty=UncertaintySpec.from_mapping(uncertainty_raw),
            correlation_group=(
                None
                if raw.get("correlation_group") in (None, "")
                else str(raw["correlation_group"])
            ),
            notes=str(raw.get("notes", "")),
        )
        quantity.validate()
        return quantity

    def validate(self) -> None:
        if not isfinite(self.nominal):
            raise UncertaintyValidationError("nominal must be finite.")
        if not self.unit.strip():
            raise UncertaintyValidationError("unit must be non-empty.")
        try:
            get_unit(self.unit)
        except UnitValidationError as exc:
            raise UncertaintyValidationError(str(exc)) from exc
        self.source.validate()
        self.uncertainty.validate()
        if self.uncertainty.distribution is DistributionKind.DISCRETE:
            raise UncertaintyValidationError(
                "Numeric quantities cannot use discrete choices; use UncertainChoice."
            )

        if self.uncertainty.distribution in {
            DistributionKind.UNIFORM,
            DistributionKind.TRIANGULAR,
            DistributionKind.TRUNCATED_NORMAL,
        }:
            lower = self.uncertainty.lower
            upper = self.uncertainty.upper
            assert lower is not None and upper is not None
            if not lower <= self.nominal <= upper:
                raise UncertaintyValidationError(
                    "nominal must lie inside the declared uncertainty bounds."
                )
        if self.uncertainty.relative_standard_deviation is not None and self.nominal == 0:
            raise UncertaintyValidationError(
                "relative_standard_deviation cannot be used with a zero nominal value."
            )

    def nominal_si(self) -> tuple[float, str]:
        return convert_to_si(self.nominal, self.unit)


@dataclass(frozen=True)
class UncertainChoice:
    """A categorical model choice with provenance and discrete alternatives."""

    nominal: str
    source: SourceSpec
    uncertainty: UncertaintySpec
    correlation_group: str | None = None
    notes: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "UncertainChoice":
        missing = [key for key in ("nominal", "source", "uncertainty") if key not in raw]
        if missing:
            raise UncertaintyValidationError(
                "Choice is missing required fields: " + ", ".join(missing)
            )
        if "unit" in raw:
            raise UncertaintyValidationError("Categorical choices must not declare a unit.")
        source_raw = raw["source"]
        uncertainty_raw = raw["uncertainty"]
        if not isinstance(source_raw, Mapping) or not isinstance(uncertainty_raw, Mapping):
            raise UncertaintyValidationError("source and uncertainty must be tables/objects.")
        choice = cls(
            nominal=str(raw["nominal"]),
            source=SourceSpec.from_mapping(source_raw),
            uncertainty=UncertaintySpec.from_mapping(uncertainty_raw),
            correlation_group=(
                None
                if raw.get("correlation_group") in (None, "")
                else str(raw["correlation_group"])
            ),
            notes=str(raw.get("notes", "")),
        )
        choice.validate()
        return choice

    def validate(self) -> None:
        if not self.nominal.strip():
            raise UncertaintyValidationError("Choice nominal must be non-empty.")
        self.source.validate()
        self.uncertainty.validate()
        if self.uncertainty.distribution is DistributionKind.FIXED:
            return
        if self.uncertainty.distribution is not DistributionKind.DISCRETE:
            raise UncertaintyValidationError(
                "Categorical choices require fixed or discrete uncertainty."
            )
        if self.nominal not in self.uncertainty.choices:
            raise UncertaintyValidationError(
                "Choice nominal must be included in uncertainty.choices."
            )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise UncertaintyValidationError(
            "Uncertainty parameters must be numeric."
        ) from exc
    if not isfinite(result):
        raise UncertaintyValidationError("Uncertainty parameters must be finite.")
    return result


def _float_sequence(value: Any, label: str) -> tuple[float, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, (list, tuple)):
        raise UncertaintyValidationError(f"{label} must be an array.")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise UncertaintyValidationError(f"{label} must contain only numbers.") from exc
    if not all(isfinite(item) for item in result):
        raise UncertaintyValidationError(f"{label} must contain only finite numbers.")
    return result


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise UncertaintyValidationError(f"{label} must be an array of strings.")
    return tuple(value)


def _optional_role(value: Any) -> UncertaintyRole | None:
    if value in (None, ""):
        return None
    try:
        return UncertaintyRole(str(value))
    except ValueError as exc:
        valid = ", ".join(item.value for item in UncertaintyRole)
        raise UncertaintyValidationError(
            f"uncertainty.role must be one of: {valid}."
        ) from exc
