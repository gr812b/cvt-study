"""Small explicit unit registry for configuration validation and SI conversion.

This is intentionally not a symbolic algebra system. It recognizes the units used
by the project contract, records their physical dimension, and converts scalar
values to a canonical SI representation. New mechanisms must register their units
when they are added rather than accepting arbitrary strings silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


class UnitValidationError(ValueError):
    pass


@dataclass(frozen=True)
class UnitDefinition:
    symbol: str
    dimension: str
    scale_to_si: float
    si_symbol: str


_DEFINITIONS = (
    UnitDefinition("1", "dimensionless", 1.0, "1"),
    UnitDefinition("%", "dimensionless", 0.01, "1"),
    UnitDefinition("kg", "mass", 1.0, "kg"),
    UnitDefinition("kg*m^2", "rotational_inertia", 1.0, "kg*m^2"),
    UnitDefinition("kg/m^3", "density", 1.0, "kg/m^3"),
    UnitDefinition("g", "mass", 1e-3, "kg"),
    UnitDefinition("m", "length", 1.0, "m"),
    UnitDefinition("cm", "length", 1e-2, "m"),
    UnitDefinition("mm", "length", 1e-3, "m"),
    UnitDefinition("in", "length", 0.0254, "m"),
    UnitDefinition("ft", "length", 0.3048, "m"),
    UnitDefinition("s", "time", 1.0, "s"),
    UnitDefinition("ms", "time", 1e-3, "s"),
    UnitDefinition("m/s", "speed", 1.0, "m/s"),
    UnitDefinition("m/s^2", "acceleration", 1.0, "m/s^2"),
    UnitDefinition("km/h", "speed", 1.0 / 3.6, "m/s"),
    UnitDefinition("mph", "speed", 0.44704, "m/s"),
    UnitDefinition("m^2", "area", 1.0, "m^2"),
    UnitDefinition("cm^2", "area", 1e-4, "m^2"),
    UnitDefinition("N", "force", 1.0, "N"),
    UnitDefinition("N*s/m", "slip_stiffness", 1.0, "N*s/m"),
    UnitDefinition("N*m", "torque", 1.0, "N*m"),
    UnitDefinition("J", "energy", 1.0, "J"),
    UnitDefinition("J/kg", "specific_energy", 1.0, "J/kg"),
    UnitDefinition("J/(kg*m)", "specific_energy_per_distance", 1.0, "J/(kg*m)"),
    UnitDefinition("kJ", "energy", 1e3, "J"),
    UnitDefinition("W", "power", 1.0, "W"),
    UnitDefinition("kW", "power", 1e3, "W"),
    UnitDefinition("hp", "power", 745.6998715822702, "W"),
    UnitDefinition("rad/s", "angular_speed", 1.0, "rad/s"),
    UnitDefinition("rpm", "angular_speed", 2.0 * 3.141592653589793 / 60.0, "rad/s"),
    UnitDefinition("deg", "angle", 3.141592653589793 / 180.0, "rad"),
    UnitDefinition("rad", "angle", 1.0, "rad"),
)

_ALIASES = {
    "": "1",
    "dimensionless": "1",
    "ratio": "1",
    "m2": "m^2",
    "m²": "m^2",
    "nm": "N*m",
    "kg m^2": "kg*m^2",
    "kg·m^2": "kg*m^2",
    "n/(m/s)": "N*s/m",
    "n s/m": "N*s/m",
    "j/kg/m": "J/(kg*m)",
    "n·m": "N*m",
    "n*m": "N*m",
    "kph": "km/h",
    "kmh": "km/h",
    "horsepower": "hp",
}

_BY_SYMBOL = {definition.symbol: definition for definition in _DEFINITIONS}


def normalize_unit(unit: str) -> str:
    stripped = unit.strip()
    alias = _ALIASES.get(stripped.lower())
    return alias if alias is not None else stripped


def get_unit(unit: str) -> UnitDefinition:
    normalized = normalize_unit(unit)
    try:
        return _BY_SYMBOL[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_BY_SYMBOL))
        raise UnitValidationError(
            f"Unknown unit {unit!r}. Supported units are: {supported}."
        ) from exc


def convert_to_si(value: float, unit: str) -> tuple[float, str]:
    if not isfinite(value):
        raise UnitValidationError("Only finite values can be converted.")
    definition = get_unit(unit)
    return value * definition.scale_to_si, definition.si_symbol


def require_dimension(unit: str, expected_dimension: str) -> None:
    definition = get_unit(unit)
    if definition.dimension != expected_dimension:
        raise UnitValidationError(
            f"Unit {unit!r} has dimension {definition.dimension!r}; "
            f"expected {expected_dimension!r}."
        )
