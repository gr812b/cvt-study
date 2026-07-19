"""Configuration loading, uncertainty contracts, units, and validation.

The package facade uses lazy imports so low-level contracts can depend on the
uncertainty and unit modules without pulling in the project resolver and its
validation graph.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "Diagnostic": (
        ".diagnostics", "Diagnostic"
    ),
    "DiagnosticBag": (
        ".diagnostics", "DiagnosticBag"
    ),
    "Severity": (
        ".diagnostics", "Severity"
    ),
    "ProjectError": (
        ".project", "ProjectError"
    ),
    "ProjectLoader": (
        ".project_v9", "ProjectLoader"
    ),
    "ProjectPaths": (
        ".project", "ProjectPaths"
    ),
    "ResolutionResult": (
        ".project", "ResolutionResult"
    ),
    "discover_project_file": (
        ".project", "discover_project_file"
    ),
    "initialize_project": (
        ".project", "initialize_project"
    ),
    "parse_override": (
        ".project", "parse_override"
    ),
    "DistributionKind": (
        ".uncertainty", "DistributionKind"
    ),
    "SourceKind": (
        ".uncertainty", "SourceKind"
    ),
    "UncertaintyRole": (
        ".uncertainty", "UncertaintyRole"
    ),
    "SourceSpec": (
        ".uncertainty", "SourceSpec"
    ),
    "UncertainChoice": (
        ".uncertainty", "UncertainChoice"
    ),
    "UncertainQuantity": (
        ".uncertainty", "UncertainQuantity"
    ),
    "UncertaintySpec": (
        ".uncertainty", "UncertaintySpec"
    ),
    "UncertaintyValidationError": (
        ".uncertainty",
        "UncertaintyValidationError",
    ),
    "UnitValidationError": (
        ".units", "UnitValidationError"
    ),
    "convert_to_si": (
        ".units", "convert_to_si"
    ),
    "get_unit": (
        ".units", "get_unit"
    ),
    "require_dimension": (
        ".units", "require_dimension"
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = (
            _EXPORTS[name]
        )
    except KeyError as exc:
        raise AttributeError(
            name
        ) from exc
    value = getattr(
        import_module(
            module_name, __name__
        ),
        attribute,
    )
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(
        set(globals()) | set(__all__)
    )
