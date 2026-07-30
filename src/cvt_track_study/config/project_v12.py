"""Validation additions for conservative route-variant reconstruction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

from .diagnostics import Diagnostic, Severity
from .project import ResolutionResult
from .project_v11 import ProjectLoader as _ProjectLoader


class ProjectLoader(_ProjectLoader):
    """Preserve v11 behavior and validate route-variant policy."""

    def resolve(
        self,
        project: str | Path,
        *,
        study: str | None = None,
        cli_overrides: Sequence[tuple[str, Any]] = (),
    ) -> ResolutionResult:
        result = super().resolve(
            project,
            study=study,
            cli_overrides=cli_overrides,
        )
        result.diagnostics = tuple(
            [*result.diagnostics, *_validate_route_variants(result)]
        )
        return result


def _validate_route_variants(result: ResolutionResult) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    track = result.data.get("track", {})
    if not isinstance(track, Mapping):
        return diagnostics
    raw = track.get("route_variants")
    if raw is None:
        return diagnostics
    path = "track.route_variants"
    if not isinstance(raw, Mapping):
        return [_error("ROUTE_VARIANTS_TABLE_INVALID", "route_variants must be a TOML table.", path)]

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        diagnostics.append(
            _error(
                "ROUTE_VARIANTS_ENABLED_NOT_BOOLEAN",
                "enabled must be true or false.",
                f"{path}.enabled",
            )
        )
        return diagnostics
    if not enabled:
        return diagnostics

    minimum = raw.get("minimum_supported_laps", 2)
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 2:
        diagnostics.append(
            _error(
                "ROUTE_VARIANT_SUPPORT_INVALID",
                "minimum_supported_laps must be an integer of at least 2.",
                f"{path}.minimum_supported_laps",
            )
        )

    for field in (
        "sample_spacing_m",
        "same_variant_p95_distance_m",
        "divergence_distance_m",
    ):
        if field not in raw:
            continue
        value = raw[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(float(value))
            or float(value) <= 0.0
        ):
            diagnostics.append(
                _error(
                    "ROUTE_VARIANT_DISTANCE_INVALID",
                    f"{field} must be positive and finite.",
                    f"{path}.{field}",
                )
            )

    for field in (
        "maximum_divergent_fraction",
        "maximum_length_relative_difference",
        "maximum_within_variant_length_deviation_fraction",
    ):
        if field not in raw:
            continue
        value = raw[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(float(value))
            or not 0.0 <= float(value) < 1.0
        ):
            diagnostics.append(
                _error(
                    "ROUTE_VARIANT_FRACTION_INVALID",
                    f"{field} must lie in [0, 1).",
                    f"{path}.{field}",
                )
            )

    selection = str(raw.get("selection", "require_explicit")).strip().lower()
    allowed = {
        "require_explicit",
        "reference_run",
        "largest_supported",
        "longest_supported",
        "variant_id",
    }
    if selection not in allowed:
        diagnostics.append(
            _error(
                "ROUTE_VARIANT_SELECTION_INVALID",
                "selection must be one of " + ", ".join(sorted(allowed)) + ".",
                f"{path}.selection",
            )
        )
    elif selection == "reference_run" and not str(
        raw.get("reference_run_id", "")
    ).strip():
        diagnostics.append(
            _error(
                "ROUTE_VARIANT_REFERENCE_RUN_MISSING",
                "reference_run_id is required for selection='reference_run'.",
                f"{path}.reference_run_id",
            )
        )
    elif selection == "variant_id" and not str(
        raw.get("selected_variant_id", "")
    ).strip():
        diagnostics.append(
            _error(
                "ROUTE_VARIANT_ID_MISSING",
                "selected_variant_id is required for selection='variant_id'.",
                f"{path}.selected_variant_id",
            )
        )

    diagnostics.append(
        Diagnostic(
            severity=Severity.INFO,
            code="ROUTE_VARIANT_DETECTION_ENABLED",
            message=(
                "Complete laps will be clustered by repeated spatial path before "
                "centreline consensus. Supported alternate routes remain auditable "
                "and are not averaged into the selected route."
            ),
            path=path,
        )
    )
    return diagnostics


def _error(code: str, message: str, path: str) -> Diagnostic:
    return Diagnostic(
        severity=Severity.ERROR,
        code=code,
        message=message,
        path=path,
    )
