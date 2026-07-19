"""Compatibility loader for all-declared structural selection."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from .diagnostics import (
    Diagnostic,
    Severity,
)
from .project import (
    ProjectLoader as _ProjectLoader,
    ResolutionResult,
)


_AUTO_SELECTIONS = {
    "*",
    "all",
    "all_declared",
    "all_declared_structural",
}


class ProjectLoader(_ProjectLoader):
    """Filter legacy validator errors for the explicit wildcard sentinel.

    Every discovered physical input is still validated by the existing
    physical-contract validation. The study planner performs the final
    structural-role and non-fixed checks after the registry is built.
    """

    def resolve(
        self,
        project: str | Path,
        *,
        study: str | None = None,
        cli_overrides: Sequence[
            tuple[str, Any]
        ] = (),
    ) -> ResolutionResult:
        result = super().resolve(
            project,
            study=study,
            cli_overrides=cli_overrides,
        )
        auto = {
            str(name)
            for name, raw in result.data.get(
                "studies", {}
            ).items()
            if _is_auto_structural(raw)
        }
        if not auto:
            return result

        retained = []
        for diagnostic in result.diagnostics:
            if _legacy_wildcard_diagnostic(
                diagnostic, auto
            ):
                continue
            retained.append(diagnostic)

        retained.append(
            Diagnostic(
                severity=Severity.INFO,
                code=(
                    "ALL_DECLARED_STRUCTURAL_SELECTION"
                ),
                message=(
                    "Structural sensitivity uses the registry wildcard and "
                    "will screen every non-fixed input with resolved "
                    "uncertainty role 'structural'."
                ),
                path=(
                    "studies."
                    + ",".join(sorted(auto))
                    + ".sensitivity.parameters"
                ),
            )
        )
        result.diagnostics = tuple(
            retained
        )
        return result


def _is_auto_structural(
    raw: Any,
) -> bool:
    if not isinstance(raw, Mapping):
        return False
    if (
        str(
            raw.get("study", {}).get(
                "type", ""
            )
        )
        != "structural_sensitivity"
    ):
        return False
    sensitivity = raw.get(
        "sensitivity", {}
    )
    if not isinstance(
        sensitivity, Mapping
    ):
        return False
    selection = str(
        sensitivity.get(
            "selection", ""
        )
    ).strip().lower()
    parameters = sensitivity.get(
        "parameters"
    )
    return (
        selection in _AUTO_SELECTIONS
        or parameters == ["*"]
        or parameters == "*"
        or parameters is None
    )


def _legacy_wildcard_diagnostic(
    diagnostic: Diagnostic,
    studies: set[str],
) -> bool:
    if diagnostic.code not in {
        "NO_SENSITIVITY_PARAMETERS",
        "SENSITIVITY_PATH_NOT_FOUND",
        "UNUSED_SENSITIVITY_QUANTILES",
    }:
        return False
    return any(
        diagnostic.path.startswith(
            f"studies.{name}.sensitivity"
        )
        for name in studies
    )
