"""Compatibility loader for the six-report contract.

The legacy schema treated ``track_robustness`` as a vehicle simulation study.
The v10 contract makes it a data-only reconstruction study, so vehicle,
base-case, and Monte-Carlo sampling diagnostics are suppressed only for that
study type.  All track, run, event, and physical-contract validation remains
active.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from math import isfinite

from .diagnostics import Diagnostic, Severity
from .project_v9 import ProjectLoader as _ProjectLoader
from .project import ResolutionResult


_LEGACY_TRACK_ROBUSTNESS_CODES = {
    "STUDY_VEHICLE_NOT_FOUND",
    "INVALID_BASE_CASE_STUDY",
    "SAMPLING_TABLE_MISSING",
    "UNKNOWN_SAMPLING_MODE",
    "INVALID_REPLICATE_COUNT",
    "UNPAIRED_DESIGN_SCENARIOS",
    "INVALID_GATE_SAMPLING_MODE",
}


class ProjectLoader(_ProjectLoader):
    """Apply the v9 wildcard compatibility plus v10 report semantics."""

    def resolve(
        self,
        project: str | Path,
        *,
        study: str | None = None,
        cli_overrides: Sequence[tuple[str, Any]] = (),
    ) -> ResolutionResult:
        result = super().resolve(project, study=study, cli_overrides=cli_overrides)
        track_studies = {
            str(name)
            for name, raw in result.data.get("studies", {}).items()
            if isinstance(raw, Mapping)
            and str(raw.get("study", {}).get("type", "")) == "track_robustness"
        }
        if not track_studies:
            result.diagnostics = tuple(
                [*result.diagnostics, *_validate_v10_report_contracts(result.data)]
            )
            return result

        retained = []
        for diagnostic in result.diagnostics:
            if diagnostic.code in _LEGACY_TRACK_ROBUSTNESS_CODES and any(
                diagnostic.path.startswith(f"studies.{name}") for name in track_studies
            ):
                continue
            retained.append(diagnostic)
        retained.append(
            Diagnostic(
                severity=Severity.INFO,
                code="TRACK_ROBUSTNESS_DATA_ONLY",
                message=(
                    "Track robustness varies telemetry reconstruction and gate-evidence "
                    "analysis only; vehicle, drivetrain, and baseline-study fields are not required."
                ),
                path="studies." + ",".join(sorted(track_studies)),
            )
        )
        retained.extend(_validate_v10_report_contracts(result.data))
        result.diagnostics = tuple(retained)
        return result


def _validate_v10_report_contracts(data: Mapping[str, Any]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    studies = data.get("studies", {})
    if not isinstance(studies, Mapping):
        return diagnostics
    for name, raw in studies.items():
        if not isinstance(raw, Mapping):
            continue
        study_type = str(raw.get("study", {}).get("type", ""))
        path = f"studies.{name}"
        if study_type == "track_robustness":
            robustness = raw.get("robustness", {})
            if robustness is not None and not isinstance(robustness, Mapping):
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="TRACK_ROBUSTNESS_TABLE_INVALID",
                        message="[robustness] must be a table.",
                        path=f"{path}.robustness",
                    )
                )
            elif isinstance(robustness, Mapping):
                maximum = robustness.get("maximum_cases", 40)
                if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
                    diagnostics.append(
                        Diagnostic(
                            severity=Severity.ERROR,
                            code="TRACK_ROBUSTNESS_CASE_COUNT_INVALID",
                            message="robustness.maximum_cases must be a positive integer.",
                            path=f"{path}.robustness.maximum_cases",
                        )
                    )
            thresholds = raw.get("thresholds", {})
            if thresholds is not None and not isinstance(thresholds, Mapping):
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="TRACK_ROBUSTNESS_THRESHOLDS_INVALID",
                        message="[thresholds] must be a table.",
                        path=f"{path}.thresholds",
                    )
                )
            elif isinstance(thresholds, Mapping):
                for key, value in thresholds.items():
                    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(float(value)) or float(value) <= 0:
                        diagnostics.append(
                            Diagnostic(
                                severity=Severity.ERROR,
                                code="TRACK_ROBUSTNESS_THRESHOLD_INVALID",
                                message=f"thresholds.{key} must be positive and finite.",
                                path=f"{path}.thresholds.{key}",
                            )
                        )
        if study_type in {"full_uncertainty", "design_sweep"}:
            ensemble = raw.get("track_ensemble")
            if ensemble is None:
                continue
            if not isinstance(ensemble, Mapping):
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="TRACK_ENSEMBLE_TABLE_INVALID",
                        message="[track_ensemble] must be a table.",
                        path=f"{path}.track_ensemble",
                    )
                )
                continue
            enabled = ensemble.get("enabled", True)
            if not isinstance(enabled, bool):
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="TRACK_ENSEMBLE_ENABLED_NOT_BOOLEAN",
                        message="track_ensemble.enabled must be true or false.",
                        path=f"{path}.track_ensemble.enabled",
                    )
                )
            maximum = ensemble.get("maximum_cases", 20)
            if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="TRACK_ENSEMBLE_CASE_COUNT_INVALID",
                        message="track_ensemble.maximum_cases must be a positive integer.",
                        path=f"{path}.track_ensemble.maximum_cases",
                    )
                )
            categories = ensemble.get("include_categories", [])
            if categories and (
                not isinstance(categories, list)
                or not all(isinstance(item, str) and item for item in categories)
            ):
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="TRACK_ENSEMBLE_CATEGORIES_INVALID",
                        message="track_ensemble.include_categories must be an array of non-empty strings.",
                        path=f"{path}.track_ensemble.include_categories",
                    )
                )
    return diagnostics
