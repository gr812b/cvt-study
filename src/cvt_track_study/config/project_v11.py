"""Validation additions for optional untimed-GPX lap-time reconstruction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from math import isfinite

from .diagnostics import Diagnostic, Severity
from .project import ResolutionResult
from .project_v10 import ProjectLoader as _ProjectLoader


class ProjectLoader(_ProjectLoader):
    """Preserve v10 behavior and validate lap-time reconstruction inputs."""

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
            [
                *result.diagnostics,
                *_validate_lap_time_reconstruction(result),
            ]
        )
        return result


def _validate_lap_time_reconstruction(
    result: ResolutionResult,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    runs = result.data.get("runs", [])
    if not isinstance(runs, list):
        return diagnostics

    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            continue
        raw = run.get("lap_time_reconstruction")
        if raw is None:
            continue
        path = f"runs.{index}.lap_time_reconstruction"
        if not isinstance(raw, Mapping):
            diagnostics.append(
                _error(
                    "LAP_TIME_RECONSTRUCTION_TABLE_INVALID",
                    "lap_time_reconstruction must be a TOML table.",
                    path,
                )
            )
            continue

        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            diagnostics.append(
                _error(
                    "LAP_TIME_RECONSTRUCTION_ENABLED_NOT_BOOLEAN",
                    "enabled must be true or false.",
                    f"{path}.enabled",
                )
            )
            continue
        if not enabled:
            continue

        source_file = Path(str(run.get("file", "")))
        if source_file.suffix.lower() != ".gpx":
            diagnostics.append(
                _error(
                    "LAP_TIME_RECONSTRUCTION_REQUIRES_GPX",
                    "Lap-time reconstruction is only valid for an untimed .gpx run.",
                    f"runs.{index}.file",
                )
            )

        csv_text = str(raw.get("lap_times_file", "")).strip()
        if not csv_text:
            diagnostics.append(
                _error(
                    "LAP_TIME_CSV_MISSING",
                    "Enabled reconstruction requires lap_times_file.",
                    f"{path}.lap_times_file",
                )
            )
        else:
            csv_path = (
                result.paths.runs_file.parent / csv_text
            ).resolve()
            if csv_path.suffix.lower() != ".csv":
                diagnostics.append(
                    _error(
                        "LAP_TIME_FILE_NOT_CSV",
                        "lap_times_file must use the .csv extension.",
                        f"{path}.lap_times_file",
                    )
                )
            if not _is_within(csv_path, result.paths.root):
                diagnostics.append(
                    _error(
                        "LAP_TIME_FILE_ESCAPES_PROJECT",
                        "lap_times_file must remain inside the project directory.",
                        f"{path}.lap_times_file",
                    )
                )
            elif not csv_path.is_file():
                diagnostics.append(
                    _error(
                        "LAP_TIME_FILE_NOT_FOUND",
                        "Declared lap-time CSV does not exist.",
                        f"{path}.lap_times_file",
                        source=str(csv_path),
                    )
                )

        for field in (
            "lap_index_column",
            "lap_time_column",
        ):
            if field in raw and not str(raw[field]).strip():
                diagnostics.append(
                    _error(
                        "LAP_TIME_COLUMN_EMPTY",
                        f"{field} must be a non-empty string.",
                        f"{path}.{field}",
                    )
                )
        for field in (
            "gate_radius_m",
            "expected_point_period_s",
            "maximum_point_period_error_fraction",
            "minimum_csv_lap_time_s",
            "maximum_csv_lap_time_s",
            "maximum_alignment_error_fraction",
            "skip_detected_lap_penalty",
            "skip_csv_lap_penalty",
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
                        "LAP_TIME_RECONSTRUCTION_NUMBER_INVALID",
                        f"{field} must be positive and finite.",
                        f"{path}.{field}",
                    )
                )
        for field in ("minimum_points_per_lap", "minimum_matched_laps"):
            if field not in raw:
                continue
            value = raw[field]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                diagnostics.append(
                    _error(
                        "LAP_TIME_RECONSTRUCTION_COUNT_INVALID",
                        f"{field} must be a positive integer.",
                        f"{path}.{field}",
                    )
                )

        mode = str(raw.get("alignment_mode", "cadence_dynamic_programming")).strip().lower()
        if mode not in {"strict_index", "cadence_dynamic_programming"}:
            diagnostics.append(
                _error(
                    "LAP_TIME_ALIGNMENT_MODE_INVALID",
                    "alignment_mode must be strict_index or cadence_dynamic_programming.",
                    f"{path}.alignment_mode",
                )
            )
        lower = raw.get("minimum_csv_lap_time_s")
        upper = raw.get("maximum_csv_lap_time_s")
        if (
            isinstance(lower, (int, float))
            and not isinstance(lower, bool)
            and isinstance(upper, (int, float))
            and not isinstance(upper, bool)
            and float(lower) >= float(upper)
        ):
            diagnostics.append(
                _error(
                    "LAP_TIME_CSV_RANGE_INVALID",
                    "minimum_csv_lap_time_s must be below maximum_csv_lap_time_s.",
                    path,
                )
            )
        for field in (
            "allow_unmatched_laps",
            "export_augmented_gpx",
        ):
            if field in raw and not isinstance(raw[field], bool):
                diagnostics.append(
                    _error(
                        "LAP_TIME_RECONSTRUCTION_FLAG_INVALID",
                        f"{field} must be true or false.",
                        f"{path}.{field}",
                    )
                )

        diagnostics.append(
            Diagnostic(
                severity=Severity.INFO,
                code="LAP_TIME_RECONSTRUCTION_SUPPLEMENTAL",
                message=(
                    "This run will reconstruct within-lap timing from ordered "
                    "GPX points and an external lap-time CSV. Its speeds remain "
                    "vehicle-specific supplemental evidence and are not pooled "
                    "as native telemetry."
                ),
                path=path,
            )
        )
    return diagnostics


def _error(
    code: str,
    message: str,
    path: str,
    *,
    source: str = "",
) -> Diagnostic:
    return Diagnostic(
        severity=Severity.ERROR,
        code=code,
        message=message,
        path=path,
        source=source,
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True
