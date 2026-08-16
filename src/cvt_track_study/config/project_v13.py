"""Native CSV and external-vehicle telemetry compatibility for v12."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

from .diagnostics import Diagnostic, Severity
from .project import ResolutionResult
from .project_v12 import ProjectLoader as _ProjectLoader


_RUN_PATH = re.compile(r"^runs\.(\d+)(?:\.|$)")


class ProjectLoader(_ProjectLoader):
    """Preserve v12 behavior while accepting native CSV/external telemetry."""

    def resolve(
        self,
        project: str | Path,
        *,
        study: str | None = None,
        cli_overrides: Sequence[tuple[str, Any]] = (),
    ) -> ResolutionResult:
        result = super().resolve(project, study=study, cli_overrides=cli_overrides)
        runs = result.data.get("runs", [])
        run_list = runs if isinstance(runs, list) else []

        retained: list[Diagnostic] = []
        for diagnostic in result.diagnostics:
            run = _run_for_diagnostic(run_list, diagnostic)
            if (
                diagnostic.code == "RUN_FORMAT_UNSUPPORTED"
                and isinstance(run, Mapping)
                and Path(str(run.get("file", ""))).suffix.lower() == ".csv"
            ):
                continue
            if (
                diagnostic.code == "RUN_VEHICLE_NOT_FOUND"
                and isinstance(run, Mapping)
                and run.get("external_vehicle") is True
            ):
                continue
            retained.append(diagnostic)

        retained.extend(_validate_csv_and_external_runs(result))
        result.diagnostics = tuple(retained)
        return result


def _run_for_diagnostic(
    runs: list[Any], diagnostic: Diagnostic
) -> Mapping[str, Any] | None:
    match = _RUN_PATH.match(str(diagnostic.path))
    if match is None:
        return None
    index = int(match.group(1))
    if not 0 <= index < len(runs):
        return None
    run = runs[index]
    return run if isinstance(run, Mapping) else None


def _validate_csv_and_external_runs(result: ResolutionResult) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    runs = result.data.get("runs", [])
    vehicles = result.data.get("vehicles", {})
    vehicle_ids = set(vehicles) if isinstance(vehicles, Mapping) else set()
    if not isinstance(runs, list):
        return diagnostics

    for index, raw in enumerate(runs):
        if not isinstance(raw, Mapping):
            continue
        path = f"runs.{index}"
        external = raw.get("external_vehicle", False)
        if not isinstance(external, bool):
            diagnostics.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    code="RUN_EXTERNAL_VEHICLE_FLAG_NOT_BOOLEAN",
                    message="external_vehicle must be true or false when declared.",
                    path=f"{path}.external_vehicle",
                )
            )
            external = False

        if Path(str(raw.get("file", ""))).suffix.lower() == ".csv":
            diagnostics.append(
                Diagnostic(
                    severity=Severity.INFO,
                    code="CSV_TELEMETRY_SUPPORTED",
                    message=(
                        "Native CSV position/speed telemetry will be ingested as ordinary "
                        "track evidence and may be used for centreline and speed gates."
                    ),
                    path=f"{path}.file",
                )
            )

        vehicle_id = str(raw.get("vehicle_id", ""))
        if external and vehicle_id not in vehicle_ids:
            diagnostics.append(
                Diagnostic(
                    severity=Severity.INFO,
                    code="EXTERNAL_TELEMETRY_VEHICLE",
                    message=(
                        f"Measured telemetry vehicle {vehicle_id!r} has no simulator "
                        "vehicle definition. Its identity is retained for track/gate "
                        "evidence only."
                    ),
                    path=f"{path}.vehicle_id",
                )
            )
    return diagnostics
