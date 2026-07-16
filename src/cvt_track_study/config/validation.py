"""Schema and cross-reference validation for resolved projects."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from math import isfinite
from statistics import NormalDist
from typing import Any, Protocol

import numpy as np

from .diagnostics import DiagnosticBag
from .uncertainty import (
    DistributionKind,
    SourceKind,
    UncertainChoice,
    UncertainQuantity,
    UncertaintyValidationError,
)
from .units import UnitValidationError, require_dimension
from cvt_track_study.contracts.obstacles import (
    obstacle_model_alternatives,
    validate_obstacle_model_contract,
)


class ProjectPathsLike(Protocol):
    root: Path
    results_directory: Path
    vehicles_directory: Path
    studies_directory: Path
    runs_file: Path


def validate_project(
    data: Mapping[str, Any],
    paths: ProjectPathsLike,
    diagnostics: DiagnosticBag,
) -> None:
    ProjectValidator()._validate_all(data, paths, diagnostics)


class ProjectValidator:
    def _validate_all(
        self,
        data: Mapping[str, Any],
        paths: ProjectPathsLike,
        diagnostics: DiagnosticBag,
    ) -> None:
        self._validate_project(data, paths, diagnostics)
        self._validate_runs(data.get("runs"), data.get("vehicles"), paths, diagnostics)
        self._validate_events(data.get("events"), diagnostics)
        self._validate_track(data.get("track"), diagnostics)
        self._validate_track_event_links(data.get("track"), data.get("events"), diagnostics)
        self._validate_vehicles(data.get("vehicles"), diagnostics)
        self._validate_studies(
            data.get("studies"),
            data.get("vehicles"),
            data.get("track"),
            data.get("events"),
            diagnostics,
        )

    def _validate_project(
        self, data: Mapping[str, Any], paths: ProjectPathsLike, diagnostics: DiagnosticBag
    ) -> None:
        project = data.get("project")
        if not isinstance(project, Mapping):
            diagnostics.error("PROJECT_TABLE_MISSING", "Resolved project table is missing.")
            return
        if not str(project.get("name", "")).strip():
            diagnostics.error("PROJECT_NAME_MISSING", "project.name must be non-empty.")
        if project.get("schema_version") != 1:
            diagnostics.error(
                "UNSUPPORTED_PROJECT_SCHEMA",
                "project.schema_version must be 1 for this checkpoint.",
                path="project.schema_version",
            )
        for directory, code in (
            (paths.results_directory, "RESULTS_DIRECTORY"),
            (paths.vehicles_directory, "VEHICLES_DIRECTORY"),
            (paths.studies_directory, "STUDIES_DIRECTORY"),
        ):
            if directory.exists() and not directory.is_dir():
                diagnostics.error(
                    f"{code}_NOT_DIRECTORY",
                    "Configured path exists but is not a directory.",
                    path=str(directory),
                )

    def _validate_runs(
        self,
        raw_runs: Any,
        raw_vehicles: Any,
        paths: ProjectPathsLike,
        diagnostics: DiagnosticBag,
    ) -> None:
        if not isinstance(raw_runs, list):
            diagnostics.error("RUNS_NOT_ARRAY", "runs.toml must define runs as an array.")
            return
        if not raw_runs:
            diagnostics.warning(
                "NO_GPX_RUNS",
                "No GPX runs are declared yet.",
                path="runs",
                hint="Add one [[runs]] entry per GPX recording before build-track.",
            )
            return
        vehicle_ids = set(raw_vehicles) if isinstance(raw_vehicles, Mapping) else set()
        seen_run_ids: set[str] = set()
        for index, run in enumerate(raw_runs):
            path = f"runs.{index}"
            if not isinstance(run, Mapping):
                diagnostics.error("INVALID_RUN", "Each run must be a TOML table.", path=path)
                continue
            required = (
                "file",
                "vehicle_id",
                "run_id",
                "driver_id",
                "use_for_centreline",
                "use_for_gate_evidence",
            )
            missing = [key for key in required if key not in run]
            if missing:
                diagnostics.error(
                    "RUN_FIELDS_MISSING",
                    "Run is missing fields: " + ", ".join(missing),
                    path=path,
                )
                continue
            file_text = str(run["file"])
            run_file = (paths.runs_file.parent / file_text).resolve()
            if Path(file_text).suffix.lower() != ".gpx":
                diagnostics.error(
                    "RUN_NOT_GPX",
                    "Raw telemetry files must use the .gpx extension.",
                    path=f"{path}.file",
                )
            if not _is_within(run_file, paths.root):
                diagnostics.error(
                    "RUN_PATH_ESCAPES_PROJECT",
                    "GPX files must remain inside the project directory.",
                    path=f"{path}.file",
                )
            elif not run_file.exists():
                diagnostics.error(
                    "GPX_FILE_MISSING",
                    "Declared GPX file does not exist.",
                    path=f"{path}.file",
                    source=str(run_file),
                )
            vehicle_id = str(run["vehicle_id"])
            if vehicle_id not in vehicle_ids:
                diagnostics.error(
                    "RUN_VEHICLE_NOT_FOUND",
                    f"Run references unknown vehicle {vehicle_id!r}.",
                    path=f"{path}.vehicle_id",
                )
            run_id = str(run["run_id"])
            if not run_id:
                diagnostics.error("RUN_ID_EMPTY", "run_id must be non-empty.", path=path)
            elif run_id in seen_run_ids:
                diagnostics.error(
                    "DUPLICATE_RUN_ID",
                    f"run_id {run_id!r} is duplicated.",
                    path=f"{path}.run_id",
                )
            seen_run_ids.add(run_id)
            for flag in ("use_for_centreline", "use_for_gate_evidence"):
                if not isinstance(run[flag], bool):
                    diagnostics.error(
                        "RUN_FLAG_NOT_BOOLEAN",
                        f"{flag} must be true or false.",
                        path=f"{path}.{flag}",
                    )
        if raw_runs and not any(
            isinstance(run, Mapping) and run.get("use_for_centreline") is True
            for run in raw_runs
        ):
            diagnostics.error(
                "NO_CENTRELINE_RUN",
                "At least one GPX run must be selected for centreline construction.",
                path="runs",
            )
        if raw_runs and not any(
            isinstance(run, Mapping) and run.get("use_for_gate_evidence") is True
            for run in raw_runs
        ):
            diagnostics.error(
                "NO_GATE_EVIDENCE_RUN",
                "At least one GPX run must be selected for event and gate evidence.",
                path="runs",
            )

    def _validate_events(self, raw_events: Any, diagnostics: DiagnosticBag) -> None:
        if not isinstance(raw_events, list):
            diagnostics.error("EVENTS_NOT_ARRAY", "events.toml must define events as an array.")
            return
        if not raw_events:
            diagnostics.warning(
                "NO_EVENTS",
                "No physical features are declared; track reconstruction needs at least one lap-gate event.",
                path="events",
            )
            return
        seen: set[str] = set()
        seen_sequences: set[int] = set()
        lap_gate_count = 0
        for index, event in enumerate(raw_events):
            path = f"events.{index}"
            if not isinstance(event, Mapping):
                diagnostics.error("INVALID_EVENT", "Each event must be a TOML table.", path=path)
                continue
            required = ("id", "name", "sequence", "kind", "analysis_role", "anchor")
            missing = [field for field in required if field not in event]
            if missing:
                diagnostics.error(
                    "EVENT_FIELDS_MISSING",
                    "Event is missing fields: " + ", ".join(missing),
                    path=path,
                )
                continue
            event_id = str(event.get("id", "")).strip()
            if not event_id:
                diagnostics.error("EVENT_ID_MISSING", "Event id is required.", path=path)
            elif event_id in seen:
                diagnostics.error(
                    "DUPLICATE_EVENT_ID",
                    f"Event id {event_id!r} is duplicated.",
                    path=f"{path}.id",
                )
            seen.add(event_id)
            if not str(event.get("name", "")).strip():
                diagnostics.error(
                    "EVENT_NAME_MISSING",
                    "Event name must be non-empty.",
                    path=f"{path}.name",
                )
            role = str(event.get("analysis_role", ""))
            if role not in {"feature", "lap_gate"}:
                diagnostics.error(
                    "UNKNOWN_EVENT_ANALYSIS_ROLE",
                    "analysis_role must be feature or lap_gate.",
                    path=f"{path}.analysis_role",
                )
            if not str(event.get("response_group_id", "")).strip():
                diagnostics.error(
                    "EVENT_RESPONSE_GROUP_MISSING",
                    "response_group_id must be non-empty.",
                    path=f"{path}.response_group_id",
                )
            if not isinstance(event.get("gate_candidate"), bool):
                diagnostics.error(
                    "EVENT_GATE_CANDIDATE_NOT_BOOLEAN",
                    "gate_candidate must be true or false.",
                    path=f"{path}.gate_candidate",
                )
            sequence = event.get("sequence")
            if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
                diagnostics.error(
                    "INVALID_EVENT_SEQUENCE",
                    "Event sequence must be a positive integer.",
                    path=f"{path}.sequence",
                )
            elif sequence in seen_sequences:
                diagnostics.error(
                    "DUPLICATE_EVENT_SEQUENCE",
                    f"Event sequence {sequence} is duplicated.",
                    path=f"{path}.sequence",
                )
            else:
                seen_sequences.add(sequence)
            if str(event.get("analysis_role")) == "lap_gate":
                lap_gate_count += 1
            if str(event.get("kind")) not in {"point", "turn", "interval", "obstacle"}:
                diagnostics.error(
                    "UNKNOWN_EVENT_KIND",
                    "Event kind must be point, turn, interval, or obstacle.",
                    path=f"{path}.kind",
                )
            anchor = event.get("anchor")
            if not isinstance(anchor, Mapping):
                diagnostics.error("EVENT_ANCHOR_MISSING", "Event anchor must be a table.", path=path)
                continue
            for field, lower, upper in (
                ("latitude_deg", -90.0, 90.0),
                ("longitude_deg", -180.0, 180.0),
            ):
                value = anchor.get(field)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not lower <= float(value) <= upper:
                    diagnostics.error(
                        "INVALID_EVENT_COORDINATE",
                        f"anchor.{field} must lie in [{lower}, {upper}].",
                        path=f"{path}.anchor.{field}",
                    )
            uncertainty = anchor.get("horizontal_uncertainty_m")
            if not isinstance(uncertainty, (int, float)) or isinstance(uncertainty, bool) or float(uncertainty) < 0:
                diagnostics.error(
                    "EVENT_COORDINATE_UNCERTAINTY_MISSING",
                    "anchor.horizontal_uncertainty_m must be explicitly declared and non-negative.",
                    path=f"{path}.anchor.horizontal_uncertainty_m",
                    hint="Use zero only with anchor.fixed_reason explaining why the coordinate is treated as exact.",
                )
            elif float(uncertainty) == 0 and not str(anchor.get("fixed_reason", "")).strip():
                diagnostics.error(
                    "FIXED_EVENT_COORDINATE_REASON_MISSING",
                    "Zero coordinate uncertainty requires anchor.fixed_reason.",
                    path=f"{path}.anchor.fixed_reason",
                )
            if not str(anchor.get("source", "")).strip():
                diagnostics.error(
                    "EVENT_COORDINATE_SOURCE_MISSING",
                    "anchor.source must identify map, survey, or video evidence.",
                    path=f"{path}.anchor.source",
                )
            for endpoint in ("start", "end"):
                value = event.get(endpoint)
                if value is None:
                    continue
                endpoint_path = f"{path}.{endpoint}"
                if not isinstance(value, Mapping):
                    diagnostics.error(
                        "INVALID_EVENT_ENDPOINT",
                        f"{endpoint} must be a coordinate table when present.",
                        path=endpoint_path,
                    )
                    continue
                if ("latitude_deg" in value) != ("longitude_deg" in value):
                    diagnostics.error(
                        "INCOMPLETE_EVENT_ENDPOINT",
                        f"{endpoint} requires both latitude_deg and longitude_deg.",
                        path=endpoint_path,
                    )
                    continue
                for field, lower, upper in (
                    ("latitude_deg", -90.0, 90.0),
                    ("longitude_deg", -180.0, 180.0),
                ):
                    coordinate = value.get(field)
                    if (
                        not isinstance(coordinate, (int, float))
                        or isinstance(coordinate, bool)
                        or not lower <= float(coordinate) <= upper
                    ):
                        diagnostics.error(
                            "INVALID_EVENT_ENDPOINT_COORDINATE",
                            f"{endpoint}.{field} must lie in [{lower}, {upper}].",
                            path=f"{endpoint_path}.{field}",
                        )
                endpoint_uncertainty = value.get("horizontal_uncertainty_m")
                if (
                    not isinstance(endpoint_uncertainty, (int, float))
                    or isinstance(endpoint_uncertainty, bool)
                    or float(endpoint_uncertainty) < 0
                ):
                    diagnostics.error(
                        "EVENT_ENDPOINT_UNCERTAINTY_MISSING",
                        f"{endpoint}.horizontal_uncertainty_m must be explicit and non-negative.",
                        path=f"{endpoint_path}.horizontal_uncertainty_m",
                    )
                elif float(endpoint_uncertainty) == 0 and not str(
                    value.get("fixed_reason", "")
                ).strip():
                    diagnostics.error(
                        "FIXED_EVENT_ENDPOINT_REASON_MISSING",
                        f"Zero uncertainty for {endpoint} requires fixed_reason.",
                        path=f"{endpoint_path}.fixed_reason",
                    )
                if not str(value.get("source", "")).strip():
                    diagnostics.error(
                        "EVENT_ENDPOINT_SOURCE_MISSING",
                        f"{endpoint}.source must identify the evidence used.",
                        path=f"{endpoint_path}.source",
                    )
            extent = event.get("extent")
            needs_before_extent = event.get("start") is None
            needs_after_extent = event.get("end") is None
            if (needs_before_extent or needs_after_extent) and not isinstance(
                extent, Mapping
            ):
                diagnostics.error(
                    "EVENT_EXTENT_MISSING",
                    "Each missing physical endpoint requires an extent fallback.",
                    path=f"{path}.extent",
                    hint=(
                        "Supply explicit start/end coordinate tables, or declare the "
                        "corresponding before/after anchor extent and uncertainty."
                    ),
                )
            elif isinstance(extent, Mapping):
                required_sides: list[tuple[str, str]] = []
                if needs_before_extent or "before_anchor_m" in extent:
                    required_sides.append(
                        ("before_anchor_m", "before_anchor_uncertainty_m")
                    )
                if needs_after_extent or "after_anchor_m" in extent:
                    required_sides.append(
                        ("after_anchor_m", "after_anchor_uncertainty_m")
                    )
                for value_field, uncertainty_field in required_sides:
                    value = extent.get(value_field)
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or float(value) < 0
                    ):
                        diagnostics.error(
                            "INVALID_EVENT_EXTENT_VALUE",
                            f"extent.{value_field} must be explicitly declared and non-negative.",
                            path=f"{path}.extent.{value_field}",
                        )
                    uncertainty_value = extent.get(uncertainty_field)
                    if (
                        not isinstance(uncertainty_value, (int, float))
                        or isinstance(uncertainty_value, bool)
                        or float(uncertainty_value) < 0
                    ):
                        diagnostics.error(
                            "EVENT_EXTENT_UNCERTAINTY_MISSING",
                            f"extent.{uncertainty_field} must be explicitly declared and non-negative.",
                            path=f"{path}.extent.{uncertainty_field}",
                        )
                    elif float(uncertainty_value) == 0 and not str(
                        extent.get("fixed_reason", "")
                    ).strip():
                        diagnostics.error(
                            "FIXED_EVENT_EXTENT_REASON_MISSING",
                            "Zero extent uncertainty requires extent.fixed_reason.",
                            path=f"{path}.extent.fixed_reason",
                        )
                if required_sides and not str(extent.get("source", "")).strip():
                    diagnostics.error(
                        "EVENT_EXTENT_SOURCE_MISSING",
                        "extent.source must identify the map, video, or measurement evidence.",
                        path=f"{path}.extent.source",
                    )
            obstacle_model = event.get("obstacle_model")
            if not isinstance(obstacle_model, Mapping):
                diagnostics.error(
                    "OBSTACLE_MODEL_MISSING",
                    "Every physical feature requires an explicit obstacle model, including an explicit none model.",
                    path=f"{path}.obstacle_model",
                    hint="Select a documented obstacle profile rather than leaving the mechanism implicit.",
                )
            else:
                try:
                    validate_obstacle_model_contract(obstacle_model)
                except (UncertaintyValidationError, UnitValidationError) as exc:
                    diagnostics.error(
                        "INVALID_OBSTACLE_MODEL",
                        str(exc),
                        path=f"{path}.obstacle_model",
                    )
        if lap_gate_count > 1:
            diagnostics.error(
                "MULTIPLE_LAP_GATE_ROLES",
                "At most one event may use analysis_role='lap_gate'.",
                path="events",
            )

    def _validate_track(self, raw_track: Any, diagnostics: DiagnosticBag) -> None:
        if not isinstance(raw_track, Mapping):
            diagnostics.error("TRACK_TABLE_MISSING", "track.toml requires a [track] table.")
            return
        if not str(raw_track.get("name", "")).strip():
            diagnostics.error("TRACK_NAME_MISSING", "track.name must be non-empty.")
        self._validate_physical_tree(
            raw_track.get("surface"), ("track", "surface"), diagnostics
        )
        if raw_track.get("closed_course") is not True:
            diagnostics.error(
                "CLOSED_COURSE_REQUIRED",
                "Phase 3 currently reconstructs closed courses only.",
                path="track.closed_course",
            )
        elevation = raw_track.get("elevation", {})
        if not isinstance(elevation, Mapping):
            diagnostics.error("INVALID_ELEVATION_CONFIG", "track.elevation must be a table.")
            return
        if elevation.get("store_from_gpx") is not True:
            diagnostics.warning(
                "GPX_ELEVATION_NOT_STORED",
                "GPX elevation is available to preserve and should normally be stored.",
                path="track.elevation.store_from_gpx",
            )
        if elevation.get("use_for_grade_force") is True:
            diagnostics.error(
                "GRADE_FORCE_NOT_IMPLEMENTED",
                "Grade force from GPX altitude is deliberately disabled until altitude processing is validated.",
                path="track.elevation.use_for_grade_force",
                hint="Set this to false for the current implementation.",
            )
        reconstruction = raw_track.get("reconstruction")
        if not isinstance(reconstruction, Mapping):
            diagnostics.error(
                "RECONSTRUCTION_CONFIG_MISSING",
                "track.toml requires [track.reconstruction].",
                path="track.reconstruction",
            )
        else:
            if not str(reconstruction.get("lap_gate_event_id", "")).strip():
                diagnostics.warning(
                    "LAP_GATE_ID_NOT_CONFIGURED",
                    "No lap_gate_event_id is configured; exactly one event must use analysis_role='lap_gate'.",
                    path="track.reconstruction.lap_gate_event_id",
                )
            positive_fields = (
                "lap_gate_radius_m",
                "minimum_lap_time_s",
                "maximum_reasonable_speed_mps",
                "maximum_normal_time_step_s",
                "centreline_spacing_m",
                "profile_spacing_m",
                "maximum_map_error_m",
            )
            for field in positive_fields:
                value = reconstruction.get(field)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
                    diagnostics.error(
                        "INVALID_RECONSTRUCTION_SETTING",
                        f"track.reconstruction.{field} must be positive.",
                        path=f"track.reconstruction.{field}",
                    )
            speed_coverage = reconstruction.get(
                "minimum_speed_coverage_fraction"
            )
            if (
                not isinstance(speed_coverage, (int, float))
                or isinstance(speed_coverage, bool)
                or not 0 < float(speed_coverage) <= 1
            ):
                diagnostics.error(
                    "INVALID_SPEED_COVERAGE_THRESHOLD",
                    "minimum_speed_coverage_fraction must lie in (0, 1].",
                    path=(
                        "track.reconstruction."
                        "minimum_speed_coverage_fraction"
                    ),
                )
        gates = raw_track.get("gate_confidence")
        if not isinstance(gates, Mapping):
            diagnostics.error(
                "GATE_CONFIDENCE_CONFIG_MISSING",
                "track.toml requires [track.gate_confidence].",
                path="track.gate_confidence",
            )
        else:
            weights = [
                gates.get("weight_pass_count"),
                gates.get("weight_speed_repeatability"),
                gates.get("weight_braking_evidence"),
                gates.get("weight_pace_independence"),
                gates.get("weight_coordinate_quality"),
                gates.get("weight_cross_vehicle_agreement"),
            ]
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0 for value in weights):
                diagnostics.error(
                    "INVALID_GATE_CONFIDENCE_WEIGHT",
                    "Every gate-confidence weight must be explicitly declared and non-negative.",
                    path="track.gate_confidence",
                )
            elif sum(float(value) for value in weights) <= 0:
                diagnostics.error(
                    "ZERO_GATE_CONFIDENCE_WEIGHT_SUM",
                    "At least one gate-confidence component must have non-zero weight.",
                    path="track.gate_confidence",
                )

        windows = raw_track.get("event_windows")
        if not isinstance(windows, Mapping):
            diagnostics.error(
                "EVENT_WINDOWS_CONFIG_MISSING",
                "track.toml requires [track.event_windows].",
                path="track.event_windows",
            )
        else:
            for field in (
                "approach_before_m",
                "approach_gap_m",
                "entry_before_m",
                "entry_gap_m",
                "exit_gap_m",
                "exit_length_m",
                "recovery_limit_m",
            ):
                value = windows.get(field)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or float(value) < 0
                ):
                    diagnostics.error(
                        "INVALID_EVENT_WINDOW",
                        f"track.event_windows.{field} must be explicit and non-negative.",
                        path=f"track.event_windows.{field}",
                    )
            if all(
                isinstance(windows.get(field), (int, float))
                and not isinstance(windows.get(field), bool)
                for field in (
                    "approach_before_m",
                    "approach_gap_m",
                    "entry_before_m",
                    "entry_gap_m",
                )
            ):
                if float(windows["approach_before_m"]) <= float(windows["approach_gap_m"]):
                    diagnostics.error(
                        "EMPTY_APPROACH_WINDOW",
                        "approach_before_m must exceed approach_gap_m.",
                        path="track.event_windows",
                    )
                if float(windows["entry_before_m"]) <= float(windows["entry_gap_m"]):
                    diagnostics.error(
                        "EMPTY_ENTRY_WINDOW",
                        "entry_before_m must exceed entry_gap_m.",
                        path="track.event_windows",
                    )
        if isinstance(gates, Mapping):
            integer_fields = ("minimum_valid_passes", "target_pass_count")
            for field in integer_fields:
                value = gates.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    diagnostics.error(
                        "INVALID_GATE_COUNT_SETTING",
                        f"track.gate_confidence.{field} must be a positive integer.",
                        path=f"track.gate_confidence.{field}",
                    )
            if (
                isinstance(gates.get("minimum_valid_passes"), int)
                and isinstance(gates.get("target_pass_count"), int)
                and gates["target_pass_count"] < gates["minimum_valid_passes"]
            ):
                diagnostics.error(
                    "GATE_TARGET_BELOW_MINIMUM",
                    "target_pass_count must be at least minimum_valid_passes.",
                    path="track.gate_confidence",
                )
            for field in (
                "braking_threshold_mps",
                "repeatability_scale_mps",
                "vehicle_agreement_scale_mps",
            ):
                value = gates.get(field)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or float(value) <= 0
                ):
                    diagnostics.error(
                        "INVALID_GATE_SCALE_SETTING",
                        f"track.gate_confidence.{field} must be positive.",
                        path=f"track.gate_confidence.{field}",
                    )
            accept = gates.get("accept_score")
            review = gates.get("review_score")
            for field, value in (("accept_score", accept), ("review_score", review)):
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not 0 <= float(value) <= 100
                ):
                    diagnostics.error(
                        "INVALID_GATE_SCORE_THRESHOLD",
                        f"track.gate_confidence.{field} must lie in [0, 100].",
                        path=f"track.gate_confidence.{field}",
                    )
            if (
                isinstance(accept, (int, float))
                and not isinstance(accept, bool)
                and isinstance(review, (int, float))
                and not isinstance(review, bool)
                and float(accept) < float(review)
            ):
                diagnostics.error(
                    "GATE_ACCEPT_BELOW_REVIEW",
                    "accept_score must be at least review_score.",
                    path="track.gate_confidence",
                )

    def _validate_track_event_links(
        self, raw_track: Any, raw_events: Any, diagnostics: DiagnosticBag
    ) -> None:
        if not isinstance(raw_track, Mapping) or not isinstance(raw_events, list):
            return
        reconstruction = raw_track.get("reconstruction", {})
        configured = (
            str(reconstruction.get("lap_gate_event_id", "")).strip()
            if isinstance(reconstruction, Mapping)
            else ""
        )
        event_ids = {
            str(event.get("id", ""))
            for event in raw_events
            if isinstance(event, Mapping)
        }
        if configured and configured not in event_ids:
            diagnostics.error(
                "LAP_GATE_EVENT_NOT_FOUND",
                f"Configured lap gate event {configured!r} does not exist.",
                path="track.reconstruction.lap_gate_event_id",
            )
        if not configured:
            role_count = sum(
                isinstance(event, Mapping) and event.get("analysis_role") == "lap_gate"
                for event in raw_events
            )
            if role_count != 1:
                diagnostics.error(
                    "LAP_GATE_EVENT_AMBIGUOUS",
                    "Without lap_gate_event_id, exactly one event must use analysis_role='lap_gate'.",
                    path="events",
                )

    def _validate_vehicles(self, raw_vehicles: Any, diagnostics: DiagnosticBag) -> None:
        if not isinstance(raw_vehicles, Mapping):
            diagnostics.error("VEHICLES_INVALID", "Resolved vehicles must be a table.")
            return
        for vehicle_id, config in raw_vehicles.items():
            if not isinstance(config, Mapping):
                diagnostics.error(
                    "VEHICLE_CONFIG_INVALID",
                    "Vehicle configuration must be a table.",
                    path=f"vehicles.{vehicle_id}",
                )
                continue
            self._validate_physical_tree(config.get("vehicle"), ("vehicles", str(vehicle_id), "vehicle"), diagnostics)
            self._validate_physical_tree(config.get("drivetrain"), ("vehicles", str(vehicle_id), "drivetrain"), diagnostics)
            self._validate_vehicle_relationships(str(vehicle_id), config, diagnostics)
            inherited = [
                path
                for path, quantity in _iter_quantities(config, ("vehicles", str(vehicle_id)))
                if quantity.source.kind is SourceKind.INHERITED_DEFAULT
            ]
            if inherited:
                preview = ", ".join(inherited[:5]) + (" ..." if len(inherited) > 5 else "")
                diagnostics.warning(
                    "INHERITED_DEFAULTS_ACTIVE",
                    f"{len(inherited)} physical inputs still use broad inherited defaults: {preview}.",
                    path=f"vehicles.{vehicle_id}",
                    hint="Review these before treating absolute simulation outputs as calibrated predictions.",
                )

    def _validate_physical_tree(
        self,
        raw: Any,
        path: tuple[str, ...],
        diagnostics: DiagnosticBag,
    ) -> None:
        if not isinstance(raw, Mapping):
            diagnostics.error(
                "PHYSICAL_SECTION_MISSING",
                "Required physical configuration section is missing.",
                path=".".join(path),
            )
            return
        for child_path, value in _walk_physical_nodes(raw, path):
            dotted = ".".join(child_path)
            if isinstance(value, Mapping) and "nominal" in value:
                try:
                    if "unit" in value:
                        quantity = UncertainQuantity.from_mapping(value)
                        expected = _expected_dimension(dotted)
                        if expected is not None:
                            require_dimension(quantity.unit, expected)
                        _validate_quantity_domain(dotted, quantity)
                        _validate_unbounded_normal_support(dotted, quantity, diagnostics)
                        source = quantity.source
                    else:
                        choice = UncertainChoice.from_mapping(value)
                        source = choice.source
                    if "replace with" in source.reference.lower():
                        diagnostics.warning(
                            "PLACEHOLDER_SOURCE_REFERENCE",
                            "Source reference still contains template placeholder text.",
                            path=dotted,
                            hint="Replace it with a measurement, document, or estimate record.",
                        )
                except (UncertaintyValidationError, UnitValidationError) as exc:
                    diagnostics.error(
                        "INVALID_UNCERTAIN_INPUT",
                        str(exc),
                        path=dotted,
                    )
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                diagnostics.error(
                    "BARE_PHYSICAL_NUMBER",
                    "Physical numeric inputs must be uncertainty-aware quantity tables, not bare numbers.",
                    path=dotted,
                    hint="Declare nominal, unit, source, and uncertainty together.",
                )

    def _validate_vehicle_relationships(
        self, vehicle_id: str, config: Mapping[str, Any], diagnostics: DiagnosticBag
    ) -> None:
        required_paths = (
            "vehicle.mass",
            "vehicle.gravity",
            "vehicle.tire_diameter",
            "vehicle.wheel_rotational_inertia",
            "vehicle.driven_normal_load_fraction",
            "vehicle.aero.drag_area",
            "vehicle.aero.air_density",
            "vehicle.rolling_resistance_coefficient",
            "vehicle.tire.peak_traction_scale",
            "vehicle.tire.slip_stiffness",
            "drivetrain.final_drive_ratio",
            "drivetrain.efficiency",
            "drivetrain.cvt.maximum_reduction_ratio",
            "drivetrain.cvt.minimum_reduction_ratio",
            "drivetrain.cvt.launch_clutch_model",
            "drivetrain.engine.model",
            "drivetrain.engine.target_speed",
            "drivetrain.engine.power_scale",
        )
        for relative in required_paths:
            if _nested(config, relative.split(".")) is None:
                diagnostics.error(
                    "SIMULATION_INPUT_MISSING",
                    "Required Phase 5 simulation input is missing.",
                    path=f"vehicles.{vehicle_id}.{relative}",
                )
        try:
            drivetrain = config["drivetrain"]
            cvt = drivetrain["cvt"]
            maximum_raw = cvt["maximum_reduction_ratio"]
            minimum_raw = cvt["minimum_reduction_ratio"]
            maximum = float(maximum_raw["nominal"])
            minimum = float(minimum_raw["nominal"])
            if maximum <= minimum:
                diagnostics.error(
                    "CVT_RATIO_ORDER",
                    "maximum_reduction_ratio must exceed minimum_reduction_ratio.",
                    path=f"vehicles.{vehicle_id}.drivetrain.cvt",
                )
            else:
                maximum_quantity = UncertainQuantity.from_mapping(maximum_raw)
                minimum_quantity = UncertainQuantity.from_mapping(minimum_raw)
                maximum_lower, _ = _bounded_support(maximum_quantity)
                _, minimum_upper = _bounded_support(minimum_quantity)
                if (
                    maximum_lower is not None
                    and minimum_upper is not None
                    and maximum_lower <= minimum_upper
                ):
                    diagnostics.error(
                        "CVT_RATIO_UNCERTAINTY_OVERLAP",
                        (
                            "Declared ratio uncertainty can produce maximum_reduction_ratio "
                            "less than or equal to minimum_reduction_ratio."
                        ),
                        path=f"vehicles.{vehicle_id}.drivetrain.cvt",
                        hint="Use non-overlapping supports or a constrained parameterization.",
                    )
            driven_raw = config["vehicle"]["driven_normal_load_fraction"]
            driven_fraction = float(driven_raw["nominal"])
            if not 0 < driven_fraction <= 1:
                diagnostics.error(
                    "DRIVEN_LOAD_FRACTION_DOMAIN",
                    "driven_normal_load_fraction must lie in (0, 1].",
                    path=f"vehicles.{vehicle_id}.vehicle.driven_normal_load_fraction",
                )
            else:
                _, driven_upper = _bounded_support(
                    UncertainQuantity.from_mapping(driven_raw)
                )
                if driven_upper is not None and driven_upper > 1:
                    diagnostics.error(
                        "DRIVEN_LOAD_FRACTION_UNCERTAINTY_DOMAIN",
                        "driven_normal_load_fraction uncertainty cannot exceed one.",
                        path=f"vehicles.{vehicle_id}.vehicle.driven_normal_load_fraction",
                    )
        except (KeyError, TypeError, ValueError):
            pass

    def _validate_studies(
        self,
        raw_studies: Any,
        raw_vehicles: Any,
        raw_track: Any,
        raw_events: Any,
        diagnostics: DiagnosticBag,
    ) -> None:
        if not isinstance(raw_studies, Mapping):
            diagnostics.error("STUDIES_INVALID", "Resolved studies must be a table.")
            return
        vehicle_ids = set(raw_vehicles) if isinstance(raw_vehicles, Mapping) else set()
        valid_types = {
            "baseline",
            "design_sweep",
            "track_robustness",
            "structural_sensitivity",
            "full_uncertainty",
        }
        valid_modes = {"nominal", "measured_track", "selected_structural", "all_declared"}
        for name, raw in raw_studies.items():
            path = f"studies.{name}"
            if not isinstance(raw, Mapping):
                diagnostics.error("INVALID_STUDY", "Study must be a table.", path=path)
                continue
            study = raw.get("study")
            if not isinstance(study, Mapping):
                diagnostics.error("STUDY_TABLE_MISSING", "Study requires [study].", path=path)
                continue
            study_type = str(study.get("type", ""))
            if study_type not in valid_types:
                diagnostics.error(
                    "UNKNOWN_STUDY_TYPE",
                    f"Unknown study type {study_type!r}.",
                    path=f"{path}.study.type",
                )
            random_seed = study.get("random_seed", 20260715)
            if (
                not isinstance(random_seed, int)
                or isinstance(random_seed, bool)
                or random_seed < 0
            ):
                diagnostics.error(
                    "INVALID_RANDOM_SEED",
                    "study.random_seed must be a non-negative integer.",
                    path=f"{path}.study.random_seed",
                )
            vehicle_id = str(study.get("vehicle_id", ""))
            if vehicle_id not in vehicle_ids:
                diagnostics.error(
                    "STUDY_VEHICLE_NOT_FOUND",
                    f"Study references unknown vehicle {vehicle_id!r}.",
                    path=f"{path}.study.vehicle_id",
                )
            if study_type in {"design_sweep", "track_robustness", "structural_sensitivity", "full_uncertainty"}:
                base_case = raw.get("base_case")
                base_name = (
                    str(base_case.get("study", ""))
                    if isinstance(base_case, Mapping)
                    else ""
                )
                base_raw = raw_studies.get(base_name) if base_name else None
                if not isinstance(base_raw, Mapping) or str(
                    base_raw.get("study", {}).get("type", "")
                ) != "baseline":
                    diagnostics.error(
                        "INVALID_BASE_CASE_STUDY",
                        "Non-baseline studies require [base_case] study to reference a baseline study.",
                        path=f"{path}.base_case.study",
                    )
            if study_type == "baseline":
                self._validate_physical_tree(
                    raw.get("driver"), ("studies", str(name), "driver"), diagnostics
                )
                self._validate_physical_tree(
                    raw.get("initial_conditions"),
                    ("studies", str(name), "initial_conditions"),
                    diagnostics,
                )
                simulation = raw.get("simulation")
                if not isinstance(simulation, Mapping):
                    diagnostics.error(
                        "SIMULATION_SETTINGS_MISSING",
                        "Baseline study requires [simulation] numerical settings.",
                        path=f"{path}.simulation",
                    )
                else:
                    required_numeric = (
                        "maximum_time_s",
                        "integration_step_s",
                        "report_step_s",
                    )
                    for field in required_numeric:
                        value = simulation.get(field)
                        if not isinstance(value, (int, float)) or isinstance(value, bool):
                            diagnostics.error(
                                "INVALID_SIMULATION_SETTING",
                                f"simulation.{field} must be numeric.",
                                path=f"{path}.simulation.{field}",
                            )
                    try:
                        if float(simulation["integration_step_s"]) <= 0 or float(simulation["report_step_s"]) < float(simulation["integration_step_s"]):
                            diagnostics.error(
                                "INVALID_SIMULATION_STEPS",
                                "integration_step_s must be positive and report_step_s must be at least as large.",
                                path=f"{path}.simulation",
                            )
                    except (KeyError, TypeError, ValueError):
                        pass
                realization = raw.get("track_realization")
                if not isinstance(realization, Mapping) or str(
                    realization.get("gate_speed_statistic", "")
                ) not in {"p10", "median", "p90"}:
                    diagnostics.error(
                        "INVALID_GATE_SPEED_STATISTIC",
                        "Baseline track_realization.gate_speed_statistic must be p10, median, or p90.",
                        path=f"{path}.track_realization.gate_speed_statistic",
                    )
            sampling = raw.get("sampling", {})
            if study_type != "structural_sensitivity":
                if not isinstance(sampling, Mapping):
                    diagnostics.error("SAMPLING_TABLE_MISSING", "Study requires [sampling].", path=path)
                else:
                    mode = str(sampling.get("mode", ""))
                    if mode not in valid_modes:
                        diagnostics.error(
                            "UNKNOWN_SAMPLING_MODE",
                            f"Unknown sampling mode {mode!r}.",
                            path=f"{path}.sampling.mode",
                        )
                    if mode == "selected_structural":
                        selected_paths = sampling.get("paths")
                        if (
                            not isinstance(selected_paths, list)
                            or not selected_paths
                            or not all(isinstance(item, str) and item for item in selected_paths)
                        ):
                            diagnostics.error(
                                "SELECTED_STRUCTURAL_PATHS_MISSING",
                                "sampling.mode='selected_structural' requires a non-empty string array sampling.paths.",
                                path=f"{path}.sampling.paths",
                            )
                        elif len(selected_paths) != len(set(selected_paths)):
                            diagnostics.error(
                                "DUPLICATE_SELECTED_STRUCTURAL_PATHS",
                                "sampling.paths must not contain duplicates.",
                                path=f"{path}.sampling.paths",
                            )
                        else:
                            vehicle = (
                                raw_vehicles.get(vehicle_id, {})
                                if isinstance(raw_vehicles, Mapping)
                                else {}
                            )
                            base_case = raw.get("base_case", {})
                            base_name = (
                                str(base_case.get("study", ""))
                                if isinstance(base_case, Mapping)
                                else ""
                            )
                            base_raw = (
                                raw_studies.get(base_name, {})
                                if isinstance(raw_studies, Mapping)
                                else {}
                            )
                            for selected_path in selected_paths:
                                selected_quantity = _study_numeric_input(
                                    selected_path,
                                    vehicle=vehicle,
                                    baseline=base_raw if isinstance(base_raw, Mapping) else {},
                                    track=raw_track if isinstance(raw_track, Mapping) else {},
                                    events=raw_events if isinstance(raw_events, list) else [],
                                )
                                if selected_quantity is None:
                                    diagnostics.error(
                                        "SELECTED_STRUCTURAL_PATH_NOT_FOUND",
                                        f"Selected uncertainty path {selected_path!r} is not a declared numeric input.",
                                        path=f"{path}.sampling.paths",
                                    )
                                    continue
                                if selected_quantity.uncertainty.distribution is DistributionKind.FIXED:
                                    diagnostics.error(
                                        "SELECTED_STRUCTURAL_PATH_FIXED",
                                        f"Selected uncertainty path {selected_path!r} is explicitly fixed.",
                                        path=f"{path}.sampling.paths",
                                        hint="Declare non-zero structural uncertainty or remove the path.",
                                    )
                                role = (
                                    selected_quantity.uncertainty.role.value
                                    if selected_quantity.uncertainty.role is not None
                                    else _default_uncertainty_role(selected_path)
                                )
                                if role != "structural":
                                    diagnostics.error(
                                        "SELECTED_STRUCTURAL_ROLE_MISMATCH",
                                        (
                                            f"Selected uncertainty path {selected_path!r} has role {role!r}; "
                                            "selected_structural accepts only structural inputs."
                                        ),
                                        path=f"{path}.sampling.paths",
                                    )
                    if mode != "nominal":
                        replicates = sampling.get("replicates")
                        if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
                            diagnostics.error(
                                "INVALID_REPLICATE_COUNT",
                                "Non-nominal sampling requires a positive integer replicates value.",
                                path=f"{path}.sampling.replicates",
                            )
                        if sampling.get("paired_scenarios") is not True:
                            diagnostics.error(
                                "UNPAIRED_DESIGN_SCENARIOS",
                                "Phase 6 comparisons require paired_scenarios=true so every design sees the same realization.",
                                path=f"{path}.sampling.paired_scenarios",
                            )
                        if str(sampling.get("gate_sampling", "paired_lap")) not in {"paired_lap", "independent"}:
                            diagnostics.error(
                                "INVALID_GATE_SAMPLING_MODE",
                                "sampling.gate_sampling must be paired_lap or independent.",
                                path=f"{path}.sampling.gate_sampling",
                            )
            if study_type == "design_sweep":
                design = raw.get("design_variable")
                if not isinstance(design, Mapping):
                    diagnostics.error(
                        "DESIGN_VARIABLE_MISSING",
                        "Design sweep requires [design_variable].",
                        path=path,
                    )
                else:
                    values = design.get("values")
                    if not isinstance(values, list) or not values or not all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and isfinite(float(value))
                        for value in values
                    ):
                        diagnostics.error(
                            "INVALID_DESIGN_VALUES",
                            "design_variable.values must be a non-empty numeric array.",
                            path=f"{path}.design_variable.values",
                        )
                    elif len(values) != len(set(float(value) for value in values)):
                        diagnostics.error(
                            "DUPLICATE_DESIGN_VALUES",
                            "design_variable.values must not contain duplicates.",
                            path=f"{path}.design_variable.values",
                        )
                    dotted = str(design.get("path", ""))
                    vehicle = (
                        raw_vehicles.get(vehicle_id, {})
                        if isinstance(raw_vehicles, Mapping)
                        else {}
                    )
                    design_quantity = (
                        _study_numeric_input(
                            dotted,
                            vehicle=vehicle if isinstance(vehicle, Mapping) else {},
                            baseline={},
                            track={},
                            events=[],
                        )
                        if dotted
                        else None
                    )
                    if design_quantity is None:
                        diagnostics.error(
                            "DESIGN_PATH_NOT_FOUND",
                            f"Design variable path {dotted!r} does not identify a numeric quantity for vehicle {vehicle_id!r}.",
                            path=f"{path}.design_variable.path",
                            hint="Paths are relative to the selected vehicle configuration.",
                        )
                    elif isinstance(values, list):
                        for value in values:
                            if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(float(value)):
                                continue
                            message = _design_value_domain_error(
                                dotted, float(value), vehicle if isinstance(vehicle, Mapping) else {}
                            )
                            if message is not None:
                                diagnostics.error(
                                    "DESIGN_VALUE_OUT_OF_DOMAIN",
                                    f"Design value {value!r} for {dotted!r} is invalid: {message}",
                                    path=f"{path}.design_variable.values",
                                )
            if study_type == "structural_sensitivity":
                sensitivity = raw.get("sensitivity")
                if not isinstance(sensitivity, Mapping):
                    diagnostics.error(
                        "SENSITIVITY_TABLE_MISSING",
                        "Structural sensitivity requires [sensitivity].",
                        path=path,
                    )
                else:
                    parameters = sensitivity.get("parameters")
                    if not isinstance(parameters, list) or not parameters or not all(
                        isinstance(item, str) and item for item in parameters
                    ):
                        diagnostics.error(
                            "NO_SENSITIVITY_PARAMETERS",
                            "Structural sensitivity requires a non-empty string array of parameters.",
                            path=f"{path}.sensitivity.parameters",
                        )
                    elif len(parameters) != len(set(parameters)):
                        diagnostics.error(
                            "DUPLICATE_SENSITIVITY_PARAMETERS",
                            "Sensitivity parameter paths must be unique.",
                            path=f"{path}.sensitivity.parameters",
                        )
                    vehicle = raw_vehicles.get(vehicle_id, {}) if isinstance(raw_vehicles, Mapping) else {}
                    base_case = raw.get("base_case", {})
                    base_name = str(base_case.get("study", "")) if isinstance(base_case, Mapping) else ""
                    base_raw = raw_studies.get(base_name, {}) if isinstance(raw_studies, Mapping) else {}
                    has_numeric_sensitivity = False
                    if isinstance(parameters, list):
                        for parameter in parameters:
                            if not isinstance(parameter, str):
                                continue
                            uncertain_input = _study_uncertain_input(
                                parameter,
                                vehicle=vehicle,
                                baseline=base_raw if isinstance(base_raw, Mapping) else {},
                                track=raw_track if isinstance(raw_track, Mapping) else {},
                                events=raw_events if isinstance(raw_events, list) else [],
                            )
                            if uncertain_input is None:
                                diagnostics.error(
                                    "SENSITIVITY_PATH_NOT_FOUND",
                                    (
                                        f"Sensitivity path {parameter!r} does not identify a declared "
                                        "numeric or categorical uncertainty input for the selected vehicle, "
                                        "baseline, track surface, or obstacle model."
                                    ),
                                    path=f"{path}.sensitivity.parameters",
                                )
                                continue
                            if isinstance(uncertain_input, UncertainQuantity):
                                has_numeric_sensitivity = True
                            if uncertain_input.uncertainty.distribution is DistributionKind.FIXED:
                                diagnostics.error(
                                    "SENSITIVITY_PARAMETER_FIXED",
                                    f"Sensitivity path {parameter!r} is explicitly fixed and has no declared range.",
                                    path=f"{path}.sensitivity.parameters",
                                    hint="Declare a non-zero uncertainty model or remove it from the sensitivity study.",
                                )
                            role = (
                                uncertain_input.uncertainty.role.value
                                if uncertain_input.uncertainty.role is not None
                                else _default_uncertainty_role(parameter)
                            )
                            if role != "structural":
                                diagnostics.error(
                                    "SENSITIVITY_PARAMETER_NOT_STRUCTURAL",
                                    f"Sensitivity path {parameter!r} must have uncertainty.role='structural'.",
                                    path=f"{path}.sensitivity.parameters",
                                )
                    quantiles = sensitivity.get("quantiles")
                    if has_numeric_sensitivity:
                        if not isinstance(quantiles, list) or len(quantiles) < 3 or not all(
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and 0 < float(value) < 1
                            for value in quantiles
                        ):
                            diagnostics.error(
                                "INVALID_SENSITIVITY_QUANTILES",
                                "Numeric sensitivity parameters require at least three probabilities strictly inside (0, 1).",
                                path=f"{path}.sensitivity.quantiles",
                            )
                        elif quantiles != sorted(quantiles) or len(quantiles) != len(set(float(v) for v in quantiles)):
                            diagnostics.error(
                                "UNORDERED_SENSITIVITY_QUANTILES",
                                "sensitivity.quantiles must be unique and increasing.",
                                path=f"{path}.sensitivity.quantiles",
                            )
                    elif quantiles not in (None, []):
                        diagnostics.warning(
                            "UNUSED_SENSITIVITY_QUANTILES",
                            "sensitivity.quantiles are ignored because every selected sensitivity input is categorical.",
                            path=f"{path}.sensitivity.quantiles",
                        )

            quality = raw.get("quality", {})
            if quality and not isinstance(quality, Mapping):
                diagnostics.error(
                    "INVALID_QUALITY_SETTINGS",
                    "quality must be a table.",
                    path=f"{path}.quality",
                )
            elif isinstance(quality, Mapping):
                for field in (
                    "maximum_abs_energy_balance_relative_error",
                    "maximum_abs_powertrain_balance_relative_error",
                ):
                    value = quality.get(field, 0.01)
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not 0 < float(value) < 1
                    ):
                        diagnostics.error(
                            "INVALID_QUALITY_THRESHOLD",
                            f"quality.{field} must lie in (0, 1).",
                            path=f"{path}.quality.{field}",
                        )

            reporting = raw.get("reporting", {})
            if reporting and (
                not isinstance(reporting, Mapping)
                or not isinstance(reporting.get("bootstrap_resamples", 1000), int)
                or isinstance(reporting.get("bootstrap_resamples", 1000), bool)
                or int(reporting.get("bootstrap_resamples", 1000)) < 100
            ):
                diagnostics.error(
                    "INVALID_BOOTSTRAP_RESAMPLES",
                    "reporting.bootstrap_resamples must be an integer of at least 100.",
                    path=f"{path}.reporting.bootstrap_resamples",
                )
            correlations = raw.get("correlations", [])
            if correlations not in (None, []):
                if not isinstance(correlations, list):
                    diagnostics.error(
                        "INVALID_CORRELATIONS",
                        "correlations must be an array of tables.",
                        path=f"{path}.correlations",
                    )
                else:
                    ids: set[str] = set()
                    used_members: set[str] = set()
                    for index, correlation in enumerate(correlations):
                        cpath = f"{path}.correlations.{index}"
                        if not isinstance(correlation, Mapping):
                            diagnostics.error(
                                "INVALID_CORRELATION",
                                "Correlation entry must be a table.",
                                path=cpath,
                            )
                            continue
                        identifier = str(correlation.get("id", ""))
                        members = correlation.get("members")
                        matrix = correlation.get("matrix")
                        if not identifier or identifier in ids:
                            diagnostics.error(
                                "INVALID_CORRELATION_ID",
                                "Correlation ids must be unique and non-empty.",
                                path=cpath,
                            )
                        ids.add(identifier)
                        members_valid = (
                            isinstance(members, list)
                            and len(members) >= 2
                            and all(isinstance(item, str) and item for item in members)
                            and len(members) == len(set(members))
                        )
                        if not members_valid:
                            diagnostics.error(
                                "INVALID_CORRELATION_MEMBERS",
                                "Correlation members require at least two unique non-empty paths.",
                                path=f"{cpath}.members",
                            )
                        elif set(members) & used_members:
                            diagnostics.error(
                                "OVERLAPPING_CORRELATION_GROUPS",
                                "An input path may appear in only one Gaussian-copula correlation group.",
                                path=f"{cpath}.members",
                            )
                        if members_valid:
                            used_members.update(members)
                            vehicle_for_correlation = (
                                raw_vehicles.get(vehicle_id, {})
                                if isinstance(raw_vehicles, Mapping)
                                else {}
                            )
                            base_case_for_correlation = raw.get("base_case", {})
                            base_name_for_correlation = (
                                str(base_case_for_correlation.get("study", ""))
                                if isinstance(base_case_for_correlation, Mapping)
                                else ""
                            )
                            baseline_for_correlation = (
                                raw_studies.get(base_name_for_correlation, {})
                                if isinstance(raw_studies, Mapping)
                                else {}
                            )
                            selected_structural_paths = (
                                set(sampling.get("paths", ()))
                                if isinstance(sampling, Mapping)
                                and str(sampling.get("mode", "")) == "selected_structural"
                                else set()
                            )
                            design_path_for_correlation = (
                                str(raw.get("design_variable", {}).get("path", ""))
                                if study_type == "design_sweep"
                                and isinstance(raw.get("design_variable"), Mapping)
                                else ""
                            )
                            for member in members:
                                uncertain_input = _study_uncertain_input(
                                    member,
                                    vehicle=(
                                        vehicle_for_correlation
                                        if isinstance(vehicle_for_correlation, Mapping)
                                        else {}
                                    ),
                                    baseline=(
                                        baseline_for_correlation
                                        if isinstance(baseline_for_correlation, Mapping)
                                        else {}
                                    ),
                                    track=(raw_track if isinstance(raw_track, Mapping) else {}),
                                    events=(raw_events if isinstance(raw_events, list) else []),
                                )
                                if uncertain_input is None:
                                    diagnostics.error(
                                        "CORRELATION_MEMBER_NOT_FOUND",
                                        f"Correlation member {member!r} is not a declared uncertainty input.",
                                        path=f"{cpath}.members",
                                    )
                                    continue
                                if uncertain_input.uncertainty.distribution is DistributionKind.FIXED:
                                    diagnostics.error(
                                        "CORRELATION_MEMBER_FIXED",
                                        f"Correlation member {member!r} is explicitly fixed and cannot be sampled.",
                                        path=f"{cpath}.members",
                                    )
                                role = (
                                    uncertain_input.uncertainty.role.value
                                    if uncertain_input.uncertainty.role is not None
                                    else _default_uncertainty_role(member)
                                )
                                selected_by_mode = (
                                    mode == "all_declared"
                                    or (mode == "measured_track" and role == "measured_track")
                                    or (
                                        mode == "selected_structural"
                                        and role == "structural"
                                        and member in selected_structural_paths
                                    )
                                )
                                if study_type == "structural_sensitivity" or mode == "nominal":
                                    selected_by_mode = False
                                if member == design_path_for_correlation:
                                    selected_by_mode = False
                                if not selected_by_mode:
                                    diagnostics.error(
                                        "CORRELATION_MEMBER_NOT_SAMPLED",
                                        (
                                            f"Correlation member {member!r} is not sampled by mode {mode!r} "
                                            "after study exclusions."
                                        ),
                                        path=f"{cpath}.members",
                                    )
                                declared_group = getattr(uncertain_input, "correlation_group", None)
                                if declared_group not in (None, identifier):
                                    diagnostics.error(
                                        "CORRELATION_GROUP_MISMATCH",
                                        (
                                            f"Correlation member {member!r} declares group "
                                            f"{declared_group!r}, not {identifier!r}."
                                        ),
                                        path=f"{cpath}.members",
                                    )
                        shape_valid = (
                            isinstance(matrix, list)
                            and isinstance(members, list)
                            and len(matrix) == len(members)
                            and all(
                                isinstance(row, list) and len(row) == len(members)
                                for row in matrix
                            )
                        )
                        if not shape_valid:
                            diagnostics.error(
                                "INVALID_CORRELATION_MATRIX_SHAPE",
                                "Correlation matrix must be square and match members.",
                                path=f"{cpath}.matrix",
                            )
                            continue
                        try:
                            array = np.asarray(matrix, dtype=float)
                        except (TypeError, ValueError):
                            diagnostics.error(
                                "INVALID_CORRELATION_MATRIX_VALUES",
                                "Correlation matrix entries must be numeric.",
                                path=f"{cpath}.matrix",
                            )
                            continue
                        if not np.all(np.isfinite(array)):
                            diagnostics.error(
                                "INVALID_CORRELATION_MATRIX_VALUES",
                                "Correlation matrix entries must be finite.",
                                path=f"{cpath}.matrix",
                            )
                        elif not np.allclose(array, array.T, atol=1e-12):
                            diagnostics.error(
                                "NONSYMMETRIC_CORRELATION_MATRIX",
                                "Correlation matrix must be symmetric.",
                                path=f"{cpath}.matrix",
                            )
                        elif not np.allclose(np.diag(array), 1.0, atol=1e-12):
                            diagnostics.error(
                                "INVALID_CORRELATION_DIAGONAL",
                                "Correlation matrix diagonal entries must all equal one.",
                                path=f"{cpath}.matrix",
                            )
                        elif float(np.min(np.linalg.eigvalsh(array))) < -1e-10:
                            diagnostics.error(
                                "NON_PSD_CORRELATION_MATRIX",
                                "Correlation matrix must be positive semidefinite.",
                                path=f"{cpath}.matrix",
                            )





def _design_value_domain_error(
    path: str, value: float, vehicle: Mapping[str, Any]
) -> str | None:
    if path.endswith(_STRICTLY_POSITIVE_SUFFIXES) and value <= 0.0:
        return "the value must be strictly positive"
    if path.endswith((".braking_trigger_margin", ".vehicle_speed", ".wheel_speed")) and value < 0.0:
        return "the value must be non-negative"
    if path.endswith(".efficiency") and value > 1.0:
        return "efficiency must not exceed one"
    if path.endswith(".driven_normal_load_fraction") and value > 1.0:
        return "driven_normal_load_fraction must not exceed one"
    if path == "drivetrain.cvt.maximum_reduction_ratio":
        other = _get_path(vehicle, "drivetrain.cvt.minimum_reduction_ratio.nominal")
        if other is not _MISSING and value <= float(other):
            return "maximum_reduction_ratio must remain above the nominal minimum_reduction_ratio"
    if path == "drivetrain.cvt.minimum_reduction_ratio":
        other = _get_path(vehicle, "drivetrain.cvt.maximum_reduction_ratio.nominal")
        if other is not _MISSING and value >= float(other):
            return "minimum_reduction_ratio must remain below the nominal maximum_reduction_ratio"
    return None

def _default_uncertainty_role(path: str) -> str:
    if path.startswith("initial_conditions."):
        return "initial_condition"
    return "structural"

def _study_uncertain_input(
    path: str,
    *,
    vehicle: Mapping[str, Any],
    baseline: Mapping[str, Any],
    track: Mapping[str, Any],
    events: list[Any],
) -> UncertainQuantity | UncertainChoice | None:
    raw: Any = _MISSING
    if path.startswith(("vehicle.", "drivetrain.")):
        raw = _get_path(vehicle, path)
    elif path.startswith("driver."):
        raw = _get_path(baseline, path)
    elif path.startswith("initial_conditions."):
        raw = _get_path(baseline, path)
    elif path.startswith("track.surface."):
        raw = _get_path(track, path.removeprefix("track."))
    elif path.startswith("obstacle."):
        parts = path.split(".")
        feature_id = parts[1] if len(parts) >= 2 else ""
        event = next(
            (
                item
                for item in events
                if isinstance(item, Mapping) and str(item.get("id")) == feature_id
            ),
            None,
        )
        if isinstance(event, Mapping) and isinstance(event.get("obstacle_model"), Mapping):
            try:
                choice, alternatives = obstacle_model_alternatives(event["obstacle_model"])
                if len(parts) == 3 and parts[2] == "model_type":
                    return choice
                if len(parts) == 4:
                    _, _, model_type, parameter = parts
                    raw = alternatives.get(model_type, {}).get(parameter, _MISSING)
            except (UncertaintyValidationError, UnitValidationError):
                raw = _MISSING
    if raw is _MISSING:
        return None
    if isinstance(raw, (UncertainQuantity, UncertainChoice)):
        return raw
    if isinstance(raw, Mapping) and "nominal" in raw:
        try:
            if "unit" in raw:
                return UncertainQuantity.from_mapping(raw)
            return UncertainChoice.from_mapping(raw)
        except UncertaintyValidationError:
            return None
    return None


def _study_numeric_input(
    path: str,
    *,
    vehicle: Mapping[str, Any],
    baseline: Mapping[str, Any],
    track: Mapping[str, Any],
    events: list[Any],
) -> UncertainQuantity | None:
    """Return whether a structural-study path identifies a numeric contract.

    Study paths use the same canonical names as the Phase 6 uncertainty
    registry. This validation runs before a track bundle exists, so obstacle
    paths are resolved from the already-profile-expanded event definitions.
    """

    value = _study_uncertain_input(
        path,
        vehicle=vehicle,
        baseline=baseline,
        track=track,
        events=events,
    )
    return value if isinstance(value, UncertainQuantity) else None


_MISSING = object()


def _get_path(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    node: Any = mapping
    for part in dotted_path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _walk_physical_nodes(
    mapping: Mapping[str, Any], path: tuple[str, ...]
) -> Iterable[tuple[tuple[str, ...], Any]]:
    for key, value in mapping.items():
        child = (*path, str(key))
        if isinstance(value, Mapping):
            if "nominal" in value:
                yield child, value
            else:
                yield from _walk_physical_nodes(value, child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield child, value


def _iter_quantities(
    mapping: Mapping[str, Any], path: tuple[str, ...]
) -> Iterable[tuple[str, UncertainQuantity]]:
    for child_path, value in _walk_physical_nodes(mapping, path):
        if isinstance(value, Mapping) and "nominal" in value:
            try:
                yield ".".join(child_path), UncertainQuantity.from_mapping(value)
            except UncertaintyValidationError:
                continue


def _expected_dimension(path: str) -> str | None:
    suffixes = {
        ".vehicle.mass": "mass",
        ".vehicle.gravity": "acceleration",
        ".vehicle.tire_diameter": "length",
        ".vehicle.aero.drag_area": "area",
        ".vehicle.rolling_resistance_coefficient": "dimensionless",
        ".vehicle.wheel_rotational_inertia": "rotational_inertia",
        ".vehicle.driven_normal_load_fraction": "dimensionless",
        ".vehicle.aero.air_density": "density",
        ".vehicle.tire.peak_traction_scale": "dimensionless",
        ".vehicle.tire.slip_stiffness": "slip_stiffness",
        ".drivetrain.final_drive_ratio": "dimensionless",
        ".drivetrain.cvt.maximum_reduction_ratio": "dimensionless",
        ".drivetrain.cvt.minimum_reduction_ratio": "dimensionless",
        ".drivetrain.efficiency": "dimensionless",
        ".drivetrain.engine.target_speed": "angular_speed",
        ".drivetrain.engine.power_scale": "dimensionless",
        ".track.surface.friction_coefficient": "dimensionless",
        ".driver.maximum_braking_deceleration": "acceleration",
        ".driver.maximum_brake_force": "force",
        ".driver.braking_trigger_margin": "speed",
        ".initial_conditions.vehicle_speed": "speed",
        ".initial_conditions.wheel_speed": "angular_speed",
    }
    for suffix, dimension in suffixes.items():
        if path.endswith(suffix):
            return dimension
    return None


_STRICTLY_POSITIVE_SUFFIXES = (
    ".mass",
    ".gravity",
    ".tire_diameter",
    ".final_drive_ratio",
    ".maximum_reduction_ratio",
    ".minimum_reduction_ratio",
    ".drag_area",
    ".rolling_resistance_coefficient",
    ".wheel_rotational_inertia",
    ".driven_normal_load_fraction",
    ".air_density",
    ".peak_traction_scale",
    ".slip_stiffness",
    ".target_speed",
    ".power_scale",
    ".friction_coefficient",
    ".maximum_braking_deceleration",
    ".maximum_brake_force",
)


def _validate_quantity_domain(path: str, quantity: UncertainQuantity) -> None:
    nonnegative_suffixes = (
        ".braking_trigger_margin",
        ".vehicle_speed",
        ".wheel_speed",
    )
    lower, upper = _bounded_support(quantity)
    if path.endswith(_STRICTLY_POSITIVE_SUFFIXES):
        if quantity.nominal <= 0:
            raise UncertaintyValidationError("nominal must be positive for this parameter.")
        if lower is not None and lower <= 0:
            raise UncertaintyValidationError(
                "the declared uncertainty support must remain strictly positive for this parameter."
            )
    if path.endswith(nonnegative_suffixes):
        if quantity.nominal < 0:
            raise UncertaintyValidationError("nominal must be non-negative for this parameter.")
        if lower is not None and lower < 0:
            raise UncertaintyValidationError(
                "the declared uncertainty support cannot extend below zero for this parameter."
            )
    if path.endswith(".efficiency"):
        if not 0 < quantity.nominal <= 1:
            raise UncertaintyValidationError("efficiency nominal must lie in (0, 1].")
        if lower is not None and lower <= 0:
            raise UncertaintyValidationError(
                "efficiency uncertainty support must remain strictly above zero."
            )
        if upper is not None and upper > 1:
            raise UncertaintyValidationError("efficiency uncertainty cannot exceed one.")


def _bounded_support(quantity: UncertainQuantity) -> tuple[float | None, float | None]:
    """Return finite declared support bounds, or ``None`` for normal tails."""

    spec = quantity.uncertainty
    if spec.distribution is DistributionKind.FIXED:
        return quantity.nominal, quantity.nominal
    if spec.distribution is DistributionKind.NORMAL:
        return None, None
    if spec.distribution in {
        DistributionKind.TRUNCATED_NORMAL,
        DistributionKind.UNIFORM,
        DistributionKind.TRIANGULAR,
    }:
        return spec.lower, spec.upper
    if spec.distribution is DistributionKind.EMPIRICAL:
        return min(spec.samples), max(spec.samples)
    return None, None


def _validate_unbounded_normal_support(
    path: str, quantity: UncertainQuantity, diagnostics: DiagnosticBag
) -> None:
    """Reject a normal prior with material probability in an impossible domain.

    Silently rejecting or resampling invalid Monte Carlo draws would change the
    declared distribution. The contract therefore requires a truncated normal
    whenever more than one part per million of the prior lies outside a hard
    physical domain. Negligible mathematical tails remain allowed.
    """

    if quantity.uncertainty.distribution is not DistributionKind.NORMAL:
        return
    sigma = quantity.uncertainty.standard_deviation_for(quantity.nominal)
    normal = NormalDist(mu=quantity.nominal, sigma=sigma)
    probability = 0.0
    invalid_region = ""
    if path.endswith(".efficiency"):
        probability = normal.cdf(0.0) + (1.0 - normal.cdf(1.0))
        invalid_region = "outside (0, 1]"
    elif path.endswith(_STRICTLY_POSITIVE_SUFFIXES):
        probability = normal.cdf(0.0)
        invalid_region = "at or below zero"
    elif path.endswith((".braking_trigger_margin", ".vehicle_speed", ".wheel_speed")):
        probability = normal.cdf(0.0)
        invalid_region = "below zero"
    if probability <= 1e-6:
        return
    diagnostics.error(
        "UNBOUNDED_NORMAL_PHYSICAL_SUPPORT",
        (
            f"The declared normal uncertainty assigns approximately "
            f"{100.0 * probability:.4g}% probability to values {invalid_region}."
        ),
        path=path,
        hint=(
            "Use distribution='truncated_normal' with physically meaningful "
            "bounds. Phase 6 does not silently reject or resample invalid draws."
        ),
    )


def _nested(raw: Mapping[str, Any], parts: Iterable[str]) -> Any:
    current: Any = raw
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

