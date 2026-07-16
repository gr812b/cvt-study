"""Project discovery, configuration resolution, and validation."""

from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from .diagnostics import Diagnostic, DiagnosticBag, Severity
from .export import export_resolution
from .merge import ProvenanceMap, deep_merge, prefix_provenance, set_existing_path
from .profiles import ProfileRegistry
from .toml_io import TomlError, load_toml
from .validation import validate_project


class ProjectError(RuntimeError):
    """Fatal error that prevents a project from being resolved at all."""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    project_file: Path
    track_file: Path
    runs_file: Path
    events_file: Path
    vehicles_directory: Path
    studies_directory: Path
    results_directory: Path
    profile_roots: tuple[Path, ...]


@dataclass
class ResolutionResult:
    paths: ProjectPaths
    data: dict[str, Any]
    provenance: ProvenanceMap
    diagnostics: tuple[Diagnostic, ...]
    loaded_files: tuple[Path, ...]
    active_study: str | None

    @property
    def error_count(self) -> int:
        return sum(item.severity is Severity.ERROR for item in self.diagnostics)

    @property
    def warning_count(self) -> int:
        return sum(item.severity is Severity.WARNING for item in self.diagnostics)

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0

    def export(self, output_directory: Path) -> None:
        export_resolution(
            output_directory,
            data=self.data,
            provenance=self.provenance,
            diagnostics=self.diagnostics,
            metadata={
                "schema_version": 1,
                "project_root": str(self.paths.root),
                "project_file": str(self.paths.project_file),
                "active_study": self.active_study,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "loaded_files": [str(path) for path in self.loaded_files],
            },
        )


class ProjectLoader:
    """Resolve one self-contained project into a validated configuration tree."""

    def __init__(self) -> None:
        self._builtin_profiles = Path(__file__).resolve().parents[1] / "builtin_profiles"

    def resolve(
        self,
        project: str | Path,
        *,
        study: str | None = None,
        cli_overrides: Sequence[tuple[str, Any]] = (),
    ) -> ResolutionResult:
        diagnostics = DiagnosticBag()
        project_file = discover_project_file(project)
        project_root = project_file.parent.resolve()
        raw_project = self._required_toml(project_file, "project", diagnostics)
        if raw_project is None:
            raise ProjectError(f"Unable to load project file: {project_file}")

        paths = self._resolve_paths(project_root, project_file, raw_project, diagnostics)
        loaded_files: list[Path] = [project_file]

        registry = ProfileRegistry(diagnostics)
        registry.add_root(self._builtin_profiles, origin="builtin", required=True)
        for root in paths.profile_roots:
            registry.add_root(root, origin="user", required=True)

        data: dict[str, Any] = {}
        provenance: ProvenanceMap = {}
        deep_merge(
            data,
            {"project": deepcopy(raw_project.get("project", {}))},
            provenance=provenance,
            diagnostics=diagnostics,
            layer="project",
            source=str(project_file),
        )
        deep_merge(
            data,
            {"profile_roots": [str(root) for root in paths.profile_roots]},
            provenance=provenance,
            diagnostics=diagnostics,
            layer="project",
            source=str(project_file),
        )

        track_raw = self._optional_toml(paths.track_file, "track", diagnostics)
        if track_raw is not None:
            loaded_files.append(paths.track_file)
            track_resolved, track_provenance = self._resolve_profiled_document(
                track_raw,
                scope="track",
                selector_path=("track", "profiles"),
                registry=registry,
                diagnostics=diagnostics,
                source=paths.track_file,
            )
            data.update(track_resolved)
            provenance.update(track_provenance)

        runs_raw = self._optional_toml(paths.runs_file, "runs", diagnostics)
        if runs_raw is not None:
            loaded_files.append(paths.runs_file)
            deep_merge(
                data,
                {"runs": deepcopy(runs_raw.get("runs", []))},
                provenance=provenance,
                diagnostics=diagnostics,
                layer="track_runs",
                source=str(paths.runs_file),
            )

        events_raw = self._optional_toml(paths.events_file, "events", diagnostics)
        if events_raw is not None:
            loaded_files.append(paths.events_file)
            resolved_events, event_provenance = self._resolve_events(
                events_raw.get("events", []),
                registry=registry,
                diagnostics=diagnostics,
                source=paths.events_file,
            )
            data["events"] = resolved_events
            provenance.update(event_provenance)

        vehicles, vehicle_files, vehicle_provenance = self._load_vehicles(
            paths, registry, diagnostics
        )
        loaded_files.extend(vehicle_files)
        data["vehicles"] = vehicles
        provenance.update(vehicle_provenance)

        studies, study_files, study_provenance = self._load_studies(paths, diagnostics)
        loaded_files.extend(study_files)
        data["studies"] = studies
        provenance.update(study_provenance)

        active_study = study
        if active_study is not None:
            selected = studies.get(active_study)
            if selected is None:
                diagnostics.error(
                    "STUDY_NOT_FOUND",
                    f"Study {active_study!r} does not exist.",
                    path=active_study,
                    hint="Available studies: " + (", ".join(sorted(studies)) or "none"),
                )
            else:
                data["active_study"] = active_study
                provenance.setdefault("active_study", []).append(
                    _provenance_step("study_selection", active_study, "set", active_study)
                )
                overrides = selected.get("config_overrides", {})
                if overrides:
                    if not isinstance(overrides, Mapping):
                        diagnostics.error(
                            "INVALID_STUDY_OVERRIDES",
                            "config_overrides must be a TOML table.",
                            path=f"studies.{active_study}.config_overrides",
                        )
                    else:
                        unknown = _unknown_override_paths(data, overrides)
                        for dotted_path in unknown:
                            diagnostics.error(
                                "UNKNOWN_STUDY_OVERRIDE_PATH",
                                "Study override does not match an existing configuration path.",
                                path=dotted_path,
                                source=active_study,
                                hint="Physical parameters belong in profiles/project files; study overrides may only replace them.",
                            )
                        if not unknown:
                            deep_merge(
                                data,
                                overrides,
                                provenance=provenance,
                                diagnostics=diagnostics,
                                layer="study_override",
                                source=active_study,
                            )

        for dotted_path, value in cli_overrides:
            set_existing_path(
                data,
                dotted_path,
                value,
                provenance=provenance,
                diagnostics=diagnostics,
                layer="command_line",
                source=f"--set {dotted_path}",
            )

        validate_project(data, paths, diagnostics)

        return ResolutionResult(
            paths=paths,
            data=data,
            provenance=provenance,
            diagnostics=diagnostics.items,
            loaded_files=tuple(dict.fromkeys(path.resolve() for path in loaded_files)),
            active_study=active_study,
        )

    def _resolve_paths(
        self,
        project_root: Path,
        project_file: Path,
        raw: Mapping[str, Any],
        diagnostics: DiagnosticBag,
    ) -> ProjectPaths:
        project_table = raw.get("project")
        if not isinstance(project_table, Mapping):
            raise ProjectError(f"{project_file} requires a [project] table.")

        def local_path(key: str, default: str) -> Path:
            raw_value = str(project_table.get(key, default))
            candidate = Path(raw_value)
            if candidate.is_absolute():
                diagnostics.error(
                    "ABSOLUTE_PROJECT_PATH",
                    "Project-owned paths must be relative to the project directory.",
                    path=f"project.{key}",
                    source=str(project_file),
                )
                candidate = Path(default)
            resolved = (project_root / candidate).resolve()
            if not _is_within(resolved, project_root):
                diagnostics.error(
                    "PROJECT_PATH_ESCAPES_ROOT",
                    "Project-owned path resolves outside the project directory.",
                    path=f"project.{key}",
                    source=str(project_file),
                )
                resolved = (project_root / default).resolve()
            return resolved

        profiles_table = raw.get("profiles", {})
        roots_raw = profiles_table.get("roots", []) if isinstance(profiles_table, Mapping) else []
        if not isinstance(roots_raw, list) or not all(isinstance(item, str) for item in roots_raw):
            diagnostics.error(
                "INVALID_PROFILE_ROOTS",
                "profiles.roots must be an array of paths.",
                path="profiles.roots",
                source=str(project_file),
            )
            roots_raw = []
        profile_roots = tuple(
            (project_root / Path(item)).resolve() if not Path(item).is_absolute() else Path(item).resolve()
            for item in roots_raw
        )

        return ProjectPaths(
            root=project_root,
            project_file=project_file,
            track_file=local_path("track", "track/track.toml"),
            runs_file=local_path("runs", "track/runs.toml"),
            events_file=local_path("events", "track/events.toml"),
            vehicles_directory=local_path("vehicles_directory", "vehicles"),
            studies_directory=local_path("studies_directory", "studies"),
            results_directory=local_path("results_directory", "results"),
            profile_roots=profile_roots,
        )

    def _resolve_profiled_document(
        self,
        raw: Mapping[str, Any],
        *,
        scope: str,
        selector_path: tuple[str, str],
        registry: ProfileRegistry,
        diagnostics: DiagnosticBag,
        source: Path,
    ) -> tuple[dict[str, Any], ProvenanceMap]:
        section = raw.get(selector_path[0], {})
        profile_ids: list[str] = []
        if isinstance(section, Mapping):
            selected = section.get(selector_path[1], [])
            if isinstance(selected, str):
                profile_ids = [selected]
            elif isinstance(selected, list) and all(isinstance(item, str) for item in selected):
                profile_ids = list(selected)
            elif selected not in (None, []):
                diagnostics.error(
                    "INVALID_PROFILE_SELECTION",
                    f"{'.'.join(selector_path)} must be a string or array of strings.",
                    source=str(source),
                )
        resolved: dict[str, Any] = {}
        provenance: ProvenanceMap = {}
        registry.resolve_into(
            resolved,
            profile_ids,
            expected_scope=scope,
            provenance=provenance,
            diagnostics=diagnostics,
        )
        deep_merge(
            resolved,
            raw,
            provenance=provenance,
            diagnostics=diagnostics,
            layer=f"project_{scope}",
            source=str(source),
        )
        return resolved, provenance

    def _resolve_events(
        self,
        raw_events: Any,
        *,
        registry: ProfileRegistry,
        diagnostics: DiagnosticBag,
        source: Path,
    ) -> tuple[list[dict[str, Any]], ProvenanceMap]:
        """Resolve per-event obstacle profiles before project validation.

        Geometry and evidence fields remain project-owned.  Obstacle profiles only
        supply the explicit ``obstacle_model`` branch, which the event may override
        atomically with a complete uncertainty-aware declaration.
        """

        if not isinstance(raw_events, list):
            diagnostics.error(
                "EVENTS_NOT_ARRAY",
                "events.toml must define events as an array.",
                path="events",
                source=str(source),
            )
            return [], {}
        resolved_events: list[dict[str, Any]] = []
        all_provenance: ProvenanceMap = {}
        for index, raw_event in enumerate(raw_events):
            if not isinstance(raw_event, Mapping):
                resolved_events.append(deepcopy(raw_event))
                continue
            selected = raw_event.get("obstacle_profiles", [])
            if isinstance(selected, str):
                profile_ids = [selected]
            elif isinstance(selected, list) and all(isinstance(item, str) for item in selected):
                profile_ids = list(selected)
            else:
                diagnostics.error(
                    "INVALID_OBSTACLE_PROFILE_SELECTION",
                    "event.obstacle_profiles must be a string or array of strings.",
                    path=f"events.{index}.obstacle_profiles",
                    source=str(source),
                )
                profile_ids = []
            resolved, local = registry.resolve(
                profile_ids, expected_scope="obstacle", diagnostics=diagnostics
            )
            deep_merge(
                resolved,
                raw_event,
                provenance=local,
                diagnostics=diagnostics,
                layer="track_event",
                source=str(source),
            )
            resolved_events.append(resolved)
            all_provenance.update(prefix_provenance(local, ("events", str(index))))
        return resolved_events, all_provenance

    def _load_vehicles(
        self,
        paths: ProjectPaths,
        registry: ProfileRegistry,
        diagnostics: DiagnosticBag,
    ) -> tuple[dict[str, Any], list[Path], ProvenanceMap]:
        vehicles: dict[str, Any] = {}
        loaded: list[Path] = []
        all_provenance: ProvenanceMap = {}
        if not paths.vehicles_directory.exists():
            diagnostics.error(
                "VEHICLES_DIRECTORY_MISSING",
                "Vehicles directory does not exist.",
                path=str(paths.vehicles_directory),
            )
            return vehicles, loaded, all_provenance

        for directory in sorted(path for path in paths.vehicles_directory.iterdir() if path.is_dir()):
            vehicle_file = directory / "vehicle.toml"
            drivetrain_file = directory / "drivetrain.toml"
            vehicle_raw = self._optional_toml(vehicle_file, "vehicle", diagnostics)
            drivetrain_raw = self._optional_toml(drivetrain_file, "drivetrain", diagnostics)
            if vehicle_raw is None or drivetrain_raw is None:
                continue
            loaded.extend((vehicle_file, drivetrain_file))
            vehicle_section = vehicle_raw.get("vehicle", {})
            vehicle_id = (
                str(vehicle_section.get("id", "")).strip()
                if isinstance(vehicle_section, Mapping)
                else ""
            )
            if not vehicle_id:
                diagnostics.error(
                    "VEHICLE_ID_MISSING",
                    "vehicle.toml requires vehicle.id.",
                    path=str(vehicle_file),
                )
                continue
            if vehicle_id != directory.name:
                diagnostics.error(
                    "VEHICLE_DIRECTORY_ID_MISMATCH",
                    f"vehicle.id {vehicle_id!r} does not match directory {directory.name!r}.",
                    path=str(vehicle_file),
                )
            if vehicle_id in vehicles:
                diagnostics.error(
                    "DUPLICATE_VEHICLE_ID",
                    f"Vehicle id {vehicle_id!r} is duplicated.",
                    path=str(vehicle_file),
                )
                continue

            combined, local_provenance = self._resolve_profiled_document(
                vehicle_raw,
                scope="vehicle",
                selector_path=("vehicle", "profiles"),
                registry=registry,
                diagnostics=diagnostics,
                source=vehicle_file,
            )
            deep_merge(
                combined,
                drivetrain_raw,
                provenance=local_provenance,
                diagnostics=diagnostics,
                layer="project_drivetrain",
                source=str(drivetrain_file),
            )
            vehicles[vehicle_id] = combined
            all_provenance.update(
                prefix_provenance(local_provenance, ("vehicles", vehicle_id))
            )
        if not vehicles:
            diagnostics.error(
                "NO_VEHICLES",
                "No complete vehicle definitions were found.",
                path=str(paths.vehicles_directory),
            )
        return vehicles, loaded, all_provenance

    def _load_studies(
        self, paths: ProjectPaths, diagnostics: DiagnosticBag
    ) -> tuple[dict[str, Any], list[Path], ProvenanceMap]:
        studies: dict[str, Any] = {}
        loaded: list[Path] = []
        provenance: ProvenanceMap = {}
        if not paths.studies_directory.exists():
            diagnostics.error(
                "STUDIES_DIRECTORY_MISSING",
                "Studies directory does not exist.",
                path=str(paths.studies_directory),
            )
            return studies, loaded, provenance
        for path in sorted(paths.studies_directory.glob("*.toml")):
            raw = self._optional_toml(path, "study", diagnostics)
            if raw is None:
                continue
            loaded.append(path)
            study_table = raw.get("study", {})
            name = str(study_table.get("name", "")).strip() if isinstance(study_table, Mapping) else ""
            if not name:
                diagnostics.error(
                    "STUDY_NAME_MISSING",
                    "Study file requires study.name.",
                    path=str(path),
                )
                continue
            if name in studies:
                diagnostics.error(
                    "DUPLICATE_STUDY_NAME",
                    f"Study name {name!r} is duplicated.",
                    path=str(path),
                )
                continue
            studies[name] = raw
            local: ProvenanceMap = {}
            scratch: dict[str, Any] = {}
            deep_merge(
                scratch,
                raw,
                provenance=local,
                diagnostics=diagnostics,
                layer="study_definition",
                source=str(path),
            )
            provenance.update(prefix_provenance(local, ("studies", name)))
        if not studies:
            diagnostics.warning(
                "NO_STUDIES",
                "No study definitions were found.",
                path=str(paths.studies_directory),
                hint="Add at least one study before running simulation commands.",
            )
        return studies, loaded, provenance

    def _required_toml(
        self, path: Path, label: str, diagnostics: DiagnosticBag
    ) -> dict[str, Any] | None:
        return self._optional_toml(path, label, diagnostics, required=True)

    def _optional_toml(
        self,
        path: Path,
        label: str,
        diagnostics: DiagnosticBag,
        *,
        required: bool = True,
    ) -> dict[str, Any] | None:
        if not path.exists():
            if required:
                diagnostics.error(
                    f"{label.upper()}_FILE_MISSING",
                    f"Required {label} TOML file does not exist.",
                    path=str(path),
                )
            return None
        try:
            return load_toml(path)
        except (OSError, TomlError) as exc:
            diagnostics.error(
                f"{label.upper()}_PARSE_ERROR",
                str(exc),
                path=str(path),
            )
            return None


def discover_project_file(project: str | Path) -> Path:
    path = Path(project).expanduser()
    if path.is_dir():
        path = path / "project.toml"
    if path.name != "project.toml":
        raise ProjectError(
            f"Expected a project directory or project.toml, received: {project}"
        )
    if not path.exists():
        raise ProjectError(f"Project file does not exist: {path}")
    return path.resolve()


def initialize_project(
    destination: str | Path,
    *,
    name: str | None = None,
    template_root: Path | None = None,
) -> Path:
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists():
        if not destination_path.is_dir():
            raise ProjectError(f"Destination exists and is not a directory: {destination_path}")
        if any(destination_path.iterdir()):
            raise ProjectError(
                f"Destination already exists and is not empty: {destination_path}"
            )
    destination_path.mkdir(parents=True, exist_ok=True)
    if template_root is None:
        from importlib.resources import files

        resource_root = files("cvt_track_study").joinpath("project_template")
        _copy_resource_tree(resource_root, destination_path)
    else:
        if not template_root.exists():
            raise ProjectError(f"Project template was not found: {template_root}")
        shutil.copytree(template_root, destination_path, dirs_exist_ok=True)
    # Empty directories are not guaranteed to survive wheel packaging.
    (destination_path / "track" / "gpx").mkdir(parents=True, exist_ok=True)
    (destination_path / "results").mkdir(parents=True, exist_ok=True)
    if name:
        project_file = destination_path / "project.toml"
        raw = load_toml(project_file)
        raw["project"]["name"] = name
        from .toml_io import dump_toml

        dump_toml(raw, project_file)
    return destination_path


def _copy_resource_tree(source: Any, destination: Path) -> None:
    """Copy an importlib Traversable tree without assuming a filesystem install."""
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            _copy_resource_tree(child, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(child.read_bytes())


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ProjectError("Overrides must use PATH=VALUE syntax.")
    path, raw_value = text.split("=", 1)
    path = path.strip()
    if not path:
        raise ProjectError("Override path cannot be empty.")
    # Parse with the same TOML scalar rules as project files. Bare non-TOML words
    # are accepted as strings for command-line convenience.
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib
    try:
        value = tomllib.loads("value = " + raw_value)["value"]
    except tomllib.TOMLDecodeError:
        value = raw_value
    return path, value


def _unknown_override_paths(
    base: Mapping[str, Any], overrides: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> list[str]:
    unknown: list[str] = []
    for key, value in overrides.items():
        path = (*prefix, str(key))
        if key not in base:
            unknown.append(".".join(path))
            continue
        existing = base[key]
        if isinstance(value, Mapping):
            if not isinstance(existing, Mapping):
                unknown.append(".".join(path))
            elif _is_complete_quantity_mapping(value):
                # A complete quantity atomically replaces the existing quantity.
                if not _is_quantity_mapping(existing):
                    unknown.append(".".join(path))
            else:
                unknown.extend(_unknown_override_paths(existing, value, path))
    return unknown


def _is_quantity_mapping(value: Mapping[str, Any]) -> bool:
    return all(key in value for key in ("nominal", "unit", "source", "uncertainty"))


def _is_complete_quantity_mapping(value: Mapping[str, Any]) -> bool:
    return _is_quantity_mapping(value)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _provenance_step(layer: str, source: str, action: str, value: Any):
    from .merge import ProvenanceStep

    return ProvenanceStep(layer=layer, source=source, action=action, value=value)
