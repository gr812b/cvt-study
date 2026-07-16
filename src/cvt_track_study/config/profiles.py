"""Profile discovery and inheritance resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .diagnostics import DiagnosticBag
from .merge import ProvenanceMap, deep_merge
from .toml_io import TomlError, load_toml


@dataclass(frozen=True)
class Profile:
    identifier: str
    scope: str
    version: int
    description: str
    extends: tuple[str, ...]
    config: Mapping[str, Any]
    path: Path
    origin: str


class ProfileRegistry:
    def __init__(self, diagnostics: DiagnosticBag) -> None:
        self._profiles: dict[str, Profile] = {}
        self._diagnostics = diagnostics

    @property
    def identifiers(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def add_root(self, root: Path, *, origin: str, required: bool = True) -> None:
        if not root.exists():
            if required:
                self._diagnostics.error(
                    "PROFILE_ROOT_MISSING",
                    "Configured profile root does not exist.",
                    path=str(root),
                    hint="Create the directory or remove it from project.toml.",
                )
            return
        if not root.is_dir():
            self._diagnostics.error(
                "PROFILE_ROOT_NOT_DIRECTORY",
                "Configured profile root is not a directory.",
                path=str(root),
            )
            return
        for path in sorted(root.rglob("*.toml")):
            self._load_profile(path, origin=origin)

    def resolve_into(
        self,
        target: dict[str, Any],
        profile_ids: list[str] | tuple[str, ...],
        *,
        expected_scope: str,
        provenance: ProvenanceMap,
        diagnostics: DiagnosticBag,
    ) -> None:
        applied: set[str] = set()
        active: list[str] = []

        def apply(identifier: str) -> None:
            if identifier in applied:
                return
            profile = self._profiles.get(identifier)
            if profile is None:
                diagnostics.error(
                    "PROFILE_NOT_FOUND",
                    f"Profile {identifier!r} was not found.",
                    path=identifier,
                    hint=(
                        "Check the profile id and configured profile roots. Available ids: "
                        + (", ".join(self.identifiers) or "none")
                    ),
                )
                return
            if identifier in active:
                cycle = " -> ".join((*active, identifier))
                diagnostics.error(
                    "PROFILE_INHERITANCE_CYCLE",
                    f"Profile inheritance contains a cycle: {cycle}.",
                    path=str(profile.path),
                )
                return
            if profile.scope not in {expected_scope, "mixed"}:
                diagnostics.error(
                    "PROFILE_SCOPE_MISMATCH",
                    f"Profile scope {profile.scope!r} cannot be applied to {expected_scope!r}.",
                    path=identifier,
                    source=str(profile.path),
                )
                return

            active.append(identifier)
            for parent in profile.extends:
                apply(parent)
            active.pop()
            if identifier in applied:
                return
            deep_merge(
                target,
                profile.config,
                provenance=provenance,
                diagnostics=diagnostics,
                layer=f"{profile.origin}_profile",
                source=f"{identifier}@{profile.version} ({profile.path})",
            )
            applied.add(identifier)

        for identifier in profile_ids:
            apply(identifier)


    def resolve(
        self,
        profile_ids: list[str] | tuple[str, ...],
        *,
        expected_scope: str,
        diagnostics: DiagnosticBag,
    ) -> tuple[dict[str, Any], ProvenanceMap]:
        """Resolve profiles into a standalone mapping and provenance tree."""

        target: dict[str, Any] = {}
        provenance: ProvenanceMap = {}
        self.resolve_into(
            target,
            profile_ids,
            expected_scope=expected_scope,
            provenance=provenance,
            diagnostics=diagnostics,
        )
        return target, provenance

    def _load_profile(self, path: Path, *, origin: str) -> None:
        try:
            raw = load_toml(path)
        except (OSError, TomlError) as exc:
            self._diagnostics.error(
                "PROFILE_PARSE_ERROR",
                str(exc),
                path=str(path),
            )
            return
        metadata = raw.get("profile")
        config = raw.get("config")
        if not isinstance(metadata, Mapping) or not isinstance(config, Mapping):
            self._diagnostics.error(
                "INVALID_PROFILE_FILE",
                "A profile requires [profile] metadata and a [config] table.",
                path=str(path),
            )
            return
        identifier = str(metadata.get("id", "")).strip()
        scope = str(metadata.get("scope", "")).strip()
        description = str(metadata.get("description", "")).strip()
        version_raw = metadata.get("version", 0)
        version = (
            version_raw
            if isinstance(version_raw, int) and not isinstance(version_raw, bool)
            else 0
        )
        raw_extends = metadata.get("extends", [])
        if isinstance(raw_extends, str):
            extends = (raw_extends,)
        elif isinstance(raw_extends, list) and all(
            isinstance(item, str) for item in raw_extends
        ):
            extends = tuple(raw_extends)
        else:
            self._diagnostics.error(
                "INVALID_PROFILE_EXTENDS",
                "profile.extends must be a string or an array of strings.",
                path=str(path),
            )
            return

        if not identifier or not scope or version < 1:
            self._diagnostics.error(
                "INVALID_PROFILE_METADATA",
                "Profile id and scope must be non-empty and version must be at least 1.",
                path=str(path),
            )
            return
        if identifier in self._profiles:
            previous = self._profiles[identifier]
            self._diagnostics.error(
                "DUPLICATE_PROFILE_ID",
                f"Profile id {identifier!r} is defined more than once.",
                path=str(path),
                hint=f"The first definition was {previous.path}.",
            )
            return
        self._profiles[identifier] = Profile(
            identifier=identifier,
            scope=scope,
            version=version,
            description=description,
            extends=extends,
            config=dict(config),
            path=path,
            origin=origin,
        )
