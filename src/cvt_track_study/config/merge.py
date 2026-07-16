"""Deterministic deep merging with per-leaf provenance."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from .diagnostics import DiagnosticBag


@dataclass(frozen=True)
class ProvenanceStep:
    layer: str
    source: str
    action: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProvenanceMap = dict[str, list[ProvenanceStep]]


def path_text(path: Sequence[str]) -> str:
    return ".".join(path)


def deep_merge(
    target: MutableMapping[str, Any],
    incoming: Mapping[str, Any],
    *,
    provenance: ProvenanceMap,
    diagnostics: DiagnosticBag,
    layer: str,
    source: str,
    prefix: Sequence[str] = (),
) -> None:
    """Merge ``incoming`` into ``target`` and record every leaf assignment.

    Tables merge recursively. Scalars and arrays replace earlier values. Quantity
    tables receive an explicit warning when a nominal is changed while inherited
    provenance or uncertainty is left untouched.
    """

    for key, value in incoming.items():
        current_path = (*prefix, str(key))
        existing = target.get(key)

        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            # A complete uncertainty-aware quantity is atomic. Replacing it as one
            # object prevents stale fields from an inherited distribution (for
            # example triangular bounds) surviving a switch to a normal model.
            if _is_complete_quantity(value) or _is_complete_choice(value):
                # Keep history for leaves that still exist, but remove provenance
                # entries for fields discarded by the atomic replacement.
                retained_paths = {
                    path_text(leaf_path)
                    for leaf_path in _leaf_paths(value, current_path)
                }
                subtree_prefix = path_text(current_path) + "."
                for recorded_path in list(provenance):
                    if (
                        recorded_path.startswith(subtree_prefix)
                        and recorded_path not in retained_paths
                    ):
                        del provenance[recorded_path]
                target[key] = deepcopy(value)
                _record_leaves(
                    value,
                    path=current_path,
                    provenance=provenance,
                    step_factory=lambda leaf: ProvenanceStep(
                        layer, source, "override", leaf
                    ),
                )
                continue
            if _looks_like_quantity(existing) and "nominal" in value:
                missing_companions = [
                    name for name in ("source", "uncertainty") if name not in value
                ]
                if missing_companions:
                    diagnostics.warning(
                        "PARTIAL_QUANTITY_OVERRIDE",
                        "The nominal value changed while inherited "
                        + " and ".join(missing_companions)
                        + " were retained.",
                        path=path_text(current_path),
                        source=source,
                        hint=(
                            "Confirm that the inherited uncertainty and source still apply, "
                            "or override the complete quantity table."
                        ),
                    )
            deep_merge(
                existing,  # type: ignore[arg-type]
                value,
                provenance=provenance,
                diagnostics=diagnostics,
                layer=layer,
                source=source,
                prefix=current_path,
            )
            continue

        action = "override" if key in target else "set"
        target[key] = deepcopy(value)
        _record_leaves(
            value,
            path=current_path,
            provenance=provenance,
            step_factory=lambda leaf: ProvenanceStep(layer, source, action, leaf),
        )


def set_existing_path(
    target: MutableMapping[str, Any],
    dotted_path: str,
    value: Any,
    *,
    provenance: ProvenanceMap,
    diagnostics: DiagnosticBag,
    layer: str,
    source: str,
) -> bool:
    parts = tuple(part for part in dotted_path.split(".") if part)
    if not parts:
        diagnostics.error("INVALID_OVERRIDE_PATH", "Override path is empty.", source=source)
        return False

    node: MutableMapping[str, Any] = target
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, MutableMapping):
            diagnostics.error(
                "UNKNOWN_OVERRIDE_PATH",
                "Override path does not identify an existing configuration value.",
                path=dotted_path,
                source=source,
                hint="Use a path shown in resolved_inputs.toml.",
            )
            return False
        node = child

    leaf = parts[-1]
    if leaf not in node:
        diagnostics.error(
            "UNKNOWN_OVERRIDE_PATH",
            "Override path does not identify an existing configuration value.",
            path=dotted_path,
            source=source,
            hint="Use a path shown in resolved_inputs.toml.",
        )
        return False

    if leaf == "nominal" and _parent_is_quantity(node):
        diagnostics.warning(
            "CLI_NOMINAL_OVERRIDE_REUSES_UNCERTAINTY",
            "The command-line override changes only the nominal value; the declared "
            "source and uncertainty remain unchanged.",
            path=dotted_path,
            source=source,
            hint="Use a project/profile edit when the uncertainty or provenance also changed.",
        )

    node[leaf] = deepcopy(value)
    _record_leaves(
        value,
        path=parts,
        provenance=provenance,
        step_factory=lambda leaf_value: ProvenanceStep(layer, source, "override", leaf_value),
    )
    return True


def prefix_provenance(provenance: ProvenanceMap, prefix: Sequence[str]) -> ProvenanceMap:
    result: ProvenanceMap = {}
    prefix_text = path_text(prefix)
    for path, steps in provenance.items():
        full_path = f"{prefix_text}.{path}" if prefix_text and path else prefix_text or path
        result[full_path] = list(steps)
    return result


def _record_leaves(
    value: Any,
    *,
    path: Sequence[str],
    provenance: ProvenanceMap,
    step_factory: Any,
) -> None:
    if isinstance(value, Mapping):
        if not value:
            provenance.setdefault(path_text(path), []).append(step_factory({}))
            return
        for key, child in value.items():
            _record_leaves(
                child,
                path=(*path, str(key)),
                provenance=provenance,
                step_factory=step_factory,
            )
        return
    provenance.setdefault(path_text(path), []).append(step_factory(deepcopy(value)))


def _looks_like_quantity(value: Mapping[str, Any]) -> bool:
    return "nominal" in value and any(
        key in value for key in ("unit", "source", "uncertainty")
    )


def _parent_is_quantity(value: Mapping[str, Any]) -> bool:
    return all(key in value for key in ("nominal", "unit", "source", "uncertainty"))


def _is_complete_quantity(value: Mapping[str, Any]) -> bool:
    return all(key in value for key in ("nominal", "unit", "source", "uncertainty"))


def _is_complete_choice(value: Mapping[str, Any]) -> bool:
    return (
        all(key in value for key in ("nominal", "source", "uncertainty"))
        and "unit" not in value
        and isinstance(value.get("nominal"), str)
    )


def _leaf_paths(value: Any, path: Sequence[str]) -> list[tuple[str, ...]]:
    if isinstance(value, Mapping):
        if not value:
            return [tuple(path)]
        result: list[tuple[str, ...]] = []
        for key, child in value.items():
            result.extend(_leaf_paths(child, (*path, str(key))))
        return result
    return [tuple(path)]
