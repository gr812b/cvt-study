"""Track reconstruction, evidence review, and track defensibility.

The public facade is intentionally lazy.  Track reconstruction is imported from
simulation/report code in several directions; eager package imports made spawned
multiprocessing workers sensitive to import order and could expose a circular
``track -> reports -> simulation -> track`` dependency before the CLI finished
bootstrapping.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "Centreline": (".geo", "Centreline"),
    "LocalFrame": (".geo", "LocalFrame"),
    "RobustnessCase": (".robustness", "RobustnessCase"),
    "TrackBuildResult": (".model", "TrackBuildResult"),
    "build_project_track": (".router_v10", "build_project_track"),
    "build_robustness_cases": (".robustness", "build_robustness_cases"),
    "run_track_robustness_project": (".robustness", "run_track_robustness_project"),
    "run_route_family_track_robustness_project": (
        ".robustness_family",
        "run_route_family_track_robustness_project",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
