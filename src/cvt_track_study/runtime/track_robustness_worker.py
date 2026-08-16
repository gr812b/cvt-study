"""Spawn-safe worker entry point for track-robustness reconstruction cases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_CONTEXT: dict[str, Any] | None = None


def initialize_track_robustness_worker(
    resolution: Any,
    parsed_runs: tuple[Any, ...],
    track_config: Mapping[str, Any],
    raw_events: tuple[Mapping[str, Any], ...],
) -> None:
    global _CONTEXT
    _CONTEXT = {
        "resolution": resolution,
        "parsed_runs": parsed_runs,
        "track_config": dict(track_config),
        "raw_events": raw_events,
    }


def execute_track_robustness_case(case: Any) -> Any:
    context = _CONTEXT
    if context is None:
        raise RuntimeError("Track-robustness worker context was not initialized.")
    from cvt_track_study.track.robustness import _execute_case

    return _execute_case(
        case=case,
        resolution=context["resolution"],
        parsed_runs=context["parsed_runs"],
        track_config=context["track_config"],
        raw_events=context["raw_events"],
    )
