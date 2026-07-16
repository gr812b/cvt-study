"""Versioned, self-contained track bundles.

Bundle reading and simulator-facing views are intentionally lightweight. Bundle
construction is imported lazily because it belongs to the GPX/reconstruction side
of the boundary and depends on pandas.
"""

from __future__ import annotations

from typing import Any

from .consumer import (
    SimulationFeature,
    SimulationSpeedGate,
    SimulationTrackInput,
    TrackInterval,
    simulation_track_from_bundle,
)
from .io import load_track_bundle, write_track_bundle
from .model import (
    CURRENT_TRACK_BUNDLE_SCHEMA,
    TRACK_BUNDLE_FORMAT,
    TrackBundle,
    TrackBundleError,
)
from .validation import validate_track_bundle

__all__ = [
    "CURRENT_TRACK_BUNDLE_SCHEMA",
    "TRACK_BUNDLE_FORMAT",
    "SimulationFeature",
    "SimulationSpeedGate",
    "SimulationTrackInput",
    "TrackBundle",
    "TrackBundleError",
    "TrackInterval",
    "build_track_bundle",
    "export_bundle_for_track_build",
    "load_track_bundle",
    "simulation_track_from_bundle",
    "validate_track_bundle",
    "write_track_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {"build_track_bundle", "export_bundle_for_track_build"}:
        from .builder import build_track_bundle, export_bundle_for_track_build

        return {
            "build_track_bundle": build_track_bundle,
            "export_bundle_for_track_build": export_bundle_for_track_build,
        }[name]
    raise AttributeError(name)
