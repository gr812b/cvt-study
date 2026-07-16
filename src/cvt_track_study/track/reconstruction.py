"""Phase 3 track-evidence orchestration.

The mechanisms live in focused modules so the future simulation-bundle boundary can
consume stable tables without depending on one monolithic reconstruction routine.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from cvt_track_study.config.diagnostics import DiagnosticBag
from cvt_track_study.gpx.model import GPXIngestionResult

from .events import _find_lap_gate, build_response_features, normalize_events, project_events
from .gates import build_gate_review, score_speed_gates
from .geo import Centreline, LocalFrame
from .laps import _clean_speed, build_centreline, build_track_profile, detect_laps, map_match_laps
from .metrics import extract_event_passes
from .settings import ReconstructionSettings


def build_track_evidence(
    ingestion_results: tuple[GPXIngestionResult, ...],
    track_config: Mapping[str, Any],
    raw_events: list[Mapping[str, Any]],
    diagnostics: DiagnosticBag,
) -> tuple[
    Centreline,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build the complete Phase 3 evidence set from canonical telemetry runs."""

    settings = ReconstructionSettings.from_mapping(track_config)
    events = normalize_events(raw_events)
    lap_gate = _find_lap_gate(events, settings.lap_gate_event_id)
    all_points = _combine_points(ingestion_results)
    frame = LocalFrame(
        float(all_points["latitude_deg"].median()),
        float(all_points["longitude_deg"].median()),
    )
    x_m, y_m = frame.to_xy(all_points["latitude_deg"], all_points["longitude_deg"])
    all_points["x_m"] = x_m
    all_points["y_m"] = y_m
    all_points["speed_analysis_mps"] = _clean_speed(all_points, settings)

    laps = detect_laps(
        all_points,
        ingestion_results,
        frame,
        float(lap_gate["anchor_latitude_deg"]),
        float(lap_gate["anchor_longitude_deg"]),
        settings,
        diagnostics,
    )
    centreline = build_centreline(all_points, laps, frame, settings)
    matched_points, laps = map_match_laps(all_points, laps, centreline, settings)
    if not laps["analysis_valid"].any():
        raise ValueError("No valid map-matched laps remain after quality checks.")
    track_profile = build_track_profile(matched_points, laps, centreline, settings)
    event_projection = project_events(events, centreline, settings)
    response_features = build_response_features(
        event_projection, centreline.length_m, settings
    )
    event_passes = extract_event_passes(
        matched_points, laps, response_features, centreline, settings
    )
    gate_evidence = score_speed_gates(
        event_passes, response_features, settings, diagnostics
    )
    gate_review = build_gate_review(gate_evidence, response_features, settings)
    return (
        centreline,
        laps,
        matched_points,
        track_profile,
        event_projection,
        response_features,
        event_passes,
        gate_evidence,
        gate_review,
    )


def _combine_points(results: tuple[GPXIngestionResult, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        frame = result.points.copy()
        frame["use_for_centreline"] = result.metadata.use_for_centreline
        frame["use_for_gate_evidence"] = result.metadata.use_for_gate_evidence
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
