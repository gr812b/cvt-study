"""Phase 3 track-evidence orchestration.

The mechanisms live in focused modules so the future simulation-bundle
boundary can consume stable tables without depending on one monolithic
reconstruction routine.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from cvt_track_study.config.diagnostics import DiagnosticBag
from cvt_track_study.gpx.model import GPXIngestionResult

from .consensus import build_iterative_consensus
from .events import (
    _find_lap_gate,
    build_response_features,
    normalize_events,
    project_events,
)
from .gates import build_gate_review, score_speed_gates
from .geo import Centreline, LocalFrame
from .laps import _clean_speed, build_track_profile, detect_laps
from .metrics import extract_event_passes
from .route_variants import (
    RouteVariantDetection,
    RouteVariantSettings,
    detect_and_select_route_variants,
    detect_laps_route_aware,
    finalize_route_variant_semantics,
)
from .settings import ReconstructionSettings


@dataclass(frozen=True)
class TrackEvidenceBuild:
    """Backward-compatible result with explicit route-variant audit tables.

    Iteration yields the historical ten values, so existing robustness code that
    unpacks ``build_track_evidence`` remains source compatible.
    """

    centreline: Centreline
    laps: pd.DataFrame
    matched_points: pd.DataFrame
    track_profile: pd.DataFrame
    event_projection: pd.DataFrame
    response_features: pd.DataFrame
    event_passes: pd.DataFrame
    gate_evidence: pd.DataFrame
    gate_review: pd.DataFrame
    rejected_map_points: pd.DataFrame
    route_variant_summary: pd.DataFrame
    route_variant_pairwise: pd.DataFrame
    route_variant_settings: RouteVariantSettings
    selected_route_variant_id: str

    def __iter__(self) -> Iterator[Any]:
        yield self.centreline
        yield self.laps
        yield self.matched_points
        yield self.track_profile
        yield self.event_projection
        yield self.response_features
        yield self.event_passes
        yield self.gate_evidence
        yield self.gate_review
        yield self.rejected_map_points


def build_track_evidence(
    ingestion_results: tuple[GPXIngestionResult, ...],
    track_config: Mapping[str, Any],
    raw_events: list[Mapping[str, Any]],
    diagnostics: DiagnosticBag,
) -> TrackEvidenceBuild:
    """Build the complete Phase 3 evidence set."""

    settings = ReconstructionSettings.from_mapping(track_config)
    variant_settings = RouteVariantSettings.from_mapping(track_config)
    events = normalize_events(raw_events)
    lap_gate = _find_lap_gate(events, settings.lap_gate_event_id)
    all_points = _combine_points(ingestion_results)
    frame = LocalFrame(
        float(all_points["latitude_deg"].median()),
        float(all_points["longitude_deg"].median()),
    )
    x_m, y_m = frame.to_xy(
        all_points["latitude_deg"],
        all_points["longitude_deg"],
    )
    all_points["x_m"] = x_m
    all_points["y_m"] = y_m
    all_points["speed_analysis_mps"] = _clean_speed(all_points, settings)

    detector = detect_laps_route_aware if variant_settings.enabled else detect_laps
    laps = detector(
        all_points,
        ingestion_results,
        frame,
        float(lap_gate["anchor_latitude_deg"]),
        float(lap_gate["anchor_longitude_deg"]),
        settings,
        diagnostics,
    )
    variants: RouteVariantDetection = detect_and_select_route_variants(
        all_points,
        laps,
        track_config,
        diagnostics,
    )
    laps = variants.laps

    consensus = build_iterative_consensus(
        all_points,
        laps,
        frame,
        settings,
        track_config,
        diagnostics,
    )
    centreline = consensus.centreline
    laps = finalize_route_variant_semantics(consensus.laps)
    matched_points = consensus.matched_points
    rejected_map_points = consensus.rejected_map_points

    if not laps["analysis_valid"].any():
        raise ValueError(
            "No valid map-matched laps remain after consensus quality checks."
        )

    track_profile = build_track_profile(
        matched_points,
        laps,
        centreline,
        settings,
    )
    event_projection = project_events(events, centreline, settings)
    response_features = build_response_features(
        event_projection,
        centreline.length_m,
        settings,
    )
    event_passes = extract_event_passes(
        matched_points,
        laps,
        response_features,
        centreline,
        settings,
    )
    gate_evidence = score_speed_gates(
        event_passes,
        response_features,
        settings,
        diagnostics,
    )
    gate_review = build_gate_review(
        gate_evidence,
        response_features,
        settings,
    )
    return TrackEvidenceBuild(
        centreline=centreline,
        laps=laps,
        matched_points=matched_points,
        track_profile=track_profile,
        event_projection=event_projection,
        response_features=response_features,
        event_passes=event_passes,
        gate_evidence=gate_evidence,
        gate_review=gate_review,
        rejected_map_points=rejected_map_points,
        route_variant_summary=variants.summary,
        route_variant_pairwise=variants.pairwise,
        route_variant_settings=variants.settings,
        selected_route_variant_id=variants.selected_variant_id,
    )


def _combine_points(
    results: tuple[GPXIngestionResult, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        frame = result.points.copy()
        frame["use_for_centreline"] = result.metadata.use_for_centreline
        frame["use_for_gate_evidence"] = result.metadata.use_for_gate_evidence
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
