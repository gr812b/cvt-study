"""Atomic Phase 3 track-evidence export."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from cvt_track_study.gpx.cleanup import create_telemetry_cleanup_map

from .model import TrackBuildResult
from .review import (
    build_event_interval_audit,
    create_elevation_profile,
    create_event_group_timeline,
    create_track_map,
    write_review_html,
    write_review_summary,
)


def export_track_build(
    output_directory: Path,
    result: TrackBuildResult,
) -> Path:
    final = output_directory.resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{final.name}-", dir=final.parent)
    )
    try:
        ingestion = temporary / "ingestion"
        ingestion.mkdir(parents=True, exist_ok=True)
        cleaned_points = pd.concat(
            [item.points for item in result.ingestion_results],
            ignore_index=True,
        )
        rejected_telemetry = _concat_nonempty(
            [item.rejected_points for item in result.ingestion_results]
        )
        _write_frame(
            cleaned_points,
            ingestion / "canonical_points.csv",
        )
        _write_frame(
            rejected_telemetry,
            ingestion / "rejected_telemetry_points.csv",
        )
        _write_frame(
            pd.concat(
                [item.segments for item in result.ingestion_results],
                ignore_index=True,
            ),
            ingestion / "segments.csv",
        )
        _write_json(
            ingestion / "run_summaries.json",
            [item.summary for item in result.ingestion_results],
        )

        track = temporary / "track"
        track.mkdir(parents=True, exist_ok=True)
        latitude, longitude = result.centreline.frame.to_latlon(
            result.centreline.x_m, result.centreline.y_m
        )
        centreline_frame = pd.DataFrame(
            {
                "s_m": result.centreline.s_m,
                "x_m": result.centreline.x_m,
                "y_m": result.centreline.y_m,
                "latitude_deg": latitude,
                "longitude_deg": longitude,
                "reference_elevation_m": result.centreline.elevation_m,
            }
        )
        _write_frame(centreline_frame, track / "centreline.csv")
        _write_frame(result.laps, track / "lap_quality.csv")
        _write_frame(
            result.matched_points,
            track / "map_matched_points.csv",
        )
        _write_frame(
            result.rejected_map_points,
            track / "rejected_map_points.csv",
        )
        _write_frame(result.track_profile, track / "track_profile.csv")
        _write_frame(
            result.event_projection,
            track / "event_projection.csv",
        )
        _write_frame(
            result.response_features,
            track / "response_features.csv",
        )
        interval_audit = build_event_interval_audit(
            result.response_features,
            result.centreline.length_m,
        )
        _write_frame(
            interval_audit,
            track / "event_interval_audit.csv",
        )
        _write_frame(result.event_passes, track / "event_passes.csv")
        _write_frame(result.gate_evidence, track / "gate_evidence.csv")
        _write_frame(result.gate_review, track / "gate_review.csv")

        review = temporary / "review"
        map_path = review / "track_map.png"
        timeline_path = review / "event_group_timeline.png"
        elevation_path = review / "elevation_profile.png"
        valid_lap_ids = set(
            result.laps.loc[result.laps["analysis_valid"], "lap_id"].astype(int)
        )
        review_matched_points = result.matched_points[
            result.matched_points["lap_id"].astype(int).isin(valid_lap_ids)
        ]
        create_track_map(
            map_path,
            result.centreline,
            review_matched_points,
            result.event_projection,
            result.gate_review,
        )
        create_event_group_timeline(
            timeline_path,
            interval_audit,
            result.centreline.length_m,
        )
        create_elevation_profile(
            elevation_path, result.track_profile
        )
        all_rejected = _concat_nonempty(
            [rejected_telemetry, result.rejected_map_points]
        )
        create_telemetry_cleanup_map(
            review / "telemetry_cleanup_map.png",
            cleaned_points,
            all_rejected,
        )
        write_review_summary(
            review / "REVIEW_SUMMARY.md",
            laps=result.laps,
            events=result.event_projection,
            gate_review=result.gate_review,
            track_length_m=result.centreline.length_m,
        )
        write_review_html(
            review / "track_review.html",
            map_path=map_path,
            timeline_path=timeline_path,
            elevation_path=elevation_path,
            interval_audit=interval_audit,
            gate_review=result.gate_review,
            laps=result.laps,
        )
        _write_json(
            temporary / "diagnostics.json",
            [item.to_dict() for item in result.diagnostics],
        )

        from cvt_track_study.bundle import export_bundle_for_track_build

        bundle = export_bundle_for_track_build(temporary, result)
        result.metadata["track_bundle_schema_version"] = (
            bundle.schema_version
        )
        result.metadata["track_bundle_sha256"] = bundle.sha256
        result.metadata[
            "track_bundle_content_fingerprint_sha256"
        ] = bundle.data["content_fingerprint_sha256"]
        _write_json(
            temporary / "track_build_manifest.json",
            result.metadata,
        )
        result.resolution.export(temporary / "configuration")
        _replace_directory(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def _concat_nonempty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate audit tables after enforcing unique column labels.

    Rejection tables come from different pipeline stages and therefore do not
    have identical schemas. Pandas supports their union, but it requires each
    individual input frame to have a uniquely valued column index.
    """

    usable = [
        _coalesce_duplicate_columns(frame)
        for frame in frames
        if frame is not None and not frame.empty
    ]
    return (
        pd.concat(usable, ignore_index=True, sort=False)
        if usable
        else pd.DataFrame()
    )


def _coalesce_duplicate_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy with duplicate labels combined left-to-right.

    Duplicate columns should not normally reach export after the cleanup schema
    fix. This remains as a defensive boundary because exported artifacts must
    never fail solely due to a repeated audit-column label.
    """

    if frame.columns.is_unique:
        return frame.copy()

    output: dict[str, pd.Series] = {}
    ordered_names: list[str] = []
    for position, raw_name in enumerate(frame.columns):
        name = str(raw_name)
        if name in output:
            continue
        positions = [
            index
            for index, candidate in enumerate(frame.columns)
            if str(candidate) == name
        ]
        values = frame.iloc[:, positions]
        combined = (
            values.iloc[:, 0].copy()
            if len(positions) == 1
            else values.bfill(axis=1).iloc[:, 0]
        )
        output[name] = combined
        ordered_names.append(name)

    return pd.DataFrame(
        {name: output[name] for name in ordered_names},
        index=frame.index,
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%fZ",
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _replace_directory(temporary: Path, final: Path) -> None:
    backup = final.with_name(f".{final.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if final.exists():
        os.replace(final, backup)
    try:
        os.replace(temporary, final)
    except Exception:
        if backup.exists() and not final.exists():
            os.replace(backup, final)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)
