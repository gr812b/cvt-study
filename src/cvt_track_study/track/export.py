"""Atomic Phase 3 track-evidence export."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .model import TrackBuildResult
from .review import (
    create_elevation_profile,
    create_track_map,
    write_review_html,
    write_review_summary,
)


def export_track_build(output_directory: Path, result: TrackBuildResult) -> Path:
    final = output_directory.resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}-", dir=final.parent))
    try:
        ingestion = temporary / "ingestion"
        ingestion.mkdir(parents=True, exist_ok=True)
        _write_frame(
            pd.concat([item.points for item in result.ingestion_results], ignore_index=True),
            ingestion / "canonical_points.csv",
        )
        _write_frame(
            pd.concat([item.segments for item in result.ingestion_results], ignore_index=True),
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
        _write_frame(result.matched_points, track / "map_matched_points.csv")
        _write_frame(result.track_profile, track / "track_profile.csv")
        _write_frame(result.event_projection, track / "event_projection.csv")
        _write_frame(result.response_features, track / "response_features.csv")
        _write_frame(result.event_passes, track / "event_passes.csv")
        _write_frame(result.gate_evidence, track / "gate_evidence.csv")
        _write_frame(result.gate_review, track / "gate_review.csv")

        review = temporary / "review"
        map_path = review / "track_map.png"
        elevation_path = review / "elevation_profile.png"
        create_track_map(
            map_path,
            result.centreline,
            result.matched_points,
            result.event_projection,
            result.gate_review,
        )
        create_elevation_profile(elevation_path, result.track_profile)
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
            elevation_path=elevation_path,
            gate_review=result.gate_review,
            laps=result.laps,
        )
        _write_json(
            temporary / "diagnostics.json",
            [item.to_dict() for item in result.diagnostics],
        )
        # The bundle is the stable Phase 4 boundary. Import locally to keep the
        # reconstruction result model independent of bundle implementation details.
        from cvt_track_study.bundle import export_bundle_for_track_build

        bundle = export_bundle_for_track_build(temporary, result)
        result.metadata["track_bundle_schema_version"] = bundle.schema_version
        result.metadata["track_bundle_sha256"] = bundle.sha256
        result.metadata["track_bundle_content_fingerprint_sha256"] = bundle.data[
            "content_fingerprint_sha256"
        ]
        _write_json(temporary / "track_build_manifest.json", result.metadata)
        result.resolution.export(temporary / "configuration")
        _replace_directory(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
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
