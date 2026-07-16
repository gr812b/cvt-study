"""Geometry and observed-profile sections of the track bundle."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from cvt_track_study.track.model import TrackBuildResult

from .serialization import (
    circular,
    interval,
    json_safe,
    optional_float,
    split_tokens,
)


def centreline_contract(result: TrackBuildResult) -> dict[str, Any]:
    latitude, longitude = result.centreline.frame.to_latlon(
        result.centreline.x_m, result.centreline.y_m
    )
    samples = []
    for s, x, y, lat, lon, elevation in zip(
        result.centreline.s_m,
        result.centreline.x_m,
        result.centreline.y_m,
        latitude,
        longitude,
        result.centreline.elevation_m,
    ):
        samples.append(
            {
                "s_m": float(s),
                "x_m": float(x),
                "y_m": float(y),
                "latitude_deg": float(lat),
                "longitude_deg": float(lon),
                "reference_elevation_m": optional_float(elevation),
            }
        )
    return {
        "sample_count": len(samples),
        "geometry_source": "reference_lap_resampled_and_map_matched",
        "elevation_source": "reference_lap_gpx_when_available",
        "samples": samples,
    }


def observed_profile_contract(frame: pd.DataFrame) -> dict[str, Any]:
    columns = (
        "s_m",
        "median_speed_mps",
        "p10_speed_mps",
        "p90_speed_mps",
        "valid_speed_lap_count",
        "median_elevation_m",
        "p10_elevation_m",
        "p90_elevation_m",
        "valid_elevation_lap_count",
    )
    samples = [
        {column: json_safe(row[column]) for column in columns}
        for _, row in frame.iterrows()
    ]
    return {
        "sample_count": len(samples),
        "speed_statistic": "interpolated_per_lap_then_across_lap_quantiles",
        "elevation_statistic": "interpolated_per_lap_then_across_lap_quantiles",
        "samples": samples,
    }


def physical_feature_contracts(
    frame: pd.DataFrame, length: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.sort_values("sequence").iterrows():
        start = circular(float(row["anchor_s_m"]) + float(row["feature_start_rel_m"]), length)
        end = circular(float(row["anchor_s_m"]) + float(row["feature_end_rel_m"]), length)
        rows.append(
            {
                "id": str(row["id"]),
                "name": str(row["name"]),
                "sequence": int(row["sequence"]),
                "kind": str(row["kind"]),
                "analysis_role": str(row["analysis_role"]),
                "response_group_id": str(row["response_group_id"]),
                "gate_candidate": bool(row["gate_candidate"]),
                "interval": interval(start, end, length),
                "anchor": {
                    "s_m": circular(float(row["anchor_s_m"]), length),
                    "latitude_deg": float(row["anchor_latitude_deg"]),
                    "longitude_deg": float(row["anchor_longitude_deg"]),
                    "projection_error_m": float(row["anchor_projection_error_m"]),
                    "horizontal_uncertainty_m": float(row["anchor_horizontal_uncertainty_m"]),
                    "effective_error_m": math.hypot(
                        float(row["anchor_projection_error_m"]),
                        float(row["anchor_horizontal_uncertainty_m"]),
                    ),
                    "source": str(row["anchor_source"]),
                },
                "geometry_uncertainty": {
                    "start": {
                        "source_type": str(row["feature_start_source"]),
                        "provenance": str(row["feature_start_provenance"]),
                        "projection_error_m": float(row["feature_start_projection_error_m"]),
                        "horizontal_uncertainty_m": float(row["feature_start_horizontal_uncertainty_m"]),
                        "effective_error_m": float(row["feature_start_effective_error_m"]),
                    },
                    "end": {
                        "source_type": str(row["feature_end_source"]),
                        "provenance": str(row["feature_end_provenance"]),
                        "projection_error_m": float(row["feature_end_projection_error_m"]),
                        "horizontal_uncertainty_m": float(row["feature_end_horizontal_uncertainty_m"]),
                        "effective_error_m": float(row["feature_end_effective_error_m"]),
                    },
                },
                "review_flags": split_tokens(row.get("review_flags")),
                "notes": str(row.get("notes", "")),
                "obstacle_model": json_safe(row.get("obstacle_model", {})),
            }
        )
    return rows


def response_group_contracts(
    frame: pd.DataFrame, length: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.sort_values("sequence").iterrows():
        start = circular(float(row["anchor_s_m"]) + float(row["feature_start_rel_m"]), length)
        end = circular(float(row["anchor_s_m"]) + float(row["feature_end_rel_m"]), length)
        rows.append(
            {
                "id": str(row["id"]),
                "name": str(row["name"]),
                "sequence": int(row["sequence"]),
                "analysis_feature_type": str(row["analysis_feature_type"]),
                "analysis_role": str(row["analysis_role"]),
                "gate_candidate": bool(row["gate_candidate"]),
                "source_feature_ids": split_tokens(row["source_event_ids"]),
                "source_feature_names": split_tokens(row["source_event_names"]),
                "interval": interval(start, end, length),
                "geometry_uncertainty": {
                    "start_effective_error_m": float(row["feature_start_effective_error_m"]),
                    "end_effective_error_m": float(row["feature_end_effective_error_m"]),
                },
                "review_flags": split_tokens(row.get("review_flags")),
                "obstacle_model": {
                    "status": "not_applicable",
                    "model_type": None,
                    "parameters": {},
                    "reason": "Response groups aggregate evidence; physical feature models are simulated individually.",
                    "source_feature_ids": split_tokens(row["source_event_ids"]),
                },
            }
        )
    return rows
