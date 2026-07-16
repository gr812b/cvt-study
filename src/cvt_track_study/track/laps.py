"""Lap detection, centreline reconstruction, map matching, and spatial profiles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cvt_track_study.config.diagnostics import DiagnosticBag
from cvt_track_study.gpx.model import GPXIngestionResult

from .geo import Centreline, LocalFrame
from .settings import ReconstructionSettings


def _clean_speed(points: pd.DataFrame, settings: ReconstructionSettings) -> pd.Series:
    raw = pd.to_numeric(points["analysis_speed_mps"], errors="coerce")
    raw = raw.where(raw.between(0.0, settings.maximum_reasonable_speed_mps))
    cleaned = raw.copy()
    for _, indices in points.groupby(
        ["run_id", "track_index", "segment_index"], sort=False
    ).groups.items():
        idx = list(indices)
        local = raw.loc[idx]
        median = local.rolling(5, center=True, min_periods=3).median()
        spike = (local - median).abs() > settings.speed_spike_threshold_mps
        cleaned.loc[idx] = local.where(~spike, median)
    return cleaned

def detect_laps(
    points: pd.DataFrame,
    ingestion_results: tuple[GPXIngestionResult, ...],
    frame: LocalFrame,
    gate_latitude_deg: float,
    gate_longitude_deg: float,
    settings: ReconstructionSettings,
    diagnostics: DiagnosticBag,
) -> pd.DataFrame:
    run_metadata = {result.metadata.run_id: result.metadata for result in ingestion_results}
    gate_x, gate_y = frame.to_xy([gate_latitude_deg], [gate_longitude_deg])
    rows: list[dict[str, Any]] = []
    global_lap_id = 0
    for (run_id, track_index, segment_index), indices in points.groupby(
        ["run_id", "track_index", "segment_index"], sort=False
    ).groups.items():
        idx = list(indices)
        segment = points.loc[idx].copy()
        distance = np.hypot(segment["x_m"] - gate_x[0], segment["y_m"] - gate_y[0])
        crossings = _gate_crossings(segment, distance.to_numpy(float), settings)
        if len(crossings) < 2:
            diagnostics.warning(
                "NO_COMPLETE_LAPS_IN_SEGMENT",
                f"Run {run_id}, segment {track_index}:{segment_index} produced fewer than two lap-gate visits.",
                path=f"runs.{run_id}",
            )
            continue
        for local_lap, (start_pos, end_pos) in enumerate(
            zip(crossings[:-1], crossings[1:]), start=1
        ):
            lap_segment = segment.iloc[start_pos : end_pos + 1]
            times = pd.to_datetime(lap_segment["timestamp_utc"], utc=True, errors="coerce")
            duration = (
                float((times.iloc[-1] - times.iloc[0]).total_seconds())
                if times.notna().all()
                else math.nan
            )
            global_lap_id += 1
            dt = lap_segment["time_step_s"].to_numpy(float)
            distance_m = float(lap_segment["step_distance_m"].fillna(0.0).sum())
            speed_values = pd.to_numeric(
                lap_segment["speed_analysis_mps"], errors="coerce"
            )
            speed_coverage_fraction = float(speed_values.notna().mean())
            stationary_fraction = float(
                (speed_values.dropna() < settings.stationary_speed_mps).mean()
            ) if speed_values.notna().any() else 1.0
            metadata = run_metadata[str(run_id)]
            rows.append(
                {
                    "lap_id": global_lap_id,
                    "run_id": str(run_id),
                    "vehicle_id": metadata.vehicle_id,
                    "driver_id": metadata.driver_id,
                    "track_index": int(track_index),
                    "segment_index": int(segment_index),
                    "local_lap_id": local_lap,
                    "start_global_index": int(lap_segment.index[0]),
                    "end_global_index": int(lap_segment.index[-1]),
                    "start_time_utc": times.iloc[0],
                    "end_time_utc": times.iloc[-1],
                    "duration_s": duration,
                    "path_distance_m": distance_m,
                    "stationary_fraction": stationary_fraction,
                    "speed_coverage_fraction": speed_coverage_fraction,
                    "time_gap_count": int(
                        np.sum(np.isfinite(dt) & (dt > settings.maximum_normal_time_step_s))
                    ),
                    "timestamp_regression_count": int(
                        np.sum(np.isfinite(dt) & (dt < 0))
                    ),
                    "median_speed_mps": float(lap_segment["speed_analysis_mps"].median()),
                    "maximum_speed_mps": float(lap_segment["speed_analysis_mps"].max()),
                    "use_for_centreline": metadata.use_for_centreline,
                    "use_for_gate_evidence": metadata.use_for_gate_evidence,
                }
            )
    laps = pd.DataFrame(rows)
    if laps.empty:
        raise ValueError("No complete laps were found between lap-gate visits.")
    median_distance = float(laps["path_distance_m"].median())
    laps["distance_ratio_to_median"] = laps["path_distance_m"] / median_distance
    laps["analysis_valid"] = (
        laps["duration_s"].notna()
        & laps["distance_ratio_to_median"].between(0.85, 1.15)
        & (laps["stationary_fraction"] <= 0.15)
        & (
            laps["speed_coverage_fraction"]
            >= settings.minimum_speed_coverage_fraction
        )
        & (laps["time_gap_count"] == 0)
        & (laps["timestamp_regression_count"] == 0)
    )
    laps["quality_flags"] = laps.apply(
        lambda row: _initial_lap_quality_flags(row, settings), axis=1
    )
    eligible_reference = laps[laps["analysis_valid"] & laps["use_for_centreline"]]
    if eligible_reference.empty:
        raise ValueError("No valid lap is enabled for centreline construction.")
    reference_index = eligible_reference["duration_s"].idxmin()
    laps["reference_lap"] = False
    laps.loc[reference_index, "reference_lap"] = True
    return laps


def _initial_lap_quality_flags(
    lap: pd.Series, settings: ReconstructionSettings
) -> str:
    flags: list[str] = []
    if not np.isfinite(lap["duration_s"]):
        flags.append("missing_or_invalid_duration")
    if not 0.85 <= float(lap["distance_ratio_to_median"]) <= 1.15:
        flags.append("path_distance_outlier")
    if float(lap["stationary_fraction"]) > 0.15:
        flags.append("excessive_stationary_fraction")
    if (
        float(lap["speed_coverage_fraction"])
        < settings.minimum_speed_coverage_fraction
    ):
        flags.append("insufficient_speed_coverage")
    if int(lap["time_gap_count"]) > 0:
        flags.append("sampling_gap")
    if int(lap["timestamp_regression_count"]) > 0:
        flags.append("timestamp_regression")
    return ";".join(flags)

def _gate_crossings(
    segment: pd.DataFrame, distance_m: np.ndarray, settings: ReconstructionSettings
) -> list[int]:
    inside = np.flatnonzero(distance_m <= settings.lap_gate_radius_m)
    groups: list[list[int]] = []
    for position in inside:
        if not groups:
            groups.append([int(position)])
            continue
        current_time = pd.to_datetime(
            segment.iloc[position]["timestamp_utc"], utc=True, errors="coerce"
        )
        prior_time = pd.to_datetime(
            segment.iloc[groups[-1][-1]]["timestamp_utc"], utc=True, errors="coerce"
        )
        gap = (current_time - prior_time).total_seconds() if pd.notna(current_time) and pd.notna(prior_time) else 0.0
        if gap > 3.0:
            groups.append([int(position)])
        else:
            groups[-1].append(int(position))
    candidates = [min(group, key=lambda item: distance_m[item]) for group in groups]
    merged: list[int] = []
    for position in candidates:
        if not merged:
            merged.append(position)
            continue
        current_time = pd.to_datetime(
            segment.iloc[position]["timestamp_utc"], utc=True, errors="coerce"
        )
        prior_time = pd.to_datetime(
            segment.iloc[merged[-1]]["timestamp_utc"], utc=True, errors="coerce"
        )
        elapsed = (current_time - prior_time).total_seconds() if pd.notna(current_time) and pd.notna(prior_time) else math.inf
        if elapsed >= settings.minimum_lap_time_s:
            merged.append(position)
        elif distance_m[position] < distance_m[merged[-1]]:
            merged[-1] = position
    return merged

def build_centreline(
    points: pd.DataFrame,
    laps: pd.DataFrame,
    frame: LocalFrame,
    settings: ReconstructionSettings,
) -> Centreline:
    reference = laps.loc[laps["reference_lap"]].iloc[0]
    segment = points.loc[
        int(reference["start_global_index"]) : int(reference["end_global_index"])
    ].copy()
    x = (
        segment["x_m"]
        .rolling(5, center=True, min_periods=1)
        .median()
        .rolling(3, center=True, min_periods=1)
        .mean()
        .to_numpy(copy=True)
    )
    y = (
        segment["y_m"]
        .rolling(5, center=True, min_periods=1)
        .median()
        .rolling(3, center=True, min_periods=1)
        .mean()
        .to_numpy(copy=True)
    )
    elevation = segment["elevation_m"].to_numpy(float, copy=True)
    endpoint = np.array([(x[0] + x[-1]) / 2, (y[0] + y[-1]) / 2])
    x[0], y[0] = endpoint
    x[-1], y[-1] = endpoint
    if np.isfinite(elevation[[0, -1]]).any():
        endpoint_elevation = float(np.nanmean(elevation[[0, -1]]))
        elevation[0] = endpoint_elevation
        elevation[-1] = endpoint_elevation

    keep = np.ones(len(x), dtype=bool)
    if len(x) > 2:
        keep[1:] = np.hypot(np.diff(x), np.diff(y)) >= 0.5
        keep[-1] = True
    x, y, elevation = x[keep], y[keep], elevation[keep]
    steps = np.hypot(np.diff(x), np.diff(y))
    valid = steps > 1e-6
    if not valid.all():
        keep_nodes = np.r_[True, valid]
        x, y, elevation = x[keep_nodes], y[keep_nodes], elevation[keep_nodes]
        steps = np.hypot(np.diff(x), np.diff(y))
    raw_s = np.r_[0.0, np.cumsum(steps)]
    total = float(raw_s[-1])
    if total <= settings.centreline_spacing_m * 2:
        raise ValueError("Reference lap is too short to build a centreline.")
    uniform_s = np.arange(0.0, total, settings.centreline_spacing_m)
    if uniform_s[-1] < total:
        uniform_s = np.r_[uniform_s, total]
    uniform_x = np.interp(uniform_s, raw_s, x)
    uniform_y = np.interp(uniform_s, raw_s, y)
    uniform_elevation = _interp_with_nan(uniform_s, raw_s, elevation)
    uniform_x[-1], uniform_y[-1] = uniform_x[0], uniform_y[0]
    if np.isfinite(uniform_elevation[[0, -1]]).any():
        uniform_elevation[-1] = uniform_elevation[0]
    exact_s = np.r_[0.0, np.cumsum(np.hypot(np.diff(uniform_x), np.diff(uniform_y)))]
    return Centreline(uniform_x, uniform_y, exact_s, uniform_elevation, frame)

def map_match_laps(
    points: pd.DataFrame,
    laps: pd.DataFrame,
    centreline: Centreline,
    settings: ReconstructionSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    updated = laps.copy()
    for row_index, lap in updated.iterrows():
        segment = points.loc[
            int(lap["start_global_index"]) : int(lap["end_global_index"])
        ].copy()
        step = segment["step_distance_m"].fillna(0.0).to_numpy(float)
        progress = np.cumsum(step)
        progress -= progress[0]
        if progress[-1] <= 0:
            guesses = np.linspace(0.0, centreline.length_m, len(segment))
        else:
            guesses = progress / progress[-1] * centreline.length_m
        matched_s: list[float] = []
        errors: list[float] = []
        qx: list[float] = []
        qy: list[float] = []
        for px, py, guess in zip(segment["x_m"], segment["y_m"], guesses):
            s, error, mx, my = centreline.project_with_progress(float(px), float(py), float(guess))
            matched_s.append(float(np.clip(s, 0.0, centreline.length_m)))
            errors.append(error)
            qx.append(mx)
            qy.append(my)
        segment["lap_id"] = int(lap["lap_id"])
        segment["s_m"] = matched_s
        segment["map_error_m"] = errors
        segment["matched_x_m"] = qx
        segment["matched_y_m"] = qy
        times = pd.to_datetime(segment["timestamp_utc"], utc=True, errors="coerce")
        segment["elapsed_lap_s"] = (times - times.iloc[0]).dt.total_seconds()
        rows.append(segment)

        error_array = np.asarray(errors)
        backward = int(np.sum(np.diff(np.asarray(matched_s)) < -20.0))
        updated.loc[row_index, "median_map_error_m"] = float(np.median(error_array))
        updated.loc[row_index, "p95_map_error_m"] = float(np.quantile(error_array, 0.95))
        updated.loc[row_index, "maximum_map_error_m"] = float(np.max(error_array))
        updated.loc[row_index, "large_backward_match_count"] = backward
        if backward:
            updated.loc[row_index, "analysis_valid"] = False
            _append_lap_quality_flag(updated, row_index, "large_backward_map_match")
        if float(np.quantile(error_array, 0.95)) > settings.maximum_map_error_m:
            updated.loc[row_index, "analysis_valid"] = False
            _append_lap_quality_flag(
                updated, row_index, "p95_map_error_exceeds_limit"
            )
    return pd.concat(rows, ignore_index=True), updated


def _append_lap_quality_flag(
    laps: pd.DataFrame, row_index: int, flag: str
) -> None:
    flags = [
        item
        for item in str(laps.loc[row_index, "quality_flags"]).split(";")
        if item
    ]
    if flag not in flags:
        flags.append(flag)
        laps.loc[row_index, "quality_flags"] = ";".join(flags)

def build_track_profile(
    matched: pd.DataFrame,
    laps: pd.DataFrame,
    centreline: Centreline,
    settings: ReconstructionSettings,
) -> pd.DataFrame:
    grid = np.arange(0.0, centreline.length_m, settings.profile_spacing_m)
    valid_ids = set(
        laps.loc[laps["analysis_valid"] & laps["use_for_gate_evidence"], "lap_id"].astype(int)
    )
    speed_rows: list[np.ndarray] = []
    elevation_rows: list[np.ndarray] = []
    for lap_id, segment in matched.groupby("lap_id"):
        if int(lap_id) not in valid_ids:
            continue
        profile = _deduplicated_profile(segment, settings.maximum_map_error_m)
        if len(profile) < 3:
            continue
        speed_rows.append(_circular_interp(profile, "speed_analysis_mps", grid, centreline.length_m))
        elevation_rows.append(_circular_interp(profile, "elevation_m", grid, centreline.length_m))
    if not speed_rows:
        raise ValueError("No gate-evidence laps can form a spatial speed profile.")
    speed = np.vstack(speed_rows)
    elevation = np.vstack(elevation_rows)
    return pd.DataFrame(
        {
            "s_m": grid,
            "median_speed_mps": _finite_column_stat(speed, "median"),
            "p10_speed_mps": _finite_column_stat(speed, "quantile", 0.10),
            "p90_speed_mps": _finite_column_stat(speed, "quantile", 0.90),
            "valid_speed_lap_count": np.sum(np.isfinite(speed), axis=0),
            "median_elevation_m": _finite_column_stat(elevation, "median"),
            "p10_elevation_m": _finite_column_stat(elevation, "quantile", 0.10),
            "p90_elevation_m": _finite_column_stat(elevation, "quantile", 0.90),
            "valid_elevation_lap_count": np.sum(np.isfinite(elevation), axis=0),
        }
    )

def _deduplicated_profile(segment: pd.DataFrame, maximum_error_m: float) -> pd.DataFrame:
    good = segment[segment["map_error_m"] <= maximum_error_m][
        ["s_m", "speed_analysis_mps", "elevation_m", "elapsed_lap_s"]
    ].copy()
    good["s_bin"] = good["s_m"].round(1)
    return (
        good.groupby("s_bin", as_index=False)
        .agg(
            s_m=("s_m", "median"),
            speed_analysis_mps=("speed_analysis_mps", "median"),
            elevation_m=("elevation_m", "median"),
            elapsed_lap_s=("elapsed_lap_s", "median"),
        )
        .sort_values("s_m")
    )

def _circular_interp(
    profile: pd.DataFrame, column: str, grid_s: np.ndarray, track_length_m: float
) -> np.ndarray:
    finite = profile[["s_m", column]].dropna()
    if len(finite) < 3:
        return np.full(len(grid_s), np.nan)
    s = finite["s_m"].to_numpy(float)
    values = finite[column].to_numpy(float)
    s_aug = np.r_[s[-1] - track_length_m, s, s[0] + track_length_m]
    value_aug = np.r_[values[-1], values, values[0]]
    return np.interp(grid_s, s_aug, value_aug)

def _interp_with_nan(target: np.ndarray, source: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.full(len(target), np.nan)
    return np.interp(target, source[finite], values[finite])


def _finite_column_stat(
    values: np.ndarray, statistic: str, quantile: float | None = None
) -> np.ndarray:
    """Summarize each spatial column without all-NaN runtime warnings."""

    output = np.full(values.shape[1], np.nan, dtype=float)
    for index in range(values.shape[1]):
        finite = values[:, index][np.isfinite(values[:, index])]
        if not len(finite):
            continue
        if statistic == "median":
            output[index] = float(np.median(finite))
        elif statistic == "quantile" and quantile is not None:
            output[index] = float(np.quantile(finite, quantile))
        else:
            raise ValueError(f"Unsupported column statistic: {statistic}")
    return output
