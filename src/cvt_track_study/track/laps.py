"""Lap detection, robust centreline reconstruction, map matching, and profiles."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from cvt_track_study.config.diagnostics import DiagnosticBag
from cvt_track_study.gpx.model import GPXIngestionResult

from .geo import Centreline, LocalFrame
from .settings import ReconstructionSettings


_MAP_QUALITY_FLAGS = {
    "large_backward_map_match",
    "p95_map_error_exceeds_limit",
    "no_map_matched_points",
    "no_finite_map_errors",
}


def _clean_speed(
    points: pd.DataFrame,
    settings: ReconstructionSettings,
) -> pd.Series:
    raw = pd.to_numeric(
        points["analysis_speed_mps"], errors="coerce"
    )
    raw = raw.where(
        raw.between(0.0, settings.maximum_reasonable_speed_mps)
    )
    cleaned = raw.copy()
    for _, indices in points.groupby(
        ["run_id", "track_index", "segment_index"], sort=False
    ).groups.items():
        idx = list(indices)
        local = raw.loc[idx]
        median = local.rolling(
            5, center=True, min_periods=3
        ).median()
        spike = (
            local - median
        ).abs() > settings.speed_spike_threshold_mps
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
    run_metadata = {
        result.metadata.run_id: result.metadata
        for result in ingestion_results
    }
    gate_x, gate_y = frame.to_xy(
        [gate_latitude_deg], [gate_longitude_deg]
    )
    rows: list[dict[str, Any]] = []
    global_lap_id = 0
    for (
        run_id,
        track_index,
        segment_index,
    ), indices in points.groupby(
        ["run_id", "track_index", "segment_index"], sort=False
    ).groups.items():
        idx = list(indices)
        segment = points.loc[idx].copy()
        distance = np.hypot(
            segment["x_m"] - gate_x[0],
            segment["y_m"] - gate_y[0],
        )
        crossings = _gate_crossings(
            segment,
            distance.to_numpy(dtype=float, copy=True),
            settings,
        )
        if len(crossings) < 2:
            diagnostics.warning(
                "NO_COMPLETE_LAPS_IN_SEGMENT",
                (
                    f"Run {run_id}, segment "
                    f"{track_index}:{segment_index} produced fewer "
                    "than two lap-gate visits."
                ),
                path=f"runs.{run_id}",
            )
            continue

        for local_lap, (
            start_pos,
            end_pos,
        ) in enumerate(
            zip(crossings[:-1], crossings[1:]), start=1
        ):
            lap_segment = segment.iloc[
                start_pos : end_pos + 1
            ]
            times = pd.to_datetime(
                lap_segment["timestamp_utc"],
                utc=True,
                errors="coerce",
            )
            duration = (
                float(
                    (
                        times.iloc[-1] - times.iloc[0]
                    ).total_seconds()
                )
                if times.notna().all()
                else math.nan
            )
            global_lap_id += 1
            dt = lap_segment["time_step_s"].to_numpy(dtype=float, copy=True)
            distance_m = float(
                lap_segment["step_distance_m"]
                .fillna(0.0)
                .sum()
            )
            speed_values = pd.to_numeric(
                lap_segment["speed_analysis_mps"],
                errors="coerce",
            )
            speed_coverage_fraction = float(
                speed_values.notna().mean()
            )
            stationary_fraction = (
                float(
                    (
                        speed_values.dropna()
                        < settings.stationary_speed_mps
                    ).mean()
                )
                if speed_values.notna().any()
                else 1.0
            )
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
                    "start_global_index": int(
                        lap_segment.index[0]
                    ),
                    "end_global_index": int(
                        lap_segment.index[-1]
                    ),
                    "start_time_utc": times.iloc[0],
                    "end_time_utc": times.iloc[-1],
                    "duration_s": duration,
                    "path_distance_m": distance_m,
                    "stationary_fraction": stationary_fraction,
                    "speed_coverage_fraction": (
                        speed_coverage_fraction
                    ),
                    "time_gap_count": int(
                        np.sum(
                            np.isfinite(dt)
                            & (
                                dt
                                > settings.maximum_normal_time_step_s
                            )
                        )
                    ),
                    "timestamp_regression_count": int(
                        np.sum(np.isfinite(dt) & (dt < 0))
                    ),
                    "median_speed_mps": float(
                        lap_segment[
                            "speed_analysis_mps"
                        ].median()
                    ),
                    "maximum_speed_mps": float(
                        lap_segment[
                            "speed_analysis_mps"
                        ].max()
                    ),
                    "use_for_centreline": (
                        metadata.use_for_centreline
                    ),
                    "use_for_gate_evidence": (
                        metadata.use_for_gate_evidence
                    ),
                }
            )

    laps = pd.DataFrame(rows)
    if laps.empty:
        raise ValueError(
            "No complete laps were found between lap-gate visits."
        )

    median_distance = float(
        laps["path_distance_m"].median()
    )
    laps["distance_ratio_to_median"] = (
        laps["path_distance_m"] / median_distance
    )
    laps["analysis_valid"] = (
        laps["duration_s"].notna()
        & laps["distance_ratio_to_median"].between(
            0.85, 1.15
        )
        & (laps["stationary_fraction"] <= 0.15)
        & (
            laps["speed_coverage_fraction"]
            >= settings.minimum_speed_coverage_fraction
        )
        & (laps["time_gap_count"] == 0)
        & (laps["timestamp_regression_count"] == 0)
    )
    laps["quality_flags"] = laps.apply(
        lambda row: _initial_lap_quality_flags(
            row, settings
        ),
        axis=1,
    )

    eligible = laps[
        laps["analysis_valid"]
        & laps["use_for_centreline"]
    ]
    if eligible.empty:
        raise ValueError(
            "No valid lap is enabled for centreline construction."
        )

    # No lap owns the geometry. This field is assigned only after the
    # final consensus, to identify the retained lap closest to it.
    laps["reference_lap"] = False
    laps["pre_consensus_valid"] = laps[
        "analysis_valid"
    ].astype(bool)
    laps["centreline_included"] = (
        laps["analysis_valid"]
        & laps["use_for_centreline"]
    )
    laps["consensus_excluded"] = False
    laps["consensus_iteration_excluded"] = np.nan
    laps["consensus_exclusion_reason"] = ""
    return laps


def _initial_lap_quality_flags(
    lap: pd.Series,
    settings: ReconstructionSettings,
) -> str:
    flags: list[str] = []
    if not np.isfinite(lap["duration_s"]):
        flags.append("missing_or_invalid_duration")
    if not 0.85 <= float(
        lap["distance_ratio_to_median"]
    ) <= 1.15:
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
    segment: pd.DataFrame,
    distance_m: np.ndarray,
    settings: ReconstructionSettings,
) -> list[int]:
    inside = np.flatnonzero(
        distance_m <= settings.lap_gate_radius_m
    )
    groups: list[list[int]] = []
    for position in inside:
        if not groups:
            groups.append([int(position)])
            continue
        current_time = pd.to_datetime(
            segment.iloc[position]["timestamp_utc"],
            utc=True,
            errors="coerce",
        )
        prior_time = pd.to_datetime(
            segment.iloc[groups[-1][-1]][
                "timestamp_utc"
            ],
            utc=True,
            errors="coerce",
        )
        gap = (
            (
                current_time - prior_time
            ).total_seconds()
            if pd.notna(current_time)
            and pd.notna(prior_time)
            else 0.0
        )
        if gap > 3.0:
            groups.append([int(position)])
        else:
            groups[-1].append(int(position))

    candidates = [
        min(group, key=lambda item: distance_m[item])
        for group in groups
    ]
    merged: list[int] = []
    for position in candidates:
        if not merged:
            merged.append(position)
            continue
        current_time = pd.to_datetime(
            segment.iloc[position]["timestamp_utc"],
            utc=True,
            errors="coerce",
        )
        prior_time = pd.to_datetime(
            segment.iloc[merged[-1]][
                "timestamp_utc"
            ],
            utc=True,
            errors="coerce",
        )
        elapsed = (
            (
                current_time - prior_time
            ).total_seconds()
            if pd.notna(current_time)
            and pd.notna(prior_time)
            else math.inf
        )
        if elapsed >= settings.minimum_lap_time_s:
            merged.append(position)
        elif (
            distance_m[position]
            < distance_m[merged[-1]]
        ):
            merged[-1] = position
    return merged


def build_centreline(
    points: pd.DataFrame,
    laps: pd.DataFrame,
    frame: LocalFrame,
    settings: ReconstructionSettings,
    *,
    lap_ids: Iterable[int] | None = None,
    alignment_centreline: Centreline | None = None,
) -> Centreline:
    """Build a pointwise-median centreline from all selected laps.

    The first pass aligns laps by normalized geometric progress. Later
    passes align each lap to the preceding consensus before taking the
    median. This prevents a fast or unusual lap from owning the track.
    """

    selected_ids = (
        {int(value) for value in lap_ids}
        if lap_ids is not None
        else _default_centreline_lap_ids(laps)
    )
    selected = laps[
        laps["lap_id"].astype(int).isin(selected_ids)
    ].copy()
    if selected.empty:
        raise ValueError(
            "Centreline consensus requires at least one selected lap."
        )

    lap_records: list[dict[str, np.ndarray | float]] = []
    for _, lap in selected.iterrows():
        segment = _lap_segment(points, lap)
        record = _resample_lap_for_consensus(
            segment,
            settings,
            alignment_centreline,
        )
        if record is not None:
            lap_records.append(record)

    if not lap_records:
        raise ValueError(
            "Selected laps contain too little usable geometry "
            "for centreline construction."
        )

    if alignment_centreline is None:
        target_length = float(
            np.median(
                [
                    float(record["length_m"])
                    for record in lap_records
                ]
            )
        )
    else:
        target_length = alignment_centreline.length_m

    node_count = max(
        4,
        int(
            math.ceil(
                target_length
                / settings.centreline_spacing_m
            )
        )
        + 1,
    )
    target_fraction = np.linspace(
        0.0, 1.0, node_count
    )

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    elevation_rows: list[np.ndarray] = []
    endpoint_x: list[float] = []
    endpoint_y: list[float] = []

    for record in lap_records:
        source_fraction = np.asarray(
            record["fraction"], dtype=float
        )
        x = np.asarray(record["x_m"], dtype=float)
        y = np.asarray(record["y_m"], dtype=float)
        elevation = np.asarray(
            record["elevation_m"], dtype=float
        )
        x_rows.append(
            np.interp(target_fraction, source_fraction, x)
        )
        y_rows.append(
            np.interp(target_fraction, source_fraction, y)
        )
        elevation_rows.append(
            _interp_with_nan(
                target_fraction,
                source_fraction,
                elevation,
            )
        )
        endpoint_x.append(float(record["endpoint_x_m"]))
        endpoint_y.append(float(record["endpoint_y_m"]))

    consensus_x = np.array(
        np.median(np.vstack(x_rows), axis=0),
        dtype=float,
        copy=True,
    )
    consensus_y = np.array(
        np.median(np.vstack(y_rows), axis=0),
        dtype=float,
        copy=True,
    )
    consensus_elevation = np.array(
        _finite_column_stat(
            np.vstack(elevation_rows), "median"
        ),
        dtype=float,
        copy=True,
    )

    consensus_x = _circular_smooth(
        consensus_x,
        settings.consensus_smoothing_window_nodes,
    )
    consensus_y = _circular_smooth(
        consensus_y,
        settings.consensus_smoothing_window_nodes,
    )

    # Preserve the physical lap-gate anchor as the median of the
    # contributing lap endpoints rather than letting smoothing move it.
    gate_x = float(np.median(endpoint_x))
    gate_y = float(np.median(endpoint_y))
    consensus_x[0] = consensus_x[-1] = gate_x
    consensus_y[0] = consensus_y[-1] = gate_y
    if np.isfinite(consensus_elevation[[0, -1]]).any():
        gate_elevation = float(
            np.nanmedian(
                consensus_elevation[[0, -1]]
            )
        )
        consensus_elevation[0] = gate_elevation
        consensus_elevation[-1] = gate_elevation

    return _uniform_centreline(
        consensus_x,
        consensus_y,
        consensus_elevation,
        frame,
        settings.centreline_spacing_m,
    )


def map_match_laps(
    points: pd.DataFrame,
    laps: pd.DataFrame,
    centreline: Centreline,
    settings: ReconstructionSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    updated = _clear_map_quality(laps)
    for row_index, lap in updated.iterrows():
        segment = _lap_segment(points, lap)
        if len(segment) < 2:
            updated.loc[
                row_index, "analysis_valid"
            ] = False
            _append_lap_quality_flag(
                updated,
                row_index,
                "no_map_matched_points",
            )
            continue

        mapped = map_segment_to_centreline(
            segment, centreline
        )
        mapped["lap_id"] = int(lap["lap_id"])
        times = pd.to_datetime(
            mapped["timestamp_utc"],
            utc=True,
            errors="coerce",
        )
        mapped["elapsed_lap_s"] = (
            times - times.iloc[0]
        ).dt.total_seconds()
        rows.append(mapped)

        error_array = pd.to_numeric(
            mapped["map_error_m"], errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        finite_errors = error_array[
            np.isfinite(error_array)
        ]
        if not len(finite_errors):
            updated.loc[
                row_index, "analysis_valid"
            ] = False
            _append_lap_quality_flag(
                updated,
                row_index,
                "no_finite_map_errors",
            )
            continue

        matched_s = pd.to_numeric(
            mapped["s_m"], errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        backward = int(
            np.sum(
                np.diff(matched_s) < -20.0
            )
        )
        p95 = float(
            np.quantile(finite_errors, 0.95)
        )
        updated.loc[
            row_index, "median_map_error_m"
        ] = float(np.median(finite_errors))
        updated.loc[
            row_index, "p95_map_error_m"
        ] = p95
        updated.loc[
            row_index, "maximum_map_error_m"
        ] = float(np.max(finite_errors))
        updated.loc[
            row_index, "large_backward_match_count"
        ] = backward
        updated.loc[
            row_index, "sustained_map_error_fraction"
        ] = _uniform_error_fraction(
            mapped,
            centreline,
            settings.maximum_map_error_m,
            settings.centreline_spacing_m,
        )

        if backward:
            updated.loc[
                row_index, "analysis_valid"
            ] = False
            _append_lap_quality_flag(
                updated,
                row_index,
                "large_backward_map_match",
            )
        if p95 > settings.maximum_map_error_m:
            updated.loc[
                row_index, "analysis_valid"
            ] = False
            _append_lap_quality_flag(
                updated,
                row_index,
                "p95_map_error_exceeds_limit",
            )

    if not rows:
        raise ValueError(
            "No lap points remain for map matching."
        )
    return pd.concat(rows, ignore_index=True), updated


def map_segment_to_centreline(
    segment: pd.DataFrame,
    centreline: Centreline,
) -> pd.DataFrame:
    output = segment.copy()
    raw_progress = _geometric_progress(output)
    if raw_progress[-1] <= 0:
        guesses = np.linspace(
            0.0, centreline.length_m, len(output)
        )
    else:
        guesses = (
            raw_progress
            / raw_progress[-1]
            * centreline.length_m
        )

    matched_s: list[float] = []
    errors: list[float] = []
    qx: list[float] = []
    qy: list[float] = []
    for px, py, guess in zip(
        output["x_m"],
        output["y_m"],
        guesses,
    ):
        s_m, error_m, matched_x, matched_y = (
            centreline.project_with_progress(
                float(px), float(py), float(guess)
            )
        )
        matched_s.append(
            float(
                np.clip(
                    s_m, 0.0, centreline.length_m
                )
            )
        )
        errors.append(float(error_m))
        qx.append(float(matched_x))
        qy.append(float(matched_y))

    output["s_m"] = matched_s
    output["map_error_m"] = errors
    output["matched_x_m"] = qx
    output["matched_y_m"] = qy
    return output


def centreline_distance_summary(
    left: Centreline,
    right: Centreline,
    *,
    sample_count: int = 400,
) -> tuple[float, float]:
    fraction = np.linspace(
        0.0, 1.0, sample_count
    )
    left_x, left_y = _sample_centreline(
        left, fraction
    )
    right_x, right_y = _sample_centreline(
        right, fraction
    )
    distance = np.hypot(
        left_x - right_x, left_y - right_y
    )
    return (
        float(np.quantile(distance, 0.95)),
        float(np.max(distance)),
    )


def _default_centreline_lap_ids(
    laps: pd.DataFrame,
) -> set[int]:
    if "centreline_included" in laps:
        mask = laps["centreline_included"].astype(bool)
    else:
        mask = (
            laps["analysis_valid"].astype(bool)
            & laps["use_for_centreline"].astype(bool)
        )
    return set(
        laps.loc[mask, "lap_id"].astype(int)
    )


def _lap_segment(
    points: pd.DataFrame,
    lap: pd.Series,
) -> pd.DataFrame:
    segment = points.loc[
        int(lap["start_global_index"])
        : int(lap["end_global_index"])
    ].copy()
    return segment.dropna(
        subset=["x_m", "y_m"]
    ).sort_index()


def _resample_lap_for_consensus(
    segment: pd.DataFrame,
    settings: ReconstructionSettings,
    alignment_centreline: Centreline | None,
) -> dict[str, np.ndarray | float] | None:
    if len(segment) < 4:
        return None

    x = (
        pd.to_numeric(segment["x_m"], errors="coerce")
        .rolling(5, center=True, min_periods=1)
        .median()
        .rolling(3, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float, copy=True)
    )
    y = (
        pd.to_numeric(segment["y_m"], errors="coerce")
        .rolling(5, center=True, min_periods=1)
        .median()
        .rolling(3, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float, copy=True)
    )
    elevation = pd.to_numeric(
        segment["elevation_m"], errors="coerce"
    ).to_numpy(dtype=float, copy=True)

    endpoint_x = float(
        np.mean([x[0], x[-1]])
    )
    endpoint_y = float(
        np.mean([y[0], y[-1]])
    )
    x[0] = x[-1] = endpoint_x
    y[0] = y[-1] = endpoint_y

    raw_s = np.r_[
        0.0,
        np.cumsum(
            np.hypot(np.diff(x), np.diff(y))
        ),
    ]
    if raw_s[-1] <= (
        2.0 * settings.centreline_spacing_m
    ):
        return None

    if alignment_centreline is None:
        source = raw_s / raw_s[-1]
    else:
        guesses = (
            raw_s
            / raw_s[-1]
            * alignment_centreline.length_m
        )
        aligned_s: list[float] = []
        for px, py, guess in zip(x, y, guesses):
            s_m, _, _, _ = (
                alignment_centreline.project_with_progress(
                    float(px),
                    float(py),
                    float(guess),
                )
            )
            aligned_s.append(
                float(
                    np.clip(
                        s_m,
                        0.0,
                        alignment_centreline.length_m,
                    )
                )
            )
        aligned = np.maximum.accumulate(
            np.asarray(aligned_s, dtype=float)
        )
        if aligned[-1] <= 0:
            source = raw_s / raw_s[-1]
        else:
            source = (
                aligned
                / alignment_centreline.length_m
            )

    source, x, y, elevation = _deduplicate_axis(
        source, x, y, elevation
    )
    if len(source) < 3:
        return None
    source[0] = 0.0
    source[-1] = 1.0

    return {
        "fraction": source,
        "x_m": x,
        "y_m": y,
        "elevation_m": elevation,
        "length_m": float(raw_s[-1]),
        "endpoint_x_m": endpoint_x,
        "endpoint_y_m": endpoint_y,
    }


def _deduplicate_axis(
    source: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    elevation: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    frame = pd.DataFrame(
        {
            "source": source,
            "x_m": x,
            "y_m": y,
            "elevation_m": elevation,
        }
    )
    frame["source_bin"] = frame[
        "source"
    ].round(6)
    grouped = (
        frame.groupby(
            "source_bin", as_index=False
        )
        .agg(
            source=("source", "median"),
            x_m=("x_m", "median"),
            y_m=("y_m", "median"),
            elevation_m=("elevation_m", "median"),
        )
        .sort_values("source")
    )
    return (
        grouped["source"].to_numpy(dtype=float, copy=True),
        grouped["x_m"].to_numpy(dtype=float, copy=True),
        grouped["y_m"].to_numpy(dtype=float, copy=True),
        grouped["elevation_m"].to_numpy(dtype=float, copy=True),
    )


def _uniform_centreline(
    x: np.ndarray,
    y: np.ndarray,
    elevation: np.ndarray,
    frame: LocalFrame,
    spacing_m: float,
) -> Centreline:
    keep = np.ones(len(x), dtype=bool)
    if len(x) > 2:
        keep[1:] = (
            np.hypot(np.diff(x), np.diff(y))
            >= 0.25
        )
        keep[-1] = True
    x = x[keep]
    y = y[keep]
    elevation = elevation[keep]

    steps = np.hypot(
        np.diff(x), np.diff(y)
    )
    valid = steps > 1e-6
    if not valid.all():
        keep_nodes = np.r_[True, valid]
        x = x[keep_nodes]
        y = y[keep_nodes]
        elevation = elevation[keep_nodes]
        steps = np.hypot(
            np.diff(x), np.diff(y)
        )

    raw_s = np.r_[
        0.0, np.cumsum(steps)
    ]
    total = float(raw_s[-1])
    if total <= 2.0 * spacing_m:
        raise ValueError(
            "Consensus centreline is too short."
        )

    uniform_s = np.arange(
        0.0, total, spacing_m
    )
    if not len(uniform_s) or uniform_s[-1] < total:
        uniform_s = np.r_[uniform_s, total]

    uniform_x = np.interp(
        uniform_s, raw_s, x
    )
    uniform_y = np.interp(
        uniform_s, raw_s, y
    )
    uniform_elevation = _interp_with_nan(
        uniform_s, raw_s, elevation
    )
    uniform_x[-1] = uniform_x[0]
    uniform_y[-1] = uniform_y[0]
    if np.isfinite(
        uniform_elevation[[0, -1]]
    ).any():
        uniform_elevation[-1] = (
            uniform_elevation[0]
        )

    exact_s = np.r_[
        0.0,
        np.cumsum(
            np.hypot(
                np.diff(uniform_x),
                np.diff(uniform_y),
            )
        ),
    ]
    return Centreline(
        uniform_x,
        uniform_y,
        exact_s,
        uniform_elevation,
        frame,
    )


def _circular_smooth(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    if len(values) <= 4 or window <= 1:
        return values.copy()

    unique = values[:-1].copy()
    half = window // 2
    extended = np.r_[
        unique[-half:],
        unique,
        unique[:half],
    ]
    median = (
        pd.Series(extended)
        .rolling(
            window,
            center=True,
            min_periods=1,
        )
        .median()
        .to_numpy(dtype=float, copy=True)
    )[half : half + len(unique)]
    extended_median = np.r_[
        median[-half:],
        median,
        median[:half],
    ]
    smooth = (
        pd.Series(extended_median)
        .rolling(
            window,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy(dtype=float, copy=True)
    )[half : half + len(unique)]
    return np.r_[smooth, smooth[0]]


def _geometric_progress(
    segment: pd.DataFrame,
) -> np.ndarray:
    x = pd.to_numeric(
        segment["x_m"], errors="coerce"
    ).to_numpy(dtype=float, copy=True)
    y = pd.to_numeric(
        segment["y_m"], errors="coerce"
    ).to_numpy(dtype=float, copy=True)
    return np.r_[
        0.0,
        np.cumsum(
            np.hypot(np.diff(x), np.diff(y))
        ),
    ]


def _uniform_error_fraction(
    mapped: pd.DataFrame,
    centreline: Centreline,
    threshold_m: float,
    spacing_m: float,
) -> float:
    profile = mapped[
        ["s_m", "map_error_m"]
    ].dropna()
    if len(profile) < 3:
        return math.nan
    profile["s_bin"] = profile[
        "s_m"
    ].round(1)
    profile = (
        profile.groupby("s_bin", as_index=False)
        .agg(
            s_m=("s_m", "median"),
            map_error_m=(
                "map_error_m", "median"
            ),
        )
        .sort_values("s_m")
    )
    grid = np.arange(
        0.0,
        centreline.length_m,
        max(spacing_m, 1.0),
    )
    if not len(grid):
        return math.nan
    s = profile["s_m"].to_numpy(dtype=float, copy=True)
    error = profile[
        "map_error_m"
    ].to_numpy(dtype=float, copy=True)
    interpolated = np.interp(
        grid, s, error
    )
    return float(
        np.mean(interpolated > threshold_m)
    )


def _sample_centreline(
    centreline: Centreline,
    fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = (
        centreline.s_m
        / centreline.length_m
    )
    return (
        np.interp(
            fraction, normalized, centreline.x_m
        ),
        np.interp(
            fraction, normalized, centreline.y_m
        ),
    )


def _clear_map_quality(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    output = laps.copy()
    if "pre_consensus_valid" in output:
        base_valid = output[
            "pre_consensus_valid"
        ].astype(bool)
    else:
        base_valid = output[
            "analysis_valid"
        ].astype(bool)

    excluded = (
        output["consensus_excluded"].astype(bool)
        if "consensus_excluded" in output
        else pd.Series(
            False, index=output.index
        )
    )
    output["analysis_valid"] = (
        base_valid & ~excluded
    )
    output["quality_flags"] = output[
        "quality_flags"
    ].map(_without_map_flags)
    return output


def _without_map_flags(value: Any) -> str:
    return ";".join(
        flag
        for flag in str(value or "").split(";")
        if flag
        and flag.lower() != "nan"
        and flag not in _MAP_QUALITY_FLAGS
    )


def _append_lap_quality_flag(
    laps: pd.DataFrame,
    row_index: int,
    flag: str,
) -> None:
    flags = [
        item
        for item in str(
            laps.loc[
                row_index, "quality_flags"
            ]
        ).split(";")
        if item and item.lower() != "nan"
    ]
    if flag not in flags:
        flags.append(flag)
        laps.loc[
            row_index, "quality_flags"
        ] = ";".join(flags)


def build_track_profile(
    matched: pd.DataFrame,
    laps: pd.DataFrame,
    centreline: Centreline,
    settings: ReconstructionSettings,
) -> pd.DataFrame:
    grid = np.arange(
        0.0,
        centreline.length_m,
        settings.profile_spacing_m,
    )
    valid_ids = set(
        laps.loc[
            laps["analysis_valid"]
            & laps["use_for_gate_evidence"],
            "lap_id",
        ].astype(int)
    )
    speed_rows: list[np.ndarray] = []
    elevation_rows: list[np.ndarray] = []
    for lap_id, segment in matched.groupby(
        "lap_id"
    ):
        if int(lap_id) not in valid_ids:
            continue
        profile = _deduplicated_profile(
            segment,
            settings.maximum_map_error_m,
        )
        if len(profile) < 3:
            continue
        speed_rows.append(
            _circular_interp(
                profile,
                "speed_analysis_mps",
                grid,
                centreline.length_m,
            )
        )
        elevation_rows.append(
            _circular_interp(
                profile,
                "elevation_m",
                grid,
                centreline.length_m,
            )
        )
    if not speed_rows:
        raise ValueError(
            "No gate-evidence laps can form a spatial speed profile."
        )
    speed = np.vstack(speed_rows)
    elevation = np.vstack(elevation_rows)
    return pd.DataFrame(
        {
            "s_m": grid,
            "median_speed_mps": _finite_column_stat(
                speed, "median"
            ),
            "p10_speed_mps": _finite_column_stat(
                speed, "quantile", 0.10
            ),
            "p90_speed_mps": _finite_column_stat(
                speed, "quantile", 0.90
            ),
            "valid_speed_lap_count": np.sum(
                np.isfinite(speed), axis=0
            ),
            "median_elevation_m": _finite_column_stat(
                elevation, "median"
            ),
            "p10_elevation_m": _finite_column_stat(
                elevation, "quantile", 0.10
            ),
            "p90_elevation_m": _finite_column_stat(
                elevation, "quantile", 0.90
            ),
            "valid_elevation_lap_count": np.sum(
                np.isfinite(elevation), axis=0
            ),
        }
    )


def _deduplicated_profile(
    segment: pd.DataFrame,
    maximum_error_m: float,
) -> pd.DataFrame:
    good = segment[
        segment["map_error_m"] <= maximum_error_m
    ][
        [
            "s_m",
            "speed_analysis_mps",
            "elevation_m",
            "elapsed_lap_s",
        ]
    ].copy()
    good["s_bin"] = good["s_m"].round(1)
    return (
        good.groupby("s_bin", as_index=False)
        .agg(
            s_m=("s_m", "median"),
            speed_analysis_mps=(
                "speed_analysis_mps", "median"
            ),
            elevation_m=(
                "elevation_m", "median"
            ),
            elapsed_lap_s=(
                "elapsed_lap_s", "median"
            ),
        )
        .sort_values("s_m")
    )


def _circular_interp(
    profile: pd.DataFrame,
    column: str,
    grid_s: np.ndarray,
    track_length_m: float,
) -> np.ndarray:
    finite = profile[
        ["s_m", column]
    ].dropna()
    if len(finite) < 3:
        return np.full(len(grid_s), np.nan)
    s = finite["s_m"].to_numpy(dtype=float, copy=True)
    values = finite[column].to_numpy(dtype=float, copy=True)
    s_aug = np.r_[
        s[-1] - track_length_m,
        s,
        s[0] + track_length_m,
    ]
    value_aug = np.r_[
        values[-1], values, values[0]
    ]
    return np.interp(
        grid_s, s_aug, value_aug
    )


def _interp_with_nan(
    target: np.ndarray,
    source: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.full(len(target), np.nan)
    return np.interp(
        target,
        source[finite],
        values[finite],
    )


def _finite_column_stat(
    values: np.ndarray,
    statistic: str,
    quantile: float | None = None,
) -> np.ndarray:
    output = np.full(
        values.shape[1], np.nan, dtype=float
    )
    for index in range(values.shape[1]):
        finite = values[:, index][
            np.isfinite(values[:, index])
        ]
        if not len(finite):
            continue
        if statistic == "median":
            output[index] = float(
                np.median(finite)
            )
        elif (
            statistic == "quantile"
            and quantile is not None
        ):
            output[index] = float(
                np.quantile(finite, quantile)
            )
        else:
            raise ValueError(
                f"Unsupported column statistic: {statistic}"
            )
    return output
