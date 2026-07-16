"""Per-lap event-response metric extraction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .geo import Centreline
from .laps import _circular_interp, _deduplicated_profile
from .settings import ReconstructionSettings

def extract_event_passes(
    matched: pd.DataFrame,
    laps: pd.DataFrame,
    events: pd.DataFrame,
    centreline: Centreline,
    settings: ReconstructionSettings,
) -> pd.DataFrame:
    valid_laps = laps[laps["analysis_valid"] & laps["use_for_gate_evidence"]]
    lap_lookup = valid_laps.set_index("lap_id")
    profiles = {
        int(lap_id): _deduplicated_profile(segment, settings.maximum_map_error_m)
        for lap_id, segment in matched.groupby("lap_id")
        if int(lap_id) in lap_lookup.index
    }
    rows: list[dict[str, Any]] = []
    step = min(1.0, settings.profile_spacing_m)
    relative_grid = np.arange(-100.0, 121.0, step)
    for _, event in events.iterrows():
        absolute_grid = (event["anchor_s_m"] + relative_grid) % centreline.length_m
        for lap_id, profile in profiles.items():
            speed = _circular_interp(profile, "speed_analysis_mps", absolute_grid, centreline.length_m)
            elevation = _circular_interp(profile, "elevation_m", absolute_grid, centreline.length_m)
            approach = _interval_stat(relative_grid, speed, event["approach_start_rel_m"], event["approach_end_rel_m"], "median")
            entry = _interval_stat(relative_grid, speed, event["entry_start_rel_m"], event["entry_end_rel_m"], "median")
            event_min, event_min_rel_m = _interval_min_with_location(
                relative_grid,
                speed,
                event["feature_start_rel_m"],
                event["feature_end_rel_m"],
            )
            exit_speed = _interval_stat(relative_grid, speed, event["exit_start_rel_m"], event["exit_end_rel_m"], "median")
            entry_elevation = _interval_stat(relative_grid, elevation, event["entry_start_rel_m"], event["entry_end_rel_m"], "median")
            recovery_distance = math.nan
            if np.isfinite(entry):
                mask = (
                    (relative_grid >= event["feature_end_rel_m"])
                    & (relative_grid <= event["feature_end_rel_m"] + event["recovery_limit_m"])
                    & np.isfinite(speed)
                    & (speed >= 0.98 * entry)
                )
                if np.any(mask):
                    recovery_distance = float(relative_grid[np.flatnonzero(mask)[0]] - event["feature_end_rel_m"])
            lap = lap_lookup.loc[lap_id]
            rows.append(
                {
                    "event_id": event["id"],
                    "event_name": event["name"],
                    "sequence": int(event["sequence"]),
                    "response_group_id": event["response_group_id"],
                    "lap_id": lap_id,
                    "run_id": lap["run_id"],
                    "vehicle_id": lap["vehicle_id"],
                    "driver_id": lap["driver_id"],
                    "lap_duration_s": float(lap["duration_s"]),
                    "lap_median_speed_mps": float(lap["median_speed_mps"]),
                    "approach_speed_mps": approach,
                    "entry_speed_mps": entry,
                    "event_min_speed_mps": event_min,
                    "event_min_rel_m": event_min_rel_m,
                    "exit_speed_mps": exit_speed,
                    "entry_elevation_m": entry_elevation,
                    "braking_drop_mps": approach - entry if np.isfinite(approach) and np.isfinite(entry) else math.nan,
                    "recovery_distance_m": recovery_distance,
                    "eligible": bool(np.isfinite(entry) and np.isfinite(approach) and np.isfinite(event_min)),
                }
            )
    return pd.DataFrame(rows)

def _interval_stat(
    relative: np.ndarray,
    values: np.ndarray,
    lower: float,
    upper: float,
    statistic: str,
) -> float:
    selected = values[(relative >= lower) & (relative <= upper) & np.isfinite(values)]
    if len(selected) == 0:
        return math.nan
    if statistic == "min":
        return float(np.min(selected))
    return float(np.median(selected))

def _interval_min_with_location(
    relative: np.ndarray, values: np.ndarray, lower: float, upper: float
) -> tuple[float, float]:
    mask = (relative >= lower) & (relative <= upper) & np.isfinite(values)
    selected_indices = np.flatnonzero(mask)
    if len(selected_indices) == 0:
        return math.nan, math.nan
    local_index = int(np.argmin(values[selected_indices]))
    absolute_index = int(selected_indices[local_index])
    return float(values[absolute_index]), float(relative[absolute_index])
