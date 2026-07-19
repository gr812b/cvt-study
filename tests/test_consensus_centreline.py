from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cvt_track_study.config.diagnostics import DiagnosticBag
from cvt_track_study.track.consensus import (
    build_iterative_consensus,
)
from cvt_track_study.track.geo import LocalFrame
from cvt_track_study.track.settings import (
    ReconstructionSettings,
)


def _sample_polyline(
    vertices: list[tuple[float, float]],
    points_per_segment: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    x: list[float] = []
    y: list[float] = []
    for segment_index, (
        start,
        end,
    ) in enumerate(zip(vertices[:-1], vertices[1:])):
        values = np.linspace(
            0.0,
            1.0,
            points_per_segment,
            endpoint=False,
        )
        if segment_index == len(vertices) - 2:
            values = np.linspace(
                0.0,
                1.0,
                points_per_segment + 1,
                endpoint=True,
            )
        for value in values:
            x.append(
                start[0]
                + value * (end[0] - start[0])
            )
            y.append(
                start[1]
                + value * (end[1] - start[1])
            )
    return np.asarray(x), np.asarray(y)


def _synthetic_points_and_laps():
    main = [
        (0.0, 0.0),
        (0.0, 100.0),
        (100.0, 100.0),
        (130.0, 70.0),
        (100.0, 40.0),
        (100.0, 0.0),
        (0.0, 0.0),
    ]
    cutter = [
        (0.0, 0.0),
        (0.0, 100.0),
        (100.0, 100.0),
        (100.0, 40.0),
        (100.0, 0.0),
        (0.0, 0.0),
    ]

    point_rows = []
    lap_rows = []
    global_index = 0
    for lap_id in range(1, 8):
        vertices = cutter if lap_id == 7 else main
        x, y = _sample_polyline(vertices)
        # Small normal run-to-run variation for the majority.
        if lap_id != 7:
            x = x + (lap_id - 3.5) * 0.15
            y = y + math.sin(lap_id) * 0.10
        start_index = global_index
        for local_index, (px, py) in enumerate(zip(x, y)):
            point_rows.append(
                {
                    "run_id": "synthetic",
                    "track_index": 0,
                    "segment_index": lap_id,
                    "point_index": local_index,
                    "x_m": px,
                    "y_m": py,
                    "latitude_deg": py / 111_000.0,
                    "longitude_deg": px / 82_000.0,
                    "timestamp_utc": pd.Timestamp(
                        "2026-01-01T00:00:00Z"
                    )
                    + pd.Timedelta(
                        seconds=global_index
                    ),
                    "step_distance_m": np.nan,
                    "speed_analysis_mps": 8.0,
                    "elevation_m": 100.0,
                }
            )
            global_index += 1
        end_index = global_index - 1
        lap_rows.append(
            {
                "lap_id": lap_id,
                "run_id": "synthetic",
                "vehicle_id": "vehicle",
                "driver_id": "driver",
                "track_index": 0,
                "segment_index": lap_id,
                "local_lap_id": lap_id,
                "start_global_index": start_index,
                "end_global_index": end_index,
                "duration_s": 90.0
                if lap_id == 7
                else 100.0 + lap_id,
                "analysis_valid": True,
                "pre_consensus_valid": True,
                "quality_flags": "",
                "use_for_centreline": True,
                "use_for_gate_evidence": True,
                "reference_lap": False,
                "centreline_included": True,
                "consensus_excluded": False,
                "consensus_iteration_excluded": np.nan,
                "consensus_exclusion_reason": "",
            }
        )

    points = pd.DataFrame(point_rows)
    for _, indices in points.groupby(
        ["run_id", "track_index", "segment_index"]
    ).groups.items():
        idx = list(indices)
        dx = np.diff(points.loc[idx, "x_m"])
        dy = np.diff(points.loc[idx, "y_m"])
        points.loc[idx, "step_distance_m"] = np.r_[
            np.nan, np.hypot(dx, dy)
        ]
    return points, pd.DataFrame(lap_rows)


def _track_config():
    return {
        "reconstruction": {
            "lap_gate_event_id": "start_finish",
            "lap_gate_radius_m": 15.0,
            "minimum_lap_time_s": 60.0,
            "maximum_reasonable_speed_mps": 25.0,
            "maximum_normal_time_step_s": 3.0,
            "stationary_speed_mps": 0.8,
            "minimum_speed_coverage_fraction": 0.80,
            "centreline_spacing_m": 2.0,
            "profile_spacing_m": 5.0,
            "maximum_map_error_m": 20.0,
            "speed_spike_threshold_mps": 4.0,
        },
        "telemetry_cleanup": {
            "enabled": True,
            "maximum_excursion_points": 3,
            "minimum_excursion_leg_m": 35.0,
            "impossible_speed_multiplier": 1.5,
            "maximum_bridge_speed_multiplier": 1.0,
            "maximum_bridge_gap_s": 8.0,
            "maximum_auto_removed_fraction": 0.10,
            "maximum_auto_removed_points": 25,
            "isolated_map_error_m": 40.0,
            "maximum_isolated_map_outlier_points": 3,
        },
        "centreline_consensus": {
            "minimum_laps": 3,
            "maximum_iterations": 6,
            "convergence_tolerance_m": 0.20,
            "smoothing_window_nodes": 3,
            "leave_one_out_p95_limit_m": 8.0,
            "sustained_error_threshold_m": 8.0,
            "minimum_sustained_outlier_fraction": 0.03,
            "strong_sustained_outlier_fraction": 0.08,
            "maximum_leave_one_out_shift_m": 3.0,
            "robust_mad_multiplier": 3.0,
            "maximum_outlier_fraction_per_iteration": 0.20,
        },
    }


def test_fast_corner_cutter_does_not_own_centreline():
    points, laps = _synthetic_points_and_laps()
    settings = ReconstructionSettings.from_mapping(
        _track_config()
    )
    result = build_iterative_consensus(
        points,
        laps,
        LocalFrame(0.0, 0.0),
        settings,
        _track_config(),
        DiagnosticBag(),
    )

    cutter = result.laps.set_index("lap_id").loc[7]
    assert bool(cutter["consensus_excluded"])
    assert not bool(cutter["centreline_included"])
    assert not bool(cutter["analysis_valid"])

    majority = result.laps[
        result.laps["lap_id"] != 7
    ]
    assert majority["centreline_included"].all()
    assert (
        result.laps.loc[
            result.laps["reference_lap"], "lap_id"
        ].iloc[0]
        != 7
    )

    # The consensus must follow the majority detour around x=130,
    # not the cutter's straight x=100 line.
    upper_detour = result.centreline.x_m[
        (result.centreline.y_m > 55.0)
        & (result.centreline.y_m < 85.0)
    ]
    assert upper_detour.max() > 120.0
