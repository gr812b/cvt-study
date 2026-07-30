from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/cvt_track_study/track/route_variants.py"
)
spec = importlib.util.spec_from_file_location("route_variants_under_test", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class Diagnostics:
    def __init__(self):
        self.messages = []

    def info(self, code, message, **kwargs):
        self.messages.append(("info", code, message))

    def warning(self, code, message, **kwargs):
        self.messages.append(("warning", code, message))


def _polyline(vertices, spacing=3.0):
    rows = []
    for start, end in zip(vertices[:-1], vertices[1:]):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        length = float(np.linalg.norm(end - start))
        count = max(2, int(np.ceil(length / spacing)) + 1)
        segment = np.linspace(start, end, count)
        if rows:
            segment = segment[1:]
        rows.extend(segment)
    return np.asarray(rows)


def _fixture():
    long_route = _polyline(
        [(0, 0), (100, 0), (100, 100), (50, 150), (0, 100), (0, 0)]
    )
    short_route = _polyline(
        [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    )
    points = []
    laps = []
    cursor = 0
    lap_id = 1
    for run_id, vehicle, route, count in (
        ("ets_reference", "ets", long_route, 3),
        ("mcmaster", "mac", short_route, 4),
    ):
        for repeat in range(count):
            jitter = np.column_stack(
                (
                    np.sin(np.arange(len(route)) + repeat) * 0.4,
                    np.cos(np.arange(len(route)) + repeat) * 0.4,
                )
            )
            local = route + jitter
            start = cursor
            points.extend(local)
            cursor += len(local)
            end = cursor - 1
            increments = np.hypot(np.diff(local[:, 0]), np.diff(local[:, 1]))
            laps.append(
                {
                    "lap_id": lap_id,
                    "run_id": run_id,
                    "vehicle_id": vehicle,
                    "driver_id": f"driver_{vehicle}",
                    "start_global_index": start,
                    "end_global_index": end,
                    "duration_s": 200.0 + repeat,
                    "path_distance_m": float(increments.sum()),
                    "stationary_fraction": 0.0,
                    "speed_coverage_fraction": 1.0,
                    "time_gap_count": 0,
                    "timestamp_regression_count": 0,
                    "distance_ratio_to_median": 1.0,
                    "analysis_valid": True,
                    "quality_flags": "",
                    "use_for_centreline": True,
                    "use_for_gate_evidence": True,
                }
            )
            lap_id += 1
    return pd.DataFrame(points, columns=["x_m", "y_m"]), pd.DataFrame(laps)


def _config(selection="reference_run"):
    return {
        "reconstruction": {"minimum_speed_coverage_fraction": 0.8},
        "route_variants": {
            "enabled": True,
            "minimum_supported_laps": 2,
            "sample_spacing_m": 4.0,
            "same_variant_p95_distance_m": 8.0,
            "divergence_distance_m": 12.0,
            "maximum_divergent_fraction": 0.03,
            "maximum_length_relative_difference": 0.12,
            "maximum_within_variant_length_deviation_fraction": 0.15,
            "selection": selection,
            "reference_run_id": "ets_reference",
        },
    }


def test_detects_two_supported_paths_and_selects_reference_run():
    points, laps = _fixture()
    result = module.detect_and_select_route_variants(
        points, laps, _config(), Diagnostics()
    )
    assert len(result.summary) == 2
    assert int(result.summary["supported"].sum()) == 2
    selected = result.summary[result.summary["selected"]].iloc[0]
    assert "ets_reference" in selected["run_ids"]
    ets = result.laps[result.laps["run_id"] == "ets_reference"]
    mcmaster = result.laps[result.laps["run_id"] == "mcmaster"]
    assert ets["analysis_valid"].all()
    assert (~mcmaster["analysis_valid"]).all()
    assert mcmaster["data_quality_valid"].all()
    assert set(mcmaster["analysis_exclusion_reason"]) == {
        "alternate_supported_route_variant"
    }


def test_multiple_supported_paths_require_explicit_policy_by_default():
    points, laps = _fixture()
    config = _config("require_explicit")
    with pytest.raises(ValueError, match="Multiple supported route variants"):
        module.detect_and_select_route_variants(
            points, laps, config, Diagnostics()
        )


def test_pairwise_audit_marks_cross_route_pairs_incompatible():
    points, laps = _fixture()
    result = module.detect_and_select_route_variants(
        points, laps, _config(), Diagnostics()
    )
    cross = result.pairwise[
        result.pairwise["left_run_id"] != result.pairwise["right_run_id"]
    ]
    within = result.pairwise[
        result.pairwise["left_run_id"] == result.pairwise["right_run_id"]
    ]
    assert not cross["same_route_compatible"].any()
    assert within["same_route_compatible"].all()


def test_bad_telemetry_does_not_create_a_route_variant():
    points, laps = _fixture()
    bad = laps.iloc[[0]].copy()
    bad["lap_id"] = 999
    bad["start_global_index"] = 0
    bad["end_global_index"] = laps.iloc[0]["end_global_index"]
    bad["time_gap_count"] = 3
    laps = pd.concat([laps, bad], ignore_index=True)
    result = module.detect_and_select_route_variants(
        points, laps, _config(), Diagnostics()
    )
    row = result.laps[result.laps["lap_id"] == 999].iloc[0]
    assert not bool(row["data_quality_valid"])
    assert row["route_variant_status"] == "data_quality_rejected"
    assert sum(int(value) for value in result.summary["lap_count"]) == 7
