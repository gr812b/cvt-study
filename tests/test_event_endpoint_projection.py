from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from cvt_track_study.track.events import (
    _event_endpoint_relative_s,
    _resolve_explicit_interval_endpoints,
)


class _IdentityFrame:
    def to_xy(self, latitude, longitude):
        return (
            np.asarray(latitude, dtype=float),
            np.asarray(longitude, dtype=float),
        )


class _FakeCentreline:
    def __init__(self, candidates, length_m=1000.0):
        self.frame = _IdentityFrame()
        self.length_m = length_m
        self._candidates = candidates

    def distinct_candidates(self, x_m, y_m):
        key = round(float(x_m), 6)
        return [dict(item) for item in self._candidates[key]]


def _event(
    *,
    start_coordinate=10.0,
    end_coordinate=20.0,
    allow_wrap=False,
):
    return pd.Series(
        {
            "id": "long_hill",
            "kind": "interval",
            "allow_start_finish_wrap": allow_wrap,
            "start_latitude_deg": start_coordinate,
            "start_longitude_deg": 0.0,
            "start_horizontal_uncertainty_m": 5.0,
            "start_source": "start coordinate",
            "end_latitude_deg": end_coordinate,
            "end_longitude_deg": 0.0,
            "end_horizontal_uncertainty_m": 5.0,
            "end_source": "end coordinate",
        }
    )


def _settings():
    return SimpleNamespace(maximum_map_error_m=20.0)


def test_geometric_fit_beats_candidate_nearest_anchor_s():
    centreline = _FakeCentreline(
        {
            10.0: [
                {"s_m": 200.0, "error_m": 0.5},
                {"s_m": 700.0, "error_m": 1.5},
            ],
            20.0: [
                {"s_m": 235.0, "error_m": 2.7},
                # This is close in s to the anchor but physically poor.
                {"s_m": 195.0, "error_m": 40.0},
            ],
        }
    )
    start, end, flags = _resolve_explicit_interval_endpoints(
        _event(),
        anchor_s_m=200.0,
        centreline=centreline,
        settings=_settings(),
    )

    assert start[0] == 0.0
    assert end[0] == 35.0
    assert end[2] == 2.7
    assert "interval_extent_implausibly_long" not in flags


def test_reversed_coordinates_are_reordered_by_course_direction():
    centreline = _FakeCentreline(
        {
            10.0: [{"s_m": 235.0, "error_m": 1.0}],
            20.0: [{"s_m": 200.0, "error_m": 1.0}],
        }
    )
    start, end, flags = _resolve_explicit_interval_endpoints(
        _event(),
        anchor_s_m=200.0,
        centreline=centreline,
        settings=_settings(),
    )

    assert end[0] - start[0] == 35.0
    assert "explicit_interval_endpoints_reordered" in flags
    assert start[4] == "end coordinate"
    assert end[4] == "start coordinate"


def test_real_start_finish_wrap_is_kept_when_near_boundary():
    centreline = _FakeCentreline(
        {
            10.0: [{"s_m": 980.0, "error_m": 1.0}],
            20.0: [{"s_m": 20.0, "error_m": 1.0}],
        }
    )
    start, end, flags = _resolve_explicit_interval_endpoints(
        _event(),
        anchor_s_m=980.0,
        centreline=centreline,
        settings=_settings(),
    )

    assert end[0] - start[0] == 40.0
    assert "interval_wraps_start_finish" in flags
    assert "explicit_interval_endpoints_reordered" not in flags


def test_single_endpoint_fallback_uses_lowest_geometric_error():
    centreline = _FakeCentreline(
        {
            20.0: [
                {"s_m": 235.0, "error_m": 2.0},
                {"s_m": 198.0, "error_m": 25.0},
            ],
        }
    )
    event = _event()
    relative, _, error, _, _ = _event_endpoint_relative_s(
        event,
        "end",
        anchor_s_m=200.0,
        centreline=centreline,
    )

    assert relative == 35.0
    assert error == 2.0
