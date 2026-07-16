from __future__ import annotations

import numpy as np
import pandas as pd

from cvt_track_study.track.grade import screen_grade_materiality


def _profile(elevation_m: np.ndarray, *, spread_m: float = 0.4) -> pd.DataFrame:
    distance_m = np.linspace(0.0, 100.0, len(elevation_m))
    return pd.DataFrame(
        {
            "s_m": distance_m,
            "median_elevation_m": elevation_m,
            "p10_elevation_m": elevation_m - 0.5 * spread_m,
            "p90_elevation_m": elevation_m + 0.5 * spread_m,
            "valid_elevation_lap_count": np.full(len(elevation_m), 6),
        }
    )


def test_grade_screen_recommends_paired_sensitivity_for_repeatable_signal() -> None:
    distance_m = np.linspace(0.0, 100.0, 41)
    profile = _profile(4.0 * np.sin(2.0 * np.pi * distance_m / 100.0))

    result = screen_grade_materiality(profile)

    assert result["status"] == "paired_grade_sensitivity_recommended"
    assert result["spatial_grade_sensitivity_recommended"] is True
    assert result["grade_force_enabled"] is False


def test_grade_screen_rejects_nonrepeatable_elevation() -> None:
    distance_m = np.linspace(0.0, 100.0, 41)
    profile = _profile(
        4.0 * np.sin(2.0 * np.pi * distance_m / 100.0), spread_m=12.0
    )

    result = screen_grade_materiality(profile)

    assert result["status"] == "elevation_not_repeatable"
    assert result["spatial_grade_sensitivity_recommended"] is False
    assert result["grade_force_enabled"] is False


def test_grade_screen_does_not_promote_missing_elevation() -> None:
    profile = _profile(np.full(21, np.nan))

    result = screen_grade_materiality(profile)

    assert result["status"] == "insufficient_elevation_evidence"
    assert result["spatial_grade_sensitivity_recommended"] is False
    assert result["grade_force_enabled"] is False
