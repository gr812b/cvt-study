"""Low-cost elevation screen used before any spatial grade model is enabled."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def screen_grade_materiality(profile: pd.DataFrame) -> dict[str, Any]:
    """Screen whether grade is both observable and plausibly material.

    This is deliberately not a grade-force model. It prevents noisy GPS/FIT
    altitude from entering vehicle dynamics without first earning a paired
    sensitivity study.
    """

    required = profile[
        [
            "s_m",
            "median_elevation_m",
            "p10_elevation_m",
            "p90_elevation_m",
            "valid_elevation_lap_count",
        ]
    ].copy()
    finite = required["median_elevation_m"].notna()
    coverage = float(finite.mean()) if len(required) else 0.0
    base = {
        "method_version": "1.0.0",
        "grade_force_enabled": False,
        "elevation_coverage_fraction": coverage,
        "minimum_elevation_lap_count": int(
            required.loc[finite, "valid_elevation_lap_count"].min()
        )
        if finite.any()
        else 0,
    }
    if finite.sum() < 5 or coverage < 0.80:
        return {
            **base,
            "status": "insufficient_elevation_evidence",
            "spatial_grade_sensitivity_recommended": False,
            "reason": "At least 80% spatial elevation coverage is required for screening.",
        }

    selected = required.loc[finite]
    distance = selected["s_m"].to_numpy(float)
    elevation = selected["median_elevation_m"].to_numpy(float)
    spacing = float(np.median(np.diff(distance)))
    window = max(3, int(round(25.0 / max(spacing, 1.0))))
    if window % 2 == 0:
        window += 1
    smoothed = (
        pd.Series(elevation)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(float)
    )
    slope = np.gradient(smoothed, distance)
    grade_degrees = np.degrees(np.arctan(slope))
    spread = (
        selected["p90_elevation_m"].to_numpy(float)
        - selected["p10_elevation_m"].to_numpy(float)
    )
    elevation_range = float(np.max(smoothed) - np.min(smoothed))
    median_spread = float(np.nanmedian(spread))
    p95_abs_grade = float(np.quantile(np.abs(grade_degrees), 0.95))
    noisy = median_spread > max(1.0, 0.5 * elevation_range)
    plausibly_material = elevation_range >= 3.0 and p95_abs_grade >= 2.0
    if noisy:
        status = "elevation_not_repeatable"
        recommended = False
        reason = "Across-lap elevation spread is too large relative to the smoothed signal."
    elif plausibly_material:
        status = "paired_grade_sensitivity_recommended"
        recommended = True
        reason = (
            "The repeatable elevation profile is large enough to justify a paired "
            "with/without-grade decision sensitivity before grade force is enabled."
        )
    else:
        status = "grade_proxy_immaterial"
        recommended = False
        reason = "The smoothed grade proxy is below the materiality screening thresholds."
    return {
        **base,
        "status": status,
        "spatial_grade_sensitivity_recommended": recommended,
        "smoothed_elevation_range_m": elevation_range,
        "median_p10_p90_elevation_spread_m": median_spread,
        "p95_absolute_grade_degrees": p95_abs_grade,
        "smoothing_window_m": window * spacing,
        "materiality_thresholds": {
            "minimum_smoothed_elevation_range_m": 3.0,
            "minimum_p95_absolute_grade_degrees": 2.0,
        },
        "reason": reason,
    }
