"""JSON-safe conversion and closed-course helpers for bundle construction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def interval(start: float, end: float, track_length: float) -> dict[str, Any]:
    interval_length = (end - start) % track_length
    return {
        "start_s_m": start,
        "end_s_m": end,
        "length_m": interval_length,
        "wraps_start_finish": bool(end < start and interval_length > 0.0),
    }


def circular(value: float, track_length: float) -> float:
    wrapped = value % track_length
    if math.isclose(wrapped, track_length, abs_tol=1.0e-9):
        return 0.0
    return float(wrapped)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [json_safe(row) for row in frame.to_dict(orient="records")]


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return value


def optional_float(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def split_tokens(value: Any, *, separator: str = ";") -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    actual_separator = "," if separator == ";" and "," in text and ";" not in text else separator
    return [token.strip() for token in text.split(actual_separator) if token.strip()]


def undeclared_obstacle_model() -> dict[str, Any]:
    return {
        "status": "undeclared",
        "model_type": None,
        "parameters": {},
        "reason": "Obstacle equations and defaults are defined in Phase 5.",
    }
