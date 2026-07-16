"""Local-map geometry and ordered projection helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class LocalFrame:
    latitude_origin_deg: float
    longitude_origin_deg: float

    def to_xy(
        self, latitude_deg: Iterable[float], longitude_deg: Iterable[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        lat = np.asarray(latitude_deg, dtype=float)
        lon = np.asarray(longitude_deg, dtype=float)
        lat0 = math.radians(self.latitude_origin_deg)
        x = np.radians(lon - self.longitude_origin_deg) * EARTH_RADIUS_M * math.cos(lat0)
        y = np.radians(lat - self.latitude_origin_deg) * EARTH_RADIUS_M
        return x, y

    def to_latlon(
        self, x_m: Iterable[float], y_m: Iterable[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x_m, dtype=float)
        y = np.asarray(y_m, dtype=float)
        lat0 = math.radians(self.latitude_origin_deg)
        latitude = self.latitude_origin_deg + np.degrees(y / EARTH_RADIUS_M)
        longitude = self.longitude_origin_deg + np.degrees(
            x / (EARTH_RADIUS_M * math.cos(lat0))
        )
        return latitude, longitude


@dataclass
class Centreline:
    x_m: np.ndarray
    y_m: np.ndarray
    s_m: np.ndarray
    elevation_m: np.ndarray
    frame: LocalFrame

    def __post_init__(self) -> None:
        if len(self.x_m) < 3 or len(self.x_m) != len(self.y_m):
            raise ValueError("Centreline requires at least three paired x/y nodes.")
        if len(self.s_m) != len(self.x_m):
            raise ValueError("Centreline s array must match node count.")
        self._ax = self.x_m[:-1]
        self._ay = self.y_m[:-1]
        self._dx = np.diff(self.x_m)
        self._dy = np.diff(self.y_m)
        self._segment_length = np.hypot(self._dx, self._dy)
        self._segment_length_sq = np.maximum(self._segment_length**2, 1e-12)
        self.length_m = float(self.s_m[-1])

    def all_projections(
        self, point_x_m: float, point_y_m: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t = (
            (point_x_m - self._ax) * self._dx
            + (point_y_m - self._ay) * self._dy
        ) / self._segment_length_sq
        t = np.clip(t, 0.0, 1.0)
        qx = self._ax + t * self._dx
        qy = self._ay + t * self._dy
        distance = np.hypot(point_x_m - qx, point_y_m - qy)
        s = self.s_m[:-1] + t * self._segment_length
        return s, distance, qx, qy

    def project_with_progress(
        self, point_x_m: float, point_y_m: float, progress_guess_m: float
    ) -> tuple[float, float, float, float]:
        raw_s, distance, qx, qy = self.all_projections(point_x_m, point_y_m)
        unwrapped = raw_s + np.round((progress_guess_m - raw_s) / self.length_m) * self.length_m
        progress_error = unwrapped - progress_guess_m
        score = distance**2 + (0.08 * progress_error) ** 2
        best = int(np.argmin(score))
        return (
            float(unwrapped[best]),
            float(distance[best]),
            float(qx[best]),
            float(qy[best]),
        )

    def distinct_candidates(
        self,
        point_x_m: float,
        point_y_m: float,
        *,
        count: int = 12,
        separation_m: float = 18.0,
    ) -> list[dict[str, float]]:
        s, distance, qx, qy = self.all_projections(point_x_m, point_y_m)
        candidates: list[dict[str, float]] = []
        for index in np.argsort(distance):
            candidate_s = float(s[index])
            if candidates:
                circular = [
                    min(
                        abs(candidate_s - item["s_m"]),
                        self.length_m - abs(candidate_s - item["s_m"]),
                    )
                    for item in candidates
                ]
                if min(circular) < separation_m:
                    continue
            candidates.append(
                {
                    "s_m": candidate_s,
                    "error_m": float(distance[index]),
                    "x_m": float(qx[index]),
                    "y_m": float(qy[index]),
                }
            )
            if len(candidates) >= count:
                break
        return candidates
