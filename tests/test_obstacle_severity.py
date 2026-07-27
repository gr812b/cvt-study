from __future__ import annotations

from dataclasses import dataclass

import pytest

from cvt_track_study.simulation.obstacle_severity import scale_runtime_track
from cvt_track_study.simulation.obstacles import ObstacleContext, ObstacleEffect


class _BaseObstacle:
    model_type = "test"

    def evaluate(self, context: ObstacleContext) -> ObstacleEffect:
        return ObstacleEffect(
            resistance_force_n=100.0,
            elevation_offset_m=0.2,
            grade_slope_addition=0.1,
            normal_load_scale=0.8,
            friction_multiplier=0.7,
        )


@dataclass(frozen=True)
class _Feature:
    model: object


@dataclass(frozen=True)
class _Track:
    features: tuple[_Feature, ...]


def test_global_severity_scales_only_dissipative_obstacle_work():
    track = _Track((_Feature(_BaseObstacle()),))
    realized = scale_runtime_track(track, 0.6)
    effect = realized.features[0].model.evaluate(
        ObstacleContext(
            local_distance_m=0.5,
            interval_length_m=1.0,
            vehicle_speed_mps=3.0,
            entry_speed_mps=3.0,
            vehicle_mass_kg=250.0,
            gravity_mps2=9.81,
        )
    )
    assert effect.resistance_force_n == pytest.approx(60.0)
    assert effect.elevation_offset_m == pytest.approx(0.2)
    assert effect.grade_slope_addition == pytest.approx(0.1)
    assert effect.normal_load_scale == pytest.approx(0.8)
    assert effect.friction_multiplier == pytest.approx(0.7)
