from dataclasses import dataclass
import multiprocessing as mp

from cvt_track_study.runtime.process_pool import interruptible_process_pool
from cvt_track_study.simulation.integrator import _record_ordered_feature_entry_crossings


@dataclass(frozen=True)
class _Interval:
    start_s_m: float


@dataclass(frozen=True)
class _Feature:
    identifier: str
    interval: _Interval


def _square(value: int) -> int:
    return value * value


def test_ordered_feature_crossings_preserve_interpolated_entry_speed():
    features = (
        _Feature("a", _Interval(2.0)),
        _Feature("b", _Interval(5.0)),
        _Feature("c", _Interval(12.0)),
    )
    recorded = {}
    index = _record_ordered_feature_entry_crossings(
        features=features,
        start_index=0,
        recorded=recorded,
        start_distance_m=0.0,
        end_distance_m=6.0,
        start_speed_mps=3.0,
        end_speed_mps=9.0,
    )
    assert index == 2
    assert recorded == {"a": 5.0, "b": 8.0}

    index = _record_ordered_feature_entry_crossings(
        features=features,
        start_index=index,
        recorded=recorded,
        start_distance_m=6.0,
        end_distance_m=13.0,
        start_speed_mps=9.0,
        end_speed_mps=16.0,
    )
    assert index == 3
    assert recorded["c"] == 15.0


def test_interruptible_process_pool_executes_picklable_work():
    with interruptible_process_pool(max_workers=2, mp_context=mp.get_context("spawn")) as executor:
        values = list(executor.map(_square, range(5)))
    assert values == [0, 1, 4, 9, 16]
