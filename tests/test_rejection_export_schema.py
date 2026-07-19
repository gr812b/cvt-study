from __future__ import annotations

import pandas as pd

from cvt_track_study.gpx.cleanup import _empty_rejections
from cvt_track_study.track.export import (
    _coalesce_duplicate_columns,
    _concat_nonempty,
)


def test_post_map_rejection_schema_is_unique() -> None:
    matched = pd.DataFrame(
        {
            "lap_id": [1],
            "point_index": [8],
            "map_error_m": [75.0],
        }
    )
    rejected = _empty_rejections(matched)

    assert rejected.columns.is_unique
    assert list(rejected.columns).count("map_error_m") == 1


def test_exporter_coalesces_legacy_duplicate_columns() -> None:
    legacy = pd.DataFrame(
        [[1, 75.0, None], [2, None, 42.0]],
        columns=["point_index", "map_error_m", "map_error_m"],
    )

    cleaned = _coalesce_duplicate_columns(legacy)

    assert cleaned.columns.is_unique
    assert cleaned["map_error_m"].tolist() == [75.0, 42.0]


def test_pre_and_post_rejection_tables_can_be_concatenated() -> None:
    pre = pd.DataFrame(
        {
            "point_index": [3],
            "rejection_stage": ["pre_lap_physical_continuity"],
            "map_error_m": [None],
        }
    )
    legacy_post = pd.DataFrame(
        [[8, "post_map_centreline_consistency", 75.0, 75.0]],
        columns=[
            "point_index",
            "rejection_stage",
            "map_error_m",
            "map_error_m",
        ],
    )

    combined = _concat_nonempty([pre, legacy_post])

    assert len(combined) == 2
    assert combined.columns.is_unique
    assert combined.loc[1, "map_error_m"] == 75.0
