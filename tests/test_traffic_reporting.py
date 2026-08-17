from __future__ import annotations

from pathlib import Path

import pandas as pd

from cvt_track_study.reports.traffic import (
    _design_traffic_stratified_summary,
    _design_traffic_winners,
    _design_world_performance,
    _independent_traffic_world_summary,
    _study_traffic_impact,
    _traffic_severity_groups,
)


def test_independent_traffic_world_summary_collapses_crossed_track_cases() -> None:
    rows = pd.DataFrame(
        [
            {
                "base_draw_id": draw,
                "track_case_id": track,
                "design_id": "nominal",
                "reference_traffic_penalty_s": penalty + jitter,
                "reference_traffic_active_time_s": active,
                "reference_traffic_event_count": events,
                "lap_time_penalty_vs_infinite_s": finite + 0.01 * jitter,
            }
            for draw, penalty, active, events, finite in (
                (0, 2.0, 4.0, 1.0, 0.4),
                (1, 12.0, 20.0, 4.0, 0.2),
            )
            for track, jitter in (("nominal", 0.0), ("alt_a", -0.2), ("alt_b", 0.2))
        ]
    )

    worlds = _independent_traffic_world_summary(rows)

    assert len(worlds) == 2
    assert worlds["crossed_row_count"].tolist() == [3, 3]
    assert worlds["track_case_count"].tolist() == [3, 3]
    assert worlds["reference_traffic_penalty_s"].tolist() == [2.0, 12.0]
    assert worlds["finite_ratio_penalty_median_s"].tolist() == [0.4, 0.2]

    impact = _study_traffic_impact(worlds)
    traffic = impact[impact["metric"] == "reference_traffic_penalty_s"].iloc[0]
    assert int(traffic["independent_world_count"]) == 2


def test_traffic_severity_is_based_on_independent_worlds() -> None:
    worlds = pd.DataFrame(
        {
            "base_draw_id": list(range(9)),
            "reference_traffic_penalty_s": [0, 1, 2, 3, 4, 5, 10, 20, 30],
        }
    )
    labels = _traffic_severity_groups(worlds)
    assert labels.value_counts().to_dict() == {"low": 3, "medium": 3, "high": 3}


def test_design_traffic_winner_can_change_with_traffic_without_pseudoreplication() -> None:
    rows = []
    # Three independent worlds, each crossed with two track reconstructions.
    # A is better in low/medium traffic; B becomes better in high traffic.
    for draw, traffic_penalty in ((0, 0.0), (1, 5.0), (2, 20.0)):
        for track in ("nominal", "alt"):
            reference = 100.0 + traffic_penalty
            rows.append(
                {
                    "base_draw_id": draw,
                    "track_case_id": track,
                    "design_id": "A",
                    "bounded_completed": True,
                    "bounded_lap_time_s": reference + 0.20,
                    "lap_time_penalty_vs_infinite_s": 0.20,
                    "finite_ratio_opportunity_loss_energy_kj": 10.0,
                    "reference_traffic_penalty_s": traffic_penalty,
                }
            )
            rows.append(
                {
                    "base_draw_id": draw,
                    "track_case_id": track,
                    "design_id": "B",
                    "bounded_completed": True,
                    "bounded_lap_time_s": reference + (0.30 if draw < 2 else 0.10),
                    "lap_time_penalty_vs_infinite_s": 0.30 if draw < 2 else 0.10,
                    "finite_ratio_opportunity_loss_energy_kj": 9.0,
                    "reference_traffic_penalty_s": traffic_penalty,
                }
            )
    frame = pd.DataFrame(rows)
    worlds = _independent_traffic_world_summary(frame)
    severity = _traffic_severity_groups(worlds)
    severity_by_draw = dict(zip(worlds["base_draw_id"], severity))
    penalties = worlds.set_index("base_draw_id")["reference_traffic_penalty_s"]

    design_world = _design_world_performance(frame, severity_by_draw, penalties)
    summary = _design_traffic_stratified_summary(design_world)
    winners = _design_traffic_winners(summary)

    assert design_world.groupby(["base_draw_id", "design_id"]).size().eq(1).all()
    winner_map = dict(zip(winners["traffic_severity"], winners["preferred_design_id"]))
    assert winner_map["low"] == "A"
    assert winner_map["medium"] == "A"
    assert winner_map["high"] == "B"
