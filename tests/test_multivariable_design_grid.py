from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from cvt_track_study.studies import design_grid
from cvt_track_study.studies.design_replay_v11 import _scenario_from_source
from cvt_track_study.reports.design_grid_report import _attach_design_values


class _FakeQuantity:
    def __init__(self, nominal: float) -> None:
        self.nominal = nominal
        self.unit = "1"


def _registry(monkeypatch):
    monkeypatch.setattr(design_grid, "UncertainQuantity", _FakeQuantity)
    return SimpleNamespace(
        by_path={
            "drivetrain.cvt.maximum_reduction_ratio": SimpleNamespace(
                value=_FakeQuantity(3.5)
            ),
            "drivetrain.cvt.minimum_reduction_ratio": SimpleNamespace(
                value=_FakeQuantity(0.9)
            ),
            "drivetrain.final_drive_ratio": SimpleNamespace(
                value=_FakeQuantity(7.556)
            ),
        }
    )


def test_cartesian_plan_prefers_modern_grid_over_legacy_validator_table(monkeypatch) -> None:
    raw = {
        "design_variable": {
            "path": "drivetrain.final_drive_ratio",
            "values": [5.0],
        },
        "design_variables": [
            {
                "path": "drivetrain.cvt.maximum_reduction_ratio",
                "values": [3.0, 3.5],
            },
            {
                "path": "drivetrain.final_drive_ratio",
                "values": [5.0, 5.5, 6.0],
            },
        ],
        "design_grid": {"maximum_candidates": 10},
        "sampling": {"mode": "all_declared", "replicates": 12},
    }
    points, mode, replicates = design_grid.design_sweep_plan(
        raw, _registry(monkeypatch), None
    )
    assert mode == "all_declared"
    assert replicates == 12
    assert len(points) == 6
    assert design_grid.design_paths(points) == (
        "drivetrain.cvt.maximum_reduction_ratio",
        "drivetrain.final_drive_ratio",
    )


def test_replayed_world_removes_all_design_controlled_paths() -> None:
    record = {
        "replicate": 91,
        "seed": 123,
        "sampling_mode": "all_declared",
        "quantity_values_si": {
            "drivetrain.cvt.maximum_reduction_ratio": 3.8,
            "drivetrain.final_drive_ratio": 7.556,
            "drivetrain.efficiency": 0.8,
        },
        "choice_values": {},
        "gate_target_speeds_mps": {"gate:E01": 4.2},
        "independently_sampled_gate_ids": [],
    }
    scenario = _scenario_from_source(
        record,
        new_replicate=0,
        design_paths=(
            "drivetrain.cvt.maximum_reduction_ratio",
            "drivetrain.final_drive_ratio",
        ),
    )
    assert scenario.quantity_values_si == {"drivetrain.efficiency": 0.8}


def test_grid_report_can_recover_both_axes() -> None:
    paths = (
        "drivetrain.cvt.maximum_reduction_ratio",
        "drivetrain.final_drive_ratio",
    )
    rows = pd.DataFrame(
        [
            {
                "design_id": "cvt_max=3 | final_drive=5",
                "design_values_json": json.dumps({paths[0]: 3.0, paths[1]: 5.0}),
            }
        ]
    )
    ranking = pd.DataFrame(
        [
            {
                "design_id": "cvt_max=3 | final_drive=5",
                "lap_time_median_s": 140.0,
            }
        ]
    )
    summary = _attach_design_values(ranking, rows, paths)
    assert summary.loc[0, paths[0]] == 3.0
    assert summary.loc[0, paths[1]] == 5.0
