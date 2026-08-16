import json

import numpy as np
import pandas as pd

from cvt_track_study.reports.design_paired import paired_design_contrasts


def _rows():
    records = []
    for replicate, world_penalty in enumerate((0.0, 10.0, -5.0)):
        for max_ratio in (3.0, 3.5):
            for final_drive in (5.0, 5.5):
                lap = (
                    100.0
                    + world_penalty
                    - 4.0 * (max_ratio - 3.0)
                    - 2.0 * (final_drive - 5.0)
                )
                design_id = f"r{max_ratio}-f{final_drive}"
                records.append(
                    {
                        "replicate": replicate,
                        "scenario_seed": 100 + replicate,
                        "design_id": design_id,
                        "design_values_json": json.dumps(
                            {
                                "drivetrain.cvt.maximum_reduction_ratio": max_ratio,
                                "drivetrain.final_drive_ratio": final_drive,
                            }
                        ),
                        "design::drivetrain.cvt.maximum_reduction_ratio": max_ratio,
                        "design::drivetrain.final_drive_ratio": final_drive,
                        "bounded_completed": True,
                        "bounded_lap_time_s": lap,
                    }
                )
    return pd.DataFrame(records)


def test_paired_contrasts_cancel_common_world_difficulty():
    rows = _rows()
    manifest = {
        "design_variable_paths": [
            "drivetrain.cvt.maximum_reduction_ratio",
            "drivetrain.final_drive_ratio",
        ],
        "random_seed": 7,
        "bootstrap_resamples": 400,
    }
    all_pairs, isolated, deltas = paired_design_contrasts(rows, manifest)

    assert len(all_pairs) == 6
    assert len(isolated) == 4
    assert len(deltas) == 18

    max_ratio = isolated[
        isolated["changed_parameter"]
        == "drivetrain.cvt.maximum_reduction_ratio"
    ]
    assert len(max_ratio) == 2
    assert np.allclose(max_ratio["time_saved_mean_s"], 2.0)
    assert np.allclose(max_ratio["time_saved_ci_low_s"], 2.0)
    assert np.allclose(max_ratio["time_saved_ci_high_s"], 2.0)
    assert (max_ratio["effect_direction_95pct"] == "to design faster").all()
    assert np.allclose(max_ratio["to_faster_world_fraction"], 1.0)
    assert np.allclose(max_ratio["paired_t_p_value"], 0.0)

    final_drive = isolated[
        isolated["changed_parameter"] == "drivetrain.final_drive_ratio"
    ]
    assert len(final_drive) == 2
    assert np.allclose(final_drive["time_saved_mean_s"], 1.0)


def test_incomplete_worlds_are_reported_but_not_used_as_lap_time_pairs():
    rows = _rows()
    mask = (rows["replicate"] == 1) & (rows["design_id"] == "r3.5-f5.0")
    rows.loc[mask, "bounded_completed"] = False
    manifest = {
        "design_variable_paths": [
            "drivetrain.cvt.maximum_reduction_ratio",
            "drivetrain.final_drive_ratio",
        ],
        "bootstrap_resamples": 200,
    }
    _, isolated, _ = paired_design_contrasts(rows, manifest)
    contrast = isolated[
        (isolated["changed_parameter"] == "drivetrain.cvt.maximum_reduction_ratio")
        & (isolated["fixed_context_json"].str.contains("5.0"))
    ].iloc[0]
    assert contrast["common_world_count"] == 3
    assert contrast["both_completed_world_count"] == 2
    assert contrast["from_only_completed_world_count"] == 1
