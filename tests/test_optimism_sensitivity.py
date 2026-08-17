from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cvt_track_study.bundle.model import TrackBundle
from cvt_track_study.simulation.integrator import _pace_envelope_ceiling
from cvt_track_study.studies.optimism_sensitivity import _build_envelopes, _report_html


def test_pace_envelope_is_cyclic_and_optional() -> None:
    assert np.isinf(_pace_envelope_ceiling(None, 25.0, 100.0))
    positions = np.array([0.0, 50.0, 100.0])
    speeds = np.array([10.0, 20.0, 10.0])
    assert np.isclose(_pace_envelope_ceiling((positions, speeds), 25.0, 100.0), 15.0)
    assert np.isclose(_pace_envelope_ceiling((positions, speeds), 125.0, 100.0), 15.0)


def test_source_profiles_are_built_per_vehicle_before_combining(tmp_path: Path) -> None:
    track = tmp_path / "track"
    track.mkdir()
    laps = pd.DataFrame(
        [
            {"lap_id": 1, "run_id": "a", "vehicle_id": "mcmaster", "duration_s": 200.0, "analysis_valid": True, "use_for_gate_evidence": True},
            {"lap_id": 2, "run_id": "b", "vehicle_id": "cornell", "duration_s": 180.0, "analysis_valid": True, "use_for_gate_evidence": True},
        ]
    )
    laps.to_csv(track / "lap_quality.csv", index=False)
    rows = []
    for lap_id, run_id, vehicle_id, speed in ((1, "a", "mcmaster", 5.0), (2, "b", "cornell", 10.0)):
        for i, s in enumerate(np.linspace(0.0, 95.0, 20)):
            rows.append(
                {
                    "lap_id": lap_id,
                    "run_id": run_id,
                    "vehicle_id": vehicle_id,
                    "s_m": s,
                    "speed_analysis_mps": speed,
                    "elapsed_lap_s": float(i),
                }
            )
    pd.DataFrame(rows).to_csv(track / "map_matched_points.csv", index=False)
    bundle = TrackBundle(
        data={
            "schema_version": "1.2.1",
            "simulation_contract": {"track_length_m": 100.0},
        }
    )
    frame, source_hist, meta = _build_envelopes(tmp_path, bundle, spacing_m=5.0, smoothing_m=15.0)
    assert {"mcmaster", "cornell"} <= set(source_hist)
    assert set(meta["vehicle_id"]) == {"mcmaster", "cornell"}
    # The upper demonstrated source pace should follow the faster Cornell source,
    # not a lap-count-weighted pooled average.
    assert np.allclose(frame["source_upper_p95_mps"], 10.0)
    assert np.all(frame["loose_ceiling_mps"] > frame["moderate_ceiling_mps"])
    assert np.all(frame["moderate_ceiling_mps"] > frame["strong_ceiling_mps"])


def test_histogram_selector_is_design_selector() -> None:
    ranking = pd.DataFrame(
        [
            {"level": "control", "design_id": "A", "world_count": 2, "completion_fraction": 1.0, "paired_regret_median_s": 0.0, "lap_time_median_s": 10.0},
            {"level": "strong", "design_id": "A", "world_count": 2, "completion_fraction": 1.0, "paired_regret_median_s": 0.0, "lap_time_median_s": 20.0},
        ]
    )
    winners = pd.DataFrame(
        [
            {"level": "control", "preferred_design_id": "A", "paired_regret_median_s": 0.0, "lap_time_median_s": 10.0},
            {"level": "strong", "preferred_design_id": "A", "paired_regret_median_s": 0.0, "lap_time_median_s": 20.0},
        ]
    )
    fidelity = pd.DataFrame(
        [{"maximum_abs_control_minus_source_s": 0.1, "maximum_abs_relative_numerical_bias_s": 0.01}]
    )
    source_meta = pd.DataFrame([{"vehicle_id": "mcmaster", "valid_lap_count": 2}])
    document = _report_html(
        ranking=ranking,
        interaction=pd.DataFrame(),
        winners=winners,
        figures=[],
        histogram_views=[{"design_id": "A", "path": "/does/not/exist.png", "caption": "x"}],
        source_meta=source_meta,
        histogram_replicate=1,
        fidelity=fidelity,
        integration_step_ms=5.0,
    )
    assert '<option value="hist-0">A</option>' in document
    assert "Design candidate:" in document
    assert "traffic option" not in document.lower()
