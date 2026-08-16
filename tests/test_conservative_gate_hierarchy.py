from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import pytest

from cvt_track_study.bundle.model import TrackBundle
from cvt_track_study.config.diagnostics import DiagnosticBag
from cvt_track_study.simulation.track import RuntimeSpeedGate, RuntimeTrack
from cvt_track_study.track.gates import score_speed_gates
from cvt_track_study.track.settings import ReconstructionSettings
from cvt_track_study.uncertainty.sampling import _gate_samples


def _settings() -> ReconstructionSettings:
    return ReconstructionSettings.from_mapping(
        {
            "reconstruction": {"maximum_reasonable_speed_mps": 25.0},
            "gate_confidence": {
                "minimum_valid_passes": 5,
                "target_pass_count": 10,
                "repeatability_scale_mps": 2.0,
                "vehicle_agreement_scale_mps": 2.0,
                "accept_score": 60.0,
                "review_score": 40.0,
            },
        }
    )


def _event() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "E1",
                "name": "Feature",
                "sequence": 1,
                "response_group_id": "E1",
                "gate_candidate": True,
                "feature_start_effective_error_m": 1.0,
                "anchor_projection_error_m": 1.0,
                "review_flags": "",
            }
        ]
    )


def _pass_rows(
    *,
    vehicle: str,
    count: int,
    entry: float,
    approach: float,
    response_ratio: float,
) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        # Tiny symmetric perturbations retain strong within-source repeatability.
        perturb = 0.03 * ((index % 3) - 1)
        rows.append(
            {
                "event_id": "E1",
                "eligible": True,
                "vehicle_id": vehicle,
                "entry_speed_mps": entry + perturb,
                "lap_median_speed_mps": approach - 2.0 + perturb,
                "approach_speed_mps": approach + perturb,
                "event_min_speed_mps": response_ratio * (approach + perturb),
                "event_min_rel_m": 8.0 + 0.1 * ((index % 3) - 1),
                "recovery_distance_m": 15.0,
                "braking_drop_mps": 1.5,
            }
        )
    return rows


def test_relative_response_can_qualify_when_absolute_entry_speed_does_not():
    # Both vehicles react to the same feature by the same fraction, but Cornell's
    # absolute pace is much higher.  This should *not* manufacture a universal
    # hard speed, while still retaining the well-supported physical response.
    passes = pd.DataFrame(
        _pass_rows(
            vehicle="mcmaster", count=6, entry=6.0, approach=10.0, response_ratio=0.60
        )
        + _pass_rows(
            vehicle="cornell", count=6, entry=10.0, approach=14.0, response_ratio=0.60
        )
    )
    row = score_speed_gates(passes, _event(), _settings(), DiagnosticBag()).iloc[0]
    assert not bool(row["hard_absolute_gate_qualified"])
    assert bool(row["sustained_gate_qualified"])
    assert row["enforcement_class"] == "guardrail_plus_relative_response"
    assert row["recommendation"] != "accepted"


def test_hard_absolute_requires_each_measured_source_to_have_support():
    # A large source is not allowed to drown out an under-sampled second vehicle.
    passes = pd.DataFrame(
        _pass_rows(
            vehicle="mcmaster", count=20, entry=6.0, approach=10.0, response_ratio=0.60
        )
        + _pass_rows(
            vehicle="cornell", count=2, entry=6.1, approach=10.1, response_ratio=0.60
        )
    )
    row = score_speed_gates(passes, _event(), _settings(), DiagnosticBag()).iloc[0]
    assert int(row["minimum_source_pass_count"]) == 2
    assert not bool(row["hard_absolute_gate_qualified"])
    assert row["recommendation"] != "accepted"


def test_guardrail_uses_source_upper_envelope_not_lap_count_weighting():
    # Many slower McMaster laps cannot pull the safety guardrail below the smaller
    # faster Cornell source.  Disagreement makes this non-fitted cap looser.
    passes = pd.DataFrame(
        _pass_rows(
            vehicle="mcmaster", count=50, entry=5.0, approach=9.0, response_ratio=0.80
        )
        + _pass_rows(
            vehicle="cornell", count=5, entry=12.0, approach=16.0, response_ratio=0.80
        )
    )
    row = score_speed_gates(passes, _event(), _settings(), DiagnosticBag()).iloc[0]
    assert not bool(row["hard_absolute_gate_qualified"])
    assert float(row["guardrail_cap_mps"]) > 13.0
    assert float(row["guardrail_cap_mps"]) <= 25.0


def _runtime_track(gates: tuple[RuntimeSpeedGate, ...]) -> RuntimeTrack:
    return RuntimeTrack(
        name="test",
        length_m=100.0,
        closed_course=True,
        centreline_s_m=(0.0, 100.0),
        centreline_x_m=(0.0, 100.0),
        centreline_y_m=(0.0, 0.0),
        centreline_curvature_1_per_m=(0.0, 0.0),
        reference_elevation_m=(None, None),
        surface_friction_coefficient=0.7,
        features=(),
        speed_gates=gates,
        global_speed_guardrail_mps=25.0,
        gpx_grade_force_enabled=False,
    )


def test_global_guardrail_prevents_an_uncapped_track():
    track = _runtime_track(())
    assert track.safe_speed_ceiling_mps(37.0, braking_deceleration_mps2=5.0) == 25.0


def test_fast_cyclic_braking_envelope_matches_brute_force():
    gates = tuple(
        RuntimeSpeedGate(
            identifier=f"g{index}",
            response_group_id=f"E{index}",
            name=f"gate {index}",
            position_s_m=position,
            target_speed_mps=target,
            confidence_score=100.0,
        )
        for index, (position, target) in enumerate(
            ((5.0, 6.0), (22.0, 9.0), (22.0, 7.0), (61.0, 4.0), (88.0, 11.0))
        )
    )
    track = _runtime_track(gates)
    positions = np.asarray([gate.position_s_m for gate in gates], dtype=float)
    targets_squared = np.asarray(
        [gate.target_speed_mps**2 for gate in gates], dtype=float
    )
    rng = random.Random(20260816)
    for _ in range(5000):
        distance = rng.random() * track.length_m
        deceleration = 10.0 ** rng.uniform(-4.0, 1.2)
        brute_squared = float(
            np.min(
                targets_squared
                + 2.0
                * deceleration
                * np.mod(positions - distance, track.length_m)
            )
        )
        brute = min(25.0, math.sqrt(max(brute_squared, 0.0)))
        fast = track.safe_speed_ceiling_mps(
            distance, braking_deceleration_mps2=deceleration
        )
        assert fast == pytest.approx(brute, abs=1.0e-11)


def test_gate_sampling_prefers_target_vehicle_empirical_speeds():
    bundle = TrackBundle(
        {
            "schema_version": "test",
            "simulation_contract": {
                "speed_gates": [
                    {
                        "id": "gate:E1:entry",
                        "active_by_default": True,
                        "target_speed_distribution": {
                            "samples": [
                                {
                                    "run_id": "m1",
                                    "lap_id": 1,
                                    "vehicle_id": "mcmaster",
                                    "driver_id": "m",
                                    "value_mps": 6.0,
                                },
                                {
                                    "run_id": "c1",
                                    "lap_id": 1,
                                    "vehicle_id": "cornell",
                                    "driver_id": "c",
                                    "value_mps": 10.0,
                                },
                            ]
                        },
                    }
                ]
            },
        }
    )
    sampled = _gate_samples(bundle, target_vehicle_id="mcmaster")
    identities = next(iter(sampled.values()))
    assert identities
    assert {identity[2] for identity in identities} == {"mcmaster"}
