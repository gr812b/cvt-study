from __future__ import annotations

import unittest

import numpy as np

from measured_track import build_track_from_bundle
from simulation import _implicit_tire_slip_step
from track_builder import EffectiveEnergyEvent, SpeedGate
from track_builder.core import TrackEvaluationContext


class SpeedGateTests(unittest.TestCase):
    def test_gate_is_one_way_braking_envelope(self) -> None:
        gate = SpeedGate(
            name="test",
            position_m=100.0,
            target_speed_mps=5.0,
            braking_deceleration_mps2=4.0,
            confidence_score=80.0,
        )
        self.assertAlmostEqual(gate.upstream_limit_mps(100.0), 5.0)
        self.assertAlmostEqual(gate.upstream_limit_mps(50.0) ** 2, 425.0)
        self.assertTrue(np.isinf(gate.upstream_limit_mps(100.1)))

    def test_bundle_filters_gate_confidence(self) -> None:
        bundle = {
            "track": {"length_m": 200.0},
            "speed_gates": [
                {
                    "analysis_group_id": "accepted",
                    "event_name": "accepted",
                    "gate_s_m": 50.0,
                    "target_speed_p10_kmh": 15.0,
                    "target_speed_median_kmh": 20.0,
                    "target_speed_p90_kmh": 25.0,
                    "braking_deceleration_mps2": 4.0,
                    "confidence_score": 80.0,
                    "confidence_class": "HIGH",
                },
                {
                    "analysis_group_id": "rejected",
                    "event_name": "rejected",
                    "gate_s_m": 100.0,
                    "target_speed_p10_kmh": 10.0,
                    "target_speed_median_kmh": 12.0,
                    "target_speed_p90_kmh": 14.0,
                    "braking_deceleration_mps2": 4.0,
                    "confidence_score": 40.0,
                    "confidence_class": "LOW",
                },
            ],
            "event_groups": [],
        }
        track = build_track_from_bundle(bundle, minimum_gate_confidence=60.0)
        self.assertEqual([gate.source_group_id for gate in track.speed_gates], ["accepted"])


class NumericalPhysicsTests(unittest.TestCase):
    def test_implicit_slip_solution_satisfies_residual(self) -> None:
        arguments = {
            "previous_slip_speed_mps": 1.2,
            "step_s": 0.01,
            "free_slip_acceleration_mps2": 90.0,
            "tire_force_coefficient": 0.2,
            "tire_limit_n": 1700.0,
            "tire_stiffness_n_per_mps": 1200.0,
        }
        slip, force = _implicit_tire_slip_step(**arguments)
        residual = (
            slip
            - arguments["previous_slip_speed_mps"]
            - arguments["step_s"]
            * (
                arguments["free_slip_acceleration_mps2"]
                - arguments["tire_force_coefficient"] * force
            )
        )
        self.assertLess(abs(residual), 1.0e-7)
        self.assertGreater(force, 0.0)
        self.assertLessEqual(abs(force), arguments["tire_limit_n"])

    def test_effective_event_integrates_to_requested_energy(self) -> None:
        event = EffectiveEnergyEvent(
            name="test",
            start_m=10.0,
            length_m=20.0,
            specific_energy_loss_j_per_kg=12.0,
        )
        mass = 300.0
        positions = np.linspace(10.0, 30.0, 20001, endpoint=False)
        forces = [
            event.evaluate(
                TrackEvaluationContext(
                    distance_m=float(position),
                    vehicle_speed_mps=5.0,
                    vehicle_mass_kg=mass,
                    gravity_mps2=9.80665,
                )
            ).additional_resistance_force_n
            for position in positions
        ]
        energy = np.trapezoid(forces, positions)
        self.assertAlmostEqual(energy, mass * 12.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()

class TirePresetTests(unittest.TestCase):
    def test_tire_axes_are_independent(self) -> None:
        from models import tire_model_from_levels
        low_peak = tire_model_from_levels(wheel_radius_m=0.2794, peak_traction="low", slip_buildup="medium")
        high_peak = tire_model_from_levels(wheel_radius_m=0.2794, peak_traction="high", slip_buildup="medium")
        slow_build = tire_model_from_levels(wheel_radius_m=0.2794, peak_traction="medium", slip_buildup="low")
        quick_build = tire_model_from_levels(wheel_radius_m=0.2794, peak_traction="medium", slip_buildup="high")
        self.assertLess(low_peak.peak_traction_scale, high_peak.peak_traction_scale)
        self.assertEqual(low_peak.slip_stiffness_n_per_mps, high_peak.slip_stiffness_n_per_mps)
        self.assertLess(slow_build.slip_stiffness_n_per_mps, quick_build.slip_stiffness_n_per_mps)
        self.assertEqual(slow_build.peak_traction_scale, quick_build.peak_traction_scale)
