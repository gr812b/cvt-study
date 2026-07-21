from __future__ import annotations

from dataclasses import dataclass

from cvt_track_study.studies.ensemble_v10 import (
    TrackVariant,
    _schedule_scenarios,
)
from cvt_track_study.uncertainty import SamplingPlan, ScenarioDraw


@dataclass
class _Bundle:
    active_speed_gates: tuple[dict, ...]
    data: dict


class _Sampler:
    def __init__(self, draws: tuple[ScenarioDraw, ...]) -> None:
        self._draws = draws
        self.paired_gate_identity_count = 2

    def draw_all(self) -> tuple[ScenarioDraw, ...]:
        return self._draws


def _gate(gate_id: str, offset: float) -> dict:
    samples = []
    for lap_id, speed in ((1, 8.0 + offset), (2, 9.0 + offset)):
        samples.append(
            {
                "run_id": "run_A",
                "lap_id": lap_id,
                "vehicle_id": "vehicle_A",
                "driver_id": "driver_A",
                "value_mps": speed,
            }
        )
    return {
        "id": gate_id,
        "active_by_default": True,
        "target_speed_distribution": {"samples": samples},
    }


def test_crossed_layout_replays_every_draw_on_every_track_case(tmp_path) -> None:
    variants = (
        TrackVariant(
            "nominal",
            "nominal",
            "Nominal",
            tmp_path / "nominal.json",
            _Bundle((_gate("E03", 0.0),), {"content_fingerprint_sha256": "nominal"}),
        ),
        TrackVariant(
            "windows_narrow",
            "event_windows",
            "Narrow windows",
            tmp_path / "narrow.json",
            _Bundle((_gate("E03", 0.2),), {"content_fingerprint_sha256": "narrow"}),
        ),
    )
    draws = tuple(
        ScenarioDraw(
            replicate=index,
            seed=100 + index,
            sampling_mode="all_declared",
            quantity_values_si={"drivetrain.efficiency": 0.8 + 0.05 * index},
            choice_values={},
        )
        for index in range(3)
    )
    samplers = {variant.case_id: _Sampler(draws) for variant in variants}
    plan = SamplingPlan(
        mode="all_declared",
        replicates=3,
        random_seed=123,
        gate_sampling="paired_lap",
    )

    scheduled, metadata = _schedule_scenarios(
        variants=variants,
        samplers=samplers,
        plan=plan,
        replicates=3,
        sampling_layout="cross_track_cases",
    )

    assert len(scheduled) == 6
    assert metadata["base_draw_count"] == 3
    assert metadata["scenarios_per_track_case"] == 3
    assert metadata["track_case_pairing_complete"] is True
    assert len({item.scenario.replicate for item in scheduled}) == 6

    for base_draw_id in range(3):
        pair = [item for item in scheduled if item.base_draw_id == base_draw_id]
        assert {item.variant.case_id for item in pair} == {"nominal", "windows_narrow"}
        assert len({item.scenario.seed for item in pair}) == 1
        assert len({tuple(item.scenario.quantity_values_si.items()) for item in pair}) == 1
        assert len({item.scenario.gate_sample_identity.key for item in pair}) == 1
        # The identity is held fixed, but each bundle supplies its own measured value.
        assert pair[0].scenario.gate_target_speeds_mps["E03"] != pair[1].scenario.gate_target_speeds_mps["E03"]
