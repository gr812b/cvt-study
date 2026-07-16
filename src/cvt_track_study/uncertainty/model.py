"""Core data structures for sampled uncertainty scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class GateSampleIdentity:
    run_id: str
    lap_id: int
    vehicle_id: str
    driver_id: str

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.run_id, self.lap_id, self.vehicle_id, self.driver_id)


@dataclass(frozen=True, slots=True)
class ScenarioDraw:
    replicate: int
    seed: int
    sampling_mode: str
    quantity_values_si: Mapping[str, float] = field(default_factory=dict)
    choice_values: Mapping[str, str] = field(default_factory=dict)
    gate_target_speeds_mps: Mapping[str, float] = field(default_factory=dict)
    gate_sample_identity: GateSampleIdentity | None = None
    independently_sampled_gate_ids: tuple[str, ...] = ()

    def serializable(self) -> dict[str, object]:
        identity = None
        if self.gate_sample_identity is not None:
            identity = {
                "run_id": self.gate_sample_identity.run_id,
                "lap_id": self.gate_sample_identity.lap_id,
                "vehicle_id": self.gate_sample_identity.vehicle_id,
                "driver_id": self.gate_sample_identity.driver_id,
            }
        return {
            "replicate": self.replicate,
            "seed": self.seed,
            "sampling_mode": self.sampling_mode,
            "quantity_values_si": dict(sorted(self.quantity_values_si.items())),
            "choice_values": dict(sorted(self.choice_values.items())),
            "gate_target_speeds_mps": dict(sorted(self.gate_target_speeds_mps.items())),
            "gate_sample_identity": identity,
            "independently_sampled_gate_ids": list(self.independently_sampled_gate_ids),
        }
