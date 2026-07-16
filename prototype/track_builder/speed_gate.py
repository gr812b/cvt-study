from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True, slots=True)
class SpeedGate:
    """Measured driver/geometry control state enforced through upstream braking.

    This is a terminal speed ceiling at one position, not a speed limit over an
    entire section. A vehicle below the envelope is never accelerated upward.
    """

    name: str
    position_m: float
    target_speed_mps: float
    braking_deceleration_mps2: float = 4.0
    confidence_score: float = 0.0
    confidence_class: str = "UNSPECIFIED"
    source_group_id: str = ""

    def __post_init__(self) -> None:
        for field_name, value in (
            ("position_m", self.position_m),
            ("target_speed_mps", self.target_speed_mps),
            ("braking_deceleration_mps2", self.braking_deceleration_mps2),
            ("confidence_score", self.confidence_score),
        ):
            if not isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.position_m < 0.0 or self.target_speed_mps < 0.0:
            raise ValueError("speed-gate position and target speed must be non-negative")
        if self.braking_deceleration_mps2 <= 0.0:
            raise ValueError("speed-gate braking deceleration must be positive")
        if not 0.0 <= self.confidence_score <= 100.0:
            raise ValueError("speed-gate confidence_score must lie in [0, 100]")

    def upstream_limit_mps(self, distance_m: float) -> float:
        if distance_m > self.position_m:
            return float("inf")
        remaining = self.position_m - distance_m
        return sqrt(
            self.target_speed_mps**2
            + 2.0 * self.braking_deceleration_mps2 * remaining
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "position_m": self.position_m,
            "target_speed_mps": self.target_speed_mps,
            "braking_deceleration_mps2": self.braking_deceleration_mps2,
            "confidence_score": self.confidence_score,
            "confidence_class": self.confidence_class,
            "source_group_id": self.source_group_id,
        }
