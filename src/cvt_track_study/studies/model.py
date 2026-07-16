"""Study-runner data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class DesignPoint:
    identifier: str
    path: str | None
    display_value: float | str
    value_si: float | None
    choice_value: str | None = None
    level_probability: float | None = None
    level_kind: str = "design"
    nominal: bool = False


@dataclass(frozen=True, slots=True)
class StudyExecution:
    rows: tuple[Mapping[str, Any], ...]
    scenario_draws: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    convergence: Mapping[str, Any]
    manifest: Mapping[str, Any]
    input_contracts: Mapping[str, Any]
