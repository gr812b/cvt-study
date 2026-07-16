"""Extension contracts for future drivetrain and tire implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@runtime_checkable
class DrivetrainAdapter(Protocol):
    """Map boundary state and demand to wheel force plus diagnostic channels."""

    identifier: str

    def evaluate(self, state: Mapping[str, float], demand: Mapping[str, float]) -> Mapping[str, float]: ...


@runtime_checkable
class TireForceAdapter(Protocol):
    """Map tire state and normal load to longitudinal force and loss."""

    identifier: str

    def evaluate(self, state: Mapping[str, float], normal_load_n: float) -> Mapping[str, float]: ...


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    identifier: str
    family: str
    factory: Callable[..., Any]
    description: str


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[tuple[str, str], ModelRegistration] = {}

    def register(self, registration: ModelRegistration) -> None:
        key = (registration.family, registration.identifier)
        if key in self._models:
            raise ValueError(f"Model already registered: {registration.family}/{registration.identifier}")
        self._models[key] = registration

    def resolve(self, family: str, identifier: str) -> ModelRegistration:
        try:
            return self._models[(family, identifier)]
        except KeyError as exc:
            available = sorted(name for fam, name in self._models if fam == family)
            raise KeyError(
                f"Unknown {family} model {identifier!r}; available: {available}"
            ) from exc

    def describe(self) -> tuple[ModelRegistration, ...]:
        return tuple(self._models[key] for key in sorted(self._models))
