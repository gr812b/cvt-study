"""Public data structures for versioned track bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


TRACK_BUNDLE_FORMAT = "cvt-track-bundle"
CURRENT_TRACK_BUNDLE_SCHEMA = "1.2.1"


class TrackBundleError(RuntimeError):
    """Raised when a bundle cannot be read or does not satisfy its contract."""


@dataclass(frozen=True)
class TrackBundle:
    """A validated, self-contained track bundle.

    ``data`` is intentionally exposed as a mapping so later simulation phases can
    consume the contract without importing reconstruction internals or pandas.
    """

    data: Mapping[str, Any]
    path: Path | None = None
    sha256: str | None = None

    @property
    def schema_version(self) -> str:
        return str(self.data["schema_version"])

    @property
    def track_length_m(self) -> float:
        return float(self.data["simulation_contract"]["track_length_m"])

    @property
    def active_speed_gates(self) -> tuple[Mapping[str, Any], ...]:
        gates = self.data["simulation_contract"]["speed_gates"]
        return tuple(gate for gate in gates if bool(gate["active_by_default"]))

    @property
    def physical_features(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.data["simulation_contract"]["physical_features"])

    @property
    def response_groups(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.data["simulation_contract"]["response_groups"])
