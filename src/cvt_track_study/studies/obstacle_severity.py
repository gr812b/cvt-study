"""Study-level integration for global obstacle severity and gate uncertainty."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import random
from typing import Any

from cvt_track_study.studies.model import DesignPoint


_PATH = "track.obstacle_energy_severity"


def install_study_patches() -> None:
    _install_structural_levels()
    _install_crossed_gate_measurement_error()


def _install_structural_levels() -> None:
    from . import planning

    original = planning._structural_design_points
    if getattr(original, "_global_severity_levels_patch", False):
        return

    def with_global_low_nominal_high(raw: Any, registry: Any) -> tuple[DesignPoint, ...]:
        points = list(original(raw, registry))
        indices = [index for index, point in enumerate(points) if point.path == _PATH]
        if not indices:
            return tuple(points)
        registered = registry.by_path.get(_PATH)
        value = registered.value
        uncertainty = value.uncertainty
        lower = uncertainty.lower
        upper = uncertainty.upper
        if lower is None or upper is None:
            return tuple(points)
        nominal_si = float(value.nominal_si()[0])
        scale_si = float(value.nominal_si()[0] / value.nominal) if value.nominal else 1.0
        replacement = [
            DesignPoint(
                identifier=f"{_PATH}@low",
                path=_PATH,
                display_value=float(lower),
                value_si=float(lower) * scale_si,
                level_probability=0.0,
                level_kind="low",
                nominal=False,
            ),
            DesignPoint(
                identifier=f"{_PATH}@nominal",
                path=_PATH,
                display_value=float(value.nominal),
                value_si=nominal_si,
                level_probability=None,
                level_kind="nominal",
                nominal=True,
            ),
            DesignPoint(
                identifier=f"{_PATH}@high",
                path=_PATH,
                display_value=float(upper),
                value_si=float(upper) * scale_si,
                level_probability=1.0,
                level_kind="high",
                nominal=False,
            ),
        ]
        first = indices[0]
        points = [point for point in points if point.path != _PATH]
        points[first:first] = replacement
        return tuple(points)

    with_global_low_nominal_high._global_severity_levels_patch = True
    planning._structural_design_points = with_global_low_nominal_high


def _install_crossed_gate_measurement_error() -> None:
    from . import ensemble_v10

    original = ensemble_v10._schedule_scenarios
    if getattr(original, "_gate_measurement_error_patch", False):
        return

    def scheduled_with_measurement_error(*args: Any, **kwargs: Any) -> Any:
        scheduled, metadata = original(*args, **kwargs)
        layout = kwargs.get("sampling_layout")
        if layout is None and len(args) >= 5:
            layout = args[4]
        if layout != "cross_track_cases":
            return scheduled, metadata

        updated = []
        for item in scheduled:
            scenario = item.scenario
            identity = scenario.gate_sample_identity
            if identity is None:
                updated.append(item)
                continue
            values = _gate_values_with_error(
                item.variant.bundle,
                identity.key,
                scenario.seed,
            )
            updated.append(
                replace(
                    item,
                    scenario=replace(
                        scenario,
                        gate_target_speeds_mps=values,
                    ),
                )
            )
        metadata = {
            **metadata,
            "gate_measurement_error_policy": (
                "one coherent normal deviate per measured traversal; sigma is "
                "source-specific, with reconstructed lap-time speed wider than FIT"
            ),
        }
        return tuple(updated), metadata

    scheduled_with_measurement_error._gate_measurement_error_patch = True
    ensemble_v10._schedule_scenarios = scheduled_with_measurement_error


def _gate_values_with_error(
    bundle: Any,
    identity_key: tuple[str, int, str, str],
    seed: int,
) -> dict[str, float]:
    digest = hashlib.sha256(
        (str(seed) + "|" + "|".join(map(str, identity_key))).encode("utf-8")
    ).digest()
    coherent_z = random.Random(int.from_bytes(digest[:8], "big")).gauss(0.0, 1.0)
    values: dict[str, float] = {}
    missing: list[str] = []
    for gate in bundle.active_speed_gates:
        gate_id = str(gate["id"])
        match = None
        for sample in gate.get("target_speed_distribution", {}).get("samples", []):
            key = (
                str(sample["run_id"]),
                int(sample["lap_id"]),
                str(sample["vehicle_id"]),
                str(sample["driver_id"]),
            )
            if key == identity_key:
                match = sample
                break
        if match is None:
            missing.append(gate_id)
            continue
        value = float(match["value_mps"])
        sigma = float(match.get("measurement_standard_deviation_mps", 0.0))
        values[gate_id] = max(0.0, value + sigma * coherent_z)
    if missing:
        raise ValueError(
            f"Measured traversal {identity_key!r} lacks gate evidence for "
            + ", ".join(sorted(missing))
        )
    return values
