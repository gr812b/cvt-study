from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from track_builder import EffectiveEnergyEvent, SpeedGate, Track, TrackSection


def load_study_bundle(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0":
        raise ValueError("Unsupported or missing measured-track bundle schema")
    return raw


def build_track_from_bundle(
    bundle: Mapping[str, object],
    *,
    gate_speed_quantile: str = "median",
    minimum_gate_confidence: float = 60.0,
    loss_quantile: str = "nominal",
    loss_scale: float = 1.0,
    gate_speed_overrides_kmh: Mapping[str, float] | None = None,
    event_loss_overrides_j_per_kg: Mapping[str, float] | None = None,
) -> Track:
    if gate_speed_quantile not in {"p10", "median", "p90"}:
        raise ValueError("gate_speed_quantile must be p10, median, or p90")
    if loss_quantile not in {"low", "nominal", "high"}:
        raise ValueError("loss_quantile must be low, nominal, or high")
    if loss_scale < 0.0:
        raise ValueError("loss_scale must be non-negative")
    track_data = dict(bundle["track"])
    length_m = float(track_data["length_m"])
    base = dict(track_data.get("base_surface_assumptions", {}))
    section = TrackSection(
        name="Measured track base",
        length_m=length_m,
        grade_degrees=float(base.get("grade_degrees", 0.0)),
        friction_coefficient=float(base.get("friction_coefficient", 0.70)),
        rolling_resistance_coefficient=float(
            base.get("rolling_resistance_coefficient", 0.03)
        ),
        surface="measured-track scenario base",
    )

    gate_speed_overrides_kmh = dict(gate_speed_overrides_kmh or {})
    gates: list[SpeedGate] = []
    speed_key = f"target_speed_{gate_speed_quantile}_kmh"
    for item in bundle.get("speed_gates", []):
        gate = dict(item)
        confidence = float(gate["confidence_score"])
        if confidence < minimum_gate_confidence:
            continue
        group_id = str(gate["analysis_group_id"])
        target_kmh = float(
            gate_speed_overrides_kmh.get(group_id, gate[speed_key])
        )
        gates.append(
            SpeedGate(
                name=str(gate["event_name"]),
                position_m=float(gate["gate_s_m"]),
                target_speed_mps=target_kmh / 3.6,
                braking_deceleration_mps2=float(gate["braking_deceleration_mps2"]),
                confidence_score=confidence,
                confidence_class=str(gate["confidence_class"]),
                source_group_id=group_id,
            )
        )

    event_loss_overrides_j_per_kg = dict(event_loss_overrides_j_per_kg or {})
    loss_key = f"effective_specific_loss_{loss_quantile}_j_per_kg"
    features: list[EffectiveEnergyEvent] = []
    for item in bundle.get("event_groups", []):
        event = dict(item)
        if str(event.get("analysis_role")) != "track_event":
            continue
        group_id = str(event["analysis_group_id"])
        total_specific_loss = float(
            event_loss_overrides_j_per_kg.get(group_id, event[loss_key])
        ) * loss_scale
        if total_specific_loss <= 0.0:
            continue
        start = float(event["start_s_m"])
        end = float(event["end_s_m"])
        total_length = max(float(event["length_m"]), 0.5)
        if end > start:
            spans = [(start, end - start)]
        else:
            spans = []
            if start < length_m:
                spans.append((start, length_m - start))
            if end > 0.0:
                spans.append((0.0, end))
        for part_index, (part_start, part_length) in enumerate(spans, start=1):
            if part_length <= 1.0e-6:
                continue
            fraction = part_length / total_length
            features.append(
                EffectiveEnergyEvent(
                    name=f"{event['name']} effective part {part_index}",
                    start_m=part_start,
                    length_m=part_length,
                    specific_energy_loss_j_per_kg=total_specific_loss * fraction,
                    model_status=str(event.get("loss_model_status", "uncalibrated")),
                )
            )

    notes = (
        "Measured single-s surrogate. Speed gates are confidence filtered and "
        "enforced through braking envelopes. Effective event losses are GPS-derived "
        "net kinetic-change scenario seeds, not calibrated terrain dissipation."
    )
    return Track(
        name=str(track_data.get("name", "Measured track surrogate")),
        sections=(section,),
        features=tuple(features),
        speed_gates=tuple(sorted(gates, key=lambda gate: gate.position_m)),
        notes=notes,
    )
