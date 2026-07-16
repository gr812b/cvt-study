"""Baseline simulation artifact writers.

Reporting is intentionally downstream of the mechanism and integrator modules: it
consumes completed traces and summaries but cannot influence the simulation state.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt

from cvt_track_study.bundle import TrackBundle

from .integrator import SimulationTrace
from .metrics import gate_compliance_rows
from .models import CVTModel, SimulationSettings, StudyCase
from .track import RuntimeTrack


def write_baseline_outputs(
    *,
    output: Path,
    bounded: SimulationTrace,
    reference: SimulationTrace,
    bounded_summary: Mapping[str, Any],
    reference_summary: Mapping[str, Any],
    comparison: Mapping[str, Any],
    bounded_case: StudyCase,
    settings: SimulationSettings,
    track: RuntimeTrack,
    bundle: TrackBundle,
    bundle_path: Path,
    study_name: str,
    vehicle_id: str,
) -> None:
    bounded.write_csv(output / "bounded_trace.csv")
    reference.write_csv(output / "infinite_reference_trace.csv")
    write_json(output / "bounded_summary.json", bounded_summary)
    write_json(output / "infinite_reference_summary.json", reference_summary)
    write_json(output / "comparison_summary.json", comparison)
    _write_bundle_snapshot(output, bundle, bundle_path)
    _write_gate_table(output / "gate_compliance.csv", bounded, reference, track)
    _write_obstacle_energy_table(
        output / "obstacle_energy_by_feature.csv", bounded, reference, track
    )
    _write_resolved_case(
        output / "resolved_simulation_case.json",
        bounded_case,
        settings,
        track,
        bundle,
    )
    _write_plots(
        output,
        bounded,
        reference,
        bounded_summary,
        track,
        bounded_case.cvt,
    )
    _write_report(
        output / "REPORT.md",
        comparison,
        bounded_summary,
        reference_summary,
        track,
    )
    write_json(
        output / "run_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "study": study_name,
            "vehicle_id": vehicle_id,
            "track_bundle": "track_bundle.json",
            "track_bundle_sha256": bundle.sha256,
            "track_bundle_content_fingerprint": bundle.data.get(
                "content_fingerprint_sha256"
            ),
            "sampling_mode": "nominal",
            "gpx_grade_force_enabled": track.gpx_grade_force_enabled,
            "lateral_traction_coupling_enabled": False,
            "physical_feature_count": len(track.features),
            "active_gate_count": len(track.speed_gates),
        },
    )


def _write_bundle_snapshot(
    output: Path, bundle: TrackBundle, bundle_path: Path
) -> None:
    """Copy the exact validated bundle into the result for portable replay."""

    source = bundle.path or bundle_path.resolve()
    payload = source.read_bytes()
    target = output / "track_bundle.json"
    target.write_bytes(payload)
    target.with_name("track_bundle.sha256").write_text(
        f"{bundle.sha256}  {target.name}\n", encoding="utf-8"
    )


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_gate_table(
    path: Path,
    bounded: SimulationTrace,
    reference: SimulationTrace,
    track: RuntimeTrack,
) -> None:
    bounded_rows = {row["gate_id"]: row for row in gate_compliance_rows(bounded, track)}
    reference_rows = {
        row["gate_id"]: row for row in gate_compliance_rows(reference, track)
    }
    fields = [
        "gate_id",
        "response_group_id",
        "name",
        "position_s_m",
        "target_speed_mps",
        "bounded_speed_mps",
        "bounded_excess_kmh",
        "bounded_compliant_0p5_kmh",
        "reference_speed_mps",
        "reference_excess_kmh",
        "reference_compliant_0p5_kmh",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for gate in track.speed_gates:
            bounded_row = bounded_rows[gate.identifier]
            reference_row = reference_rows[gate.identifier]
            writer.writerow(
                {
                    "gate_id": gate.identifier,
                    "response_group_id": gate.response_group_id,
                    "name": gate.name,
                    "position_s_m": gate.position_s_m,
                    "target_speed_mps": gate.target_speed_mps,
                    "bounded_speed_mps": bounded_row["simulated_speed_mps"],
                    "bounded_excess_kmh": bounded_row["excess_over_ceiling_kmh"],
                    "bounded_compliant_0p5_kmh": bounded_row[
                        "compliant_within_0p5_kmh"
                    ],
                    "reference_speed_mps": reference_row["simulated_speed_mps"],
                    "reference_excess_kmh": reference_row["excess_over_ceiling_kmh"],
                    "reference_compliant_0p5_kmh": reference_row[
                        "compliant_within_0p5_kmh"
                    ],
                }
            )



def _write_obstacle_energy_table(
    path: Path,
    bounded: SimulationTrace,
    reference: SimulationTrace,
    track: RuntimeTrack,
) -> None:
    fields = [
        "feature_id",
        "name",
        "response_group_id",
        "model_type",
        "bounded_entry_speed_mps",
        "reference_entry_speed_mps",
        "bounded_obstacle_energy_kj",
        "reference_obstacle_energy_kj",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for feature in track.features:
            writer.writerow(
                {
                    "feature_id": feature.identifier,
                    "name": feature.name,
                    "response_group_id": feature.response_group_id,
                    "model_type": feature.model.model_type,
                    "bounded_entry_speed_mps": bounded.feature_entry_speeds_mps.get(
                        feature.identifier
                    ),
                    "reference_entry_speed_mps": reference.feature_entry_speeds_mps.get(
                        feature.identifier
                    ),
                    "bounded_obstacle_energy_kj": bounded.feature_obstacle_energy_j.get(
                        feature.identifier, 0.0
                    )
                    / 1000.0,
                    "reference_obstacle_energy_kj": reference.feature_obstacle_energy_j.get(
                        feature.identifier, 0.0
                    )
                    / 1000.0,
                }
            )

def _write_resolved_case(
    path: Path,
    case: StudyCase,
    settings: SimulationSettings,
    track: RuntimeTrack,
    bundle: TrackBundle,
) -> None:
    data = {
        "engine": {
            "target_rpm": case.engine.target_rpm,
            "target_power_w": case.engine.target_power_w,
            "power_scale": case.engine.power_scale,
            "torque_curve": [asdict(point) for point in case.engine.points],
        },
        "vehicle": asdict(case.vehicle),
        "tire": asdict(case.tire),
        "cvt": asdict(case.cvt),
        "driver": asdict(case.driver),
        "simulation": asdict(settings),
        "track": {
            "name": track.name,
            "length_m": track.length_m,
            "surface_friction_coefficient": track.surface_friction_coefficient,
            "gpx_grade_force_enabled": track.gpx_grade_force_enabled,
            "lateral_traction_coupling_enabled": False,
            "physical_feature_count": len(track.features),
            "active_gate_count": len(track.speed_gates),
            "bundle_schema_version": bundle.schema_version,
        },
    }
    write_json(path, data)


def _write_plots(
    output: Path,
    bounded: SimulationTrace,
    reference: SimulationTrace,
    bounded_summary: Mapping[str, Any],
    track: RuntimeTrack,
    cvt: CVTModel,
) -> None:
    distance = bounded.numeric["distance_m"]
    figure, axes = plt.subplots(figsize=(12, 5.5))
    axes.plot(distance, bounded.numeric["vehicle_speed_kmh"], label="Bounded CVT")
    axes.plot(
        reference.numeric["distance_m"],
        reference.numeric["vehicle_speed_kmh"],
        label="Infinite reference",
    )
    for gate in track.speed_gates:
        axes.scatter([gate.position_s_m], [3.6 * gate.target_speed_mps], marker="v")
    axes.set_xlabel("Track distance, s [m]")
    axes.set_ylabel("Vehicle speed [km/h]")
    axes.set_title("Bounded and infinite-CVT speed traces with active gate ceilings")
    axes.grid(True, alpha=0.25)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output / "01_speed_comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(12, 5.2))
    axes.plot(distance, bounded.numeric["cvt_ratio"], label="Selected ratio")
    axes.axhline(
        cvt.maximum_reduction_ratio, linestyle=":", label="Maximum reduction"
    )
    axes.axhline(
        cvt.minimum_reduction_ratio, linestyle="--", label="Minimum reduction"
    )
    axes.set_xlabel("Track distance, s [m]")
    axes.set_ylabel("CVT ratio, engine/secondary")
    axes.set_title("Bounded-CVT ratio occupancy")
    axes.grid(True, alpha=0.25)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output / "02_ratio_trace.png", dpi=180)
    plt.close(figure)

    labels = [
        "Drivetrain",
        "Clutch",
        "Off-peak",
        "Tire slip",
        "Braking",
        "Obstacles",
        "Rolling",
        "Aero",
    ]
    values = [
        bounded_summary["drivetrain_loss_energy_kj"],
        bounded_summary["clutch_loss_energy_kj"],
        bounded_summary["engine_operating_shortfall_energy_kj"],
        bounded_summary["tire_slip_loss_energy_kj"],
        bounded_summary["brake_loss_energy_kj"],
        bounded_summary["obstacle_loss_energy_kj"],
        bounded_summary["rolling_loss_energy_kj"],
        bounded_summary["aerodynamic_loss_energy_kj"],
    ]
    figure, axes = plt.subplots(figsize=(10, 5.5))
    bars = axes.bar(labels, values)
    axes.bar_label(bars, fmt="%.1f")
    axes.set_ylabel("Energy [kJ]")
    axes.set_title("Bounded-CVT losses and opportunity diagnostic (not additive)")
    axes.tick_params(axis="x", rotation=20)
    axes.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "03_energy_accounting.png", dpi=180)
    plt.close(figure)


def _write_report(
    path: Path,
    comparison: Mapping[str, Any],
    bounded: Mapping[str, Any],
    reference: Mapping[str, Any],
    track: RuntimeTrack,
) -> None:
    text = f"""# Phase 5 nominal baseline comparison

- Track: {track.name}
- Physical feature models: {len(track.features)}
- Active measured speed gates: {len(track.speed_gates)}
- Bounded lap time: {float(bounded['lap_time_s']):.3f} s
- Infinite-reference lap time: {float(reference['lap_time_s']):.3f} s
- Time penalty: {float(comparison['lap_time_penalty_vs_infinite_s']):.3f} s
- Bounded clutch-plus-off-peak loss: {float(comparison['bounded_total_opportunity_loss_energy_kj']):.3f} kJ
- Shared infinite-reference launch loss: {float(comparison['reference_shared_launch_loss_energy_kj']):.3f} kJ
- Finite-ratio-only opportunity loss: {float(comparison['finite_ratio_opportunity_loss_energy_kj']):.3f} kJ
- Reference dominance check: {comparison['reference_dominance_pass']}
- Bounded energy-balance relative error: {float(bounded['energy_balance_relative_error']):.6f}
- Infinite-reference energy-balance relative error: {float(reference['energy_balance_relative_error']):.6f}

This run uses nominal values from every uncertainty declaration. Confidence bands
are produced in Phase 6. Obstacle losses come only from the explicit per-feature
model contracts in the track bundle; measured speed reductions are not silently
converted into terrain dissipation. GPX elevation remains stored for review and
does not create grade force. The reported finite-ratio opportunity loss subtracts
the launch-clutch loss shared by the infinite reference, while drivetrain-efficiency
loss remains a separate physical term common to both designs. Centreline curvature
and lateral-force demand are reported as diagnostics only; measured gates, rather
than an unvalidated lateral vehicle model, define driver-limited corner-entry
ceilings in Phase 5.
"""
    path.write_text(text, encoding="utf-8")
