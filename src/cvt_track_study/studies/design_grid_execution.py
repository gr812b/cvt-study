"""Execution support for simultaneous multi-variable drivetrain design points.

This module deliberately lives beside the existing Phase-8 study runner.  It
uses the same simulation, cache, comparison, and summary helpers, but applies a
complete mapping of design overrides for each Cartesian candidate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from cvt_track_study.bundle import TrackBundle
from cvt_track_study.runtime import SimulationCache
from cvt_track_study.simulation.metrics import compare_summaries
from cvt_track_study.simulation.service import SimulationError, resolve_simulation_cases
from cvt_track_study.uncertainty import ScenarioDraw

from . import service as phase6_service
from . import service_v8
from .analysis import METRICS
from .design_grid import (
    design_choice_values,
    design_display_values,
    design_paths,
    design_quantity_values_si,
    reference_can_be_shared,
)
from .planning import reference_cache_key


def execute_design_grid_scenario(
    *,
    scenario: ScenarioDraw,
    design_points: Sequence[Any],
    study_type: str,
    vehicle_id: str,
    vehicle_raw: Mapping[str, Any],
    base_study: Mapping[str, Any],
    track_raw: Mapping[str, Any],
    bundle: TrackBundle,
    cache: SimulationCache,
) -> dict[str, Any]:
    """Run one paired uncertainty world across every design-grid candidate."""

    points = tuple(design_points)
    rows: list[dict[str, Any]] = []
    references: dict[tuple[int, str], tuple[dict[str, Any], str]] = {}
    bounded_runs = reference_runs = reference_reuses = persistent_hits = 0

    all_design_paths = design_paths(points)
    share_reference = (
        study_type == "design_sweep"
        and reference_can_be_shared(all_design_paths)
    )

    for design in points:
        design_values = design_quantity_values_si(design)
        choices = dict(scenario.choice_values)
        choices.update(design_choice_values(design))
        try:
            bounded_case, reference_case, settings, runtime_track = resolve_simulation_cases(
                vehicle_id=vehicle_id,
                vehicle_raw=vehicle_raw,
                study_raw=base_study,
                track_raw=track_raw,
                bundle=bundle,
                quantity_values_si=scenario.quantity_values_si,
                choice_values=choices,
                gate_target_speeds_mps=scenario.gate_target_speeds_mps,
                design_values_si=design_values,
                shared_reference=share_reference,
            )
        except Exception as exc:
            raise SimulationError(
                f"Scenario {scenario.replicate}, design {design.identifier!r} "
                f"could not form a valid physical case: {exc}"
            ) from exc

        bounded_record, cached = service_v8._run_case_summary_cached(
            bounded_case, settings, runtime_track, cache
        )
        bounded_runs += int(not cached)
        persistent_hits += int(cached)

        key = reference_cache_key(
            scenario.replicate,
            design,
            share_across_designs=share_reference,
        )
        if key in references:
            reference_record, reference_fingerprint = references[key]
            reference_reuses += 1
        else:
            reference_record, cached = service_v8._run_case_summary_cached(
                reference_case, settings, runtime_track, cache
            )
            reference_runs += int(not cached)
            persistent_hits += int(cached)
            reference_fingerprint = phase6_service._reference_fingerprint(
                scenario, design, reference_case, runtime_track
            )
            references[key] = (reference_record, reference_fingerprint)

        bounded_summary = bounded_record["summary"]
        reference_summary = reference_record["summary"]
        comparison = compare_summaries(bounded_summary, reference_summary)
        display_values = design_display_values(design)
        quantity_values_si = design_quantity_values_si(design)
        point_paths = tuple(display_values)
        design_path_text = (
            str(getattr(design, "path", None))
            if getattr(design, "path", None)
            else " + ".join(point_paths) if point_paths else "nominal"
        )

        row: dict[str, Any] = {
            "replicate": scenario.replicate,
            "scenario_seed": scenario.seed,
            "design_id": design.identifier,
            "design_path": design_path_text,
            "design_value": design.display_value,
            "design_value_si": design.value_si,
            "design_choice_value": getattr(design, "choice_value", None),
            "design_dimension_count": len(point_paths),
            "design_paths_json": json.dumps(
                list(point_paths), separators=(",", ":"), allow_nan=False
            ),
            "design_values_json": json.dumps(
                display_values,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "design_values_si_json": json.dumps(
                quantity_values_si,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "level_probability": getattr(design, "level_probability", None),
            "level_kind": getattr(design, "level_kind", "design_grid"),
            "parameter_path": (
                getattr(design, "path", None)
                if study_type == "structural_sensitivity"
                else None
            ),
            "reference_fingerprint": reference_fingerprint,
            "bounded_completed": bool(bounded_summary["completed"]),
            "reference_completed": bool(reference_summary["completed"]),
            "reference_dominance_pass": bool(comparison["reference_dominance_pass"]),
            "bounded_energy_balance_relative_error": float(
                comparison["bounded_energy_balance_relative_error"]
            ),
            "reference_energy_balance_relative_error": float(
                comparison["reference_energy_balance_relative_error"]
            ),
            "bounded_powertrain_energy_balance_relative_error": float(
                bounded_summary["powertrain_energy_balance_relative_error"]
            ),
            "reference_powertrain_energy_balance_relative_error": float(
                reference_summary["powertrain_energy_balance_relative_error"]
            ),
            "bounded_max_gate_excess_kmh": bounded_record[
                "maximum_gate_excess_kmh"
            ],
            "reference_max_gate_excess_kmh": reference_record[
                "maximum_gate_excess_kmh"
            ],
            "bounded_gates_compliant_0p5_kmh": bounded_record[
                "gates_compliant_0p5_kmh"
            ],
            "reference_gates_compliant_0p5_kmh": reference_record[
                "gates_compliant_0p5_kmh"
            ],
        }
        for index, (path, value) in enumerate(display_values.items()):
            row[f"design_axis_{index}_path"] = path
            row[f"design_axis_{index}_value"] = value
            row[f"design::{path}"] = value
        row.update({metric: float(comparison[metric]) for metric in METRICS})
        service_v8._add_summary_fields(row, "bounded", bounded_summary)
        service_v8._add_summary_fields(row, "reference", reference_summary)
        rows.append(row)

    return {
        "rows": rows,
        "bounded_case_count": len(points),
        "reference_case_count": 1 if share_reference and points else len(points),
        "bounded_simulation_count": bounded_runs,
        "reference_simulation_count": reference_runs,
        "reference_cache_hits": reference_reuses,
        "simulation_cache_hits": persistent_hits,
        "resumed": False,
    }
