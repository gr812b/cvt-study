"""Track-only defensibility study over telemetry reconstruction choices.

This module deliberately does not resolve a vehicle, run the longitudinal model,
or compare drivetrain designs.  It asks only whether the track inferred from the
supplied telemetry is stable under defensible analysis choices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvt_track_study.config import ProjectError, ProjectLoader
from cvt_track_study.config.diagnostics import DiagnosticBag
from cvt_track_study.gpx.cleanup import apply_telemetry_cleanup
from cvt_track_study.gpx.ingestion import TelemetryParseError, ingest_telemetry_run
from cvt_track_study.gpx.model import GPXRunMetadata
from cvt_track_study.runtime import ProgressReporter
from cvt_track_study.runtime.provenance import canonical_fingerprint

from .model import TrackBuildResult
from .reconstruction import build_track_evidence
from .service import build_project_track as _build_nominal_track
from cvt_track_study.reports.catalog import REPORTS
from cvt_track_study.reports.html import dataframe_table, figure, metric_cards, render_page, write_json


@dataclass(frozen=True)
class RobustnessCase:
    identifier: str
    category: str
    label: str
    rationale: str
    overrides: Mapping[str, Any]
    excluded_run_ids: tuple[str, ...] = ()


@dataclass
class CaseResult:
    case: RobustnessCase
    success: bool
    error: str
    centreline: Any | None = None
    laps: pd.DataFrame | None = None
    event_projection: pd.DataFrame | None = None
    gate_review: pd.DataFrame | None = None
    diagnostics: tuple[Any, ...] = ()
    rejected_map_points: pd.DataFrame | None = None
    track_build: TrackBuildResult | None = None


_DEFAULT_THRESHOLDS = {
    "maximum_centreline_p95_shift_m": 5.0,
    "maximum_event_p95_shift_m": 10.0,
    "maximum_track_length_relative_shift": 0.02,
    "minimum_gate_classification_agreement": 0.80,
    "maximum_gate_target_p80_range_mps": 2.0,
    "core_gate_definition_acceptance_fraction": 0.75,
    "near_miss_gate_definition_acceptance_fraction": 0.20,
    "high_speed_gate_review_kmh": 30.0,
    "minimum_braking_evidence_score_for_high_speed_gate": 50.0,
    "minimum_gate_set_jaccard_for_ensemble": 0.50,
    "minimum_nominal_gate_retention_fraction_for_ensemble": 0.60,
    "maximum_new_gate_count_for_ensemble": 8.0,
}

_GATE_DEFINITION_CATEGORIES = {"gate_policy", "gate_weighting", "event_windows"}
_PRIMARY_GEOMETRY_CATEGORIES = {"centreline", "telemetry_cleanup"}
_ALL_GEOMETRY_CATEGORIES = {"data_support", "centreline", "telemetry_cleanup"}
_EVENT_GEOMETRY_CATEGORIES = {
    "data_support", "centreline", "telemetry_cleanup", "event_windows"
}

_STATUS_CODE = {
    "accepted": 3,
    "recommended_review": 2,
    "must_fix": 1,
    "rejected": 0,
    "not_a_candidate": -1,
}


def run_track_robustness_project(
    project: str | Path,
    *,
    study: str = "track_robustness",
    output_directory: Path | None = None,
    workers: int = 1,
    resume: bool = False,
    restart: bool = False,
    progress: bool = True,
    run_name: str | None = None,
    command: tuple[str, ...] = (),
) -> Path:
    """Run the complete data-only track defensibility report."""

    if workers < 1:
        raise ValueError("workers must be at least one")
    resolution = ProjectLoader().resolve(project)
    if resolution.error_count:
        details = "\n".join(item.format() for item in resolution.diagnostics)
        raise ProjectError(f"Project validation failed:\n{details}")
    raw_study = resolution.data.get("studies", {}).get(study, {})
    if not isinstance(raw_study, Mapping):
        raise ProjectError(f"Track robustness study {study!r} was not found.")
    if str(raw_study.get("study", {}).get("type", "")) != "track_robustness":
        raise ProjectError(f"Study {study!r} is not type='track_robustness'.")

    track_config = resolution.data.get("track", {})
    raw_runs = tuple(raw for raw in resolution.data.get("runs", ()) if isinstance(raw, Mapping))
    raw_events = resolution.data.get("events", [])
    cases = build_robustness_cases(track_config, raw_runs, raw_study)
    thresholds = _thresholds(raw_study)
    fingerprint = canonical_fingerprint(
        {
            "schema": "track-defensibility-v1",
            "study": raw_study,
            "track": track_config,
            "runs": raw_runs,
            "events": raw_events,
            "cases": [case_to_dict(case) for case in cases],
        }
    )
    default_output = (
        resolution.paths.results_directory
        / "track_robustness"
        / f"{_safe_name(run_name or study)}--{fingerprint[:10]}"
    )
    output = (output_directory or default_output).resolve()
    if output.exists():
        existing = _read_json(output / "track_robustness_manifest.json", {})
        if resume and existing.get("study_fingerprint_sha256") == fingerprint:
            return output
        if restart:
            shutil.rmtree(output)
        else:
            raise ProjectError(
                f"Output directory already exists: {output}. Use --resume or --restart."
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    reporter = ProgressReporter(total=len(cases) + 1, label="track cases", enabled=progress)
    reporter.begin(
        f"1 nominal reconstruction plus {len(cases)} robustness cases; "
        f"vehicle and drivetrain simulation disabled; {workers} worker(s) requested"
    )
    try:
        nominal_dir = staging / "nominal_track"
        nominal = _build_nominal_track(project, output_directory=nominal_dir)
        reporter.advance("nominal reconstruction")
        parsed_runs = _parse_runs(resolution, raw_runs)
        case_results: list[CaseResult] = []
        # Track reconstruction currently contains internal shared objects that are
        # safest to evaluate serially.  The workers value is retained in the manifest
        # and CLI contract; parallel case execution can be enabled later after the
        # reconstruction core is made explicitly re-entrant.
        for case in cases:
            result = _execute_case(
                case=case,
                resolution=resolution,
                parsed_runs=parsed_runs,
                track_config=track_config,
                raw_events=raw_events,
            )
            case_results.append(result)
            _write_case_artifacts(staging / "cases" / _safe_name(case.identifier), result)
            reporter.advance(case.label if result.success else f"{case.label} (failed)")

        tables = summarize_track_robustness(nominal, case_results, thresholds)
        ensemble_manifest = _write_track_ensemble_manifest(
            staging, tables["robustness_case_summary"], thresholds
        )
        tables["ensemble_eligibility"] = pd.DataFrame(ensemble_manifest["cases"])
        for name, frame in tables.items():
            frame.to_csv(staging / f"{name}.csv", index=False)
        write_json(staging / "robustness_cases.json", [case_to_dict(case) for case in cases])
        _write_robustness_plots(staging, tables)
        _write_centreline_overlay(staging, nominal, case_results)
        _write_centreline_stability_corridor(staging, tables["robustness_case_summary"])
        _write_track_robustness_html(
            staging,
            nominal_track_length_m=float(nominal.centreline.length_m),
            tables=tables,
            thresholds=thresholds,
            ensemble_manifest=ensemble_manifest,
        )

        successful = sum(result.success for result in case_results)
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "study_name": study,
            "study_type": "track_robustness",
            "study_fingerprint_sha256": fingerprint,
            "question": REPORTS["track_robustness"].question,
            "track_only": True,
            "vehicle_simulation_count": 0,
            "drivetrain_design_count": 0,
            "nominal_reconstruction_count": 1,
            "robustness_case_count": len(cases),
            "successful_case_count": successful,
            "failed_case_count": len(cases) - successful,
            "requested_workers": workers,
            "effective_workers": 1,
            "case_categories": sorted({case.category for case in cases}),
            "thresholds": thresholds,
            "command": list(command),
            "nominal_track_bundle": "nominal_track/track_bundle.json",
            "primary_report": REPORTS["track_robustness"].html_filename,
            "track_ensemble_manifest": "track_ensemble_manifest.json",
            "eligible_track_ensemble_case_count": len(
                ensemble_manifest["eligible_cases"]
            ),
        }
        write_json(staging / "track_robustness_manifest.json", manifest)
        write_json(
            staging / "report_manifest.json",
            {
                "schema_version": 1,
                "report_key": "track_robustness",
                "title": REPORTS["track_robustness"].title,
                "question": REPORTS["track_robustness"].question,
                "fixed": REPORTS["track_robustness"].fixed,
                "varied": REPORTS["track_robustness"].varied,
                "html_file": REPORTS["track_robustness"].html_filename,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        resolution.export(staging / "resolved_inputs")
        reporter.finish(f"{successful}/{len(cases)} alternative cases succeeded")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_robustness_cases(
    track_config: Mapping[str, Any],
    raw_runs: Sequence[Mapping[str, Any]],
    raw_study: Mapping[str, Any],
) -> tuple[RobustnessCase, ...]:
    """Construct explicit, auditable robustness cases from the nominal settings."""

    options = raw_study.get("robustness", {})
    if not isinstance(options, Mapping):
        options = {}
    cases: list[RobustnessCase] = []

    if bool(options.get("include_leave_one_run_out", True)):
        for run in raw_runs:
            run_id = str(run.get("run_id", ""))
            if run_id and _remaining_run_contract_valid(raw_runs, {run_id}):
                cases.append(
                    RobustnessCase(
                        f"leave_run_out__{run_id}",
                        "data_support",
                        f"Leave out run {run_id}",
                        "Checks whether one recording controls the centreline or gate evidence.",
                        {},
                        (run_id,),
                    )
                )

    if bool(options.get("include_leave_one_vehicle_out", True)):
        vehicles = sorted({str(run.get("vehicle_id", "")) for run in raw_runs if run.get("vehicle_id")})
        if len(vehicles) > 1:
            for vehicle in vehicles:
                excluded = {str(run.get("run_id")) for run in raw_runs if str(run.get("vehicle_id")) == vehicle}
                if _remaining_run_contract_valid(raw_runs, excluded):
                    cases.append(RobustnessCase(f"leave_vehicle_out__{vehicle}", "data_support", f"Leave out vehicle {vehicle}", "Tests cross-vehicle dependence of the inferred track.", {}, tuple(sorted(excluded))))

    if bool(options.get("include_leave_one_driver_out", True)):
        drivers = sorted({str(run.get("driver_id", "")) for run in raw_runs if run.get("driver_id")})
        if len(drivers) > 1:
            for driver in drivers:
                excluded = {str(run.get("run_id")) for run in raw_runs if str(run.get("driver_id")) == driver}
                if _remaining_run_contract_valid(raw_runs, excluded):
                    cases.append(RobustnessCase(f"leave_driver_out__{driver}", "data_support", f"Leave out driver {driver}", "Tests whether one driver's line or braking style controls the inference.", {}, tuple(sorted(excluded))))

    gates = _mapping(track_config.get("gate_confidence"))
    windows = _mapping(track_config.get("event_windows"))
    reconstruction = _mapping(track_config.get("reconstruction"))
    consensus = _mapping(track_config.get("centreline_consensus"))
    cleanup = _mapping(track_config.get("telemetry_cleanup"))

    if bool(options.get("include_gate_policy_cases", True)):
        minimum = int(gates.get("minimum_valid_passes", 5))
        target = int(gates.get("target_pass_count", 10))
        accept = float(gates.get("accept_score", 60.0))
        review = float(gates.get("review_score", 40.0))
        braking = float(gates.get("braking_threshold_mps", 0.8))
        cases.extend(
            [
                _case("gate_strict", "gate_policy", "Strict gate policy", "Requires more repeated evidence and a higher confidence score.", {
                    "gate_confidence.minimum_valid_passes": minimum + 2,
                    "gate_confidence.target_pass_count": max(target + 2, minimum + 2),
                    "gate_confidence.accept_score": min(95.0, accept + 10.0),
                    "gate_confidence.review_score": min(min(94.0, accept + 9.0), review + 10.0),
                    "gate_confidence.braking_threshold_mps": 1.25 * braking,
                }),
                _case("gate_permissive", "gate_policy", "Permissive gate policy", "Tests supported review-level gates without pretending the score is a probability.", {
                    "gate_confidence.minimum_valid_passes": max(3, minimum - 2),
                    "gate_confidence.target_pass_count": max(5, target - 2),
                    "gate_confidence.accept_score": max(20.0, accept - 10.0),
                    "gate_confidence.review_score": max(10.0, min(review - 10.0, accept - 11.0)),
                    "gate_confidence.braking_threshold_mps": max(0.05, 0.75 * braking),
                }),
                _case("gate_equal_weights", "gate_weighting", "Equal confidence weights", "Removes the nominal preference among confidence components.", _weight_overrides({name: 1.0 / 6.0 for name in _weight_names()})),
                _case("gate_no_pace_weight", "gate_weighting", "No pace-independence weight", "Checks whether the pace-adjustment component controls qualification.", _weight_overrides(_renormalized_weights(gates, omitted="pace_independence"))),
                _case("gate_no_coordinate_weight", "gate_weighting", "No coordinate-quality weight", "Checks whether manually declared coordinate quality controls qualification.", _weight_overrides(_renormalized_weights(gates, omitted="coordinate_quality"))),
                _case(
                    "gate_stronger_cross_vehicle",
                    "gate_weighting",
                    "Stronger weight on unavailable cross-vehicle evidence",
                    "Tests sensitivity to the neutral cross-vehicle placeholder when only one vehicle is represented; it is not evidence of actual cross-vehicle agreement.",
                    _weight_overrides(_boost_weight(gates, "cross_vehicle_agreement", 0.25)),
                ),
            ]
        )

    if bool(options.get("include_event_window_cases", True)):
        cases.extend(
            [
                _case("event_windows_narrow", "event_windows", "Narrow event windows", "Moves approach, entry, exit, and recovery windows closer to each feature.", _scaled_overrides("event_windows", windows, 0.75, exclude={"entry_gap_m", "approach_gap_m", "exit_gap_m"})),
                _case("event_windows_wide", "event_windows", "Wide event windows", "Tests whether gate evidence depends on generous windows around each feature.", _scaled_overrides("event_windows", windows, 1.25, exclude={"entry_gap_m", "approach_gap_m", "exit_gap_m"})),
            ]
        )

    if bool(options.get("include_centreline_cases", True)):
        map_error = float(reconstruction.get("maximum_map_error_m", 20.0))
        smoothing = int(consensus.get("smoothing_window_nodes", 5))
        cases.extend(
            [
                _case("centreline_strict_outlier", "centreline", "Strict centreline outlier policy", "Rejects map and consensus disagreement earlier.", {
                    "reconstruction.maximum_map_error_m": 0.75 * map_error,
                    "centreline_consensus.leave_one_out_p95_limit_m": 0.75 * float(consensus.get("leave_one_out_p95_limit_m", 15.0)),
                    "centreline_consensus.sustained_error_threshold_m": 0.75 * float(consensus.get("sustained_error_threshold_m", 15.0)),
                }),
                _case("centreline_permissive_outlier", "centreline", "Permissive centreline outlier policy", "Retains more lap geometry before declaring it inconsistent.", {
                    "reconstruction.maximum_map_error_m": 1.25 * map_error,
                    "centreline_consensus.leave_one_out_p95_limit_m": 1.25 * float(consensus.get("leave_one_out_p95_limit_m", 15.0)),
                    "centreline_consensus.sustained_error_threshold_m": 1.25 * float(consensus.get("sustained_error_threshold_m", 15.0)),
                }),
                _case("centreline_less_smoothing", "centreline", "Less centreline smoothing", "Retains more local line variation.", {"centreline_consensus.smoothing_window_nodes": _odd(max(1, smoothing - 2))}),
                _case("centreline_more_smoothing", "centreline", "More centreline smoothing", "Tests whether event s coordinates depend on local GPS wiggle.", {"centreline_consensus.smoothing_window_nodes": _odd(smoothing + 2)}),
                _case("centreline_fine_spacing", "centreline", "Finer centreline spacing", "Tests discretization sensitivity with more centreline nodes.", {"reconstruction.centreline_spacing_m": max(1.0, 0.67 * float(reconstruction.get("centreline_spacing_m", 3.0)))}),
                _case("centreline_coarse_spacing", "centreline", "Coarser centreline spacing", "Tests discretization sensitivity with fewer centreline nodes.", {"reconstruction.centreline_spacing_m": 1.50 * float(reconstruction.get("centreline_spacing_m", 3.0))}),
            ]
        )

    if bool(options.get("include_cleanup_cases", True)) and bool(cleanup.get("enabled", True)):
        cases.extend(
            [
                _case("cleanup_conservative", "telemetry_cleanup", "Conservative telemetry cleanup", "Automatically removes only single-point excursions and fewer map outliers.", {
                    "telemetry_cleanup.maximum_excursion_points": 1,
                    "telemetry_cleanup.minimum_excursion_leg_m": 1.25 * float(cleanup.get("minimum_excursion_leg_m", 35.0)),
                    "telemetry_cleanup.maximum_isolated_map_outlier_points": 1,
                }),
                _case("cleanup_permissive", "telemetry_cleanup", "Permissive telemetry cleanup", "Allows slightly longer isolated bursts while retaining the global removal caps.", {
                    "telemetry_cleanup.maximum_excursion_points": max(5, int(cleanup.get("maximum_excursion_points", 3)) + 2),
                    "telemetry_cleanup.minimum_excursion_leg_m": 0.75 * float(cleanup.get("minimum_excursion_leg_m", 35.0)),
                    "telemetry_cleanup.maximum_isolated_map_outlier_points": max(5, int(cleanup.get("maximum_isolated_map_outlier_points", 3)) + 2),
                }),
            ]
        )

    maximum = int(options.get("maximum_cases", 40))
    return tuple(cases[:maximum])


def summarize_track_robustness(
    nominal: TrackBuildResult,
    case_results: Sequence[CaseResult],
    thresholds: Mapping[str, float],
) -> dict[str, pd.DataFrame]:
    """Build machine-readable stability tables from successful cases.

    Geometry, event interpretation, and gate-definition cases are kept
    distinguishable. Gate-policy fractions are stress-case frequencies, not
    probabilities that a gate is true.
    """

    case_rows: list[dict[str, Any]] = []
    centreline_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    nominal_events = nominal.event_projection.set_index("id", drop=False)
    nominal_gate_review = nominal.gate_review.copy()
    nominal_gate_review.attrs["accept_score"] = float(
        _mapping(
            nominal.resolution.data.get("track", {}).get("gate_confidence", {})
        ).get("accept_score", 60.0)
    )
    nominal_gates = nominal_gate_review.set_index("response_group_id", drop=False)
    nominal_accepted = {
        str(gate_id)
        for gate_id, row in nominal_gates.iterrows()
        if str(row.get("recommendation", "")) == "accepted"
    }
    nominal_length = float(nominal.centreline.length_m)

    for result in case_results:
        base = {
            "case_id": result.case.identifier,
            "category": result.case.category,
            "label": result.case.label,
            "rationale": result.case.rationale,
            "excluded_run_ids": ";".join(result.case.excluded_run_ids),
            "overrides_json": json.dumps(dict(result.case.overrides), sort_keys=True),
            "stress_only": _is_stress_only_case(
                result.case.identifier, result.case.category
            ),
            "success": result.success,
            "error": result.error,
        }
        if not result.success:
            case_rows.append(base)
            continue

        displacement = _centreline_displacement(nominal.centreline, result.centreline)
        shifts = np.asarray([row[1] for row in displacement], dtype=float)
        for progress_value, shift in displacement:
            centreline_rows.append(
                {
                    "case_id": result.case.identifier,
                    "case_category": result.case.category,
                    "progress_fraction": progress_value,
                    "nominal_s_m": progress_value * nominal_length,
                    "shift_m": shift,
                }
            )

        case_gates = result.gate_review.set_index("response_group_id", drop=False)
        case_accepted = {
            str(gate_id)
            for gate_id, row in case_gates.iterrows()
            if str(row.get("recommendation", "")) == "accepted"
        }
        union = nominal_accepted | case_accepted
        intersection = nominal_accepted & case_accepted
        gate_jaccard = len(intersection) / len(union) if union else 1.0
        retention = (
            len(intersection) / len(nominal_accepted) if nominal_accepted else 1.0
        )
        newly_accepted = len(case_accepted - nominal_accepted)
        length_delta = float(result.centreline.length_m - nominal_length)
        length_relative = abs(length_delta) / nominal_length if nominal_length else math.nan
        p95 = float(np.quantile(shifts, 0.95))

        case_rows.append(
            {
                **base,
                "track_length_m": float(result.centreline.length_m),
                "track_length_delta_m": length_delta,
                "track_length_relative_shift": length_relative,
                "track_length_stable": bool(
                    np.isfinite(length_relative)
                    and length_relative
                    <= float(thresholds["maximum_track_length_relative_shift"])
                ),
                "valid_lap_count": int(result.laps["analysis_valid"].sum()),
                "centreline_included_lap_count": int(
                    result.laps.get(
                        "centreline_included", result.laps["analysis_valid"]
                    ).sum()
                ),
                "accepted_gate_count": len(case_accepted),
                "nominal_gate_retention_fraction": retention,
                "gate_set_jaccard": gate_jaccard,
                "newly_accepted_gate_count": newly_accepted,
                "gate_set_stable_for_ensemble": bool(
                    gate_jaccard
                    >= float(thresholds["minimum_gate_set_jaccard_for_ensemble"])
                    and retention
                    >= float(
                        thresholds[
                            "minimum_nominal_gate_retention_fraction_for_ensemble"
                        ]
                    )
                    and newly_accepted
                    <= int(thresholds["maximum_new_gate_count_for_ensemble"])
                ),
                "centreline_shift_median_m": float(np.median(shifts)),
                "centreline_shift_p95_m": p95,
                "centreline_shift_max_m": float(np.max(shifts)),
                "centreline_stable": p95
                <= float(thresholds["maximum_centreline_p95_shift_m"]),
                "warning_count": sum(
                    getattr(item.severity, "value", "") == "warning"
                    for item in result.diagnostics
                ),
                "error_count": sum(
                    getattr(item.severity, "value", "") == "error"
                    for item in result.diagnostics
                ),
            }
        )

        case_events = result.event_projection.set_index("id", drop=False)
        for event_id in sorted(set(nominal_events.index) & set(case_events.index)):
            nominal_row = nominal_events.loc[event_id]
            case_row = case_events.loc[event_id]
            anchor_shift = _circular_s_shift(
                float(nominal_row["anchor_s_m"]),
                nominal_length,
                float(case_row["anchor_s_m"]),
                float(result.centreline.length_m),
            )
            nominal_interval_length = float(
                nominal_row["feature_end_rel_m"] - nominal_row["feature_start_rel_m"]
            )
            case_interval_length = float(
                case_row["feature_end_rel_m"] - case_row["feature_start_rel_m"]
            )
            event_rows.append(
                {
                    "case_id": result.case.identifier,
                    "case_category": result.case.category,
                    "case_label": result.case.label,
                    "event_id": event_id,
                    "event_name": nominal_row.get("name", event_id),
                    "anchor_shift_m": anchor_shift,
                    "nominal_interval_length_m": nominal_interval_length,
                    "case_interval_length_m": case_interval_length,
                    "interval_length_delta_m": case_interval_length
                    - nominal_interval_length,
                    "case_projection_error_m": case_row.get(
                        "anchor_projection_error_m", math.nan
                    ),
                    "case_review_flags": case_row.get("review_flags", ""),
                }
            )

        for gate_id in sorted(set(nominal_gates.index) | set(case_gates.index)):
            nominal_row = (
                nominal_gates.loc[gate_id] if gate_id in nominal_gates.index else None
            )
            case_row = case_gates.loc[gate_id] if gate_id in case_gates.index else None
            gate_rows.append(
                {
                    "case_id": result.case.identifier,
                    "case_category": result.case.category,
                    "case_label": result.case.label,
                    "gate_id": gate_id,
                    "gate_name": (
                        nominal_row.get("event_name", gate_id)
                        if nominal_row is not None
                        else case_row.get("event_name", gate_id)
                    ),
                    "nominal_recommendation": (
                        nominal_row.get("recommendation", "missing")
                        if nominal_row is not None
                        else "missing"
                    ),
                    "case_recommendation": (
                        case_row.get("recommendation", "missing")
                        if case_row is not None
                        else "missing"
                    ),
                    "classification_matches_nominal": bool(
                        nominal_row is not None
                        and case_row is not None
                        and nominal_row.get("recommendation")
                        == case_row.get("recommendation")
                    ),
                    "accepted": bool(
                        case_row is not None
                        and case_row.get("recommendation") == "accepted"
                    ),
                    "overall_confidence_score": (
                        case_row.get("overall_confidence_score", math.nan)
                        if case_row is not None
                        else math.nan
                    ),
                    "valid_pass_count": (
                        case_row.get("valid_pass_count", math.nan)
                        if case_row is not None
                        else math.nan
                    ),
                    "entry_speed_median_mps": (
                        case_row.get("entry_speed_median_mps", math.nan)
                        if case_row is not None
                        else math.nan
                    ),
                    "entry_speed_p10_mps": (
                        case_row.get("entry_speed_p10_mps", math.nan)
                        if case_row is not None
                        else math.nan
                    ),
                    "entry_speed_p90_mps": (
                        case_row.get("entry_speed_p90_mps", math.nan)
                        if case_row is not None
                        else math.nan
                    ),
                    "review_flags": (
                        case_row.get("review_flags", "")
                        if case_row is not None
                        else "missing"
                    ),
                }
            )

    cases = pd.DataFrame(case_rows)
    centreline = pd.DataFrame(centreline_rows)
    event_cases = pd.DataFrame(event_rows)
    gate_cases = pd.DataFrame(gate_rows)
    cases = _enrich_case_summary(cases, event_cases, gate_cases, nominal_gate_review, thresholds)
    event_stability = _aggregate_event_stability(event_cases, thresholds)
    gate_stability = _aggregate_gate_stability(
        gate_cases, thresholds, nominal_gate_review
    )
    return {
        "robustness_case_summary": cases,
        "centreline_displacement": centreline,
        "event_case_results": event_cases,
        "event_stability": event_stability,
        "gate_case_results": gate_cases,
        "gate_stability": gate_stability,
    }

def case_to_dict(case: RobustnessCase) -> dict[str, Any]:
    return {
        "id": case.identifier,
        "category": case.category,
        "label": case.label,
        "rationale": case.rationale,
        "overrides": dict(case.overrides),
        "excluded_run_ids": list(case.excluded_run_ids),
    }


def _execute_case(
    *,
    case: RobustnessCase,
    resolution: Any,
    parsed_runs: Sequence[Any],
    track_config: Mapping[str, Any],
    raw_events: Sequence[Mapping[str, Any]],
) -> CaseResult:
    try:
        config = deepcopy(dict(track_config))
        for path, value in case.overrides.items():
            _set_nested(config, str(path).split("."), value)
        selected = [item for item in parsed_runs if item.metadata.run_id not in set(case.excluded_run_ids)]
        cleaned = tuple(apply_telemetry_cleanup(item, config) for item in selected)
        diagnostics = DiagnosticBag()
        for item in cleaned:
            diagnostics.extend(item.diagnostics)
        (
            centreline, laps, matched, profile, event_projection, response_features,
            event_passes, gate_evidence, gate_review, rejected_map_points,
        ) = build_track_evidence(cleaned, config, raw_events, diagnostics)
        case_resolution = deepcopy(resolution)
        case_resolution.data["track"] = config
        build = TrackBuildResult(
            resolution=case_resolution,
            ingestion_results=cleaned,
            centreline=centreline,
            laps=laps,
            matched_points=matched,
            track_profile=profile,
            event_projection=event_projection,
            response_features=response_features,
            event_passes=event_passes,
            gate_evidence=gate_evidence,
            gate_review=gate_review,
            rejected_map_points=rejected_map_points,
            diagnostics=diagnostics.items,
            metadata={
                "schema_version": 1,
                "phase": "track_robustness_case",
                "case_id": case.identifier,
                "track_length_m": float(centreline.length_m),
                "valid_lap_count": int(laps["analysis_valid"].sum()),
                "accepted_gate_count": int(
                    (gate_review["recommendation"] == "accepted").sum()
                ),
            },
        )
        return CaseResult(
            case=case,
            success=True,
            error="",
            centreline=centreline,
            laps=laps,
            event_projection=event_projection,
            gate_review=gate_review,
            diagnostics=diagnostics.items,
            rejected_map_points=rejected_map_points,
            track_build=build,
        )
    except Exception as exc:  # a failed robustness case is itself reportable evidence
        return CaseResult(case=case, success=False, error=f"{type(exc).__name__}: {exc}")


def _parse_runs(resolution: Any, raw_runs: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    parsed = []
    for raw in raw_runs:
        metadata = GPXRunMetadata(
            run_id=str(raw["run_id"]),
            vehicle_id=str(raw["vehicle_id"]),
            driver_id=str(raw["driver_id"]),
            source_file=(resolution.paths.runs_file.parent / str(raw["file"])).resolve(),
            use_for_centreline=bool(raw["use_for_centreline"]),
            use_for_gate_evidence=bool(raw["use_for_gate_evidence"]),
        )
        try:
            parsed.append(ingest_telemetry_run(metadata))
        except (TelemetryParseError, ValueError) as exc:
            raise ProjectError(str(exc)) from exc
    return tuple(parsed)


def _write_case_artifacts(directory: Path, result: CaseResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "case_manifest.json", {**case_to_dict(result.case), "success": result.success, "error": result.error})
    if not result.success:
        return
    latitude, longitude = result.centreline.frame.to_latlon(result.centreline.x_m, result.centreline.y_m)
    pd.DataFrame({"s_m": result.centreline.s_m, "latitude_deg": latitude, "longitude_deg": longitude}).to_csv(directory / "centreline.csv", index=False)
    result.laps.to_csv(directory / "lap_quality.csv", index=False)
    result.event_projection.to_csv(directory / "event_projection.csv", index=False)
    result.gate_review.to_csv(directory / "gate_review.csv", index=False)
    write_json(directory / "diagnostics.json", [item.to_dict() for item in result.diagnostics])
    if result.track_build is not None:
        from cvt_track_study.bundle import export_bundle_for_track_build

        export_bundle_for_track_build(directory, result.track_build)


def _centreline_displacement(nominal: Any, candidate: Any, nodes: int = 500) -> list[tuple[float, float]]:
    progress = np.linspace(0.0, 1.0, nodes, endpoint=False)
    nominal_lat, nominal_lon = nominal.frame.to_latlon(nominal.x_m, nominal.y_m)
    candidate_lat, candidate_lon = candidate.frame.to_latlon(candidate.x_m, candidate.y_m)
    nominal_x, nominal_y = nominal.frame.to_xy(nominal_lat, nominal_lon)
    candidate_x_in_nominal, candidate_y_in_nominal = nominal.frame.to_xy(candidate_lat, candidate_lon)
    nominal_u = np.asarray(nominal.s_m, float) / float(nominal.length_m)
    candidate_u = np.asarray(candidate.s_m, float) / float(candidate.length_m)
    nx = np.interp(progress, nominal_u, nominal_x)
    ny = np.interp(progress, nominal_u, nominal_y)
    cx = np.interp(progress, candidate_u, candidate_x_in_nominal)
    cy = np.interp(progress, candidate_u, candidate_y_in_nominal)
    shifts = np.hypot(cx - nx, cy - ny)
    return list(zip(progress.tolist(), shifts.tolist()))


def _enrich_case_summary(
    cases: pd.DataFrame,
    event_cases: pd.DataFrame,
    gate_cases: pd.DataFrame,
    nominal_gate_review: pd.DataFrame,
    thresholds: Mapping[str, float],
) -> pd.DataFrame:
    """Add case-level geometry, gate-set, and downstream-eligibility evidence."""

    if cases.empty:
        return cases
    result = cases.copy()
    category_by_case = result.set_index("case_id")["category"].astype(str).to_dict()
    label_by_case = result.set_index("case_id")["label"].astype(str).to_dict()
    for frame in (event_cases, gate_cases):
        if frame.empty or "case_id" not in frame:
            continue
        if "case_category" not in frame:
            frame["case_category"] = frame["case_id"].map(category_by_case)
        if "case_label" not in frame:
            frame["case_label"] = frame["case_id"].map(label_by_case)

    nominal_length_values = pd.to_numeric(
        result.get("track_length_m", pd.Series(dtype=float)), errors="coerce"
    ) - pd.to_numeric(
        result.get("track_length_delta_m", pd.Series(dtype=float)), errors="coerce"
    )
    nominal_length_values = nominal_length_values.dropna()
    nominal_length = float(nominal_length_values.median()) if len(nominal_length_values) else math.nan
    if "track_length_relative_shift" not in result:
        result["track_length_relative_shift"] = (
            pd.to_numeric(result.get("track_length_delta_m"), errors="coerce").abs()
            / nominal_length
        )
    result["track_length_stable"] = (
        pd.to_numeric(result["track_length_relative_shift"], errors="coerce")
        <= float(thresholds["maximum_track_length_relative_shift"])
    )

    if not event_cases.empty:
        event_summary = (
            event_cases.groupby("case_id", as_index=False)["anchor_shift_m"]
            .agg(
                event_anchor_shift_p95_m=lambda values: float(
                    pd.to_numeric(values, errors="coerce").quantile(0.95)
                ),
                maximum_event_anchor_shift_m=lambda values: float(
                    pd.to_numeric(values, errors="coerce").max()
                ),
            )
        )
        result = result.drop(
            columns=[
                column
                for column in (
                    "event_anchor_shift_p95_m",
                    "maximum_event_anchor_shift_m",
                    "event_projection_stable",
                )
                if column in result
            ]
        ).merge(event_summary, on="case_id", how="left")
        result["event_projection_stable"] = (
            pd.to_numeric(result["event_anchor_shift_p95_m"], errors="coerce")
            <= float(thresholds["maximum_event_p95_shift_m"])
        )

    nominal_index = nominal_gate_review.set_index("response_group_id", drop=False)
    nominal_accepted = {
        str(gate_id)
        for gate_id, row in nominal_index.iterrows()
        if str(row.get("recommendation", "")) == "accepted"
    }
    gate_metrics: list[dict[str, Any]] = []
    if not gate_cases.empty:
        for case_id, group in gate_cases.groupby("case_id", sort=False):
            accepted = {
                str(value)
                for value in group.loc[group["accepted"].astype(bool), "gate_id"]
            }
            union = nominal_accepted | accepted
            intersection = nominal_accepted & accepted
            jaccard = len(intersection) / len(union) if union else 1.0
            retention = (
                len(intersection) / len(nominal_accepted) if nominal_accepted else 1.0
            )
            new_count = len(accepted - nominal_accepted)
            gate_metrics.append(
                {
                    "case_id": case_id,
                    "accepted_gate_count": len(accepted),
                    "nominal_gate_retention_fraction": retention,
                    "gate_set_jaccard": jaccard,
                    "newly_accepted_gate_count": new_count,
                    "gate_set_stable_for_ensemble": bool(
                        jaccard
                        >= float(thresholds["minimum_gate_set_jaccard_for_ensemble"])
                        and retention
                        >= float(
                            thresholds[
                                "minimum_nominal_gate_retention_fraction_for_ensemble"
                            ]
                        )
                        and new_count
                        <= int(thresholds["maximum_new_gate_count_for_ensemble"])
                    ),
                }
            )
    if gate_metrics:
        metrics = pd.DataFrame(gate_metrics)
        result = result.drop(
            columns=[column for column in metrics.columns if column != "case_id" and column in result]
        ).merge(metrics, on="case_id", how="left")

    result["stress_only"] = [
        _is_stress_only_case(str(row.case_id), str(row.category))
        for row in result.itertuples()
    ]
    return result


def _is_stress_only_case(case_id: str, category: str) -> bool:
    return category == "data_support" or case_id in {"gate_strict", "gate_permissive"}


def _aggregate_event_stability(
    frame: pd.DataFrame, thresholds: Mapping[str, float]
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for event_id, group in frame.groupby("event_id", sort=False):
        category = group.get("case_category", pd.Series("", index=group.index)).astype(str)
        primary = group[category.isin(_PRIMARY_GEOMETRY_CATEGORIES)]
        geometry = group[category.isin(_ALL_GEOMETRY_CATEGORIES)]
        if primary.empty:
            primary = geometry if not geometry.empty else group
        if geometry.empty:
            geometry = group
        primary_shift = pd.to_numeric(primary["anchor_shift_m"], errors="coerce").dropna()
        all_shift = pd.to_numeric(geometry["anchor_shift_m"], errors="coerce").dropna()
        delta = pd.to_numeric(geometry["interval_length_delta_m"], errors="coerce").dropna()
        p95 = float(primary_shift.quantile(0.95)) if len(primary_shift) else math.nan
        rows.append(
            {
                "event_id": event_id,
                "event_name": group["event_name"].iloc[0],
                "geometry_case_count": len(geometry),
                "primary_geometry_case_count": len(primary),
                "anchor_shift_primary_median_m": (
                    float(primary_shift.median()) if len(primary_shift) else math.nan
                ),
                "anchor_shift_primary_p95_m": p95,
                "anchor_shift_geometry_max_m": (
                    float(all_shift.max()) if len(all_shift) else math.nan
                ),
                "interval_length_delta_p10_m": (
                    float(delta.quantile(0.1)) if len(delta) else math.nan
                ),
                "interval_length_delta_median_m": (
                    float(delta.median()) if len(delta) else math.nan
                ),
                "interval_length_delta_p90_m": (
                    float(delta.quantile(0.9)) if len(delta) else math.nan
                ),
                "projection_stable": bool(
                    np.isfinite(p95)
                    and p95 <= float(thresholds["maximum_event_p95_shift_m"])
                ),
                # Compatibility aliases retained for downstream notebooks.
                "anchor_shift_median_m": (
                    float(primary_shift.median()) if len(primary_shift) else math.nan
                ),
                "anchor_shift_p95_m": p95,
                "anchor_shift_max_m": (
                    float(all_shift.max()) if len(all_shift) else math.nan
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("anchor_shift_primary_p95_m", ascending=False)
        .reset_index(drop=True)
    )


def _aggregate_gate_stability(
    frame: pd.DataFrame,
    thresholds: Mapping[str, float],
    nominal_gate_review: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    nominal_index = nominal_gate_review.set_index("response_group_id", drop=False)
    accept_score = float(
        nominal_gate_review.attrs.get("accept_score", math.nan)
    )
    if not np.isfinite(accept_score):
        accepted_scores = pd.to_numeric(
            nominal_gate_review.loc[
                nominal_gate_review["recommendation"] == "accepted",
                "overall_confidence_score",
            ],
            errors="coerce",
        ).dropna()
        accept_score = 60.0 if accepted_scores.empty else min(60.0, float(accepted_scores.min()))

    rows: list[dict[str, Any]] = []
    for gate_id, group in frame.groupby("gate_id", sort=False):
        nominal_row = nominal_index.loc[gate_id] if gate_id in nominal_index.index else None
        if nominal_row is not None and not bool(nominal_row.get("gate_candidate", True)):
            continue
        definition = group[
            group.get("case_category", pd.Series("", index=group.index))
            .astype(str)
            .isin(_GATE_DEFINITION_CATEGORIES)
        ]
        if definition.empty:
            definition = group
        score = pd.to_numeric(group["overall_confidence_score"], errors="coerce").dropna()
        speed = pd.to_numeric(group["entry_speed_median_mps"], errors="coerce").dropna()
        definition_accepted = float(definition["accepted"].astype(bool).mean())
        definition_agreement = float(
            definition["classification_matches_nominal"].astype(bool).mean()
        )
        all_agreement = float(group["classification_matches_nominal"].astype(bool).mean())
        speed_range = (
            float(speed.quantile(0.9) - speed.quantile(0.1)) if len(speed) else math.nan
        )
        nominal_recommendation = (
            str(nominal_row.get("recommendation", "missing"))
            if nominal_row is not None
            else str(group["nominal_recommendation"].iloc[0])
        )
        if nominal_recommendation == "accepted":
            frontier = (
                "core"
                if definition_accepted
                >= float(thresholds["core_gate_definition_acceptance_fraction"])
                else "conditional"
            )
        elif definition_accepted >= float(
            thresholds["near_miss_gate_definition_acceptance_fraction"]
        ):
            frontier = "near_miss"
        else:
            frontier = "unsupported"

        nominal_speed = (
            float(nominal_row.get("entry_speed_median_mps", math.nan))
            if nominal_row is not None
            else math.nan
        )
        nominal_score = (
            float(nominal_row.get("overall_confidence_score", math.nan))
            if nominal_row is not None
            else math.nan
        )
        braking_score = (
            float(nominal_row.get("braking_evidence_score", math.nan))
            if nominal_row is not None
            else math.nan
        )
        slowdown = (
            float(nominal_row.get("median_approach_to_min_slowdown_mps", math.nan))
            if nominal_row is not None
            else math.nan
        )
        speed_kmh = nominal_speed * 3.6 if np.isfinite(nominal_speed) else math.nan
        high_speed = bool(
            np.isfinite(speed_kmh)
            and speed_kmh >= float(thresholds["high_speed_gate_review_kmh"])
        )
        weak_braking = bool(
            not np.isfinite(braking_score)
            or braking_score
            < float(
                thresholds["minimum_braking_evidence_score_for_high_speed_gate"]
            )
        )
        cap_review = bool(high_speed and weak_braking)
        if cap_review:
            interpretation = (
                "High measured speed with weak repeatable braking; review whether the "
                "observation is vehicle-speed-limited rather than a feature constraint."
            )
        elif high_speed:
            interpretation = (
                "High measured speed, but repeatable braking evidence supports treating "
                "the feature as a genuine constraint."
            )
        else:
            interpretation = "Measured speed is below the high-speed review threshold."

        rows.append(
            {
                "gate_id": gate_id,
                "gate_name": group["gate_name"].iloc[0],
                "frontier_classification": frontier,
                "nominal_recommendation": nominal_recommendation,
                "nominal_confidence_score": nominal_score,
                "nominal_score_margin_to_accept": (
                    nominal_score - accept_score if np.isfinite(nominal_score) else math.nan
                ),
                "gate_definition_case_count": len(definition),
                "gate_definition_accepted_fraction": definition_accepted,
                "gate_definition_classification_agreement_fraction": definition_agreement,
                "all_case_count": len(group),
                "all_case_classification_agreement_fraction": all_agreement,
                "score_min": float(score.min()) if len(score) else math.nan,
                "score_median": float(score.median()) if len(score) else math.nan,
                "score_max": float(score.max()) if len(score) else math.nan,
                "nominal_target_speed_mps": nominal_speed,
                "nominal_target_speed_kmh": speed_kmh,
                "target_speed_p10_mps": (
                    float(speed.quantile(0.1)) if len(speed) else math.nan
                ),
                "target_speed_median_mps": (
                    float(speed.median()) if len(speed) else math.nan
                ),
                "target_speed_p90_mps": (
                    float(speed.quantile(0.9)) if len(speed) else math.nan
                ),
                "target_speed_p80_range_mps": speed_range,
                "target_speed_p80_range_kmh": (
                    speed_range * 3.6 if np.isfinite(speed_range) else math.nan
                ),
                "nominal_braking_evidence_score": braking_score,
                "nominal_approach_to_min_slowdown_mps": slowdown,
                "nominal_approach_to_min_slowdown_kmh": (
                    slowdown * 3.6 if np.isfinite(slowdown) else math.nan
                ),
                "nominal_slowdown_lap_fraction": (
                    float(nominal_row.get("slowdown_lap_fraction", math.nan))
                    if nominal_row is not None
                    else math.nan
                ),
                "nominal_pace_correlation": (
                    float(nominal_row.get("pace_correlation", math.nan))
                    if nominal_row is not None
                    else math.nan
                ),
                "high_speed_weak_braking_review": cap_review,
                "speed_interpretation": interpretation,
                "nominal_reasons": (
                    str(nominal_row.get("reasons", ""))
                    if nominal_row is not None
                    else ""
                ),
                "suggested_action": (
                    str(nominal_row.get("suggested_action", ""))
                    if nominal_row is not None
                    else ""
                ),
                "target_speed_stable": bool(
                    not np.isfinite(speed_range)
                    or speed_range
                    <= float(thresholds["maximum_gate_target_p80_range_mps"])
                ),
                "status": frontier,
            }
        )
    order = {"core": 0, "conditional": 1, "near_miss": 2, "unsupported": 3}
    result = pd.DataFrame(rows)
    result["_order"] = result["frontier_classification"].map(order)
    return (
        result.sort_values(
            ["_order", "gate_definition_accepted_fraction", "nominal_confidence_score"],
            ascending=[True, False, False],
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def _write_centreline_overlay(
    output: Path,
    nominal: TrackBuildResult,
    case_results: Sequence[CaseResult],
) -> None:
    figure_obj, axis = plt.subplots(figsize=(11, 8.5))
    nominal_lat, nominal_lon = nominal.centreline.frame.to_latlon(
        nominal.centreline.x_m, nominal.centreline.y_m
    )
    nominal_x, nominal_y = nominal.centreline.frame.to_xy(nominal_lat, nominal_lon)
    for result in case_results:
        if not result.success or result.centreline is None:
            continue
        if result.case.category not in _ALL_GEOMETRY_CATEGORIES:
            continue
        latitude, longitude = result.centreline.frame.to_latlon(
            result.centreline.x_m, result.centreline.y_m
        )
        x_m, y_m = nominal.centreline.frame.to_xy(latitude, longitude)
        axis.plot(x_m, y_m, linewidth=0.75, alpha=0.24)
    axis.plot(nominal_x, nominal_y, linewidth=2.2, label="nominal consensus")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Local east [m]")
    axis.set_ylabel("Local north [m]")
    axis.set_title("Top-down centreline alternatives that can change geometry")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure_obj.tight_layout()
    figure_obj.savefig(output / "centreline_overlay.png", dpi=180)
    plt.close(figure_obj)


def _write_centreline_stability_corridor(
    output: Path, cases: pd.DataFrame, nodes: int = 500
) -> None:
    """Write a signed p10-p90 top-down corridor from retained centreline files."""

    nominal_path = output / "nominal_track" / "track" / "centreline.csv"
    if not nominal_path.is_file() or cases.empty:
        return
    nominal = pd.read_csv(nominal_path)
    required = {"s_m", "latitude_deg", "longitude_deg"}
    if not required <= set(nominal.columns):
        return
    progress = np.linspace(0.0, 1.0, nodes, endpoint=False)
    nominal_x, nominal_y = _latlon_to_local_xy(
        nominal["latitude_deg"].to_numpy(float),
        nominal["longitude_deg"].to_numpy(float),
        float(nominal["latitude_deg"].iloc[0]),
        float(nominal["longitude_deg"].iloc[0]),
    )
    nominal_u = nominal["s_m"].to_numpy(float) / float(nominal["s_m"].iloc[-1])
    nx = np.interp(progress, nominal_u, nominal_x)
    ny = np.interp(progress, nominal_u, nominal_y)
    tangent_x = np.gradient(nx)
    tangent_y = np.gradient(ny)
    norm = np.hypot(tangent_x, tangent_y)
    norm[norm <= 1e-12] = 1.0
    normal_x = -tangent_y / norm
    normal_y = tangent_x / norm

    offsets_by_case: dict[str, np.ndarray] = {}
    candidates_xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in cases.itertuples():
        if not bool(getattr(row, "success", False)):
            continue
        if str(getattr(row, "category", "")) not in _ALL_GEOMETRY_CATEGORIES:
            continue
        path = output / "cases" / _safe_name(str(row.case_id)) / "centreline.csv"
        if not path.is_file():
            continue
        candidate = pd.read_csv(path)
        if not required <= set(candidate.columns):
            continue
        cx_raw, cy_raw = _latlon_to_local_xy(
            candidate["latitude_deg"].to_numpy(float),
            candidate["longitude_deg"].to_numpy(float),
            float(nominal["latitude_deg"].iloc[0]),
            float(nominal["longitude_deg"].iloc[0]),
        )
        candidate_u = candidate["s_m"].to_numpy(float) / float(candidate["s_m"].iloc[-1])
        cx = np.interp(progress, candidate_u, cx_raw)
        cy = np.interp(progress, candidate_u, cy_raw)
        offsets_by_case[str(row.case_id)] = (cx - nx) * normal_x + (cy - ny) * normal_y
        candidates_xy[str(row.case_id)] = (cx, cy)

    if not offsets_by_case:
        return
    case_index = cases.set_index("case_id", drop=False)
    primary_ids = [
        case_id
        for case_id in offsets_by_case
        if str(case_index.loc[case_id].get("category", ""))
        in _PRIMARY_GEOMETRY_CATEGORIES
        and bool(case_index.loc[case_id].get("centreline_stable", False))
        and bool(case_index.loc[case_id].get("track_length_stable", False))
    ]
    if not primary_ids:
        primary_ids = [
            case_id
            for case_id in offsets_by_case
            if str(case_index.loc[case_id].get("category", ""))
            in _PRIMARY_GEOMETRY_CATEGORIES
        ]
    all_ids = list(offsets_by_case)
    primary_values = np.vstack([offsets_by_case[case_id] for case_id in primary_ids])
    all_values = np.vstack([offsets_by_case[case_id] for case_id in all_ids])
    primary_p10 = np.nanquantile(primary_values, 0.10, axis=0)
    primary_p90 = np.nanquantile(primary_values, 0.90, axis=0)
    all_p10 = np.nanquantile(all_values, 0.10, axis=0)
    all_p90 = np.nanquantile(all_values, 0.90, axis=0)

    lower_x = nx + primary_p10 * normal_x
    lower_y = ny + primary_p10 * normal_y
    upper_x = nx + primary_p90 * normal_x
    upper_y = ny + primary_p90 * normal_y
    all_lower_x = nx + all_p10 * normal_x
    all_lower_y = ny + all_p10 * normal_y
    all_upper_x = nx + all_p90 * normal_x
    all_upper_y = ny + all_p90 * normal_y

    corridor = pd.DataFrame(
        {
            "progress_fraction": progress,
            "nominal_s_m": progress * float(nominal["s_m"].iloc[-1]),
            "nominal_x_m": nx,
            "nominal_y_m": ny,
            "primary_normal_offset_p10_m": primary_p10,
            "primary_normal_offset_p90_m": primary_p90,
            "all_geometry_normal_offset_p10_m": all_p10,
            "all_geometry_normal_offset_p90_m": all_p90,
        }
    )
    corridor.to_csv(output / "centreline_stability_corridor.csv", index=False)

    figure_obj, axis = plt.subplots(figsize=(11.5, 8.8))
    polygon_x = np.concatenate([lower_x, upper_x[::-1]])
    polygon_y = np.concatenate([lower_y, upper_y[::-1]])
    axis.fill(
        polygon_x,
        polygon_y,
        alpha=0.28,
        label=f"p10-p90 stable reconstruction corridor ({len(primary_ids)} cases)",
    )
    axis.plot(
        all_lower_x,
        all_lower_y,
        linestyle="--",
        linewidth=0.9,
        alpha=0.75,
        label="p10-p90 including data-support stress cases",
    )
    axis.plot(all_upper_x, all_upper_y, linestyle="--", linewidth=0.9, alpha=0.75)
    axis.plot(nx, ny, linewidth=2.2, label="nominal consensus")
    axis.scatter([nx[0]], [ny[0]], marker="o", s=36, zorder=5, label="start / finish")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Local east [m]")
    axis.set_ylabel("Local north [m]")
    axis.set_title("Top-down centreline stability corridor")
    axis.grid(True, alpha=0.22)
    axis.legend(loc="best")
    figure_obj.tight_layout()
    figure_obj.savefig(output / "centreline_stability_corridor.png", dpi=180)
    plt.close(figure_obj)


def _latlon_to_local_xy(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius_m = 6_371_000.0
    latitude = np.radians(np.asarray(latitude_deg, dtype=float))
    longitude = np.radians(np.asarray(longitude_deg, dtype=float))
    origin_latitude = math.radians(origin_latitude_deg)
    origin_longitude = math.radians(origin_longitude_deg)
    x = radius_m * math.cos(origin_latitude) * (longitude - origin_longitude)
    y = radius_m * (latitude - origin_latitude)
    return x, y


def _write_robustness_plots(output: Path, tables: Mapping[str, pd.DataFrame]) -> None:
    cases = tables["robustness_case_summary"]
    displacement = tables["centreline_displacement"]
    events = tables["event_stability"]
    gates = tables["gate_stability"]
    gate_cases = tables["gate_case_results"]

    if not displacement.empty:
        category_by_case = cases.set_index("case_id")["category"].astype(str).to_dict()
        primary_ids = [
            case_id
            for case_id in displacement["case_id"].unique()
            if category_by_case.get(case_id) in _PRIMARY_GEOMETRY_CATEGORIES
        ]
        stress_ids = [
            case_id
            for case_id in displacement["case_id"].unique()
            if category_by_case.get(case_id) in _ALL_GEOMETRY_CATEGORIES
        ]
        primary = displacement[displacement["case_id"].isin(primary_ids)]
        stress = displacement[displacement["case_id"].isin(stress_ids)]
        if primary.empty:
            primary = stress
        pivot = primary.pivot(index="nominal_s_m", columns="case_id", values="shift_m").sort_index()
        x = pivot.index.to_numpy(float)
        values = pivot.to_numpy(float)
        fig, ax = plt.subplots(figsize=(12, 5.6))
        ax.fill_between(
            x,
            0.0,
            np.nanquantile(values, 0.90, axis=1),
            alpha=0.23,
            label="p90 stable reconstruction envelope",
        )
        ax.plot(x, np.nanmedian(values, axis=1), linewidth=1.7, label="median")
        ax.plot(x, np.nanquantile(values, 0.90, axis=1), linewidth=1.4, label="p90")
        if not stress.empty:
            stress_pivot = stress.pivot(index="nominal_s_m", columns="case_id", values="shift_m").sort_index()
            ax.plot(
                stress_pivot.index.to_numpy(float),
                np.nanquantile(stress_pivot.to_numpy(float), 0.90, axis=1),
                linestyle="--",
                linewidth=1.0,
                label="p90 including data-support stress cases",
            )
        ax.set_xlabel("Nominal along-track coordinate, s [m]")
        ax.set_ylabel("Centreline displacement [m]")
        ax.set_title("Local centreline displacement under geometry-affecting choices")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "centreline_stability.png", dpi=180)
        plt.close(fig)

    successful = cases.loc[cases["success"].astype(bool)].copy() if not cases.empty else cases
    if not successful.empty and "track_length_delta_m" in successful:
        successful["display_label"] = successful.apply(
            lambda row: str(row["label"]) + (" [stress only]" if bool(row.get("stress_only", False)) else ""),
            axis=1,
        )
        data = successful.sort_values("track_length_delta_m")
        fig, ax = plt.subplots(figsize=(11, max(5.5, 0.38 * len(data))))
        ax.barh(data["display_label"], data["track_length_delta_m"])
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Track length change from nominal [m]")
        ax.set_title("Reconstructed track-length sensitivity")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "track_length_sensitivity.png", dpi=180)
        plt.close(fig)

    if not events.empty:
        data = events.head(30).sort_values("anchor_shift_primary_p95_m")
        y = np.arange(len(data))
        fig, ax = plt.subplots(figsize=(11, max(5.5, 0.38 * len(data))))
        ax.barh(y, data["anchor_shift_primary_p95_m"], label="p95 stable reconstruction shift")
        ax.scatter(data["anchor_shift_geometry_max_m"], y, marker="|", s=70, label="worst geometry stress case")
        ax.set_yticks(y, data["event_name"])
        ax.set_xlabel("Movement on common s coordinate [m]")
        ax.set_title("Event projection stability")
        ax.grid(True, axis="x", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "event_projection_stability.png", dpi=180)
        plt.close(fig)

    if not gates.empty:
        frontier_order = {"core": 0, "conditional": 1, "near_miss": 2, "unsupported": 3}
        data = gates.copy()
        data["_order"] = data["frontier_classification"].map(frontier_order)
        data = data.sort_values(["_order", "gate_definition_accepted_fraction"]).drop(columns="_order")
        fig, ax = plt.subplots(figsize=(11, max(6.0, 0.4 * len(data))))
        ax.barh(data["gate_name"], data["gate_definition_accepted_fraction"])
        ax.axvline(
            float(_DEFAULT_THRESHOLDS["core_gate_definition_acceptance_fraction"]),
            linestyle="--",
            linewidth=1,
            label="core threshold",
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Accepted fraction across gate-definition cases")
        ax.set_title("Gate frontier: stable core, conditional gates, and near misses")
        ax.grid(True, axis="x", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "gate_qualification_stability.png", dpi=180)
        plt.close(fig)

        frontier = gates.dropna(
            subset=["nominal_score_margin_to_accept", "gate_definition_accepted_fraction"]
        ).copy()
        if not frontier.empty:
            fig, ax = plt.subplots(figsize=(11, 7.2))
            ax.scatter(
                frontier["nominal_score_margin_to_accept"],
                frontier["gate_definition_accepted_fraction"],
                s=48,
                alpha=0.8,
            )
            for row in frontier.itertuples():
                if row.frontier_classification != "unsupported" or abs(float(row.nominal_score_margin_to_accept)) <= 5.0:
                    ax.annotate(
                        str(row.gate_id),
                        (float(row.nominal_score_margin_to_accept), float(row.gate_definition_accepted_fraction)),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                    )
            ax.axvline(0.0, linewidth=1)
            ax.axhline(
                float(_DEFAULT_THRESHOLDS["core_gate_definition_acceptance_fraction"]),
                linestyle="--",
                linewidth=1,
            )
            ax.axhline(
                float(_DEFAULT_THRESHOLDS["near_miss_gate_definition_acceptance_fraction"]),
                linestyle=":",
                linewidth=1,
            )
            ax.set_xlabel("Nominal confidence-score margin above acceptance cutoff")
            ax.set_ylabel("Accepted fraction across gate-definition cases")
            ax.set_ylim(-0.03, 1.03)
            ax.set_title("Gate qualification frontier")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(output / "gate_frontier.png", dpi=180)
            plt.close(fig)

        speed = gates.dropna(subset=["nominal_target_speed_kmh"]).copy().sort_values(
            "nominal_target_speed_kmh"
        )
        if not speed.empty:
            y = np.arange(len(speed))
            med = speed["target_speed_median_mps"].to_numpy(float) * 3.6
            low = speed["target_speed_p10_mps"].to_numpy(float) * 3.6
            high = speed["target_speed_p90_mps"].to_numpy(float) * 3.6
            fig, ax = plt.subplots(figsize=(11, max(6.0, 0.4 * len(speed))))
            ax.errorbar(med, y, xerr=np.vstack((med - low, high - med)), fmt="o", capsize=3)
            review = speed["high_speed_weak_braking_review"].astype(bool).to_numpy()
            if review.any():
                ax.scatter(med[review], y[review], marker="x", s=70, label="high speed + weak braking review")
            ax.axvline(
                float(_DEFAULT_THRESHOLDS["high_speed_gate_review_kmh"]),
                linestyle="--",
                linewidth=1,
                label="high-speed review threshold",
            )
            ax.set_yticks(y, speed["gate_name"])
            ax.set_xlabel("Entry target speed [km/h]")
            ax.set_title("Gate target-speed stability and high-speed review")
            ax.grid(True, axis="x", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / "gate_target_speed_stability.png", dpi=180)
            plt.close(fig)

    if not gate_cases.empty:
        definition = gate_cases[
            gate_cases.get("case_category", pd.Series("", index=gate_cases.index))
            .astype(str)
            .isin(_GATE_DEFINITION_CATEGORIES)
        ]
        if not definition.empty:
            ordered_gate_ids = gates["gate_id"].astype(str).tolist() if not gates.empty else None
            matrix = definition.pivot_table(
                index="case_label",
                columns="gate_id",
                values="case_recommendation",
                aggfunc="first",
            )
            if ordered_gate_ids:
                matrix = matrix.reindex(columns=[gate for gate in ordered_gate_ids if gate in matrix.columns])
            numeric = matrix.map(lambda value: _STATUS_CODE.get(str(value), -2))
            fig, ax = plt.subplots(
                figsize=(max(11, 0.4 * len(matrix.columns)), max(5.5, 0.42 * len(matrix.index)))
            )
            image = ax.imshow(numeric.to_numpy(), aspect="auto", interpolation="nearest", vmin=-2, vmax=3)
            ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=90)
            ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
            ax.set_title("Gate classification under threshold, weighting, and window choices")
            fig.colorbar(image, ax=ax, label="accepted 3 · review 2 · must-fix 1 · rejected 0")
            fig.tight_layout()
            fig.savefig(output / "gate_policy_matrix.png", dpi=180)
            plt.close(fig)


def _write_track_robustness_html(
    output: Path,
    *,
    nominal_track_length_m: float,
    tables: Mapping[str, pd.DataFrame],
    thresholds: Mapping[str, float],
    ensemble_manifest: Mapping[str, Any],
) -> None:
    cases = tables["robustness_case_summary"]
    events = tables["event_stability"]
    gates = tables["gate_stability"]
    gate_cases = tables["gate_case_results"]
    event_cases = tables["event_case_results"]
    eligibility = tables.get("ensemble_eligibility", pd.DataFrame())

    primary_geometry = cases[
        cases["category"].astype(str).isin(_PRIMARY_GEOMETRY_CATEGORIES)
        & cases["success"].astype(bool)
    ] if not cases.empty else cases
    primary_p95 = (
        float(primary_geometry["centreline_shift_p95_m"].max())
        if not primary_geometry.empty
        else math.nan
    )
    primary_length = (
        float(primary_geometry["track_length_relative_shift"].max())
        if not primary_geometry.empty
        else math.nan
    )
    counts = gates["frontier_classification"].value_counts().to_dict() if not gates.empty else {}
    high_speed_review = int(gates["high_speed_weak_braking_review"].astype(bool).sum()) if not gates.empty else 0
    target_stable = int(gates["target_speed_stable"].astype(bool).sum()) if not gates.empty else 0
    eligible_count = len(ensemble_manifest.get("eligible_cases", ()))

    nav = (
        '<nav class="report-nav" aria-label="Report sections">'
        '<a href="#summary">Summary</a><a href="#centreline">Centreline</a>'
        '<a href="#events">Events</a><a href="#gates">Gates</a>'
        '<a href="#ensemble">Track ensemble</a><a href="#appendix">Detailed tables</a></nav>'
    )
    body = nav
    body += (
        '<div class="scope">'
        f'<div class="card note"><div class="label">Question</div><div>{html.escape(REPORTS["track_robustness"].question)}</div></div>'
        '<div class="card"><div class="label">Held fixed</div><div>Raw telemetry and reviewed physical-event declarations. No vehicle or drivetrain simulation is run.</div></div>'
        '<div class="card"><div class="label">Varied</div><div>Cleanup, centreline, event-window, gate-threshold, confidence-weight, and limited data-support choices.</div></div></div>'
    )
    body += '<h2 id="summary">Executive summary</h2>'
    body += '<div class="section-intro"><strong>What this section shows.</strong>The headline answer separates stable measured geometry and target speeds from the more policy-sensitive decision to promote an observation into a hard gate.</div>'
    body += metric_cards(
        [
            (
                "Primary geometry p95 shift",
                f"{primary_p95:.2f} m" if np.isfinite(primary_p95) else "n/a",
                "good" if np.isfinite(primary_p95) and primary_p95 <= thresholds["maximum_centreline_p95_shift_m"] else "warning",
            ),
            (
                "Largest primary length change",
                f"{100.0 * primary_length:.2f}%" if np.isfinite(primary_length) else "n/a",
                "good" if np.isfinite(primary_length) and primary_length <= thresholds["maximum_track_length_relative_shift"] else "warning",
            ),
            ("Core gates", str(int(counts.get("core", 0))), "good"),
            ("Conditional nominal gates", str(int(counts.get("conditional", 0))), "warning"),
            ("Near-miss candidates", str(int(counts.get("near_miss", 0))), "warning"),
            ("High-speed weak-braking reviews", str(high_speed_review), "warning" if high_speed_review else "good"),
            ("Stable target-speed ranges", f"{target_stable}/{len(gates)}", "good" if target_stable == len(gates) else "warning"),
            ("Downstream track cases", str(eligible_count), "note"),
        ]
    )
    body += (
        '<div class="finding-grid">'
        '<div class="finding"><strong>Centreline</strong>The ordinary cleanup and reconstruction alternatives are summarized as a top-down p10-p90 corridor. Leave-one-run/vehicle/driver cases remain visible as stress tests but do not dominate the primary corridor.</div>'
        '<div class="finding"><strong>Event placement</strong>Movement is measured on the shared s coordinate because it determines where entry, response, obstacle, and recovery windows are sampled. Physical extent stability is reported separately.</div>'
        '<div class="finding"><strong>Gate evidence</strong>Target-speed repeatability and gate qualification are intentionally separated. A stable speed observation can still be a conditional gate if its acceptance depends on cutoff or weighting choices.</div>'
        '<div class="finding"><strong>No fake probabilities</strong>Acceptance fractions are frequencies across selected analysis policies. They are not probabilities that a gate is true or false.</div>'
        '</div>'
    )

    body += '<h2 id="centreline">1. Centreline and reconstructed length</h2>'
    body += '<div class="section-intro"><strong>What this section tests.</strong>Whether reasonable telemetry cleanup, consensus, smoothing, and discretization choices move the measured driving line or materially change the reconstructed course length.</div>'
    body += figure(
        output / "centreline_stability_corridor.png",
        "Top-down signed p10-p90 corridor from stable cleanup/centreline cases. Dashed boundaries additionally include data-support stress cases. This shows where the inferred line is wide or narrow directly on the map.",
    )
    body += figure(
        output / "centreline_stability.png",
        "Local displacement versus s using only geometry-affecting cases; gate-only cases are excluded so zeros cannot artificially narrow the envelope.",
    )
    body += figure(
        output / "track_length_sensitivity.png",
        "Change in reconstructed track length under every successful case. Stress-only cases are labelled separately.",
    )
    centreline_columns = [
        "case_id", "label", "category", "track_length_delta_m",
        "track_length_relative_shift", "centreline_shift_p95_m",
        "centreline_shift_max_m", "track_length_stable", "centreline_stable",
        "stress_only",
    ]
    body += '<p class="table-note"><strong>Underlying summary.</strong> Key geometry outcomes by analysis case. Click any header to cycle ascending, descending, and original order.</p>'
    body += dataframe_table(
        cases,
        columns=centreline_columns,
        sticky_columns=("case_id", "label"),
        max_rows=250,
        compact=True,
        table_id="centreline-case-summary",
    )

    body += '<h2 id="events">2. Event placement and physical extent</h2>'
    body += '<div class="section-intro"><strong>Why projection stability matters.</strong>The geographic event coordinate is approximate, while simulation and gate extraction operate on s. A few metres of movement is usually harmless; a large movement can shift the entry or obstacle window into a different section. Primary p95 movement excludes gate-only policies, while the worst geometry stress case remains visible.</div>'
    body += figure(
        output / "event_projection_stability.png",
        "For each event, the bar is p95 movement under primary geometry choices and the marker is the worst geometry/data-support stress case. Interval-length changes are available in the detailed table.",
    )
    event_summary = events.head(20)
    body += '<p class="table-note"><strong>Most movement-sensitive events.</strong> This compact table is the evidence behind the plot; the complete event table is in the appendix.</p>'
    body += dataframe_table(
        event_summary,
        columns=(
            "event_id", "event_name", "anchor_shift_primary_p95_m",
            "anchor_shift_geometry_max_m", "interval_length_delta_p10_m",
            "interval_length_delta_p90_m", "projection_stable",
        ),
        sticky_columns=("event_id", "event_name"),
        table_id="event-summary",
        compact=True,
    )

    body += '<h2 id="gates">3. Gate evidence and qualification frontier</h2>'
    body += '<div class="section-intro"><strong>What this section tests.</strong>Whether a repeated speed observation remains a formal gate under strict, permissive, alternative-weight, and alternative-window definitions. The target speed and the binary qualification decision are shown separately. Gates above the configured high-speed review threshold receive extra scrutiny when repeatable braking evidence is weak.</div>'
    body += figure(
        output / "gate_frontier.png",
        "Nominal confidence-score margin versus acceptance frequency across gate-definition cases. The upper-right region contains the strongest core gates; nominal gates near the cutoff or with low policy agreement are conditional; excluded observations that repeatedly qualify are near misses.",
    )
    body += figure(
        output / "gate_target_speed_stability.png",
        "Nominal target speed with p10-p90 movement across policies, in km/h. Cross markers identify high-speed observations with weak braking evidence that may reflect the vehicle-speed ceiling rather than a local track constraint.",
    )
    body += figure(
        output / "gate_qualification_stability.png",
        "Acceptance frequency across only the cases that change gate thresholds, component weights, or measurement windows. This is a robustness frequency, not a calibrated probability.",
    )
    frontier_display = gates.copy()
    if not frontier_display.empty:
        frontier_display["definition_acceptance_percent"] = 100.0 * frontier_display["gate_definition_accepted_fraction"]
        frontier_display["target_speed_range_kmh"] = frontier_display["target_speed_p80_range_kmh"]
    body += '<p class="table-note"><strong>Gate frontier summary.</strong> Core gates survive most tested definitions; conditional gates are nominally accepted but policy-sensitive; near misses are excluded nominally but qualify under multiple reasonable alternatives. The first two columns remain visible while horizontally scrolling.</p>'
    body += dataframe_table(
        frontier_display,
        columns=(
            "gate_id", "gate_name", "frontier_classification",
            "nominal_recommendation", "nominal_confidence_score",
            "nominal_score_margin_to_accept", "definition_acceptance_percent",
            "nominal_target_speed_kmh", "target_speed_range_kmh",
            "nominal_braking_evidence_score",
            "nominal_approach_to_min_slowdown_kmh",
            "high_speed_weak_braking_review", "speed_interpretation",
        ),
        sticky_columns=("gate_id", "gate_name"),
        table_id="gate-frontier-summary",
        max_rows=250,
    )

    body += '<h3>Gate-definition case matrix</h3>'
    body += '<div class="section-intro"><strong>What this plot supports.</strong>It shows exactly which gates change classification under each threshold, weighting, or event-window policy. Centreline and leave-one-out cases are omitted because they answer different questions.</div>'
    body += figure(
        output / "gate_policy_matrix.png",
        "Gate classification matrix restricted to the cases that alter the definition of a good gate.",
    )

    body += '<h2 id="ensemble">4. Downstream track-interpretation ensemble</h2>'
    body += '<div class="section-intro"><strong>Purpose.</strong>Not every useful stress test should be propagated as an equally defensible track. Downstream eligibility now requires stable centreline position, stable length, stable event projection, and a sufficiently similar gate set. Strict/permissive extremes and data-support removals remain reportable stress tests but are not treated as ordinary alternatives.</div>'
    if not eligibility.empty:
        body += dataframe_table(
            eligibility,
            columns=(
                "case_id", "label", "category", "eligible_for_answer_uncertainty",
                "stress_only", "centreline_stable", "track_length_stable",
                "event_projection_stable", "gate_set_stable_for_ensemble",
                "eligibility_reasons",
            ),
            sticky_columns=("case_id", "label"),
            table_id="ensemble-eligibility",
            max_rows=250,
        )

    body += '<h2>5. Interpretation and decision rules</h2>'
    body += '<div class="section-intro"><strong>How to read the report.</strong>A stable centreline does not prove every gate. A stable target speed does not prove the observed speed is track-driven. Policy acceptance frequencies are selected stress-test results, not independent trials or probabilities. Cross-vehicle weighting remains a sensitivity test until multiple vehicles supply evidence.</div>'
    body += '<h3>Configured thresholds</h3><pre>' + html.escape(json.dumps(dict(thresholds), indent=2, sort_keys=True)) + '</pre>'

    body += '<h2 id="appendix">Detailed supporting tables</h2>'
    body += '<div class="section-intro"><strong>Underlying data.</strong>The report keeps the exhaustive tables for auditability, but moves them below the major plots and conclusions. Identity columns remain fixed on the left, and every column is sortable.</div>'
    body += '<details><summary>Appendix A — complete gate frontier evidence</summary>'
    body += '<p class="table-note">All calculated gate-stability, braking, speed, and interpretation fields.</p>'
    body += dataframe_table(
        gates,
        sticky_columns=("gate_id", "gate_name"),
        table_id="gate-detail",
        max_rows=500,
    ) + '</details>'
    body += '<details><summary>Appendix B — gate results for every definition case</summary>'
    body += '<p class="table-note">One row per gate and robustness case. Gate identity and case identity remain visible while scrolling.</p>'
    body += dataframe_table(
        gate_cases,
        sticky_columns=("gate_id", "gate_name", "case_id"),
        table_id="gate-case-detail",
        max_rows=2000,
    ) + '</details>'
    body += '<details><summary>Appendix C — complete event stability</summary>'
    body += '<p class="table-note">Aggregated event movement and interval-length stability.</p>'
    body += dataframe_table(
        events,
        sticky_columns=("event_id", "event_name"),
        table_id="event-detail",
        max_rows=500,
    )
    body += '<p class="table-note">One row per event and geometry case.</p>'
    body += dataframe_table(
        event_cases,
        sticky_columns=("event_id", "event_name", "case_id"),
        table_id="event-case-detail",
        max_rows=2000,
    ) + '</details>'
    body += '<details><summary>Appendix D — complete robustness case audit</summary>'
    body += '<p class="table-note">All cases, rationales, overrides, warnings, geometry changes, gate-set changes, and eligibility fields.</p>'
    body += dataframe_table(
        cases,
        sticky_columns=("case_id", "label"),
        table_id="case-detail",
        max_rows=500,
    ) + '</details>'
    body += figure(
        output / "centreline_overlay.png",
        "Audit overlay of every successful geometry-affecting alternative centreline. The corridor plot above is the preferred summary.",
    )

    target = output / REPORTS["track_robustness"].html_filename
    target.write_text(
        render_page(
            title=REPORTS["track_robustness"].title,
            subtitle=REPORTS["track_robustness"].question,
            body=body,
            report_key="track_robustness",
            source_note="Every case configuration and compact reconstruction artifact is retained under cases/.",
        ),
        encoding="utf-8",
    )


def _write_track_ensemble_manifest(
    output: Path,
    case_summary: pd.DataFrame,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Separate useful stress tests from defensible downstream alternatives."""

    records: list[dict[str, Any]] = [
        {
            "case_id": "nominal",
            "category": "nominal",
            "label": "Nominal reconstructed track",
            "bundle": "nominal_track/track_bundle.json",
            "include_in_robustness_report": True,
            "eligible_for_answer_uncertainty": True,
            "stress_only": False,
            "centreline_stable": True,
            "track_length_stable": True,
            "event_projection_stable": True,
            "gate_set_stable_for_ensemble": True,
            "eligibility_reasons": "nominal reference",
            "probability_interpretation": "none",
        }
    ]
    for row in case_summary.itertuples():
        success = bool(getattr(row, "success", False))
        stress_only = bool(getattr(row, "stress_only", False))
        checks = {
            "centreline_stable": bool(getattr(row, "centreline_stable", False)),
            "track_length_stable": bool(getattr(row, "track_length_stable", False)),
            "event_projection_stable": bool(getattr(row, "event_projection_stable", False)),
            "gate_set_stable_for_ensemble": bool(getattr(row, "gate_set_stable_for_ensemble", False)),
        }
        eligible = bool(success and not stress_only and all(checks.values()))
        reasons: list[str] = []
        if not success:
            reasons.append("case failed")
        if stress_only:
            reasons.append("explicit stress-only case")
        reasons.extend(name.replace("_", " ") for name, passed in checks.items() if not passed)
        if not reasons:
            reasons.append("all downstream eligibility checks passed")
        records.append(
            {
                "case_id": str(row.case_id),
                "category": str(row.category),
                "label": str(row.label),
                "rationale": str(getattr(row, "rationale", "")),
                "bundle": f"cases/{_safe_name(str(row.case_id))}/track_bundle.json",
                "include_in_robustness_report": True,
                "eligible_for_answer_uncertainty": eligible,
                "stress_only": stress_only,
                **checks,
                "gate_set_jaccard": float(getattr(row, "gate_set_jaccard", math.nan)),
                "nominal_gate_retention_fraction": float(getattr(row, "nominal_gate_retention_fraction", math.nan)),
                "newly_accepted_gate_count": int(getattr(row, "newly_accepted_gate_count", 0) or 0),
                "eligibility_reasons": "; ".join(reasons),
                "probability_interpretation": "none",
            }
        )
    manifest = {
        "schema_version": 2,
        "interpretation": (
            "Unweighted epistemic track scenarios. Percentiles across these cases are "
            "scenario-envelope summaries, not calibrated probabilities. Stress-only "
            "cases remain in this report but are not propagated as ordinary alternatives."
        ),
        "eligibility_thresholds": dict(thresholds),
        "cases": records,
        "eligible_cases": [
            record["case_id"]
            for record in records
            if record["eligible_for_answer_uncertainty"]
        ],
    }
    write_json(output / "track_ensemble_manifest.json", manifest)
    return manifest


def regenerate_track_robustness_report(output: Path) -> Path:
    """Regenerate the organized HTML report from an existing robustness result."""

    output = output.resolve()
    manifest = _read_json(output / "track_robustness_manifest.json", {})
    thresholds = {
        key: float(manifest.get("thresholds", {}).get(key, default))
        for key, default in _DEFAULT_THRESHOLDS.items()
    }
    cases = _read_csv_frame(output / "robustness_case_summary.csv")
    displacement = _read_csv_frame(output / "centreline_displacement.csv")
    event_cases = _read_csv_frame(output / "event_case_results.csv")
    gate_cases = _read_csv_frame(output / "gate_case_results.csv")
    nominal_gates = _read_csv_frame(output / "nominal_track" / "track" / "gate_review.csv")
    if cases.empty or nominal_gates.empty:
        raise FileNotFoundError(
            "Track robustness regeneration requires robustness_case_summary.csv and nominal_track/track/gate_review.csv."
        )
    category_by_case = cases.set_index("case_id")["category"].astype(str).to_dict()
    label_by_case = cases.set_index("case_id")["label"].astype(str).to_dict()
    for frame in (event_cases, gate_cases):
        if not frame.empty:
            frame["case_category"] = frame["case_id"].map(category_by_case)
            frame["case_label"] = frame["case_id"].map(label_by_case)
    cases = _enrich_case_summary(cases, event_cases, gate_cases, nominal_gates, thresholds)
    events = _aggregate_event_stability(event_cases, thresholds)
    gates = _aggregate_gate_stability(gate_cases, thresholds, nominal_gates)
    ensemble_manifest = _write_track_ensemble_manifest(output, cases, thresholds)
    tables = {
        "robustness_case_summary": cases,
        "centreline_displacement": displacement,
        "event_case_results": event_cases,
        "event_stability": events,
        "gate_case_results": gate_cases,
        "gate_stability": gates,
        "ensemble_eligibility": pd.DataFrame(ensemble_manifest["cases"]),
    }
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    _write_robustness_plots(output, tables)
    _write_centreline_stability_corridor(output, cases)
    nominal_length_values = pd.to_numeric(cases["track_length_m"], errors="coerce") - pd.to_numeric(cases["track_length_delta_m"], errors="coerce")
    nominal_length = float(nominal_length_values.dropna().median())
    _write_track_robustness_html(
        output,
        nominal_track_length_m=nominal_length,
        tables=tables,
        thresholds=thresholds,
        ensemble_manifest=ensemble_manifest,
    )
    write_json(
        output / "report_manifest.json",
        {
            "schema_version": 1,
            "report_key": "track_robustness",
            "title": REPORTS["track_robustness"].title,
            "question": REPORTS["track_robustness"].question,
            "fixed": REPORTS["track_robustness"].fixed,
            "varied": REPORTS["track_robustness"].varied,
            "html_file": REPORTS["track_robustness"].html_filename,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return output / REPORTS["track_robustness"].html_filename


def _read_csv_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def _thresholds(raw_study: Mapping[str, Any]) -> dict[str,float]:
    raw=raw_study.get("thresholds",{}); raw=raw if isinstance(raw,Mapping) else {}
    return {key:float(raw.get(key,default)) for key,default in _DEFAULT_THRESHOLDS.items()}


def _case(identifier:str,category:str,label:str,rationale:str,overrides:Mapping[str,Any])->RobustnessCase:
    return RobustnessCase(identifier,category,label,rationale,dict(overrides))


def _mapping(value:Any)->Mapping[str,Any]: return value if isinstance(value,Mapping) else {}

def _set_nested(mapping:dict[str,Any],path:Sequence[str],value:Any)->None:
    current=mapping
    for key in path[:-1]:
        child=current.get(key)
        if not isinstance(child,dict): child={}; current[key]=child
        current=child
    current[path[-1]]=value


def _remaining_run_contract_valid(raw_runs:Sequence[Mapping[str,Any]],excluded:set[str])->bool:
    remaining=[run for run in raw_runs if str(run.get("run_id")) not in excluded]
    return bool(remaining) and any(bool(run.get("use_for_centreline")) for run in remaining) and any(bool(run.get("use_for_gate_evidence")) for run in remaining)


def _weight_names()->tuple[str,...]: return ("pass_count","speed_repeatability","braking_evidence","pace_independence","coordinate_quality","cross_vehicle_agreement")

def _nominal_weights(gates:Mapping[str,Any])->dict[str,float]:
    defaults={"pass_count":.15,"speed_repeatability":.25,"braking_evidence":.20,"pace_independence":.15,"coordinate_quality":.15,"cross_vehicle_agreement":.10}
    return {name:float(gates.get(f"weight_{name}",defaults[name])) for name in _weight_names()}

def _renormalized_weights(gates:Mapping[str,Any],omitted:str)->dict[str,float]:
    weights=_nominal_weights(gates); weights[omitted]=0.0; total=sum(weights.values()); return {k:(v/total if total else 1/len(weights)) for k,v in weights.items()}

def _boost_weight(gates:Mapping[str,Any],name:str,target:float)->dict[str,float]:
    weights=_nominal_weights(gates); other_total=sum(v for k,v in weights.items() if k!=name); scale=(1-target)/other_total if other_total else 0; return {k:(target if k==name else v*scale) for k,v in weights.items()}

def _weight_overrides(weights:Mapping[str,float])->dict[str,float]: return {f"gate_confidence.weight_{name}":float(value) for name,value in weights.items()}

def _scaled_overrides(prefix:str,mapping:Mapping[str,Any],scale:float,exclude:set[str])->dict[str,float]:
    return {f"{prefix}.{key}":max(.01,float(value)*scale) for key,value in mapping.items() if key.endswith("_m") and key not in exclude and isinstance(value,(int,float))}

def _odd(value:int)->int: return value if value%2 else value+1

def _circular_s_shift(nominal_s:float,nominal_length:float,case_s:float,case_length:float)->float:
    mapped=(case_s/case_length)*nominal_length; difference=abs(mapped-nominal_s); return float(min(difference,nominal_length-difference))

def _safe_name(value:str)->str:
    text="".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in str(value)).strip("-."); return text or "run"

def _read_json(path:Path,default:Any)->Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default
