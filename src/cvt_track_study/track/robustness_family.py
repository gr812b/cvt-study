"""Route-family aware track robustness and downstream track-ensemble export."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cvt_track_study.config import ProjectError, ProjectLoader
from cvt_track_study.config.diagnostics import DiagnosticBag
from cvt_track_study.gpx.cleanup import apply_telemetry_cleanup
from cvt_track_study.reports.catalog import REPORTS
from cvt_track_study.reports.html import (
    dataframe_table,
    figure,
    metric_cards,
    render_page,
    write_json,
)
from cvt_track_study.runtime.provenance import canonical_fingerprint

from . import robustness as _robustness
from .family import build_track_family_evidence
from .model import TrackBuildResult
from .service import build_project_track


def route_family_robustness_enabled(resolution: Any) -> bool:
    """Whether normal track robustness should expand over route families."""

    track = resolution.data.get("track", {})
    if not isinstance(track, Mapping):
        return False
    variants = track.get("route_variants", {})
    if not isinstance(variants, Mapping) or not bool(variants.get("enabled", False)):
        return False
    family = variants.get("family", {})
    if isinstance(family, Mapping) and "enabled" in family:
        return bool(family.get("enabled"))
    return True


def run_route_family_track_robustness_project(
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
    """Run the canonical robustness study on every supported course traversal.

    The output is deliberately shaped like an ordinary ``track_robustness``
    result: it has one top-level HTML report and one top-level
    ``track_ensemble_manifest.json``.  The full per-route reports remain nested
    underneath for auditability, while downstream uncertainty sees a balanced
    route-family ensemble rather than a lap-count-weighted mixture.
    """

    if workers < 1:
        raise ValueError("workers must be at least one")
    resolution = ProjectLoader().resolve(project)
    if resolution.error_count:
        details = "\n".join(item.format() for item in resolution.diagnostics)
        raise ProjectError(f"Project validation failed:\n{details}")
    raw_study = resolution.data.get("studies", {}).get(study, {})
    if not isinstance(raw_study, Mapping):
        raise ProjectError(f"Track robustness study {study!r} was not found.")

    build = build_project_track(project)
    if build.output_directory is None:
        raise RuntimeError("Track build did not expose an output directory.")
    family_path = build.output_directory / "route_family_manifest.json"
    if not family_path.is_file():
        # A project may have route-family support enabled but only one route may
        # survive. In that case the stock robustness engine is the correct result.
        return _robustness.run_track_robustness_project(
            project,
            study=study,
            output_directory=output_directory,
            workers=workers,
            resume=resume,
            restart=restart,
            progress=progress,
            run_name=run_name,
            command=command,
        )
    family = json.loads(family_path.read_text(encoding="utf-8"))
    routes = list(family.get("routes", []))
    if not routes:
        raise ValueError("The track build did not export any route-family members.")
    if len(routes) == 1:
        return _robustness.run_track_robustness_project(
            project,
            study=study,
            output_directory=output_directory,
            workers=workers,
            resume=resume,
            restart=restart,
            progress=progress,
            run_name=run_name,
            command=command,
        )

    fingerprint = canonical_fingerprint(
        {
            "schema": "route-family-track-robustness-v3",
            "study": raw_study,
            "track": resolution.data.get("track", {}),
            "runs": resolution.data.get("runs", ()),
            "events": resolution.data.get("events", ()),
            "routes": [
                {
                    "route_variant_id": item.get("route_variant_id"),
                    "nominal": bool(item.get("nominal", False)),
                    "track_length_m": item.get("track_length_m"),
                    "valid_geometry_lap_count": item.get("valid_geometry_lap_count"),
                }
                for item in routes
            ],
        }
    )
    default_output = (
        resolution.paths.results_directory
        / "track_robustness"
        / f"{_robustness._safe_name(run_name or study)}--route-family-{fingerprint[:10]}"
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
    try:
        rows: list[dict[str, Any]] = []
        combined_case_summaries: list[pd.DataFrame] = []
        route_manifests: dict[str, dict[str, Any]] = {}
        for route in routes:
            route_id = str(route["route_variant_id"])
            with tempfile.TemporaryDirectory(
                prefix=f"cvt-route-{route_id}-"
            ) as temporary:
                temporary_root = Path(temporary) / resolution.paths.root.name
                ignored_results = resolution.paths.results_directory.resolve()

                def ignore(directory: str, names: list[str]) -> set[str]:
                    current = Path(directory).resolve()
                    ignored: set[str] = set()
                    for name in names:
                        if (current / name).resolve() == ignored_results:
                            ignored.add(name)
                    return ignored

                shutil.copytree(
                    resolution.paths.root, temporary_root, ignore=ignore
                )
                project_relative = resolution.paths.project_file.relative_to(
                    resolution.paths.root
                )
                track_relative = resolution.paths.track_file.relative_to(
                    resolution.paths.root
                )
                copied_project = temporary_root / project_relative
                copied_track = temporary_root / track_relative
                _select_route_variant(copied_track, route_id)
                case_output = staging / route_id
                with _family_aware_robustness_runtime():
                    _robustness.run_track_robustness_project(
                        copied_project,
                        study=study,
                        output_directory=case_output,
                        workers=workers,
                        progress=progress,
                        command=command,
                    )

            normalized_weight = 1.0 / len(routes)
            route_manifest = _read_json(
                case_output / "track_robustness_manifest.json", {}
            )
            route_manifests[route_id] = _read_json(
                case_output / "track_ensemble_manifest.json", {}
            )
            rows.append(
                {
                    "course_case_id": route_id,
                    "route_variant_id": route_id,
                    "nominal": bool(route.get("nominal", False)),
                    "case_weight": 1.0,
                    "normalized_equal_weight": normalized_weight,
                    "track_length_m": float(route.get("track_length_m", math.nan)),
                    "valid_geometry_lap_count": int(
                        route.get("valid_geometry_lap_count", 0)
                    ),
                    "successful_robustness_cases": int(
                        route_manifest.get("successful_case_count", 0)
                    ),
                    "failed_robustness_cases": int(
                        route_manifest.get("failed_case_count", 0)
                    ),
                    "eligible_track_interpretations": len(
                        route_manifests[route_id].get("eligible_cases", ())
                    ),
                    "robustness_output": str(case_output.relative_to(staging)),
                    "detailed_report": str(
                        (case_output / REPORTS["track_robustness"].html_filename)
                        .relative_to(staging)
                    ),
                }
            )
            summary_path = case_output / "robustness_case_summary.csv"
            if summary_path.exists() and summary_path.stat().st_size:
                frame = pd.read_csv(summary_path)
                frame.insert(0, "route_variant_id", route_id)
                frame.insert(1, "course_case_weight", 1.0)
                frame.insert(
                    2, "normalized_course_case_weight", normalized_weight
                )
                combined_case_summaries.append(frame)

        combined = (
            pd.concat(combined_case_summaries, ignore_index=True, sort=False)
            if combined_case_summaries
            else pd.DataFrame()
        )
        combined.to_csv(
            staging / "route_family_robustness_case_summary.csv", index=False
        )
        course_cases = pd.DataFrame(rows)
        course_cases.to_csv(staging / "route_family_course_cases.csv", index=False)

        ensemble = _write_balanced_family_ensemble_manifest(
            staging,
            family=family,
            route_manifests=route_manifests,
        )
        _write_family_robustness_report(
            staging,
            course_cases=course_cases,
            combined=combined,
            ensemble=ensemble,
        )

        manifest = {
            "schema_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "study_name": study,
            "study_type": "track_robustness",
            "study_fingerprint_sha256": fingerprint,
            "route_family_aware": True,
            "aggregation_policy": "equal_route_weight_then_equal_interpretations_within_route",
            "route_family_count": len(routes),
            "route_family_ids": [str(item["route_variant_id"]) for item in routes],
            "nominal_route_variant_id": str(
                family.get("nominal_route_variant_id", "")
            ),
            "source_route_family_manifest": str(family_path),
            "combined_case_summary": "route_family_robustness_case_summary.csv",
            "course_case_summary": "route_family_course_cases.csv",
            "track_ensemble_manifest": "track_ensemble_manifest.json",
            "eligible_track_ensemble_case_count": len(
                ensemble.get("eligible_cases", ())
            ),
            "requested_workers": workers,
            "effective_workers_per_route": 1,
            "command": list(command),
            "primary_report": REPORTS["track_robustness"].html_filename,
        }
        write_json(staging / "track_robustness_manifest.json", manifest)
        # Compatibility alias retained for users of the earlier wrapper.
        write_json(staging / "route_family_track_robustness_manifest.json", manifest)
        write_json(
            staging / "report_manifest.json",
            {
                "schema_version": 1,
                "report_key": "track_robustness",
                "title": "Track robustness — route family",
                "question": REPORTS["track_robustness"].question,
                "fixed": REPORTS["track_robustness"].fixed,
                "varied": REPORTS["track_robustness"].varied,
                "html_file": REPORTS["track_robustness"].html_filename,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _execute_family_case(
    *,
    case: Any,
    resolution: Any,
    parsed_runs: Any,
    track_config: Any,
    raw_events: Any,
) -> Any:
    """Run one robustness perturbation through the topology-family builder."""

    try:
        config = deepcopy(dict(track_config))
        for path, value in case.overrides.items():
            _robustness._set_nested(config, str(path).split("."), value)

        excluded = set(case.excluded_run_ids)
        selected = [
            item for item in parsed_runs if item.metadata.run_id not in excluded
        ]
        cleaned = tuple(apply_telemetry_cleanup(item, config) for item in selected)
        diagnostics = DiagnosticBag()
        for item in cleaned:
            diagnostics.extend(item.diagnostics)

        family = build_track_family_evidence(
            cleaned, config, tuple(raw_events), diagnostics
        )
        route_id = str(family.nominal_route_variant_id)
        evidence = family.routes[route_id]

        case_resolution = deepcopy(resolution)
        case_resolution.data["track"] = config
        build = TrackBuildResult(
            resolution=case_resolution,
            ingestion_results=cleaned,
            centreline=evidence.centreline,
            laps=evidence.laps,
            matched_points=evidence.matched_points,
            track_profile=evidence.track_profile,
            event_projection=evidence.event_projection,
            response_features=evidence.response_features,
            event_passes=evidence.event_passes,
            gate_evidence=evidence.gate_evidence,
            gate_review=evidence.gate_review,
            rejected_map_points=evidence.rejected_map_points,
            route_variant_summary=family.route_variant_summary,
            route_variant_pairwise=family.route_variant_pairwise,
            route_variant_id=route_id,
            nominal_route_variant_id=route_id,
            shared_gate_evidence=family.shared_gate_evidence,
            event_route_applicability=family.event_route_applicability,
            course_cases=family.course_cases,
            branch_summary=family.branch_summary,
            event_route_sections=family.event_route_sections,
            diagnostics=diagnostics.items,
            metadata={
                "schema_version": 2,
                "phase": "track_robustness_case",
                "case_id": case.identifier,
                "route_variant_id": route_id,
                "track_length_m": float(evidence.centreline.length_m),
                "valid_lap_count": int(evidence.laps["analysis_valid"].sum()),
                "accepted_gate_count": int(
                    (evidence.gate_review["recommendation"] == "accepted").sum()
                ),
                "supported_route_family_count": int(len(family.routes)),
            },
        )
        return _robustness.CaseResult(
            case=case,
            success=True,
            error="",
            centreline=evidence.centreline,
            laps=evidence.laps,
            event_projection=evidence.event_projection,
            gate_review=evidence.gate_review,
            diagnostics=diagnostics.items,
            rejected_map_points=evidence.rejected_map_points,
            track_build=build,
        )
    except Exception as exc:
        return _robustness.CaseResult(
            case=case,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _prepare_case_summary_for_enrichment(cases: pd.DataFrame) -> pd.DataFrame:
    prepared = cases.copy()
    for column in ("track_length_m", "track_length_delta_m"):
        if column not in prepared:
            prepared[column] = math.nan
    return prepared


def _spacing_decoupled_cases(
    cases: Sequence[Any], track_config: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Keep physical smoothing width fixed in discretization-only cases.

    ``smoothing_window_nodes`` is a node count, so changing node spacing without
    changing that count also changes the physical smoothing scale.  The fine and
    coarse spacing robustness cases now adjust the odd node count inversely so
    they isolate discretization.  The explicit less/more smoothing cases remain
    the only tests that intentionally change physical smoothing strength.
    """

    reconstruction = track_config.get("reconstruction", {})
    reconstruction = reconstruction if isinstance(reconstruction, Mapping) else {}
    consensus = track_config.get("centreline_consensus", {})
    consensus = consensus if isinstance(consensus, Mapping) else {}
    nominal_spacing = float(reconstruction.get("centreline_spacing_m", 3.0))
    nominal_nodes = int(consensus.get("smoothing_window_nodes", 5))
    nominal_nodes = max(1, nominal_nodes if nominal_nodes % 2 else nominal_nodes + 1)
    physical_width = nominal_spacing * nominal_nodes

    output: list[Any] = []
    for case in cases:
        if str(case.identifier) not in {
            "centreline_fine_spacing",
            "centreline_coarse_spacing",
        }:
            output.append(case)
            continue
        overrides = dict(case.overrides)
        spacing = float(
            overrides.get("reconstruction.centreline_spacing_m", nominal_spacing)
        )
        nodes = _nearest_odd(max(1.0, physical_width / max(spacing, 1e-9)))
        overrides["centreline_consensus.smoothing_window_nodes"] = nodes
        actual_width = nodes * spacing
        output.append(
            _robustness.RobustnessCase(
                case.identifier,
                case.category,
                case.label + " (fixed smoothing width)",
                (
                    case.rationale
                    + f" Physical smoothing is held near the nominal {physical_width:.2f} m "
                    + f"scale ({nodes} nodes × {spacing:.3g} m = {actual_width:.2f} m)."
                ),
                overrides,
                tuple(case.excluded_run_ids),
            )
        )
    return tuple(output)


def _nearest_odd(value: float) -> int:
    rounded = max(1, int(round(float(value))))
    if rounded % 2:
        return rounded
    lower = max(1, rounded - 1)
    upper = rounded + 1
    return lower if abs(value - lower) <= abs(upper - value) else upper


@contextmanager
def _family_aware_robustness_runtime():
    """Temporarily adapt the stock robustness engine to route families."""

    original_execute = _robustness._execute_case
    original_enrich = _robustness._enrich_case_summary
    original_build_cases = _robustness.build_robustness_cases

    def enrich_failure_safe(
        cases: pd.DataFrame,
        event_cases: pd.DataFrame,
        gate_cases: pd.DataFrame,
        nominal_gate_review: pd.DataFrame,
        thresholds: Any,
    ) -> pd.DataFrame:
        return original_enrich(
            _prepare_case_summary_for_enrichment(cases),
            event_cases,
            gate_cases,
            nominal_gate_review,
            thresholds,
        )

    def build_spacing_decoupled_cases(
        track_config: Mapping[str, Any],
        raw_runs: Sequence[Mapping[str, Any]],
        raw_study: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        return _spacing_decoupled_cases(
            original_build_cases(track_config, raw_runs, raw_study),
            track_config,
        )

    _robustness._execute_case = _execute_family_case
    _robustness._enrich_case_summary = enrich_failure_safe
    _robustness.build_robustness_cases = build_spacing_decoupled_cases
    try:
        yield
    finally:
        _robustness._execute_case = original_execute
        _robustness._enrich_case_summary = original_enrich
        _robustness.build_robustness_cases = original_build_cases


def _write_balanced_family_ensemble_manifest(
    output: Path,
    *,
    family: Mapping[str, Any],
    route_manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Export all route-balanced eligible interpretations for downstream studies.

    Only robustness case IDs that are eligible on *every* route are propagated.
    This prevents one route receiving more epistemic weight merely because one of
    its perturbations happened to pass an eligibility cutoff that failed on the
    other route.  The uncertainty runner may later apply its configured maximum
    case count while preserving this route balance.
    """

    route_ids = [str(item["route_variant_id"]) for item in family.get("routes", [])]
    nominal_route_id = str(family.get("nominal_route_variant_id", ""))
    eligible_maps: dict[str, dict[str, Mapping[str, Any]]] = {}
    for route_id in route_ids:
        records = {
            str(record.get("case_id")): record
            for record in route_manifests.get(route_id, {}).get("cases", [])
            if isinstance(record, Mapping)
            and bool(record.get("eligible_for_answer_uncertainty", False))
        }
        eligible_maps[route_id] = records
    non_nominal_sets = [set(records) - {"nominal"} for records in eligible_maps.values()]
    common = set.intersection(*non_nominal_sets) if non_nominal_sets else set()
    # Preserve the stock robustness case order from the first route rather than
    # alphabetizing, because that order groups related interpretation choices.
    first_records = list(eligible_maps.get(route_ids[0], {}).values()) if route_ids else []
    common_order = [
        str(record.get("case_id"))
        for record in first_records
        if str(record.get("case_id")) in common
    ]

    records: list[dict[str, Any]] = []
    for route_id in route_ids:
        if route_id != nominal_route_id:
            records.append(
                {
                    "case_id": f"{route_id}__nominal",
                    "route_variant_id": route_id,
                    "robustness_case_id": "nominal",
                    "category": "nominal",
                    "label": f"{route_id} — nominal reconstruction",
                    "bundle": f"{route_id}/nominal_track/track_bundle.json",
                    "eligible_for_answer_uncertainty": True,
                    "stress_only": False,
                    "family_pairing_group": "nominal",
                }
            )
        for case_id in common_order:
            source = eligible_maps[route_id][case_id]
            records.append(
                {
                    **{
                        key: value
                        for key, value in source.items()
                        if key not in {"case_id", "bundle", "label"}
                    },
                    "case_id": f"{route_id}__{case_id}",
                    "route_variant_id": route_id,
                    "robustness_case_id": case_id,
                    "label": f"{route_id} — {source.get('label', case_id)}",
                    "bundle": f"{route_id}/{source.get('bundle')}",
                    "eligible_for_answer_uncertainty": True,
                    "family_pairing_group": case_id,
                }
            )

    implicit_nominal = {
        "case_id": "nominal",
        "route_variant_id": nominal_route_id,
        "robustness_case_id": "nominal",
        "category": "nominal",
        "label": f"{nominal_route_id} — nominal reconstruction",
        "bundle": "external_nominal_track_bundle",
        "eligible_for_answer_uncertainty": True,
        "stress_only": False,
        "family_pairing_group": "nominal",
    }
    manifest = {
        "schema_version": 3,
        "route_family_balanced": True,
        "route_variant_ids": route_ids,
        "nominal_route_variant_id": nominal_route_id,
        "interpretation": (
            "Balanced route-family epistemic ensemble. Only robustness case IDs "
            "eligible on every supported traversal are propagated. The selected "
            "nominal traversal is supplied externally by the uncertainty runner; "
            "the alternate nominal traversal is listed explicitly. Route families "
            "receive equal weight before interpretation alternatives are compared."
        ),
        "family_pairing_policy": "same_eligible_robustness_case_ids_on_every_route",
        "common_eligible_robustness_case_ids": common_order,
        "cases": [implicit_nominal, *records],
        "eligible_cases": ["nominal", *[str(item["case_id"]) for item in records]],
    }
    write_json(output / "track_ensemble_manifest.json", manifest)
    return manifest


def _write_family_robustness_report(
    output: Path,
    *,
    course_cases: pd.DataFrame,
    combined: pd.DataFrame,
    ensemble: Mapping[str, Any],
) -> None:
    route_ids = course_cases["route_variant_id"].astype(str).tolist()
    successful = int(course_cases["successful_robustness_cases"].sum())
    failed = int(course_cases["failed_robustness_cases"].sum())
    common = list(ensemble.get("common_eligible_robustness_case_ids", ()))
    cards = metric_cards(
        [
            ("Supported traversals", str(len(route_ids)), "good"),
            ("Route weights", "equal", "good"),
            ("Successful perturbations", str(successful), "good" if not failed else "warning"),
            ("Failed perturbations", str(failed), "good" if not failed else "warning"),
            ("Common downstream alternatives", str(len(common)), "note"),
        ]
    )
    nav = (
        '<nav class="report-nav" aria-label="Report sections">'
        '<a href="#summary">Summary</a><a href="#routes">Route cases</a>'
        '<a href="#smoothing">Spacing vs smoothing</a><a href="#downstream">Downstream ensemble</a>'
        '<a href="#details">Detailed route reports</a></nav>'
    )
    body = nav + cards
    body += (
        '<h2 id="summary">Route-family robustness summary</h2>'
        '<div class="section-intro"><strong>What this report tests.</strong> Each supported '
        'long/short traversal is reconstructed under the same robustness choices. Shared-course '
        'gate contracts are pooled once across traversals; branch-local events remain route-specific. '
        'The two course traversals have equal weight regardless of observed lap count.</div>'
    )
    body += '<h2 id="routes">Route cases</h2>'
    body += dataframe_table(
        course_cases,
        sticky_columns=("route_variant_id",),
        searchable=True,
        table_id="route-family-cases",
        max_rows=50,
    )
    body += (
        '<h2 id="smoothing">Spacing and smoothing are tested independently</h2>'
        '<div class="section-intro"><strong>Important cleanup.</strong> Fine/coarse centreline '
        'spacing cases now adjust the odd smoothing-node count inversely so the physical smoothing '
        'width remains near nominal. The separate less/more-smoothing cases are therefore the only '
        'tests that intentionally change physical smoothing strength.</div>'
    )
    spacing = combined[
        combined.get("case_id", pd.Series(dtype=str)).astype(str).isin(
            {"centreline_fine_spacing", "centreline_coarse_spacing", "centreline_less_smoothing", "centreline_more_smoothing"}
        )
    ].copy() if not combined.empty else pd.DataFrame()
    if not spacing.empty:
        body += dataframe_table(
            spacing,
            columns=(
                "route_variant_id", "case_id", "label", "success",
                "track_length_delta_m", "track_length_relative_shift",
                "centreline_shift_p95_m", "event_anchor_shift_p95_m",
                "centreline_stable", "track_length_stable", "event_projection_stable",
            ),
            sticky_columns=("route_variant_id", "case_id"),
            table_id="spacing-smoothing-cases",
            max_rows=50,
        )
    body += '<h2 id="downstream">Balanced downstream track ensemble</h2>'
    body += (
        '<div class="section-intro"><strong>Propagation rule.</strong> Only robustness case IDs '
        'that remain eligible on every supported route are available to full uncertainty. This '
        'prevents one route from receiving extra weight merely because it retained more acceptable '
        'perturbations. The uncertainty runner further respects its configured maximum case count '
        'while keeping the route counts balanced.</div>'
    )
    body += dataframe_table(
        pd.DataFrame(
            {
                "common_eligible_robustness_case_id": common,
            }
        ),
        table_id="common-downstream-cases",
        max_rows=100,
    )
    body += '<h2 id="details">Detailed route reports</h2>'
    for route_id in route_ids:
        body += f'<h3>{html.escape(route_id)}</h3>'
        body += (
            f'<p><a href="{html.escape(route_id)}/{html.escape(REPORTS["track_robustness"].html_filename)}">'
            'Open the complete per-route robustness report</a>.</p>'
        )
        body += figure(
            output / route_id / "centreline_stability_corridor.png",
            f"{route_id}: top-down reconstruction corridor.",
        )
        body += figure(
            output / route_id / "gate_frontier.png",
            f"{route_id}: gate qualification frontier. Shared-course gates use the common pooled contract.",
        )
        body += figure(
            output / route_id / "gate_target_speed_stability.png",
            f"{route_id}: target-speed stability across robustness policies.",
        )
    body += '<details><summary>Complete combined robustness case table</summary>'
    body += dataframe_table(
        combined,
        sticky_columns=("route_variant_id", "case_id"),
        searchable=True,
        table_id="combined-robustness-cases",
        max_rows=1000,
    ) + '</details>'

    target = output / REPORTS["track_robustness"].html_filename
    target.write_text(
        render_page(
            title="Track robustness — route family",
            subtitle=REPORTS["track_robustness"].question,
            body=body,
            report_key="track_robustness",
            source_note=(
                "Per-route robustness artifacts remain under each route directory; "
                "the top-level track ensemble is balanced across route families."
            ),
        ),
        encoding="utf-8",
    )


def _select_route_variant(track_file: Path, route_id: str) -> None:
    text = track_file.read_text(encoding="utf-8")
    header = "[track.route_variants]"
    start = text.find(header)
    if start < 0:
        suffix = (
            "\n\n[track.route_variants]\n"
            "enabled = true\n"
            "selection = \"variant_id\"\n"
            f"selected_variant_id = {json.dumps(route_id)}\n"
        )
        track_file.write_text(text.rstrip() + suffix, encoding="utf-8")
        return
    content_start = start + len(header)
    next_header = re.search(r"(?m)^\s*\[[^\n]+\]\s*$", text[content_start:])
    end = content_start + next_header.start() if next_header is not None else len(text)
    section = text[content_start:end]
    section = _replace_or_append_key(section, "enabled", "true")
    section = _replace_or_append_key(section, "selection", '"variant_id"')
    section = _replace_or_append_key(
        section, "selected_variant_id", json.dumps(route_id)
    )
    track_file.write_text(
        text[:content_start] + section + text[end:], encoding="utf-8"
    )


def _replace_or_append_key(section: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=.*$")
    replacement = f"{key} = {value}"
    if pattern.search(section):
        return pattern.sub(replacement, section, count=1)
    return section.rstrip() + "\n" + replacement + "\n"


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default
