"""Post-process machine artifacts into the six canonical HTML reports."""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .catalog import REPORTS
from .html import (
    dataframe_table,
    figure,
    metric_cards,
    nested_get,
    read_json,
    render_page,
    write_json,
)


def _register(output: Path, report_key: str, html_path: Path) -> Path:
    definition = REPORTS[report_key]
    manifest = {
        "schema_version": 1,
        "report_key": report_key,
        "title": definition.title,
        "question": definition.question,
        "fixed": definition.fixed,
        "varied": definition.varied,
        "html_file": str(html_path.relative_to(output)),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "report_manifest.json", manifest)
    return html_path


def primary_report_path(output: Path) -> Path | None:
    manifest = read_json(output / "report_manifest.json", {})
    relative = manifest.get("html_file") if isinstance(manifest, Mapping) else None
    if relative:
        path = output / str(relative)
        if path.is_file():
            return path
    candidates = (
        output / "track_robustness_report.html",
        output / "structural_sensitivity_report.html",
        output / "full_uncertainty_report.html",
        output / "design_comparison_report.html",
        output / "nominal_simulation_report.html",
        output / "review" / "track_evidence_report.html",
        output / "review" / "track_review.html",
        output / "REPORT.md",
    )
    return next((path for path in candidates if path.is_file()), None)


def write_track_evidence_report(output: Path) -> Path:
    """Render the nominal track-evidence package using the shared report shell.

    ``track_review.html`` remains as a legacy filename, but it is rewritten to the
    same self-contained document so users do not encounter two visual systems in
    one track-build result.
    """

    output = output.resolve()
    review = output / "review"
    review.mkdir(parents=True, exist_ok=True)

    manifest = read_json(output / "track_build_manifest.json", {})
    diagnostics_raw = read_json(output / "diagnostics.json", [])
    run_summaries_raw = read_json(output / "ingestion" / "run_summaries.json", [])
    laps = _read_csv(output / "track" / "lap_quality.csv")
    events = _read_csv(output / "track" / "event_projection.csv")
    intervals = _read_csv(output / "track" / "event_interval_audit.csv")
    gates = _read_csv(output / "track" / "gate_review.csv")
    rejected_telemetry = _read_csv(output / "ingestion" / "rejected_telemetry_points.csv")
    rejected_map = _read_csv(output / "track" / "rejected_map_points.csv")

    diagnostics = pd.DataFrame(diagnostics_raw if isinstance(diagnostics_raw, list) else [])
    run_summaries = pd.DataFrame(
        run_summaries_raw if isinstance(run_summaries_raw, list) else []
    )

    track_length = float(manifest.get("track_length_m", math.nan))
    if not laps.empty and "analysis_valid" in laps:
        valid_flags = laps["analysis_valid"]
        if pd.api.types.is_bool_dtype(valid_flags.dtype):
            valid_laps = int(valid_flags.fillna(False).sum())
        else:
            valid_laps = int(
                valid_flags.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).sum()
            )
    else:
        valid_laps = int(manifest.get("valid_lap_count", 0))
    accepted = int((gates.get("recommendation", pd.Series(dtype=str)).astype(str) == "accepted").sum())
    review_count = int((gates.get("recommendation", pd.Series(dtype=str)).astype(str) == "recommended_review").sum())
    must_fix = int((gates.get("recommendation", pd.Series(dtype=str)).astype(str) == "must_fix").sum())

    cards = metric_cards(
        [
            ("Reconstructed length", _fmt(track_length, " m"), "note"),
            ("Valid evidence laps", str(valid_laps), "good" if valid_laps else "warning"),
            ("Physical features", str(len(events)), "note"),
            ("Accepted gates", str(accepted), "good"),
            ("Recommended review", str(review_count), "warning" if review_count else "good"),
            ("Must fix", str(must_fix), "bad" if must_fix else "good"),
        ]
    )

    body = _scope_html("track_evidence") + cards
    body += (
        '<div class="card note"><strong>Interpretation boundary.</strong> This is the selected '
        'nominal reconstruction from the supplied telemetry and reviewed event evidence. It documents '
        'what was inferred; sensitivity to reconstruction choices is handled separately by the track '
        'robustness report.</div>'
    )

    body += '<h2>Executive evidence summary</h2>'
    body += (
        '<div class="section-intro"><strong>What this section shows.</strong> The central evidence '
        'package: the retained telemetry, consensus centreline, interpreted events, and selected speed '
        'constraints. Review warnings below are retained rather than hidden.</div>'
    )
    body += figure(
        review / "track_map.png",
        "Consensus centreline, retained evidence laps, event anchors and gate-response markers.",
    )
    body += figure(
        review / "telemetry_cleanup_map.png",
        "Retained and rejected telemetry points. Rejections remain available in the machine-readable audit tables.",
    )

    body += '<h2>Event interpretation</h2>'
    body += (
        '<div class="section-intro"><strong>What this section shows.</strong> Every physical response '
        'group is placed on the common along-track coordinate. Long, wrapped, or otherwise suspicious '
        'intervals are surfaced explicitly before the full supporting table.</div>'
    )
    suspicious = pd.DataFrame()
    if not intervals.empty and "interval_audit_flags" in intervals:
        suspicious = intervals[
            intervals["interval_audit_flags"].fillna("").astype(str).str.strip().ne("")
        ].copy()
    if suspicious.empty:
        body += '<div class="card good"><strong>Interval audit.</strong> No event-interval flags were raised.</div>'
    else:
        body += (
            f'<div class="card warning"><strong>Interval audit.</strong> {len(suspicious)} response '
            'group(s) carry wrap, extent, or reconstruction-review flags. Inspect the timeline and '
            'flagged table before using them as physical obstacles.</div>'
        )
    body += figure(
        review / "event_group_timeline.png",
        "Resolved event-group extents along the common s coordinate. Hatching marks groups with interval-review flags.",
    )
    if not suspicious.empty:
        body += dataframe_table(
            suspicious,
            columns=(
                "sequence", "response_group_id", "name", "feature_start_s_m",
                "feature_end_s_m", "feature_length_m", "wraps_start_finish",
                "interval_audit_flags",
            ),
            sticky_columns=("sequence", "response_group_id", "name"),
            searchable=True,
            max_rows=200,
            table_id="flagged-event-intervals",
        )

    body += '<h2>Gate evidence and qualification</h2>'
    body += (
        '<div class="section-intro"><strong>What this section shows.</strong> Gate confidence is an '
        'evidence score, not a probability that a gate is true. The compact table keeps the decision, '
        'speed range, pass support, coordinate quality, and braking signature visible; the complete '
        'table remains in the appendix.</div>'
    )
    gate_columns = (
        "response_group_id", "event_name", "sequence", "recommendation",
        "overall_confidence_score", "valid_pass_count", "entry_speed_median_mps",
        "entry_speed_p10_mps", "entry_speed_p90_mps", "coordinate_effective_error_m",
        "slowdown_signature", "cross_vehicle_status", "reasons", "suggested_action",
    )
    body += dataframe_table(
        gates,
        columns=gate_columns,
        sticky_columns=("response_group_id", "event_name"),
        searchable=True,
        max_rows=300,
        table_id="track-evidence-gates",
    )

    body += '<h2>Elevation and lap support</h2>'
    body += (
        '<div class="section-intro"><strong>What this section shows.</strong> Elevation and lap-quality '
        'evidence are preserved for review. Telemetry elevation is not converted into vehicle grade '
        'force unless that capability is explicitly enabled elsewhere.</div>'
    )
    body += figure(
        review / "elevation_profile.png",
        "Median telemetry elevation with the p10–p90 between-lap band; retained for evidence review only.",
    )
    lap_columns = (
        "lap_id", "run_id", "vehicle_id", "duration_s", "analysis_valid",
        "speed_coverage_fraction", "p95_map_error_m", "time_gap_count", "quality_flags",
    )
    body += dataframe_table(
        laps,
        columns=lap_columns,
        sticky_columns=("lap_id", "run_id"),
        searchable=True,
        max_rows=300,
        table_id="track-evidence-laps",
    )

    body += '<h2>Evidence provenance and diagnostics</h2>'
    body += (
        '<div class="section-intro"><strong>What this section shows.</strong> The source runs, cleanup '
        'counts, reconstruction diagnostics, and bundle fingerprints needed to reproduce or challenge '
        'the nominal track.</div>'
    )
    if not diagnostics.empty:
        body += dataframe_table(
            diagnostics,
            sticky_columns=("severity", "code"),
            searchable=True,
            max_rows=300,
            table_id="track-evidence-diagnostics",
        )
    if not run_summaries.empty:
        body += '<h3>Telemetry run summaries</h3>'
        body += dataframe_table(
            run_summaries,
            sticky_columns=("run_id", "vehicle_id", "driver_id"),
            searchable=True,
            max_rows=100,
            table_id="track-evidence-runs",
        )
    body += '<h3>Track-build contract</h3>'
    body += f'<pre>{html.escape(json.dumps(manifest, indent=2, sort_keys=True))}</pre>'

    body += '<h2>Detailed supporting tables and files</h2>'
    body += (
        '<div class="section-intro"><strong>Underlying data.</strong> These appendices retain the full '
        'event and rejection tables without forcing them ahead of the major plots and conclusions.</div>'
    )
    body += '<details><summary>Complete event-interval audit</summary>'
    body += dataframe_table(
        intervals,
        sticky_columns=("sequence", "response_group_id", "name"),
        searchable=True,
        max_rows=1000,
        table_id="complete-event-intervals",
    ) + '</details>'
    body += '<details><summary>Complete event projection table</summary>'
    body += dataframe_table(
        events,
        sticky_columns=("sequence", "response_group_id", "name"),
        searchable=True,
        max_rows=1000,
        table_id="complete-event-projection",
    ) + '</details>'
    body += '<details><summary>Rejected telemetry points</summary>'
    body += dataframe_table(
        rejected_telemetry, searchable=True, max_rows=max(2000, len(rejected_telemetry)), table_id="rejected-telemetry"
    ) + '</details>'
    body += '<details><summary>Rejected map-matched points</summary>'
    body += dataframe_table(
        rejected_map, searchable=True, max_rows=max(2000, len(rejected_map)), table_id="rejected-map-points"
    ) + '</details>'
    body += (
        '<ul><li><code>track_bundle.json</code></li><li><code>track/centreline.csv</code></li>'
        '<li><code>track/gate_evidence.csv</code></li><li><code>track/event_passes.csv</code></li>'
        '<li><code>configuration/resolved_inputs.toml</code></li></ul>'
    )

    target = review / REPORTS["track_evidence"].html_filename
    document = render_page(
        title=REPORTS["track_evidence"].title,
        subtitle=REPORTS["track_evidence"].question,
        body=body,
        report_key="track_evidence",
        source_note="Track-build CSV, JSON, image, and bundle artifacts remain the source of truth.",
    )
    target.write_text(document, encoding="utf-8")

    # Preserve the historical path while preventing it from remaining the odd
    # visual outlier in a freshly generated track-build directory.
    (review / "track_review.html").write_text(document, encoding="utf-8")
    return _register(output, "track_evidence", target)


def write_nominal_simulation_report(output: Path) -> Path:
    output = output.resolve()
    bounded = read_json(output / "bounded_summary.json", {})
    reference = read_json(output / "infinite_reference_summary.json", {})
    comparison = read_json(output / "comparison_summary.json", {})
    manifest = read_json(output / "run_manifest.json", {})
    gates = _read_csv(output / "gate_compliance.csv")
    obstacles = _read_csv(output / "obstacle_energy_by_feature.csv")
    bounded_trace = _read_csv(output / "bounded_trace.csv")
    reference_trace = _read_csv(output / "infinite_reference_trace.csv")
    plots = output / "report_plots"
    plots.mkdir(exist_ok=True)
    _nominal_trace_plots(bounded_trace, reference_trace, plots)

    completed = bool(bounded.get("completed", bounded.get("lap_completed", True)))
    dominance = bool(comparison.get("reference_dominance_pass", False))
    gate_ok = _bool_fraction(gates, "bounded_compliant_0p5_kmh") if not gates.empty else math.nan
    cards = metric_cards(
        [
            ("Bounded lap time", _fmt(bounded.get("lap_time_s"), " s"), "good" if completed else "bad"),
            ("Infinite-reference lap time", _fmt(reference.get("lap_time_s"), " s"), "note"),
            ("Finite-range time penalty", _fmt(comparison.get("lap_time_penalty_vs_infinite_s"), " s"), "warning"),
            ("Finite-range opportunity", _fmt(comparison.get("finite_ratio_opportunity_loss_energy_kj"), " kJ"), "warning"),
            ("Reference dominance", "pass" if dominance else "FAIL", "good" if dominance else "bad"),
            ("Gate compliance", _percent(gate_ok), "good" if not np.isfinite(gate_ok) or gate_ok >= 0.999 else "bad"),
        ]
    )
    scope = _scope_html("nominal_simulation")
    health = (
        '<div class="card note"><strong>Interpretation boundary.</strong> This is one fixed baseline. '
        "It explains the mechanism and establishes the comparison point; it does not quantify the uncertainty "
        "of the answer or rank alternative designs.</div>"
    )
    body = scope + cards + health
    body += "<h2>Primary performance traces</h2>"
    body += figure(output / "01_speed_comparison.png", "Bounded and infinite-reference vehicle speed on the common track coordinate.")
    body += figure(output / "02_ratio_trace.png", "Bounded-CVT ratio demand and time spent at the available ratio limits.")
    body += figure(output / "03_energy_accounting.png", "Nominal physical energy accounting and finite-ratio opportunity diagnostics.")
    body += figure(plots / "speed_target_by_s.png", "Vehicle speed and the active empirical target along the common track coordinate.")
    body += figure(plots / "engine_and_ratio_by_s.png", "Engine-speed regulation and bounded-CVT ratio demand along the lap.")
    body += figure(plots / "longitudinal_force_balance.png", "Delivered tire force and non-braking longitudinal resistance components along the lap.")
    body += figure(plots / "braking_demand_by_s.png", "Brake-force command shown separately so short braking events do not hide the lower-magnitude road-load forces.")
    body += figure(plots / "loss_power_by_s.png", "Continuous physical loss power around the lap, excluding braking for scale clarity. Opportunity loss is not included because it is counterfactual rather than dissipative.")
    body += figure(plots / "brake_loss_power_by_s.png", "Brake dissipation shown separately from the continuous loss mechanisms.")

    body += "<h2>Mechanism summary</h2>"
    bounded_rows = _summary_rows(bounded, "bounded")
    reference_rows = _summary_rows(reference, "infinite_reference")
    body += dataframe_table(pd.DataFrame(bounded_rows + reference_rows), columns=["case", "metric", "value"])

    body += "<h2>Gate compliance</h2>"
    body += dataframe_table(gates, max_rows=200)
    body += "<h2>Obstacle energy by feature</h2>"
    if not obstacles.empty:
        energy_column = _first_existing(obstacles, ("bounded_energy_kj", "energy_kj", "bounded_obstacle_energy_kj"))
        if energy_column:
            obstacles = obstacles.sort_values(energy_column, ascending=False)
    body += dataframe_table(obstacles, max_rows=200)

    body += "<h2>Numerical and evidence status</h2>"
    body += dataframe_table(
        pd.DataFrame(
            [
                {"check": "bounded completed", "value": completed},
                {"check": "infinite reference completed", "value": bool(reference.get("completed", reference.get("lap_completed", True)))},
                {"check": "reference dominance", "value": dominance},
                {"check": "track bundle fingerprint", "value": manifest.get("track_bundle_content_fingerprint") or manifest.get("track_bundle_sha256", "")},
                {"check": "study", "value": manifest.get("study_name", "baseline")},
            ]
        )
    )
    target = output / REPORTS["nominal_simulation"].html_filename
    target.write_text(
        render_page(
            title=REPORTS["nominal_simulation"].title,
            subtitle=REPORTS["nominal_simulation"].question,
            body=body,
            report_key="nominal_simulation",
            source_note="Machine artifacts remain the source of truth.",
        ),
        encoding="utf-8",
    )
    return _register(output, "nominal_simulation", target)


def write_full_uncertainty_report(output: Path) -> Path:
    """Regenerate the full report from saved artifacts without simulation."""

    from .full_uncertainty import regenerate_full_uncertainty_report

    target = regenerate_full_uncertainty_report(output)
    return _register(output.resolve(), "full_uncertainty", target)

def write_design_comparison_report(output: Path) -> Path:
    output = output.resolve()
    rows = _read_csv(output / "replicate_results.csv")
    manifest = read_json(output / "run_manifest.json", {})
    plots = output / "report_plots"
    plots.mkdir(exist_ok=True)
    ranking = _design_ranking(rows)
    _design_plots(rows, ranking, plots)

    winner = ranking.iloc[0]["design_id"] if not ranking.empty else "unresolved"
    completion = float(ranking.iloc[0]["completion_fraction"]) if not ranking.empty else math.nan
    cards = metric_cards(
        [
            ("Candidates", str(len(ranking)), "note"),
            ("Scenario draws", str(int(rows["replicate"].nunique()) if "replicate" in rows else 0), "note"),
            ("Best median lap time", str(winner), "good" if len(ranking) else "warning"),
            ("Winner completion", _percent(completion), "good" if completion >= 0.95 else "warning"),
        ]
    )
    body = _scope_html("design_comparison") + cards
    body += (
        '<div class="card note"><strong>Paired comparison.</strong> Every candidate must see the same scenario '
        "realizations. The ranking table reports absolute performance, completion, and paired reference penalty; "
        "a candidate is not preferred merely because it fails difficult scenarios.</div>"
    )
    body += "<h2>Decision table</h2>" + dataframe_table(ranking, max_rows=100)
    body += (
        '<p class="subtitle">Paired win fraction and regret compare candidates within the same '
        'scenario. Non-completing candidates cannot win a scenario and are not rewarded by being '
        'absent from difficult cases.</p>'
    )
    body += "<h2>Absolute performance</h2>"
    body += figure(plots / "design_lap_time.png", "Median bounded lap time with p10–p90 scenario ranges.")
    body += figure(plots / "design_completion.png", "Completion probability for every candidate.")
    body += "<h2>CVT mechanism and reference comparison</h2>"
    body += figure(plots / "design_penalty.png", "Paired lap-time penalty relative to the infinite-ratio reference.")
    body += figure(plots / "design_ratio_time.png", "Median time spent at the maximum ratio, in the variable region, and at the minimum ratio.")
    body += figure(plots / "design_track_case_matrix.png", "Median bounded lap time for every design and reconstructed-track case represented in the paired scenarios.")
    body += "<h2>Candidate-level raw summaries</h2>" + dataframe_table(rows, max_rows=250)
    body += "<h2>Study contract</h2>" + f"<pre>{html.escape(json.dumps(_manifest_subset(manifest), indent=2, sort_keys=True))}</pre>"

    target = output / REPORTS["design_comparison"].html_filename
    target.write_text(
        render_page(
            title=REPORTS["design_comparison"].title,
            subtitle=REPORTS["design_comparison"].question,
            body=body,
            report_key="design_comparison",
            source_note="Rankings use paired scenarios and preserve incomplete cases.",
        ),
        encoding="utf-8",
    )
    return _register(output, "design_comparison", target)


def write_structural_report_manifest(output: Path) -> Path:
    """Regenerate and register the structural report from saved artifacts.

    The function name is retained for compatibility with the six-report router,
    but it now rebuilds plots and HTML rather than merely registering an old
    document. No simulation is executed.
    """

    output = output.resolve()
    from cvt_track_study.studies.structural_reporting import (
        regenerate_structural_outputs,
    )

    target = regenerate_structural_outputs(output)
    return _register(output, "structural_sensitivity", target)


def regenerate_framework_report(output: Path) -> Path:
    output = output.resolve()
    if (output / "track_build_manifest.json").is_file():
        return write_track_evidence_report(output)
    if (output / "track_robustness_manifest.json").is_file():
        from cvt_track_study.track.robustness import regenerate_track_robustness_report

        return regenerate_track_robustness_report(output)
    manifest = read_json(output / "run_manifest.json", {})
    study_type = str(manifest.get("study_type", ""))
    if (output / "comparison_summary.json").is_file() or study_type == "baseline":
        return write_nominal_simulation_report(output)
    if study_type == "track_robustness":
        from cvt_track_study.track.robustness import regenerate_track_robustness_report

        return regenerate_track_robustness_report(output)
    if study_type == "structural_sensitivity":
        return write_structural_report_manifest(output)
    if study_type == "full_uncertainty":
        return write_full_uncertainty_report(output)
    if study_type == "design_sweep":
        return write_design_comparison_report(output)
    raise ValueError(f"Could not determine report type for {output}")


def _scope_html(key: str) -> str:
    definition = REPORTS[key]
    return (
        '<div class="scope">'
        f'<div class="card note"><div class="label">Question</div><div>{html.escape(definition.question)}</div></div>'
        f'<div class="card"><div class="label">Held fixed</div><div>{html.escape(definition.fixed)}</div></div>'
        f'<div class="card"><div class="label">Varied</div><div>{html.escape(definition.varied)}</div></div>'
        "</div>"
    )


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _fmt(value: Any, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(number) else f"{number:.3f}{suffix}"


def _percent(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * value:.1f}%"


def _bool_fraction(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return math.nan
    values = frame[column]
    if values.dtype == object:
        values = values.astype(str).str.lower().map({"true": True, "false": False})
    values = values.dropna().astype(bool)
    return float(values.mean()) if len(values) else math.nan


def _summary_rows(summary: Mapping[str, Any], case: str) -> list[dict[str, Any]]:
    preferred = (
        "lap_time_s", "distance_m", "average_speed_kmh", "maximum_speed_kmh",
        "minimum_engine_rpm", "maximum_engine_rpm", "engine_energy_kj",
        "transmitted_energy_kj", "drivetrain_loss_energy_kj", "clutch_loss_energy_kj",
        "tire_slip_loss_energy_kj", "brake_loss_energy_kj", "rolling_loss_energy_kj",
        "aerodynamic_loss_energy_kj", "obstacle_loss_energy_kj", "time_maximum_ratio_s",
        "time_variable_ratio_s", "time_minimum_ratio_s",
    )
    return [
        {"case": case, "metric": key, "value": summary[key]}
        for key in preferred if key in summary
    ]


def _first_existing(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _quality_table(rows: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for label, column in (
        ("bounded completion", "bounded_completed"),
        ("reference completion", "reference_completed"),
        ("reference dominance", "reference_dominance_pass"),
        ("bounded gate compliance", "bounded_gates_compliant_0p5_kmh"),
        ("reference gate compliance", "reference_gates_compliant_0p5_kmh"),
    ):
        records.append(
            {"check": label, "value": _bool_fraction(rows, column), "unit": "fraction"}
        )
    records.extend(
        [
            {"check": "sampled input count", "value": manifest.get("sampled_input_count"), "unit": "count"},
            {"check": "sampled gate count", "value": manifest.get("sampled_gate_count"), "unit": "count"},
            {"check": "paired gate identities", "value": manifest.get("paired_gate_identity_count"), "unit": "count"},
            {"check": "track ensemble cases", "value": manifest.get("track_ensemble_case_count", 1), "unit": "count"},
            {"check": "parallel workers", "value": manifest.get("parallel_workers"), "unit": "count"},
        ]
    )
    return pd.DataFrame(records)


def _scenario_explorer(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    columns = [
        column
        for column in (
            "replicate",
            "track_case_id",
            "design_id",
            "bounded_completed",
            "bounded_lap_time_s",
            "infinite_reference_lap_time_s",
            "lap_time_penalty_vs_infinite_s",
            "finite_ratio_opportunity_loss_energy_kj",
            "bounded_time_minimum_ratio_s",
            "bounded_max_gate_excess_kmh",
            "reference_dominance_pass",
        )
        if column in rows
    ]
    if "bounded_completed" in rows:
        completed_mask = rows["bounded_completed"].astype(bool)
        completed = rows.loc[completed_mask].copy()
    else:
        completed = rows.copy()
    pieces: list[pd.DataFrame] = []
    if not completed.empty and "bounded_lap_time_s" in completed:
        ordered = completed.sort_values("bounded_lap_time_s")
        midpoint = len(ordered) // 2
        pieces.extend(
            [
                ordered.head(3),
                ordered.iloc[max(0, midpoint - 1) : midpoint + 2],
                ordered.tail(3),
            ]
        )
    if "bounded_completed" in rows:
        pieces.append(rows.loc[~rows["bounded_completed"].astype(bool)].head(10))
    if not pieces:
        return rows[columns].head(20)
    selected = pd.concat(pieces, ignore_index=True).drop_duplicates()
    return selected[columns]


def _manifest_subset(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "study_name", "study_type", "vehicle_id", "sampling_mode", "scenario_count",
        "design_point_count", "sampled_input_paths", "sampled_gate_ids", "gate_sampling_policy",
        "paired_gate_identity_count", "independent_gate_fallback_ids", "uncertainty_not_propagated",
        "track_ensemble_case_count", "track_ensemble_case_ids", "track_ensemble_policy",
        "numerical_quality", "evidence_assessment", "track_bundle_content_fingerprint",
    )
    return {key: manifest.get(key) for key in keys if key in manifest}


def _nominal_trace_plots(
    bounded: pd.DataFrame,
    reference: pd.DataFrame,
    plots: Path,
) -> None:
    if bounded.empty:
        return
    x_key = "distance_m" if "distance_m" in bounded else "time_s"
    x_label = "Along-track coordinate, s [m]" if x_key == "distance_m" else "Time [s]"
    x = pd.to_numeric(bounded[x_key], errors="coerce")

    if "vehicle_speed_kmh" in bounded:
        figure_obj, axis = plt.subplots(figsize=(12, 5.4))
        axis.plot(x, bounded["vehicle_speed_kmh"], label="bounded CVT")
        if not reference.empty and x_key in reference and "vehicle_speed_kmh" in reference:
            axis.plot(reference[x_key], reference["vehicle_speed_kmh"], label="infinite reference", alpha=0.8)
        if "target_speed_mps" in bounded:
            axis.plot(x, 3.6 * pd.to_numeric(bounded["target_speed_mps"], errors="coerce"), label="active target", linewidth=1.0, alpha=0.75)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Speed [km/h]")
        axis.set_title("Nominal speed response on the reconstructed track")
        axis.legend()
        axis.grid(True, alpha=0.25)
        figure_obj.tight_layout()
        figure_obj.savefig(plots / "speed_target_by_s.png", dpi=180)
        plt.close(figure_obj)

    if "engine_speed_rpm" in bounded and "cvt_ratio" in bounded:
        figure_obj, axis = plt.subplots(figsize=(12, 5.4))
        axis.plot(x, bounded["engine_speed_rpm"], label="engine speed")
        if "engine_speed_rpm" in reference and x_key in reference:
            axis.plot(reference[x_key], reference["engine_speed_rpm"], label="infinite-reference engine speed", alpha=0.65)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Engine speed [rpm]")
        axis.grid(True, alpha=0.25)
        ratio_axis = axis.twinx()
        ratio_axis.plot(x, bounded["cvt_ratio"], label="bounded CVT ratio", linewidth=1.2, alpha=0.75)
        bounded_ratio = pd.to_numeric(bounded["cvt_ratio"], errors="coerce")
        ratio_ceiling = max(4.0, 1.25 * float(bounded_ratio.replace([np.inf, -np.inf], np.nan).max()))
        if "ratio_required" in bounded:
            required = pd.to_numeric(bounded["ratio_required"], errors="coerce")
            required = required.where(required.between(0.0, ratio_ceiling))
            ratio_axis.plot(x, required, label="required ratio", linewidth=0.9, alpha=0.5)
        ratio_axis.set_ylim(0.0, ratio_ceiling)
        ratio_axis.set_ylabel("CVT reduction ratio")
        handles_a, labels_a = axis.get_legend_handles_labels()
        handles_b, labels_b = ratio_axis.get_legend_handles_labels()
        axis.legend(handles_a + handles_b, labels_a + labels_b, loc="best")
        axis.set_title("Engine regulation and CVT ratio demand")
        figure_obj.tight_layout()
        figure_obj.savefig(plots / "engine_and_ratio_by_s.png", dpi=180)
        plt.close(figure_obj)

    force_columns = [
        column
        for column in (
            "tire_force_n",
            "rolling_force_n",
            "aerodynamic_force_n",
            "obstacle_force_n",
            "grade_force_n",
        )
        if column in bounded
    ]
    if force_columns:
        figure_obj, axis = plt.subplots(figsize=(12, 5.7))
        for column in force_columns:
            axis.plot(x, bounded[column], label=column.replace("_force_n", "").replace("_command_n", "").replace("_", " "), linewidth=1.0)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Force [N]")
        axis.set_title("Longitudinal force balance by track position")
        axis.legend(ncol=3)
        axis.grid(True, alpha=0.25)
        figure_obj.tight_layout()
        figure_obj.savefig(plots / "longitudinal_force_balance.png", dpi=180)
        plt.close(figure_obj)

    if "brake_force_command_n" in bounded:
        brake = pd.to_numeric(bounded["brake_force_command_n"], errors="coerce")
        figure_obj, axis = plt.subplots(figsize=(12, 4.8))
        axis.plot(x, brake, label="brake force command")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Brake-force command [N]")
        axis.set_title("Braking demand around the nominal lap")
        axis.grid(True, alpha=0.25)
        figure_obj.tight_layout()
        figure_obj.savefig(plots / "braking_demand_by_s.png", dpi=180)
        plt.close(figure_obj)

    loss_columns = [
        column
        for column in (
            "drivetrain_loss_power_w",
            "clutch_loss_power_w",
            "tire_slip_loss_power_w",
            "rolling_loss_power_w",
            "aerodynamic_loss_power_w",
            "obstacle_loss_power_w",
        )
        if column in bounded
    ]
    if loss_columns:
        figure_obj, axis = plt.subplots(figsize=(12, 5.7))
        for column in loss_columns:
            axis.plot(x, bounded[column], label=column.replace("_loss_power_w", "").replace("_", " "), linewidth=0.95)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Loss power [W]")
        axis.set_title("Physical loss power around the nominal lap")
        axis.legend(ncol=3)
        axis.grid(True, alpha=0.25)
        figure_obj.tight_layout()
        figure_obj.savefig(plots / "loss_power_by_s.png", dpi=180)
        plt.close(figure_obj)

    if "brake_loss_power_w" in bounded:
        figure_obj, axis = plt.subplots(figsize=(12, 4.8))
        axis.plot(x, bounded["brake_loss_power_w"], label="brake loss power")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Brake loss power [W]")
        axis.set_title("Brake dissipation around the nominal lap")
        axis.grid(True, alpha=0.25)
        figure_obj.tight_layout()
        figure_obj.savefig(plots / "brake_loss_power_by_s.png", dpi=180)
        plt.close(figure_obj)


def _uncertainty_plots(rows: pd.DataFrame, attribution: pd.DataFrame, energy: pd.DataFrame, plots: Path) -> None:
    if rows.empty:
        return
    if "bounded_lap_time_s" in rows and "infinite_reference_lap_time_s" in rows:
        figure_obj, axis = plt.subplots(figsize=(10, 5.5))
        for column, label in (("bounded_lap_time_s", "bounded"), ("infinite_reference_lap_time_s", "infinite reference")):
            values = pd.to_numeric(rows[column], errors="coerce").dropna()
            axis.hist(values, bins=min(24, max(6, int(np.sqrt(len(values))))), alpha=0.55, label=label)
        axis.set_xlabel("Lap time [s]"); axis.set_ylabel("Scenario count"); axis.set_title("Absolute lap-time uncertainty"); axis.legend(); axis.grid(True, axis="y", alpha=.25)
        figure_obj.tight_layout(); figure_obj.savefig(plots / "absolute_lap_time_distribution.png", dpi=180); plt.close(figure_obj)
    if "design_id" in rows and "bounded_completed" in rows:
        grouped = rows.groupby("design_id")["bounded_completed"].apply(lambda s: s.astype(bool).mean()).sort_values()
        figure_obj, axis = plt.subplots(figsize=(10, max(4.5, .42 * len(grouped))))
        axis.barh(grouped.index.astype(str), grouped.values); axis.set_xlim(0, 1); axis.set_xlabel("Completion fraction"); axis.set_title("Completion by design"); axis.grid(True, axis="x", alpha=.25)
        figure_obj.tight_layout(); figure_obj.savefig(plots / "completion_by_design.png", dpi=180); plt.close(figure_obj)
    if "lap_time_penalty_vs_infinite_s" in rows:
        values = pd.to_numeric(rows["lap_time_penalty_vs_infinite_s"], errors="coerce").dropna()
        figure_obj, axis = plt.subplots(figsize=(10, 5.2)); axis.hist(values, bins=min(24, max(6, int(np.sqrt(len(values)))))); axis.axvline(0, linewidth=1); axis.set_xlabel("Bounded minus infinite lap time [s]"); axis.set_ylabel("Scenario count"); axis.set_title("Paired finite-ratio time penalty"); axis.grid(True, axis="y", alpha=.25)
        figure_obj.tight_layout(); figure_obj.savefig(plots / "paired_penalty_distribution.png", dpi=180); plt.close(figure_obj)
    ratio_cols = [c for c in ("bounded_time_maximum_ratio_s", "bounded_time_variable_ratio_s", "bounded_time_minimum_ratio_s") if c in rows]
    if ratio_cols:
        figure_obj, axis = plt.subplots(figsize=(10, 5.5)); axis.boxplot([pd.to_numeric(rows[c], errors="coerce").dropna() for c in ratio_cols], tick_labels=[c.replace("bounded_time_", "").replace("_s", "").replace("_", " ") for c in ratio_cols], showfliers=False); axis.set_ylabel("Time [s]"); axis.set_title("CVT ratio-region occupancy"); axis.grid(True, axis="y", alpha=.25)
        figure_obj.tight_layout(); figure_obj.savefig(plots / "ratio_occupancy_distribution.png", dpi=180); plt.close(figure_obj)
    loss_cols = [c for c in ("bounded_drivetrain_loss_energy_kj", "bounded_clutch_loss_energy_kj", "bounded_tire_slip_loss_energy_kj", "bounded_brake_loss_energy_kj", "bounded_rolling_loss_energy_kj", "bounded_aerodynamic_loss_energy_kj", "bounded_obstacle_loss_energy_kj") if c in rows]
    if loss_cols:
        med = np.array([pd.to_numeric(rows[c], errors="coerce").median() for c in loss_cols]); low=np.array([pd.to_numeric(rows[c], errors="coerce").quantile(.1) for c in loss_cols]); high=np.array([pd.to_numeric(rows[c], errors="coerce").quantile(.9) for c in loss_cols]); x=np.arange(len(loss_cols))
        figure_obj, axis = plt.subplots(figsize=(11,5.5)); axis.bar(x,med); axis.errorbar(x,med,yerr=np.vstack((med-low,high-med)),fmt="none",capsize=3); axis.set_xticks(x,[c.replace("bounded_","").replace("_loss_energy_kj","").replace("_"," ") for c in loss_cols],rotation=25,ha="right"); axis.set_ylabel("Energy [kJ]"); axis.set_title("Physical loss uncertainty"); axis.grid(True,axis="y",alpha=.25)
        figure_obj.tight_layout(); figure_obj.savefig(plots / "physical_loss_distribution.png", dpi=180); plt.close(figure_obj)
    if "track_case_id" in rows and "bounded_lap_time_s" in rows:
        grouped = []
        for case_id, group in rows.groupby("track_case_id", sort=False):
            values = pd.to_numeric(group["bounded_lap_time_s"], errors="coerce").dropna()
            penalty = pd.to_numeric(group.get("lap_time_penalty_vs_infinite_s"), errors="coerce").dropna()
            if len(values):
                grouped.append(
                    {
                        "case_id": str(case_id),
                        "lap_p10": float(values.quantile(0.1)),
                        "lap_median": float(values.median()),
                        "lap_p90": float(values.quantile(0.9)),
                        "penalty_p10": float(penalty.quantile(0.1)) if len(penalty) else math.nan,
                        "penalty_median": float(penalty.median()) if len(penalty) else math.nan,
                        "penalty_p90": float(penalty.quantile(0.9)) if len(penalty) else math.nan,
                    }
                )
        track_summary = pd.DataFrame(grouped)
        if not track_summary.empty:
            track_summary = track_summary.sort_values("lap_median")
            y = np.arange(len(track_summary))
            figure_obj, axis = plt.subplots(figsize=(11, max(5.5, 0.4 * len(track_summary))))
            med = track_summary["lap_median"].to_numpy(float)
            low = track_summary["lap_p10"].to_numpy(float)
            high = track_summary["lap_p90"].to_numpy(float)
            axis.errorbar(med, y, xerr=np.vstack((med - low, high - med)), fmt="o", capsize=3)
            axis.set_yticks(y, track_summary["case_id"])
            axis.set_xlabel("Bounded lap time [s]")
            axis.set_title("Absolute performance across track reconstructions")
            axis.grid(True, axis="x", alpha=0.25)
            figure_obj.tight_layout()
            figure_obj.savefig(plots / "track_case_lap_time.png", dpi=180)
            plt.close(figure_obj)

            valid = track_summary.dropna(subset=["penalty_median"]).sort_values("penalty_median")
            if not valid.empty:
                y = np.arange(len(valid))
                figure_obj, axis = plt.subplots(figsize=(11, max(5.5, 0.4 * len(valid))))
                med = valid["penalty_median"].to_numpy(float)
                low = valid["penalty_p10"].to_numpy(float)
                high = valid["penalty_p90"].to_numpy(float)
                axis.errorbar(med, y, xerr=np.vstack((med - low, high - med)), fmt="o", capsize=3)
                axis.axvline(0, linewidth=1)
                axis.set_yticks(y, valid["case_id"])
                axis.set_xlabel("Bounded minus infinite lap time [s]")
                axis.set_title("Finite-range penalty across track reconstructions")
                axis.grid(True, axis="x", alpha=0.25)
                figure_obj.tight_layout()
                figure_obj.savefig(plots / "track_case_penalty.png", dpi=180)
                plt.close(figure_obj)

    if not attribution.empty:
        metric_col = _first_existing(attribution, ("metric", "output_metric")); path_col=_first_existing(attribution,("path","input_path")); importance_col=_first_existing(attribution,("relative_screening_importance","importance","absolute_rank_score"))
        data=attribution
        if metric_col and (data[metric_col].astype(str)=="bounded_lap_time_s").any(): data=data[data[metric_col].astype(str)=="bounded_lap_time_s"]
        if path_col and importance_col:
            data=data.assign(_importance=pd.to_numeric(data[importance_col],errors="coerce")).dropna(subset=["_importance"]).nlargest(15,"_importance").sort_values("_importance")
            if not data.empty:
                figure_obj, axis=plt.subplots(figsize=(11,max(5.5,.42*len(data)))); axis.barh(data[path_col].astype(str),data["_importance"]); axis.set_xlabel("Relative screening importance"); axis.set_title("Top associations with bounded lap time"); axis.grid(True,axis="x",alpha=.25); figure_obj.tight_layout(); figure_obj.savefig(plots / "top_uncertainty_drivers.png",dpi=180); plt.close(figure_obj)


def _design_ranking(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "design_id" not in rows:
        return pd.DataFrame()

    paired = rows.copy()
    paired["_lap_time"] = pd.to_numeric(
        paired.get("bounded_lap_time_s"), errors="coerce"
    )
    if "bounded_completed" in paired:
        paired.loc[~paired["bounded_completed"].astype(bool), "_lap_time"] = math.inf
    scenario_keys = ["replicate"]
    if "scenario_seed" in paired:
        scenario_keys.append("scenario_seed")
    best = paired.groupby(scenario_keys)["_lap_time"].transform("min")
    paired["_regret_s"] = paired["_lap_time"] - best
    paired["_paired_win"] = np.isfinite(paired["_lap_time"]) & np.isclose(
        paired["_lap_time"], best, rtol=0.0, atol=1e-9
    )

    records = []
    for design_id, group in rows.groupby("design_id", sort=False):
        lap = pd.to_numeric(group.get("bounded_lap_time_s"), errors="coerce")
        penalty = pd.to_numeric(group.get("lap_time_penalty_vs_infinite_s"), errors="coerce")
        energy = pd.to_numeric(
            group.get("finite_ratio_opportunity_loss_energy_kj"), errors="coerce"
        )
        completed = (
            group["bounded_completed"].astype(bool)
            if "bounded_completed" in group
            else pd.Series(True, index=group.index)
        )
        paired_group = paired[paired["design_id"] == design_id]
        finite_regret = pd.to_numeric(
            paired_group["_regret_s"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        records.append(
            {
                "design_id": design_id,
                "design_value": group["design_value"].iloc[0] if "design_value" in group else "",
                "scenario_count": int(group["replicate"].nunique()) if "replicate" in group else len(group),
                "completion_fraction": float(completed.mean()),
                "paired_win_fraction": float(paired_group["_paired_win"].mean()),
                "paired_regret_median_s": float(finite_regret.median()) if len(finite_regret) else math.inf,
                "paired_regret_p90_s": float(finite_regret.quantile(0.9)) if len(finite_regret) else math.inf,
                "lap_time_p10_s": float(lap.quantile(0.1)),
                "lap_time_median_s": float(lap.median()),
                "lap_time_p90_s": float(lap.quantile(0.9)),
                "penalty_median_s": float(penalty.median()),
                "opportunity_loss_median_kj": float(energy.median()),
                "minimum_ratio_time_median_s": (
                    float(
                        pd.to_numeric(
                            group.get("bounded_time_minimum_ratio_s"), errors="coerce"
                        ).median()
                    )
                    if "bounded_time_minimum_ratio_s" in group
                    else math.nan
                ),
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "completion_fraction",
                "paired_regret_median_s",
                "lap_time_median_s",
            ],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )


def _design_plots(rows: pd.DataFrame, ranking: pd.DataFrame, plots: Path) -> None:
    if ranking.empty: return
    labels=ranking["design_id"].astype(str).tolist(); x=np.arange(len(ranking))
    fig,ax=plt.subplots(figsize=(max(9,.8*len(labels)),5.5)); med=ranking["lap_time_median_s"].to_numpy(float); low=ranking["lap_time_p10_s"].to_numpy(float); high=ranking["lap_time_p90_s"].to_numpy(float); ax.bar(x,med); ax.errorbar(x,med,yerr=np.vstack((med-low,high-med)),fmt="none",capsize=3); ax.set_xticks(x,labels,rotation=30,ha="right"); ax.set_ylabel("Lap time [s]"); ax.set_title("Absolute bounded performance by design"); ax.grid(True,axis="y",alpha=.25); fig.tight_layout(); fig.savefig(plots/"design_lap_time.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(max(9,.8*len(labels)),4.8)); ax.bar(x,ranking["completion_fraction"]); ax.set_ylim(0,1); ax.set_xticks(x,labels,rotation=30,ha="right"); ax.set_ylabel("Completion fraction"); ax.set_title("Completion reliability"); ax.grid(True,axis="y",alpha=.25); fig.tight_layout(); fig.savefig(plots/"design_completion.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(max(9,.8*len(labels)),4.8)); ax.bar(x,ranking["penalty_median_s"]); ax.axhline(0,linewidth=1); ax.set_xticks(x,labels,rotation=30,ha="right"); ax.set_ylabel("Median penalty [s]"); ax.set_title("Paired bounded-versus-infinite penalty"); ax.grid(True,axis="y",alpha=.25); fig.tight_layout(); fig.savefig(plots/"design_penalty.png",dpi=180); plt.close(fig)
    ratio_cols=[c for c in ("bounded_time_maximum_ratio_s","bounded_time_variable_ratio_s","bounded_time_minimum_ratio_s") if c in rows]
    if ratio_cols:
        medians=rows.groupby("design_id")[ratio_cols].median().reindex(labels); fig,ax=plt.subplots(figsize=(max(9,.85*len(labels)),5.5)); bottom=np.zeros(len(labels));
        for c in ratio_cols:
            vals=medians[c].to_numpy(float); ax.bar(x,vals,bottom=bottom,label=c.replace("bounded_time_","").replace("_s","").replace("_"," ")); bottom+=vals
        ax.set_xticks(x,labels,rotation=30,ha="right"); ax.set_ylabel("Median time [s]"); ax.set_title("CVT ratio-region occupancy by design"); ax.legend(); ax.grid(True,axis="y",alpha=.25); fig.tight_layout(); fig.savefig(plots/"design_ratio_time.png",dpi=180); plt.close(fig)
    if "track_case_id" in rows and "bounded_lap_time_s" in rows:
        matrix = rows.pivot_table(
            index="track_case_id",
            columns="design_id",
            values="bounded_lap_time_s",
            aggfunc="median",
        )
        if not matrix.empty:
            fig, ax = plt.subplots(
                figsize=(max(9, 0.8 * len(matrix.columns)), max(5.5, 0.42 * len(matrix.index)))
            )
            image = ax.imshow(matrix.to_numpy(float), aspect="auto", interpolation="nearest")
            ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
            ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
            ax.set_title("Median bounded lap time by design and track reconstruction")
            fig.colorbar(image, ax=ax, label="Lap time [s]")
            fig.tight_layout()
            fig.savefig(plots / "design_track_case_matrix.png", dpi=180)
            plt.close(fig)
