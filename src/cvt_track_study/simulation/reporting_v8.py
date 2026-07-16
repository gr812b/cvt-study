"""Phase 8 decision-first reports for the nominal baseline comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from cvt_track_study.runtime.evidence import unavailable_evidence_assessment


def write_baseline_hierarchy(
    *,
    output: Path,
    bounded: Mapping[str, Any],
    reference: Mapping[str, Any],
    comparison: Mapping[str, Any],
    manifest: dict[str, Any],
    evidence_assessment: Mapping[str, Any] | None = None,
) -> None:
    """Add a compact summary, decision trace, and navigable appendix.

    The infinite-ratio case is a counterfactual mechanism reference, not a
    buildable design recommendation.  The wording below keeps that distinction
    explicit while still making the nominal comparison easy to consume.
    """

    gate_quality = _gate_quality(output / "gate_compliance.csv")
    vehicle_limit = 0.01
    powertrain_limit = 0.01
    vehicle_error = max(
        abs(float(bounded.get("energy_balance_relative_error", float("inf")))),
        abs(float(reference.get("energy_balance_relative_error", float("inf")))),
    )
    powertrain_error = max(
        abs(float(bounded.get("powertrain_energy_balance_relative_error", float("inf")))),
        abs(float(reference.get("powertrain_energy_balance_relative_error", float("inf")))),
    )
    quality = {
        "all_cases_completed": bool(bounded.get("completed")) and bool(reference.get("completed")),
        "all_reference_dominance_checks_pass": bool(comparison.get("reference_dominance_pass")),
        "all_gate_compliance_checks_pass": gate_quality,
        "maximum_abs_vehicle_energy_balance_relative_error": vehicle_error,
        "maximum_allowed_vehicle_energy_balance_relative_error": vehicle_limit,
        "vehicle_energy_balance_pass": vehicle_error <= vehicle_limit,
        "maximum_abs_powertrain_energy_balance_relative_error": powertrain_error,
        "maximum_allowed_powertrain_energy_balance_relative_error": powertrain_limit,
        "powertrain_energy_balance_pass": powertrain_error <= powertrain_limit,
    }
    quality["numerically_valid"] = all(
        quality[key]
        for key in (
            "all_cases_completed",
            "all_reference_dominance_checks_pass",
            "all_gate_compliance_checks_pass",
            "vehicle_energy_balance_pass",
            "powertrain_energy_balance_pass",
        )
    )
    evidence = dict(
        evidence_assessment
        or manifest.get("evidence_assessment", {})
        or unavailable_evidence_assessment()
    )
    readiness = {
        "numerically_valid": bool(quality["numerically_valid"]),
        "evidence_ready": bool(evidence.get("ready", False)),
        "statistically_ready": False,
        "decision_ready": False,
        "blocking_reasons": list(
            dict.fromkeys(
                [
                    *(str(item) for item in evidence.get("blocking_reasons", [])),
                    "A nominal baseline is a mechanism comparison, not a design recommendation.",
                    "Track and model uncertainty have not been propagated in a baseline run.",
                ]
            )
        ),
    }
    manifest.update(
        {
            "framework_contract": "measured-track-drivetrain-framework-v0.8",
            "study_name": manifest.get("study", "baseline"),
            "study_type": "baseline",
            "scenario_count": 1,
            "design_point_count": 1,
            "numerical_quality": quality,
            "evidence_assessment": evidence,
            "decision_readiness": readiness,
        }
    )
    _write_json(output / "run_manifest.json", manifest)

    warnings: list[str] = []
    if not quality["numerically_valid"]:
        warnings.append("At least one numerical quality check failed; inspect the detailed report before using the comparison.")
    warnings.extend(str(item) for item in evidence.get("warnings", []))
    warnings.extend(
        [
            "This is one nominal scenario, so it does not quantify track or model uncertainty.",
            "The infinite-ratio result is a counterfactual opportunity bound, not a realizable drivetrain.",
        ]
    )
    next_actions = _baseline_next_actions(evidence)
    summary = _summary_text(
        comparison, quality, evidence, readiness, warnings, next_actions
    )
    report = _report_text(bounded, reference, comparison, quality, evidence, readiness, warnings)
    trace = _decision_trace(comparison, quality, evidence, readiness, warnings)
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "decision_trace.md").write_text(trace, encoding="utf-8")
    appendix = output / "appendix"
    appendix.mkdir(exist_ok=True)
    (appendix / "README.md").write_text(_appendix_text(), encoding="utf-8")


def regenerate_baseline_reports(output: Path) -> None:
    """Regenerate baseline Markdown reports from preserved JSON/CSV artifacts."""

    bounded = _read_json(output / "bounded_summary.json")
    reference = _read_json(output / "infinite_reference_summary.json")
    comparison = _read_json(output / "comparison_summary.json")
    manifest = _read_json(output / "run_manifest.json")
    write_baseline_hierarchy(
        output=output,
        bounded=bounded,
        reference=reference,
        comparison=comparison,
        manifest=manifest,
        evidence_assessment=manifest.get("evidence_assessment"),
    )


def _gate_quality(path: Path) -> bool:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return all(
        row.get("bounded_compliant_0p5_kmh", "").lower() == "true"
        and row.get("reference_compliant_0p5_kmh", "").lower() == "true"
        for row in rows
    )


def _summary_text(
    comparison: Mapping[str, Any],
    quality: Mapping[str, Any],
    evidence: Mapping[str, Any],
    readiness: Mapping[str, Any],
    warnings: list[str],
    next_actions: list[str],
) -> str:
    return "\n".join(
        [
            "# Engineering decision summary",
            "",
            "**Finding:** The bounded ideal CVT is compared with an otherwise-identical infinite-ratio opportunity reference.",
            "",
            f"**Nominal lap-time penalty:** {float(comparison['lap_time_penalty_vs_infinite_s']):.3f} s  ",
            f"**Finite-ratio opportunity loss:** {float(comparison['finite_ratio_opportunity_loss_energy_kj']):.3f} kJ  ",
            f"**Numerically valid:** `{quality['numerically_valid']}`  ",
            f"**Evidence ready:** `{evidence.get('ready', False)}`  ",
            f"**Decision ready:** `{readiness['decision_ready']}`  ",
            "**Confidence:** `nominal mechanism check`",
            "",
            "## Warnings",
            "",
            *[f"- {item}" for item in warnings],
            "",
            "## Recommended next actions",
            "",
            *[f"- {item}" for item in next_actions],
            "",
            "## Drill down",
            "",
            "- [Full technical report](REPORT.md)",
            "- [Decision trace](decision_trace.md)",
            "- [Machine-readable appendix](appendix/README.md)",
            "",
        ]
    )


def _baseline_next_actions(evidence: Mapping[str, Any]) -> list[str]:
    if not bool(evidence.get("ready", False)):
        return [
            "Resolve or explicitly disposition the project-input and track-review blockers listed above.",
            "Rebuild and review the Track Evidence Bundle after correcting source track inputs.",
            "Run the nominal baseline again before launching paired studies.",
        ]
    return [
        "Run measured-track robustness to test whether the result survives plausible measured laps.",
        "Run structural sensitivity to identify assumptions that move the result.",
        "Run a design sweep before treating a ratio choice as a recommendation.",
    ]


def _report_text(
    bounded: Mapping[str, Any],
    reference: Mapping[str, Any],
    comparison: Mapping[str, Any],
    quality: Mapping[str, Any],
    evidence: Mapping[str, Any],
    readiness: Mapping[str, Any],
    warnings: list[str],
) -> str:
    components = (
        ("Drivetrain efficiency", "drivetrain_loss_energy_kj"),
        ("Launch clutch", "clutch_loss_energy_kj"),
        ("Tire slip", "tire_slip_loss_energy_kj"),
        ("Braking", "brake_loss_energy_kj"),
        ("Rolling resistance", "rolling_loss_energy_kj"),
        ("Aerodynamic drag", "aerodynamic_loss_energy_kj"),
        ("Obstacles", "obstacle_loss_energy_kj"),
    )
    lines = [
        "# Nominal bounded-CVT baseline report",
        "",
        "This report starts with the engineering comparison and then links to the trace-level evidence. The infinite-ratio case removes finite ratio bounds while preserving the same vehicle, track evidence, driver gates, and launch contract.",
        "",
        "## 1. Result",
        "",
        "| Metric | Bounded ideal CVT | Infinite-ratio reference | Difference |",
        "| --- | ---: | ---: | ---: |",
        f"| Lap time [s] | {float(bounded['lap_time_s']):.4f} | {float(reference['lap_time_s']):.4f} | {float(comparison['lap_time_penalty_vs_infinite_s']):+.4f} |",
        f"| Average speed [km/h] | {float(bounded['average_speed_kmh']):.3f} | {float(reference['average_speed_kmh']):.3f} | {float(bounded['average_speed_kmh']) - float(reference['average_speed_kmh']):+.3f} |",
        f"| Opportunity loss [kJ] | {float(bounded['finite_ratio_opportunity_loss_energy_kj']):.3f} | {float(reference['finite_ratio_opportunity_loss_energy_kj']):.3f} | {float(comparison['finite_ratio_opportunity_loss_energy_kj']):+.3f} |",
        "",
        "## 2. Numerical quality",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Both cases completed | `{quality['all_cases_completed']}` |",
        f"| Reference dominance | `{quality['all_reference_dominance_checks_pass']}` |",
        f"| Accepted-gate compliance | `{quality['all_gate_compliance_checks_pass']}` |",
        f"| Vehicle energy closure | `{quality['vehicle_energy_balance_pass']}` ({float(quality['maximum_abs_vehicle_energy_balance_relative_error']):.3e}) |",
        f"| Powertrain energy closure | `{quality['powertrain_energy_balance_pass']}` ({float(quality['maximum_abs_powertrain_energy_balance_relative_error']):.3e}) |",
        "",
        "A failed quality check overrides an attractive performance number.",
        "",
        "## 3. Evidence readiness",
        "",
        f"Evidence ready: `{evidence.get('ready', False)}`. Decision ready: `{readiness.get('decision_ready', False)}`.",
        "",
        *_evidence_table(evidence),
        "",
        "A numerically valid run remains exploratory while project-input or track-review warnings are unresolved.",
        "",
        "## 4. Physical energy accounting",
        "",
        "These rows are physical loss channels. Finite-ratio opportunity loss is counterfactual and is deliberately not added to this balance.",
        "",
        "| Component | Bounded [kJ] | Reference [kJ] |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {label} | {float(bounded.get(key, 0.0)):.3f} | {float(reference.get(key, 0.0)):.3f} |"
        for label, key in components
    )
    lines.extend(
        [
            "",
            "Feature-level obstacle energy is in [obstacle_energy_by_feature.csv](obstacle_energy_by_feature.csv).",
            "",
            "## 5. Ratio occupancy and trace evidence",
            "",
            f"The bounded case spent {float(bounded.get('time_minimum_ratio_s', 0.0)):.3f} s at minimum reduction, {float(bounded.get('time_maximum_ratio_s', 0.0)):.3f} s at maximum reduction, and {float(bounded.get('time_variable_ratio_s', 0.0)):.3f} s in the variable-ratio region.",
            "",
            "Use [01_speed_comparison.png](01_speed_comparison.png), [02_ratio_trace.png](02_ratio_trace.png), and the two trace CSV files to inspect where the comparison was created.",
            "",
            "## 6. Gate behavior",
            "",
            "[gate_compliance.csv](gate_compliance.csv) records every accepted measured speed gate, its target, the two simulated speeds, and the compliance tolerance.",
            "",
            "## 7. Warnings and scope",
            "",
            *[f"- {item}" for item in warnings],
            "- Telemetry elevation is screened, but grade force remains disabled pending a material paired sensitivity.",
            "- The current tire model is longitudinal and intentionally compact.",
            "- The bounded ideal CVT is an intentionally reduced-order comparison mechanism.",
            "",
            "## 8. Provenance",
            "",
            "The exact evidence bundle, resolved inputs, run identity, and hashes are recorded in `track_bundle.json`, `resolved_inputs/`, `run_manifest.json`, `provenance.json`, and [provenance_graph.svg](provenance_graph.svg).",
            "",
            "## 9. Appendix",
            "",
            "The [appendix index](appendix/README.md) maps this report to all machine-readable outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def _decision_trace(
    comparison: Mapping[str, Any],
    quality: Mapping[str, Any],
    evidence: Mapping[str, Any],
    readiness: Mapping[str, Any],
    warnings: list[str],
) -> str:
    lines = [
        "# Decision trace",
        "",
        "1. **Question** — What is lost because the declared ideal CVT has finite ratio bounds?",
        f"2. **Observed nominal time penalty** — {float(comparison['lap_time_penalty_vs_infinite_s']):.4f} s.",
        f"3. **Observed nominal opportunity loss** — {float(comparison['finite_ratio_opportunity_loss_energy_kj']):.4f} kJ.",
        f"4. **Reference dominance check** — `{quality['all_reference_dominance_checks_pass']}`.",
        f"5. **Numerically valid** — `{quality['numerically_valid']}`.",
        f"6. **Evidence ready** — `{evidence.get('ready', False)}`.",
        f"7. **Decision ready** — `{readiness.get('decision_ready', False)}`.",
        "8. **Interpretation** — This establishes a nominal mechanism comparison, not a robust design ranking.",
        "9. **Constraints on interpretation**",
        *[f"   - {item}" for item in warnings],
    ]
    return "\n".join(lines) + "\n"


def _evidence_table(evidence: Mapping[str, Any]) -> list[str]:
    counts = evidence.get("track_review_status_counts", {})
    identities = evidence.get("evidence_identity_counts", {})
    return [
        "| Evidence check | Result |",
        "| --- | --- |",
        f"| Project validation warnings | {int(evidence.get('project_warning_count', 0))} |",
        f"| Track events accepted | {int(counts.get('accepted', 0))} |",
        f"| Track events recommended for review | {int(counts.get('recommended_review', 0))} |",
        f"| Track events marked must-fix | {int(counts.get('must_fix', 0))} |",
        f"| Run identities | {int(identities.get('run_id', 0))} |",
        f"| Vehicle identities | {int(identities.get('vehicle_id', 0))} |",
        f"| Driver identities | {int(identities.get('driver_id', 0))} |",
    ]


def _appendix_text() -> str:
    return """# Machine-readable appendix

- `../bounded_summary.json` and `../infinite_reference_summary.json` — scalar case outputs
- `../comparison_summary.json` — finite-ratio deltas and dominance check
- `../bounded_trace.csv` and `../infinite_reference_trace.csv` — trace-level state and energy history
- `../gate_compliance.csv` — measured-gate targets and simulated compliance
- `../obstacle_energy_by_feature.csv` — feature-level energy accounting
- `../resolved_simulation_case.json` — fully resolved mechanism inputs
- `../run_manifest.json` — run identity and numerical quality
- `../provenance.json` — version, command, and evidence hashes
- `../track_bundle.json` and `../track_bundle.sha256` — exact Track Evidence Bundle
- `../resolved_inputs/` — complete resolved project configuration
"""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot regenerate baseline report; missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
