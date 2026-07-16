"""Hierarchical human-facing study reports generated from machine artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def write_hierarchical_reports(
    *,
    output: Path,
    decision: Mapping[str, Any],
    summary: Mapping[str, Any],
    convergence: Mapping[str, Any],
    energy: Mapping[str, Any],
    attribution: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    (output / "SUMMARY.md").write_text(
        _summary_text(decision, manifest), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        _report_text(decision, summary, convergence, energy, attribution, manifest),
        encoding="utf-8",
    )
    (output / "decision_trace.md").write_text(
        _decision_trace(decision), encoding="utf-8"
    )
    appendix = output / "appendix"
    appendix.mkdir(exist_ok=True)
    (appendix / "README.md").write_text(_appendix_text(), encoding="utf-8")


def _summary_text(decision: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    warnings = decision.get("warnings", [])
    actions = decision.get("recommended_next_actions", [])
    lines = [
        "# Engineering decision summary",
        "",
        f"**Recommendation:** {decision.get('recommendation')}",
        "",
        f"**Confidence:** `{decision.get('confidence')}`  ",
        f"**Numerically valid:** `{decision.get('numerically_valid', decision.get('numerical_quality_valid', False))}`  ",
        f"**Evidence ready:** `{decision.get('evidence_ready', False)}`  ",
        f"**Statistically ready:** `{decision.get('statistically_ready', False)}`  ",
        f"**Directionally robust recommendation:** `{decision.get('directionally_robust', False)}`  ",
        f"**Decision ready:** `{decision.get('decision_ready', False)}`  ",
        f"**Study:** `{manifest.get('study_name', manifest.get('study'))}`  ",
        f"**Scenarios:** {manifest.get('scenario_count', 1)}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- No unresolved report warning.")
    lines.extend(["", "## Recommended next actions", ""])
    lines.extend(f"- {action}" for action in actions)
    if not actions:
        lines.append("- Preserve this result with its manifest and evidence bundle.")
    lines.extend(
        [
            "",
            "## Drill down",
            "",
            "- [Full technical report](REPORT.md)",
            "- [Decision trace](decision_trace.md)",
            "- [Machine-readable appendix](appendix/README.md)",
        ]
    )
    return "\n".join(lines) + "\n"


def _report_text(
    decision: Mapping[str, Any],
    summary: Mapping[str, Any],
    convergence: Mapping[str, Any],
    energy: Mapping[str, Any],
    attribution: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    quality = summary.get("numerical_quality", {})
    lines = [
        "# Measured track-based drivetrain study report",
        "",
        "This report moves from the engineering decision to the evidence and numerical detail used to support it.",
        "",
        "## 1. Decision",
        "",
        f"{decision.get('recommendation')}",
        "",
        f"Confidence classification: `{decision.get('confidence')}`.",
        "",
        "See [decision_trace.md](decision_trace.md) for the compact reasoning chain.",
        "",
        "## 2. Study contract",
        "",
        f"- Study type: `{manifest.get('study_type')}`",
        f"- Sampling mode: `{manifest.get('sampling_mode')}`",
        f"- Paired scenarios: `{manifest.get('paired_scenarios', False)}`",
        f"- Scenario count: {manifest.get('scenario_count', 1)}",
        f"- Design points: {manifest.get('design_point_count', 1)}",
        f"- Random seed: {manifest.get('random_seed', 'nominal')}",
        "",
        "The physical and statistical mechanisms are defined in the methods document; this report records the resolved run, not a new model definition.",
        "",
        "## 3. Numerical quality",
        "",
        f"Numerically valid: `{quality.get('numerically_valid', quality.get('valid_for_decision', False))}`.",
        "",
        *_quality_table(quality),
        "",
        "## 4. Evidence readiness",
        "",
        *_evidence_table(manifest),
        "",
        "Project and track warnings do not prevent exploratory execution, but they prevent the result from being labelled decision-ready.",
        "",
        "## 5. Headline results",
        "",
        "Physical scenario p10–p90 bands describe variation across declared scenarios. Bootstrap bounds describe finite-sample uncertainty in estimated summaries; they are not another physical uncertainty source.",
        "",
        *_headline_table(summary),
        "",
        "Full bootstrap bounds, threshold probabilities, paired regrets, and every metric are in [summary.csv](summary.csv) and `summary.json`.",
        "",
        "## 6. Physical energy accounting",
        "",
        "Engine-side and vehicle-side balances are additive partitions. Off-peak and finite-ratio opportunity losses are counterfactual diagnostics and are reported separately.",
        "",
        "See [energy_accounting.csv](energy_accounting.csv), [feature_energy_results.csv](feature_energy_results.csv), and `energy_accounting.json` for design-level bands and scenario detail.",
        "",
        "## 7. Uncertainty attribution",
        "",
        f"Attribution status: `{attribution.get('status')}`.",
        "",
        "Marginal slopes, rank associations, and uncertainty-weighted effects are screening indicators. They are not exact causal variance fractions, particularly when inputs are correlated.",
        "",
        "See [uncertainty_attribution.csv](uncertainty_attribution.csv) and `uncertainty_attribution.json`.",
        "",
        "## 8. Convergence",
        "",
        *_convergence_table(convergence),
        "",
        "Full diagnostics are retained in `convergence.json`.",
        "",
        "## 9. Provenance and reproducibility",
        "",
        "The exact evidence bundle, resolved configuration, seed, framework version, hashes, and command are recorded in `run_manifest.json`, `provenance.json`, and [provenance_graph.svg](provenance_graph.svg).",
        "",
        "## 10. Interpretation limits",
        "",
        "- Telemetry elevation is screened, but grade force remains disabled pending a material paired sensitivity.",
        "- Obstacle mechanics are explicit approximations and require calibration.",
        "- The current tire model is longitudinal and deliberately compact.",
        "- The bounded ideal CVT is an intentionally reduced-order comparison mechanism.",
        "- Gate ceilings reproduce measured entry behavior; they do not optimize a driver policy.",
        "",
        "## 11. Appendix",
        "",
        "The [appendix index](appendix/README.md) maps the report to every machine-readable artifact.",
    ]
    return "\n".join(lines) + "\n"


def _quality_table(quality: Mapping[str, Any]) -> list[str]:
    labels = (
        ("all_cases_completed", "All cases completed"),
        ("all_reference_dominance_checks_pass", "Infinite-reference dominance"),
        ("all_gate_compliance_checks_pass", "Accepted-gate compliance"),
        ("vehicle_energy_balance_pass", "Vehicle energy closure"),
        ("powertrain_energy_balance_pass", "Powertrain energy closure"),
    )
    lines = ["| Check | Result |", "| --- | --- |"]
    lines.extend(f"| {label} | `{quality.get(key)}` |" for key, label in labels)
    lines.extend(
        [
            "",
            f"Maximum vehicle relative residual: `{float(quality.get('maximum_abs_vehicle_energy_balance_relative_error', 0.0)):.3e}`.  ",
            f"Maximum powertrain relative residual: `{float(quality.get('maximum_abs_powertrain_energy_balance_relative_error', 0.0)):.3e}`.",
        ]
    )
    return lines


def _evidence_table(manifest: Mapping[str, Any]) -> list[str]:
    evidence = manifest.get("evidence_assessment", {})
    readiness = manifest.get("decision_readiness", {})
    counts = evidence.get("track_review_status_counts", {}) if isinstance(evidence, Mapping) else {}
    identities = evidence.get("evidence_identity_counts", {}) if isinstance(evidence, Mapping) else {}
    lines = [
        "| Evidence/readiness check | Result |",
        "| --- | --- |",
        f"| Evidence ready | `{bool(evidence.get('ready', False)) if isinstance(evidence, Mapping) else False}` |",
        f"| Project validation warnings | {int(evidence.get('project_warning_count', 0)) if isinstance(evidence, Mapping) else 0} |",
        f"| Accepted track events | {int(counts.get('accepted', 0))} |",
        f"| Recommended-review events | {int(counts.get('recommended_review', 0))} |",
        f"| Must-fix events | {int(counts.get('must_fix', 0))} |",
        f"| Run / vehicle / driver identities | {int(identities.get('run_id', 0))} / {int(identities.get('vehicle_id', 0))} / {int(identities.get('driver_id', 0))} |",
        f"| Statistically ready | `{bool(readiness.get('statistically_ready', False)) if isinstance(readiness, Mapping) else False}` |",
        f"| Decision ready | `{bool(readiness.get('decision_ready', False)) if isinstance(readiness, Mapping) else False}` |",
    ]
    blockers = evidence.get("blocking_reasons", []) if isinstance(evidence, Mapping) else []
    if blockers:
        lines.extend(["", "Blocking evidence findings:"])
        lines.extend(f"- {item}" for item in blockers)
    return lines


def _headline_table(summary: Mapping[str, Any]) -> list[str]:
    designs = summary.get("designs", {})
    if isinstance(designs, Mapping) and designs:
        lines = [
            "| Design | Time penalty p10 / median / p90 [s] | Opportunity loss p10 / median / p90 [kJ] |",
            "| --- | ---: | ---: |",
        ]
        for design_id in sorted(designs):
            record = designs[design_id]
            lines.append(
                f"| `{design_id}` | {_interval(record.get('lap_time_penalty_vs_infinite_s', {}))} | "
                f"{_interval(record.get('finite_ratio_opportunity_loss_energy_kj', {}))} |"
            )
        return lines
    parameters = summary.get("parameters", {})
    if isinstance(parameters, Mapping) and parameters:
        lines = [
            "| Structural parameter | Time-penalty span [s] | Opportunity-loss span [kJ] |",
            "| --- | ---: | ---: |",
        ]
        for path in sorted(parameters):
            record = parameters[path]
            lines.append(
                f"| `{path}` | {float(record.get('time_penalty_span_s', 0.0)):.4g} | "
                f"{float(record.get('opportunity_loss_span_kj', 0.0)):.4g} |"
            )
        return lines
    return ["No headline table was available."]


def _convergence_table(convergence: Mapping[str, Any]) -> list[str]:
    if convergence.get("status") == "not_applicable":
        return [f"Not applicable: {convergence.get('reason')}"]
    lines = [
        "| Design | Metric | Samples | Status | Split-half relative difference |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for design_id in sorted(convergence):
        metrics = convergence[design_id]
        if not isinstance(metrics, Mapping):
            continue
        for metric in sorted(metrics):
            record = metrics[metric]
            if not isinstance(record, Mapping):
                continue
            lines.append(
                f"| `{design_id}` | `{metric}` | {record.get('sample_count')} | "
                f"`{record.get('status')}` | {float(record.get('split_half_relative_difference', 0.0)):.3g} |"
            )
    return lines


def _interval(record: Mapping[str, Any]) -> str:
    return " / ".join(
        f"{float(record.get(key, 0.0)):.4g}" for key in ("p10", "median", "p90")
    )


def _decision_trace(decision: Mapping[str, Any]) -> str:
    lines = [
        "# Decision trace",
        "",
        f"1. **Recommendation** — {decision.get('recommendation')}",
        f"2. **Confidence gate** — `{decision.get('confidence')}`",
        f"3. **Numerically valid** — `{decision.get('numerically_valid', decision.get('numerical_quality_valid', False))}`",
        f"4. **Evidence ready** — `{decision.get('evidence_ready', False)}`",
        f"5. **Statistically ready** — `{decision.get('statistically_ready', False)}`",
        f"6. **Directionally robust** — `{decision.get('directionally_robust', False)}`",
        f"7. **Decision ready** — `{decision.get('decision_ready', False)}`",
    ]
    winners = decision.get("metric_winners", {})
    for metric in sorted(winners):
        winner = winners[metric]
        lines.append(f"8. **{metric}** — best tested design: `{winner}`")
    if decision.get("warnings"):
        lines.append("9. **Constraints on interpretation**")
        lines.extend(f"   - {item}" for item in decision["warnings"])
    if decision.get("recommended_next_actions"):
        lines.append("10. **Next evidence to collect or run**")
        lines.extend(f"   - {item}" for item in decision["recommended_next_actions"])
    return "\n".join(lines) + "\n"


def _appendix_text() -> str:
    return """# Machine-readable appendix

- `../run_manifest.json` — run identity, execution counts, quality, and caching
- `../provenance.json` — hashes, version, resolved command, and evidence lineage
- `../summary.json` / `../summary.csv` — output distributions and paired rankings
- `../replicate_results.csv` — one bounded/reference comparison per design and scenario
- `../scenario_draws.jsonl` — exact paired physical inputs
- `../energy_accounting.json` / `../energy_accounting.csv` — physical partitions
- `../feature_energy_results.csv` — obstacle energy by feature and scenario
- `../uncertainty_attribution.json` / `../uncertainty_attribution.csv` — screening attribution
- `../convergence.json` — finite-sample stability diagnostics
- `../input_contracts.json` — declared input distributions and roles
- `../track_bundle.json` / `../track_bundle.sha256` — exact track evidence bundle
- `../resolved_inputs/` — fully resolved project configuration
"""
