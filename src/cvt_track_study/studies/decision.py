"""Decision synthesis with explicit quality and evidence gates."""

from __future__ import annotations

from typing import Any, Mapping

from cvt_track_study.runtime.evidence import numerical_valid


TIME_METRIC = "lap_time_penalty_vs_infinite_s"
ENERGY_METRIC = "finite_ratio_opportunity_loss_energy_kj"


def synthesize_decision(
    *,
    summary: Mapping[str, Any],
    convergence: Mapping[str, Any],
    attribution: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    study_type = str(manifest.get("study_type", "unknown"))
    quality = numerical_valid(summary.get("numerical_quality", {}))
    evidence = manifest.get("evidence_assessment", {})
    evidence_ready = bool(
        isinstance(evidence, Mapping) and evidence.get("ready", False)
    )
    warnings: list[str] = []
    next_actions: list[str] = []
    recommendation = "No design recommendation is produced by this study type."
    confidence = "exploratory"
    metric_winners: dict[str, str] = {}
    statistically_ready = False
    boundary_winner = False
    common_winner: str | None = None

    if not quality:
        warnings.append("At least one numerical quality check failed; results are not valid for a design decision.")
        next_actions.append("Correct the numerical-quality failure before interpreting the ranking.")
    if not evidence_ready:
        warnings.append("Project inputs or track evidence have unresolved review warnings.")
        next_actions.append("Resolve the evidence warnings recorded in the run manifest and resolved-input report.")
    if isinstance(evidence, Mapping):
        warnings.extend(str(item) for item in evidence.get("warnings", []))

    if study_type == "design_sweep":
        designs = summary.get("designs", {})
        if designs:
            for metric in (TIME_METRIC, ENERGY_METRIC):
                metric_winners[metric] = min(
                    designs,
                    key=lambda design_id: float(designs[design_id][metric]["median"]),
                )
            same_winner = len(set(metric_winners.values())) == 1
            winner = metric_winners[TIME_METRIC]
            if same_winner:
                common_winner = winner
                recommendation = f"Current tested winner: {winner}; no design recommendation yet."
            else:
                recommendation = "Time and opportunity-loss metrics prefer different tested designs."
                warnings.append("Headline metrics do not identify the same winner.")
                next_actions.append("Review the time-versus-energy tradeoff before selecting a design.")

            win_bounds_ok = True
            for metric in (TIME_METRIC, ENERGY_METRIC):
                record = designs[winner].get(f"paired_ranking.{metric}", {})
                low = record.get("paired_win_fraction_bootstrap_95_low")
                if low is None or float(low) <= 0.5:
                    win_bounds_ok = False
            convergence_ok = _convergence_ok(convergence)
            values = sorted(
                (
                    float(record["design_value_si"]),
                    design_id,
                )
                for design_id, record in designs.items()
                if record.get("design_value_si") is not None
            )
            if values and winner in {values[0][1], values[-1][1]}:
                boundary_winner = True
                warnings.append("The apparent optimum lies on a tested sweep boundary.")
                next_actions.append("Extend the sweep beyond the winning boundary.")
            statistically_ready = (
                same_winner and win_bounds_ok and convergence_ok and not boundary_winner
            )
            robust = quality and statistically_ready
            if robust and evidence_ready:
                confidence = "directionally_robust"
            elif robust:
                confidence = "evidence_limited"
            else:
                confidence = "provisional"
            if not convergence_ok:
                warnings.append("Monte Carlo convergence checks recommend more scenarios.")
                next_actions.append("Increase paired scenario count and rerun the same declared study.")
            if not win_bounds_ok:
                warnings.append("The paired win-fraction interval does not remain above 0.5.")
    else:
        scenario_count = int(manifest.get("scenario_count", 0))
        if study_type != "structural_sensitivity" and scenario_count < 20:
            warnings.append("The scenario count is below the screening recommendation of twenty.")
            next_actions.append("Use this run as a mechanism check, then repeat with at least twenty paired scenarios.")
        if quality and study_type == "structural_sensitivity":
            confidence = "one_at_a_time_screening"
            strongest = _strongest_structural(attribution)
            if strongest:
                recommendation = (
                    "Strongest tested one-at-a-time driver: "
                    f"{strongest['path']} for {strongest['metric']}."
                )
        elif quality:
            confidence = "exploratory_distribution" if scenario_count < 20 else "screening_distribution"

    warnings.extend(str(item) for item in attribution.get("warnings", []))
    directionally_robust = quality and statistically_ready
    decision_ready = (
        study_type == "design_sweep"
        and quality
        and evidence_ready
        and statistically_ready
    )
    if decision_ready and common_winner is not None:
        recommendation = f"Decision-ready best tested design: {common_winner}."
    return {
        "study_name": manifest.get("study_name", manifest.get("study")),
        "study_type": study_type,
        "recommendation": recommendation,
        "confidence": confidence,
        "numerically_valid": quality,
        "numerical_quality_valid": quality,
        "evidence_ready": evidence_ready,
        "statistically_ready": statistically_ready,
        "directionally_robust": directionally_robust,
        "decision_ready": decision_ready,
        "metric_winners": metric_winners,
        "warnings": list(dict.fromkeys(warnings)),
        "recommended_next_actions": list(dict.fromkeys(next_actions)),
    }


def _convergence_ok(convergence: Mapping[str, Any]) -> bool:
    records = []
    for design in convergence.values():
        if not isinstance(design, Mapping):
            continue
        records.extend(record for record in design.values() if isinstance(record, Mapping))
    return bool(records) and all(record.get("status") == "adequate_quick_check" for record in records)


def _strongest_structural(attribution: Mapping[str, Any]) -> Mapping[str, Any] | None:
    rows = attribution.get("parameters", [])
    if not rows:
        return None
    return max(rows, key=lambda item: abs(float(item.get("response_span", 0.0))))
