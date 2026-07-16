"""Decision synthesis with explicit quality and evidence gates."""

from __future__ import annotations

from typing import Any, Mapping


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
    quality = bool(summary.get("numerical_quality", {}).get("valid_for_decision", False))
    warnings: list[str] = []
    next_actions: list[str] = []
    recommendation = "No design recommendation is produced by this study type."
    confidence = "exploratory"
    metric_winners: dict[str, str] = {}

    if not quality:
        warnings.append("At least one numerical quality check failed; results are not valid for a design decision.")
        next_actions.append("Correct the numerical-quality failure before interpreting the ranking.")

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
                recommendation = f"Best tested design: {winner}."
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
                warnings.append("The apparent optimum lies on a tested sweep boundary.")
                next_actions.append("Extend the sweep beyond the winning boundary.")
            robust = quality and same_winner and win_bounds_ok and convergence_ok
            confidence = "directionally_robust" if robust else "provisional"
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
    return {
        "study_name": manifest.get("study_name", manifest.get("study")),
        "study_type": study_type,
        "recommendation": recommendation,
        "confidence": confidence,
        "numerical_quality_valid": quality,
        "directionally_robust": quality and confidence == "directionally_robust",
        "valid_for_decision": quality,
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
