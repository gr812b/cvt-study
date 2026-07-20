"""Canonical six-report contract for the measured-track framework."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportDefinition:
    key: str
    title: str
    question: str
    fixed: str
    varied: str
    html_filename: str


REPORTS: dict[str, ReportDefinition] = {
    "track_evidence": ReportDefinition(
        key="track_evidence",
        title="Track evidence and reconstruction",
        question="What track can be inferred from the supplied telemetry and reviewed events?",
        fixed="Raw source files and declared event evidence.",
        varied="Nothing; this is the selected nominal reconstruction.",
        html_filename="track_evidence_report.html",
    ),
    "nominal_simulation": ReportDefinition(
        key="nominal_simulation",
        title="Nominal vehicle simulation",
        question="What does the selected vehicle and drivetrain do on the nominal reconstructed track?",
        fixed="Nominal track, vehicle, drivetrain, driver, and physics contracts.",
        varied="Nothing; bounded and infinite-CVT cases share one nominal realization.",
        html_filename="nominal_simulation_report.html",
    ),
    "track_robustness": ReportDefinition(
        key="track_robustness",
        title="Track defensibility and robustness",
        question="Is the inferred track stable under reasonable alternative analysis choices supported by the same telemetry?",
        fixed="Raw telemetry, reviewed event declarations, and the meaning of the evidence.",
        varied="Run support, cleanup, centreline, projection-window, gate-threshold, and confidence-weighting choices.",
        html_filename="track_robustness_report.html",
    ),
    "structural_sensitivity": ReportDefinition(
        key="structural_sensitivity",
        title="Structural sensitivity",
        question="Which physical or modelling assumptions materially move the nominal simulation result?",
        fixed="Nominal reconstructed track and all non-varied nominal inputs.",
        varied="One declared structural input at a time across its defensible levels.",
        html_filename="structural_sensitivity_report.html",
    ),
    "full_uncertainty": ReportDefinition(
        key="full_uncertainty",
        title="Full uncertainty and answer robustness",
        question="When defensible uncertainties vary together, what range of simulation answers should be believed?",
        fixed="The design definition being evaluated.",
        varied="Track realization, measured traversal, and structural uncertainty jointly and coherently.",
        html_filename="full_uncertainty_report.html",
    ),
    "design_comparison": ReportDefinition(
        key="design_comparison",
        title="Design comparison",
        question="Which drivetrain design performs best, by how much, and does the ranking survive uncertainty?",
        fixed="Common scenario draws and evidence contracts for every candidate.",
        varied="The declared design variable or candidate set.",
        html_filename="design_comparison_report.html",
    ),
}

ALIASES = {
    "track-evidence": "track_evidence",
    "build-track": "track_evidence",
    "nominal": "nominal_simulation",
    "baseline": "nominal_simulation",
    "track-robustness": "track_robustness",
    "structural-sensitivity": "structural_sensitivity",
    "full-uncertainty": "full_uncertainty",
    "uncertainty": "full_uncertainty",
    "design-comparison": "design_comparison",
    "sweep": "design_comparison",
}


def canonical_report_key(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    key = ALIASES.get(normalized, normalized.replace("-", "_"))
    if key not in REPORTS:
        raise ValueError(
            f"Unknown report {value!r}; expected one of {', '.join(sorted(REPORTS))}."
        )
    return key
