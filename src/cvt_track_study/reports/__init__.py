"""Six-report framework facade."""

from .catalog import REPORTS, ReportDefinition, canonical_report_key
from .postprocess import (
    primary_report_path,
    regenerate_framework_report,
    write_design_comparison_report,
    write_full_uncertainty_report,
    write_nominal_simulation_report,
    write_structural_report_manifest,
    write_track_evidence_report,
)

__all__ = [
    "REPORTS",
    "ReportDefinition",
    "canonical_report_key",
    "primary_report_path",
    "regenerate_framework_report",
    "write_design_comparison_report",
    "write_full_uncertainty_report",
    "write_nominal_simulation_report",
    "write_structural_report_manifest",
    "write_track_evidence_report",
]
