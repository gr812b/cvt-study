"""Six-report framework facade."""

from pathlib import Path

from .catalog import REPORTS, ReportDefinition, canonical_report_key
from .postprocess import (
    primary_report_path,
    regenerate_framework_report as _regenerate_framework_report,
    write_design_comparison_report,
    write_full_uncertainty_report as _write_full_uncertainty_report,
    write_nominal_simulation_report,
    write_structural_report_manifest,
    write_track_evidence_report,
)
from .uncertainty_mechanism import enhance_full_uncertainty_report


def write_full_uncertainty_report(output: Path) -> Path:
    target = _write_full_uncertainty_report(output)
    return enhance_full_uncertainty_report(Path(output), target)


def regenerate_framework_report(output: Path) -> Path:
    target = _regenerate_framework_report(output)
    manifest = Path(output) / "run_manifest.json"
    if manifest.is_file():
        import json

        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if str(raw.get("study_type", "")) == "full_uncertainty":
            target = enhance_full_uncertainty_report(Path(output), target)
    return target


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
