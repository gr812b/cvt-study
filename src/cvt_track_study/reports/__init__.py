"""Six-report framework facade."""

from pathlib import Path

# Install this before importing any report plotting module.  Python always
# executes the package initializer before a ``cvt_track_study.reports.*``
# submodule, so the policy also covers direct submodule imports.
from .contrast_errorbars import install_contrast_errorbars

install_contrast_errorbars()

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
from .traffic import augment_full_uncertainty_traffic_report


def write_full_uncertainty_report(output: Path) -> Path:
    target = _write_full_uncertainty_report(output)
    target = enhance_full_uncertainty_report(Path(output), target)
    return augment_full_uncertainty_traffic_report(Path(output)) or target


def regenerate_framework_report(output: Path) -> Path:
    target = _regenerate_framework_report(output)
    manifest = Path(output) / "run_manifest.json"
    if manifest.is_file():
        import json

        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if str(raw.get("study_type", "")) == "full_uncertainty":
            target = enhance_full_uncertainty_report(Path(output), target)
            target = augment_full_uncertainty_traffic_report(Path(output)) or target
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
