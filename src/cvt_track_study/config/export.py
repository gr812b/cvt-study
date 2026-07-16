"""Resolved-configuration and validation artifact export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .diagnostics import Diagnostic
from .merge import ProvenanceMap
from .toml_io import dump_toml


def export_resolution(
    output_directory: Path,
    *,
    data: Mapping[str, Any],
    provenance: ProvenanceMap,
    diagnostics: tuple[Diagnostic, ...],
    metadata: Mapping[str, Any],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    dump_toml(data, output_directory / "resolved_inputs.toml")
    _write_json(
        output_directory / "provenance.json",
        {
            path: [step.to_dict() for step in steps]
            for path, steps in sorted(provenance.items())
        },
    )
    _write_json(
        output_directory / "validation_report.json",
        {
            "summary": {
                "errors": sum(item.severity.value == "error" for item in diagnostics),
                "warnings": sum(item.severity.value == "warning" for item in diagnostics),
                "information": sum(item.severity.value == "info" for item in diagnostics),
            },
            "diagnostics": [item.to_dict() for item in diagnostics],
        },
    )
    _write_json(output_directory / "resolution_manifest.json", dict(metadata))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
