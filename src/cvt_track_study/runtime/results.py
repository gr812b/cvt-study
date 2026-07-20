"""Discovery and indexing for the six canonical framework reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def discover_results(results_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not results_root.exists():
        return records

    report_directories: set[Path] = set()
    for report_path in sorted(results_root.rglob("report_manifest.json")):
        if _is_incomplete(report_path):
            continue
        manifest = _read_json(report_path)
        if not manifest:
            continue
        directory = report_path.parent
        html_file = str(manifest.get("html_file", ""))
        html_path = directory / html_file if html_file else None
        run = _read_json(directory / "run_manifest.json")
        track = _read_json(directory / "track_robustness_manifest.json")
        source = run or track
        records.append(
            {
                "path": str(directory.relative_to(results_root)),
                "study": source.get("study_name", manifest.get("report_key", "unknown")),
                "type": manifest.get("report_key", source.get("study_type", "unknown")),
                "title": manifest.get("title", manifest.get("report_key", "unknown")),
                "created_utc": manifest.get(
                    "generated_utc", source.get("created_utc", "unknown")
                ),
                "numerically_valid": source.get("numerical_quality", {}).get(
                    "numerically_valid",
                    source.get("numerical_quality", {}).get("valid_for_decision"),
                ),
                "evidence_ready": source.get("evidence_assessment", {}).get("ready"),
                "decision_ready": source.get("decision_readiness", {}).get(
                    "decision_ready"
                ),
                "primary_report": (
                    str(html_path.relative_to(results_root))
                    if html_path is not None and html_path.is_file()
                    else None
                ),
            }
        )
        report_directories.add(directory.resolve())

    # Compatibility for old completed results that predate report_manifest.json.
    for manifest_path in sorted(results_root.rglob("run_manifest.json")):
        if _is_incomplete(manifest_path) or manifest_path.parent.resolve() in report_directories:
            continue
        manifest = _read_json(manifest_path)
        if not manifest:
            continue
        directory = manifest_path.parent
        records.append(
            {
                "path": str(directory.relative_to(results_root)),
                "study": manifest.get("study_name", manifest.get("study", "unknown")),
                "type": manifest.get("study_type", "baseline"),
                "title": manifest.get("study_name", manifest.get("study_type", "result")),
                "created_utc": manifest.get("created_utc", "unknown"),
                "numerically_valid": manifest.get("numerical_quality", {}).get(
                    "numerically_valid",
                    manifest.get("numerical_quality", {}).get("valid_for_decision"),
                ),
                "evidence_ready": manifest.get("evidence_assessment", {}).get("ready"),
                "decision_ready": manifest.get("decision_readiness", {}).get(
                    "decision_ready"
                ),
                "primary_report": _legacy_report_path(directory, results_root),
            }
        )
    return sorted(
        records,
        key=lambda row: (str(row["created_utc"]), str(row["path"])),
        reverse=True,
    )


def write_results_index(results_root: Path) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    records = discover_results(results_root)
    lines = [
        "# Framework report index",
        "",
        "The six report types answer different questions; numerical and evidence status are not interchangeable.",
        "",
        "| Created (UTC) | Report | Study | Numerical | Evidence | Decision ready | Open |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in records:
        numerical = _status(row["numerically_valid"], "valid", "failed")
        evidence = _status(row["evidence_ready"], "ready", "review required")
        decision = _status(row["decision_ready"], "yes", "no")
        target = row.get("primary_report")
        link = f"[{row['path']}]({str(target).replace(chr(92), '/')})" if target else str(row["path"])
        lines.append(
            f"| {row['created_utc']} | {row.get('title', row['type'])} | {row['study']} | "
            f"{numerical} | {evidence} | {decision} | {link} |"
        )
    if not records:
        lines.append("| — | — | — | — | — | — | No completed reports |")
    path = results_root / "INDEX.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_incomplete(path: Path) -> bool:
    return any(part.startswith(".") and part.endswith(".incomplete") for part in path.parts)


def _legacy_report_path(directory: Path, results_root: Path) -> str | None:
    candidates = (
        directory / "full_uncertainty_report.html",
        directory / "design_comparison_report.html",
        directory / "structural_sensitivity_report.html",
        directory / "nominal_simulation_report.html",
        directory / "REPORT.md",
        directory / "SUMMARY.md",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    return str(path.relative_to(results_root)) if path is not None else None


def _status(value: Any, true_text: str, false_text: str) -> str:
    if value is None:
        return "n/a"
    return true_text if bool(value) else false_text
