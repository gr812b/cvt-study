"""Project result discovery and lightweight index generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def discover_results(results_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not results_root.exists():
        return records
    for manifest_path in sorted(results_root.rglob("run_manifest.json")):
        if any(part.startswith(".") and part.endswith(".incomplete") for part in manifest_path.parts):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append(
            {
                "path": str(manifest_path.parent.relative_to(results_root)),
                "study": manifest.get("study_name", manifest.get("study", "unknown")),
                "type": manifest.get("study_type", "baseline"),
                "created_utc": manifest.get("created_utc", "unknown"),
                "numerically_valid": manifest.get("numerical_quality", {}).get(
                    "numerically_valid",
                    manifest.get("numerical_quality", {}).get("valid_for_decision"),
                ),
                "evidence_ready": manifest.get("evidence_assessment", {}).get("ready"),
                "decision_ready": manifest.get("decision_readiness", {}).get("decision_ready"),
            }
        )
    return sorted(records, key=lambda row: (str(row["created_utc"]), str(row["path"])), reverse=True)


def write_results_index(results_root: Path) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    records = discover_results(results_root)
    lines = [
        "# Result index",
        "",
        "Completed framework results discovered below this directory.",
        "",
        "| Created (UTC) | Study | Type | Numerical | Evidence | Decision ready | Result |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in records:
        numerical = _status(row["numerically_valid"], "valid", "failed")
        evidence = _status(row["evidence_ready"], "ready", "review required")
        decision = _status(row["decision_ready"], "yes", "no")
        relative = str(row["path"]).replace("\\", "/")
        lines.append(
            f"| {row['created_utc']} | {row['study']} | {row['type']} | "
            f"{numerical} | {evidence} | {decision} | [{relative}]({relative}/SUMMARY.md) |"
        )
    if not records:
        lines.append("| — | — | — | — | — | — | No completed results |")
    path = results_root / "INDEX.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _status(value: Any, true_text: str, false_text: str) -> str:
    if value is None:
        return "n/a"
    return true_text if bool(value) else false_text
