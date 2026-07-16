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
                "valid_for_decision": manifest.get("numerical_quality", {}).get(
                    "valid_for_decision"
                ),
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
        "| Created (UTC) | Study | Type | Decision quality | Result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in records:
        quality = row["valid_for_decision"]
        quality_text = "n/a" if quality is None else ("valid" if quality else "review required")
        relative = str(row["path"]).replace("\\", "/")
        lines.append(
            f"| {row['created_utc']} | {row['study']} | {row['type']} | "
            f"{quality_text} | [{relative}]({relative}/SUMMARY.md) |"
        )
    if not records:
        lines.append("| — | — | — | — | No completed results |")
    path = results_root / "INDEX.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
