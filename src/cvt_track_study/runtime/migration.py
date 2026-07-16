"""Conservative migration helpers for prototype event tables."""

from __future__ import annotations

import csv
from pathlib import Path


def migrate_prototype_events(source: Path, destination: Path) -> int:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Prototype event table is empty.")
    lines = [
        "# Migrated geometry anchors only. Review every row before track build.",
        "# Obstacle physics is deliberately not inferred by this migration.",
        "",
    ]
    count = 0
    for index, row in enumerate(rows, start=1):
        name = _first(row, "name", "event_name", "obstacle", "label") or f"event_{index}"
        anchor = _first(row, "anchor_s_m", "anchor_s", "entry_s_m", "entry_s")
        if anchor in (None, ""):
            continue
        try:
            anchor_value = float(anchor)
        except ValueError as exc:
            raise ValueError(f"Row {index} has non-numeric anchor {anchor!r}.") from exc
        identifier = _slug(_first(row, "id", "event_id") or name)
        lines.extend(
            [
                "[[feature]]",
                f'id = "{identifier}"',
                f'name = "{name.replace(chr(34), chr(39))}"',
                f"anchor_s_m = {anchor_value:.6g}",
                'review_status = "needs_engineering_review"',
                'obstacle_model = "unset"',
                "",
            ]
        )
        count += 1
    if not count:
        raise ValueError("No row contained an anchor/entry distance that could be migrated.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return count


def _first(row: dict[str, str], *names: str) -> str | None:
    normalized = {key.strip().lower(): value.strip() for key, value in row.items() if key}
    for name in names:
        value = normalized.get(name)
        if value:
            return value
    return None


def _slug(value: str) -> str:
    text = "".join(char.lower() if char.isalnum() else "_" for char in value)
    text = "_".join(part for part in text.split("_") if part)
    return text or "event"
