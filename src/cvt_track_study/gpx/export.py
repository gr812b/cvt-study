"""Atomic GPX ingestion artifact export."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd


from .model import GPXIngestionResult


def export_ingestion_results(
    output_directory: Path,
    results: Iterable[GPXIngestionResult],
    *,
    manifest: dict[str, Any],
    export_configuration: Callable[[Path], None] | None = None,
) -> Path:
    """Write one complete ingestion result tree and atomically publish it."""

    final = output_directory.resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}-", dir=final.parent))
    try:
        run_summaries: list[dict[str, Any]] = []
        all_points: list[pd.DataFrame] = []
        all_segments: list[pd.DataFrame] = []
        all_diagnostics: list[dict[str, Any]] = []
        for result in results:
            run_dir = temporary / "runs" / result.metadata.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_dataframe(result.points, run_dir / "canonical_points.csv")
            _write_dataframe(result.segments, run_dir / "segments.csv")
            _write_json(run_dir / "summary.json", result.summary)
            _write_json(
                run_dir / "diagnostics.json",
                [item.to_dict() for item in result.diagnostics],
            )
            run_summaries.append(result.summary)
            all_points.append(result.points)
            all_segments.append(result.segments)
            all_diagnostics.extend(
                {"run_id": result.metadata.run_id, **item.to_dict()}
                for item in result.diagnostics
            )

        _write_dataframe(pd.DataFrame(run_summaries), temporary / "run_summaries.csv")
        _write_dataframe(pd.concat(all_points, ignore_index=True), temporary / "canonical_points.csv")
        _write_dataframe(pd.concat(all_segments, ignore_index=True), temporary / "segments.csv")
        _write_json(temporary / "diagnostics.json", all_diagnostics)
        _write_json(temporary / "ingestion_manifest.json", manifest)
        if export_configuration is not None:
            export_configuration(temporary / "configuration")
        _replace_directory(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def _write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _replace_directory(temporary: Path, final: Path) -> None:
    backup = final.with_name(f".{final.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if final.exists():
        os.replace(final, backup)
    try:
        os.replace(temporary, final)
    except Exception:
        if backup.exists() and not final.exists():
            os.replace(backup, final)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)
