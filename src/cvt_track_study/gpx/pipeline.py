"""Shared configured-run ingestion pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .cleanup import apply_telemetry_cleanup
from .ingestion import ingest_telemetry_run
from .lap_time_reconstruction import apply_optional_lap_time_reconstruction
from .model import GPXIngestionResult
from .run_config import metadata_from_run_config


def ingest_configured_run(
    raw: Mapping[str, Any],
    *,
    runs_directory: Path,
    track_config: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> GPXIngestionResult:
    metadata = metadata_from_run_config(
        raw,
        runs_directory=runs_directory,
    )
    parsed = ingest_telemetry_run(metadata)
    cleaned = apply_telemetry_cleanup(parsed, track_config)
    return apply_optional_lap_time_reconstruction(
        cleaned,
        track_config=track_config,
        events=events,
    )
