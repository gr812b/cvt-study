"""Format dispatch for declared raw telemetry runs."""

from __future__ import annotations

from .csv_parser import CSVParseError, ingest_csv_run
from .fit_parser import FITParseError, ingest_fit_run
from .model import GPXIngestionResult, GPXRunMetadata
from .parser import GPXParseError, TelemetryParseError, ingest_gpx_run


def ingest_telemetry_run(metadata: GPXRunMetadata) -> GPXIngestionResult:
    suffix = metadata.source_file.suffix.lower()
    if suffix == ".gpx":
        return ingest_gpx_run(metadata)
    if suffix == ".fit":
        return ingest_fit_run(metadata)
    if suffix == ".csv":
        return ingest_csv_run(metadata)
    raise TelemetryParseError(
        f"Unsupported telemetry format {suffix or '<none>'!r}; "
        "expected .gpx, .fit, or .csv."
    )


__all__ = [
    "CSVParseError",
    "FITParseError",
    "GPXParseError",
    "TelemetryParseError",
    "ingest_csv_run",
    "ingest_fit_run",
    "ingest_gpx_run",
    "ingest_telemetry_run",
]
