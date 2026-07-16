"""Garmin FIT activity parser normalized to the canonical telemetry schema."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from garmin_fit_sdk import Decoder, Profile, Stream

from cvt_track_study.config.diagnostics import DiagnosticBag

from .model import CANONICAL_POINT_COLUMNS, GPXIngestionResult, GPXRunMetadata
from .parser import (
    TelemetryParseError,
    _derive_kinematics,
    _float_or_none,
    _normalize_numeric_columns,
    _sha256,
    _summarize_segments,
)


class FITParseError(TelemetryParseError):
    """Fatal FIT parsing or contract failure."""


def ingest_fit_run(metadata: GPXRunMetadata) -> GPXIngestionResult:
    """Decode FIT record messages without discarding native device channels."""

    diagnostics = DiagnosticBag()
    path = metadata.source_file
    records: list[dict[str, Any]] = []

    def collect_record(message_number: int, message: dict[str, Any]) -> None:
        if message_number == Profile["mesg_num"]["RECORD"]:
            records.append(dict(message))

    try:
        stream = Stream.from_file(str(path))
        _, errors = Decoder(stream).read(mesg_listener=collect_record)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise FITParseError(f"Unable to parse FIT file {path}: {exc}") from exc

    if errors:
        diagnostics.warning(
            "FIT_DECODE_WARNINGS",
            f"The FIT decoder reported {len(errors)} warning(s); retained valid record messages.",
            source=str(path),
            hint="Inspect the ingestion diagnostics and source-device export if fields are missing.",
        )
    if not records:
        raise FITParseError(f"FIT file {path} contains no record messages.")

    sha256 = _sha256(path)
    rows: list[dict[str, Any]] = []
    invalid_coordinates = 0
    missing_time = 0
    missing_elevation = 0
    for point_index, record in enumerate(records):
        latitude = _semicircles_to_degrees(record.get("position_lat"), latitude=True)
        longitude = _semicircles_to_degrees(record.get("position_long"), latitude=False)
        if latitude is None or longitude is None:
            invalid_coordinates += 1
            diagnostics.warning(
                "INVALID_FIT_COORDINATE",
                "FIT record has a missing or out-of-range position and was excluded.",
                path=f"record[{point_index}]",
                source=str(path),
            )
            continue

        timestamp = _fit_timestamp(record.get("timestamp"))
        if timestamp is None:
            missing_time += 1
        elevation, elevation_source = _native_altitude(record)
        if elevation is None:
            missing_elevation += 1
        speed = _first_valid_nonnegative(
            record.get("enhanced_speed"), record.get("speed")
        )
        distance = _first_valid_nonnegative(record.get("distance"))
        course = _first_valid_nonnegative(
            record.get("enhanced_heading"), record.get("heading")
        )
        rows.append(
            {
                "run_id": metadata.run_id,
                "vehicle_id": metadata.vehicle_id,
                "driver_id": metadata.driver_id,
                "source_file": str(path),
                "source_sha256": sha256,
                "source_format": "fit",
                "track_index": 0,
                "segment_index": 0,
                "point_index": point_index,
                "timestamp_utc": timestamp,
                "latitude_deg": latitude,
                "longitude_deg": longitude,
                "elevation_m": elevation,
                "elevation_source": elevation_source,
                "device_distance_m": distance,
                "device_speed_mps": speed,
                "reported_speed_mps": math.nan,
                "derived_speed_mps": math.nan,
                "analysis_speed_mps": math.nan,
                "analysis_speed_source": "unavailable",
                "speed_certainty": "unavailable",
                "course_deg": course,
                "fix_type": record.get("gps_fix"),
                "satellites": record.get("num_satellites"),
                "horizontal_accuracy_m": record.get("gps_accuracy"),
                "hdop": math.nan,
                "vdop": math.nan,
                "pdop": math.nan,
                "step_distance_m": math.nan,
                "time_step_s": math.nan,
                "extension_json": json.dumps(
                    _json_safe_record(record), sort_keys=True, default=str
                ),
            }
        )

    if not rows:
        raise FITParseError(f"FIT file {path} contains no valid positioned record messages.")

    points = _normalize_numeric_columns(pd.DataFrame(rows))
    points = _derive_kinematics(points, diagnostics)
    points = points.reindex(columns=CANONICAL_POINT_COLUMNS)
    segment_records = [
        {
            "run_id": metadata.run_id,
            "track_index": 0,
            "segment_index": 0,
            "start_row": 0,
            "point_count": len(points),
        }
    ]
    segments = _summarize_segments(points, segment_records)

    if missing_time:
        diagnostics.warning(
            "FIT_TIMESTAMPS_MISSING",
            f"{missing_time} positioned record(s) have no usable timestamp.",
            source=str(path),
        )
    if missing_elevation:
        diagnostics.warning(
            "FIT_ELEVATION_INCOMPLETE",
            f"Elevation is missing for {missing_elevation} of {len(points)} valid point(s).",
            source=str(path),
            hint="Elevation is retained as evidence; grade force remains conditional on the grade screen.",
        )

    summary = {
        "run_id": metadata.run_id,
        "vehicle_id": metadata.vehicle_id,
        "driver_id": metadata.driver_id,
        "source_file": str(path),
        "source_sha256": sha256,
        "source_format": "fit",
        "track_count": 1,
        "segment_count": 1,
        "record_message_count": len(records),
        "valid_point_count": len(points),
        "invalid_coordinate_count": invalid_coordinates,
        "missing_timestamp_count": missing_time,
        "invalid_timestamp_count": 0,
        "unusable_timestamp_count": int(points["timestamp_utc"].isna().sum()),
        "missing_elevation_count": int(points["elevation_m"].isna().sum()),
        "reported_speed_count": 0,
        "device_speed_count": int(points["device_speed_mps"].notna().sum()),
        "derived_speed_count": int(points["derived_speed_mps"].notna().sum()),
        "device_distance_count": int(points["device_distance_m"].notna().sum()),
        "decoder_warning_count": len(errors),
        "total_path_distance_m": float(points["step_distance_m"].fillna(0.0).sum()),
    }
    return GPXIngestionResult(
        metadata=metadata,
        points=points,
        segments=segments,
        summary=summary,
        diagnostics=diagnostics.items,
    )


def _semicircles_to_degrees(value: Any, *, latitude: bool) -> float | None:
    number = _float_or_none(value)
    if number is None:
        return None
    limit = 90.0 if latitude else 180.0
    degrees = number if abs(number) <= limit else number * 180.0 / (2**31)
    return degrees if -limit <= degrees <= limit else None


def _fit_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return None


def _native_altitude(record: dict[str, Any]) -> tuple[float | None, str]:
    enhanced = _float_or_none(record.get("enhanced_altitude"))
    if enhanced is not None:
        return enhanced, "fit_enhanced_altitude"
    altitude = _float_or_none(record.get("altitude"))
    if altitude is not None:
        return altitude, "fit_altitude"
    return None, "unavailable"


def _first_valid_nonnegative(*values: Any) -> float | None:
    for value in values:
        number = _float_or_none(value)
        if number is not None and number >= 0:
            return number
    return None


def _json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in record.items()
        if key not in {"developer_fields"}
    }
