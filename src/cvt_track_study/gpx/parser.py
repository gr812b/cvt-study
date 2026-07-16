"""Secure GPX 1.0/1.1 track parser and canonical point normalization."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from cvt_track_study.config.diagnostics import DiagnosticBag

from .model import CANONICAL_POINT_COLUMNS, GPXIngestionResult, GPXRunMetadata

EARTH_RADIUS_M = 6_371_008.8


class GPXParseError(RuntimeError):
    """Fatal GPX parsing or contract failure."""


def ingest_gpx_run(metadata: GPXRunMetadata) -> GPXIngestionResult:
    diagnostics = DiagnosticBag()
    path = metadata.source_file
    try:
        root = SafeET.parse(path).getroot()
    except (OSError, SafeET.ParseError, DefusedXmlException) as exc:
        raise GPXParseError(f"Unable to parse GPX file {path}: {exc}") from exc

    if _local_name(root.tag) != "gpx":
        raise GPXParseError(f"Root element in {path} is not <gpx>.")

    version = str(root.attrib.get("version", "")).strip()
    if version not in {"1.0", "1.1"}:
        diagnostics.warning(
            "GPX_VERSION_UNTESTED",
            f"GPX version {version!r} is not one of the tested 1.0/1.1 versions.",
            source=str(path),
        )

    tracks = [node for node in root if _local_name(node.tag) == "trk"]
    route_count = sum(_local_name(node.tag) == "rte" for node in root)
    waypoint_count = sum(_local_name(node.tag) == "wpt" for node in root)
    if route_count or waypoint_count:
        diagnostics.info(
            "NON_TRACK_GPX_CONTENT_IGNORED",
            f"Ignored {route_count} route(s) and {waypoint_count} waypoint(s); raw runs are built from GPX tracks only.",
            source=str(path),
            hint="Convert route-only recordings to GPX track segments before ingestion.",
        )
    if not tracks:
        raise GPXParseError(f"GPX file {path} contains no <trk> elements.")

    sha256 = _sha256(path)
    rows: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    invalid_coordinates = 0
    missing_time = 0
    invalid_time = 0
    naive_time = 0
    missing_elevation = 0

    for track_index, track in enumerate(tracks):
        segments = [node for node in track if _local_name(node.tag) == "trkseg"]
        if not segments:
            diagnostics.warning(
                "GPX_TRACK_WITHOUT_SEGMENTS",
                f"Track {track_index} contains no track segments.",
                source=str(path),
            )
        for segment_index, segment in enumerate(segments):
            start_row = len(rows)
            for point_index, point in enumerate(
                node for node in segment if _local_name(node.tag) == "trkpt"
            ):
                lat = _float_or_none(point.attrib.get("lat"))
                lon = _float_or_none(point.attrib.get("lon"))
                if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    invalid_coordinates += 1
                    diagnostics.warning(
                        "INVALID_GPX_COORDINATE",
                        "Track point has a missing or out-of-range latitude/longitude and was excluded.",
                        path=f"track[{track_index}].segment[{segment_index}].point[{point_index}]",
                        source=str(path),
                    )
                    continue
                timestamp_text = _child_text(point, "time")
                timestamp, was_naive = _parse_timestamp(timestamp_text)
                if timestamp_text is None:
                    missing_time += 1
                elif timestamp is None:
                    invalid_time += 1
                elif was_naive:
                    naive_time += 1
                elevation = _float_or_none(_child_text(point, "ele"))
                if elevation is None:
                    missing_elevation += 1
                extension_values = _flatten_extensions(point)
                reported_speed = _first_float(
                    _child_text(point, "speed"),
                    extension_values.get("speed"),
                    extension_values.get("gpxtpx:speed"),
                )
                course = _first_float(
                    _child_text(point, "course"),
                    extension_values.get("course"),
                )
                rows.append(
                    {
                        "run_id": metadata.run_id,
                        "vehicle_id": metadata.vehicle_id,
                        "driver_id": metadata.driver_id,
                        "source_file": str(path),
                        "source_sha256": sha256,
                        "track_index": track_index,
                        "segment_index": segment_index,
                        "point_index": point_index,
                        "timestamp_utc": timestamp,
                        "latitude_deg": lat,
                        "longitude_deg": lon,
                        "elevation_m": elevation,
                        "reported_speed_mps": reported_speed,
                        "derived_speed_mps": math.nan,
                        "analysis_speed_mps": math.nan,
                        "course_deg": course,
                        "fix_type": _child_text(point, "fix"),
                        "satellites": _int_or_none(_child_text(point, "sat")),
                        "hdop": _float_or_none(_child_text(point, "hdop")),
                        "vdop": _float_or_none(_child_text(point, "vdop")),
                        "pdop": _float_or_none(_child_text(point, "pdop")),
                        "step_distance_m": math.nan,
                        "time_step_s": math.nan,
                        "extension_json": json.dumps(extension_values, sort_keys=True),
                    }
                )
            segment_records.append(
                {
                    "run_id": metadata.run_id,
                    "track_index": track_index,
                    "segment_index": segment_index,
                    "start_row": start_row,
                    "point_count": len(rows) - start_row,
                }
            )

    if not rows:
        raise GPXParseError(f"GPX file {path} contains no valid track points.")

    points = pd.DataFrame(rows)
    points = _normalize_numeric_columns(points)
    points = _derive_kinematics(points, diagnostics)
    points = points.reindex(columns=CANONICAL_POINT_COLUMNS)
    segments = _summarize_segments(points, segment_records)

    if missing_time:
        diagnostics.warning(
            "GPX_TIMESTAMPS_MISSING",
            f"{missing_time} valid point(s) have no timestamp; they remain available for geometry but not timing metrics.",
            source=str(path),
        )
    if invalid_time:
        diagnostics.warning(
            "GPX_TIMESTAMPS_INVALID",
            f"{invalid_time} point timestamp(s) could not be parsed; those points remain available for geometry only.",
            source=str(path),
            hint="Use ISO-8601 timestamps with an explicit timezone when timing evidence is required.",
        )
    if naive_time:
        diagnostics.warning(
            "GPX_TIMEZONE_ASSUMED_UTC",
            f"{naive_time} timestamp(s) omitted a timezone and were interpreted as UTC.",
            source=str(path),
        )
    if missing_elevation:
        diagnostics.warning(
            "GPX_ELEVATION_INCOMPLETE",
            f"Elevation is missing for {missing_elevation} of {len(points)} valid point(s).",
            source=str(path),
            hint="Elevation is preserved for review only; no grade force is inferred.",
        )

    summary = {
        "run_id": metadata.run_id,
        "vehicle_id": metadata.vehicle_id,
        "driver_id": metadata.driver_id,
        "source_file": str(path),
        "source_sha256": sha256,
        "gpx_version": version,
        "track_count": len(tracks),
        "segment_count": len(segments),
        "valid_point_count": len(points),
        "invalid_coordinate_count": invalid_coordinates,
        "missing_timestamp_count": missing_time,
        "invalid_timestamp_count": invalid_time,
        "unusable_timestamp_count": int(points["timestamp_utc"].isna().sum()),
        "missing_elevation_count": int(points["elevation_m"].isna().sum()),
        "reported_speed_count": int(points["reported_speed_mps"].notna().sum()),
        "derived_speed_count": int(points["derived_speed_mps"].notna().sum()),
        "route_count_ignored": route_count,
        "waypoint_count_ignored": waypoint_count,
        "total_path_distance_m": float(points["step_distance_m"].fillna(0.0).sum()),
    }
    return GPXIngestionResult(
        metadata=metadata,
        points=points,
        segments=segments,
        summary=summary,
        diagnostics=diagnostics.items,
    )


def _normalize_numeric_columns(points: pd.DataFrame) -> pd.DataFrame:
    """Keep optional GPX numerics numeric even when an entire column is absent."""

    output = points.copy()
    numeric_columns = (
        "latitude_deg",
        "longitude_deg",
        "elevation_m",
        "reported_speed_mps",
        "derived_speed_mps",
        "analysis_speed_mps",
        "course_deg",
        "satellites",
        "hdop",
        "vdop",
        "pdop",
        "step_distance_m",
        "time_step_s",
    )
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _derive_kinematics(points: pd.DataFrame, diagnostics: DiagnosticBag) -> pd.DataFrame:
    output = points.copy()
    for (_, track_index, segment_index), indices in output.groupby(
        ["run_id", "track_index", "segment_index"], sort=False
    ).groups.items():
        idx = list(indices)
        segment = output.loc[idx]
        lat = segment["latitude_deg"].to_numpy(float)
        lon = segment["longitude_deg"].to_numpy(float)
        distance = np.r_[math.nan, _haversine_steps(lat, lon)]
        times = pd.to_datetime(segment["timestamp_utc"], utc=True, errors="coerce")
        dt = times.diff().dt.total_seconds().to_numpy(float)
        derived = np.divide(
            distance,
            dt,
            out=np.full(len(distance), np.nan),
            where=np.isfinite(distance) & np.isfinite(dt) & (dt > 0),
        )
        output.loc[idx, "step_distance_m"] = distance
        output.loc[idx, "time_step_s"] = dt
        output.loc[idx, "derived_speed_mps"] = derived
        reported = output.loc[idx, "reported_speed_mps"].to_numpy(float)
        analysis = np.where(np.isfinite(reported) & (reported >= 0), reported, derived)
        output.loc[idx, "analysis_speed_mps"] = analysis

        duplicate = int(np.sum(np.isfinite(dt) & (dt == 0)))
        regressions = int(np.sum(np.isfinite(dt) & (dt < 0)))
        gaps = int(np.sum(np.isfinite(dt) & (dt > 5.0)))
        if duplicate:
            diagnostics.warning(
                "DUPLICATE_GPX_TIMESTAMPS",
                f"Segment {track_index}:{segment_index} contains {duplicate} duplicate timestamp step(s).",
            )
        if regressions:
            diagnostics.error(
                "GPX_TIMESTAMP_REGRESSION",
                f"Segment {track_index}:{segment_index} contains {regressions} backward timestamp step(s).",
                hint="Fix the source recording or split it into separate GPX segments.",
            )
        if gaps:
            diagnostics.warning(
                "GPX_SAMPLING_GAPS",
                f"Segment {track_index}:{segment_index} contains {gaps} time gap(s) longer than 5 s.",
            )
    return output


def _summarize_segments(
    points: pd.DataFrame, segment_records: Iterable[dict[str, Any]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in segment_records:
        mask = (
            (points["track_index"] == record["track_index"])
            & (points["segment_index"] == record["segment_index"])
        )
        segment = points.loc[mask]
        times = pd.to_datetime(segment["timestamp_utc"], utc=True, errors="coerce")
        dt = segment["time_step_s"].to_numpy(float)
        monotonic = bool(not np.any(np.isfinite(dt) & (dt < 0)))
        duration_s: float | None = None
        if len(times) and times.notna().all() and monotonic:
            duration_s = float((times.iloc[-1] - times.iloc[0]).total_seconds())
        rows.append(
            {
                **record,
                "start_time_utc": times.iloc[0] if len(times) else pd.NaT,
                "end_time_utc": times.iloc[-1] if len(times) else pd.NaT,
                "timestamps_monotonic": monotonic,
                "duration_s": duration_s,
                "path_distance_m": float(segment["step_distance_m"].fillna(0.0).sum()),
                "missing_timestamp_count": int(times.isna().sum()),
                "missing_elevation_count": int(segment["elevation_m"].isna().sum()),
                "median_sample_period_s": float(
                    np.nanmedian(dt[np.isfinite(dt) & (dt > 0)])
                )
                if np.any(np.isfinite(dt) & (dt > 0))
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _parse_timestamp(text: str | None) -> tuple[datetime | None, bool]:
    if not text:
        return None, False
    value = text.strip()
    if value.endswith(("Z", "z")):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None, False
    naive = parsed.tzinfo is None
    if naive:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc), naive


def _haversine_steps(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat1 = np.radians(lat[:-1])
    lat2 = np.radians(lat[1:])
    dlat = lat2 - lat1
    dlon = np.radians(lon[1:] - lon[:-1])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _flatten_extensions(point: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    extensions = next(
        (child for child in point if _local_name(child.tag) == "extensions"), None
    )
    if extensions is None:
        return result
    for node in extensions.iter():
        if node is extensions or len(node):
            continue
        text = (node.text or "").strip()
        if not text:
            continue
        local = _local_name(node.tag)
        prefix = _namespace_hint(node.tag)
        key = f"{prefix}:{local}" if prefix else local
        if key in result:
            suffix = 2
            while f"{key}_{suffix}" in result:
                suffix += 1
            key = f"{key}_{suffix}"
        result[key] = text
    return result


def _namespace_hint(tag: str) -> str:
    if not tag.startswith("{"):
        return ""
    uri = tag[1:].split("}", 1)[0].lower()
    if "trackpointextension" in uri:
        return "gpxtpx"
    return "ext"


def _child_text(node: Any, local_name: str) -> str | None:
    for child in node:
        if _local_name(child.tag) == local_name:
            text = (child.text or "").strip()
            return text or None
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    return int(number) if number is not None else None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _float_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
