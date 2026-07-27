"""Optional speed reconstruction for untimed GPX recordings.

This is intentionally an evidence reconstruction, not a claim that local speed
was directly measured. Point order is preserved, lap duration is externally
supplied, and every output point is tagged with its reconstructed source and
confidence.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta, timezone
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from cvt_track_study.config.diagnostics import DiagnosticBag

from .model import CANONICAL_POINT_COLUMNS, GPXIngestionResult
from .parser import _derive_kinematics, _summarize_segments


EARTH_RADIUS_M = 6_371_008.8
_CVT_NS = "https://github.com/gr812b/cvt-study/gpx/reconstruction/1"


def apply_optional_lap_time_reconstruction(
    result: GPXIngestionResult,
    *,
    track_config: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> GPXIngestionResult:
    """Attach auditable timing to an ordered, untimed GPX.

    Spatial laps are detected first. CSV lap durations are then aligned in order.
    The dynamic-programming mode is deliberately cadence-aware: a nominal 1 Hz
    export should contain roughly one point per second, so point count is useful
    evidence for matching CSV rows while still allowing pit/service intervals or
    extra spatial loops to remain unmatched and visible in the audit.
    """

    config = result.metadata.lap_time_reconstruction
    if config is None:
        return result
    if result.metadata.source_file.suffix.lower() != ".gpx":
        raise ValueError(
            "Lap-time reconstruction is only valid for an untimed GPX; "
            "native FIT and timed GPX data must remain unchanged."
        )

    points = result.points.copy()
    timestamp_coverage = float(
        pd.to_datetime(points["timestamp_utc"], utc=True, errors="coerce")
        .notna()
        .mean()
    )
    if timestamp_coverage > 0.05:
        raise ValueError(
            f"Run {result.metadata.run_id!r} already has usable timestamps on "
            f"{timestamp_coverage:.1%} of points. Remove lap_time_reconstruction "
            "and ingest its native timing instead."
        )

    gate_lat, gate_lon = _lap_gate(events, track_config)
    gate_radius_m = (
        config.gate_radius_m
        if config.gate_radius_m is not None
        else _track_gate_radius(track_config)
    )
    crossings = _point_order_gate_crossings(
        points,
        gate_latitude_deg=gate_lat,
        gate_longitude_deg=gate_lon,
        gate_radius_m=gate_radius_m,
        minimum_points_between_crossings=config.minimum_points_per_lap,
    )
    detected_lap_count = max(0, len(crossings) - 1)
    if detected_lap_count < 1:
        raise ValueError(
            f"Run {result.metadata.run_id!r} has fewer than two spatial "
            "start/finish visits, so no complete lap can receive a CSV time."
        )

    detected = pd.DataFrame(
        [
            {
                "detected_lap_index": lap_index,
                "start_source_point_index": int(crossings[lap_index - 1]),
                "end_source_point_index": int(crossings[lap_index]),
                "point_count": int(crossings[lap_index] - crossings[lap_index - 1] + 1),
                "point_interval_count": int(crossings[lap_index] - crossings[lap_index - 1]),
            }
            for lap_index in range(1, detected_lap_count + 1)
        ]
    )
    declared = _read_lap_times(config.lap_times_file, config)
    # Preserve every CSV row during alignment, including pit/service laps.
    # Excluded rows still occupy their chronological lap position; they are
    # simply withheld from pointwise speed reconstruction after matching.
    alignment = _align_lap_times(detected, declared, config)
    matched_all = alignment[alignment["alignment_status"] == "matched"].copy()
    matched = matched_all[matched_all["csv_included"].astype(bool)].copy()
    matched_excluded = matched_all[~matched_all["csv_included"].astype(bool)].copy()
    if not matched_excluded.empty:
        matched_excluded["alignment_status"] = "matched_excluded_csv_lap"
    if len(matched) < config.minimum_matched_laps:
        raise ValueError(
            f"Lap-time reconstruction matched only {len(matched)} lap(s); "
            f"minimum_matched_laps={config.minimum_matched_laps}. Review the "
            "start gate, CSV filtering, and expected 1 Hz cadence."
        )

    unmatched_detected = alignment[
        alignment["alignment_status"] == "unmatched_detected_lap"
    ]
    unmatched_csv = alignment[
        alignment["alignment_status"] == "unmatched_csv_lap"
    ].copy()
    if (len(unmatched_detected) or len(unmatched_csv)) and not config.allow_unmatched_laps:
        raise ValueError(
            "Lap-time reconstruction did not produce a one-to-one alignment: "
            f"{len(unmatched_detected)} detected GPX lap(s) and "
            f"{len(unmatched_csv)} CSV row(s) remain unmatched/excluded. "
            "Correct the gate or CSV, or set allow_unmatched_laps=true for an "
            "explicitly audited partial reconstruction."
        )

    bag = DiagnosticBag(result.diagnostics)
    if len(unmatched_detected) or len(unmatched_csv) or len(matched_excluded):
        bag.warning(
            "LAP_TIME_RECONSTRUCTION_PARTIAL",
            (
                f"Run {result.metadata.run_id} aligned {len(matched_all)} of "
                f"{detected_lap_count} detected spatial laps to {len(declared)} "
                f"CSV rows; {len(matched)} aligned lap(s) are eligible for speed "
                f"reconstruction and {len(matched_excluded)} aligned pit/service "
                "lap(s) remain excluded. Any unmatched rows remain in the audit."
            ),
            path=f"runs.{result.metadata.run_id}.lap_time_reconstruction",
            source=str(config.lap_times_file),
            hint=(
                "Review lap_time_reconstruction.csv before using reconstructed "
                "speed as gate evidence. Pit/service laps should remain excluded, "
                "not silently compressed into race laps."
            ),
        )

    cursor = config.synthetic_start_time_utc
    assert cursor is not None
    output_segments: list[pd.DataFrame] = []
    matched_audit_rows: list[dict[str, Any]] = []

    for output_segment, row in enumerate(
        matched.sort_values("detected_lap_index").itertuples(index=False)
    ):
        lap_index = int(row.detected_lap_index)
        start_position = int(row.start_source_point_index)
        end_position = int(row.end_source_point_index)
        lap = points.iloc[start_position : end_position + 1].copy()
        point_count = len(lap)
        if point_count < config.minimum_points_per_lap:
            message = (
                f"Detected lap {lap_index} contains only {point_count} points; "
                f"minimum_points_per_lap={config.minimum_points_per_lap}."
            )
            if not config.allow_unmatched_laps:
                raise ValueError(message)
            bag.warning(
                "RECONSTRUCTED_LAP_TOO_SHORT",
                message,
                path=f"runs.{result.metadata.run_id}.lap_time_reconstruction",
            )
            continue

        duration_s = float(row.lap_time_s)
        inferred_period_s = duration_s / float(point_count - 1)
        cadence_error_fraction = abs(
            inferred_period_s / config.expected_point_period_s - 1.0
        )
        certainty = _certainty(
            cadence_error_fraction,
            config.maximum_point_period_error_fraction,
        )

        lap["track_index"] = 0
        lap["segment_index"] = output_segment
        lap["point_index"] = np.arange(point_count, dtype=int)
        lap["timestamp_utc"] = [
            cursor + timedelta(seconds=float(value))
            for value in np.linspace(0.0, duration_s, point_count)
        ]
        lap["source_format"] = "gpx_lap_time_reconstructed"
        for column in (
            "step_distance_m",
            "time_step_s",
            "derived_speed_mps",
            "analysis_speed_mps",
        ):
            lap[column] = np.nan
        lap["analysis_speed_source"] = "unavailable"
        lap["speed_certainty"] = "unavailable"

        lap = _derive_kinematics(lap, bag)
        reconstructed_speed = pd.to_numeric(
            lap["derived_speed_mps"], errors="coerce"
        )
        lap["reported_speed_mps"] = reconstructed_speed
        lap["analysis_speed_mps"] = reconstructed_speed
        lap["analysis_speed_source"] = "lap_time_reconstructed"
        # Deliberately below FIT's native_high certainty. The downstream gate
        # sampler converts this label to a larger measurement-error distribution.
        lap["speed_certainty"] = certainty
        lap["extension_json"] = [
            _merged_extension(
                value,
                {
                    "cvt:timing_source": "lap_time_reconstructed",
                    "cvt:detected_lap_index": lap_index,
                    "cvt:csv_row_index": int(row.csv_row_index),
                    "cvt:csv_lap_value": str(row.csv_lap_value),
                    "cvt:declared_lap_time_s": duration_s,
                    "cvt:inferred_point_period_s": inferred_period_s,
                    "cvt:speed_certainty": certainty,
                },
            )
            for value in lap["extension_json"]
        ]
        lap = lap.reindex(columns=CANONICAL_POINT_COLUMNS)
        output_segments.append(lap)

        path_distance_m = float(lap["step_distance_m"].fillna(0.0).sum())
        flags: list[str] = []
        if certainty == "reconstructed_low":
            flags.append("point_period_outside_expected_band")
        elif certainty == "reconstructed_very_low":
            flags.append("point_period_far_outside_expected_band")
        matched_audit_rows.append(
            {
                **row._asdict(),
                "run_id": result.metadata.run_id,
                "vehicle_id": result.metadata.vehicle_id,
                "driver_id": result.metadata.driver_id,
                "output_segment_index": output_segment,
                "path_distance_m": path_distance_m,
                "average_lap_speed_mps": path_distance_m / duration_s,
                "speed_certainty": certainty,
                "quality_flags": ";".join(flags),
            }
        )
        cursor = cursor + timedelta(seconds=duration_s + 1.0)

    if not output_segments:
        raise ValueError(
            f"Run {result.metadata.run_id!r} produced no usable reconstructed laps."
        )

    reconstructed = pd.concat(output_segments, ignore_index=True)
    audit = pd.concat(
        [
            pd.DataFrame(matched_audit_rows),
            unmatched_detected.assign(
                run_id=result.metadata.run_id,
                vehicle_id=result.metadata.vehicle_id,
                driver_id=result.metadata.driver_id,
                output_segment_index=np.nan,
                path_distance_m=np.nan,
                average_lap_speed_mps=np.nan,
                speed_certainty="unavailable",
                quality_flags="unmatched_detected_lap",
            ),
            matched_excluded.assign(
                run_id=result.metadata.run_id,
                vehicle_id=result.metadata.vehicle_id,
                driver_id=result.metadata.driver_id,
                output_segment_index=np.nan,
                path_distance_m=np.nan,
                average_lap_speed_mps=np.nan,
                speed_certainty="unavailable",
                quality_flags=matched_excluded.get(
                    "csv_exclusion_reason",
                    pd.Series("matched_excluded_csv_lap", index=matched_excluded.index),
                ).replace("", "matched_excluded_csv_lap"),
            ),
            unmatched_csv.assign(
                run_id=result.metadata.run_id,
                vehicle_id=result.metadata.vehicle_id,
                driver_id=result.metadata.driver_id,
                output_segment_index=np.nan,
                path_distance_m=np.nan,
                average_lap_speed_mps=np.nan,
                speed_certainty="unavailable",
                quality_flags="unmatched_csv_lap",
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    matched_mask = audit["alignment_status"].astype(str).eq("matched")
    median_distance = float(audit.loc[matched_mask, "path_distance_m"].median())
    audit["distance_ratio_to_median"] = np.nan
    if median_distance > 0.0:
        audit.loc[matched_mask, "distance_ratio_to_median"] = (
            audit.loc[matched_mask, "path_distance_m"] / median_distance
        )
    for index, row in audit.loc[matched_mask].iterrows():
        flags = [item for item in str(row["quality_flags"]).split(";") if item]
        ratio = float(row["distance_ratio_to_median"])
        if math.isfinite(ratio) and not 0.85 <= ratio <= 1.15:
            flags.append("lap_path_distance_outlier")
        audit.at[index, "quality_flags"] = ";".join(flags)

    segment_records = [
        {
            "run_id": result.metadata.run_id,
            "track_index": 0,
            "segment_index": int(segment_index),
            "start_row": int(indices[0]),
            "point_count": len(indices),
        }
        for segment_index, indices in reconstructed.groupby(
            "segment_index", sort=True
        ).groups.items()
    ]
    segments = _summarize_segments(reconstructed, segment_records)

    matched_audit = audit[matched_mask]
    summary = dict(result.summary)
    summary.update(
        {
            "source_format": "gpx_lap_time_reconstructed",
            "lap_time_reconstruction_applied": True,
            "lap_time_reconstruction_method": (
                "segment_crossing_spatial_laps_aligned_one_to_one_with_ordered_csv"
            ),
            "lap_time_alignment_mode": config.alignment_mode,
            "lap_time_csv": str(config.lap_times_file),
            "lap_time_csv_sha256": _sha256(config.lap_times_file),
            "spatial_gate_radius_m": gate_radius_m,
            "detected_spatial_lap_count": detected_lap_count,
            "declared_csv_lap_count": len(declared),
            "included_csv_lap_count": int(declared["csv_included"].sum()),
            "excluded_csv_lap_count": int((~declared["csv_included"]).sum()),
            "matched_csv_lap_count": len(matched_all),
            "matched_excluded_csv_lap_count": len(matched_excluded),
            "unmatched_detected_lap_count": len(unmatched_detected),
            "unmatched_csv_lap_count": int(
                (alignment["alignment_status"] == "unmatched_csv_lap").sum()
            ),
            "reconstructed_lap_count": len(matched_audit),
            "reconstructed_point_count": len(reconstructed),
            "reconstructed_speed_count": int(
                reconstructed["analysis_speed_mps"].notna().sum()
            ),
            "median_inferred_point_period_s": float(
                matched_audit["inferred_point_period_s"].median()
            ),
            "minimum_inferred_point_period_s": float(
                matched_audit["inferred_point_period_s"].min()
            ),
            "maximum_inferred_point_period_s": float(
                matched_audit["inferred_point_period_s"].max()
            ),
            "reconstructed_speed_certainty_contract": (
                "always below FIT native_high; propagated as larger gate-speed "
                "measurement uncertainty"
            ),
            "valid_point_count": len(reconstructed),
            "clean_positioned_point_count": len(reconstructed),
            "unusable_timestamp_count": 0,
            "reported_speed_count": int(
                reconstructed["reported_speed_mps"].notna().sum()
            ),
            "derived_speed_count": int(
                reconstructed["derived_speed_mps"].notna().sum()
            ),
            "total_path_distance_m": float(
                reconstructed["step_distance_m"].fillna(0.0).sum()
            ),
        }
    )

    very_low = int(
        (matched_audit["speed_certainty"] == "reconstructed_very_low").sum()
    )
    if very_low:
        bag.warning(
            "RECONSTRUCTED_POINT_CADENCE_IMPLAUSIBLE",
            (
                f"{very_low} reconstructed lap(s) imply a point period far from "
                f"the expected {config.expected_point_period_s:g} s cadence."
            ),
            path=f"runs.{result.metadata.run_id}.lap_time_reconstruction",
            source=str(config.lap_times_file),
            hint=(
                "Review lap_time_reconstruction.csv. The GPX may have been "
                "spatially simplified or a CSV row may still be misaligned."
            ),
        )
    bag.info(
        "LAP_TIME_SPEED_RECONSTRUCTED",
        (
            f"Reconstructed timing and speed for {len(matched_audit)} lap(s) in "
            f"{result.metadata.run_id} from ordered GPX points and external lap times."
        ),
        path=f"runs.{result.metadata.run_id}.lap_time_reconstruction",
        source=str(config.lap_times_file),
        hint=(
            "This is supplemental vehicle-specific speed evidence. It remains "
            "lower-certainty than native FIT speed and is sampled with a wider "
            "measurement-error contract downstream."
        ),
    )

    augmented = (
        _augmented_gpx(reconstructed, matched_audit, result.metadata.run_id)
        if config.export_augmented_gpx
        else None
    )
    return replace(
        result,
        points=reconstructed,
        segments=segments,
        summary=summary,
        diagnostics=bag.items,
        reconstruction_laps=audit,
        augmented_gpx_text=augmented,
    )

def _lap_gate(
    events: Sequence[Mapping[str, Any]],
    track_config: Mapping[str, Any],
) -> tuple[float, float]:
    reconstruction = track_config.get("reconstruction", {})
    event_id = (
        str(reconstruction.get("lap_gate_event_id", "")).strip()
        if isinstance(reconstruction, Mapping)
        else ""
    )
    candidates = [
        event
        for event in events
        if isinstance(event, Mapping)
        and (
            str(event.get("id", "")) == event_id
            or (
                not event_id
                and str(event.get("analysis_role", "")) == "lap_gate"
            )
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Lap-time reconstruction requires exactly one resolvable lap-gate "
            "event from track.reconstruction.lap_gate_event_id."
        )
    anchor = candidates[0].get("anchor")
    if not isinstance(anchor, Mapping):
        raise ValueError("The lap-gate event is missing its anchor coordinate.")
    return float(anchor["latitude_deg"]), float(anchor["longitude_deg"])


def _track_gate_radius(track_config: Mapping[str, Any]) -> float:
    reconstruction = track_config.get("reconstruction", {})
    if isinstance(reconstruction, Mapping):
        value = reconstruction.get("lap_gate_radius_m", 15.0)
    else:
        value = 15.0
    number = float(value)
    if number <= 0.0:
        raise ValueError("track.reconstruction.lap_gate_radius_m must be positive.")
    return number


def _point_order_gate_crossings(
    points: pd.DataFrame,
    *,
    gate_latitude_deg: float,
    gate_longitude_deg: float,
    gate_radius_m: float,
    minimum_points_between_crossings: int = 1,
) -> list[int]:
    """Detect start/finish passages from GPX *segments*, not sampled points.

    Untimed Strava GPX exports are often spatially simplified. A vehicle can cross
    the start/finish coordinate between two retained points while neither endpoint
    enters a small radius. Point-only detection therefore merges several physical
    laps. This routine measures the shortest distance from every consecutive GPX
    segment to the gate anchor, groups adjacent candidate segments into one visit,
    and returns one source-point boundary per visit.
    """

    lat = pd.to_numeric(points["latitude_deg"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(points["longitude_deg"], errors="coerce").to_numpy(float)
    if len(lat) < 2:
        return []

    valid = np.isfinite(lat) & np.isfinite(lon)
    lat0 = math.radians(gate_latitude_deg)
    x = (
        EARTH_RADIUS_M
        * np.radians(lon - gate_longitude_deg)
        * math.cos(lat0)
    )
    y = EARTH_RADIUS_M * np.radians(lat - gate_latitude_deg)

    p0 = np.column_stack((x[:-1], y[:-1]))
    p1 = np.column_stack((x[1:], y[1:]))
    segment_valid = valid[:-1] & valid[1:]
    delta = p1 - p0
    length_squared = np.sum(delta * delta, axis=1)
    projection = np.divide(
        -np.sum(p0 * delta, axis=1),
        length_squared,
        out=np.zeros(len(delta), dtype=float),
        where=length_squared > 0.0,
    )
    projection = np.clip(projection, 0.0, 1.0)
    closest = p0 + projection[:, None] * delta
    segment_distance = np.linalg.norm(closest, axis=1)
    segment_distance[~segment_valid] = np.nan

    candidates = np.flatnonzero(
        np.isfinite(segment_distance) & (segment_distance <= gate_radius_m)
    )
    groups: list[list[int]] = []
    for position in candidates:
        position = int(position)
        if not groups or position > groups[-1][-1] + 1:
            groups.append([position])
        else:
            groups[-1].append(position)

    visits: list[tuple[int, float]] = []
    point_distance = np.hypot(x, y)
    for group in groups:
        segment_index = min(
            group, key=lambda index: float(segment_distance[index])
        )
        boundary_index = (
            segment_index
            if point_distance[segment_index] <= point_distance[segment_index + 1]
            else segment_index + 1
        )
        visits.append((int(boundary_index), float(segment_distance[segment_index])))

    # Merge only implausibly close repeat detections from one noisy gate visit.
    # Real lap separation remains governed by ordered GPX geometry, not CSV times.
    minimum_gap = max(1, int(minimum_points_between_crossings))
    merged: list[tuple[int, float]] = []
    for boundary, distance in visits:
        if not merged or boundary - merged[-1][0] >= minimum_gap:
            merged.append((boundary, distance))
        elif distance < merged[-1][1]:
            merged[-1] = (boundary, distance)

    return [boundary for boundary, _ in merged]


def _read_lap_times(path: Path, config: Any) -> pd.DataFrame:
    """Read ordinary lap CSVs and race exports with metadata rows before header.

    The ÉTS race export begins with ``Car Number`` and ``Team Name`` rows, then a
    real ``timestamp,lap_time_s`` header. Header discovery is explicit and the
    skipped metadata is retained in ``csv_metadata_json`` for auditability.
    """

    if not path.is_file():
        raise ValueError(f"Lap-time CSV does not exist: {path}")
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.reader(handle))
    if not raw_rows:
        raise ValueError(f"Lap-time CSV is empty: {path}")

    recognized_time = {
        str(config.lap_time_column).strip().lower(),
        "lap_time",
        "lap_time_s",
        "duration",
        "duration_s",
    }
    header_index = None
    for index, row in enumerate(raw_rows):
        normalized = {str(value).strip().lower() for value in row}
        if normalized & recognized_time:
            header_index = index
            break
    if header_index is None:
        raise ValueError(
            "Could not find a lap-time header row. Expected one of: "
            + ", ".join(sorted(recognized_time))
        )

    metadata = {
        str(row[0]).strip(): str(row[1]).strip()
        for row in raw_rows[:header_index]
        if len(row) >= 2 and str(row[0]).strip()
    }
    frame = pd.read_csv(path, skiprows=header_index, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError(f"Lap-time CSV contains a header but no data rows: {path}")

    lap_column = _resolve_column(
        frame,
        config.lap_index_column,
        ("lap", "lap_index", "lap_number"),
        required=False,
    )
    time_column = _resolve_column(
        frame,
        config.lap_time_column,
        ("lap_time", "lap_time_s", "duration", "duration_s"),
        required=True,
    )
    timestamp_column = next(
        (candidate for candidate in ("timestamp", "lap_timestamp", "time") if candidate in frame.columns),
        None,
    )

    selected = frame.copy().reset_index(drop=True)
    selected["csv_row_index"] = np.arange(1, len(selected) + 1, dtype=int)
    include_column = config.include_column
    explicit_include = (
        selected[include_column].map(_included)
        if include_column and include_column in selected.columns
        else pd.Series(True, index=selected.index)
    )

    durations = selected[time_column].map(_parse_lap_duration)
    if durations.isna().any() or (durations <= 0.0).any():
        bad = selected.loc[durations.isna() | (durations <= 0.0), "csv_row_index"].tolist()
        raise ValueError(
            f"Lap time column {time_column!r} contains invalid/non-positive values at CSV rows {bad}."
        )

    if lap_column is None:
        lap_indices = np.arange(1, len(selected) + 1, dtype=int)
        csv_lap_values = (
            selected[timestamp_column].astype(str).to_numpy()
            if timestamp_column is not None
            else lap_indices.astype(object)
        )
    else:
        numeric = pd.to_numeric(selected[lap_column], errors="coerce")
        if numeric.isna().any() or (numeric < 1).any():
            raise ValueError(
                f"Lap index column {lap_column!r} must contain positive integers."
            )
        lap_indices = numeric.astype(int).to_numpy()
        csv_lap_values = selected[lap_column].astype(str).to_numpy()

    reasons: list[str] = []
    included: list[bool] = []
    for explicit, duration in zip(explicit_include.astype(bool), durations.astype(float)):
        reason = ""
        keep = bool(explicit)
        if not keep:
            reason = "explicitly_excluded"
        if keep and config.minimum_csv_lap_time_s is not None and duration < config.minimum_csv_lap_time_s:
            keep = False
            reason = "below_minimum_csv_lap_time"
        if keep and config.maximum_csv_lap_time_s is not None and duration > config.maximum_csv_lap_time_s:
            keep = False
            reason = "above_maximum_csv_lap_time"
        included.append(keep)
        reasons.append(reason)

    output = pd.DataFrame(
        {
            "csv_row_index": selected["csv_row_index"].astype(int),
            "csv_lap_index": lap_indices,
            "csv_lap_value": csv_lap_values,
            "csv_timestamp": (
                selected[timestamp_column].astype(str)
                if timestamp_column is not None
                else ""
            ),
            "lap_time_s": durations.astype(float),
            "csv_included": included,
            "csv_exclusion_reason": reasons,
            "csv_metadata_json": json.dumps(metadata, sort_keys=True, ensure_ascii=False),
        }
    )
    if lap_column is not None and output["csv_lap_index"].duplicated().any():
        duplicates = sorted(
            output.loc[
                output["csv_lap_index"].duplicated(keep=False), "csv_lap_index"
            ].unique()
        )
        raise ValueError(f"Lap-time CSV repeats lap index values: {duplicates}")
    return output


def _align_lap_times(
    detected: pd.DataFrame,
    declared: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    if config.alignment_mode == "strict_index":
        by_index = declared.set_index("csv_lap_index", drop=False)
        rows: list[dict[str, Any]] = []
        for d in detected.itertuples(index=False):
            if int(d.detected_lap_index) not in by_index.index:
                rows.append({**d._asdict(), "alignment_status": "unmatched_detected_lap"})
                continue
            c = by_index.loc[int(d.detected_lap_index)]
            period = float(c["lap_time_s"]) / max(1, int(d.point_interval_count))
            error = abs(period / config.expected_point_period_s - 1.0)
            rows.append(
                {
                    **d._asdict(),
                    **c.to_dict(),
                    "alignment_status": "matched",
                    "inferred_point_period_s": period,
                    "point_period_error_fraction": error,
                    "alignment_cost": error**2,
                }
            )
        matched_csv = {int(row["csv_row_index"]) for row in rows if row.get("alignment_status") == "matched"}
        for c in declared.itertuples(index=False):
            if int(c.csv_row_index) not in matched_csv:
                rows.append({**c._asdict(), "alignment_status": "unmatched_csv_lap"})
        return pd.DataFrame(rows)

    if config.alignment_mode != "cadence_dynamic_programming":
        raise ValueError(f"Unsupported alignment_mode {config.alignment_mode!r}.")

    detected_rows = list(detected.itertuples(index=False))
    csv_rows = list(declared.itertuples(index=False))
    n, m = len(csv_rows), len(detected_rows)
    inf = float("inf")
    cost = np.full((n + 1, m + 1), inf, dtype=float)
    action = np.full((n + 1, m + 1), "", dtype=object)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        cost[i, 0] = cost[i - 1, 0] + config.skip_csv_lap_penalty
        action[i, 0] = "skip_csv"
    for j in range(1, m + 1):
        cost[0, j] = cost[0, j - 1] + config.skip_detected_lap_penalty
        action[0, j] = "skip_detected"

    for i in range(1, n + 1):
        csv_row = csv_rows[i - 1]
        for j in range(1, m + 1):
            detected_row = detected_rows[j - 1]
            period = float(csv_row.lap_time_s) / max(1, int(detected_row.point_interval_count))
            error = abs(period / config.expected_point_period_s - 1.0)
            # A smooth, bounded cadence cost: implausible matches remain possible
            # only when skipping both sides would be even worse, and they are
            # subsequently labelled very low certainty.
            match_penalty = (error / config.maximum_alignment_error_fraction) ** 2
            candidates = (
                (cost[i - 1, j - 1] + match_penalty, "match"),
                (cost[i - 1, j] + config.skip_csv_lap_penalty, "skip_csv"),
                (cost[i, j - 1] + config.skip_detected_lap_penalty, "skip_detected"),
            )
            best_cost, best_action = min(candidates, key=lambda item: item[0])
            cost[i, j] = best_cost
            action[i, j] = best_action

    matches: list[tuple[int, int]] = []
    unmatched_csv: list[int] = []
    unmatched_detected: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = action[i, j]
        if step == "match":
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif step == "skip_csv":
            unmatched_csv.append(i - 1)
            i -= 1
        elif step == "skip_detected":
            unmatched_detected.append(j - 1)
            j -= 1
        else:
            raise ValueError("Internal lap-alignment backtracking failure.")
    matches.reverse()
    unmatched_csv.reverse()
    unmatched_detected.reverse()

    rows: list[dict[str, Any]] = []
    for csv_index, detected_index in matches:
        c = csv_rows[csv_index]
        d = detected_rows[detected_index]
        period = float(c.lap_time_s) / max(1, int(d.point_interval_count))
        error = abs(period / config.expected_point_period_s - 1.0)
        rows.append(
            {
                **d._asdict(),
                **c._asdict(),
                "alignment_status": "matched",
                "inferred_point_period_s": period,
                "point_period_error_fraction": error,
                "alignment_cost": (error / config.maximum_alignment_error_fraction) ** 2,
            }
        )
    for index in unmatched_detected:
        d = detected_rows[index]
        rows.append({**d._asdict(), "alignment_status": "unmatched_detected_lap"})
    for index in unmatched_csv:
        c = csv_rows[index]
        rows.append({**c._asdict(), "alignment_status": "unmatched_csv_lap"})
    return pd.DataFrame(rows)

def _resolve_column(
    frame: pd.DataFrame,
    configured: str,
    alternatives: tuple[str, ...],
    *,
    required: bool,
) -> str | None:
    if configured in frame.columns:
        return configured
    for candidate in alternatives:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ValueError(
            f"CSV column {configured!r} was not found. Available columns: "
            + ", ".join(map(str, frame.columns))
        )
    return None


def _parse_lap_duration(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return math.nan
    if len(numbers) == 2:
        minutes, seconds = numbers
        return 60.0 * minutes + seconds
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
        return 3600.0 * hours + 60.0 * minutes + seconds
    return math.nan


def _included(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    return str(value).strip().lower() not in {
        "0", "false", "no", "exclude", "excluded", "skip"
    }


def _certainty(error_fraction: float, allowed: float) -> str:
    if error_fraction <= allowed:
        return "reconstructed_low_medium"
    if error_fraction <= 2.0 * allowed:
        return "reconstructed_low"
    return "reconstructed_very_low"


def _merged_extension(raw: Any, additions: Mapping[str, Any]) -> str:
    try:
        parsed = json.loads(str(raw)) if str(raw).strip() else {}
    except json.JSONDecodeError:
        parsed = {"original_extension_text": str(raw)}
    if not isinstance(parsed, dict):
        parsed = {"original_extension": parsed}
    parsed.update(additions)
    return json.dumps(parsed, sort_keys=True)


def _haversine_to_point(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    point_latitude_deg: float,
    point_longitude_deg: float,
) -> np.ndarray:
    lat1 = np.radians(latitude_deg)
    lon1 = np.radians(longitude_deg)
    lat2 = math.radians(point_latitude_deg)
    lon2 = math.radians(point_longitude_deg)
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    value = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * math.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * np.arcsin(
        np.sqrt(np.clip(value, 0.0, 1.0))
    )


def _augmented_gpx(
    points: pd.DataFrame,
    audit: pd.DataFrame,
    run_id: str,
) -> str:
    ET.register_namespace("cvt", _CVT_NS)
    root = ET.Element(
        "gpx",
        {
            "version": "1.1",
            "creator": "cvt-track-study lap-time reconstruction",
            "xmlns": "http://www.topografix.com/GPX/1/1",
        },
    )
    metadata = ET.SubElement(root, "metadata")
    ET.SubElement(metadata, "name").text = f"{run_id} reconstructed timing"
    description = ET.SubElement(metadata, "desc")
    description.text = (
        "Synthetic within-lap timestamps and speeds reconstructed from "
        "ordered GPX points plus externally supplied lap durations."
    )
    track = ET.SubElement(root, "trk")
    ET.SubElement(track, "name").text = f"{run_id} reconstructed"
    audit_lookup = {
        int(row.output_segment_index): row
        for row in audit.itertuples(index=False)
        if pd.notna(row.output_segment_index)
    }
    for segment_index, segment in points.groupby("segment_index", sort=True):
        trkseg = ET.SubElement(track, "trkseg")
        audit_row = audit_lookup.get(int(segment_index))
        for row in segment.itertuples(index=False):
            trkpt = ET.SubElement(
                trkseg,
                "trkpt",
                {
                    "lat": f"{float(row.latitude_deg):.9f}",
                    "lon": f"{float(row.longitude_deg):.9f}",
                },
            )
            if math.isfinite(float(row.elevation_m)):
                ET.SubElement(trkpt, "ele").text = f"{float(row.elevation_m):.3f}"
            timestamp = pd.Timestamp(row.timestamp_utc)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(timezone.utc)
            timestamp = timestamp.tz_convert(timezone.utc)
            ET.SubElement(trkpt, "time").text = (
                timestamp.isoformat().replace("+00:00", "Z")
            )
            speed = float(row.analysis_speed_mps)
            if math.isfinite(speed):
                ET.SubElement(trkpt, "speed").text = f"{speed:.6f}"
            extensions = ET.SubElement(trkpt, "extensions")
            ET.SubElement(
                extensions, f"{{{_CVT_NS}}}timing_source"
            ).text = "lap_time_reconstructed"
            ET.SubElement(
                extensions, f"{{{_CVT_NS}}}speed_certainty"
            ).text = str(row.speed_certainty)
            if audit_row is not None:
                ET.SubElement(
                    extensions, f"{{{_CVT_NS}}}declared_lap_time_s"
                ).text = f"{float(audit_row.lap_time_s):.6f}"
    return ET.tostring(
        root,
        encoding="unicode",
        xml_declaration=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
