"""Conservative, auditable cleanup of isolated GPS coordinate excursions.

The cleanup is intentionally narrow. It does not smooth a route, snap points to
a centreline, or repair sustained off-course travel. It removes only short
bursts that are physically impossible on both sides while the direct bridge
between their valid neighbours is physically plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cvt_track_study.config.diagnostics import Diagnostic, DiagnosticBag, Severity

from .model import CANONICAL_POINT_COLUMNS, GPXIngestionResult
from .parser import _derive_kinematics, _summarize_segments


EARTH_RADIUS_M = 6_371_008.8

_REJECTION_COLUMNS = (
    "rejection_stage",
    "rejection_reason",
    "candidate_group_size",
    "left_leg_distance_m",
    "right_leg_distance_m",
    "bridge_distance_m",
    "left_leg_speed_mps",
    "right_leg_speed_mps",
    "bridge_speed_mps",
    "map_error_m",
)

_INVALID_COORDINATE_CODES = {
    "INVALID_FIT_COORDINATE",
    "INVALID_GPX_COORDINATE",
}


@dataclass(frozen=True)
class TelemetryCleanupSettings:
    """Resolved telemetry-cleanup policy.

    Values are deliberately conservative. The speed thresholds are relative to
    the project's declared maximum reasonable vehicle speed.
    """

    enabled: bool = True
    maximum_reasonable_speed_mps: float = 25.0
    maximum_map_error_m: float = 20.0
    maximum_excursion_points: int = 3
    minimum_excursion_leg_m: float = 35.0
    impossible_speed_multiplier: float = 1.5
    maximum_bridge_speed_multiplier: float = 1.0
    maximum_bridge_gap_s: float = 8.0
    maximum_auto_removed_fraction: float = 0.005
    maximum_auto_removed_points: int = 25
    isolated_map_error_m: float = 40.0
    maximum_isolated_map_outlier_points: int = 3

    @classmethod
    def from_mapping(cls, track: Mapping[str, Any]) -> "TelemetryCleanupSettings":
        reconstruction = _mapping(track.get("reconstruction"))
        cleanup = _mapping(track.get("telemetry_cleanup"))
        settings = cls(
            enabled=_boolean(cleanup.get("enabled"), True),
            maximum_reasonable_speed_mps=_positive_float(
                reconstruction.get("maximum_reasonable_speed_mps"), 25.0
            ),
            maximum_map_error_m=_positive_float(
                reconstruction.get("maximum_map_error_m"), 20.0
            ),
            maximum_excursion_points=_positive_int(
                cleanup.get("maximum_excursion_points"), 3
            ),
            minimum_excursion_leg_m=_positive_float(
                cleanup.get("minimum_excursion_leg_m"), 35.0
            ),
            impossible_speed_multiplier=_positive_float(
                cleanup.get("impossible_speed_multiplier"), 1.5
            ),
            maximum_bridge_speed_multiplier=_positive_float(
                cleanup.get("maximum_bridge_speed_multiplier"), 1.0
            ),
            maximum_bridge_gap_s=_positive_float(
                cleanup.get("maximum_bridge_gap_s"), 8.0
            ),
            maximum_auto_removed_fraction=_fraction(
                cleanup.get("maximum_auto_removed_fraction"), 0.005
            ),
            maximum_auto_removed_points=_positive_int(
                cleanup.get("maximum_auto_removed_points"), 25
            ),
            isolated_map_error_m=_positive_float(
                cleanup.get("isolated_map_error_m"), 40.0
            ),
            maximum_isolated_map_outlier_points=_positive_int(
                cleanup.get("maximum_isolated_map_outlier_points"), 3
            ),
        )
        if settings.isolated_map_error_m <= settings.maximum_map_error_m:
            raise ValueError(
                "track.telemetry_cleanup.isolated_map_error_m must exceed "
                "track.reconstruction.maximum_map_error_m"
            )
        return settings


def apply_telemetry_cleanup(
    result: GPXIngestionResult,
    track_config: Mapping[str, Any],
) -> GPXIngestionResult:
    """Return an ingestion result with isolated coordinate excursions excluded."""

    settings = TelemetryCleanupSettings.from_mapping(track_config)
    points = result.points.copy()
    empty = _empty_rejections(points)
    summary = dict(result.summary)
    summary["raw_positioned_point_count"] = len(points)
    summary["telemetry_cleanup_enabled"] = settings.enabled

    if not settings.enabled or len(points) < 3:
        diagnostics = _collapse_invalid_coordinate_diagnostics(
            result.diagnostics, result.summary, result.metadata.source_file
        )
        summary["clean_positioned_point_count"] = len(points)
        summary["isolated_excursion_point_count"] = 0
        return replace(
            result,
            summary=summary,
            diagnostics=diagnostics,
            rejected_points=empty,
        )

    candidates = _isolated_excursion_candidates(points, settings)
    candidate_indices = sorted(
        {index for candidate in candidates for index in candidate["indices"]}
    )
    allowed = _automatic_removal_limit(len(points), settings)

    base_diagnostics = _collapse_invalid_coordinate_diagnostics(
        result.diagnostics, result.summary, result.metadata.source_file
    )
    bag = DiagnosticBag(base_diagnostics)

    if len(candidate_indices) > allowed:
        bag.warning(
            "TELEMETRY_CLEANUP_LIMIT_EXCEEDED",
            (
                f"Detected {len(candidate_indices)} isolated excursion point(s), "
                f"exceeding the automatic-removal limit of {allowed}; no valid "
                "coordinate points were removed."
            ),
            path=f"runs.{result.metadata.run_id}",
            source=str(result.metadata.source_file),
            hint=(
                "Inspect the cleanup diagnostics. A substantial excursion may be "
                "real off-course travel, a recording failure, or a run that should "
                "be split or excluded."
            ),
        )
        summary["clean_positioned_point_count"] = len(points)
        summary["isolated_excursion_point_count"] = 0
        summary["cleanup_candidate_point_count"] = len(candidate_indices)
        return replace(
            result,
            summary=summary,
            diagnostics=bag.items,
            rejected_points=empty,
        )

    if not candidate_indices:
        summary["clean_positioned_point_count"] = len(points)
        summary["isolated_excursion_point_count"] = 0
        summary["cleanup_candidate_point_count"] = 0
        return replace(
            result,
            summary=summary,
            diagnostics=bag.items,
            rejected_points=empty,
        )

    rejected = _build_pre_ingestion_rejections(points, candidates)
    cleaned = points.drop(index=candidate_indices).reset_index(drop=True)

    # Recompute every distance/speed field after deletion. This is critical:
    # simply hiding the bad coordinates would leave false path distance behind.
    cleaned = cleaned.reindex(columns=CANONICAL_POINT_COLUMNS)
    for column in (
        "step_distance_m",
        "time_step_s",
        "derived_speed_mps",
        "analysis_speed_mps",
    ):
        cleaned[column] = np.nan
    cleaned["analysis_speed_source"] = "unavailable"
    cleaned["speed_certainty"] = "unavailable"

    recompute_bag = DiagnosticBag(
        item for item in bag.items if not _is_recomputed_kinematic_diagnostic(item)
    )
    cleaned = _derive_kinematics(cleaned, recompute_bag)
    cleaned = cleaned.reindex(columns=CANONICAL_POINT_COLUMNS)
    segments = _summarize_segments(cleaned, _segment_records(cleaned))

    summary["clean_positioned_point_count"] = len(cleaned)
    summary["valid_point_count"] = len(cleaned)
    summary["isolated_excursion_point_count"] = len(rejected)
    summary["cleanup_candidate_point_count"] = len(candidate_indices)
    summary["total_path_distance_m"] = float(
        cleaned["step_distance_m"].fillna(0.0).sum()
    )
    summary["derived_speed_count"] = int(cleaned["derived_speed_mps"].notna().sum())
    summary["unusable_timestamp_count"] = int(cleaned["timestamp_utc"].isna().sum())
    summary["missing_elevation_count"] = int(cleaned["elevation_m"].isna().sum())

    recompute_bag.warning(
        "ISOLATED_TELEMETRY_EXCURSIONS_REMOVED",
        (
            f"Removed {len(rejected)} isolated coordinate excursion point(s) "
            f"from {result.metadata.run_id}; raw source telemetry was not modified."
        ),
        path=f"runs.{result.metadata.run_id}",
        source=str(result.metadata.source_file),
        hint=(
            "Review rejected_telemetry_points.csv and telemetry_cleanup_map.png. "
            "Longer or ambiguous excursions are intentionally retained."
        ),
    )
    return replace(
        result,
        points=cleaned,
        segments=segments,
        summary=summary,
        diagnostics=recompute_bag.items,
        rejected_points=rejected,
    )


def remove_isolated_map_outliers(
    matched_points: pd.DataFrame,
    laps: pd.DataFrame,
    track_config: Mapping[str, Any],
    diagnostics: DiagnosticBag,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove tiny, bracketed map-error bursts after a centreline exists.

    This second pass catches isolated position errors that were not fast enough
    to violate the pre-lap physical-continuity test, commonly because a source
    timestamp gap accompanied the GPS error.
    """

    settings = TelemetryCleanupSettings.from_mapping(track_config)
    empty = _empty_rejections(matched_points)
    if (
        not settings.enabled
        or matched_points.empty
        or "map_error_m" not in matched_points
    ):
        return matched_points, laps, empty

    candidates: list[dict[str, Any]] = []
    for _, indices in matched_points.groupby("lap_id", sort=False).groups.items():
        positions = list(indices)
        segment = matched_points.loc[positions]
        errors = pd.to_numeric(segment["map_error_m"], errors="coerce").to_numpy(float)
        bad = np.isfinite(errors) & (errors >= settings.isolated_map_error_m)
        for start, end in _true_runs(bad):
            size = end - start
            if size > settings.maximum_isolated_map_outlier_points:
                continue
            if start == 0 or end >= len(segment):
                continue
            left_error = errors[start - 1]
            right_error = errors[end]
            if (
                not np.isfinite(left_error)
                or not np.isfinite(right_error)
                or left_error > settings.maximum_map_error_m
                or right_error > settings.maximum_map_error_m
            ):
                continue
            local_indices = positions[start:end]
            bridge_dt = _elapsed_seconds(
                segment.iloc[start - 1]["timestamp_utc"],
                segment.iloc[end]["timestamp_utc"],
            )
            if np.isfinite(bridge_dt) and bridge_dt > settings.maximum_bridge_gap_s:
                continue
            candidates.append(
                {
                    "indices": local_indices,
                    "candidate_group_size": size,
                    "left_leg_distance_m": math.nan,
                    "right_leg_distance_m": math.nan,
                    "bridge_distance_m": math.nan,
                    "left_leg_speed_mps": math.nan,
                    "right_leg_speed_mps": math.nan,
                    "bridge_speed_mps": math.nan,
                }
            )

    candidate_indices = sorted(
        {index for candidate in candidates for index in candidate["indices"]}
    )
    allowed = _automatic_removal_limit(len(matched_points), settings)
    if len(candidate_indices) > allowed:
        diagnostics.warning(
            "MAP_CLEANUP_LIMIT_EXCEEDED",
            (
                f"Detected {len(candidate_indices)} isolated map outlier point(s), "
                f"exceeding the automatic-removal limit of {allowed}; none were removed."
            ),
            hint=(
                "Inspect the lap and centreline. Sustained map disagreement must not "
                "be silently repaired."
            ),
        )
        return matched_points, laps, empty
    if not candidate_indices:
        return matched_points, laps, empty

    rejected = _build_map_rejections(matched_points, candidates)
    cleaned = matched_points.drop(index=candidate_indices).reset_index(drop=True)
    updated = _refresh_map_quality(cleaned, laps, settings)
    removed_by_lap = (
        rejected.groupby("lap_id").size().to_dict()
        if "lap_id" in rejected.columns and not rejected.empty
        else {}
    )
    for row_index, lap in updated.iterrows():
        updated.loc[row_index, "map_cleanup_removed_point_count"] = int(
            removed_by_lap.get(int(lap["lap_id"]), 0)
        )

    diagnostics.warning(
        "ISOLATED_MAP_OUTLIERS_REMOVED",
        (
            f"Removed {len(rejected)} isolated post-map outlier point(s); longer "
            "or unbracketed map excursions remain reviewable and can still reject a lap."
        ),
        hint="Review track/rejected_map_points.csv and telemetry_cleanup_map.png.",
    )
    return cleaned, updated, rejected


def create_telemetry_cleanup_map(
    path: Any,
    retained_points: pd.DataFrame,
    rejected_points: pd.DataFrame,
) -> None:
    """Plot retained and rejected telemetry without requiring a built centreline."""

    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 8))

    retained = retained_points.dropna(
        subset=["latitude_deg", "longitude_deg"]
    ).copy()
    rejected = rejected_points.dropna(
        subset=["latitude_deg", "longitude_deg"]
    ).copy()

    combined = pd.concat(
        [retained[["latitude_deg", "longitude_deg"]],
         rejected[["latitude_deg", "longitude_deg"]]],
        ignore_index=True,
    )
    if combined.empty:
        axis.text(0.5, 0.5, "No positioned telemetry", ha="center", va="center")
    else:
        lat0 = float(combined["latitude_deg"].median())
        lon0 = float(combined["longitude_deg"].median())

        def xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            lat = np.deg2rad(frame["latitude_deg"].to_numpy(float))
            lon = np.deg2rad(frame["longitude_deg"].to_numpy(float))
            x = EARTH_RADIUS_M * (lon - math.radians(lon0)) * math.cos(
                math.radians(lat0)
            )
            y = EARTH_RADIUS_M * (lat - math.radians(lat0))
            return x, y

        for _, segment in retained.groupby(
            ["run_id", "track_index", "segment_index"], sort=False
        ):
            x, y = xy(segment)
            axis.plot(x, y, linewidth=0.7, alpha=0.35)

        if not rejected.empty:
            x, y = xy(rejected)
            axis.scatter(x, y, marker="x", s=48, label="automatically excluded")
            axis.legend(loc="best")

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Local east [m]")
    axis.set_ylabel("Local north [m]")
    axis.set_title("Telemetry cleanup: retained points and explicit exclusions")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _isolated_excursion_candidates(
    points: pd.DataFrame,
    settings: TelemetryCleanupSettings,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for _, indices in points.groupby(
        ["run_id", "track_index", "segment_index"], sort=False
    ).groups.items():
        positions = list(indices)
        segment = points.loc[positions].reset_index()
        cursor = 1
        while cursor < len(segment) - 1:
            accepted: dict[str, Any] | None = None
            for size in range(1, settings.maximum_excursion_points + 1):
                right = cursor + size
                if right >= len(segment):
                    break
                metrics = _excursion_metrics(segment, cursor, right, settings)
                if metrics is not None:
                    accepted = {
                        **metrics,
                        "indices": segment.loc[cursor:right - 1, "index"]
                        .astype(int)
                        .tolist(),
                        "candidate_group_size": size,
                    }
                    break
            if accepted is None:
                cursor += 1
            else:
                candidates.append(accepted)
                cursor += int(accepted["candidate_group_size"])
    return candidates


def _excursion_metrics(
    segment: pd.DataFrame,
    start: int,
    right: int,
    settings: TelemetryCleanupSettings,
) -> dict[str, float] | None:
    left = start - 1
    last = right - 1
    left_leg = _point_distance(segment.iloc[left], segment.iloc[start])
    right_leg = _point_distance(segment.iloc[last], segment.iloc[right])
    bridge = _point_distance(segment.iloc[left], segment.iloc[right])

    dt_left = _elapsed_seconds(
        segment.iloc[left]["timestamp_utc"],
        segment.iloc[start]["timestamp_utc"],
    )
    dt_right = _elapsed_seconds(
        segment.iloc[last]["timestamp_utc"],
        segment.iloc[right]["timestamp_utc"],
    )
    dt_bridge = _elapsed_seconds(
        segment.iloc[left]["timestamp_utc"],
        segment.iloc[right]["timestamp_utc"],
    )

    left_speed = _safe_speed(left_leg, dt_left)
    right_speed = _safe_speed(right_leg, dt_right)
    bridge_speed = _safe_speed(bridge, dt_bridge)

    timed = all(np.isfinite(value) for value in (left_speed, right_speed, bridge_speed))
    if timed:
        impossible = (
            settings.maximum_reasonable_speed_mps
            * settings.impossible_speed_multiplier
        )
        bridge_limit = (
            settings.maximum_reasonable_speed_mps
            * settings.maximum_bridge_speed_multiplier
        )
        suspicious = (
            left_leg >= settings.minimum_excursion_leg_m
            and right_leg >= settings.minimum_excursion_leg_m
            and left_speed > impossible
            and right_speed > impossible
        )
        plausible_bridge = (
            0 < dt_bridge <= settings.maximum_bridge_gap_s
            and bridge_speed <= bridge_limit
        )
    else:
        # Untimed removal requires an unmistakably large out-and-back excursion.
        untimed_leg = max(100.0, 3.0 * settings.minimum_excursion_leg_m)
        suspicious = left_leg >= untimed_leg and right_leg >= untimed_leg
        plausible_bridge = bridge <= settings.minimum_excursion_leg_m

    if not (suspicious and plausible_bridge):
        return None
    return {
        "left_leg_distance_m": left_leg,
        "right_leg_distance_m": right_leg,
        "bridge_distance_m": bridge,
        "left_leg_speed_mps": left_speed,
        "right_leg_speed_mps": right_speed,
        "bridge_speed_mps": bridge_speed,
    }


def _build_pre_ingestion_rejections(
    points: pd.DataFrame,
    candidates: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for index in candidate["indices"]:
            row = points.loc[index].to_dict()
            row.update(
                {
                    "rejection_stage": "pre_lap_physical_continuity",
                    "rejection_reason": "isolated_bracketed_coordinate_excursion",
                    **{
                        key: candidate.get(key, math.nan)
                        for key in _REJECTION_COLUMNS
                        if key not in {"rejection_stage", "rejection_reason", "map_error_m"}
                    },
                    "map_error_m": math.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).reindex(
        columns=_rejection_schema_columns(points)
    )


def _build_map_rejections(
    points: pd.DataFrame,
    candidates: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for index in candidate["indices"]:
            row = points.loc[index].to_dict()
            row.update(
                {
                    "rejection_stage": "post_map_centreline_consistency",
                    "rejection_reason": "isolated_bracketed_map_outlier",
                    **{
                        key: candidate.get(key, math.nan)
                        for key in _REJECTION_COLUMNS
                        if key not in {"rejection_stage", "rejection_reason", "map_error_m"}
                    },
                    "map_error_m": points.loc[index].get("map_error_m", math.nan),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).reindex(
        columns=_rejection_schema_columns(points)
    )


def _refresh_map_quality(
    matched: pd.DataFrame,
    laps: pd.DataFrame,
    settings: TelemetryCleanupSettings,
) -> pd.DataFrame:
    updated = laps.copy()
    updated["map_cleanup_removed_point_count"] = 0
    for row_index, lap in updated.iterrows():
        lap_id = int(lap["lap_id"])
        segment = matched[matched["lap_id"] == lap_id]
        if segment.empty:
            updated.loc[row_index, "analysis_valid"] = False
            _set_quality_flags(updated, row_index, ["no_map_matched_points"])
            continue

        errors = pd.to_numeric(segment["map_error_m"], errors="coerce").dropna().to_numpy(float)
        matched_s = pd.to_numeric(segment["s_m"], errors="coerce").dropna().to_numpy(float)
        if not len(errors):
            updated.loc[row_index, "analysis_valid"] = False
            _set_quality_flags(updated, row_index, ["no_finite_map_errors"])
            continue

        flags = _quality_flags(updated.loc[row_index, "quality_flags"])
        flags = [
            flag for flag in flags
            if flag not in {"p95_map_error_exceeds_limit", "large_backward_map_match"}
        ]
        backward = int(np.sum(np.diff(matched_s) < -20.0)) if len(matched_s) > 1 else 0
        p95 = float(np.quantile(errors, 0.95))
        if backward:
            flags.append("large_backward_map_match")
        if p95 > settings.maximum_map_error_m:
            flags.append("p95_map_error_exceeds_limit")

        updated.loc[row_index, "median_map_error_m"] = float(np.median(errors))
        updated.loc[row_index, "p95_map_error_m"] = p95
        updated.loc[row_index, "maximum_map_error_m"] = float(np.max(errors))
        updated.loc[row_index, "large_backward_match_count"] = backward
        _set_quality_flags(updated, row_index, flags)
        updated.loc[row_index, "analysis_valid"] = len(flags) == 0

    return updated


def _collapse_invalid_coordinate_diagnostics(
    diagnostics: tuple[Diagnostic, ...],
    summary: Mapping[str, Any],
    source_file: Any,
) -> tuple[Diagnostic, ...]:
    details = [item for item in diagnostics if item.code in _INVALID_COORDINATE_CODES]
    retained = [item for item in diagnostics if item.code not in _INVALID_COORDINATE_CODES]
    count = int(summary.get("invalid_coordinate_count", len(details)) or 0)
    if not count:
        return tuple(retained)

    indices = sorted(
        value
        for item in details
        for value in _diagnostic_indices(item.path)
    )
    location = ""
    if indices:
        location = f" Affected source-record span: {indices[0]}–{indices[-1]}."
    retained.append(
        Diagnostic(
            severity=Severity.WARNING,
            code="INVALID_TELEMETRY_COORDINATES_SUMMARY",
            message=(
                f"{count} missing or out-of-range coordinate record(s) were "
                f"excluded during parsing.{location}"
            ),
            source=str(source_file),
            hint=(
                "Detailed invalid-coordinate counts remain in the run summary; "
                "valid positioned records were retained."
            ),
        )
    )
    return tuple(retained)


def _diagnostic_indices(path: str) -> list[int]:
    return [int(value) for value in re.findall(r"\[(\d+)\]", path)]


def _is_recomputed_kinematic_diagnostic(item: Diagnostic) -> bool:
    return (
        item.code.startswith("DUPLICATE_")
        or item.code.endswith("_SAMPLING_GAPS")
        or item.code.endswith("_TIMESTAMP_REGRESSION")
    )


def _segment_records(points: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    start_row = 0
    for (run_id, track_index, segment_index), indices in points.groupby(
        ["run_id", "track_index", "segment_index"], sort=False
    ).groups.items():
        count = len(indices)
        records.append(
            {
                "run_id": str(run_id),
                "track_index": int(track_index),
                "segment_index": int(segment_index),
                "start_row": start_row,
                "point_count": count,
            }
        )
        start_row += count
    return records


def _automatic_removal_limit(
    point_count: int,
    settings: TelemetryCleanupSettings,
) -> int:
    fraction_limit = max(
        1, int(math.floor(point_count * settings.maximum_auto_removed_fraction))
    )
    return min(settings.maximum_auto_removed_points, fraction_limit)


def _empty_rejections(points: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        columns=_rejection_schema_columns(points)
    )


def _rejection_schema_columns(
    points: pd.DataFrame,
) -> list[str]:
    """Return stable rejection columns with each label appearing once.

    Matched-point tables already include fields such as ``map_error_m``.
    Rejection metadata may name the same field, so concatenating the two raw
    column lists would create duplicate labels and later break pandas concat.
    """

    return list(
        dict.fromkeys(
            [
                *(str(column) for column in points.columns),
                *_REJECTION_COLUMNS,
            ]
        )
    )


def _point_distance(left: pd.Series, right: pd.Series) -> float:
    lat1 = math.radians(float(left["latitude_deg"]))
    lat2 = math.radians(float(right["latitude_deg"]))
    dlat = lat2 - lat1
    dlon = math.radians(float(right["longitude_deg"]) - float(left["longitude_deg"]))
    value = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(value)))


def _elapsed_seconds(left: Any, right: Any) -> float:
    left_time = pd.to_datetime(left, utc=True, errors="coerce")
    right_time = pd.to_datetime(right, utc=True, errors="coerce")
    if pd.isna(left_time) or pd.isna(right_time):
        return math.nan
    return float((right_time - left_time).total_seconds())


def _safe_speed(distance_m: float, elapsed_s: float) -> float:
    if not np.isfinite(elapsed_s) or elapsed_s <= 0:
        return math.nan
    return distance_m / elapsed_s


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current != previous + 1:
            runs.append((start, previous + 1))
            start = current
        previous = current
    runs.append((start, previous + 1))
    return runs


def _quality_flags(value: Any) -> list[str]:
    return [
        item
        for item in str(value or "").split(";")
        if item and item.lower() != "nan"
    ]


def _set_quality_flags(frame: pd.DataFrame, row_index: int, flags: list[str]) -> None:
    unique = list(dict.fromkeys(flags))
    frame.loc[row_index, "quality_flags"] = ";".join(unique)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _boolean(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"Expected boolean telemetry-cleanup value, got {value!r}")


def _positive_float(value: Any, default: float) -> float:
    number = default if value is None else float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Expected a positive finite value, got {value!r}")
    return number


def _positive_int(value: Any, default: int) -> int:
    number = default if value is None else int(value)
    if number <= 0:
        raise ValueError(f"Expected a positive integer, got {value!r}")
    return number


def _fraction(value: Any, default: float) -> float:
    number = default if value is None else float(value)
    if not math.isfinite(number) or not 0 < number <= 1:
        raise ValueError(f"Expected a fraction in (0, 1], got {value!r}")
    return number
