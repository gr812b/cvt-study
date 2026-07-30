"""Conservative detection and selection of genuine closed-course route variants.

A route variant is a repeatedly observed spatial path, not a long/short-lap
heuristic and not a generic centreline outlier.  Detection occurs before the
single-route consensus so supported alternatives cannot be averaged together.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


_MAP_ONLY_FLAGS = {
    "large_backward_map_match",
    "p95_map_error_exceeds_limit",
    "no_map_matched_points",
    "no_finite_map_errors",
    "consensus_geometry_outlier",
}


@dataclass(frozen=True, slots=True)
class RouteVariantSettings:
    enabled: bool = False
    minimum_supported_laps: int = 2
    sample_spacing_m: float = 5.0
    same_variant_p95_distance_m: float = 10.0
    divergence_distance_m: float = 15.0
    maximum_divergent_fraction: float = 0.03
    maximum_length_relative_difference: float = 0.08
    maximum_within_variant_length_deviation_fraction: float = 0.15
    selection: str = "require_explicit"
    reference_run_id: str = ""
    selected_variant_id: str = ""

    @classmethod
    def from_mapping(cls, track: Mapping[str, Any]) -> "RouteVariantSettings":
        raw = track.get("route_variants", {})
        raw = raw if isinstance(raw, Mapping) else {}
        settings = cls(
            enabled=bool(raw.get("enabled", False)),
            minimum_supported_laps=int(raw.get("minimum_supported_laps", 2)),
            sample_spacing_m=float(raw.get("sample_spacing_m", 5.0)),
            same_variant_p95_distance_m=float(
                raw.get("same_variant_p95_distance_m", 10.0)
            ),
            divergence_distance_m=float(raw.get("divergence_distance_m", 15.0)),
            maximum_divergent_fraction=float(
                raw.get("maximum_divergent_fraction", 0.03)
            ),
            maximum_length_relative_difference=float(
                raw.get("maximum_length_relative_difference", 0.08)
            ),
            maximum_within_variant_length_deviation_fraction=float(
                raw.get("maximum_within_variant_length_deviation_fraction", 0.15)
            ),
            selection=str(raw.get("selection", "require_explicit")).strip().lower(),
            reference_run_id=str(raw.get("reference_run_id", "")).strip(),
            selected_variant_id=str(raw.get("selected_variant_id", "")).strip(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.minimum_supported_laps < 2:
            raise ValueError(
                "track.route_variants.minimum_supported_laps must be at least 2"
            )
        for name, value in (
            ("sample_spacing_m", self.sample_spacing_m),
            ("same_variant_p95_distance_m", self.same_variant_p95_distance_m),
            ("divergence_distance_m", self.divergence_distance_m),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"track.route_variants.{name} must be positive and finite")
        for name, value in (
            ("maximum_divergent_fraction", self.maximum_divergent_fraction),
            (
                "maximum_length_relative_difference",
                self.maximum_length_relative_difference,
            ),
            (
                "maximum_within_variant_length_deviation_fraction",
                self.maximum_within_variant_length_deviation_fraction,
            ),
        ):
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"track.route_variants.{name} must be in [0, 1)")
        allowed = {
            "require_explicit",
            "reference_run",
            "largest_supported",
            "longest_supported",
            "variant_id",
        }
        if self.selection not in allowed:
            raise ValueError(
                "track.route_variants.selection must be one of "
                + ", ".join(sorted(allowed))
            )
        if self.selection == "reference_run" and not self.reference_run_id:
            raise ValueError(
                "track.route_variants.reference_run_id is required when "
                "selection='reference_run'"
            )
        if self.selection == "variant_id" and not self.selected_variant_id:
            raise ValueError(
                "track.route_variants.selected_variant_id is required when "
                "selection='variant_id'"
            )


@dataclass(frozen=True, slots=True)
class RouteVariantDetection:
    laps: pd.DataFrame
    summary: pd.DataFrame
    pairwise: pd.DataFrame
    settings: RouteVariantSettings
    selected_variant_id: str


def detect_laps_route_aware(
    points: pd.DataFrame,
    ingestion_results: tuple[Any, ...],
    frame: Any,
    gate_latitude_deg: float,
    gate_longitude_deg: float,
    settings: Any,
    diagnostics: Any,
) -> pd.DataFrame:
    """Detect laps without declaring a genuine route length an invalid lap.

    This mirrors the canonical detector, but the global median-distance test is
    deliberately deferred until after spatial route variants have been found.
    """

    from .laps import _gate_crossings

    run_metadata = {result.metadata.run_id: result.metadata for result in ingestion_results}
    gate_x, gate_y = frame.to_xy([gate_latitude_deg], [gate_longitude_deg])
    rows: list[dict[str, Any]] = []
    global_lap_id = 0
    for (run_id, track_index, segment_index), indices in points.groupby(
        ["run_id", "track_index", "segment_index"], sort=False
    ).groups.items():
        idx = list(indices)
        segment = points.loc[idx].copy()
        distance = np.hypot(segment["x_m"] - gate_x[0], segment["y_m"] - gate_y[0])
        crossings = _gate_crossings(
            segment, distance.to_numpy(dtype=float, copy=True), settings
        )
        if len(crossings) < 2:
            diagnostics.warning(
                "NO_COMPLETE_LAPS_IN_SEGMENT",
                f"Run {run_id}, segment {track_index}:{segment_index} produced fewer "
                "than two lap-gate visits.",
                path=f"runs.{run_id}",
            )
            continue
        for local_lap, (start_pos, end_pos) in enumerate(
            zip(crossings[:-1], crossings[1:]), start=1
        ):
            lap_segment = segment.iloc[start_pos : end_pos + 1]
            times = pd.to_datetime(
                lap_segment["timestamp_utc"], utc=True, errors="coerce"
            )
            duration = (
                float((times.iloc[-1] - times.iloc[0]).total_seconds())
                if times.notna().all()
                else math.nan
            )
            global_lap_id += 1
            dt = lap_segment["time_step_s"].to_numpy(dtype=float, copy=True)
            distance_m = float(lap_segment["step_distance_m"].fillna(0.0).sum())
            speed_values = pd.to_numeric(
                lap_segment["speed_analysis_mps"], errors="coerce"
            )
            speed_coverage_fraction = float(speed_values.notna().mean())
            stationary_fraction = (
                float((speed_values.dropna() < settings.stationary_speed_mps).mean())
                if speed_values.notna().any()
                else 1.0
            )
            metadata = run_metadata[str(run_id)]
            rows.append(
                {
                    "lap_id": global_lap_id,
                    "run_id": str(run_id),
                    "vehicle_id": metadata.vehicle_id,
                    "driver_id": metadata.driver_id,
                    "track_index": int(track_index),
                    "segment_index": int(segment_index),
                    "local_lap_id": local_lap,
                    "start_global_index": int(lap_segment.index[0]),
                    "end_global_index": int(lap_segment.index[-1]),
                    "start_time_utc": times.iloc[0],
                    "end_time_utc": times.iloc[-1],
                    "duration_s": duration,
                    "path_distance_m": distance_m,
                    "stationary_fraction": stationary_fraction,
                    "speed_coverage_fraction": speed_coverage_fraction,
                    "time_gap_count": int(
                        np.sum(
                            np.isfinite(dt)
                            & (dt > settings.maximum_normal_time_step_s)
                        )
                    ),
                    "timestamp_regression_count": int(
                        np.sum(np.isfinite(dt) & (dt < 0))
                    ),
                    "median_speed_mps": float(
                        lap_segment["speed_analysis_mps"].median()
                    ),
                    "maximum_speed_mps": float(
                        lap_segment["speed_analysis_mps"].max()
                    ),
                    "use_for_centreline": metadata.use_for_centreline,
                    "use_for_gate_evidence": metadata.use_for_gate_evidence,
                }
            )
    laps = pd.DataFrame(rows)
    if laps.empty:
        raise ValueError("No complete laps were found between lap-gate visits.")

    median_distance = float(laps["path_distance_m"].median())
    laps["distance_ratio_to_median"] = laps["path_distance_m"] / median_distance
    laps["data_quality_valid"] = _data_quality_mask(laps, settings)
    laps["analysis_valid"] = laps["data_quality_valid"]
    laps["quality_flags"] = laps.apply(
        lambda row: _data_quality_flags(row, settings), axis=1
    )
    laps["reference_lap"] = False
    laps["pre_consensus_valid"] = laps["analysis_valid"].astype(bool)
    laps["centreline_included"] = (
        laps["analysis_valid"] & laps["use_for_centreline"]
    )
    laps["consensus_excluded"] = False
    laps["consensus_iteration_excluded"] = np.nan
    laps["consensus_exclusion_reason"] = ""
    if not (laps["data_quality_valid"] & laps["use_for_centreline"]).any():
        raise ValueError(
            "No data-quality-valid lap is enabled for route-variant detection."
        )
    return laps


def detect_and_select_route_variants(
    points: pd.DataFrame,
    laps: pd.DataFrame,
    track_config: Mapping[str, Any],
    diagnostics: Any,
) -> RouteVariantDetection:
    settings = RouteVariantSettings.from_mapping(track_config)
    output = laps.copy()
    if "data_quality_valid" not in output:
        # Legacy detector compatibility. Its analysis_valid includes a global
        # distance check, so reconstruct the data-quality portion explicitly.
        reconstruction = track_config.get("reconstruction", {})
        reconstruction = reconstruction if isinstance(reconstruction, Mapping) else {}
        proxy = type(
            "SettingsProxy",
            (),
            {
                "stationary_speed_mps": float(
                    reconstruction.get("stationary_speed_mps", 0.8)
                ),
                "minimum_speed_coverage_fraction": float(
                    reconstruction.get("minimum_speed_coverage_fraction", 0.80)
                ),
                "maximum_normal_time_step_s": float(
                    reconstruction.get("maximum_normal_time_step_s", 3.0)
                ),
            },
        )()
        output["data_quality_valid"] = _data_quality_mask(output, proxy)

    if not settings.enabled:
        output["route_variant_id"] = "route_001"
        output["route_variant_supported"] = True
        output["route_variant_selected"] = True
        output["route_variant_status"] = "single_route_detection_disabled"
        output["route_variant_distance_ratio"] = output["distance_ratio_to_median"]
        output["analysis_exclusion_reason"] = np.where(
            output["data_quality_valid"], "", "data_quality_rejected"
        )
        summary = _disabled_summary(output)
        return RouteVariantDetection(
            output, summary, pd.DataFrame(), settings, "route_001"
        )

    candidate_rows = output[output["data_quality_valid"].astype(bool)].copy()
    if candidate_rows.empty:
        raise ValueError("Route-variant detection has no data-quality-valid laps.")

    representations: dict[int, np.ndarray] = {}
    geometric_lengths: dict[int, float] = {}
    unusable: set[int] = set()
    for row in candidate_rows.itertuples(index=False):
        lap_id = int(row.lap_id)
        representation, length = _lap_representation(points, row, settings)
        if representation is None:
            unusable.add(lap_id)
            continue
        representations[lap_id] = representation
        geometric_lengths[lap_id] = length

    usable_ids = sorted(representations)
    if not usable_ids:
        raise ValueError(
            "Route-variant detection found no laps with sufficient finite geometry."
        )

    pairwise = _pairwise_table(
        output, usable_ids, representations, geometric_lengths, settings
    )
    clusters = _complete_link_clusters(usable_ids, pairwise)
    cluster_records = _cluster_records(output, clusters, geometric_lengths, settings)
    cluster_records.sort(
        key=lambda row: (
            -float(row["median_path_distance_m"]),
            -int(row["lap_count"]),
            int(row["minimum_lap_id"]),
        )
    )
    for index, record in enumerate(cluster_records, 1):
        record["route_variant_id"] = f"route_{index:03d}"

    selected_variant_id, selection_reason = _select_variant(
        output, cluster_records, settings, diagnostics
    )
    cluster_by_lap: dict[int, dict[str, Any]] = {}
    for record in cluster_records:
        for lap_id in record["lap_ids"]:
            cluster_by_lap[int(lap_id)] = record

    output["route_variant_id"] = ""
    output["route_variant_supported"] = False
    output["route_variant_selected"] = False
    output["route_variant_status"] = "data_quality_rejected"
    output["route_variant_distance_ratio"] = np.nan
    output["analysis_exclusion_reason"] = "data_quality_rejected"

    for row_index, row in output.iterrows():
        lap_id = int(row["lap_id"])
        if not bool(row["data_quality_valid"]):
            continue
        if lap_id in unusable or lap_id not in cluster_by_lap:
            output.loc[row_index, "route_variant_status"] = "unclassifiable_geometry"
            output.loc[row_index, "analysis_exclusion_reason"] = (
                "route_variant_unclassifiable_geometry"
            )
            continue
        record = cluster_by_lap[lap_id]
        variant_id = str(record["route_variant_id"])
        supported = bool(record["supported"])
        selected = variant_id == selected_variant_id
        median_length = float(record["median_path_distance_m"])
        ratio = float(row["path_distance_m"]) / max(median_length, 1e-12)
        within = abs(ratio - 1.0) <= settings.maximum_within_variant_length_deviation_fraction
        output.loc[row_index, "route_variant_id"] = variant_id
        output.loc[row_index, "route_variant_supported"] = supported
        output.loc[row_index, "route_variant_selected"] = selected
        output.loc[row_index, "route_variant_distance_ratio"] = ratio
        if not supported:
            status = "insufficient_repeated_support"
            exclusion = "unsupported_route_variant"
        elif not selected:
            status = "alternate_supported_route"
            exclusion = "alternate_supported_route_variant"
        elif not within:
            status = "selected_route_length_outlier"
            exclusion = "within_variant_distance_outlier"
        else:
            status = "selected_supported_route"
            exclusion = ""
        output.loc[row_index, "route_variant_status"] = status
        output.loc[row_index, "analysis_exclusion_reason"] = exclusion

    output["pre_consensus_valid"] = (
        output["data_quality_valid"].astype(bool)
        & output["route_variant_selected"].astype(bool)
        & output["route_variant_supported"].astype(bool)
        & (
            (output["route_variant_distance_ratio"] - 1.0).abs()
            <= settings.maximum_within_variant_length_deviation_fraction
        )
    )
    output["analysis_valid"] = output["pre_consensus_valid"]
    output["centreline_included"] = (
        output["pre_consensus_valid"] & output["use_for_centreline"].astype(bool)
    )
    if not output["centreline_included"].any():
        raise ValueError(
            f"Selected route variant {selected_variant_id!r} has no data-quality-valid "
            "lap enabled for centreline construction."
        )

    summary = pd.DataFrame(
        [
            {
                **{key: value for key, value in record.items() if key != "lap_ids"},
                "lap_ids": ";".join(map(str, record["lap_ids"])),
                "run_ids": ";".join(record["run_ids"]),
                "vehicle_ids": ";".join(record["vehicle_ids"]),
                "driver_ids": ";".join(record["driver_ids"]),
                "selected": record["route_variant_id"] == selected_variant_id,
                "selection_reason": (
                    selection_reason
                    if record["route_variant_id"] == selected_variant_id
                    else ""
                ),
            }
            for record in cluster_records
        ]
    )
    diagnostics.info(
        "ROUTE_VARIANTS_DETECTED",
        (
            f"Detected {len(cluster_records)} geometric route cluster(s), including "
            f"{int(summary['supported'].sum())} with at least "
            f"{settings.minimum_supported_laps} repeated lap(s). Selected "
            f"{selected_variant_id} for the nominal track build."
        ),
        hint=(
            "Alternate supported routes remain in lap_quality.csv and the route "
            "variant audit; they are not averaged into the selected centreline."
        ),
    )
    return RouteVariantDetection(
        output, summary, pairwise, settings, selected_variant_id
    )


def finalize_route_variant_semantics(laps: pd.DataFrame) -> pd.DataFrame:
    """Keep alternate valid routes from being mislabeled as bad nominal map matches."""

    if "route_variant_selected" not in laps:
        return laps
    output = laps.copy()
    alternate = (
        output.get("data_quality_valid", False).astype(bool)
        & output.get("route_variant_supported", False).astype(bool)
        & ~output["route_variant_selected"].astype(bool)
    )
    unsupported = (
        output.get("data_quality_valid", False).astype(bool)
        & ~output.get("route_variant_supported", False).astype(bool)
    )
    for index in output.index[alternate | unsupported]:
        output.loc[index, "analysis_valid"] = False
        output.loc[index, "centreline_included"] = False
        output.loc[index, "consensus_excluded"] = False
        output.loc[index, "consensus_exclusion_reason"] = ""
        output.loc[index, "quality_flags"] = _remove_flags(
            output.loc[index, "quality_flags"], _MAP_ONLY_FLAGS
        )
    output.loc[alternate, "analysis_exclusion_reason"] = (
        "alternate_supported_route_variant"
    )
    output.loc[unsupported, "analysis_exclusion_reason"] = "unsupported_route_variant"
    output["nominal_route_map_match_applicable"] = output[
        "route_variant_selected"
    ].astype(bool)
    return output


def _data_quality_mask(laps: pd.DataFrame, settings: Any) -> pd.Series:
    return (
        pd.to_numeric(laps["duration_s"], errors="coerce").notna()
        & (
            pd.to_numeric(laps["stationary_fraction"], errors="coerce")
            <= 0.15
        )
        & (
            pd.to_numeric(laps["speed_coverage_fraction"], errors="coerce")
            >= float(settings.minimum_speed_coverage_fraction)
        )
        & (pd.to_numeric(laps["time_gap_count"], errors="coerce").fillna(1) == 0)
        & (
            pd.to_numeric(laps["timestamp_regression_count"], errors="coerce")
            .fillna(1)
            == 0
        )
        & (pd.to_numeric(laps["path_distance_m"], errors="coerce") > 0.0)
    )


def _data_quality_flags(lap: pd.Series, settings: Any) -> str:
    flags: list[str] = []
    if not np.isfinite(float(lap["duration_s"])):
        flags.append("missing_or_invalid_duration")
    if float(lap["stationary_fraction"]) > 0.15:
        flags.append("excessive_stationary_fraction")
    if float(lap["speed_coverage_fraction"]) < float(
        settings.minimum_speed_coverage_fraction
    ):
        flags.append("insufficient_speed_coverage")
    if int(lap["time_gap_count"]) > 0:
        flags.append("sampling_gap")
    if int(lap["timestamp_regression_count"]) > 0:
        flags.append("timestamp_regression")
    return ";".join(flags)


def _lap_representation(
    points: pd.DataFrame, lap: Any, settings: RouteVariantSettings
) -> tuple[np.ndarray | None, float]:
    start = int(lap.start_global_index)
    end = int(lap.end_global_index)
    segment = points.loc[start:end, ["x_m", "y_m"]].apply(
        pd.to_numeric, errors="coerce"
    )
    segment = segment.dropna().to_numpy(float)
    if len(segment) < 4:
        return None, math.nan
    keep = np.ones(len(segment), dtype=bool)
    keep[1:] = np.hypot(
        np.diff(segment[:, 0]), np.diff(segment[:, 1])
    ) > 1.0e-6
    segment = segment[keep]
    if len(segment) < 4:
        return None, math.nan
    increments = np.hypot(np.diff(segment[:, 0]), np.diff(segment[:, 1]))
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    length = float(cumulative[-1])
    if not math.isfinite(length) or length <= 3.0 * settings.sample_spacing_m:
        return None, length
    targets = np.arange(0.0, length, settings.sample_spacing_m)
    if len(targets) == 0 or targets[-1] < length:
        targets = np.append(targets, length)
    representation = np.column_stack(
        (
            np.interp(targets, cumulative, segment[:, 0]),
            np.interp(targets, cumulative, segment[:, 1]),
        )
    )
    return representation, length


def _pairwise_table(
    laps: pd.DataFrame,
    lap_ids: list[int],
    representations: Mapping[int, np.ndarray],
    lengths: Mapping[int, float],
    settings: RouteVariantSettings,
) -> pd.DataFrame:
    lookup = laps.set_index("lap_id")
    trees = {lap_id: cKDTree(representations[lap_id]) for lap_id in lap_ids}
    records: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(lap_ids):
        for right_id in lap_ids[left_index + 1 :]:
            left_to_right = trees[right_id].query(
                representations[left_id], k=1
            )[0]
            right_to_left = trees[left_id].query(
                representations[right_id], k=1
            )[0]
            p95 = max(
                float(np.quantile(left_to_right, 0.95)),
                float(np.quantile(right_to_left, 0.95)),
            )
            divergent_fraction = max(
                float(np.mean(left_to_right > settings.divergence_distance_m)),
                float(np.mean(right_to_left > settings.divergence_distance_m)),
            )
            length_relative = abs(lengths[left_id] - lengths[right_id]) / max(
                lengths[left_id], lengths[right_id], 1.0e-12
            )
            score = max(
                p95 / settings.same_variant_p95_distance_m,
                divergent_fraction
                / max(settings.maximum_divergent_fraction, 1.0e-12),
                length_relative
                / max(settings.maximum_length_relative_difference, 1.0e-12),
            )
            records.append(
                {
                    "left_lap_id": left_id,
                    "right_lap_id": right_id,
                    "left_run_id": str(lookup.loc[left_id, "run_id"]),
                    "right_run_id": str(lookup.loc[right_id, "run_id"]),
                    "symmetric_p95_nearest_path_distance_m": p95,
                    "symmetric_divergent_fraction": divergent_fraction,
                    "length_relative_difference": length_relative,
                    "complete_link_score": score,
                    "same_route_compatible": bool(score <= 1.0),
                }
            )
    return pd.DataFrame(records)


def _complete_link_clusters(lap_ids: list[int], pairwise: pd.DataFrame) -> list[set[int]]:
    scores: dict[tuple[int, int], float] = {}
    for row in pairwise.itertuples(index=False):
        key = tuple(sorted((int(row.left_lap_id), int(row.right_lap_id))))
        scores[key] = float(row.complete_link_score)
    clusters = [{lap_id} for lap_id in lap_ids]
    while True:
        candidates: list[tuple[float, tuple[int, ...], int, int]] = []
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                score = max(
                    scores[tuple(sorted((left, right)))]
                    for left in clusters[left_index]
                    for right in clusters[right_index]
                )
                if score <= 1.0:
                    identity = tuple(sorted(clusters[left_index] | clusters[right_index]))
                    candidates.append((score, identity, left_index, right_index))
        if not candidates:
            break
        _, _, left_index, right_index = min(candidates)
        merged = clusters[left_index] | clusters[right_index]
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left_index, right_index}
        ]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: min(cluster))
    return clusters


def _cluster_records(
    laps: pd.DataFrame,
    clusters: list[set[int]],
    lengths: Mapping[int, float],
    settings: RouteVariantSettings,
) -> list[dict[str, Any]]:
    lookup = laps.set_index("lap_id")
    records: list[dict[str, Any]] = []
    for cluster in clusters:
        lap_ids = sorted(cluster)
        rows = lookup.loc[lap_ids]
        values = np.asarray([lengths[lap_id] for lap_id in lap_ids], dtype=float)
        records.append(
            {
                "lap_ids": lap_ids,
                "minimum_lap_id": min(lap_ids),
                "lap_count": len(lap_ids),
                "supported": len(lap_ids) >= settings.minimum_supported_laps,
                "run_count": int(rows["run_id"].astype(str).nunique()),
                "vehicle_count": int(rows["vehicle_id"].astype(str).nunique()),
                "driver_count": int(rows["driver_id"].astype(str).nunique()),
                "run_ids": sorted(rows["run_id"].astype(str).unique()),
                "vehicle_ids": sorted(rows["vehicle_id"].astype(str).unique()),
                "driver_ids": sorted(rows["driver_id"].astype(str).unique()),
                "median_path_distance_m": float(np.median(values)),
                "p10_path_distance_m": float(np.quantile(values, 0.10)),
                "p90_path_distance_m": float(np.quantile(values, 0.90)),
            }
        )
    return records


def _select_variant(
    laps: pd.DataFrame,
    records: list[dict[str, Any]],
    settings: RouteVariantSettings,
    diagnostics: Any,
) -> tuple[str, str]:
    supported = [record for record in records if bool(record["supported"])]
    if len(supported) == 1:
        return str(supported[0]["route_variant_id"]), "only_supported_route_variant"
    if not supported:
        if len(records) == 1:
            diagnostics.warning(
                "ROUTE_VARIANT_SUPPORT_BELOW_MINIMUM",
                "Only one route cluster was observed, but it has fewer repeated laps "
                "than the configured support threshold. It is selected provisionally.",
            )
            return str(records[0]["route_variant_id"]), "single_provisional_cluster"
        raise ValueError(
            "Route-variant detection found multiple clusters but none has the "
            "configured repeated-lap support. Review route_variant audit thresholds."
        )

    if settings.selection == "require_explicit":
        description = ", ".join(
            f"{record['route_variant_id']} ({record['lap_count']} laps, "
            f"median {record['median_path_distance_m']:.1f} m)"
            for record in supported
        )
        raise ValueError(
            "Multiple supported route variants were detected: "
            + description
            + ". Set track.route_variants.selection to reference_run, variant_id, "
            "largest_supported, or longest_supported."
        )
    if settings.selection == "reference_run":
        matches = [
            record
            for record in supported
            if settings.reference_run_id in record["run_ids"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"reference_run_id={settings.reference_run_id!r} appears in "
                f"{len(matches)} supported route variants; expected exactly one."
            )
        return str(matches[0]["route_variant_id"]), (
            f"supported_variant_containing_reference_run:{settings.reference_run_id}"
        )
    if settings.selection == "variant_id":
        matches = [
            record
            for record in supported
            if record["route_variant_id"] == settings.selected_variant_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"selected_variant_id={settings.selected_variant_id!r} is not a "
                "supported route variant in this build."
            )
        return settings.selected_variant_id, "explicit_variant_id"
    if settings.selection == "largest_supported":
        maximum = max(int(record["lap_count"]) for record in supported)
        matches = [record for record in supported if int(record["lap_count"]) == maximum]
        if len(matches) != 1:
            raise ValueError(
                "largest_supported route selection is tied; use reference_run or variant_id."
            )
        diagnostics.warning(
            "ROUTE_VARIANT_AUTOMATIC_SELECTION",
            "Multiple genuine route variants were detected; the most-supported one "
            "was selected by explicit project policy.",
        )
        return str(matches[0]["route_variant_id"]), "largest_supported_policy"
    if settings.selection == "longest_supported":
        maximum = max(float(record["median_path_distance_m"]) for record in supported)
        matches = [
            record
            for record in supported
            if math.isclose(
                float(record["median_path_distance_m"]), maximum, abs_tol=1.0e-9
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                "longest_supported route selection is tied; use reference_run or variant_id."
            )
        diagnostics.warning(
            "ROUTE_VARIANT_AUTOMATIC_SELECTION",
            "Multiple genuine route variants were detected; the longest supported "
            "one was selected by explicit project policy.",
        )
        return str(matches[0]["route_variant_id"]), "longest_supported_policy"
    raise AssertionError(settings.selection)


def _disabled_summary(laps: pd.DataFrame) -> pd.DataFrame:
    valid = laps[laps["data_quality_valid"].astype(bool)]
    distances = pd.to_numeric(valid["path_distance_m"], errors="coerce").dropna()
    return pd.DataFrame(
        [
            {
                "route_variant_id": "route_001",
                "lap_count": int(len(valid)),
                "supported": True,
                "selected": True,
                "run_count": int(valid["run_id"].astype(str).nunique()),
                "vehicle_count": int(valid["vehicle_id"].astype(str).nunique()),
                "driver_count": int(valid["driver_id"].astype(str).nunique()),
                "median_path_distance_m": (
                    float(distances.median()) if len(distances) else math.nan
                ),
                "p10_path_distance_m": (
                    float(distances.quantile(0.10)) if len(distances) else math.nan
                ),
                "p90_path_distance_m": (
                    float(distances.quantile(0.90)) if len(distances) else math.nan
                ),
                "lap_ids": ";".join(map(str, valid["lap_id"].astype(int))),
                "run_ids": ";".join(sorted(valid["run_id"].astype(str).unique())),
                "vehicle_ids": ";".join(
                    sorted(valid["vehicle_id"].astype(str).unique())
                ),
                "driver_ids": ";".join(
                    sorted(valid["driver_id"].astype(str).unique())
                ),
                "selection_reason": "route_variant_detection_disabled",
            }
        ]
    )


def _remove_flags(value: Any, removed: set[str]) -> str:
    flags = [
        item.strip()
        for item in str(value or "").split(";")
        if item.strip() and item.strip() not in removed
    ]
    return ";".join(dict.fromkeys(flags))
