"""Iterative robust centreline consensus and leave-one-out screening."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cvt_track_study.config.diagnostics import DiagnosticBag
from cvt_track_study.gpx.cleanup import (
    remove_isolated_map_outliers,
)

from .geo import Centreline, LocalFrame
from .laps import (
    _append_lap_quality_flag,
    _circular_smooth,
    _finite_column_stat,
    _interp_with_nan,
    _lap_segment,
    _resample_lap_for_consensus,
    _uniform_centreline,
    build_centreline,
    centreline_distance_summary,
    map_match_laps,
    map_segment_to_centreline,
)
from .settings import ReconstructionSettings


_SOURCE_KEY_COLUMNS = (
    "run_id",
    "track_index",
    "segment_index",
    "point_index",
)


@dataclass(frozen=True)
class ConsensusBuildResult:
    centreline: Centreline
    laps: pd.DataFrame
    matched_points: pd.DataFrame
    rejected_map_points: pd.DataFrame
    iteration_count: int
    converged: bool


def build_iterative_consensus(
    points: pd.DataFrame,
    laps: pd.DataFrame,
    frame: LocalFrame,
    settings: ReconstructionSettings,
    track_config: Mapping[str, Any],
    diagnostics: DiagnosticBag,
) -> ConsensusBuildResult:
    """Build, screen, and rebuild the shared track geometry."""

    state = _initialize_state(laps)
    active = set(
        state.loc[
            state["pre_consensus_valid"]
            & state["use_for_centreline"],
            "lap_id",
        ].astype(int)
    )
    if not active:
        raise ValueError(
            "No provisionally valid laps are enabled for "
            "centreline consensus."
        )

    active_rows = state[state["lap_id"].astype(int).isin(active)]
    source_count = int(active_rows["vehicle_id"].astype(str).nunique())
    if source_count >= 2:
        diagnostics.info(
            "CENTRELINE_SOURCE_BALANCED",
            (
                f"Centreline consensus balances {source_count} measured vehicle/source "
                "consensuses equally. Repeated laps refine each source but do not give "
                "one vehicle proportional voting power over the physical course."
            ),
        )

    if len(active) < settings.consensus_minimum_laps:
        diagnostics.warning(
            "CENTRELINE_CONSENSUS_SMALL_SAMPLE",
            (
                f"Only {len(active)} lap(s) are available for "
                "centreline consensus. A median centreline will "
                "still be built, but leave-one-out lap exclusion "
                f"requires at least "
                f"{settings.consensus_minimum_laps} laps."
            ),
        )

    working_points = points.copy()
    prior: Centreline | None = None
    rejected_batches: list[pd.DataFrame] = []
    converged = False
    final_metrics = pd.DataFrame()
    iteration_count = 0

    for iteration in range(
        1, settings.consensus_maximum_iterations + 1
    ):
        iteration_count = iteration
        centreline = build_centreline(
            working_points,
            state,
            frame,
            settings,
            lap_ids=active,
            alignment_centreline=prior,
        )
        prepared = _prepare_for_map_match(
            state, active
        )
        matched, scored = map_match_laps(
            working_points,
            prepared,
            centreline,
            settings,
        )

        matched, scored, rejected = (
            remove_isolated_map_outliers(
                matched,
                scored,
                track_config,
                diagnostics,
            )
        )
        if not rejected.empty:
            working_points, removed_count = (
                _drop_rejected_source_points(
                    working_points, rejected
                )
            )
            if removed_count:
                rejected_batches.append(rejected)
                prior = centreline
                diagnostics.info(
                    "CENTRELINE_CONSENSUS_REBUILD_AFTER_POINT_CLEANUP",
                    (
                        f"Consensus iteration {iteration} removed "
                        f"{removed_count} isolated map point(s); "
                        "the centreline will be rebuilt before "
                        "lap-level outlier screening."
                    ),
                )
                continue

        metrics = evaluate_leave_one_out(
            working_points,
            state,
            active,
            centreline,
            frame,
            settings,
        )
        final_metrics = metrics
        state = _merge_consensus_metrics(
            state, metrics
        )
        outliers = _select_outliers(
            metrics,
            active,
            settings,
        )

        if outliers:
            for lap_id in outliers:
                row_index = state.index[
                    state["lap_id"].astype(int)
                    == int(lap_id)
                ][0]
                state.loc[
                    row_index, "consensus_excluded"
                ] = True
                state.loc[
                    row_index,
                    "consensus_iteration_excluded",
                ] = iteration
                state.loc[
                    row_index,
                    "consensus_exclusion_reason",
                ] = "leave_one_out_geometry_outlier"
                _append_lap_quality_flag(
                    state,
                    row_index,
                    "consensus_geometry_outlier",
                )
            active -= set(outliers)
            diagnostics.warning(
                "CENTRELINE_CONSENSUS_LAPS_EXCLUDED",
                (
                    f"Consensus iteration {iteration} excluded "
                    f"{len(outliers)} clear geometry outlier "
                    f"lap(s): {', '.join(map(str, outliers))}. "
                    "The consensus will be rebuilt without them."
                ),
                hint=(
                    "Excluded laps remain in lap_quality.csv and "
                    "the full cleanup evidence; they are not "
                    "silently deleted."
                ),
            )
            prior = centreline
            continue

        if prior is not None:
            p95_shift, maximum_shift = (
                centreline_distance_summary(
                    prior, centreline
                )
            )
            state[
                "consensus_iteration_p95_shift_m"
            ] = p95_shift
            state[
                "consensus_iteration_maximum_shift_m"
            ] = maximum_shift
            if (
                maximum_shift
                <= settings.consensus_convergence_tolerance_m
            ):
                converged = True
                prior = centreline
                break
        prior = centreline

    if prior is None:
        raise ValueError(
            "Centreline consensus produced no geometry."
        )

    # One final build after all exclusions. This ensures the exported
    # centreline never contains a lap removed on the last iteration.
    final_centreline = build_centreline(
        working_points,
        state,
        frame,
        settings,
        lap_ids=active,
        alignment_centreline=prior,
    )
    prepared = _prepare_for_map_match(
        state, active
    )
    final_matched, final_laps = map_match_laps(
        working_points,
        prepared,
        final_centreline,
        settings,
    )
    final_matched, final_laps, final_rejected = (
        remove_isolated_map_outliers(
            final_matched,
            final_laps,
            track_config,
            diagnostics,
        )
    )

    if not final_rejected.empty:
        (
            working_points,
            removed_count,
        ) = _drop_rejected_source_points(
            working_points, final_rejected
        )
        if removed_count:
            rejected_batches.append(final_rejected)
            final_centreline = build_centreline(
                working_points,
                state,
                frame,
                settings,
                lap_ids=active,
                alignment_centreline=final_centreline,
            )
            prepared = _prepare_for_map_match(
                state, active
            )
            final_matched, final_laps = (
                map_match_laps(
                    working_points,
                    prepared,
                    final_centreline,
                    settings,
                )
            )

    final_metrics = evaluate_leave_one_out(
        working_points,
        state,
        active,
        final_centreline,
        frame,
        settings,
    )
    final_laps = _merge_consensus_metrics(
        final_laps, final_metrics
    )
    final_laps["centreline_included"] = (
        final_laps["lap_id"]
        .astype(int)
        .isin(active)
    )
    final_laps["consensus_excluded"] = (
        final_laps["lap_id"]
        .astype(int)
        .isin(
            set(
                state.loc[
                    state["consensus_excluded"],
                    "lap_id",
                ].astype(int)
            )
        )
    )
    final_laps = _copy_exclusion_state(
        final_laps, state
    )
    final_laps["analysis_valid"] = (
        final_laps["analysis_valid"]
        & ~final_laps["consensus_excluded"]
    )
    final_laps["consensus_iteration_count"] = (
        iteration_count
    )
    final_laps["reference_lap"] = False

    representative = final_laps[
        final_laps["centreline_included"]
        & final_laps["analysis_valid"]
    ]
    if representative.empty:
        representative = final_laps[
            final_laps["centreline_included"]
        ]
    if representative.empty:
        raise ValueError(
            "No retained lap remains to support the final consensus."
        )
    representative_index = (
        representative["p95_map_error_m"]
        .astype(float)
        .idxmin()
    )
    final_laps.loc[
        representative_index, "reference_lap"
    ] = True

    rejected_map_points = _concat_rejections(
        rejected_batches
    )
    excluded_count = int(
        final_laps["consensus_excluded"].sum()
    )
    diagnostics.info(
        "CENTRELINE_CONSENSUS_BUILT",
        (
            "Built the final centreline from "
            f"{len(active)} retained lap(s) after "
            f"{iteration_count} iteration(s); "
            f"{excluded_count} lap(s) were excluded by "
            "leave-one-out geometry screening."
        ),
        hint=(
            "reference_lap now identifies the retained lap "
            "closest to the final consensus; it does not define "
            "the centreline geometry."
        ),
    )
    if not converged:
        diagnostics.warning(
            "CENTRELINE_CONSENSUS_ITERATION_LIMIT",
            (
                "The centreline reached the configured iteration "
                "limit before satisfying the strict convergence "
                "tolerance. The final rebuilt consensus is still "
                "exported with its diagnostics."
            ),
        )

    return ConsensusBuildResult(
        centreline=final_centreline,
        laps=final_laps,
        matched_points=final_matched,
        rejected_map_points=rejected_map_points,
        iteration_count=iteration_count,
        converged=converged,
    )


def evaluate_leave_one_out(
    points: pd.DataFrame,
    laps: pd.DataFrame,
    active_lap_ids: set[int],
    full_centreline: Centreline,
    frame: LocalFrame,
    settings: ReconstructionSettings,
) -> pd.DataFrame:
    """Evaluate each lap against a centreline built from all other laps.

    The historical implementation called :func:`build_centreline` once per
    held-out lap.  With N laps that re-aligned essentially the same N-1 lap
    traces N times, turning leave-one-out screening into an O(N^2) geometry
    alignment step.  The alignment target is the same ``full_centreline`` for
    every held-out case, so each lap can be aligned and resampled once.  The
    pointwise median with one row removed is then exactly the same construction
    as the historical call.

    This is a computational refactor only: the smoothing, lap-gate anchor,
    uniform re-spacing, map matching, and screening metrics are unchanged.
    """
    rows: list[dict[str, Any]] = []
    if len(active_lap_ids) < 2:
        return pd.DataFrame(columns=_metric_columns())

    prepared = _prepare_leave_one_out_centreline_inputs(
        points,
        laps,
        active_lap_ids,
        full_centreline,
        settings,
    )

    for lap_id in sorted(active_lap_ids):
        leave_one_out = _build_leave_one_out_centreline(
            prepared,
            held_out_lap_id=lap_id,
            frame=frame,
            settings=settings,
        )
        lap = laps[
            laps["lap_id"].astype(int) == int(lap_id)
        ].iloc[0]
        segment = _lap_segment(points, lap)
        mapped = map_segment_to_centreline(
            segment, leave_one_out
        )
        errors = pd.to_numeric(
            mapped["map_error_m"], errors="coerce"
        ).to_numpy(float)
        finite = errors[np.isfinite(errors)]
        if not len(finite):
            continue
        p95_shift, maximum_shift = (
            centreline_distance_summary(
                full_centreline,
                leave_one_out,
            )
        )
        sustained_fraction = _uniform_sustained_fraction(
            mapped,
            leave_one_out,
            settings.consensus_sustained_error_threshold_m,
            settings.centreline_spacing_m,
        )
        matched_s = pd.to_numeric(
            mapped["s_m"], errors="coerce"
        ).to_numpy(float)
        rows.append(
            {
                "lap_id": int(lap_id),
                "loo_median_map_error_m": float(
                    np.median(finite)
                ),
                "loo_p95_map_error_m": float(
                    np.quantile(finite, 0.95)
                ),
                "loo_maximum_map_error_m": float(
                    np.max(finite)
                ),
                "loo_sustained_outlier_fraction": (
                    sustained_fraction
                ),
                "loo_large_backward_match_count": int(
                    np.sum(np.diff(matched_s) < -20.0)
                ),
                "loo_p95_centreline_shift_m": p95_shift,
                "loo_maximum_centreline_shift_m": (
                    maximum_shift
                ),
            }
        )
    return pd.DataFrame(rows).reindex(
        columns=_metric_columns()
    )


def _prepare_leave_one_out_centreline_inputs(
    points: pd.DataFrame,
    laps: pd.DataFrame,
    active_lap_ids: set[int],
    full_centreline: Centreline,
    settings: ReconstructionSettings,
) -> dict[str, Any]:
    """Align each active lap once to the common full-consensus centreline."""
    selected = laps[
        laps["lap_id"].astype(int).isin(active_lap_ids)
    ]
    records: list[tuple[int, dict[str, np.ndarray | float]]] = []
    for _, lap in selected.iterrows():
        record = _resample_lap_for_consensus(
            _lap_segment(points, lap),
            settings,
            full_centreline,
        )
        if record is not None:
            records.append((int(lap["lap_id"]), record))
    if len(records) < 2:
        raise ValueError(
            "Leave-one-out consensus requires at least two usable lap traces."
        )

    target_length = full_centreline.length_m
    node_count = max(
        4,
        int(math.ceil(target_length / settings.centreline_spacing_m)) + 1,
    )
    target_fraction = np.linspace(0.0, 1.0, node_count)

    lap_ids: list[int] = []
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    elevation_rows: list[np.ndarray] = []
    endpoint_x: list[float] = []
    endpoint_y: list[float] = []
    for lap_id, record in records:
        source_fraction = np.asarray(record["fraction"], dtype=float)
        x = np.asarray(record["x_m"], dtype=float)
        y = np.asarray(record["y_m"], dtype=float)
        elevation = np.asarray(record["elevation_m"], dtype=float)
        lap_ids.append(lap_id)
        x_rows.append(np.interp(target_fraction, source_fraction, x))
        y_rows.append(np.interp(target_fraction, source_fraction, y))
        elevation_rows.append(
            _interp_with_nan(
                target_fraction,
                source_fraction,
                elevation,
            )
        )
        endpoint_x.append(float(record["endpoint_x_m"]))
        endpoint_y.append(float(record["endpoint_y_m"]))

    return {
        "lap_ids": np.asarray(lap_ids, dtype=int),
        "x_rows": np.vstack(x_rows),
        "y_rows": np.vstack(y_rows),
        "elevation_rows": np.vstack(elevation_rows),
        "endpoint_x": np.asarray(endpoint_x, dtype=float),
        "endpoint_y": np.asarray(endpoint_y, dtype=float),
    }


def _build_leave_one_out_centreline(
    prepared: Mapping[str, Any],
    *,
    held_out_lap_id: int,
    frame: LocalFrame,
    settings: ReconstructionSettings,
) -> Centreline:
    """Reproduce ``build_centreline(..., lap_ids=others)`` from cached rows."""
    lap_ids = np.asarray(prepared["lap_ids"], dtype=int)
    keep = lap_ids != int(held_out_lap_id)
    if not np.any(keep):
        raise ValueError(
            "Leave-one-out consensus cannot remove its only usable lap."
        )

    x_rows = np.asarray(prepared["x_rows"], dtype=float)
    y_rows = np.asarray(prepared["y_rows"], dtype=float)
    elevation_rows = np.asarray(
        prepared["elevation_rows"], dtype=float
    )
    endpoint_x = np.asarray(prepared["endpoint_x"], dtype=float)
    endpoint_y = np.asarray(prepared["endpoint_y"], dtype=float)

    consensus_x = np.array(
        np.median(x_rows[keep], axis=0), dtype=float, copy=True
    )
    consensus_y = np.array(
        np.median(y_rows[keep], axis=0), dtype=float, copy=True
    )
    consensus_elevation = np.array(
        _finite_column_stat(elevation_rows[keep], "median"),
        dtype=float,
        copy=True,
    )
    consensus_x = _circular_smooth(
        consensus_x,
        settings.consensus_smoothing_window_nodes,
    )
    consensus_y = _circular_smooth(
        consensus_y,
        settings.consensus_smoothing_window_nodes,
    )

    gate_x = float(np.median(endpoint_x[keep]))
    gate_y = float(np.median(endpoint_y[keep]))
    consensus_x[0] = consensus_x[-1] = gate_x
    consensus_y[0] = consensus_y[-1] = gate_y
    if np.isfinite(consensus_elevation[[0, -1]]).any():
        gate_elevation = float(
            np.nanmedian(consensus_elevation[[0, -1]])
        )
        consensus_elevation[0] = gate_elevation
        consensus_elevation[-1] = gate_elevation

    return _uniform_centreline(
        consensus_x,
        consensus_y,
        consensus_elevation,
        frame,
        settings.centreline_spacing_m,
    )

def _select_outliers(
    metrics: pd.DataFrame,
    active_lap_ids: set[int],
    settings: ReconstructionSettings,
) -> list[int]:
    if (
        len(active_lap_ids)
        <= settings.consensus_minimum_laps
        or metrics.empty
    ):
        return []

    p95_threshold = _robust_upper_threshold(
        metrics["loo_p95_map_error_m"],
        settings.consensus_leave_one_out_p95_limit_m,
        settings.consensus_robust_mad_multiplier,
    )
    shift_threshold = _robust_upper_threshold(
        metrics["loo_maximum_centreline_shift_m"],
        settings.consensus_maximum_leave_one_out_shift_m,
        settings.consensus_robust_mad_multiplier,
    )

    candidates: list[tuple[float, int]] = []
    for _, row in metrics.iterrows():
        lap_id = int(row["lap_id"])
        p95 = float(row["loo_p95_map_error_m"])
        shift = float(
            row[
                "loo_maximum_centreline_shift_m"
            ]
        )
        sustained = float(
            row[
                "loo_sustained_outlier_fraction"
            ]
        )
        backward = int(
            row[
                "loo_large_backward_match_count"
            ]
        )

        strong_sustained = (
            sustained
            >= settings.consensus_strong_sustained_outlier_fraction
        )
        sustained_and_bad = (
            sustained
            >= settings.consensus_minimum_sustained_outlier_fraction
            and (
                p95 > p95_threshold
                or shift > shift_threshold
            )
        )
        backward_and_bad = (
            backward > 0
            and p95
            > settings.consensus_leave_one_out_p95_limit_m
        )
        if not (
            strong_sustained
            or sustained_and_bad
            or backward_and_bad
        ):
            continue

        severity = (
            p95 / max(p95_threshold, 1e-9)
            + shift / max(shift_threshold, 1e-9)
            + sustained
            / max(
                settings.consensus_minimum_sustained_outlier_fraction,
                1e-9,
            )
            + float(backward > 0)
        )
        candidates.append((severity, lap_id))

    if not candidates:
        return []

    maximum_by_fraction = max(
        1,
        int(
            math.floor(
                len(active_lap_ids)
                * settings.consensus_maximum_outlier_fraction_per_iteration
            )
        ),
    )
    maximum_preserving_minimum = max(
        0,
        len(active_lap_ids)
        - settings.consensus_minimum_laps,
    )
    maximum_remove = min(
        maximum_by_fraction,
        maximum_preserving_minimum,
    )
    if maximum_remove <= 0:
        return []

    candidates.sort(reverse=True)
    return [
        lap_id
        for _, lap_id in candidates[
            :maximum_remove
        ]
    ]


def _initialize_state(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    state = laps.copy()
    if "pre_consensus_valid" not in state:
        state["pre_consensus_valid"] = state[
            "analysis_valid"
        ].astype(bool)
    state["consensus_excluded"] = False
    state["consensus_iteration_excluded"] = np.nan
    state["consensus_exclusion_reason"] = ""
    state["centreline_included"] = (
        state["pre_consensus_valid"]
        & state["use_for_centreline"]
    )
    state["reference_lap"] = False
    return state


def _prepare_for_map_match(
    state: pd.DataFrame,
    active: set[int],
) -> pd.DataFrame:
    prepared = state.copy()
    prepared["centreline_included"] = (
        prepared["lap_id"]
        .astype(int)
        .isin(active)
    )
    prepared["analysis_valid"] = (
        prepared["pre_consensus_valid"]
        & ~prepared["consensus_excluded"]
    )
    prepared["reference_lap"] = False
    return prepared


def _merge_consensus_metrics(
    laps: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    metric_names = [
        column
        for column in _metric_columns()
        if column != "lap_id"
    ]
    output = laps.drop(
        columns=[
            column
            for column in metric_names
            if column in laps.columns
        ],
        errors="ignore",
    )
    if metrics.empty:
        for column in metric_names:
            output[column] = np.nan
        return output
    return output.merge(
        metrics, on="lap_id", how="left"
    )


def _copy_exclusion_state(
    target: pd.DataFrame,
    state: pd.DataFrame,
) -> pd.DataFrame:
    lookup = state.set_index("lap_id")
    output = target.copy()
    for column in (
        "consensus_excluded",
        "consensus_iteration_excluded",
        "consensus_exclusion_reason",
    ):
        if column in lookup:
            output[column] = output[
                "lap_id"
            ].map(lookup[column])

    # Preserve final map-quality flags and add any consensus-specific
    # flags from the iterative state. Neither audit trail overwrites
    # the other.
    state_flags = output["lap_id"].map(
        lookup["quality_flags"]
    )
    output["quality_flags"] = [
        _merge_quality_flags(final_value, state_value)
        for final_value, state_value in zip(
            output["quality_flags"], state_flags
        )
    ]
    return output


def _merge_quality_flags(
    *values: Any,
) -> str:
    flags: list[str] = []
    for value in values:
        for flag in str(value or "").split(";"):
            if (
                flag
                and flag.lower() != "nan"
                and flag not in flags
            ):
                flags.append(flag)
    return ";".join(flags)


def _drop_rejected_source_points(
    points: pd.DataFrame,
    rejected: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    if rejected.empty or not set(
        _SOURCE_KEY_COLUMNS
    ).issubset(rejected.columns):
        return points, 0

    keys = set(
        tuple(row)
        for row in rejected[
            list(_SOURCE_KEY_COLUMNS)
        ].itertuples(index=False, name=None)
    )
    point_keys = points[
        list(_SOURCE_KEY_COLUMNS)
    ].apply(tuple, axis=1)
    keep = ~point_keys.isin(keys)
    removed = int((~keep).sum())
    return points.loc[keep].copy(), removed


def _uniform_sustained_fraction(
    mapped: pd.DataFrame,
    centreline: Centreline,
    threshold_m: float,
    spacing_m: float,
) -> float:
    profile = mapped[
        ["s_m", "map_error_m"]
    ].dropna()
    if len(profile) < 3:
        return math.nan
    profile["s_bin"] = profile[
        "s_m"
    ].round(1)
    profile = (
        profile.groupby("s_bin", as_index=False)
        .agg(
            s_m=("s_m", "median"),
            map_error_m=(
                "map_error_m", "median"
            ),
        )
        .sort_values("s_m")
    )
    grid = np.arange(
        0.0,
        centreline.length_m,
        max(1.0, spacing_m),
    )
    if not len(grid):
        return math.nan
    errors = np.interp(
        grid,
        profile["s_m"].to_numpy(float),
        profile["map_error_m"].to_numpy(float),
    )
    return float(
        np.mean(errors > threshold_m)
    )


def _robust_upper_threshold(
    values: pd.Series,
    absolute_floor: float,
    mad_multiplier: float,
) -> float:
    finite = pd.to_numeric(
        values, errors="coerce"
    ).dropna().to_numpy(float)
    if not len(finite):
        return absolute_floor
    median = float(np.median(finite))
    mad = float(
        np.median(np.abs(finite - median))
    )
    robust = median + mad_multiplier * 1.4826 * mad
    return max(absolute_floor, robust)


def _concat_rejections(
    batches: list[pd.DataFrame],
) -> pd.DataFrame:
    usable = [
        frame
        for frame in batches
        if frame is not None and not frame.empty
    ]
    if not usable:
        return pd.DataFrame()
    combined = pd.concat(
        usable, ignore_index=True
    )
    dedupe = [
        column
        for column in (
            "rejection_stage",
            *_SOURCE_KEY_COLUMNS,
        )
        if column in combined.columns
    ]
    return combined.drop_duplicates(
        subset=dedupe
    ).reset_index(drop=True)


def _metric_columns() -> list[str]:
    return [
        "lap_id",
        "loo_median_map_error_m",
        "loo_p95_map_error_m",
        "loo_maximum_map_error_m",
        "loo_sustained_outlier_fraction",
        "loo_large_backward_match_count",
        "loo_p95_centreline_shift_m",
        "loo_maximum_centreline_shift_m",
    ]
