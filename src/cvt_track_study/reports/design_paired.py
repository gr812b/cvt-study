"""Paired design-effect contrasts across identical uncertainty worlds.

The design sweep is deliberately paired: every candidate is replayed in the same
world.  This module exploits that contract by differencing candidates *within*
each world before estimating uncertainty.  Shared world difficulty therefore
cancels from the contrast instead of inflating between-design uncertainty.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp


_WORLD_METADATA = (
    "scenario_seed",
    "base_draw_id",
    "track_pair_id",
    "track_case_id",
    "track_case_category",
    "source_uncertainty_replicate",
)


def paired_design_contrasts(
    rows: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    bootstrap_resamples: int | None = None,
    confidence_level: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return all pairwise contrasts, isolated adjacent contrasts, and world deltas.

    Time-saving convention:

    ``time_saved_s = lap_time(from) - lap_time(to)``

    so positive values mean the ``to`` candidate is faster.  Relative speedup is
    the same paired difference divided by the ``from`` candidate's lap time.
    Statistical summaries use only worlds where both candidates completed and
    both lap times are finite; completion discordance is reported separately.
    """

    if rows.empty or "design_id" not in rows or "replicate" not in rows:
        empty = pd.DataFrame()
        return empty, empty, empty

    bootstrap = int(
        bootstrap_resamples
        if bootstrap_resamples is not None
        else manifest.get("bootstrap_resamples", 2000)
    )
    bootstrap = max(200, bootstrap)
    seed = int(manifest.get("random_seed", 0)) ^ 0x504149524544

    design_ids = [str(value) for value in rows["design_id"].drop_duplicates()]
    if len(design_ids) < 2:
        empty = pd.DataFrame()
        return empty, empty, empty

    paths = _design_paths(rows, manifest)
    values = _candidate_values(rows, design_ids, paths)
    levels = _axis_levels(values, design_ids, paths)

    duplicate_counts = rows.groupby(["replicate", "design_id"], dropna=False).size()
    if (duplicate_counts > 1).any():
        raise ValueError(
            "Paired design inference requires one result row per design per replicate."
        )

    records: list[dict[str, Any]] = []
    world_records: list[dict[str, Any]] = []
    by_design = {
        design_id: rows[rows["design_id"].astype(str) == design_id].copy()
        for design_id in design_ids
    }

    for pair_index, (first, second) in enumerate(combinations(design_ids, 2)):
        from_id, to_id, changed = _orient_pair(
            first, second, values=values, paths=paths, design_ids=design_ids
        )
        from_group = by_design[from_id].copy()
        to_group = by_design[to_id].copy()
        merged = from_group.merge(
            to_group,
            on="replicate",
            suffixes=("_from", "_to"),
            how="inner",
            validate="one_to_one",
        )
        if merged.empty:
            continue

        from_lap = pd.to_numeric(merged.get("bounded_lap_time_s_from"), errors="coerce")
        to_lap = pd.to_numeric(merged.get("bounded_lap_time_s_to"), errors="coerce")
        from_completed = _bool_series(merged, "bounded_completed_from")
        to_completed = _bool_series(merged, "bounded_completed_to")
        finite = np.isfinite(from_lap.to_numpy(float)) & np.isfinite(to_lap.to_numpy(float))
        pairable = from_completed.to_numpy(bool) & to_completed.to_numpy(bool) & finite

        time_saved = from_lap.to_numpy(float) - to_lap.to_numpy(float)
        relative = np.divide(
            100.0 * time_saved,
            from_lap.to_numpy(float),
            out=np.full(len(merged), np.nan, dtype=float),
            where=np.isfinite(from_lap.to_numpy(float)) & (from_lap.to_numpy(float) > 0.0),
        )
        paired_time = time_saved[pairable]
        paired_relative = relative[pairable]

        rng = np.random.default_rng(seed + 104729 * (pair_index + 1))
        time_ci = _bootstrap_mean_interval(
            paired_time, rng, bootstrap, confidence_level
        )
        relative_ci = _bootstrap_mean_interval(
            paired_relative, rng, bootstrap, confidence_level
        )
        p_value = _paired_t_p_value(paired_time)
        sign_p = _sign_test_p_value(paired_time)

        changed_paths = _changed_paths(values.get(from_id, {}), values.get(to_id, {}), paths)
        changed_parameter = changed_paths[0] if len(changed_paths) == 1 else " + ".join(changed_paths)
        one_axis = len(changed_paths) == 1
        adjacent = (
            _is_adjacent(
                changed_parameter,
                values.get(from_id, {}).get(changed_parameter),
                values.get(to_id, {}).get(changed_parameter),
                levels,
            )
            if one_axis
            else False
        )
        context = {
            path: values.get(from_id, {}).get(path)
            for path in paths
            if path not in changed_paths
            and _values_equal(
                values.get(from_id, {}).get(path), values.get(to_id, {}).get(path)
            )
        }

        from_only = int(np.sum(from_completed.to_numpy(bool) & ~to_completed.to_numpy(bool)))
        to_only = int(np.sum(~from_completed.to_numpy(bool) & to_completed.to_numpy(bool)))
        both_failed = int(np.sum(~from_completed.to_numpy(bool) & ~to_completed.to_numpy(bool)))
        positive = int(np.sum(paired_time > 0.0))
        negative = int(np.sum(paired_time < 0.0))
        ties = int(np.sum(np.isclose(paired_time, 0.0, atol=1e-12, rtol=0.0)))

        record = {
            "from_design_id": from_id,
            "to_design_id": to_id,
            "changed_parameter_count": len(changed_paths),
            "changed_parameter": changed_parameter,
            "one_axis_change": one_axis,
            "adjacent_on_axis": adjacent,
            "from_value": values.get(from_id, {}).get(changed_parameter) if one_axis else math.nan,
            "to_value": values.get(to_id, {}).get(changed_parameter) if one_axis else math.nan,
            "fixed_context_json": json.dumps(context, sort_keys=True, separators=(",", ":"), default=str),
            "common_world_count": int(len(merged)),
            "both_completed_world_count": int(np.sum(pairable)),
            "pairable_world_fraction": float(np.mean(pairable)),
            "from_only_completed_world_count": from_only,
            "to_only_completed_world_count": to_only,
            "both_failed_world_count": both_failed,
            "completion_advantage_to_minus_from_worlds": to_only - from_only,
            "time_saved_mean_s": _mean(paired_time),
            "time_saved_median_s": _median(paired_time),
            "time_saved_std_s": _std(paired_time),
            "time_saved_ci_low_s": time_ci[0],
            "time_saved_ci_high_s": time_ci[1],
            "relative_speedup_mean_pct": _mean(paired_relative),
            "relative_speedup_median_pct": _median(paired_relative),
            "relative_speedup_ci_low_pct": relative_ci[0],
            "relative_speedup_ci_high_pct": relative_ci[1],
            "to_faster_world_fraction": (positive / len(paired_time) if len(paired_time) else math.nan),
            "to_slower_world_fraction": (negative / len(paired_time) if len(paired_time) else math.nan),
            "tie_world_fraction": (ties / len(paired_time) if len(paired_time) else math.nan),
            "paired_t_p_value": p_value,
            "sign_test_p_value": sign_p,
            "effect_direction_95pct": _effect_direction(time_ci),
        }
        records.append(record)

        for index, row in merged.iterrows():
            item: dict[str, Any] = {
                "replicate": int(row["replicate"]),
                "from_design_id": from_id,
                "to_design_id": to_id,
                "changed_parameter": changed_parameter,
                "one_axis_change": one_axis,
                "adjacent_on_axis": adjacent,
                "from_value": record["from_value"],
                "to_value": record["to_value"],
                "fixed_context_json": record["fixed_context_json"],
                "from_completed": bool(from_completed.loc[index]),
                "to_completed": bool(to_completed.loc[index]),
                "paired_complete": bool(pairable[list(merged.index).index(index)]),
                "from_lap_time_s": float(from_lap.loc[index]) if np.isfinite(from_lap.loc[index]) else math.nan,
                "to_lap_time_s": float(to_lap.loc[index]) if np.isfinite(to_lap.loc[index]) else math.nan,
                "time_saved_s": float(time_saved[list(merged.index).index(index)]) if pairable[list(merged.index).index(index)] else math.nan,
                "relative_speedup_pct": float(relative[list(merged.index).index(index)]) if pairable[list(merged.index).index(index)] else math.nan,
            }
            for name in _WORLD_METADATA:
                from_name = f"{name}_from"
                to_name = f"{name}_to"
                if from_name in merged:
                    item[name] = row[from_name]
                    if to_name in merged and not _values_equal(row[from_name], row[to_name]):
                        item[f"{name}_to"] = row[to_name]
            world_records.append(item)

    stats = pd.DataFrame(records)
    if stats.empty:
        empty = pd.DataFrame()
        return stats, empty, pd.DataFrame(world_records)

    stats["paired_t_q_value_fdr_all_pairs"] = _benjamini_hochberg(
        pd.to_numeric(stats["paired_t_p_value"], errors="coerce").to_numpy(float)
    )
    isolated = stats[stats["one_axis_change"] & stats["adjacent_on_axis"]].copy()
    if not isolated.empty:
        isolated["paired_t_q_value_fdr_isolated"] = _benjamini_hochberg(
            pd.to_numeric(isolated["paired_t_p_value"], errors="coerce").to_numpy(float)
        )
        isolated = isolated.sort_values(
            ["changed_parameter", "from_value", "to_value", "fixed_context_json"],
            kind="stable",
        ).reset_index(drop=True)

    stats = stats.sort_values(
        ["changed_parameter_count", "changed_parameter", "from_design_id", "to_design_id"],
        kind="stable",
    ).reset_index(drop=True)
    return stats, isolated, pd.DataFrame(world_records)


def _design_paths(rows: pd.DataFrame, manifest: Mapping[str, Any]) -> tuple[str, ...]:
    declared = manifest.get("design_variable_paths", ())
    if isinstance(declared, (list, tuple)) and declared:
        return tuple(str(value) for value in declared)
    columns = sorted(
        (column for column in rows.columns if column.startswith("design::")),
        key=str,
    )
    if columns:
        return tuple(column.removeprefix("design::") for column in columns)
    if "design_path" in rows:
        paths = [
            str(value)
            for value in rows["design_path"].dropna().drop_duplicates()
            if str(value) not in {"", "nominal"}
        ]
        if len(paths) == 1:
            return tuple(paths)
    return ()


def _candidate_values(
    rows: pd.DataFrame,
    design_ids: list[str],
    paths: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for design_id in design_ids:
        group = rows[rows["design_id"].astype(str) == design_id]
        values: dict[str, Any] = {}
        if "design_values_json" in group and not group.empty:
            try:
                parsed = json.loads(str(group["design_values_json"].iloc[0]))
                if isinstance(parsed, Mapping):
                    values.update({str(key): value for key, value in parsed.items()})
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        for path in paths:
            column = f"design::{path}"
            if path not in values and column in group and not group.empty:
                values[path] = group[column].iloc[0]
        if len(paths) == 1 and paths[0] not in values and "design_value" in group:
            values[paths[0]] = group["design_value"].iloc[0]
        output[design_id] = {path: _scalar(values.get(path)) for path in paths}
    return output


def _axis_levels(
    values: Mapping[str, Mapping[str, Any]],
    design_ids: list[str],
    paths: tuple[str, ...],
) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for path in paths:
        seen: list[Any] = []
        for design_id in design_ids:
            value = values.get(design_id, {}).get(path)
            if not any(_values_equal(value, existing) for existing in seen):
                seen.append(value)
        try:
            numeric = [float(value) for value in seen]
            order = np.argsort(numeric)
            seen = [seen[int(index)] for index in order]
        except (TypeError, ValueError):
            pass
        result[path] = seen
    return result


def _orient_pair(
    first: str,
    second: str,
    *,
    values: Mapping[str, Mapping[str, Any]],
    paths: tuple[str, ...],
    design_ids: list[str],
) -> tuple[str, str, tuple[str, ...]]:
    changed = _changed_paths(values.get(first, {}), values.get(second, {}), paths)
    if len(changed) == 1:
        path = changed[0]
        left = values.get(first, {}).get(path)
        right = values.get(second, {}).get(path)
        try:
            if float(left) > float(right):
                return second, first, changed
            return first, second, changed
        except (TypeError, ValueError):
            pass
    return (
        (first, second, changed)
        if design_ids.index(first) <= design_ids.index(second)
        else (second, first, changed)
    )


def _changed_paths(
    left: Mapping[str, Any], right: Mapping[str, Any], paths: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        path for path in paths if not _values_equal(left.get(path), right.get(path))
    )


def _is_adjacent(path: str, left: Any, right: Any, levels: Mapping[str, list[Any]]) -> bool:
    axis = levels.get(path, [])
    left_index = next((i for i, value in enumerate(axis) if _values_equal(value, left)), None)
    right_index = next((i for i, value in enumerate(axis) if _values_equal(value, right)), None)
    return (
        left_index is not None
        and right_index is not None
        and abs(left_index - right_index) == 1
    )


def _bootstrap_mean_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
    confidence_level: float,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return math.nan, math.nan
    if clean.size == 1 or np.allclose(clean, clean[0], rtol=0.0, atol=1e-15):
        value = float(np.mean(clean))
        return value, value
    indices = rng.integers(0, clean.size, size=(resamples, clean.size))
    means = clean[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def _paired_t_p_value(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return math.nan
    spread = float(np.std(clean, ddof=1))
    if spread <= 1e-15:
        return 1.0 if abs(float(np.mean(clean))) <= 1e-15 else 0.0
    result = ttest_1samp(clean, popmean=0.0, nan_policy="omit")
    return float(result.pvalue)


def _sign_test_p_value(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    positive = int(np.sum(clean > 1e-12))
    negative = int(np.sum(clean < -1e-12))
    n = positive + negative
    if n == 0:
        return 1.0 if clean.size else math.nan
    return float(binomtest(positive, n=n, p=0.5, alternative="two-sided").pvalue)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    output = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return output
    order = valid[np.argsort(p[valid])]
    m = len(order)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = m - reverse_rank + 1
        value = min(running, float(p[index]) * m / rank)
        adjusted[rank - 1] = value
        running = value
    for rank, index in enumerate(order):
        output[index] = min(1.0, adjusted[rank])
    return output


def _effect_direction(interval: tuple[float, float]) -> str:
    low, high = interval
    if not np.isfinite(low) or not np.isfinite(high):
        return "insufficient paired completions"
    if low > 0.0:
        return "to design faster"
    if high < 0.0:
        return "to design slower"
    return "uncertain / overlaps zero"


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(True, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _values_equal(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    try:
        lf = float(left)
        rf = float(right)
        if np.isfinite(lf) and np.isfinite(rf):
            return bool(np.isclose(lf, rf, rtol=0.0, atol=1e-12))
    except (TypeError, ValueError):
        pass
    return str(left) == str(right)


def _mean(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    return float(np.mean(clean)) if clean.size else math.nan


def _median(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    return float(np.median(clean)) if clean.size else math.nan


def _std(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    return float(np.std(clean, ddof=1)) if clean.size >= 2 else math.nan
