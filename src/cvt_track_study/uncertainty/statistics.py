"""Summary intervals, paired design rankings, bootstrap error, and convergence checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class SummaryInterval:
    count: int
    mean: float
    standard_deviation: float
    p10: float
    median: float
    p90: float
    median_bootstrap_low: float
    median_bootstrap_high: float
    p10_bootstrap_low: float
    p10_bootstrap_high: float
    p90_bootstrap_low: float
    p90_bootstrap_high: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "p10": self.p10,
            "median": self.median,
            "p90": self.p90,
            "median_bootstrap_95_low": self.median_bootstrap_low,
            "median_bootstrap_95_high": self.median_bootstrap_high,
            "p10_bootstrap_95_low": self.p10_bootstrap_low,
            "p10_bootstrap_95_high": self.p10_bootstrap_high,
            "p90_bootstrap_95_low": self.p90_bootstrap_low,
            "p90_bootstrap_95_high": self.p90_bootstrap_high,
        }


def summarize_samples(
    values: Sequence[float] | np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int = 1000,
) -> SummaryInterval:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Summary samples must be a non-empty finite 1-D array.")
    ddof = 1 if array.size > 1 else 0
    p10, median, p90 = np.quantile(array, (0.1, 0.5, 0.9), method="linear")
    boot = _bootstrap_quantiles(
        array,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    intervals = np.quantile(boot, (0.025, 0.975), axis=0, method="linear")
    return SummaryInterval(
        count=int(array.size),
        mean=float(np.mean(array)),
        standard_deviation=float(np.std(array, ddof=ddof)),
        p10=float(p10),
        median=float(median),
        p90=float(p90),
        p10_bootstrap_low=float(intervals[0, 0]),
        p10_bootstrap_high=float(intervals[1, 0]),
        median_bootstrap_low=float(intervals[0, 1]),
        median_bootstrap_high=float(intervals[1, 1]),
        p90_bootstrap_low=float(intervals[0, 2]),
        p90_bootstrap_high=float(intervals[1, 2]),
    )


def paired_design_statistics(
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    design_key: str,
    replicate_key: str,
    metric: str,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 1000,
) -> dict[str, dict[str, float]]:
    """Summarize paired design rankings and finite-sample estimation error.

    Each bootstrap resample selects complete scenario rows, never individual
    design results. This preserves the common-random-number pairing used by the
    study and keeps win fractions and regret intervals statistically coherent.
    """

    designs = sorted({str(row[design_key]) for row in rows})
    replicates = sorted({int(row[replicate_key]) for row in rows})
    if not designs or not replicates:
        raise ValueError("Paired design statistics require at least one design and replicate.")
    if bootstrap_resamples < 100:
        raise ValueError("At least 100 bootstrap resamples are required.")
    values: dict[str, dict[int, float]] = {design: {} for design in designs}
    for row in rows:
        design = str(row[design_key])
        replicate = int(row[replicate_key])
        value = float(row[metric])
        if not np.isfinite(value):
            raise ValueError("Paired design metrics must be finite.")
        if replicate in values[design]:
            raise ValueError(
                f"Design {design!r} repeats paired replicate {replicate}."
            )
        values[design][replicate] = value
    for design in designs:
        missing = set(replicates) - set(values[design])
        if missing:
            raise ValueError(f"Design {design!r} is missing paired replicates {sorted(missing)}.")

    matrix = np.asarray(
        [[values[design][replicate] for design in designs] for replicate in replicates],
        dtype=float,
    )
    win_shares, regrets = _paired_win_shares_and_regrets(matrix)
    rng = np.random.default_rng(bootstrap_seed)
    boot = {
        design: np.empty((bootstrap_resamples, 4), dtype=float)
        for design in designs
    }
    chunk = 250
    for start in range(0, bootstrap_resamples, chunk):
        stop = min(start + chunk, bootstrap_resamples)
        indices = rng.integers(0, matrix.shape[0], size=(stop - start, matrix.shape[0]))
        for offset, sample_indices in enumerate(indices, start=start):
            sample_wins = win_shares[sample_indices]
            sample_regrets = regrets[sample_indices]
            for column, design in enumerate(designs):
                design_regret = sample_regrets[:, column]
                boot[design][offset] = (
                    float(np.mean(sample_wins[:, column])),
                    float(np.mean(design_regret)),
                    float(np.median(design_regret)),
                    float(np.quantile(design_regret, 0.9, method="linear")),
                )

    result: dict[str, dict[str, float]] = {}
    for column, design in enumerate(designs):
        design_regret = regrets[:, column]
        estimate = np.asarray(
            [
                float(np.mean(win_shares[:, column])),
                float(np.mean(design_regret)),
                float(np.median(design_regret)),
                float(np.quantile(design_regret, 0.9, method="linear")),
            ]
        )
        intervals = np.quantile(
            boot[design], (0.025, 0.975), axis=0, method="linear"
        )
        names = (
            "paired_win_fraction",
            "mean_paired_regret",
            "median_paired_regret",
            "p90_paired_regret",
        )
        record: dict[str, float] = {}
        for index, name in enumerate(names):
            record[name] = float(estimate[index])
            record[f"{name}_bootstrap_95_low"] = float(intervals[0, index])
            record[f"{name}_bootstrap_95_high"] = float(intervals[1, index])
        result[design] = record
    return result


def _paired_win_shares_and_regrets(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best = np.min(matrix, axis=1, keepdims=True)
    tied = np.isclose(matrix, best, rtol=0.0, atol=1e-12)
    win_shares = tied / np.sum(tied, axis=1, keepdims=True)
    regrets = matrix - best
    return win_shares.astype(float), regrets.astype(float)

def convergence_diagnostics(
    values: Sequence[float],
) -> dict[str, float | int | bool | str | None]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Convergence diagnostics require samples.")
    half = array.size // 2
    first = array[:half] if half else array
    second = array[half:] if half else array
    overall = float(np.median(array))
    split_difference = abs(float(np.median(first)) - float(np.median(second)))
    scale = max(abs(overall), float(np.std(array)), 1e-9)
    relative = split_difference / scale
    mean_se = (
        float(np.std(array, ddof=1) / np.sqrt(array.size))
        if array.size > 1
        else None
    )
    enough_count = array.size >= 20
    stable_split = relative <= 0.05
    return {
        "sample_count": int(array.size),
        "minimum_recommended_count_met": enough_count,
        "split_half_median_difference": split_difference,
        "split_half_relative_difference": relative,
        "mean_monte_carlo_standard_error": mean_se,
        "split_half_stable_5_percent": stable_split,
        "status": "adequate_quick_check" if enough_count and stable_split else "more_replicates_recommended",
    }


def _bootstrap_quantiles(array: np.ndarray, *, seed: int, resamples: int) -> np.ndarray:
    if resamples < 100:
        raise ValueError("At least 100 bootstrap resamples are required.")
    rng = np.random.default_rng(seed)
    # Chunking avoids a large temporary allocation for eventual long studies.
    result = np.empty((resamples, 3), dtype=float)
    chunk = 250
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        samples = array[indices]
        result[start:stop] = np.quantile(samples, (0.1, 0.5, 0.9), axis=1).T
    return result
