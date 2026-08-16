"""Audit reports for empirical endurance traffic calibration and propagation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvt_track_study.simulation.traffic import (
    TrafficCalibration,
    _draw_nhpp_window,
    traffic_model_from_project,
)


def write_traffic_calibration_project(
    project: str | Path, *, output_directory: Path | None = None
) -> Path:
    project_root = Path(project).resolve()
    if project_root.is_file():
        project_root = project_root.parent
    model = traffic_model_from_project(project_root)
    if model is None:
        raise ValueError(f"No enabled track/traffic.toml was found under {project_root}")
    output = (
        output_directory.resolve()
        if output_directory is not None
        else project_root
        / "results"
        / "traffic_calibration"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output.mkdir(parents=True, exist_ok=True)
    return write_traffic_calibration_report(model, output, project_root=project_root)


def write_traffic_calibration_report(
    model: TrafficCalibration, output: Path, *, project_root: Path | None = None
) -> Path:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)

    contract = model.contract()
    (output / "traffic_calibration.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    events = _events_frame(model)
    source_events = pd.DataFrame([event.serializable() for event in model.source_observations])
    survival = _survival_frame(model)
    rate_bins = _rate_bins_frame(model)
    source_rate_bins = _source_rate_bins_frame(model)
    marks = _mark_summary(events)
    predictive = _predictive_validation(model)
    speed_trace_summary, speed_trace_laps = _speed_trace_validation(project_root)
    events.to_csv(output / "traffic_events_normalized.csv", index=False)
    source_events.to_csv(output / "traffic_source_events_normalized.csv", index=False)
    survival.to_csv(output / "traffic_field_survival.csv", index=False)
    rate_bins.to_csv(output / "traffic_rate_by_15min.csv", index=False)
    source_rate_bins.to_csv(output / "traffic_source_rate_by_15min.csv", index=False)
    marks.to_csv(output / "traffic_mark_summary.csv", index=False)
    predictive.to_csv(output / "traffic_predictive_validation.csv", index=False)
    if not speed_trace_summary.empty:
        speed_trace_summary.to_csv(
            output / "traffic_speed_trace_validation_summary.csv", index=False
        )
        speed_trace_laps.to_csv(
            output / "traffic_speed_trace_validation_laps.csv", index=False
        )
        _plot_speed_trace_laps(
            speed_trace_laps, plots / "traffic_speed_trace_laps.png"
        )

    _plot_rate_fit(
        model,
        source_rate_bins if model.arrival_model != "homogeneous_poisson" else rate_bins,
        plots / "traffic_rate_fit.png",
    )
    _plot_survival(model, plots / "traffic_field_survival.png")
    _plot_marks(events, plots / "traffic_marks.png")

    target = output / "traffic_calibration_report.html"
    target.write_text(
        _calibration_html(
            model, rate_bins, source_rate_bins, marks, predictive, speed_trace_summary
        ),
        encoding="utf-8",
    )
    return target


def augment_full_uncertainty_traffic_report(output: Path) -> Path | None:
    """Append traffic audit/impact sections to an already-generated full uncertainty report."""

    output = output.resolve()
    manifest_path = output / "run_manifest.json"
    scenario_path = output / "scenario_draws.jsonl"
    rows_path = output / "replicate_results.csv"
    if not manifest_path.is_file() or not scenario_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    traffic_contract = manifest.get("traffic_model")
    if not isinstance(traffic_contract, Mapping) or not traffic_contract.get("enabled", False):
        return None

    worlds, events = _study_traffic_frames(scenario_path)
    rows = pd.read_csv(rows_path) if rows_path.is_file() else pd.DataFrame()
    worlds.to_csv(output / "full_uncertainty_traffic_worlds.csv", index=False)
    events.to_csv(output / "full_uncertainty_traffic_events.csv", index=False)
    impact = _study_traffic_impact(rows)
    impact.to_csv(output / "full_uncertainty_traffic_impact.csv", index=False)

    report = output / "full_uncertainty_report.html"
    if not report.is_file():
        candidates = sorted(output.glob("*uncertainty*.html"))
        if not candidates:
            return None
        report = candidates[0]
    text = report.read_text(encoding="utf-8")
    marker = "<!-- TRAFFIC_AUDIT_SECTION -->"
    end_marker = "<!-- /TRAFFIC_AUDIT_SECTION -->"
    section = _study_traffic_html(traffic_contract, worlds, events, impact)
    if marker in text and end_marker in text:
        before, remainder = text.split(marker, 1)
        _, after = remainder.split(end_marker, 1)
        text = before + section + after
    else:
        lower = text.lower()
        index = lower.rfind("</body>")
        text = text[:index] + section + text[index:] if index >= 0 else text + section
    report.write_text(text, encoding="utf-8")
    return report


def _events_frame(model: TrafficCalibration) -> pd.DataFrame:
    return pd.DataFrame([event.serializable() for event in model.observations])


def _survival_frame(model: TrafficCalibration) -> pd.DataFrame:
    grid = np.arange(0.0, model.race_duration_s + 1.0, 900.0)
    return pd.DataFrame(
        {
            "race_time_s": grid,
            "race_time_h": grid / 3600.0,
            "field_survival_fraction": [model.survival_fraction(t) for t in grid],
            "fitted_event_rate_per_hour": [model.rate_per_hour(t) for t in grid],
        }
    )


def _rate_bins_frame(model: TrafficCalibration) -> pd.DataFrame:
    end = max(stream.exposure_end_s for stream in model.streams)
    bins = np.arange(0.0, end + 900.0, 900.0)
    records: list[dict[str, Any]] = []
    for start, stop in zip(bins[:-1], bins[1:]):
        exposure_s = sum(
            max(0.0, min(stop, stream.exposure_end_s) - max(start, stream.exposure_start_s))
            for stream in model.streams
        )
        if exposure_s <= 0.0:
            continue
        observed = sum(start <= event.time_s < stop for event in model.observations)
        # integrate expected count with a fine enough piecewise grid through the KM jumps
        sample = np.linspace(start, stop, 901)
        rates = np.asarray([model.rate_per_hour(t) / 3600.0 for t in sample])
        exposure_count = np.asarray(
            [sum(stream.exposure_start_s <= t < stream.exposure_end_s for stream in model.streams) for t in sample],
            dtype=float,
        )
        expected = float(np.trapezoid(rates * exposure_count, sample))
        records.append(
            {
                "start_s": start,
                "end_s": stop,
                "midpoint_min": (start + stop) / 120.0,
                "observer_exposure_h": exposure_s / 3600.0,
                "observed_event_count": observed,
                "observed_rate_per_hour": observed / (exposure_s / 3600.0),
                "fitted_expected_event_count": expected,
                "fitted_average_rate_per_hour": expected / (exposure_s / 3600.0),
            }
        )
    return pd.DataFrame(records)



def _source_rate_bins_frame(model: TrafficCalibration) -> pd.DataFrame:
    end = max(stream.exposure_end_s for stream in model.source_streams)
    bins = np.arange(0.0, end + 900.0, 900.0)
    records: list[dict[str, Any]] = []
    for start, stop in zip(bins[:-1], bins[1:]):
        exposure_s = sum(
            max(0.0, min(stop, stream.exposure_end_s) - max(start, stream.exposure_start_s))
            for stream in model.source_streams
        )
        if exposure_s <= 0.0:
            continue
        observed = sum(start <= event.time_s < stop for event in model.source_observations)
        sample = np.linspace(start, stop, 901)
        source_rates = dict(model.source_stream_initial_rates_per_hour)
        total_rate = np.asarray(
            [
                sum(
                    source_rates[stream.identifier]
                    * model.survival_fraction(t) ** model.field_exponent
                    / 3600.0
                    for stream in model.source_streams
                    if stream.exposure_start_s <= t < stream.exposure_end_s
                )
                for t in sample
            ],
            dtype=float,
        )
        expected = float(np.trapezoid(total_rate, sample))
        records.append(
            {
                "start_s": start,
                "end_s": stop,
                "midpoint_min": (start + stop) / 120.0,
                "observer_exposure_h": exposure_s / 3600.0,
                "observed_event_count": observed,
                "observed_rate_per_hour": observed / (exposure_s / 3600.0),
                "fitted_expected_event_count": expected,
                "fitted_average_rate_per_hour": expected / (exposure_s / 3600.0),
            }
        )
    return pd.DataFrame(records)

def _mark_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    records = []
    for event_type, group in events.groupby("event_type", sort=True):
        records.append(
            {
                "event_type": event_type,
                "count": len(group),
                "fraction": len(group) / len(events),
                "duration_median_s": float(group["duration_s"].median()),
                "duration_mean_s": float(group["duration_s"].mean()),
                "retained_fraction_median": float(group["retained_speed_fraction"].median()),
                "retained_fraction_mean": float(group["retained_speed_fraction"].mean()),
            }
        )
    return pd.DataFrame(records)



def _predictive_validation(model: TrafficCalibration, *, replicates: int = 3000) -> pd.DataFrame:
    """Posterior/parametric predictive check against this project's observer streams.

    For the equal-stream mixture, every observed stream is treated as an independent
    draw from the same empirical population of traffic-exposure regimes. That keeps a
    short high-rate stream from being averaged away merely because another car has a
    longer reviewed video.
    """

    seed = int(model.fingerprint[:16], 16) ^ 0x5052454449435456
    rng = np.random.default_rng(seed)
    km_t = np.asarray(model.km_time_s, dtype=float)
    km_s = np.asarray(model.km_survival, dtype=float)
    marks = model.observations
    total_exposure = model.total_observer_exposure_s
    records: list[dict[str, float]] = []

    def calibration_draw() -> tuple[float, float]:
        if model.bootstrap_draws:
            draw = model.bootstrap_draws[int(rng.integers(0, len(model.bootstrap_draws)))]
            return draw.initial_rate_per_hour, draw.field_exponent
        return model.initial_rate_per_hour, model.field_exponent

    for _ in range(max(200, int(replicates))):
        shared_rate, shared_exponent = calibration_draw()
        generated: list[tuple[float, float, str]] = []
        counts: dict[str, int] = {}
        busy_s = 0.0
        for stream in model.streams:
            if model.generation_rate_mode == "equal_observer_stream_mixture":
                initial_rate, exponent = calibration_draw()
            else:
                initial_rate, exponent = shared_rate, shared_exponent
            offsets = _draw_nhpp_window(
                rng=rng,
                start_race_time_s=stream.exposure_start_s,
                horizon_s=stream.exposure_s,
                initial_rate_per_hour=initial_rate,
                field_exponent=exponent,
                km_time_s=km_t,
                km_survival=km_s,
                race_duration_s=model.race_duration_s,
            )
            counts[stream.identifier] = len(offsets)
            intervals: list[tuple[float, float]] = []
            for offset in offsets:
                mark = marks[int(rng.integers(0, len(marks)))]
                start = stream.exposure_start_s + float(offset)
                stop = min(stream.exposure_end_s, start + mark.duration_s)
                intervals.append((start, stop))
                generated.append(
                    (mark.duration_s, mark.retained_speed_fraction, mark.event_type)
                )
            busy_s += _union_interval_length(intervals)

        count = len(generated)
        records.append(
            {
                "total_event_count": float(count),
                "busy_fraction": busy_s / total_exposure,
                "mean_duration_s": (
                    float(np.mean([row[0] for row in generated])) if generated else np.nan
                ),
                "mean_retained_fraction": (
                    float(np.mean([row[1] for row in generated])) if generated else np.nan
                ),
                "yellow_fraction": (
                    sum(row[2] == "yellow flag" for row in generated) / count if count else np.nan
                ),
                "stop_like_fraction": (
                    sum(row[1] <= 1.0e-12 for row in generated) / count if count else np.nan
                ),
                **{f"stream_{key}_event_count": float(value) for key, value in counts.items()},
            }
        )

    generated = pd.DataFrame(records)
    observed_busy_s = 0.0
    for stream in model.streams:
        intervals = [
            (event.time_s, min(stream.exposure_end_s, event.time_s + event.duration_s))
            for event in stream.observations
        ]
        observed_busy_s += _union_interval_length(intervals)
    observed: dict[str, float] = {
        "total_event_count": float(len(model.observations)),
        "busy_fraction": observed_busy_s / total_exposure,
        "mean_duration_s": float(np.mean([event.duration_s for event in model.observations])),
        "mean_retained_fraction": float(
            np.mean([event.retained_speed_fraction for event in model.observations])
        ),
        "yellow_fraction": sum(event.event_type == "yellow flag" for event in model.observations)
        / len(model.observations),
        "stop_like_fraction": sum(
            event.retained_speed_fraction <= 1.0e-12 for event in model.observations
        ) / len(model.observations),
    }
    for stream in model.streams:
        observed[f"stream_{stream.identifier}_event_count"] = float(len(stream.observations))

    labels = {
        "total_event_count": "Total event count",
        "busy_fraction": "Fraction of observer time in annotated traffic",
        "mean_duration_s": "Mean event duration (s)",
        "mean_retained_fraction": "Mean retained speed fraction",
        "yellow_fraction": "Yellow-flag event fraction",
        "stop_like_fraction": "Stop-like restriction fraction (retained speed = 0)",
    }
    labels.update(
        {f"stream_{stream.identifier}_event_count": f"{stream.identifier} event count" for stream in model.streams}
    )

    output: list[dict[str, Any]] = []
    for key, label in labels.items():
        values = pd.to_numeric(generated[key], errors="coerce").dropna()
        value = observed[key]
        output.append(
            {
                "metric": label,
                "observed": value,
                "predictive_mean": float(values.mean()),
                "predictive_p05": float(values.quantile(0.05)),
                "predictive_p50": float(values.quantile(0.50)),
                "predictive_p95": float(values.quantile(0.95)),
                "observed_percentile": float((values <= value).mean()),
            }
        )
    return pd.DataFrame(output)

def _speed_trace_validation(
    project_root: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize optional independent GPS/speed traces without fitting traffic.

    Repeated laps expose the observed tail of on-track delay. Raw slowdowns are
    deliberately not converted into traffic-event labels because driver error,
    terrain, mechanical trouble and service activity can produce the same signal.
    """

    if project_root is None:
        return pd.DataFrame(), pd.DataFrame()
    config_path = Path(project_root) / "track" / "traffic.toml"
    if not config_path.is_file():
        return pd.DataFrame(), pd.DataFrame()
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return pd.DataFrame(), pd.DataFrame()
    traffic = raw.get("traffic", {})
    rows = (
        traffic.get("validation_speed_traces", ())
        if isinstance(traffic, Mapping)
        else ()
    )
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return pd.DataFrame(), pd.DataFrame()

    summaries: list[dict[str, Any]] = []
    all_laps: list[pd.DataFrame] = []
    root = Path(project_root).resolve()
    for index, item in enumerate(rows, 1):
        if not isinstance(item, Mapping):
            continue
        identifier = str(item.get("id", f"trace_{index:02d}"))
        source = (root / str(item.get("file", ""))).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "traffic validation speed trace must remain inside the project"
            ) from exc
        if not source.is_file():
            raise FileNotFoundError(
                f"Traffic validation speed trace does not exist: {source}"
            )
        frame = pd.read_csv(source)
        timestamp_col = str(item.get("timestamp_column", "timestamp"))
        latitude_col = str(item.get("latitude_column", "lat"))
        longitude_col = str(item.get("longitude_column", "lon"))
        speed_col = str(item.get("speed_column", "speed_kmh"))
        required = (timestamp_col, latitude_col, longitude_col, speed_col)
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(
                f"Traffic validation trace {identifier!r} lacks columns: "
                + ", ".join(missing)
            )
        parsed = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(frame[timestamp_col], errors="coerce"),
                "lat": pd.to_numeric(frame[latitude_col], errors="coerce"),
                "lon": pd.to_numeric(frame[longitude_col], errors="coerce"),
                "speed_kmh": pd.to_numeric(frame[speed_col], errors="coerce"),
            }
        ).dropna()
        parsed = (
            parsed.sort_values("timestamp", kind="stable")
            .drop_duplicates(
                subset=["timestamp", "lat", "lon", "speed_kmh"], keep="first"
            )
            .reset_index(drop=True)
        )
        if len(parsed) < 20:
            raise ValueError(
                f"Traffic validation trace {identifier!r} has too few valid rows"
            )

        lap_frame = _segment_speed_trace_laps(
            parsed,
            identifier=identifier,
            gate_latitude_deg=float(item["start_finish_latitude_deg"]),
            gate_longitude_deg=float(item["start_finish_longitude_deg"]),
            gate_radius_m=float(item.get("gate_radius_m", 15.0)),
            minimum_lap_time_s=float(item.get("minimum_lap_time_s", 120.0)),
            maximum_sample_gap_s=float(item.get("maximum_sample_gap_s", 5.0)),
        )
        if lap_frame.empty:
            continue
        running = lap_frame[lap_frame["usable_running_lap"]].copy()
        if running.empty:
            running = lap_frame.copy()
        baseline = float(running["lap_time_s"].quantile(0.20))
        lap_frame["excess_vs_running_p20_s"] = lap_frame["lap_time_s"] - baseline
        running = lap_frame[lap_frame["usable_running_lap"]].copy()
        if running.empty:
            running = lap_frame.copy()
        summaries.append(
            {
                "trace_id": identifier,
                "source_file": str(source.relative_to(root)),
                "valid_sample_count": len(parsed),
                "complete_lap_count": len(lap_frame),
                "usable_running_lap_count": int(
                    lap_frame["usable_running_lap"].sum()
                ),
                "service_or_gap_lap_count": int(
                    (~lap_frame["usable_running_lap"]).sum()
                ),
                "running_lap_time_p10_s": float(
                    running["lap_time_s"].quantile(0.10)
                ),
                "running_lap_time_p20_s": baseline,
                "running_lap_time_median_s": float(running["lap_time_s"].median()),
                "running_lap_time_p90_s": float(
                    running["lap_time_s"].quantile(0.90)
                ),
                "running_lap_time_p95_s": float(
                    running["lap_time_s"].quantile(0.95)
                ),
                "running_mean_speed_median_kmh": float(
                    running["mean_speed_kmh"].median()
                ),
                "running_stopped_time_fraction_median": float(
                    running["stopped_time_fraction"].median()
                ),
                "running_low_speed_time_fraction_median": float(
                    running["low_speed_time_fraction"].median()
                ),
                "use_in_traffic_fit": False,
                "interpretation": (
                    "Independent speed/GPS validation only; raw slowdowns are not "
                    "assumed to be traffic without event labels."
                ),
            }
        )
        all_laps.append(lap_frame)

    laps = pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()
    return pd.DataFrame(summaries), laps


def _segment_speed_trace_laps(
    frame: pd.DataFrame,
    *,
    identifier: str,
    gate_latitude_deg: float,
    gate_longitude_deg: float,
    gate_radius_m: float,
    minimum_lap_time_s: float,
    maximum_sample_gap_s: float,
) -> pd.DataFrame:
    lat = frame["lat"].to_numpy(dtype=float)
    lon = frame["lon"].to_numpy(dtype=float)
    distance_to_gate = _haversine_arrays_m(
        lat,
        lon,
        np.full(len(frame), gate_latitude_deg, dtype=float),
        np.full(len(frame), gate_longitude_deg, dtype=float),
    )
    inside = np.flatnonzero(distance_to_gate <= gate_radius_m)
    if inside.size < 2:
        return pd.DataFrame()

    timestamps = frame["timestamp"].reset_index(drop=True)
    groups: list[list[int]] = []
    current = [int(inside[0])]
    for raw_index in inside[1:]:
        i = int(raw_index)
        previous = current[-1]
        dt = (timestamps.iloc[i] - timestamps.iloc[previous]).total_seconds()
        if i <= previous + 3 and dt <= 5.0:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    crossing_indices = [
        min(group, key=lambda i: float(distance_to_gate[i])) for group in groups
    ]
    accepted: list[int] = []
    for i in crossing_indices:
        if not accepted:
            accepted.append(i)
            continue
        dt = (timestamps.iloc[i] - timestamps.iloc[accepted[-1]]).total_seconds()
        if dt >= minimum_lap_time_s:
            accepted.append(i)
    if len(accepted) < 2:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for lap_number, (left, right) in enumerate(
        zip(accepted[:-1], accepted[1:]), 1
    ):
        lap = frame.iloc[left : right + 1].reset_index(drop=True)
        duration = (
            lap["timestamp"].iloc[-1] - lap["timestamp"].iloc[0]
        ).total_seconds()
        gaps = lap["timestamp"].diff().dt.total_seconds().dropna()
        maximum_gap = float(gaps.max()) if not gaps.empty else 0.0
        step_distance = _haversine_arrays_m(
            lap["lat"].to_numpy(dtype=float)[:-1],
            lap["lon"].to_numpy(dtype=float)[:-1],
            lap["lat"].to_numpy(dtype=float)[1:],
            lap["lon"].to_numpy(dtype=float)[1:],
        )
        length = float(np.sum(step_distance))
        dt = (
            lap["timestamp"]
            .shift(-1)
            .sub(lap["timestamp"])
            .dt.total_seconds()
            .to_numpy()
        )
        speed = lap["speed_kmh"].to_numpy(dtype=float)
        valid_dt = (
            np.isfinite(dt)
            & (dt > 0.0)
            & (dt <= maximum_sample_gap_s)
        )
        weighted_time = float(np.sum(dt[valid_dt]))
        stopped = (speed < 1.0) & valid_dt
        low_speed = (speed < 5.0) & valid_dt
        records.append(
            {
                "trace_id": identifier,
                "lap_index": lap_number,
                "start_timestamp": lap["timestamp"].iloc[0].isoformat(),
                "end_timestamp": lap["timestamp"].iloc[-1].isoformat(),
                "lap_time_s": float(duration),
                "sample_count": len(lap),
                "maximum_sample_gap_s": maximum_gap,
                "gps_path_length_m": length,
                "mean_speed_kmh": float(lap["speed_kmh"].mean()),
                "median_speed_kmh": float(lap["speed_kmh"].median()),
                "minimum_speed_kmh": float(lap["speed_kmh"].min()),
                "maximum_speed_kmh": float(lap["speed_kmh"].max()),
                "stopped_time_fraction": (
                    float(np.sum(dt[stopped])) / weighted_time
                    if weighted_time > 0.0
                    else np.nan
                ),
                "low_speed_time_fraction": (
                    float(np.sum(dt[low_speed])) / weighted_time
                    if weighted_time > 0.0
                    else np.nan
                ),
            }
        )
    result = pd.DataFrame(records)
    continuous = result[
        result["maximum_sample_gap_s"] <= maximum_sample_gap_s
    ]
    base = continuous if not continuous.empty else result
    median_time = float(base["lap_time_s"].median())
    median_length = float(base["gps_path_length_m"].median())
    time_limit = max(2.0 * median_time, median_time + 180.0)
    result["usable_running_lap"] = (
        (result["maximum_sample_gap_s"] <= maximum_sample_gap_s)
        & (result["lap_time_s"] <= time_limit)
        & result["gps_path_length_m"].between(
            0.85 * median_length, 1.15 * median_length
        )
    )
    return result


def _haversine_arrays_m(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    radius = 6_371_000.0
    lat1 = np.radians(np.asarray(lat1_deg, dtype=float))
    lon1 = np.radians(np.asarray(lon1_deg, dtype=float))
    lat2 = np.radians(np.asarray(lat2_deg, dtype=float))
    lon2 = np.radians(np.asarray(lon2_deg, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _plot_speed_trace_laps(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for trace_id, group in frame.groupby("trace_id", sort=True):
        x = np.arange(1, len(group) + 1)
        normal = group["usable_running_lap"].to_numpy(dtype=bool)
        ax.scatter(
            x[normal],
            group.loc[normal, "lap_time_s"],
            label=f"{trace_id} running laps",
        )
        if (~normal).any():
            ax.scatter(
                x[~normal],
                group.loc[~normal, "lap_time_s"],
                marker="x",
                label=f"{trace_id} service/gap-like laps",
            )
    ax.set_xlabel("Complete lap index in speed trace")
    ax.set_ylabel("Observed lap time (s)")
    ax.set_title("Independent GPS/speed-trace validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _union_interval_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted((float(start), float(stop)) for start, stop in intervals if stop > start)
    if not ordered:
        return 0.0
    start, stop = ordered[0]
    total = 0.0
    for next_start, next_stop in ordered[1:]:
        if next_start <= stop:
            stop = max(stop, next_stop)
        else:
            total += stop - start
            start, stop = next_start, next_stop
    return total + stop - start

def _plot_rate_fit(model: TrafficCalibration, frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    label = (
        "homogeneous Poisson"
        if model.arrival_model == "homogeneous_poisson"
        else "field-survival NHPP population mean"
    )
    ax.plot(frame["midpoint_min"], frame["fitted_average_rate_per_hour"], marker="o", label=label)
    ax.scatter(frame["midpoint_min"], frame["observed_rate_per_hour"], label="observed exposure-normalized rate")
    ax.set_xlabel("Race time (min)")
    ax.set_ylabel("Traffic events / observer-hour")
    ax.set_title("Observed traffic rate and production encounter process")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_survival(model: TrafficCalibration, path: Path) -> None:
    grid = np.linspace(0.0, model.race_duration_s, 500)
    rate = [model.rate_per_hour(t) for t in grid]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    if model.arrival_model == "homogeneous_poisson":
        ax.plot(grid / 3600.0, rate, label="production traffic event rate")
        ax.set_xlabel("Race time (h)")
        ax.set_ylabel("Events / observer-hour")
        ax.set_title("Production arrival rate is intentionally time-homogeneous")
        ax.legend()
    else:
        survival = [model.survival_fraction(t) for t in grid]
        ax.step(grid / 3600.0, survival, where="post", label="surviving field fraction")
        ax.set_xlabel("Race time (h)")
        ax.set_ylabel("Field survival fraction")
        ax2 = ax.twinx()
        ax2.plot(grid / 3600.0, rate, linestyle="--", label="population-mean traffic event rate")
        ax2.set_ylabel("Events / observer-hour")
        ax.set_title("Same-race field attrition anchors late-race traffic extrapolation")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def _plot_marks(events: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for event_type, group in events.groupby("event_type", sort=True):
        ax.scatter(group["duration_s"], group["retained_speed_fraction"], label=event_type, alpha=0.8)
    ax.set_xlabel("Observed event duration (s)")
    ax.set_ylabel("Retained unobstructed speed fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Joint empirical traffic marks retained by the simulator")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _calibration_html(
    model: TrafficCalibration,
    rate_bins: pd.DataFrame,
    source_rate_bins: pd.DataFrame,
    marks: pd.DataFrame,
    predictive: pd.DataFrame,
    speed_trace_summary: pd.DataFrame,
) -> str:
    contract = model.contract()
    quantiles = contract.get("bootstrap_quantiles", {})
    rate_q = quantiles.get("initial_rate_per_hour", [float("nan")] * 5)
    exponent_q = quantiles.get("field_exponent", [float("nan")] * 5)
    average_rate_q = quantiles.get("average_rate_per_hour_4h", [float("nan")] * 5)
    same_race = model.source_race_id == model.target_race_id
    source_initial_rates = dict(model.source_stream_initial_rates_per_hour)
    stream_rows = []
    for stream in model.source_streams:
        stream_rows.append(
            {
                "Race": model.source_race_id,
                "Stream": stream.identifier,
                "Exposure (min)": stream.exposure_s / 60.0,
                "Events": len(stream.observations),
                "Observed average rate (/h)": 3600.0 * len(stream.observations) / stream.exposure_s,
                "Fitted stream initial rate (/h)": source_initial_rates.get(stream.identifier, float("nan")),
            }
        )

    if model.arrival_model == "homogeneous_poisson":
        intro = (
            f"This project uses only <strong>{html.escape(model.target_race_id)}</strong> traffic evidence. "
            "A homogeneous Poisson process is the production arrival model because the available "
            "observer window does not justify extrapolating a race-time trend. Complete observed "
            "event marks are resampled jointly, so duration, event type and retained-speed severity "
            "stay empirically coupled."
        )
        model_rows = [
            {
                "Model": "Homogeneous Poisson",
                "Parameters": 1,
                "Log likelihood": model.homogeneous_log_likelihood,
                "AIC": 2 - 2 * model.homogeneous_log_likelihood,
                "Use": "Production model",
            },
            {
                "Model": "Clock-time exponential",
                "Parameters": len(model.source_streams) + 1,
                "Log likelihood": model.exponential_log_likelihood,
                "AIC": 2 * (len(model.source_streams) + 1) - 2 * model.exponential_log_likelihood,
                "Use": "Trend sensitivity comparator only",
            },
        ]
        model_note = (
            "The production model deliberately does not borrow another event's track, field-survival "
            "or severity data. A time trend can be revisited when this event has longer observer exposure."
        )
        third_card = '<div class="card"><div>Arrival model</div><div class="big">constant λ</div><div>no transferred decay law</div></div>'
        survival_heading = "Race-time arrival assumption"
        survival_note = "The constant line is the explicit extrapolation beyond the observed window."
    else:
        intro = (
            f"This project uses only <strong>{html.escape(model.source_race_id)}</strong> evidence: its "
            "traffic observer streams provide arrival timing and event marks, and its own field-survival "
            "records provide the four-hour attrition shape. Each observer stream receives its own nuisance "
            "rate while a shared exponent captures only the within-stream relationship to field survival. "
            "A generic simulated car samples the observer-stream rate regimes equally, preserving the "
            "large between-car traffic variation instead of averaging it away."
        )
        model_rows = [
            {
                "Model": "Field survival: λ₀ S(t)^α",
                "Parameters": len(model.source_streams) + 1,
                "Log likelihood": model.log_likelihood,
                "AIC": 2 * (len(model.source_streams) + 1) - 2 * model.log_likelihood,
                "Use": "Production four-hour extrapolation",
            },
            {
                "Model": "Clock-time exponential",
                "Parameters": len(model.source_streams) + 1,
                "Log likelihood": model.exponential_log_likelihood,
                "AIC": 2 * (len(model.source_streams) + 1) - 2 * model.exponential_log_likelihood,
                "Use": "Clock-time sensitivity comparator",
            },
            {
                "Model": "Per-stream homogeneous Poisson",
                "Parameters": len(model.source_streams),
                "Log likelihood": model.homogeneous_log_likelihood,
                "AIC": 2 * len(model.source_streams) - 2 * model.homogeneous_log_likelihood,
                "Use": "No-decay comparator",
            },
        ]
        model_note = (
            "These models are close in AIC over the observed window, so the data do not strongly identify "
            "a decay law. Field survival is retained only because it is measured in this same race and gives "
            "a physical four-hour extrapolation; bootstrap draws preserve near-flat alternatives."
        )
        third_card = (
            f'<div class="card"><div>Field exponent α</div><div class="big">{model.field_exponent:.2f}</div>'
            f'<div>bootstrap 95% {exponent_q[0]:.2f}–{exponent_q[4]:.2f}</div></div>'
        )
        survival_heading = "Same-race field attrition"
        survival_note = "No field-survival information is transferred from another competition."

    models = pd.DataFrame(model_rows)
    if speed_trace_summary.empty:
        speed_trace_section = ""
    else:
        speed_trace_section = (
            "<h2>Independent GPS/speed-trace validation</h2>"
            "<p>This evidence is intentionally held out of the traffic-event fit. "
            "Repeated laps provide an independent check on the observed tail of "
            "on-track delay and low-speed operation, but raw slowdowns are not "
            "automatically labelled as traffic because driver error, terrain, "
            "mechanical trouble and service activity can produce the same signal.</p>"
            + speed_trace_summary.to_html(
                index=False, float_format=lambda x: f"{x:.3f}"
            )
            + '<img src="plots/traffic_speed_trace_laps.png" '
              'alt="Independent speed trace lap times">'
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Traffic calibration</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#f6f7f9;color:#1f2937}}
main{{max-width:1100px;margin:auto;padding:28px}} h1,h2{{margin-top:1.35em}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.card{{background:white;border:1px solid #d9dde5;border-radius:12px;padding:14px}} .big{{font-size:1.55rem;font-weight:650}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:8px 10px;border:1px solid #dde2ea;text-align:left}}th{{background:#eef1f5}}
img{{max-width:100%;height:auto;background:white;border:1px solid #dde2ea;border-radius:10px}} .note{{padding:12px;border-left:4px solid #6b7280;background:#fff}}
code{{background:#eef1f5;padding:1px 4px;border-radius:4px}}
</style></head><body><main>
<h1>Empirical endurance traffic calibration — {html.escape(model.target_race_id)}</h1>
<p>{intro}</p>
<div class="cards">
<div class="card"><div>Observed events</div><div class="big">{len(model.observations)}</div><div>{model.total_observer_exposure_s/3600:.2f} observer-hours</div></div>
<div class="card"><div>Population-mean initial rate</div><div class="big">{model.initial_rate_per_hour:.1f}/h</div><div>bootstrap 95% {rate_q[0]:.1f}–{rate_q[4]:.1f}/h</div></div>
{third_card}
<div class="card"><div>4 h mean rate</div><div class="big">{model.average_rate_per_hour():.1f}/h</div><div>bootstrap 95% {average_rate_q[0]:.1f}–{average_rate_q[4]:.1f}/h</div></div>
<div class="card"><div>Arrival timing check</div><div class="big">p={model.time_rescaling_ks_pvalue:.3f}</div><div>time-rescaling Exp(1) test</div></div>
</div>
<h2>Project-local evidence</h2>
{pd.DataFrame(stream_rows).to_html(index=False, float_format=lambda x:f'{x:.3f}')}
<p class="note">Source race <code>{html.escape(model.source_race_id)}</code>; target race <code>{html.escape(model.target_race_id)}</code>. Cross-race traffic-event pooling: <strong>{'none' if same_race else 'present'}</strong>.</p>
<h2>Arrival-model check</h2>{models.to_html(index=False, float_format=lambda x:f'{x:.3f}')}
<p class="note">{model_note}</p>
<h2>Predictive validation</h2>
<p>The table re-simulates this project's actual observer exposure windows 3,000 times. It checks event counts, occupied traffic time and empirical mark summaries against the data the generator is intended to reproduce.</p>
{predictive.to_html(index=False, float_format=lambda x:f'{x:.3f}')}
{speed_trace_section}
<img src="plots/traffic_rate_fit.png" alt="Observed and fitted traffic rate">
<h2>{survival_heading}</h2><p>{survival_note}</p><img src="plots/traffic_field_survival.png" alt="Traffic rate through the endurance clock">
<h2>Event severity marks</h2>{marks.to_html(index=False, float_format=lambda x:f'{x:.3f}')}
<img src="plots/traffic_marks.png" alt="Observed traffic duration versus retained speed fraction">
<p class="note">{('The cleaned Arizona spreadsheet heading calls the numeric speed field a “reduction”, but its ordinary slowdown/yellow annotations behave as retained-speed fractions. Rows explicitly labelled <code>full stop</code> are normalized to retained fraction 0. ' if 'arizona' in model.target_race_id.lower() else 'The normalized project observations store retained speed fraction directly. ')}Confidence labels are preserved for audit and are not converted into invented standard deviations.</p>
<h2>15-minute production-rate check</h2>{rate_bins.to_html(index=False, float_format=lambda x:f'{x:.3f}')}
{('<h2>15-minute source-fit check</h2>' + source_rate_bins.to_html(index=False, float_format=lambda x:f'{x:.3f}')) if model.arrival_model != "homogeneous_poisson" else ''}
</main></body></html>"""

def _study_traffic_frames(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    world_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        traffic = record.get("traffic_realization")
        if not isinstance(traffic, Mapping):
            continue
        base = {
            "replicate": record.get("replicate"),
            "base_draw_id": record.get("base_draw_id", record.get("replicate")),
            "track_case_id": record.get("track_case_id", ""),
            "scenario_seed": traffic.get("scenario_seed"),
            "lap_start_race_time_s": traffic.get("lap_start_race_time_s"),
            "initial_rate_per_hour": traffic.get("initial_rate_per_hour"),
            "field_exponent": traffic.get("field_exponent"),
            "calibration_draw_index": traffic.get("calibration_draw_index"),
            "event_count": traffic.get("event_count", len(traffic.get("events", ()))),
        }
        world_rows.append(base)
        for event in traffic.get("events", ()):
            if isinstance(event, Mapping):
                event_rows.append({**base, **{str(k): v for k, v in event.items()}})
    return pd.DataFrame(world_rows), pd.DataFrame(event_rows)


def _study_traffic_impact(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "reference_traffic_penalty_s",
            "bounded_traffic_active_time_s",
            "bounded_traffic_event_count",
            "bounded_traffic_minimum_retained_fraction",
        )
        if column in rows
    ]
    if not columns:
        return pd.DataFrame()
    records = []
    for column in columns:
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        if values.empty:
            continue
        records.append(
            {
                "metric": column,
                "count": len(values),
                "mean": values.mean(),
                "p10": values.quantile(0.1),
                "median": values.median(),
                "p90": values.quantile(0.9),
            }
        )
    return pd.DataFrame(records)


def _study_traffic_html(
    contract: Mapping[str, Any], worlds: pd.DataFrame, events: pd.DataFrame, impact: pd.DataFrame
) -> str:
    if worlds.empty:
        return ""
    unique_worlds = (
        worlds.drop_duplicates(subset=["base_draw_id"])
        if "base_draw_id" in worlds
        else worlds
    )
    event_count = pd.to_numeric(unique_worlds["event_count"], errors="coerce")
    phase_h = pd.to_numeric(unique_worlds["lap_start_race_time_s"], errors="coerce") / 3600.0
    impact_html = impact.to_html(index=False, float_format=lambda x: f"{x:.3f}") if not impact.empty else "<p>Traffic impact columns were not found.</p>"
    source_race = html.escape(str(contract.get("source_race_id", "project")))
    target_race = html.escape(str(contract.get("target_race_id", source_race)))
    arrival = html.escape(str(contract.get("arrival_model", contract.get("model", "traffic"))))
    field_count = int(contract.get("field_record_count", 0))
    field_text = (
        f"; {field_count} same-race field-survival records"
        if field_count
        else "; no field-survival transfer or late-race attrition model"
    )
    return f"""\n<!-- TRAFFIC_AUDIT_SECTION -->
<section id="traffic-audit" style="margin:2rem 0;padding-top:1rem;border-top:2px solid #d1d5db">
<h2>Endurance traffic exposure</h2>
<p>Traffic is an in-simulation external speed constraint, not a post-processing lap-time penalty. One reproducible traffic world is paired across the bounded CVT, infinite reference, design candidates and crossed track cases. The absolute traffic ceiling is based on the shared traffic-free infinite-CVT reference speed profile, so drivetrain changes do not alter the same external traffic restriction.</p>
<ul>
<li>Traffic evidence: <code>{target_race}</code>, {int(contract.get('target_event_count', contract.get('event_count',0)))} events over {float(contract.get('target_observer_exposure_h', contract.get('observer_exposure_h',0))):.2f} observer-hours{field_text}.</li>
<li>Arrival model: <code>{arrival}</code>; generation rate mode <code>{html.escape(str(contract.get('generation_rate_mode','')))}</code>.</li>
<li>Population-mean initial rate: {float(contract.get('initial_rate_per_hour',0)):.1f}/h; four-hour mean {float(contract.get('average_rate_per_hour_4h',0)):.1f}/h.</li>
<li>Cross-race traffic evidence: <strong>{'none' if source_race == target_race else source_race + ' → ' + target_race}</strong>.</li>
<li>Unique simulated traffic worlds: {len(unique_worlds)} ({len(worlds)} route-case scenario rows); event count mean {event_count.mean():.2f}, median {event_count.median():.0f}, p90 {event_count.quantile(.9):.0f}.</li>
<li>Representative lap-start phase is sampled uniformly over the endurance clock (observed median {phase_h.median():.2f} h).</li>
</ul>
<h3>Traffic impact on the simulated drivetrain</h3>{impact_html}
<p><strong>Interpretation limits:</strong> duration/severity marks and arrival calibration come from the project-local traffic evidence declared in <code>track/traffic.toml</code>. Traffic locations are not modeled because they were not measured consistently.</p>
</section>
<!-- /TRAFFIC_AUDIT_SECTION -->\n"""
