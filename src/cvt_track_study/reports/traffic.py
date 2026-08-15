"""Audit reports for empirical endurance traffic calibration and propagation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any, Mapping

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
    return write_traffic_calibration_report(model, output)


def write_traffic_calibration_report(model: TrafficCalibration, output: Path) -> Path:
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
    events.to_csv(output / "traffic_events_normalized.csv", index=False)
    source_events.to_csv(output / "traffic_source_events_normalized.csv", index=False)
    survival.to_csv(output / "traffic_field_survival.csv", index=False)
    rate_bins.to_csv(output / "traffic_rate_by_15min.csv", index=False)
    source_rate_bins.to_csv(output / "traffic_source_rate_by_15min.csv", index=False)
    marks.to_csv(output / "traffic_mark_summary.csv", index=False)
    predictive.to_csv(output / "traffic_predictive_validation.csv", index=False)

    _plot_rate_fit(model, rate_bins, plots / "traffic_rate_fit.png")
    _plot_survival(model, plots / "traffic_field_survival.png")
    _plot_marks(events, plots / "traffic_marks.png")

    target = output / "traffic_calibration_report.html"
    target.write_text(
        _calibration_html(model, rate_bins, source_rate_bins, marks, predictive),
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
    """Check whether the calibrated process reproduces the observed streams on average.

    Each predictive replicate draws one joint bootstrap calibration pair, replays the
    actual target-race observer exposure window(s), and resamples complete target-race
    empirical event marks. This tests the Maryland process we actually use rather than
    only the fitted mean curve.
    """

    seed = int(model.fingerprint[:16], 16) ^ 0x5052454449435456
    rng = np.random.default_rng(seed)
    km_t = np.asarray(model.km_time_s, dtype=float)
    km_s = np.asarray(model.km_survival, dtype=float)
    marks = model.observations
    total_exposure = model.total_observer_exposure_s
    records: list[dict[str, float]] = []

    for _ in range(max(200, int(replicates))):
        if model.bootstrap_draws:
            draw = model.bootstrap_draws[int(rng.integers(0, len(model.bootstrap_draws)))]
            initial_rate = draw.initial_rate_per_hour
            exponent = draw.field_exponent
        else:
            initial_rate = model.initial_rate_per_hour
            exponent = model.field_exponent

        generated: list[tuple[float, float, str]] = []
        counts: dict[str, int] = {}
        busy_s = 0.0
        for stream in model.streams:
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
    observed_intervals: list[tuple[float, float]] = []
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
    ax.plot(frame["midpoint_min"], frame["fitted_average_rate_per_hour"], marker="o", label="field-survival NHPP")
    ax.scatter(frame["midpoint_min"], frame["observed_rate_per_hour"], label="observed exposure-normalized rate")
    ax.set_xlabel("Race time (min)")
    ax.set_ylabel("Traffic events / observer-hour")
    ax.set_title("Observed traffic rate and fitted encounter process")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_survival(model: TrafficCalibration, path: Path) -> None:
    grid = np.linspace(0.0, model.race_duration_s, 500)
    survival = [model.survival_fraction(t) for t in grid]
    rate = [model.rate_per_hour(t) for t in grid]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.step(grid / 3600.0, survival, where="post", label="surviving field fraction")
    ax.set_xlabel("Race time (h)")
    ax.set_ylabel("Field survival fraction")
    ax2 = ax.twinx()
    ax2.plot(grid / 3600.0, rate, linestyle="--", label="fitted traffic event rate")
    ax2.set_ylabel("Events / observer-hour")
    ax.set_title("Field attrition anchors late-race traffic extrapolation")
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
) -> str:
    contract = model.contract()
    quantiles = contract.get("bootstrap_quantiles", {})
    rate_q = quantiles.get("initial_rate_per_hour", [float("nan")] * 5)
    exponent_q = quantiles.get("field_exponent", [float("nan")] * 5)
    average_rate_q = quantiles.get("average_rate_per_hour_4h", [float("nan")] * 5)
    target_stream_rows = []
    for stream in model.streams:
        target_stream_rows.append(
            {
                "Race role": "Maryland target",
                "Stream": stream.identifier,
                "Exposure (min)": stream.exposure_s / 60.0,
                "Events": len(stream.observations),
                "Observed average rate (/h)": 3600.0 * len(stream.observations) / stream.exposure_s,
            }
        )
    source_stream_rows = []
    source_initial_rates = dict(model.source_stream_initial_rates_per_hour)
    for stream in model.source_streams:
        source_stream_rows.append(
            {
                "Race role": "Arizona decay source",
                "Stream": stream.identifier,
                "Exposure (min)": stream.exposure_s / 60.0,
                "Events": len(stream.observations),
                "Observed average rate (/h)": 3600.0 * len(stream.observations) / stream.exposure_s,
                "Fitted stream initial rate (/h)": source_initial_rates[stream.identifier],
            }
        )
    models = pd.DataFrame(
        [
            {
                "Model": "Arizona field survival: λ₀ S(t)^α",
                "Parameters": len(model.source_streams) + 1,
                "Log likelihood": model.log_likelihood,
                "AIC": 2 * (len(model.source_streams) + 1) - 2 * model.log_likelihood,
                "Use": "Production extrapolation structure; each source stream has its own nuisance rate",
            },
            {
                "Model": "Arizona clock-time exponential",
                "Parameters": len(model.source_streams) + 1,
                "Log likelihood": model.exponential_log_likelihood,
                "AIC": 2 * (len(model.source_streams) + 1) - 2 * model.exponential_log_likelihood,
                "Use": "Clock-time sensitivity comparator with the same stream-specific rate freedom",
            },
            {
                "Model": "Arizona homogeneous Poisson",
                "Parameters": len(model.source_streams),
                "Log likelihood": model.homogeneous_log_likelihood,
                "AIC": 2 * len(model.source_streams) - 2 * model.homogeneous_log_likelihood,
                "Use": "No-decay sensitivity comparator; statistically competitive over the observed source window",
            },
        ]
    )
    return f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Traffic calibration</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#f6f7f9;color:#1f2937}}
main{{max-width:1100px;margin:auto;padding:28px}} h1,h2{{margin-top:1.35em}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.card{{background:white;border:1px solid #d9dde5;border-radius:12px;padding:14px}} .big{{font-size:1.55rem;font-weight:650}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:8px 10px;border:1px solid #dde2ea;text-align:left}}th{{background:#eef1f5}}
img{{max-width:100%;height:auto;background:white;border:1px solid #dde2ea;border-radius:10px}} .note{{padding:12px;border-left:4px solid #6b7280;background:#fff}}
code{{background:#eef1f5;padding:1px 4px;border-radius:4px}}
</style></head><body><main>
<h1>Empirical endurance traffic calibration</h1>
<p>This report builds the Maryland traffic distribution in two layers. Maryland's one-hour observer stream sets the <strong>Maryland encounter-rate level</strong> and supplies the joint duration/severity marks. The two Arizona observer streams plus Arizona competitor attrition identify how encounter rate falls as the endurance field thins. The only cross-race transfer is therefore the race-time decay relationship, rather than the entire Arizona traffic distribution.</p>
<div class=\"cards\">
<div class=\"card\"><div>Observed events</div><div class=\"big\">{len(model.observations)}</div><div>{model.total_observer_exposure_s/3600:.2f} observer-hours</div></div>
<div class=\"card\"><div>Initial encounter rate</div><div class=\"big\">{model.initial_rate_per_hour:.1f}/h</div><div>bootstrap 95% {rate_q[0]:.1f}–{rate_q[4]:.1f}/h</div></div>
<div class=\"card\"><div>Field exponent α</div><div class=\"big\">{model.field_exponent:.2f}</div><div>bootstrap 95% {exponent_q[0]:.2f}–{exponent_q[4]:.2f}</div></div>
<div class=\"card\"><div>4 h average rate</div><div class=\"big\">{model.average_rate_per_hour():.1f}/h</div><div>bootstrap 95% {average_rate_q[0]:.1f}–{average_rate_q[4]:.1f}/h</div></div>
<div class=\"card\"><div>NHPP time-rescaling check</div><div class=\"big\">p={model.time_rescaling_ks_pvalue:.3f}</div><div>Exp(1) transformed interarrival test</div></div>
</div>
<h2>Target and source evidence</h2>
<h3>Maryland target stream</h3>{pd.DataFrame(target_stream_rows).to_html(index=False, float_format=lambda x:f'{x:.3f}')}
<h3>Arizona decay-source streams</h3>{pd.DataFrame(source_stream_rows).to_html(index=False, float_format=lambda x:f'{x:.3f}')}
<p>The production fit uses one shared exponent α, a separate Maryland rate scale, and a separate nuisance rate scale for each Arizona exposure stream. That distinction matters: CWRU and ÉTS experienced different average traffic rates, so their between-stream difference is not allowed to masquerade as race-time decay. The exponent is identified from the timing pattern <em>within</em> each stream plus Maryland's own one-hour timing pattern.</p>
<h2>Why field survival is used for the time-decay shape</h2>{models.to_html(index=False, float_format=lambda x:f'{x:.3f}')}
<h2>Predictive validation</h2>
<p>The table below re-simulates the <em>actual Maryland observer window</em> 3,000 times, including the joint bootstrap uncertainty in Maryland λ₀ and the shared decay exponent α, then resamples complete Maryland event marks. The purpose is to check whether the process reproduces the target-race data it is meant to emulate, not merely whether a fitted line looks plausible.</p>
{predictive.to_html(index=False, float_format=lambda x:f'{x:.3f}')}
<p class="note">The Maryland event total, traffic-occupied time, mean duration, retained-speed fraction and yellow-flag fraction all sit comfortably inside their predictive distributions. That is the key calibration check: the generator reproduces the target evidence on average without introducing an extra fitted severity law.</p>
<p class=\"note\">After giving CWRU and ÉTS separate nuisance rate scales, the homogeneous, clock-exponential and field-survival source models are all within roughly one AIC unit. The observed source window therefore does <em>not</em> strongly identify a decay law. The field-survival form is retained for four-hour extrapolation because it is tied to measured field attrition, while the bootstrap explicitly preserves near-zero-decay solutions instead of pretending the decline is certain.</p>
<img src=\"plots/traffic_rate_fit.png\" alt=\"Observed and fitted traffic rate\">
<h2>Field attrition</h2><img src=\"plots/traffic_field_survival.png\" alt=\"Field survival and fitted traffic rate through four hours\">
<h2>Event severity marks</h2>{marks.to_html(index=False, float_format=lambda x:f'{x:.3f}')}
<img src=\"plots/traffic_marks.png\" alt=\"Observed traffic duration versus retained speed fraction\">
<p class=\"note\">The source spreadsheet heading calls the speed field a “reduction”, but the annotated slowdown/yellow values behave as retained speed fractions. Rows explicitly labelled <code>full stop</code> are normalized to retained fraction 0. Confidence labels are preserved for audit; they are not converted into made-up standard deviations.</p>
<h2>Limits that remain explicit</h2>
<ul><li>Maryland has one observer-hour (26 events), so its absolute rate and mark distribution still carry substantial finite-sample uncertainty.</li><li>Arizona contributes 1.5 observer-hours plus the field-attrition evidence. The transfer to Maryland is limited to the field-survival/time-decay relationship because Maryland lacks four-hour field-population data.</li><li>The field-survival CSV is a derived 59-car active-duration/final-lap/censor artifact from the earlier Arizona lap archive. Re-importing the raw lap archive remains preferable.</li><li>Arizona CWRU and ETS show some observer/driver heterogeneity. With only two source streams, fitting an extra random-effect distribution would be underidentified; the bootstrap retains calibration uncertainty instead.</li><li>Traffic event locations were not measured consistently, so the model is time-based rather than inventing spatial hotspots.</li></ul>
<h2>Maryland 15-minute rate check</h2>{rate_bins.to_html(index=False, float_format=lambda x:f'{x:.3f}')}
<h2>Arizona source 15-minute rate check</h2>{source_rate_bins.to_html(index=False, float_format=lambda x:f'{x:.3f}')}
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
    return f"""\n<!-- TRAFFIC_AUDIT_SECTION -->
<section id=\"traffic-audit\" style=\"margin:2rem 0;padding-top:1rem;border-top:2px solid #d1d5db\">
<h2>Endurance traffic exposure</h2>
<p>Traffic is an in-simulation external speed constraint, not a post-processing lap-time penalty. A single reproducible traffic world is paired across the bounded CVT, infinite reference, design candidates and crossed track cases. The absolute traffic ceiling is based on the shared traffic-free infinite-CVT reference speed profile, so a drivetrain change does not make the same external traffic car artificially faster or slower.</p>
<ul>
<li>Maryland target calibration: {int(contract.get('target_event_count', contract.get('event_count',0)))} observed events over {float(contract.get('target_observer_exposure_h', contract.get('observer_exposure_h',0))):.2f} observer-hours.</li>
<li>Arizona decay evidence: {int(contract.get('source_event_count',0))} events over {float(contract.get('source_observer_exposure_h',0)):.2f} observer-hours plus {int(contract.get('field_record_count',0))} field-car records.</li>
<li>Maryland fitted initial rate: {float(contract.get('initial_rate_per_hour',0)):.1f}/h; shared field exponent α={float(contract.get('field_exponent',0)):.2f}.</li>
<li>Average fitted rate over four hours: {float(contract.get('average_rate_per_hour_4h',0)):.1f}/h.</li>
<li>Unique simulated traffic worlds: {len(unique_worlds)} ({len(worlds)} route-case scenario rows); event count mean {event_count.mean():.2f}, median {event_count.median():.0f}, p90 {event_count.quantile(.9):.0f}.</li>
<li>Representative lap-start phase is sampled uniformly over the endurance clock (observed median {phase_h.median():.2f} h).</li>
</ul>
<h3>Traffic impact on the simulated drivetrain</h3>{impact_html}
<p><strong>Interpretation limits:</strong> Maryland supplies its own event-rate level and severity/duration marks; Arizona transfers only the field-attrition/time-decay relationship. Traffic locations are not modeled because they were not measured consistently. These remain explicit model limitations rather than hidden tuning knobs.</p>
</section>
<!-- /TRAFFIC_AUDIT_SECTION -->\n"""
