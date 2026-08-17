"""Continuous-pace optimism sensitivity for a completed paired design sweep.

This is deliberately a model-form stress test, not a replacement track model.  It
constructs loose continuous upper speed envelopes from valid source laps and asks
whether the *relative* drivetrain conclusion changes when optimistic between-feature
pace is suppressed.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import base64
import html
import json
import math
import os
from pathlib import Path
import shutil
import tomllib
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvt_track_study.bundle.model import TrackBundle
from cvt_track_study.simulation.integrator import run_simulation
from cvt_track_study.simulation.metrics import compare_summaries, summarize_trace
from cvt_track_study.simulation.service import resolve_simulation_cases
from cvt_track_study.simulation.traffic import (
    TrafficReferenceProfile,
    traffic_realization_from_mapping,
)
from cvt_track_study.uncertainty.model import ScenarioDraw


LEVELS: tuple[tuple[str, float, float], ...] = (
    ("loose", 1.20, 2.0),
    ("moderate", 1.10, 1.0),
    ("strong", 1.00, 0.5),
)


@dataclass(frozen=True)
class _DesignSpec:
    identifier: str
    values_si: Mapping[str, float]
    display_values: Mapping[str, float]


@dataclass(frozen=True)
class _Task:
    result_path: str
    replicate: int
    scenario_payload: Mapping[str, Any]
    level: str
    positions_m: tuple[float, ...]
    speeds_mps: tuple[float, ...]
    designs: tuple[_DesignSpec, ...]
    histogram_replicate: int
    histogram_bins_kmh: tuple[float, ...]
    integration_step_s: float


def run_optimism_sensitivity(
    result: str | Path,
    *,
    project: str | Path | None = None,
    workers: int = 1,
    restart: bool = False,
    output_directory: Path | None = None,
    integration_step_ms: float = 5.0,
) -> Path:
    result_path = Path(result).resolve()
    _validate_design_result(result_path)
    if not np.isfinite(integration_step_ms) or integration_step_ms <= 0.0:
        raise ValueError("integration_step_ms must be positive")
    integration_step_s = float(integration_step_ms) / 1000.0
    project_root = _resolve_project_root(result_path, project)
    output = (
        output_directory.resolve()
        if output_directory is not None
        else result_path / "optimism_sensitivity"
    )
    if output.exists():
        if not restart:
            report = output / "optimism_sensitivity_report.html"
            if report.is_file():
                return report
            raise ValueError(
                f"Optimism sensitivity output already exists: {output}. Use --restart to replace it."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)
    plots = output / "plots"
    plots.mkdir()

    nominal_bundle = _nominal_bundle(result_path)
    evidence_root = _locate_matching_evidence(project_root, nominal_bundle)
    envelope_frame, source_hist, source_meta = _build_envelopes(evidence_root, nominal_bundle)
    envelope_frame.to_csv(output / "pace_envelopes.csv", index=False)
    source_meta.to_csv(output / "source_lap_summary.csv", index=False)

    source_rows = pd.read_csv(result_path / "replicate_results.csv")
    nominal_rows = source_rows.copy()
    if "track_case_id" in nominal_rows and (
        nominal_rows["track_case_id"].astype(str) == "nominal"
    ).any():
        nominal_rows = nominal_rows[nominal_rows["track_case_id"].astype(str) == "nominal"].copy()
    if nominal_rows.empty:
        raise ValueError("The design result does not contain a nominal track case for optimism sensitivity.")

    designs = _design_specs(nominal_rows)
    scenarios = _scenario_payloads(result_path, set(nominal_rows["replicate"].astype(int)))
    if not scenarios:
        raise ValueError("No nominal paired scenario draws were found in scenario_draws.jsonl.")

    histogram_replicate = _median_traffic_replicate(nominal_rows)
    hist_bins = tuple(np.linspace(0.0, 90.0, 31))
    envelopes: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
        "control": ((), ()),
        **{
            name: (
                tuple(envelope_frame["s_m"].to_numpy(float)),
                tuple(envelope_frame[f"{name}_ceiling_mps"].to_numpy(float)),
            )
            for name, _, _ in LEVELS
        },
    }
    all_levels = ("control", *[name for name, _, _ in LEVELS])

    tasks = [
        _Task(
            result_path=str(result_path),
            replicate=replicate,
            scenario_payload=payload,
            level=level,
            positions_m=envelopes[level][0],
            speeds_mps=envelopes[level][1],
            designs=tuple(designs),
            histogram_replicate=histogram_replicate,
            histogram_bins_kmh=hist_bins,
            integration_step_s=integration_step_s,
        )
        for level in all_levels
        for replicate, payload in sorted(scenarios.items())
    ]

    print(
        f"Optimism sensitivity: {len(designs)} design(s) × {len(scenarios)} paired world(s) × "
        f"{len(all_levels)} pace assumptions = {len(designs) * len(scenarios) * len(all_levels)} bounded runs "
        f"at {integration_step_ms:g} ms."
    )
    results: list[dict[str, Any]] = []
    histogram_records: list[dict[str, Any]] = []
    max_workers = max(1, min(int(workers), len(tasks)))
    if max_workers == 1:
        for index, task in enumerate(tasks, 1):
            rows, hist = _run_task(task)
            results.extend(rows)
            histogram_records.extend(hist)
            print(f"  completed {index}/{len(tasks)} world-level tasks", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                rows, hist = future.result()
                results.extend(rows)
                histogram_records.extend(hist)
                print(f"  completed {index}/{len(tasks)} world-level tasks", flush=True)

    sensitivity_rows = pd.DataFrame(results)
    sensitivity_rows.to_csv(output / "optimism_sensitivity_results.csv", index=False)

    combined = sensitivity_rows.copy()
    ranking = _ranking_by_level(combined)
    ranking.to_csv(output / "optimism_design_ranking.csv", index=False)
    interaction = _design_interaction_summary(combined)
    interaction.to_csv(output / "optimism_design_interaction.csv", index=False)
    winners = _winner_summary(ranking)
    winners.to_csv(output / "optimism_winner_summary.csv", index=False)
    fidelity = _numerical_fidelity(nominal_rows, combined)
    fidelity.to_csv(output / "optimism_control_numerical_fidelity.csv", index=False)

    hist_frame = pd.DataFrame(histogram_records)
    hist_frame.to_csv(output / "optimism_speed_histograms.csv", index=False)

    figures = _write_figures(
        output=output,
        plots=plots,
        envelope_frame=envelope_frame,
        ranking=ranking,
        interaction=interaction,
        combined=combined,
    )
    histogram_views = _write_histogram_views(
        plots=plots,
        hist_frame=hist_frame,
        source_hist=source_hist,
        designs=designs,
        bins=np.asarray(hist_bins, dtype=float),
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "continuous_pace_optimism_sensitivity",
        "source_design_result": str(result_path),
        "project": str(project_root),
        "paired_world_count": len(scenarios),
        "design_count": len(designs),
        "levels": [
            {"id": name, "p95_multiplier": mult, "additive_margin_mps": add}
            for name, mult, add in LEVELS
        ],
        "source_profile": {
            "per_vehicle_quantile": 0.95,
            "cross_vehicle_combination": "positionwise maximum of each vehicle's p95 profile",
            "cyclic_smoothing_m": 100.0,
            "spacing_m": 5.0,
        },
        "histogram_replicate": histogram_replicate,
        "integration_step_ms": integration_step_ms,
        "histogram_semantics": "one median-traffic paired world; dropdown selects drivetrain design; control and all optimism levels are overlaid",
        "scope": "nominal track reconstruction only; this is a model-form stress test, not a replacement probability model",
    }
    (output / "optimism_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    report = output / "optimism_sensitivity_report.html"
    report.write_text(
        _report_html(
            ranking=ranking,
            interaction=interaction,
            winners=winners,
            figures=figures,
            histogram_views=histogram_views,
            source_meta=source_meta,
            histogram_replicate=histogram_replicate,
            fidelity=fidelity,
            integration_step_ms=integration_step_ms,
        ),
        encoding="utf-8",
    )
    return report


def _validate_design_result(result: Path) -> None:
    required = (
        "replicate_results.csv",
        "scenario_draws.jsonl",
        "run_manifest.json",
        "resolved_inputs/resolved_inputs.toml",
    )
    missing = [name for name in required if not (result / name).is_file()]
    if missing:
        raise ValueError("Not a completed design result; missing: " + ", ".join(missing))
    manifest = json.loads((result / "run_manifest.json").read_text(encoding="utf-8"))
    if str(manifest.get("study_type", "")) != "design_sweep":
        raise ValueError("Optimism sensitivity currently requires a completed design_sweep result.")


def _resolve_project_root(result: Path, project: str | Path | None) -> Path:
    if project is not None:
        root = Path(project).resolve()
    else:
        provenance_path = result / "provenance.json"
        if not provenance_path.is_file():
            raise ValueError("Supply --project because the result has no provenance.json project path.")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        raw = provenance.get("project")
        if not raw:
            raise ValueError("Supply --project because the result provenance has no project path.")
        root = Path(str(raw)).resolve()
    if not (root / "track" / "track.toml").is_file():
        raise FileNotFoundError(f"Project root does not contain track/track.toml: {root}")
    return root


def _nominal_bundle(result: Path) -> TrackBundle:
    path = result / "track_ensemble" / "nominal.json"
    if not path.is_file():
        path = result / "track_bundle.json"
    return TrackBundle(data=json.loads(path.read_text(encoding="utf-8")), path=path)


def _bundle_fingerprint(bundle_path: Path) -> str | None:
    try:
        value = json.loads(bundle_path.read_text(encoding="utf-8"))
        return str(value.get("content_fingerprint_sha256") or "") or None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _locate_matching_evidence(project: Path, bundle: TrackBundle) -> Path:
    fingerprint = str(bundle.data.get("content_fingerprint_sha256") or "")
    candidates: list[Path] = []
    candidates.extend(sorted((project / "results" / "track_robustness").glob("*/nominal_track"), reverse=True))
    candidates.extend(sorted((project / "results" / "track_build").glob("*"), reverse=True))
    for candidate in candidates:
        bundle_path = candidate / "track_bundle.json"
        track_dir = candidate / "track"
        if not bundle_path.is_file() or not (track_dir / "map_matched_points.csv").is_file():
            continue
        if fingerprint and _bundle_fingerprint(bundle_path) == fingerprint:
            return candidate
    # Fall back to the newest nominal evidence package if the content fingerprint is
    # unavailable, but require the same nominal track length within 1 m.
    for candidate in candidates:
        bundle_path = candidate / "track_bundle.json"
        track_dir = candidate / "track"
        if not bundle_path.is_file() or not (track_dir / "map_matched_points.csv").is_file():
            continue
        try:
            other = json.loads(bundle_path.read_text(encoding="utf-8"))
            length = float(other["simulation_contract"]["track_length_m"])
        except Exception:
            continue
        if abs(length - bundle.track_length_m) <= 1.0:
            return candidate
    raise FileNotFoundError(
        "Could not find map-matched valid-lap artifacts matching the design sweep's nominal track. "
        "Run track robustness/build-track with the current project first."
    )


def _build_envelopes(
    evidence_root: Path,
    bundle: TrackBundle,
    *,
    spacing_m: float = 5.0,
    smoothing_m: float = 100.0,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    track = evidence_root / "track"
    points = pd.read_csv(track / "map_matched_points.csv")
    laps = pd.read_csv(track / "lap_quality.csv")
    maximum_map_error_m = math.inf
    resolved = evidence_root / "configuration" / "resolved_inputs.toml"
    if resolved.is_file():
        try:
            raw = tomllib.loads(resolved.read_text(encoding="utf-8"))
            maximum_map_error_m = float(raw.get("track", {}).get("reconstruction", {}).get("maximum_map_error_m", math.inf))
        except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError):
            maximum_map_error_m = math.inf
    if "map_error_m" in points and np.isfinite(maximum_map_error_m):
        points = points[pd.to_numeric(points["map_error_m"], errors="coerce") <= maximum_map_error_m].copy()
    valid = laps[
        laps["analysis_valid"].astype(bool)
        & laps["use_for_gate_evidence"].astype(bool)
    ][["lap_id", "run_id", "vehicle_id", "duration_s"]]
    points = points.merge(valid[["lap_id", "run_id", "vehicle_id"]], on=["lap_id", "run_id", "vehicle_id"], how="inner")
    length = float(bundle.track_length_m)
    grid = np.arange(0.0, length, spacing_m)
    by_vehicle: dict[str, list[np.ndarray]] = {}
    for (vehicle, run_id, lap_id), group in points.groupby(["vehicle_id", "run_id", "lap_id"], sort=False):
        s = pd.to_numeric(group["s_m"], errors="coerce").to_numpy(float) % length
        speed = pd.to_numeric(group["speed_analysis_mps"], errors="coerce").to_numpy(float)
        good = np.isfinite(s) & np.isfinite(speed) & (speed >= 0.0)
        s, speed = s[good], speed[good]
        if len(s) < 5:
            continue
        order = np.argsort(s)
        s, speed = s[order], speed[order]
        unique_s, inverse = np.unique(s, return_inverse=True)
        unique_speed = np.zeros_like(unique_s)
        counts = np.zeros_like(unique_s)
        np.add.at(unique_speed, inverse, speed)
        np.add.at(counts, inverse, 1.0)
        unique_speed /= np.maximum(counts, 1.0)
        curve = np.interp(
            grid,
            np.r_[unique_s - length, unique_s, unique_s + length],
            np.r_[unique_speed, unique_speed, unique_speed],
        )
        by_vehicle.setdefault(str(vehicle), []).append(curve)
    if len(by_vehicle) < 2:
        raise ValueError("Optimism sensitivity requires valid-lap speed evidence from at least two vehicles.")
    profiles = {
        vehicle: np.quantile(np.vstack(curves), 0.95, axis=0)
        for vehicle, curves in by_vehicle.items()
        if curves
    }
    source_upper = np.maximum.reduce([profiles[key] for key in sorted(profiles)])
    window = max(3, int(round(smoothing_m / spacing_m)))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    extended = np.r_[source_upper[-pad:], source_upper, source_upper[:pad]]
    smooth = np.convolve(extended, np.ones(window) / window, mode="valid")
    frame = pd.DataFrame({"s_m": np.r_[grid, length], "source_upper_p95_mps": np.r_[source_upper, source_upper[0]], "source_upper_p95_kmh": np.r_[source_upper * 3.6, source_upper[0] * 3.6]})
    for vehicle, profile in sorted(profiles.items()):
        frame[f"{vehicle}_p95_mps"] = np.r_[profile, profile[0]]
        frame[f"{vehicle}_p95_kmh"] = np.r_[profile * 3.6, profile[0] * 3.6]
    for name, multiplier, additive in LEVELS:
        ceiling = smooth * multiplier + additive
        frame[f"{name}_ceiling_mps"] = np.r_[ceiling, ceiling[0]]
        frame[f"{name}_ceiling_kmh"] = np.r_[ceiling * 3.6, ceiling[0] * 3.6]

    # Valid-lap raw-source histogram profiles, time weighted.  These are used only
    # as a sanity overlay and never to fit the vehicle simulation.
    source_hist: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    bins = np.linspace(0.0, 90.0, 31)
    for vehicle, vehicle_points in points.groupby("vehicle_id", sort=True):
        speeds: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for _, lap_points in vehicle_points.groupby(["run_id", "lap_id"], sort=False):
            speed = pd.to_numeric(lap_points["speed_analysis_mps"], errors="coerce").to_numpy(float) * 3.6
            elapsed = pd.to_numeric(lap_points["elapsed_lap_s"], errors="coerce").to_numpy(float)
            good = np.isfinite(speed) & np.isfinite(elapsed) & (speed >= 0.0) & (speed <= 90.0)
            speed, elapsed = speed[good], elapsed[good]
            if len(speed) < 2:
                continue
            order = np.argsort(elapsed)
            speed, elapsed = speed[order], elapsed[order]
            dt = np.diff(elapsed, append=elapsed[-1] + np.median(np.diff(elapsed)))
            dt = np.where((dt > 0.0) & np.isfinite(dt), dt, 0.0)
            speeds.append(speed)
            weights.append(dt)
        if speeds:
            speed_all = np.concatenate(speeds)
            weight_all = np.concatenate(weights)
            hist, _ = np.histogram(speed_all, bins=bins, weights=weight_all)
            hist = hist / max(hist.sum(), 1.0)
            source_hist[str(vehicle)] = (bins, hist)

    meta = (
        valid.groupby("vehicle_id", as_index=False)
        .agg(valid_lap_count=("lap_id", "count"), median_lap_time_s=("duration_s", "median"), fastest_lap_time_s=("duration_s", "min"))
        .sort_values("vehicle_id")
    )
    return frame, source_hist, meta


def _design_specs(rows: pd.DataFrame) -> list[_DesignSpec]:
    designs: list[_DesignSpec] = []
    for design_id, group in rows.groupby("design_id", sort=False):
        values_si = json.loads(str(group["design_values_si_json"].iloc[0]))
        display = json.loads(str(group["design_values_json"].iloc[0]))
        designs.append(
            _DesignSpec(
                identifier=str(design_id),
                values_si={str(k): float(v) for k, v in values_si.items()},
                display_values={str(k): float(v) for k, v in display.items()},
            )
        )
    return designs


def _scenario_payloads(result: Path, nominal_replicates: set[int]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    with (result / "scenario_draws.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            replicate = int(payload["replicate"])
            if replicate in nominal_replicates:
                output[replicate] = payload
    return output


def _scenario_from_payload(payload: Mapping[str, Any]) -> ScenarioDraw:
    return ScenarioDraw(
        replicate=int(payload["replicate"]),
        seed=int(payload["seed"]),
        sampling_mode=str(payload.get("sampling_mode", "uncertainty")),
        quantity_values_si=dict(payload.get("quantity_values_si", {})),
        choice_values=dict(payload.get("choice_values", {})),
        gate_target_speeds_mps=dict(payload.get("gate_target_speeds_mps", {})),
        independently_sampled_gate_ids=tuple(payload.get("independently_sampled_gate_ids", ())),
        sampling_design=str(payload.get("sampling_design", "latin_hypercube_rank_correlated")),
        traffic_realization=payload.get("traffic_realization"),
    )


def _resolved_context(result: Path) -> tuple[str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], TrackBundle]:
    data = tomllib.loads((result / "resolved_inputs" / "resolved_inputs.toml").read_text(encoding="utf-8"))
    manifest = json.loads((result / "run_manifest.json").read_text(encoding="utf-8"))
    study_name = str(manifest.get("study_name") or data.get("active_study"))
    current = data["studies"][study_name]
    base_name = str(current.get("base_case", {}).get("study") or study_name)
    base_study = data["studies"][base_name]
    vehicle_id = str(current["study"]["vehicle_id"])
    vehicle = data["vehicles"][vehicle_id]
    track = data["track"]
    bundle = _nominal_bundle(result)
    return vehicle_id, vehicle, base_study, track, bundle


def _run_task(task: _Task) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result = Path(task.result_path)
    vehicle_id, vehicle, base_study, track_raw, bundle = _resolved_context(result)
    scenario = _scenario_from_payload(task.scenario_payload)
    pace = (
        None
        if task.level == "control"
        else (np.asarray(task.positions_m, dtype=float), np.asarray(task.speeds_mps, dtype=float))
    )
    traffic = traffic_realization_from_mapping(scenario.traffic_realization)
    first = task.designs[0]
    _, reference_case, settings, runtime_track = resolve_simulation_cases(
        vehicle_id=vehicle_id,
        vehicle_raw=vehicle,
        study_raw=base_study,
        track_raw=track_raw,
        bundle=bundle,
        quantity_values_si=scenario.quantity_values_si,
        choice_values=scenario.choice_values,
        gate_target_speeds_mps=scenario.gate_target_speeds_mps,
        design_values_si=first.values_si,
        shared_reference=True,
    )
    settings = replace(
        settings,
        integration_step_s=task.integration_step_s,
        report_step_s=max(float(settings.report_step_s), 0.05),
    )
    reference_free = run_simulation(case=reference_case, track=runtime_track, settings=settings, pace_envelope=pace)
    traffic_reference = TrafficReferenceProfile.from_trace(reference_free, spacing_m=1.0)
    reference = run_simulation(
        case=reference_case,
        track=runtime_track,
        settings=settings,
        traffic=traffic,
        traffic_reference=(traffic_reference if traffic is not None else None),
        pace_envelope=pace,
    )
    reference_summary = summarize_trace(reference, target_engine_rpm=reference_case.engine.target_rpm, target_power_w=reference_case.engine.target_power_w)
    rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    bins = np.asarray(task.histogram_bins_kmh, dtype=float)
    for design in task.designs:
        bounded_case, _, settings_b, runtime_b = resolve_simulation_cases(
            vehicle_id=vehicle_id,
            vehicle_raw=vehicle,
            study_raw=base_study,
            track_raw=track_raw,
            bundle=bundle,
            quantity_values_si=scenario.quantity_values_si,
            choice_values=scenario.choice_values,
            gate_target_speeds_mps=scenario.gate_target_speeds_mps,
            design_values_si=design.values_si,
            shared_reference=True,
        )
        settings_b = replace(
            settings_b,
            integration_step_s=task.integration_step_s,
            report_step_s=max(float(settings_b.report_step_s), 0.05),
        )
        trace = run_simulation(
            case=bounded_case,
            track=runtime_b,
            settings=settings_b,
            traffic=traffic,
            traffic_reference=(traffic_reference if traffic is not None else None),
            pace_envelope=pace,
        )
        summary = summarize_trace(trace, target_engine_rpm=bounded_case.engine.target_rpm, target_power_w=bounded_case.engine.target_power_w)
        comparison = compare_summaries(summary, reference_summary)
        record: dict[str, Any] = {
            "level": task.level,
            "replicate": scenario.replicate,
            "base_draw_id": task.scenario_payload.get("base_draw_id"),
            "design_id": design.identifier,
            "bounded_completed": bool(summary["completed"]),
            "bounded_lap_time_s": float(summary["lap_time_s"]),
            "infinite_reference_lap_time_s": float(reference_summary["lap_time_s"]),
            "lap_time_penalty_vs_infinite_s": float(comparison["lap_time_penalty_vs_infinite_s"]),
            "finite_ratio_opportunity_loss_energy_kj": float(comparison["finite_ratio_opportunity_loss_energy_kj"]),
            "bounded_time_minimum_ratio_s": float(summary.get("time_minimum_ratio_s", math.nan)),
            "bounded_time_maximum_ratio_s": float(summary.get("time_maximum_ratio_s", math.nan)),
            "bounded_maximum_speed_kmh": float(summary.get("maximum_speed_kmh", math.nan)),
        }
        record.update({f"design::{key}": value for key, value in design.display_values.items()})
        rows.append(record)
        if scenario.replicate == task.histogram_replicate:
            hist = _trace_hist(trace, bins)
            for index, fraction in enumerate(hist):
                hist_rows.append(
                    {
                        "design_id": design.identifier,
                        "level": task.level,
                        "bin_left_kmh": float(bins[index]),
                        "bin_right_kmh": float(bins[index + 1]),
                        "time_fraction": float(fraction),
                    }
                )
    return rows, hist_rows


def _trace_hist(trace: Any, bins: np.ndarray) -> np.ndarray:
    speed = np.asarray(trace.numeric["vehicle_speed_kmh"], dtype=float)
    time = np.asarray(trace.numeric["time_s"], dtype=float)
    if len(speed) < 2:
        return np.zeros(len(bins) - 1)
    dt = np.diff(time, append=time[-1] + np.median(np.diff(time)))
    good = np.isfinite(speed) & np.isfinite(dt) & (dt > 0.0) & (speed >= bins[0]) & (speed <= bins[-1])
    hist, _ = np.histogram(speed[good], bins=bins, weights=dt[good])
    return hist / max(hist.sum(), 1.0)


def _median_traffic_replicate(rows: pd.DataFrame) -> int:
    unique = (
        rows[["replicate", "reference_traffic_penalty_s"]]
        .drop_duplicates("replicate")
        .assign(reference_traffic_penalty_s=lambda frame: pd.to_numeric(frame["reference_traffic_penalty_s"], errors="coerce"))
        .dropna()
        .sort_values("reference_traffic_penalty_s")
        .reset_index(drop=True)
    )
    if unique.empty:
        return int(rows["replicate"].drop_duplicates().iloc[len(rows["replicate"].drop_duplicates()) // 2])
    return int(unique.iloc[len(unique) // 2]["replicate"])


def _ranking_by_level(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    axis_columns = [column for column in rows.columns if column.startswith("design::")]
    for level, level_rows in rows.groupby("level", sort=False):
        work = level_rows.copy()
        work["_lap"] = pd.to_numeric(work["bounded_lap_time_s"], errors="coerce")
        work.loc[~work["bounded_completed"].astype(bool), "_lap"] = np.inf
        best = work.groupby("replicate")["_lap"].transform("min")
        work["_regret"] = work["_lap"] - best
        for design_id, group in work.groupby("design_id", sort=False):
            regret = pd.to_numeric(group["_regret"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            rec: dict[str, Any] = {
                "level": level,
                "design_id": design_id,
                "world_count": int(group["replicate"].nunique()),
                "completion_fraction": float(group["bounded_completed"].astype(bool).mean()),
                "lap_time_median_s": float(pd.to_numeric(group["bounded_lap_time_s"], errors="coerce").median()),
                "paired_regret_median_s": float(regret.median()) if len(regret) else math.inf,
                "paired_regret_p90_s": float(regret.quantile(0.90)) if len(regret) else math.inf,
                "finite_ratio_penalty_median_s": float(pd.to_numeric(group["lap_time_penalty_vs_infinite_s"], errors="coerce").median()),
                "minimum_ratio_time_median_s": float(pd.to_numeric(group.get("bounded_time_minimum_ratio_s"), errors="coerce").median()),
            }
            for column in axis_columns:
                rec[column] = float(group[column].iloc[0])
            records.append(rec)
    return pd.DataFrame(records)


def _design_interaction_summary(rows: pd.DataFrame) -> pd.DataFrame:
    control = rows[rows["level"] == "control"][["replicate", "design_id", "bounded_lap_time_s"]].rename(columns={"bounded_lap_time_s": "control_lap_s"})
    records: list[dict[str, Any]] = []
    for level in [name for name, _, _ in LEVELS]:
        stressed = rows[rows["level"] == level][["replicate", "design_id", "bounded_lap_time_s"]].rename(columns={"bounded_lap_time_s": "stress_lap_s"})
        merged = stressed.merge(control, on=["replicate", "design_id"], how="inner")
        merged["added_lap_time_s"] = pd.to_numeric(merged["stress_lap_s"], errors="coerce") - pd.to_numeric(merged["control_lap_s"], errors="coerce")
        for design_id, group in merged.groupby("design_id", sort=False):
            value = group["added_lap_time_s"].dropna()
            records.append(
                {
                    "level": level,
                    "design_id": design_id,
                    "world_count": int(group["replicate"].nunique()),
                    "added_lap_time_mean_s": float(value.mean()),
                    "added_lap_time_median_s": float(value.median()),
                    "added_lap_time_p10_s": float(value.quantile(0.10)),
                    "added_lap_time_p90_s": float(value.quantile(0.90)),
                }
            )
    return pd.DataFrame(records)



def _numerical_fidelity(source_rows: pd.DataFrame, sensitivity_rows: pd.DataFrame) -> pd.DataFrame:
    control = sensitivity_rows[sensitivity_rows["level"] == "control"][["replicate", "design_id", "bounded_lap_time_s"]].rename(columns={"bounded_lap_time_s": "sensitivity_control_lap_s"})
    source = source_rows[["replicate", "design_id", "bounded_lap_time_s"]].rename(columns={"bounded_lap_time_s": "source_1ms_lap_s"})
    merged = control.merge(source, on=["replicate", "design_id"], how="inner")
    merged["lap_time_delta_s"] = pd.to_numeric(merged["sensitivity_control_lap_s"], errors="coerce") - pd.to_numeric(merged["source_1ms_lap_s"], errors="coerce")
    common_by_world = merged.groupby("replicate")["lap_time_delta_s"].transform("median")
    merged["relative_numerical_bias_s"] = merged["lap_time_delta_s"] - common_by_world
    records: list[dict[str, Any]] = []
    for design_id, group in merged.groupby("design_id", sort=False):
        delta = group["lap_time_delta_s"].dropna()
        relative = group["relative_numerical_bias_s"].dropna()
        records.append({
            "design_id": design_id,
            "paired_world_count": int(group["replicate"].nunique()),
            "control_minus_source_mean_s": float(delta.mean()),
            "control_minus_source_median_s": float(delta.median()),
            "maximum_abs_control_minus_source_s": float(delta.abs().max()),
            "maximum_abs_relative_numerical_bias_s": float(relative.abs().max()),
            "median_relative_numerical_bias_s": float(relative.median()),
        })
    return pd.DataFrame(records)

def _winner_summary(ranking: pd.DataFrame) -> pd.DataFrame:
    records = []
    for level, group in ranking.groupby("level", sort=False):
        ordered = group.sort_values(["completion_fraction", "paired_regret_median_s", "lap_time_median_s"], ascending=[False, True, True])
        row = ordered.iloc[0]
        records.append({"level": level, "preferred_design_id": row["design_id"], "paired_regret_median_s": row["paired_regret_median_s"], "lap_time_median_s": row["lap_time_median_s"]})
    return pd.DataFrame(records)


def _write_figures(
    *,
    output: Path,
    plots: Path,
    envelope_frame: pd.DataFrame,
    ranking: pd.DataFrame,
    interaction: pd.DataFrame,
    combined: pd.DataFrame,
) -> list[tuple[Path, str]]:
    figures: list[tuple[Path, str]] = []
    # Envelope profile.
    fig, ax = plt.subplots(figsize=(11, 5.4))
    vehicle_cols = [column for column in envelope_frame.columns if column.endswith("_p95_kmh") and column != "source_upper_p95_kmh"]
    for column in vehicle_cols:
        ax.plot(envelope_frame["s_m"], envelope_frame[column], alpha=0.55, linewidth=1, label=column.replace("_p95_kmh", " p95"))
    for level, _, _ in LEVELS:
        ax.plot(envelope_frame["s_m"], envelope_frame[f"{level}_ceiling_kmh"], linewidth=1.6, label=level)
    ax.set_xlabel("Track position s [m]")
    ax.set_ylabel("Speed [km/h]")
    ax.set_title("Source-derived continuous pace stress envelopes")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    path = plots / "pace_envelope_profiles.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append((path, "Valid-lap per-vehicle p95 pace and the three intentionally conservative continuous upper-speed stress envelopes."))

    # Added time by design / level.
    if not interaction.empty:
        matrix = interaction.pivot_table(index="design_id", columns="level", values="added_lap_time_median_s", aggfunc="median")
        ordered = matrix.mean(axis=1).sort_values().index
        matrix = matrix.reindex(ordered)
        fig, ax = plt.subplots(figsize=(10, max(5.5, 0.34 * len(matrix))))
        image = ax.imshow(matrix.to_numpy(float), aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
        ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
        ax.set_xlabel("Optimism stress level")
        ax.set_ylabel("Design")
        ax.set_title("Added lap time from continuous pace suppression")
        fig.colorbar(image, ax=ax, label="Added lap time [s]")
        fig.tight_layout()
        path = plots / "optimism_added_time_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figures.append((path, "How much absolute lap time each pace envelope adds to each design. Similar values across a column indicate common-mode track optimism."))

    # Ranking heatmaps by level for the two-dimensional grid.
    axis_cols = [column for column in ranking.columns if column.startswith("design::")]
    if len(axis_cols) == 2:
        x_col, y_col = axis_cols
        levels = ["control", *[name for name, _, _ in LEVELS]]
        for level in levels:
            group = ranking[ranking["level"] == level]
            matrix = group.pivot_table(index=y_col, columns=x_col, values="paired_regret_median_s", aggfunc="median").sort_index().sort_index(axis=1)
            if matrix.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 5.8))
            data = matrix.to_numpy(float)
            image = ax.imshow(data, aspect="auto", interpolation="nearest")
            ax.set_xticks(np.arange(len(matrix.columns)), [f"{value:g}" for value in matrix.columns])
            ax.set_yticks(np.arange(len(matrix.index)), [f"{value:g}" for value in matrix.index])
            ax.set_xlabel(x_col.replace("design::", ""))
            ax.set_ylabel(y_col.replace("design::", ""))
            ax.set_title(f"Median paired regret — {level}")
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if np.isfinite(data[i, j]):
                        ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", bbox={"boxstyle":"round,pad=0.12","facecolor":"white","edgecolor":"none","alpha":0.7})
            fig.colorbar(image, ax=ax, label="Median paired regret [s]")
            fig.tight_layout()
            path = plots / f"optimism_regret_{level}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            figures.append((path, f"The same paired design-regret surface under the {level} pace assumption."))
    return figures


def _write_histogram_views(
    *,
    plots: Path,
    hist_frame: pd.DataFrame,
    source_hist: Mapping[str, tuple[np.ndarray, np.ndarray]],
    designs: Sequence[_DesignSpec],
    bins: np.ndarray,
) -> list[dict[str, str]]:
    views: list[dict[str, str]] = []
    centers = 0.5 * (bins[:-1] + bins[1:])
    for design in designs:
        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        for vehicle, (_, hist) in sorted(source_hist.items()):
            ax.step(centers, hist, where="mid", linewidth=2, label=f"{vehicle} valid-lap telemetry")
        group = hist_frame[hist_frame["design_id"] == design.identifier]
        for level in ("control", "loose", "moderate", "strong"):
            level_group = group[group["level"] == level].sort_values("bin_left_kmh")
            if level_group.empty:
                continue
            ax.plot(
                0.5 * (level_group["bin_left_kmh"].to_numpy(float) + level_group["bin_right_kmh"].to_numpy(float)),
                level_group["time_fraction"].to_numpy(float),
                marker="o" if level != "control" else None,
                linewidth=1.5,
                label=f"simulation — {level}",
            )
        ax.set_xlabel("Vehicle speed [km/h]")
        ax.set_ylabel("Fraction of time")
        ax.set_title(f"Speed-occupancy sanity check — {design.identifier}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        filename = "speed_hist_" + _slug(design.identifier) + ".png"
        path = plots / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        views.append({"design_id": design.identifier, "path": str(path), "caption": "One representative median-traffic paired world. The dropdown changes drivetrain design; all optimism levels are overlaid against valid-lap Cornell/McMaster telemetry."})
    return views


def _report_html(
    *,
    ranking: pd.DataFrame,
    interaction: pd.DataFrame,
    winners: pd.DataFrame,
    figures: Sequence[tuple[Path, str]],
    histogram_views: Sequence[Mapping[str, str]],
    source_meta: pd.DataFrame,
    histogram_replicate: int,
    fidelity: pd.DataFrame,
    integration_step_ms: float,
) -> str:
    control = ranking[ranking["level"] == "control"]
    strong = ranking[ranking["level"] == "strong"]
    control_winner = str(winners[winners["level"] == "control"]["preferred_design_id"].iloc[0]) if (winners["level"] == "control").any() else "unresolved"
    strong_winner = str(winners[winners["level"] == "strong"]["preferred_design_id"].iloc[0]) if (winners["level"] == "strong").any() else "unresolved"
    strong_span = math.nan
    fidelity_absolute = float(pd.to_numeric(fidelity.get("maximum_abs_control_minus_source_s"), errors="coerce").max()) if not fidelity.empty else math.nan
    fidelity_relative = float(pd.to_numeric(fidelity.get("maximum_abs_relative_numerical_bias_s"), errors="coerce").max()) if not fidelity.empty else math.nan
    if not interaction.empty:
        values = interaction[interaction["level"] == "strong"]["added_lap_time_median_s"]
        if len(values):
            strong_span = float(values.max() - values.min())
    options = []
    hist_html = []
    for index, view in enumerate(histogram_views):
        key = f"hist-{index}"
        options.append(f'<option value="{key}">{html.escape(str(view["design_id"]))}</option>')
        hidden = "" if index == 0 else ' hidden="hidden"'
        hist_html.append(
            f'<div data-optimism-hist-view="{key}"{hidden}>'
            + _figure(Path(str(view["path"])), str(view["caption"]))
            + "</div>"
        )
    figure_html = "".join(_figure(path, caption) for path, caption in figures)
    winner_table = winners.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}")
    source_table = source_meta.to_html(index=False, border=0, float_format=lambda x: f"{x:.2f}")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Design optimism sensitivity</title><style>
body{{font-family:Inter,Arial,sans-serif;max-width:1280px;margin:0 auto;padding:28px;color:#20242a;line-height:1.5}}
h1,h2,h3{{line-height:1.2}} h2{{margin-top:34px;border-bottom:1px solid #ccc;padding-bottom:6px}}
.card{{border:1px solid #d5d9df;border-radius:9px;padding:14px 16px;margin:12px 0;background:#f7f8fa}} .good{{border-left:5px solid #2c7a4b}} .note{{border-left:5px solid #3576a8}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}} .metric{{font-size:1.35rem;font-weight:700}} .label{{font-size:.82rem;text-transform:uppercase;color:#59636e}}
figure{{margin:20px 0 30px}} img{{max-width:100%;border:1px solid #ddd;border-radius:7px}} figcaption{{font-size:.92rem;color:#59636e;margin-top:6px}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}} th,td{{border:1px solid #ddd;padding:7px 9px;text-align:left}} th{{background:#f0f2f5}}
select{{padding:7px 9px;min-width:320px}} .small{{color:#59636e;font-size:.92rem}}
</style></head><body>
<h1>Design-sweep optimism sensitivity</h1>
<div class="card note"><strong>Purpose.</strong> This is a model-form stress test for the completed paired design sweep. It does not replace the track model or fit the car to Cornell/McMaster pace. It asks whether suppressing the simulator's optimistic between-feature speed changes the drivetrain conclusion. All four sensitivity assumptions are rerun at {integration_step_ms:g} ms, including a no-envelope control, so stress effects are compared at identical numerical resolution; the control is separately checked against the source 1 ms sweep.</div>
<div class="grid">
<div class="card"><div class="label">Control winner</div><div class="metric">{html.escape(control_winner)}</div></div>
<div class="card"><div class="label">Strong-stress winner</div><div class="metric">{html.escape(strong_winner)}</div></div>
<div class="card"><div class="label">Strong stress design-dependence</div><div class="metric">{strong_span:.3f} s</div><div>span in median added lap time across designs</div></div>
<div class="card"><div class="label">Paired worlds</div><div class="metric">{int(control['world_count'].max()) if not control.empty else 0}</div><div>nominal reconstruction only</div></div>
<div class="card"><div class="label">{integration_step_ms:g} ms relative numerical bias</div><div class="metric">{fidelity_relative:.3f} s</div><div>largest design-dependent control bias after removing each world's common integration shift; largest absolute control shift {fidelity_absolute:.3f} s</div></div>
</div>
<h2>Source evidence and stress definition</h2>
{source_table}
<p>Each vehicle first gets its own valid-lap p95 spatial pace profile. The faster source at each position defines a conservative upper demonstrated pace before smoothing. The three sensitivity levels then add progressively less headroom: loose = 1.20× + 2.0 m/s, moderate = 1.10× + 1.0 m/s, strong = 1.00× + 0.5 m/s.</p>
{figure_html}
<h2>Preferred design by optimism level</h2>
{winner_table}
<p class="small">The exact cell can move inside a near-tied plateau without implying a meaningful engineering reversal. Read the regret surfaces and the common-mode added-time heatmap together.</p>
<h2>Speed-occupancy sanity check</h2>
<div class="card note"><strong>How to read this.</strong> The dropdown selects the <em>design-sweep candidate</em>. For that design, one representative median-traffic world (replicate {histogram_replicate}) is shown under control, loose, moderate and strong pace assumptions, all against the valid-lap McMaster/Cornell telemetry distributions. This is deliberately a sanity check, not a fit target.</div>
<label for="optimism-design-select"><strong>Design candidate:</strong></label>
<select id="optimism-design-select">{''.join(options)}</select>
{''.join(hist_html)}
<script>
(function(){{const s=document.getElementById('optimism-design-select');function a(){{document.querySelectorAll('[data-optimism-hist-view]').forEach(n=>n.hidden=n.getAttribute('data-optimism-hist-view')!==s.value);}}s.addEventListener('change',a);a();}})();
</script>
</body></html>"""


def _figure(path: Path, caption: str) -> str:
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def _slug(value: str) -> str:
    return "-".join(part for part in "".join(char.lower() if char.isalnum() else "-" for char in value).split("-") if part) or "design"
