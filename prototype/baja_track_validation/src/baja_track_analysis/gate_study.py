from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .gps_core import (
    Centreline,
    build_lap_table,
    build_track_profile,
    detect_gate_crossings,
    load_and_clean_gps,
    map_match_laps,
)
from .metrics import extract_event_passes, summarize_event_passes
from .pipeline import AnalysisResult, run_analysis
from .telemetry import attach_optional_telemetry


@dataclass
class GateStudyResult:
    primary: AnalysisResult
    dataset_summary: pd.DataFrame
    laps: pd.DataFrame
    event_passes: pd.DataFrame
    event_summary: pd.DataFrame
    speed_gates: pd.DataFrame
    measurement_locations: pd.DataFrame
    bundle: dict[str, object]


def _with_dataset_context(
    passes: pd.DataFrame,
    laps: pd.DataFrame,
    *,
    dataset_id: str,
    vehicle_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_laps = laps.copy()
    out_laps["source_lap_id"] = out_laps["lap_id"]
    out_laps["dataset_id"] = dataset_id
    out_laps["vehicle_id"] = vehicle_id
    out_laps["global_lap_id"] = out_laps["source_lap_id"].map(
        lambda value: f"{dataset_id}__lap_{int(value):03d}"
    )

    duration = out_laps.set_index("source_lap_id")["duration_s"]
    median_speed = out_laps.set_index("source_lap_id")["median_speed_kmh"]
    out = passes.copy()
    out["source_lap_id"] = out["lap_id"]
    out["dataset_id"] = dataset_id
    out["vehicle_id"] = vehicle_id
    out["lap_duration_s"] = out["source_lap_id"].map(duration)
    out["lap_median_speed_kmh"] = out["source_lap_id"].map(median_speed)
    out["case_id"] = dataset_id + "__" + out["case_id"].astype(str)
    out["lap_id"] = out["source_lap_id"].map(
        lambda value: f"{dataset_id}__lap_{int(value):03d}"
    )
    return out, out_laps


def _secondary_dataset_passes(
    gps_csv: Path,
    *,
    dataset_id: str,
    vehicle_id: str,
    primary: AnalysisResult,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    gps, cleaning = load_and_clean_gps(gps_csv, config.gps)
    gps, telemetry_channels = attach_optional_telemetry(gps, gps_csv)
    cleaning["optional_telemetry_channels"] = ";".join(telemetry_channels)
    centreline = primary.centreline
    x, y = centreline.frame.to_xy(gps["lat"], gps["lon"])
    gps["x_m"] = x
    gps["y_m"] = y

    gate = primary.projected_definitions[
        primary.projected_definitions["name"].str.casefold()
        == config.gps.lap_gate_name.casefold()
    ].iloc[0]
    crossings = detect_gate_crossings(
        gps,
        centreline.frame,
        float(gate["latitude"]),
        float(gate["longitude"]),
        config.gps,
    )
    if len(crossings) < 3:
        raise ValueError(
            f"Dataset {dataset_id!r} found only {len(crossings)} lap-gate visits"
        )
    laps = build_lap_table(gps, crossings, config.gps)
    matched, laps = map_match_laps(gps, laps, centreline)
    _, lap_profiles = build_track_profile(matched, laps, centreline, config.gps)
    sample_period = float(cleaning["median_sample_period_s"])
    passes = extract_event_passes(
        matched,
        laps,
        primary.analysis_features,
        lap_profiles,
        centreline,
        config,
        sample_period,
    )
    passes, laps = _with_dataset_context(
        passes,
        laps,
        dataset_id=dataset_id,
        vehicle_id=vehicle_id,
    )
    matched["dataset_id"] = dataset_id
    matched["vehicle_id"] = vehicle_id
    return passes, laps, cleaning, matched


def _circular_s(value: float, length_m: float) -> float:
    return float(value % length_m)


def _score_speed_gates(
    passes: pd.DataFrame,
    features: pd.DataFrame,
    projected: pd.DataFrame,
    config: PipelineConfig,
    track_length_m: float,
) -> pd.DataFrame:
    cfg = config.gate
    eligible = passes[passes["aggregate_eligible"].astype(bool)].copy()
    rows: list[dict[str, object]] = []
    for _, feature in features.sort_values("sequence").iterrows():
        group_id = str(feature["analysis_group_id"])
        group = eligible[eligible["analysis_group_id"].astype(str) == group_id].copy()
        role = str(feature.get("analysis_role", "track_event"))
        is_turn = role == "turn_context"
        target_column = "event_min_speed_kmh" if is_turn else "entry_speed_kmh"
        measurement_kind = "turn_control_point" if is_turn else "physical_entry"
        target = pd.to_numeric(group.get(target_column), errors="coerce").dropna()
        n = int(len(target))

        if n:
            median_target = float(target.median())
            std_target = float(target.std(ddof=1)) if n > 1 else 0.0
            target_cv = std_target / max(median_target, 1.0e-9)
            percentiles = target.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        else:
            median_target = float("nan")
            target_cv = float("nan")
            percentiles = pd.Series({q: np.nan for q in [0.10, 0.25, 0.50, 0.75, 0.90]})

        if is_turn:
            reduction = pd.to_numeric(
                group.get("ordered_entry_to_min_change_kmh"), errors="coerce"
            )
            braking_evidence = float(
                (reduction > cfg.meaningful_speed_reduction_kmh).mean()
            ) if len(group) else 0.0
            distance_to_control = float(
                pd.to_numeric(group.get("distance_to_min_m"), errors="coerce").median()
            ) if len(group) else 0.0
            gate_s_m = _circular_s(
                float(feature["event_start_s_m"]) + distance_to_control,
                track_length_m,
            )
        else:
            approach_accel = pd.to_numeric(
                group.get("approach_acceleration_mps2"), errors="coerce"
            )
            approach_drop = pd.to_numeric(group.get("approach_speed_kmh"), errors="coerce") - pd.to_numeric(
                group.get("entry_speed_kmh"), errors="coerce"
            )
            evidence = (approach_accel < -cfg.deceleration_threshold_mps2) | (
                approach_drop > cfg.meaningful_speed_reduction_kmh
            )
            braking_evidence = float(evidence.mean()) if len(evidence) else 0.0
            distance_to_control = 0.0
            gate_s_m = _circular_s(float(feature["event_start_s_m"]), track_length_m)

        paired = group[[target_column, "lap_duration_s"]].dropna() if len(group) else pd.DataFrame()
        if len(paired) >= 4 and paired[target_column].nunique() > 1:
            pace_correlation = float(
                paired[target_column].corr(-paired["lap_duration_s"], method="spearman")
            )
        else:
            pace_correlation = float("nan")
        positive_pace_correlation = max(0.0, pace_correlation) if math.isfinite(pace_correlation) else 0.5

        vehicle_medians = group.groupby("vehicle_id")[target_column].median().dropna()
        vehicle_count = int(len(vehicle_medians))
        if vehicle_count >= 2 and median_target > 0:
            spread_fraction = float(
                (vehicle_medians.max() - vehicle_medians.min()) / median_target
            )
            cross_vehicle_score = float(np.clip(1.0 - spread_fraction / 0.20, 0.0, 1.0))
            cross_vehicle_status = "measured"
        else:
            spread_fraction = float("nan")
            cross_vehicle_score = 0.5
            cross_vehicle_status = "single_vehicle_unverified"

        members = projected[projected["final_group_id"].astype(str) == group_id]
        explicit_fraction = float(
            (members["extent_source"] == "explicit_start_end_gps").mean()
        ) if len(members) else 0.0
        map_error = float(pd.to_numeric(members["anchor_projection_error_m"], errors="coerce").max()) if len(members) else 20.0
        position_score = float(
            np.clip(0.45 + 0.40 * explicit_fraction + 0.15 * (1.0 - map_error / 20.0), 0.0, 1.0)
        )
        sample_score = float(np.clip(n / max(2 * cfg.minimum_valid_passes, 1), 0.0, 1.0))
        stability_score = float(np.clip(1.0 - target_cv / 0.25, 0.0, 1.0)) if math.isfinite(target_cv) else 0.0
        pace_score = float(np.clip(1.0 - positive_pace_correlation / 0.80, 0.0, 1.0))
        confidence_score = 100.0 * (
            0.15 * sample_score
            + 0.20 * stability_score
            + 0.25 * braking_evidence
            + 0.15 * pace_score
            + 0.15 * position_score
            + 0.10 * cross_vehicle_score
        )

        if n < cfg.minimum_valid_passes:
            confidence_class = "INSUFFICIENT"
        elif confidence_score >= cfg.high_confidence_score:
            confidence_class = "HIGH"
        elif confidence_score >= cfg.medium_confidence_score:
            confidence_class = "MEDIUM"
        else:
            confidence_class = "LOW"
        accepted = bool(
            n >= cfg.minimum_valid_passes
            and confidence_score >= cfg.default_acceptance_score
            and braking_evidence >= cfg.default_braking_evidence_fraction
            and target_cv <= cfg.maximum_default_entry_cv
            and positive_pace_correlation <= cfg.maximum_default_pace_correlation
        )
        if accepted:
            recommendation = "SPEED_GATE"
        elif braking_evidence < 0.40:
            recommendation = "NOT_GATE"
        else:
            recommendation = "REVIEW"

        reasons: list[str] = []
        if target_cv <= 0.10:
            reasons.append("stable_speed")
        elif target_cv > cfg.maximum_default_entry_cv:
            reasons.append("variable_speed")
        if braking_evidence >= 0.70:
            reasons.append("repeatable_state_convergence")
        elif braking_evidence < 0.40:
            reasons.append("little_braking_evidence")
        if positive_pace_correlation > cfg.maximum_default_pace_correlation:
            reasons.append("entry_rises_with_lap_pace")
        if explicit_fraction < 1.0:
            reasons.append("assumed_or_mixed_extent")
        if vehicle_count < 2:
            reasons.append("single_vehicle_only")

        rows.append(
            {
                "analysis_group_id": group_id,
                "sequence": int(feature["sequence"]),
                "event_name": str(feature["name"]),
                "source_members": str(feature.get("source_members", feature["name"])),
                "analysis_role": role,
                "measurement_kind": measurement_kind,
                "gate_s_m": gate_s_m,
                "distance_from_event_start_m": distance_to_control,
                "target_speed_p10_kmh": float(percentiles.loc[0.10]),
                "target_speed_p25_kmh": float(percentiles.loc[0.25]),
                "target_speed_median_kmh": float(percentiles.loc[0.50]),
                "target_speed_p75_kmh": float(percentiles.loc[0.75]),
                "target_speed_p90_kmh": float(percentiles.loc[0.90]),
                "target_speed_iqr_kmh": float(percentiles.loc[0.75] - percentiles.loc[0.25]),
                "target_speed_cv": target_cv,
                "valid_passes": n,
                "vehicle_count": vehicle_count,
                "cross_vehicle_status": cross_vehicle_status,
                "cross_vehicle_median_spread_fraction": spread_fraction,
                "braking_evidence_fraction": braking_evidence,
                "entry_speed_vs_faster_lap_spearman": pace_correlation,
                "explicit_extent_fraction": explicit_fraction,
                "maximum_anchor_projection_error_m": map_error,
                "sample_score": sample_score,
                "stability_score": stability_score,
                "braking_score": braking_evidence,
                "pace_independence_score": pace_score,
                "position_score": position_score,
                "cross_vehicle_score": cross_vehicle_score,
                "confidence_score": confidence_score,
                "confidence_class": confidence_class,
                "gate_recommendation": recommendation,
                "accepted_by_default": accepted,
                "braking_deceleration_mps2": cfg.default_braking_deceleration_mps2,
                "confidence_reasons": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows).sort_values("sequence").reset_index(drop=True)


def _measurement_locations(projected: pd.DataFrame, track_length_m: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    definitions = (
        ("entry_measurement", "entry_start_rel_m", "entry_end_rel_m"),
        ("event_start", "feature_start_rel_m", "feature_start_rel_m"),
        ("anchor", None, None),
        ("event_end", "feature_end_rel_m", "feature_end_rel_m"),
        ("exit_measurement", "exit_start_rel_m", "exit_end_rel_m"),
    )
    for _, feature in projected.sort_values("sequence").iterrows():
        anchor = float(feature["anchor_s_m"])
        for marker, low_column, high_column in definitions:
            if low_column is None:
                s_m = anchor
                zone_start = anchor
                zone_end = anchor
            else:
                low = float(feature[low_column])
                high = float(feature[high_column])
                zone_start = _circular_s(anchor + low, track_length_m)
                zone_end = _circular_s(anchor + high, track_length_m)
                midpoint = low + 0.5 * (high - low)
                s_m = _circular_s(anchor + midpoint, track_length_m)
            rows.append(
                {
                    "sequence": int(feature["sequence"]),
                    "name": str(feature["name"]),
                    "final_group_id": str(feature["final_group_id"]),
                    "marker_type": marker,
                    "s_m": s_m,
                    "zone_start_s_m": zone_start,
                    "zone_end_s_m": zone_end,
                    "extent_source": str(feature["extent_source"]),
                }
            )
    return pd.DataFrame(rows)


def _xy_at_s(centreline: Centreline, values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    s = np.mod(np.asarray(values, dtype=float), centreline.length_m)
    return (
        np.interp(s, centreline.s_nodes_m, centreline.x),
        np.interp(s, centreline.s_nodes_m, centreline.y),
    )


def _write_gate_plots(
    centreline: Centreline,
    locations: pd.DataFrame,
    gates: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.plot(centreline.x, centreline.y, color="0.72", linewidth=2.0, label="Reference centreline")
    styles = {
        "entry_measurement": ("^", "#2a9d8f", 34, "Entry measurement"),
        "event_start": (">", "#f4a261", 30, "Physical/analysis start"),
        "anchor": ("o", "#222222", 24, "Anchor"),
        "event_end": ("s", "#457b9d", 28, "Physical/analysis end"),
        "exit_measurement": ("v", "#8e5ea2", 34, "Exit measurement"),
    }
    for marker_type, (marker, color, size, label) in styles.items():
        part = locations[locations["marker_type"] == marker_type]
        x, y = _xy_at_s(centreline, part["s_m"])
        ax.scatter(x, y, marker=marker, color=color, s=size, alpha=0.78, label=label, zorder=3)
    anchors = locations[locations["marker_type"] == "anchor"]
    ax_x, ax_y = _xy_at_s(centreline, anchors["s_m"])
    for sequence, x, y in zip(anchors["sequence"], ax_x, ax_y):
        ax.annotate(str(sequence), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    accepted = gates[gates["accepted_by_default"].astype(bool)]
    colors = {"HIGH": "#1b9e77", "MEDIUM": "#e6ab02", "LOW": "#d95f02", "INSUFFICIENT": "#7570b3"}
    for confidence, part in accepted.groupby("confidence_class"):
        x, y = _xy_at_s(centreline, part["gate_s_m"])
        ax.scatter(
            x,
            y,
            marker="*",
            s=160,
            facecolor=colors.get(str(confidence), "#d95f02"),
            edgecolor="white",
            linewidth=0.8,
            label=f"Accepted {str(confidence).lower()} gate",
            zorder=5,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Local east [m]")
    ax.set_ylabel("Local north [m]")
    ax.set_title("Measured track locations and accepted speed-convergence gates\nNumbers identify physical feature sequence")
    ax.grid(True, alpha=0.18)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "01_track_measurement_and_gate_map.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for (confidence, accepted), part in gates.groupby(["confidence_class", "accepted_by_default"]):
        color = colors.get(str(confidence), "#7570b3")
        x = part["sequence"].to_numpy(dtype=float)
        y = part["target_speed_median_kmh"].to_numpy(dtype=float)
        low = y - part["target_speed_p10_kmh"].to_numpy(dtype=float)
        high = part["target_speed_p90_kmh"].to_numpy(dtype=float) - y
        marker = "*" if bool(accepted) else "o"
        status = "accepted" if bool(accepted) else "evidence only"
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([low, high]),
            fmt=marker,
            color=color,
            capsize=2,
            alpha=0.85 if accepted else 0.55,
            label=f"{confidence} — {status}",
        )
    ax.set_xlabel("Feature sequence")
    ax.set_ylabel("Measured control/entry speed [km/h]")
    ax.set_title("Gate speed evidence: median with observed 10th–90th percentile range")
    ax.set_xticks(gates["sequence"])
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, alpha=0.22)
    ax.legend(title="Confidence")
    fig.tight_layout()
    fig.savefig(output_dir / "02_gate_speed_and_confidence.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def _build_bundle(
    *,
    primary: AnalysisResult,
    gps_files: Sequence[Path],
    vehicle_ids: Sequence[str],
    passes: pd.DataFrame,
    summary: pd.DataFrame,
    gates: pd.DataFrame,
    locations: pd.DataFrame,
) -> dict[str, object]:
    centreline = primary.centreline
    lat, lon = centreline.frame.to_latlon(centreline.x, centreline.y)
    event_rows: list[dict[str, object]] = []
    eligible = passes[passes["aggregate_eligible"].astype(bool)]
    for _, event in summary.sort_values("sequence").iterrows():
        gid = str(event["analysis_group_id"])
        group = eligible[eligible["analysis_group_id"].astype(str) == gid]
        ke = pd.to_numeric(group["specific_ke_change_to_end_j_per_kg"], errors="coerce").dropna().clip(lower=0.0)
        quantiles = ke.quantile([0.25, 0.50, 0.75]) if len(ke) else pd.Series({0.25: 0.0, 0.50: 0.0, 0.75: 0.0})
        event_rows.append(
            {
                "analysis_group_id": gid,
                "sequence": int(event["sequence"]),
                "name": str(event["event_name"]),
                "analysis_role": str(event["analysis_role"]),
                "start_s_m": float(event["event_start_s_m"]),
                "end_s_m": float(event["event_end_s_m"]),
                "length_m": float(event["event_length_m"]),
                "observed_entry_median_kmh": float(event["median_entry_speed_kmh"]),
                "observed_min_median_kmh": float(event["median_event_min_speed_kmh"]),
                "observed_end_median_kmh": float(event["median_end_speed_kmh"]),
                "observed_time_median_s": float(event["median_event_time_s"]),
                "effective_specific_loss_low_j_per_kg": float(quantiles.loc[0.25]),
                "effective_specific_loss_nominal_j_per_kg": float(quantiles.loc[0.50]),
                "effective_specific_loss_high_j_per_kg": float(quantiles.loc[0.75]),
                "loss_model_status": "observed_net_kinetic_change_seed_not_physical_calibration",
            }
        )
    accepted_gates = gates[gates["accepted_by_default"].astype(bool)].copy()
    bundle: dict[str, object] = {
        "schema_version": "1.0",
        "track": {
            "name": "Measured Baja track surrogate",
            "length_m": centreline.length_m,
            "coordinate_system": "single_s_metres",
            "base_surface_assumptions": {
                "grade_degrees": 0.0,
                "friction_coefficient": 0.70,
                "rolling_resistance_coefficient": 0.03,
                "status": "scenario_assumptions_not_measured_by_gps",
            },
        },
        "source_datasets": [
            {"vehicle_id": vehicle_id, "gps_file": str(path)}
            for vehicle_id, path in zip(vehicle_ids, gps_files)
        ],
        "centreline": {
            "s_m": centreline.s_nodes_m.tolist(),
            "latitude": lat.tolist(),
            "longitude": lon.tolist(),
            "local_x_m": centreline.x.tolist(),
            "local_y_m": centreline.y.tolist(),
        },
        "speed_gates": _records(accepted_gates),
        "all_gate_evidence": _records(gates),
        "measurement_locations": _records(locations),
        "event_groups": event_rows,
        "interpretation": {
            "gate_rule": "physical backward braking envelope; never reset a slow vehicle upward",
            "confidence_fallthrough": "only accepted_by_default gates constrain the primary simulation",
            "loss_warning": "GPS-only effective loss seeds include unresolved braking, grade, propulsion, and terrain effects",
        },
    }
    return bundle


def run_gate_study(
    gps_files: Sequence[Path],
    definition_csv: Path,
    output_dir: Path,
    *,
    vehicle_ids: Sequence[str] | None = None,
    config: PipelineConfig | None = None,
) -> GateStudyResult:
    if not gps_files:
        raise ValueError("At least one GPS file is required")
    config = config or PipelineConfig()
    vehicle_ids = tuple(vehicle_ids or [f"vehicle_{index + 1}" for index in range(len(gps_files))])
    if len(vehicle_ids) != len(gps_files):
        raise ValueError("vehicle_ids must have the same length as gps_files")
    if len(set(vehicle_ids)) != len(vehicle_ids):
        raise ValueError("vehicle_ids must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)

    primary = run_analysis(Path(gps_files[0]), definition_csv, None, config=config)
    primary_passes, primary_laps = _with_dataset_context(
        primary.event_passes,
        primary.laps,
        dataset_id="dataset_01",
        vehicle_id=str(vehicle_ids[0]),
    )
    all_passes = [primary_passes]
    all_laps = [primary_laps]
    dataset_rows = [
        {
            "dataset_id": "dataset_01",
            "vehicle_id": str(vehicle_ids[0]),
            "gps_file": str(gps_files[0]),
            "complete_laps": len(primary.laps),
            "retained_laps": int(primary.laps["analysis_valid"].sum()),
            "eligible_event_passes": int(primary.event_passes["aggregate_eligible"].sum()),
            "median_sample_period_s": primary.cleaning["median_sample_period_s"],
        }
    ]
    matched_frames = [primary.matched_gps.assign(dataset_id="dataset_01", vehicle_id=str(vehicle_ids[0]))]

    for index, (gps_file, vehicle_id) in enumerate(zip(gps_files[1:], vehicle_ids[1:]), start=2):
        dataset_id = f"dataset_{index:02d}"
        passes, laps, cleaning, matched = _secondary_dataset_passes(
            Path(gps_file),
            dataset_id=dataset_id,
            vehicle_id=str(vehicle_id),
            primary=primary,
            config=config,
        )
        all_passes.append(passes)
        all_laps.append(laps)
        matched_frames.append(matched)
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "vehicle_id": str(vehicle_id),
                "gps_file": str(gps_file),
                "complete_laps": len(laps),
                "retained_laps": int(laps["analysis_valid"].sum()),
                "eligible_event_passes": int(passes["aggregate_eligible"].sum()),
                "median_sample_period_s": cleaning["median_sample_period_s"],
            }
        )

    combined_passes = pd.concat(all_passes, ignore_index=True)
    combined_laps = pd.concat(all_laps, ignore_index=True)
    combined_summary = summarize_event_passes(combined_passes, primary.analysis_features)
    gates = _score_speed_gates(
        combined_passes,
        primary.analysis_features,
        primary.projected_definitions,
        config,
        primary.centreline.length_m,
    )
    locations = _measurement_locations(primary.projected_definitions, primary.centreline.length_m)
    dataset_summary = pd.DataFrame(dataset_rows)
    bundle = _build_bundle(
        primary=primary,
        gps_files=[Path(path) for path in gps_files],
        vehicle_ids=vehicle_ids,
        passes=combined_passes,
        summary=combined_summary,
        gates=gates,
        locations=locations,
    )

    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    combined_laps.to_csv(output_dir / "combined_laps.csv", index=False)
    combined_passes.to_csv(output_dir / "combined_event_passes.csv", index=False)
    combined_summary.to_csv(output_dir / "combined_event_summary.csv", index=False)
    gates.to_csv(output_dir / "speed_gate_confidence.csv", index=False)
    locations.to_csv(output_dir / "measurement_locations.csv", index=False)
    pd.concat(matched_frames, ignore_index=True).to_csv(output_dir / "combined_map_matched_gps.csv", index=False)
    (output_dir / "simulator_track_bundle.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )
    _write_gate_plots(primary.centreline, locations, gates, output_dir)

    accepted = gates[gates["accepted_by_default"].astype(bool)]
    report = [
        "# Measured speed-gate study",
        "",
        f"- Input datasets: {len(gps_files)}",
        f"- Vehicles represented: {len(set(vehicle_ids))}",
        f"- Retained laps: {int(combined_laps['analysis_valid'].sum())}",
        f"- Event groups evaluated: {len(gates)}",
        f"- Default accepted speed gates: {len(accepted)}",
        "",
        "## Confidence classes",
        "",
    ]
    for name, count in gates["confidence_class"].value_counts().items():
        report.append(f"- {name}: {int(count)}")
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "Accepted gates are supported driver/geometry state-convergence constraints, not track-wide speed caps. The simulator must apply them through a physical backward braking envelope and must never increase a slow vehicle to the target.",
            "",
            "GPS-only effective event-loss seeds are included only to exercise uncertainty plumbing. They are observed net kinetic-state changes, not calibrated terrain dissipation.",
        ]
    )
    (output_dir / "GATE_STUDY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    return GateStudyResult(
        primary=primary,
        dataset_summary=dataset_summary,
        laps=combined_laps,
        event_passes=combined_passes,
        event_summary=combined_summary,
        speed_gates=gates,
        measurement_locations=locations,
        bundle=bundle,
    )
