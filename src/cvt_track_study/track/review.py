"""Static review plots and human-readable review report generation."""

from __future__ import annotations

import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .geo import Centreline


def create_track_map(
    path: Path,
    centreline: Centreline,
    matched_points: pd.DataFrame,
    events: pd.DataFrame,
    gate_review: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 9))
    for _, segment in matched_points.groupby("lap_id"):
        axis.plot(segment["x_m"], segment["y_m"], linewidth=0.6, alpha=0.22)
    axis.plot(
        centreline.x_m,
        centreline.y_m,
        linewidth=2.0,
        label="reference centreline",
    )
    review_lookup = gate_review.set_index("response_group_id")
    marker_map = {
        "accepted": "o",
        "recommended_review": "s",
        "must_fix": "X",
        "rejected": "v",
        "not_a_candidate": ".",
    }
    for _, event in events.iterrows():
        recommendation = (
            str(review_lookup.loc[event["response_group_id"], "recommendation"])
            if event["response_group_id"] in review_lookup.index
            else "not_a_candidate"
        )
        anchor_x, anchor_y = _point_at_s(centreline, float(event["anchor_s_m"]))
        axis.scatter(
            anchor_x,
            anchor_y,
            marker=marker_map.get(recommendation, "o"),
            s=58,
            zorder=5,
        )
        start_x, start_y = _point_at_s(
            centreline,
            float(event["anchor_s_m"] + event["feature_start_rel_m"]),
        )
        end_x, end_y = _point_at_s(
            centreline,
            float(event["anchor_s_m"] + event["feature_end_rel_m"]),
        )
        axis.scatter(start_x, start_y, marker="|", s=80, zorder=4)
        axis.scatter(end_x, end_y, marker="_", s=80, zorder=4)
        axis.annotate(
            f"{int(event['sequence'])}: {event['name']}",
            (anchor_x, anchor_y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    for _, gate in gate_review.iterrows():
        if not bool(gate["gate_candidate"]):
            continue
        anchor_s = float(gate["anchor_s_m"])
        entry_s = anchor_s + 0.5 * (
            float(gate["entry_start_rel_m"]) + float(gate["entry_end_rel_m"])
        )
        entry_x, entry_y = _point_at_s(centreline, entry_s)
        axis.scatter(entry_x, entry_y, marker=">", s=55, zorder=6)
        if np.isfinite(gate.get("median_event_min_rel_m", np.nan)):
            minimum_x, minimum_y = _point_at_s(
                centreline,
                anchor_s + float(gate["median_event_min_rel_m"]),
            )
            axis.scatter(minimum_x, minimum_y, marker="v", s=50, zorder=6)
        if np.isfinite(gate.get("median_recovery_distance_m", np.nan)):
            recovery_s = (
                anchor_s
                + float(gate["feature_end_rel_m"])
                + float(gate["median_recovery_distance_m"])
            )
            recovery_x, recovery_y = _point_at_s(centreline, recovery_s)
            axis.scatter(recovery_x, recovery_y, marker="^", s=50, zorder=6)

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Local east [m]")
    axis.set_ylabel("Local north [m]")
    axis.set_title(
        "Track reconstruction: anchor labels, physical starts/ends, entry gates, observed minima, and recovery"
    )
    axis.grid(True, alpha=0.25)
    axis.text(
        0.01,
        0.01,
        "Markers: | physical start, _ physical end, > entry window, v median minimum, ^ median recovery",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    legend_handles = [
        Line2D([], [], linestyle="-", label="reference centreline"),
        Line2D([], [], marker="o", linestyle="None", label="accepted anchor"),
        Line2D([], [], marker="s", linestyle="None", label="review anchor"),
        Line2D([], [], marker="X", linestyle="None", label="must-fix anchor"),
        Line2D([], [], marker="v", linestyle="None", label="rejected anchor"),
    ]
    axis.legend(handles=legend_handles, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _point_at_s(centreline: Centreline, s_m: float) -> tuple[float, float]:
    wrapped = s_m % centreline.length_m
    x = float(np.interp(wrapped, centreline.s_m, centreline.x_m))
    y = float(np.interp(wrapped, centreline.s_m, centreline.y_m))
    return x, y


def create_elevation_profile(path: Path, track_profile: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 4.5))
    s = track_profile["s_m"].to_numpy(float)
    median = track_profile["median_elevation_m"].to_numpy(float)
    lower = track_profile["p10_elevation_m"].to_numpy(float)
    upper = track_profile["p90_elevation_m"].to_numpy(float)
    if np.isfinite(median).any():
        axis.plot(s, median, linewidth=1.7, label="median GPX elevation")
        axis.fill_between(s, lower, upper, alpha=0.2, label="p10-p90 between laps")
    else:
        axis.text(0.5, 0.5, "No usable GPX elevation", transform=axis.transAxes, ha="center")
    axis.set_xlabel("Along-track coordinate, s [m]")
    axis.set_ylabel("GPX elevation [m]")
    axis.set_title("Elevation retained for review only — grade force remains disabled")
    axis.grid(True, alpha=0.25)
    if np.isfinite(median).any():
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_review_summary(
    path: Path,
    *,
    laps: pd.DataFrame,
    events: pd.DataFrame,
    gate_review: pd.DataFrame,
    track_length_m: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = gate_review["recommendation"].value_counts().to_dict()
    must_fix = gate_review[gate_review["recommendation"] == "must_fix"]
    recommended = gate_review[gate_review["recommendation"] == "recommended_review"]
    accepted_with_assumptions = gate_review[
        (gate_review["recommendation"] == "accepted")
        & gate_review["review_flags"].fillna("").astype(str).ne("")
    ]
    invalid_laps = laps[~laps["analysis_valid"]]
    lines = [
        "# Track review summary",
        "",
        f"- Reconstructed length: **{track_length_m:.1f} m**",
        f"- Complete laps: **{len(laps)}**",
        f"- Valid evidence laps: **{int(laps['analysis_valid'].sum())}**",
        f"- Declared physical features: **{len(events)}**",
        f"- Accepted speed gates: **{int(counts.get('accepted', 0))}**",
        f"- Recommended review: **{int(counts.get('recommended_review', 0))}**",
        f"- Must fix: **{int(counts.get('must_fix', 0))}**",
        "",
        "Gate confidence is an evidence score, not a probability that a gate is true. "
        "The component scores expose the supporting pass count, speed repeatability, "
        "braking evidence, pace independence, coordinate quality, and cross-vehicle agreement.",
        "",
    ]
    if len(must_fix):
        lines.extend(["## Must fix before simulation", ""])
        for _, row in must_fix.iterrows():
            lines.append(f"- **{row['event_name']}** — {row['suggested_action']}")
        lines.append("")
    if len(recommended):
        lines.extend(["## Recommended review", ""])
        for _, row in recommended.iterrows():
            lines.append(
                f"- **{row['event_name']}** ({row['overall_confidence_score']:.1f}/100) — "
                f"{row['suggested_action']}"
            )
        lines.append("")
    if len(accepted_with_assumptions):
        lines.extend(["## Accepted gates with retained assumptions", ""])
        for _, row in accepted_with_assumptions.iterrows():
            lines.append(f"- **{row['event_name']}** — {row['suggested_action']}")
        lines.append("")
    if len(invalid_laps):
        lines.extend(["## Excluded laps", ""])
        for _, row in invalid_laps.iterrows():
            lines.append(
                f"- Lap {int(row['lap_id'])} ({row['run_id']}): "
                f"{row['quality_flags'] or 'unspecified quality failure'}"
            )
        lines.append("")
    lines.extend(
        [
            "## Review order",
            "",
            "1. Resolve all `must_fix` coordinate or contract problems.",
            "2. Inspect `recommended_review` events against video and the map.",
            "3. Confirm response groups for adjacent or overlapping physical features.",
            "4. Treat accepted gate speeds as empirical distributions, not exact constants.",
            "5. Do not infer road grade from the raw elevation profile yet.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_review_html(
    path: Path,
    *,
    map_path: Path,
    elevation_path: Path,
    gate_review: pd.DataFrame,
    laps: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    map_data = base64.b64encode(map_path.read_bytes()).decode("ascii")
    elevation_data = base64.b64encode(elevation_path.read_bytes()).decode("ascii")
    gate_columns = [
        "sequence",
        "event_name",
        "overall_confidence_score",
        "recommendation",
        "valid_pass_count",
        "entry_speed_median_mps",
        "entry_speed_p10_mps",
        "entry_speed_p90_mps",
        "coordinate_effective_error_m",
        "slowdown_signature",
        "cross_vehicle_status",
        "reasons",
        "suggested_action",
    ]
    lap_columns = [
        "lap_id",
        "run_id",
        "vehicle_id",
        "duration_s",
        "analysis_valid",
        "speed_coverage_fraction",
        "quality_flags",
        "p95_map_error_m",
        "time_gap_count",
    ]
    gate_table = gate_review[gate_columns].to_html(index=False, escape=True, float_format=lambda x: f"{x:.3g}")
    lap_table = laps[lap_columns].to_html(index=False, escape=True, float_format=lambda x: f"{x:.3g}")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CVT track review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; line-height: 1.45; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.35rem; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #f4f4f4; }}
.note {{ padding: 0.8rem; background: #f6f6f6; border-left: 4px solid #777; }}
</style></head><body>
<h1>Track reconstruction review</h1>
<p class="note">Elevation is preserved and visualized, but is not converted to road grade or vehicle force in this phase.</p>
<h2>Map</h2><img alt="Track review map" src="data:image/png;base64,{map_data}">
<h2>Elevation</h2><img alt="Elevation profile" src="data:image/png;base64,{elevation_data}">
<h2>Gate review</h2>{gate_table}
<h2>Lap quality</h2>{lap_table}
</body></html>"""
    path.write_text(document, encoding="utf-8")
