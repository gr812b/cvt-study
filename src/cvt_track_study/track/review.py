"""Static review plots and human-readable review report generation."""

from __future__ import annotations

import base64
import html
import math
from pathlib import Path
from typing import Any

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


def build_event_interval_audit(
    response_features: pd.DataFrame,
    track_length_m: float,
) -> pd.DataFrame:
    """Resolve each response group's physical interval into explicit s fields.

    The source response-feature table stores an anchor plus relative start/end
    positions. This audit makes the interpreted absolute interval, length, and
    start/finish wrapping visible without changing the reconstruction itself.
    """

    if not np.isfinite(track_length_m) or track_length_m <= 0:
        raise ValueError("Event timeline requires a positive finite track length.")

    rows: list[dict[str, Any]] = []
    ordered = response_features.sort_values("sequence").reset_index(drop=True)
    for _, feature in ordered.iterrows():
        anchor_s = float(feature["anchor_s_m"])
        start_unwrapped = anchor_s + float(feature["feature_start_rel_m"])
        end_unwrapped = anchor_s + float(feature["feature_end_rel_m"])
        while end_unwrapped < start_unwrapped:
            end_unwrapped += track_length_m

        length_m = max(0.0, end_unwrapped - start_unwrapped)
        start_wrapped = start_unwrapped % track_length_m
        end_wrapped = end_unwrapped % track_length_m
        wraps = _interval_wraps_start_finish(
            start_unwrapped,
            end_unwrapped,
            track_length_m,
        )
        track_fraction = length_m / track_length_m

        flags = [
            item
            for item in str(feature.get("review_flags", "") or "").split(";")
            if item and item.lower() != "nan"
        ]
        if wraps and "wraps_start_finish" not in flags:
            flags.append("wraps_start_finish")
        if length_m > 150.0 and "extent_very_long" not in flags:
            flags.append("extent_very_long")
        if track_fraction > 0.10 and "covers_more_than_10_percent_of_track" not in flags:
            flags.append("covers_more_than_10_percent_of_track")
        if track_fraction > 0.50 and "covers_more_than_half_of_track" not in flags:
            flags.append("covers_more_than_half_of_track")

        rows.append(
            {
                "sequence": int(feature["sequence"]),
                "response_group_id": str(feature["response_group_id"]),
                "name": str(feature["name"]),
                "source_event_ids": str(feature.get("source_event_ids", "")),
                "source_event_names": str(feature.get("source_event_names", "")),
                "analysis_feature_type": str(
                    feature.get("analysis_feature_type", "individual")
                ),
                "anchor_s_m": anchor_s % track_length_m,
                "feature_start_s_unwrapped_m": start_unwrapped,
                "feature_end_s_unwrapped_m": end_unwrapped,
                "feature_start_s_m": start_wrapped,
                "feature_end_s_m": end_wrapped,
                "feature_length_m": length_m,
                "track_fraction": track_fraction,
                "wraps_start_finish": wraps,
                "review_flags": str(feature.get("review_flags", "") or ""),
                "interval_audit_flags": ";".join(dict.fromkeys(flags)),
            }
        )

    return pd.DataFrame(rows)


def create_event_group_timeline(
    path: Path,
    interval_audit: pd.DataFrame,
    track_length_m: float,
) -> None:
    """Create a one-dimensional review of resolved event-group extents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = interval_audit.sort_values("sequence").reset_index(drop=True)
    figure_height = max(7.0, 0.38 * max(len(ordered), 1) + 2.2)
    figure, axis = plt.subplots(figsize=(15, figure_height))

    if ordered.empty:
        axis.text(
            0.5,
            0.5,
            "No response groups available",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    else:
        for lane, (_, feature) in enumerate(ordered.iterrows()):
            start = float(feature["feature_start_s_unwrapped_m"])
            end = float(feature["feature_end_s_unwrapped_m"])
            suspicious = bool(str(feature["interval_audit_flags"]).strip())

            for segment_start, segment_length in _split_interval_for_plot(
                start,
                end,
                track_length_m,
            ):
                bars = axis.barh(
                    lane,
                    segment_length,
                    left=segment_start,
                    height=0.58,
                    alpha=0.72,
                    linewidth=1.0,
                )
                if suspicious:
                    for patch in bars.patches:
                        patch.set_hatch("///")

            axis.scatter(
                float(feature["anchor_s_m"]),
                lane,
                marker="o",
                s=24,
                zorder=5,
            )

        labels = []
        for _, feature in ordered.iterrows():
            wrap_marker = " ↻" if bool(feature["wraps_start_finish"]) else ""
            labels.append(
                f"{int(feature['sequence'])}: {feature['name']}"
                f"  [{float(feature['feature_length_m']):.1f} m]{wrap_marker}"
            )
        axis.set_yticks(np.arange(len(ordered)))
        axis.set_yticklabels(labels, fontsize=8)
        axis.invert_yaxis()

    axis.axvline(0.0, linewidth=1.2)
    axis.axvline(track_length_m, linewidth=1.2)
    axis.set_xlim(0.0, track_length_m)
    axis.set_xlabel("Along-track coordinate, s [m]")
    axis.set_ylabel("Response group")
    axis.set_title(
        "Resolved physical extent of each event response group along the track"
    )
    axis.grid(True, axis="x", alpha=0.25)
    axis.text(
        0.01,
        -0.035,
        "Bars: resolved physical extent; circle: anchor; hatch: interval review flag; ↻: crosses start/finish",
        transform=axis.transAxes,
        fontsize=8,
        va="top",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _interval_wraps_start_finish(
    start_unwrapped_m: float,
    end_unwrapped_m: float,
    track_length_m: float,
) -> bool:
    if end_unwrapped_m <= start_unwrapped_m:
        return False
    epsilon = min(1e-9, track_length_m * 1e-12)
    start_lap = math.floor(start_unwrapped_m / track_length_m)
    end_lap = math.floor((end_unwrapped_m - epsilon) / track_length_m)
    return start_lap != end_lap


def _split_interval_for_plot(
    start_unwrapped_m: float,
    end_unwrapped_m: float,
    track_length_m: float,
) -> list[tuple[float, float]]:
    """Split an unwrapped interval into visible [0, track_length] pieces."""

    if end_unwrapped_m <= start_unwrapped_m:
        return []

    length_m = end_unwrapped_m - start_unwrapped_m
    if length_m >= track_length_m:
        return [(0.0, track_length_m)]

    start = start_unwrapped_m % track_length_m
    end = end_unwrapped_m % track_length_m
    if start + length_m <= track_length_m + 1e-9:
        return [(start, min(length_m, track_length_m - start))]

    first = track_length_m - start
    second = max(0.0, length_m - first)
    return [(start, first), (0.0, second)]


def create_elevation_profile(path: Path, track_profile: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 4.5))
    s = track_profile["s_m"].to_numpy(float)
    median = track_profile["median_elevation_m"].to_numpy(float)
    lower = track_profile["p10_elevation_m"].to_numpy(float)
    upper = track_profile["p90_elevation_m"].to_numpy(float)
    if np.isfinite(median).any():
        axis.plot(s, median, linewidth=1.7, label="median telemetry elevation")
        axis.fill_between(s, lower, upper, alpha=0.2, label="p10-p90 between laps")
    else:
        axis.text(
            0.5,
            0.5,
            "No usable telemetry elevation",
            transform=axis.transAxes,
            ha="center",
        )
    axis.set_xlabel("Along-track coordinate, s [m]")
    axis.set_ylabel("Telemetry elevation [m]")
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
    recommended = gate_review[
        gate_review["recommendation"] == "recommended_review"
    ]
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
                f"- **{row['event_name']}** "
                f"({row['overall_confidence_score']:.1f}/100) — "
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
    timeline_path: Path,
    elevation_path: Path,
    interval_audit: pd.DataFrame,
    gate_review: pd.DataFrame,
    laps: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    map_data = base64.b64encode(map_path.read_bytes()).decode("ascii")
    timeline_data = base64.b64encode(timeline_path.read_bytes()).decode("ascii")
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
    interval_columns = [
        "sequence",
        "response_group_id",
        "name",
        "source_event_names",
        "feature_start_s_m",
        "feature_end_s_m",
        "feature_length_m",
        "track_fraction",
        "wraps_start_finish",
        "interval_audit_flags",
    ]

    gate_table = gate_review[gate_columns].to_html(
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.3g}",
    )
    lap_table = laps[lap_columns].to_html(
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.3g}",
    )
    interval_table = interval_audit[interval_columns].to_html(
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.3g}",
    )

    suspicious = interval_audit[
        interval_audit["interval_audit_flags"].fillna("").astype(str).ne("")
    ]
    if suspicious.empty:
        interval_warning = (
            '<p class="note">No event interval audit flags were raised.</p>'
        )
    else:
        items = "".join(
            "<li><strong>"
            + html.escape(f"{int(row['sequence'])}: {row['name']}")
            + "</strong> — "
            + html.escape(str(row["interval_audit_flags"]))
            + f"; resolved length {float(row['feature_length_m']):.1f} m"
            + "</li>"
            for _, row in suspicious.iterrows()
        )
        interval_warning = (
            '<div class="warning"><strong>Event interval review required.</strong>'
            "<p>The following groups wrap start/finish, occupy a large share of "
            "the lap, or already carry reconstruction review flags:</p>"
            f"<ul>{items}</ul></div>"
        )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CVT track review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; line-height: 1.45; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.35rem; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #f4f4f4; }}
.note {{ padding: 0.8rem; background: #f6f6f6; border-left: 4px solid #777; }}
.warning {{ padding: 0.8rem; background: #fff7e6; border-left: 4px solid #a66a00; }}
</style></head><body>
<h1>Track reconstruction review</h1>
<p class="note">Elevation is preserved and visualized, but is not converted to road grade or vehicle force in this phase.</p>
<h2>Map</h2><img alt="Track review map" src="data:image/png;base64,{map_data}">
<h2>Along-track event-group timeline</h2>
<p class="note">This view shows the exact physical extent assigned to every response group on the common s coordinate. A nearly full-width bar means the reconstruction really interpreted that feature as occupying nearly the whole lap.</p>
{interval_warning}
<img alt="Along-track event-group timeline" src="data:image/png;base64,{timeline_data}">
<h3>Event interval audit</h3>{interval_table}
<h2>Elevation</h2><img alt="Elevation profile" src="data:image/png;base64,{elevation_data}">
<h2>Gate review</h2>{gate_table}
<h2>Lap quality</h2>{lap_table}
</body></html>"""
    path.write_text(document, encoding="utf-8")
