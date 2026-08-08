"""Branch-aware route-family visualization and HTML augmentation.

Supported whole-lap variants are displayed as alternate traversals through one
shared course.  Heavy geometry therefore shows the common backbone once and only
splits inside sustained divergence/re-merge corridors.
"""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def augment_track_review_with_route_family(
    report_path: Path,
    map_path: Path,
    result: Any,
) -> None:
    members = dict(getattr(result, "route_family_members", {}) or {})
    if len(members) <= 1 or not report_path.exists():
        return
    create_route_family_map(map_path, result)
    section = _route_family_section(result, map_path)
    document = report_path.read_text(encoding="utf-8")
    document = document.replace(
        '<div class="label">Valid evidence laps</div>',
        '<div class="label">Nominal branch valid laps</div>',
        1,
    )
    document = document.replace(
        '<div class="label">Reconstructed length</div>',
        '<div class="label">Nominal traversal length</div>',
        1,
    )
    nav_anchor = '<nav class="report-nav" aria-label="Report sections">'
    nav_link = '<a href="#route-family-reconstruction">Route branches</a>'
    if nav_anchor in document and nav_link not in document:
        start = document.index(nav_anchor)
        end = document.find("</nav>", start)
        if end >= 0:
            document = document[:end] + nav_link + document[end:]
    for marker in (
        '<h2 id="executive-evidence-summary">',
        "<h2>Executive evidence summary</h2>",
        "</header>",
    ):
        position = document.find(marker)
        if position < 0:
            continue
        if marker == "</header>":
            position += len(marker)
        document = document[:position] + section + document[position:]
        break
    else:
        body_end = document.rfind("</main>")
        position = body_end if body_end >= 0 else len(document)
        document = document[:position] + section + document[position:]
    report_path.write_text(document, encoding="utf-8")


def create_route_family_map(path: Path, result: Any) -> None:
    members = dict(getattr(result, "route_family_members", {}) or {})
    nominal_id = str(getattr(result, "nominal_route_variant_id", ""))
    branches = _branch_frame(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12.0, 7.2))

    # Retained raw laps remain faint so ordinary line choice is visible as evidence,
    # but heavy geometry is a course graph rather than two complete coloured loops.
    for route_id, member in members.items():
        valid_ids = set(
            pd.to_numeric(
                member.laps.loc[
                    member.laps.get("analysis_valid", False).astype(bool), "lap_id"
                ],
                errors="coerce",
            ).dropna().astype(int)
        )
        matched = member.matched_points
        if not valid_ids or matched is None or matched.empty:
            continue
        visible = matched[
            pd.to_numeric(matched["lap_id"], errors="coerce")
            .fillna(-1).astype(int).isin(valid_ids)
        ]
        for _, lap in visible.groupby("lap_id", sort=False):
            axis.plot(
                pd.to_numeric(lap["x_m"], errors="coerce"),
                pd.to_numeric(lap["y_m"], errors="coerce"),
                linewidth=0.65,
                alpha=0.12,
                zorder=1,
            )

    if nominal_id in members and not branches.empty:
        nominal = members[nominal_id]
        nominal_intervals = [
            (
                float(row.nominal_branch_start_s_m),
                float(row.nominal_branch_end_s_m),
            )
            for row in branches.itertuples(index=False)
            if str(row.nominal_route_variant_id) == nominal_id
        ]
        shared_mask = _outside_intervals_mask(
            np.asarray(nominal.centreline.s_m, float),
            nominal.centreline.length_m,
            nominal_intervals,
        )
        shared_count = int(
            pd.to_numeric(
                branches.get(
                    "shared_section_valid_lap_count", pd.Series([0])
                ),
                errors="coerce",
            ).max()
        )
        _plot_masked_path(
            axis,
            nominal.centreline,
            shared_mask,
            linewidth=3.3,
            label=f"shared course — {shared_count} valid laps",
            zorder=5,
        )

        for row in branches.itertuples(index=False):
            branch_id = str(row.branch_id)
            # Nominal/short option through the divergence corridor.
            nominal_member = members[str(row.nominal_route_variant_id)]
            nominal_mask = _inside_interval_mask(
                np.asarray(nominal_member.centreline.s_m, float),
                nominal_member.centreline.length_m,
                float(row.nominal_branch_start_s_m),
                float(row.nominal_branch_end_s_m),
            )
            nominal_label = (
                f"{branch_id} short / {row.nominal_route_variant_id}"
                f" — {int(row.nominal_branch_valid_geometry_lap_count)} laps"
            )
            _plot_masked_path(
                axis,
                nominal_member.centreline,
                nominal_mask,
                linewidth=3.1,
                label=nominal_label,
                zorder=6,
            )

            alternate_member = members[str(row.alternate_route_variant_id)]
            alternate_mask = _inside_interval_mask(
                np.asarray(alternate_member.centreline.s_m, float),
                alternate_member.centreline.length_m,
                float(row.alternate_branch_start_s_m),
                float(row.alternate_branch_end_s_m),
            )
            alternate_label = (
                f"{branch_id} long / {row.alternate_route_variant_id}"
                f" — {int(row.alternate_branch_valid_geometry_lap_count)} laps"
            )
            _plot_masked_path(
                axis,
                alternate_member.centreline,
                alternate_mask,
                linewidth=3.1,
                label=alternate_label,
                zorder=6,
            )
    else:
        # Fallback if the branch detector cannot resolve a sustained corridor.
        for route_id, member in members.items():
            label = route_id + (" (nominal)" if route_id == nominal_id else "")
            label += f" — {_valid_geometry_lap_count(member)} valid laps"
            axis.plot(
                np.asarray(member.centreline.x_m, float),
                np.asarray(member.centreline.y_m, float),
                linewidth=2.8,
                label=label,
                zorder=4,
            )

    axis.set_title("Shared course with sustained alternate branch")
    axis.set_xlabel("Local east / x [m]")
    axis.set_ylabel("Local north / y [m]")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def route_family_review_rows(result: Any) -> pd.DataFrame:
    members = dict(getattr(result, "route_family_members", {}) or {})
    summary = getattr(result, "route_variant_summary", pd.DataFrame())
    summary = summary.copy() if isinstance(summary, pd.DataFrame) else pd.DataFrame()
    summary_by_id: Mapping[str, pd.Series] = {}
    if not summary.empty and "route_variant_id" in summary:
        summary_by_id = {
            str(row["route_variant_id"]): row for _, row in summary.iterrows()
        }
    common_count = sum(_valid_geometry_lap_count(member) for member in members.values())
    nominal = str(getattr(result, "nominal_route_variant_id", ""))
    rows: list[dict[str, Any]] = []
    for route_id, member in members.items():
        source = summary_by_id.get(str(route_id))
        assigned = _row_int(source, "lap_count")
        if assigned is None:
            assigned = _pre_consensus_geometry_lap_count(member)
        rows.append(
            {
                "route_variant_id": str(route_id),
                "role": "nominal branch" if str(route_id) == nominal else "alternate branch",
                "assigned_topology_laps": int(assigned),
                "branch_valid_geometry_laps": _valid_geometry_lap_count(member),
                "shared_section_support_laps": int(common_count),
                "consensus_excluded_laps": _consensus_excluded_lap_count(member),
                "traversal_length_m": float(member.centreline.length_m),
                "merged_line_clusters": _row_int(source, "geometric_seed_cluster_count") or "",
                "accepted_gates": _accepted_gate_count(member),
                "eligible_gate_evidence_laps": _eligible_gate_lap_count(member),
            }
        )
    return pd.DataFrame(rows)


def _route_family_section(result: Any, map_path: Path) -> str:
    rows = route_family_review_rows(result)
    branches = _branch_frame(result)
    summary = getattr(result, "route_variant_summary", pd.DataFrame())
    summary = summary.copy() if isinstance(summary, pd.DataFrame) else pd.DataFrame()
    nominal = str(getattr(result, "nominal_route_variant_id", ""))
    shared_laps = (
        int(rows["shared_section_support_laps"].max()) if not rows.empty else 0
    )
    event_sections = _event_section_summary(result)
    shared_events = int((event_sections.get("course_section", pd.Series(dtype=str)) == "shared").sum())
    branch_events = int((event_sections.get("course_section", pd.Series(dtype=str)) == "branch").sum())
    unsupported_rows = pd.DataFrame()
    unsupported_laps = 0
    if not summary.empty and "supported" in summary:
        supported = summary["supported"].fillna(False).astype(bool)
        unsupported_rows = summary.loc[~supported].copy()
        if "lap_count" in unsupported_rows:
            unsupported_laps = int(
                pd.to_numeric(unsupported_rows["lap_count"], errors="coerce")
                .fillna(0).sum()
            )

    image = ""
    if map_path.exists():
        encoded = base64.b64encode(map_path.read_bytes()).decode("ascii")
        image = (
            '<div class="figure"><img alt="Shared course backbone and local alternate '
            'branch corridors with retained lap evidence." '
            f'src="data:image/png;base64,{encoded}">'
            '<div class="caption">The heavy shared backbone is drawn only once. '
            'All supported branch laps contribute evidence there. Heavy branch traces '
            'appear only between the detected divergence and re-merge points; events '
            'inside that corridor use only laps that traversed the corresponding branch.</div></div>'
        )

    branch_table = _branch_table_html(branches)
    return (
        '<h2 id="route-family-reconstruction">Course branches</h2>'
        '<div class="section-intro"><strong>Interpretation.</strong> The supported '
        'whole-lap clusters are not treated as separate tracks. They are alternate '
        'traversals of one shared closed course. Ordinary inside/outside turn choice '
        'is absorbed into the common evidence pool; only a sustained spatial deviation '
        'that later rejoins the course creates a branch.</div>'
        '<div class="scope">'
        f'<div class="card good"><div class="label">Shared-section valid laps</div><div class="value">{shared_laps}</div></div>'
        f'<div class="card note"><div class="label">Alternate branch corridors</div><div class="value">{len(branches)}</div></div>'
        f'<div class="card note"><div class="label">Shared-section events</div><div class="value">{shared_events}</div></div>'
        f'<div class="card note"><div class="label">Branch-local events</div><div class="value">{branch_events}</div></div>'
        f'<div class="card warning"><div class="label">Unsupported / ambiguous laps</div><div class="value">{unsupported_laps}</div></div>'
        '</div>'
        + image
        + '<h3>Branch corridors</h3>'
        + branch_table
        + '<h3>Evidence pooling by event</h3>'
        '<p class="table-note">This is the key branch-aware evidence view. Events on the '
        'shared course use one deduplicated empirical gate contract, scored once from '
        'compatible passes across every supported traversal and then reused on each route. Local '
        'coverage is evaluated by continuous bracketing between acceptable telemetry '
        'samples, so a short window does not require a raw GPS/FIT sample to land inside '
        'it. Events inside a divergence corridor are restricted to the laps that actually '
        'used that branch.</p>'
        + _event_evidence_table_html(result)
        + '<h3>Traversal support</h3>'
        '<p class="table-note">Shared-section support is the sum of valid '
        'laps across the supported branch choices. Branch-valid geometry is restricted '
        'to laps that actually used that branch. Shared-course local projection allows '
        'ordinary line variation while remaining below the detected genuine-branch '
        'separation threshold.</p>'
        + _family_table_html(rows)
        + _unsupported_table_html(unsupported_rows)
    )


def _branch_table_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<p class="table-note">No sustained divergence/re-merge corridor was resolved.</p>'
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('branch_id', '')))}</td>"
            f"<td>{escape(str(row.get('nominal_route_variant_id', '')))}</td>"
            f"<td>{escape(str(row.get('alternate_route_variant_id', '')))}</td>"
            f"<td>{_safe_int(row.get('shared_section_valid_lap_count', 0))}</td>"
            f"<td>{_safe_int(row.get('nominal_branch_valid_geometry_lap_count', 0))}</td>"
            f"<td>{_safe_int(row.get('alternate_branch_valid_geometry_lap_count', 0))}</td>"
            f"<td>{float(row.get('nominal_branch_length_m', 0.0)):.1f}</td>"
            f"<td>{float(row.get('alternate_branch_length_m', 0.0)):.1f}</td>"
            f"<td>{float(row.get('divergence_distance_threshold_m', 0.0)):.1f}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap compact"><table><thead><tr>'
        '<th>Branch</th><th>Nominal option</th><th>Alternate option</th>'
        '<th>Shared laps</th><th>Nominal-branch laps</th><th>Alternate-branch laps</th>'
        '<th>Nominal branch [m]</th><th>Alternate branch [m]</th><th>Divergence threshold [m]</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
    )


def _family_table_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<p class="table-note">No exported branch traversals were found.</p>'
    headings = [
        ("route_variant_id", "Traversal"),
        ("role", "Role"),
        ("assigned_topology_laps", "Assigned laps"),
        ("branch_valid_geometry_laps", "Branch-valid laps"),
        ("shared_section_support_laps", "Shared-section laps"),
        ("consensus_excluded_laps", "Consensus excluded"),
        ("traversal_length_m", "Traversal length [m]"),
        ("merged_line_clusters", "Merged line clusters"),
        ("accepted_gates", "Accepted gates"),
        ("eligible_gate_evidence_laps", "Gate-evidence laps"),
    ]
    head = "".join(f"<th>{escape(label)}</th>" for _, label in headings)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for key, _ in headings:
            value = row[key]
            text = f"{float(value):.1f}" if key == "traversal_length_m" else str(value)
            cells.append(f"<td>{escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return '<div class="table-wrap compact"><table><thead><tr>' + head + '</tr></thead><tbody>' + "".join(body) + '</tbody></table></div>'


def _unsupported_table_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('route_variant_id', '')))}</td>"
            f"<td>{_safe_int(row.get('lap_count', 0))}</td>"
            f"<td>{escape(str(row.get('lap_ids', '')))}</td>"
            f"<td>{escape(str(row.get('topology_component_status', 'unsupported')))}</td>"
            "</tr>"
        )
    return (
        '<details><summary>Unsupported or ambiguous topology components</summary>'
        '<p class="table-note">These laps remain auditable and are not forced into '
        'either branch.</p><div class="table-wrap compact"><table><thead><tr>'
        '<th>Component</th><th>Lap count</th><th>Lap IDs</th><th>Status</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div></details>'
    )



def _event_section_summary(result: Any) -> pd.DataFrame:
    """Collapse per-traversal event classification to one course-section row per event."""
    sections = getattr(result, "event_route_sections", pd.DataFrame())
    if not isinstance(sections, pd.DataFrame) or sections.empty:
        return pd.DataFrame(
            columns=["event_id", "event_name", "course_section", "branch_ids"]
        )
    rows: list[dict[str, Any]] = []
    for event_id, group in sections.groupby("event_id", sort=False):
        kinds = set(group.get("route_section_kind", pd.Series(dtype=str)).astype(str))
        branch_ids = sorted(
            {
                token
                for value in group.get("branch_ids", pd.Series(dtype=str)).fillna("").astype(str)
                for token in value.split(";")
                if token
            }
        )
        rows.append(
            {
                "event_id": str(event_id),
                "event_name": str(group.get("event_name", pd.Series([event_id])).iloc[0]),
                "course_section": "branch" if "branch" in kinds else "shared",
                "branch_ids": ";".join(branch_ids),
            }
        )
    return pd.DataFrame(rows)


def _event_evidence_table_html(result: Any) -> str:
    sections = _event_section_summary(result)
    evidence = getattr(result, "shared_gate_evidence", pd.DataFrame())
    evidence = evidence.copy() if isinstance(evidence, pd.DataFrame) else pd.DataFrame()

    if sections.empty and evidence.empty:
        return '<p class="table-note">No branch-aware event-evidence audit was exported.</p>'

    if sections.empty:
        frame = evidence.copy()
        frame["course_section"] = np.where(
            frame.get("evidence_scope", pd.Series(dtype=str)).astype(str).str.contains("branch_specific"),
            "branch",
            "shared",
        )
        frame["branch_ids"] = frame.get("branch_ids", "")
    elif evidence.empty:
        frame = sections.copy()
        frame["measured_unique_lap_count_before_local_compatibility"] = ""
        frame["eligible_unique_lap_count"] = ""
        frame["compatible_unique_lap_count"] = ""
        frame["evidence_pooling"] = ""
        frame["compatible_route_variant_ids"] = ""
    else:
        keep = [
            column
            for column in (
                "event_id",
                "measured_unique_lap_count_before_local_compatibility",
                "eligible_unique_lap_count",
                "compatible_unique_lap_count",
                "evidence_scope",
                "evidence_pooling",
                "compatible_route_variant_ids",
                "contributing_source_route_variant_ids",
            )
            if column in evidence.columns
        ]
        frame = sections.merge(evidence[keep], on="event_id", how="left")

    order = {"shared": 0, "branch": 1}
    frame["_section_order"] = frame["course_section"].map(order).fillna(9)
    frame = frame.sort_values(["_section_order", "event_id"], kind="stable")

    rows: list[str] = []
    for _, row in frame.iterrows():
        pooling = str(row.get("evidence_pooling", ""))
        if not pooling:
            pooling = (
                "all supported traversals"
                if str(row.get("course_section", "")) == "shared"
                else "matching branch only"
            )
        pooling = pooling.replace("all_supported_branches_on_shared_course_section", "all supported traversals")
        pooling = pooling.replace("single_shared_contract_reused_across_supported_traversals", "one shared contract across all supported traversals")
        pooling = pooling.replace("route_branch_only_inside_divergence_corridor", "matching branch only")
        compatible_routes = str(row.get("compatible_route_variant_ids", ""))
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('event_id', '')))}</td>"
            f"<td>{escape(str(row.get('event_name', '')))}</td>"
            f"<td><strong>{escape(str(row.get('course_section', '')))}</strong></td>"
            f"<td>{escape(str(row.get('branch_ids', '')))}</td>"
            f"<td>{_display_count(row.get('measured_unique_lap_count_before_local_compatibility', ''))}</td>"
            f"<td>{_display_count(row.get('eligible_unique_lap_count', ''))}</td>"
            f"<td>{_display_count(row.get('compatible_unique_lap_count', ''))}</td>"
            f"<td>{escape(pooling)}</td>"
            f"<td>{escape(compatible_routes)}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap compact"><table><thead><tr>'
        '<th>Event</th><th>Name</th><th>Course section</th><th>Branch</th>'
        '<th>Measured before local check</th><th>Used unique laps</th><th>Compatible unique laps</th>'
        '<th>Evidence pool</th><th>Compatible traversals</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
    )


def _display_count(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return str(int(numeric)) if np.isfinite(numeric) else ""

def _branch_frame(result: Any) -> pd.DataFrame:
    frame = getattr(result, "branch_summary", pd.DataFrame())
    return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _inside_interval_mask(s: np.ndarray, length: float, start: float, end: float) -> np.ndarray:
    values = np.mod(np.asarray(s, float), float(length))
    start = float(start) % float(length)
    end = float(end) % float(length)
    if start <= end:
        return (values >= start) & (values <= end)
    return (values >= start) | (values <= end)


def _outside_intervals_mask(
    s: np.ndarray, length: float, intervals: list[tuple[float, float]]
) -> np.ndarray:
    mask = np.ones(len(s), dtype=bool)
    for start, end in intervals:
        mask &= ~_inside_interval_mask(s, length, start, end)
    return mask


def _plot_masked_path(axis: Any, centreline: Any, mask: np.ndarray, **kwargs: Any) -> None:
    x = np.asarray(centreline.x_m, float).copy()
    y = np.asarray(centreline.y_m, float).copy()
    x[~mask] = np.nan
    y[~mask] = np.nan
    axis.plot(x, y, **kwargs)


def _row_int(row: pd.Series | None, key: str) -> int | None:
    if row is None or key not in row:
        return None
    value = pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0]
    return int(value) if pd.notna(value) else None


def _pre_consensus_geometry_lap_count(member: Any) -> int:
    laps = member.laps
    if "pre_consensus_valid" in laps:
        return int(laps["pre_consensus_valid"].fillna(False).astype(bool).sum())
    return _valid_geometry_lap_count(member)


def _valid_geometry_lap_count(member: Any) -> int:
    laps = member.laps
    return int(laps["analysis_valid"].fillna(False).astype(bool).sum()) if "analysis_valid" in laps else 0


def _consensus_excluded_lap_count(member: Any) -> int:
    laps = member.laps
    return int(laps["consensus_excluded"].fillna(False).astype(bool).sum()) if "consensus_excluded" in laps else 0


def _accepted_gate_count(member: Any) -> int:
    review = member.gate_review
    if review is None or review.empty or "recommendation" not in review:
        return 0
    return int((review["recommendation"].astype(str) == "accepted").sum())


def _eligible_gate_lap_count(member: Any) -> int:
    passes = member.event_passes
    if passes is None or passes.empty or not {"eligible", "lap_id"}.issubset(passes.columns):
        return 0
    return int(passes.loc[passes["eligible"].fillna(False).astype(bool), "lap_id"].nunique())


def _safe_int(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return int(number) if np.isfinite(number) else 0
