from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cvt_track_study.track.family_review import (
    augment_track_review_with_route_family,
    route_family_review_rows,
)


class _Centreline:
    def __init__(self, offset: float, length: float) -> None:
        t = np.linspace(0.0, 2.0 * np.pi, 49)
        self.x_m = 100.0 * np.cos(t) + offset
        self.y_m = 60.0 * np.sin(t)
        step = np.hypot(np.diff(self.x_m), np.diff(self.y_m))
        raw = np.concatenate(([0.0], np.cumsum(step)))
        self.s_m = raw / raw[-1] * length
        self.length_m = length


def _member(offset: float, valid_ids: set[int], length: float) -> SimpleNamespace:
    laps = pd.DataFrame(
        {
            "lap_id": [1, 2, 3, 4],
            "analysis_valid": [lap_id in valid_ids for lap_id in [1, 2, 3, 4]],
            "pre_consensus_valid": [lap_id in valid_ids for lap_id in [1, 2, 3, 4]],
            "consensus_excluded": [False, False, False, False],
        }
    )
    points: list[dict[str, float | int]] = []
    for lap_id in [1, 2, 3, 4]:
        t = np.linspace(0.0, 2.0 * np.pi, 36)
        for x, y in zip(100.0 * np.cos(t) + offset, 60.0 * np.sin(t)):
            points.append({"lap_id": lap_id, "x_m": x, "y_m": y})
    return SimpleNamespace(
        centreline=_Centreline(offset, length),
        laps=laps,
        matched_points=pd.DataFrame(points),
        event_projection=pd.DataFrame({"id": ["a", "b"]}),
        gate_review=pd.DataFrame(
            {"recommendation": ["accepted", "recommended_review"]}
        ),
        event_passes=pd.DataFrame(
            {"eligible": [True, True, False], "lap_id": [1, 2, 3]}
        ),
    )


def _result() -> SimpleNamespace:
    route_001 = _member(0.0, {1, 2}, 2100.0)
    route_002 = _member(30.0, {3, 4}, 1870.0)
    summary = pd.DataFrame(
        [
            {
                "route_variant_id": "route_001",
                "supported": True,
                "lap_count": 15,
                "lap_ids": "1;2",
                "geometric_seed_cluster_count": 3,
                "median_path_distance_m": 2110.0,
                "topology_component_status": "supported_course_route",
            },
            {
                "route_variant_id": "route_002",
                "supported": True,
                "lap_count": 23,
                "lap_ids": "3;4",
                "geometric_seed_cluster_count": 4,
                "median_path_distance_m": 1875.0,
                "topology_component_status": "supported_course_route",
            },
            {
                "route_variant_id": "route_003",
                "supported": False,
                "lap_count": 2,
                "lap_ids": "20;21",
                "geometric_seed_cluster_count": 2,
                "median_path_distance_m": 1950.0,
                "topology_component_status": "isolated_or_ambiguous_outlier",
            },
        ]
    )
    branches = pd.DataFrame(
        [
            {
                "branch_id": "branch_001",
                "nominal_route_variant_id": "route_002",
                "alternate_route_variant_id": "route_001",
                "shared_section_valid_lap_count": 4,
                "nominal_branch_valid_geometry_lap_count": 2,
                "alternate_branch_valid_geometry_lap_count": 2,
                "nominal_branch_start_s_m": 250.0,
                "nominal_branch_end_s_m": 520.0,
                "alternate_branch_start_s_m": 260.0,
                "alternate_branch_end_s_m": 720.0,
                "nominal_branch_length_m": 270.0,
                "alternate_branch_length_m": 460.0,
                "divergence_distance_threshold_m": 25.0,
            }
        ]
    )
    event_sections = pd.DataFrame(
        [
            {"route_variant_id": "route_001", "event_id": "shared_gate", "event_name": "Shared gate", "route_section_kind": "shared", "branch_ids": ""},
            {"route_variant_id": "route_002", "event_id": "shared_gate", "event_name": "Shared gate", "route_section_kind": "shared", "branch_ids": ""},
            {"route_variant_id": "route_001", "event_id": "branch_gate", "event_name": "Branch gate", "route_section_kind": "branch", "branch_ids": "branch_001"},
            {"route_variant_id": "route_002", "event_id": "branch_gate", "event_name": "Branch gate", "route_section_kind": "branch", "branch_ids": "branch_001"},
        ]
    )
    shared_gate_evidence = pd.DataFrame(
        [
            {"event_id": "shared_gate", "measured_unique_lap_count_before_local_compatibility": 4, "eligible_unique_lap_count": 4, "compatible_unique_lap_count": 4, "evidence_scope": "shared_course", "evidence_pooling": "all_supported_branches_on_shared_course_section", "compatible_route_variant_ids": "route_001;route_002"},
            {"event_id": "branch_gate", "measured_unique_lap_count_before_local_compatibility": 4, "eligible_unique_lap_count": 2, "compatible_unique_lap_count": 4, "evidence_scope": "branch_specific", "evidence_pooling": "route_branch_only_inside_divergence_corridor", "compatible_route_variant_ids": "route_001;route_002"},
        ]
    )
    return SimpleNamespace(
        route_family_members={"route_001": route_001, "route_002": route_002},
        nominal_route_variant_id="route_002",
        route_variant_summary=summary,
        branch_summary=branches,
        event_route_sections=event_sections,
        shared_gate_evidence=shared_gate_evidence,
    )


def test_family_review_rows_show_shared_and_branch_support_separately() -> None:
    rows = route_family_review_rows(_result()).set_index("route_variant_id")
    assert int(rows.loc["route_001", "assigned_topology_laps"]) == 15
    assert int(rows.loc["route_002", "assigned_topology_laps"]) == 23
    assert int(rows.loc["route_001", "branch_valid_geometry_laps"]) == 2
    assert int(rows.loc["route_002", "branch_valid_geometry_laps"]) == 2
    assert int(rows.loc["route_001", "shared_section_support_laps"]) == 4
    assert rows.loc["route_002", "role"] == "nominal branch"


def test_family_review_is_inserted_into_main_track_html(tmp_path: Path) -> None:
    report = tmp_path / "track_evidence_report.html"
    report.write_text(
        '<html><body><main><nav class="report-nav" aria-label="Report sections">'
        '<a href="#executive-evidence-summary">Executive evidence summary</a></nav>'
        '<div class="label">Reconstructed length</div>'
        '<div class="label">Valid evidence laps</div>'
        '<h2 id="executive-evidence-summary">Executive evidence summary</h2>'
        '</main></body></html>',
        encoding="utf-8",
    )
    image = tmp_path / "route_family_map.png"
    augment_track_review_with_route_family(report, image, _result())
    text = report.read_text(encoding="utf-8")
    assert image.exists()
    assert 'href="#route-family-reconstruction"' in text
    assert 'id="route-family-reconstruction"' in text
    assert "Shared-section valid laps" in text
    assert "Evidence pooling by event" in text
    assert "Measured before local check" in text
    assert "continuous bracketing" in text
    assert "Shared gate" in text and "Branch gate" in text
    assert "all supported traversals" in text
    assert "matching branch only" in text
    assert "branch_001" in text
    assert "15" in text and "23" in text
    assert "20;21" in text
    assert "Nominal branch valid laps" in text
    assert "Nominal traversal length" in text
