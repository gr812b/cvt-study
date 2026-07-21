from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cvt_track_study.reports.html import render_page
from cvt_track_study.reports.postprocess import write_track_evidence_report


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(2, 1.2))
    axis.plot([0, 1], [0, 1])
    figure.tight_layout()
    figure.savefig(path, dpi=60)
    plt.close(figure)


def test_render_page_adds_one_quick_link_bar_and_stable_heading_ids() -> None:
    page = render_page(
        title="Example",
        subtitle="Example question",
        body="<h2>Executive summary</h2><p>A</p><h2>Detailed tables</h2><p>B</p>",
        report_key="example",
    )
    assert page.count('class="report-nav"') == 1
    assert 'href="#executive-summary"' in page
    assert 'href="#detailed-tables"' in page
    assert '<h2 id="executive-summary">Executive summary</h2>' in page
    assert 'id="top"' in page


def test_render_page_preserves_deliberate_navigation_without_duplication() -> None:
    body = (
        '<nav class="report-nav"><a href="#summary">Summary</a></nav>'
        '<h2 id="summary">Summary</h2>'
    )
    page = render_page(
        title="Example",
        subtitle="Example question",
        body=body,
        report_key="example",
    )
    assert page.count('class="report-nav"') == 1


def test_track_evidence_uses_shared_shell_and_rewrites_legacy_alias(tmp_path: Path) -> None:
    (tmp_path / "review").mkdir()
    (tmp_path / "track").mkdir()
    (tmp_path / "ingestion").mkdir()

    for name in (
        "track_map.png",
        "telemetry_cleanup_map.png",
        "event_group_timeline.png",
        "elevation_profile.png",
    ):
        _png(tmp_path / "review" / name)

    (tmp_path / "track_build_manifest.json").write_text(
        json.dumps({"track_length_m": 1000.0, "valid_lap_count": 2}),
        encoding="utf-8",
    )
    (tmp_path / "diagnostics.json").write_text("[]", encoding="utf-8")
    (tmp_path / "ingestion" / "run_summaries.json").write_text("[]", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "lap_id": 1,
                "run_id": "run-a",
                "vehicle_id": "car",
                "duration_s": 100.0,
                "analysis_valid": True,
                "speed_coverage_fraction": 1.0,
                "p95_map_error_m": 1.0,
                "time_gap_count": 0,
                "quality_flags": "",
            }
        ]
    ).to_csv(tmp_path / "track" / "lap_quality.csv", index=False)
    pd.DataFrame(
        [{"sequence": 1, "response_group_id": "E01", "name": "Turn"}]
    ).to_csv(tmp_path / "track" / "event_projection.csv", index=False)
    pd.DataFrame(
        [
            {
                "sequence": 1,
                "response_group_id": "E01",
                "name": "Turn",
                "feature_start_s_m": 10.0,
                "feature_end_s_m": 20.0,
                "feature_length_m": 10.0,
                "wraps_start_finish": False,
                "interval_audit_flags": "",
            }
        ]
    ).to_csv(tmp_path / "track" / "event_interval_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "response_group_id": "E01",
                "event_name": "Turn",
                "sequence": 1,
                "recommendation": "accepted",
                "overall_confidence_score": 80.0,
                "valid_pass_count": 8,
            }
        ]
    ).to_csv(tmp_path / "track" / "gate_review.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "ingestion" / "rejected_telemetry_points.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "track" / "rejected_map_points.csv", index=False)

    path = write_track_evidence_report(tmp_path)
    legacy = tmp_path / "review" / "track_review.html"
    text = path.read_text(encoding="utf-8")

    assert path.name == "track_evidence_report.html"
    assert legacy.read_text(encoding="utf-8") == text
    assert text.count('class="report-nav"') == 1
    assert 'href="#gate-evidence-and-qualification"' in text
    assert 'data-sortable="true"' in text
    assert 'class="sticky-col identity-column"' in text
    assert "Measured-track drivetrain framework" in text
