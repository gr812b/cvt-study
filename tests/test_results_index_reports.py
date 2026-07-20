from __future__ import annotations

import json
from pathlib import Path

from cvt_track_study.runtime.results import discover_results, write_results_index


def test_index_discovers_track_evidence_and_other_canonical_reports(tmp_path: Path) -> None:
    track = tmp_path / "track_build" / "run"
    (track / "review").mkdir(parents=True)
    (track / "review" / "track_evidence_report.html").write_text("track", encoding="utf-8")
    (track / "report_manifest.json").write_text(
        json.dumps(
            {
                "report_key": "track_evidence",
                "title": "Track evidence and reconstruction",
                "html_file": "review/track_evidence_report.html",
                "generated_utc": "2026-07-19T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    full = tmp_path / "full_uncertainty" / "run"
    full.mkdir(parents=True)
    (full / "full_uncertainty_report.html").write_text("full", encoding="utf-8")
    (full / "report_manifest.json").write_text(
        json.dumps(
            {
                "report_key": "full_uncertainty",
                "title": "Full uncertainty",
                "html_file": "full_uncertainty_report.html",
                "generated_utc": "2026-07-19T01:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    records = discover_results(tmp_path)
    assert [record["type"] for record in records] == ["full_uncertainty", "track_evidence"]
    index = write_results_index(tmp_path).read_text(encoding="utf-8")
    assert "review/track_evidence_report.html" in index
    assert "full_uncertainty_report.html" in index
