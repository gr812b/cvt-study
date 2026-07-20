from __future__ import annotations

from pathlib import Path

from cvt_track_study.reports.postprocess import write_structural_report_manifest


def test_postprocess_regenerates_before_registering(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "structural_sensitivity_report.html"
    calls: list[Path] = []

    def fake_regenerate(output: Path) -> Path:
        calls.append(output)
        target.write_text("<html></html>", encoding="utf-8")
        return target

    monkeypatch.setattr(
        "cvt_track_study.studies.structural_reporting.regenerate_structural_outputs",
        fake_regenerate,
    )

    report_path = write_structural_report_manifest(tmp_path)

    assert calls == [tmp_path.resolve()]
    assert report_path == target
    assert report_path.is_file()
    manifest_path = tmp_path / "report_manifest.json"
    assert manifest_path.is_file()
    assert "structural_sensitivity_report.html" in manifest_path.read_text(encoding="utf-8")
