from __future__ import annotations

from pathlib import Path


def test_repository_and_packaged_project_templates_are_identical() -> None:
    root = Path(__file__).resolve().parents[1]
    repository = root / "templates" / "project"
    packaged = root / "src" / "cvt_track_study" / "project_template"
    repository_files = sorted(
        path.relative_to(repository) for path in repository.rglob("*") if path.is_file()
    )
    packaged_files = sorted(
        path.relative_to(packaged) for path in packaged.rglob("*") if path.is_file()
    )
    assert repository_files == packaged_files
    for relative in repository_files:
        assert (repository / relative).read_bytes() == (packaged / relative).read_bytes()
