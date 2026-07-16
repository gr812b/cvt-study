from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from cvt_track_study.runtime.cache import SimulationCache
from cvt_track_study.runtime.migration import migrate_prototype_events
from cvt_track_study.runtime.models import ModelRegistration, ModelRegistry
from cvt_track_study.runtime.provenance import canonical_fingerprint, write_provenance
from cvt_track_study.runtime.results import discover_results, write_results_index
from cvt_track_study.runtime.workspace import ResultWorkspace, WorkspaceError


def test_content_cache_is_stable_and_strict(tmp_path: Path) -> None:
    cache = SimulationCache(tmp_path / "cache")
    key = cache.key({"b": 2, "a": 1})
    assert key == cache.key({"a": 1, "b": 2})
    assert cache.get(key) is None
    cache.put(key, {"finite": 1.25})
    assert cache.get(key) == {"finite": 1.25}
    assert cache.status()["entry_count"] == 1
    with pytest.raises(ValueError):
        cache.get("not-a-sha")


def test_cache_session_counters_are_thread_safe(tmp_path: Path) -> None:
    cache = SimulationCache(tmp_path / "cache", enabled=False)
    key = cache.key({"case": "shared"})
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(cache.get, [key] * 500)) == [None] * 500
    assert cache.status()["session_misses"] == 500


def test_result_workspace_resume_and_commit(tmp_path: Path) -> None:
    output = tmp_path / "result"
    first = ResultWorkspace(output, fingerprint="abc")
    first.write_checkpoint(2, {"result": {"answer": 42}})
    with pytest.raises(WorkspaceError, match="--resume"):
        ResultWorkspace(output, fingerprint="abc")
    resumed = ResultWorkspace(output, fingerprint="abc", resume=True)
    assert resumed.load_checkpoint(2)["result"]["answer"] == 42
    committed = resumed.commit()
    assert committed == output
    assert not (output / "checkpoints").exists()
    assert json.loads((output / "workspace.json").read_text())["state"] == "complete"


def test_result_workspace_rejects_mismatched_resume(tmp_path: Path) -> None:
    output = tmp_path / "result"
    ResultWorkspace(output, fingerprint="first")
    with pytest.raises(WorkspaceError, match="different resolved inputs"):
        ResultWorkspace(output, fingerprint="second", resume=True)
    restarted = ResultWorkspace(output, fingerprint="second", restart=True)
    assert restarted.path.exists()


def test_model_registry_rejects_duplicate_and_reports_available() -> None:
    registry = ModelRegistry()
    registration = ModelRegistration("ideal", "drivetrain", dict, "test")
    registry.register(registration)
    assert registry.resolve("drivetrain", "ideal") is registration
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registration)
    with pytest.raises(KeyError, match="ideal"):
        registry.resolve("drivetrain", "missing")


def test_prototype_migration_preserves_geometry_but_not_physics(tmp_path: Path) -> None:
    source = tmp_path / "events.csv"
    source.write_text("event_name,entry_s_m,coefficient\nLog Row,12.5,99\n", encoding="utf-8")
    destination = tmp_path / "events.toml"
    assert migrate_prototype_events(source, destination) == 1
    text = destination.read_text(encoding="utf-8")
    assert 'id = "log_row"' in text
    assert "anchor_s_m = 12.5" in text
    assert 'obstacle_model = "unset"' in text
    assert "99" not in text


def test_results_index_ignores_incomplete_workspaces(tmp_path: Path) -> None:
    complete = tmp_path / "design_sweep" / "run"
    complete.mkdir(parents=True)
    (complete / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": "2026-01-01T00:00:00+00:00",
                "study_name": "gearing",
                "study_type": "design_sweep",
                "numerical_quality": {"numerically_valid": True},
                "evidence_assessment": {"ready": False},
                "decision_readiness": {"decision_ready": False},
            }
        ),
        encoding="utf-8",
    )
    incomplete = tmp_path / "full_uncertainty" / ".run.incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "run_manifest.json").write_text("{}", encoding="utf-8")
    records = discover_results(tmp_path)
    assert len(records) == 1
    index = write_results_index(tmp_path)
    assert "gearing" in index.read_text(encoding="utf-8")


def test_provenance_outputs_strict_json_and_svg(tmp_path: Path) -> None:
    provenance = {
        "study_name": "study",
        "study_fingerprint_sha256": canonical_fingerprint({"x": 1}),
        "track_bundle_sha256": "a" * 64,
        "track_bundle_file": "track_bundle.json",
    }
    write_provenance(tmp_path, provenance)
    assert json.loads((tmp_path / "provenance.json").read_text())["study_name"] == "study"
    assert (tmp_path / "provenance_graph.svg").read_text().startswith("<svg")
