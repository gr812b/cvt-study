from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from cvt_track_study.bundle import (
    TrackBundleError,
    build_track_bundle,
    load_track_bundle,
    validate_track_bundle,
)
from cvt_track_study.bundle.canonical import content_fingerprint
from cvt_track_study.track import build_project_track


REFERENCE = Path(__file__).resolve().parents[1] / "examples" / "reference_project"
ARIZONA = Path(__file__).resolve().parents[1] / "examples" / "arizona_endurance_project"


def _built_bundle(project: Path, tmp_path: Path):
    result = build_project_track(project, output_directory=tmp_path / "build")
    path = tmp_path / "build" / "track_bundle.json"
    return result, path, load_track_bundle(path)


def test_track_build_exports_valid_self_contained_bundle(tmp_path: Path) -> None:
    result, path, bundle = _built_bundle(ARIZONA, tmp_path)
    assert path.exists()
    assert path.with_name("track_bundle.sha256").exists()
    assert bundle.schema_version == "1.2.0"
    assert bundle.track_length_m == pytest.approx(result.centreline.length_m)
    assert len(bundle.physical_features) == 40
    assert len(bundle.response_groups) == 37
    assert len(bundle.active_speed_gates) == 13
    assert bundle.data["simulation_contract"]["grade_force_enabled"] is False
    assert bundle.data["simulation_contract"]["capabilities"] == {
        "grade_force_ready": False,
        "obstacle_models_ready": True,
        "speed_gates_ready": True,
        "uncertainty_roles_ready": True,
    }



def test_bundle_json_schema_matches_phase6_role_contract() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "cvt_track_study"
        / "bundle"
        / "track_bundle_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$id"].endswith(":1.2.0")
    assert schema["properties"]["schema_version"]["pattern"] == r"^1\.2\.[0-9]+$"
    required = schema["properties"]["simulation_contract"]["properties"]["capabilities"]["required"]
    assert "uncertainty_roles_ready" in required


def test_bundle_rejects_missing_uncertainty_role_capability(tmp_path: Path) -> None:
    result = build_project_track(REFERENCE, output_directory=tmp_path / "build")
    data = build_track_bundle(result)
    del data["simulation_contract"]["capabilities"]["uncertainty_roles_ready"]
    data["content_fingerprint_sha256"] = content_fingerprint(data)
    with pytest.raises(TrackBundleError, match="uncertainty_roles_ready"):
        validate_track_bundle(data)

def test_bundle_gate_distribution_retains_paired_lap_evidence(tmp_path: Path) -> None:
    _, _, bundle = _built_bundle(REFERENCE, tmp_path)
    gate = bundle.active_speed_gates[0]
    distribution = gate["target_speed_distribution"]
    samples = distribution["samples"]
    assert distribution["distribution"] == "empirical"
    assert distribution["sampling_unit"] == "eligible_lap_pass"
    assert len(samples) == distribution["summary"]["sample_count"]
    assert {"lap_id", "run_id", "vehicle_id", "driver_id", "value_mps"} <= set(
        samples[0]
    )
    assert gate["position_semantics"] == "physical_feature_entry_boundary"
    assert gate["enforcement_contract"]["slow_vehicle_reset_allowed"] is False


def test_bundle_carries_declared_physical_obstacle_models(tmp_path: Path) -> None:
    _, _, bundle = _built_bundle(REFERENCE, tmp_path)
    assert all(
        feature["obstacle_model"]["status"] == "declared"
        for feature in bundle.physical_features
    )
    assert all(
        group["obstacle_model"]["status"] == "not_applicable"
        for group in bundle.response_groups
    )
    assert (
        bundle.data["uncertainty_contract"]["obstacle_models"]["propagation_status"]
        == "ready_for_role_separated_sampling"
    )


def test_bundle_remains_loadable_without_project_or_gpx_files(tmp_path: Path) -> None:
    _, path, _ = _built_bundle(REFERENCE, tmp_path)
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    shutil.copy2(path, standalone / path.name)
    shutil.copy2(path.with_name("track_bundle.sha256"), standalone / "track_bundle.sha256")
    shutil.rmtree(tmp_path / "build")
    loaded = load_track_bundle(standalone / "track_bundle.json")
    assert loaded.track_length_m > 500.0
    assert loaded.active_speed_gates


def test_bundle_contains_no_absolute_project_path(tmp_path: Path) -> None:
    _, path, _ = _built_bundle(REFERENCE, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert str(REFERENCE.resolve()) not in text
    assert str(tmp_path.resolve()) not in text
    data = json.loads(text)
    assert all(item["scope"] in {"project", "external_profile"} for item in data["provenance"]["source_files"])



def test_bundle_provenance_source_order_uses_portable_identity(tmp_path: Path) -> None:
    _, _, bundle = _built_bundle(REFERENCE, tmp_path)
    source_files = bundle.data["provenance"]["source_files"]
    identities = [
        (item["scope"], item["path"], item["sha256"])
        for item in source_files
    ]
    assert identities == sorted(identities)


def test_bundle_content_fingerprint_is_reproducible(tmp_path: Path) -> None:
    first = build_project_track(REFERENCE, output_directory=tmp_path / "first")
    second = build_project_track(REFERENCE, output_directory=tmp_path / "second")
    first_data = build_track_bundle(first)
    second_data = build_track_bundle(second)
    assert first_data["created_utc"] != second_data["created_utc"]
    assert first_data["content_fingerprint_sha256"] == second_data["content_fingerprint_sha256"]


def test_bundle_checksum_detects_byte_level_tampering(tmp_path: Path) -> None:
    _, path, _ = _built_bundle(REFERENCE, tmp_path)
    path.write_bytes(path.read_bytes() + b" \n")
    with pytest.raises(TrackBundleError, match="checksum mismatch"):
        load_track_bundle(path)
    # JSON content remains valid when the explicit integrity check is disabled.
    assert load_track_bundle(path, verify_checksum=False).track_length_m > 0.0


def test_bundle_rejects_internal_content_tampering_without_sidecar(tmp_path: Path) -> None:
    _, path, _ = _built_bundle(REFERENCE, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["identity"]["track_name"] = "silently changed"
    path.write_text(json.dumps(data), encoding="utf-8")
    path.with_name("track_bundle.sha256").unlink()
    with pytest.raises(TrackBundleError, match="content_fingerprint"):
        load_track_bundle(path)


def test_bundle_version_policy_accepts_patch_but_rejects_new_minor(tmp_path: Path) -> None:
    result = build_project_track(REFERENCE, output_directory=tmp_path / "build")
    original = build_track_bundle(result)

    patch = deepcopy(original)
    patch["schema_version"] = "1.2.9"
    patch["content_fingerprint_sha256"] = content_fingerprint(patch)
    validate_track_bundle(patch)

    future_minor = deepcopy(original)
    future_minor["schema_version"] = "1.3.0"
    future_minor["content_fingerprint_sha256"] = content_fingerprint(future_minor)
    with pytest.raises(TrackBundleError, match="newer than supported"):
        validate_track_bundle(future_minor)

    legacy_minor = deepcopy(original)
    legacy_minor["schema_version"] = "1.1.9"
    legacy_minor["content_fingerprint_sha256"] = content_fingerprint(legacy_minor)
    with pytest.raises(TrackBundleError, match="older than supported"):
        validate_track_bundle(legacy_minor)


def test_bundle_rejects_inconsistent_closed_course_interval(tmp_path: Path) -> None:
    result = build_project_track(REFERENCE, output_directory=tmp_path / "build")
    data = build_track_bundle(result)
    data["simulation_contract"]["physical_features"][0]["interval"]["length_m"] += 1.0
    data["content_fingerprint_sha256"] = content_fingerprint(data)
    with pytest.raises(TrackBundleError, match="length_m is inconsistent"):
        validate_track_bundle(data)


def test_bundle_rejects_active_unaccepted_gate(tmp_path: Path) -> None:
    result = build_project_track(REFERENCE, output_directory=tmp_path / "build")
    data = build_track_bundle(result)
    gate = data["simulation_contract"]["speed_gates"][0]
    gate["status"] = "recommended_review"
    gate["active_by_default"] = True
    data["content_fingerprint_sha256"] = content_fingerprint(data)
    with pytest.raises(TrackBundleError, match="active only when status is accepted"):
        validate_track_bundle(data)


def test_bundle_fingerprint_is_independent_of_vehicle_physical_config(
    tmp_path: Path,
) -> None:
    from cvt_track_study.config.toml_io import dump_toml, load_toml

    project = tmp_path / "project"
    shutil.copytree(REFERENCE, project)
    original_result = build_project_track(project, output_directory=tmp_path / "original")
    original = build_track_bundle(original_result)

    vehicle_path = project / "vehicles" / "vehicle_A" / "vehicle.toml"
    vehicle = load_toml(vehicle_path)
    vehicle["vehicle"]["mass"]["nominal"] += 50.0
    vehicle["vehicle"]["mass"]["source"]["reference"] = "deliberate unrelated change"
    dump_toml(vehicle, vehicle_path)

    changed_result = build_project_track(project, output_directory=tmp_path / "changed")
    changed = build_track_bundle(changed_result)
    assert original["content_fingerprint_sha256"] == changed["content_fingerprint_sha256"]
    provenance_text = json.dumps(changed["provenance"])
    assert "vehicles.vehicle_A" not in provenance_text
    assert "studies." not in provenance_text


def test_simulator_view_depends_only_on_loaded_bundle(tmp_path: Path) -> None:
    from cvt_track_study.bundle import simulation_track_from_bundle

    _, path, bundle = _built_bundle(REFERENCE, tmp_path)
    view = simulation_track_from_bundle(bundle)
    shutil.rmtree(tmp_path / "build" / "ingestion")
    shutil.rmtree(tmp_path / "build" / "track")
    assert view.length_m == pytest.approx(bundle.track_length_m)
    assert len(view.active_speed_gates) == len(bundle.active_speed_gates)
    assert len(view.centreline_s_m) > 100
    assert view.grade_force_enabled is False
    assert view.ready_for_full_vehicle_simulation is True
    assert all(gate.empirical_speed_samples_mps for gate in view.active_speed_gates)


def test_bundle_reader_and_simulator_view_do_not_import_reconstruction_stack(
    tmp_path: Path,
) -> None:
    import os
    import subprocess
    import sys

    script = r'''
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pandas" or fullname.startswith("cvt_track_study.track") or fullname.startswith("cvt_track_study.gpx"):
            raise RuntimeError(f"forbidden import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
from cvt_track_study.bundle import TrackBundle, simulation_track_from_bundle
print(TrackBundle.__name__, callable(simulation_track_from_bundle))
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "TrackBundle True" in completed.stdout


def test_bundle_non_finite_values_fail_with_contract_error(tmp_path: Path) -> None:
    result = build_project_track(REFERENCE, output_directory=tmp_path / "build")
    data = build_track_bundle(result)
    data["evidence"]["event_passes"][0]["entry_speed_mps"] = float("nan")
    with pytest.raises(TrackBundleError, match="non-finite"):
        validate_track_bundle(data)


def test_empty_checksum_file_is_actionable_error(tmp_path: Path) -> None:
    _, path, _ = _built_bundle(REFERENCE, tmp_path)
    path.with_name("track_bundle.sha256").write_text("", encoding="utf-8")
    with pytest.raises(TrackBundleError, match="checksum file is empty"):
        load_track_bundle(path)


def test_public_bundle_builder_annotations_resolve() -> None:
    import typing

    hints = typing.get_type_hints(build_track_bundle)
    assert typing.get_origin(hints["return"]) is dict
