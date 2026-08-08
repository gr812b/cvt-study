from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cvt_track_study.bundle.builder import _finalize_bundle, _simulatable_speed_gates
from cvt_track_study.track.family import (
    RouteFamilySettings,
    _course_cases,
    _event_compatibility,
    _event_is_route_applicable,
    _interval_coverage,
    _map_segment_nearest,
    _merge_geometric_clusters_into_topology_families,
    _shared_course_projection_error_limit,
    _shared_contract_event_ids,
    _deduplicated_shared_contract_passes,
    _shared_gate_evidence,
    _sustained_divergence_interval,
    _s_in_circular_interval,
    _supported_variant_ids,
)
from cvt_track_study.track.geo import Centreline, LocalFrame
from cvt_track_study.track.robustness_family import (
    _execute_family_case,
    _prepare_case_summary_for_enrichment,
    _select_route_variant,
    _spacing_decoupled_cases,
    route_family_robustness_enabled,
)
from cvt_track_study.studies.route_family import (
    _balanced_family_variants,
    _equal_course_summary,
)



def test_branch_only_event_is_not_relocated_to_another_route():
    applicable, reasons = _event_is_route_applicable(
        role="feature",
        anchor_error_m=24.0,
        start_error_m=23.0,
        end_error_m=26.0,
        maximum_error_m=12.0,
    )
    assert not applicable
    assert "anchor_not_on_route" in reasons

    shared, reasons = _event_is_route_applicable(
        role="feature",
        anchor_error_m=2.0,
        start_error_m=3.0,
        end_error_m=2.5,
        maximum_error_m=12.0,
    )
    assert shared
    assert reasons == []

def test_interval_coverage_uses_continuous_bracketing_not_raw_bin_occupancy():
    # The 4 m entry window contains no raw point, but it lies between two valid
    # consecutive 1 Hz samples and is therefore fully bracketed.
    relative = np.array([-12.0, -7.0, 1.0, 8.0, 16.0])
    assert _interval_coverage(relative, -5.0, -1.0, 5.0) == 1.0
    assert _interval_coverage(np.array([-10.0]), -10.0, 10.0, 5.0) == 0.0


def test_interval_coverage_does_not_bridge_invalid_or_large_gaps():
    with_invalid = np.array([-12.0, -7.0, np.nan, 1.0, 8.0])
    assert _interval_coverage(with_invalid, -5.0, -1.0, 5.0) == 0.0
    large_gap = np.array([-12.0, -7.0, 30.0])
    assert (
        _interval_coverage(
            large_gap, -5.0, -1.0, 5.0, maximum_gap_m=25.0
        )
        == 0.0
    )


def test_shared_course_projection_limit_stays_below_branch_scale():
    branches = pd.DataFrame(
        [{"divergence_distance_threshold_m": 25.0}]
    )
    assert _shared_course_projection_error_limit(
        strict_error_m=10.0, branch_summary=branches
    ) == 15.0
    assert _shared_course_projection_error_limit(
        strict_error_m=10.0, branch_summary=pd.DataFrame()
    ) == 10.0
    applicable, reasons = _event_is_route_applicable(
        role="feature",
        anchor_error_m=10.48,
        start_error_m=10.48,
        end_error_m=10.32,
        maximum_error_m=15.0,
    )
    assert applicable
    assert reasons == []


def test_event_compatibility_accepts_bracketed_short_window_on_shared_course():
    mapped = pd.DataFrame(
        {
            "lap_id": [1] * 7,
            "s_m": [70.0, 80.0, 88.0, 93.0, 101.0, 108.0, 118.0],
            "map_error_m": [12.0] * 7,
        }
    )
    laps = pd.DataFrame(
        {
            "lap_id": [1],
            "analysis_valid": [True],
            "use_for_gate_evidence": [True],
        }
    )
    events = pd.DataFrame(
        [
            {
                "id": "shared",
                "anchor_s_m": 100.0,
                "approach_start_rel_m": -30.0,
                "approach_end_rel_m": -10.0,
                "entry_start_rel_m": -5.0,
                "entry_end_rel_m": -1.0,
                "feature_start_rel_m": 0.0,
                "feature_end_rel_m": 5.0,
                "exit_start_rel_m": 7.0,
                "exit_end_rel_m": 17.0,
            }
        ]
    )
    shared = _event_compatibility(
        mapped=mapped,
        laps=laps,
        events=events,
        track_length_m=500.0,
        maximum_error_m=10.0,
        shared_maximum_error_m=15.0,
        event_section_lookup={"shared": "shared"},
        minimum_coverage=0.60,
        bin_size_m=5.0,
        maximum_interpolation_gap_m=25.0,
        maximum_backward=1,
    ).iloc[0]
    assert bool(shared["route_compatible"])
    assert shared["route_projection_entry_coverage"] == 1.0
    assert shared["route_projection_error_limit_m"] == 15.0

    branch = _event_compatibility(
        mapped=mapped,
        laps=laps,
        events=events,
        track_length_m=500.0,
        maximum_error_m=10.0,
        shared_maximum_error_m=15.0,
        event_section_lookup={"shared": "branch"},
        minimum_coverage=0.60,
        bin_size_m=5.0,
        maximum_interpolation_gap_m=25.0,
        maximum_backward=1,
    ).iloc[0]
    assert not bool(branch["route_compatible"])


def test_nearest_projection_is_not_forced_by_whole_lap_progress():
    centreline = Centreline(
        x_m=np.array([0.0, 10.0, 10.0, 0.0, 0.0]),
        y_m=np.array([0.0, 0.0, 10.0, 10.0, 0.0]),
        s_m=np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
        elevation_m=np.zeros(5),
        frame=LocalFrame(0.0, 0.0),
    )
    segment = pd.DataFrame({"x_m": [9.8], "y_m": [8.0]})
    mapped = _map_segment_nearest(segment, centreline)
    assert mapped.loc[0, "map_error_m"] < 0.3
    assert 17.0 < mapped.loc[0, "s_m"] < 19.0


def test_shared_gate_evidence_deduplicates_same_lap_across_routes():
    passes_a = pd.DataFrame(
        [
            {
                "event_id": "gate_shared",
                "event_name": "Shared gate",
                "lap_id": 1,
                "eligible": True,
                "route_compatible": True,
                "source_route_variant_id": "route_001",
            },
            {
                "event_id": "gate_shared",
                "event_name": "Shared gate",
                "lap_id": 2,
                "eligible": True,
                "route_compatible": True,
                "source_route_variant_id": "route_002",
            },
        ]
    )
    passes_b = passes_a.copy()
    result = _shared_gate_evidence(
        {
            "route_001": SimpleNamespace(event_passes=passes_a),
            "route_002": SimpleNamespace(event_passes=passes_b),
        }
    )
    row = result.iloc[0]
    assert row["eligible_unique_lap_count"] == 2
    assert row["compatible_route_variant_ids"] == "route_001;route_002"


def test_course_cases_are_equal_and_not_lap_count_weighted():
    routes = {
        "route_001": SimpleNamespace(centreline=SimpleNamespace(length_m=100.0)),
        "route_002": SimpleNamespace(centreline=SimpleNamespace(length_m=125.0)),
    }
    cases = _course_cases(routes, "route_001")
    assert list(cases["case_weight"]) == [1.0, 1.0]
    assert list(cases["normalized_equal_weight"]) == [0.5, 0.5]


def test_family_settings_are_nested_below_route_variants():
    settings = RouteFamilySettings.from_mapping(
        {
            "route_variants": {
                "family": {
                    "minimum_event_window_coverage_fraction": 0.7,
                    "event_projection_bin_size_m": 4.0,
                    "maximum_event_interpolation_gap_m": 22.0,
                }
            }
        }
    )
    assert settings.minimum_event_window_coverage_fraction == 0.7
    assert settings.event_projection_bin_size_m == 4.0
    assert settings.maximum_event_interpolation_gap_m == 22.0


def test_robustness_copy_selects_one_route_without_duplicate_table(tmp_path: Path):
    track = tmp_path / "track.toml"
    track.write_text(
        "[track]\nname = \"Maryland\"\n\n"
        "[track.route_variants]\n"
        "enabled = true\n"
        "selection = \"reference_run\"\n"
        "reference_run_id = \"run_a\"\n\n"
        "[track.reconstruction]\nprofile_spacing_m = 2.0\n",
        encoding="utf-8",
    )
    _select_route_variant(track, "route_002")
    text = track.read_text(encoding="utf-8")
    assert text.count("[track.route_variants]") == 1
    assert 'selection = "variant_id"' in text
    assert 'selected_variant_id = "route_002"' in text



def test_failed_only_robustness_summary_keeps_numeric_series_columns():
    frame = pd.DataFrame(
        [
            {
                "case_id": "gate_strict",
                "category": "gate_policy",
                "label": "Strict gate policy",
                "success": False,
                "error": "example",
            }
        ]
    )
    prepared = _prepare_case_summary_for_enrichment(frame)
    assert "track_length_m" in prepared
    assert "track_length_delta_m" in prepared
    assert prepared["track_length_m"].isna().all()
    assert prepared["track_length_delta_m"].isna().all()


def test_family_robustness_case_uses_topology_family_builder(monkeypatch):
    from cvt_track_study.track import robustness_family as module

    case = SimpleNamespace(
        identifier="centreline_case",
        category="centreline",
        label="Centreline case",
        rationale="test",
        overrides={},
        excluded_run_ids=(),
    )
    ingestion = SimpleNamespace(
        metadata=SimpleNamespace(run_id="run_a"),
        diagnostics=(),
    )
    evidence = SimpleNamespace(
        centreline=SimpleNamespace(length_m=123.0),
        laps=pd.DataFrame(
            {
                "analysis_valid": [True, True],
                "reference_lap": [True, False],
            }
        ),
        matched_points=pd.DataFrame(),
        track_profile=pd.DataFrame(),
        event_projection=pd.DataFrame(),
        response_features=pd.DataFrame(),
        event_passes=pd.DataFrame(),
        gate_evidence=pd.DataFrame(),
        gate_review=pd.DataFrame(
            {"recommendation": ["accepted", "recommended_review"]}
        ),
        rejected_map_points=pd.DataFrame(),
    )
    family = SimpleNamespace(
        nominal_route_variant_id="route_002",
        routes={"route_002": evidence},
        route_variant_summary=pd.DataFrame(),
        route_variant_pairwise=pd.DataFrame(),
        shared_gate_evidence=pd.DataFrame(),
        event_route_applicability=pd.DataFrame(),
        course_cases=pd.DataFrame(),
        branch_summary=pd.DataFrame(),
        event_route_sections=pd.DataFrame(),
    )
    called = {}

    def fake_family_builder(cleaned, config, raw_events, diagnostics):
        called["selection"] = config["route_variants"]["selected_variant_id"]
        return family

    monkeypatch.setattr(module, "apply_telemetry_cleanup", lambda item, config: item)
    monkeypatch.setattr(module, "build_track_family_evidence", fake_family_builder)
    resolution = SimpleNamespace(
        data={
            "track": {
                "route_variants": {
                    "enabled": True,
                    "selection": "variant_id",
                    "selected_variant_id": "route_002",
                }
            }
        }
    )
    result = _execute_family_case(
        case=case,
        resolution=resolution,
        parsed_runs=(ingestion,),
        track_config=resolution.data["track"],
        raw_events=(),
    )
    assert result.success
    assert called["selection"] == "route_002"
    assert result.track_build.route_variant_id == "route_002"
    assert result.track_build.metadata["supported_route_family_count"] == 1


def test_robustness_route_selection_preserves_header_newline_and_parses_toml(tmp_path: Path):
    import tomllib

    track = tmp_path / "track.toml"
    track.write_text(
        "[track]\nname = \"Maryland\"\n\n"
        "# Route detection comment immediately above the table.\n"
        "[track.route_variants]\n"
        "enabled = true\n"
        "minimum_supported_laps = 2\n"
        "selection = \"largest_supported\"\n"
        "selected_variant_id = \"\"\n\n"
        "[track.telemetry_cleanup]\nenabled = true\n",
        encoding="utf-8",
    )
    _select_route_variant(track, "route_001")
    text = track.read_text(encoding="utf-8")
    assert "[track.route_variants]\nenabled = true" in text
    assert "[track.route_variants]enabled" not in text
    parsed = tomllib.loads(text)
    route = parsed["track"]["route_variants"]
    assert route["enabled"] is True
    assert route["selection"] == "variant_id"
    assert route["selected_variant_id"] == "route_001"

def test_only_repeated_supported_clusters_become_family_members():
    summary = pd.DataFrame(
        [
            {"route_variant_id": "route_001", "supported": True},
            {"route_variant_id": "route_002", "supported": True},
            {"route_variant_id": "route_003", "supported": False},
        ]
    )
    assert _supported_variant_ids(summary) == ["route_001", "route_002"]


def test_equal_course_study_summary_uses_both_routes_equally():
    frame = pd.DataFrame(
        [
            {"route_variant_id": "route_001", "design_id": "d", "replicate": 0, "metric_x": 1.0},
            {"route_variant_id": "route_001", "design_id": "d", "replicate": 1, "metric_x": 3.0},
            {"route_variant_id": "route_002", "design_id": "d", "replicate": 0, "metric_x": 9.0},
            {"route_variant_id": "route_002", "design_id": "d", "replicate": 1, "metric_x": 11.0},
        ]
    )
    summary = _equal_course_summary(frame)
    row = summary[summary["metric"] == "metric_x"].iloc[0]
    assert row["course_case_count"] == 2
    assert row["mean"] == 6.0
    assert row["median"] == 6.0


def test_bundle_fingerprint_sanitizes_nan_before_strict_json():
    bundle = {
        "format": "test",
        "schema_version": 1,
        "identity": {"route_variant_id": "route_001"},
        "simulation_contract": {"optional_metric": float("nan")},
        "evidence": {},
        "uncertainty_contract": {},
        "provenance": {},
    }
    finalized = _finalize_bundle(bundle)
    assert finalized["simulation_contract"]["optional_metric"] is None
    assert len(finalized["content_fingerprint_sha256"]) == 64


def test_zero_sample_gate_is_not_exported_as_a_simulator_target():
    unavailable = {
        "id": "gate:start_finish",
        "target_speed_distribution": {"distribution": "empirical", "samples": []},
    }
    usable = {
        "id": "gate:shared",
        "target_speed_distribution": {
            "distribution": "empirical",
            "samples": [{"lap_id": 1, "value_mps": 4.2}],
        },
    }
    assert _simulatable_speed_gates([unavailable, usable]) == [usable]



def test_strict_line_clusters_merge_into_two_topology_families():
    cluster_specs = {
        "seed_long_a": (6, 2240.0, True),
        "seed_long_b": (5, 2220.0, True),
        "line_long_extra": (1, 2260.0, False),
        "seed_short_a": (12, 2025.0, True),
        "seed_short_b": (4, 1995.0, True),
        "line_short_extra": (1, 2040.0, False),
        "bad_excursion": (1, 6800.0, False),
    }
    lap_rows = []
    summary_rows = []
    first_lap_by_cluster = {}
    lap_id = 1
    for cluster_id, (count, distance, supported) in cluster_specs.items():
        ids = []
        for offset in range(count):
            ids.append(lap_id)
            lap_rows.append(
                {
                    "lap_id": lap_id,
                    "route_variant_id": cluster_id,
                    "path_distance_m": distance + offset,
                    "run_id": "run",
                    "vehicle_id": "vehicle",
                    "driver_id": "driver",
                }
            )
            lap_id += 1
        first_lap_by_cluster[cluster_id] = ids[0]
        summary_rows.append(
            {
                "route_variant_id": cluster_id,
                "lap_count": count,
                "supported": supported,
            }
        )

    def pair(left, right, p95, divergent, length):
        return {
            "left_lap_id": first_lap_by_cluster[left],
            "right_lap_id": first_lap_by_cluster[right],
            "symmetric_p95_nearest_path_distance_m": p95,
            "symmetric_divergent_fraction": divergent,
            "length_relative_difference": length,
        }

    pairwise = pd.DataFrame(
        [
            pair("seed_long_a", "seed_long_b", 8.0, 0.002, 0.01),
            pair("line_long_extra", "seed_long_a", 10.0, 0.02, 0.02),
            pair("seed_short_a", "seed_short_b", 8.5, 0.001, 0.02),
            pair("line_short_extra", "seed_short_a", 11.0, 0.01, 0.02),
            pair("seed_long_a", "seed_short_a", 68.0, 0.15, 0.10),
            pair("bad_excursion", "seed_long_a", 65.0, 0.25, 0.65),
            pair("bad_excursion", "seed_short_a", 31.0, 0.18, 0.70),
        ]
    )
    seed = SimpleNamespace(
        laps=pd.DataFrame(lap_rows),
        route_variant_summary=pd.DataFrame(summary_rows),
        route_variant_pairwise=pairwise,
    )
    detector_settings = SimpleNamespace(
        same_variant_p95_distance_m=10.0,
        maximum_divergent_fraction=0.03,
        maximum_length_relative_difference=0.08,
        minimum_supported_laps=2,
    )
    merged = _merge_geometric_clusters_into_topology_families(
        seed=seed,
        original_settings=detector_settings,
        family_settings=RouteFamilySettings(),
    )
    supported = merged.summary[merged.summary["supported"]]
    assert list(supported["route_variant_id"]) == ["route_001", "route_002"]
    assert list(supported["lap_count"]) == [12, 17]
    assert set(
        supported.iloc[0]["geometric_seed_route_variant_ids"].split(";")
    ) == {"line_long_extra", "seed_long_a", "seed_long_b"}
    assert set(
        supported.iloc[1]["geometric_seed_route_variant_ids"].split(";")
    ) == {"line_short_extra", "seed_short_a", "seed_short_b"}
    outlier = merged.summary[~merged.summary["supported"]].iloc[0]
    assert outlier["lap_count"] == 1
    assert outlier["geometric_seed_route_variant_ids"] == "bad_excursion"



def _polyline_centreline(vertices: list[tuple[float, float]], spacing: float = 5.0) -> Centreline:
    pts = np.asarray(vertices, dtype=float)
    pieces = []
    for left, right in zip(pts[:-1], pts[1:]):
        length = float(np.hypot(*(right - left)))
        count = max(2, int(np.ceil(length / spacing)) + 1)
        t = np.linspace(0.0, 1.0, count, endpoint=False)
        pieces.append(left[None, :] + t[:, None] * (right - left)[None, :])
    pieces.append(pts[-1:])
    xy = np.vstack(pieces)
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])) > 1e-9
    xy = xy[keep]
    ds = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    s = np.concatenate(([0.0], np.cumsum(ds)))
    return Centreline(
        x_m=xy[:, 0],
        y_m=xy[:, 1],
        s_m=s,
        elevation_m=np.zeros(len(xy)),
        frame=LocalFrame(0.0, 0.0),
    )


def test_branch_detector_finds_local_divergence_and_remerge_not_whole_track():
    # Both traversals share the entire rectangle except the bottom edge. The
    # nominal uses the short bottom edge; the alternate makes one large detour.
    nominal = _polyline_centreline(
        [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    )
    alternate = _polyline_centreline(
        [(0, 0), (50, -100), (100, 0), (100, 100), (0, 100), (0, 0)]
    )
    nominal_branch = _sustained_divergence_interval(
        nominal,
        alternate,
        divergence_threshold_m=25.0,
        minimum_length_m=30.0,
        gap_tolerance_m=10.0,
        padding_m=5.0,
    )
    alternate_branch = _sustained_divergence_interval(
        alternate,
        nominal,
        divergence_threshold_m=25.0,
        minimum_length_m=30.0,
        gap_tolerance_m=10.0,
        padding_m=5.0,
    )
    assert nominal_branch is not None
    assert alternate_branch is not None
    assert nominal_branch["length_m"] < nominal.length_m * 0.45
    assert alternate_branch["length_m"] < alternate.length_m * 0.60
    # A point on the opposite/top edge remains shared.
    assert not _s_in_circular_interval(
        200.0,
        nominal_branch["start_s_m"],
        nominal_branch["end_s_m"],
        nominal.length_m,
    )



def test_shared_contract_events_require_shared_section_on_every_route():
    sections = pd.DataFrame(
        [
            {"route_variant_id": "route_001", "event_id": "a", "route_section_kind": "shared"},
            {"route_variant_id": "route_002", "event_id": "a", "route_section_kind": "shared"},
            {"route_variant_id": "route_001", "event_id": "b", "route_section_kind": "branch"},
            {"route_variant_id": "route_002", "event_id": "b", "route_section_kind": "branch"},
            {"route_variant_id": "route_001", "event_id": "c", "route_section_kind": "shared"},
        ]
    )
    assert _shared_contract_event_ids(
        sections, route_ids=("route_001", "route_002")
    ) == {"a"}


def test_shared_contract_deduplicates_one_pass_per_lap_using_best_local_projection():
    base = {
        "event_id": "a",
        "event_name": "A",
        "lap_id": 7,
        "eligible": True,
        "eligible_before_route_compatibility": True,
        "route_compatible": True,
        "source_route_variant_id": "route_001",
        "route_section_kind": "shared",
        "evidence_scope": "shared_course",
    }
    route_a = pd.DataFrame(
        [{**base, "target_route_variant_id": "route_001", "route_projection_p95_error_m": 6.0, "entry_speed_mps": 4.0}]
    )
    route_b = pd.DataFrame(
        [{**base, "target_route_variant_id": "route_002", "route_projection_p95_error_m": 2.0, "entry_speed_mps": 4.1}]
    )
    pooled = _deduplicated_shared_contract_passes(
        [route_a, route_b], shared_event_ids={"a"}
    )
    assert len(pooled) == 1
    assert pooled.iloc[0]["projection_source_route_variant_id"] == "route_002"
    assert bool(pooled.iloc[0]["shared_evidence_contract"])
    assert bool(pooled.iloc[0]["eligible"])


def test_spacing_robustness_holds_physical_smoothing_width_near_nominal():
    case = SimpleNamespace(
        identifier="centreline_coarse_spacing",
        category="centreline",
        label="Coarser centreline spacing",
        rationale="test",
        overrides={"reconstruction.centreline_spacing_m": 4.5},
        excluded_run_ids=(),
    )
    updated = _spacing_decoupled_cases(
        [case],
        {
            "reconstruction": {"centreline_spacing_m": 3.0},
            "centreline_consensus": {"smoothing_window_nodes": 5},
        },
    )[0]
    assert updated.overrides["centreline_consensus.smoothing_window_nodes"] == 3
    assert abs(
        updated.overrides["centreline_consensus.smoothing_window_nodes"]
        * updated.overrides["reconstruction.centreline_spacing_m"]
        - 15.0
    ) <= 1.5
    assert "fixed smoothing width" in updated.label


def test_normal_robustness_detects_route_family_capability():
    resolution = SimpleNamespace(
        data={
            "track": {
                "route_variants": {
                    "enabled": True,
                    "family": {"enabled": True},
                }
            }
        }
    )
    assert route_family_robustness_enabled(resolution)
    resolution.data["track"]["route_variants"]["family"]["enabled"] = False
    assert not route_family_robustness_enabled(resolution)


def test_uncertainty_track_case_cap_remains_balanced_across_routes():
    def variant(case_id, category):
        return SimpleNamespace(case_id=case_id, category=category, label=case_id)

    variants = [variant("nominal", "nominal"), variant("route_001__nominal", "nominal")]
    groups = [
        ("centreline_a", "centreline"),
        ("centreline_b", "centreline"),
        ("cleanup_a", "telemetry_cleanup"),
        ("event_windows_a", "event_windows"),
        ("gate_a", "gate_weighting"),
        ("gate_b", "gate_weighting"),
    ]
    for group, category in groups:
        variants.extend(
            [
                variant(f"route_002__{group}", category),
                variant(f"route_001__{group}", category),
            ]
        )
    manifest = {
        "route_variant_ids": ["route_001", "route_002"],
        "nominal_route_variant_id": "route_002",
        "common_eligible_robustness_case_ids": [group for group, _ in groups],
    }
    selected = _balanced_family_variants(variants, manifest=manifest, maximum=8)
    route_counts = {"route_001": 0, "route_002": 0}
    for item in selected:
        if item.case_id == "nominal":
            route_counts["route_002"] += 1
        else:
            route_counts[item.case_id.split("__", 1)[0]] += 1
    assert route_counts == {"route_001": 4, "route_002": 4}
    selected_ids = {item.case_id for item in selected}
    assert "route_001__nominal" in selected_ids
    # Physical reconstruction categories are selected before policy-only variants.
    assert "route_001__centreline_a" in selected_ids
    assert "route_001__cleanup_a" in selected_ids
