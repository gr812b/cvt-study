"""Apply the reviewed endurance-traffic model to the current drivetrain-study tree.

The repository has evolved through route-family and performance overlays, so this
installer changes only narrow, verified source blocks and refuses ambiguous shapes.
New traffic/calibration modules are already present in the overlay; this script wires
them into the existing uncertainty and simulation pipeline without replacing whole
older source files.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PatchError(RuntimeError):
    pass


def read(rel: str) -> tuple[Path, str]:
    path = ROOT / rel
    if not path.is_file():
        raise PatchError(f"Required file not found: {rel}")
    return path, path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise PatchError(f"{label}: expected one regex match, found {count}")
    return updated


def patch_uncertainty_model() -> None:
    rel = "src/cvt_track_study/uncertainty/model.py"
    path, text = read(rel)
    if "traffic_realization" in text:
        print(f"already patched: {rel}")
        return
    text = replace_once(
        text,
        '    sampling_design: str = "latin_hypercube_rank_correlated"\n',
        '    sampling_design: str = "latin_hypercube_rank_correlated"\n'
        '    traffic_realization: Mapping[str, object] | None = None\n',
        "ScenarioDraw traffic field",
    )
    text = replace_once(
        text,
        '            "independently_sampled_gate_ids": list(\n'
        '                self.independently_sampled_gate_ids\n'
        '            ),\n'
        '        }\n',
        '            "independently_sampled_gate_ids": list(\n'
        '                self.independently_sampled_gate_ids\n'
        '            ),\n'
        '            "traffic_realization": (\n'
        '                None if self.traffic_realization is None else dict(self.traffic_realization)\n'
        '            ),\n'
        '        }\n',
        "ScenarioDraw serialization",
    )
    write(path, text)
    print(f"patched: {rel}")


def patch_dynamics() -> None:
    rel = "src/cvt_track_study/simulation/dynamics.py"
    path, text = read(rel)
    if "external_speed_ceiling_mps" in text:
        print(f"already patched: {rel}")
        return
    text = replace_once(
        text,
        '    braking_deceleration_mps2: float,\n) -> DriverCommand:\n',
        '    braking_deceleration_mps2: float,\n'
        '    external_speed_ceiling_mps: float | None = None,\n'
        ') -> DriverCommand:\n',
        "driver command signature",
    )
    text = replace_once(
        text,
        '    target = track.safe_speed_ceiling_mps(\n'
        '        distance_m,\n'
        '        braking_deceleration_mps2=braking_deceleration_mps2,\n'
        '    )\n',
        '    target = track.safe_speed_ceiling_mps(\n'
        '        distance_m,\n'
        '        braking_deceleration_mps2=braking_deceleration_mps2,\n'
        '    )\n'
        '    if external_speed_ceiling_mps is not None and isfinite(float(external_speed_ceiling_mps)):\n'
        '        target = min(target, max(0.0, float(external_speed_ceiling_mps)))\n',
        "external driver target",
    )
    text = replace_once(
        text,
        '    feature_entry_speeds_mps: Mapping[str, float] | None = None,\n) -> DynamicsSample:\n',
        '    feature_entry_speeds_mps: Mapping[str, float] | None = None,\n'
        '    external_speed_ceiling_mps: float | None = None,\n'
        ') -> DynamicsSample:\n',
        "evaluate dynamics signature",
    )
    text = replace_once(
        text,
        '        braking_deceleration_mps2=max(effective_braking_deceleration, 1.0e-9),\n'
        '    )\n',
        '        braking_deceleration_mps2=max(effective_braking_deceleration, 1.0e-9),\n'
        '        external_speed_ceiling_mps=external_speed_ceiling_mps,\n'
        '    )\n',
        "external driver target pass-through",
    )
    write(path, text)
    print(f"patched: {rel}")


def patch_integrator() -> None:
    rel = "src/cvt_track_study/simulation/integrator.py"
    path, text = read(rel)
    if "traffic_reference" in text and "traffic_limit_state" in text:
        print(f"already patched: {rel}")
        return

    anchor = "from .track import RuntimeTrack\n"
    traffic_import = (
        "from .traffic import (\n"
        "    TrafficRealization,\n"
        "    TrafficReferenceProfile,\n"
        "    traffic_limit_state,\n"
        ")\n"
    )
    text = replace_once(text, anchor, anchor + traffic_import, "integrator traffic imports")

    text = replace_once(
        text,
        '    feature_obstacle_energy_j: Mapping[str, float]\n',
        '    feature_obstacle_energy_j: Mapping[str, float]\n'
        '    traffic_realization: Mapping[str, Any] | None = None\n',
        "SimulationTrace traffic payload",
    )

    # Signature has existed both on one line and multiple lines; normalize with regex.
    text = regex_once(
        text,
        r"def run_simulation\(\n\s*\*,\s*case: StudyCase,\s*track: RuntimeTrack,\s*settings: SimulationSettings\s*\n?\) -> SimulationTrace:",
        "def run_simulation(\n"
        "    *,\n"
        "    case: StudyCase,\n"
        "    track: RuntimeTrack,\n"
        "    settings: SimulationSettings,\n"
        "    traffic: TrafficRealization | None = None,\n"
        "    traffic_reference: TrafficReferenceProfile | None = None,\n"
        ") -> SimulationTrace:",
        "run_simulation signature",
    )

    text = replace_once(
        text,
        '        "grade_work_j": 0.0,\n'
        '    }\n',
        '        "grade_work_j": 0.0,\n'
        '        "traffic_active_time_s": 0.0,\n'
        '    }\n',
        "traffic active integral",
    )

    eval_anchor = '        dynamics = evaluate_dynamics(\n'
    traffic_state = (
        '        traffic_ceiling, _traffic_retained = traffic_limit_state(\n'
        '            traffic=traffic,\n'
        '            reference=traffic_reference,\n'
        '            lap_time_s=time_s,\n'
        '            distance_m=distance,\n'
        '        )\n'
    )
    text = replace_once(text, eval_anchor, traffic_state + eval_anchor, "integration traffic state")
    text = replace_once(
        text,
        '            feature_entry_speeds_mps=feature_entry_speeds,\n'
        '        )\n',
        '            feature_entry_speeds_mps=feature_entry_speeds,\n'
        '            external_speed_ceiling_mps=(\n'
        '                None if not np.isfinite(traffic_ceiling) else traffic_ceiling\n'
        '            ),\n'
        '        )\n',
        "integration external traffic ceiling",
    )
    text = replace_once(
        text,
        '        actual_step = new_time - time_s\n'
        '        average_speed = 0.5 * (speed + new_speed)\n',
        '        actual_step = new_time - time_s\n'
        '        if np.isfinite(traffic_ceiling):\n'
        '            integrals["traffic_active_time_s"] += actual_step\n'
        '        average_speed = 0.5 * (speed + new_speed)\n',
        "traffic active integration",
    )

    text = replace_once(
        text,
        '        feature_obstacle_energy_j=feature_obstacle_energy,\n'
        '    )\n',
        '        feature_obstacle_energy_j=feature_obstacle_energy,\n'
        '        traffic=traffic,\n'
        '        traffic_reference=traffic_reference,\n'
        '    )\n',
        "report traffic args",
    )
    text = replace_once(
        text,
        '    feature_obstacle_energy_j: Mapping[str, float],\n'
        ') -> SimulationTrace:\n',
        '    feature_obstacle_energy_j: Mapping[str, float],\n'
        '    traffic: TrafficRealization | None,\n'
        '    traffic_reference: TrafficReferenceProfile | None,\n'
        ') -> SimulationTrace:\n',
        "report signature traffic",
    )
    text = replace_once(
        text,
        '        "wheel_kinetic_energy_j", "total_kinetic_energy_j",\n'
        '    )\n',
        '        "wheel_kinetic_energy_j", "total_kinetic_energy_j",\n'
        '        "traffic_speed_ceiling_mps", "traffic_retained_fraction", "race_time_s",\n'
        '    )\n',
        "report traffic keys",
    )
    report_eval = '    for t, s, v, omega in zip(times, distances, speeds, wheel_speeds):\n        d = evaluate_dynamics(\n'
    report_state = (
        '    for t, s, v, omega in zip(times, distances, speeds, wheel_speeds):\n'
        '        traffic_ceiling, traffic_retained = traffic_limit_state(\n'
        '            traffic=traffic,\n'
        '            reference=traffic_reference,\n'
        '            lap_time_s=float(t),\n'
        '            distance_m=float(s),\n'
        '        )\n'
        '        d = evaluate_dynamics(\n'
    )
    text = replace_once(text, report_eval, report_state, "report traffic state")
    # This is the second feature-entry argument occurrence, because the first was patched above.
    text = replace_once(
        text,
        '            feature_entry_speeds_mps=feature_entry_speeds_mps,\n'
        '        )\n',
        '            feature_entry_speeds_mps=feature_entry_speeds_mps,\n'
        '            external_speed_ceiling_mps=(\n'
        '                None if not np.isfinite(traffic_ceiling) else traffic_ceiling\n'
        '            ),\n'
        '        )\n',
        "report external traffic ceiling",
    )
    text = replace_once(
        text,
        '            "total_kinetic_energy_j": vehicle_ke + wheel_ke,\n'
        '        }\n',
        '            "total_kinetic_energy_j": vehicle_ke + wheel_ke,\n'
        '            "traffic_speed_ceiling_mps": (\n'
        '                np.nan if not np.isfinite(traffic_ceiling) else traffic_ceiling\n'
        '            ),\n'
        '            "traffic_retained_fraction": traffic_retained,\n'
        '            "race_time_s": (\n'
        '                float(t) if traffic is None else traffic.lap_start_race_time_s + float(t)\n'
        '            ),\n'
        '        }\n',
        "report traffic values",
    )
    text = replace_once(
        text,
        '        feature_obstacle_energy_j={\n'
        '            key: float(value) for key, value in feature_obstacle_energy_j.items()\n'
        '        },\n'
        '    )\n',
        '        feature_obstacle_energy_j={\n'
        '            key: float(value) for key, value in feature_obstacle_energy_j.items()\n'
        '        },\n'
        '        traffic_realization=(None if traffic is None else traffic.serializable()),\n'
        '    )\n',
        "SimulationTrace traffic serialization",
    )
    write(path, text)
    print(f"patched: {rel}")


def patch_metrics() -> None:
    rel = "src/cvt_track_study/simulation/metrics.py"
    path, text = read(rel)
    if '"traffic_event_count"' in text:
        print(f"already patched: {rel}")
        return
    text = replace_once(
        text,
        '    summary: dict[str, Any] = {\n'
        '        "case": trace.case_name,\n',
        '    traffic_payload = trace.traffic_realization or {}\n'
        '    traffic_events = traffic_payload.get("events", ()) if isinstance(traffic_payload, dict) else ()\n'
        '    traffic_min_retained = min(\n'
        '        (float(item.get("retained_speed_fraction", 1.0)) for item in traffic_events if isinstance(item, dict)),\n'
        '        default=1.0,\n'
        '    )\n'
        '    summary: dict[str, Any] = {\n'
        '        "case": trace.case_name,\n',
        "traffic summary setup",
    )
    text = replace_once(
        text,
        '        "time_traction_limited_s": _time(n["tire_utilization"] >= 0.95, t),\n',
        '        "time_traction_limited_s": _time(n["tire_utilization"] >= 0.95, t),\n'
        '        "traffic_active_time_s": float(integrated.get("traffic_active_time_s", 0.0)),\n'
        '        "traffic_event_count": float(len(traffic_events)),\n'
        '        "traffic_lap_start_race_time_s": float(traffic_payload.get("lap_start_race_time_s", -1.0)) if isinstance(traffic_payload, dict) else -1.0,\n'
        '        "traffic_minimum_retained_fraction": float(traffic_min_retained),\n',
        "traffic summary fields",
    )
    write(path, text)
    print(f"patched: {rel}")


def patch_service_v8() -> None:
    rel = "src/cvt_track_study/studies/service_v8.py"
    path, text = read(rel)
    if "traffic_model_from_project" in text and "reference_no_traffic_lap_time_s" in text:
        print(f"already patched: {rel}")
        return

    # replace/extend dataclass import without disturbing performance overlay
    text = re.sub(
        r"from dataclasses import asdict(?:, replace)?",
        "from dataclasses import asdict, replace",
        text,
        count=1,
    )
    traffic_import = (
        "from cvt_track_study.simulation.traffic import (\n"
        "    TrafficReferenceProfile,\n"
        "    traffic_model_from_project,\n"
        "    traffic_realization_from_mapping,\n"
        ")\n"
    )
    anchor = "from cvt_track_study.simulation.service import SimulationError, resolve_simulation_cases\n"
    text = replace_once(text, anchor, anchor + traffic_import, "service traffic import")

    registry_block = '''    registry = build_input_registry(
        vehicle_raw=vehicle_raw,
        base_study_raw=base_study,
        track_raw=resolution.data["track"],
        bundle=bundle,
    )
'''
    text = replace_once(
        text,
        registry_block,
        registry_block
        + '    traffic_model = (\n'
          '        traffic_model_from_project(resolution.paths.root)\n'
          '        if study_type in {"full_uncertainty", "design_sweep"}\n'
          '        else None\n'
          '    )\n',
        "service traffic model load",
    )
    text = replace_once(
        text,
        '            "replicates_override": replicates_override,\n'
        '        }\n',
        '            "replicates_override": replicates_override,\n'
        '            "traffic_model": (None if traffic_model is None else traffic_model.contract()),\n'
        '        }\n',
        "service traffic fingerprint",
    )
    text = replace_once(
        text,
        '        study_fingerprint=fingerprint,\n'
        '        evidence_assessment=assess_evidence(\n',
        '        study_fingerprint=fingerprint,\n'
        '        traffic_model=traffic_model,\n'
        '        evidence_assessment=assess_evidence(\n',
        "service execute traffic arg",
    )
    text = replace_once(
        text,
        '    study_fingerprint: str,\n'
        '    evidence_assessment: Mapping[str, Any],\n',
        '    study_fingerprint: str,\n'
        '    traffic_model: Any | None,\n'
        '    evidence_assessment: Mapping[str, Any],\n',
        "service execute signature traffic",
    )
    scenario_block = '''    scenarios = (
        sampler.draw_all()
        if study_type != "structural_sensitivity"
        else (ScenarioDraw(0, seed, "nominal"),)
    )
'''
    text = replace_once(
        text,
        scenario_block,
        scenario_block
        + '    if traffic_model is not None:\n'
          '        horizon_s = float(base_study.get("simulation", {}).get("maximum_time_s", 300.0))\n'
          '        scenarios = tuple(\n'
          '            replace(\n'
          '                scenario,\n'
          '                traffic_realization=traffic_model.draw_realization(\n'
          '                    scenario_seed=scenario.seed, horizon_s=horizon_s\n'
          '                ).serializable(),\n'
          '            )\n'
          '            for scenario in scenarios\n'
          '        )\n',
        "service traffic scenario worlds",
    )
    text = replace_once(
        text,
        '        "paired_scenarios": True,\n',
        '        "paired_scenarios": True,\n'
        '        "traffic_model": ({"enabled": False} if traffic_model is None else {"enabled": True, **traffic_model.contract()}),\n',
        "service traffic manifest",
    )

    # Replace the complete scenario executor. Performance-overlay worker helpers call
    # this function, so they automatically inherit traffic without another patch.
    pattern = r"def _execute_scenario\(\n.*?\n\n_SUMMARY_FIELDS = \("
    replacement = '''def _execute_scenario(
    *,
    scenario: ScenarioDraw,
    design_points: tuple[DesignPoint, ...],
    study_type: str,
    vehicle_id: str,
    vehicle_raw: Mapping[str, Any],
    base_study: Mapping[str, Any],
    track_raw: Mapping[str, Any],
    bundle: TrackBundle,
    cache: SimulationCache,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    # With traffic active, the traffic-free infinite-CVT trace is required to
    # define the shared absolute traffic speed profile. The traffic-aware infinite
    # reference is then paired with every bounded/design case. We intentionally do
    # not run an additional traffic-free bounded case for every scenario: it is not
    # required by the comparison and would add avoidable study runtime.
    references: dict[
        tuple[int, str],
        tuple[dict[str, Any], dict[str, Any], str, TrafficReferenceProfile],
    ] = {}
    bounded_runs = reference_runs = reference_reuses = persistent_hits = 0
    traffic = traffic_realization_from_mapping(scenario.traffic_realization)

    for design in design_points:
        design_values = (
            {design.path: float(design.value_si)}
            if design.path is not None and design.value_si is not None
            else {}
        )
        choices = dict(scenario.choice_values)
        if design.path is not None and design.choice_value is not None:
            choices[design.path] = design.choice_value
        try:
            bounded_case, reference_case, settings, runtime_track = resolve_simulation_cases(
                vehicle_id=vehicle_id,
                vehicle_raw=vehicle_raw,
                study_raw=base_study,
                track_raw=track_raw,
                bundle=bundle,
                quantity_values_si=scenario.quantity_values_si,
                choice_values=choices,
                gate_target_speeds_mps=scenario.gate_target_speeds_mps,
                design_values_si=design_values,
                shared_reference=study_type == "design_sweep",
            )
        except Exception as exc:
            raise SimulationError(
                f"Scenario {scenario.replicate}, design {design.identifier!r} could not form a valid physical case: {exc}"
            ) from exc

        key = reference_cache_key(
            scenario.replicate,
            design,
            share_across_designs=study_type == "design_sweep",
        )
        if key in references:
            reference_free_record, reference_record, reference_fingerprint, traffic_reference = references[key]
            reference_reuses += 1
        else:
            reference_free_record, cached = _run_case_summary_cached(
                reference_case, settings, runtime_track, cache
            )
            reference_runs += int(not cached)
            persistent_hits += int(cached)
            traffic_reference = TrafficReferenceProfile.from_mapping(
                reference_free_record["traffic_reference_profile"]
            )
            if traffic is None:
                reference_record = reference_free_record
            else:
                reference_record, cached = _run_case_summary_cached(
                    reference_case,
                    settings,
                    runtime_track,
                    cache,
                    traffic=traffic,
                    traffic_reference=traffic_reference,
                )
                reference_runs += int(not cached)
                persistent_hits += int(cached)
            reference_fingerprint = phase6_service._reference_fingerprint(
                scenario, design, reference_case, runtime_track
            )
            references[key] = (
                reference_free_record,
                reference_record,
                reference_fingerprint,
                traffic_reference,
            )

        bounded_record, cached = _run_case_summary_cached(
            bounded_case,
            settings,
            runtime_track,
            cache,
            traffic=traffic,
            traffic_reference=(traffic_reference if traffic is not None else None),
        )
        bounded_runs += int(not cached)
        persistent_hits += int(cached)

        bounded_summary = bounded_record["summary"]
        reference_summary = reference_record["summary"]
        reference_free_summary = reference_free_record["summary"]
        comparison = compare_summaries(bounded_summary, reference_summary)
        row: dict[str, Any] = {
            "replicate": scenario.replicate,
            "scenario_seed": scenario.seed,
            "design_id": design.identifier,
            "design_path": design.path or "nominal",
            "design_value": design.display_value,
            "design_value_si": design.value_si,
            "design_choice_value": design.choice_value,
            "level_probability": design.level_probability,
            "level_kind": design.level_kind,
            "parameter_path": design.path if study_type == "structural_sensitivity" else None,
            "reference_fingerprint": reference_fingerprint,
            "bounded_completed": bool(bounded_summary["completed"]),
            "reference_completed": bool(reference_summary["completed"]),
            "reference_dominance_pass": bool(comparison["reference_dominance_pass"]),
            "bounded_energy_balance_relative_error": float(comparison["bounded_energy_balance_relative_error"]),
            "reference_energy_balance_relative_error": float(comparison["reference_energy_balance_relative_error"]),
            "bounded_powertrain_energy_balance_relative_error": float(
                bounded_summary["powertrain_energy_balance_relative_error"]
            ),
            "reference_powertrain_energy_balance_relative_error": float(
                reference_summary["powertrain_energy_balance_relative_error"]
            ),
            "bounded_max_gate_excess_kmh": bounded_record["maximum_gate_excess_kmh"],
            "reference_max_gate_excess_kmh": reference_record["maximum_gate_excess_kmh"],
            "bounded_gates_compliant_0p5_kmh": bounded_record["gates_compliant_0p5_kmh"],
            "reference_gates_compliant_0p5_kmh": reference_record["gates_compliant_0p5_kmh"],
            "reference_no_traffic_lap_time_s": float(reference_free_summary["lap_time_s"]),
            "reference_traffic_penalty_s": float(
                reference_summary["lap_time_s"] - reference_free_summary["lap_time_s"]
            ),
        }
        row.update({metric: float(comparison[metric]) for metric in METRICS})
        _add_summary_fields(row, "bounded", bounded_summary)
        _add_summary_fields(row, "reference", reference_summary)
        rows.append(row)
    return {
        "rows": rows,
        "bounded_case_count": len(design_points),
        "reference_case_count": (
            1 if study_type == "design_sweep" and design_points else len(design_points)
        ),
        "bounded_simulation_count": bounded_runs,
        "reference_simulation_count": reference_runs,
        "reference_cache_hits": reference_reuses,
        "simulation_cache_hits": persistent_hits,
        "resumed": False,
    }


_SUMMARY_FIELDS = ('''
    text = regex_once(text, pattern, replacement, "traffic-aware scenario executor")

    text = replace_once(
        text,
        '    "time_minimum_ratio_s",\n'
        ')\n',
        '    "time_minimum_ratio_s",\n'
        '    "traffic_active_time_s",\n'
        '    "traffic_event_count",\n'
        '    "traffic_lap_start_race_time_s",\n'
        '    "traffic_minimum_retained_fraction",\n'
        ')\n',
        "traffic summary export fields",
    )

    # Cache now stores a compact traffic-free reference profile and keys traffic
    # runs separately from ordinary runs.
    cache_pattern = r"def _run_case_summary_cached\(\n.*?\n\ndef _track_cache_contract"
    cache_replacement = '''def _run_case_summary_cached(
    case: Any,
    settings: Any,
    track: Any,
    cache: SimulationCache,
    *,
    traffic: Any | None = None,
    traffic_reference: TrafficReferenceProfile | None = None,
) -> tuple[dict[str, Any], bool]:
    traffic_contract = None if traffic is None else traffic.serializable()
    reference_fingerprint = (
        None
        if traffic_reference is None
        else canonical_fingerprint(traffic_reference.serializable())
    )
    key = SimulationCache.key(
        {
            "schema": "simulation-summary-v2-traffic",
            "case": asdict(case),
            "settings": asdict(settings),
            "track": _track_cache_contract(track),
            "traffic": traffic_contract,
            "traffic_reference_fingerprint": reference_fingerprint,
        }
    )
    cached = cache.get(key)
    if cached is not None:
        return cached, True
    trace = run_simulation(
        case=case,
        track=track,
        settings=settings,
        traffic=traffic,
        traffic_reference=traffic_reference,
    )
    summary = summarize_trace(
        trace,
        target_engine_rpm=case.engine.target_rpm,
        target_power_w=case.engine.target_power_w,
    )
    gate_rows = gate_compliance_rows(trace, track)
    record = {
        "summary": summary,
        "maximum_gate_excess_kmh": max(
            (float(item["excess_over_ceiling_kmh"]) for item in gate_rows), default=0.0
        ),
        "gates_compliant_0p5_kmh": all(
            bool(item["compliant_within_0p5_kmh"]) for item in gate_rows
        ),
    }
    if traffic is None:
        record["traffic_reference_profile"] = TrafficReferenceProfile.from_trace(
            trace, spacing_m=1.0
        ).serializable()
    cache.put(key, record)
    return record, False


def _track_cache_contract'''
    text = regex_once(text, cache_pattern, cache_replacement, "traffic-aware simulation cache")

    write(path, text)
    print(f"patched: {rel}")


def patch_ensemble_v10() -> None:
    rel = "src/cvt_track_study/studies/ensemble_v10.py"
    path, text = read(rel)
    if "traffic_by_draw" in text and "traffic_model_from_project" in text:
        print(f"already patched: {rel}")
        return
    text = re.sub(
        r"from dataclasses import dataclass(?:, replace)?",
        "from dataclasses import dataclass, replace",
        text,
        count=1,
    )
    anchor = "from cvt_track_study.simulation.service import SimulationError\n"
    text = replace_once(
        text,
        anchor,
        anchor + "from cvt_track_study.simulation.traffic import traffic_model_from_project\n",
        "ensemble traffic import",
    )
    registry_block = '''    registry = build_input_registry(
        vehicle_raw=vehicle_raw,
        base_study_raw=base_study,
        track_raw=resolution.data["track"],
        bundle=nominal_bundle,
    )
'''
    text = replace_once(
        text,
        registry_block,
        registry_block + '    traffic_model = traffic_model_from_project(resolution.paths.root)\n',
        "ensemble traffic model load",
    )
    schedule_block = '''    selected, schedule_metadata = _schedule_scenarios(
        variants=variants,
        samplers=samplers,
        plan=plan,
        replicates=replicates,
        sampling_layout=sampling_layout,
    )
'''
    text = replace_once(
        text,
        schedule_block,
        schedule_block
        + '    if traffic_model is not None:\n'
          '        horizon_s = float(base_study.get("simulation", {}).get("maximum_time_s", 300.0))\n'
          '        traffic_by_draw: dict[int, Mapping[str, object]] = {}\n'
          '        enriched: list[ScheduledScenario] = []\n'
          '        for item in selected:\n'
          '            payload = traffic_by_draw.get(item.base_draw_id)\n'
          '            if payload is None:\n'
          '                payload = traffic_model.draw_realization(\n'
          '                    scenario_seed=item.scenario.seed, horizon_s=horizon_s\n'
          '                ).serializable()\n'
          '                traffic_by_draw[item.base_draw_id] = payload\n'
          '            enriched.append(\n'
          '                ScheduledScenario(\n'
          '                    variant=item.variant,\n'
          '                    scenario=replace(item.scenario, traffic_realization=payload),\n'
          '                    base_draw_id=item.base_draw_id,\n'
          '                )\n'
          '            )\n'
          '        selected = tuple(enriched)\n',
        "ensemble paired traffic worlds",
    )
    text = replace_once(
        text,
        '            "base_draw_count": schedule_metadata["base_draw_count"],\n'
        '        }\n',
        '            "base_draw_count": schedule_metadata["base_draw_count"],\n'
        '            "traffic_model": (None if traffic_model is None else traffic_model.contract()),\n'
        '        }\n',
        "ensemble traffic fingerprint",
    )
    text = replace_once(
        text,
        '        "paired_scenarios": True,\n',
        '        "paired_scenarios": True,\n'
        '        "traffic_model": ({"enabled": False} if traffic_model is None else {"enabled": True, **traffic_model.contract()}),\n',
        "ensemble traffic manifest",
    )
    write(path, text)
    print(f"patched: {rel}")


def patch_design_grid_execution() -> None:
    """Make multi-variable design replay use the same paired traffic world."""
    rel = "src/cvt_track_study/studies/design_grid_execution.py"
    path, text = read(rel)
    if "traffic_realization_from_mapping" in text and "reference_no_traffic_lap_time_s" in text:
        print(f"already patched: {rel}")
        return
    anchor = "from cvt_track_study.simulation.service import SimulationError, resolve_simulation_cases\n"
    traffic_import = (
        "from cvt_track_study.simulation.traffic import (\n"
        "    TrafficReferenceProfile,\n"
        "    traffic_realization_from_mapping,\n"
        ")\n"
    )
    text = replace_once(text, anchor, anchor + traffic_import, "design-grid traffic imports")
    pattern = r"def execute_design_grid_scenario\(\n.*\Z"
    replacement = r'''def execute_design_grid_scenario(
    *,
    scenario: ScenarioDraw,
    design_points: Sequence[Any],
    study_type: str,
    vehicle_id: str,
    vehicle_raw: Mapping[str, Any],
    base_study: Mapping[str, Any],
    track_raw: Mapping[str, Any],
    bundle: TrackBundle,
    cache: SimulationCache,
) -> dict[str, Any]:
    """Run one paired uncertainty world across every design-grid candidate."""

    points = tuple(design_points)
    rows: list[dict[str, Any]] = []
    references: dict[
        tuple[int, str],
        tuple[dict[str, Any], dict[str, Any], str, TrafficReferenceProfile],
    ] = {}
    bounded_runs = reference_runs = reference_reuses = persistent_hits = 0
    traffic = traffic_realization_from_mapping(scenario.traffic_realization)

    all_design_paths = design_paths(points)
    share_reference = (
        study_type == "design_sweep"
        and reference_can_be_shared(all_design_paths)
    )

    for design in points:
        design_values = design_quantity_values_si(design)
        choices = dict(scenario.choice_values)
        choices.update(design_choice_values(design))
        try:
            bounded_case, reference_case, settings, runtime_track = resolve_simulation_cases(
                vehicle_id=vehicle_id,
                vehicle_raw=vehicle_raw,
                study_raw=base_study,
                track_raw=track_raw,
                bundle=bundle,
                quantity_values_si=scenario.quantity_values_si,
                choice_values=choices,
                gate_target_speeds_mps=scenario.gate_target_speeds_mps,
                design_values_si=design_values,
                shared_reference=share_reference,
            )
        except Exception as exc:
            raise SimulationError(
                f"Scenario {scenario.replicate}, design {design.identifier!r} "
                f"could not form a valid physical case: {exc}"
            ) from exc

        key = reference_cache_key(
            scenario.replicate,
            design,
            share_across_designs=share_reference,
        )
        if key in references:
            reference_free_record, reference_record, reference_fingerprint, traffic_reference = references[key]
            reference_reuses += 1
        else:
            reference_free_record, cached = service_v8._run_case_summary_cached(
                reference_case, settings, runtime_track, cache
            )
            reference_runs += int(not cached)
            persistent_hits += int(cached)
            traffic_reference = TrafficReferenceProfile.from_mapping(
                reference_free_record["traffic_reference_profile"]
            )
            if traffic is None:
                reference_record = reference_free_record
            else:
                reference_record, cached = service_v8._run_case_summary_cached(
                    reference_case,
                    settings,
                    runtime_track,
                    cache,
                    traffic=traffic,
                    traffic_reference=traffic_reference,
                )
                reference_runs += int(not cached)
                persistent_hits += int(cached)
            reference_fingerprint = phase6_service._reference_fingerprint(
                scenario, design, reference_case, runtime_track
            )
            references[key] = (
                reference_free_record,
                reference_record,
                reference_fingerprint,
                traffic_reference,
            )

        bounded_record, cached = service_v8._run_case_summary_cached(
            bounded_case,
            settings,
            runtime_track,
            cache,
            traffic=traffic,
            traffic_reference=(traffic_reference if traffic is not None else None),
        )
        bounded_runs += int(not cached)
        persistent_hits += int(cached)

        bounded_summary = bounded_record["summary"]
        reference_summary = reference_record["summary"]
        reference_free_summary = reference_free_record["summary"]
        comparison = compare_summaries(bounded_summary, reference_summary)
        display_values = design_display_values(design)
        quantity_values_si = design_quantity_values_si(design)
        point_paths = tuple(display_values)
        design_path_text = (
            str(getattr(design, "path", None))
            if getattr(design, "path", None)
            else " + ".join(point_paths) if point_paths else "nominal"
        )

        row: dict[str, Any] = {
            "replicate": scenario.replicate,
            "scenario_seed": scenario.seed,
            "design_id": design.identifier,
            "design_path": design_path_text,
            "design_value": design.display_value,
            "design_value_si": design.value_si,
            "design_choice_value": getattr(design, "choice_value", None),
            "design_dimension_count": len(point_paths),
            "design_paths_json": json.dumps(
                list(point_paths), separators=(",", ":"), allow_nan=False
            ),
            "design_values_json": json.dumps(
                display_values,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "design_values_si_json": json.dumps(
                quantity_values_si,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "level_probability": getattr(design, "level_probability", None),
            "level_kind": getattr(design, "level_kind", "design_grid"),
            "parameter_path": (
                getattr(design, "path", None)
                if study_type == "structural_sensitivity"
                else None
            ),
            "reference_fingerprint": reference_fingerprint,
            "bounded_completed": bool(bounded_summary["completed"]),
            "reference_completed": bool(reference_summary["completed"]),
            "reference_dominance_pass": bool(comparison["reference_dominance_pass"]),
            "bounded_energy_balance_relative_error": float(
                comparison["bounded_energy_balance_relative_error"]
            ),
            "reference_energy_balance_relative_error": float(
                comparison["reference_energy_balance_relative_error"]
            ),
            "bounded_powertrain_energy_balance_relative_error": float(
                bounded_summary["powertrain_energy_balance_relative_error"]
            ),
            "reference_powertrain_energy_balance_relative_error": float(
                reference_summary["powertrain_energy_balance_relative_error"]
            ),
            "bounded_max_gate_excess_kmh": bounded_record[
                "maximum_gate_excess_kmh"
            ],
            "reference_max_gate_excess_kmh": reference_record[
                "maximum_gate_excess_kmh"
            ],
            "bounded_gates_compliant_0p5_kmh": bounded_record[
                "gates_compliant_0p5_kmh"
            ],
            "reference_gates_compliant_0p5_kmh": reference_record[
                "gates_compliant_0p5_kmh"
            ],
            "reference_no_traffic_lap_time_s": float(reference_free_summary["lap_time_s"]),
            "reference_traffic_penalty_s": float(
                reference_summary["lap_time_s"] - reference_free_summary["lap_time_s"]
            ),
        }
        for index, (axis_path, value) in enumerate(display_values.items()):
            row[f"design_axis_{index}_path"] = axis_path
            row[f"design_axis_{index}_value"] = value
            row[f"design::{axis_path}"] = value
        row.update({metric: float(comparison[metric]) for metric in METRICS})
        service_v8._add_summary_fields(row, "bounded", bounded_summary)
        service_v8._add_summary_fields(row, "reference", reference_summary)
        rows.append(row)

    return {
        "rows": rows,
        "bounded_case_count": len(points),
        "reference_case_count": 1 if share_reference and points else len(points),
        "bounded_simulation_count": bounded_runs,
        "reference_simulation_count": reference_runs,
        "reference_cache_hits": reference_reuses,
        "simulation_cache_hits": persistent_hits,
        "resumed": False,
    }
'''
    text = regex_once(text, pattern, replacement, "traffic-aware design-grid executor")
    write(path, text)
    print(f"patched: {rel}")


def patch_design_replay_v11() -> None:
    """Replay the exact traffic world saved by the source uncertainty result."""
    rel = "src/cvt_track_study/studies/design_replay_v11.py"
    path, text = read(rel)
    if "exact_source_traffic_world_replay" in text and "traffic_realization=traffic_payload" in text:
        print(f"already patched: {rel}")
        return

    scenario_anchor = '''    gate_values = {
        str(path): float(value)
        for path, value in dict(record.get("gate_target_speeds_mps", {})).items()
    }
    return ScenarioDraw(
'''
    scenario_insert = '''    gate_values = {
        str(path): float(value)
        for path, value in dict(record.get("gate_target_speeds_mps", {})).items()
    }
    traffic_raw = record.get("traffic_realization")
    traffic_payload = dict(traffic_raw) if isinstance(traffic_raw, Mapping) else None
    return ScenarioDraw(
'''
    text = replace_once(text, scenario_anchor, scenario_insert, "design replay traffic payload setup")
    text = replace_once(
        text,
        '        sampling_design="replayed_from_completed_full_uncertainty_result",\n'
        '    )\n',
        '        sampling_design="replayed_from_completed_full_uncertainty_result",\n'
        '        traffic_realization=traffic_payload,\n'
        '    )\n',
        "design replay traffic field",
    )
    text = replace_once(
        text,
        '        "paired_scenarios": True,\n',
        '        "paired_scenarios": True,\n'
        '        "traffic_model": source.manifest.get("traffic_model", {"enabled": False}),\n'
        '        "traffic_replay_policy": "exact_source_traffic_world_replay",\n',
        "design replay traffic manifest",
    )
    write(path, text)
    print(f"patched: {rel}")


def patch_scenario_reduction() -> None:
    """Include saved traffic-world structure in representative-world selection."""
    rel = "src/cvt_track_study/studies/scenario_reduction.py"
    path, text = read(rel)
    if 'features["traffic:restriction_burden_s"]' in text:
        print(f"already patched: {rel}")
        return
    anchor = (
        '        for path, alternatives in choice_values.items():\n'
        '            selected = str(scenario.get("choice_values", {}).get(path, ""))\n'
        '            for alternative in sorted(alternatives):\n'
        '                features[f"choice:{path}={alternative}"] = float(selected == alternative)\n'
        '\n'
        '        efficiency = _number(quantities.get("drivetrain.efficiency"))\n'
    )
    replacement = (
        '        for path, alternatives in choice_values.items():\n'
        '            selected = str(scenario.get("choice_values", {}).get(path, ""))\n'
        '            for alternative in sorted(alternatives):\n'
        '                features[f"choice:{path}={alternative}"] = float(selected == alternative)\n'
        '\n'
        '        # Traffic is a stochastic source-world input even though it is generated\n'
        '        # from an empirical point process rather than the ordinary quantity\n'
        '        # registry. Include compact descriptors in the diversity calculation.\n'
        '        # The exact traffic realization is still replayed; these are selection\n'
        '        # features only.\n'
        '        traffic = scenario.get("traffic_realization")\n'
        '        if isinstance(traffic, Mapping):\n'
        '            for field in (\n'
        '                "lap_start_race_time_s",\n'
        '                "initial_rate_per_hour",\n'
        '                "field_exponent",\n'
        '            ):\n'
        '                number = _number(traffic.get(field))\n'
        '                if math.isfinite(number):\n'
        '                    features[f"traffic:{field}"] = number\n'
        '            raw_events = traffic.get("events", ())\n'
        '            if isinstance(raw_events, Sequence) and not isinstance(raw_events, (str, bytes)):\n'
        '                event_count = 0\n'
        '                total_duration = 0.0\n'
        '                minimum_retained = 1.0\n'
        '                restriction_burden = 0.0\n'
        '                for raw_event in raw_events:\n'
        '                    if not isinstance(raw_event, Mapping):\n'
        '                        continue\n'
        '                    duration = _number(raw_event.get("duration_s"))\n'
        '                    retained = _number(raw_event.get("retained_speed_fraction"))\n'
        '                    if not (math.isfinite(duration) and math.isfinite(retained)):\n'
        '                        continue\n'
        '                    duration = max(0.0, duration)\n'
        '                    retained = min(1.0, max(0.0, retained))\n'
        '                    event_count += 1\n'
        '                    total_duration += duration\n'
        '                    minimum_retained = min(minimum_retained, retained)\n'
        '                    restriction_burden += duration * (1.0 - retained)\n'
        '                features["traffic:event_count"] = float(event_count)\n'
        '                features["traffic:total_annotated_duration_s"] = total_duration\n'
        '                features["traffic:minimum_retained_fraction"] = minimum_retained\n'
        '                features["traffic:restriction_burden_s"] = restriction_burden\n'
        '\n'
        '        efficiency = _number(quantities.get("drivetrain.efficiency"))\n'
    )
    text = replace_once(text, anchor, replacement, "scenario reduction traffic features")
    write(path, text)
    print(f"patched: {rel}")

def patch_reports_facade() -> None:
    rel = "src/cvt_track_study/reports/__init__.py"
    path, text = read(rel)
    if "augment_full_uncertainty_traffic_report" in text:
        print(f"already patched: {rel}")
        return
    anchor = "from .uncertainty_mechanism import enhance_full_uncertainty_report\n"
    text = replace_once(
        text,
        anchor,
        anchor + "from .traffic import augment_full_uncertainty_traffic_report\n",
        "reports traffic augmentation import",
    )
    text = replace_once(
        text,
        '''def write_full_uncertainty_report(output: Path) -> Path:
    target = _write_full_uncertainty_report(output)
    return enhance_full_uncertainty_report(Path(output), target)
''',
        '''def write_full_uncertainty_report(output: Path) -> Path:
    target = _write_full_uncertainty_report(output)
    target = enhance_full_uncertainty_report(Path(output), target)
    return augment_full_uncertainty_traffic_report(Path(output)) or target
''',
        "write full uncertainty traffic augmentation",
    )
    text = replace_once(
        text,
        '''        if str(raw.get("study_type", "")) == "full_uncertainty":
            target = enhance_full_uncertainty_report(Path(output), target)
''',
        '''        if str(raw.get("study_type", "")) == "full_uncertainty":
            target = enhance_full_uncertainty_report(Path(output), target)
            target = augment_full_uncertainty_traffic_report(Path(output)) or target
''',
        "regenerated full uncertainty traffic augmentation",
    )
    write(path, text)
    print(f"patched: {rel}")

def patch_cli() -> None:
    rel = "src/cvt_track_study/cli.py"
    path, text = read(rel)
    if "calibrate-traffic" in text:
        print(f"already patched: {rel}")
        return
    anchor = "from .runtime.results import discover_results, write_results_index\n"
    text = replace_once(
        text,
        anchor,
        anchor + "from .reports.traffic import write_traffic_calibration_project\n",
        "CLI traffic report import",
    )
    parser_anchor = '''    build_track_parser.add_argument("--output", type=Path)

    review_parser = subparsers.add_parser(
'''
    parser_insert = '''    build_track_parser.add_argument("--output", type=Path)

    traffic_parser = subparsers.add_parser(
        "calibrate-traffic",
        help="Calibrate and audit the empirical endurance-traffic model without running vehicle simulation.",
    )
    traffic_parser.add_argument("project", type=Path)
    traffic_parser.add_argument("--output", type=Path)

    review_parser = subparsers.add_parser(
'''
    text = replace_once(text, parser_anchor, parser_insert, "CLI traffic parser")
    main_anchor = '''        if args.command == "run" and args.run_command in {"nominal", "baseline"}:
'''
    traffic_main = '''        if args.command == "calibrate-traffic":
            report = write_traffic_calibration_project(
                args.project, output_directory=args.output
            )
            print(f"Traffic calibration report: {report}")
            return 0
        if args.command == "run" and args.run_command in {"nominal", "baseline"}:
'''
    text = replace_once(text, main_anchor, traffic_main, "CLI traffic command")
    write(path, text)
    print(f"patched: {rel}")


def main() -> int:
    targets = (
        "src/cvt_track_study/uncertainty/model.py",
        "src/cvt_track_study/simulation/dynamics.py",
        "src/cvt_track_study/simulation/integrator.py",
        "src/cvt_track_study/simulation/metrics.py",
        "src/cvt_track_study/studies/service_v8.py",
        "src/cvt_track_study/studies/ensemble_v10.py",
        "src/cvt_track_study/studies/design_grid_execution.py",
        "src/cvt_track_study/studies/design_replay_v11.py",
        "src/cvt_track_study/studies/scenario_reduction.py",
        "src/cvt_track_study/reports/__init__.py",
        "src/cvt_track_study/cli.py",
    )
    originals: dict[Path, str] = {}
    try:
        # Snapshot first so an unexpected newer source shape cannot leave a
        # half-applied overlay. Git remains the ultimate rollback, but the installer
        # itself is transactional for the files it edits.
        for rel in targets:
            path, text = read(rel)
            originals[path] = text
        patch_uncertainty_model()
        patch_dynamics()
        patch_integrator()
        patch_metrics()
        patch_service_v8()
        patch_ensemble_v10()
        patch_design_grid_execution()
        patch_design_replay_v11()
        patch_scenario_reduction()
        patch_reports_facade()
        patch_cli()
        for rel in (
            *targets,
            "src/cvt_track_study/simulation/traffic.py",
            "src/cvt_track_study/reports/traffic.py",
        ):
            source_path = ROOT / rel
            compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")
    except Exception as exc:
        for path, text in originals.items():
            write(path, text)
        print(
            f"Traffic overlay refused to patch and restored edited files: {exc}",
            file=sys.stderr,
        )
        return 2
    print("Traffic model wiring complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
