"""Apply the reviewed Max performance changes without importing unrelated work.

This transformer is intentionally narrow. It edits only three existing modules:

* studies/service_v8.py     - CPU-bound scenarios use worker processes;
* studies/ensemble_v10.py   - joint track ensembles use the same process workers;
* simulation/integrator.py  - feature-entry detection no longer scans every feature
                               on every integration step.

The script refuses to continue if an expected source shape is not found. That is
preferable to silently corrupting a newer local version of the repository.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PatchError(RuntimeError):
    pass


def _read(relative: str) -> tuple[Path, str]:
    path = ROOT / relative
    if not path.is_file():
        raise PatchError(f"Required file not found: {relative}")
    return path, path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count == 0:
        raise PatchError(f"Could not find expected source block for {label}.")
    if count != 1:
        raise PatchError(f"Expected one source block for {label}, found {count}.")
    return text.replace(old, new, 1)


def _replace_regex_once(text: str, pattern: str, replacement: str, *, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise PatchError(f"Could not uniquely patch {label}; matches={count}.")
    return updated


def patch_service_v8() -> None:
    relative = "src/cvt_track_study/studies/service_v8.py"
    path, text = _read(relative)
    if "_SCENARIO_PROCESS_CONTEXT" in text and "interruptible_process_pool" in text:
        print(f"already patched: {relative}")
        return

    # Remove only ThreadPoolExecutor; as_completed remains useful for process futures.
    text, count = re.subn(
        r"from concurrent\.futures import ([^\n]*?)ThreadPoolExecutor,\s*as_completed",
        lambda m: "from concurrent.futures import " + ((m.group(1) or "") + "as_completed").replace(", ,", ","),
        text,
        count=1,
    )
    if count == 0:
        text, count = re.subn(
            r"from concurrent\.futures import as_completed,\s*ThreadPoolExecutor",
            "from concurrent.futures import as_completed",
            text,
            count=1,
        )
    if count != 1:
        raise PatchError("Could not find the ThreadPoolExecutor import in service_v8.py.")

    runtime_import = "from cvt_track_study.runtime.process_pool import interruptible_process_pool\n"
    anchor = "from cvt_track_study.runtime.results import write_results_index\n"
    if runtime_import not in text:
        text = _replace_once(text, anchor, anchor + runtime_import, label="service_v8 process-pool import")

    worker_support = r'''

_SCENARIO_PROCESS_CONTEXT: dict[str, Any] | None = None


def _initialize_scenario_process(
    design_points: tuple[DesignPoint, ...],
    study_type: str,
    vehicle_id: str,
    vehicle_raw: Mapping[str, Any],
    base_study: Mapping[str, Any],
    track_raw: Mapping[str, Any],
    bundles: Mapping[str, TrackBundle],
    cache_root: Path,
    cache_enabled: bool,
) -> None:
    """Load static scenario data once per worker process.

    The old thread runner shared these objects implicitly. With process workers we
    deliberately initialize them once per process instead of serializing the full
    study context with every submitted scenario.
    """

    global _SCENARIO_PROCESS_CONTEXT
    _SCENARIO_PROCESS_CONTEXT = {
        "design_points": design_points,
        "study_type": study_type,
        "vehicle_id": vehicle_id,
        "vehicle_raw": vehicle_raw,
        "base_study": base_study,
        "track_raw": track_raw,
        "bundles": dict(bundles),
        "cache": SimulationCache(cache_root, enabled=cache_enabled),
    }


def _execute_scenario_process_initialized(
    scenario: ScenarioDraw, bundle_key: str = "default"
) -> dict[str, Any]:
    """Execute one scenario using worker-local static context."""

    context = _SCENARIO_PROCESS_CONTEXT
    if context is None:
        raise RuntimeError("Scenario worker was not initialized.")
    cache = context["cache"]
    before = (cache.hits, cache.misses, cache.writes)
    result = _execute_scenario(
        scenario=scenario,
        design_points=context["design_points"],
        study_type=context["study_type"],
        vehicle_id=context["vehicle_id"],
        vehicle_raw=context["vehicle_raw"],
        base_study=context["base_study"],
        track_raw=context["track_raw"],
        bundle=context["bundles"][bundle_key],
        cache=cache,
    )
    result["_process_cache_counts"] = {
        "hits": cache.hits - before[0],
        "misses": cache.misses - before[1],
        "writes": cache.writes - before[2],
    }
    return result


def _merge_process_cache_counts(cache: SimulationCache, result: dict[str, Any]) -> None:
    """Fold worker-local cache counters back into the parent manifest counters."""

    counts = result.pop("_process_cache_counts", None)
    if not isinstance(counts, Mapping):
        return
    cache.hits += int(counts.get("hits", 0))
    cache.misses += int(counts.get("misses", 0))
    cache.writes += int(counts.get("writes", 0))
'''

    marker = "\n_SUPPORTED_TYPES = {"
    if marker not in text:
        raise PatchError("Could not find _SUPPORTED_TYPES insertion point in service_v8.py.")
    text = text.replace(marker, worker_support + marker, 1)

    old_parallel = '''    scenario_results: list[dict[str, Any]] = []
    if workers == 1 or len(scenarios) == 1:
        for scenario in scenarios:
            scenario_results.append(execute_or_resume(scenario))
            reporter.advance(f"replicate {scenario.replicate}")
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(scenarios))) as executor:
            futures = {
                executor.submit(execute_or_resume, scenario): scenario
                for scenario in scenarios
            }
            for future in as_completed(futures):
                scenario = futures[future]
                scenario_results.append(future.result())
                reporter.advance(f"replicate {scenario.replicate}")
'''

    new_parallel = '''    scenario_results: list[dict[str, Any]] = []
    if workers == 1 or len(scenarios) == 1:
        for scenario in scenarios:
            scenario_results.append(execute_or_resume(scenario))
            reporter.advance(f"replicate {scenario.replicate}")
    else:
        # Checkpoints are owned by the parent process. Only genuinely pending
        # scenarios are submitted, and workers never write shared workspace state.
        pending: list[ScenarioDraw] = []
        for scenario in scenarios:
            checkpoint = workspace.load_checkpoint(scenario.replicate)
            if checkpoint is None:
                pending.append(scenario)
                continue
            result = dict(checkpoint["result"])
            result["resumed"] = True
            scenario_results.append(result)
            reporter.advance(f"replicate {scenario.replicate} (resumed)")

        if pending:
            with interruptible_process_pool(
                max_workers=min(workers, len(pending)),
                initializer=_initialize_scenario_process,
                initargs=(
                    design_points,
                    study_type,
                    vehicle_id,
                    vehicle_raw,
                    base_study,
                    track_raw,
                    {"default": bundle},
                    cache.root,
                    cache.enabled,
                ),
            ) as executor:
                futures = {
                    executor.submit(
                        _execute_scenario_process_initialized, scenario, "default"
                    ): scenario
                    for scenario in pending
                }
                for future in as_completed(futures):
                    scenario = futures[future]
                    result = future.result()
                    _merge_process_cache_counts(cache, result)
                    workspace.write_checkpoint(scenario.replicate, {"result": result})
                    scenario_results.append(result)
                    reporter.advance(f"replicate {scenario.replicate}")
'''
    text = _replace_once(text, old_parallel, new_parallel, label="service_v8 parallel execution")

    if '"parallel_backend":' not in text:
        text = _replace_once(
            text,
            '        "parallel_workers": workers,\n',
            '        "parallel_workers": workers,\n'
            '        "parallel_backend": "process" if workers > 1 and len(scenarios) > 1 else "serial",\n',
            label="service_v8 parallel backend manifest",
        )

    _write(path, text)
    print(f"patched: {relative}")


def patch_ensemble_v10() -> None:
    relative = "src/cvt_track_study/studies/ensemble_v10.py"
    path, text = _read(relative)
    if "interruptible_process_pool" in text and "_execute_scenario_process_initialized" in text:
        print(f"already patched: {relative}")
        return

    # The exact import order has varied; remove ThreadPoolExecutor wherever it sits.
    text = re.sub(
        r"from concurrent\.futures import ThreadPoolExecutor,\s*as_completed",
        "from concurrent.futures import as_completed",
        text,
        count=1,
    )
    text = re.sub(
        r"from concurrent\.futures import as_completed,\s*ThreadPoolExecutor",
        "from concurrent.futures import as_completed",
        text,
        count=1,
    )
    if "ThreadPoolExecutor" in text.split("\n", 30)[0:30]:
        raise PatchError("Could not cleanly remove ThreadPoolExecutor import from ensemble_v10.py.")

    runtime_import = "from cvt_track_study.runtime.process_pool import interruptible_process_pool\n"
    if runtime_import not in text:
        # ensemble_v10 imports several cvt_track_study.runtime modules. This anchor
        # exists in the current framework and keeps the new dependency explicit.
        anchor = "from cvt_track_study.runtime.results import write_results_index\n"
        text = _replace_once(text, anchor, anchor + runtime_import, label="ensemble_v10 process-pool import")

    # Replace only the execution section; scheduling, sampling, equal-route
    # weighting, result annotation, ordering, and reporting remain untouched.
    pattern = r'''    results: list\[dict\[str, Any\]\] = \[\]\n    if workers == 1 or len\(selected\) == 1:\n.*?\n    order = \{point\.identifier: index for index, point in enumerate\(design_points\)\}'''
    replacement = '''    results: list[dict[str, Any]] = []
    if workers == 1 or len(selected) == 1:
        for item in selected:
            result = execute_or_resume(item)
            results.append(result)
            reporter.advance(
                f"scenario {item.scenario.replicate}; draw={item.base_draw_id}; track={result['track_case_id']}"
            )
    else:
        # Resume/checkpoint IO remains parent-only. Workers receive only the
        # ScenarioDraw and a small bundle key; all static study data is initialized
        # once per process by service_v8._initialize_scenario_process.
        pending: list[ScheduledScenario] = []
        for item in selected:
            checkpoint = workspace.load_checkpoint(item.scenario.replicate)
            if checkpoint is None:
                pending.append(item)
                continue
            result = dict(checkpoint["result"])
            result["resumed"] = True
            results.append(result)
            reporter.advance(
                f"scenario {item.scenario.replicate}; draw={item.base_draw_id}; "
                f"track={result['track_case_id']} (resumed)"
            )

        if pending:
            bundles_by_case = {
                item.variant.case_id: item.variant.bundle for item in selected
            }
            with interruptible_process_pool(
                max_workers=min(workers, len(pending)),
                initializer=service_v8._initialize_scenario_process,
                initargs=(
                    design_points,
                    study_type,
                    vehicle_id,
                    vehicle_raw,
                    base_study,
                    resolution.data["track"],
                    bundles_by_case,
                    cache.root,
                    cache.enabled,
                ),
            ) as executor:
                futures = {
                    executor.submit(
                        service_v8._execute_scenario_process_initialized,
                        item.scenario,
                        item.variant.case_id,
                    ): item
                    for item in pending
                }
                for future in as_completed(futures):
                    item = futures[future]
                    result = future.result()
                    service_v8._merge_process_cache_counts(cache, result)
                    annotate_result(item, result)
                    workspace.write_checkpoint(
                        item.scenario.replicate, {"result": result}
                    )
                    results.append(result)
                    reporter.advance(
                        f"scenario {item.scenario.replicate}; draw={item.base_draw_id}; "
                        f"track={result['track_case_id']}"
                    )

    order = {point.identifier: index for index, point in enumerate(design_points)}'''
    text = _replace_regex_once(text, pattern, replacement, label="ensemble_v10 parallel execution")

    if '"parallel_backend":' not in text:
        text = _replace_once(
            text,
            '        "parallel_workers": workers,\n',
            '        "parallel_workers": workers,\n'
            '        "parallel_backend": "process" if workers > 1 and len(selected) > 1 else "serial",\n',
            label="ensemble_v10 parallel backend manifest",
        )

    _write(path, text)
    print(f"patched: {relative}")


def patch_integrator() -> None:
    relative = "src/cvt_track_study/simulation/integrator.py"
    path, text = _read(relative)
    if "_record_ordered_feature_entry_crossings" in text:
        print(f"already patched: {relative}")
        return

    old_setup = '''    feature_entry_speeds: dict[str, float] = {}
    feature_obstacle_energy = {feature.identifier: 0.0 for feature in track.features}
    completed = False
'''
    new_setup = '''    feature_entry_speeds: dict[str, float] = {
        feature.identifier: speed
        for feature in track.features
        if feature.interval.local_distance(distance, track.length_m) is not None
    }
    feature_obstacle_energy = {feature.identifier: 0.0 for feature in track.features}
    ordered_feature_starts = tuple(
        sorted(track.features, key=lambda feature: feature.interval.start_s_m)
    )
    next_feature_start_index = 0
    while (
        next_feature_start_index < len(ordered_feature_starts)
        and ordered_feature_starts[next_feature_start_index].interval.start_s_m <= distance
    ):
        next_feature_start_index += 1
    completed = False
'''
    text = _replace_once(text, old_setup, new_setup, label="integrator ordered feature setup")

    scan_pattern = r'''        for feature in track\.features:\n            if \(\n                feature\.identifier not in feature_entry_speeds\n                and feature\.interval\.local_distance\(distance, track\.length_m\) is not None\n            \):\n                feature_entry_speeds\[feature\.identifier\] = speed\n'''
    text = _replace_regex_once(text, scan_pattern, "", label="integrator per-step feature scan")

    old_call = '''        _record_feature_entry_crossings(
            features=track.features,
            recorded=feature_entry_speeds,
            start_distance_m=distance,
            end_distance_m=new_distance,
            start_speed_mps=speed,
            end_speed_mps=new_speed,
        )
'''
    new_call = '''        next_feature_start_index = _record_ordered_feature_entry_crossings(
            features=ordered_feature_starts,
            start_index=next_feature_start_index,
            recorded=feature_entry_speeds,
            start_distance_m=distance,
            end_distance_m=new_distance,
            start_speed_mps=speed,
            end_speed_mps=new_speed,
        )
'''
    text = _replace_once(text, old_call, new_call, label="integrator ordered crossing call")

    helper = r'''

def _record_ordered_feature_entry_crossings(
    *,
    features: tuple[Any, ...],
    start_index: int,
    recorded: dict[str, float],
    start_distance_m: float,
    end_distance_m: float,
    start_speed_mps: float,
    end_speed_mps: float,
) -> int:
    """Capture crossed feature entries in O(crossed features), not O(all features).

    Vehicle distance is monotone in this lap integrator. Once a feature start lies
    behind the current distance it never needs to be considered again. The speed at
    the exact boundary is still linearly interpolated exactly as in the legacy
    helper, so this changes search cost rather than the physical/numerical model.
    """

    span = end_distance_m - start_distance_m
    if span <= 0.0:
        return start_index
    index = start_index
    while index < len(features):
        feature = features[index]
        boundary = float(feature.interval.start_s_m)
        if boundary > end_distance_m:
            break
        if boundary > start_distance_m and feature.identifier not in recorded:
            fraction = (boundary - start_distance_m) / span
            fraction = min(1.0, max(0.0, fraction))
            recorded[feature.identifier] = (
                start_speed_mps + fraction * (end_speed_mps - start_speed_mps)
            )
        index += 1
    return index
'''
    marker = "\ndef _record_feature_entry_crossings("
    if marker not in text:
        raise PatchError("Could not find legacy feature-crossing helper in integrator.py.")
    text = text.replace(marker, helper + marker, 1)

    _write(path, text)
    print(f"patched: {relative}")


def main() -> int:
    try:
        patch_service_v8()
        patch_ensemble_v10()
        patch_integrator()
    except PatchError as exc:
        print(f"PERFORMANCE PATCH FAILED: {exc}", file=sys.stderr)
        return 2
    print("Performance speedups applied successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
