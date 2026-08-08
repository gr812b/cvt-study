"""Run an existing study independently for every route-family member.

This is intentionally a thin orchestration layer: every route bundle is a
separate course case, all study settings and random seeds remain unchanged, and
no route is probability-weighted above another.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cvt_track_study.track.router_v10 import build_project_track



def run_route_family_study_project(
    project: str | Path,
    *,
    study: str,
    family_manifest_path: Path | None = None,
    output_directory: Path | None = None,
    replicates_override: int | None = None,
) -> Path:
    from .router_v10 import run_study_project

    if family_manifest_path is None:
        build = build_project_track(project)
        if build.output_directory is None:
            raise RuntimeError("Track build did not expose an output directory.")
        family_manifest_path = build.output_directory / "route_family_manifest.json"
    manifest_path = family_manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    routes = list(manifest.get("routes", []))
    if not routes:
        raise ValueError(f"No route members were found in {manifest_path}.")

    output = (
        output_directory
        or manifest_path.parent
        / "route_family_studies"
        / study
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)

    case_rows: list[dict[str, Any]] = []
    combined_frames: list[pd.DataFrame] = []
    for route in routes:
        route_id = str(route["route_variant_id"])
        bundle = (manifest_path.parent / str(route["track_bundle"])).resolve()
        case_output = output / route_id
        result_path = run_study_project(
            project,
            study=study,
            bundle_path=bundle,
            output_directory=case_output,
            replicates_override=replicates_override,
        )
        normalized_weight = 1.0 / len(routes)
        case_rows.append(
            {
                "course_case_id": route_id,
                "route_variant_id": route_id,
                "nominal": bool(route.get("nominal", False)),
                "case_weight": 1.0,
                "normalized_equal_weight": normalized_weight,
                "track_bundle": str(bundle),
                "study_output": str(result_path.relative_to(output)),
            }
        )
        replicate_path = result_path / "replicate_results.csv"
        if replicate_path.exists() and replicate_path.stat().st_size:
            frame = pd.read_csv(replicate_path)
            frame.insert(0, "route_variant_id", route_id)
            frame.insert(1, "course_case_weight", 1.0)
            frame.insert(2, "normalized_course_case_weight", normalized_weight)
            combined_frames.append(frame)

    combined = (
        pd.concat(combined_frames, ignore_index=True, sort=False)
        if combined_frames
        else pd.DataFrame()
    )
    combined.to_csv(output / "route_family_replicate_results.csv", index=False)
    equal_summary = _equal_course_summary(combined)
    equal_summary.to_csv(output / "route_family_summary.csv", index=False)

    family_output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_name": study,
        "aggregation_policy": "unweighted_equal_course_cases",
        "route_family_manifest": str(manifest_path),
        "course_cases": case_rows,
        "combined_replicate_results": "route_family_replicate_results.csv",
        "equal_course_summary": "route_family_summary.csv",
        "interpretation": (
            "Each route is simulated as an independent course case with identical "
            "study settings. Any cross-route summary should give every route one "
            "equal case contribution rather than weighting by observed lap count."
        ),
    }
    (output / "route_family_study_manifest.json").write_text(
        json.dumps(family_output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _equal_course_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize the equal-mixture distribution over route course cases.

    Every route runner uses the same study definition and replicate count. The
    combined table is therefore an equal-size mixture of course cases, not a lap-
    count-weighted sample. A mismatch is rejected rather than silently reweighted.
    """

    if frame.empty:
        return pd.DataFrame()
    grouping_candidates = (
        "design_id",
        "design_path",
        "design_value",
        "design_value_si",
        "design_choice_value",
        "parameter_path",
        "level_probability",
        "level_kind",
    )
    group_columns = [name for name in grouping_candidates if name in frame.columns]
    identity = {
        "replicate",
        "scenario_seed",
        "course_case_weight",
        "normalized_course_case_weight",
    } | set(group_columns)
    metric_columns = [
        name
        for name in frame.select_dtypes(include=[np.number, "bool"]).columns
        if name not in identity
    ]
    grouped = [((), frame)] if not group_columns else frame.groupby(
        group_columns, dropna=False, sort=False
    )
    rows: list[dict[str, Any]] = []
    for key, segment in grouped:
        route_counts = segment.groupby("route_variant_id").size()
        if route_counts.nunique() > 1:
            raise ValueError(
                "Route-family study produced unequal row counts across course cases "
                f"for group {key!r}: {route_counts.to_dict()}."
            )
        base: dict[str, Any] = {}
        if group_columns:
            values = key if isinstance(key, tuple) else (key,)
            base.update(dict(zip(group_columns, values)))
        for metric in metric_columns:
            values = pd.to_numeric(segment[metric], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                continue
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "course_case_count": int(segment["route_variant_id"].nunique()),
                    "sample_count": int(len(values)),
                    "mean": float(np.mean(values)),
                    "p10": float(np.quantile(values, 0.10)),
                    "median": float(np.median(values)),
                    "p90": float(np.quantile(values, 0.90)),
                    "aggregation_policy": "equal_mixture_of_course_cases",
                }
            )
    return pd.DataFrame(rows)


@contextmanager
def family_balanced_track_ensemble_runtime():
    """Keep normal uncertainty/design studies balanced across route families.

    The stock joint-ensemble loader truncates a flat list of track cases at
    ``track_ensemble.maximum_cases``.  A route-family robustness result is
    hierarchical, so flat truncation can accidentally give one long/short route
    more interpretations than the other.  This adapter expands the family
    manifest first, then selects the same interpretation groups on every route.
    """

    from . import ensemble_v10 as ensemble

    original = ensemble._track_variants

    def balanced_track_variants(
        *,
        project: str | Path,
        resolution: Any,
        raw: Mapping[str, Any],
        nominal_path: Path,
        nominal_bundle: Any,
        progress: bool,
    ):
        config = raw.get("track_ensemble", {})
        config = config if isinstance(config, Mapping) else {}
        maximum = int(config.get("maximum_cases", 20))
        source_value = config.get("result_directory", "latest")
        if (
            source_value == "latest"
            and bool(config.get("auto_build", True))
            and ensemble._latest_track_robustness_result(
                resolution.paths.results_directory
            )
            is None
        ):
            from cvt_track_study.track.robustness_family import (
                route_family_robustness_enabled,
                run_route_family_track_robustness_project,
            )

            if route_family_robustness_enabled(resolution):
                run_route_family_track_robustness_project(
                    project,
                    study=str(config.get("study", "track_robustness")),
                    resume=True,
                    progress=progress,
                    run_name=str(config.get("study", "track_robustness")),
                    command=(
                        "drivetrain-study",
                        "run",
                        "track-robustness",
                        str(project),
                    ),
                )
        expanded_raw = deepcopy(dict(raw))
        expanded_config = dict(config)
        expanded_config["maximum_cases"] = max(10000, maximum)
        expanded_raw["track_ensemble"] = expanded_config
        variants, source = original(
            project=project,
            resolution=resolution,
            raw=expanded_raw,
            nominal_path=nominal_path,
            nominal_bundle=nominal_bundle,
            progress=progress,
        )
        if source is None:
            return tuple(variants[:maximum]), source
        manifest_path = source / "track_ensemble_manifest.json"
        if not manifest_path.is_file():
            return tuple(variants[:maximum]), source
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not bool(manifest.get("route_family_balanced", False)):
            return tuple(variants[:maximum]), source
        return _balanced_family_variants(
            variants,
            manifest=manifest,
            maximum=maximum,
        ), source

    ensemble._track_variants = balanced_track_variants
    try:
        yield
    finally:
        ensemble._track_variants = original


def _balanced_family_variants(
    variants: tuple[Any, ...] | list[Any],
    *,
    manifest: Mapping[str, Any],
    maximum: int,
) -> tuple[Any, ...]:
    route_ids = [str(value) for value in manifest.get("route_variant_ids", ())]
    nominal_route = str(manifest.get("nominal_route_variant_id", ""))
    if not variants or not route_ids or maximum < len(route_ids):
        return tuple(variants[:maximum])

    by_route_group: dict[str, dict[str, Any]] = {route_id: {} for route_id in route_ids}
    implicit_nominal = variants[0]
    by_route_group.setdefault(nominal_route, {})["nominal"] = implicit_nominal
    for variant in variants[1:]:
        case_id = str(getattr(variant, "case_id", ""))
        if "__" not in case_id:
            continue
        route_id, group = case_id.split("__", 1)
        if route_id in by_route_group:
            by_route_group[route_id][group] = variant

    common_groups = set.intersection(
        *(set(groups) for groups in by_route_group.values())
    ) if by_route_group else set()
    common_groups.discard("nominal")
    order_from_manifest = [
        str(value)
        for value in manifest.get("common_eligible_robustness_case_ids", ())
        if str(value) in common_groups
    ]
    missing = sorted(common_groups - set(order_from_manifest))
    order = order_from_manifest + missing
    original_position = {group: index for index, group in enumerate(order)}
    order.sort(
        key=lambda group: (
            _family_group_priority(group), original_position.get(group, 10**9)
        )
    )

    per_route = max(1, maximum // len(route_ids))
    selected_groups = order[: max(0, per_route - 1)]
    selected: list[Any] = [implicit_nominal]
    # Add the alternate nominal traversal(s) so every route begins with one
    # nominal interpretation before robustness variants are added.
    for route_id in route_ids:
        if route_id == nominal_route:
            continue
        variant = by_route_group.get(route_id, {}).get("nominal")
        if variant is not None:
            selected.append(variant)
    for group in selected_groups:
        for route_id in route_ids:
            variant = by_route_group.get(route_id, {}).get(group)
            if variant is not None:
                selected.append(variant)
    return tuple(selected[:maximum])


def _family_group_priority(group: str) -> tuple[int, str]:
    """Prefer physical reconstruction diversity before policy-only variants."""

    if group.startswith("centreline_"):
        bucket = 0
    elif group.startswith("cleanup_"):
        bucket = 1
    elif group.startswith("event_windows_"):
        bucket = 2
    elif group.startswith("gate_"):
        bucket = 3
    else:
        bucket = 4
    return bucket, group
