"""Build and audit a family of genuinely different closed-course routes.

The strict detector first identifies tight geometric line clusters.  This module
merges compatible clusters into topology-level route families, builds one robust
consensus per family, and then evaluates gate evidence locally so a lap may support
a physical gate on a shared section even when it uses another branch elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from cvt_track_study.config.diagnostics import DiagnosticBag

from .consensus import build_iterative_consensus
from .events import build_response_features, normalize_events, project_events
from .gates import build_gate_review, score_speed_gates
from .geo import LocalFrame
from .laps import _clean_speed, build_track_profile
from .metrics import extract_event_passes
from .reconstruction import TrackEvidenceBuild, _combine_points, build_track_evidence
from .route_variants import (
    RouteVariantSettings,
    finalize_route_variant_semantics,
)
from .settings import ReconstructionSettings


_MAP_ONLY_QUALITY_FLAGS = {
    "large_backward_map_match",
    "p95_map_error_exceeds_limit",
    "no_map_matched_points",
    "no_finite_map_errors",
    "consensus_geometry_outlier",
}


@dataclass(frozen=True)
class RouteFamilyEvidenceBuild:
    nominal_route_variant_id: str
    routes: Mapping[str, TrackEvidenceBuild]
    route_variant_summary: pd.DataFrame
    route_variant_pairwise: pd.DataFrame
    shared_gate_evidence: pd.DataFrame
    event_route_applicability: pd.DataFrame
    course_cases: pd.DataFrame
    branch_summary: pd.DataFrame
    event_route_sections: pd.DataFrame

    @property
    def nominal(self) -> TrackEvidenceBuild:
        return self.routes[self.nominal_route_variant_id]


@dataclass(frozen=True)
class RouteFamilySettings:
    enabled: bool = True
    merge_line_clusters: bool = True
    pool_shared_gate_evidence: bool = True
    event_projection_max_error_m: float | None = None
    minimum_event_window_coverage_fraction: float = 0.60
    event_projection_bin_size_m: float = 5.0
    maximum_event_interpolation_gap_m: float = 25.0
    maximum_local_backward_matches: int = 1
    topology_same_route_p95_distance_m: float | None = None
    topology_maximum_divergent_fraction: float | None = None
    topology_maximum_length_relative_difference: float | None = None
    topology_attachment_ambiguity_ratio: float = 0.80
    branch_divergence_distance_m: float | None = None
    branch_minimum_sustained_length_m: float = 60.0
    branch_gap_tolerance_m: float = 25.0
    branch_boundary_padding_m: float = 15.0

    @classmethod
    def from_mapping(cls, track: Mapping[str, Any]) -> "RouteFamilySettings":
        raw_variants = track.get("route_variants", {})
        raw_variants = raw_variants if isinstance(raw_variants, Mapping) else {}
        raw = raw_variants.get("family", {})
        raw = raw if isinstance(raw, Mapping) else {}
        value = raw.get("event_projection_max_error_m")
        maximum_error = None if value in (None, "") else float(value)
        result = cls(
            enabled=bool(raw.get("enabled", True)),
            merge_line_clusters=bool(raw.get("merge_line_clusters", True)),
            pool_shared_gate_evidence=bool(
                raw.get("pool_shared_gate_evidence", True)
            ),
            event_projection_max_error_m=maximum_error,
            minimum_event_window_coverage_fraction=float(
                raw.get("minimum_event_window_coverage_fraction", 0.60)
            ),
            event_projection_bin_size_m=float(
                raw.get("event_projection_bin_size_m", 5.0)
            ),
            maximum_event_interpolation_gap_m=float(
                raw.get("maximum_event_interpolation_gap_m", 25.0)
            ),
            maximum_local_backward_matches=int(
                raw.get("maximum_local_backward_matches", 1)
            ),
            topology_same_route_p95_distance_m=_optional_positive_float(
                raw.get("topology_same_route_p95_distance_m")
            ),
            topology_maximum_divergent_fraction=_optional_fraction(
                raw.get("topology_maximum_divergent_fraction")
            ),
            topology_maximum_length_relative_difference=_optional_fraction(
                raw.get("topology_maximum_length_relative_difference")
            ),
            topology_attachment_ambiguity_ratio=float(
                raw.get("topology_attachment_ambiguity_ratio", 0.80)
            ),
            branch_divergence_distance_m=_optional_positive_float(
                raw.get("branch_divergence_distance_m")
            ),
            branch_minimum_sustained_length_m=float(
                raw.get("branch_minimum_sustained_length_m", 60.0)
            ),
            branch_gap_tolerance_m=float(
                raw.get("branch_gap_tolerance_m", 25.0)
            ),
            branch_boundary_padding_m=float(
                raw.get("branch_boundary_padding_m", 15.0)
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.event_projection_max_error_m is not None:
            if not math.isfinite(self.event_projection_max_error_m) or self.event_projection_max_error_m <= 0:
                raise ValueError(
                    "track.route_variants.family.event_projection_max_error_m "
                    "must be positive and finite"
                )
        if not 0.0 < self.minimum_event_window_coverage_fraction <= 1.0:
            raise ValueError(
                "track.route_variants.family.minimum_event_window_coverage_fraction "
                "must be in (0, 1]"
            )
        if not math.isfinite(self.event_projection_bin_size_m) or self.event_projection_bin_size_m <= 0:
            raise ValueError(
                "track.route_variants.family.event_projection_bin_size_m "
                "must be positive and finite"
            )
        if (
            not math.isfinite(self.maximum_event_interpolation_gap_m)
            or self.maximum_event_interpolation_gap_m <= 0.0
        ):
            raise ValueError(
                "track.route_variants.family.maximum_event_interpolation_gap_m "
                "must be positive and finite"
            )
        if self.maximum_local_backward_matches < 0:
            raise ValueError(
                "track.route_variants.family.maximum_local_backward_matches "
                "must be non-negative"
            )
        if not 0.0 < self.topology_attachment_ambiguity_ratio < 1.0:
            raise ValueError(
                "track.route_variants.family.topology_attachment_ambiguity_ratio "
                "must be in (0, 1)"
            )
        for name, value in (
            ("branch_minimum_sustained_length_m", self.branch_minimum_sustained_length_m),
            ("branch_gap_tolerance_m", self.branch_gap_tolerance_m),
            ("branch_boundary_padding_m", self.branch_boundary_padding_m),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"track.route_variants.family.{name} must be finite and non-negative"
                )


def build_track_family_evidence(
    ingestion_results: tuple[Any, ...],
    track_config: Mapping[str, Any],
    raw_events: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    diagnostics: Any,
) -> RouteFamilyEvidenceBuild:
    """Build topology-level route families from strict geometric line clusters.

    The base detector intentionally creates tight complete-link clusters.  Those
    clusters are useful centreline seeds, but ordinary driving-line variation can
    split one physical course into several seeds.  The family layer therefore
    performs a second, more topology-oriented merge before constructing the final
    per-route consensuses.
    """

    family_settings = RouteFamilySettings.from_mapping(track_config)
    original_variant_settings = RouteVariantSettings.from_mapping(track_config)
    use_topology_merge = (
        family_settings.enabled
        and family_settings.merge_line_clusters
        and original_variant_settings.enabled
    )

    seed_config = deepcopy(dict(track_config))
    seed_diagnostics = DiagnosticBag() if use_topology_merge else diagnostics
    if use_topology_merge:
        # The seed pass exists only to obtain strict line clusters and lap-pair
        # geometry.  Final nominal selection is applied to topology families below,
        # so a topology route id must never be interpreted as a seed-cluster id.
        raw_variants = seed_config.get("route_variants", {})
        raw_variants = (
            dict(raw_variants) if isinstance(raw_variants, Mapping) else {}
        )
        raw_variants["selection"] = "largest_supported"
        raw_variants.pop("selected_variant_id", None)
        seed_config["route_variants"] = raw_variants

    seed = build_track_evidence(
        ingestion_results,
        seed_config,
        list(raw_events),
        seed_diagnostics,
    )

    if use_topology_merge:
        topology = _merge_geometric_clusters_into_topology_families(
            seed=seed,
            original_settings=original_variant_settings,
            family_settings=family_settings,
        )
        supported = _supported_variant_ids(topology.summary)
        if not supported:
            raise ValueError(
                "Route-family topology merge found no repeatedly supported route."
            )
        nominal_id, selection_reason = _select_topology_family(
            topology.summary,
            seed.laps,
            original_variant_settings,
        )
        topology.summary["selected"] = (
            topology.summary["route_variant_id"].astype(str) == nominal_id
        )
        topology.summary["selection_reason"] = ""
        topology.summary.loc[
            topology.summary["route_variant_id"].astype(str) == nominal_id,
            "selection_reason",
        ] = selection_reason

        unsupported_count = int((~topology.summary["supported"].astype(bool)).sum())
        unsupported_laps = int(
            topology.summary.loc[
                ~topology.summary["supported"].astype(bool), "lap_count"
            ].sum()
        )
        diagnostics.info(
            "ROUTE_TOPOLOGY_FAMILIES_DETECTED",
            (
                f"Merged {len(seed.route_variant_summary)} strict geometric line "
                f"cluster(s) into {len(supported)} supported course-topology "
                f"route family/families. Selected {nominal_id} for the nominal build."
            ),
            hint=(
                "Ordinary line choice is absorbed within each topology family; "
                "sustained branch divergence remains separated."
            ),
        )
        if unsupported_count:
            diagnostics.warning(
                "ROUTE_TOPOLOGY_OUTLIERS_RETAINED",
                (
                    f"Retained {unsupported_laps} lap(s) in {unsupported_count} "
                    "isolated or ambiguous topology component(s); they remain in "
                    "the audit but do not define a route centreline."
                ),
            )

        all_points, frame = _prepare_all_points(
            ingestion_results, track_config
        )
        routes = {
            route_id: _build_topology_route(
                route_id=route_id,
                seed=seed,
                topology=topology,
                all_points=all_points,
                frame=frame,
                track_config=track_config,
                raw_events=list(raw_events),
                diagnostics=diagnostics,
                original_variant_settings=original_variant_settings,
            )
            for route_id in supported
        }
        support_text = ", ".join(
            f"{route_id}={int(route.laps['analysis_valid'].sum())} valid lap(s)"
            for route_id, route in routes.items()
        )
        diagnostics.info(
            "ROUTE_FAMILY_GEOMETRY_SUPPORT",
            f"Final topology-family geometry support: {support_text}.",
            hint=(
                "The CLI's final valid-lap count refers only to the selected nominal "
                "route; all exported family members are listed here."
            ),
        )
        summary = topology.summary
        pairwise = topology.pairwise
    else:
        nominal_id = seed.selected_route_variant_id
        summary = seed.route_variant_summary.copy()
        pairwise = seed.route_variant_pairwise.copy()
        supported = _supported_variant_ids(summary)
        if not supported:
            supported = [nominal_id]
        if nominal_id not in supported:
            supported.insert(0, nominal_id)
        routes: dict[str, TrackEvidenceBuild] = {nominal_id: seed}
        build_family = (
            family_settings.enabled
            and original_variant_settings.enabled
            and len(supported) > 1
        )
        if build_family:
            for route_id in supported:
                if route_id == nominal_id:
                    continue
                member_config = deepcopy(dict(track_config))
                route_raw = member_config.get("route_variants", {})
                route_raw = (
                    dict(route_raw) if isinstance(route_raw, Mapping) else {}
                )
                route_raw["enabled"] = True
                route_raw["selection"] = "variant_id"
                route_raw["selected_variant_id"] = route_id
                member_config["route_variants"] = route_raw
                routes[route_id] = build_track_evidence(
                    ingestion_results,
                    member_config,
                    list(raw_events),
                    diagnostics,
                )

    build_family = family_settings.enabled and len(routes) > 1
    applicability = pd.DataFrame()
    branch_summary = pd.DataFrame()
    event_sections = pd.DataFrame()
    if build_family:
        reconstruction = ReconstructionSettings.from_mapping(track_config)
        maximum_error = (
            family_settings.event_projection_max_error_m
            if family_settings.event_projection_max_error_m is not None
            else min(
                reconstruction.maximum_map_error_m,
                original_variant_settings.same_variant_p95_distance_m,
            )
        )
        # Resolve the physical branch graph from geometry before filtering events.
        # Event projection never influences a centreline, so this lets applicability
        # distinguish a genuine branch from ordinary shared-course line variation.
        branch_summary, _ = _detect_route_branch_network(
            routes=routes,
            nominal_id=nominal_id,
            family_settings=family_settings,
            variant_settings=original_variant_settings,
            diagnostics=diagnostics,
            classify_events=False,
        )
        shared_maximum_error = _shared_course_projection_error_limit(
            strict_error_m=maximum_error,
            branch_summary=branch_summary,
        )
        routes, applicability = _restrict_route_events(
            routes=routes,
            normalized_events=normalize_events(list(raw_events)),
            reconstruction=reconstruction,
            maximum_error_m=maximum_error,
            shared_maximum_error_m=shared_maximum_error,
            branch_summary=branch_summary,
            diagnostics=diagnostics,
        )
        event_sections = _classify_route_event_sections(routes, branch_summary)
        if family_settings.pool_shared_gate_evidence:
            routes = _pool_gate_evidence(
                ingestion_results=ingestion_results,
                track_config=track_config,
                routes=routes,
                source_laps=next(iter(routes.values())).laps,
                settings=family_settings,
                maximum_error_m=maximum_error,
                shared_maximum_error_m=shared_maximum_error,
                diagnostics=diagnostics,
                event_route_sections=event_sections,
            )

    shared = _shared_gate_evidence(routes)
    cases = _course_cases(routes, nominal_id)
    summary = _augment_summary(summary, routes, nominal_id)
    return RouteFamilyEvidenceBuild(
        nominal_route_variant_id=nominal_id,
        routes=routes,
        route_variant_summary=summary,
        route_variant_pairwise=pairwise,
        shared_gate_evidence=shared,
        event_route_applicability=applicability,
        course_cases=cases,
        branch_summary=branch_summary,
        event_route_sections=event_sections,
    )


@dataclass(frozen=True)
class _TopologyMerge:
    summary: pd.DataFrame
    pairwise: pd.DataFrame
    primitive_to_topology: Mapping[str, str]


def _optional_positive_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("topology distance thresholds must be positive and finite")
    return number


def _optional_fraction(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number < 1.0:
        raise ValueError("topology fraction thresholds must be in [0, 1)")
    return number


def _prepare_all_points(
    ingestion_results: tuple[Any, ...],
    track_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, LocalFrame]:
    reconstruction = ReconstructionSettings.from_mapping(track_config)
    all_points = _combine_points(ingestion_results)
    frame = LocalFrame(
        float(all_points["latitude_deg"].median()),
        float(all_points["longitude_deg"].median()),
    )
    x_m, y_m = frame.to_xy(
        all_points["latitude_deg"], all_points["longitude_deg"]
    )
    all_points["x_m"] = x_m
    all_points["y_m"] = y_m
    all_points["speed_analysis_mps"] = _clean_speed(
        all_points, reconstruction
    )
    return all_points, frame


def _merge_geometric_clusters_into_topology_families(
    *,
    seed: TrackEvidenceBuild,
    original_settings: RouteVariantSettings,
    family_settings: RouteFamilySettings,
) -> _TopologyMerge:
    laps = seed.laps.copy()
    primitive_summary = seed.route_variant_summary.copy()
    pairwise = seed.route_variant_pairwise.copy()
    primitive_ids = [
        str(value)
        for value in primitive_summary.get(
            "route_variant_id", pd.Series(dtype=str)
        ).tolist()
        if str(value)
    ]
    if not primitive_ids:
        return _TopologyMerge(primitive_summary, pairwise, {})

    p95_limit = (
        family_settings.topology_same_route_p95_distance_m
        if family_settings.topology_same_route_p95_distance_m is not None
        else 1.5 * original_settings.same_variant_p95_distance_m
    )
    divergent_limit = (
        family_settings.topology_maximum_divergent_fraction
        if family_settings.topology_maximum_divergent_fraction is not None
        else original_settings.maximum_divergent_fraction
    )
    length_limit = (
        family_settings.topology_maximum_length_relative_difference
        if family_settings.topology_maximum_length_relative_difference is not None
        else original_settings.maximum_length_relative_difference
    )

    lap_to_primitive = laps.set_index("lap_id")["route_variant_id"].astype(str)
    aggregate = _primitive_pair_aggregate(
        pairwise, lap_to_primitive, p95_limit, divergent_limit, length_limit
    )
    score_lookup = {
        tuple(sorted((str(row.left_primitive_route_id), str(row.right_primitive_route_id)))): float(row.topology_merge_score)
        for row in aggregate.itertuples(index=False)
    }
    supported_mask = (
        primitive_summary["supported"].astype(bool)
        if "supported" in primitive_summary
        else pd.Series(False, index=primitive_summary.index)
    )
    supported_seed_ids = set(
        primitive_summary.loc[supported_mask, "route_variant_id"].astype(str)
    )
    if supported_seed_ids:
        cores = _connected_components(
            sorted(supported_seed_ids), score_lookup
        )
        unresolved = set(primitive_ids) - supported_seed_ids
        for primitive_id in sorted(unresolved):
            candidates: list[tuple[float, int]] = []
            for index, core in enumerate(cores):
                scores = [
                    score_lookup.get(tuple(sorted((primitive_id, member))), math.inf)
                    for member in core
                ]
                candidates.append((min(scores, default=math.inf), index))
            candidates.sort()
            if not candidates or candidates[0][0] > 1.0:
                continue
            if (
                len(candidates) > 1
                and candidates[1][0] <= 1.0
                and candidates[0][0]
                > family_settings.topology_attachment_ambiguity_ratio
                * candidates[1][0]
            ):
                continue
            cores[candidates[0][1]].add(primitive_id)
    else:
        cores = _connected_components(primitive_ids, score_lookup)

    assigned = set().union(*cores) if cores else set()
    unresolved_components = [{item} for item in primitive_ids if item not in assigned]
    component_rows: list[tuple[set[str], bool]] = []
    for component in cores:
        lap_count = int(
            laps["route_variant_id"].astype(str).isin(component).sum()
        )
        component_rows.append(
            (component, lap_count >= original_settings.minimum_supported_laps)
        )
    component_rows.extend((component, False) for component in unresolved_components)

    supported_components = [item for item in component_rows if item[1]]
    unsupported_components = [item for item in component_rows if not item[1]]
    supported_components.sort(
        key=lambda item: _component_sort_key(laps, item[0])
    )
    unsupported_components.sort(
        key=lambda item: _component_sort_key(laps, item[0])
    )

    primitive_to_topology: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for index, (component, _) in enumerate(supported_components, 1):
        route_id = f"route_{index:03d}"
        primitive_to_topology.update({item: route_id for item in component})
        records.append(
            _topology_summary_record(laps, component, route_id, supported=True)
        )
    for index, (component, _) in enumerate(unsupported_components, 1):
        route_id = f"route_outlier_{index:03d}"
        primitive_to_topology.update({item: route_id for item in component})
        records.append(
            _topology_summary_record(laps, component, route_id, supported=False)
        )

    output_pairwise = pairwise.copy()
    output_pairwise["left_geometric_route_variant_id"] = (
        output_pairwise["left_lap_id"].map(lap_to_primitive).fillna("")
    )
    output_pairwise["right_geometric_route_variant_id"] = (
        output_pairwise["right_lap_id"].map(lap_to_primitive).fillna("")
    )
    output_pairwise["left_topology_route_variant_id"] = output_pairwise[
        "left_geometric_route_variant_id"
    ].map(primitive_to_topology).fillna("")
    output_pairwise["right_topology_route_variant_id"] = output_pairwise[
        "right_geometric_route_variant_id"
    ].map(primitive_to_topology).fillna("")
    output_pairwise["same_topology_route_family"] = (
        output_pairwise["left_topology_route_variant_id"].astype(str)
        == output_pairwise["right_topology_route_variant_id"].astype(str)
    )
    output_pairwise["topology_same_route_p95_limit_m"] = p95_limit
    output_pairwise["topology_maximum_divergent_fraction"] = divergent_limit
    output_pairwise["topology_maximum_length_relative_difference"] = length_limit
    return _TopologyMerge(
        pd.DataFrame(records), output_pairwise, primitive_to_topology
    )


def _primitive_pair_aggregate(
    pairwise: pd.DataFrame,
    lap_to_primitive: pd.Series,
    p95_limit: float,
    divergent_limit: float,
    length_limit: float,
) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame(
            columns=[
                "left_primitive_route_id",
                "right_primitive_route_id",
                "topology_merge_score",
            ]
        )
    frame = pairwise.copy()
    frame["left_primitive_route_id"] = frame["left_lap_id"].map(
        lap_to_primitive
    )
    frame["right_primitive_route_id"] = frame["right_lap_id"].map(
        lap_to_primitive
    )
    frame = frame[
        frame["left_primitive_route_id"].notna()
        & frame["right_primitive_route_id"].notna()
        & (
            frame["left_primitive_route_id"].astype(str)
            != frame["right_primitive_route_id"].astype(str)
        )
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    ordered = frame.apply(
        lambda row: tuple(
            sorted(
                (
                    str(row["left_primitive_route_id"]),
                    str(row["right_primitive_route_id"]),
                )
            )
        ),
        axis=1,
    )
    frame["left_primitive_route_id"] = [item[0] for item in ordered]
    frame["right_primitive_route_id"] = [item[1] for item in ordered]
    aggregate = (
        frame.groupby(
            ["left_primitive_route_id", "right_primitive_route_id"],
            as_index=False,
        )
        .agg(
            lap_pair_count=("left_lap_id", "size"),
            median_symmetric_p95_nearest_path_distance_m=(
                "symmetric_p95_nearest_path_distance_m", "median"
            ),
            median_symmetric_divergent_fraction=(
                "symmetric_divergent_fraction", "median"
            ),
            median_length_relative_difference=(
                "length_relative_difference", "median"
            ),
        )
    )
    aggregate["topology_merge_score"] = aggregate.apply(
        lambda row: max(
            float(row["median_symmetric_p95_nearest_path_distance_m"])
            / max(p95_limit, 1.0e-12),
            float(row["median_symmetric_divergent_fraction"])
            / max(divergent_limit, 1.0e-12),
            float(row["median_length_relative_difference"])
            / max(length_limit, 1.0e-12),
        ),
        axis=1,
    )
    aggregate["topology_merge_compatible"] = (
        aggregate["topology_merge_score"] <= 1.0
    )
    return aggregate


def _connected_components(
    identifiers: list[str],
    score_lookup: Mapping[tuple[str, str], float],
) -> list[set[str]]:
    remaining = set(identifiers)
    components: list[set[str]] = []
    while remaining:
        start = min(remaining)
        component = {start}
        frontier = [start]
        remaining.remove(start)
        while frontier:
            current = frontier.pop()
            neighbours = [
                candidate
                for candidate in sorted(remaining)
                if score_lookup.get(
                    tuple(sorted((current, candidate))), math.inf
                )
                <= 1.0
            ]
            for candidate in neighbours:
                remaining.remove(candidate)
                component.add(candidate)
                frontier.append(candidate)
        components.append(component)
    return components


def _component_sort_key(
    laps: pd.DataFrame, component: set[str]
) -> tuple[float, int, int]:
    rows = laps[laps["route_variant_id"].astype(str).isin(component)]
    median = float(
        pd.to_numeric(rows["path_distance_m"], errors="coerce").median()
    )
    minimum_lap = int(rows["lap_id"].min()) if not rows.empty else 10**9
    return (-median, -len(rows), minimum_lap)


def _topology_summary_record(
    laps: pd.DataFrame,
    component: set[str],
    route_id: str,
    *,
    supported: bool,
) -> dict[str, Any]:
    rows = laps[laps["route_variant_id"].astype(str).isin(component)].copy()
    values = pd.to_numeric(rows["path_distance_m"], errors="coerce").dropna()
    lap_ids = sorted(rows["lap_id"].astype(int).tolist())
    return {
        "minimum_lap_id": min(lap_ids) if lap_ids else 0,
        "lap_count": len(lap_ids),
        "supported": supported,
        "run_count": int(rows["run_id"].astype(str).nunique()),
        "vehicle_count": int(rows["vehicle_id"].astype(str).nunique()),
        "driver_count": int(rows["driver_id"].astype(str).nunique()),
        "run_ids": ";".join(sorted(rows["run_id"].astype(str).unique())),
        "vehicle_ids": ";".join(sorted(rows["vehicle_id"].astype(str).unique())),
        "driver_ids": ";".join(sorted(rows["driver_id"].astype(str).unique())),
        "median_path_distance_m": float(values.median()),
        "p10_path_distance_m": float(values.quantile(0.10)),
        "p90_path_distance_m": float(values.quantile(0.90)),
        "route_variant_id": route_id,
        "lap_ids": ";".join(map(str, lap_ids)),
        "selected": False,
        "selection_reason": "",
        "topology_family": supported,
        "geometric_seed_cluster_count": len(component),
        "geometric_seed_route_variant_ids": ";".join(sorted(component)),
        "topology_component_status": (
            "supported_course_route" if supported else "isolated_or_ambiguous_outlier"
        ),
    }


def _select_topology_family(
    summary: pd.DataFrame,
    seed_laps: pd.DataFrame,
    settings: RouteVariantSettings,
) -> tuple[str, str]:
    supported = summary[summary["supported"].astype(bool)].copy()
    if supported.empty:
        raise ValueError("No supported topology route family is available.")
    if len(supported) == 1:
        return str(supported.iloc[0]["route_variant_id"]), "only_supported_topology_family"
    if settings.selection == "largest_supported":
        row = supported.sort_values(
            ["lap_count", "median_path_distance_m", "minimum_lap_id"],
            ascending=[False, False, True],
        ).iloc[0]
        return str(row["route_variant_id"]), "largest_supported_topology_family"
    if settings.selection == "longest_supported":
        row = supported.sort_values(
            ["median_path_distance_m", "lap_count", "minimum_lap_id"],
            ascending=[False, False, True],
        ).iloc[0]
        return str(row["route_variant_id"]), "longest_supported_topology_family"
    if settings.selection == "variant_id":
        requested = settings.selected_variant_id
        direct = supported[
            supported["route_variant_id"].astype(str) == requested
        ]
        if not direct.empty:
            return requested, "explicit_topology_family_id"
        primitive_match = supported[
            supported["geometric_seed_route_variant_ids"]
            .astype(str)
            .str.split(";")
            .map(lambda values: requested in values)
        ]
        if len(primitive_match) == 1:
            return (
                str(primitive_match.iloc[0]["route_variant_id"]),
                "explicit_geometric_seed_mapped_to_topology_family",
            )
        raise ValueError(
            f"Requested route variant {requested!r} is not a supported topology family."
        )
    if settings.selection == "reference_run":
        family_rows = []
        for row in supported.itertuples(index=False):
            run_ids = set(str(row.run_ids).split(";"))
            if settings.reference_run_id in run_ids:
                family_rows.append(row)
        if len(family_rows) == 1:
            return str(family_rows[0].route_variant_id), "reference_run_topology_family"
        raise ValueError(
            "reference_run selects multiple topology families; use selection='variant_id'."
        )
    raise ValueError(
        "Multiple supported topology route families were detected; set "
        "track.route_variants.selection to largest_supported, longest_supported, "
        "reference_run, or variant_id."
    )


def _build_topology_route(
    *,
    route_id: str,
    seed: TrackEvidenceBuild,
    topology: _TopologyMerge,
    all_points: pd.DataFrame,
    frame: LocalFrame,
    track_config: Mapping[str, Any],
    raw_events: list[Mapping[str, Any]],
    diagnostics: Any,
    original_variant_settings: RouteVariantSettings,
) -> TrackEvidenceBuild:
    reconstruction = ReconstructionSettings.from_mapping(track_config)
    laps = seed.laps.copy()
    if "quality_flags" in laps:
        laps["quality_flags"] = laps["quality_flags"].map(
            _strip_map_only_quality_flags
        )
    laps["geometric_route_variant_id"] = laps["route_variant_id"].astype(str)
    laps["route_variant_id"] = laps["geometric_route_variant_id"].map(
        topology.primitive_to_topology
    ).fillna("")
    support_lookup = topology.summary.set_index("route_variant_id")[
        "supported"
    ].astype(bool)
    median_lookup = topology.summary.set_index("route_variant_id")[
        "median_path_distance_m"
    ].astype(float)
    laps["route_variant_supported"] = laps["route_variant_id"].map(
        support_lookup
    ).fillna(False).astype(bool)
    laps["route_variant_selected"] = (
        laps["route_variant_id"].astype(str) == route_id
    )
    medians = laps["route_variant_id"].map(median_lookup)
    laps["route_variant_distance_ratio"] = (
        pd.to_numeric(laps["path_distance_m"], errors="coerce")
        / medians.replace(0.0, np.nan)
    )
    data_quality = laps["data_quality_valid"].astype(bool)
    within = (
        (laps["route_variant_distance_ratio"] - 1.0).abs()
        <= original_variant_settings.maximum_within_variant_length_deviation_fraction
    ).fillna(False)
    selected = laps["route_variant_selected"].astype(bool)
    supported = laps["route_variant_supported"].astype(bool)
    laps["route_variant_status"] = np.select(
        [
            ~data_quality,
            data_quality & ~supported,
            data_quality & supported & ~selected,
            data_quality & supported & selected & ~within,
            data_quality & supported & selected & within,
        ],
        [
            "data_quality_rejected",
            "unsupported_topology_component",
            "alternate_supported_route",
            "selected_route_length_outlier",
            "selected_supported_route",
        ],
        default="unclassifiable_geometry",
    )
    laps["analysis_exclusion_reason"] = np.select(
        [
            ~data_quality,
            data_quality & ~supported,
            data_quality & supported & ~selected,
            data_quality & supported & selected & ~within,
        ],
        [
            "data_quality_rejected",
            "unsupported_route_variant",
            "alternate_supported_route_variant",
            "within_variant_distance_outlier",
        ],
        default="",
    )
    laps["pre_consensus_valid"] = (
        data_quality & selected & supported & within
    )
    laps["analysis_valid"] = laps["pre_consensus_valid"].astype(bool)
    laps["centreline_included"] = (
        laps["pre_consensus_valid"].astype(bool)
        & laps["use_for_centreline"].astype(bool)
    )
    laps["consensus_excluded"] = False
    laps["consensus_iteration_excluded"] = np.nan
    laps["consensus_exclusion_reason"] = ""
    laps["reference_lap"] = False

    consensus = build_iterative_consensus(
        all_points,
        laps,
        frame,
        reconstruction,
        track_config,
        diagnostics,
    )
    final_laps = finalize_route_variant_semantics(consensus.laps)
    if not final_laps["analysis_valid"].any():
        raise ValueError(
            f"No valid map-matched laps remain for topology route {route_id}."
        )
    centreline = consensus.centreline
    track_profile = build_track_profile(
        consensus.matched_points, final_laps, centreline, reconstruction
    )
    events = normalize_events(raw_events)
    event_projection = project_events(events, centreline, reconstruction)
    response_features = build_response_features(
        event_projection, centreline.length_m, reconstruction
    )
    event_passes = extract_event_passes(
        consensus.matched_points,
        final_laps,
        response_features,
        centreline,
        reconstruction,
    )
    gate_evidence = score_speed_gates(
        event_passes, response_features, reconstruction, diagnostics
    )
    gate_review = build_gate_review(
        gate_evidence, response_features, reconstruction
    )
    return TrackEvidenceBuild(
        centreline=centreline,
        laps=final_laps,
        matched_points=consensus.matched_points,
        track_profile=track_profile,
        event_projection=event_projection,
        response_features=response_features,
        event_passes=event_passes,
        gate_evidence=gate_evidence,
        gate_review=gate_review,
        rejected_map_points=consensus.rejected_map_points,
        route_variant_summary=topology.summary,
        route_variant_pairwise=topology.pairwise,
        route_variant_settings=original_variant_settings,
        selected_route_variant_id=route_id,
    )


def _strip_map_only_quality_flags(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    tokens = [item.strip() for item in str(value).split(";") if item.strip()]
    return ";".join(
        item for item in tokens if item not in _MAP_ONLY_QUALITY_FLAGS
    )


def _supported_variant_ids(summary: pd.DataFrame) -> list[str]:
    if summary.empty or "route_variant_id" not in summary:
        return []
    mask = (
        summary["supported"].astype(bool)
        if "supported" in summary
        else pd.Series(True, index=summary.index)
    )
    return [
        str(value)
        for value in summary.loc[mask, "route_variant_id"].tolist()
        if str(value)
    ]



def _shared_course_projection_error_limit(
    *,
    strict_error_m: float,
    branch_summary: pd.DataFrame,
) -> float:
    """Return a shared-course tolerance that remains below a genuine branch split.

    The strict route/event projection limit is intentionally conservative.  Once a
    sustained branch corridor has been resolved, ordinary shared-course line choice
    can be somewhat wider without being confused with that branch.  The relaxed
    limit is capped at 60% of the detected divergence threshold and at 1.5x the
    strict limit, while never becoming stricter than the original limit.
    """

    strict = float(strict_error_m)
    if branch_summary is None or branch_summary.empty:
        return strict
    thresholds = pd.to_numeric(
        branch_summary.get(
            "divergence_distance_threshold_m", pd.Series(dtype=float)
        ),
        errors="coerce",
    ).dropna()
    if thresholds.empty:
        return strict
    branch_threshold = float(thresholds.min())
    relaxed = min(1.5 * strict, 0.60 * branch_threshold)
    return float(max(strict, relaxed))


def _branch_interval_lookup(
    branch_summary: pd.DataFrame,
) -> dict[str, list[tuple[str, float, float]]]:
    lookup: dict[str, list[tuple[str, float, float]]] = {}
    if branch_summary is None or branch_summary.empty:
        return lookup
    for row in branch_summary.itertuples(index=False):
        branch_id = str(getattr(row, "branch_id"))
        nominal_id = str(getattr(row, "nominal_route_variant_id"))
        alternate_id = str(getattr(row, "alternate_route_variant_id"))
        lookup.setdefault(nominal_id, []).append(
            (
                branch_id,
                float(getattr(row, "nominal_branch_start_s_m")),
                float(getattr(row, "nominal_branch_end_s_m")),
            )
        )
        lookup.setdefault(alternate_id, []).append(
            (
                branch_id,
                float(getattr(row, "alternate_branch_start_s_m")),
                float(getattr(row, "alternate_branch_end_s_m")),
            )
        )
    return lookup


def _declared_event_branch_membership(
    *,
    event: Any,
    routes: Mapping[str, TrackEvidenceBuild],
    branch_summary: pd.DataFrame,
    shared_maximum_error_m: float,
) -> dict[str, set[str]]:
    """Map a declared event anchor to genuine branch corridors, if any.

    Classification uses the declared physical anchor rather than ordered event
    projection.  A point is considered branch-local only when it is both close to
    a route and falls inside that route's sustained divergence corridor.
    """

    role = str(getattr(event, "analysis_role", "feature"))
    if role == "lap_gate":
        return {}
    intervals = _branch_interval_lookup(branch_summary)
    memberships: dict[str, set[str]] = {}
    for route_id, route in routes.items():
        s_m, error_m = _declared_coordinate_projection(
            route.centreline,
            getattr(event, "anchor_latitude_deg", math.nan),
            getattr(event, "anchor_longitude_deg", math.nan),
        )
        if not np.isfinite(s_m) or not np.isfinite(error_m):
            continue
        if error_m > shared_maximum_error_m:
            continue
        for branch_id, start_s, end_s in intervals.get(str(route_id), []):
            if _s_in_circular_interval(
                s_m, start_s, end_s, route.centreline.length_m
            ):
                memberships.setdefault(branch_id, set()).add(str(route_id))
    return memberships


def _restrict_route_events(
    *,
    routes: Mapping[str, TrackEvidenceBuild],
    normalized_events: pd.DataFrame,
    reconstruction: ReconstructionSettings,
    maximum_error_m: float,
    shared_maximum_error_m: float,
    branch_summary: pd.DataFrame,
    diagnostics: Any,
) -> tuple[dict[str, TrackEvidenceBuild], pd.DataFrame]:
    """Remove physical events that do not exist on a route's geometry.

    Applicability is evaluated from the original declared coordinates before the
    ordered projection is solved.  Once a sustained divergence/re-merge corridor
    has been identified, events outside that corridor are treated as shared-course
    objects and receive a modestly relaxed projection tolerance.  Branch-local
    events retain the strict route-specific tolerance and are only placed on the
    branch(es) their declared anchor physically occupies.
    """

    output: dict[str, TrackEvidenceBuild] = {}
    audit_rows: list[dict[str, Any]] = []
    branch_membership_by_event: dict[str, dict[str, set[str]]] = {}
    for event in normalized_events.itertuples(index=False):
        event_id = str(getattr(event, "id"))
        branch_membership_by_event[event_id] = _declared_event_branch_membership(
            event=event,
            routes=routes,
            branch_summary=branch_summary,
            shared_maximum_error_m=shared_maximum_error_m,
        )

    for route_id, route in routes.items():
        applicable_mask: list[bool] = []
        for event in normalized_events.itertuples(index=False):
            event_id = str(getattr(event, "id"))
            role = str(getattr(event, "analysis_role", "feature"))
            branch_memberships = branch_membership_by_event.get(event_id, {})
            branch_ids = sorted(branch_memberships)
            route_section_kind = "branch" if branch_ids else "shared"
            allowed_error = (
                maximum_error_m
                if route_section_kind == "branch"
                else shared_maximum_error_m
            )
            anchor_s, anchor_error = _declared_coordinate_projection(
                route.centreline,
                getattr(event, "anchor_latitude_deg", math.nan),
                getattr(event, "anchor_longitude_deg", math.nan),
            )
            start_error = _optional_endpoint_error(
                route.centreline, event, "start", fallback=anchor_error
            )
            end_error = _optional_endpoint_error(
                route.centreline, event, "end", fallback=anchor_error
            )
            branch_route_allowed = (
                not branch_ids
                or any(
                    str(route_id) in branch_memberships.get(branch_id, set())
                    for branch_id in branch_ids
                )
            )
            applicable, reasons = _event_is_route_applicable(
                role=role,
                anchor_error_m=anchor_error,
                start_error_m=start_error,
                end_error_m=end_error,
                maximum_error_m=allowed_error,
                branch_route_allowed=branch_route_allowed,
            )
            notes: list[str] = []
            if (
                applicable
                and route_section_kind == "shared"
                and allowed_error > maximum_error_m
                and max(anchor_error, start_error, end_error) > maximum_error_m
            ):
                notes.append("shared_course_relaxed_projection_tolerance")
            applicable_mask.append(applicable)
            audit_rows.append(
                {
                    "route_variant_id": route_id,
                    "event_id": event_id,
                    "event_name": str(getattr(event, "name")),
                    "analysis_role": role,
                    "declared_route_section_kind": route_section_kind,
                    "declared_branch_ids": ";".join(branch_ids),
                    "route_applicable": applicable,
                    "route_applicability_reason": ";".join(reasons),
                    "route_applicability_note": ";".join(notes),
                    "strict_projection_error_limit_m": maximum_error_m,
                    "shared_course_projection_error_limit_m": shared_maximum_error_m,
                    "applied_projection_error_limit_m": allowed_error,
                    "declared_anchor_nearest_route_s_m": anchor_s,
                    "declared_anchor_nearest_route_error_m": anchor_error,
                    "declared_feature_start_nearest_route_error_m": start_error,
                    "declared_feature_end_nearest_route_error_m": end_error,
                }
            )

        applicable_events = normalized_events.loc[applicable_mask].copy()
        if applicable_events.empty:
            raise ValueError(
                f"Route {route_id!r} has no applicable events, including no lap gate."
            )
        event_projection = project_events(
            applicable_events, route.centreline, reconstruction
        )
        response_features = build_response_features(
            event_projection, route.centreline.length_m, reconstruction
        )
        event_passes = extract_event_passes(
            route.matched_points,
            route.laps,
            response_features,
            route.centreline,
            reconstruction,
        )
        gate_evidence = score_speed_gates(
            event_passes, response_features, reconstruction, diagnostics
        )
        gate_review = build_gate_review(
            gate_evidence, response_features, reconstruction
        )
        output[route_id] = replace(
            route,
            event_projection=event_projection,
            response_features=response_features,
            event_passes=event_passes,
            gate_evidence=gate_evidence,
            gate_review=gate_review,
        )
    return output, pd.DataFrame(audit_rows)


def _declared_coordinate_projection(
    centreline: Any, latitude_deg: Any, longitude_deg: Any
) -> tuple[float, float]:
    latitude = _finite_float(latitude_deg)
    longitude = _finite_float(longitude_deg)
    if not np.isfinite(latitude) or not np.isfinite(longitude):
        return math.nan, math.nan
    x, y = centreline.frame.to_xy([latitude], [longitude])
    s_values, distances, _, _ = centreline.all_projections(
        float(x[0]), float(y[0])
    )
    if not len(distances):
        return math.nan, math.nan
    best = int(np.argmin(distances))
    return float(s_values[best]), float(distances[best])


def _declared_coordinate_error(
    centreline: Any, latitude_deg: Any, longitude_deg: Any
) -> float:
    return _declared_coordinate_projection(
        centreline, latitude_deg, longitude_deg
    )[1]


def _optional_endpoint_error(
    centreline: Any, event: Any, endpoint: str, *, fallback: float
) -> float:
    latitude = _finite_float(getattr(event, f"{endpoint}_latitude_deg", math.nan))
    longitude = _finite_float(getattr(event, f"{endpoint}_longitude_deg", math.nan))
    if not np.isfinite(latitude) or not np.isfinite(longitude):
        return fallback
    return _declared_coordinate_error(centreline, latitude, longitude)


def _event_is_route_applicable(
    *,
    role: str,
    anchor_error_m: float,
    start_error_m: float,
    end_error_m: float,
    maximum_error_m: float,
    branch_route_allowed: bool = True,
) -> tuple[bool, list[str]]:
    # Every route shares the declared lap gate by construction.  Other physical
    # events require their anchor and resolved endpoints to lie on this route. A
    # branch-local event must also belong to this specific branch traversal.
    if role == "lap_gate":
        return True, []
    reasons: list[str] = []
    if not branch_route_allowed:
        reasons.append("branch_event_not_on_route")
    if not np.isfinite(anchor_error_m) or anchor_error_m > maximum_error_m:
        reasons.append("anchor_not_on_route")
    if not np.isfinite(start_error_m) or start_error_m > maximum_error_m:
        reasons.append("feature_start_not_on_route")
    if not np.isfinite(end_error_m) or end_error_m > maximum_error_m:
        reasons.append("feature_end_not_on_route")
    return not reasons, reasons


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if np.isfinite(number) else math.nan



def _detect_route_branch_network(
    *,
    routes: Mapping[str, TrackEvidenceBuild],
    nominal_id: str,
    family_settings: RouteFamilySettings,
    variant_settings: RouteVariantSettings,
    diagnostics: Any,
    classify_events: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Represent alternate routes as local branches of one shared course.

    Whole-lap topology families remain useful for classifying traversals, but the
    physical course is a graph: most of the lap is shared and only a sustained
    divergence corridor is branch-specific.  This routine identifies that corridor
    against the nominal backbone and classifies every projected event as shared or
    branch-local.
    """

    if nominal_id not in routes or len(routes) <= 1:
        return pd.DataFrame(), pd.DataFrame()

    divergence_threshold = (
        family_settings.branch_divergence_distance_m
        if family_settings.branch_divergence_distance_m is not None
        else max(25.0, 2.5 * variant_settings.same_variant_p95_distance_m)
    )
    nominal = routes[nominal_id]
    total_valid = sum(
        int(route.laps["analysis_valid"].fillna(False).astype(bool).sum())
        for route in routes.values()
    )
    interval_lookup: dict[str, list[tuple[str, float, float]]] = {
        route_id: [] for route_id in routes
    }
    branch_rows: list[dict[str, Any]] = []

    alternate_ids = [route_id for route_id in routes if route_id != nominal_id]
    for branch_index, alternate_id in enumerate(alternate_ids, 1):
        alternate = routes[alternate_id]
        nominal_interval = _sustained_divergence_interval(
            nominal.centreline,
            alternate.centreline,
            divergence_threshold_m=divergence_threshold,
            minimum_length_m=family_settings.branch_minimum_sustained_length_m,
            gap_tolerance_m=family_settings.branch_gap_tolerance_m,
            padding_m=family_settings.branch_boundary_padding_m,
        )
        alternate_interval = _sustained_divergence_interval(
            alternate.centreline,
            nominal.centreline,
            divergence_threshold_m=divergence_threshold,
            minimum_length_m=family_settings.branch_minimum_sustained_length_m,
            gap_tolerance_m=family_settings.branch_gap_tolerance_m,
            padding_m=family_settings.branch_boundary_padding_m,
        )
        if nominal_interval is None or alternate_interval is None:
            diagnostics.warning(
                "ROUTE_BRANCH_CORRIDOR_NOT_RESOLVED",
                (
                    f"Could not resolve one sustained divergence/re-merge corridor "
                    f"between {nominal_id} and {alternate_id}; whole-route local "
                    "compatibility remains available for gate evidence."
                ),
            )
            continue

        branch_id = f"branch_{branch_index:03d}"
        interval_lookup[nominal_id].append(
            (
                branch_id,
                float(nominal_interval["start_s_m"]),
                float(nominal_interval["end_s_m"]),
            )
        )
        interval_lookup[alternate_id].append(
            (
                branch_id,
                float(alternate_interval["start_s_m"]),
                float(alternate_interval["end_s_m"]),
            )
        )
        nominal_valid = int(
            nominal.laps["analysis_valid"].fillna(False).astype(bool).sum()
        )
        alternate_valid = int(
            alternate.laps["analysis_valid"].fillna(False).astype(bool).sum()
        )
        branch_rows.append(
            {
                "branch_id": branch_id,
                "nominal_route_variant_id": nominal_id,
                "alternate_route_variant_id": alternate_id,
                "divergence_distance_threshold_m": divergence_threshold,
                "shared_section_valid_lap_count": total_valid,
                "nominal_branch_valid_geometry_lap_count": nominal_valid,
                "alternate_branch_valid_geometry_lap_count": alternate_valid,
                "nominal_branch_start_s_m": nominal_interval["start_s_m"],
                "nominal_branch_end_s_m": nominal_interval["end_s_m"],
                "nominal_branch_length_m": nominal_interval["length_m"],
                "alternate_branch_start_s_m": alternate_interval["start_s_m"],
                "alternate_branch_end_s_m": alternate_interval["end_s_m"],
                "alternate_branch_length_m": alternate_interval["length_m"],
                "nominal_maximum_separation_m": nominal_interval["maximum_distance_m"],
                "alternate_maximum_separation_m": alternate_interval["maximum_distance_m"],
                "nominal_median_separation_in_branch_m": nominal_interval["median_distance_m"],
                "alternate_median_separation_in_branch_m": alternate_interval["median_distance_m"],
                "network_interpretation": (
                    "one shared closed course with alternate paths only inside this "
                    "divergence/re-merge corridor"
                ),
            }
        )

    branch_frame = pd.DataFrame(branch_rows)
    event_frame = (
        _classify_route_event_sections(routes, branch_frame)
        if classify_events and not branch_frame.empty
        else pd.DataFrame()
    )

    if branch_rows:
        details = ", ".join(
            (
                f"{row['branch_id']} {row['nominal_route_variant_id']}↔"
                f"{row['alternate_route_variant_id']} "
                f"({row['shared_section_valid_lap_count']} shared-section "
                "valid laps)"
            )
            for row in branch_rows
        )
        diagnostics.info(
            "ROUTE_BRANCH_NETWORK_DETECTED",
            (
                "Interpreted supported route families as one shared course with "
                f"{len(branch_rows)} sustained divergence/re-merge branch corridor(s): "
                f"{details}."
            ),
            hint=(
                "Shared-section events pool evidence across every supported branch. "
                "Only events inside a divergence corridor are branch-specific."
            ),
        )
    return branch_frame, event_frame


def _classify_route_event_sections(
    routes: Mapping[str, TrackEvidenceBuild],
    branch_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Classify projected response features on the shared course or a branch.

    The physical feature extent, rather than the entire approach/exit analysis
    window, determines the course section.  This keeps an obstacle immediately
    before a divergence shared while allowing its later exit/recovery evidence to
    become less transferable when the traversals begin to separate.
    """

    intervals = _branch_interval_lookup(branch_summary)
    rows: list[dict[str, Any]] = []
    for route_id, route in routes.items():
        route_intervals = intervals.get(str(route_id), [])
        length = float(route.centreline.length_m)
        for event in route.response_features.itertuples(index=False):
            anchor = _finite_float(getattr(event, "anchor_s_m", math.nan))
            feature_start = (
                (anchor + _finite_float(getattr(event, "feature_start_rel_m", 0.0)))
                % length
                if np.isfinite(anchor)
                else math.nan
            )
            feature_end = (
                (anchor + _finite_float(getattr(event, "feature_end_rel_m", 0.0)))
                % length
                if np.isfinite(anchor)
                else math.nan
            )
            branch_ids = [
                branch_id
                for branch_id, start_s, end_s in route_intervals
                if np.isfinite(feature_start)
                and np.isfinite(feature_end)
                and _circular_intervals_overlap(
                    feature_start,
                    feature_end,
                    start_s,
                    end_s,
                    length,
                )
            ]
            rows.append(
                {
                    "route_variant_id": str(route_id),
                    "event_id": str(getattr(event, "id")),
                    "event_name": str(getattr(event, "name")),
                    "anchor_s_m": anchor,
                    "feature_start_s_m": feature_start,
                    "feature_end_s_m": feature_end,
                    "route_section_kind": "branch" if branch_ids else "shared",
                    "branch_ids": ";".join(sorted(set(branch_ids))),
                    "shared_course_evidence_expected": not branch_ids,
                    "branch_specific_evidence_expected": bool(branch_ids),
                }
            )
    return pd.DataFrame(rows)


def _circular_intervals_overlap(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
    length: float,
) -> bool:
    def pieces(start: float, end: float) -> list[tuple[float, float]]:
        start = float(start) % float(length)
        end = float(end) % float(length)
        if start <= end:
            return [(start, end)]
        return [(start, float(length)), (0.0, end)]

    return any(
        max(a0, b0) <= min(a1, b1)
        for a0, a1 in pieces(left_start, left_end)
        for b0, b1 in pieces(right_start, right_end)
    )


def _sustained_divergence_interval(
    left: Any,
    right: Any,
    *,
    divergence_threshold_m: float,
    minimum_length_m: float,
    gap_tolerance_m: float,
    padding_m: float,
) -> dict[str, float] | None:
    # Ignore the duplicated closed-course endpoint for circular run detection.
    left_xy = np.column_stack(
        (np.asarray(left.x_m[:-1], float), np.asarray(left.y_m[:-1], float))
    )
    right_xy = np.column_stack(
        (np.asarray(right.x_m[:-1], float), np.asarray(right.y_m[:-1], float))
    )
    s = np.asarray(left.s_m[:-1], float)
    if len(left_xy) < 4 or len(right_xy) < 4 or len(s) != len(left_xy):
        return None
    distances = cKDTree(right_xy).query(left_xy, k=1)[0]
    spacing = float(np.median(np.diff(np.asarray(left.s_m, float))))
    spacing = spacing if np.isfinite(spacing) and spacing > 0.0 else 1.0
    mask = distances > float(divergence_threshold_m)
    mask = _close_short_circular_gaps(
        mask, int(math.ceil(gap_tolerance_m / spacing))
    )
    mask = _remove_short_circular_runs(
        mask, int(math.ceil(minimum_length_m / spacing))
    )
    runs = _circular_true_runs(mask)
    if not runs:
        return None
    run = max(runs, key=len)
    pad_nodes = int(math.ceil(padding_m / spacing))
    start_index = (run[0] - pad_nodes) % len(mask)
    end_index = (run[-1] + pad_nodes) % len(mask)
    start_s = float(s[start_index])
    end_s = float(s[end_index])
    length = float((end_s - start_s) % left.length_m)
    if length <= 0.0:
        length = float(len(run) * spacing + 2.0 * padding_m)
    run_distances = distances[np.asarray(run, dtype=int)]
    return {
        "start_s_m": start_s,
        "end_s_m": end_s,
        "length_m": min(length, float(left.length_m)),
        "maximum_distance_m": float(np.max(run_distances)),
        "median_distance_m": float(np.median(run_distances)),
    }


def _circular_true_runs(mask: np.ndarray) -> list[list[int]]:
    values = np.asarray(mask, dtype=bool)
    n = len(values)
    if n == 0 or not values.any():
        return []
    if values.all():
        return [list(range(n))]
    false_index = int(np.flatnonzero(~values)[0])
    order = [(false_index + 1 + offset) % n for offset in range(n)]
    runs: list[list[int]] = []
    current: list[int] = []
    for index in order:
        if values[index]:
            current.append(index)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _close_short_circular_gaps(mask: np.ndarray, maximum_gap_nodes: int) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    if maximum_gap_nodes <= 0 or not output.any() or output.all():
        return output
    for run in _circular_true_runs(~output):
        if len(run) <= maximum_gap_nodes:
            output[np.asarray(run, dtype=int)] = True
    return output


def _remove_short_circular_runs(mask: np.ndarray, minimum_run_nodes: int) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    if minimum_run_nodes <= 1:
        return output
    for run in _circular_true_runs(output):
        if len(run) < minimum_run_nodes:
            output[np.asarray(run, dtype=int)] = False
    return output


def _s_in_circular_interval(
    value: float, start: float, end: float, length: float
) -> bool:
    value = float(value) % float(length)
    start = float(start) % float(length)
    end = float(end) % float(length)
    if start <= end:
        return start <= value <= end
    return value >= start or value <= end

def _pool_gate_evidence(
    *,
    ingestion_results: tuple[Any, ...],
    track_config: Mapping[str, Any],
    routes: Mapping[str, TrackEvidenceBuild],
    source_laps: pd.DataFrame,
    settings: RouteFamilySettings,
    maximum_error_m: float,
    shared_maximum_error_m: float,
    diagnostics: Any,
    event_route_sections: pd.DataFrame,
) -> dict[str, TrackEvidenceBuild]:
    """Pool local gate evidence while keeping branch-only evidence route-specific.

    A shared physical section has one evidence contract, not one contract per
    complete traversal.  We therefore map every supported lap against every
    route, choose one locally best compatible pass for each shared
    ``(event_id, lap_id)`` pair, score that shared pass set once, and reuse the
    resulting confidence/target distribution on every route.  Only events whose
    analysis windows overlap a genuine divergence corridor remain branch-specific.
    """

    reconstruction = ReconstructionSettings.from_mapping(track_config)
    all_points = _combine_points(ingestion_results)
    frame = LocalFrame(
        float(all_points["latitude_deg"].median()),
        float(all_points["longitude_deg"].median()),
    )
    x_m, y_m = frame.to_xy(
        all_points["latitude_deg"], all_points["longitude_deg"]
    )
    all_points["x_m"] = x_m
    all_points["y_m"] = y_m
    all_points["speed_analysis_mps"] = _clean_speed(all_points, reconstruction)

    candidate_laps = source_laps.copy()
    data_quality = (
        candidate_laps["data_quality_valid"].astype(bool)
        if "data_quality_valid" in candidate_laps
        else candidate_laps["analysis_valid"].astype(bool)
    )
    supported_source = (
        candidate_laps["route_variant_supported"].fillna(False).astype(bool)
        if "route_variant_supported" in candidate_laps
        else pd.Series(True, index=candidate_laps.index)
    )
    candidate_laps["analysis_valid"] = data_quality & supported_source
    candidate_laps["reference_lap"] = False

    prepared: dict[str, dict[str, Any]] = {}
    for route_id, route in routes.items():
        mapped = _map_all_candidate_laps(
            all_points,
            candidate_laps,
            route.centreline,
            route_id=route_id,
            preferred_route_matched=route.matched_points,
        )
        passes = extract_event_passes(
            mapped,
            candidate_laps,
            route.response_features,
            route.centreline,
            reconstruction,
        )
        section_lookup: dict[str, str] = {}
        branch_lookup: dict[str, str] = {}
        if event_route_sections is not None and not event_route_sections.empty:
            selected_sections = event_route_sections[
                event_route_sections["route_variant_id"].astype(str)
                == str(route_id)
            ]
            section_lookup = selected_sections.set_index("event_id")[
                "route_section_kind"
            ].astype(str).to_dict()
            branch_lookup = selected_sections.set_index("event_id")[
                "branch_ids"
            ].astype(str).to_dict()
        compatibility = _event_compatibility(
            mapped=mapped,
            laps=candidate_laps,
            events=route.response_features,
            track_length_m=route.centreline.length_m,
            maximum_error_m=maximum_error_m,
            shared_maximum_error_m=shared_maximum_error_m,
            event_section_lookup=section_lookup,
            minimum_coverage=settings.minimum_event_window_coverage_fraction,
            bin_size_m=settings.event_projection_bin_size_m,
            maximum_interpolation_gap_m=settings.maximum_event_interpolation_gap_m,
            maximum_backward=settings.maximum_local_backward_matches,
        )
        passes = passes.merge(
            compatibility,
            on=["event_id", "lap_id"],
            how="left",
            validate="one_to_one",
        )
        passes["route_compatible"] = (
            passes["route_compatible"].fillna(False).astype(bool)
        )
        passes["eligible_before_route_compatibility"] = passes["eligible"].astype(
            bool
        )
        passes["target_route_variant_id"] = route_id
        source_variant = candidate_laps.set_index("lap_id").get(
            "route_variant_id", pd.Series(dtype=str)
        )
        passes["source_route_variant_id"] = passes["lap_id"].map(
            source_variant
        ).fillna("")
        passes["route_section_kind"] = passes["event_id"].astype(str).map(
            section_lookup
        ).fillna("shared")
        passes["branch_ids"] = passes["event_id"].astype(str).map(
            branch_lookup
        ).fillna("")
        branch_local = passes["route_section_kind"].astype(str) == "branch"
        source_on_target_branch = (
            passes["source_route_variant_id"].astype(str) == str(route_id)
        )
        passes["eligible"] = (
            passes["eligible"].astype(bool)
            & passes["route_compatible"]
            & (~branch_local | source_on_target_branch)
        )
        passes["evidence_scope"] = np.where(
            branch_local, "branch_specific", "shared_course"
        )
        passes["shared_evidence_contract"] = False
        passes["projection_source_route_variant_id"] = route_id
        prepared[route_id] = {
            "route": route,
            "passes": passes,
            "section_lookup": section_lookup,
        }

    shared_event_ids = _shared_contract_event_ids(
        event_route_sections, route_ids=tuple(routes)
    )
    shared_passes = _deduplicated_shared_contract_passes(
        [item["passes"] for item in prepared.values()],
        shared_event_ids=shared_event_ids,
    )
    canonical_shared_events = _canonical_shared_event_rows(
        routes, shared_event_ids=shared_event_ids
    )
    shared_evidence = (
        score_speed_gates(
            shared_passes,
            canonical_shared_events,
            reconstruction,
            diagnostics,
        )
        if not shared_passes.empty and not canonical_shared_events.empty
        else pd.DataFrame()
    )
    if shared_event_ids:
        diagnostics.info(
            "ROUTE_SHARED_GATE_CONTRACTS",
            (
                f"Built {len(shared_event_ids)} shared-course event contract(s) "
                "once from deduplicated locally compatible laps and reused them "
                "across every supported traversal."
            ),
            hint=(
                "Only events overlapping a sustained divergence/re-merge corridor "
                "retain route-specific gate evidence."
            ),
        )

    output: dict[str, TrackEvidenceBuild] = {}
    for route_id, item in prepared.items():
        route = item["route"]
        passes = item["passes"]
        if shared_event_ids and not shared_passes.empty:
            branch_rows = passes[
                ~passes["event_id"].astype(str).isin(shared_event_ids)
            ].copy()
            shared_rows = shared_passes.copy()
            shared_rows["target_route_variant_id"] = route_id
            shared_rows["shared_evidence_contract"] = True
            passes = pd.concat(
                [branch_rows, shared_rows], ignore_index=True, sort=False
            )

        local_evidence = score_speed_gates(
            passes,
            route.response_features,
            reconstruction,
            diagnostics,
        )
        if not shared_evidence.empty:
            local_evidence = pd.concat(
                [
                    local_evidence[
                        ~local_evidence["event_id"].astype(str).isin(
                            shared_event_ids
                        )
                    ],
                    shared_evidence,
                ],
                ignore_index=True,
                sort=False,
            ).sort_values("sequence").reset_index(drop=True)
        review = build_gate_review(
            local_evidence,
            route.response_features,
            reconstruction,
        )
        output[route_id] = replace(
            route,
            event_passes=passes,
            gate_evidence=local_evidence,
            gate_review=review,
        )
    return output


def _shared_contract_event_ids(
    event_route_sections: pd.DataFrame,
    *,
    route_ids: tuple[str, ...],
) -> set[str]:
    """Events present on every route and outside every branch corridor."""

    if event_route_sections is None or event_route_sections.empty or not route_ids:
        return set()
    expected = {str(value) for value in route_ids}
    shared: set[str] = set()
    for event_id, group in event_route_sections.groupby("event_id", sort=False):
        present = set(group["route_variant_id"].astype(str))
        section = set(group["route_section_kind"].astype(str))
        if present == expected and section == {"shared"}:
            shared.add(str(event_id))
    return shared


def _deduplicated_shared_contract_passes(
    route_pass_frames: list[pd.DataFrame],
    *,
    shared_event_ids: set[str],
) -> pd.DataFrame:
    """Choose one locally best pass per lap for each shared physical event."""

    frames = [frame for frame in route_pass_frames if frame is not None and not frame.empty]
    if not frames or not shared_event_ids:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[
        combined["event_id"].astype(str).isin(shared_event_ids)
    ].copy()
    if combined.empty:
        return combined
    combined["_compatible_rank"] = combined.get(
        "route_compatible", pd.Series(False, index=combined.index)
    ).fillna(False).astype(bool).astype(int)
    combined["_pre_eligible_rank"] = combined.get(
        "eligible_before_route_compatibility",
        combined.get("eligible", pd.Series(False, index=combined.index)),
    ).fillna(False).astype(bool).astype(int)
    combined["_projection_error_rank"] = pd.to_numeric(
        combined.get(
            "route_projection_p95_error_m",
            pd.Series(math.inf, index=combined.index),
        ),
        errors="coerce",
    ).fillna(math.inf)
    combined["_own_route_rank"] = (
        combined.get("source_route_variant_id", pd.Series("", index=combined.index))
        .astype(str)
        .eq(
            combined.get(
                "target_route_variant_id", pd.Series("", index=combined.index)
            ).astype(str)
        )
        .astype(int)
    )
    combined = combined.sort_values(
        [
            "event_id",
            "lap_id",
            "_pre_eligible_rank",
            "_compatible_rank",
            "_projection_error_rank",
            "_own_route_rank",
            "target_route_variant_id",
        ],
        ascending=[True, True, False, False, True, False, True],
    )
    selected = combined.drop_duplicates(
        subset=["event_id", "lap_id"], keep="first"
    ).copy()
    selected["projection_source_route_variant_id"] = selected[
        "target_route_variant_id"
    ].astype(str)
    selected["eligible"] = (
        selected["eligible_before_route_compatibility"].astype(bool)
        & selected["route_compatible"].astype(bool)
    )
    selected["evidence_scope"] = "shared_course"
    selected["shared_evidence_contract"] = True
    return selected.drop(
        columns=[
            "_compatible_rank",
            "_pre_eligible_rank",
            "_projection_error_rank",
            "_own_route_rank",
        ]
    ).reset_index(drop=True)


def _canonical_shared_event_rows(
    routes: Mapping[str, TrackEvidenceBuild],
    *,
    shared_event_ids: set[str],
) -> pd.DataFrame:
    """Select one geometry row for scoring each shared event contract.

    Route-centreline offset is a driving-line property on shared course sections,
    not uncertainty about the physical event itself.  The locally best projected
    representation therefore supplies the coordinate-quality component while the
    resulting evidence statistics are reused on every route.  Route-specific
    ``s`` coordinates are restored later by ``build_gate_review``.
    """

    rows: list[pd.DataFrame] = []
    for route_id, route in routes.items():
        if route.response_features.empty:
            continue
        frame = route.response_features[
            route.response_features["id"].astype(str).isin(shared_event_ids)
        ].copy()
        if frame.empty:
            continue
        frame["_route_variant_id"] = route_id
        frame["_canonical_error"] = pd.to_numeric(
            frame.get(
                "feature_start_effective_error_m",
                frame.get("anchor_projection_error_m", math.inf),
            ),
            errors="coerce",
        ).fillna(math.inf)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["id", "_canonical_error", "_route_variant_id"]
    )
    return combined.drop_duplicates("id", keep="first").drop(
        columns=["_route_variant_id", "_canonical_error"]
    ).sort_values("sequence").reset_index(drop=True)

def _map_all_candidate_laps(
    points: pd.DataFrame,
    laps: pd.DataFrame,
    centreline: Any,
    *,
    route_id: str,
    preferred_route_matched: pd.DataFrame,
) -> pd.DataFrame:
    """Use ordered native matching on a lap's own route and nearest matching across routes."""

    rows: list[pd.DataFrame] = []
    preferred = {
        int(lap_id): segment.copy()
        for lap_id, segment in preferred_route_matched.groupby("lap_id", sort=False)
    }
    for lap in laps.itertuples(index=False):
        if not bool(getattr(lap, "analysis_valid")):
            continue
        lap_id = int(getattr(lap, "lap_id"))
        source_route = str(getattr(lap, "route_variant_id", ""))
        if source_route == route_id and lap_id in preferred:
            mapped = preferred[lap_id].copy()
        else:
            segment = points.loc[
                int(getattr(lap, "start_global_index"))
                : int(getattr(lap, "end_global_index"))
            ].copy()
            segment = segment.dropna(subset=["x_m", "y_m"]).sort_index()
            if len(segment) < 2:
                continue
            mapped = _map_segment_nearest(segment, centreline)
            mapped["lap_id"] = lap_id
            times = pd.to_datetime(
                mapped["timestamp_utc"], utc=True, errors="coerce"
            )
            mapped["elapsed_lap_s"] = (
                times - times.iloc[0]
            ).dt.total_seconds()
        rows.append(mapped)
    if not rows:
        raise ValueError(
            "No data-quality-valid lap points remain for route-family projection."
        )
    return pd.concat(rows, ignore_index=True)



def _map_segment_nearest(segment: pd.DataFrame, centreline: Any) -> pd.DataFrame:
    """Project by physical proximity, not whole-lap normalized progress.

    A branch lap can have a very different total length, so normalized progress is
    intentionally not used for cross-route evidence. Ambiguous crossings are later
    rejected by the local coverage/direction checks.
    """

    output = segment.copy()
    matched_s: list[float] = []
    errors: list[float] = []
    matched_x: list[float] = []
    matched_y: list[float] = []
    for point_x, point_y in zip(output["x_m"], output["y_m"]):
        s_values, distance, qx, qy = centreline.all_projections(
            float(point_x), float(point_y)
        )
        best = int(np.argmin(distance))
        matched_s.append(float(s_values[best]))
        errors.append(float(distance[best]))
        matched_x.append(float(qx[best]))
        matched_y.append(float(qy[best]))
    output["s_m"] = matched_s
    output["map_error_m"] = errors
    output["matched_x_m"] = matched_x
    output["matched_y_m"] = matched_y
    return output

def _event_compatibility(
    *,
    mapped: pd.DataFrame,
    laps: pd.DataFrame,
    events: pd.DataFrame,
    track_length_m: float,
    maximum_error_m: float,
    shared_maximum_error_m: float,
    event_section_lookup: Mapping[str, str],
    minimum_coverage: float,
    bin_size_m: float,
    maximum_interpolation_gap_m: float,
    maximum_backward: int,
) -> pd.DataFrame:
    """Audit whether each lap locally supports each projected event.

    Coverage is measured continuously between consecutive acceptable projected
    samples rather than by requiring a raw sample inside every short analysis bin.
    This matches the event extractor's interpolation semantics and is important for
    1 Hz telemetry, where a valid 4 m entry window can lie entirely between two
    successive samples.  Interpolation is never allowed across an invalid sample,
    an excessive spatial gap, or a strongly backward local projection.
    """

    rows: list[dict[str, Any]] = []
    eligible_lap_ids = set(
        laps.loc[
            laps["analysis_valid"].astype(bool)
            & laps["use_for_gate_evidence"].astype(bool),
            "lap_id",
        ].astype(int)
    )
    by_lap = {
        int(lap_id): segment.sort_index()
        for lap_id, segment in mapped.groupby("lap_id", sort=False)
        if int(lap_id) in eligible_lap_ids
    }
    windows = (
        ("approach", "approach_start_rel_m", "approach_end_rel_m"),
        ("entry", "entry_start_rel_m", "entry_end_rel_m"),
        ("feature", "feature_start_rel_m", "feature_end_rel_m"),
        ("exit", "exit_start_rel_m", "exit_end_rel_m"),
    )
    for event in events.itertuples(index=False):
        event_id = str(getattr(event, "id"))
        anchor = float(getattr(event, "anchor_s_m"))
        section_kind = str(event_section_lookup.get(event_id, "shared"))
        applied_error_limit = (
            maximum_error_m
            if section_kind == "branch"
            else shared_maximum_error_m
        )
        lower = min(float(getattr(event, item[1])) for item in windows)
        upper = max(float(getattr(event, item[2])) for item in windows)
        wraps_track_boundary = (
            anchor + lower < 0.0 or anchor + upper > track_length_m
        )
        for lap_id, segment in by_lap.items():
            s = pd.to_numeric(segment["s_m"], errors="coerce").to_numpy(float)
            error = pd.to_numeric(
                segment["map_error_m"], errors="coerce"
            ).to_numpy(float)
            relative = (
                (s - anchor + 0.5 * track_length_m) % track_length_m
            ) - 0.5 * track_length_m
            valid = (
                np.isfinite(relative)
                & np.isfinite(error)
                & (error <= applied_error_limit)
            )
            bracketed_relative = np.where(valid, relative, np.nan)
            coverage: dict[str, float] = {}
            for label, lower_name, upper_name in windows:
                window_lower = float(getattr(event, lower_name))
                window_upper = float(getattr(event, upper_name))
                coverage[label] = _interval_coverage(
                    bracketed_relative,
                    window_lower,
                    window_upper,
                    bin_size_m,
                    maximum_gap_m=maximum_interpolation_gap_m,
                )
            local = valid & (relative >= lower) & (relative <= upper)
            local_relative = relative[local]
            local_error = error[local]
            raw_backward = int(
                np.sum(np.diff(local_relative) < -2.0 * bin_size_m)
            )
            # A complete lap whose analysis window straddles start/finish naturally
            # appears in two local chunks.  Discount that one known circular wrap;
            # any additional backward jump remains an ambiguity failure.
            backward = (
                max(0, raw_backward - 1)
                if wraps_track_boundary and raw_backward
                else raw_backward
            )
            p95_error = (
                float(np.quantile(local_error, 0.95))
                if len(local_error)
                else math.nan
            )
            coverage_ok = all(
                value >= minimum_coverage for value in coverage.values()
            )
            compatible = (
                coverage_ok
                and np.isfinite(p95_error)
                and p95_error <= applied_error_limit
                and backward <= maximum_backward
            )
            reasons: list[str] = []
            if not coverage_ok:
                reasons.append("insufficient_bracketed_local_window_coverage")
            if not np.isfinite(p95_error) or p95_error > applied_error_limit:
                reasons.append("local_projection_error_exceeds_limit")
            if backward > maximum_backward:
                reasons.append("ambiguous_or_backward_local_projection")
            rows.append(
                {
                    "event_id": event_id,
                    "lap_id": lap_id,
                    "route_compatible": compatible,
                    "route_compatibility_reason": ";".join(reasons),
                    "route_projection_coverage_method": (
                        "continuous_bracketed_interpolation"
                    ),
                    "route_projection_error_limit_m": applied_error_limit,
                    "route_projection_maximum_interpolation_gap_m": (
                        maximum_interpolation_gap_m
                    ),
                    "route_projection_p95_error_m": p95_error,
                    "route_projection_raw_local_backward_count": raw_backward,
                    "route_projection_local_backward_count": backward,
                    "route_projection_approach_coverage": coverage["approach"],
                    "route_projection_entry_coverage": coverage["entry"],
                    "route_projection_feature_coverage": coverage["feature"],
                    "route_projection_exit_coverage": coverage["exit"],
                }
            )
    return pd.DataFrame(rows)


def _interval_coverage(
    relative: np.ndarray,
    lower: float,
    upper: float,
    bin_size_m: float,
    *,
    maximum_gap_m: float | None = None,
) -> float:
    """Fraction of an interval continuously bracketed by projected samples.

    ``relative`` is kept in traversal order and may contain NaNs marking rejected
    samples.  Consecutive finite samples contribute the spatial span between them
    only when progress is forward and the gap is small enough to interpolate.
    This intentionally gives full coverage to a narrow window that lies between
    two valid 1 Hz samples while refusing to bridge a telemetry dropout or a
    missing alternate branch.
    """

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return 0.0
    values = np.asarray(relative, dtype=float)
    if len(values) < 2:
        return 0.0
    maximum_gap = (
        float(maximum_gap_m)
        if maximum_gap_m is not None
        else max(20.0, 4.0 * float(bin_size_m))
    )
    spans: list[tuple[float, float]] = []
    for left, right in zip(values[:-1], values[1:]):
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        delta = float(right - left)
        if delta < 0.0 or delta > maximum_gap:
            continue
        start = max(float(lower), float(left))
        end = min(float(upper), float(right))
        if end > start:
            spans.append((start, end))
    if not spans:
        return 0.0
    spans.sort()
    merged: list[list[float]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    covered = sum(end - start for start, end in merged)
    return float(np.clip(covered / (upper - lower), 0.0, 1.0))


def _shared_gate_evidence(
    routes: Mapping[str, TrackEvidenceBuild],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for route_id, route in routes.items():
        if route.event_passes.empty:
            continue
        frame = route.event_passes.copy()
        frame["target_route_variant_id"] = route_id
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    passes = pd.concat(frames, ignore_index=True, sort=False)
    rows: list[dict[str, Any]] = []
    for event_id, group in passes.groupby("event_id", sort=False):
        eligible = group[group["eligible"].astype(bool)]
        pre_compatibility_mask = (
            group["eligible_before_route_compatibility"].astype(bool)
            if "eligible_before_route_compatibility" in group
            else group["eligible"].astype(bool)
        )
        pre_compatibility = group[pre_compatibility_mask]
        compatible_mask = (
            group["route_compatible"].astype(bool)
            if "route_compatible" in group
            else group["eligible"].astype(bool)
        )
        compatible = group[compatible_mask]
        route_counts = {
            str(route_id): int(segment.loc[segment["eligible"].astype(bool), "lap_id"].nunique())
            for route_id, segment in group.groupby("target_route_variant_id")
        }
        unique_eligible = eligible.drop_duplicates(subset=["lap_id"])
        scopes = (
            sorted(set(group["evidence_scope"].astype(str)))
            if "evidence_scope" in group
            else ["legacy_local_compatibility"]
        )
        branch_ids = (
            sorted(
                token
                for value in group.get("branch_ids", pd.Series(dtype=str)).astype(str)
                for token in value.split(";")
                if token
            )
            if "branch_ids" in group
            else []
        )
        rows.append(
            {
                "event_id": str(event_id),
                "event_name": str(group["event_name"].iloc[0]),
                "measured_unique_lap_count_before_local_compatibility": int(
                    pre_compatibility["lap_id"].nunique()
                ),
                "eligible_unique_lap_count": int(unique_eligible["lap_id"].nunique()),
                "compatible_unique_lap_count": int(compatible["lap_id"].nunique()),
                "compatible_route_variant_ids": ";".join(
                    sorted(set(compatible["target_route_variant_id"].astype(str)))
                ),
                "contributing_source_route_variant_ids": ";".join(
                    sorted(
                        set(
                            (
                                unique_eligible["source_route_variant_id"]
                                if "source_route_variant_id" in unique_eligible
                                else unique_eligible["target_route_variant_id"]
                            ).astype(str)
                        )
                        - {""}
                    )
                ),
                "route_eligible_lap_counts_json": json.dumps(route_counts, sort_keys=True),
                "evidence_scope": ";".join(scopes),
                "branch_ids": ";".join(branch_ids),
                "evidence_pooling": (
                    "single_shared_contract_reused_across_supported_traversals"
                    if scopes == ["shared_course"]
                    else "route_branch_only_inside_divergence_corridor"
                    if scopes == ["branch_specific"]
                    else "mixed_route_section_scope"
                ),
                "shared_contract_reused_across_routes": bool(
                    scopes == ["shared_course"]
                    and (
                        "shared_evidence_contract" not in group
                        or group["shared_evidence_contract"].fillna(False).astype(bool).any()
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _course_cases(
    routes: Mapping[str, TrackEvidenceBuild],
    nominal_id: str,
) -> pd.DataFrame:
    count = max(1, len(routes))
    return pd.DataFrame(
        [
            {
                "course_case_id": route_id,
                "route_variant_id": route_id,
                "nominal": route_id == nominal_id,
                "case_weight": 1.0,
                "normalized_equal_weight": 1.0 / count,
                "aggregation_policy": "unweighted_equal_course_cases",
                "track_bundle_relative_path": f"track/{route_id}/track_bundle.json",
                "track_length_m": float(route.centreline.length_m),
            }
            for route_id, route in routes.items()
        ]
    )


def _augment_summary(
    summary: pd.DataFrame,
    routes: Mapping[str, TrackEvidenceBuild],
    nominal_id: str,
) -> pd.DataFrame:
    output = summary.copy()
    if output.empty:
        output = pd.DataFrame(
            {"route_variant_id": list(routes), "supported": True}
        )
    output["family_exported"] = output["route_variant_id"].astype(str).isin(routes)
    output["nominal"] = output["route_variant_id"].astype(str) == nominal_id
    output["track_bundle_relative_path"] = output["route_variant_id"].map(
        lambda value: f"track/{value}/track_bundle.json" if str(value) in routes else ""
    )
    output["track_length_m"] = output["route_variant_id"].map(
        {key: float(value.centreline.length_m) for key, value in routes.items()}
    )
    output["applicable_physical_event_count"] = output["route_variant_id"].map(
        {key: int(len(value.event_projection)) for key, value in routes.items()}
    )
    output["pre_consensus_geometry_lap_count"] = output["route_variant_id"].map(
        {
            key: int(
                value.laps["pre_consensus_valid"].fillna(False).astype(bool).sum()
            )
            if "pre_consensus_valid" in value.laps
            else int(value.laps["analysis_valid"].fillna(False).astype(bool).sum())
            for key, value in routes.items()
        }
    )
    output["valid_geometry_lap_count"] = output["route_variant_id"].map(
        {
            key: int(value.laps["analysis_valid"].fillna(False).astype(bool).sum())
            for key, value in routes.items()
        }
    )
    output["consensus_excluded_lap_count"] = output["route_variant_id"].map(
        {
            key: int(
                value.laps["consensus_excluded"].fillna(False).astype(bool).sum()
            )
            if "consensus_excluded" in value.laps
            else 0
            for key, value in routes.items()
        }
    )
    output["accepted_gate_count"] = output["route_variant_id"].map(
        {
            key: int(
                (value.gate_review["recommendation"] == "accepted").sum()
            )
            if not value.gate_review.empty
            and "recommendation" in value.gate_review
            else 0
            for key, value in routes.items()
        }
    )
    output["eligible_gate_evidence_unique_lap_count"] = output[
        "route_variant_id"
    ].map(
        {
            key: int(
                value.event_passes.loc[
                    value.event_passes["eligible"].astype(bool), "lap_id"
                ].nunique()
            )
            if not value.event_passes.empty
            and {"eligible", "lap_id"}.issubset(value.event_passes.columns)
            else 0
            for key, value in routes.items()
        }
    )
    return output
