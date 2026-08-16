"""Immutable runtime track assembled only from a validated track bundle."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from math import atan, degrees, isfinite, sqrt
from typing import Any, Mapping

import numpy as np

from cvt_track_study.bundle import TrackBundle

from .models import SimulationInputError
from .obstacles import ObstacleContext, ObstacleModel, obstacle_model_from_contract


@dataclass(frozen=True, slots=True)
class RuntimeInterval:
    start_s_m: float
    end_s_m: float
    length_m: float
    wraps_start_finish: bool

    def local_distance(self, s_m: float, track_length_m: float) -> float | None:
        s = float(s_m) % track_length_m
        if not self.wraps_start_finish:
            if self.start_s_m <= s < self.end_s_m:
                return s - self.start_s_m
            return None
        if s >= self.start_s_m:
            return s - self.start_s_m
        if s < self.end_s_m:
            return track_length_m - self.start_s_m + s
        return None


@dataclass(frozen=True, slots=True)
class RuntimeFeature:
    identifier: str
    name: str
    response_group_id: str
    interval: RuntimeInterval
    model: ObstacleModel


@dataclass(frozen=True, slots=True)
class RuntimeSpeedGate:
    identifier: str
    response_group_id: str
    name: str
    position_s_m: float
    target_speed_mps: float
    confidence_score: float
    gate_type: str = "entry_speed"
    enforcement_class: str = "hard_absolute"


@dataclass(frozen=True, slots=True)
class TrackSample:
    s_m: float
    reference_elevation_m: float | None
    modeled_elevation_offset_m: float
    modeled_grade_degrees: float
    curvature_1_per_m: float
    friction_coefficient: float
    normal_load_scale: float
    obstacle_resistance_force_n: float
    feature_resistance_forces_n: tuple[tuple[str, float], ...]
    active_feature_ids: tuple[str, ...]
    active_feature_names: tuple[str, ...]

    @property
    def modeled_grade_radians(self) -> float:
        return np.deg2rad(self.modeled_grade_degrees).item()


@dataclass(frozen=True, slots=True)
class RuntimeTrack:
    name: str
    length_m: float
    closed_course: bool
    centreline_s_m: tuple[float, ...]
    centreline_x_m: tuple[float, ...]
    centreline_y_m: tuple[float, ...]
    centreline_curvature_1_per_m: tuple[float, ...]
    reference_elevation_m: tuple[float | None, ...]
    surface_friction_coefficient: float
    features: tuple[RuntimeFeature, ...]
    speed_gates: tuple[RuntimeSpeedGate, ...]
    global_speed_guardrail_mps: float
    gpx_grade_force_enabled: bool
    _speed_gate_positions_sorted_m: tuple[float, ...] = field(
        init=False, repr=False, compare=False
    )
    _speed_gate_envelopes: tuple[
        tuple[tuple[tuple[float, float], ...], tuple[float, ...]], ...
    ] = field(init=False, repr=False, compare=False)
    _centreline_s_array: np.ndarray = field(init=False, repr=False, compare=False)
    _curvature_array: np.ndarray = field(init=False, repr=False, compare=False)
    _elevation_s_array: np.ndarray = field(init=False, repr=False, compare=False)
    _elevation_array: np.ndarray = field(init=False, repr=False, compare=False)
    _feature_bin_width_m: float = field(init=False, repr=False, compare=False)
    _feature_bins: tuple[tuple[RuntimeFeature, ...], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.closed_course:
            raise SimulationInputError("Phase 5 supports closed-course track bundles only.")
        if not isfinite(self.length_m) or self.length_m <= 0.0:
            raise SimulationInputError("Track length must be positive and finite.")
        lengths = {len(self.centreline_s_m), len(self.centreline_x_m), len(self.centreline_y_m), len(self.centreline_curvature_1_per_m), len(self.reference_elevation_m)}
        if len(lengths) != 1:
            raise SimulationInputError("Centreline distance, geometry, curvature, and elevation arrays must align.")
        if len(self.centreline_s_m) < 2:
            raise SimulationInputError("Track runtime requires at least two centreline samples.")
        if self.surface_friction_coefficient <= 0.0:
            raise SimulationInputError("Surface friction coefficient must be positive.")
        if (
            not isfinite(self.global_speed_guardrail_mps)
            or self.global_speed_guardrail_mps <= 0.0
        ):
            raise SimulationInputError(
                "Global speed guardrail must be positive and finite."
            )
        ordered_gates = sorted(
            self.speed_gates, key=lambda gate: float(gate.position_s_m)
        )
        positions = tuple(float(gate.position_s_m) for gate in ordered_gates)
        target_squared = tuple(
            float(gate.target_speed_mps) ** 2 for gate in ordered_gates
        )
        object.__setattr__(self, "_speed_gate_positions_sorted_m", positions)
        object.__setattr__(
            self,
            "_speed_gate_envelopes",
            _build_cyclic_speed_gate_envelopes(
                positions, target_squared, self.length_m
            ),
        )
        centreline_s = np.asarray(self.centreline_s_m, dtype=float)
        curvature = np.asarray(self.centreline_curvature_1_per_m, dtype=float)
        elevation_pairs = [
            (float(s_m), float(elevation_m))
            for s_m, elevation_m in zip(
                self.centreline_s_m, self.reference_elevation_m
            )
            if elevation_m is not None
        ]
        if elevation_pairs:
            elevation_s = np.asarray([item[0] for item in elevation_pairs], dtype=float)
            elevation = np.asarray([item[1] for item in elevation_pairs], dtype=float)
        else:
            elevation_s = np.asarray([], dtype=float)
            elevation = np.asarray([], dtype=float)
        object.__setattr__(self, "_centreline_s_array", centreline_s)
        object.__setattr__(self, "_curvature_array", curvature)
        object.__setattr__(self, "_elevation_s_array", elevation_s)
        object.__setattr__(self, "_elevation_array", elevation)
        feature_bin_width, feature_bins = _build_feature_bins(
            self.features, self.length_m
        )
        object.__setattr__(self, "_feature_bin_width_m", feature_bin_width)
        object.__setattr__(self, "_feature_bins", feature_bins)


    def sample(
        self,
        distance_m: float,
        *,
        vehicle_speed_mps: float,
        vehicle_mass_kg: float,
        gravity_mps2: float,
        feature_entry_speeds_mps: Mapping[str, float] | None = None,
    ) -> TrackSample:
        s = min(max(float(distance_m), 0.0), self.length_m)
        loop_s = 0.0 if s >= self.length_m else s
        reference_elevation = (
            None
            if self._elevation_s_array.size < 2
            else float(
                np.interp(loop_s, self._elevation_s_array, self._elevation_array)
            )
        )
        curvature = float(
            np.interp(loop_s, self._centreline_s_array, self._curvature_array)
        )
        resistance = 0.0
        elevation_offset = 0.0
        slope = 0.0
        normal_scale = 1.0
        friction = self.surface_friction_coefficient
        active_ids: list[str] = []
        active_names: list[str] = []
        feature_forces: list[tuple[str, float]] = []
        entry_speeds = feature_entry_speeds_mps or {}
        bin_index = min(
            int(loop_s / self._feature_bin_width_m), len(self._feature_bins) - 1
        )
        for feature in self._feature_bins[bin_index]:
            local = feature.interval.local_distance(loop_s, self.length_m)
            if local is None:
                continue
            effect = feature.model.evaluate(
                ObstacleContext(
                    local_distance_m=local,
                    interval_length_m=feature.interval.length_m,
                    vehicle_speed_mps=max(0.0, vehicle_speed_mps),
                    entry_speed_mps=max(
                        0.0,
                        float(entry_speeds.get(feature.identifier, vehicle_speed_mps)),
                    ),
                    vehicle_mass_kg=vehicle_mass_kg,
                    gravity_mps2=gravity_mps2,
                )
            )
            feature_force = max(0.0, effect.resistance_force_n)
            resistance += feature_force
            feature_forces.append((feature.identifier, feature_force))
            elevation_offset += effect.elevation_offset_m
            slope += effect.grade_slope_addition
            normal_scale *= max(0.0, effect.normal_load_scale)
            friction *= max(0.0, effect.friction_multiplier)
            active_ids.append(feature.identifier)
            active_names.append(feature.name)
        return TrackSample(
            s_m=loop_s,
            reference_elevation_m=reference_elevation,
            modeled_elevation_offset_m=elevation_offset,
            modeled_grade_degrees=degrees(atan(slope)),
            curvature_1_per_m=curvature,
            friction_coefficient=max(0.0, friction),
            normal_load_scale=max(0.0, normal_scale),
            obstacle_resistance_force_n=max(0.0, resistance),
            feature_resistance_forces_n=tuple(feature_forces),
            active_feature_ids=tuple(active_ids),
            active_feature_names=tuple(active_names),
        )

    def safe_speed_ceiling_mps(
        self, distance_m: float, *, braking_deceleration_mps2: float
    ) -> float:
        """Backward-propagate every accepted gate through a finite braking envelope."""

        if braking_deceleration_mps2 <= 0.0:
            raise SimulationInputError("braking_deceleration_mps2 must be positive.")
        s = min(max(float(distance_m), 0.0), self.length_m)
        loop_s = 0.0 if s >= self.length_m else s
        # The global guardrail is intentionally very permissive, but it makes the
        # contract total: even a sparse or pathological reconstruction never leaves
        # a simulated vehicle with an infinite speed target. Local hard gates,
        # relative-response gates and candidate guardrails can only tighten it.
        result = float(self.global_speed_guardrail_mps)
        if not self.speed_gates:
            return result
        # For a fixed interval between consecutive gate positions, every cyclic
        # braking constraint is linear in the effective braking deceleration:
        #
        #   v^2 <= q_i + 2 a ((p_i - s) mod L)
        #        = (q_i + 2 a p_i*) - 2 a s.
        #
        # ``_speed_gate_envelopes`` stores the exact lower envelope of the lines
        # q_i + 2 a p_i* for each interval.  Querying it avoids allocating and
        # reducing a ~50-element NumPy array at every millisecond integration
        # step while preserving the same gate-by-gate braking constraint.
        interval_index = bisect_left(self._speed_gate_positions_sorted_m, loop_s)
        lines, starts = self._speed_gate_envelopes[interval_index]
        line_index = bisect_right(starts, braking_deceleration_mps2) - 1
        slope, intercept = lines[max(line_index, 0)]
        minimum_squared = (
            slope * braking_deceleration_mps2
            + intercept
            - 2.0 * braking_deceleration_mps2 * loop_s
        )
        return min(result, sqrt(max(minimum_squared, 0.0)))



def _build_feature_bins(
    features: tuple[RuntimeFeature, ...],
    track_length_m: float,
    *,
    bin_count: int = 256,
) -> tuple[float, tuple[tuple[RuntimeFeature, ...], ...]]:
    """Build conservative spatial candidate bins for exact feature lookup.

    A feature is placed into every bin whose open spatial interval can intersect
    it. ``sample`` still calls ``local_distance`` on candidates, so binning only
    removes impossible features and cannot change obstacle activation semantics.
    """

    count = max(1, int(bin_count))
    width = float(track_length_m) / count
    bins: list[list[RuntimeFeature]] = [[] for _ in range(count)]
    for feature in features:
        interval = feature.interval
        ranges = (
            ((interval.start_s_m, interval.end_s_m),)
            if not interval.wraps_start_finish
            else (
                (interval.start_s_m, track_length_m),
                (0.0, interval.end_s_m),
            )
        )
        touched: set[int] = set()
        for start, end in ranges:
            if end <= start:
                continue
            first = min(int(max(start, 0.0) / width), count - 1)
            # The interval is end-exclusive. A tiny downward shift keeps a
            # boundary exactly on a bin edge out of the following untouched bin.
            end_inside = max(start, min(end, track_length_m) - 1.0e-12)
            last = min(int(end_inside / width), count - 1)
            touched.update(range(first, last + 1))
        for index in touched:
            bins[index].append(feature)
    return width, tuple(tuple(items) for items in bins)


def _build_cyclic_speed_gate_envelopes(
    positions_m: tuple[float, ...],
    target_squared_mps2: tuple[float, ...],
    track_length_m: float,
) -> tuple[tuple[tuple[tuple[float, float], ...], tuple[float, ...]], ...]:
    """Precompute exact lower line envelopes for cyclic braking constraints.

    Interval ``k`` corresponds to ``bisect_left(positions_m, s) == k``.  Gates
    before ``k`` are one lap ahead; gates at/after ``k`` are on the current lap.
    Each candidate is represented by ``m*a + b`` with ``m = 2*p*`` and
    ``b = target_speed**2``.
    """

    count = len(positions_m)
    if count == 0:
        return tuple()
    envelopes = []
    for split in range(count + 1):
        adjusted = [
            (
                2.0 * (position + (track_length_m if index < split else 0.0)),
                target_squared_mps2[index],
            )
            for index, position in enumerate(positions_m)
        ]
        # Minimum-envelope construction is simplest with strictly descending
        # slopes. Equal-position gates share a slope, so retain only the lowest
        # intercept before building the hull.
        by_slope: dict[float, float] = {}
        for slope, intercept in adjusted:
            previous = by_slope.get(slope)
            if previous is None or intercept < previous:
                by_slope[slope] = intercept
        candidates = sorted(by_slope.items(), reverse=True)

        hull: list[tuple[float, float]] = []
        starts: list[float] = []
        for slope, intercept in candidates:
            start = float("-inf")
            while hull:
                previous_slope, previous_intercept = hull[-1]
                start = (intercept - previous_intercept) / (
                    previous_slope - slope
                )
                if start > starts[-1]:
                    break
                hull.pop()
                starts.pop()
            if not hull:
                start = float("-inf")
            hull.append((slope, intercept))
            starts.append(start)
        envelopes.append((tuple(hull), tuple(starts)))
    return tuple(envelopes)


def runtime_track_from_bundle(
    bundle: TrackBundle,
    *,
    surface_friction_coefficient: float,
    gate_speed_statistic: str = "median",
    gate_target_speeds_mps: Mapping[str, float] | None = None,
    target_vehicle_id: str | None = None,
    obstacle_model_types: Mapping[str, str] | None = None,
    obstacle_parameters_si: Mapping[str, Mapping[str, float]] | None = None,
) -> RuntimeTrack:
    """Resolve one nominal or sampled runtime track from a validated bundle."""

    if gate_speed_statistic not in {"p10", "median", "p90"}:
        raise SimulationInputError("gate_speed_statistic must be p10, median, or p90.")
    simulation = bundle.data["simulation_contract"]
    centreline = simulation["centreline"]["samples"]
    centreline_s = tuple(float(row["s_m"]) for row in centreline)
    centreline_x = tuple(float(row["x_m"]) for row in centreline)
    centreline_y = tuple(float(row["y_m"]) for row in centreline)
    centreline_curvature = _closed_curvature(centreline_s, centreline_x, centreline_y)
    features: list[RuntimeFeature] = []
    for raw in simulation["physical_features"]:
        interval = raw["interval"]
        features.append(
            RuntimeFeature(
                identifier=str(raw["id"]),
                name=str(raw["name"]),
                response_group_id=str(raw["response_group_id"]),
                interval=RuntimeInterval(
                    start_s_m=float(interval["start_s_m"]),
                    end_s_m=float(interval["end_s_m"]),
                    length_m=float(interval["length_m"]),
                    wraps_start_finish=bool(interval["wraps_start_finish"]),
                ),
                model=obstacle_model_from_contract(
                    raw["obstacle_model"],
                    model_type_override=(obstacle_model_types or {}).get(str(raw["id"])),
                    parameter_overrides_si=(obstacle_parameters_si or {}).get(str(raw["id"])),
                ),
            )
        )
    gate_overrides = gate_target_speeds_mps or {}
    speed_key = {
        "p10": "p10_mps",
        "median": "median_mps",
        "p90": "p90_mps",
    }[gate_speed_statistic]
    gates: list[RuntimeSpeedGate] = []
    for raw in simulation["speed_gates"]:
        if not bool(raw["active_by_default"]):
            continue
        distribution = raw["target_speed_distribution"]
        summary = distribution["summary"]
        vehicle_summaries = distribution.get("vehicle_summaries", {})
        if (
            target_vehicle_id is not None
            and isinstance(vehicle_summaries, Mapping)
            and str(target_vehicle_id) in vehicle_summaries
            and str(raw.get("enforcement_class", "hard_absolute"))
            in {"hard_absolute", "relative_response"}
        ):
            summary = vehicle_summaries[str(target_vehicle_id)]
        gates.append(
            RuntimeSpeedGate(
                identifier=str(raw["id"]),
                response_group_id=str(raw["response_group_id"]),
                name=str(raw["name"]),
                position_s_m=float(raw["position_s_m"]),
                target_speed_mps=float(
                    gate_overrides.get(str(raw["id"]), summary[speed_key])
                ),
                confidence_score=float(raw["confidence"]["overall_score"]),
                gate_type=str(raw.get("gate_type", "entry_speed")),
                enforcement_class=str(raw.get("enforcement_class", "hard_absolute")),
            )
        )
    capabilities = simulation["capabilities"]
    if not bool(capabilities.get("obstacle_models_ready")):
        raise SimulationInputError(
            "Track bundle does not declare complete obstacle models. Rebuild it with Phase 5 event profiles."
        )
    return RuntimeTrack(
        name=str(bundle.data["identity"]["track_name"]),
        length_m=float(simulation["track_length_m"]),
        closed_course=bool(bundle.data["identity"]["closed_course"]),
        centreline_s_m=centreline_s,
        centreline_x_m=centreline_x,
        centreline_y_m=centreline_y,
        centreline_curvature_1_per_m=centreline_curvature,
        reference_elevation_m=tuple(
            None if row.get("reference_elevation_m") is None else float(row["reference_elevation_m"])
            for row in centreline
        ),
        surface_friction_coefficient=surface_friction_coefficient,
        features=tuple(features),
        speed_gates=tuple(sorted(gates, key=lambda gate: gate.position_s_m)),
        global_speed_guardrail_mps=float(
            simulation.get("global_speed_guardrail_mps", 25.0)
        ),
        gpx_grade_force_enabled=bool(simulation["grade_force_enabled"]),
    )


def _interpolate_optional(
    x: float, sample_x: tuple[float, ...], sample_y: tuple[float | None, ...]
) -> float | None:
    valid = [(sx, sy) for sx, sy in zip(sample_x, sample_y) if sy is not None]
    if len(valid) < 2:
        return None
    xs = np.asarray([item[0] for item in valid], dtype=float)
    ys = np.asarray([item[1] for item in valid], dtype=float)
    return float(np.interp(x, xs, ys))


def _closed_curvature(
    s_m: tuple[float, ...], x_m: tuple[float, ...], y_m: tuple[float, ...]
) -> tuple[float, ...]:
    """Estimate signed planar curvature from the published centreline samples."""

    s = np.asarray(s_m, dtype=float)
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    edge_order = 2 if len(s) >= 3 else 1
    dx = np.gradient(x, s, edge_order=edge_order)
    dy = np.gradient(y, s, edge_order=edge_order)
    ddx = np.gradient(dx, s, edge_order=edge_order)
    ddy = np.gradient(dy, s, edge_order=edge_order)
    denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1.0e-12)
    curvature = (dx * ddy - dy * ddx) / denominator
    curvature[~np.isfinite(curvature)] = 0.0
    return tuple(float(value) for value in curvature)
