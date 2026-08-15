"""Empirical endurance-traffic calibration and reproducible traffic worlds.

The model is intentionally small and auditable:

* arrival times follow a non-homogeneous Poisson process (NHPP);
* the encounter rate follows the empirical surviving-field fraction ``S(t)`` as
  ``lambda(t) = lambda0 * S(t)**alpha``;
* ``lambda0`` and ``alpha`` are fitted from unbinned event times with the actual
  observer exposure windows, not from arbitrary time bins;
* calibration uncertainty is carried as joint bootstrap draws of
  ``(lambda0, alpha)`` so their correlation is preserved;
* event type, duration and retained-speed fraction are resampled jointly from the
  observed events. No invented parametric severity distribution or confidence-to-
  sigma conversion is added.

The model describes an exogenous traffic world.  The simulator maps each event's
retained-speed fraction onto a shared traffic-free reference speed profile, so the
same traffic world imposes the same absolute speed ceiling on every drivetrain
candidate in a paired comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from math import isfinite, log
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import tomllib

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kstest


TRAFFIC_SCHEMA_VERSION = 3


class TrafficDataError(ValueError):
    """Raised when traffic evidence cannot form a defensible calibration."""


@dataclass(frozen=True, slots=True)
class TrafficObservation:
    stream_id: str
    source_row: int
    time_s: float
    event_type: str
    duration_s: float
    retained_speed_fraction: float
    baseline_speed_ordinal: float | None
    confidence: str
    source_file: str

    def serializable(self) -> dict[str, object]:
        return {
            "stream_id": self.stream_id,
            "source_row": self.source_row,
            "time_s": self.time_s,
            "event_type": self.event_type,
            "duration_s": self.duration_s,
            "retained_speed_fraction": self.retained_speed_fraction,
            "baseline_speed_ordinal": self.baseline_speed_ordinal,
            "confidence": self.confidence,
            "source_file": self.source_file,
        }


@dataclass(frozen=True, slots=True)
class ObservationStream:
    identifier: str
    source_file: str
    exposure_start_s: float
    exposure_end_s: float
    observations: tuple[TrafficObservation, ...]

    @property
    def exposure_s(self) -> float:
        return self.exposure_end_s - self.exposure_start_s


@dataclass(frozen=True, slots=True)
class FieldPopulationRecord:
    observed_active_time_s: float
    last_lap_duration_s: float
    censored: bool
    source_row: int


@dataclass(frozen=True, slots=True)
class TrafficCalibrationDraw:
    initial_rate_per_hour: float
    field_exponent: float

    def serializable(self) -> dict[str, float]:
        return {
            "initial_rate_per_hour": self.initial_rate_per_hour,
            "field_exponent": self.field_exponent,
        }


@dataclass(frozen=True, slots=True)
class TrafficEvent:
    identifier: str
    start_offset_s: float
    duration_s: float
    retained_speed_fraction: float
    event_type: str
    source_stream_id: str
    source_row: int
    source_confidence: str
    baseline_speed_ordinal: float | None

    @property
    def end_offset_s(self) -> float:
        return self.start_offset_s + self.duration_s

    def serializable(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "start_offset_s": self.start_offset_s,
            "duration_s": self.duration_s,
            "retained_speed_fraction": self.retained_speed_fraction,
            "event_type": self.event_type,
            "source_stream_id": self.source_stream_id,
            "source_row": self.source_row,
            "source_confidence": self.source_confidence,
            "baseline_speed_ordinal": self.baseline_speed_ordinal,
        }


@dataclass(frozen=True, slots=True)
class TrafficRealization:
    scenario_seed: int
    lap_start_race_time_s: float
    race_duration_s: float
    initial_rate_per_hour: float
    field_exponent: float
    calibration_draw_index: int
    events: tuple[TrafficEvent, ...]
    model_fingerprint: str

    def active_events(self, lap_time_s: float) -> tuple[TrafficEvent, ...]:
        t = float(lap_time_s)
        return tuple(event for event in self.events if event.start_offset_s <= t < event.end_offset_s)

    def retained_fraction(self, lap_time_s: float) -> float:
        active = self.active_events(lap_time_s)
        if not active:
            return 1.0
        return min(event.retained_speed_fraction for event in active)

    def serializable(self) -> dict[str, object]:
        return {
            "scenario_seed": self.scenario_seed,
            "lap_start_race_time_s": self.lap_start_race_time_s,
            "race_duration_s": self.race_duration_s,
            "initial_rate_per_hour": self.initial_rate_per_hour,
            "field_exponent": self.field_exponent,
            "calibration_draw_index": self.calibration_draw_index,
            "event_count": len(self.events),
            "model_fingerprint": self.model_fingerprint,
            "events": [event.serializable() for event in self.events],
        }


@dataclass(frozen=True, slots=True)
class TrafficReferenceProfile:
    """Traffic-free speed versus distance used to create an absolute traffic cap."""

    distance_m: tuple[float, ...]
    speed_mps: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.distance_m) != len(self.speed_mps) or len(self.distance_m) < 2:
            raise TrafficDataError("traffic reference profile requires matching arrays with >=2 points")
        distance = np.asarray(self.distance_m, dtype=float)
        speed = np.asarray(self.speed_mps, dtype=float)
        if not np.all(np.isfinite(distance)) or not np.all(np.isfinite(speed)):
            raise TrafficDataError("traffic reference profile must be finite")
        if np.any(np.diff(distance) < 0.0):
            raise TrafficDataError("traffic reference profile distance must be nondecreasing")

    def speed_at_distance(self, distance_m: float) -> float:
        return max(
            0.0,
            float(np.interp(float(distance_m), self.distance_m, self.speed_mps)),
        )

    def serializable(self) -> dict[str, object]:
        return {"distance_m": list(self.distance_m), "speed_mps": list(self.speed_mps)}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TrafficReferenceProfile":
        return cls(
            distance_m=tuple(float(value) for value in raw.get("distance_m", ())),
            speed_mps=tuple(float(value) for value in raw.get("speed_mps", ())),
        )

    @classmethod
    def from_trace(cls, trace: Any, *, spacing_m: float = 1.0) -> "TrafficReferenceProfile":
        distance = np.asarray(trace.numeric["distance_m"], dtype=float)
        speed = np.asarray(trace.numeric["vehicle_speed_mps"], dtype=float)
        if len(distance) < 2:
            raise TrafficDataError("simulation trace is too short to form traffic reference")
        # Collapse duplicate distances before interpolation. A one-metre profile is
        # compact enough for the simulation cache and much finer than any traffic
        # evidence resolution.
        keep = np.concatenate(([True], np.diff(distance) > 1.0e-9))
        d = distance[keep]
        v = speed[keep]
        if d[-1] <= d[0]:
            return cls((float(d[0]), float(d[0] + 1.0e-6)), (float(v[0]), float(v[0])))
        spacing = max(0.1, float(spacing_m))
        grid = np.arange(d[0], d[-1], spacing, dtype=float)
        if grid.size == 0 or grid[-1] < d[-1]:
            grid = np.append(grid, d[-1])
        values = np.interp(grid, d, v)
        return cls(tuple(float(x) for x in grid), tuple(float(x) for x in values))


@dataclass(frozen=True, slots=True)
class TrafficCalibration:
    config_path: str
    race_duration_s: float
    # ``streams``/``observations`` are the target-race evidence used by generated
    # worlds. ``source_*`` identify the race that supplies the field-decay shape.
    streams: tuple[ObservationStream, ...]
    source_streams: tuple[ObservationStream, ...]
    field_records: tuple[FieldPopulationRecord, ...]
    dropout_offset_fraction: float
    km_time_s: tuple[float, ...]
    km_survival: tuple[float, ...]
    initial_rate_per_hour: float
    source_initial_rate_per_hour: float
    source_stream_initial_rates_per_hour: tuple[tuple[str, float], ...]
    field_exponent: float
    log_likelihood: float
    target_log_likelihood: float
    exponential_initial_rate_per_hour: float
    exponential_decay_per_s: float
    exponential_log_likelihood: float
    homogeneous_rate_per_hour: float
    homogeneous_log_likelihood: float
    bootstrap_draws: tuple[TrafficCalibrationDraw, ...]
    observations: tuple[TrafficObservation, ...]
    source_observations: tuple[TrafficObservation, ...]
    time_rescaling_ks_statistic: float
    time_rescaling_ks_pvalue: float
    target_time_rescaling_ks_statistic: float
    target_time_rescaling_ks_pvalue: float
    source_race_id: str
    target_race_id: str
    fingerprint: str

    @property
    def total_observer_exposure_s(self) -> float:
        return sum(stream.exposure_s for stream in self.streams)

    @property
    def source_observer_exposure_s(self) -> float:
        return sum(stream.exposure_s for stream in self.source_streams)

    def survival_fraction(self, race_time_s: float) -> float:
        t = min(max(0.0, float(race_time_s)), self.race_duration_s)
        index = int(np.searchsorted(self.km_time_s, t, side="right") - 1)
        index = max(0, min(index, len(self.km_survival) - 1))
        return float(self.km_survival[index])

    def rate_per_hour(
        self,
        race_time_s: float,
        *,
        initial_rate_per_hour: float | None = None,
        field_exponent: float | None = None,
    ) -> float:
        initial = self.initial_rate_per_hour if initial_rate_per_hour is None else float(initial_rate_per_hour)
        exponent = self.field_exponent if field_exponent is None else float(field_exponent)
        return max(0.0, initial * self.survival_fraction(race_time_s) ** max(0.0, exponent))

    def average_rate_per_hour(self) -> float:
        integral = _integral_survival_power(
            np.asarray(self.km_time_s),
            np.asarray(self.km_survival),
            0.0,
            self.race_duration_s,
            self.field_exponent,
        )
        return self.initial_rate_per_hour * integral / self.race_duration_s

    def draw_realization(self, *, scenario_seed: int, horizon_s: float) -> TrafficRealization:
        rng = np.random.default_rng(int(scenario_seed) ^ 0x545241464649435F)
        draws = self.bootstrap_draws
        if draws:
            draw_index = int(rng.integers(0, len(draws)))
            draw = draws[draw_index]
            initial_rate = draw.initial_rate_per_hour
            exponent = draw.field_exponent
        else:
            draw_index = -1
            initial_rate = self.initial_rate_per_hour
            exponent = self.field_exponent

        # We want the distribution of a complete representative lap across the
        # endurance clock. Start time is therefore uniform over the race. Events a
        # few minutes beyond the nominal checkered flag use the final observed field
        # survival instead of truncating the lap and biasing late-race traffic low.
        lap_start = float(rng.uniform(0.0, self.race_duration_s))
        event_times = _draw_nhpp_window(
            rng=rng,
            start_race_time_s=lap_start,
            horizon_s=max(0.0, float(horizon_s)),
            initial_rate_per_hour=initial_rate,
            field_exponent=exponent,
            km_time_s=np.asarray(self.km_time_s),
            km_survival=np.asarray(self.km_survival),
            race_duration_s=self.race_duration_s,
        )
        marks = self.observations
        events: list[TrafficEvent] = []
        for index, start_offset in enumerate(event_times, 1):
            source = marks[int(rng.integers(0, len(marks)))]
            events.append(
                TrafficEvent(
                    identifier=f"traffic_{index:02d}",
                    start_offset_s=float(start_offset),
                    duration_s=source.duration_s,
                    retained_speed_fraction=source.retained_speed_fraction,
                    event_type=source.event_type,
                    source_stream_id=source.stream_id,
                    source_row=source.source_row,
                    source_confidence=source.confidence,
                    baseline_speed_ordinal=source.baseline_speed_ordinal,
                )
            )
        return TrafficRealization(
            scenario_seed=int(scenario_seed),
            lap_start_race_time_s=lap_start,
            race_duration_s=self.race_duration_s,
            initial_rate_per_hour=float(initial_rate),
            field_exponent=float(exponent),
            calibration_draw_index=draw_index,
            events=tuple(events),
            model_fingerprint=self.fingerprint,
        )

    def contract(self) -> dict[str, object]:
        quantiles = _bootstrap_quantiles(
            self.bootstrap_draws,
            km_time_s=np.asarray(self.km_time_s, dtype=float),
            km_survival=np.asarray(self.km_survival, dtype=float),
            race_duration_s=self.race_duration_s,
        )
        return {
            "schema_version": TRAFFIC_SCHEMA_VERSION,
            "model": "target_scaled_field_survival_power_nhpp_empirical_marks_v3",
            "config_path": self.config_path,
            "fingerprint": self.fingerprint,
            "race_duration_s": self.race_duration_s,
            "source_race_id": self.source_race_id,
            "target_race_id": self.target_race_id,
            "source_event_count": len(self.source_observations),
            "source_observer_stream_count": len(self.source_streams),
            "source_observer_exposure_h": self.source_observer_exposure_s / 3600.0,
            "target_event_count": len(self.observations),
            "target_observer_stream_count": len(self.streams),
            "target_observer_exposure_h": self.total_observer_exposure_s / 3600.0,
            # Backward-friendly aliases used by the report augmentation.
            "event_count": len(self.observations),
            "observer_exposure_h": self.total_observer_exposure_s / 3600.0,
            "field_record_count": len(self.field_records),
            "dropout_offset_fraction": self.dropout_offset_fraction,
            "initial_rate_per_hour": self.initial_rate_per_hour,
            "source_initial_rate_per_hour": self.source_initial_rate_per_hour,
            "source_stream_initial_rates_per_hour": {
                key: value for key, value in self.source_stream_initial_rates_per_hour
            },
            "field_exponent": self.field_exponent,
            "average_rate_per_hour_4h": self.average_rate_per_hour(),
            "bootstrap_draw_count": len(self.bootstrap_draws),
            "bootstrap_quantiles": quantiles,
            "time_rescaling_ks_statistic": self.time_rescaling_ks_statistic,
            "time_rescaling_ks_pvalue": self.time_rescaling_ks_pvalue,
            "target_time_rescaling_ks_statistic": self.target_time_rescaling_ks_statistic,
            "target_time_rescaling_ks_pvalue": self.target_time_rescaling_ks_pvalue,
            "alternative_exponential": {
                "initial_rate_per_hour": self.exponential_initial_rate_per_hour,
                "decay_per_s": self.exponential_decay_per_s,
                "log_likelihood": self.exponential_log_likelihood,
                "aic": 2.0 * (len(self.source_streams) + 1) - 2.0 * self.exponential_log_likelihood,
            },
            "selected_source_field_survival_model": {
                "log_likelihood": self.log_likelihood,
                "aic": 2.0 * (len(self.source_streams) + 1) - 2.0 * self.log_likelihood,
            },
            "target_rate_fit": {
                "log_likelihood": self.target_log_likelihood,
                "initial_rate_per_hour": self.initial_rate_per_hour,
                "shared_field_exponent": self.field_exponent,
            },
            "homogeneous_source_poisson": {
                "pooled_rate_per_hour": self.homogeneous_rate_per_hour,
                "stream_rates_per_hour": {
                    stream.identifier: 3600.0 * len(stream.observations) / stream.exposure_s
                    for stream in self.source_streams
                },
                "log_likelihood": self.homogeneous_log_likelihood,
                "aic": 2.0 * len(self.source_streams) - 2.0 * self.homogeneous_log_likelihood,
            },
            "source_rate_scaling": "one nuisance initial-rate scale per Arizona exposure stream; shared time-decay exponent",
            "mark_sampling": "joint empirical resampling of target-race observed event rows",
            "confidence_handling": "retained for audit; no invented numeric noise model",
            "transfer_assumption": (
                "Arizona field-survival curve is the transferred proxy; Maryland rate level and marks are measured directly, and the survival-to-traffic exponent is fitted jointly"
            ),
        }


def traffic_realization_from_mapping(raw: Mapping[str, Any] | None) -> TrafficRealization | None:
    if not isinstance(raw, Mapping):
        return None
    events_raw = raw.get("events", ())
    events: list[TrafficEvent] = []
    if isinstance(events_raw, Sequence) and not isinstance(events_raw, (str, bytes)):
        for item in events_raw:
            if not isinstance(item, Mapping):
                continue
            ordinal = item.get("baseline_speed_ordinal")
            events.append(
                TrafficEvent(
                    identifier=str(item.get("id", f"traffic_{len(events)+1:02d}")),
                    start_offset_s=float(item.get("start_offset_s", 0.0)),
                    duration_s=float(item.get("duration_s", 0.0)),
                    retained_speed_fraction=float(item.get("retained_speed_fraction", 1.0)),
                    event_type=str(item.get("event_type", "slowdown")),
                    source_stream_id=str(item.get("source_stream_id", "")),
                    source_row=int(item.get("source_row", 0)),
                    source_confidence=str(item.get("source_confidence", "")),
                    baseline_speed_ordinal=(None if ordinal is None else float(ordinal)),
                )
            )
    return TrafficRealization(
        scenario_seed=int(raw.get("scenario_seed", 0)),
        lap_start_race_time_s=float(raw.get("lap_start_race_time_s", 0.0)),
        race_duration_s=float(raw.get("race_duration_s", 14400.0)),
        initial_rate_per_hour=float(raw.get("initial_rate_per_hour", 0.0)),
        field_exponent=float(raw.get("field_exponent", 0.0)),
        calibration_draw_index=int(raw.get("calibration_draw_index", -1)),
        events=tuple(events),
        model_fingerprint=str(raw.get("model_fingerprint", "")),
    )


def traffic_limit_state(
    *,
    traffic: TrafficRealization | None,
    reference: TrafficReferenceProfile | None,
    lap_time_s: float,
    distance_m: float,
) -> tuple[float, float]:
    """Return absolute traffic speed ceiling and retained fraction at one state."""

    if traffic is None:
        return float("inf"), 1.0
    if reference is None:
        raise TrafficDataError("traffic realization requires a shared traffic-free reference profile")
    retained = traffic.retained_fraction(lap_time_s)
    if retained >= 1.0:
        return float("inf"), 1.0
    ceiling = reference.speed_at_distance(distance_m) * max(0.0, min(1.0, retained))
    return float(ceiling), float(retained)


def traffic_model_from_project(project_root: Path) -> TrafficCalibration | None:
    path = project_root.resolve() / "track" / "traffic.toml"
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    table = raw.get("traffic")
    if not isinstance(table, Mapping):
        raise TrafficDataError(f"{path} requires a [traffic] table")
    if not bool(table.get("enabled", True)):
        return None
    return calibrate_traffic(project_root=project_root.resolve(), config_path=path, raw=table)


def calibrate_traffic(*, project_root: Path, config_path: Path, raw: Mapping[str, Any]) -> TrafficCalibration:
    schema_version = int(raw.get("schema_version", TRAFFIC_SCHEMA_VERSION))
    if schema_version != TRAFFIC_SCHEMA_VERSION:
        raise TrafficDataError(
            f"traffic schema_version must be {TRAFFIC_SCHEMA_VERSION}; got {schema_version}"
        )
    arrival_model = str(raw.get("arrival_model", "target_scaled_field_survival_power_nhpp")).strip()
    if arrival_model != "target_scaled_field_survival_power_nhpp":
        raise TrafficDataError(f"unsupported traffic arrival_model {arrival_model!r}")
    mark_model = str(raw.get("mark_model", "target_joint_empirical_resampling")).strip()
    if mark_model != "target_joint_empirical_resampling":
        raise TrafficDataError(f"unsupported traffic mark_model {mark_model!r}")
    race_duration = _positive(raw.get("race_duration_s", 14400.0), "traffic.race_duration_s")
    bootstrap_replicates = int(raw.get("bootstrap_replicates", 512))
    if bootstrap_replicates < 0:
        raise TrafficDataError("traffic.bootstrap_replicates must be nonnegative")
    bootstrap_seed = int(raw.get("bootstrap_seed", 20260814))
    source_race_id = str(raw.get("source_race_id", "source")).strip() or "source"
    target_race_id = str(raw.get("target_race_id", source_race_id)).strip() or source_race_id

    source_streams, source_observations, source_hashes = _load_stream_group(
        project_root=project_root,
        rows=raw.get("source_observation_streams", raw.get("observation_streams", ())),
        label="source",
    )
    target_rows = raw.get("target_observation_streams", ())
    if target_rows:
        target_streams, target_observations, target_hashes = _load_stream_group(
            project_root=project_root,
            rows=target_rows,
            label="target",
        )
    else:
        target_streams = list(source_streams)
        target_observations = list(source_observations)
        target_hashes = {}

    population_raw = raw.get("field_population")
    if not isinstance(population_raw, Mapping):
        raise TrafficDataError("traffic requires [traffic.field_population]")
    population_file_text = str(population_raw.get("file", "")).strip()
    population_file = _project_file(project_root, population_file_text)
    field_records = _read_field_population(population_file)
    if len(field_records) < 10:
        raise TrafficDataError("traffic field population requires at least 10 records")
    dropout_offset_fraction = float(population_raw.get("dropout_offset_fraction", 0.5))
    if not isfinite(dropout_offset_fraction) or not 0.0 <= dropout_offset_fraction <= 1.0:
        raise TrafficDataError("traffic.field_population.dropout_offset_fraction must lie in [0,1]")

    km_t, km_s = _kaplan_meier(
        field_records,
        race_duration_s=race_duration,
        dropout_offset_fraction=dropout_offset_fraction,
    )
    source_windows = {
        stream.identifier: (stream.exposure_start_s, stream.exposure_end_s)
        for stream in source_streams
    }
    source_times = np.asarray([event.time_s for event in source_observations], dtype=float)
    source_times_by_stream = {
        stream.identifier: np.asarray(
            [event.time_s for event in stream.observations], dtype=float
        )
        for stream in source_streams
    }
    # CWRU and ETS are different traffic-exposure streams. Giving each source
    # stream its own nuisance rate scale prevents a high-rate short stream from
    # masquerading as evidence for stronger race-time decay. Only the within-stream
    # timing pattern identifies the decay exponent.
    _, _, source_comparison_ll = _fit_source_stream_scaled_field_nhpp(
        event_times_by_stream=source_times_by_stream,
        km_time_s=km_t,
        km_survival=km_s,
        exposure_windows=source_windows,
    )
    target_windows = {
        stream.identifier: (stream.exposure_start_s, stream.exposure_end_s)
        for stream in target_streams
    }
    target_times = np.asarray([event.time_s for event in target_observations], dtype=float)
    source_rates_per_s, target_rate_per_s, exponent, source_ll, target_ll = (
        _fit_shared_exponent_nhpp(
            source_event_times_by_stream=source_times_by_stream,
            target_event_times=target_times,
            km_time_s=km_t,
            km_survival=km_s,
            source_exposure_windows=source_windows,
            target_exposure_windows=target_windows,
        )
    )

    source_exposure = sum(stream.exposure_s for stream in source_streams)
    homogeneous_rates_per_s = {
        stream.identifier: len(stream.observations) / stream.exposure_s
        for stream in source_streams
    }
    homogeneous_rate = len(source_times) / source_exposure
    homogeneous_ll = sum(
        len(stream.observations) * log(homogeneous_rates_per_s[stream.identifier])
        - homogeneous_rates_per_s[stream.identifier] * stream.exposure_s
        for stream in source_streams
    )
    exp_rates_per_s, exp_decay, exp_ll = _fit_source_stream_scaled_exponential_nhpp(
        source_streams
    )
    exp_rate = sum(
        exp_rates_per_s[stream.identifier] * stream.exposure_s
        for stream in source_streams
    ) / source_exposure
    source_rate_per_s = sum(
        source_rates_per_s[stream.identifier] * stream.exposure_s
        for stream in source_streams
    ) / source_exposure
    ks_stat, ks_pvalue = _time_rescaling_test(
        streams=source_streams,
        initial_rate_per_s=source_rates_per_s,
        exponent=exponent,
        km_time_s=km_t,
        km_survival=km_s,
    )
    target_ks_stat, target_ks_pvalue = _time_rescaling_test(
        streams=target_streams,
        initial_rate_per_s=target_rate_per_s,
        exponent=exponent,
        km_time_s=km_t,
        km_survival=km_s,
    )

    bootstrap = _bootstrap_calibration(
        records=field_records,
        source_streams=source_streams,
        target_streams=target_streams,
        source_initial_rates_per_s=source_rates_per_s,
        target_initial_rate_per_s=target_rate_per_s,
        field_exponent=exponent,
        km_time_s=km_t,
        km_survival=km_s,
        race_duration_s=race_duration,
        dropout_offset_fraction=dropout_offset_fraction,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    source_hashes.update(target_hashes)
    source_hashes[population_file_text] = _sha256(population_file)
    contract_source = {
        "schema": TRAFFIC_SCHEMA_VERSION,
        "race_duration_s": race_duration,
        "source_race_id": source_race_id,
        "target_race_id": target_race_id,
        "source_hashes": source_hashes,
        "source_stream_windows": source_windows,
        "target_stream_windows": target_windows,
        "dropout_offset_fraction": dropout_offset_fraction,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "source_initial_rate_per_hour": source_rate_per_s * 3600.0,
        "source_stream_initial_rates_per_hour": {
            key: value * 3600.0 for key, value in source_rates_per_s.items()
        },
        "target_initial_rate_per_hour": target_rate_per_s * 3600.0,
        "field_exponent": exponent,
        "bootstrap": [draw.serializable() for draw in bootstrap],
    }
    fingerprint = hashlib.sha256(
        json.dumps(contract_source, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()

    return TrafficCalibration(
        config_path=str(config_path.relative_to(project_root)),
        race_duration_s=race_duration,
        streams=tuple(target_streams),
        source_streams=tuple(source_streams),
        field_records=tuple(field_records),
        dropout_offset_fraction=dropout_offset_fraction,
        km_time_s=tuple(float(x) for x in km_t),
        km_survival=tuple(float(x) for x in km_s),
        initial_rate_per_hour=float(target_rate_per_s * 3600.0),
        source_initial_rate_per_hour=float(source_rate_per_s * 3600.0),
        source_stream_initial_rates_per_hour=tuple(
            sorted((key, float(value * 3600.0)) for key, value in source_rates_per_s.items())
        ),
        field_exponent=float(exponent),
        log_likelihood=float(source_comparison_ll),
        target_log_likelihood=float(target_ll),
        exponential_initial_rate_per_hour=float(exp_rate * 3600.0),
        exponential_decay_per_s=float(exp_decay),
        exponential_log_likelihood=float(exp_ll),
        homogeneous_rate_per_hour=float(homogeneous_rate * 3600.0),
        homogeneous_log_likelihood=float(homogeneous_ll),
        bootstrap_draws=tuple(bootstrap),
        observations=tuple(target_observations),
        source_observations=tuple(source_observations),
        time_rescaling_ks_statistic=float(ks_stat),
        time_rescaling_ks_pvalue=float(ks_pvalue),
        target_time_rescaling_ks_statistic=float(target_ks_stat),
        target_time_rescaling_ks_pvalue=float(target_ks_pvalue),
        source_race_id=source_race_id,
        target_race_id=target_race_id,
        fingerprint=fingerprint,
    )


def _load_stream_group(
    *, project_root: Path, rows: Any, label: str
) -> tuple[list[ObservationStream], list[TrafficObservation], dict[str, str]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise TrafficDataError(f"traffic requires [[traffic.{label}_observation_streams]]")
    streams: list[ObservationStream] = []
    observations: list[TrafficObservation] = []
    source_hashes: dict[str, str] = {}
    for item in rows:
        if not isinstance(item, Mapping):
            raise TrafficDataError(f"traffic {label} observation stream must be a table")
        identifier = str(item.get("id", "")).strip()
        if not identifier:
            raise TrafficDataError(f"traffic {label} observation stream requires id")
        file_text = str(item.get("file", "")).strip()
        source = _project_file(project_root, file_text)
        start = float(item.get("exposure_start_s", 0.0))
        end = _positive(item.get("exposure_end_s"), f"traffic {label} stream {identifier} exposure_end_s")
        if start < 0.0 or end <= start:
            raise TrafficDataError(f"traffic {label} stream {identifier} has invalid exposure window")
        parsed = _read_observation_csv(
            source,
            stream_id=identifier,
            exposure_start_s=start,
            exposure_end_s=end,
        )
        if not parsed:
            raise TrafficDataError(f"traffic {label} stream {identifier} has no usable events")
        streams.append(ObservationStream(identifier, file_text, start, end, tuple(parsed)))
        observations.extend(parsed)
        source_hashes[file_text] = _sha256(source)
    return streams, observations, source_hashes


def _read_observation_csv(
    path: Path,
    *,
    stream_id: str,
    exposure_start_s: float,
    exposure_end_s: float,
) -> list[TrafficObservation]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []

    headers = set(rows[0])
    normalized_schema = {
        "event_timestamp",
        "event_type",
        "duration_s",
        "retained_speed_fraction",
        "confidence",
    }
    raw_schema = {
        "Event timestamp",
        "Traffic Type",
        "Event Duration",
        "Event Speed Reduction(Estimated % ofspeed without traffic)",
        "Speed if there wasn't traffic(4 is max speed)",
        "Confidence in columns E to G",
    }
    if normalized_schema.issubset(headers):
        timestamp_key = "event_timestamp"
        type_key = "event_type"
        duration_key = "duration_s"
        retained_key = "retained_speed_fraction"
        ordinal_key = None
        confidence_key = "confidence"
        normalized = True
    elif raw_schema.issubset(headers):
        timestamp_key = "Event timestamp"
        type_key = "Traffic Type"
        duration_key = "Event Duration"
        retained_key = "Event Speed Reduction(Estimated % ofspeed without traffic)"
        ordinal_key = "Speed if there wasn't traffic(4 is max speed)"
        confidence_key = "Confidence in columns E to G"
        normalized = False
    else:
        expected = sorted(normalized_schema | raw_schema)
        raise TrafficDataError(
            f"traffic CSV {path} does not match a supported schema; expected normalized columns or cleaned observer columns"
        )

    parsed_times = _parse_stream_times([row.get(timestamp_key, "") for row in rows])
    observations: list[TrafficObservation] = []
    for index, (row, time_s) in enumerate(zip(rows, parsed_times), 2):
        if time_s is None:
            continue
        if not (exposure_start_s <= time_s <= exposure_end_s + 1.0e-9):
            raise TrafficDataError(
                f"traffic event {path.name}:{index} time {time_s:.1f}s lies outside declared exposure"
            )
        event_type = str(row.get(type_key, "")).strip().lower().replace("_", " ")
        if event_type not in {"slowdown", "yellow flag", "full stop"}:
            raise TrafficDataError(f"unsupported traffic type {event_type!r} at {path.name}:{index}")
        duration = _duration_seconds(row.get(duration_key))
        raw_retained = _float(row.get(retained_key), f"{path.name}:{index} retained fraction")
        # The cleaned Arizona heading says "reduction", but the slowdown/yellow
        # annotations behave as retained speed fractions. Full-stop rows are the
        # known encoding exception: source cell 1.0 means physically retained=0.
        retained = (
            raw_retained
            if normalized
            else (0.0 if event_type == "full stop" else raw_retained)
        )
        if event_type == "full stop":
            retained = 0.0
        if not 0.0 <= retained <= 1.0:
            raise TrafficDataError(f"retained speed fraction outside [0,1] at {path.name}:{index}")
        ordinal = None
        if ordinal_key is not None:
            ordinal_raw = str(row.get(ordinal_key, "")).strip()
            ordinal = None if not ordinal_raw else float(ordinal_raw)
        confidence = str(row.get(confidence_key, "")).strip().lower()
        if confidence not in {"high", "medium", "low"}:
            raise TrafficDataError(f"unsupported confidence {confidence!r} at {path.name}:{index}")
        observations.append(
            TrafficObservation(
                stream_id=stream_id,
                source_row=index,
                time_s=float(time_s),
                event_type=event_type,
                duration_s=duration,
                retained_speed_fraction=float(retained),
                baseline_speed_ordinal=ordinal,
                confidence=confidence,
                source_file=path.name,
            )
        )
    observations.sort(key=lambda item: item.time_s)
    return observations


def _parse_stream_times(values: Sequence[Any]) -> list[float | None]:
    """Normalize the mixed spreadsheet time encodings in the cleaned evidence.

    The files contain normal ``m:ss`` values, true ``h:mm:ss`` values after one
    hour, and Excel-export artefacts such as ``29:39:00`` which mean 29m39s.  A
    two-field ``1:14`` after true one-hour timestamps is interpreted as 1h14m;
    before that point a value such as ``1:45`` remains 1m45s.
    """

    result: list[float | None] = []
    seen_true_hour = False
    for raw in values:
        text = "" if raw is None else str(raw).strip()
        if not text or text.lower() == "nan":
            result.append(None)
            continue
        parts = text.split(":")
        try:
            numbers = [int(float(part)) for part in parts]
        except ValueError as exc:
            raise TrafficDataError(f"invalid traffic timestamp {text!r}") from exc
        if len(numbers) == 3:
            first, second, third = numbers
            if first >= 2:
                # Spreadsheet exported m:ss as h:mm:ss-looking text.
                value = first * 60.0 + second + third / 60.0
            else:
                value = first * 3600.0 + second * 60.0 + third
                seen_true_hour = seen_true_hour or first >= 1
        elif len(numbers) == 2:
            first, second = numbers
            if seen_true_hour and first == 1:
                value = first * 3600.0 + second * 60.0
            else:
                value = first * 60.0 + second
        else:
            raise TrafficDataError(f"invalid traffic timestamp {text!r}")
        result.append(float(value))
    return result


def _read_field_population(path: Path) -> list[FieldPopulationRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"observed_active_time_s", "censored"}
    if not rows or not required.issubset(rows[0]):
        raise TrafficDataError(f"field-population CSV {path} requires columns {sorted(required)}")
    output: list[FieldPopulationRecord] = []
    for index, row in enumerate(rows, 2):
        time = _positive(row.get("observed_active_time_s"), f"{path.name}:{index} active time")
        last_lap_text = str(row.get("last_lap_duration_s", "0")).strip() or "0"
        last_lap = _float(last_lap_text, f"{path.name}:{index} last lap duration")
        if last_lap < 0.0:
            raise TrafficDataError(f"last_lap_duration_s cannot be negative at {path.name}:{index}")
        censored_text = str(row.get("censored", "")).strip().lower()
        if censored_text in {"true", "1", "yes"}:
            censored = True
        elif censored_text in {"false", "0", "no"}:
            censored = False
        else:
            raise TrafficDataError(f"invalid censored flag at {path.name}:{index}")
        output.append(FieldPopulationRecord(time, last_lap, censored, index))
    return output


def _kaplan_meier(
    records: Sequence[FieldPopulationRecord],
    *,
    race_duration_s: float,
    dropout_offset_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    # The last completed lap precedes the actual retirement for a non-finisher.
    # Shift failures by the declared fraction of one final lap; censored finishers
    # are not shifted. This preserves the physical interpretation used in the
    # original Arizona lap-data reduction.
    adjusted: list[tuple[float, bool]] = []
    for record in records:
        time = record.observed_active_time_s
        if not record.censored:
            time += dropout_offset_fraction * record.last_lap_duration_s
        adjusted.append((min(max(float(time), 0.0), race_duration_s), record.censored))

    at_risk = len(adjusted)
    survival = 1.0
    out_t = [0.0]
    out_s = [1.0]
    for time in sorted({row[0] for row in adjusted}):
        failures = sum(row_time == time and not censored for row_time, censored in adjusted)
        withdrawals = sum(row_time == time and censored for row_time, censored in adjusted)
        if failures and at_risk > 0:
            survival *= 1.0 - failures / at_risk
            if time > out_t[-1]:
                out_t.append(float(time))
                out_s.append(float(survival))
            else:
                out_s[-1] = float(survival)
        at_risk -= failures + withdrawals
    return np.asarray(out_t, dtype=float), np.asarray(out_s, dtype=float)


def _survival_at(time_s: Any, km_time_s: np.ndarray, km_survival: np.ndarray) -> np.ndarray:
    values = np.asarray(time_s, dtype=float)
    indices = np.searchsorted(km_time_s, values, side="right") - 1
    indices = np.clip(indices, 0, len(km_survival) - 1)
    return km_survival[indices]


def _exposure_integral_survival_power(
    *,
    km_time_s: np.ndarray,
    km_survival: np.ndarray,
    exposure_windows: Mapping[str, tuple[float, float]],
    exponent: float,
) -> float:
    total = 0.0
    for start, end in exposure_windows.values():
        total += _integral_survival_power(km_time_s, km_survival, start, end, exponent)
    return total


def _integral_survival_power(
    km_time_s: np.ndarray,
    km_survival: np.ndarray,
    start_s: float,
    end_s: float,
    exponent: float,
) -> float:
    if end_s <= start_s:
        return 0.0
    breaks = [float(start_s)]
    breaks.extend(float(value) for value in km_time_s if start_s < value < end_s)
    breaks.append(float(end_s))
    total = 0.0
    for start, end in zip(breaks[:-1], breaks[1:]):
        midpoint = 0.5 * (start + end)
        survival = float(_survival_at(midpoint, km_time_s, km_survival))
        total += (end - start) * survival ** exponent
    return total


def _fit_source_stream_scaled_field_nhpp(
    *,
    event_times_by_stream: Mapping[str, np.ndarray],
    km_time_s: np.ndarray,
    km_survival: np.ndarray,
    exposure_windows: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, float], float, float]:
    """Source-only field-survival fit with one nuisance rate per exposure stream."""

    if sum(len(values) for values in event_times_by_stream.values()) < 2:
        raise TrafficDataError("at least two traffic events are required")

    def profile(exponent: float) -> tuple[float, dict[str, float]]:
        total_ll = 0.0
        rates: dict[str, float] = {}
        for stream_id, times in event_times_by_stream.items():
            if stream_id not in exposure_windows or len(times) < 1:
                continue
            start, end = exposure_windows[stream_id]
            integral = _integral_survival_power(
                km_time_s, km_survival, start, end, exponent
            )
            if integral <= 0.0:
                return float("-inf"), {}
            rate = len(times) / integral
            event_survival = np.maximum(
                _survival_at(times, km_time_s, km_survival), 1.0e-12
            )
            total_ll += (
                len(times) * log(rate)
                + exponent * float(np.sum(np.log(event_survival)))
                - rate * integral
            )
            rates[stream_id] = float(rate)
        return float(total_ll), rates

    result = minimize_scalar(
        lambda exponent: -profile(exponent)[0], bounds=(0.0, 30.0), method="bounded"
    )
    if not result.success:
        raise TrafficDataError("traffic field-survival NHPP calibration did not converge")
    ll, rates = profile(float(result.x))
    return rates, float(result.x), ll


def _fit_shared_exponent_nhpp(
    *,
    source_event_times_by_stream: Mapping[str, np.ndarray],
    target_event_times: np.ndarray,
    km_time_s: np.ndarray,
    km_survival: np.ndarray,
    source_exposure_windows: Mapping[str, tuple[float, float]],
    target_exposure_windows: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, float], float, float, float, float]:
    """Fit a shared field-decay exponent without confounding source stream rates.

    Each Arizona source stream receives its own nuisance intercept.  The Maryland
    target stream(s) share one target rate scale because that is the production
    distribution we need to generate.  This makes the decay exponent depend on
    within-stream event timing rather than on CWRU/ETS having different average
    encounter rates.
    """

    if sum(len(values) for values in source_event_times_by_stream.values()) < 2:
        raise TrafficDataError("shared traffic fit requires source observations")
    if len(target_event_times) < 1:
        raise TrafficDataError("shared traffic fit requires target observations")

    def source_profile(exponent: float) -> tuple[float, dict[str, float]]:
        total_ll = 0.0
        rates: dict[str, float] = {}
        for stream_id, times in source_event_times_by_stream.items():
            if len(times) < 1:
                continue
            start, end = source_exposure_windows[stream_id]
            integral = _integral_survival_power(
                km_time_s, km_survival, start, end, exponent
            )
            if integral <= 0.0:
                return float("-inf"), {}
            rate = len(times) / integral
            event_survival = np.maximum(
                _survival_at(times, km_time_s, km_survival), 1.0e-12
            )
            total_ll += (
                len(times) * log(rate)
                + exponent * float(np.sum(np.log(event_survival)))
                - rate * integral
            )
            rates[stream_id] = float(rate)
        return float(total_ll), rates

    target_survival = np.maximum(
        _survival_at(target_event_times, km_time_s, km_survival), 1.0e-12
    )

    def target_profile(exponent: float) -> tuple[float, float]:
        integral = _exposure_integral_survival_power(
            km_time_s=km_time_s,
            km_survival=km_survival,
            exposure_windows=target_exposure_windows,
            exponent=exponent,
        )
        if integral <= 0.0:
            return float("-inf"), 0.0
        rate = len(target_event_times) / integral
        ll = (
            len(target_event_times) * log(rate)
            + exponent * float(np.sum(np.log(target_survival)))
            - rate * integral
        )
        return float(ll), float(rate)

    def profile(exponent: float) -> tuple[float, dict[str, float], float, float, float]:
        source_ll, source_rates = source_profile(exponent)
        target_ll, target_rate = target_profile(exponent)
        return source_ll + target_ll, source_rates, target_rate, source_ll, target_ll

    result = minimize_scalar(
        lambda exponent: -profile(exponent)[0], bounds=(0.0, 30.0), method="bounded"
    )
    if not result.success:
        raise TrafficDataError("shared source/target traffic NHPP calibration did not converge")
    _, source_rates, target_rate, source_ll, target_ll = profile(float(result.x))
    return source_rates, target_rate, float(result.x), source_ll, target_ll


def _fit_rate_given_exponent(
    *,
    event_times: np.ndarray,
    km_time_s: np.ndarray,
    km_survival: np.ndarray,
    exposure_windows: Mapping[str, tuple[float, float]],
    exponent: float,
) -> tuple[float, float]:
    if len(event_times) < 1:
        raise TrafficDataError("target traffic rate requires at least one observed event")
    integral = _exposure_integral_survival_power(
        km_time_s=km_time_s,
        km_survival=km_survival,
        exposure_windows=exposure_windows,
        exponent=exponent,
    )
    if integral <= 0.0:
        raise TrafficDataError("target traffic exposure integral is zero")
    rate = len(event_times) / integral
    event_survival = np.maximum(_survival_at(event_times, km_time_s, km_survival), 1.0e-12)
    ll = (
        len(event_times) * log(rate)
        + exponent * float(np.sum(np.log(event_survival)))
        - rate * integral
    )
    return float(rate), float(ll)


def _fit_source_stream_scaled_exponential_nhpp(
    streams: Sequence[ObservationStream],
) -> tuple[dict[str, float], float, float]:
    """Clock-time comparator with the same per-source-stream intercept freedom."""

    def profile(decay: float) -> tuple[float, dict[str, float]]:
        total_ll = 0.0
        rates: dict[str, float] = {}
        for stream in streams:
            start, end = stream.exposure_start_s, stream.exposure_end_s
            if abs(decay) < 1.0e-12:
                integral = end - start
            else:
                integral = (np.exp(-decay * start) - np.exp(-decay * end)) / decay
            times = np.asarray([event.time_s for event in stream.observations], dtype=float)
            rate = len(times) / integral
            total_ll += (
                len(times) * log(rate)
                - decay * float(np.sum(times))
                - rate * integral
            )
            rates[stream.identifier] = float(rate)
        return float(total_ll), rates

    result = minimize_scalar(
        lambda value: -profile(value)[0], bounds=(0.0, 0.003), method="bounded"
    )
    if not result.success:
        raise TrafficDataError("traffic exponential comparator did not converge")
    ll, rates = profile(float(result.x))
    return rates, float(result.x), ll


def _time_rescaling_test(
    *,
    streams: Sequence[ObservationStream],
    initial_rate_per_s: float | Mapping[str, float],
    exponent: float,
    km_time_s: np.ndarray,
    km_survival: np.ndarray,
) -> tuple[float, float]:
    increments: list[float] = []
    for stream in streams:
        previous = stream.exposure_start_s
        stream_rate = (
            float(initial_rate_per_s[stream.identifier])
            if isinstance(initial_rate_per_s, Mapping)
            else float(initial_rate_per_s)
        )
        for event in sorted(stream.observations, key=lambda item: item.time_s):
            integral = _integral_survival_power(
                km_time_s, km_survival, previous, event.time_s, exponent
            )
            increments.append(stream_rate * integral)
            previous = event.time_s
    if not increments:
        return float("nan"), float("nan")
    result = kstest(np.asarray(increments, dtype=float), "expon")
    return float(result.statistic), float(result.pvalue)


def _bootstrap_calibration(
    *,
    records: Sequence[FieldPopulationRecord],
    source_streams: Sequence[ObservationStream],
    target_streams: Sequence[ObservationStream],
    source_initial_rates_per_s: Mapping[str, float],
    target_initial_rate_per_s: float,
    field_exponent: float,
    km_time_s: np.ndarray,
    km_survival: np.ndarray,
    race_duration_s: float,
    dropout_offset_fraction: float,
    replicates: int,
    seed: int,
) -> list[TrafficCalibrationDraw]:
    if replicates <= 0:
        return []
    rng = np.random.default_rng(seed)
    source_windows = {
        stream.identifier: (stream.exposure_start_s, stream.exposure_end_s)
        for stream in source_streams
    }
    target_windows = {
        stream.identifier: (stream.exposure_start_s, stream.exposure_end_s)
        for stream in target_streams
    }
    draws: list[TrafficCalibrationDraw] = []
    record_array = np.asarray(records, dtype=object)
    for _ in range(replicates):
        indices = rng.integers(0, len(records), size=len(records))
        sampled_records = tuple(record_array[indices])
        boot_t, boot_s = _kaplan_meier(
            sampled_records,
            race_duration_s=race_duration_s,
            dropout_offset_fraction=dropout_offset_fraction,
        )
        source_event_times_by_stream = {
            stream.identifier: _simulate_observation_streams(
                rng=rng,
                streams=(stream,),
                initial_rate_per_s=source_initial_rates_per_s[stream.identifier],
                exponent=field_exponent,
                km_time_s=km_time_s,
                km_survival=km_survival,
            )
            for stream in source_streams
        }
        target_event_times = _simulate_observation_streams(
            rng=rng,
            streams=target_streams,
            initial_rate_per_s=target_initial_rate_per_s,
            exponent=field_exponent,
            km_time_s=km_time_s,
            km_survival=km_survival,
        )
        if (
            sum(len(values) for values in source_event_times_by_stream.values()) < 2
            or len(target_event_times) < 1
        ):
            continue
        try:
            _, target_rate, exponent, _, _ = _fit_shared_exponent_nhpp(
                source_event_times_by_stream=source_event_times_by_stream,
                target_event_times=target_event_times,
                km_time_s=boot_t,
                km_survival=boot_s,
                source_exposure_windows=source_windows,
                target_exposure_windows=target_windows,
            )
        except TrafficDataError:
            continue
        draws.append(TrafficCalibrationDraw(target_rate * 3600.0, exponent))
    if replicates and len(draws) < max(20, int(0.8 * replicates)):
        raise TrafficDataError(
            f"traffic bootstrap produced only {len(draws)}/{replicates} valid calibration draws"
        )
    return draws


def _simulate_observation_streams(
    *,
    rng: np.random.Generator,
    streams: Sequence[ObservationStream],
    initial_rate_per_s: float,
    exponent: float,
    km_time_s: np.ndarray,
    km_survival: np.ndarray,
) -> np.ndarray:
    times: list[float] = []
    for stream in streams:
        breaks = [stream.exposure_start_s]
        breaks.extend(float(value) for value in km_time_s if stream.exposure_start_s < value < stream.exposure_end_s)
        breaks.append(stream.exposure_end_s)
        for start, end in zip(breaks[:-1], breaks[1:]):
            survival = float(_survival_at(0.5 * (start + end), km_time_s, km_survival))
            mean = initial_rate_per_s * survival ** exponent * (end - start)
            count = int(rng.poisson(mean))
            if count:
                times.extend(float(value) for value in rng.uniform(start, end, size=count))
    return np.sort(np.asarray(times, dtype=float))


def _draw_nhpp_window(
    *,
    rng: np.random.Generator,
    start_race_time_s: float,
    horizon_s: float,
    initial_rate_per_hour: float,
    field_exponent: float,
    km_time_s: np.ndarray,
    km_survival: np.ndarray,
    race_duration_s: float,
) -> tuple[float, ...]:
    if horizon_s <= 0.0 or initial_rate_per_hour <= 0.0:
        return ()
    absolute_end = start_race_time_s + horizon_s
    breaks = [start_race_time_s]
    breaks.extend(
        float(value)
        for value in km_time_s
        if start_race_time_s < value < min(absolute_end, race_duration_s)
    )
    if start_race_time_s < race_duration_s < absolute_end:
        breaks.append(race_duration_s)
    breaks.append(absolute_end)
    breaks = sorted(set(breaks))
    rate0 = initial_rate_per_hour / 3600.0
    output: list[float] = []
    for start, end in zip(breaks[:-1], breaks[1:]):
        midpoint = min(0.5 * (start + end), race_duration_s)
        survival = float(_survival_at(midpoint, km_time_s, km_survival))
        mean = rate0 * survival ** field_exponent * (end - start)
        count = int(rng.poisson(mean))
        if count:
            output.extend(float(value - start_race_time_s) for value in rng.uniform(start, end, size=count))
    return tuple(sorted(output))


def _bootstrap_quantiles(
    draws: Sequence[TrafficCalibrationDraw],
    *,
    km_time_s: np.ndarray | None = None,
    km_survival: np.ndarray | None = None,
    race_duration_s: float | None = None,
) -> dict[str, Any]:
    if not draws:
        return {}
    levels = (0.025, 0.1, 0.5, 0.9, 0.975)
    rate = np.asarray([draw.initial_rate_per_hour for draw in draws], dtype=float)
    exponent = np.asarray([draw.field_exponent for draw in draws], dtype=float)
    output: dict[str, Any] = {
        "levels": list(levels),
        "initial_rate_per_hour": [float(value) for value in np.quantile(rate, levels)],
        "field_exponent": [float(value) for value in np.quantile(exponent, levels)],
        "correlation": float(np.corrcoef(rate, exponent)[0, 1]),
    }
    if (
        km_time_s is not None
        and km_survival is not None
        and race_duration_s is not None
        and race_duration_s > 0.0
    ):
        average_rates = np.asarray(
            [
                draw.initial_rate_per_hour
                * _integral_survival_power(
                    km_time_s,
                    km_survival,
                    0.0,
                    float(race_duration_s),
                    draw.field_exponent,
                )
                / float(race_duration_s)
                for draw in draws
            ],
            dtype=float,
        )
        output["average_rate_per_hour_4h"] = [
            float(value) for value in np.quantile(average_rates, levels)
        ]
        output["average_rate_per_hour_4h_mean"] = float(np.mean(average_rates))
    return output


def _project_file(project_root: Path, text: str) -> Path:
    if not text:
        raise TrafficDataError("traffic evidence file path is empty")
    candidate = (project_root / Path(text)).resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError as exc:
        raise TrafficDataError(f"traffic evidence path escapes project: {text}") from exc
    if not candidate.is_file():
        raise TrafficDataError(f"traffic evidence file not found: {candidate}")
    return candidate


def _duration_seconds(value: Any) -> float:
    text = "" if value is None else str(value).strip().lower()
    # Most rows are simple values such as ``3s``. One reviewed row records an
    # observed lower bound followed by the annotator's best estimate ("at least
    # 2s ... I'd guess 6s"). The final numeric token is therefore the intended
    # nominal estimate; the free-form wording remains in the source CSV for audit.
    import re

    numbers = re.findall(r"(?:\d+(?:\.\d*)?|\.\d+)", text)
    if not numbers:
        raise TrafficDataError("traffic event duration must contain a numeric estimate")
    return _positive(numbers[-1], "traffic event duration")


def _float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrafficDataError(f"{label} must be numeric") from exc
    if not isfinite(number):
        raise TrafficDataError(f"{label} must be finite")
    return number


def _positive(value: Any, label: str) -> float:
    number = _float(value, label)
    if number <= 0.0:
        raise TrafficDataError(f"{label} must be positive")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
