"""Paired scenarios with Latin-hypercube marginals and coherent gate errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.stats import norm

from cvt_track_study.bundle import TrackBundle
from cvt_track_study.config.uncertainty import UncertainChoice, UncertainQuantity

from .distributions import choice_from_uniform, is_stochastic, quantity_from_uniform
from .model import GateSampleIdentity, ScenarioDraw
from .registry import InputRegistry, RegisteredInput


class SamplingError(ValueError):
    """Raised when a declared uncertainty study cannot produce valid scenarios."""


@dataclass(frozen=True, slots=True)
class CorrelationGroup:
    identifier: str
    members: tuple[str, ...]
    matrix: np.ndarray


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    mode: str
    replicates: int
    random_seed: int
    selected_paths: tuple[str, ...] = ()
    excluded_paths: tuple[str, ...] = ()
    correlation_groups: tuple[CorrelationGroup, ...] = ()
    gate_sampling: str = "paired_lap"
    sampling_design: str = "latin_hypercube"
    target_vehicle_id: str | None = None


@dataclass(frozen=True, slots=True)
class _GateObservation:
    value_mps: float
    measurement_sigma_mps: float
    speed_certainty: str


class ScenarioSampler:
    def __init__(
        self,
        *,
        registry: InputRegistry,
        bundle: TrackBundle,
        plan: SamplingPlan,
    ) -> None:
        self.registry = registry
        self.bundle = bundle
        self.plan = plan
        if plan.replicates < 1:
            raise SamplingError("Sampling plans require at least one replicate.")
        if plan.gate_sampling not in {"paired_lap", "independent"}:
            raise SamplingError("gate_sampling must be 'paired_lap' or 'independent'.")
        if plan.sampling_design not in {"latin_hypercube", "independent_random"}:
            raise SamplingError(
                "sampling_design must be 'latin_hypercube' or 'independent_random'."
            )
        self._selected = self._select_inputs()
        self._groups = self._validate_correlations()
        self._gate_samples = _gate_samples(
            bundle, target_vehicle_id=plan.target_vehicle_id
        )

    @property
    def sampled_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self._selected if is_stochastic(item.value))

    @property
    def paired_gate_identity_count(self) -> int:
        if not self._gate_samples:
            return 0
        identities = [set(samples) for samples in self._gate_samples.values()]
        return len(set.intersection(*identities)) if identities else 0

    def draw_all(self) -> tuple[ScenarioDraw, ...]:
        uniforms = self._uniform_matrix()
        seed_sequence = np.random.SeedSequence(self.plan.random_seed)
        children = seed_sequence.spawn(self.plan.replicates)
        return tuple(
            self._draw_one(index, child, uniforms[index])
            for index, child in enumerate(children)
        )

    def _uniform_matrix(self) -> list[dict[str, float]]:
        n = self.plan.replicates
        stochastic = [item for item in self._selected if is_stochastic(item.value)]
        if not stochastic:
            return [{} for _ in range(n)]
        rng = np.random.default_rng(self.plan.random_seed ^ 0x4C48535F435654)
        columns: dict[str, np.ndarray] = {}
        for item in stochastic:
            if self.plan.sampling_design == "latin_hypercube":
                strata = (np.arange(n, dtype=float) + rng.random(n)) / n
                columns[item.path] = strata[rng.permutation(n)]
            else:
                columns[item.path] = rng.random(n)

        # Iman-Conover rank reordering preserves every Latin-hypercube marginal
        # while imposing each declared Gaussian-copula rank structure.
        for group in self._groups:
            z = rng.multivariate_normal(
                np.zeros(len(group.members), dtype=float),
                group.matrix,
                size=n,
                check_valid="raise",
            )
            for member_index, path in enumerate(group.members):
                source = np.sort(columns[path])
                ranks = np.argsort(np.argsort(z[:, member_index], kind="mergesort"))
                columns[path] = source[ranks]

        return [
            {path: float(values[index]) for path, values in columns.items()}
            for index in range(n)
        ]

    def _draw_one(
        self,
        replicate: int,
        seed_sequence: np.random.SeedSequence,
        uniforms: Mapping[str, float],
    ) -> ScenarioDraw:
        rng = np.random.default_rng(seed_sequence)
        seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
        quantities: dict[str, float] = {}
        choices: dict[str, str] = {}
        for item in self._selected:
            if not is_stochastic(item.value):
                continue
            u = float(uniforms[item.path])
            if isinstance(item.value, UncertainQuantity):
                quantities[item.path] = quantity_from_uniform(item.value, u)
            elif isinstance(item.value, UncertainChoice):
                choices[item.path] = choice_from_uniform(item.value, u)

        gate_values, gate_identity, independent = self._draw_gates(rng)
        return ScenarioDraw(
            replicate=replicate,
            seed=seed,
            sampling_mode=self.plan.mode,
            quantity_values_si=quantities,
            choice_values=choices,
            gate_target_speeds_mps=gate_values,
            gate_sample_identity=gate_identity,
            independently_sampled_gate_ids=independent,
            sampling_design=(
                "latin_hypercube_rank_correlated"
                if self.plan.sampling_design == "latin_hypercube"
                else "independent_random"
            ),
        )

    def _draw_gates(
        self, rng: np.random.Generator
    ) -> tuple[dict[str, float], GateSampleIdentity | None, tuple[str, ...]]:
        if self.plan.mode not in {"measured_track", "all_declared"}:
            return {}, None, ()
        if not self._gate_samples:
            return {}, None, ()
        if self.plan.gate_sampling == "independent":
            values: dict[str, float] = {}
            for gate_id, samples in self._gate_samples.items():
                keys = tuple(samples)
                observation = samples[keys[int(rng.integers(0, len(keys)))]]
                values[gate_id] = _perturb_observation(observation, float(rng.normal()))
            return values, None, tuple(sorted(values))

        common = set.intersection(*(set(samples) for samples in self._gate_samples.values()))
        values: dict[str, float] = {}
        independently_sampled: list[str] = []
        identity: GateSampleIdentity | None = None
        shared_measurement_error = float(rng.normal())
        if common:
            key_list = sorted(common)
            selected_key = key_list[int(rng.integers(0, len(key_list)))]
            identity = GateSampleIdentity(*selected_key)
            for gate_id, samples in self._gate_samples.items():
                values[gate_id] = _perturb_observation(
                    samples[selected_key], shared_measurement_error
                )
            return values, identity, ()

        coverage: dict[tuple[str, int, str, str], int] = {}
        for samples in self._gate_samples.values():
            for key in samples:
                coverage[key] = coverage.get(key, 0) + 1
        best_count = max(coverage.values())
        candidates = sorted(key for key, count in coverage.items() if count == best_count)
        selected_key = candidates[int(rng.integers(0, len(candidates)))]
        identity = GateSampleIdentity(*selected_key)
        for gate_id, samples in self._gate_samples.items():
            if selected_key in samples:
                values[gate_id] = _perturb_observation(
                    samples[selected_key], shared_measurement_error
                )
            else:
                keys = tuple(samples)
                observation = samples[keys[int(rng.integers(0, len(keys)))]]
                values[gate_id] = _perturb_observation(
                    observation, float(rng.normal())
                )
                independently_sampled.append(gate_id)
        return values, identity, tuple(sorted(independently_sampled))

    def _select_inputs(self) -> tuple[RegisteredInput, ...]:
        excluded = set(self.plan.excluded_paths)
        if self.plan.mode == "nominal":
            return ()
        if self.plan.mode == "measured_track":
            rows = [item for item in self.registry.inputs if item.category == "measured_track"]
        elif self.plan.mode == "all_declared":
            rows = list(self.registry.inputs)
        elif self.plan.mode == "selected_structural":
            selected = set(self.plan.selected_paths)
            missing = selected - set(self.registry.by_path)
            if missing:
                raise SamplingError(
                    f"Selected uncertainty paths do not exist: {sorted(missing)}"
                )
            rows = [self.registry.by_path[path] for path in self.plan.selected_paths]
            non_structural = [item.path for item in rows if item.category != "structural"]
            if non_structural:
                raise SamplingError(
                    "selected_structural may sample only inputs with "
                    "uncertainty.role='structural': " + ", ".join(non_structural)
                )
        else:
            raise SamplingError(f"Unsupported sampling mode {self.plan.mode!r}.")
        return tuple(item for item in rows if item.path not in excluded)

    def _validate_correlations(self) -> tuple[CorrelationGroup, ...]:
        selected_by_path = {
            item.path: item for item in self._selected if is_stochastic(item.value)
        }
        used: set[str] = set()
        groups: list[CorrelationGroup] = []
        for group in self.plan.correlation_groups:
            if len(group.members) < 2:
                raise SamplingError(
                    f"Correlation group {group.identifier!r} needs at least two members."
                )
            if len(group.members) != len(set(group.members)):
                raise SamplingError(
                    f"Correlation group {group.identifier!r} repeats a member."
                )
            missing = set(group.members) - set(selected_by_path)
            if missing:
                raise SamplingError(
                    f"Correlation group {group.identifier!r} references non-sampled paths: "
                    f"{sorted(missing)}"
                )
            overlap = set(group.members) & used
            if overlap:
                raise SamplingError(
                    f"Correlation members may appear in only one group: {sorted(overlap)}"
                )
            matrix = np.asarray(group.matrix, dtype=float)
            expected = (len(group.members), len(group.members))
            if matrix.shape != expected:
                raise SamplingError(
                    f"Correlation matrix {group.identifier!r} must have shape {expected}."
                )
            if not np.all(np.isfinite(matrix)) or not np.allclose(
                matrix, matrix.T, atol=1e-12
            ):
                raise SamplingError(
                    f"Correlation matrix {group.identifier!r} must be finite and symmetric."
                )
            if not np.allclose(np.diag(matrix), 1.0, atol=1e-12):
                raise SamplingError(
                    f"Correlation matrix {group.identifier!r} must have unit diagonal."
                )
            if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
                raise SamplingError(
                    f"Correlation matrix {group.identifier!r} must be positive semidefinite."
                )
            for path in group.members:
                declared = selected_by_path[path].correlation_group
                if declared not in (None, group.identifier):
                    raise SamplingError(
                        f"Input {path!r} declares correlation_group={declared!r}, "
                        f"not {group.identifier!r}."
                    )
            used.update(group.members)
            groups.append(CorrelationGroup(group.identifier, group.members, matrix))
        undeclared = {
            item.path: item.correlation_group
            for item in selected_by_path.values()
            if item.correlation_group and item.path not in used
        }
        if undeclared:
            raise SamplingError(
                "Inputs declare correlation groups but no matching matrix was supplied: "
                + ", ".join(
                    f"{path} -> {group}" for path, group in sorted(undeclared.items())
                )
            )
        return tuple(groups)


def correlation_groups_from_study(raw: Mapping[str, Any]) -> tuple[CorrelationGroup, ...]:
    source = raw.get("correlations", [])
    if source in (None, []):
        return ()
    if not isinstance(source, list):
        raise SamplingError("correlations must be an array of tables.")
    groups: list[CorrelationGroup] = []
    identifiers: set[str] = set()
    for index, item in enumerate(source):
        if not isinstance(item, Mapping):
            raise SamplingError(f"correlations[{index}] must be a table.")
        identifier = str(item.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            raise SamplingError(
                "Every correlation group requires a unique non-empty id."
            )
        members_raw = item.get("members")
        matrix_raw = item.get("matrix")
        if not isinstance(members_raw, list) or not all(
            isinstance(x, str) for x in members_raw
        ):
            raise SamplingError(
                f"Correlation group {identifier!r} members must be strings."
            )
        groups.append(
            CorrelationGroup(
                identifier=identifier,
                members=tuple(members_raw),
                matrix=np.asarray(matrix_raw, dtype=float),
            )
        )
        identifiers.add(identifier)
    return tuple(groups)


def _gate_samples(
    bundle: TrackBundle,
    *,
    target_vehicle_id: str | None = None,
) -> dict[str, dict[tuple[str, int, str, str], _GateObservation]]:
    rows: dict[str, dict[tuple[str, int, str, str], _GateObservation]] = {}
    for gate in bundle.active_speed_gates:
        samples = list(gate["target_speed_distribution"]["samples"])
        if target_vehicle_id is not None:
            matching = [
                sample
                for sample in samples
                if str(sample.get("vehicle_id", "")) == str(target_vehicle_id)
            ]
            if matching:
                samples = matching
        parsed: dict[tuple[str, int, str, str], _GateObservation] = {}
        for sample in samples:
            key = (
                str(sample["run_id"]),
                int(sample["lap_id"]),
                str(sample["vehicle_id"]),
                str(sample["driver_id"]),
            )
            if key in parsed:
                raise SamplingError(
                    f"Gate {gate['id']!r} repeats empirical identity {key!r}."
                )
            value = float(sample["value_mps"])
            sigma = float(sample.get("measurement_standard_deviation_mps", 0.0))
            if not np.isfinite(value) or value < 0.0:
                raise SamplingError(
                    f"Gate {gate['id']!r} has an invalid empirical speed {value!r}."
                )
            if not np.isfinite(sigma) or sigma < 0.0:
                raise SamplingError(
                    f"Gate {gate['id']!r} has an invalid measurement sigma {sigma!r}."
                )
            parsed[key] = _GateObservation(
                value_mps=value,
                measurement_sigma_mps=sigma,
                speed_certainty=str(sample.get("speed_certainty", "legacy_unspecified")),
            )
        if parsed:
            rows[str(gate["id"])] = parsed
    return rows


def _perturb_observation(observation: _GateObservation, z: float) -> float:
    return max(
        0.0,
        float(observation.value_mps + observation.measurement_sigma_mps * z),
    )
