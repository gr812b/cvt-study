"""Paired scenario generation, gate-lap sampling, and Gaussian-copula correlations."""

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
        self._selected = self._select_inputs()
        self._groups = self._validate_correlations()
        self._gate_samples = _gate_samples(bundle)

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
        seed_sequence = np.random.SeedSequence(self.plan.random_seed)
        children = seed_sequence.spawn(self.plan.replicates)
        return tuple(
            self._draw_one(index, child)
            for index, child in enumerate(children)
        )

    def _draw_one(self, replicate: int, seed_sequence: np.random.SeedSequence) -> ScenarioDraw:
        rng = np.random.default_rng(seed_sequence)
        seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
        uniforms = {item.path: float(rng.random()) for item in self._selected}
        for group in self._groups:
            z = rng.multivariate_normal(
                np.zeros(len(group.members), dtype=float),
                group.matrix,
                check_valid="raise",
            )
            correlated_uniforms = norm.cdf(z)
            for path, uniform in zip(group.members, correlated_uniforms):
                uniforms[path] = float(uniform)

        quantities: dict[str, float] = {}
        choices: dict[str, str] = {}
        for item in self._selected:
            if not is_stochastic(item.value):
                continue
            u = uniforms[item.path]
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
        )

    def _draw_gates(
        self, rng: np.random.Generator
    ) -> tuple[dict[str, float], GateSampleIdentity | None, tuple[str, ...]]:
        if self.plan.mode not in {"measured_track", "all_declared"}:
            return {}, None, ()
        if not self._gate_samples:
            return {}, None, ()
        if self.plan.gate_sampling == "independent":
            values = {
                gate_id: float(rng.choice(tuple(samples.values())))
                for gate_id, samples in self._gate_samples.items()
            }
            return values, None, tuple(sorted(values))
        if self.plan.gate_sampling != "paired_lap":
            raise SamplingError("gate_sampling must be 'paired_lap' or 'independent'.")
        common = set.intersection(*(set(samples) for samples in self._gate_samples.values()))
        values: dict[str, float] = {}
        independently_sampled: list[str] = []
        identity: GateSampleIdentity | None = None
        if common:
            key_list = sorted(common)
            selected_key = key_list[int(rng.integers(0, len(key_list)))]
            identity = GateSampleIdentity(*selected_key)
            for gate_id, samples in self._gate_samples.items():
                values[gate_id] = samples[selected_key]
            return values, identity, ()
        # A bundle assembled from heterogeneous vehicles may have no one lap with
        # evidence at every active gate. Preserve pairing wherever possible by
        # selecting one identity from the largest coverage set, then resample only
        # the gates missing that identity and report them explicitly.
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
                values[gate_id] = samples[selected_key]
            else:
                values[gate_id] = float(rng.choice(tuple(samples.values())))
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
                raise SamplingError(f"Selected uncertainty paths do not exist: {sorted(missing)}")
            rows = [self.registry.by_path[path] for path in self.plan.selected_paths]
            non_structural = [item.path for item in rows if item.category != "structural"]
            if non_structural:
                raise SamplingError(
                    "selected_structural may sample only inputs with uncertainty.role='structural': "
                    + ", ".join(non_structural)
                )
        else:
            raise SamplingError(f"Unsupported sampling mode {self.plan.mode!r}.")
        return tuple(item for item in rows if item.path not in excluded)

    def _validate_correlations(self) -> tuple[CorrelationGroup, ...]:
        selected_by_path = {item.path: item for item in self._selected if is_stochastic(item.value)}
        used: set[str] = set()
        groups: list[CorrelationGroup] = []
        for group in self.plan.correlation_groups:
            if len(group.members) < 2:
                raise SamplingError(f"Correlation group {group.identifier!r} needs at least two members.")
            if len(group.members) != len(set(group.members)):
                raise SamplingError(f"Correlation group {group.identifier!r} repeats a member.")
            missing = set(group.members) - set(selected_by_path)
            if missing:
                raise SamplingError(
                    f"Correlation group {group.identifier!r} references non-sampled paths: {sorted(missing)}"
                )
            overlap = set(group.members) & used
            if overlap:
                raise SamplingError(f"Correlation members may appear in only one group: {sorted(overlap)}")
            matrix = np.asarray(group.matrix, dtype=float)
            expected = (len(group.members), len(group.members))
            if matrix.shape != expected:
                raise SamplingError(
                    f"Correlation matrix {group.identifier!r} must have shape {expected}."
                )
            if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, atol=1e-12):
                raise SamplingError(f"Correlation matrix {group.identifier!r} must be finite and symmetric.")
            if not np.allclose(np.diag(matrix), 1.0, atol=1e-12):
                raise SamplingError(f"Correlation matrix {group.identifier!r} must have unit diagonal.")
            if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
                raise SamplingError(f"Correlation matrix {group.identifier!r} must be positive semidefinite.")
            for path in group.members:
                declared = selected_by_path[path].correlation_group
                if declared not in (None, group.identifier):
                    raise SamplingError(
                        f"Input {path!r} declares correlation_group={declared!r}, not {group.identifier!r}."
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
                + ", ".join(f"{path} -> {group}" for path, group in sorted(undeclared.items()))
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
            raise SamplingError("Every correlation group requires a unique non-empty id.")
        members_raw = item.get("members")
        matrix_raw = item.get("matrix")
        if not isinstance(members_raw, list) or not all(isinstance(x, str) for x in members_raw):
            raise SamplingError(f"Correlation group {identifier!r} members must be strings.")
        groups.append(
            CorrelationGroup(
                identifier=identifier,
                members=tuple(members_raw),
                matrix=np.asarray(matrix_raw, dtype=float),
            )
        )
        identifiers.add(identifier)
    return tuple(groups)


def _gate_samples(bundle: TrackBundle) -> dict[str, dict[tuple[str, int, str, str], float]]:
    rows: dict[str, dict[tuple[str, int, str, str], float]] = {}
    for gate in bundle.active_speed_gates:
        samples = gate["target_speed_distribution"]["samples"]
        parsed: dict[tuple[str, int, str, str], float] = {}
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
            if not np.isfinite(value) or value < 0.0:
                raise SamplingError(
                    f"Gate {gate['id']!r} has an invalid empirical speed {value!r}."
                )
            parsed[key] = value
        if parsed:
            rows[str(gate["id"])] = parsed
    return rows
