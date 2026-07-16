"""Strict validation for the Phase 4 track-bundle contract."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import content_fingerprint
from cvt_track_study.contracts.obstacles import validate_obstacle_model_contract
from .model import CURRENT_TRACK_BUNDLE_SCHEMA, TRACK_BUNDLE_FORMAT, TrackBundleError

_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_ALLOWED_GATE_STATUS = {"accepted", "recommended_review", "must_fix", "rejected", "not_a_candidate"}
_ALLOWED_MODEL_STATUS = {"undeclared", "declared", "not_applicable"}


def validate_track_bundle(data: Mapping[str, Any]) -> None:
    """Validate a parsed track bundle or raise :class:`TrackBundleError`.

    The validator is deliberately independent of the reconstruction package. A
    simulator can therefore trust a loaded bundle without having pandas, GPX, or
    map-matching objects in its call path.
    """

    errors: list[str] = []
    if data.get("format") != TRACK_BUNDLE_FORMAT:
        errors.append(f"format must be {TRACK_BUNDLE_FORMAT!r}")
    _validate_version(data.get("schema_version"), errors)

    for key in (
        "created_utc",
        "generator",
        "identity",
        "coordinate_contract",
        "simulation_contract",
        "evidence",
        "uncertainty_contract",
        "provenance",
    ):
        if key not in data:
            errors.append(f"missing top-level field {key!r}")

    simulation = _mapping(data.get("simulation_contract"), "simulation_contract", errors)
    coordinate = _mapping(data.get("coordinate_contract"), "coordinate_contract", errors)
    identity = _mapping(data.get("identity"), "identity", errors)
    evidence = _mapping(data.get("evidence"), "evidence", errors)

    length = _positive_number(simulation.get("track_length_m"), "simulation_contract.track_length_m", errors)
    if length is not None:
        _validate_coordinate_contract(coordinate, length, errors)
        _validate_centreline(simulation.get("centreline"), length, errors)
        _validate_profile(simulation.get("observed_profile"), length, errors)
        feature_ids = _validate_features(simulation.get("physical_features"), length, errors)
        feature_rows = _sequence(
            simulation.get("physical_features"),
            "simulation_contract.physical_features",
            errors,
        )
        capabilities_here = _mapping(
            simulation.get("capabilities"),
            "simulation_contract.capabilities",
            errors,
        )
        all_models_declared = bool(feature_rows) and all(
            isinstance(row, Mapping)
            and isinstance(row.get("obstacle_model"), Mapping)
            and row["obstacle_model"].get("status") == "declared"
            for row in feature_rows
        )
        if capabilities_here and bool(
            capabilities_here.get("obstacle_models_ready")
        ) != all_models_declared:
            errors.append(
                "capabilities.obstacle_models_ready does not match physical feature declarations"
            )
        group_ids = _validate_groups(
            simulation.get("response_groups"), feature_ids, length, errors
        )
        _validate_gates(simulation.get("speed_gates"), group_ids, length, errors)

    if simulation.get("grade_force_enabled") is not False:
        errors.append(
            "simulation_contract.grade_force_enabled must remain false in schema 1.2.x"
        )
    _validate_grade_screen(simulation.get("grade_screen"), errors)
    capabilities = _mapping(
        simulation.get("capabilities"), "simulation_contract.capabilities", errors
    )
    if capabilities and capabilities.get("grade_force_ready") is not False:
        errors.append("schema 1.2.x does not support grade force")
    if capabilities and capabilities.get("uncertainty_roles_ready") is not True:
        errors.append(
            "schema 1.2.x requires capabilities.uncertainty_roles_ready = true"
        )

    if identity and identity.get("closed_course") is not True:
        errors.append("schema 1.2.x requires identity.closed_course = true")

    gate_method = _mapping(evidence.get("gate_confidence_method"), "evidence.gate_confidence_method", errors)
    if gate_method:
        weights = _mapping(gate_method.get("weights"), "evidence.gate_confidence_method.weights", errors)
        if weights:
            numeric = [_number(value, "gate confidence weight", errors) for value in weights.values()]
            if all(value is not None for value in numeric):
                if not math.isclose(sum(numeric), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
                    errors.append("gate confidence weights must sum to 1.0")

    _reject_non_finite(data, "$", errors)
    declared_fingerprint = data.get("content_fingerprint_sha256")
    if not isinstance(declared_fingerprint, str):
        errors.append("content_fingerprint_sha256 must be present")
    else:
        try:
            computed_fingerprint = content_fingerprint(data)
        except (TypeError, ValueError) as exc:
            errors.append(f"bundle content cannot be canonically serialized: {exc}")
        else:
            if declared_fingerprint != computed_fingerprint:
                errors.append("content_fingerprint_sha256 does not match bundle content")

    if errors:
        raise TrackBundleError("Invalid track bundle:\n- " + "\n- ".join(errors))


def _validate_version(raw: Any, errors: list[str]) -> None:
    if not isinstance(raw, str) or not _SEMVER.match(raw):
        errors.append("schema_version must be a semantic version such as '1.0.0'")
        return
    current = tuple(int(part) for part in CURRENT_TRACK_BUNDLE_SCHEMA.split("."))
    candidate = tuple(int(part) for part in raw.split("."))
    if candidate[0] != current[0]:
        errors.append(
            f"unsupported track-bundle major version {candidate[0]}; reader supports {current[0]}"
        )
    elif candidate[1] != current[1]:
        direction = "newer" if candidate[1] > current[1] else "older"
        errors.append(
            f"track-bundle minor version {raw} is {direction} than supported {CURRENT_TRACK_BUNDLE_SCHEMA}; "
            "this reader intentionally supports only the current minor contract"
        )


def _validate_grade_screen(raw: Any, errors: list[str]) -> None:
    screen = _mapping(raw, "simulation_contract.grade_screen", errors)
    if not screen:
        return
    if screen.get("grade_force_enabled") is not False:
        errors.append("grade_screen.grade_force_enabled must remain false")
    allowed_statuses = {
        "insufficient_elevation_evidence",
        "elevation_not_repeatable",
        "paired_grade_sensitivity_recommended",
        "grade_proxy_immaterial",
    }
    if screen.get("status") not in allowed_statuses:
        errors.append("grade_screen.status is not a supported materiality-screen result")
    if not isinstance(screen.get("spatial_grade_sensitivity_recommended"), bool):
        errors.append(
            "grade_screen.spatial_grade_sensitivity_recommended must be boolean"
        )


def _validate_coordinate_contract(
    coordinate: Mapping[str, Any], length: float, errors: list[str]
) -> None:
    if coordinate.get("coordinate") != "s":
        errors.append("coordinate_contract.coordinate must be 's'")
    if coordinate.get("unit") != "m":
        errors.append("coordinate_contract.unit must be 'm'")
    domain = _mapping(coordinate.get("domain"), "coordinate_contract.domain", errors)
    if domain:
        lower = _number(domain.get("minimum"), "coordinate_contract.domain.minimum", errors)
        upper = _number(domain.get("maximum"), "coordinate_contract.domain.maximum", errors)
        if lower is not None and not math.isclose(lower, 0.0, abs_tol=1.0e-9):
            errors.append("coordinate domain minimum must be 0")
        if upper is not None and not math.isclose(upper, length, rel_tol=0.0, abs_tol=1.0e-6):
            errors.append("coordinate domain maximum must equal track length")


def _validate_centreline(raw: Any, length: float, errors: list[str]) -> None:
    centreline = _mapping(raw, "simulation_contract.centreline", errors)
    if not centreline:
        return
    samples = _sequence(centreline.get("samples"), "simulation_contract.centreline.samples", errors)
    if len(samples) < 2:
        errors.append("centreline requires at least two samples")
        return
    declared_count = centreline.get("sample_count")
    if declared_count != len(samples):
        errors.append("centreline.sample_count does not match samples")
    previous = -math.inf
    for index, sample_raw in enumerate(samples):
        sample = _mapping(sample_raw, f"centreline.samples[{index}]", errors)
        s = _number(sample.get("s_m"), f"centreline.samples[{index}].s_m", errors)
        if s is None:
            continue
        if s <= previous:
            errors.append("centreline s values must be strictly increasing")
            break
        previous = s
        _bounded_coordinate(s, length, f"centreline.samples[{index}].s_m", errors, allow_end=True)
        latitude = _number(sample.get("latitude_deg"), f"centreline.samples[{index}].latitude_deg", errors)
        longitude = _number(sample.get("longitude_deg"), f"centreline.samples[{index}].longitude_deg", errors)
        if latitude is not None and not -90.0 <= latitude <= 90.0:
            errors.append(f"centreline sample {index} latitude is outside [-90, 90]")
        if longitude is not None and not -180.0 <= longitude <= 180.0:
            errors.append(f"centreline sample {index} longitude is outside [-180, 180]")
    first = _number(samples[0].get("s_m"), "centreline first s", errors) if isinstance(samples[0], Mapping) else None
    last = _number(samples[-1].get("s_m"), "centreline last s", errors) if isinstance(samples[-1], Mapping) else None
    if first is not None and not math.isclose(first, 0.0, abs_tol=1.0e-9):
        errors.append("centreline must begin at s=0")
    if last is not None and not math.isclose(last, length, rel_tol=0.0, abs_tol=1.0e-6):
        errors.append("centreline must end at track length")


def _validate_profile(raw: Any, length: float, errors: list[str]) -> None:
    profile = _mapping(raw, "simulation_contract.observed_profile", errors)
    if not profile:
        return
    samples = _sequence(profile.get("samples"), "observed_profile.samples", errors)
    if profile.get("sample_count") != len(samples):
        errors.append("observed_profile.sample_count does not match samples")
    previous = -math.inf
    for index, sample_raw in enumerate(samples):
        sample = _mapping(sample_raw, f"observed_profile.samples[{index}]", errors)
        s = _number(sample.get("s_m"), f"observed_profile.samples[{index}].s_m", errors)
        if s is not None:
            if s <= previous:
                errors.append("observed profile s values must be strictly increasing")
                break
            previous = s
            _bounded_coordinate(s, length, f"observed_profile.samples[{index}].s_m", errors)


def _validate_features(raw: Any, length: float, errors: list[str]) -> set[str]:
    rows = _sequence(raw, "simulation_contract.physical_features", errors)
    ids: set[str] = set()
    sequences: set[int] = set()
    for index, row_raw in enumerate(rows):
        row = _mapping(row_raw, f"physical_features[{index}]", errors)
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            errors.append(f"physical_features[{index}].id must be non-empty text")
        elif identifier in ids:
            errors.append(f"duplicate physical feature id {identifier!r}")
        else:
            ids.add(identifier)
        sequence = row.get("sequence")
        if not isinstance(sequence, int) or sequence <= 0:
            errors.append(f"physical feature {identifier!r} sequence must be a positive integer")
        elif sequence in sequences:
            errors.append(f"duplicate physical feature sequence {sequence}")
        else:
            sequences.add(sequence)
        _validate_interval(row.get("interval"), length, f"physical feature {identifier!r}", errors)
        model = _mapping(row.get("obstacle_model"), f"physical feature {identifier!r}.obstacle_model", errors)
        if model and model.get("status") not in _ALLOWED_MODEL_STATUS:
            errors.append(f"physical feature {identifier!r} has invalid obstacle model status")
        if model.get("status") == "declared":
            try:
                validate_obstacle_model_contract(model)
            except ValueError as exc:
                errors.append(f"physical feature {identifier!r} obstacle model: {exc}")
    return ids


def _validate_groups(
    raw: Any, feature_ids: set[str], length: float, errors: list[str]
) -> set[str]:
    rows = _sequence(raw, "simulation_contract.response_groups", errors)
    ids: set[str] = set()
    for index, row_raw in enumerate(rows):
        row = _mapping(row_raw, f"response_groups[{index}]", errors)
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            errors.append(f"response_groups[{index}].id must be non-empty text")
            continue
        if identifier in ids:
            errors.append(f"duplicate response group id {identifier!r}")
        ids.add(identifier)
        source_ids = _sequence(row.get("source_feature_ids"), f"response group {identifier!r}.source_feature_ids", errors)
        for source_id in source_ids:
            if source_id not in feature_ids:
                errors.append(
                    f"response group {identifier!r} references unknown physical feature {source_id!r}"
                )
        _validate_interval(row.get("interval"), length, f"response group {identifier!r}", errors)
        model = _mapping(row.get("obstacle_model"), f"response group {identifier!r}.obstacle_model", errors)
        if model and model.get("status") not in _ALLOWED_MODEL_STATUS:
            errors.append(f"response group {identifier!r} has invalid obstacle model status")
    return ids


def _validate_gates(raw: Any, group_ids: set[str], length: float, errors: list[str]) -> None:
    rows = _sequence(raw, "simulation_contract.speed_gates", errors)
    ids: set[str] = set()
    for index, row_raw in enumerate(rows):
        row = _mapping(row_raw, f"speed_gates[{index}]", errors)
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            errors.append(f"speed_gates[{index}].id must be non-empty text")
            continue
        if identifier in ids:
            errors.append(f"duplicate speed gate id {identifier!r}")
        ids.add(identifier)
        group_id = row.get("response_group_id")
        if group_id not in group_ids:
            errors.append(f"speed gate {identifier!r} references unknown response group {group_id!r}")
        status = row.get("status")
        if status not in _ALLOWED_GATE_STATUS:
            errors.append(f"speed gate {identifier!r} has invalid status {status!r}")
        gate_type = row.get("gate_type", "entry_speed")
        if gate_type not in {"entry_speed", "sustained_response"}:
            errors.append(
                f"speed gate {identifier!r} has invalid gate_type {gate_type!r}"
            )
        position = _number(row.get("position_s_m"), f"speed gate {identifier!r}.position_s_m", errors)
        if position is not None:
            _bounded_coordinate(position, length, f"speed gate {identifier!r}.position_s_m", errors)
        _validate_interval(row.get("measurement_window"), length, f"speed gate {identifier!r} measurement window", errors)
        distribution = _mapping(row.get("target_speed_distribution"), f"speed gate {identifier!r}.target_speed_distribution", errors)
        samples = _sequence(distribution.get("samples"), f"speed gate {identifier!r} samples", errors)
        if distribution.get("distribution") != "empirical":
            errors.append(f"speed gate {identifier!r} target distribution must be empirical")
        if distribution.get("unit") != "m/s":
            errors.append(f"speed gate {identifier!r} target speed unit must be m/s")
        for sample_index, sample_raw in enumerate(samples):
            sample = _mapping(sample_raw, f"speed gate {identifier!r} sample {sample_index}", errors)
            value = _number(sample.get("value_mps"), f"speed gate {identifier!r} sample value", errors)
            if value is not None and value < 0.0:
                errors.append(f"speed gate {identifier!r} contains a negative speed sample")
        summary = _mapping(distribution.get("summary"), f"speed gate {identifier!r} summary", errors)
        if summary:
            p10 = _number(summary.get("p10_mps"), f"speed gate {identifier!r} p10", errors)
            median = _number(summary.get("median_mps"), f"speed gate {identifier!r} median", errors)
            p90 = _number(summary.get("p90_mps"), f"speed gate {identifier!r} p90", errors)
            if None not in (p10, median, p90) and not (p10 <= median <= p90):
                errors.append(f"speed gate {identifier!r} speed summary is not ordered p10 <= median <= p90")
        active = row.get("active_by_default")
        if not isinstance(active, bool):
            errors.append(f"speed gate {identifier!r}.active_by_default must be boolean")
        elif active and status != "accepted":
            errors.append(f"speed gate {identifier!r} may be active only when status is accepted")
        elif active and not samples:
            errors.append(f"active speed gate {identifier!r} requires empirical samples")


def _validate_interval(raw: Any, length: float, label: str, errors: list[str]) -> None:
    interval = _mapping(raw, f"{label}.interval", errors)
    if not interval:
        return
    start = _number(interval.get("start_s_m"), f"{label}.start_s_m", errors)
    end = _number(interval.get("end_s_m"), f"{label}.end_s_m", errors)
    interval_length = _number(interval.get("length_m"), f"{label}.length_m", errors)
    wraps = interval.get("wraps_start_finish")
    if not isinstance(wraps, bool):
        errors.append(f"{label}.wraps_start_finish must be boolean")
    if start is None or end is None or interval_length is None:
        return
    _bounded_coordinate(start, length, f"{label}.start_s_m", errors)
    _bounded_coordinate(end, length, f"{label}.end_s_m", errors)
    expected = (end - start) % length
    if math.isclose(expected, 0.0, abs_tol=1.0e-9) and interval_length > length - 1.0e-6:
        expected = length
    if not math.isclose(interval_length, expected, rel_tol=0.0, abs_tol=1.0e-5):
        errors.append(f"{label}.length_m is inconsistent with start/end on the closed course")
    expected_wrap = end < start and interval_length > 0.0
    if isinstance(wraps, bool) and wraps != expected_wrap:
        errors.append(f"{label}.wraps_start_finish is inconsistent with start/end")


def _mapping(raw: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return raw


def _sequence(raw: Any, label: str, errors: list[str]) -> Sequence[Any]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        errors.append(f"{label} must be an array")
        return ()
    return raw


def _number(raw: Any, label: str, errors: list[str]) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        errors.append(f"{label} must be numeric")
        return None
    value = float(raw)
    if not math.isfinite(value):
        errors.append(f"{label} must be finite")
        return None
    return value


def _positive_number(raw: Any, label: str, errors: list[str]) -> float | None:
    value = _number(raw, label, errors)
    if value is not None and value <= 0.0:
        errors.append(f"{label} must be greater than zero")
        return None
    return value


def _bounded_coordinate(
    value: float,
    length: float,
    label: str,
    errors: list[str],
    *,
    allow_end: bool = False,
) -> None:
    upper_ok = value <= length + 1.0e-9 if allow_end else value < length
    if value < -1.0e-9 or not upper_ok:
        errors.append(f"{label} must lie within the track coordinate domain")


def _reject_non_finite(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_non_finite(child, f"{path}.{key}", errors)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_non_finite(child, f"{path}[{index}]", errors)
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path} contains a non-finite number")
