"""Select a small, auditable set of worlds from a completed uncertainty study.

The reducer never invents an averaged scenario.  It selects actual completed
scenarios from ``scenario_draws.jsonl`` and keeps their exact structural values,
measured traversal, gate values, and track-case identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ScenarioReductionError(ValueError):
    """Raised when a full-uncertainty result cannot support a design replay."""


_DEFAULT_OUTPUT_METRICS = (
    "bounded_lap_time_s",
    "lap_time_penalty_vs_infinite_s",
    "finite_ratio_opportunity_loss_energy_kj",
    "bounded_time_maximum_ratio_s",
    "bounded_time_minimum_ratio_s",
    "bounded_obstacle_loss_energy_kj",
    "bounded_tire_slip_loss_energy_kj",
)

_TRACK_EFFECT_METRICS = (
    "bounded_lap_time_s",
    "lap_time_penalty_vs_infinite_s",
    "finite_ratio_opportunity_loss_energy_kj",
    "bounded_time_maximum_ratio_s",
    "bounded_time_minimum_ratio_s",
)


@dataclass(frozen=True, slots=True)
class ReductionSettings:
    enabled: bool = False
    source_result: str = "latest"
    selection: str = "representative"
    maximum_base_draws: int = 12
    maximum_track_cases: int = 5
    require_completed: bool = True
    maximum_abs_energy_balance_relative_error: float = 0.02
    maximum_abs_powertrain_balance_relative_error: float = 0.02
    include_output_extremes: bool = True

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        base_draw_override: int | None = None,
    ) -> "ReductionSettings":
        raw = raw if isinstance(raw, Mapping) else {}
        settings = cls(
            enabled=bool(raw.get("enabled", False)),
            source_result=str(raw.get("source_result", "latest")).strip() or "latest",
            selection=str(raw.get("selection", "representative")).strip().lower(),
            maximum_base_draws=int(
                base_draw_override
                if base_draw_override is not None
                else raw.get("maximum_base_draws", 12)
            ),
            maximum_track_cases=int(raw.get("maximum_track_cases", 5)),
            require_completed=bool(raw.get("require_completed", True)),
            maximum_abs_energy_balance_relative_error=float(
                raw.get("maximum_abs_energy_balance_relative_error", 0.02)
            ),
            maximum_abs_powertrain_balance_relative_error=float(
                raw.get("maximum_abs_powertrain_balance_relative_error", 0.02)
            ),
            include_output_extremes=bool(raw.get("include_output_extremes", True)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.selection not in {"representative", "all"}:
            raise ScenarioReductionError(
                "uncertainty_informed_design.selection must be 'representative' or 'all'."
            )
        if self.maximum_base_draws < 1:
            raise ScenarioReductionError(
                "uncertainty_informed_design.maximum_base_draws must be positive."
            )
        if self.maximum_track_cases < 1:
            raise ScenarioReductionError(
                "uncertainty_informed_design.maximum_track_cases must be positive."
            )
        for name, value in (
            (
                "maximum_abs_energy_balance_relative_error",
                self.maximum_abs_energy_balance_relative_error,
            ),
            (
                "maximum_abs_powertrain_balance_relative_error",
                self.maximum_abs_powertrain_balance_relative_error,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ScenarioReductionError(
                    f"uncertainty_informed_design.{name} must be positive and finite."
                )


@dataclass(frozen=True, slots=True)
class SourceTrackCase:
    case_id: str
    category: str
    label: str
    bundle_path: Path


@dataclass(frozen=True)
class UncertaintySource:
    result_directory: Path
    manifest: Mapping[str, Any]
    scenarios: tuple[dict[str, Any], ...]
    results: pd.DataFrame
    track_cases: tuple[SourceTrackCase, ...]


@dataclass(frozen=True)
class ReducedScenarioSet:
    source: UncertaintySource
    settings: ReductionSettings
    selected_base_draw_ids: tuple[int, ...]
    selected_track_case_ids: tuple[str, ...]
    selected_scenarios: tuple[dict[str, Any], ...]
    draw_audit: pd.DataFrame
    track_case_audit: pd.DataFrame
    metadata: Mapping[str, Any]


def uncertainty_informed_design_enabled(raw_study: Mapping[str, Any]) -> bool:
    config = raw_study.get("uncertainty_informed_design", {})
    return isinstance(config, Mapping) and bool(config.get("enabled", False))


def resolve_source_result(
    *,
    project_root: Path,
    results_directory: Path,
    source_value: str,
) -> Path:
    """Resolve an explicit or ``latest`` full-uncertainty result directory."""

    if source_value == "latest":
        root = results_directory / "full_uncertainty"
        candidates = (
            [
                path
                for path in root.iterdir()
                if path.is_dir() and _looks_like_uncertainty_result(path)
            ]
            if root.is_dir()
            else []
        )
        if not candidates:
            raise ScenarioReductionError(
                f"No completed full-uncertainty result was found under {root}."
            )
        return max(candidates, key=lambda path: path.stat().st_mtime).resolve()

    source = Path(source_value)
    if not source.is_absolute():
        source = (project_root / source).resolve()
    if not _looks_like_uncertainty_result(source):
        raise ScenarioReductionError(
            "The configured uncertainty source must contain run_manifest.json, "
            f"scenario_draws.jsonl, and replicate_results.csv: {source}"
        )
    return source


def load_uncertainty_source(result_directory: Path) -> UncertaintySource:
    result_directory = result_directory.resolve()
    manifest = _read_json(result_directory / "run_manifest.json")
    if str(manifest.get("study_type", "")) != "full_uncertainty":
        raise ScenarioReductionError(
            f"Source result is not a full_uncertainty study: {result_directory}"
        )

    scenarios = tuple(_read_jsonl(result_directory / "scenario_draws.jsonl"))
    if not scenarios:
        raise ScenarioReductionError("The source uncertainty result contains no scenarios.")
    normalized = tuple(_normalize_scenario(record) for record in scenarios)

    try:
        results = pd.read_csv(result_directory / "replicate_results.csv")
    except pd.errors.EmptyDataError as exc:
        raise ScenarioReductionError(
            "The source uncertainty result contains no replicate rows."
        ) from exc
    if results.empty or "replicate" not in results:
        raise ScenarioReductionError(
            "The source uncertainty result lacks usable replicate_results.csv rows."
        )

    track_cases = _load_track_cases(result_directory, normalized)
    return UncertaintySource(
        result_directory=result_directory,
        manifest=manifest,
        scenarios=normalized,
        results=results,
        track_cases=track_cases,
    )


def reduce_uncertainty_source(
    source: UncertaintySource,
    *,
    settings: ReductionSettings,
    design_path: str | None = None,
    design_paths: Sequence[str] | None = None,
) -> ReducedScenarioSet:
    """Select paired source worlds and materially distinct track cases.

    ``design_path`` remains accepted for compatibility with older callers. New
    multi-variable studies pass ``design_paths`` so every design-controlled input
    is removed from the source uncertainty features before scenario reduction.
    """

    removed_design_paths = tuple(
        dict.fromkeys(
            str(path)
            for path in (design_paths or ((design_path,) if design_path else ()))
            if str(path)
        )
    )
    if not removed_design_paths:
        raise ScenarioReductionError("At least one design path is required for replay.")

    scenarios = source.scenarios
    scenario_by_replicate = {
        int(record["replicate"]): record for record in scenarios
    }
    scenario_by_pair = {
        (int(record["base_draw_id"]), str(record["track_case_id"])): record
        for record in scenarios
    }

    rows = _normalize_result_rows(source.results, scenario_by_replicate)
    valid = _valid_result_rows(rows, settings)
    if valid.empty:
        raise ScenarioReductionError(
            "No completed, finite source scenarios satisfy the replay quality filters."
        )

    case_ids, track_audit = _select_track_cases(
        source=source,
        valid_rows=valid,
        maximum=settings.maximum_track_cases,
    )
    candidate_draws = _paired_candidate_draws(
        valid_rows=valid,
        scenario_by_pair=scenario_by_pair,
        selected_case_ids=case_ids,
    )
    if not candidate_draws:
        raise ScenarioReductionError(
            "No source base draw has a valid scenario for every selected track case. "
            "Reduce maximum_track_cases or review incomplete source scenarios."
        )

    feature_frame, outcome_frame = _feature_frame(
        base_draw_ids=candidate_draws,
        selected_case_ids=case_ids,
        scenario_by_pair=scenario_by_pair,
        valid_rows=valid,
        design_paths=removed_design_paths,
    )
    selected_draws, reasons = _select_base_draws(
        feature_frame=feature_frame,
        outcome_frame=outcome_frame,
        settings=settings,
    )

    selected_records: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    order_by_draw = {draw_id: index for index, draw_id in enumerate(selected_draws)}
    case_order = {case_id: index for index, case_id in enumerate(case_ids)}
    source_rows = valid.set_index(["base_draw_id", "track_case_id"], drop=False)
    for draw_id in selected_draws:
        for case_id in case_ids:
            record = dict(scenario_by_pair[(draw_id, case_id)])
            selected_records.append(record)
            result = source_rows.loc[(draw_id, case_id)]
            if isinstance(result, pd.DataFrame):
                result = result.iloc[0]
            draw_rows.append(
                {
                    "selection_order": order_by_draw[draw_id],
                    "selection_reason": reasons.get(draw_id, "diversity"),
                    "base_draw_id": draw_id,
                    "track_case_order": case_order[case_id],
                    "track_case_id": case_id,
                    "source_replicate": int(record["replicate"]),
                    "source_seed": int(record.get("seed", 0)),
                    "bounded_lap_time_s": _number(result.get("bounded_lap_time_s")),
                    "lap_time_penalty_vs_infinite_s": _number(
                        result.get("lap_time_penalty_vs_infinite_s")
                    ),
                    "finite_ratio_opportunity_loss_energy_kj": _number(
                        result.get("finite_ratio_opportunity_loss_energy_kj")
                    ),
                    "bounded_time_maximum_ratio_s": _number(
                        result.get("bounded_time_maximum_ratio_s")
                    ),
                    "bounded_time_minimum_ratio_s": _number(
                        result.get("bounded_time_minimum_ratio_s")
                    ),
                }
            )

    metadata = {
        "schema_version": 1,
        "method": (
            "all_valid_paired_source_worlds"
            if settings.selection == "all"
            else "actual_source_worlds_selected_by_center_extremes_and_farthest_point_diversity"
        ),
        "source_result": str(source.result_directory),
        "source_study_fingerprint_sha256": source.manifest.get(
            "study_fingerprint_sha256"
        ),
        "source_scenario_count": len(source.scenarios),
        "valid_source_scenario_count": int(len(valid)),
        "candidate_base_draw_count": len(candidate_draws),
        "selected_base_draw_count": len(selected_draws),
        "selected_track_case_count": len(case_ids),
        "selected_base_draw_ids": list(selected_draws),
        "selected_track_case_ids": list(case_ids),
        "design_path_removed_from_source_draws": (
            removed_design_paths[0] if len(removed_design_paths) == 1 else None
        ),
        "design_paths_removed_from_source_draws": list(removed_design_paths),
        "interpretation": (
            "The selected rows are real completed worlds from the source full-uncertainty "
            "study. They are a decision-screening set, not a probability-calibrated replacement "
            "for the complete uncertainty distribution."
        ),
    }
    return ReducedScenarioSet(
        source=source,
        settings=settings,
        selected_base_draw_ids=selected_draws,
        selected_track_case_ids=case_ids,
        selected_scenarios=tuple(selected_records),
        draw_audit=pd.DataFrame(draw_rows),
        track_case_audit=track_audit,
        metadata=metadata,
    )


def write_reduction_artifacts(output: Path, reduction: ReducedScenarioSet) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "uncertainty_informed_design_manifest.json").write_text(
        json.dumps(reduction.metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    reduction.draw_audit.to_csv(
        output / "selected_uncertainty_scenarios.csv", index=False
    )
    reduction.track_case_audit.to_csv(
        output / "selected_uncertainty_track_cases.csv", index=False
    )
    lines = [
        "# Uncertainty-informed design scenario reduction",
        "",
        f"Source: `{reduction.source.result_directory}`",
        "",
        f"Selected **{len(reduction.selected_base_draw_ids)}** actual uncertainty draws "
        f"across **{len(reduction.selected_track_case_ids)}** track cases, for "
        f"**{len(reduction.selected_scenarios)}** paired design worlds.",
        "",
        "The reducer keeps the exact sampled structural inputs, measured traversal, "
        "gate values, and track bundle from the source uncertainty result. Only the "
        "configured design variables are removed so each candidate can replace them.",
        "",
        "This reduced set is intended for broad design screening. Verify the leading "
        "two or three candidates on a larger source subset when their paired results are close.",
        "",
        "## Selected base draws",
        "",
    ]
    for draw_id in reduction.selected_base_draw_ids:
        subset = reduction.draw_audit[
            reduction.draw_audit["base_draw_id"] == draw_id
        ]
        reason = str(subset.iloc[0]["selection_reason"]) if not subset.empty else ""
        lines.append(f"- `{draw_id}` — {reason}")
    lines.extend(["", "## Selected track cases", ""])
    for case_id in reduction.selected_track_case_ids:
        subset = reduction.track_case_audit[
            reduction.track_case_audit["track_case_id"] == case_id
        ]
        label = str(subset.iloc[0]["label"]) if not subset.empty else case_id
        lines.append(f"- `{case_id}` — {label}")
    (output / "SCENARIO_REDUCTION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _looks_like_uncertainty_result(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in (
            "run_manifest.json",
            "scenario_draws.jsonl",
            "replicate_results.csv",
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioReductionError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ScenarioReductionError(f"Expected a JSON object in {path}.")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ScenarioReductionError(f"Could not read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScenarioReductionError(
                f"Malformed JSONL at {path}:{line_number}."
            ) from exc
        if not isinstance(value, Mapping):
            raise ScenarioReductionError(
                f"Expected an object at {path}:{line_number}."
            )
        records.append(dict(value))
    return records


def _normalize_scenario(record: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(record)
    try:
        replicate = int(output["replicate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScenarioReductionError(
            "Every source scenario requires an integer replicate identifier."
        ) from exc
    output["replicate"] = replicate
    output["base_draw_id"] = int(output.get("base_draw_id", replicate))
    output["track_case_id"] = str(output.get("track_case_id", "nominal"))
    output["track_case_category"] = str(
        output.get("track_case_category", "nominal")
    )
    output["quantity_values_si"] = dict(output.get("quantity_values_si", {}))
    output["choice_values"] = dict(output.get("choice_values", {}))
    output["gate_target_speeds_mps"] = dict(
        output.get("gate_target_speeds_mps", {})
    )
    output["independently_sampled_gate_ids"] = list(
        output.get("independently_sampled_gate_ids", [])
    )
    return output


def _load_track_cases(
    result_directory: Path,
    scenarios: Sequence[Mapping[str, Any]],
) -> tuple[SourceTrackCase, ...]:
    manifest_path = result_directory / "track_ensemble" / "manifest.json"
    records: list[SourceTrackCase] = []
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        for raw in manifest.get("cases", []):
            if not isinstance(raw, Mapping):
                continue
            path = (result_directory / str(raw.get("file", ""))).resolve()
            if not path.is_file():
                continue
            records.append(
                SourceTrackCase(
                    case_id=str(raw.get("case_id", "")),
                    category=str(raw.get("category", "unknown")),
                    label=str(raw.get("label", raw.get("case_id", ""))),
                    bundle_path=path,
                )
            )
    if not records:
        path = result_directory / "track_bundle.json"
        if not path.is_file():
            raise ScenarioReductionError(
                "The uncertainty result contains neither a track ensemble snapshot nor track_bundle.json."
            )
        records.append(
            SourceTrackCase(
                case_id="nominal",
                category="nominal",
                label="Nominal reconstructed track",
                bundle_path=path.resolve(),
            )
        )

    used = {str(record.get("track_case_id", "nominal")) for record in scenarios}
    filtered = tuple(record for record in records if record.case_id in used)
    if not filtered:
        raise ScenarioReductionError(
            "No saved track bundle matches the track_case_id values in scenario_draws.jsonl."
        )
    return filtered


def _normalize_result_rows(
    raw: pd.DataFrame,
    scenario_by_replicate: Mapping[int, Mapping[str, Any]],
) -> pd.DataFrame:
    rows = raw.copy()
    rows["replicate"] = pd.to_numeric(rows["replicate"], errors="coerce")
    rows = rows[rows["replicate"].notna()].copy()
    rows["replicate"] = rows["replicate"].astype(int)

    if "design_id" in rows and rows["design_id"].nunique() > 1:
        nominal = rows[rows["design_id"].astype(str) == "nominal"]
        if not nominal.empty:
            rows = nominal.copy()
        else:
            rows = rows.sort_values("replicate").drop_duplicates("replicate")

    def source_field(replicate: int, field: str, default: Any) -> Any:
        source = scenario_by_replicate.get(int(replicate), {})
        return source.get(field, default)

    if "base_draw_id" not in rows:
        rows["base_draw_id"] = [
            source_field(value, "base_draw_id", value) for value in rows["replicate"]
        ]
    if "track_case_id" not in rows:
        rows["track_case_id"] = [
            source_field(value, "track_case_id", "nominal")
            for value in rows["replicate"]
        ]
    rows["base_draw_id"] = pd.to_numeric(
        rows["base_draw_id"], errors="coerce"
    ).astype("Int64")
    rows = rows[rows["base_draw_id"].notna()].copy()
    rows["base_draw_id"] = rows["base_draw_id"].astype(int)
    rows["track_case_id"] = rows["track_case_id"].astype(str)
    return rows


def _valid_result_rows(
    rows: pd.DataFrame,
    settings: ReductionSettings,
) -> pd.DataFrame:
    mask = pd.Series(True, index=rows.index)
    if settings.require_completed:
        for column in ("bounded_completed", "reference_completed"):
            if column not in rows:
                raise ScenarioReductionError(
                    f"Source replicate results lack required completion column {column!r}."
                )
            mask &= _boolean_series(rows[column])

    for column in (
        "bounded_lap_time_s",
        "lap_time_penalty_vs_infinite_s",
        "finite_ratio_opportunity_loss_energy_kj",
    ):
        if column not in rows:
            raise ScenarioReductionError(
                f"Source replicate results lack required metric {column!r}."
            )
        mask &= pd.to_numeric(rows[column], errors="coerce").notna()

    for column, limit in (
        (
            "bounded_energy_balance_relative_error",
            settings.maximum_abs_energy_balance_relative_error,
        ),
        (
            "reference_energy_balance_relative_error",
            settings.maximum_abs_energy_balance_relative_error,
        ),
        (
            "bounded_powertrain_energy_balance_relative_error",
            settings.maximum_abs_powertrain_balance_relative_error,
        ),
        (
            "reference_powertrain_energy_balance_relative_error",
            settings.maximum_abs_powertrain_balance_relative_error,
        ),
    ):
        if column in rows:
            mask &= pd.to_numeric(rows[column], errors="coerce").abs() <= limit
    return rows[mask].copy()


def _select_track_cases(
    *,
    source: UncertaintySource,
    valid_rows: pd.DataFrame,
    maximum: int,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    case_lookup = {case.case_id: case for case in source.track_cases}
    available = [
        case.case_id
        for case in source.track_cases
        if case.case_id in set(valid_rows["track_case_id"].astype(str))
    ]
    if not available:
        raise ScenarioReductionError("No saved track case has valid uncertainty rows.")

    nominal_id = "nominal" if "nominal" in available else available[0]
    nominal = valid_rows[valid_rows["track_case_id"] == nominal_id]
    records: list[dict[str, Any]] = []
    for case_id in available:
        case_rows = valid_rows[valid_rows["track_case_id"] == case_id]
        if case_id == nominal_id:
            score = 0.0
            paired_count = int(case_rows["base_draw_id"].nunique())
            reason = "nominal_anchor"
        else:
            paired = nominal.merge(
                case_rows,
                on="base_draw_id",
                suffixes=("_nominal", "_case"),
            )
            paired_count = int(len(paired))
            effects: list[float] = []
            for metric in _TRACK_EFFECT_METRICS:
                left = f"{metric}_nominal"
                right = f"{metric}_case"
                if left not in paired or right not in paired:
                    continue
                nominal_values = pd.to_numeric(paired[left], errors="coerce")
                case_values = pd.to_numeric(paired[right], errors="coerce")
                delta = (case_values - nominal_values).dropna()
                if delta.empty:
                    continue
                scale = _robust_scale(nominal_values.dropna(), floor=_metric_floor(metric))
                effects.append(float(delta.abs().median()) / scale)
            score = float(np.mean(effects)) if effects else -math.inf
            reason = "ranked_by_paired_effect_on_source_outputs"
        case = case_lookup[case_id]
        records.append(
            {
                "track_case_id": case_id,
                "category": case.category,
                "label": case.label,
                "paired_valid_draw_count": paired_count,
                "paired_effect_score": score,
                "selection_reason": reason,
            }
        )

    alternatives = sorted(
        (record for record in records if record["track_case_id"] != nominal_id),
        key=lambda record: (
            -float(record["paired_effect_score"]),
            -int(record["paired_valid_draw_count"]),
            str(record["track_case_id"]),
        ),
    )
    selected = [nominal_id]
    selected.extend(
        record["track_case_id"] for record in alternatives[: max(0, maximum - 1)]
    )
    selected_set = set(selected)
    for record in records:
        record["selected"] = record["track_case_id"] in selected_set
        record["selection_order"] = (
            selected.index(record["track_case_id"])
            if record["track_case_id"] in selected_set
            else -1
        )
    audit = pd.DataFrame(records).sort_values(
        ["selected", "selection_order", "paired_effect_score"],
        ascending=[False, True, False],
    )
    return tuple(selected), audit


def _paired_candidate_draws(
    *,
    valid_rows: pd.DataFrame,
    scenario_by_pair: Mapping[tuple[int, str], Mapping[str, Any]],
    selected_case_ids: Sequence[str],
) -> tuple[int, ...]:
    sets: list[set[int]] = []
    for case_id in selected_case_ids:
        valid_ids = set(
            int(value)
            for value in valid_rows.loc[
                valid_rows["track_case_id"] == case_id, "base_draw_id"
            ]
        )
        scenario_ids = {
            int(draw_id)
            for draw_id, scenario_case_id in scenario_by_pair
            if scenario_case_id == case_id
        }
        sets.append(valid_ids & scenario_ids)
    return tuple(sorted(set.intersection(*sets))) if sets else ()


def _feature_frame(
    *,
    base_draw_ids: Sequence[int],
    selected_case_ids: Sequence[str],
    scenario_by_pair: Mapping[tuple[int, str], Mapping[str, Any]],
    valid_rows: pd.DataFrame,
    design_paths: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    removed_design_paths = set(str(path) for path in design_paths)
    preferred_case = "nominal" if "nominal" in selected_case_ids else selected_case_ids[0]
    result_lookup = valid_rows.set_index(["base_draw_id", "track_case_id"], drop=False)
    feature_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    choice_values: dict[str, set[str]] = {}

    for draw_id in base_draw_ids:
        scenario = scenario_by_pair[(int(draw_id), preferred_case)]
        for path, value in scenario.get("choice_values", {}).items():
            if str(path) not in removed_design_paths:
                choice_values.setdefault(str(path), set()).add(str(value))

    for draw_id in base_draw_ids:
        scenario = scenario_by_pair[(int(draw_id), preferred_case)]
        features: dict[str, Any] = {"base_draw_id": int(draw_id)}
        quantities = scenario.get("quantity_values_si", {})
        for path, value in quantities.items():
            if str(path) in removed_design_paths:
                continue
            number = _number(value)
            if math.isfinite(number):
                features[f"quantity:{path}"] = number
        for path, value in scenario.get("gate_target_speeds_mps", {}).items():
            number = _number(value)
            if math.isfinite(number):
                features[f"gate:{path}"] = number
        for path, alternatives in choice_values.items():
            selected = str(scenario.get("choice_values", {}).get(path, ""))
            for alternative in sorted(alternatives):
                features[f"choice:{path}={alternative}"] = float(selected == alternative)

        efficiency = _number(quantities.get("drivetrain.efficiency"))
        power_scale = _number(quantities.get("drivetrain.engine.power_scale"))
        if math.isfinite(efficiency) and math.isfinite(power_scale):
            features["derived:effective_wheel_power_multiplier"] = efficiency * power_scale
        gate_values = [
            _number(value)
            for value in scenario.get("gate_target_speeds_mps", {}).values()
            if math.isfinite(_number(value))
        ]
        if gate_values:
            features["derived:measured_gate_pace_median_mps"] = float(
                np.median(gate_values)
            )

        outcomes: dict[str, Any] = {"base_draw_id": int(draw_id)}
        for metric in _DEFAULT_OUTPUT_METRICS:
            values: list[float] = []
            for case_id in selected_case_ids:
                row = result_lookup.loc[(int(draw_id), case_id)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                value = _number(row.get(metric))
                if math.isfinite(value):
                    values.append(value)
            if values:
                mean = float(np.mean(values))
                maximum = float(np.max(values))
                features[f"output:{metric}:mean"] = mean
                features[f"output:{metric}:max"] = maximum
                outcomes[f"{metric}_mean"] = mean
                outcomes[f"{metric}_max"] = maximum
        feature_rows.append(features)
        outcome_rows.append(outcomes)

    features = pd.DataFrame(feature_rows).set_index("base_draw_id").sort_index()
    outcomes = pd.DataFrame(outcome_rows).set_index("base_draw_id").sort_index()
    return features, outcomes


def _select_base_draws(
    *,
    feature_frame: pd.DataFrame,
    outcome_frame: pd.DataFrame,
    settings: ReductionSettings,
) -> tuple[tuple[int, ...], dict[int, str]]:
    ids = [int(value) for value in feature_frame.index]
    if settings.selection == "all" or len(ids) <= settings.maximum_base_draws:
        return tuple(ids), {draw_id: "all_valid_paired_source_worlds" for draw_id in ids}

    matrix = _standardized_matrix(feature_frame)
    maximum = min(settings.maximum_base_draws, len(ids))
    selected: list[int] = []
    reasons: dict[int, str] = {}

    center_position = int(np.argmin(np.linalg.norm(matrix, axis=1)))
    center_id = ids[center_position]
    selected.append(center_id)
    reasons[center_id] = "representative_center"

    if settings.include_output_extremes:
        priority_columns = [
            column
            for column in outcome_frame.columns
            if column.endswith("_mean")
            and any(
                token in column
                for token in (
                    "bounded_lap_time_s",
                    "lap_time_penalty_vs_infinite_s",
                    "finite_ratio_opportunity_loss_energy_kj",
                    "bounded_time_maximum_ratio_s",
                    "bounded_time_minimum_ratio_s",
                )
            )
        ]
        for column in priority_columns:
            values = pd.to_numeric(outcome_frame[column], errors="coerce")
            for label, draw_id in (
                ("low", int(values.idxmin())),
                ("high", int(values.idxmax())),
            ):
                if len(selected) >= maximum:
                    break
                if draw_id not in selected:
                    selected.append(draw_id)
                    reasons[draw_id] = f"source_output_extreme:{column}:{label}"
            if len(selected) >= maximum:
                break

    id_to_position = {draw_id: index for index, draw_id in enumerate(ids)}
    while len(selected) < maximum:
        selected_positions = [id_to_position[draw_id] for draw_id in selected]
        minimum_distances = np.full(len(ids), np.inf, dtype=float)
        for position in selected_positions:
            distances = np.linalg.norm(matrix - matrix[position], axis=1)
            minimum_distances = np.minimum(minimum_distances, distances)
        for draw_id in selected:
            minimum_distances[id_to_position[draw_id]] = -np.inf
        next_position = int(np.argmax(minimum_distances))
        next_id = ids[next_position]
        selected.append(next_id)
        reasons[next_id] = "farthest_point_diversity"

    return tuple(selected), reasons


def _standardized_matrix(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    columns: list[np.ndarray] = []
    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        median = float(np.nanmedian(values))
        values = np.where(finite, values, median)
        scale = _robust_scale(pd.Series(values), floor=1.0e-12)
        if scale <= 1.0e-12 or np.allclose(values, values[0]):
            continue
        columns.append(np.clip((values - median) / scale, -8.0, 8.0))
    if not columns:
        return np.zeros((len(frame), 1), dtype=float)
    matrix = np.column_stack(columns)
    # Prevent a family with hundreds of obstacle parameters from overwhelming the
    # explicit source-output dimensions purely by column count.
    norms = np.sqrt(np.maximum(np.sum(matrix * matrix, axis=0), 1.0e-12))
    return matrix / norms


def _robust_scale(values: pd.Series, *, floor: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return floor
    q25 = float(numeric.quantile(0.25))
    q75 = float(numeric.quantile(0.75))
    scale = q75 - q25
    if scale <= floor:
        scale = float(numeric.std(ddof=0))
    return max(scale, floor)


def _metric_floor(metric: str) -> float:
    if metric.endswith("_time_s") or "lap_time" in metric:
        return 0.25
    if "energy" in metric or metric.endswith("_kj"):
        return 1.0
    return 0.1


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
