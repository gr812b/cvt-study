from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import textwrap

MARKER = "multivariable-design-grid-v1"


def replace_top_level_function(source: str, name: str, replacement: str) -> str:
    lines = source.splitlines(keepends=True)
    start = None
    for index, line in enumerate(lines):
        if line.startswith(f"def {name}("):
            start = index
            break
    if start is None:
        raise RuntimeError(f"Could not find top-level function {name}")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("def ") or line.startswith("class "):
            end = index
            break
    block = textwrap.dedent(replacement).strip() + "\n\n"
    return "".join(lines[:start]) + block + "".join(lines[end:])


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} block, found {count}")
    return source.replace(old, new, 1)


def patch_planning(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    start = source.find('    if study_type == "design_sweep":')
    end = source.find('\n    if study_type == "structural_sensitivity":', start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not locate design-sweep planning branch")
    replacement = '''    if study_type == "design_sweep":
        # multivariable-design-grid-v1
        from .design_grid import design_sweep_plan

        return design_sweep_plan(raw, registry, replicates_override)
'''
    source = source[:start] + replacement + source[end:]
    path.write_text(source, encoding="utf-8")


def patch_service_v8(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    import_line = "from .planning import reference_cache_key, study_plan\n"
    new_import = import_line + '''from .design_grid import (
    design_choice_values,
    design_display_values,
    design_paths as configured_design_paths,
    design_quantity_values_si,
    reference_can_be_shared,
)
'''
    if "from .design_grid import (" not in source:
        source = replace_once(source, import_line, new_import, "service_v8 import")

    source = source.replace(
        '    design_path = design_points[0].path if study_type == "design_sweep" else None\n',
        '    design_paths = configured_design_paths(design_points) if study_type == "design_sweep" else ()\n',
        1,
    )
    source = source.replace(
        '        excluded_paths=(design_path,) if design_path else (),\n',
        '        excluded_paths=design_paths,\n',
        1,
    )

    replacement = r'''
def _execute_scenario(
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
    """Execute one paired world across every design candidate.

    Multi-variable candidates carry a complete override mapping.  All candidates
    still share one reference only when every varied path is known to leave the
    infinite-CVT reference invariant.
    """

    rows: list[dict[str, Any]] = []
    references: dict[tuple[int, str], tuple[dict[str, Any], str]] = {}
    bounded_runs = reference_runs = reference_reuses = persistent_hits = 0
    all_design_paths = configured_design_paths(design_points)
    share_reference = (
        study_type == "design_sweep"
        and reference_can_be_shared(all_design_paths)
    )

    for design in design_points:
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

        bounded_record, cached = _run_case_summary_cached(
            bounded_case, settings, runtime_track, cache
        )
        bounded_runs += int(not cached)
        persistent_hits += int(cached)
        key = reference_cache_key(
            scenario.replicate,
            design,
            share_across_designs=share_reference,
        )
        if key in references:
            reference_record, reference_fingerprint = references[key]
            reference_reuses += 1
        else:
            reference_record, cached = _run_case_summary_cached(
                reference_case, settings, runtime_track, cache
            )
            reference_runs += int(not cached)
            persistent_hits += int(cached)
            reference_fingerprint = phase6_service._reference_fingerprint(
                scenario, design, reference_case, runtime_track
            )
            references[key] = (reference_record, reference_fingerprint)

        bounded_summary = bounded_record["summary"]
        reference_summary = reference_record["summary"]
        comparison = compare_summaries(bounded_summary, reference_summary)
        display_values = design_display_values(design)
        quantity_values_si = design_quantity_values_si(design)
        design_paths = tuple(display_values)
        design_path_text = (
            str(design.path)
            if getattr(design, "path", None)
            else " + ".join(design_paths) if design_paths else "nominal"
        )
        row: dict[str, Any] = {
            "replicate": scenario.replicate,
            "scenario_seed": scenario.seed,
            "design_id": design.identifier,
            "design_path": design_path_text,
            "design_value": design.display_value,
            "design_value_si": design.value_si,
            "design_choice_value": design.choice_value,
            "design_dimension_count": len(design_paths),
            "design_paths_json": json.dumps(
                list(design_paths), separators=(",", ":"), allow_nan=False
            ),
            "design_values_json": json.dumps(
                display_values, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
            "design_values_si_json": json.dumps(
                quantity_values_si,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "level_probability": design.level_probability,
            "level_kind": design.level_kind,
            "parameter_path": design.path if study_type == "structural_sensitivity" else None,
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
            "bounded_max_gate_excess_kmh": bounded_record["maximum_gate_excess_kmh"],
            "reference_max_gate_excess_kmh": reference_record["maximum_gate_excess_kmh"],
            "bounded_gates_compliant_0p5_kmh": bounded_record[
                "gates_compliant_0p5_kmh"
            ],
            "reference_gates_compliant_0p5_kmh": reference_record[
                "gates_compliant_0p5_kmh"
            ],
        }
        for index, (path, value) in enumerate(display_values.items()):
            row[f"design_axis_{index}_path"] = path
            row[f"design_axis_{index}_value"] = value
            row[f"design::{path}"] = value
        row.update({metric: float(comparison[metric]) for metric in METRICS})
        _add_summary_fields(row, "bounded", bounded_summary)
        _add_summary_fields(row, "reference", reference_summary)
        rows.append(row)

    return {
        "rows": rows,
        "bounded_case_count": len(design_points),
        "reference_case_count": (
            1 if share_reference and design_points else len(design_points)
        ),
        "bounded_simulation_count": bounded_runs,
        "reference_simulation_count": reference_runs,
        "reference_cache_hits": reference_reuses,
        "simulation_cache_hits": persistent_hits,
        "resumed": False,
    }
'''
    source = replace_top_level_function(source, "_execute_scenario", replacement)
    path.write_text(source, encoding="utf-8")


def patch_ensemble(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    import_line = "from .planning import study_plan\n"
    if "configured_design_paths" not in source:
        source = replace_once(
            source,
            import_line,
            import_line + "from .design_grid import design_paths as configured_design_paths\n",
            "ensemble design-grid import",
        )
    source = source.replace(
        '    design_path = design_points[0].path if study_type == "design_sweep" else None\n',
        '    design_paths = configured_design_paths(design_points) if study_type == "design_sweep" else ()\n',
        1,
    )
    source = source.replace(
        '        excluded_paths=(design_path,) if design_path else (),\n',
        '        excluded_paths=design_paths,\n',
        1,
    )
    path.write_text(source, encoding="utf-8")


def patch_validation(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if "def _validate_design_sweep(" not in source:
        match = re.search(r"^    def _validate_studies\(", source, flags=re.MULTILINE)
        if match is None:
            raise RuntimeError("Could not locate ProjectValidator._validate_studies")
        method = r'''
    def _validate_design_sweep(
        self,
        *,
        raw: Mapping[str, Any],
        path: str,
        vehicle_id: str,
        raw_vehicles: Any,
        diagnostics: DiagnosticBag,
    ) -> None:
        """Validate one-dimensional or Cartesian transmission design grids."""

        single = raw.get("design_variable")
        multiple = raw.get("design_variables")
        if single is not None and multiple is not None:
            diagnostics.error(
                "AMBIGUOUS_DESIGN_VARIABLES",
                "Use either [design_variable] or [[design_variables]], not both.",
                path=path,
            )
            return
        if multiple is not None:
            if (
                not isinstance(multiple, list)
                or not multiple
                or not all(isinstance(item, Mapping) for item in multiple)
            ):
                diagnostics.error(
                    "DESIGN_VARIABLES_INVALID",
                    "[[design_variables]] must contain one or more design-variable tables.",
                    path=f"{path}.design_variables",
                )
                return
            variables = list(multiple)
        elif isinstance(single, Mapping):
            variables = [single]
        else:
            diagnostics.error(
                "DESIGN_VARIABLE_MISSING",
                "Design sweep requires [design_variable] or [[design_variables]].",
                path=path,
            )
            return

        configured_paths = [str(item.get("path", "")).strip() for item in variables]
        if any(not value for value in configured_paths):
            diagnostics.error(
                "DESIGN_PATH_MISSING",
                "Every design variable requires a non-empty path.",
                path=f"{path}.design_variables",
            )
        if len(configured_paths) != len(set(configured_paths)):
            diagnostics.error(
                "DUPLICATE_DESIGN_PATHS",
                "Design-variable paths must be unique.",
                path=f"{path}.design_variables",
            )

        supported_design_paths = {
            "drivetrain.final_drive_ratio",
            "drivetrain.cvt.minimum_reduction_ratio",
            "drivetrain.cvt.maximum_reduction_ratio",
        }
        vehicle = (
            raw_vehicles.get(vehicle_id, {})
            if isinstance(raw_vehicles, Mapping)
            else {}
        )
        values_by_path: dict[str, list[float]] = {}
        candidate_count = 1
        for index, design in enumerate(variables):
            item_path = (
                f"{path}.design_variable"
                if len(variables) == 1 and multiple is None
                else f"{path}.design_variables.{index}"
            )
            dotted = str(design.get("path", "")).strip()
            values = design.get("values")
            if not isinstance(values, list) or not values or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isfinite(float(value))
                for value in values
            ):
                diagnostics.error(
                    "INVALID_DESIGN_VALUES",
                    "Each design-variable values field must be a non-empty finite numeric array.",
                    path=f"{item_path}.values",
                )
                continue
            numeric_values = [float(value) for value in values]
            if len(numeric_values) != len(set(numeric_values)):
                diagnostics.error(
                    "DUPLICATE_DESIGN_VALUES",
                    "Values for one design variable must not contain duplicates.",
                    path=f"{item_path}.values",
                )
            candidate_count *= len(numeric_values)
            values_by_path[dotted] = numeric_values

            if dotted and dotted not in supported_design_paths:
                diagnostics.error(
                    "DESIGN_PATH_NOT_TRANSMISSION",
                    "Design grids currently support final-drive and CVT ratio variables only.",
                    path=f"{item_path}.path",
                    hint="The Cartesian grid engine is generic, but reference policy is currently approved only for transmission-ratio paths.",
                )
            design_quantity = (
                _study_numeric_input(
                    dotted,
                    vehicle=vehicle if isinstance(vehicle, Mapping) else {},
                    baseline={},
                    track={},
                    events=[],
                )
                if dotted
                else None
            )
            if design_quantity is None:
                diagnostics.error(
                    "DESIGN_PATH_NOT_FOUND",
                    f"Design variable path {dotted!r} does not identify a numeric quantity for vehicle {vehicle_id!r}.",
                    path=f"{item_path}.path",
                    hint="Paths are relative to the selected vehicle configuration.",
                )
            else:
                for value in numeric_values:
                    message = _design_value_domain_error(
                        dotted,
                        value,
                        vehicle if isinstance(vehicle, Mapping) else {},
                    )
                    if message is not None:
                        diagnostics.error(
                            "DESIGN_VALUE_OUT_OF_DOMAIN",
                            f"Design value {value!r} for {dotted!r} is invalid: {message}",
                            path=f"{item_path}.values",
                        )

        grid = raw.get("design_grid", {})
        maximum_candidates = 500
        if grid is not None and not isinstance(grid, Mapping):
            diagnostics.error(
                "DESIGN_GRID_TABLE_INVALID",
                "[design_grid] must be a table when present.",
                path=f"{path}.design_grid",
            )
        elif isinstance(grid, Mapping):
            maximum_candidates = grid.get("maximum_candidates", 500)
            if (
                not isinstance(maximum_candidates, int)
                or isinstance(maximum_candidates, bool)
                or maximum_candidates < 1
            ):
                diagnostics.error(
                    "INVALID_DESIGN_GRID_LIMIT",
                    "design_grid.maximum_candidates must be a positive integer.",
                    path=f"{path}.design_grid.maximum_candidates",
                )
                maximum_candidates = 500
        if candidate_count > int(maximum_candidates):
            diagnostics.error(
                "DESIGN_GRID_TOO_LARGE",
                f"Cartesian grid contains {candidate_count} candidates, above the configured limit of {maximum_candidates}.",
                path=f"{path}.design_variables",
                hint="Reduce axis values or raise design_grid.maximum_candidates deliberately.",
            )

        # Validate the complete Cartesian CVT ratio ordering, including a fixed
        # counterpart taken from the selected vehicle when only one limit varies.
        try:
            drivetrain = vehicle["drivetrain"]
            cvt = drivetrain["cvt"]
            nominal_maximum = float(cvt["maximum_reduction_ratio"]["nominal"])
            nominal_minimum = float(cvt["minimum_reduction_ratio"]["nominal"])
            maximum_values = values_by_path.get(
                "drivetrain.cvt.maximum_reduction_ratio", [nominal_maximum]
            )
            minimum_values = values_by_path.get(
                "drivetrain.cvt.minimum_reduction_ratio", [nominal_minimum]
            )
            invalid_pairs = [
                (maximum, minimum)
                for maximum in maximum_values
                for minimum in minimum_values
                if maximum <= minimum
            ]
            if invalid_pairs:
                maximum, minimum = invalid_pairs[0]
                diagnostics.error(
                    "DESIGN_GRID_CVT_RATIO_ORDER",
                    "Every grid candidate requires maximum_reduction_ratio greater than minimum_reduction_ratio; "
                    f"found maximum={maximum:g}, minimum={minimum:g}.",
                    path=f"{path}.design_variables",
                )
        except (KeyError, TypeError, ValueError):
            pass

'''
        source = source[: match.start()] + method.lstrip("\n") + source[match.start() :]

    start = source.find('            if study_type == "design_sweep":')
    end = source.find('\n            if study_type == "structural_sensitivity":', start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not locate validation design-sweep block")
    replacement = '''            if study_type == "design_sweep":
                # multivariable-design-grid-v1
                self._validate_design_sweep(
                    raw=raw,
                    path=path,
                    vehicle_id=vehicle_id,
                    raw_vehicles=raw_vehicles,
                    diagnostics=diagnostics,
                )
'''
    source = source[:start] + replacement + source[end:]
    path.write_text(source, encoding="utf-8")


def patch_postprocess(path: Path) -> None:
    source = path.read_text(encoding="utf-8")

    ranking = r'''
def _design_ranking(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "design_id" not in rows:
        return pd.DataFrame()

    def bool_values(series: pd.Series) -> pd.Series:
        if series.dtype == object:
            return series.astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            ).fillna(False)
        return series.fillna(False).astype(bool)

    paired = rows.copy()
    paired["_lap_time"] = pd.to_numeric(
        paired.get("bounded_lap_time_s"), errors="coerce"
    )
    if "bounded_completed" in paired:
        paired.loc[~bool_values(paired["bounded_completed"]), "_lap_time"] = math.inf
    scenario_keys = ["replicate"]
    if "scenario_seed" in paired:
        scenario_keys.append("scenario_seed")
    best = paired.groupby(scenario_keys)["_lap_time"].transform("min")
    paired["_regret_s"] = paired["_lap_time"] - best
    paired["_paired_win"] = np.isfinite(paired["_lap_time"]) & np.isclose(
        paired["_lap_time"], best, rtol=0.0, atol=1e-9
    )

    records: list[dict[str, Any]] = []
    for design_id, group in rows.groupby("design_id", sort=False):
        lap = pd.to_numeric(group.get("bounded_lap_time_s"), errors="coerce")
        penalty = pd.to_numeric(
            group.get("lap_time_penalty_vs_infinite_s"), errors="coerce"
        )
        energy = pd.to_numeric(
            group.get("finite_ratio_opportunity_loss_energy_kj"), errors="coerce"
        )
        completed = (
            bool_values(group["bounded_completed"])
            if "bounded_completed" in group
            else pd.Series(True, index=group.index)
        )
        paired_group = paired[paired["design_id"] == design_id]
        finite_regret = pd.to_numeric(
            paired_group["_regret_s"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        record: dict[str, Any] = {
            "design_id": design_id,
            "design_value": group["design_value"].iloc[0] if "design_value" in group else "",
            "scenario_count": int(group["replicate"].nunique()) if "replicate" in group else len(group),
            "completion_fraction": float(completed.mean()),
            "paired_win_fraction": float(paired_group["_paired_win"].mean()),
            "paired_regret_median_s": float(finite_regret.median()) if len(finite_regret) else math.inf,
            "paired_regret_p90_s": float(finite_regret.quantile(0.9)) if len(finite_regret) else math.inf,
            "lap_time_p10_s": float(lap.quantile(0.1)),
            "lap_time_median_s": float(lap.median()),
            "lap_time_p90_s": float(lap.quantile(0.9)),
            "penalty_median_s": float(penalty.median()),
            "opportunity_loss_median_kj": float(energy.median()),
            "maximum_ratio_time_median_s": (
                float(pd.to_numeric(group["bounded_time_maximum_ratio_s"], errors="coerce").median())
                if "bounded_time_maximum_ratio_s" in group else math.nan
            ),
            "minimum_ratio_time_median_s": (
                float(pd.to_numeric(group["bounded_time_minimum_ratio_s"], errors="coerce").median())
                if "bounded_time_minimum_ratio_s" in group else math.nan
            ),
        }
        values: dict[str, Any] = {}
        if "design_values_json" in group:
            try:
                parsed = json.loads(str(group["design_values_json"].iloc[0]))
                if isinstance(parsed, Mapping):
                    values = {str(key): value for key, value in parsed.items()}
            except (TypeError, ValueError, json.JSONDecodeError):
                values = {}
        if not values and "design_path" in group and "design_value" in group:
            path_value = str(group["design_path"].iloc[0])
            if path_value and path_value != "nominal":
                values[path_value] = group["design_value"].iloc[0]
        for design_path, value in values.items():
            record[f"design::{design_path}"] = value
        records.append(record)

    return (
        pd.DataFrame(records)
        .sort_values(
            ["completion_fraction", "paired_regret_median_s", "lap_time_median_s"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
'''
    source = replace_top_level_function(source, "_design_ranking", ranking)

    plots = r'''
def _design_plots(rows: pd.DataFrame, ranking: pd.DataFrame, plots: Path) -> None:
    if ranking.empty:
        return

    def compact_label(value: str) -> str:
        return value.replace("maximum_reduction_ratio", "cvt max").replace(
            "final_drive_ratio", "final drive"
        )

    labels = [compact_label(value) for value in ranking["design_id"].astype(str)]
    x = np.arange(len(ranking))
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(labels)), 5.5))
    med = ranking["lap_time_median_s"].to_numpy(float)
    low = ranking["lap_time_p10_s"].to_numpy(float)
    high = ranking["lap_time_p90_s"].to_numpy(float)
    ax.bar(x, med)
    ax.errorbar(
        x,
        med,
        yerr=np.vstack((med - low, high - med)),
        fmt="none",
        ecolor="black",
        elinewidth=1.6,
        capthick=1.6,
        capsize=4,
        zorder=5,
    )
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Lap time [s]")
    ax.set_title("Absolute bounded performance by design")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "design_lap_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(labels)), 4.8))
    ax.bar(x, ranking["completion_fraction"])
    ax.set_ylim(0, 1)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Completion fraction")
    ax.set_title("Completion reliability")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "design_completion.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(labels)), 4.8))
    ax.bar(x, ranking["penalty_median_s"])
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Median penalty [s]")
    ax.set_title("Paired bounded-versus-infinite penalty")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "design_penalty.png", dpi=180)
    plt.close(fig)

    ratio_cols = [
        column
        for column in (
            "bounded_time_maximum_ratio_s",
            "bounded_time_variable_ratio_s",
            "bounded_time_minimum_ratio_s",
        )
        if column in rows
    ]
    if ratio_cols:
        medians = rows.groupby("design_id")[ratio_cols].median().reindex(
            ranking["design_id"].astype(str)
        )
        fig, ax = plt.subplots(figsize=(max(9, 0.85 * len(labels)), 5.5))
        bottom = np.zeros(len(labels))
        for column in ratio_cols:
            values = medians[column].to_numpy(float)
            ax.bar(
                x,
                values,
                bottom=bottom,
                label=column.replace("bounded_time_", "").replace("_s", "").replace("_", " "),
            )
            bottom += values
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylabel("Median time [s]")
        ax.set_title("CVT ratio-region occupancy by design")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / "design_ratio_time.png", dpi=180)
        plt.close(fig)

    if "track_case_id" in rows and "bounded_lap_time_s" in rows:
        matrix = rows.pivot_table(
            index="track_case_id",
            columns="design_id",
            values="bounded_lap_time_s",
            aggfunc="median",
        )
        if not matrix.empty:
            fig, ax = plt.subplots(
                figsize=(max(9, 0.8 * len(matrix.columns)), max(5.5, 0.42 * len(matrix.index)))
            )
            image = ax.imshow(matrix.to_numpy(float), aspect="auto", interpolation="nearest")
            ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
            ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
            ax.set_title("Median bounded lap time by design and track reconstruction")
            fig.colorbar(image, ax=ax, label="Lap time [s]")
            fig.tight_layout()
            fig.savefig(plots / "design_track_case_matrix.png", dpi=180)
            plt.close(fig)

    design_columns = [column for column in ranking if column.startswith("design::")]
    if len(design_columns) != 2:
        return

    final_drive = "design::drivetrain.final_drive_ratio"
    if final_drive in design_columns:
        x_column = final_drive
        y_column = next(column for column in design_columns if column != final_drive)
    else:
        x_column, y_column = design_columns

    grid_summary = ranking.copy()
    grid_summary.to_csv(plots.parent / "design_grid_summary.csv", index=False)

    def axis_label(column: str) -> str:
        path = column.removeprefix("design::")
        aliases = {
            "drivetrain.final_drive_ratio": "Final-drive ratio",
            "drivetrain.cvt.maximum_reduction_ratio": "Maximum CVT reduction ratio",
            "drivetrain.cvt.minimum_reduction_ratio": "Minimum CVT reduction ratio",
        }
        return aliases.get(path, path)

    def heatmap(metric: str, filename: str, title: str, colourbar: str, digits: int = 2) -> None:
        if metric not in grid_summary:
            return
        table = grid_summary.pivot_table(
            index=y_column,
            columns=x_column,
            values=metric,
            aggfunc="median",
        ).sort_index().sort_index(axis=1)
        if table.empty:
            return
        values = table.to_numpy(float)
        fig, ax = plt.subplots(
            figsize=(max(7.5, 1.05 * len(table.columns)), max(5.5, 0.75 * len(table.index)))
        )
        image = ax.imshow(values, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(table.columns)), [f"{value:g}" for value in table.columns])
        ax.set_yticks(np.arange(len(table.index)), [f"{value:g}" for value in table.index])
        ax.set_xlabel(axis_label(x_column))
        ax.set_ylabel(axis_label(y_column))
        ax.set_title(title)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if not np.isfinite(value):
                    continue
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.{digits}f}",
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.72,
                    },
                )
        fig.colorbar(image, ax=ax, label=colourbar)
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=180)
        plt.close(fig)

    heatmap(
        "lap_time_median_s",
        "design_grid_lap_time.png",
        "Median lap time across the two-dimensional design grid",
        "Median lap time [s]",
    )
    heatmap(
        "paired_regret_median_s",
        "design_grid_regret.png",
        "Median paired regret across the design grid",
        "Median paired regret [s]",
    )
    heatmap(
        "paired_win_fraction",
        "design_grid_win_fraction.png",
        "Paired win fraction within the selected worlds",
        "Paired win fraction",
        digits=3,
    )
    heatmap(
        "opportunity_loss_median_kj",
        "design_grid_opportunity_loss.png",
        "Finite-ratio opportunity loss across the design grid",
        "Median opportunity loss [kJ]",
    )
    heatmap(
        "maximum_ratio_time_median_s",
        "design_grid_max_ratio_time.png",
        "Time at maximum CVT reduction across the design grid",
        "Median time at maximum ratio [s]",
    )
    heatmap(
        "minimum_ratio_time_median_s",
        "design_grid_min_ratio_time.png",
        "Time at minimum CVT reduction across the design grid",
        "Median time at minimum ratio [s]",
    )
    heatmap(
        "completion_fraction",
        "design_grid_completion.png",
        "Completion fraction across the design grid",
        "Completion fraction",
        digits=3,
    )
'''
    source = replace_top_level_function(source, "_design_plots", plots)
    source = source.replace(
        '("Best median lap time", str(winner), "good" if len(ranking) else "warning"),',
        '("Top paired design", str(winner), "good" if len(ranking) else "warning"),',
    )
    old = '    body += "<h2>Absolute performance</h2>"\n'
    new = '''    grid_lap_time = plots / "design_grid_lap_time.png"
    if grid_lap_time.is_file():
        body += "<h2>Two-dimensional design grid</h2>"
        body += (
            '<p class="subtitle">Each cell is one simultaneous combination of the two configured '
            'design variables, evaluated on the same paired uncertainty-informed worlds.</p>'
        )
        body += figure(grid_lap_time, "Median bounded lap time for every Cartesian design combination.")
        body += figure(plots / "design_grid_regret.png", "Median paired regret relative to the best candidate in each selected world.")
        body += figure(plots / "design_grid_win_fraction.png", "Fraction of selected paired worlds won by each combination; this screening fraction is not a calibrated probability.")
        body += figure(plots / "design_grid_opportunity_loss.png", "Median finite-ratio opportunity loss for every combination.")
        body += figure(plots / "design_grid_max_ratio_time.png", "Median time pinned at maximum CVT reduction, indicating low-speed ratio demand.")
        body += figure(plots / "design_grid_min_ratio_time.png", "Median time pinned at minimum CVT reduction, indicating high-speed ratio demand.")
        body += figure(plots / "design_grid_completion.png", "Completion reliability for every design-grid cell.")
    body += "<h2>Absolute performance</h2>"
'''
    if 'grid_lap_time = plots / "design_grid_lap_time.png"' not in source:
        if old not in source:
            raise RuntimeError("Could not locate design report absolute-performance section")
        source = source.replace(old, new, 1)
    path.write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    paths = {
        "planning": repo / "src/cvt_track_study/studies/planning.py",
        "service": repo / "src/cvt_track_study/studies/service_v8.py",
        "ensemble": repo / "src/cvt_track_study/studies/ensemble_v10.py",
        "validation": repo / "src/cvt_track_study/config/validation.py",
        "postprocess": repo / "src/cvt_track_study/reports/postprocess.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("Required repository files are missing:\n" + "\n".join(missing))

    patch_planning(paths["planning"])
    patch_service_v8(paths["service"])
    patch_ensemble(paths["ensemble"])
    patch_validation(paths["validation"])
    patch_postprocess(paths["postprocess"])
    print("Patched Cartesian design planning, execution, validation, and reporting.")


if __name__ == "__main__":
    main()
