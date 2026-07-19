"""Absolute one-at-a-time structural response summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cvt_track_study.simulation.service import (
    SimulationError,
)


METRIC_DEFINITIONS: dict[
    str, tuple[str, str]
] = {
    "bounded_lap_time_s": (
        "Bounded lap time",
        "s",
    ),
    "bounded_maximum_speed_kmh": (
        "Maximum speed",
        "km/h",
    ),
    "bounded_average_speed_kmh": (
        "Average speed",
        "km/h",
    ),
    "bounded_distance_m": (
        "Distance completed",
        "m",
    ),
    "bounded_engine_energy_kj": (
        "Engine energy",
        "kJ",
    ),
    "bounded_transmitted_energy_kj": (
        "Transmitted energy",
        "kJ",
    ),
    "bounded_drivetrain_loss_energy_kj": (
        "Drivetrain loss",
        "kJ",
    ),
    "bounded_clutch_loss_energy_kj": (
        "Clutch loss",
        "kJ",
    ),
    "bounded_engine_operating_shortfall_energy_kj": (
        "Engine operating shortfall",
        "kJ",
    ),
    "bounded_tire_slip_loss_energy_kj": (
        "Tire-slip loss",
        "kJ",
    ),
    "bounded_brake_loss_energy_kj": (
        "Brake loss",
        "kJ",
    ),
    "bounded_rolling_loss_energy_kj": (
        "Rolling loss",
        "kJ",
    ),
    "bounded_aerodynamic_loss_energy_kj": (
        "Aerodynamic loss",
        "kJ",
    ),
    "bounded_obstacle_loss_energy_kj": (
        "Obstacle loss",
        "kJ",
    ),
    "bounded_time_maximum_ratio_s": (
        "Time at maximum ratio",
        "s",
    ),
    "bounded_time_variable_ratio_s": (
        "Time in variable ratio",
        "s",
    ),
    "bounded_time_minimum_ratio_s": (
        "Time at minimum ratio",
        "s",
    ),
    "bounded_time_braking_s": (
        "Braking time",
        "s",
    ),
    "bounded_time_traction_limited_s": (
        "Traction-limited time",
        "s",
    ),
    "lap_time_penalty_vs_infinite_s": (
        "Finite-ratio lap-time penalty",
        "s",
    ),
    "finite_ratio_opportunity_loss_energy_kj": (
        "Finite-ratio opportunity loss",
        "kJ",
    ),
}

HEADLINE_METRICS = (
    "bounded_lap_time_s",
    "bounded_maximum_speed_kmh",
    "bounded_engine_energy_kj",
    "bounded_obstacle_loss_energy_kj",
    "lap_time_penalty_vs_infinite_s",
    "finite_ratio_opportunity_loss_energy_kj",
)


def summarize_structural_screening(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise SimulationError(
            "Structural sensitivity cannot be summarized without cases."
        )

    parameters: dict[str, Any] = {}
    paths = sorted(
        {
            str(row["parameter_path"])
            for row in rows
        }
    )
    for path in paths:
        subset = [
            row
            for row in rows
            if str(
                row["parameter_path"]
            )
            == path
        ]
        nominal_rows = [
            row
            for row in subset
            if row.get("level_kind")
            == "nominal"
        ]
        if len(nominal_rows) != 1:
            raise SimulationError(
                f"Structural parameter {path!r} must have exactly one nominal case."
            )
        nominal = nominal_rows[0]
        metric_ranges: dict[str, Any] = {}
        for metric, (
            label,
            unit,
        ) in METRIC_DEFINITIONS.items():
            values = [
                float(row[metric])
                for row in subset
                if _finite(
                    row.get(metric)
                )
            ]
            if not values:
                continue
            nominal_value = float(
                nominal[metric]
            )
            minimum_index = int(
                np.argmin(values)
            )
            maximum_index = int(
                np.argmax(values)
            )
            valid_rows = [
                row
                for row in subset
                if _finite(
                    row.get(metric)
                )
            ]
            changes = [
                float(value)
                - nominal_value
                for value in values
            ]
            percent_changes = [
                (
                    100.0
                    * change
                    / abs(nominal_value)
                    if abs(nominal_value)
                    > 1e-12
                    else None
                )
                for change in changes
            ]
            metric_ranges[metric] = {
                "label": label,
                "unit": unit,
                "nominal": nominal_value,
                "minimum": min(values),
                "maximum": max(values),
                "span": (
                    max(values)
                    - min(values)
                ),
                "minimum_change_from_nominal": min(
                    changes
                ),
                "maximum_change_from_nominal": max(
                    changes
                ),
                "maximum_abs_change_from_nominal": max(
                    abs(value)
                    for value in changes
                ),
                "maximum_abs_percent_change_from_nominal": (
                    max(
                        abs(value)
                        for value in percent_changes
                        if value is not None
                    )
                    if any(
                        value is not None
                        for value in percent_changes
                    )
                    else None
                ),
                "minimum_design_id": valid_rows[
                    minimum_index
                ]["design_id"],
                "maximum_design_id": valid_rows[
                    maximum_index
                ]["design_id"],
            }

        levels: list[dict[str, Any]] = []
        for row in _ordered_levels(subset):
            record: dict[str, Any] = {
                "design_id": row["design_id"],
                "level_kind": row[
                    "level_kind"
                ],
                "level_probability": row.get(
                    "level_probability"
                ),
                "value": row[
                    "design_value"
                ],
                "value_si": row.get(
                    "design_value_si"
                ),
                "choice_value": row.get(
                    "design_choice_value"
                ),
                "bounded_completed": bool(
                    row["bounded_completed"]
                ),
                "bounded_termination_reason": row.get(
                    "bounded_termination_reason",
                    "",
                ),
            }
            for metric in METRIC_DEFINITIONS:
                if metric not in row:
                    continue
                value = float(row[metric])
                record[metric] = value
                record[
                    f"{metric}.change_from_nominal"
                ] = (
                    value
                    - float(nominal[metric])
                )
            levels.append(record)

        completed_count = sum(
            bool(row["bounded_completed"])
            for row in subset
        )
        parameters[path] = {
            "category": _parameter_category(
                path
            ),
            "nominal_value": nominal[
                "design_value"
            ],
            "nominal_value_si": nominal.get(
                "design_value_si"
            ),
            "nominal_choice_value": nominal.get(
                "design_choice_value"
            ),
            "level_count": len(subset),
            "completed_level_count": (
                completed_count
            ),
            "all_levels_completed": (
                completed_count == len(subset)
            ),
            "metrics": metric_ranges,
            "levels": levels,
            # Backward-compatible headline fields.
            "time_penalty_span_s": float(
                metric_ranges.get(
                    "lap_time_penalty_vs_infinite_s",
                    {},
                ).get("span", 0.0)
            ),
            "opportunity_loss_span_kj": float(
                metric_ranges.get(
                    "finite_ratio_opportunity_loss_energy_kj",
                    {},
                ).get("span", 0.0)
            ),
            "absolute_lap_time_span_s": float(
                metric_ranges.get(
                    "bounded_lap_time_s",
                    {},
                ).get("span", 0.0)
            ),
            "maximum_speed_span_kmh": float(
                metric_ranges.get(
                    "bounded_maximum_speed_kmh",
                    {},
                ).get("span", 0.0)
            ),
        }

    rankings: dict[str, list[dict[str, Any]]] = {}
    for metric, (
        label,
        unit,
    ) in METRIC_DEFINITIONS.items():
        entries = []
        for path, parameter in parameters.items():
            record = parameter[
                "metrics"
            ].get(metric)
            if not record:
                continue
            entries.append(
                {
                    "path": path,
                    "category": parameter[
                        "category"
                    ],
                    "label": label,
                    "unit": unit,
                    "nominal": record[
                        "nominal"
                    ],
                    "minimum_change_from_nominal": record[
                        "minimum_change_from_nominal"
                    ],
                    "maximum_change_from_nominal": record[
                        "maximum_change_from_nominal"
                    ],
                    "span": record[
                        "span"
                    ],
                    "maximum_abs_change_from_nominal": record[
                        "maximum_abs_change_from_nominal"
                    ],
                    "maximum_abs_percent_change_from_nominal": record[
                        "maximum_abs_percent_change_from_nominal"
                    ],
                }
            )
        entries.sort(
            key=lambda item: float(
                item[
                    "maximum_abs_change_from_nominal"
                ]
            ),
            reverse=True,
        )
        denominator = max(
            (
                float(
                    item[
                        "maximum_abs_change_from_nominal"
                    ]
                )
                for item in entries
            ),
            default=0.0,
        )
        for rank, item in enumerate(
            entries, start=1
        ):
            item["rank"] = rank
            item[
                "relative_screening_importance"
            ] = (
                float(
                    item[
                        "maximum_abs_change_from_nominal"
                    ]
                )
                / denominator
                if denominator > 0.0
                else 0.0
            )
        rankings[metric] = entries

    return {
        "study_type": "structural_sensitivity",
        "method": (
            "deterministic one-at-a-time screening over every selected "
            "declared structural uncertainty level"
        ),
        "statistical_error_bars_applicable": False,
        "parameter_count": len(parameters),
        "level_count": len(rows),
        "completed_level_count": sum(
            bool(row["bounded_completed"])
            for row in rows
        ),
        "parameters": parameters,
        "rankings": rankings,
        "headline_metrics": list(
            HEADLINE_METRICS
        ),
        "metric_definitions": {
            key: {
                "label": value[0],
                "unit": value[1],
            }
            for key, value in METRIC_DEFINITIONS.items()
        },
    }


def metric_range_rows(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, parameter in summary.get(
        "parameters", {}
    ).items():
        for metric, record in parameter.get(
            "metrics", {}
        ).items():
            rows.append(
                {
                    "parameter_path": path,
                    "category": parameter.get(
                        "category"
                    ),
                    "metric": metric,
                    **record,
                }
            )
    return rows


def parameter_level_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "parameter_path",
        "design_id",
        "level_kind",
        "level_probability",
        "design_value",
        "design_value_si",
        "design_choice_value",
        "bounded_completed",
        "bounded_termination_reason",
        *METRIC_DEFINITIONS.keys(),
    )
    return [
        {
            key: row.get(key)
            for key in fields
        }
        for row in rows
    ]


def _ordered_levels(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            0
            if item.get("level_kind")
            == "nominal"
            else 1,
            (
                float(
                    item[
                        "level_probability"
                    ]
                )
                if item.get(
                    "level_probability"
                )
                not in (None, "")
                else str(
                    item.get(
                        "design_value", ""
                    )
                )
            ),
        ),
    )


def _parameter_category(
    path: str,
) -> str:
    if path.startswith("vehicle."):
        return "vehicle"
    if path.startswith("drivetrain."):
        return "drivetrain"
    if path.startswith("driver."):
        return "driver"
    if path.startswith("track.surface."):
        return "surface"
    if path.startswith("obstacle."):
        return "obstacle"
    return path.split(".", 1)[0]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
