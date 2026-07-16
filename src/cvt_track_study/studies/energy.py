"""Physical energy-accounting summaries for sampled studies."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np

from cvt_track_study.uncertainty import summarize_samples


ENGINE_PARTITION = (
    "engine_energy_kj",
    "transmitted_energy_kj",
    "drivetrain_loss_energy_kj",
    "clutch_loss_energy_kj",
    "powertrain_energy_balance_residual_kj",
)
VEHICLE_PARTITION = (
    "transmitted_energy_kj",
    "initial_total_kinetic_energy_kj",
    "final_total_kinetic_energy_kj",
    "tire_slip_loss_energy_kj",
    "brake_loss_energy_kj",
    "rolling_loss_energy_kj",
    "aerodynamic_loss_energy_kj",
    "obstacle_loss_energy_kj",
    "net_grade_work_kj",
    "energy_balance_residual_kj",
)
OPPORTUNITY_DIAGNOSTICS = (
    "engine_operating_shortfall_energy_kj",
    "finite_ratio_opportunity_loss_energy_kj",
)
LOSS_COMPONENTS = (
    "drivetrain_loss_energy_kj",
    "clutch_loss_energy_kj",
    "tire_slip_loss_energy_kj",
    "brake_loss_energy_kj",
    "rolling_loss_energy_kj",
    "aerodynamic_loss_energy_kj",
    "obstacle_loss_energy_kj",
)


def build_energy_accounting(
    rows: Sequence[Mapping[str, Any]], *, seed: int, bootstrap_resamples: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    designs: dict[str, Any] = {}
    shares: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for design_index, design_id in enumerate(
        sorted({str(row["design_id"]) for row in rows})
    ):
        subset = [row for row in rows if str(row["design_id"]) == design_id]
        design_record: dict[str, Any] = {}
        for side_index, side in enumerate(("bounded", "reference")):
            side_record: dict[str, Any] = {}
            for component_index, component in enumerate(
                dict.fromkeys((*ENGINE_PARTITION, *VEHICLE_PARTITION, *OPPORTUNITY_DIAGNOSTICS))
            ):
                key = f"{side}_{component}"
                values = [float(row[key]) for row in subset if row.get(key) is not None]
                if not values:
                    continue
                summary = summarize_samples(
                    values,
                    bootstrap_seed=(
                        seed
                        + 100003 * (design_index + 1)
                        + 1009 * (side_index + 1)
                        + component_index
                    ),
                    bootstrap_resamples=max(100, bootstrap_resamples),
                )
                side_record[component] = summary.as_dict()
            design_record[side] = side_record
        designs[design_id] = design_record

        for row in subset:
            losses = {
                component: max(0.0, float(row.get(f"bounded_{component}", 0.0)))
                for component in LOSS_COMPONENTS
            }
            total = sum(losses.values())
            for component, value in losses.items():
                shares.append(
                    {
                        "replicate": int(row["replicate"]),
                        "design_id": design_id,
                        "component": component,
                        "energy_kj": value,
                        "share_of_reported_physical_losses": value / total if total > 0.0 else 0.0,
                    }
                )
            for side in ("bounded", "reference"):
                raw = row.get(f"{side}_obstacle_energy_by_feature_kj_json", "{}")
                mapping = json.loads(str(raw)) if isinstance(raw, str) else dict(raw or {})
                for feature_id, value in sorted(mapping.items()):
                    feature_rows.append(
                        {
                            "replicate": int(row["replicate"]),
                            "design_id": design_id,
                            "side": side,
                            "feature_id": feature_id,
                            "obstacle_energy_kj": float(value),
                        }
                    )

    return (
        {
            "contract": {
                "engine_partition": list(ENGINE_PARTITION),
                "vehicle_partition": list(VEHICLE_PARTITION),
                "opportunity_diagnostics_non_additive": list(OPPORTUNITY_DIAGNOSTICS),
                "note": (
                    "Opportunity diagnostics are counterfactual measures and are not added "
                    "to either physical energy balance."
                ),
            },
            "designs": designs,
        },
        shares,
        feature_rows,
    )
