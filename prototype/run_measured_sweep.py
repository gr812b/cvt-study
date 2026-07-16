from __future__ import annotations

import argparse
from pathlib import Path

from standard_case import (
    STANDARD_FINAL_DRIVE_RATIO, STANDARD_INTEGRATION_STEP_S,
    STANDARD_MAXIMUM_CVT_RATIO, STANDARD_MINIMUM_CVT_RATIO,
    STANDARD_WHEEL_RADIUS_IN,
)

from decision_study import run_measured_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep one CVT/gearing variable with paired measured-input uncertainty")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--parameter", choices=["final_drive_ratio", "minimum_speed_ratio", "maximum_speed_ratio"], default="final_drive_ratio")
    parser.add_argument("--values", type=float, nargs="+", required=True)
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=20260714)
    parser.add_argument("--minimum-cvt-ratio", type=float, default=STANDARD_MINIMUM_CVT_RATIO)
    parser.add_argument("--maximum-cvt-ratio", type=float, default=STANDARD_MAXIMUM_CVT_RATIO)
    parser.add_argument("--final-drive-ratio", type=float, default=STANDARD_FINAL_DRIVE_RATIO)
    parser.add_argument("--wheel-radius-in", type=float, default=STANDARD_WHEEL_RADIUS_IN)
    parser.add_argument("--vehicle-mass-kg", type=float, default=300.0)
    parser.add_argument("--tire-peak-traction-scale", type=float, default=1.0)
    parser.add_argument("--tire-slip-stiffness", type=float, default=2500.0)
    parser.add_argument("--minimum-gate-confidence", type=float, default=60.0)
    parser.add_argument("--integration-step-s", type=float, default=STANDARD_INTEGRATION_STEP_S)
    parser.add_argument("--report-step-s", type=float, default=0.05)
    args = parser.parse_args()
    _, summary = run_measured_sweep(
        bundle_path=args.bundle,
        output_dir=args.output_dir,
        parameter=args.parameter,
        values=args.values,
        replicates=args.replicates,
        random_seed=args.random_seed,
        minimum_speed_ratio=args.minimum_cvt_ratio,
        maximum_speed_ratio=args.maximum_cvt_ratio,
        final_drive_ratio=args.final_drive_ratio,
        wheel_radius_in=args.wheel_radius_in,
        vehicle_mass_kg=args.vehicle_mass_kg,
        tire_peak_traction_scale=args.tire_peak_traction_scale,
        tire_slip_stiffness=args.tire_slip_stiffness,
        minimum_gate_confidence=args.minimum_gate_confidence,
        integration_step_s=args.integration_step_s,
        report_step_s=args.report_step_s,
    )
    best = summary.sort_values("opportunity_loss_energy_kj_median").iloc[0]
    print(f"Completed uncertainty sweep: {args.output_dir.resolve()}")
    print(
        f"best median opportunity-loss value: {float(best['parameter_value']):g}; "
        f"median={float(best['opportunity_loss_energy_kj_median']):.3f} kJ"
    )


if __name__ == "__main__":
    main()

