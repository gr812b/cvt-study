from __future__ import annotations

import argparse
from pathlib import Path

from standard_case import (
    STANDARD_FINAL_DRIVE_RATIO, STANDARD_INTEGRATION_STEP_S,
    STANDARD_MAXIMUM_CVT_RATIO, STANDARD_MINIMUM_CVT_RATIO,
    STANDARD_WHEEL_RADIUS_IN,
)

from decision_study import run_measured_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one bounded ideal CVT with the measured-gate unbounded reference")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-cvt-ratio", type=float, default=STANDARD_MINIMUM_CVT_RATIO)
    parser.add_argument("--maximum-cvt-ratio", type=float, default=STANDARD_MAXIMUM_CVT_RATIO)
    parser.add_argument("--final-drive-ratio", type=float, default=STANDARD_FINAL_DRIVE_RATIO)
    parser.add_argument("--wheel-radius-in", type=float, default=STANDARD_WHEEL_RADIUS_IN)
    parser.add_argument("--vehicle-mass-kg", type=float, default=300.0)
    parser.add_argument("--tire-peak-traction-scale", type=float, default=1.0)
    parser.add_argument("--tire-slip-stiffness", type=float, default=2500.0)
    parser.add_argument("--minimum-gate-confidence", type=float, default=60.0)
    parser.add_argument("--gate-speed-quantile", choices=["p10", "median", "p90"], default="median")
    parser.add_argument("--loss-quantile", choices=["low", "nominal", "high"], default="nominal")
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--integration-step-s", type=float, default=STANDARD_INTEGRATION_STEP_S)
    parser.add_argument("--report-step-s", type=float, default=0.02)
    args = parser.parse_args()
    result = run_measured_comparison(
        bundle_path=args.bundle,
        output_dir=args.output_dir,
        minimum_speed_ratio=args.minimum_cvt_ratio,
        maximum_speed_ratio=args.maximum_cvt_ratio,
        final_drive_ratio=args.final_drive_ratio,
        wheel_radius_in=args.wheel_radius_in,
        vehicle_mass_kg=args.vehicle_mass_kg,
        tire_peak_traction_scale=args.tire_peak_traction_scale,
        tire_slip_stiffness=args.tire_slip_stiffness,
        minimum_gate_confidence=args.minimum_gate_confidence,
        gate_speed_quantile=args.gate_speed_quantile,
        loss_quantile=args.loss_quantile,
        loss_scale=args.loss_scale,
        integration_step_s=args.integration_step_s,
        report_step_s=args.report_step_s,
    )
    summary = result["bounded_summary"]
    print(f"Completed measured comparison: {args.output_dir.resolve()}")
    print(
        f"time penalty={float(summary['lap_time_penalty_vs_infinite_s']):.3f} s, "
        f"opportunity loss={float(summary['finite_ratio_opportunity_loss_energy_kj']):.3f} kJ, "
        f"reference dominance={summary['reference_dominance_pass']}"
    )


if __name__ == "__main__":
    main()

