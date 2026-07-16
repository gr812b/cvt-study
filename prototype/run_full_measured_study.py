from __future__ import annotations

import argparse
from pathlib import Path

from standard_case import (
    STANDARD_FINAL_DRIVE_RATIO, STANDARD_INTEGRATION_STEP_S,
    STANDARD_MAXIMUM_CVT_RATIO, STANDARD_MINIMUM_CVT_RATIO,
    STANDARD_WHEEL_RADIUS_IN,
)
import sys


ROOT = Path(__file__).resolve().parent
VALIDATION_SOURCE = ROOT / "baja_track_validation" / "src"
if str(VALIDATION_SOURCE) not in sys.path:
    sys.path.insert(0, str(VALIDATION_SOURCE))

from baja_track_analysis import PipelineConfig, run_gate_study  # noqa: E402
from decision_study import run_measured_comparison, run_measured_sweep  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run gate inference, bounded/unbounded comparison, and an optional paired sweep"
    )
    parser.add_argument("--gps", type=Path, nargs="+", required=True)
    parser.add_argument("--vehicle-ids", nargs="+")
    parser.add_argument("--definitions", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-cvt-ratio", type=float, default=STANDARD_MINIMUM_CVT_RATIO)
    parser.add_argument("--maximum-cvt-ratio", type=float, default=STANDARD_MAXIMUM_CVT_RATIO)
    parser.add_argument("--final-drive-ratio", type=float, default=STANDARD_FINAL_DRIVE_RATIO)
    parser.add_argument("--wheel-radius-in", type=float, default=STANDARD_WHEEL_RADIUS_IN)
    parser.add_argument("--vehicle-mass-kg", type=float, default=300.0)
    parser.add_argument("--minimum-gate-confidence", type=float, default=60.0)
    parser.add_argument("--integration-step-s", type=float, default=STANDARD_INTEGRATION_STEP_S)
    parser.add_argument(
        "--sweep-parameter",
        choices=["final_drive_ratio", "minimum_speed_ratio", "maximum_speed_ratio"],
    )
    parser.add_argument("--sweep-values", type=float, nargs="+")
    parser.add_argument("--sweep-replicates", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=20260714)
    args = parser.parse_args()

    if (args.sweep_parameter is None) != (args.sweep_values is None):
        parser.error("--sweep-parameter and --sweep-values must be supplied together")
    if args.vehicle_ids is not None and len(args.vehicle_ids) != len(args.gps):
        parser.error("--vehicle-ids must contain one ID per --gps file")

    validation_dir = args.output_dir / "01_track_validation"
    comparison_dir = args.output_dir / "02_bounded_vs_unbounded"
    sweep_dir = args.output_dir / "03_design_sweep"
    config = PipelineConfig.from_toml(args.config)
    gate_result = run_gate_study(
        args.gps,
        args.definitions,
        validation_dir,
        vehicle_ids=args.vehicle_ids,
        config=config,
    )
    bundle = validation_dir / "simulator_track_bundle.json"
    comparison = run_measured_comparison(
        bundle_path=bundle,
        output_dir=comparison_dir,
        minimum_speed_ratio=args.minimum_cvt_ratio,
        maximum_speed_ratio=args.maximum_cvt_ratio,
        final_drive_ratio=args.final_drive_ratio,
        wheel_radius_in=args.wheel_radius_in,
        vehicle_mass_kg=args.vehicle_mass_kg,
        minimum_gate_confidence=args.minimum_gate_confidence,
        integration_step_s=args.integration_step_s,
    )
    if args.sweep_parameter is not None:
        run_measured_sweep(
            bundle_path=bundle,
            output_dir=sweep_dir,
            parameter=args.sweep_parameter,
            values=args.sweep_values,
            replicates=args.sweep_replicates,
            random_seed=args.random_seed,
            minimum_speed_ratio=args.minimum_cvt_ratio,
            maximum_speed_ratio=args.maximum_cvt_ratio,
            final_drive_ratio=args.final_drive_ratio,
            wheel_radius_in=args.wheel_radius_in,
            vehicle_mass_kg=args.vehicle_mass_kg,
            minimum_gate_confidence=args.minimum_gate_confidence,
            integration_step_s=args.integration_step_s,
        )

    summary = comparison["bounded_summary"]
    print(f"Completed full measured study: {args.output_dir.resolve()}")
    print(
        f"datasets={len(args.gps)}, retained_laps={len(gate_result.laps)}, "
        f"accepted_gates={int(gate_result.speed_gates['accepted_by_default'].sum())}"
    )
    print(
        f"time_penalty={float(summary['lap_time_penalty_vs_infinite_s']):.3f} s, "
        f"opportunity_loss={float(summary['finite_ratio_opportunity_loss_energy_kj']):.3f} kJ"
    )


if __name__ == "__main__":
    main()

