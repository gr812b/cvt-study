# Repository guide

## Root simulator

- `standard_case.py` — canonical drivetrain and tire defaults
- `models.py` — engine, vehicle, tire, CVT, driver, and settings dataclasses
- `simulation.py` — longitudinal simulation, robust implicit tire-slip solve, measured braking envelopes, and unbounded reference
- `metrics.py` — energy, time, ratio occupancy, gate compliance, and reference comparisons
- `decision_study.py` — measured comparison and paired uncertainty orchestration
- `run_measured.py` — one bounded versus unbounded measured-track run
- `run_measured_sweep.py` — paired design sweep with confidence intervals
- `run_structural_sensitivity.py` — seven-level outer vehicle-model sensitivity
- `run_single.py` / `run_sweep.py` — synthetic JSON-track studies

## GPS and gate inference

`baja_track_validation/` is installable on its own. Its pipeline performs intake validation, GPS cleaning, lap selection, shared-centerline construction, map matching, event metrics, grouping QA, slowdown signatures, confidence-scored gate inference, and bundle export.

## Track representations

- `sample_outputs/measured_gate_study_vehicle_A/simulator_track_bundle.json` is the bridge from measured GPS analysis to the simulator.
- `tracks/*.json` are synthetic tracks for isolated model studies.
- `track_builder/` defines reusable synthetic and effective-energy feature types.

## Included reference outputs

- `sample_outputs/measured_gate_study_vehicle_A/` — one-vehicle gate evidence, maps, tables, and simulator bundle
- `sample_outputs/measured_single_standard_case/` — standard 3.5–0.9, 7.556, 22-in-diameter measured comparison
- `sample_outputs/structural_sensitivity_drag_example/` — seven-case smoke example for the structural sensitivity path

Large full sensitivity studies are intentionally not bundled; regenerate them locally using the top-level README command.
