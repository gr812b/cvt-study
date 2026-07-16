# Baseline simulation outputs

`cvt-study run baseline` publishes one atomic result directory. A failed run does
not leave a partially named final result.

- `bounded_trace.csv`: full reported bounded-CVT trajectory.
- `infinite_reference_trace.csv`: otherwise-identical ideal-ratio trajectory.
- `bounded_summary.json`, `infinite_reference_summary.json`: time, ratio occupancy,
  traction, energy, and numerical checks.
- `comparison_summary.json`: lap-time penalty, bounded total opportunity loss,
  shared infinite-reference launch loss, finite-ratio-only opportunity loss,
  dominance, and energy-balance errors.
- `gate_compliance.csv`: exact crossing speed at every active gate.
- `obstacle_energy_by_feature.csv`: physical feature model, captured entry speed,
  and bounded/reference energy.
- `resolved_simulation_case.json`: nominal mechanism inputs actually used.
- `resolved_inputs/`: complete configuration and provenance.
- `track_bundle.json`, `track_bundle.sha256`: exact validated bundle copied into
  the result so the run can be replayed without the original project workspace.
- `run_manifest.json`: portable bundle reference, bundle identity, sampling mode,
  and capability flags.

The trace contains reference GPX elevation, modeled feature elevation, grade,
curvature, longitudinal tire state, controls, CVT state, engine state, individual
resistance forces and powers, and kinetic energies.

A negative gate excess means the simulation crossed below the ceiling. The
0.5 km/h compliance flag is a numerical/controller check, not a confidence
statement about the measured target.

The energy plot intentionally places physical losses beside the off-peak
opportunity diagnostic, but those bars are not a single additive energy balance.
Use the explicit residual fields for closure.

The finite-ratio opportunity metric subtracts the launch-clutch loss shared by the
infinite reference from the bounded case's clutch-plus-off-peak total. Physical
drivetrain-efficiency loss remains a separate energy-accounting term because the
same efficiency applies to both designs.

The vehicle-level and engine-to-wheel energy-balance errors should be inspected
together with a timestep comparison. Phase 5 does not create statistical output
intervals; all values are nominal.
