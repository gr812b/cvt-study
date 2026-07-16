# Phase 5 checkpoint — explicit obstacle and vehicle simulation

Phase 5 moves the nominal track study onto the versioned Phase 4 bundle boundary.
The simulator no longer reads GPX, reconstruction frames, event spreadsheets, or
map-matching internals. It combines one validated `track_bundle.json` with one
resolved vehicle, drivetrain, driver, and study configuration.

Phase 6 has not been started. Every Phase 5 run uses the nominal value from each
uncertainty declaration.

## User entry point

```powershell
cvt-study run baseline .\examples\arizona_endurance_project
```

By default the command rebuilds the track first so stale evidence is not used
silently. A reviewed bundle may be supplied explicitly:

```powershell
cvt-study run baseline .\examples\arizona_endurance_project `
  --bundle .\path\to\track_bundle.json `
  --output .\examples\arizona_endurance_project\results\baseline\reviewed_run
```

A successful run atomically publishes bounded and infinite-reference traces,
summary JSON, gate compliance, per-feature obstacle energy, plots, resolved
inputs, a portable run manifest, and a copy of the exact bundle used.

## Implemented mechanisms

### Track and gate boundary

- Reads only schema-compatible `cvt-track-bundle` version `1.1.x`.
- Uses physical-feature intervals independently from GPS response groups.
- Uses only accepted, active speed gates.
- Selects gate targets from the declared empirical statistic (`p10`, `median`,
  or `p90`).
- Enforces each gate as a one-way ceiling through a finite backward braking
  envelope. A slow vehicle is never accelerated to a gate target.
- Inserts exact gate-crossing samples into the exported trace.
- Retains GPX elevation and centreline curvature as diagnostics only.

### Explicit obstacle models

Every physical feature must explicitly select one model, including `none`:

- `none`
- `fixed_specific_energy`
- `speed_quadratic_energy`
- `distributed_resistance`
- `roughness_energy_density`
- `smooth_profile`

Each model has a strict parameter list, dimensions, provenance, and uncertainty.
Lumped energy is distributed with a normalized raised-cosine spatial density.
Speed-dependent impact loss captures vehicle speed once at physical feature entry;
it does not repeatedly reduce the requested loss as the vehicle slows inside the
feature.

Built-in profiles provide broad uncertainty-aware starting priors for common
feature classes. They are defaults, not calibration.

### Vehicle and tire host

The reduced state contains track distance, vehicle speed, and driven-wheel speed.
The model includes:

- vehicle longitudinal inertia;
- driven wheel/axle rotational inertia;
- aerodynamic drag;
- rolling resistance;
- explicit feature grade and normal-load effects from `smooth_profile` only;
- a two-parameter saturating longitudinal tire law;
- tire-slip energy;
- finite brake force and tire-limited braking.

The tire slip coordinate is advanced by backward Euler. Its monotone scalar
residual is solved with deterministic bracketed bisection. SciPy is allowed in the
repository, but a general nonlinear solver is not advantageous for this particular
one-dimensional guaranteed-bracket problem.

### Bounded and infinite ideal-CVT cases

The bounded ideal CVT attempts to hold the declared engine target speed and clips
the required ratio to the declared ratio range. At low wheel speed, the optional
ideal launch clutch permits target engine speed while transmitting only the torque
available at maximum reduction.

The infinite reference keeps the same:

- engine curve and target;
- drivetrain efficiency;
- finite launch-torque capability;
- vehicle, tire, driver, obstacles, and gates.

It removes only the finite CVT ratio window. The reference therefore cannot create
infinite launch torque. Launch-clutch loss shared by both cases is measured in
both and removed from the reported finite-ratio-only opportunity loss.

### Energy accounting

Energy is integrated at the solver step, not reconstructed from the downsampled
report trace. Outputs distinguish:

- engine energy;
- mechanical work delivered to the driven wheel;
- drivetrain-efficiency loss;
- launch-clutch loss;
- engine off-target operating shortfall;
- tire-slip loss;
- braking loss;
- rolling loss;
- aerodynamic loss;
- obstacle loss by physical feature;
- conservative grade work;
- initial and final kinetic energy.

For one case,

\[
E_{\mathrm{opp,case}} = E_{\mathrm{clutch}} + E_{\mathrm{off\text{-}peak}}.
\]

For the paired comparison,

\[
E_{\mathrm{opp,finite}}=
\max\left(0,E_{\mathrm{opp,bounded}}-E_{\mathrm{opp,infinite}}\right).
\]

The subtraction removes the launch loss common to both cases. Physical loss bars
and opportunity diagnostics are not all additive; explicit vehicle-level and
engine-to-wheel residuals provide the closure checks.

## Real Arizona nominal result

The canonical example uses the supplied 6,822-point GPX recording and the reviewed
Phase 4 track definition:

- track length: 1,773.6 m;
- physical features: 40;
- response groups: 37;
- active accepted gates: 13;
- valid evidence laps: 11;
- vehicle mass: 245 kg;
- tire diameter: 22 in;
- CVT range: 3.5 to 0.9;
- final drive: 7.556;
- drivetrain efficiency: 0.80.

At the 1 ms engineering timestep:

| Metric | Result |
|---|---:|
| Bounded lap time | 139.313 s |
| Infinite-reference lap time | 134.284 s |
| Time penalty | 5.029 s |
| Bounded clutch + off-peak loss | 173.379 kJ |
| Shared reference launch loss | 6.287 kJ |
| Finite-ratio-only opportunity loss | 167.091 kJ |
| Bounded vehicle-energy residual | -0.0497% |
| Reference vehicle-energy residual | -0.0431% |
| Bounded engine-to-wheel residual | -0.4708% |
| Reference engine-to-wheel residual | -0.4274% |

A complete 2 ms artifact was also generated for portable output inspection:

- 13/13 bounded gates compliant within 0.5 km/h;
- 13/13 reference gates compliant within 0.5 km/h;
- bounded vehicle-energy residual: -0.2732%;
- reference vehicle-energy residual: -0.2425%;
- per-feature obstacle energies sum to total obstacle energy to floating-point
  precision.

The 2 ms minus 1 ms comparison was:

| Metric | Difference |
|---|---:|
| Bounded lap time | +0.0390 s |
| Reference lap time | +0.0081 s |
| Time penalty | +0.0309 s |
| Finite-ratio opportunity loss | -0.1617 kJ |

These values are a deterministic mechanism check, not calibrated confidence
bounds. The Arizona validation still warns that seven vehicle inputs use broad
inherited priors, including wheel inertia, driven-load fraction, air density, and
tire parameters.

## Output contract

A baseline directory contains:

```text
bounded_trace.csv
infinite_reference_trace.csv
bounded_summary.json
infinite_reference_summary.json
comparison_summary.json
gate_compliance.csv
obstacle_energy_by_feature.csv
resolved_simulation_case.json
resolved_inputs/
track_bundle.json
track_bundle.sha256
run_manifest.json
REPORT.md
01_speed_comparison.png
02_ratio_trace.png
03_energy_accounting.png
```

The copied bundle and its checksum make the result replayable without the original
project workspace. The manifest references the local `track_bundle.json`, not a
machine-specific absolute path.

## Verification

- 101 clean-package tests passed in reviewed split suites.
- 16 preserved prototype regression tests passed.
- Python source compilation passed.
- Public simulation and bundle annotations resolve at runtime.
- Real Arizona nominal runs completed at 1 ms and 2 ms.
- Fresh wheel build and installation passed outside the source tree.
- Installed-wheel `init`, `validate`, `build-track`, `validate-bundle`, and
  `run baseline` passed.
- Source and installed-wheel 2 ms comparison summaries were identical.
- Generated plots and result tables were manually inspected.

## Deliberate limitations retained for later phases

- No statistical sampling or confidence bands yet.
- No structural sensitivity or uncertainty attribution yet.
- GPX altitude is not used as road grade.
- Centreline curvature does not consume tire capacity; no lateral/yaw model exists.
- Tire behavior is a reduced saturating law without post-peak force drop,
  temperature, soil state, or weight transfer.
- The engine curve and several vehicle/obstacle profiles are broad engineering
  priors rather than vehicle-specific calibration.
- The CVT is an ideal bounded-ratio mechanism, not the transient rubber-belt
  mechanics model.
- The driver is a simple full-propulsion/finite-braking controller around accepted
  gates.
