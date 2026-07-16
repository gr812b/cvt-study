# Phase 5 code review

## Scope reviewed

The review covered the new simulation boundary, obstacle contracts, runtime track
adapter, vehicle/tire dynamics, ideal-CVT models, integrator, energy accounting,
reporting, CLI, templates, built-in profiles, tests, real Arizona output, and
isolated-wheel behavior.

## Findings corrected before release

### 1. Obstacle behavior was initially too easy to leave implicit

A physical feature without a declared model could otherwise inherit old behavior
or be mistaken for a calibrated loss. The final contract rejects undeclared
models. `none` is an explicit uncertainty-aware choice.

### 2. Speed-dependent loss must use physical entry speed

Re-evaluating \(k_{impact}v^2\) from local speed inside the interval makes the
requested event energy shrink as the event itself slows the vehicle. Entry speed
is now captured once and used for the whole feature. Regression tests hold local
speed fixed and varying to prove this behavior.

### 3. Raw GPX elevation and centreline curvature were overreaching

An early implementation allowed centreline curvature to consume longitudinal tire
capacity. That quietly inserted an unvalidated lateral vehicle model and defeated
the intended free-propagation-between-gates architecture. Curvature and lateral
force demand are now diagnostics only. Raw GPX elevation remains stored but does
not create grade force.

### 4. The infinite reference could imply unlimited launch torque

Holding finite power at zero wheel speed formally requests infinite wheel torque.
The reference now shares the bounded design's maximum-ratio launch torque cap.
This isolates removal of the ratio window rather than granting a fictitious launch
actuator.

### 5. Shared launch loss distorted opportunity-loss attribution

Once the reference received a finite launch-torque cap, its uncoupled launch power
also had to be accounted for. Both cases now record launch-clutch loss. The paired
finite-ratio metric subtracts the reference's shared launch loss from the bounded
clutch-plus-off-peak total.

### 6. Drivetrain efficiency was not visible in the energy ledger

The same efficiency affected wheel torque but was not separately reported.
Drivetrain-efficiency loss is now integrated explicitly and included in the
engine-to-wheel residual.

### 7. Report-grid integration hid numerical behavior

Energy reconstructed from a coarser exported trace changed with the reporting
interval. All work and loss terms are now integrated at the solver step. The
report grid is presentation-only.

### 8. Gate compliance depended on interpolation resolution

A gate could be physically crossed between report samples and then appear to miss
its target. Exact crossing times are inserted into the public trace, and gate
compliance is evaluated from those samples.

### 9. Wheel-speed clipping left an impossible slip state

When braking locked the driven wheel, carrying the unconstrained implicit slip
root into the next step could create a large nonphysical slip state. After the
non-negative wheel-speed constraint is applied, slip is reconstructed from the
constrained vehicle and wheel speeds.

### 10. Result directories were not fully portable

A manifest originally retained the development machine's absolute bundle path.
Each baseline result now contains the exact `track_bundle.json` and checksum used,
and the manifest references that local snapshot.

### 11. Output publication needed failure isolation

Baseline output is written to a unique staging directory and atomically renamed
only after every trace, table, plot, report, and manifest succeeds. Exceptions
remove the staging directory.

### 12. Module ownership was tightened

Configuration resolution, runtime-track adaptation, obstacle equations,
powertrain behavior, dynamics, integration, metrics, reporting, and project-level
orchestration remain separate. Reporting receives completed traces and cannot
alter simulation state.

### 13. The scalar tire solve did not justify a general dependency

SciPy is acceptable for this repository and may be used where its algorithms add
value. The tire residual here is scalar, continuous, monotone, and has an exact
force-derived bracket. Deterministic bisection gives predictable branch behavior
and no solver-tolerance coupling. Scalar `math.tanh` also removed substantial
NumPy scalar-call overhead.

### 14. Energy visualization could be misread as additive

The plot includes physical dissipations and the counterfactual off-peak diagnostic
side by side. Its title and documentation now state that the bars are not a single
additive balance; closure is read from the residual fields.

## Verification observations

- The 1 ms Arizona vehicle-level energy residuals are below 0.05% in magnitude.
- Engine-to-wheel residuals are below 0.5% in magnitude at 1 ms.
- All 13 active gates pass for both cases in the inspected full output.
- Per-feature obstacle energy exactly reconciles with the total obstacle term.
- The infinite reference is faster than the bounded case.
- The 1 ms to 2 ms change is small relative to the reported design gap.
- Source-tree and installed-wheel runs produce identical 2 ms comparison JSON.

## Remaining risks and consciously deferred work

### Broad priors are not calibration

The clean example still relies on inherited assumptions for several tire,
inertia, air-density, and obstacle inputs. Phase 6 must propagate these rather
than treating the nominal baseline as truth.

### The ideal CVT is a study reference, not CINDER

The ratio is selected algebraically, with an idealized launch clutch. It does not
contain sheave inertia, mechanical actuation, belt tension distribution, or
stick/slip branch dynamics. A later integration may compare this track demand
layer with the full mechanical CVT model, but Phase 5 deliberately keeps that
boundary explicit. The broader dynamic CVT formulation remains documented
separately.

### Longitudinal tire physics is deliberately reduced

The tanh law is suitable for studying uncertainty in force ceiling and slip
buildup without pretending that available GPX identifies a full tire/soil model.
It does not represent post-peak decline, changing terrain, wheel sinkage, thermal
behavior, or load transfer.

### Driver behavior is reduced to accepted gates

The controller applies full propulsion unless a finite braking envelope requires
braking. It is not a driver model for throttle modulation, line choice, tactical
passing, or fatigue.

### Elevation and lateral dynamics remain disabled

Raw GPX elevation is too noisy to use directly as grade without a validated
processing method. Curvature remains diagnostic until a lateral/yaw and combined-
slip contract is defined.

### Powertrain closure is numerical, not algebraically forced

Wheel work uses actual average wheel speed over each step while the powertrain
sample is evaluated at the step start. The resulting residual is intentionally
reported rather than hidden by defining a loss term to force exact closure.
Timestep comparison is therefore part of acceptance.

## Review conclusion

No unresolved high-severity Phase 5 contract or implementation issue was found.
The nominal simulator is suitable as the deterministic mechanism underneath
Phase 6 paired uncertainty propagation, subject to the limitations above.
