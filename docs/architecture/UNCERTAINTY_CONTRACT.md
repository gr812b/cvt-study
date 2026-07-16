# Uncertainty-first input contract

## Core rule

A physical numeric input is invalid unless it declares:

- a nominal value;
- a recognized unit;
- provenance;
- uncertainty.

A missing uncertainty field is an error unless a complete inherited quantity is
being deliberately retained. Zero uncertainty is represented only by an explicit
fixed declaration with a reason.

```toml
[vehicle.mass]
nominal = 245.0
unit = "kg"

[vehicle.mass.source]
kind = "measured"
reference = "four-corner scales, 2026-07-10"

[vehicle.mass.uncertainty]
distribution = "normal"
confidence_half_width = 0.1
confidence_level = 0.95
```

A consciously fixed value looks like:

```toml
[drivetrain.final_drive_ratio]
nominal = 7.556
unit = "1"

[drivetrain.final_drive_ratio.source]
kind = "derived"
reference = "integer tooth counts"

[drivetrain.final_drive_ratio.uncertainty]
distribution = "fixed"
reason = "Exact ratio derived from integer tooth counts."
```

## Semantic role is separate from distribution shape

Every non-fixed uncertainty may declare a semantic role:

```toml
[vehicle.aero.drag_area.uncertainty]
distribution = "truncated_normal"
role = "structural"
standard_deviation = 0.03
lower = 0.2
upper = 1.5
```

The role controls which study samples the value:

- `structural` means uncertainty in a physical/model input;
- `measured_track` means repeatable variability supported by track evidence;
- `initial_condition` means uncertainty in the simulation starting state.

Obstacles are the ambiguous case. A broad default `impact_coefficient` is
structural uncertainty, even though it lives inside the track bundle. An
empirical severity sample measured across laps may instead be `measured_track`.
Every stochastic obstacle coefficient or model choice must therefore state its
role explicitly. A measured-track robustness study will not sample broad
structural obstacle priors.

Fixed inputs do not require a role because no study samples them. Vehicle,
drivetrain, driver, and surface quantities use structural as their location-based
default; initial conditions default to `initial_condition`. Explicit roles are
still recommended in user-edited templates because they make intent visible.

## Numeric distributions

Numeric quantities support:

- `fixed`;
- `normal`;
- `truncated_normal`;
- `uniform`;
- `triangular`;
- `empirical`.

A normal spread may be entered as exactly one of:

- absolute `standard_deviation`;
- `relative_standard_deviation`;
- symmetric `confidence_half_width` plus `confidence_level`.

The nominal must lie inside explicit uniform, triangular, or truncated-normal
bounds. A relative spread cannot be used with a zero nominal value.

## Categorical model-form uncertainty

A model choice is not a unit-bearing numeric quantity. Categorical alternatives
use an `UncertainChoice` contract with a string nominal and either fixed or
discrete uncertainty. For example, a later obstacle model may choose among
`impact`, `distributed_roughness`, and `fixed_energy` scenarios.

This separation prevents model alternatives from being disguised as arbitrary
numeric coefficients.

## Complete quantities are atomic

When a later profile or project file provides all four parts—nominal, unit, source,
and uncertainty—the quantity replaces the inherited one as an atomic object.
This is essential when changing distribution families: old triangular bounds must
not survive a switch to a normal model.

Changing only the nominal is allowed but produces a warning because inherited
source and uncertainty are retained. Command-line overrides use this behavior for
quick experiments.

## Defaults are priors, not facts

Hard-to-measure inputs may inherit broad, versioned engineering defaults. The
resolved input still reports:

- `source.kind = "inherited_default"`;
- exact profile ID and version;
- uncertainty declaration;
- every later override in the provenance chain.

Validation lists inherited defaults still active for each vehicle. Users therefore
do not need to invent every value, but absolute outputs cannot quietly appear more
certain than the inputs justify.

Obstacle-specific defaults such as a future `k_impact` remain deferred until the
corresponding equation and units are frozen.

## Correlation

A quantity may declare a `correlation_group`. Sampling semantics begin in the
uncertainty propagation phase, but the field exists now so the data contract does
not assume independence. Examples include engine-map points from one test session
or drag parameters derived from the same coast-down experiment.

## Study-level behavior

A study may later choose among:

- deterministic nominal evaluation;
- measured-track variability only;
- selected structural uncertainties;
- all declared uncertainty jointly.

A deterministic run does not delete uncertainty from the project. It uses nominal
values and records that sampling was disabled for that study.
