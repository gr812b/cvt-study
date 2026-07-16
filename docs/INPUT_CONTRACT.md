# Input contract reference

Configuration is TOML. Paths below use dot notation after profile resolution.
An inherited value and a local value have the same resolved contract; the
resolved-input export records which layer supplied it.

## Quantity envelope

Every physical scalar uses:

```toml
[vehicle.mass]
nominal = 245.0
unit = "kg"

[vehicle.mass.source]
kind = "measured"
reference = "four-corner scale record, 2026-04-12"
notes = "race-ready wet mass"

[vehicle.mass.uncertainty]
distribution = "normal"
role = "structural"
confidence_half_width = 0.5
confidence_level = 0.95
```

Required concepts:

| Field | Meaning |
| --- | --- |
| `nominal` | deterministic baseline and centre of the engineering declaration |
| `unit` | supported unit converted to SI during resolution |
| `source.kind` | `measured`, `derived`, `engineering_estimate`, or `inherited_default` |
| `source.reference` | enough detail to locate or reproduce the evidence |
| `source.notes` | optional qualification |
| `uncertainty.distribution` | sampling contract, including explicit `fixed` |
| `uncertainty.role` | semantic role for stochastic values |

Supported numeric distributions are `fixed`, `normal`, `truncated_normal`,
`uniform`, `triangular`, and `empirical`. Categorical inputs use fixed or
discrete choices. Use the parameters appropriate to the distribution:

| Distribution | Parameters |
| --- | --- |
| `fixed` | `reason` |
| `normal` | standard deviation, or confidence half-width plus confidence level |
| `truncated_normal` | normal scale plus lower/upper support |
| `uniform` | `lower`, `upper` |
| `triangular` | `lower`, `mode`, `upper` |
| `empirical` | declared samples |
| discrete choice | choices and probabilities |

Stochastic roles are:

- `measured_track`: real lap/event variability in the measured course;
- `structural`: uncertainty in physics, vehicle properties, or model form;
- `initial_condition`: uncertainty in the declared initial state.

Physical support is validated before simulation. Invalid mass, efficiency,
ratio ordering, probability, or distribution support is rejected rather than
silently clipped or redrawn.

## `project.toml`

| Path | Required | Contract |
| --- | --- | --- |
| `project.name` | yes | stable human-readable project name |
| `project.schema_version` | yes | project schema integer supported by this package |
| `project.track` | yes | path to track TOML, relative to project root |
| `project.runs` | yes | path to run manifest |
| `project.events` | yes | path to event manifest |
| `project.vehicles_directory` | yes | directory of vehicle subdirectories |
| `project.studies_directory` | yes | directory of study TOML files |
| `project.results_directory` | yes | output root |
| `profiles.roots` | no | additional profile search roots |

Paths may not escape the intended project contract accidentally. Local values
override inherited profile leaves; a CLI `--set PATH=VALUE` override applies
only to an existing resolved leaf and is exported in provenance.

## Track fields

| Path | Unit/type | Meaning |
| --- | --- | --- |
| `track.name` | text | report label |
| `track.profiles` | list | inherited track profiles |
| `track.closed_course` | bool | enables closed-lap reconstruction |
| `track.surface_class` | text | descriptive surface class |
| `track.surface.friction_coefficient` | quantity, `1` | longitudinal friction prior |
| `track.elevation.store_from_gpx` | bool | retain GPX/FIT elevation evidence (legacy field name) |
| `track.elevation.use_for_grade_force` | bool | currently must remain false |
| `track.reconstruction.lap_gate_event_id` | id | event defining `s=0` and lap crossing |
| `lap_gate_radius_m` | m | geographic crossing region |
| `minimum_lap_time_s` | s | rejects implausibly short crossings |
| `maximum_reasonable_speed_mps` | m/s | ingestion/reconstruction quality limit |
| `maximum_normal_time_step_s` | s | gap diagnostic threshold |
| `stationary_speed_mps` | m/s | moving-coverage threshold |
| `minimum_speed_coverage_fraction` | 0--1 | required valid moving coverage |
| `centreline_spacing_m` | m | common centreline station spacing |
| `profile_spacing_m` | m | output/resampling spacing |
| `maximum_map_error_m` | m | pass-to-centreline validity limit |
| `speed_spike_threshold_mps` | m/s | local speed discontinuity diagnostic |

Event-window fields are distances relative to a feature boundary:
`approach_before_m`, `approach_gap_m`, `entry_before_m`, `entry_gap_m`,
`exit_gap_m`, `exit_length_m`, and `recovery_limit_m`.

Gate-confidence fields are `minimum_valid_passes`, `target_pass_count`,
`braking_threshold_mps`, `repeatability_scale_mps`,
`vehicle_agreement_scale_mps`, `accept_score`, `review_score`, and the six
weights for pass count, speed repeatability, braking evidence, pace
independence, coordinate quality, and cross-vehicle agreement. Weights are
validated as a transparent evidence score, not a probability model.

## Run manifest (`runs.toml`)

Each `[[runs]]` record contains:

| Field | Meaning |
| --- | --- |
| `file` | GPX or FIT path relative to the track file/project |
| `vehicle_id` | vehicle that created the recording |
| `run_id` | unique recording/session identity |
| `driver_id` | driver identity used for evidence grouping |
| `use_for_centreline` | permit geometry contribution |
| `use_for_gate_evidence` | permit speed/gate contribution |

Do not reuse a `run_id` for distinct recordings. Lap identities produced from
these records are the unit used for paired empirical gate draws. If GPX and FIT
are exports of one device session, declare the FIT version once rather than
counting the two files as independent evidence.

## Events and features (`events.toml`)

Each `[[events]]` record contains:

| Field | Meaning |
| --- | --- |
| `id` | stable identifier |
| `name` | human label |
| `sequence` | review order around the course |
| `kind` | `point` or `interval` |
| `analysis_role` | `lap_gate` or physical `feature` |
| `response_group_id` | groups physical geometry with one behavioral response |
| `gate_candidate` | whether entry-speed evidence should be scored |
| `obstacle_profiles` | explicit obstacle model/profile list |
| `notes` | optional engineering context |
| `anchor.latitude_deg`, `longitude_deg` | geographic anchor |
| `anchor.horizontal_uncertainty_m` | location uncertainty |
| `anchor.source` | anchor evidence |
| `extent.before_anchor_m`, `after_anchor_m` | feature geometry around anchor |
| `extent.*_uncertainty_m` | extent uncertainty |
| `extent.source` | geometry evidence |

One response group may contain multiple physical features while producing one
measured entry behavior. Conversely, two distinct behavioral responses must not
be collapsed merely because the obstacles are nearby.

Obstacle mechanisms:

| `model_type` | Required parameters | Interpretation |
| --- | --- | --- |
| `none` | none | geometry/gate only, no obstacle loss |
| `roughness_energy_density` | `specific_energy_per_distance` in J/(kg m) | distributed dissipative work over interval |
| `speed_quadratic_energy` | `specific_fixed_energy` in J/kg, `impact_coefficient` in kg | localized fixed plus speed-squared work |
| `smooth_profile` | vertical amplitude, loss coefficients, traction and load scales | explicit smooth geometry/normal-load approximation |

Broad obstacle defaults are structural unless a project explicitly declares an
observed event-to-event quantity as `measured_track`.

## Vehicle and drivetrain fields

| Path | SI dimension | Constraint/meaning |
| --- | --- | --- |
| `vehicle.id` | id | matches study and run vehicle IDs |
| `vehicle.profiles` | list | inherited vehicle profiles |
| `vehicle.mass` | kg | positive total modeled mass |
| `vehicle.gravity` | m/s² | normally fixed conventional gravity |
| `vehicle.tire_diameter` | length | positive loaded diameter |
| `vehicle.wheel_rotational_inertia` | kg m² | nonnegative driven rotational inertia |
| `vehicle.driven_normal_load_fraction` | 1 | strictly bounded physical fraction |
| `vehicle.aero.drag_area` | m² | nonnegative `CdA`-style area |
| `vehicle.aero.air_density` | kg/m³ | positive ambient density |
| `vehicle.rolling_resistance_coefficient` | 1 | nonnegative coefficient |
| `vehicle.tire.peak_traction_scale` | 1 | force-ceiling scale |
| `vehicle.tire.slip_stiffness` | N s/m | positive slip buildup scale |
| `drivetrain.efficiency` | 1 | greater than zero and no more than one |
| `drivetrain.final_drive_ratio` | 1 | positive engine-to-wheel reduction |
| `drivetrain.engine.model` | choice | currently `baja_br10_reference_v1` |
| `drivetrain.engine.target_speed` | angular speed | ideal-CVT target |
| `drivetrain.engine.power_scale` | 1 | positive curve scale |
| `drivetrain.cvt.maximum_reduction_ratio` | 1 | launch/high-reduction bound |
| `drivetrain.cvt.minimum_reduction_ratio` | 1 | overdrive/low-reduction bound |
| `drivetrain.cvt.launch_clutch_model` | choice | `ideal_slip` or `disabled` |

The minimum reduction ratio must not exceed the maximum reduction ratio.

## Baseline study fields

| Path | Meaning |
| --- | --- |
| `study.name`, `type`, `vehicle_id`, `random_seed` | run identity; type is `baseline` |
| `sampling.mode` | `nominal` |
| `driver.maximum_braking_deceleration` | braking envelope acceleration |
| `driver.maximum_brake_force` | force cap |
| `driver.braking_trigger_margin` | anti-chatter speed margin |
| `simulation.maximum_time_s` | hard termination limit |
| `simulation.integration_step_s` | solver step |
| `simulation.report_step_s` | trace output spacing |
| `track_realization.gate_speed_statistic` | nominal gate statistic, normally median |
| `initial_conditions.vehicle_speed` | launch state |
| `initial_conditions.wheel_speed` | driven-wheel launch state |
| `quality.maximum_abs_energy_balance_relative_error` | vehicle closure tolerance |
| `quality.maximum_abs_powertrain_balance_relative_error` | engine/wheel closure tolerance |

## Study fields

All non-baseline studies use `base_case.study` to select the nominal mechanism.

Design sweep adds `design_variable.path` and `design_variable.values`, optional
decision thresholds, and measured-track or other declared sampling.

Track robustness uses `sampling.mode = "measured_track"`. Full uncertainty uses
`all_declared`. Both declare `replicates`, `paired_scenarios = true`, and
`gate_sampling = "paired_lap"`.

Structural sensitivity uses `sensitivity.method = "one_at_a_time"`, a list of
registered parameter paths, and quantiles. The exact nominal is always included
even when it differs from a distribution median.

`reporting.bootstrap_resamples` controls estimation intervals. Low values speed
smoke checks but should not be used for a final report.

Correlation groups, when declared, must reference known stochastic inputs with
consistent semantic roles. Matrices must be symmetric, have unit diagonal, be
positive semidefinite, and not overlap. Gaussian-copula correlation is latent
dependence; final Pearson correlation need not equal the matrix entry for
nonnormal marginals.
