# User guide

This guide takes a project from raw GPX evidence to a reviewed engineering
result. The framework separates evidence preparation, mechanism simulation,
uncertainty studies, and decision reporting so each stage can be checked before
the next one is trusted.

## 1. Install and check the environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"
drivetrain-study doctor
```

`doctor PROJECT` also resolves a project and distinguishes errors from warnings.
A warning permits exploratory work but should be reviewed before a decision run.

## 2. Create a project

```powershell
drivetrain-study init .\projects\arizona --name "Arizona endurance"
```

The command creates configuration, track, vehicle, study, results, and GPX
directories. It does not invent event geometry or vehicle measurements.

Edit in this order:

1. `project.toml`: paths and profile roots.
2. `track/runs.toml`: each GPX file and its vehicle/driver/run identity.
3. `track/events.toml`: lap gate, features, response groups, and obstacle models.
4. `vehicles/<id>/vehicle.toml` and `drivetrain.toml`: physical inputs.
5. `studies/*.toml`: numerical settings, study question, samples, and thresholds.

Every physical quantity needs a nominal value, unit, source, and uncertainty.
See `INPUT_CONTRACT.md` for the full pattern.

## 3. Add and inspect GPX evidence

Copy GPX files into `track/gpx/`, then make each file explicit in `runs.toml`.
Do not merge files manually; run identity is needed for lap-paired sampling.

```powershell
drivetrain-study ingest .\projects\arizona
```

Inspect the ingestion diagnostics for:

- missing or non-monotonic timestamps;
- invalid coordinates;
- large time gaps or speed spikes;
- whether speed was reported or derived from position and time;
- retained elevation and segment boundaries.

The ingestion output is diagnostic. Simulation consumes the reviewed Track
Evidence Bundle created by the next stage.

## 4. Build and review track evidence

```powershell
drivetrain-study build-track .\projects\arizona
# `review` is an alias that emphasizes the review package.
drivetrain-study review .\projects\arizona
```

The build detects complete laps, constructs a common centreline, map-matches
each valid pass, projects features onto distance `s`, computes response windows,
scores gate evidence, and writes a versioned bundle.

Review before simulation:

- map alignment and centreline shape;
- lap inclusion/exclusion and lap-length distribution;
- each feature's start, anchor, and end;
- response-group membership;
- entry-speed distributions by run, lap, vehicle, and driver;
- all gate-confidence components, not only the total score;
- inherited obstacle priors and unresolved warnings;
- the bundle checksum and schema version.

An accepted gate is a one-way measured speed ceiling in the simulation. It does
not prescribe the whole speed trace. Rejected gates remain evidence but are not
enforced.

## 5. Run the nominal mechanism check

```powershell
drivetrain-study run baseline .\projects\arizona `
  --bundle .\projects\arizona\results\track_build\<run>\track_bundle.json
```

The bounded ideal CVT and infinite-ratio reference share the same vehicle,
track, gates, driver envelope, tire law, efficiency, and finite launch-torque
contract. Only the finite operating ratio bounds are removed in the reference.

Open `SUMMARY.md` first. Then inspect:

- `REPORT.md` for the full comparison and quality table;
- `decision_trace.md` for the reasoning chain;
- speed and ratio plots to locate the penalty;
- energy accounting and feature tables;
- gate compliance;
- `provenance.json` and the resolved inputs.

A baseline is one nominal mechanism comparison, not a robust design ranking.

## 6. Choose and run a study

### Measured-track robustness

```powershell
drivetrain-study run track-robustness PROJECT --workers 4
```

This samples empirical gate behavior and only inputs explicitly assigned the
`measured_track` role. Paired-lap sampling preserves the observed relationship
between gates on the same lap. Use it to ask whether a conclusion survives
plausible versions of the measured course.

### Structural sensitivity

```powershell
drivetrain-study run structural-sensitivity PROJECT
```

This holds everything else nominal and changes one declared structural input at
a time at the exact nominal and selected quantiles. Use it to learn direction,
local slope, elasticity, and which assumptions merit better measurement.

### Design sweep

```powershell
drivetrain-study run sweep PROJECT --workers 4
```

All designs receive identical sampled scenarios. The report includes output
bands, paired wins, paired regret, thresholds, convergence, and boundary-optimum
warnings. Extend the sweep if the apparent winner lies at an edge.

### Full uncertainty

```powershell
drivetrain-study run uncertainty PROJECT --workers 4
```

This jointly samples all declared stochastic measured-track, structural, and
initial-condition inputs. It estimates the output distribution under the full
uncertainty contract; it does not diagnose model-form errors that were never
declared.

For quick mechanism checks, `--replicates N` temporarily overrides the study
file. Reports retain the actual count and warn when it is inadequate.

## 7. Resume, restart, and cache

If a study stops, the unpublished workspace remains beside the final path:

```powershell
drivetrain-study run uncertainty PROJECT --resume
```

Resume is permitted only when the resolved study fingerprint matches. Use
`--restart` when deliberately discarding the incomplete result. A completed
result directory is never overwritten.

The content-addressed cache stores deterministic case summaries. Use
`--no-cache` for a forced recomputation, and inspect or clear it with the `cache`
commands. Cached and uncached execution should produce identical scientific
artifacts; only manifest execution counts differ.

## 8. Read a result from high to low level

1. `SUMMARY.md`: recommendation/finding, confidence, warnings, next actions.
2. `decision_trace.md`: explicit reasoning gates.
3. `REPORT.md`: study contract, quality, result bands, energy, attribution,
   convergence, provenance, and limits.
4. Plots: locate relationships and distributions visually.
5. `appendix/README.md`: map each claim to CSV/JSON/JSONL evidence.
6. `replicate_results.csv` and `scenario_draws.jsonl`: audit individual cases.
7. traces (baseline): inspect solver history.

Regenerate Markdown without rerunning physics:

```powershell
drivetrain-study report PATH_TO_RESULT
```

List completed project results and rebuild the project index:

```powershell
drivetrain-study results PROJECT
```

## 9. Interpret confidence correctly

- `valid_for_decision` in the numerical-quality section means the simulated
  cases passed completion, dominance, gate, and energy checks.
- `directionally_robust` is stronger: headline metrics agree on a winner, paired
  evidence clears its threshold, convergence is adequate, and quality passes.
- Physical p10--p90 describes declared scenario variation.
- A bootstrap interval describes uncertainty in an estimated statistic because
  only finitely many scenarios were run.
- Screening attribution describes marginal association and sensitivity. It is
  not automatically causal, especially with correlated inputs.

## 10. Common failure modes

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Project validation error | missing field, unit, source, role, or invalid path | Read the exact diagnostic and `INPUT_CONTRACT.md` |
| No valid laps | lap gate, timing limits, or map quality do not fit the data | Inspect ingestion and reconstruction artifacts |
| Gate rejected | too few passes, weak braking evidence, poor repeatability, or map error | Review components; do not lower thresholds only to force acceptance |
| Energy closure fails | integration step too coarse or an invalid mechanism input | Reduce the step and inspect traces/limits |
| Reference dominance fails | numerical issue or mismatched comparison contract | Treat result as invalid and inspect case resolution |
| Attribution suppressed | fewer than eight scenarios | Run more paired scenarios |
| Optimum on boundary | design domain too narrow | Extend the tested design range |
| Incomplete workspace exists | prior interrupted run | Use a matching `--resume` or intentional `--restart` |

## 11. What is not yet modeled

GPX altitude does not create grade force, lateral/yaw dynamics are not active,
obstacles are uncalibrated unless the project supplies evidence, the tire model
is reduced-order, and the current CVT is idealized. These limits are repeated in
every technical report because they constrain the decision.
