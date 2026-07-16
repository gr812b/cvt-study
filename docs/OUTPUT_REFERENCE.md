# Output reference

Outputs are split into human interpretation, machine-readable evidence, plots,
and provenance. Markdown can be regenerated from machine artifacts without
rerunning simulation.

## Human-facing result files

| File | Audience and purpose |
| --- | --- |
| `SUMMARY.md` | one-page finding/recommendation, confidence, warnings, next actions |
| `REPORT.md` | technical narrative from decision through evidence and limitations |
| `decision_trace.md` | compact reasoning chain and evidence gates |
| `appendix/README.md` | links each report section to detailed artifacts |
| `provenance_graph.svg` | visual GPX → build → bundle → study → report lineage |
| `results/INDEX.md` | project-level list of completed result directories |

## Universal provenance and configuration

| File | Key content |
| --- | --- |
| `run_manifest.json` | study identity, counts, seed, cache/resume use, quality, bundle hashes |
| `provenance.json` | framework version, command, configuration/study fingerprints, evidence hash |
| `track_bundle.json` | exact Track Evidence Bundle used by simulation |
| `track_bundle.sha256` | adjacent bundle checksum |
| `resolved_inputs/resolved_inputs.toml` | fully merged SI-aware configuration |
| `resolved_inputs/resolution_manifest.json` | contributing project/profile layers |
| `resolved_inputs/validation_report.json` | validation diagnostics |
| `resolved_inputs/provenance.json` | configuration-resolution provenance |

JSON is strict: non-standard `NaN` and infinity tokens are not written.

## Baseline outputs

| File | Meaning |
| --- | --- |
| `bounded_trace.csv` | complete bounded-case report-step history |
| `infinite_reference_trace.csv` | matched reference history |
| `bounded_summary.json` | scalar bounded metrics and energy totals |
| `infinite_reference_summary.json` | scalar reference metrics |
| `comparison_summary.json` | time penalty, opportunity loss, dominance, closure |
| `gate_compliance.csv` | target and simulated speed at each accepted gate |
| `obstacle_energy_by_feature.csv` | entry speed and obstacle energy per feature |
| `resolved_simulation_case.json` | exact mechanism objects and numerical settings |
| `01_speed_comparison.png` | speed traces and gate ceilings |
| `02_ratio_trace.png` | bounded ratio occupancy and limits |
| `03_energy_accounting.png` | physical loss channels |

Important baseline summary fields:

- `lap_time_s`, `average_speed_kmh`, and maximum speed;
- completion and termination reason;
- time at minimum, variable, and maximum ratio;
- time braking and traction limited;
- engine, transmitted, and physical-loss energies;
- finite-ratio opportunity loss and engine operating shortfall;
- vehicle and powertrain residuals and relative errors;
- feature entry speeds and feature obstacle energy.

Opportunity loss and engine shortfall are counterfactual diagnostics. They are
not extra physical-loss rows and must not be summed into an energy balance.

## Study outputs

| File | Grain |
| --- | --- |
| `replicate_results.csv` | one row per scenario and design point |
| `scenario_draws.jsonl` | one exact sampled physical scenario per line |
| `summary.json` / `summary.csv` | distributions, rankings, regret, thresholds, bootstrap intervals |
| `convergence.json` | Monte Carlo error and split-half stability by design/metric |
| `input_contracts.json` | sampled and nominal input declarations |
| `energy_accounting.json` / `.csv` | physical energy bands by design and side |
| `physical_loss_shares.csv` | scenario-level physical shares |
| `feature_energy_results.csv` | scenario/design/feature obstacle energy |
| `uncertainty_attribution.json` / `.csv` | numeric and categorical screening indicators |
| `decision_summary.json` | machine-readable recommendation and warning synthesis |
| `scenario checkpoints` | exist only inside an unpublished `.incomplete` workspace |

### `replicate_results.csv` field groups

Identity fields include `replicate`, `design_id`, design path/value, scenario
identity, and completion. Comparison fields include bounded/reference lap time,
time penalty, opportunity loss, and reference dominance. Physical fields include
the two energy partitions, ratio occupancy, gate compliance, and feature energy.
Input columns retain the sampled values needed for attribution and replay.

The exact columns may grow additively. Consumers should select by name and must
not rely on column position.

### Scenario draws

Each JSONL record includes replicate/seed identity, sampled numeric and
categorical values, gate target speeds, the common run/lap/vehicle/driver
identity when paired gate sampling succeeds, and any independently sampled gate
fallback IDs. A fallback is evidence, not an invisible implementation detail.

### Statistical summary

For each design and metric, physical scenario statistics include p10, median,
p90, mean, and standard deviation. Bootstrap intervals quantify finite-sample
uncertainty for p10/median/p90 and paired ranking metrics. Threshold fractions
use binomial intervals.

Paired ranking fields distinguish:

- paired win fraction: fraction of common scenarios where the design wins;
- paired regret: within-scenario difference from the best tested design;
- probability-of-best estimate: empirical best fraction among tested designs;
- bootstrap bounds: uncertainty in those estimates from finite scenarios.

Ties and comparison direction are declared in the machine record rather than
inferred from plot labels.

### Convergence

`convergence.json` records sample count, Monte Carlo standard error where
defined, split-half median difference, and a status. Below twenty scenarios,
the report recommends more replicates even if the mechanism itself is valid.

### Energy accounting

Engine side:

```text
engine = wheel-transmitted + drivetrain loss + clutch loss + residual
```

Vehicle side:

```text
wheel + initial kinetic = final kinetic + tire slip + braking + rolling
                          + aerodynamic + obstacle + grade + residual
```

The JSON carries p10/median/p90 and bootstrap bounds by design/component. The
feature table reconciles to total obstacle work within numerical tolerance.

### Uncertainty attribution

Numeric rows may include response slope, slope bootstrap bounds, Pearson and
Spearman association, normalized elasticity, observed input spread,
uncertainty-weighted effect, and relative screening importance. Categorical rows
contain level counts, output by choice, span, and an eta-squared-style separation
measure.

Status values matter:

- fewer than 8 scenarios: suppressed;
- 8--19: exploratory;
- 20 or more: screening, subject to quality/convergence warnings.

## Track-build outputs

The exact filenames are indexed in the build report and include canonical GPX
telemetry, lap tables, centreline/profile tables, matched pass evidence,
feature/response-group tables, gate evidence with every confidence component,
review maps/plots, bundle JSON, and checksum. See
`output/TRACK_REVIEW_OUTPUTS.md` and `output/TRACK_BUNDLE_CONTRACT.md` for the
schema-level reference.

## Quality fields

`numerical_quality.numerically_valid` is true only when all declared numerical
checks pass. `evidence_assessment.ready` separately records whether project and
track-review blockers are absent. `decision_readiness` and
`decision_summary.json` separately expose statistical, directional, and final
decision readiness. A numerical pass alone is never a design recommendation.

## Result ownership

Completed outputs are immutable directories. A run is assembled in a hidden
sibling `.incomplete` directory, then atomically renamed after reports and
provenance succeed. Failed generation therefore does not publish a partial
completed result. `--restart` is the explicit way to replace an incomplete
workspace; completed directories are never silently overwritten.
