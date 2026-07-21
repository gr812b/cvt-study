# Full-uncertainty v2 incremental drop-in

This archive is an incremental overlay for the repository after applying:

1. the six-report refactor;
2. the track-robustness report v2 drop-in; and
3. the structural-sensitivity report v2 drop-in.

Extract it directly over the repository root and approve replacement.

## Scientific change to future runs

The recommended full-uncertainty configuration now uses:

```toml
[sampling]
mode = "all_declared"
replicates = 30
layout = "cross_track_cases"
paired_scenarios = true
gate_sampling = "paired_lap"
```

Under `cross_track_cases`, `replicates` is the number of **common structural and
measured-traversal draws per track interpretation**, not the total number of
joint scenarios. With 14 admitted track cases, 30 draws produce 420 joint
scenarios.

For every base draw, the runner replays the same:

- structural quantities and categorical choices;
- measured run/lap/vehicle/driver identity;
- design candidates;

on every admitted track reconstruction. This makes track-case differences paired
and directly interpretable. It also writes `base_draw_id`, `track_pair_id`, and
track-case metadata into the scenario and result artifacts.

The legacy round-robin layout remains readable and runnable as:

```toml
layout = "round_robin_track_cases"
```

but it cannot isolate track-case effects because each case receives different
random draws.

A crossed run requires at least one measured lap identity with evidence at every
active gate across the admitted track ensemble. The runner stops with a clear
error rather than silently sampling unmatched traversals independently.

## Exact nominal reference

Future full-uncertainty runs also execute and save one exact nominal bounded and
infinite-reference comparison in:

```text
nominal_reference.json
```

This provides an explicit nominal marker in the report without estimating it
from the joint scenario distribution.

## Report changes

`full_uncertainty_report.html` now begins with:

- a study-adequacy panel;
- a complete inventory of every varied structural parameter and declared range;
- every measured gate-related target and observed range;
- every admitted epistemic track interpretation.

The report then separates:

- absolute vehicle performance;
- finite-CVT-range limitation;
- uncertainty families and every individual driver;
- paired or unpaired track-case effects;
- physical losses and compounded input families;
- convergence of medians and tails;
- representative scenario cases;
- detailed searchable appendices.

All driver inputs are preserved in searchable, sortable tables. Top-driver plots
remain concise, and companion `*_all.png` plots include every input. Table headers
cycle ascending, descending, and original order. Identity columns remain sticky
while horizontally scrolling.

The report explicitly distinguishes unweighted epistemic track alternatives from
calibrated probabilities. A pooled percentile is labelled a joint-scenario
percentile, not a real-world credible interval.

## Regenerate an existing result without simulations

For the previously completed result:

```powershell
drivetrain-study report `
  .\projects\arizona\results\full_uncertainty\full-uncertainty--69f39476db
```

This reads the existing CSV and JSON artifacts, regenerates all report plots and
tables, and does not run a vehicle simulation.

The old result will correctly remain labelled as a legacy unpaired track-case
layout. Reporting can clarify that limitation, but only a new crossed run can
produce isolated paired track-case effects.

## Configure the next Arizona run

```powershell
Copy-Item .\project_updates\arizona\studies\full_uncertainty.toml `
  .\projects\arizona\studies\full_uncertainty.toml -Force

.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"

drivetrain-study validate .\projects\arizona --study full_uncertainty

drivetrain-study run full-uncertainty `
  .\projects\arizona `
  --workers 6
```

With 14 track cases and 30 configured draws, expect 420 joint scenarios. The
console reports the number of common draws and total crossed scenarios before
execution.

## Focused checks

```powershell
pytest -q `
  tests/test_full_uncertainty_report_v2.py `
  tests/test_full_uncertainty_crossed_schedule.py
```

The complete repository test suite should also be run before merging.

## New machine-readable artifacts

```text
full_uncertainty_parameter_inventory.csv
full_uncertainty_gate_target_inventory.csv
full_uncertainty_track_case_inventory.csv
full_uncertainty_driver_explorer.csv
full_uncertainty_family_screening.csv
full_uncertainty_derived_inputs.csv
full_uncertainty_convergence.csv
full_uncertainty_adequacy.csv
full_uncertainty_track_case_summary.csv
full_uncertainty_paired_track_effects.csv
full_uncertainty_scenario_explorer.csv
```
