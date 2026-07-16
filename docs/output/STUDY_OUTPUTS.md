# Phase 6 study outputs

Each study writes an atomic, self-contained result directory:

```text
results/<study_type>/<timestamp>/
├── track_bundle.json
├── track_bundle.sha256
├── resolved_inputs/
├── input_contracts.json
├── scenario_draws.jsonl
├── replicate_results.csv
├── summary.json
├── summary.csv
├── convergence.json
├── run_manifest.json
├── REPORT.md
└── *.png
```

## `scenario_draws.jsonl`

One line per scenario. It records the deterministic scenario seed, sampled SI
quantities, categorical choices, gate targets, paired lap identity, and any gates
that required independent fallback sampling.

## `replicate_results.csv`

One row per scenario and design point. It contains bounded/reference lap times,
opportunity loss, energy residuals, gate compliance, reference fingerprint, and
design metadata. This is the primary audit and re-analysis table.

## `summary.json`

For stochastic studies, each design contains mean, standard deviation, p10,
median, p90, and bootstrap 95% intervals for each primary output. Design sweeps
also include paired win fractions and paired-regret statistics, with bootstrap
95% estimation intervals formed by resampling whole paired scenarios.

For structural sensitivity, each parameter instead contains the exact nominal
case, quantile cases, signed changes from nominal, and total output span.

## `convergence.json`

Distinguishes a quick run from a sufficiently stable Monte Carlo study. A null
standard error for a single sample is intentional and valid JSON; no output file
contains NaN or Infinity.

## `input_contracts.json`

A frozen record of every quantity and categorical uncertainty contract available
to the study, including units, provenance, distribution, semantic uncertainty role,
correlation group, and obstacle-model branch.

## `run_manifest.json`

Records the study type, seed, scenario count, simulation counts, cache policy,
sampled scalar/categorical paths, stochastic paths grouped by semantic role,
sampled gate IDs, gate-pairing status, bundle hashes, numerical quality, and
declared uncertainties that were not propagated.
