# All-declared structural sensitivity drop-in

Extract this archive directly over the repository root and approve file
replacement.

This upgrade:

- discovers every non-fixed registered input with structural uncertainty;
- retains explicit focused parameter lists for later follow-up runs;
- evaluates nominal plus declared numeric quantiles or categorical choices;
- reports progress and ETA per parameter level;
- checkpoints each level independently;
- reports absolute lap time, speed, completion, energy/loss mechanisms,
  ratio-region times, and bounded-versus-infinite diagnostics;
- writes `structural_sensitivity_report.html`.

## Existing Arizona project

Replace:

```text
projects/arizona/studies/structural_sensitivity.toml
```

with the included Arizona-ready root file (`vehicle_id = "mcmaster"`):

```text
structural_sensitivity_all_declared_arizona.toml
```

Copy it with:

```powershell
Copy-Item .\structural_sensitivity_all_declared_arizona.toml `
  .\projects\arizona\studies\structural_sensitivity.toml -Force
```

Then run:

```powershell
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"

pytest -q tests/test_structural_auto_discovery.py `
  tests/test_structural_absolute_summary.py `
  tests/test_progress_reporter_eta.py `
  tests/test_structural_html_report.py

drivetrain-study validate .\projects\arizona --study structural_sensitivity
drivetrain-study run structural-sensitivity .\projects\arizona --workers 4
```

The `--workers` value can be adjusted for the machine. The console prints
total parameter levels, elapsed time, ETA, throughput, the current parameter,
and cache usage.

Start with:

```text
results/structural_sensitivity/<run>/structural_sensitivity_report.html
```

Main machine-readable outputs:

```text
structural_metric_ranges.csv
structural_parameter_levels.csv
replicate_results.csv
summary.json
run_manifest.json
input_contracts.json
```
