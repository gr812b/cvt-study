# Structural-sensitivity report v2 drop-in

This is an incremental reporting upgrade intended to be extracted over the
six-report refactor and the track-robustness-report-v2 drop-in.

It does **not** change structural study execution, sampling, simulation physics,
checkpoints, or saved machine artifacts. It replaces only structural reporting
and the completed-result report-regeneration route.

## What changes

The structural HTML now provides:

- a nominal-result strip before any sensitivity ranking;
- major plots before detailed tables;
- the exact tested input range on every tornado-plot label;
- all prior headline tornado plots and the mechanism heatmap;
- correct `Maximum speed` and `Engine energy` names while retaining legacy plot
  filenames as compatibility aliases;
- a direct comparison of drivers of absolute lap time versus drivers of the
  bounded-versus-infinite finite-ratio penalty;
- response curves for the leading six lap-time parameters using every completed
  evaluated level;
- a ratio-occupancy mechanism plot;
- the existing physical-energy attribution plot when present;
- an explicit warning when nominal drivetrain efficiency is 1.0 but the tested
  range extends substantially lower;
- measurement-priority and potentially compounding uncertainty-family tables;
- sortable tables whose headers cycle ascending, descending, and original order;
- sticky parameter/evaluated-level identity columns during horizontal scrolling;
- full supporting tables in collapsible appendices.

The older `structural_sensitivity.png` is retained in a legacy-diagnostic
section when it exists. Existing headline plots remain available, including the
old misspelled/short filenames as aliases so external links do not break.

## Apply

Extract this archive directly over the repository root and approve replacement.

Then reinstall and run the focused checks:

```powershell
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"

pytest -q `
  tests/test_structural_report_v2.py `
  tests/test_structural_report_postprocess.py
```

## Regenerate an existing completed run

No simulation is repeated. The renderer reads:

- `replicate_results.csv`
- `summary.json`
- `run_manifest.json`
- `input_contracts.json`

Run:

```powershell
drivetrain-study report `
  .\projects\arizona\results\structural_sensitivity\structural-sensitivity--0f47c968d4
```

The main result is rewritten at:

```text
structural_sensitivity_report.html
```

New machine-readable reporting artifacts include:

```text
structural_input_ranges.csv
structural_measurement_priorities.csv
structural_uncertainty_families.csv
structural_nominal_summary.csv
structural_response_curves_manifest.json
```

New plots include:

```text
structural_absolute_vs_ratio_drivers.png
structural_ratio_occupancy_sensitivity.png
structural_response_*.png
```

## Scientific interpretation

This remains deterministic one-at-a-time structural screening. Plot sizes are
responses across the declared input ranges, not intrinsic parameter constants,
probabilities, or Monte Carlo confidence intervals. Joint effects and
interactions remain the responsibility of the full-uncertainty study.
