# Uncertainty-informed design screening

This drop-in adds a reduced replay mode for design sweeps.

The full-uncertainty study remains responsible for exploring the admitted input
space. After that result is reviewed, the design sweep reads its saved
`scenario_draws.jsonl`, `replicate_results.csv`, and track-bundle snapshots. It
selects actual source worlds rather than creating averaged or synthetic worlds.

The selection keeps:

- a central representative draw;
- important high/low source-output cases;
- additional farthest-point draws covering the remaining joint input/output space;
- the nominal track plus the track cases with the largest paired effect in the
  source uncertainty result.

Every candidate design is run against the exact same selected worlds. The design
variable is removed from the source draws and replaced by each configured design
point. The implementation is generic for numeric design variables; the supplied
Arizona configuration uses `drivetrain.final_drive_ratio`.

## Intended workflow

```powershell
drivetrain-study build-track .\projects\arizona

drivetrain-study run track-robustness .\projects\arizona `
  --workers 6 `
  --restart `
  --run-name arizona-track-approved

drivetrain-study run full-uncertainty .\projects\arizona `
  --workers 6 `
  --restart `
  --run-name arizona-uncertainty

# After reviewing the uncertainty report:
drivetrain-study run design-comparison .\projects\arizona `
  --study final_drive_sweep `
  --workers 6 `
  --restart `
  --run-name arizona-final-drive-screen
```

In replay mode, `--replicates N` means “select N source base draws”; it does not
generate N new random samples. For example, this forces a 16-draw screen:

```powershell
drivetrain-study run design-comparison .\projects\arizona `
  --study final_drive_sweep `
  --replicates 16 `
  --workers 6 `
  --restart
```

## Produced audit files

The design result includes:

- `uncertainty_informed_design_manifest.json`;
- `selected_uncertainty_scenarios.csv`;
- `selected_uncertainty_track_cases.csv`;
- `SCENARIO_REDUCTION.md`;
- the standard design-comparison report and machine artifacts.

The reduced screen is designed to eliminate poor candidates cheaply. When two or
three candidates remain close, increase `maximum_base_draws` or use
`--replicates` for a larger verification subset.
