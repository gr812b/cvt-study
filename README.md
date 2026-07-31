# CVT-study multi-variable design-grid drop-in

This drop-in adds Cartesian `[[design_variables]]` support while preserving all
existing one-variable studies.

It includes:

- simultaneous numeric design overrides;
- exclusion of every design path from uncertainty sampling/replay;
- safe shared-reference policy for final-drive and CVT-ratio grids;
- validation of candidate count and CVT max/min ordering;
- two-dimensional report heatmaps;
- black, visible uncertainty bars on bar charts;
- an Arizona maximum-CVT-ratio × final-drive example.

## Apply

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\cvt-study-multivariable-design-grid-dropin\apply_multivariable_design_grid.ps1" `
  -RepoRoot "."
```

## Run

```powershell
drivetrain-study validate .\projects\arizona --study cvt_range_final_drive_sweep

drivetrain-study run design-comparison .\projects\arizona `
  --study cvt_range_final_drive_sweep `
  --workers 6 `
  --restart `
  --run-name arizona-cvt-range-final-drive
```

The supplied grid has 5 × 6 = 30 candidates. With 12 source draws and 5 selected
track cases, it runs at most 1,800 bounded simulations plus 60 shared references.
