# Multivariable design grid — direct root overlay

This ZIP is intentionally laid out as a repository-root overlay.

Extract/copy its contents directly into the root of `cvt-study`, allowing files
under `src/`, `projects/`, `docs/`, and `tests/` to overwrite or merge with the
matching folders.

No installer or patch script is required.

The included Arizona study keeps the legacy `[design_variable]` table only for
compatibility with the older project validator. The runtime grid uses the two
`[[design_variables]]` tables and evaluates their full Cartesian product.

After copying, run:

```powershell
drivetrain-study validate .\projects\arizona `
  --study cvt_range_final_drive_sweep
```

Then:

```powershell
drivetrain-study run design-comparison .\projects\arizona `
  --study cvt_range_final_drive_sweep `
  --workers 6 `
  --restart `
  --run-name arizona-cvt-range-final-drive
```

The study uses the saved full-uncertainty result through
`[uncertainty_informed_design]`, so each CVT-maximum/final-drive combination is
run against the same selected source worlds.
