# Broad final-drive sweep + optimism sensitivity

This overlay keeps the existing paired design/traffic reporting and adds one new model-form stress test.

## Arizona design grid

`projects/arizona/studies/cvt_range_final_drive_sweep.toml` now uses:

- maximum CVT reduction: `2.5, 3.0, 3.5, 4.0, 4.5`
- final drive: `4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0`

That is 35 Cartesian drivetrain designs. The CVT minimum-reduction ratio remains fixed by the vehicle configuration.

Run the broad design sweep normally:

```powershell
drivetrain-study run design-comparison .\projects\arizona `
  --study cvt_range_final_drive_sweep `
  --workers 8 `
  --restart
```

## Continuous-pace optimism sensitivity

After the design sweep completes, run:

```powershell
drivetrain-study optimism-sensitivity <DESIGN_RESULT_DIRECTORY> `
  --workers 8 `
  --restart
```

The project path is inferred from the design result provenance. If the result was moved to another machine/location, pass it explicitly:

```powershell
drivetrain-study optimism-sensitivity <DESIGN_RESULT_DIRECTORY> `
  --project .\projects\arizona `
  --workers 8 `
  --restart
```

The sensitivity uses the same paired base uncertainty worlds on the nominal track reconstruction and reruns every design under four assumptions at the same numerical step:

1. `control`: no added continuous pace ceiling
2. `loose`: valid-lap per-vehicle p95 upper pace, cyclically smoothed, then `1.20 x + 2.0 m/s`
3. `moderate`: `1.10 x + 1.0 m/s`
4. `strong`: `1.00 x + 0.5 m/s`

Each source vehicle first gets its own p95 spatial pace profile. The positionwise faster source is retained before smoothing, so the larger Cornell lap count does not vote down McMaster or vice versa. This is intentionally an *upper envelope* stress test, not a fitted target-speed profile.

The default sensitivity integration step is 5 ms. It reruns a no-envelope control at that same 5 ms and compares that control back to the original design sweep, so any design-dependent numerical bias is visible. You can change it with `--integration-step-ms`, e.g. `--integration-step-ms 10` for a quicker screen.

### Report outputs

The new `optimism_sensitivity/optimism_sensitivity_report.html` contains:

- source-derived pace-envelope profiles;
- control/loose/moderate/strong paired-regret design heatmaps;
- absolute lap-time increase by design to show whether optimism correction is common-mode;
- preferred design by optimism level;
- numerical-fidelity comparison against the original design result;
- a speed-occupancy sanity section.

The speed histogram selector is **by drivetrain design candidate**. For the selected design, the plot overlays:

- valid-lap Cornell telemetry;
- valid-lap McMaster telemetry;
- normal/control simulation;
- loose optimism correction;
- moderate optimism correction;
- strong optimism correction.

All simulated histogram traces use the same representative median-traffic paired world. Traffic is not the dropdown dimension.
