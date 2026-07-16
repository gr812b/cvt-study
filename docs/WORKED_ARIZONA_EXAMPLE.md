# Worked Arizona measured-track example

The included `examples/arizona_endurance_project` demonstrates the complete
evidence path on measured GPX. It is a method example, not a universal gearing
recommendation.

## Run the evidence pipeline

```powershell
drivetrain-study doctor .\examples\arizona_endurance_project
drivetrain-study validate .\examples\arizona_endurance_project
drivetrain-study ingest .\examples\arizona_endurance_project
drivetrain-study build-track .\examples\arizona_endurance_project
```

The GPX contains timestamps and elevation but no direct speed field. The parser
therefore derives speed from position and time and records that provenance.
Elevation is retained for review but does not create grade force.

Before using the bundle, inspect complete/valid laps, map error, feature
projection, response groups, gate pass counts, and inherited obstacle priors.
The project notes identify which values are example assumptions rather than team
measurements.

## Nominal comparison

```powershell
drivetrain-study run baseline .\examples\arizona_endurance_project `
  --bundle PATH_TO_REVIEWED_TRACK_BUNDLE
```

Read `SUMMARY.md`, then verify the speed/ratio traces, accepted-gate compliance,
reference dominance, and both energy residuals. The infinite result is an
opportunity bound: it estimates what finite operating ratio range prevented
under the same launch and track constraints.

## Short integration checks

During development it is reasonable to use:

```powershell
drivetrain-study run track-robustness PROJECT --replicates 2 --workers 2
```

This validates sampling, pairing, execution, reporting, and provenance. It does
not validate a distributional conclusion. The generated summary should say so.

## Decision studies

Run the declared final-drive sweep with enough paired scenarios to stabilize
win/regret intervals. If the winner is on a boundary, expand the range before
selecting hardware. Run structural sensitivity to identify assumptions worth
measuring. Run track robustness to separate measured course variation from
model uncertainty. Finally, use full uncertainty to communicate admitted total
spread.

## Evidence to replace for a team decision

- vehicle mass and loaded tire diameter;
- engine curve/power scale and CVT ratio range;
- drivetrain efficiency;
- surface traction and tire-slip behavior;
- obstacle geometry and loss calibration;
- run/driver identities and gate review decisions;
- study thresholds and design domain.

Replacing a broad default narrows uncertainty only if the new evidence supports
that narrower declaration. Preserve the measurement reference and method in the
quantity source block.
