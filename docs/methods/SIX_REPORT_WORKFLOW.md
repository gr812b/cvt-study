# Six-report workflow

The framework has six major reports. Each report has one question and should not
silently absorb the purpose of another report.

## 1. Track evidence and reconstruction

**Command**

```powershell
drivetrain-study build-track .\projects\arizona
```

**Question:** What track can be inferred from the supplied telemetry and reviewed
events?

Primary output: `review/track_evidence_report.html`.

This is the nominal evidence package: cleaned telemetry, excluded points and
laps, consensus centreline, event projection, along-track timeline, gate evidence,
confidence components, and the selected track bundle.

## 2. Nominal vehicle simulation

```powershell
drivetrain-study run nominal .\projects\arizona --bundle <track_bundle.json>
```

**Question:** What does one fixed vehicle and drivetrain do on the nominal track?

Primary output: `nominal_simulation_report.html`.

This report emphasizes absolute performance, force/energy mechanisms, ratio-bound
occupancy, obstacle behavior, gate compliance, and the bounded/infinite comparison.

## 3. Track defensibility and robustness

```powershell
drivetrain-study run track-robustness .\projects\arizona
```

**Question:** Is the inferred track stable under reasonable alternative analysis
choices supported by the same telemetry?

Primary output: `track_robustness_report.html`.

This report never runs a vehicle model. It reconstructs the track under:

- leave-one-run/vehicle/driver-out support checks;
- strict and permissive gate policies;
- alternative confidence weights;
- narrow and wide event windows;
- centreline smoothing, spacing, and outlier choices;
- conservative and permissive isolated-point cleanup.

It reports centreline displacement, length movement, event projection movement,
gate qualification frequency, target-speed movement, failed cases, and exact
case settings.

## 4. Structural sensitivity

```powershell
drivetrain-study run structural-sensitivity .\projects\arizona --bundle <track_bundle.json>
```

**Question:** Which physical or modelling assumptions materially move the nominal
answer?

Primary output: `structural_sensitivity_report.html`.

Each structural input is changed one at a time. Results are response spans and
mechanism changes, not fake stochastic error bars around deterministic runs.

## 5. Full uncertainty and answer robustness

```powershell
drivetrain-study run full-uncertainty .\projects\arizona --bundle <track_bundle.json>
```

**Question:** When defensible uncertainties vary together, what range of answers
should be believed?

Primary output: `full_uncertainty_report.html`.

The report begins with completion and numerical health, then absolute lap-time
and energy distributions, paired bounded/infinite differences, ratio occupancy,
physical loss distributions, attribution, scenario explorer, and convergence.

By default it consumes the latest successful `track_robustness` ensemble as an
outer set of **unweighted epistemic scenarios**. These alternatives are not
pretended to be probability-calibrated random draws. Each scenario records its
`track_case_id`, while structural inputs and coherent measured-lap gate speeds
are sampled inside that track interpretation.

## 6. Design comparison

```powershell
drivetrain-study run design-comparison .\projects\arizona --bundle <track_bundle.json>
```

**Question:** Which design performs best, by how much, and does the ranking survive
uncertainty?

Primary output: `design_comparison_report.html`.

All candidates share paired scenario draws, including the same selected track
interpretation. The ranking includes completion, absolute lap time, paired
infinite-reference penalty, energy opportunity, and ratio-bound occupancy.

## Legacy aliases

These remain valid:

- `run baseline` → `run nominal`
- `run uncertainty` → `run full-uncertainty`
- `run sweep` → `run design-comparison`

The old vehicle-simulation behavior previously called `track_robustness` is no
longer used. Measured traversal variation belongs inside full uncertainty.
