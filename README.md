# Measured Track-Based Drivetrain Design Framework

Research software for turning repeated closed-course GPX recordings into a
reviewed **Track Evidence Bundle**, then using that evidence to compare
drivetrain designs under explicit physical and statistical assumptions.

The current mechanism compares a bounded ideal CVT with an otherwise-identical
infinite-ratio opportunity reference. This repository owns the measured-track
evidence, reduced-order simulation, paired-study, and decision-reporting workflow.
Detailed physical drivetrain development is outside this project's scope.

This is a working Phase 8 checkpoint (`0.8.0.dev0`). Start with the one-page
`SUMMARY.md` produced by each run, then expand into `REPORT.md`,
`decision_trace.md`, the plots, and the machine-readable appendix.

## Install

Python 3.10 or newer is required.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"
drivetrain-study doctor
```

On macOS or Linux, activate with `source .venv/bin/activate`.

`cvt-study` remains an alias for compatibility, but new examples use
`drivetrain-study`.

## Two-minute workflow

```powershell
drivetrain-study init .\projects\my_track --name "My endurance track"

# Put GPX files in projects/my_track/track/gpx, then edit the TOML contracts.
drivetrain-study validate .\projects\my_track
drivetrain-study build-track .\projects\my_track

# Review the map, gates, features, and warnings before simulation.
drivetrain-study run baseline .\projects\my_track
drivetrain-study run sweep .\projects\my_track --workers 4
drivetrain-study results .\projects\my_track
```

When a reviewed bundle already exists, pass it explicitly so the run is tied to
that evidence snapshot:

```powershell
drivetrain-study run uncertainty .\projects\my_track `
  --bundle .\projects\my_track\results\track_build\<run>\track_bundle.json `
  --workers 4
```

## Choose the study that answers the question

| Command | What changes? | Engineering question |
| --- | --- | --- |
| `run baseline` | Nothing; nominal inputs only | What does finite ratio range cost in the declared nominal case? |
| `run track-robustness` | Measured gates and inputs marked `measured_track` | Does the conclusion survive plausible measured laps? |
| `run structural-sensitivity` | One declared structural assumption at a time | Which assumptions move the answer, and in what direction? |
| `run sweep` | One declared design variable across paired scenarios | Which tested design is best under this study contract? |
| `run uncertainty` | All declared stochastic roles jointly | What output distribution follows from everything admitted as uncertain? |

These studies are deliberately separate. A broad obstacle coefficient is not
measured lap variability simply because it belongs to the track. Its
uncertainty role must say what it means.

## Result hierarchy

Completed baseline and study results use the same human-first structure:

```text
result/
├── SUMMARY.md                 # recommendation, confidence, warnings, next actions
├── REPORT.md                  # technical narrative and compact evidence tables
├── decision_trace.md          # reasoning chain behind the headline
├── appendix/README.md         # map to every detailed artifact
├── provenance.json
├── provenance_graph.svg
├── run_manifest.json
├── track_bundle.json
├── resolved_inputs/
└── CSV, JSON, JSONL, plots, and traces
```

`SUMMARY.md` is designed for a two-minute review. It never silently upgrades a
short smoke run into a settled recommendation. Numerical validity, evidence
readiness, statistical readiness, directional robustness, and final decision
readiness are reported as distinct ideas.

## Safe interrupted runs and caching

Study runs write to a sibling `.incomplete` workspace and publish the final
directory only after reports and provenance are complete. Per-scenario
checkpoints permit an exact resume:

```powershell
drivetrain-study run uncertainty .\projects\my_track --resume
```

Use `--restart` to discard a matching incomplete workspace. `--resume` and
`--restart` are mutually exclusive, and a checkpoint from different resolved
inputs is rejected.

Simulation summaries are cached by a content fingerprint that includes the
resolved mechanism, scenario, design, and evidence contract. Inspect or clear
the project-local cache with:

```powershell
drivetrain-study cache status .\projects\my_track
drivetrain-study cache clear .\projects\my_track
```

Caching changes execution counts, not scientific results. Serial and parallel
execution are sorted back into deterministic scenario/design order.

## Project layout

```text
my_track/
├── project.toml
├── profiles/
├── track/
│   ├── track.toml
│   ├── runs.toml
│   ├── events.toml
│   └── gpx/
├── vehicles/
│   └── vehicle_A/
│       ├── vehicle.toml
│       └── drivetrain.toml
├── studies/
└── results/
```

Every physical scalar carries a nominal value, unit, source, and uncertainty
declaration. A value treated as exact still requires a `fixed` distribution and
a reason. Profile inheritance and command-line overrides retain provenance in
the resolved-input export.

## Track Evidence Bundle

The bundle is the simulator-facing evidence boundary. It contains only what a
downstream model is allowed to know about the measured track:

- a common distance coordinate and centreline diagnostics;
- reviewed feature geometry and response groups;
- accepted speed-gate evidence and confidence components;
- explicit obstacle-model contracts and uncertainties;
- run/lap identities used for paired empirical gate sampling;
- schema version, content fingerprint, source provenance, and checksum.

A baseline rebuilds the track when `--bundle` is omitted. Supplying a bundle is
the explicit reproducibility path; stale evidence is never chosen implicitly.

## Physical and statistical quality gates

A study is numerically valid only when all cases complete and pass:

- infinite-reference dominance;
- accepted-gate compliance;
- vehicle energy closure;
- engine-to-wheel energy closure.

The reports separately show physical scenario bands, bootstrap uncertainty on
estimated statistics, paired win/regret measures, threshold probabilities, and
convergence warnings. Attribution is suppressed below eight scenarios,
exploratory from eight to nineteen, and enabled as screening evidence at twenty
or more. Screening importance is not presented as an exact causal variance
fraction.

Project validation warnings and unresolved track-review records are carried into
every result. They do not block exploratory execution, but they prevent a result
from being marked decision-ready. A nominal baseline is never a design
recommendation.

## Model boundary and extension path

The current simulator includes one-dimensional translation, driven-wheel
rotation, a compact saturating longitudinal tire law, rolling and aerodynamic
resistance, explicit obstacles, braking gates, a bounded ideal CVT, and a
finite-launch-torque infinite reference.

The runtime model registry records small drivetrain and tire adapter protocols.
It is an API boundary, not a claim that external models are dynamically selected
by the current runner. Any external mechanism integration should preserve the
existing evidence and reporting contracts. In particular:

- a brush or measured tire model can replace the compact tire force law;
- calibrated obstacle models can retain feature and response-group identities;
- a new driveline can participate in paired studies if it returns the declared
  state, force, energy, and quality channels.

## Other commands

```powershell
drivetrain-study doctor [PROJECT]
drivetrain-study validate PROJECT [--study NAME] [--strict]
drivetrain-study ingest PROJECT
drivetrain-study review PROJECT
drivetrain-study validate-bundle TRACK_BUNDLE.json
drivetrain-study report RESULT_DIRECTORY
drivetrain-study results PROJECT [--json]
drivetrain-study migrate prototype-events old.csv events_migrated.toml
```

Migration intentionally copies geometry anchors only. It does not infer
obstacle physics from prototype coefficients.

## Documentation map

- `docs/USER_GUIDE.md` — practical end-to-end workflow
- `docs/DATA_HANDOFF_GUIDE.md` — track-first and vehicle-second data-entry checklist
- `docs/INPUT_CONTRACT.md` — field, unit, provenance, and uncertainty reference
- `docs/OUTPUT_REFERENCE.md` — files, plots, JSON, CSV, and column meanings
- `docs/DEVELOPER_GUIDE.md` — architecture, extension points, tests, versioning
- `docs/WORKED_ARIZONA_EXAMPLE.md` — measured-data walkthrough
- `docs/RELEASE_AND_REPRODUCIBILITY.md` — release and replay checklist
- `docs/methods/framework_methods.tex` — cross-referenced full methods document
- `output/pdf/framework_methods.pdf` — compiled and visually checked methods document
- `docs/input/`, `docs/output/`, and `docs/methods/` — focused contracts

## Declared limitations

- GPX elevation is preserved but not yet differentiated into road grade force.
- Obstacle equations are explicit approximations and need calibration for a
  specific vehicle and course.
- Lateral/yaw dynamics and curvature-derived corner-speed prediction are not
  active vehicle forces.
- The tire law is longitudinal and compact.
- The bounded ideal CVT is an intentionally reduced-order comparison mechanism.
- Measured speed gates reproduce evidence; they do not optimize a driver.
- A result cannot be more trustworthy than its declared inputs, reviewed track
  evidence, numerical checks, and sample count.

The preserved `prototype/` directory is a migration reference, not the active
implementation.
