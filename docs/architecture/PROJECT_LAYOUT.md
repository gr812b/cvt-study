# Project layout contract

Every command takes a project directory or its `project.toml`. Track data,
vehicles, study definitions, and generated results remain organized together.

```text
my_endurance_project/
├── project.toml
├── profiles/
│   └── vehicles/
├── track/
│   ├── track.toml
│   ├── runs.toml
│   ├── events.toml
│   └── gpx/
├── vehicles/
│   ├── vehicle_A/
│   │   ├── vehicle.toml
│   │   └── drivetrain.toml
│   └── vehicle_B/
├── studies/
└── results/
```

## Project-owned paths

Track, runs, events, vehicle, study, and result paths must be relative and remain
inside the project directory. This keeps a copied project coherent and prevents a
result from depending on an accidentally selected file elsewhere.

Profile roots are the exception: they may be project-relative or absolute because
a team may deliberately maintain one shared profile library for several tracks.
Configured missing profile roots are errors rather than silently ignored inputs.

## GPX/FIT raw telemetry

`track/runs.toml` may reference `.gpx` or `.fit` files. Each declared run identifies its
vehicle, run, and driver and states whether it contributes to centreline and gate
evidence. The validator checks paths and references, and ingestion normalizes both
formats while retaining source channels and certainty.

An empty run list is allowed while a project is being configured and produces a
clear warning. A later `build-track` command will require at least one run selected
for centreline construction.

## Reusable profiles

The package ships versioned built-in profiles. `project.toml` adds user roots:

```toml
[profiles]
roots = ["profiles", "C:/Baja/shared_cvt_profiles"]
```

Project-local profiles make a repository self-contained. Shared roots let the same
measured or estimated vehicle assumptions be reused from track to track.

## Results

Commands never modify source track, vehicle, or study inputs. Validation, ingestion,
and track building write timestamped artifacts under the project-owned `results/`
tree. Every track build includes `track_bundle.json` and `track_bundle.sha256`; these
files may then be copied or archived independently because they contain the complete
track-side simulation contract and evidence.
