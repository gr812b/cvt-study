# Track-bundle contract

`track_bundle.json` is the only supported boundary between track reconstruction
and the vehicle/CVT simulator. It is produced by `cvt-study build-track`
and can be checked independently with:

```powershell
cvt-study validate-bundle .\results\track_build\<timestamp>\track_bundle.json
```

The bundle is self-contained: loading it does not require the original GPX,
project TOML files, pandas data frames, or map-matching code.

## Version and integrity

The current role-aware format is:

```json
{
  "format": "cvt-track-bundle",
  "schema_version": "1.2.0"
}
```

The reader accepts patch releases in the supported `1.2.x` family. A new minor
or major version is rejected until the reader explicitly supports it. This avoids
silently interpreting a newer contract with older code.

Each build writes:

```text
track_bundle.json
track_bundle.sha256
```

The sidecar SHA-256 detects byte-level changes. The JSON also contains a
`content_fingerprint_sha256` computed from the meaningful track content while
excluding creation time and generator version. This fingerprint is reproducible
for the same reconstruction evidence and changes when the track contract changes.

## Scope

The bundle contains track evidence only. Vehicle mass, drag, tire, drivetrain,
and study settings are deliberately excluded. Therefore one unchanged bundle can
be used with vehicle A, vehicle B, several CVT designs, or several study definitions.

Track provenance includes:

- project, track, runs, events, and track-profile source hashes;
- GPX source hashes and run/vehicle/driver identifiers;
- track-related resolved configuration provenance;
- reconstruction counts, length, and reference-lap identity.

Paths are project-relative or reduced to an external profile filename. Absolute
machine paths are not part of the portable contract.

## Coordinate contract

The bundle uses one closed-course coordinate `s` in metres:

\[
0 \le s < L,
\]

where `L` is the reconstructed track length. The origin is the configured lap-gate
location projected onto the reference centreline, and positive `s` follows the
recorded driving direction.

Every interval contains:

```json
{
  "start_s_m": 100.0,
  "end_s_m": 120.0,
  "length_m": 20.0,
  "wraps_start_finish": false
}
```

Intervals follow the driving direction. When an interval crosses the origin,
`end_s_m < start_s_m` and `wraps_start_finish` is true. `length_m` is checked
against the start/end pair on the closed course.

## Simulation contract

`simulation_contract` contains only quantities a simulator may consume:

- track length and coordinate geometry;
- reference centreline samples;
- observed speed/elevation profile;
- physical-feature intervals;
- response-group intervals;
- speed-gate definitions and empirical target distributions;
- explicit capability flags.

The current capability flags are intentionally explicit:

```json
{
  "speed_gates_ready": true,
  "obstacle_models_ready": true,
  "uncertainty_roles_ready": true,
  "grade_force_ready": false
}
```

The bundle is ready to supply geometry, gate evidence, explicit physical-feature
models, and role-separated uncertainty contracts. GPX elevation is still not validated as road grade, so `grade_force_ready`
remains false.

## Physical features and response groups

A physical feature remains a distinct map object. It carries:

- permanent ID, name, sequence, and kind;
- analysis role and response-group ID;
- absolute closed-course interval;
- anchor coordinate and projection error;
- start/end geometry uncertainty and provenance;
- review flags;
- obstacle-model declaration status.

A response group combines features only when their GPS response cannot be separated.
It preserves the source physical-feature IDs and uses the union interval. This means
several map objects can share one measured speed distribution without being treated
as one physical obstacle.

Every physical feature contains an uncertainty-aware declared model. Even a feature
with no modeled resistance uses an explicit `none` declaration. Response groups do
not own physical models because they are evidence aggregations, not physical objects.

The simulator validates exact parameter names, dimensions, model relationships, and
`obstacle_models_ready` before accepting the bundle. Measured GPS speed change is
never promoted automatically to obstacle energy.

## Speed gates

The bundle includes one gate record per response group so accepted and rejected
evidence remains auditable. Only records with:

```json
{
  "status": "accepted",
  "active_by_default": true
}
```

enter the default simulator view.

The gate position is the physical feature-entry boundary. Its target is the median
speed measured over the configured window immediately before that boundary. Each
eligible lap contributes one empirical sample with its lap, run, vehicle, and driver
identity.

The gate contract is one-way:

- an upstream braking envelope may limit an approaching vehicle;
- a vehicle already below the sampled target is never reset upward;
- the braking-envelope parameter itself remains a vehicle/simulation input and is
  not invented by track reconstruction.

The p10, median, p90, mean, standard deviation, and IQR are stored for review, but
paired uncertainty studies should sample the retained empirical pass values rather
than reconstructing a distribution from only those summaries.

## Evidence section

`evidence` retains the audit trail behind the simulation contract:

- lap-quality records;
- gate-confidence weights and thresholds;
- component and overall gate scores;
- all event-pass measurements;
- review priorities, reasons, and actions.

The evidence section is not required in the vehicle ODE loop. It is present so a
result can always be traced back to the observations that justified the active
constraints.

## Simulator-facing Python view

The package exposes:

```python
from cvt_track_study.bundle import (
    load_track_bundle,
    simulation_track_from_bundle,
)

bundle = load_track_bundle("track_bundle.json")
track = simulation_track_from_bundle(bundle)
```

The returned immutable view imports neither GPX nor reconstruction modules. The
Phase 5 runtime builds nominal obstacle and vehicle mechanisms against this boundary.
