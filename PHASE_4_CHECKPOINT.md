# Phase 4 checkpoint — versioned track bundle and simulation boundary

## Status

Phase 4 is complete. Every track build now emits a validated, self-contained,
versioned `track_bundle.json` plus a SHA-256 sidecar. The bundle separates the
track-evidence pipeline from future vehicle/CVT simulation.

Phase 5 has not started. Obstacle equations and the migrated vehicle simulator are
still deliberately absent.

## User-facing entry points

```powershell
cvt-study build-track .\examples\arizona_endurance_project
cvt-study validate-bundle .\path\to\track_bundle.json
```

A successful build prints both the review-package path and bundle path.

## Bundle contents

The `1.0.0` contract contains:

- closed-course `s` coordinate and interval rules;
- complete centreline geometry and retained reference elevation;
- p10/median/p90 observed speed and elevation profiles;
- all physical features and response groups;
- geometry uncertainty and review flags;
- all gate statuses, confidence components, and empirical lap samples;
- active-by-default gate selection;
- lap, pass, confidence, and review evidence;
- project-relative source hashes and track-only provenance;
- explicit capability and uncertainty-status declarations.

## Important boundary decisions

- Vehicle and study configuration are excluded from bundle provenance and content
  fingerprint. Changing mass or final drive does not rebuild the track.
- Physical features remain separate from response groups.
- Accepted gates are active; review, rejected, must-fix, and non-candidate records
  remain visible but inactive.
- A gate is located at physical entry and its empirical target comes from the
  immediately upstream measurement window.
- Gate enforcement is one-way and may never accelerate a slow vehicle.
- The braking-deceleration assumption belongs to the vehicle/simulation layer.
- Obstacle models are explicitly `undeclared`; no net GPS speed change is mislabeled
  as calibrated terrain energy loss.
- Elevation is retained, but grade force remains disabled.

## Integrity and compatibility

- Semantic schema version: `1.0.0`.
- Reader accepts supported patch versions and rejects newer minor/major versions.
- Strict validation checks coordinate domains, interval closure, unique identities,
  feature/group references, gate activation rules, empirical distributions, finite
  JSON numbers, confidence weights, and capability limits.
- The adjacent SHA-256 detects byte changes.
- The internal content fingerprint detects semantic changes even without the sidecar.
- The content fingerprint is stable across repeated builds and unrelated vehicle
  configuration edits.

## Simulator-facing view

`simulation_track_from_bundle()` creates immutable geometry, feature, and active-gate
objects using only the validated JSON bundle. It imports no GPX, pandas, or track
reconstruction code. `ready_for_full_vehicle_simulation` remains false until Phase 5
resolves obstacle models.

## Verification

- 92 clean-package tests passed.
- 5 preserved simulator tests passed.
- 11 preserved GPS-analysis tests passed.
- Source compilation passed.
- Real Arizona `validate -> ingest -> build-track -> validate-bundle` passed.
- Isolated built-wheel installation and packaged example workflow passed.
- Bundle JSON was parsed with strict finite-number output.
- Bundle and checksum were visually/structurally inspected.
