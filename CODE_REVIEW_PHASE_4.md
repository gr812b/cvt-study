# Phase 4 code review

## Scope reviewed

- bundle construction and JSON normalization;
- version and semantic validation;
- checksum and content fingerprinting;
- portable provenance;
- simulator-facing immutable view;
- CLI integration;
- source and isolated-wheel workflows;
- real Arizona output.

## Findings corrected during review

### Bundle fingerprint initially depended on build time

The first draft included volatile track-build metadata in provenance, making
identical evidence produce different fingerprints. Volatile creation paths/times
are now excluded. Repeated builds produce the same content fingerprint.

### Bundle provenance initially included vehicle and study files

That would have forced a track rebuild after changing mass, drag, gearing, or a
study definition. Provenance is now filtered to track reconstruction inputs only.
A regression test changes vehicle mass by 50 kg and verifies the bundle fingerprint
is unchanged.

### Absolute source paths leaked into profile provenance

Profile labels embedded absolute paths inside descriptive strings and profile-root
values. Both source labels and nested values are now made project-relative or
reduced to an external filename. The portable bundle contains no build-machine path.

### One construction module became too large

The first implementation placed geometry, gates, provenance, JSON normalization,
and orchestration in one file. It was split into focused modules:

- `geometry.py`;
- `gates.py`;
- `provenance.py`;
- `serialization.py`;
- `builder.py`.

### JSON output could not rely on permissive NaN behavior

Pandas and NumPy missing values are now converted to JSON `null`; non-finite values
are rejected and serialization uses `allow_nan=False`.

### File checksum alone was insufficient

A missing sidecar would otherwise remove integrity checking. The bundle now also
contains a reproducible semantic content fingerprint which is recomputed during
validation.

### Version strings needed an explicit policy

The reader accepts patch changes in `1.0.x`, rejects newer minor versions, and
rejects a different major version. Tests cover each behavior.

### Isolated installation exposed an undeclared SciPy dependency

Pandas delegates `Series.corr(..., method="spearman")` to SciPy. The development
environment happened to contain SciPy even though the project did not declare it.
The gate scorer now computes average ranks and applies ordinary Pearson correlation
to those ranks, preserving Spearman behavior without adding a large hidden
dependency. The wheel was rebuilt and the Arizona workflow passed in a fresh
environment containing only declared dependencies.

### The simulation boundary needed executable form

A typed immutable simulator view was added. It is produced entirely from a loaded
bundle and explicitly reports that full vehicle simulation is not ready while
obstacle models remain undeclared.

## Remaining intentional limitations

- The schema supports closed courses only.
- Obstacle models are not declared or resolved.
- Grade force is disabled.
- Gate braking-envelope magnitude is not a track property and will be added through
  the Phase 5 vehicle/simulation configuration.
- The JSON Schema file is a published structural aid; the Python semantic validator
  remains authoritative for cross-field rules.
- No migration exists from prototype `simulator_track_bundle.json`; backward
  compatibility was intentionally not preserved.

## Review conclusion

The Phase 4 boundary is suitable for Phase 5. It is portable, reproducible,
self-contained, independent of vehicle design, strict about unsupported physics,
and retains enough evidence to audit every active speed gate.
