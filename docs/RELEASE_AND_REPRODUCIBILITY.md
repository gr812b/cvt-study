# Release and reproducibility

## What makes a run replayable

A completed result preserves:

- the exact Track Evidence Bundle and checksum;
- fully resolved configuration and its provenance;
- package/framework version;
- study type, seed, sample count, and design domain;
- study/configuration fingerprints;
- exact sampled scenarios and gate-pairing identities;
- mechanism summaries, numerical quality, and execution policy;
- the invoked command and a provenance graph.

Cache contents are an optimization and are not required to interpret a result.
Raw GPX/FIT remains project evidence; the bundle is the exact simulator input.

## Before a decision run

- [ ] `drivetrain-study doctor PROJECT` has no failures.
- [ ] project validation warnings have been reviewed.
- [ ] GPX timestamps, segments, and speed provenance are understood.
- [ ] lap inclusion and centreline/map-match diagnostics are acceptable.
- [ ] every feature and response group is reviewed.
- [ ] accepted gates have enough passes and credible component scores.
- [ ] inherited defaults are identified and not described as measurements.
- [ ] uncertainty roles reflect physical meaning.
- [ ] the design domain and decision thresholds answer the intended question.
- [ ] numerical step has passed refinement checks for the declared mechanism.

## Release verification

- [ ] all tests pass from a clean source tree;
- [ ] source compiles without syntax errors;
- [ ] all four study runners complete a short valid integration path;
- [ ] JSON and JSONL parse under a strict parser;
- [ ] feature obstacle energy reconciles with the physical total;
- [ ] serial and parallel scientific artifacts match;
- [ ] cached and uncached scientific artifacts match;
- [ ] interruption leaves an unpublished workspace and resume reuses checkpoints;
- [ ] baseline and study reports regenerate without rerunning physics;
- [ ] a fresh wheel builds and installs;
- [ ] installed CLI runs outside the source checkout;
- [ ] source and installed-wheel scientific outputs match;
- [ ] representative SUMMARY, REPORT, trace, appendix, plots, and provenance are inspected;
- [ ] archives pass ZIP integrity and SHA-256 checks.

## Replaying a result

1. Install the recorded framework version.
2. Verify `track_bundle.sha256`.
3. Inspect `resolved_inputs/` and `provenance.json`.
4. Recreate the project or use the archived project tree.
5. invoke the recorded study with the preserved bundle and seed.
6. compare row-level scientific artifacts, allowing only documented timestamp,
   path, cache-count, or package-build differences.

`drivetrain-study report RESULT` is not a replay of physics. It is a safe
regeneration of the human Markdown projection from existing machine evidence.

## Expected non-equivalences

Manifests may differ in creation time, command spelling (`cvt-study` alias),
worker count, cache hit/miss counts, and local path. Scientific summaries,
scenario draws, energy accounting, attribution, convergence, and decisions must
not change because of workers or cache use.

## Archive contents

A release archive should include active source, tests, templates, profiles,
documentation, examples, changelog, license if supplied, and build metadata. It
should exclude virtual environments, bytecode, test caches, build output,
project result caches, and incomplete result workspaces. A separate optional
archive may preserve the legacy prototype for migration comparison.
