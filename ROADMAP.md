# Implementation roadmap

The implementation stops at the end of every core phase for review. A phase is
not considered complete until its contract tests pass and its generated artifacts
have been inspected.

## Phase 0 — Architecture and uncertainty-first contracts **[implemented]**

- Freeze terminology and ownership boundaries.
- Adopt a self-contained project directory.
- Use GPX as the only raw GPS input format.
- Put uncertainty beside every physical numeric input.
- Require an explicit reason for zero uncertainty.
- Define reusable profile and override semantics.
- Implement typed uncertainty primitives and contract tests.

**Review gate:** approve terminology, project layout, uncertainty syntax, and
configuration precedence before implementing loaders.

## Phase 1 — Project workspace, configuration resolution, and validation **[implemented]**

- Implement `project.toml` discovery from a project directory.
- Load project, track, vehicle, study, and profile TOML files.
- Resolve built-in profile → user profile → project file → study override → CLI override.
- Validate units, references, missing uncertainty, and fixed-value reasons.
- Export a fully resolved configuration with provenance for every value.
- Add `init` and `validate` commands with useful error messages.

**Review gate:** inspect the generated workspace, inherited-default warnings,
resolved TOML, provenance chains, and deliberate-failure diagnostics before Phase 2.

## Phase 2 — GPX ingestion and canonical telemetry **[implemented]**

- Parse GPX tracks, segments, timestamps, and elevation.
- Preserve segment breaks and raw source values.
- Attach run, vehicle, and driver metadata from `track/runs.toml`.
- Produce one canonical telemetry table.
- Report missing timestamps/elevation and suspicious sampling gaps.
- Store elevation for review and future grade work without yet applying grade force.

**Review gate:** compare canonical output from several GPX files and inspect the
elevation and timestamp quality report.

## Phase 3 — Track reconstruction, evidence, and review **[implemented]**

- GPS cleaning, lap detection, centreline construction, and projection to `s`.
- Event geometry, response groups, pass metrics, slowdown signatures, and gate evidence.
- Decomposed confidence scores and actionable review status.
- Map and elevation-profile review package.

**Review gate:** manually inspect every `must_fix` and representative accepted,
review-only, and rejected gates.

## Phase 4 — Versioned track bundle and simulation boundary **[implemented]**

- Produce a versioned track bundle containing geometry, elevation, events, gates,
  empirical distributions, provenance, and declared models.
- Publish an immutable simulator-facing view that consumes only the bundle.
- Add semantic schema, integrity, portability, and compatibility tests.

**Review gate:** verify the same bundle fingerprint survives unrelated vehicle/study
changes, loads without source GPX, and rejects unsupported or tampered contracts.
Actual vehicle simulation migration belongs to Phase 5.

## Phase 5 — Explicit obstacle and vehicle mechanisms **[implemented]**

- Freeze obstacle interfaces and equations.
- Add versioned broad default profiles for hard-to-measure coefficients such as
  impact severity only after their physical meaning and units are fixed.
- Refactor vehicle, tire, resistance, CVT, gate-control, and energy accounting.
- Carry elevation through; keep grade force disabled until altitude processing is validated.

**Review gate:** equation-level tests, energy closure, timestep convergence, and
traceability from every output term to a resolved input.

## Phase 6 — Paired uncertainty propagation and studies **[implemented]**

- Joint scenario sampling with correlation groups.
- Baseline, design sweep, measured-track robustness, structural sensitivity, and
  full uncertainty propagation.
- Separate physical output variability from Monte Carlo estimation error.
- Cache one shared unbounded reference wherever mathematically valid.

**Review gate:** convergence checks, paired-design invariance tests, and comparison
against deterministic zero-uncertainty cases.

## Phase 7 — Sensitivity attribution and reporting **[implemented]**

- Physical energy attribution.
- Signed sensitivity and normalized elasticity.
- Uncertainty-weighted importance.
- Optional global attribution for sufficiently large studies.
- Final reports that distinguish track variability, model uncertainty, and sampling error.

**Review gate:** verify attribution using synthetic problems with known sensitivities.

## Phase 8 — Operational reporting and handoff **[implemented]**

- Compact operational README.
- Full LaTeX methods document in data-flow order.
- Input/output reference and developer guide.
- Migration utility for the prototype project.
- Progress reporting, caching, run manifests, hashes, deterministic seeds, and release tests.

**Review gate:** a new user completes the reference workflow from GPX to report
without reading source code.

## Current readiness closure

Implemented in the current checkpoint:

- numerical, evidence, statistical, directional, and final decision-readiness
  fields have distinct meanings;
- project validation warnings and unresolved track-review findings propagate to
  baseline and study summaries, reports, manifests, and decision traces;
- result indexes distinguish numerical validity from evidence and decision readiness;
- the track-first/vehicle-second user handoff is documented;
- the Arizona example remains explicitly exploratory until its inputs and review
  findings are replaced by a user.

Remaining before a team decision run:

- a user re-enters and reviews current Arizona track, vehicle, and drivetrain data;
- all `must_fix` and `recommended_review` track records are dispositioned;
- inherited defaults are replaced or explicitly accepted with defensible sources;
- the design domain and thresholds are confirmed by the study owner;
- production-size paired studies are run only after the evidence handoff is complete.

Tracked functional follow-ups that do not block the data handoff:

- benchmark thread-based study workers on the intended compute environment and
  evaluate a process-backed runner before production-size studies if CPU scaling
  is inadequate;
- consolidate the retained Phase 6 study helpers and Phase 8 orchestration after
  output compatibility is frozen;
- defer plotting imports so lightweight CLI commands do not initialize the
  plotting stack;
- decide whether physical feature-coordinate uncertainty should become sampled
  or remain a carried evidence limitation.

Intentional model-scope limitations remain documented: grade force is disabled,
feature-coordinate uncertainty is carried but not sampled, obstacle and tire
mechanisms are reduced-order unless calibrated, and lateral/yaw dynamics are not
active. Detailed physical drivetrain-model development is outside this repository.
