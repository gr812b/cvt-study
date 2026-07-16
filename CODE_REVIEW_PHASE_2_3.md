# Code review — Phases 2 and 3

## Review scope

The review covered GPX parsing, project-level ingestion, lap reconstruction,
centreline creation, map matching, event projection, response grouping, event
metrics, speed-gate scoring, review generation, exports, packaging, and the
reference project.

The preserved prototype was treated as a regression fixture only; no backward
compatibility requirements were imposed on the clean package.

## Findings corrected

### Phase 2

1. **Read-only/ambiguous exported arrays**
   Optional numeric GPX fields are normalized to numeric dtypes, and centreline
   construction later requests writable NumPy copies explicitly.

2. **Misleading duration after timestamp regression**
   Segment duration is now exported only with complete, monotonic timestamps.
   Backward timestamps generate fatal ingestion diagnostics for track building.

3. **Missing and invalid timestamps were conflated**
   They now have separate counts and diagnostic codes, plus a combined unusable
   timestamp count.

4. **GPX scope wording was too broad**
   Routes and waypoints are reported but explicitly ignored. Only track segments
   become driven telemetry.

5. **Unsafe XML inputs**
   The parser uses `defusedxml` and regression tests reject entity expansion.

6. **Packaging risk**
   A wheel is built and installed outside the source tree; its project template
   and built-in profiles are exercised from the installed package.

### Phase 3

1. **Monolithic reconstruction module**
   The initial implementation exceeded 1,100 lines. It was split into focused
   modules for settings, laps/centreline, event geometry, pass metrics, gate
   evidence, review, export, and orchestration. The orchestration module is now a
   small data-flow composition layer.

2. **Uncertainty stopped at the anchor**
   Physical start/end and extent uncertainty are now explicit. Effective boundary
   errors combine projection and declared terms and are carried into gate review.

3. **Redundant endpoint input**
   Events with explicit start and end coordinates no longer need an extent table.
   Extent data is required only for a missing side.

4. **Gate quality used the wrong geometry**
   Entry gates now use physical-start effective error, not anchor uncertainty
   alone. Separate projection, declared uncertainty, and combined values are
   exported for both boundaries.

5. **Non-candidates could hide geometry failures**
   `must_fix` geometry status now takes precedence over `not_a_candidate`.

6. **Missing speed could pass lap quality silently**
   Laps now export speed-coverage fraction and require a configurable minimum.

7. **Closed-loop overlap check omitted the last-to-first pair**
   Adjacent feature overlap is now checked across start/finish as well as within
   the linear event order.

8. **One vehicle appeared like agreement evidence**
   The cross-vehicle component is marked `single_vehicle_neutral`; two-vehicle
   behavior has an integration test.

9. **Lap gate could become a speed gate implicitly**
   Gate candidacy remains an explicit user declaration.

10. **User review lacked enough context**
    HTML and CSV review outputs now include empirical intervals, slowdown class,
    coordinate error, cross-vehicle status, speed coverage, reasons, and actions.

## Verification performed

- clean-package unit and integration suite;
- deliberate malformed-input and contract-failure tests;
- secure-XML test;
- missing-elevation test with runtime warnings promoted to errors;
- timestamp-regression rejection;
- explicit-endpoint/no-extent build;
- cross-start/finish overlap test;
- multi-vehicle gate-agreement test;
- configuration template synchronization test;
- Python bytecode compilation;
- runtime resolution of all public type annotations;
- strict reference-project validation;
- source-checkout reference ingestion and track build;
- wheel build, isolated installation, packaged `init`, validation, ingestion,
  and track build;
- preserved prototype simulator and GPS-analysis regression suites;
- visual inspection of the generated map and elevation profile.

## Known limitations retained deliberately

- The local tangent approximation is intended for compact courses, not regional
  routes.
- The reference centreline currently comes from the fastest clean eligible lap,
  not a robust multi-lap geometric average.
- The progress-aware map matcher is heuristic and should still be reviewed at
  crossovers or tightly parallel lanes.
- Gate confidence is a transparent engineering score, not a calibrated posterior
  probability.
- GPX speed-field semantics depend on the recorder; raw reported and derived
  values remain separate for audit.
- GPX altitude is preserved and summarized but not converted to grade.
- The HTML review is intentionally static in this checkpoint.
- Declared geometry and empirical gate uncertainty are exposed but not yet
  sampled through simulation.
- No obstacle force/energy model or simulation-ready track bundle exists until
  Phases 4 and 5.

## Review conclusion

Phases 2 and 3 are suitable as the foundation for the Phase 4 bundle boundary.
The source and packaged workflows agree on the deterministic reference fixture,
and the remaining limitations are explicit model-scope choices rather than
hidden implementation failures.
