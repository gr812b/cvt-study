# Phase 1 code review

## Scope reviewed

- project discovery and owned-path resolution;
- profile scanning, inheritance, and scope checks;
- deep merge and provenance behavior;
- uncertainty and unit validation;
- run, vehicle, track, and study cross references;
- TOML export;
- packaged CLI template behavior.

## Review method

- unit and integration tests covering normal and deliberately invalid projects;
- compile-all syntax check;
- editable-install CLI smoke test;
- wheel-build and isolated virtual-environment smoke test;
- manual inspection of resolved TOML and representative provenance chains;
- regression execution of the preserved prototype tests.

## Defects found and fixed

- Complete uncertainty-aware quantities were being recursively merged, allowing
  obsolete fields from the inherited distribution to survive.
- Provenance retained removed fields after an atomic replacement.
- Study override typos could create new configuration branches silently.
- The project template lookup assumed a filesystem-style editable installation.
- Numeric and categorical uncertainty were represented by one class despite
  different unit and nominal-value contracts.
- Loader and validation responsibilities were initially concentrated in one large
  module and were separated.

## Remaining limits, intentionally deferred

- The unit registry is explicit rather than a general dimensional-algebra engine.
  New mechanisms must register their accepted units and expected dimensions.
- GPX files are checked for location and extension only; XML/track-point validation
  begins in Phase 2.
- Event geometry receives only minimal ID checks until the Phase 3 contract.
- Profiles can supply future physical parameters, but only currently known paths
  receive specialized domain checks.
- Input hashes and immutable run IDs remain Phase 8 release work.

No known Phase 1 correctness issue remains after the final test and packaging pass.

## Final verification results

- Clean Phase 1 package: **46 tests passed**.
- Preserved simulator integration suite: **5 tests passed**.
- Preserved GPS-analysis suite: **11 tests passed**.
- `python -m compileall -q src`: passed.
- Editable-install `cvt-study init` and `validate`: passed.
- Built wheel installed into a fresh virtual environment: passed.
- Wheel-installed package contained the built-in profiles and complete project
  template; generated resolved artifacts successfully.
