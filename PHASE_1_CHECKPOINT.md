# Phase 1 checkpoint — workspace, resolution, and validation

## Implemented

### Workspace and CLI

- Installable `cvt-study` command.
- `cvt-study init <directory>` creates the bundled project template.
- `cvt-study validate <project>` accepts a directory or `project.toml`.
- Optional study selection, strict warning mode, explicit output directory, and
  repeatable `--set PATH=VALUE` overrides.

### Configuration resolution

- Built-in profiles bundled with the package.
- Project-local and external shared profile roots.
- Multi-level profile inheritance.
- Duplicate-profile and inheritance-cycle detection.
- Track and vehicle profile scopes.
- Project vehicle/drivetrain files applied after profiles.
- Selected study overrides applied only to existing paths.
- Command-line overrides limited to existing leaves.

### Uncertainty-first validation

- Bare numeric values in vehicle/drivetrain physical sections are rejected.
- Every numeric physical quantity requires nominal, unit, source, and uncertainty.
- Fixed uncertainty requires a reason.
- Complete quantities replace atomically across inheritance layers.
- Partial nominal overrides produce an explicit warning.
- Numeric uncertainty and categorical model-form uncertainty are separate types.
- Known parameters receive dimension and physical-domain checks.
- Broad inherited defaults are listed prominently.

### Cross references and exports

- Project-owned paths are constrained to the project directory.
- GPX-only run references, unique run IDs, vehicle references, and centreline
  selection are checked.
- Study type, sampling mode, replicate count, vehicle, and design-variable paths
  are checked.
- Resolved TOML, provenance chain, validation report, and resolution manifest are
  exported beside the project.

## Important intentional behavior

- A newly initialized project validates with warnings because it has no real GPX
  runs and still contains clearly labeled placeholder evidence/defaults.
- Warnings do not fail normal validation. `--strict` makes them fail for release or
  CI workflows.
- No backward compatibility is provided for the prototype file formats or scripts.
- GPX content is not parsed yet; Phase 1 validates only the workspace contract and
  file references.

## Review findings corrected during implementation

1. **Stale distribution fields during inheritance.** Deep-merging a local normal
   uncertainty into an inherited triangular uncertainty initially retained
   `lower`, `mode`, and `upper`. Complete quantities are now atomic replacements.
2. **Provenance after atomic replacement.** Removed uncertainty fields initially
   remained in the provenance map. Stale leaf records are now pruned while history
   for retained leaves is preserved.
3. **Categorical versus numeric uncertainty.** Discrete model alternatives no
   longer masquerade as unit-bearing numeric quantities. `UncertainChoice` now owns
   fixed/discrete categorical declarations.
4. **Installed-template reliability.** `init` now copies package resources without
   assuming the package is an editable filesystem checkout and recreates required
   empty directories.
5. **Silent typo risk in study overrides.** Study overrides may replace only
   existing paths; unknown paths are errors and are not merged.
6. **Monolithic loader.** Validation was separated from project loading/resolution
   to keep ownership clearer and make Phase 2 extension safer.

## Verification

The final checkpoint is accepted only after:

- clean package tests pass;
- deliberate malformed-project tests pass;
- source compilation passes;
- editable-install CLI smoke test passes;
- built wheel installs in a fresh virtual environment and its bundled `init`
  template validates;
- preserved prototype integration and GPS-analysis tests still pass.

Detailed commands and results are recorded in `CODE_REVIEW_PHASE_1.md`.

## Next review gate

Before Phase 2, inspect:

1. the generated project folder;
2. the active inherited-default warnings;
3. `resolved_inputs.toml`;
4. the profile/project provenance chain for mass, drag area, and efficiency;
5. the behavior of one deliberate invalid unit or missing uncertainty.

### Completed verification totals

- Phase 1 tests: **46 passed**.
- Prototype simulator tests: **5 passed**.
- Prototype GPS-analysis tests: **11 passed**.
- Isolated wheel installation and CLI smoke test: passed.
