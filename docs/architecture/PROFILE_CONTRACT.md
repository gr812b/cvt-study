# Profile contract

Profiles are reusable, versioned configuration layers. They are intended for
values that remain reasonably stable across tracks or vehicles, especially broad
estimates that are expensive to remeasure for every project.

## File shape

```toml
[profile]
id = "team.vehicle.baja_defaults_v1"
scope = "vehicle"
version = 1
description = "Team estimates for the current vehicle generation."
extends = ["builtin.vehicle.baja_generic_v1"]

[config.drivetrain.efficiency]
nominal = 0.80
unit = "1"
source = { kind = "engineering_estimate", reference = "2026 drivetrain review" }
uncertainty = { distribution = "triangular", lower = 0.75, mode = 0.80, upper = 0.86 }
```

Required metadata:

- `id`: globally unique stable identifier;
- `scope`: currently `vehicle`, `track`, or `mixed`;
- `version`: positive integer;
- `extends`: optional parent profile IDs;
- `config`: configuration subtree contributed by the profile.

## Discovery

Built-in profiles ship with the package. Additional roots are listed in
`project.toml`:

```toml
[profiles]
roots = ["profiles", "C:/Baja/shared_profiles"]
```

Roots are scanned recursively for `.toml` files. Relative roots resolve from the
project. Missing configured roots, duplicate IDs, malformed metadata, missing
parents, scope mismatches, and inheritance cycles are validation errors.

## Inheritance order

Parents are applied before children. Profiles selected by a vehicle or track are
then applied in the order listed, followed by the local project file. Applying a
parent through more than one selected child does not duplicate it.

A profile should extend a built-in under a new ID rather than redefine the same ID.
This makes the provenance chain unambiguous.

## What belongs in a profile

Good profile candidates:

- a measured tire diameter reused for the same tire/pressure setup;
- broad drag-area or drivetrain-efficiency estimates;
- manufacturer engine data;
- team-calibrated tire or resistance parameters;
- later, documented obstacle-model priors.

Values that belong in a track or vehicle file instead:

- one specific vehicle's measured mass;
- one project's sprocket selection;
- track-specific event geometry;
- a temporary study design candidate.

## Provenance and uncertainty

Profiles do not bypass the uncertainty contract. Every physical value still
requires units, source, and uncertainty. Broad package defaults use
`source.kind = "inherited_default"`, and validation reports every such default
remaining in the resolved vehicle.
