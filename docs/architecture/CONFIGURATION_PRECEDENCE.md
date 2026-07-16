# Configuration precedence and provenance

Configuration resolves from broad reusable assumptions to narrow local choices:

```text
built-in profile
→ inherited parent profile(s)
→ selected user/project profile(s)
→ project track or vehicle file
→ selected study override
→ command-line leaf override
```

## Profile selection

A vehicle selects profiles in `vehicle.toml`:

```toml
[vehicle]
id = "vehicle_A"
profiles = ["team.vehicle.baja_defaults_v1"]
```

A profile is self-describing and may extend other profiles:

```toml
[profile]
id = "team.vehicle.baja_defaults_v1"
scope = "vehicle"
version = 1
extends = ["builtin.vehicle.baja_generic_v1"]

[config.drivetrain.efficiency]
nominal = 0.80
unit = "1"
source = { kind = "engineering_estimate", reference = "team prior" }
uncertainty = { distribution = "triangular", lower = 0.75, mode = 0.80, upper = 0.86 }
```

Profile IDs are globally unique across built-in and configured roots. A user
profile should extend a built-in profile under a new ID rather than silently
shadowing it. Duplicate IDs and inheritance cycles are errors.

## Quantity replacement

A complete physical quantity is atomic. When a later layer provides `nominal`,
`unit`, `source`, and `uncertainty`, the earlier quantity is replaced as one object.
This prevents stale triangular bounds surviving when a project switches an
inherited value to a normal distribution.

A partial override is permitted but warned about. For example, changing only
`nominal` keeps the inherited source and uncertainty. That can be useful for a
quick command-line test, but it must be a conscious choice.

## Study overrides

Study files may contain a nested `[config_overrides]` tree. Every override path
must already exist in the resolved project; studies cannot silently invent new
physical parameters. Canonical physical values belong in profiles or project
files.

## Provenance

Every resolved leaf records an ordered chain containing:

- layer type;
- source profile/file/command;
- set or override action;
- value supplied by that layer.

The chain is exported to `provenance.json`. Fields removed by an atomic quantity
replacement are removed from the resolved provenance map as well.
