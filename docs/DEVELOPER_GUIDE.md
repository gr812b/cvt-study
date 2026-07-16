# Developer guide

## Architectural rule

The project is a measured track-based drivetrain design framework. The bounded
ideal CVT is one mechanism implementation, not the identity of the upstream
evidence system.

```text
configuration → GPX/FIT ingestion → track reconstruction → Track Evidence Bundle
              → mechanism adapter → paired studies → attribution → reports
```

Each arrow is a contract. Downstream code may consume the prior contract but
must not reach backward and reinterpret raw evidence silently.

## Package map

| Package | Ownership |
| --- | --- |
| `config` | project/profile merge, units, uncertainty validation, resolved export |
| `gpx` | safe XML parsing, canonical telemetry, ingestion diagnostics |
| `track` | laps, centreline, map matching, features, gate evidence, review |
| `bundle` | versioned simulator-facing evidence schema and checksum |
| `contracts` | obstacle declarations shared across configuration/bundle/runtime |
| `simulation` | longitudinal state, tire/obstacle/powertrain mechanisms, integrator, baseline |
| `uncertainty` | registered inputs, distributions, copula sampling, bootstrap statistics |
| `studies` | planning, paired execution, summaries, attribution, decision synthesis |
| `runtime` | cache, progress, workspaces, provenance, result index, extension registry |

The preserved `prototype` package is outside the active architecture.

## Dependency direction

Mechanism and integrator code must not import plotting or report writers.
Reporting consumes completed summaries/traces. Study sampling creates a scenario
before designs are executed so all compared designs share identical physical
inputs. The bundle consumer validates schema/capability before exposing runtime
track objects.

## External mechanism boundary

`runtime.models.DrivetrainAdapter` defines a small evaluation boundary:

```python
class DrivetrainAdapter(Protocol):
    identifier: str
    def evaluate(self, state, demand) -> Mapping[str, float]: ...
```

An external mechanism adapter should:

1. declare a versioned identifier and resolved configuration;
2. consume only the runtime track/vehicle state and driver demand;
3. return wheel-force capability plus state and energy diagnostic channels;
4. make every stored-energy and dissipative-loss channel explicit;
5. support deterministic replay from serialized state/inputs;
6. provide a valid comparison/reference policy rather than reusing ideal-CVT
   caching assumptions;
7. satisfy completion, gate, vehicle-energy, and powertrain-energy checks.

The current runner does not dynamically select implementations from the registry;
the protocol is an integration boundary for separate model work. External model
integration must not change telemetry ingestion, gate evidence, bundle geometry, or
uncertainty semantics.

## Tire extension

`TireForceAdapter` maps tire state and normal load to longitudinal force and loss.
A richer tire implementation must declare its state variables, valid domain,
force sign, dissipated-energy channel, and whether additional uncertain inputs
are structural or measured. If it couples lateral and longitudinal demand, the
bundle remains unchanged; curvature may be consumed as evidence only after the
vehicle model declares how it is used.

## Obstacle extension

Obstacle models are selected by explicit model type and parameters. New models
must define:

- geometry/feature applicability;
- force or energy equation and units;
- handling at zero speed and feature boundaries;
- additive physical energy channel;
- uncertainty support and model alternatives;
- validation tests with known limiting cases.

Do not calibrate obstacle physics from a single GPS speed loss without declaring
the other forces and uncertainty. Migration helpers intentionally refuse to
copy prototype coefficients as trusted physics.

## Study extension

A study type should separate planning from execution:

1. resolve a base mechanism and design domain;
2. register every sampleable input with role and physical support;
3. create paired `ScenarioDraw` objects;
4. execute every design against the same draw;
5. generate row-level comparisons and quality channels;
6. summarize physical variation separately from estimator uncertainty;
7. synthesize a decision only after numerical, evidence, and statistical gates;
8. publish machine artifacts before regenerable Markdown.

Design sweeps use one scenario-level infinite reference whose launch cap is frozen
before candidate transmission values are applied. Structural sensitivity does not
share references across levels. Tests must preserve both rules.

## Cache and run fingerprints

Simulation cache keys are canonical strict-JSON SHA-256 fingerprints of the
resolved case payload. They must include every value that can affect a summary.
Do not include timestamps, process IDs, machine paths, or installed build
metadata. Adding a mechanism field requires adding it to the key payload.

Study fingerprints own result directories and resume workspaces. A mismatched
fingerprint must fail closed. Per-scenario checkpoints are strict JSON and are
removed on successful commit.

## Reporting contract

Human reports are projections of machine artifacts and must be regenerable.
Keep the hierarchy:

1. `SUMMARY.md` — answer and caveats;
2. `decision_trace.md` — reasoning gates;
3. `REPORT.md` — compact technical evidence;
4. appendix — full audit trail.

Never hide a quality failure, low sample count, fallback gate draw, correlation
warning, or boundary optimum. Do not add counterfactual opportunity loss to a
physical energy partition.

## Strict serialization

All JSON/JSONL writers use `allow_nan=False`. Represent unavailable values with
`null` plus a status/reason. CSV column names are stable API; append columns but
do not reorder-dependent consumers. Timestamps are UTC ISO 8601.

## Tests

Run:

```powershell
py -m pytest -q
py -m compileall -q src
```

High-value tests include:

- configuration precedence, units, provenance, and invalid support;
- malformed GPX and adversarial lap/gate evidence;
- bundle canonicalization and checksum stability;
- analytic tire/obstacle/powertrain limits;
- timestep refinement, exact gate crossings, and both energy closures;
- paired sampling, correlations, reproducibility, bootstrap behavior;
- serial/parallel and uncached/cached scientific equivalence;
- interruption/resume and atomic result ownership;
- report regeneration from machine artifacts;
- wheel installation and CLI execution outside the source tree.

Use short deliberately adversarial runs for integration tests. A coarse run may
exercise a code path, but its quality failure must remain visible.

## Versioning

Package version, project schema, bundle schema, and model identifiers are
separate. Use semantic versioning for the package. A bundle reader may accept a
documented compatible minor family; it must reject unsupported contracts rather
than guess semantics. Model behavior changes require a new model identifier or
versioned contract and new cache fingerprints.

## Release flow

1. run source tests and compilation;
2. build a clean track and short runs for all study types;
3. validate strict JSON/JSONL and energy reconciliation;
4. compare serial/parallel and cached/uncached outputs;
5. build a wheel and install it into a fresh environment;
6. run the packaged CLI outside the source tree;
7. compare source and wheel scientific artifacts;
8. inspect representative reports and provenance SVGs;
9. archive source, methods, validation outputs, and checksums.

See `RELEASE_AND_REPRODUCIBILITY.md` for the operational checklist.
