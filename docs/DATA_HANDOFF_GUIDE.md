# Track and vehicle data handoff guide

This is the practical checklist for replacing example assumptions with current
team evidence. It complements the field-by-field `INPUT_CONTRACT.md`; it does
not replace the detailed contracts linked below.

Do the track handoff first. The reviewed Track Evidence Bundle is the boundary
consumed by every vehicle and study run.

## 1. Track-data handoff

The track-data owner supplies and reviews these source files:

| File | What the user enters | Detailed reference |
| --- | --- | --- |
| `track/gpx/*.gpx` | Original closed-course recordings with track points and usable timestamps | [`input/GPX_CONTRACT.md`](input/GPX_CONTRACT.md) |
| `track/runs.toml` | One unique run ID plus vehicle/driver identity and evidence-use flags per GPX | [`input/GPX_CONTRACT.md`](input/GPX_CONTRACT.md) |
| `track/track.toml` | Course identity, surface declaration, lap/reconstruction limits, review thresholds, and window settings | [`INPUT_CONTRACT.md`](INPUT_CONTRACT.md#track-fields) |
| `track/events.toml` | Lap gate, physical features, response groups, coordinates/extents, source statements, uncertainty, and obstacle declarations | [`input/EVENT_CONTRACT.md`](input/EVENT_CONTRACT.md) |

Recommended sequence:

```powershell
drivetrain-study validate PROJECT
drivetrain-study ingest PROJECT
drivetrain-study build-track PROJECT
```

The track-data owner should then review, in this order:

1. `review/REVIEW_SUMMARY.md` for required fixes and recommended reviews;
2. `track/lap_quality.csv` for excluded laps and quality flags;
3. `review/track_map.png` or `review/track_review.html` for centreline and feature placement;
4. `track/event_projection.csv` for each physical start/end and uncertainty;
5. `track/response_features.csv` for response-group membership;
6. `track/gate_review.csv` and `track/event_passes.csv` for pass-level gate evidence;
7. `track_bundle.json` and its checksum after the human review is complete.

The bundle is generated output. Do not edit it by hand. Correct `runs.toml`,
`track.toml`, `events.toml`, or the source evidence and rebuild instead.

A decision run remains evidence-blocked while any project validation warning,
`must_fix` event, or `recommended_review` event remains. Single-run,
single-vehicle, and single-driver evidence is retained as an advisory limitation
even when the configured review rules accept the gates.

## 2. Vehicle-data handoff

After the track package is reviewed, the vehicle-data owner updates:

| File | What the user enters | Detailed reference |
| --- | --- | --- |
| `vehicles/<id>/vehicle.toml` | Mass, tire diameter/inertia, driven load fraction, rolling resistance, drag area, air density, and tire-force parameters | [`input/VEHICLE_AND_STUDY_CONTRACT.md`](input/VEHICLE_AND_STUDY_CONTRACT.md) |
| `vehicles/<id>/drivetrain.toml` | Efficiency, final drive, engine declaration, ideal-CVT ratio bounds, target speed, power scale, and launch-clutch contract | [`input/VEHICLE_AND_STUDY_CONTRACT.md`](input/VEHICLE_AND_STUDY_CONTRACT.md) |
| optional `profiles/` | Reusable team measurements or assumptions shared by more than one project | [`architecture/PROFILE_CONTRACT.md`](architecture/PROFILE_CONTRACT.md) |

Every physical scalar must carry all four parts of the quantity envelope:

1. nominal value;
2. unit;
3. source kind and reproducible reference;
4. uncertainty distribution and, when stochastic, its semantic role.

Use `source.kind = "measured"` only for an actual measurement. Manufacturer
data, derived calculations, engineering estimates, and inherited defaults must
retain their real provenance. A fixed value still needs an explicit reason.
The complete syntax and supported distributions are in
[`INPUT_CONTRACT.md`](INPUT_CONTRACT.md#quantity-envelope).

Run validation after each group of edits:

```powershell
drivetrain-study validate PROJECT --study baseline --strict
```

`--strict` returns a failure while warnings remain. It is useful as a handoff
check; warnings do not prevent exploratory simulations. After a run, inspect
`resolved_inputs/validation_report.json`, `resolved_inputs/resolved_inputs.toml`,
and `resolved_inputs/provenance.json` to confirm what was actually used.

## 3. Study-definition handoff

Once track and vehicle evidence are current, the study owner reviews
`studies/*.toml`:

- the vehicle and baseline references;
- the design variable and tested domain;
- uncertainty sampling mode and roles;
- scenario count and random seed;
- decision thresholds;
- integration and numerical-quality limits.

The baseline is a nominal mechanism check. A final design recommendation requires
a design sweep whose numerical, evidence, and statistical readiness gates all
pass. Robustness, structural-sensitivity, and full-uncertainty studies answer
supporting questions; they do not independently choose a design.

## 4. Handoff completion record

Record these items with the project before launching large studies:

- name/date of the track-data review;
- GPX file list and run/vehicle/driver identities;
- disposition of every `must_fix` and `recommended_review` event;
- vehicle measurement references and dates;
- accepted inherited defaults, with reasons for accepting them;
- selected bundle path and SHA-256 checksum;
- study file and intended engineering question.

The Arizona example is intentionally not a completed handoff. Its warnings are
preserved so the next user can replace the example assumptions without mistaking
them for current measurements.
