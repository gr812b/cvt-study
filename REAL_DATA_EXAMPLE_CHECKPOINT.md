# Real-data example checkpoint

The supplied McMaster Arizona endurance GPX has been added as the primary
user-facing example under `examples/arizona_endurance_project`.

## Source check

- GPX 1.1
- one track and one segment
- 6,822 track points
- timestamps at every point
- elevation at every point
- no reported speed field
- one duplicate timestamp step, retained and reported
- latitude/longitude/timestamps match the prior CSV prototype recording

## Phase 3 smoke result

- validation: 0 errors, 0 warnings
- ingestion: 6,822 valid points
- complete laps: 13
- valid evidence laps: 11
- reconstructed centreline: 1,773.6 m
- physical events: 40
- response groups: 37
- accepted speed gates: 13
- recommended review: 16
- must-fix geometry reviews: 5

The review findings are intentionally not edited away. The real example now
demonstrates the actual user workflow: ingest raw evidence, inspect warnings,
review map/event placement, and only then freeze a simulation-ready bundle.
