# Example projects

## `arizona_endurance_project`

The primary user-facing example. It uses a real McMaster Baja Arizona endurance
GPX recording and the 40 reviewed physical-event definitions from the earlier
prototype workflow. The events resolve into 37 response groups. Running the
Phase 3 build currently finds 13 complete laps, retains 11 evidence-quality laps,
and creates a roughly 1.77 km centreline.

The raw GPX is intentionally preserved exactly as supplied. It contains one
duplicate timestamp step, which is reported as a warning. It includes elevation
at every point but no reported speed field; analysis speed is therefore derived
from position and time. Elevation remains review-only and does not create a grade
force.

## `reference_project`

A small deterministic synthetic project used by automated regression tests. It
is not measurement or calibration data.
