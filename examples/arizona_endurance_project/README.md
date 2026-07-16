# McMaster Arizona endurance example

This is the primary real-data example for the clean GPX workflow. The GPX is the
same 6,822-point Arizona endurance recording previously represented by the CSV
prototype example, now used directly without a CSV conversion step. It includes
1 Hz timestamps and elevation at every point.

The 40 physical event definitions were migrated from the previously reviewed
cleaned event file. They form 37 response groups. Coordinates and endpoint
uncertainties are explicit; automatically sized point/turn extents are marked as
items for user review rather than presented as exact measurements.

Elevation is retained and plotted but is not yet converted to road grade or used
as a force in the vehicle simulation.

Run from the repository root:

```powershell
cvt-study validate .\examples\arizona_endurance_project
cvt-study ingest .\examples\arizona_endurance_project
cvt-study build-track .\examples\arizona_endurance_project
```
