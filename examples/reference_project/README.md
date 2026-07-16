# Synthetic reference project

This project is a deterministic integration fixture for Phases 1–3. It is not a
measured Baja course and its physical values are not calibration data.

The GPX contains one track segment with 1,081 valid points and nine complete laps
of a roughly 600 m compact closed course. Synthetic elevation is included to
exercise preservation and profile plotting; grade force remains disabled.

Expected Phase 3 behavior:

- nine complete, valid laps;
- centreline length near 600 m;
- three accepted gate candidates (`turn_north`, `logs_west`, `ruts_south`);
- `start_finish` used only to separate laps, not as a speed gate;
- physical intervals and their uncertainty visible in the review outputs.

Run from the repository root after installation:

```powershell
cvt-study validate .\examples\reference_project
cvt-study ingest .\examples\reference_project
cvt-study build-track .\examples\reference_project
```
