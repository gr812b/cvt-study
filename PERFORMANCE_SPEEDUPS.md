# CVT-study performance-only transplant

This overlay extracts only the defensible runtime optimizations found in Max's
`maxChanges` snapshot. It deliberately does **not** import the traffic model,
project parameter changes, generated cache files, numerical-tolerance changes, or
README replacement.

## Included

1. **Process workers for CPU-bound scenario simulation**
   - replaces Python thread parallelism in `studies/service_v8.py`;
   - replaces Python thread parallelism in `studies/ensemble_v10.py`;
   - static vehicle/study/bundle context is initialized once per worker;
   - only small scenario objects are submitted per job;
   - checkpoint writes remain in the parent process;
   - worker-local cache counters are merged into parent reporting;
   - result rows are still sorted deterministically after execution.

2. **Prompt process-pool cancellation**
   - Ctrl+C/error paths cancel queued work and terminate active worker processes;
   - normal completion still performs orderly pool shutdown.

3. **Ordered feature-entry detection in the 1 ms integrator**
   - feature starts are sorted once;
   - the integrator advances one monotone index as distance increases;
   - exact feature-entry speed remains linearly interpolated at the boundary;
   - no timestep, solver, force, energy, CVT, tire, or obstacle equation changes.

## Why the ZIP uses a narrow transformer

The current working tree contains the route-family v11 overlay that is newer than
Max's branch. Replacing whole old source files would risk deleting those newer
changes. The included transformer edits only known execution blocks and refuses to
run if the source no longer matches the expected framework shape.

## Apply

Extract this ZIP directly over the repository root, then run:

```powershell
.\apply_performance_speedups.ps1
```

Then optionally run:

```powershell
python -m pytest .\tests\test_performance_speedups.py -q
```

Your normal command remains:

```powershell
drivetrain-study run full-uncertainty .\projects\maryland --workers 8
```

The run manifest will record `parallel_backend = "process"` when more than one
worker is active.

## Commit scope

After applying, the meaningful runtime changes are:

```text
src/cvt_track_study/runtime/process_pool.py
src/cvt_track_study/studies/service_v8.py
src/cvt_track_study/studies/ensemble_v10.py
src/cvt_track_study/simulation/integrator.py
tests/test_performance_speedups.py
PERFORMANCE_SPEEDUPS.md
```

The `tools/apply_performance_speedups.py` and PowerShell launcher are intentionally
included for audit/replay; keep or omit them from the final commit as you prefer.
