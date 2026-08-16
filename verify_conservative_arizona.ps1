$ErrorActionPreference = "Stop"

Write-Host "=== Conservative Arizona gate-stack verification ==="
python -m pip install -e .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Targeted regression tests ==="
python -m pytest `
  .\tests\test_route_provenance_fallback.py `
  .\tests\test_performance_speedups.py `
  .\tests\test_design_paired_contrasts.py `
  .\tests\test_conservative_gate_hierarchy.py -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Strict Arizona validation ==="
drivetrain-study validate .\projects\arizona --strict
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nVerified source stack. Recommended next command:"
Write-Host "drivetrain-study run track-robustness .\projects\arizona --workers 8 --restart"
