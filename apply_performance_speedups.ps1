$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot
try {
    python .\tools\apply_performance_speedups.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python -m compileall -q .\src\cvt_track_study
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Speedups applied and Python compilation passed."
} finally {
    Pop-Location
}
