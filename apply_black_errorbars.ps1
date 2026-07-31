param(
    [Parameter(Mandatory = $false)]
    [string]$RepoRoot = "."
)

$ErrorActionPreference = "Stop"
$DropInRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path $RepoRoot).Path

$Python = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { throw "Python was not found." }

Write-Host "Patching Matplotlib error-bar contrast across the report source tree..."
& $Python "$DropInRoot\patch_black_errorbars.py" --repo-root "$RepoRoot"

$TestTarget = Join-Path $RepoRoot "tests\test_report_errorbar_contrast.py"
Copy-Item "$DropInRoot\tests\test_report_errorbar_contrast.py" $TestTarget -Force

Write-Host "Compiling source..."
& $Python -m compileall -q (Join-Path $RepoRoot "src")

Write-Host "Running contrast regression test..."
& $Python -m pytest -q $TestTarget

Write-Host "Done. Existing reports can be regenerated from saved artifacts without rerunning simulations:"
Write-Host "  drivetrain-study report <RESULT_DIRECTORY>"
