param(
    [string]$RepoRoot = "."
)

$ErrorActionPreference = "Stop"
$dropInRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path $RepoRoot).Path

if (-not (Test-Path (Join-Path $repo "pyproject.toml"))) {
    throw "RepoRoot does not look like the cvt-study repository: $repo"
}

$required = @(
    "src\cvt_track_study\studies\planning.py",
    "src\cvt_track_study\studies\service_v8.py",
    "src\cvt_track_study\studies\ensemble_v10.py",
    "src\cvt_track_study\config\validation.py",
    "src\cvt_track_study\reports\postprocess.py"
)
foreach ($relative in $required) {
    if (-not (Test-Path (Join-Path $repo $relative))) {
        throw "Required repository file is missing: $relative"
    }
}

$stamp = Get-Date -Format "yyyyMMddTHHmmss"
$backup = Join-Path $repo ".dropin-backups\multivariable-design-grid-$stamp"
New-Item -ItemType Directory -Force -Path $backup | Out-Null

$affected = @(
    "src\cvt_track_study\studies\planning.py",
    "src\cvt_track_study\studies\service_v8.py",
    "src\cvt_track_study\studies\ensemble_v10.py",
    "src\cvt_track_study\config\validation.py",
    "src\cvt_track_study\reports\postprocess.py",
    "src\cvt_track_study\studies\design_grid.py",
    "src\cvt_track_study\studies\design_replay_v11.py",
    "src\cvt_track_study\studies\scenario_reduction.py",
    "src\cvt_track_study\studies\router_v10.py",
    "tests\test_multivariable_design_grid.py",
    "docs\MULTIVARIABLE_DESIGN_GRIDS.md",
    "projects\arizona\studies\cvt_range_final_drive_sweep.toml"
)

foreach ($relative in $affected) {
    $existing = Join-Path $repo $relative
    if (Test-Path $existing) {
        $backupTarget = Join-Path $backup $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $backupTarget) | Out-Null
        Copy-Item $existing $backupTarget -Force
    }
}

$overlay = @(
    "src\cvt_track_study\studies\design_grid.py",
    "src\cvt_track_study\studies\design_replay_v11.py",
    "src\cvt_track_study\studies\scenario_reduction.py",
    "src\cvt_track_study\studies\router_v10.py",
    "tests\test_multivariable_design_grid.py",
    "docs\MULTIVARIABLE_DESIGN_GRIDS.md",
    "projects\arizona\studies\cvt_range_final_drive_sweep.toml"
)
foreach ($relative in $overlay) {
    $source = Join-Path $dropInRoot $relative
    if (-not (Test-Path $source)) {
        throw "Drop-in file missing: $source"
    }
    $destination = Join-Path $repo $relative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item $source $destination -Force
}

$pythonCommand = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $pythonCommand) {
    $pythonCommand = Get-Command python -ErrorAction Stop
}
$python = $pythonCommand.Source

Push-Location $repo
try {
    & $python (Join-Path $dropInRoot "patch_multivariable_design_grid.py") --repo $repo
    & $python (Join-Path $dropInRoot "patch_black_errorbars.py") --repo-root $repo
    & $python -m compileall -q "src\cvt_track_study"
    & $python -m pip install -e ".[dev]"
    $testPaths = @(
        "tests\test_multivariable_design_grid.py",
        "tests\test_report_postprocessing.py"
    )
    if (Test-Path "tests\test_uncertainty_informed_design.py") {
        $testPaths += "tests\test_uncertainty_informed_design.py"
    }
    & $python -m pytest -q @testPaths
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Multi-variable design-grid support applied."
Write-Host "Backup: $backup"
Write-Host ""
Write-Host "Validate:"
Write-Host "drivetrain-study validate .\projects\arizona --study cvt_range_final_drive_sweep"
Write-Host ""
Write-Host "Run:"
Write-Host "drivetrain-study run design-comparison .\projects\arizona --study cvt_range_final_drive_sweep --workers 6 --restart --run-name arizona-cvt-range-final-drive"
