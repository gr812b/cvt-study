param(
    [string]$RepoRoot = "."
)

$ErrorActionPreference = "Stop"
$dropInRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path $RepoRoot).Path

if (-not (Test-Path (Join-Path $repo "pyproject.toml"))) {
    throw "RepoRoot does not look like the cvt-study repository: $repo"
}

$paths = @(
    "src\cvt_track_study\studies\scenario_reduction.py",
    "src\cvt_track_study\studies\design_replay_v11.py",
    "src\cvt_track_study\studies\router_v10.py",
    "tests\test_uncertainty_informed_design.py",
    "docs\UNCERTAINTY_INFORMED_DESIGN.md",
    "projects\arizona\studies\gearing_sweep.toml"
)

$stamp = Get-Date -Format "yyyyMMddTHHmmss"
$backup = Join-Path $repo ".uncertainty-informed-design-backup-$stamp"
New-Item -ItemType Directory -Force -Path $backup | Out-Null

foreach ($relative in $paths) {
    $source = Join-Path $dropInRoot $relative
    if (-not (Test-Path $source)) {
        throw "Drop-in file missing: $source"
    }
    $destination = Join-Path $repo $relative
    if (Test-Path $destination) {
        $backupTarget = Join-Path $backup $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $backupTarget) | Out-Null
        Copy-Item $destination $backupTarget -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item $source $destination -Force
}

Push-Location $repo
try {
    py -m pip install -e ".[dev]"
    py -m pytest -q tests/test_uncertainty_informed_design.py
}
finally {
    Pop-Location
}

Write-Host "Uncertainty-informed design drop-in applied. Backup: $backup"
Write-Host "After the full-uncertainty result is approved, run:"
Write-Host "drivetrain-study run design-comparison .\projects\arizona --study final_drive_sweep --workers 6 --restart"
