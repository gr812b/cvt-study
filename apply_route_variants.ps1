param(
    [string]$RepoRoot = "."
)

$ErrorActionPreference = "Stop"
$dropInRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path $RepoRoot).Path

$paths = @(
    "src\cvt_track_study\config\__init__.py",
    "src\cvt_track_study\config\project_v12.py",
    "src\cvt_track_study\track\export.py",
    "src\cvt_track_study\track\model.py",
    "src\cvt_track_study\track\reconstruction.py",
    "src\cvt_track_study\track\route_variants.py",
    "src\cvt_track_study\track\service.py",
    "src\cvt_track_study\project_template\track\track.toml",
    "tests\test_route_variants.py",
    "docs\ROUTE_VARIANTS.md",
    "docs\MARYLAND_REVIEW_QUESTIONS.md"
)

if (-not (Test-Path (Join-Path $repo "pyproject.toml"))) {
    throw "RepoRoot does not look like the cvt-study repository: $repo"
}

$stamp = Get-Date -Format "yyyyMMddTHHmmss"
$backup = Join-Path $repo ".route-variant-backup-$stamp"
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
    py -m pytest -q tests/test_route_variants.py
}
finally {
    Pop-Location
}

Write-Host "Route-variant drop-in applied. Backup: $backup"
