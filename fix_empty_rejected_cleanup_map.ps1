param(
    [string]$RepoRoot = "."
)

$cleanupPath = Join-Path $RepoRoot "src\cvt_track_study\gpx\cleanup.py"

if (-not (Test-Path $cleanupPath)) {
    throw "Could not find $cleanupPath. Run this from the cvt-study repository root or pass -RepoRoot."
}

$text = Get-Content $cleanupPath -Raw

$old = @'
    retained = retained_points.dropna(
        subset=["latitude_deg", "longitude_deg"]
    ).copy()
    rejected = rejected_points.dropna(
        subset=["latitude_deg", "longitude_deg"]
    ).copy()
'@

$new = @'
    coordinate_columns = ["latitude_deg", "longitude_deg"]

    retained = (
        retained_points.dropna(subset=coordinate_columns).copy()
        if set(coordinate_columns).issubset(retained_points.columns)
        else pd.DataFrame(columns=coordinate_columns)
    )
    rejected = (
        rejected_points.dropna(subset=coordinate_columns).copy()
        if set(coordinate_columns).issubset(rejected_points.columns)
        else pd.DataFrame(columns=coordinate_columns)
    )
'@

if (-not $text.Contains($old)) {
    if ($text.Contains('coordinate_columns = ["latitude_deg", "longitude_deg"]')) {
        Write-Host "Patch is already applied."
        exit 0
    }
    throw "Expected cleanup-map block was not found. The file differs from the version this patch targets."
}

$text = $text.Replace($old, $new)
Set-Content -Path $cleanupPath -Value $text -Encoding UTF8

Write-Host "Patched $cleanupPath"
Write-Host "Now rerun:"
Write-Host "  drivetrain-study ingest .\projects\arizona --run ets_78_maryland_reconstructed"
