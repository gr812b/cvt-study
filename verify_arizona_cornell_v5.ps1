$ErrorActionPreference = "Stop"

Write-Host "=== Arizona Cornell V5 verification ==="
Write-Host "Refreshing editable install..."
python -m pip install -e .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$staleCornellTraffic = ".\projects\arizona\track\traffic\cornell_az_2025.csv"
if (Test-Path $staleCornellTraffic) {
    Write-Host "Removing stale V4 Cornell traffic artifact..."
    Remove-Item $staleCornellTraffic -Force
}

Write-Host "`n=== Validate Arizona ==="
drivetrain-study validate .\projects\arizona
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== CSV telemetry regression tests ==="
python -m pytest .\tests\test_csv_telemetry.py .\tests\test_arizona_cornell_project.py -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Build Arizona track ==="
drivetrain-study build-track .\projects\arizona
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$latest = Get-ChildItem ".\projects\arizona\results\track_build" -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $latest) {
    throw "build-track completed but no track_build result directory was found."
}

$zip = Join-Path $latest.Parent.FullName ("arizona-cornell-v5-" + $latest.Name + ".zip")
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $latest.FullName "*") -DestinationPath $zip -Force

Write-Host "`nSUCCESS"
Write-Host ("Track build: " + $latest.FullName)
Write-Host ("Shareable ZIP: " + $zip)
