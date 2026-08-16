$ErrorActionPreference = "Stop"

python .\tools\apply_traffic_model.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# V3 correction: there is no directly sourced Maryland traffic dataset in the
# evidence supplied for this project. Remove traffic files propagated by older
# overlays so Maryland uncertainty remains traffic-free until real evidence is added.
$marylandTrafficToml = ".\projects\maryland\track\traffic.toml"
$marylandTrafficDir = ".\projects\maryland\track\traffic"
if (Test-Path $marylandTrafficToml) { Remove-Item $marylandTrafficToml -Force }
if (Test-Path $marylandTrafficDir) { Remove-Item $marylandTrafficDir -Recurse -Force }


# V4 correction: Cornell is normal Arizona track telemetry, never traffic evidence.
$oldCornellTraffic = ".\projects\arizona\track\traffic\cornell_az_2025.csv"
if (Test-Path $oldCornellTraffic) { Remove-Item $oldCornellTraffic -Force }

python -m compileall -q .\src .\tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Traffic model v4 applied: Cornell is normal Arizona CSV telemetry for centreline + gates; traffic remains CWRU/ETS + Arizona field attrition only."
