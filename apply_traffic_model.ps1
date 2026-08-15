$ErrorActionPreference = "Stop"
python .\tools\apply_traffic_model.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m compileall -q .\src .\tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Traffic model applied and Python sources compiled."
