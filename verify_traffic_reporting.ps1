param(
    [string]$ResultDirectory = ""
)

$ErrorActionPreference = "Stop"
python -m pytest .\tests\test_traffic_reporting.py -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($ResultDirectory -ne "") {
    drivetrain-study report $ResultDirectory
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
