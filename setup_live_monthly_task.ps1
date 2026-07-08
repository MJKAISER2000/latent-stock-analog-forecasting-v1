# Latent Market Twin — Live Monthly Rebuild Task
# Runs the full live data refresh + feature rebuild + live signal generation pipeline.

$ProjectRoot = $PSScriptRoot
$BatPath = Join-Path $ProjectRoot "run_live_rebuild.bat"
$TaskName = "LatentMarketTwinLiveMonthlyRebuild"

Write-Host ""
Write-Host "Setting up scheduled task:"
Write-Host "Task name: $TaskName"
Write-Host "Project root: $ProjectRoot"
Write-Host "Batch path: $BatPath"
Write-Host ""

if (!(Test-Path $ProjectRoot)) {
    Write-Host "ERROR: Project root not found: $ProjectRoot"
    exit 1
}

if (!(Test-Path $BatPath)) {
    Write-Host "ERROR: Batch file not found: $BatPath"
    Write-Host "Expected file: run_live_rebuild.bat"
    exit 1
}

Write-Host "Removing old task if it exists..."
schtasks /Delete /TN $TaskName /F | Out-Null

Write-Host "Creating live monthly rebuild scheduled task..."

# Runs monthly on the 2nd day at 9:30 AM.
# This gives market data a little time to settle after month-end.
$TaskRunCommand = "`"$BatPath`""

schtasks /Create `
    /TN $TaskName `
    /TR $TaskRunCommand `
    /SC MONTHLY `
    /D 2 `
    /ST 09:30 `
    /RL LIMITED `
    /F

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to create scheduled task."
    Write-Host "Try running PowerShell as Administrator, then rerun:"
    Write-Host "powershell -ExecutionPolicy Bypass -File .\setup_live_monthly_task.ps1"
    exit 1
}

Write-Host ""
Write-Host "Scheduled task created successfully."
Write-Host ""
Write-Host "It will run monthly on the 2nd day of each month at 9:30 AM."
Write-Host ""
Write-Host "To test it manually:"
Write-Host "schtasks /Run /TN $TaskName"
Write-Host ""
Write-Host "To check it:"
Write-Host "schtasks /Query /TN $TaskName /V /FO LIST"
Write-Host ""
Write-Host "To remove it:"
Write-Host "schtasks /Delete /TN $TaskName /F"