# Latent Market Twin — Daily Live Portfolio Value Task
# Uses schtasks.exe for Windows compatibility.

$ProjectRoot = "C:\ResearchCode\latent_market_twin"
$BatPath = Join-Path $ProjectRoot "run_daily_value_check.bat"
$TaskName = "LatentMarketTwinDailyValueCheck"

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
    Write-Host "Expected file: run_daily_value_check.bat"
    exit 1
}

Write-Host "Removing old task if it exists..."
schtasks /Delete /TN $TaskName /F | Out-Null

Write-Host "Creating daily scheduled task..."

# Runs every weekday at 4:30 PM local time.
# This is intended to run after the US market close.
$TaskRunCommand = "`"$BatPath`""

schtasks /Create `
    /TN $TaskName `
    /TR $TaskRunCommand `
    /SC WEEKLY `
    /D MON,TUE,WED,THU,FRI `
    /ST 16:30 `
    /RL LIMITED `
    /F

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to create scheduled task."
    Write-Host "Try running PowerShell as Administrator, then rerun:"
    Write-Host "powershell -ExecutionPolicy Bypass -File .\setup_daily_value_task.ps1"
    exit 1
}

Write-Host ""
Write-Host "Scheduled task created successfully."
Write-Host ""
Write-Host "It will run Monday-Friday at 4:30 PM."
Write-Host ""
Write-Host "To test it manually:"
Write-Host "schtasks /Run /TN $TaskName"
Write-Host ""
Write-Host "To check it:"
Write-Host "schtasks /Query /TN $TaskName /V /FO LIST"
Write-Host ""
Write-Host "To remove it:"
Write-Host "schtasks /Delete /TN $TaskName /F"