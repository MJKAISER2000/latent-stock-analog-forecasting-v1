# Latent Market Twin — Monthly Task Scheduler Setup
# Uses schtasks.exe because some PowerShell versions do not support New-ScheduledTaskTrigger -Monthly.

$ProjectRoot = "C:\ResearchCode\latent_market_twin"
$BatPath = Join-Path $ProjectRoot "run_monthly_update.bat"
$TaskName = "LatentMarketTwinMonthlyUpdate"

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
    Write-Host "Expected file: run_monthly_update.bat"
    exit 1
}

# Delete old task if it exists.
Write-Host "Removing old task if it exists..."
schtasks /Delete /TN $TaskName /F | Out-Null

# Create monthly task.
# Runs on the 1st day of every month at 9:00 AM.
Write-Host "Creating monthly scheduled task..."

$TaskRunCommand = "`"$BatPath`""

schtasks /Create `
    /TN $TaskName `
    /TR $TaskRunCommand `
    /SC MONTHLY `
    /D 1 `
    /ST 09:00 `
    /RL LIMITED `
    /F

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to create scheduled task."
    Write-Host "Try running PowerShell as Administrator, then rerun:"
    Write-Host "powershell -ExecutionPolicy Bypass -File .\setup_monthly_task.ps1"
    exit 1
}

Write-Host ""
Write-Host "Scheduled task created successfully."
Write-Host ""
Write-Host "It will run monthly on the 1st day of each month at 9:00 AM."
Write-Host ""
Write-Host "To test it manually:"
Write-Host "schtasks /Run /TN $TaskName"
Write-Host ""
Write-Host "To check it:"
Write-Host "schtasks /Query /TN $TaskName /V /FO LIST"
Write-Host ""
Write-Host "To remove it:"
Write-Host "schtasks /Delete /TN $TaskName /F"