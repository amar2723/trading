# Setup Windows Scheduled Tasks for Trading AI
$root = Resolve-Path "$PSScriptRoot\.."
$scannerBat = Join-Path $root "scripts\run_live_scanner.bat"
$retrainBat = Join-Path $root "scripts\run_retraining.bat"

Write-Host "Creating Windows Task Scheduler jobs for Trading AI..." -ForegroundColor Cyan

# 1. Scanner task (on user logon)
$actionScanner = New-ScheduledTaskAction -Execute $scannerBat
$triggerScanner = New-ScheduledTaskTrigger -AtLogOn
$principalScanner = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$settingsScanner = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName "TradingAI_LiveScanner" -Action $actionScanner -Trigger $triggerScanner -Principal $principalScanner -Settings $settingsScanner -Description "XAUUSD Trading AI Live Scanner" -Force
Write-Host " Registered 'TradingAI_LiveScanner' (starts automatically at login)" -ForegroundColor Green

# 2. Daily Model Retraining task (runs daily at 02:00 AM)
$actionRetrain = New-ScheduledTaskAction -Execute $retrainBat
$triggerRetrain = New-ScheduledTaskTrigger -Daily -At "02:00AM"
$principalRetrain = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$settingsRetrain = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName "TradingAI_DailyRetraining" -Action $actionRetrain -Trigger $triggerRetrain -Principal $principalRetrain -Settings $settingsRetrain -Description "XAUUSD Trading AI Daily Retraining" -Force
Write-Host " Registered 'TradingAI_DailyRetraining' (runs daily at 02:00 AM)" -ForegroundColor Green

Write-Host "`nAutomation tasks configured successfully!" -ForegroundColor Cyan
