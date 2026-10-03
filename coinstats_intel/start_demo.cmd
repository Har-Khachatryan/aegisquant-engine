@echo off
rem One-click demo for the PRIVATE Cloud Run deployment.
rem Opens two authenticated tunnels (dashboard + API), waits until both services
rem have woken up (Cloud Run scales to zero, so the first request can take ~20 s),
rem then opens the browser. Requires: gcloud, logged in with an account that has
rem access to the project. Close the two tunnel windows when you are done.
setlocal
set "PROJECT=coinstats-intel-demo"
set "REGION=europe-west1"
if /i "%~1"=="tunnel" goto tunnel

echo Starting tunnels to the private Cloud Run services...
start "CoinStats DASHBOARD tunnel - keep this window open" cmd /c ""%~f0" tunnel coinstats-intel-dashboard 8502"
start "CoinStats API tunnel - keep this window open" cmd /c ""%~f0" tunnel coinstats-intel-api 8000"
echo Waiting for both services to wake up...
call :wait http://localhost:8502/_stcore/health
call :wait http://localhost:8000/health
start "" http://localhost:8502
start "" http://localhost:8000/docs
echo.
echo Ready:  dashboard http://localhost:8502   API http://localhost:8000/docs
echo Close the two tunnel windows when you are done.
timeout /t 10
exit /b 0

:wait
powershell -NoProfile -Command "for ($i = 0; $i -lt 60; $i++) { try { Invoke-WebRequest '%~1' -UseBasicParsing -TimeoutSec 10 | Out-Null; exit 0 } catch { Start-Sleep 2 } }; exit 1"
if errorlevel 1 echo WARNING: %~1 is not answering - look at the tunnel windows for errors.
exit /b 0

:tunnel
rem gcloud ends a proxy after ~55 minutes (identity-token lifetime): restart it automatically.
title CoinStats %2 tunnel - keep this window open
:loop
call gcloud run services proxy %2 --project %PROJECT% --region %REGION% --port %3
echo Tunnel stopped - restarting in 3 seconds. Close this window to stop it.
timeout /t 3 /nobreak >nul
goto loop
