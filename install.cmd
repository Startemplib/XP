@echo off
cd /d "%~dp0"
rem Install XP. No admin unless it truly needs it. Arguments are passed through
rem (-CheckOnly, -NoDeps, -WithGpu, -Uninstall).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
exit /b %errorlevel%
