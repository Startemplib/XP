@echo off
cd /d "%~dp0"
rem Remove what install.cmd added: the Explorer actions, the shortcuts and
rem the keys they created. None of your own documents are touched.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Uninstall %*
exit /b %errorlevel%
