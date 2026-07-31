@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Refuse elevated Administrator (optional soft warning)
net session >nul 2>&1
if %errorlevel%==0 (
  echo Error: Do not run this script as Administrator.
  echo Example:  install.bat
  echo           python install.py
  echo (To override: python install.py --allow-root)
  exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
  where py >nul 2>&1
  if errorlevel 1 (
    echo Python not found. Install Python 3.10+ from https://www.python.org/downloads/
    echo and check "Add python.exe to PATH".
    exit /b 1
  )
  py -3 install.py %*
  exit /b %errorlevel%
)

python install.py %*
exit /b %errorlevel%
