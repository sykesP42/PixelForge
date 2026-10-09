@echo off
chcp 65001 >nul
title PixelForge Studio
cd /d "%~dp0"

rem Python: conda base (project convention, do not change)
set PY=C:\Users\24311\anaconda3\python.exe

if not exist "%PY%" (
    echo [ERROR] Python not found: %PY%
    echo Please install Anaconda or edit the PY path in this file.
    pause
    exit /b 1
)

if not exist "%~dp0main.py" (
    echo [ERROR] main.py not found in %~dp0
    pause
    exit /b 1
)

echo Starting PixelForge Studio ...
"%PY%" main.py

if errorlevel 1 (
    echo.
    echo [ERROR] PixelForge exited with an error.
    echo Check logs\ and outputs\ for details, then press any key to close.
    pause >nul
)
