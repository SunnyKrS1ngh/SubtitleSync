@echo off
REM =========================================================================
REM SubtitleSync launcher
REM =========================================================================

setlocal

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "MARKER=%VENV%\.installed"
set "REQUIREMENTS=%ROOT%requirements.txt"

:: Locate Python 3.10+
set "PY_CMD="

where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys; v=sys.version_info; sys.exit(0 if v.major==3 and v.minor>=10 else 1)" >nul 2>&1
    if not errorlevel 1 set "PY_CMD=py -3"
)

if not defined PY_CMD (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; v=sys.version_info; sys.exit(0 if v.major==3 and v.minor>=10 else 1)" >nul 2>&1
        if not errorlevel 1 set "PY_CMD=python"
    )
)

if not defined PY_CMD (
    echo Python 3.10+ was not found on your system.
    echo Please install Python 3.10 or newer and make sure it is added to your PATH.
    pause
    exit /b 1
)

:: If already installed and virtual environment exists, skip straight to launch
if exist "%MARKER%" (
    if exist "%PYTHON%" goto :start_app
)

echo First run - setting up SubtitleSync...
echo.

echo [1/5] Creating Python environment...
%PY_CMD% -m venv "%VENV%"
if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/5] Upgrading pip...
"%PYTHON%" -m pip install --upgrade pip >nul 2>&1

echo [3/5] Installing dependencies...
where uv >nul 2>&1
if not errorlevel 1 (
    echo Using uv for fast installation...
    uv pip install --python "%PYTHON%" -r "%REQUIREMENTS%"
) else (
    echo Installing components with pip - this may take a few minutes...
    "%PYTHON%" -m pip install -r "%REQUIREMENTS%"
)
if errorlevel 1 (
    echo Installation failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo [4/5] Downloading speech model (~141 MB, one time)...
"%PYTHON%" -c "from modules.transcribe import WhisperSession; WhisperSession('base', 'auto', None); print('Model ready.')"
if errorlevel 1 (
    echo Model download failed. The app will try again on first use.
)

echo.>"%MARKER%"
echo [5/5] Setup complete.
echo.

:start_app
echo Starting SubtitleSync...
echo.
echo A browser window will open. Leave this window open while you work.
echo Press Ctrl+C here to stop the app.
echo.

"%PYTHON%" "%ROOT%app.py"

if errorlevel 1 (
    echo.
    echo The app stopped with an error.
    pause
)