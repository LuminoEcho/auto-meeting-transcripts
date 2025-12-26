@echo off
setlocal EnableExtensions

REM Change to the directory of this script (project root)
pushd "%~dp0" >nul 2>&1

REM -------------------------------
REM Settings (customize if desired)
REM -------------------------------
set "VENV_DIR=.venv"
set "PY_FILE=meetings_pipeline.py"

REM Flags
set "SKIP_INSTALL=0"
set "NO_PROMPT=0"
set "PASSTHRU="

REM Robust arg parsing (handles no-arg case)
:parse_args
if "%~1"=="" goto after_args
if /I "%~1"=="--skip-install" (
  set "SKIP_INSTALL=1"
  shift
  goto parse_args
)
if /I "%~1"=="--no-prompt" (
  set "NO_PROMPT=1"
  shift
  goto parse_args
)
set PASSTHRU=%PASSTHRU% "%~1"
shift
goto parse_args
:after_args

echo === Meetings Pipeline Launcher ===
echo Root: %CD%
echo Venv: %VENV_DIR%

REM -------------------------------
REM Locate Python launcher
REM -------------------------------
set "PY_CMD="
where py >nul 2>&1 && set "PY_CMD=py -3"
if not defined PY_CMD (
  where python >nul 2>&1 && set "PY_CMD=python"
)
if not defined PY_CMD (
  echo [ERROR] Python not found on PATH. Install Python 3.9+ and retry.
  exit /b 1
)

REM -------------------------------
REM Create venv if needed
REM -------------------------------
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating virtual environment in %VENV_DIR% ...
  %PY_CMD% -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
  )
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
if not exist "%VENV_PY%" (
  echo [ERROR] Virtual environment looks invalid: %VENV_PY% not found.
  exit /b 1
)

REM -------------------------------
REM Install/upgrade dependencies (unless skipped)
REM -------------------------------
if "%SKIP_INSTALL%"=="0" (
  echo Upgrading pip...
  "%VENV_PY%" -m pip install --upgrade pip

  if exist requirements.txt (
    echo Installing requirements.txt ...
    "%VENV_PIP%" install -r requirements.txt
  ) else (
    echo Installing base dependencies ...
    "%VENV_PIP%" install pyyaml watchdog google-genai openai-whisper pyannote.audio

    REM Try installing torch: prefer user-provided index if set
    if defined TORCH_INDEX_URL (
      echo Installing torch from %TORCH_INDEX_URL% ...
      "%VENV_PIP%" install torch --index-url "%TORCH_INDEX_URL%"
      if errorlevel 1 echo [WARN] torch install via TORCH_INDEX_URL failed. You may need to install manually.
    ) else (
    echo Installing CPU-only torch ^(override with TORCH_INDEX_URL if needed^) ...
      "%VENV_PIP%" install torch --index-url https://download.pytorch.org/whl/cpu
      if errorlevel 1 echo [WARN] torch CPU build install failed. See https://pytorch.org/get-started/locally/ for instructions.
    )
  )
)

REM -------------------------------
REM Quick FFmpeg check
REM -------------------------------
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo [WARN] ffmpeg not found on PATH. Install FFmpeg and ensure it is on PATH.
)

REM -------------------------------
REM Ensure input/output folders exist
REM -------------------------------
if not exist "input" mkdir "input" >nul 2>&1
if not exist "output" mkdir "output" >nul 2>&1

REM -------------------------------
REM Prompt for optional env vars
REM -------------------------------
if "%NO_PROMPT%"=="0" (
  if not defined GEMINI_API_KEY (
    set /p GEMINI_API_KEY=Enter GEMINI_API_KEY ^(leave blank to skip^): 
  )
  if not defined HF_TOKEN (
    set /p HF_TOKEN=Enter HF_TOKEN for diarization ^(leave blank to skip^): 
  )
)

REM Echo key config summary (obscure secrets)
set "_G=%GEMINI_API_KEY%"
if defined _G set "_G=***set***"
set "_H=%HF_TOKEN%"
if defined _H set "_H=***set***"
echo Env: GEMINI_API_KEY=%_G%  HF_TOKEN=%_H%
set "_G="
set "_H="

REM -------------------------------
REM Run the pipeline (forward remaining args)
REM -------------------------------
if not exist "%PY_FILE%" (
  echo [ERROR] %PY_FILE% not found in %CD%.
  exit /b 1
)

echo Launching: %PY_FILE% %PASSTHRU%
"%VENV_PY%" "%PY_FILE%" %PASSTHRU%
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
  echo [ERROR] meetings_pipeline exited with code %EXITCODE%.
)

popd >nul 2>&1
exit /b %EXITCODE%
