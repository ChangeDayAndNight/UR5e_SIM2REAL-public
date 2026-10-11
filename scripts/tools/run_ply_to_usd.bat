@echo off
setlocal
set "REPO_ROOT=%~dp0..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

if not "%~1"=="" set "THREEDGRUT_ROOT=%~f1"
if not "%~2"=="" set "CUDA_TOOLKIT_ROOT=%~f2"
if not "%~3"=="" set "VCVARS64_BAT=%~f3"

if "%THREEDGRUT_ROOT%"=="" (
  echo [ply_to_usd][error] THREEDGRUT_ROOT is required and must point to a 3dgrut checkout.
  exit /b 2
)
if "%CUDA_TOOLKIT_ROOT%"=="" (
  echo [ply_to_usd][error] CUDA_TOOLKIT_ROOT is required.
  exit /b 2
)
if "%VCVARS64_BAT%"=="" (
  echo [ply_to_usd][error] VCVARS64_BAT is required.
  exit /b 2
)
if not exist "%VCVARS64_BAT%" (
  echo [ply_to_usd][error] Visual Studio environment script not found: %VCVARS64_BAT%
  exit /b 2
)
if not exist "%THREEDGRUT_ROOT%\.venv\Scripts\python.exe" (
  echo [ply_to_usd][error] 3dgrut virtualenv not found: %THREEDGRUT_ROOT%\.venv\Scripts\python.exe
  exit /b 2
)
if not exist "%REPO_ROOT%\assets\Scene\Workplace.ply" (
  echo [ply_to_usd][error] optional input missing: %REPO_ROOT%\assets\Scene\Workplace.ply
  exit /b 2
)

set "CUDA_PATH=%CUDA_TOOLKIT_ROOT%"
set "PATH=%THREEDGRUT_ROOT%\.venv\Lib\site-packages\slangtorch\bin;%CUDA_TOOLKIT_ROOT%\bin;%THREEDGRUT_ROOT%\.venv\Scripts;%PATH%"
set "INCLUDE=%CUDA_TOOLKIT_ROOT%\include;%CUDA_TOOLKIT_ROOT%\include\targets\x64;%INCLUDE%"
set "LIB=%CUDA_TOOLKIT_ROOT%\lib;%CUDA_TOOLKIT_ROOT%\lib\x64;%LIB%"
call "%VCVARS64_BAT%" >nul
if errorlevel 1 exit /b %ERRORLEVEL%
pushd "%THREEDGRUT_ROOT%"
"%THREEDGRUT_ROOT%\.venv\Scripts\python.exe" -u -m threedgrut.export.scripts.ply_to_usd "%REPO_ROOT%\assets\Scene\Workplace.ply" --output_file "%REPO_ROOT%\assets\Scene\Workplace.usdz"
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
