@echo off
setlocal
if defined ISAAC_WHIP_PYTHON (
  set "ISAAC_WHIP_PYTHON_RESOLVED=%ISAAC_WHIP_PYTHON%"
) else (
  set "ISAAC_WHIP_PYTHON_RESOLVED=%USERPROFILE%\env_isaaclab\Scripts\python.exe"
)
if not exist "%ISAAC_WHIP_PYTHON_RESOLVED%" (
  echo Isaac Lab Python was not found at "%ISAAC_WHIP_PYTHON_RESOLVED%".
  echo Install it under "%%USERPROFILE%%\env_isaaclab" or set ISAAC_WHIP_PYTHON.
  exit /b 1
)
pushd "%~dp0"
"%ISAAC_WHIP_PYTHON_RESOLVED%" -m isaac_whip.launcher %*
set "ISAAC_WHIP_EXIT=%ERRORLEVEL%"
popd
endlocal & exit /b %ISAAC_WHIP_EXIT%
