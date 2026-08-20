@echo off
rem === PyShape inditasa Windowson ===
rem Megfelelo Python keresese: 3.10+ NORMAL (nem "free-threaded") valtozat,
rem tkinterrel. A 3.13t/3.14t jelu free-threaded Pythonnal a numpy/opencv
rem nem mukodik, ezert azt kihagyjuk.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "CHECK=import sys,sysconfig;assert sys.version_info>=(3,10);assert not sysconfig.get_config_var('Py_GIL_DISABLED');import tkinter"
set "PYEXE="

for %%V in (-3.12 -3.11 -3.13 -3.10 -3.14) do (
  if not defined PYEXE (
    py %%V -c "%CHECK%" >nul 2>nul
    if !errorlevel!==0 set "PYEXE=py %%V"
  )
)
if not defined PYEXE (
  python -c "%CHECK%" >nul 2>nul
  if !errorlevel!==0 set "PYEXE=python"
)

if not defined PYEXE (
  echo.
  echo ============================================================
  echo  Nem talaltam megfelelo Pythont ezen a gepen.
  echo.
  echo  Megoldas: telepitsd a normal Python 3.12-t innen:
  echo    https://www.python.org/downloads/release/python-31210/
  echo  ...lent a "Windows installer 64-bit" linket valaszd.
  echo.
  echo  Telepiteskor pipald be: "Add python.exe to PATH"
  echo  FONTOS: a "free-threaded binaries" opciot NE valaszd.
  echo  Utana inditsd ujra ezt a fajlt.
  echo ============================================================
  pause
  exit /b 1
)

echo Hasznalt Python: %PYEXE%
%PYEXE% -c "import sys; print(sys.version)"

rem Gyors ellenorzes: megvannak-e a csomagok ES epek-e a binaris moduljaik.
rem Ha egy korabbi hibas Python-valtozat felkesz csomagokat hagyott hatra,
rem a force-reinstall kijavitja oket.
set "IMPCHECK=import numpy,scipy,cv2,PIL,rasterio,pyproj,pyproj.network,tkinter"
%PYEXE% -c "%IMPCHECK%" >nul 2>nul
if errorlevel 1 (
  echo Szukseges csomagok telepitese / javitasa - par perc, kerlek varj...
  %PYEXE% -m pip install --upgrade --force-reinstall -r requirements.txt
  %PYEXE% -c "%IMPCHECK%"
  if errorlevel 1 (
    echo.
    echo HIBA: a csomagok ujratelepites utan sem toltodnek be.
    echo Masold be a fenti hibauzenetet Claude-nak, es megoldja.
    pause
    exit /b 1
  )
) else (
  echo Csomagok rendben.
)

echo PyShape inditasa...
%PYEXE% -m pyshape gui
if errorlevel 1 pause
