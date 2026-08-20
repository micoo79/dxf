@echo off
rem === PyShape inditasa Windowson ===
rem Elofeltetel: Python 3.10+ telepitve a python.org-rol
rem (telepiteskor pipald be: "Add python.exe to PATH")
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)

echo Szukseges csomagok telepitese (elso inditaskor par perc)...
%PY% -m pip install --quiet -r requirements.txt
if errorlevel 1 (
  echo.
  echo HIBA: nem sikerult a csomagok telepitese.
  echo Ellenorizd, hogy a Python telepitve van-e: https://www.python.org/downloads/
  pause
  exit /b 1
)

echo PyShape inditasa...
%PY% -m pyshape gui
pause
