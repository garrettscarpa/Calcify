@echo off
REM Build Calcify.exe on Windows
REM Usage:  double-click this file, or run  build_windows.bat  from cmd

echo ==^> Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo ==^> Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ==^> Building Calcify with PyInstaller...
pyinstaller Calcify.spec --noconfirm --clean

echo.
echo ==^> Done!  Your app is at:  dist\Calcify\Calcify.exe
echo     Double-click Calcify.exe to run, or make a shortcut to it.
echo     To distribute, zip the entire dist\Calcify folder.
pause
