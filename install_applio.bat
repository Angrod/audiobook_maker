@echo off
rem One-time setup for Tools > Train RVC Voice.
rem Downloads Applio 3.6.5 (https://github.com/IAHispano/Applio, MIT license) into modules\Applio
rem and runs its own installer, which creates a separate Python environment in modules\Applio\env
rem (about 6 GB). Audiobook Maker's own venv is not touched.
rem Already have Applio? Skip this and choose its folder in the Train RVC Voice window instead.
setlocal
cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
    echo git was not found. Install it from https://git-scm.com/ and run this again.
    pause
    exit /b 1
)

if exist "modules\Applio\core.py" (
    echo Applio is already downloaded in modules\Applio.
) else (
    echo Downloading Applio 3.6.5...
    git clone --depth 1 --branch 3.6.5 https://github.com/IAHispano/Applio.git modules\Applio
    if errorlevel 1 (
        echo Download failed. Check your internet connection and run this again.
        pause
        exit /b 1
    )
)

rem Applio's installer works in the current folder, so run it from inside modules\Applio
cd /d "%~dp0modules\Applio"
call run-install.bat
