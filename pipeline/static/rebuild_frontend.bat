@echo off
echo Rebuilding the Lung Nodule Detection React frontend...
echo.

REM Check if npm is installed
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: npm is not installed or not in the PATH.
    echo Please install Node.js and npm from https://nodejs.org/
    pause
    exit /b 1
)

REM Navigate to the static directory
cd /d "%~dp0"

REM Install required dependencies if needed
echo Installing required dependencies...
call npm install --save tailwindcss postcss autoprefixer
call npx tailwindcss init -p

REM Build the project
echo Building the project...
call npm run build

echo.
if %errorlevel% equ 0 (
    echo Frontend rebuilt successfully!
    echo.
    echo The build files are located in the 'build' directory.
    echo Copy these files to your Flask app's static directory for deployment.
) else (
    echo Error: Build failed. Please check the error messages above.
)

pause 