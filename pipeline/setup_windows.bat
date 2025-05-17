@echo off
echo Setting up the Lung Nodule Detection Web Application for Windows...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in the PATH. Please install Python and try again.
    pause
    exit /b 1
)

REM Check if virtual environment exists, create if it doesn't
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Please install venv package: 'pip install virtualenv'
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install packages with pre-built wheels for Windows
echo Installing dependencies...
pip install flask==2.3.3 flask-cors==4.0.0 flask-jwt-extended==4.5.3 Werkzeug==2.3.7
pip install numpy==1.26.2
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
pip install SimpleITK==2.5.0 Pillow==10.0.0 tqdm==4.66.1
pip install matplotlib==3.7.2 scipy==1.11.2 --only-binary=matplotlib,scipy
pip install gunicorn==21.2.0

echo.
echo Dependencies installed successfully.
echo.

REM Run the verification script
echo Running verification script to check the required files and directories...
python verify_models.py

echo.
echo Setting up directory structure...
python setup_structure.py

echo.
echo Setup completed!
echo.
echo To run the application, use: python app.py
echo.

pause 