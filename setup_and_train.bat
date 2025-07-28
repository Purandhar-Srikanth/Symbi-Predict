@echo off
echo ===================================================
echo    SymbiPredict - Model Training Setup
echo ===================================================
echo.

REM Check if Python is installed
python --version > NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH.
    echo.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    echo After installing Python, run this script again.
    pause
    exit /b
)

echo Setting up virtual environment...
REM Create virtual environment if it doesn't exist
if not exist venv (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Installing required packages...
pip install pandas numpy scikit-learn flask

echo.
echo ===================================================
echo Training the model and creating pickle files...
echo ===================================================

REM Create the models directory if it doesn't exist
if not exist models mkdir models

REM Run the training script
python train_logistic_regression.py

echo.
echo ===================================================
echo Setup complete!
echo ===================================================
echo.

if exist models\logistic_regression_model.pkl (
    echo Model pickle file created successfully at:
    echo models\logistic_regression_model.pkl
    echo.
    echo Symptoms list pickle file created at:
    echo models\symptoms_list.pkl
    echo.
    echo You can now run the Flask API using:
    echo python api.py
) else (
    echo Error: Model pickle file was not created.
    echo Please check the error messages above.
)

pause
