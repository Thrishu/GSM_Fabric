@echo off
REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt -q

REM Run Flask app
echo Starting Flask app...
python app.py
