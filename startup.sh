#!/bin/bash
echo "=== Starting Sales Copilot ==="

# Try Oryx-built venv first
if [ -d "/home/site/wwwroot/antenv" ]; then
    echo "Found Oryx virtual environment, activating..."
    source /home/site/wwwroot/antenv/bin/activate
elif [ -d "/home/site/wwwroot/.python_packages" ]; then
    echo "Found python packages directory..."
    export PYTHONPATH="/home/site/wwwroot/.python_packages/lib/site-packages:$PYTHONPATH"
else
    echo "No virtual environment found. Installing packages..."
    cd /home/site/wwwroot
    python -m venv antenv
    source antenv/bin/activate
    pip install --no-cache-dir -r requirements.txt
fi

echo "Starting uvicorn server..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000
