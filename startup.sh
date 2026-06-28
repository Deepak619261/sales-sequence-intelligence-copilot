#!/usr/bin/env bash
set -e

echo "=== Starting Sales Copilot ==="
cd /home/site/wwwroot

# Try Oryx-built venv first.
if [ -d "/home/site/wwwroot/antenv" ]; then
    echo "Found Oryx virtual environment, activating..."
    . /home/site/wwwroot/antenv/bin/activate
elif [ -d "/home/site/wwwroot/.python_packages" ]; then
    echo "Found python packages directory..."
    export PYTHONPATH="/home/site/wwwroot/.python_packages/lib/site-packages:$PYTHONPATH"
else
    echo "No virtual environment found. Installing packages..."
    python -m venv antenv
    . antenv/bin/activate
    pip install --no-cache-dir -r requirements.txt
fi

echo "Starting uvicorn server on port ${PORT:-8000}..."
exec python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
