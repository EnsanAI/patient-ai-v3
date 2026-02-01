#!/bin/bash
# Start the Patient AI Service with proper PYTHONPATH

cd "$(dirname "$0")"
source venv/bin/activate

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Kill any existing server on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 1

# Start the server
echo "Starting Patient AI Service..."
python run.py





