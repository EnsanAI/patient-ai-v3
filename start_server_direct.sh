#!/bin/bash
# Alternative: Start server directly with uvicorn

cd "$(dirname "$0")"
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Kill any existing server
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 1

# Start directly with uvicorn (alternative to run.py)
echo "Starting server with uvicorn..."
uvicorn patient_ai_service.api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info





