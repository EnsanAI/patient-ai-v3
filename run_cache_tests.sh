#!/bin/bash
# Script to run cache-related tests with proper environment setup

# Change to project directory
cd /Users/omar/Downloads/The\ Future/carebot_dev/patient-ai-v3

# Set PYTHONPATH to include src and venv packages
export PYTHONPATH="src:venv/lib/python3.13/site-packages:$PYTHONPATH"

# Use Python 3.13 from venv
PYTHON_CMD="python3.13"

# Run tests based on argument
if [ "$1" == "all" ]; then
    echo "Running all cache tests..."
    $PYTHON_CMD -m pytest tests/unit/test_token_usage_cache.py \
        tests/unit/test_cost_calculator_cache.py \
        tests/unit/test_prompt_cache.py \
        tests/integration/test_think_caching.py \
        tests/integration/test_llm_client_cache.py \
        tests/performance/test_cache_performance.py \
        -v
elif [ "$1" == "unit" ]; then
    echo "Running unit tests..."
    $PYTHON_CMD -m pytest tests/unit/test_token_usage_cache.py \
        tests/unit/test_cost_calculator_cache.py \
        tests/unit/test_prompt_cache.py \
        -v
elif [ "$1" == "integration" ]; then
    echo "Running integration tests..."
    $PYTHON_CMD -m pytest tests/integration/test_think_caching.py \
        tests/integration/test_llm_client_cache.py \
        -v
elif [ "$1" == "performance" ]; then
    echo "Running performance tests..."
    $PYTHON_CMD -m pytest tests/performance/test_cache_performance.py -v -s
else
    echo "Usage: $0 [all|unit|integration|performance]"
    echo ""
    echo "Examples:"
    echo "  $0 all          # Run all cache tests"
    echo "  $0 unit         # Run only unit tests"
    echo "  $0 integration  # Run only integration tests"
    echo "  $0 performance  # Run only performance tests (requires API key)"
    exit 1
fi







