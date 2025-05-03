#!/bin/bash

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Run unit tests with coverage
echo "Running unit tests with coverage..."
python -m pytest tests/test_black_scholes.py tests/test_options_data.py -v --cov=backend.models --cov=python --cov-report=term --cov-report=html

# Run API tests
echo "Running API tests..."
python -m pytest tests/test_api.py -v

# Run integration tests
echo "Running integration tests..."
python -m pytest tests/test_cache_manager.py -v

# Run E2E tests if --all flag is passed
if [[ "$1" == "--all" ]]; then
    echo "Running E2E tests... (requires running server)"
    echo "Make sure you have servers running:"
    echo "  Terminal 1: python -m backend.app"
    python -m pytest tests/test_e2e.py -v
else
    echo "Skipping E2E tests. Use --all flag to run them."
    echo "Note: E2E tests require running servers:"
    echo "  Terminal 1: python -m backend.app"
fi

# Print coverage report summary
echo "Coverage report summary:"
coverage report --fail-under=90

echo "All tests completed!" 