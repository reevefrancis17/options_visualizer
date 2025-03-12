#!/bin/bash

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Run unit tests with coverage
echo "Running unit tests with coverage..."
python -m pytest tests/test_black_scholes.py tests/test_options_data.py -v --cov=options_visualizer_backend.models --cov=python --cov-report=term --cov-report=html

# Run integration tests
echo "Running integration tests..."
python -m pytest tests/test_api.py -v

# Run end-to-end tests (only if both servers are running)
echo "Checking if servers are running for E2E tests..."
if curl -s http://localhost:5001/health > /dev/null && curl -s http://localhost:5002/health > /dev/null; then
  echo "Both servers are running. Running end-to-end tests..."
  python -m pytest tests/test_e2e.py -v
else
  echo "Servers are not running. Skipping end-to-end tests."
  echo "To run E2E tests, start both servers in separate terminals:"
  echo "  Terminal 1: python -m options_visualizer_backend.app"
  echo "  Terminal 2: python -m options_visualizer_web.app"
  echo "Then run: python -m pytest tests/test_e2e.py -v"
fi

# Print coverage report summary
echo "Coverage report summary:"
coverage report --fail-under=90

echo "Tests completed!" 