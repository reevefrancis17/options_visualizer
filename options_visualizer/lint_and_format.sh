#!/bin/bash

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

echo "Running Black formatter on Python code..."
black .

echo "Running Flake8 linter on Python code..."
flake8 .

# Check if we have Node.js installed for frontend linting
if command -v node &> /dev/null; then
  echo "Running ESLint on JavaScript code..."
  cd options_visualizer_web && npm run lint
  
  echo "Running Prettier on JavaScript and CSS code..."
  cd options_visualizer_web && npm run format
else
  echo "Node.js not found. Skipping JavaScript/CSS linting and formatting."
fi

echo "Linting and formatting complete!" 