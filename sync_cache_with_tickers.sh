#!/bin/bash
# Sync cache with tickers.csv
# This script runs the update_cache.py script to ensure the cache only contains
# tickers from the tickers.csv file.

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project root directory
cd "$SCRIPT_DIR"

# Run the update_cache.py script
echo "Starting cache sync with tickers.csv at $(date)"
python3 update_cache.py

# Check the exit code
if [ $? -eq 0 ]; then
    echo "Cache sync completed successfully at $(date)"
    exit 0
else
    echo "Cache sync failed at $(date)"
    exit 1
fi 