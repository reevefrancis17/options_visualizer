#!/bin/bash
# Setup cron job to sync cache with tickers.csv daily
# This script sets up a cron job to run the sync_cache_with_tickers.sh script daily at 1:00 AM

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Full path to the sync script
SYNC_SCRIPT="$SCRIPT_DIR/sync_cache_with_tickers.sh"

# Full path to the log file
LOG_FILE="$SCRIPT_DIR/logs/cache_sync.log"

# Create logs directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/logs"

# Create a temporary file for the new crontab
TEMP_CRONTAB=$(mktemp)

# Export current crontab
crontab -l > "$TEMP_CRONTAB" 2>/dev/null || echo "# Options Visualizer cron jobs" > "$TEMP_CRONTAB"

# Check if the cron job already exists
if grep -q "$SYNC_SCRIPT" "$TEMP_CRONTAB"; then
    echo "Cron job already exists. Updating..."
    # Remove existing cron job
    grep -v "$SYNC_SCRIPT" "$TEMP_CRONTAB" > "${TEMP_CRONTAB}.new"
    mv "${TEMP_CRONTAB}.new" "$TEMP_CRONTAB"
else
    echo "Adding new cron job..."
fi

# Add the new cron job (run daily at 1:00 AM)
echo "0 1 * * * $SYNC_SCRIPT >> $LOG_FILE 2>&1" >> "$TEMP_CRONTAB"

# Install the new crontab
crontab "$TEMP_CRONTAB"

# Clean up
rm "$TEMP_CRONTAB"

echo "Cron job installed successfully. The cache will be synced with tickers.csv daily at 1:00 AM."
echo "Logs will be written to $LOG_FILE" 