# Cache Management for Options Visualizer

This document explains how to manage the options data cache using the provided scripts.

## Overview

The Options Visualizer application caches options data to improve performance and reduce API calls. The cache is managed by the `OptionsCache` class in `python/cache_manager.py`.

The scripts in this directory allow you to:

1. Sync the cache with the tickers in `data/tickers.csv`
2. Set up a cron job to automatically sync the cache daily

## Tickers CSV File

The `data/tickers.csv` file contains the list of tickers that should be cached. The file has the following format:

```csv
ticker,timestamp,ticker_exists,chain_exists
SPY,2025-03-14T17:30:03.819014,True,True
AAPL,2025-03-13T22:00:00,True,True
...
```

To add a new ticker to the cache, simply add it to this file. To remove a ticker from the cache, remove it from this file and run the sync script.

## Sync Script

The `sync_cache_with_tickers.sh` script syncs the cache with the tickers in `data/tickers.csv`. It:

1. Removes tickers from the cache that are not in the CSV file
2. Updates the registry for tickers that are in both the cache and the CSV file
3. Starts fetching data for tickers that are in the CSV file but not in the cache

To run the script:

```bash
./sync_cache_with_tickers.sh
```

## Cron Job Setup

The `setup_cron.sh` script sets up a cron job to run the sync script daily at 1:00 AM. This ensures that the cache is always in sync with the tickers in the CSV file.

To set up the cron job:

```bash
./setup_cron.sh
```

The cron job will write logs to `logs/cache_sync.log`.

## Manual Cache Management

If you need to manually manage the cache, you can use the `update_cache.py` script directly:

```bash
python3 update_cache.py
```

This script provides more detailed logging and can be useful for debugging.

## Cache Location

The cache is stored in the following location:

- On macOS/Linux: `~/.cache/options_visualizer/options_cache.db`
- On Windows: `%APPDATA%\options_visualizer\options_cache.db`

## Troubleshooting

If you encounter issues with the cache:

1. Check the logs in `logs/cache_sync.log`
2. Run the sync script manually to see more detailed output
3. Delete the cache file and run the sync script to rebuild the cache from scratch 