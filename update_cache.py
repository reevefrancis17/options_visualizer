#!/usr/bin/env python3
"""
Cache Update Script

This script updates the options cache to only include tickers from the tickers.csv file
and removes any tickers that are not in the file.
"""
import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the cache manager
from python.cache_manager import OptionsCache
from python.options_data import OptionsDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_tickers_from_csv(csv_path):
    """Load tickers from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if 'ticker' not in df.columns:
            logger.error(f"CSV file {csv_path} does not have a 'ticker' column")
            return []
        
        # Extract tickers and filter out any empty values
        tickers = [ticker.strip() for ticker in df['ticker'].tolist() if ticker and isinstance(ticker, str)]
        logger.info(f"Loaded {len(tickers)} tickers from {csv_path}")
        return tickers
    except Exception as e:
        logger.error(f"Error loading tickers from {csv_path}: {str(e)}")
        return []

def update_cache_with_csv_tickers(csv_path):
    """Update the cache to only include tickers from the CSV file."""
    # Load tickers from CSV
    csv_tickers = load_tickers_from_csv(csv_path)
    if not csv_tickers:
        logger.error("No tickers loaded from CSV, aborting cache update")
        return False
    
    # Initialize the cache
    cache = OptionsCache()
    
    # Get all tickers currently in the cache
    cached_tickers = cache.get_all_tickers()
    logger.info(f"Found {len(cached_tickers)} tickers in the cache")
    
    # Find tickers to remove (in cache but not in CSV)
    tickers_to_remove = [ticker for ticker in cached_tickers if ticker not in csv_tickers]
    logger.info(f"Will remove {len(tickers_to_remove)} tickers from cache")
    
    # Find tickers to keep (in both cache and CSV)
    tickers_to_keep = [ticker for ticker in cached_tickers if ticker in csv_tickers]
    logger.info(f"Will keep {len(tickers_to_keep)} tickers in cache")
    
    # Find tickers to add (in CSV but not in cache)
    tickers_to_add = [ticker for ticker in csv_tickers if ticker not in cached_tickers]
    logger.info(f"Will add {len(tickers_to_add)} new tickers to cache")
    
    # Remove tickers not in CSV
    for ticker in tickers_to_remove:
        try:
            logger.info(f"Removing ticker {ticker} from cache")
            cache.delete(ticker)
        except Exception as e:
            logger.error(f"Error removing ticker {ticker} from cache: {str(e)}")
    
    # Update registry for tickers to keep
    for ticker in tickers_to_keep:
        try:
            logger.info(f"Updating registry for ticker {ticker}")
            cache.update_registry(ticker)
        except Exception as e:
            logger.error(f"Error updating registry for ticker {ticker}: {str(e)}")
    
    # Initialize options data manager to trigger fetching for new tickers
    if tickers_to_add:
        try:
            logger.info(f"Initializing OptionsDataManager to fetch new tickers")
            options_data_manager = OptionsDataManager(cache_duration=600)
            
            # Start fetching data for new tickers
            for ticker in tickers_to_add:
                try:
                    logger.info(f"Starting to fetch data for new ticker {ticker}")
                    options_data_manager.start_fetching(ticker)
                except Exception as e:
                    logger.error(f"Error fetching data for ticker {ticker}: {str(e)}")
        except Exception as e:
            logger.error(f"Error initializing OptionsDataManager: {str(e)}")
    
    # Perform cache maintenance
    try:
        logger.info("Performing cache maintenance")
        cache.maintenance()
    except Exception as e:
        logger.error(f"Error during cache maintenance: {str(e)}")
    
    logger.info("Cache update completed")
    return True

if __name__ == "__main__":
    # Path to the tickers CSV file
    csv_path = os.path.join(os.path.dirname(__file__), "data", "tickers.csv")
    
    # Check if the file exists
    if not os.path.exists(csv_path):
        logger.error(f"Tickers CSV file not found at {csv_path}")
        sys.exit(1)
    
    logger.info(f"Starting cache update with tickers from {csv_path}")
    
    # Update the cache
    success = update_cache_with_csv_tickers(csv_path)
    
    if success:
        logger.info("Cache update completed successfully")
        sys.exit(0)
    else:
        logger.error("Cache update failed")
        sys.exit(1) 