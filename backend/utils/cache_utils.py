#!/usr/bin/env python3
"""
Cache Utility Script

This script provides command-line utilities for managing the options data cache.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Import the cache manager
from backend.utils.cache_manager import OptionsCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_cache(args):
    """Clear the cache for a specific ticker or all tickers."""
    cache = OptionsCache()
    
    if args.ticker:
        logger.info(f"Clearing cache for ticker: {args.ticker}")
        cache.clear(args.ticker)
    else:
        logger.info("Clearing entire cache")
        cache.clear()
    
    logger.info("Cache cleared successfully")

def show_stats(args):
    """Show statistics about the cache."""
    cache = OptionsCache()
    stats = cache.get_stats()
    
    logger.info("Cache Statistics:")
    logger.info(f"  Cache path: {stats.get('cache_path', 'Unknown')}")
    logger.info(f"  Total entries: {stats.get('total_entries', 0)}")
    logger.info(f"  Valid entries: {stats.get('valid_entries', 0)}")
    logger.info(f"  Expired entries: {stats.get('expired_entries', 0)}")
    logger.info(f"  Database size: {stats.get('database_size_mb', 0):.2f} MB")

def run_maintenance(args):
    """Run maintenance on the cache."""
    cache = OptionsCache()
    logger.info("Running cache maintenance")
    cache.maintenance()
    logger.info("Maintenance completed")

def main():
    """Main entry point for the cache utility script."""
    parser = argparse.ArgumentParser(description="Options Data Cache Utility")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Clear cache command
    clear_parser = subparsers.add_parser("clear", help="Clear the cache")
    clear_parser.add_argument("--ticker", "-t", help="Ticker symbol to clear (if not specified, clears all)")
    clear_parser.set_defaults(func=clear_cache)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")
    stats_parser.set_defaults(func=show_stats)
    
    # Maintenance command
    maintenance_parser = subparsers.add_parser("maintenance", help="Run cache maintenance")
    maintenance_parser.set_defaults(func=run_maintenance)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)

if __name__ == "__main__":
    main() 