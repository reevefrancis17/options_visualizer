import os
import json
import sqlite3
import logging
import time
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Set up logger
logger = logging.getLogger(__name__)

class OptionsCache:
    """SQLite-based persistent cache for options data with crash recovery."""
    
    def __init__(self, cache_duration=600):
        """Initialize the cache with the specified duration.
        
        Args:
            cache_duration: Cache validity duration in seconds (default: 10 minutes)
        """
        self.cache_duration = cache_duration
        self.db_path = self._get_cache_path()
        self._initialize_db()
    
    def _get_cache_path(self) -> Path:
        """Get the path to the cache database file."""
        # Determine appropriate cache directory based on platform
        app_name = "options_visualizer"
        
        if os.name == 'nt':  # Windows
            base_dir = os.environ.get('APPDATA', os.path.expanduser('~'))
            cache_dir = os.path.join(base_dir, app_name)
        else:  # macOS/Linux
            base_dir = os.path.expanduser('~')
            cache_dir = os.path.join(base_dir, '.cache', app_name)
        
        # Create directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        return Path(cache_dir) / "options_cache.db"
    
    def _initialize_db(self):
        """Initialize the SQLite database with the required schema."""
        try:
            # Connect to the database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Enable WAL mode for better crash recovery
            cursor.execute("PRAGMA journal_mode=WAL;")
            
            # Create the options_cache table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS options_cache (
                    ticker TEXT PRIMARY KEY,
                    data BLOB,
                    current_price REAL,
                    last_updated REAL,
                    processed_dates INTEGER,
                    total_dates INTEGER
                );
            """)
            
            # Create index on ticker for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON options_cache (ticker);")
            
            # Commit changes and close connection
            conn.commit()
            conn.close()
            
            logger.info(f"Initialized cache database at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing cache database: {str(e)}")
            # If initialization fails, try to recover by recreating the database
            self._recover_database()
    
    def _recover_database(self):
        """Attempt to recover the database if it's corrupted."""
        try:
            logger.warning("Attempting to recover corrupted cache database")
            
            # Backup the corrupted database if it exists
            if self.db_path.exists():
                backup_path = self.db_path.with_suffix(f".bak.{int(time.time())}")
                self.db_path.rename(backup_path)
                logger.info(f"Backed up corrupted database to {backup_path}")
            
            # Create a new database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Enable WAL mode
            cursor.execute("PRAGMA journal_mode=WAL;")
            
            # Create the options_cache table
            cursor.execute("""
                CREATE TABLE options_cache (
                    ticker TEXT PRIMARY KEY,
                    data BLOB,
                    current_price REAL,
                    last_updated REAL,
                    processed_dates INTEGER,
                    total_dates INTEGER
                );
            """)
            
            # Create index on ticker
            cursor.execute("CREATE INDEX idx_ticker ON options_cache (ticker);")
            
            # Commit changes and close connection
            conn.commit()
            conn.close()
            
            logger.info("Successfully recovered cache database")
        except Exception as e:
            logger.error(f"Failed to recover cache database: {str(e)}")
    
    def get(self, ticker: str) -> Tuple[Optional[Dict], Optional[float], str, float, int, int]:
        """Get cached data for a ticker.
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            Tuple of (options_data, current_price, status, progress, processed_dates, total_dates)
            where status is one of 'complete', 'partial', or 'not_found'
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Query the cache for the ticker
            cursor.execute(
                "SELECT data, current_price, last_updated, processed_dates, total_dates FROM options_cache WHERE ticker = ?",
                (ticker,)
            )
            result = cursor.fetchone()
            conn.close()
            
            # If no result or cache is expired, return not found
            if not result or (time.time() - result[2] > self.cache_duration):
                return None, None, 'not_found', 0, 0, 0
            
            # Unpack the result
            data_blob, current_price, timestamp, processed_dates, total_dates = result
            
            # Deserialize the data
            options_data = pickle.loads(data_blob)
            
            # Calculate progress
            progress = (processed_dates / total_dates) * 100 if total_dates > 0 else 0
            
            # Determine status based on processed vs total dates
            status = 'complete' if processed_dates >= total_dates else 'partial'
            
            return options_data, current_price, status, progress, processed_dates, total_dates
        
        except Exception as e:
            logger.error(f"Error retrieving data from cache for {ticker}: {str(e)}")
            return None, None, 'not_found', 0, 0, 0
    
    def set(self, ticker: str, options_data: Dict, current_price: float, processed_dates: int, total_dates: int):
        """Store data in the cache.
        
        Args:
            ticker: The stock ticker symbol
            options_data: The options data to cache
            current_price: The current stock price
            processed_dates: Number of expiration dates processed
            total_dates: Total number of expiration dates
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Serialize the data
            data_blob = pickle.dumps(options_data)
            
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION;")
            
            # Insert or replace the data
            cursor.execute(
                """
                INSERT OR REPLACE INTO options_cache 
                (ticker, data, current_price, last_updated, processed_dates, total_dates)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (ticker, data_blob, current_price, time.time(), processed_dates, total_dates)
            )
            
            # Commit the transaction
            cursor.execute("COMMIT;")
            conn.close()
            
            logger.info(f"Cached data for {ticker} ({processed_dates}/{total_dates} dates)")
        except Exception as e:
            logger.error(f"Error caching data for {ticker}: {str(e)}")
    
    def clear(self, ticker: Optional[str] = None):
        """Clear cache entries.
        
        Args:
            ticker: If provided, clear only this ticker's data. Otherwise, clear all cache.
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION;")
            
            if ticker:
                # Delete specific ticker
                cursor.execute("DELETE FROM options_cache WHERE ticker = ?;", (ticker,))
                logger.info(f"Cleared cache for {ticker}")
            else:
                # Delete all entries
                cursor.execute("DELETE FROM options_cache;")
                logger.info("Cleared entire cache")
            
            # Commit the transaction
            cursor.execute("COMMIT;")
            
            # Optimize the database
            cursor.execute("VACUUM;")
            
            conn.close()
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    def maintenance(self):
        """Perform maintenance tasks on the cache database."""
        try:
            # Connect to the database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Remove expired entries
            expiry_time = time.time() - self.cache_duration
            cursor.execute("DELETE FROM options_cache WHERE last_updated < ?;", (expiry_time,))
            deleted_count = cursor.rowcount
            
            # Optimize the database if entries were deleted
            if deleted_count > 0:
                cursor.execute("VACUUM;")
                logger.info(f"Removed {deleted_count} expired cache entries and optimized database")
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error during cache maintenance: {str(e)}")
    
    def get_stats(self):
        """Get statistics about the cache.
        
        Returns:
            Dict with cache statistics
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM options_cache;")
            total_count = cursor.fetchone()[0]
            
            # Get expired count
            expiry_time = time.time() - self.cache_duration
            cursor.execute("SELECT COUNT(*) FROM options_cache WHERE last_updated < ?;", (expiry_time,))
            expired_count = cursor.fetchone()[0]
            
            # Get database size
            cursor.execute("PRAGMA page_count;")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size;")
            page_size = cursor.fetchone()[0]
            db_size = page_count * page_size
            
            conn.close()
            
            return {
                "total_entries": total_count,
                "expired_entries": expired_count,
                "valid_entries": total_count - expired_count,
                "database_size_bytes": db_size,
                "database_size_mb": db_size / (1024 * 1024),
                "cache_path": str(self.db_path)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"error": str(e)} 