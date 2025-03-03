import os
import json
import sqlite3
import logging
import time
import pickle
import zlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import threading
import shutil

# Set up logger
logger = logging.getLogger(__name__)

class OptionsCache:
    """SQLite-based persistent cache for options data with crash recovery and compression."""
    
    def __init__(self, cache_duration=600, compression_level=6):
        """Initialize the cache with the specified duration.
        
        Args:
            cache_duration: Cache validity duration in seconds (default: 10 minutes)
            compression_level: Compression level for pickle data (0-9, higher = more compression)
        """
        self.cache_duration = cache_duration
        self.compression_level = compression_level
        self.db_path = self._get_cache_path()
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self._initialize_db()
        self._migrate_db_if_needed()  # Add migration step
        self.cached_tickers = set()
        self._load_cached_tickers()
        
        # Start background polling for cache updates
        self._start_polling()
    
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
            with self.lock:
                # Connect to the database
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Enable WAL mode for better crash recovery
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")  # Slightly faster with good safety
                cursor.execute("PRAGMA temp_store=MEMORY;")   # Store temp tables in memory
                
                # Create the options_cache table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS options_cache (
                        ticker TEXT PRIMARY KEY,
                        data BLOB,
                        current_price REAL,
                        last_updated REAL,
                        processed_dates INTEGER,
                        total_dates INTEGER,
                        is_compressed INTEGER DEFAULT 1
                    );
                """)
                
                # Create index on ticker for faster lookups
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON options_cache (ticker);")
                
                # Create a metadata table for cache info
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    );
                """)
                
                # Insert or update version info
                cursor.execute("""
                    INSERT OR REPLACE INTO cache_metadata (key, value)
                    VALUES ('version', '2.0');
                """)
                
                # Commit changes and close connection
                conn.commit()
                conn.close()
                
                logger.info(f"Initialized cache database at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing cache database: {str(e)}")
            # If initialization fails, try to recover by recreating the database
            self._recover_database()
    
    def _migrate_db_if_needed(self):
        """Check if database needs migration and apply necessary changes."""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Check if is_compressed column exists
                cursor.execute("PRAGMA table_info(options_cache)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'is_compressed' not in columns:
                    logger.info("Migrating database: Adding is_compressed column")
                    cursor.execute("ALTER TABLE options_cache ADD COLUMN is_compressed INTEGER DEFAULT 0")
                    
                    # Update existing records to mark them as uncompressed
                    cursor.execute("UPDATE options_cache SET is_compressed = 0")
                    
                    # Update version in metadata
                    cursor.execute("""
                        INSERT OR REPLACE INTO cache_metadata (key, value)
                        VALUES ('version', '2.0');
                    """)
                    
                    conn.commit()
                    logger.info("Database migration completed successfully")
                
                conn.close()
        except Exception as e:
            logger.error(f"Error during database migration: {str(e)}")
            # If migration fails, try to recover
            self._recover_database()
    
    def _load_cached_tickers(self):
        """Load the list of cached tickers from the database."""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT ticker FROM options_cache")
                tickers = cursor.fetchall()
                conn.close()
                
                self.cached_tickers = {ticker[0] for ticker in tickers}
                logger.info(f"Loaded {len(self.cached_tickers)} tickers from cache")
        except Exception as e:
            logger.error(f"Error loading cached tickers: {str(e)}")
            self.cached_tickers = set()
    
    def _recover_database(self):
        """Attempt to recover the database if it's corrupted."""
        try:
            logger.warning(f"Attempting to recover database at {self.db_path}")
            
            # Create a backup of the corrupted database
            backup_path = f"{self.db_path}.bak.{int(time.time())}"
            try:
                shutil.copy2(self.db_path, backup_path)
                logger.info(f"Created backup of corrupted database at {backup_path}")
            except Exception as e:
                logger.error(f"Failed to create backup: {str(e)}")
            
            # Try to open and close the database to see if it's accessible
            try:
                conn = sqlite3.connect(str(self.db_path))
                conn.close()
                logger.info("Database is accessible, no recovery needed")
                return
            except sqlite3.Error:
                logger.warning("Database is not accessible, attempting recovery")
            
            # Try to recover using SQLite's recovery mode
            try:
                conn = sqlite3.connect(str(self.db_path))
                conn.execute("PRAGMA integrity_check;")
                conn.close()
                logger.info("Database integrity check passed")
                return
            except sqlite3.Error as e:
                logger.error(f"Integrity check failed: {str(e)}")
            
            # If we get here, we need to recreate the database
            logger.warning("Recreating database from scratch")
            try:
                # Delete the corrupted database
                os.remove(self.db_path)
            except Exception as e:
                logger.error(f"Failed to delete corrupted database: {str(e)}")
            
            # Reinitialize the database
            self._initialize_db()
            logger.info("Database recovery completed")
            
        except Exception as e:
            logger.error(f"Error during database recovery: {str(e)}")
    
    def _vacuum_db(self):
        """Vacuum the database to reclaim space and optimize performance."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("VACUUM;")
            conn.close()
            logger.info("Database vacuumed successfully")
        except Exception as e:
            logger.error(f"Error vacuuming database: {str(e)}")
    
    def get(self, ticker: str) -> Tuple[Optional[Dict], Optional[float], str, float, int, int]:
        """Get cached data for a ticker.
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            Tuple of (options_data, current_price, status, progress, processed_dates, total_dates)
            where status is one of 'complete', 'partial', or 'not_found'
        """
        try:
            with self.lock:
                # Connect to the database
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Query the cache for the ticker
                cursor.execute(
                    """SELECT data, current_price, last_updated, processed_dates, total_dates, 
                       CASE WHEN is_compressed IS NULL THEN 0 ELSE is_compressed END 
                       FROM options_cache WHERE ticker = ?""",
                    (ticker,)
                )
                result = cursor.fetchone()
                conn.close()
                
                # If no result, return not found
                if not result:
                    return None, None, 'not_found', 0, 0, 0
                
                # Unpack the result
                data_blob, current_price, timestamp, processed_dates, total_dates, is_compressed = result
                
                # Add ticker to cached tickers set if not already there
                if ticker not in self.cached_tickers:
                    self.cached_tickers.add(ticker)
                
                # Check if cache is stale (older than cache_duration)
                is_stale = (time.time() - timestamp > self.cache_duration)
                
                # Deserialize the data - use a more efficient approach
                try:
                    if is_compressed:
                        # Use a memory-efficient approach for large data
                        options_data = pickle.loads(zlib.decompress(data_blob))
                    else:
                        # Just unpickle (for backward compatibility)
                        options_data = pickle.loads(data_blob)
                except Exception as deserialize_error:
                    logger.error(f"Error deserializing data for {ticker}: {str(deserialize_error)}")
                    return None, None, 'not_found', 0, 0, 0
                
                # Calculate progress
                progress = (processed_dates / total_dates) * 100 if total_dates > 0 else 0
                
                # Determine status based on processed vs total dates and staleness
                if is_stale:
                    # Data is stale but we'll return it anyway and trigger a refresh in the background
                    status = 'stale'
                    # Trigger a background refresh
                    self._trigger_refresh(ticker)
                else:
                    status = 'complete' if processed_dates >= total_dates else 'partial'
                
                return options_data, current_price, status, progress, processed_dates, total_dates
        
        except Exception as e:
            logger.error(f"Error retrieving data from cache for {ticker}: {str(e)}")
            return None, None, 'not_found', 0, 0, 0
    
    def _trigger_refresh(self, ticker: str):
        """Trigger a background refresh of a stale ticker."""
        # This will be called by the OptionsDataManager
        logger.info(f"Triggering background refresh for stale ticker: {ticker}")
    
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
            with self.lock:
                # Connect to the database
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Serialize and compress the data
                try:
                    # Use highest compression level (9) for better storage efficiency
                    # Use protocol 5 for better performance with large objects
                    pickled_data = pickle.dumps(options_data, protocol=pickle.HIGHEST_PROTOCOL)
                    data_blob = zlib.compress(pickled_data, level=9)
                    is_compressed = 1
                except Exception as serialize_error:
                    logger.error(f"Error compressing data for {ticker}, falling back to uncompressed: {str(serialize_error)}")
                    # Fall back to uncompressed pickle if compression fails
                    data_blob = pickle.dumps(options_data, protocol=pickle.HIGHEST_PROTOCOL)
                    is_compressed = 0
                
                # Begin transaction
                cursor.execute("BEGIN IMMEDIATE TRANSACTION;")
                
                # Insert or replace the data
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO options_cache 
                    (ticker, data, current_price, last_updated, processed_dates, total_dates, is_compressed)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (ticker, data_blob, current_price, time.time(), processed_dates, total_dates, is_compressed)
                )
                
                # Commit the transaction
                cursor.execute("COMMIT;")
                conn.close()
                
                # Add to cached tickers set
                self.cached_tickers.add(ticker)
                
                # Log the size of the data for monitoring
                data_size_kb = len(data_blob) / 1024
                logger.info(f"Cached data for {ticker} ({processed_dates}/{total_dates} dates), size: {data_size_kb:.2f} KB")
            
        except Exception as e:
            logger.error(f"Error caching data for {ticker}: {str(e)}")
    
    def batch_set(self, items: List[Tuple[str, Dict, float, int, int]]):
        """Store multiple items in the cache in a single transaction.
        
        Args:
            items: List of tuples (ticker, options_data, current_price, processed_dates, total_dates)
        """
        if not items:
            return
            
        try:
            with self.lock:
                # Connect to the database
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Begin transaction
                cursor.execute("BEGIN IMMEDIATE TRANSACTION;")
                
                # Process each item
                for ticker, options_data, current_price, processed_dates, total_dates in items:
                    try:
                        # Serialize and compress the data
                        pickled_data = pickle.dumps(options_data, protocol=pickle.HIGHEST_PROTOCOL)
                        data_blob = zlib.compress(pickled_data, level=9)
                        is_compressed = 1
                    except Exception:
                        # Fall back to uncompressed pickle if compression fails
                        data_blob = pickle.dumps(options_data, protocol=pickle.HIGHEST_PROTOCOL)
                        is_compressed = 0
                    
                    # Insert or replace the data
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO options_cache 
                        (ticker, data, current_price, last_updated, processed_dates, total_dates, is_compressed)
                        VALUES (?, ?, ?, ?, ?, ?, ?);
                        """,
                        (ticker, data_blob, current_price, time.time(), processed_dates, total_dates, is_compressed)
                    )
                    
                    # Add to cached tickers set
                    self.cached_tickers.add(ticker)
                
                # Commit the transaction
                cursor.execute("COMMIT;")
                conn.close()
                
                logger.info(f"Batch cached {len(items)} items")
            
        except Exception as e:
            logger.error(f"Error in batch cache operation: {str(e)}")
    
    def clear(self, ticker: Optional[str] = None):
        """Clear the cache for a specific ticker or all tickers."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                if ticker:
                    # Clear specific ticker
                    cursor.execute("DELETE FROM options_cache WHERE ticker = ?", (ticker,))
                    logger.info(f"Cleared cache for {ticker}")
                    
                    # Remove from cached_tickers set
                    if ticker in self.cached_tickers:
                        self.cached_tickers.remove(ticker)
                else:
                    # Clear all tickers
                    cursor.execute("DELETE FROM options_cache")
                    logger.info("Cleared entire cache")
                    self.cached_tickers.clear()
                
                conn.commit()
                conn.close()
                
                # Vacuum the database to reclaim space
                self._vacuum_db()
                
            except Exception as e:
                logger.error(f"Error clearing cache: {str(e)}")
                # Try to recover the database
                self._recover_database()
    
    def delete(self, ticker: str):
        """Delete a specific ticker from the cache."""
        # This is just an alias for clear(ticker)
        self.clear(ticker)
    
    def maintenance(self):
        """Perform maintenance tasks on the cache database.
        
        Note: This will NOT delete any entries to ensure the cache never shrinks.
        It will only update the list of cached tickers.
        """
        try:
            with self.lock:
                # Connect to the database
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Check database integrity
                cursor.execute("PRAGMA integrity_check;")
                integrity_result = cursor.fetchone()[0]
                if integrity_result != "ok":
                    logger.error(f"Database integrity check failed: {integrity_result}")
                    self._recover_database()
                    return
                
                # Get all tickers in the cache
                cursor.execute("SELECT ticker FROM options_cache;")
                tickers = cursor.fetchall()
                
                # Update the cached tickers set
                self.cached_tickers = {ticker[0] for ticker in tickers}
                
                # Close connection
                conn.close()
                
                logger.info(f"Cache maintenance completed. {len(self.cached_tickers)} tickers in cache.")
        except Exception as e:
            logger.error(f"Error during cache maintenance: {str(e)}")
            # Try to recover the database
            self._recover_database()
    
    def get_stats(self):
        """Get statistics about the cache."""
        try:
            with self.lock:
                # Connect to the database
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Get count of entries
                cursor.execute("SELECT COUNT(*) FROM options_cache;")
                count = cursor.fetchone()[0]
                
                # Get total size of data
                cursor.execute("SELECT SUM(LENGTH(data)) FROM options_cache;")
                total_size = cursor.fetchone()[0] or 0
                
                # Get compression stats - safely check if column exists first
                try:
                    cursor.execute("PRAGMA table_info(options_cache)")
                    columns = [column[1] for column in cursor.fetchall()]
                    
                    if 'is_compressed' in columns:
                        cursor.execute("SELECT COUNT(*) FROM options_cache WHERE is_compressed = 1;")
                        compressed_count = cursor.fetchone()[0]
                    else:
                        compressed_count = 0
                except Exception as column_error:
                    logger.warning(f"Error checking compression stats: {str(column_error)}")
                    compressed_count = 0
                
                # Get oldest and newest entries
                cursor.execute("SELECT MIN(last_updated), MAX(last_updated) FROM options_cache;")
                min_time, max_time = cursor.fetchone()
                
                # Get database file size
                db_size = os.path.getsize(str(self.db_path)) if os.path.exists(str(self.db_path)) else 0
                
                # Get WAL file size if it exists
                wal_path = str(self.db_path) + "-wal"
                wal_size = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
                
                # Close connection
                conn.close()
                
                # Format times as readable strings
                oldest = datetime.fromtimestamp(min_time).strftime('%Y-%m-%d %H:%M:%S') if min_time else 'N/A'
                newest = datetime.fromtimestamp(max_time).strftime('%Y-%m-%d %H:%M:%S') if max_time else 'N/A'
                
                # Return stats
                return {
                    'entries': count,
                    'data_size_mb': total_size / (1024 * 1024) if total_size else 0,
                    'db_size_mb': db_size / (1024 * 1024),
                    'wal_size_mb': wal_size / (1024 * 1024),
                    'compressed_entries': compressed_count,
                    'compression_ratio': f"{compressed_count}/{count}" if count else "0/0",
                    'oldest_entry': oldest,
                    'newest_entry': newest,
                    'cached_tickers': list(self.cached_tickers)
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {
                'entries': 0,
                'data_size_mb': 0,
                'db_size_mb': 0,
                'wal_size_mb': 0,
                'compressed_entries': 0,
                'compression_ratio': "0/0",
                'oldest_entry': 'N/A',
                'newest_entry': 'N/A',
                'error': str(e),
                'cached_tickers': []
            }
    
    def get_all_tickers(self) -> List[str]:
        """Get a list of all tickers in the cache."""
        return list(self.cached_tickers)
    
    def _start_polling(self):
        """Start a background thread to poll for cache updates every 10 minutes."""
        def poll_cache():
            while True:
                try:
                    logger.info("Starting cache polling cycle")
                    # Sleep first to allow the application to initialize
                    time.sleep(self.cache_duration)
                    
                    # This will be used by the OptionsDataManager to refresh all cached tickers
                    logger.info("Cache polling cycle complete")
                except Exception as e:
                    logger.error(f"Error in cache polling: {str(e)}")
                    # Sleep before retrying
                    time.sleep(60)
        
        # Start polling thread
        polling_thread = threading.Thread(target=poll_cache, daemon=True)
        polling_thread.start()
        logger.info("Started cache polling thread") 