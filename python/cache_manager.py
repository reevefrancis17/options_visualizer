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
        self.ticker_locks = {}  # Per-ticker locks for thread safety
        
        # Path to the ticker registry file - single source of truth for tickers
        self.registry_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ticker_registry.json')
        self.registry_lock = threading.RLock()  # Lock for thread-safe registry access
        
        # Ensure the registry file exists
        self._ensure_registry_exists()
        
        # Initialize the database
        self._initialize_db()
        self._migrate_db_if_needed()
        
        # Synchronize the cache with the registry (bidirectional)
        self._sync_cache_with_registry()
        
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
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        return Path(os.path.join(cache_dir, 'options_cache.db'))

    def _ensure_registry_exists(self):
        """Ensure the ticker registry file exists."""
        # Create the data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        
        # Create an empty registry file if it doesn't exist
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, 'w') as f:
                json.dump({}, f, indent=2)
            logger.info(f"Created new ticker registry at {self.registry_path}")

    def _load_registry(self) -> Dict:
        """Load the ticker registry from file."""
        with self.registry_lock:
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error loading ticker registry: {str(e)}")
                return {}

    def _save_registry(self, registry: Dict):
        """Save the ticker registry to file."""
        with self.registry_lock:
            try:
                with open(self.registry_path, 'w') as f:
                    json.dump(registry, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving ticker registry: {str(e)}")

    def update_registry(self, ticker: str):
        """Update the ticker registry with a new or existing ticker."""
        with self.registry_lock:
            registry = self._load_registry()
            now = datetime.now().isoformat()
            
            if ticker in registry:
                # Update existing ticker
                registry[ticker]["last_accessed"] = now
                registry[ticker]["access_count"] += 1
            else:
                # Add new ticker
                registry[ticker] = {
                    "first_added": now,
                    "last_accessed": now,
                    "access_count": 1
                }
                logger.info(f"Added new ticker {ticker} to registry")
            
            self._save_registry(registry)

    def get_registry_tickers(self) -> List[str]:
        """Get all tickers from the registry."""
        registry = self._load_registry()
        return list(registry.keys())

    def _sync_cache_with_registry(self):
        """Synchronize the cache with the registry bidirectionally.
        
        This ensures that:
        1. All tickers in the cache are also in the registry
        2. All tickers in the registry are loaded into the cache
        3. The registry is the single source of truth for tickers
        """
        try:
            # Get all tickers from the cache
            cached_tickers = self._get_cached_tickers_from_db()
            
            # Get all tickers from the registry
            registry_tickers = set(self.get_registry_tickers())
            
            # Add any cached tickers that are not in the registry
            with self.registry_lock:
                registry = self._load_registry()
                now = datetime.now().isoformat()
                
                for ticker in cached_tickers:
                    if ticker not in registry:
                        logger.info(f"Adding cached ticker {ticker} to registry")
                        registry[ticker] = {
                            "first_added": now,
                            "last_accessed": now,
                            "access_count": 1
                        }
                
                self._save_registry(registry)
            
            # Trigger loading of registry tickers that aren't in the cache
            for ticker in registry_tickers:
                if ticker not in cached_tickers:
                    logger.info(f"Ticker {ticker} from registry not in cache - will be loaded during next refresh cycle")
            
            logger.info(f"Synchronized cache with registry: {len(registry_tickers)} registry tickers, {len(cached_tickers)} cached tickers")
        except Exception as e:
            logger.error(f"Error synchronizing cache with registry: {str(e)}")

    def _get_cached_tickers_from_db(self) -> set:
        """Get all tickers from the cache database."""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT ticker FROM options_cache")
                tickers = cursor.fetchall()
                conn.close()
                
                return {ticker[0] for ticker in tickers}
        except Exception as e:
            logger.error(f"Error getting cached tickers from database: {str(e)}")
            return set()

    def _initialize_db(self):
        """Initialize the SQLite database with the required tables."""
        try:
            with self.lock:
                # Create a backup of the database if it exists
                if os.path.exists(self.db_path):
                    backup_path = f"{self.db_path}.bak"
                    try:
                        shutil.copy2(self.db_path, backup_path)
                        logger.info(f"Created backup of cache database at {backup_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create backup of cache database: {str(e)}")
                
                # Connect to the database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create the options_cache table if it doesn't exist
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS options_cache (
                    ticker TEXT PRIMARY KEY,
                    data BLOB,
                    current_price REAL,
                    timestamp REAL,
                    processed_dates INTEGER,
                    total_dates INTEGER
                )
                ''')
                
                # Create the metadata table if it doesn't exist
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                ''')
                
                # Set the schema version
                cursor.execute('''
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('schema_version', '2')
                ''')
                
                # Commit the changes and close the connection
                conn.commit()
                conn.close()
                
                logger.info("Cache database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing cache database: {str(e)}")
            # Try to recover the database if initialization fails
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
                        INSERT OR REPLACE INTO metadata (key, value)
                        VALUES ('version', '2.0')
                    """)
                    
                    conn.commit()
                    logger.info("Database migration completed successfully")
                
                # Check if timestamp column exists
                if 'timestamp' not in columns:
                    logger.info("Migrating database: Adding timestamp column")
                    cursor.execute("ALTER TABLE options_cache ADD COLUMN timestamp REAL DEFAULT 0")
                    
                    # Update existing records with current timestamp
                    cursor.execute("UPDATE options_cache SET timestamp = ?", (time.time(),))
                    
                    conn.commit()
                    logger.info("Timestamp column added successfully")
                
                conn.close()
        except Exception as e:
            logger.error(f"Error during database migration: {str(e)}")
            # If migration fails, try to recover
            self._recover_database()
    
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
    
    def get(self, ticker: str) -> Tuple[Optional[Dict], Optional[float], float, float, int, int]:
        """Get options data from the cache.
        
        Args:
            ticker: The ticker symbol to get data for
        
        Returns:
            Tuple of (options_data, current_price, timestamp, age, processed_dates, total_dates)
            or (None, None, 0, 0, 0, 0) if not found
        """
        # Update the ticker registry - single source of truth
        self.update_registry(ticker)
        
        ticker_lock = self.get_lock(ticker)
        with ticker_lock:
            try:
                with self.lock:
                    # Connect to the database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # Get the options data for the ticker
                    cursor.execute('''
                    SELECT data, current_price, timestamp, processed_dates, total_dates
                    FROM options_cache
                    WHERE ticker = ?
                    ''', (ticker,))
                    
                    result = cursor.fetchone()
                    conn.close()
                    
                    if result:
                        # Decompress and unpickle the data
                        compressed_data, current_price, timestamp, processed_dates, total_dates = result
                        try:
                            data = pickle.loads(zlib.decompress(compressed_data))
                            
                            # Calculate the age of the data
                            now = time.time()
                            age = now - timestamp if timestamp is not None else float('inf')
                            
                            # Check if the data is stale
                            if age > self.cache_duration:
                                # Trigger a refresh in the background
                                self._trigger_refresh(ticker)
                            
                            return data, current_price, timestamp, age, processed_dates, total_dates
                        except Exception as e:
                            logger.error(f"Error decompressing/unpickling cached data for {ticker}: {str(e)}")
                            # Delete the corrupted data
                            self.delete(ticker)
                            return None, None, 0, 0, 0, 0
                    else:
                        return None, None, 0, 0, 0, 0
            except Exception as e:
                logger.error(f"Error getting cached data for {ticker}: {str(e)}")
                return None, None, 0, 0, 0, 0
    
    def _trigger_refresh(self, ticker: str):
        """Trigger a background refresh of a stale ticker."""
        logger.info(f"Triggering background refresh for stale ticker: {ticker}")
        
        # Call the refresh callback if registered
        if hasattr(self, 'refresh_callback') and self.refresh_callback:
            try:
                self.refresh_callback(ticker)
                logger.info(f"Called refresh callback for ticker: {ticker}")
            except Exception as e:
                logger.error(f"Error calling refresh callback for {ticker}: {str(e)}")
    
    def register_refresh_callback(self, callback):
        """Register a callback function to be called when a ticker needs refreshing.
        
        Args:
            callback: A function that takes a ticker symbol as its only argument
        """
        self.refresh_callback = callback
        logger.info("Registered refresh callback")
    
    def set(self, ticker: str, options_data: Dict, current_price: float, processed_dates: int, total_dates: int):
        """Store data in the cache.
        
        Args:
            ticker: The stock ticker symbol
            options_data: The options data to cache
            current_price: The current stock price
            processed_dates: Number of expiration dates processed
            total_dates: Total number of expiration dates
        """
        # Update the ticker registry - single source of truth
        self.update_registry(ticker)
        
        try:
            # Use the ticker-specific lock for thread safety
            with self.get_lock(ticker):
                # Connect to the database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Serialize and compress the data
                try:
                    # Use highest compression level for better storage efficiency
                    pickled_data = pickle.dumps(options_data, protocol=pickle.HIGHEST_PROTOCOL)
                    data_blob = zlib.compress(pickled_data, level=self.compression_level)
                except Exception as serialize_error:
                    logger.error(f"Error compressing data for {ticker}: {str(serialize_error)}")
                    # Fall back to uncompressed pickle if compression fails
                    data_blob = pickle.dumps(options_data, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Insert or replace the data
                cursor.execute('''
                INSERT OR REPLACE INTO options_cache 
                (ticker, data, current_price, timestamp, processed_dates, total_dates)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (ticker, data_blob, current_price, time.time(), processed_dates, total_dates))
                
                # Commit the changes and close the connection
                conn.commit()
                conn.close()
                
                # Log the size of the data for monitoring
                data_size_kb = len(data_blob) / 1024
                logger.info(f"Cached data for {ticker} ({processed_dates}/{total_dates} dates), size: {data_size_kb:.2f} KB")
        except Exception as e:
            logger.error(f"Error caching data for {ticker}: {str(e)}")
    
    def clear(self, ticker: Optional[str] = None):
        """Clear the cache for a specific ticker or all tickers."""
        try:
            if ticker:
                # Use the ticker-specific lock for thread safety
                with self.get_lock(ticker):
                    # Connect to the database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # Delete the ticker from the cache
                    cursor.execute("DELETE FROM options_cache WHERE ticker = ?", (ticker,))
                    
                    # Commit changes and close connection
                    conn.commit()
                    conn.close()
                    
                    logger.info(f"Cleared cache for {ticker}")
            else:
                # Clear all tickers - use the global lock
                with self.lock:
                    # Connect to the database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # Delete all data from the cache
                    cursor.execute("DELETE FROM options_cache")
                    
                    # Commit changes and close connection
                    conn.commit()
                    conn.close()
                    
                    logger.info("Cleared entire cache")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    def delete(self, ticker: str):
        """Delete a specific ticker from the cache."""
        # Use the ticker-specific lock for thread safety
        with self.get_lock(ticker):
            self.clear(ticker)
    
    def maintenance(self):
        """Perform maintenance tasks on the cache database.
        
        This includes:
        1. Checking database integrity
        2. Synchronizing with the registry
        3. Updating the list of cached tickers
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
                
                # Close connection
                conn.close()
                
                # Synchronize with the registry
                self._sync_cache_with_registry()
                
                logger.info("Cache maintenance completed successfully")
        except Exception as e:
            logger.error(f"Error during cache maintenance: {str(e)}")
            # Try to recover the database
            self._recover_database()
    
    def get_all_tickers(self) -> List[str]:
        """Get a list of all tickers in the cache and registry.
        
        This returns the union of tickers in both the cache and registry,
        ensuring we have a complete list of all tickers.
        """
        # Get tickers from the registry (single source of truth)
        registry_tickers = set(self.get_registry_tickers())
        
        # Get tickers from the cache
        cached_tickers = self._get_cached_tickers_from_db()
        
        # Return the union of both sets
        all_tickers = registry_tickers.union(cached_tickers)
        return list(all_tickers)
    
    def _start_polling(self):
        """Start a background thread to poll and refresh the cache periodically."""
        def poll_cache():
            while True:
                try:
                    # Sleep first to allow the application to initialize
                    time.sleep(self.cache_duration / 2)
                    
                    # Synchronize the cache with the registry (bidirectional)
                    self._sync_cache_with_registry()
                    
                    # Get all tickers from the registry (single source of truth)
                    registry_tickers = self.get_registry_tickers()
                    
                    # Get all tickers currently in the cache
                    cached_tickers = self._get_cached_tickers_from_db()
                    
                    # Load all tickers from the registry into the cache
                    for ticker in registry_tickers:
                        # Check if the ticker is already in the cache
                        if ticker not in cached_tickers:
                            logger.info(f"Auto-loading ticker {ticker} from registry")
                            # Trigger a refresh for this ticker
                            self._trigger_refresh(ticker)
                            # Sleep briefly to avoid overwhelming the API
                            time.sleep(1)
                    
                    # Perform cache maintenance
                    self.maintenance()
                except Exception as e:
                    logger.error(f"Error in cache polling thread: {str(e)}")
        
        # Start the polling thread
        thread = threading.Thread(target=poll_cache, daemon=True)
        thread.start()
        logger.info("Cache polling thread started")
    
    def get_lock(self, ticker):
        """Get a lock for a specific ticker.
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            A threading.Lock object for the ticker
        """
        with self.lock:  # Protect the ticker_locks dictionary
            if ticker not in self.ticker_locks:
                self.ticker_locks[ticker] = threading.RLock()  # Use RLock to allow reentrant locking
            return self.ticker_locks[ticker] 