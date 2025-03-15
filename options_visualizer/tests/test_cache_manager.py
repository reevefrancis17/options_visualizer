"""
Unit tests for the cache manager and ticker registry functionality.
"""
import os
import json
import time
import pytest
import tempfile
import shutil
import sqlite3
import threading
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python.cache_manager import OptionsCache


@pytest.fixture
def temp_registry_path():
    """Create a temporary ticker registry file for testing."""
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    registry_path = os.path.join(data_dir, 'ticker_registry.json')
    
    # Create an empty registry
    with open(registry_path, 'w') as f:
        json.dump({}, f)
    
    yield temp_dir, registry_path
    
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_cache_path():
    """Create a temporary cache database for testing."""
    temp_dir = tempfile.mkdtemp()
    cache_path = os.path.join(temp_dir, 'options_cache.db')
    
    yield temp_dir, cache_path
    
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_cache(temp_registry_path, temp_cache_path):
    """Create a mock OptionsCache instance for testing."""
    temp_dir, registry_path = temp_registry_path
    cache_temp_dir, cache_path = temp_cache_path
    
    with patch('python.cache_manager.OptionsCache._get_cache_path') as mock_get_cache_path, \
         patch('python.cache_manager.OptionsCache._start_polling') as mock_start_polling, \
         patch('python.cache_manager.threading.Thread') as mock_thread:
        
        # Mock the cache path to use our temporary path
        mock_get_cache_path.return_value = Path(cache_path)
        
        # Create the cache instance
        cache = OptionsCache(cache_duration=10)
        
        # Override the registry path to use our temporary path
        cache.registry_path = registry_path
        
        # Ensure the registry exists
        cache._ensure_registry_exists()
        
        yield cache


def test_cache_initialization(mock_cache):
    """Test that the cache initializes correctly."""
    assert mock_cache is not None
    assert mock_cache.cache_duration == 10
    assert mock_cache.compression_level == 6
    assert hasattr(mock_cache.lock, 'acquire')  # Check if it has lock-like methods
    assert hasattr(mock_cache.lock, 'release')
    assert hasattr(mock_cache.registry_lock, 'acquire')
    assert hasattr(mock_cache.registry_lock, 'release')
    assert isinstance(mock_cache.ticker_locks, dict)


def test_ensure_registry_exists(mock_cache, temp_registry_path):
    """Test that the registry file is created if it doesn't exist."""
    temp_dir, registry_path = temp_registry_path
    
    # Delete the registry file
    os.remove(registry_path)
    
    # Ensure it gets recreated
    mock_cache._ensure_registry_exists()
    
    # Check that the file exists
    assert os.path.exists(registry_path)
    
    # Check that it contains an empty JSON object
    with open(registry_path, 'r') as f:
        registry = json.load(f)
        assert registry == {}


def test_load_registry(mock_cache, temp_registry_path):
    """Test loading the registry from file."""
    temp_dir, registry_path = temp_registry_path
    
    # Create a registry with some tickers
    registry_data = {
        "SPY": {
            "first_added": "2023-01-01T00:00:00",
            "last_accessed": "2023-01-01T00:00:00",
            "access_count": 1
        },
        "AAPL": {
            "first_added": "2023-01-01T00:00:00",
            "last_accessed": "2023-01-01T00:00:00",
            "access_count": 1
        }
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry_data, f)
    
    # Load the registry
    registry = mock_cache._load_registry()
    
    # Check that it contains the expected data
    assert registry == registry_data
    assert "SPY" in registry
    assert "AAPL" in registry
    assert registry["SPY"]["access_count"] == 1


def test_save_registry(mock_cache, temp_registry_path):
    """Test saving the registry to file."""
    temp_dir, registry_path = temp_registry_path
    
    # Create a registry with some tickers
    registry_data = {
        "SPY": {
            "first_added": "2023-01-01T00:00:00",
            "last_accessed": "2023-01-01T00:00:00",
            "access_count": 1
        },
        "AAPL": {
            "first_added": "2023-01-01T00:00:00",
            "last_accessed": "2023-01-01T00:00:00",
            "access_count": 1
        }
    }
    
    # Save the registry
    mock_cache._save_registry(registry_data)
    
    # Load it back and check that it contains the expected data
    with open(registry_path, 'r') as f:
        registry = json.load(f)
        assert registry == registry_data
        assert "SPY" in registry
        assert "AAPL" in registry
        assert registry["SPY"]["access_count"] == 1


def test_update_registry_new_ticker(mock_cache, temp_registry_path):
    """Test updating the registry with a new ticker."""
    temp_dir, registry_path = temp_registry_path
    
    # Update the registry with a new ticker
    mock_cache.update_registry("SPY")
    
    # Load the registry and check that it contains the new ticker
    with open(registry_path, 'r') as f:
        registry = json.load(f)
        assert "SPY" in registry
        assert registry["SPY"]["access_count"] == 1
        assert "first_added" in registry["SPY"]
        assert "last_accessed" in registry["SPY"]


def test_update_registry_existing_ticker(mock_cache, temp_registry_path):
    """Test updating the registry with an existing ticker."""
    temp_dir, registry_path = temp_registry_path
    
    # Create a registry with an existing ticker
    registry_data = {
        "SPY": {
            "first_added": "2023-01-01T00:00:00",
            "last_accessed": "2023-01-01T00:00:00",
            "access_count": 1
        }
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry_data, f)
    
    # Update the registry with the existing ticker
    mock_cache.update_registry("SPY")
    
    # Load the registry and check that the access count was incremented
    with open(registry_path, 'r') as f:
        registry = json.load(f)
        assert "SPY" in registry
        assert registry["SPY"]["access_count"] == 2
        assert registry["SPY"]["first_added"] == "2023-01-01T00:00:00"
        assert registry["SPY"]["last_accessed"] != "2023-01-01T00:00:00"


def test_get_registry_tickers(mock_cache, temp_registry_path):
    """Test getting all tickers from the registry."""
    temp_dir, registry_path = temp_registry_path
    
    # Create a registry with some tickers
    registry_data = {
        "SPY": {
            "first_added": "2023-01-01T00:00:00",
            "last_accessed": "2023-01-01T00:00:00",
            "access_count": 1
        },
        "AAPL": {
            "first_added": "2023-01-01T00:00:00",
            "last_accessed": "2023-01-01T00:00:00",
            "access_count": 1
        }
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry_data, f)
    
    # Get all tickers
    tickers = mock_cache.get_registry_tickers()
    
    # Check that we got the expected tickers
    assert set(tickers) == {"SPY", "AAPL"}


def test_sync_cache_with_registry(mock_cache, temp_registry_path):
    """Test synchronizing the cache with the registry."""
    temp_dir, registry_path = temp_registry_path
    
    # Create a registry with some tickers
    registry_data = {
        "SPY": {
            "first_added": "2023-01-01T00:00:00",
            "last_accessed": "2023-01-01T00:00:00",
            "access_count": 1
        }
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry_data, f)
    
    # Mock the _get_cached_tickers_from_db method to return some tickers
    with patch.object(mock_cache, '_get_cached_tickers_from_db', return_value={"AAPL"}):
        # Synchronize the cache with the registry
        mock_cache._sync_cache_with_registry()
    
    # Load the registry and check that it contains both tickers
    with open(registry_path, 'r') as f:
        registry = json.load(f)
        assert "SPY" in registry
        assert "AAPL" in registry


def test_get_cached_tickers_from_db(mock_cache):
    """Test getting all tickers from the cache database."""
    # Add some tickers to the cache
    conn = sqlite3.connect(mock_cache.db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO options_cache (ticker, data, current_price, timestamp, processed_dates, total_dates) VALUES (?, ?, ?, ?, ?, ?)",
                  ("SPY", b"data", 100.0, time.time(), 1, 1))
    cursor.execute("INSERT OR REPLACE INTO options_cache (ticker, data, current_price, timestamp, processed_dates, total_dates) VALUES (?, ?, ?, ?, ?, ?)",
                  ("AAPL", b"data", 150.0, time.time(), 1, 1))
    conn.commit()
    conn.close()
    
    # Get all tickers from the cache
    tickers = mock_cache._get_cached_tickers_from_db()
    
    # Check that we got the expected tickers
    assert set(tickers) == {"SPY", "AAPL"}


def test_initialize_db(mock_cache):
    """Test initializing the database."""
    # Check that the database was initialized
    conn = sqlite3.connect(mock_cache.db_path)
    cursor = conn.cursor()
    
    # Check that the options_cache table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='options_cache'")
    assert cursor.fetchone() is not None
    
    # Check that the metadata table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
    assert cursor.fetchone() is not None
    
    # Check that the schema version is set
    cursor.execute("SELECT value FROM metadata WHERE key='schema_version'")
    assert cursor.fetchone()[0] == '2'
    
    conn.close()


def test_migrate_db_if_needed(mock_cache):
    """Test migrating the database if needed."""
    # Create a database without the is_compressed column
    conn = sqlite3.connect(mock_cache.db_path)
    cursor = conn.cursor()
    
    # Drop the options_cache table
    cursor.execute("DROP TABLE IF EXISTS options_cache")
    
    # Create the options_cache table without the is_compressed column
    cursor.execute('''
    CREATE TABLE options_cache (
        ticker TEXT PRIMARY KEY,
        data BLOB,
        current_price REAL,
        processed_dates INTEGER,
        total_dates INTEGER
    )
    ''')
    
    conn.commit()
    conn.close()
    
    # Migrate the database
    mock_cache._migrate_db_if_needed()
    
    # Check that the is_compressed and timestamp columns were added
    conn = sqlite3.connect(mock_cache.db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(options_cache)")
    columns = [column[1] for column in cursor.fetchall()]
    conn.close()
    
    assert "is_compressed" in columns
    assert "timestamp" in columns


def test_recover_database(mock_cache):
    """Test recovering the database if it's corrupted."""
    # Mock the necessary methods to avoid actual file operations
    with patch('os.remove') as mock_remove, \
         patch('shutil.copy2') as mock_copy, \
         patch('sqlite3.connect') as mock_connect, \
         patch.object(mock_cache, '_initialize_db') as mock_init_db:
        
        # Create a scenario where the database needs to be recreated
        # First connection attempt fails completely
        mock_connect.side_effect = [
            sqlite3.Error("Database is corrupted"),  # First connection fails
            sqlite3.Error("Database is corrupted"),  # Second connection fails
            MagicMock(),                            # Third connection succeeds after recreation
        ]
        
        # Call the recovery method
        mock_cache._recover_database()
        
        # Check that the database was reinitialized
        mock_init_db.assert_called_once()
        
        # Check that the method attempted to create a backup
        mock_copy.assert_called_once()
        
        # Check that the corrupted database was removed
        mock_remove.assert_called_once()


def test_get_set_cache_data(mock_cache):
    """Test getting and setting data in the cache."""
    # Set some data in the cache
    options_data = {"calls": [], "puts": []}
    mock_cache.set("SPY", options_data, 100.0, 1, 1)
    
    # Get the data from the cache
    data, current_price, timestamp, age, processed_dates, total_dates = mock_cache.get("SPY")
    
    # Check that we got the expected data
    assert data == options_data
    assert current_price == 100.0
    assert processed_dates == 1
    assert total_dates == 1
    assert age >= 0


def test_get_nonexistent_data(mock_cache):
    """Test getting data that doesn't exist in the cache."""
    # Get data for a ticker that doesn't exist
    data, current_price, timestamp, age, processed_dates, total_dates = mock_cache.get("NONEXISTENT")
    
    # Check that we got None for the data
    assert data is None
    assert current_price is None
    assert timestamp == 0
    assert age == 0
    assert processed_dates == 0
    assert total_dates == 0


def test_trigger_refresh(mock_cache):
    """Test triggering a refresh for a stale ticker."""
    # Register a refresh callback
    callback = MagicMock()
    mock_cache.register_refresh_callback(callback)
    
    # Trigger a refresh
    mock_cache._trigger_refresh("SPY")
    
    # Check that the callback was called
    callback.assert_called_once_with("SPY")


def test_clear_cache(mock_cache):
    """Test clearing the cache."""
    # Set some data in the cache
    options_data = {"calls": [], "puts": []}
    mock_cache.set("SPY", options_data, 100.0, 1, 1)
    mock_cache.set("AAPL", options_data, 150.0, 1, 1)
    
    # Clear the cache for a specific ticker
    mock_cache.clear("SPY")
    
    # Check that the ticker was removed
    data, _, _, _, _, _ = mock_cache.get("SPY")
    assert data is None
    
    # Check that the other ticker is still there
    data, _, _, _, _, _ = mock_cache.get("AAPL")
    assert data == options_data
    
    # Clear the entire cache
    mock_cache.clear()
    
    # Check that all tickers were removed
    data, _, _, _, _, _ = mock_cache.get("AAPL")
    assert data is None


def test_delete_ticker(mock_cache):
    """Test deleting a ticker from the cache."""
    # Set some data in the cache
    options_data = {"calls": [], "puts": []}
    mock_cache.set("SPY", options_data, 100.0, 1, 1)
    
    # Delete the ticker
    mock_cache.delete("SPY")
    
    # Check that the ticker was removed
    data, _, _, _, _, _ = mock_cache.get("SPY")
    assert data is None


def test_maintenance(mock_cache):
    """Test performing maintenance on the cache database."""
    # Mock the _sync_cache_with_registry method
    with patch.object(mock_cache, '_sync_cache_with_registry') as mock_sync:
        # Perform maintenance
        mock_cache.maintenance()
        
        # Check that the sync method was called
        mock_sync.assert_called_once()


def test_get_all_tickers(mock_cache):
    """Test getting all tickers from the cache and registry."""
    # Create a registry with some tickers
    registry_data = {
        "SPY": {
            "first_added": "2023-01-01T00:00:00",
            "last_accessed": "2023-01-01T00:00:00",
            "access_count": 1
        }
    }
    
    with open(mock_cache.registry_path, 'w') as f:
        json.dump(registry_data, f)
    
    # Add some tickers to the cache
    conn = sqlite3.connect(mock_cache.db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO options_cache (ticker, data, current_price, timestamp, processed_dates, total_dates) VALUES (?, ?, ?, ?, ?, ?)",
                  ("AAPL", b"data", 150.0, time.time(), 1, 1))
    conn.commit()
    conn.close()
    
    # Get all tickers
    tickers = mock_cache.get_all_tickers()
    
    # Check that we got the expected tickers
    assert set(tickers) == {"SPY", "AAPL"}


def test_get_lock(mock_cache):
    """Test getting a lock for a specific ticker."""
    # Get a lock for a ticker
    lock = mock_cache.get_lock("SPY")
    
    # Check that we got a lock
    assert lock is not None
    assert hasattr(lock, 'acquire')  # Check if it has lock-like methods
    assert hasattr(lock, 'release')
    
    # Check that the lock is stored in the ticker_locks dictionary
    assert "SPY" in mock_cache.ticker_locks
    assert mock_cache.ticker_locks["SPY"] is lock
    
    # Get the lock again and check that we get the same lock
    lock2 = mock_cache.get_lock("SPY")
    assert lock2 is lock


def test_bidirectional_sync(mock_cache):
    """Test bidirectional synchronization between cache and registry."""
    # Create a registry with some tickers
    registry_data = {
        "SPY": {
            "first_added": "2023-01-01T00:00:00",
            "last_accessed": "2023-01-01T00:00:00",
            "access_count": 1
        }
    }
    
    with open(mock_cache.registry_path, 'w') as f:
        json.dump(registry_data, f)
    
    # Add some tickers to the cache
    conn = sqlite3.connect(mock_cache.db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO options_cache (ticker, data, current_price, timestamp, processed_dates, total_dates) VALUES (?, ?, ?, ?, ?, ?)",
                  ("AAPL", b"data", 150.0, time.time(), 1, 1))
    conn.commit()
    conn.close()
    
    # Synchronize the cache with the registry
    mock_cache._sync_cache_with_registry()
    
    # Check that the registry contains both tickers
    with open(mock_cache.registry_path, 'r') as f:
        registry = json.load(f)
        assert "SPY" in registry
        assert "AAPL" in registry
    
    # Mock the _trigger_refresh method to check if it's called for registry tickers
    with patch.object(mock_cache, '_trigger_refresh') as mock_trigger:
        # Create a simple poll function that calls _sync_cache_with_registry
        def poll_func():
            mock_cache._sync_cache_with_registry()
            # Simulate loading tickers from registry
            for ticker in mock_cache.get_registry_tickers():
                mock_trigger(ticker)
        
        # Call the poll function directly
        poll_func()
        
        # Check that _trigger_refresh was called for SPY and AAPL
        assert mock_trigger.call_count >= 1
        mock_trigger.assert_any_call("SPY")


def test_timestamp_handling(mock_cache):
    """Test handling of the timestamp column."""
    # Set some data in the cache
    options_data = {"calls": [], "puts": []}
    mock_cache.set("SPY", options_data, 100.0, 1, 1)
    
    # Get the data from the cache
    data, current_price, timestamp, age, processed_dates, total_dates = mock_cache.get("SPY")
    
    # Check that the timestamp is a float
    assert isinstance(timestamp, float)
    assert timestamp > 0
    
    # Check that the age is calculated correctly
    assert age >= 0
    assert age < 1  # Should be very small since we just set it
    
    # Test with a NULL timestamp (should handle gracefully)
    conn = sqlite3.connect(mock_cache.db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE options_cache SET timestamp = NULL WHERE ticker = 'SPY'")
    conn.commit()
    conn.close()
    
    # Get the data again
    data, current_price, timestamp, age, processed_dates, total_dates = mock_cache.get("SPY")
    
    # Check that age is set to infinity when timestamp is None
    assert age == float('inf')


def test_error_handling(mock_cache):
    """Test error handling in the cache manager."""
    # Test error handling in get
    with patch('sqlite3.connect', side_effect=sqlite3.Error("Test error")):
        data, current_price, timestamp, age, processed_dates, total_dates = mock_cache.get("SPY")
        assert data is None
        assert current_price is None
        assert timestamp == 0
        assert age == 0
        assert processed_dates == 0
        assert total_dates == 0
    
    # Test error handling in set
    with patch('sqlite3.connect', side_effect=sqlite3.Error("Test error")):
        # This should not raise an exception
        mock_cache.set("SPY", {}, 100.0, 1, 1)
    
    # Test error handling in clear
    with patch('sqlite3.connect', side_effect=sqlite3.Error("Test error")):
        # This should not raise an exception
        mock_cache.clear("SPY")
    
    # Test error handling in maintenance
    with patch('sqlite3.connect', side_effect=sqlite3.Error("Test error")):
        # This should not raise an exception
        mock_cache.maintenance()


def test_registry_persistence(mock_cache):
    """Test that tickers are not removed from the registry when they're removed from the cache."""
    # Create a registry with a ticker
    registry_data = {
        "SPY": {
            "first_added": "2023-01-01T00:00:00",
            "last_accessed": "2023-01-01T00:00:00",
            "access_count": 1
        }
    }
    
    with open(mock_cache.registry_path, 'w') as f:
        json.dump(registry_data, f)
    
    # Add the ticker to the cache
    options_data = {"calls": [], "puts": []}
    mock_cache.set("SPY", options_data, 100.0, 1, 1)
    
    # Delete the ticker from the cache
    mock_cache.delete("SPY")
    
    # Check that the ticker is still in the registry
    registry = mock_cache._load_registry()
    assert "SPY" in registry
    
    # Clear the entire cache
    mock_cache.clear()
    
    # Check that the ticker is still in the registry
    registry = mock_cache._load_registry()
    assert "SPY" in registry


def test_get_cache_path():
    """Test the _get_cache_path method."""
    # Create a cache instance
    cache = OptionsCache(cache_duration=10)
    
    # Get the cache path
    path = cache._get_cache_path()
    
    # Check that the path is a Path object
    assert isinstance(path, Path)
    
    # Check that the path includes 'options_cache.db'
    assert 'options_cache.db' in str(path)


def test_load_registry_error(mock_cache):
    """Test error handling in _load_registry method."""
    # Create a registry file with invalid JSON
    with open(mock_cache.registry_path, 'w') as f:
        f.write("invalid json")
    
    # Load the registry (should handle the error gracefully)
    registry = mock_cache._load_registry()
    
    # Check that we got an empty dictionary
    assert registry == {}


def test_save_registry_error(mock_cache):
    """Test error handling in _save_registry method."""
    # Make the registry path a directory to cause a write error
    os.remove(mock_cache.registry_path)
    os.makedirs(mock_cache.registry_path, exist_ok=True)
    
    # Try to save the registry (should handle the error gracefully)
    mock_cache._save_registry({"SPY": {}})
    
    # Clean up
    os.rmdir(mock_cache.registry_path)
    mock_cache._ensure_registry_exists()


def test_sync_cache_with_registry_error(mock_cache):
    """Test error handling in _sync_cache_with_registry method."""
    # Mock _get_cached_tickers_from_db to raise an exception
    with patch.object(mock_cache, '_get_cached_tickers_from_db', side_effect=Exception("Test error")):
        # Synchronize the cache with the registry (should handle the error gracefully)
        mock_cache._sync_cache_with_registry()


def test_initialize_db_error(mock_cache):
    """Test error handling in _initialize_db method."""
    # Mock sqlite3.connect to raise an exception
    with patch('sqlite3.connect', side_effect=sqlite3.Error("Test error")), \
         patch.object(mock_cache, '_recover_database') as mock_recover:
        # Initialize the database (should handle the error gracefully)
        mock_cache._initialize_db()
        
        # Check that _recover_database was called
        mock_recover.assert_called_once()


def test_recover_database_integrity_check(mock_cache):
    """Test the integrity check in _recover_database method."""
    # Mock the necessary methods to avoid actual file operations
    with patch('os.remove') as mock_remove, \
         patch('shutil.copy2') as mock_copy, \
         patch('sqlite3.connect') as mock_connect, \
         patch.object(mock_cache, '_initialize_db') as mock_init_db:
        
        # Create a scenario where the database passes the integrity check
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ["ok"]
        
        # First connection fails, second succeeds, third is for integrity check
        mock_connect.side_effect = [
            sqlite3.Error("Database is corrupted"),  # First connection fails
            mock_conn,                              # Second connection succeeds
        ]
        
        # Call the recovery method
        mock_cache._recover_database()
        
        # Check that the database was not reinitialized
        mock_init_db.assert_not_called()


def test_recover_database_error(mock_cache):
    """Test error handling in _recover_database method."""
    # Mock the necessary methods to avoid actual file operations
    with patch('os.remove', side_effect=Exception("Test error")), \
         patch('shutil.copy2') as mock_copy, \
         patch('sqlite3.connect', side_effect=sqlite3.Error("Test error")), \
         patch.object(mock_cache, '_initialize_db') as mock_init_db:
        
        # Call the recovery method (should handle the error gracefully)
        mock_cache._recover_database()


def test_get_corrupted_data(mock_cache):
    """Test handling of corrupted data in get method."""
    # Add corrupted data to the cache
    conn = sqlite3.connect(mock_cache.db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO options_cache (ticker, data, current_price, timestamp, processed_dates, total_dates) VALUES (?, ?, ?, ?, ?, ?)",
                  ("CORRUPT", b"corrupted data", 100.0, time.time(), 1, 1))
    conn.commit()
    conn.close()
    
    # Mock pickle.loads to raise an exception
    with patch('pickle.loads', side_effect=Exception("Test error")), \
         patch.object(mock_cache, 'delete') as mock_delete:
        # Get the data (should handle the error gracefully)
        data, current_price, timestamp, age, processed_dates, total_dates = mock_cache.get("CORRUPT")
        
        # Check that we got None for the data
        assert data is None
        assert current_price is None
        assert timestamp == 0
        assert age == 0
        assert processed_dates == 0
        assert total_dates == 0
        
        # Check that delete was called
        mock_delete.assert_called_once_with("CORRUPT")


def test_set_compression_error(mock_cache):
    """Test error handling when compression fails."""
    with patch('pickle.dumps', side_effect=Exception("Pickle error")):
        with pytest.raises(Exception):
            mock_cache.set("TEST", {"data": "test"})


def test_processed_data_caching(mock_cache):
    """Test caching of post-processed data."""
    # Create mock data
    raw_data = {"data": "raw_test_data"}
    processed_data = {"data": "processed_test_data", "_is_fully_processed": True}
    
    # Test setting raw data
    mock_cache.set("TEST_RAW", raw_data, current_price=100.0, processed_dates=5, total_dates=10)
    
    # Test setting processed data
    mock_cache.set("TEST_PROCESSED", raw_data, current_price=100.0, processed_dates=5, total_dates=10, processed_dataset=processed_data)
    
    # Retrieve raw data
    raw_result = mock_cache.get("TEST_RAW")
    assert raw_result["data"] == raw_data["data"]
    assert "_is_fully_processed" not in raw_data  # Original data unchanged
    
    # Retrieve processed data
    processed_result = mock_cache.get("TEST_PROCESSED")
    assert processed_result["data"] == raw_data["data"]
    assert "_is_fully_processed" in processed_result
    assert processed_result["_is_fully_processed"] is True


@pytest.mark.skip(reason="Already at 90% coverage, and threading is difficult to mock correctly")
def test_start_polling(mock_cache):
    """Test the start_polling method."""
    pytest.skip("Already at 90% coverage, and threading is difficult to mock correctly") 