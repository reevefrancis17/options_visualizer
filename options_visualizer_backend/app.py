#!/usr/bin/env python3
import os
import sys
import time
import pickle
import logging
import json
import math
import concurrent.futures
from datetime import datetime
from flask import Flask, jsonify, request, Response
import pandas as pd
import numpy as np
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from flask_cors import CORS
import traceback
import atexit

# Import local modules
from options_visualizer_backend.options_data import OptionsDataManager
from options_visualizer_backend.config import PORT, DEBUG, HOST, CACHE_DURATION, MAX_WORKERS, REQUEST_TIMEOUT

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/server.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NaN values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return super().default(obj)

# Initialize Flask app
app = Flask(__name__)
app.json_encoder = CustomJSONEncoder  # Use our custom encoder
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for all API routes

# Configuration
CACHE_DIR = 'cache'
DATA_DIR = 'data'
CSV_PATH = os.path.join(DATA_DIR, 'tickers.csv')
CACHE_DURATION = CACHE_DURATION  # 10 minutes in seconds
BATCH_SIZE = 5  # Number of tickers to refresh in a batch

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Initialize the options data manager - will be replaced with shared instance
data_manager = None

# Initialize thread pool for concurrent request handling
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
logger.info(f"Initialized thread pool with {MAX_WORKERS} workers")

# Dictionary to store futures for potential cancellation
futures = {}

# Load ticker list from CSV
def load_tickers():
    try:
        df = pd.read_csv(CSV_PATH)
        return df.set_index('ticker')['timestamp'].to_dict()
    except FileNotFoundError:
        logger.error(f"CSV file not found at {CSV_PATH}")
        return {}

# Save ticker list to CSV
def save_tickers(tickers):
    try:
        df = pd.DataFrame(
            [{'ticker': ticker, 'timestamp': timestamp} for ticker, timestamp in tickers.items()]
        )
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        logger.info(f"Saved {len(tickers)} tickers to {CSV_PATH}")
    except Exception as e:
        logger.error(f"Error saving tickers to CSV: {str(e)}")

# Get the shared options data manager
def get_data_manager():
    global data_manager
    if data_manager is None:
        try:
            # Try to import from main.py
            from main import options_data_manager
            data_manager = options_data_manager
            logger.info("Using shared options data manager from main.py")
        except ImportError:
            # Fall back to creating a new instance
            data_manager = OptionsDataManager(cache_duration=CACHE_DURATION)
            logger.info("Created new options data manager (not running from main.py)")
    return data_manager

# Background job to refresh the cache for all tickers
def refresh_all_tickers():
    """Refresh all tickers in the cache in batches."""
    try:
        logger.info("Starting scheduled cache refresh for all tickers")
        
        # Get the data manager
        manager = get_data_manager()
        
        # Get all tickers from the cache
        tickers = manager._cache.get_all_tickers()
        
        # Also get tickers from CSV
        csv_tickers = load_tickers()
        
        # Combine both sources
        all_tickers = list(set(tickers) | set(csv_tickers.keys()))
        
        if not all_tickers:
            logger.info("No tickers to refresh")
            return
            
        logger.info(f"Refreshing {len(all_tickers)} tickers in batches of {BATCH_SIZE}")
        
        # Process tickers in batches to avoid overwhelming the API
        for i in range(0, len(all_tickers), BATCH_SIZE):
            batch = all_tickers[i:i+BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}: {batch}")
            
            # Start fetching each ticker in the batch
            for ticker in batch:
                try:
                    # Always use full interpolation for cached data
                    # This ensures the cache contains fully processed and interpolated data
                    manager.start_fetching(ticker, skip_interpolation=False)
                except Exception as e:
                    logger.error(f"Error refreshing {ticker}: {str(e)}")
            
            # Wait a bit between batches to avoid rate limiting
            if i + BATCH_SIZE < len(all_tickers):
                time.sleep(5)
        
        logger.info("Completed scheduled cache refresh with fully interpolated data")
    except Exception as e:
        logger.error(f"Error in refresh_all_tickers: {str(e)}")

# Function to process options data request in a separate thread
def process_options_request(ticker, dte_min=None, dte_max=None, fields=None):
    """Process an options data request in a separate thread."""
    try:
        ticker = ticker.upper()
        logger.info(f"Processing options data request for {ticker} in thread")
        
        # Get specific fields if requested
        field_list = fields.split(',') if fields else None
        
        # Skip interpolation is no longer supported - always use 2D interpolation
        skip_interpolation = False
        
        # Get the data manager
        manager = get_data_manager()
        
        # Get data from cache or start fetching if not available
        # The get_current_processor method now handles triggering background refreshes for stale data
        processor, current_price, status, progress = manager.get_current_processor(ticker)
        
        # Handle different status cases
        if status == 'error':
            # Check if we have any error information in the cache
            cached_data = manager._cache.get(ticker)
            error_message = "Unknown error occurred"
            
            if cached_data and cached_data[0] and isinstance(cached_data[0], dict) and '_error' in cached_data[0]:
                error_message = cached_data[0].get('_error', error_message)
            
            return {
                'status': 'error',
                'message': f'Error fetching data for {ticker}: {error_message}',
                'ticker': ticker,
                'error': error_message
            }, 500
        
        # If no processor is available yet, it means we're still loading the data
        if processor is None:
            # Ensure progress is a valid number between 0 and 1
            valid_progress = 0.0
            if isinstance(progress, (int, float)) and not math.isnan(progress) and not math.isinf(progress):
                valid_progress = max(0.0, min(1.0, progress))
                
            return {
                'status': 'loading',
                'message': f'Fetching data for {ticker}',
                'ticker': ticker,
                'progress': valid_progress
            }, 202  # Accepted
        
        # Get the data frame
        df = processor.get_data_frame() if processor else None
        
        # If the processor exists but the DataFrame is None or empty, 
        # it means the data is still being processed
        if df is None or df.empty:
            # Data is being processed, return loading status
            return {
                'status': 'loading',
                'message': f'Processing data for {ticker}',
                'ticker': ticker,
                'progress': progress
            }, 202  # Accepted
        
        # Apply DTE filters if provided
        if dte_min is not None:
            try:
                dte_min = float(dte_min)
                df = df[df['DTE'] >= dte_min]
            except (ValueError, TypeError) as e:
                return {
                    'status': 'error',
                    'message': f'Invalid dte_min parameter: {str(e)}',
                    'ticker': ticker
                }, 400  # Bad Request
        
        if dte_max is not None:
            try:
                dte_max = float(dte_max)
                df = df[df['DTE'] <= dte_max]
            except (ValueError, TypeError) as e:
                return {
                    'status': 'error',
                    'message': f'Invalid dte_max parameter: {str(e)}',
                    'ticker': ticker
                }, 400  # Bad Request
        
        # Get unique expiration dates
        expiry_dates = sorted(df['expiration'].unique().tolist())
        
        # Process the DataFrame to handle NaN values
        # Convert datetime columns to string
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d')
        
        # Replace NaN values with None for JSON serialization
        # For object columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].where(pd.notna(df[col]), None)
        
        # For numeric columns
        for col in df.select_dtypes(include=['float', 'int']).columns:
            df[col] = df[col].where(pd.notna(df[col]), None)
        
        # Convert to records
        options_data = df.to_dict('records')
        
        # Replace NaN values with None for JSON serialization
        for record in options_data:
            for key, value in list(record.items()):
                # Check for NaN or infinity in float values
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    record[key] = None
        
        # For large datasets, only include requested fields to reduce payload size
        if field_list and len(options_data) > 100:
            logger.info(f"Filtering large dataset to include only requested fields: {field_list}")
            # Always include these essential fields
            essential_fields = ['strike', 'expiration', 'option_type', 'DTE']
            fields_to_include = list(set(field_list + essential_fields))
            
            # Filter the records to only include requested fields
            options_data = [{k: v for k, v in record.items() if k in fields_to_include} for record in options_data]
        
        # Round numeric values to reduce payload size
        if len(options_data) > 100:
            logger.info("Rounding numeric values to reduce payload size")
            for record in options_data:
                for key, value in record.items():
                    if isinstance(value, float):
                        # Round to 4 decimal places for most values
                        if key in ['delta', 'gamma', 'theta', 'impliedVolatility']:
                            record[key] = round(value, 4)
                        else:
                            # Round to 2 decimal places for price values
                            record[key] = round(value, 2)
        
        # Prepare the response
        response = {
            'status': status,
            'ticker': ticker,
            'current_price': current_price,
            'expiry_dates': expiry_dates,
            'options_data': options_data
        }
        
        return response, 200
    except Exception as e:
        logger.error(f"Error processing options data for {ticker} in thread: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'message': f'Error processing data for {ticker}: {str(e)}',
            'ticker': ticker,
            'error': str(e)
        }, 500

# API endpoint to get options data for a ticker
@app.route('/api/options/<ticker>', methods=['GET'])
def get_options(ticker):
    """Get options data for a ticker."""
    try:
        ticker = ticker.upper()
        logger.info(f"API request for options data: {ticker}")
        
        # Check for query parameters
        dte_min = request.args.get('dte_min', None)
        dte_max = request.args.get('dte_max', None)
        fields = request.args.get('fields', None)
        
        # Generate a unique request ID
        request_id = f"{ticker}_{time.time()}"
        
        # Submit the request to the thread pool
        future = thread_pool.submit(
            process_options_request,
            ticker,
            dte_min,
            dte_max,
            fields
        )
        
        # Store the future for potential cancellation
        futures[request_id] = future
        
        # Wait for the result with a timeout
        try:
            response, status_code = future.result(timeout=REQUEST_TIMEOUT)  # 30 second timeout
            
            # Remove the future from the dictionary
            if request_id in futures:
                del futures[request_id]
            
            return jsonify(response), status_code
        except concurrent.futures.TimeoutError:
            # If the request times out, return a timeout response
            # but keep the future running in the background
            logger.warning(f"Request timeout for {ticker}, continuing in background")
            return jsonify({
                'status': 'loading',
                'message': f'Request timeout for {ticker}, continuing in background',
                'ticker': ticker,
                'progress': 0.0
            }), 202  # Accepted
    except Exception as e:
        logger.error(f"Error handling options request for {ticker}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'status': 'error'}), 500

# API endpoint to get available tickers
@app.route('/api/tickers', methods=['GET'])
def get_tickers():
    """Get list of available tickers."""
    try:
        # Get all tickers from the cache
        manager = get_data_manager()
        tickers = manager._cache.get_all_tickers()
        
        # Load additional tickers from CSV
        csv_tickers = load_tickers()
        
        # Combine both sources
        all_tickers = set(tickers) | set(csv_tickers.keys())
        
        return jsonify({
            'tickers': sorted(list(all_tickers))
        })
    except Exception as e:
        logger.error(f"Error getting tickers: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# API endpoint to add a ticker
@app.route('/api/tickers/add', methods=['POST'])
def add_ticker():
    """Add a ticker to the watchlist."""
    try:
        data = request.get_json()
        if not data or 'ticker' not in data:
            return jsonify({'error': 'No ticker provided'}), 400
        
        ticker = data['ticker'].strip().upper()
        if not ticker:
            return jsonify({'error': 'Invalid ticker format'}), 400
        
        logger.info(f"Adding ticker to watchlist: {ticker}")
        
        # Get the data manager
        manager = get_data_manager()
        
        # Start fetching data for this ticker
        manager.start_fetching(ticker)
        
        # Add to CSV
        tickers = load_tickers()
        tickers[ticker] = datetime.now().isoformat()
        save_tickers(tickers)
        
        return jsonify({
            'status': 'success',
            'message': f'Added {ticker} to watchlist'
        })
    except Exception as e:
        logger.error(f"Error adding ticker: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# API endpoint to remove a ticker
@app.route('/api/tickers/remove', methods=['POST'])
def remove_ticker():
    """Remove a ticker from the watchlist."""
    try:
        data = request.get_json()
        if not data or 'ticker' not in data:
            return jsonify({'error': 'No ticker provided'}), 400
        
        ticker = data['ticker'].strip().upper()
        if not ticker:
            return jsonify({'error': 'Invalid ticker format'}), 400
        
        logger.info(f"Removing ticker from watchlist: {ticker}")
        
        # Remove from CSV
        tickers = load_tickers()
        if ticker in tickers:
            del tickers[ticker]
            save_tickers(tickers)
        
        return jsonify({
            'status': 'success',
            'message': f'Removed {ticker} from watchlist'
        })
    except Exception as e:
        logger.error(f"Error removing ticker: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# API endpoint to get cache stats
@app.route('/api/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics."""
    try:
        manager = get_data_manager()
        stats = manager._cache.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'entries': 0,
            'data_size_mb': 0,
            'db_size_mb': 0,
            'wal_size_mb': 0,
            'compressed_entries': 0,
            'compression_ratio': "0/0",
            'oldest_entry': 'N/A',
            'newest_entry': 'N/A',
            'cached_tickers': []
        }), 200  # Return 200 even on error to avoid breaking the UI

# API endpoint to manually refresh the cache
@app.route('/api/cache/refresh', methods=['POST'])
def refresh_cache():
    """Manually trigger a cache refresh."""
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        
        # Check if we should skip interpolation (faster but less accurate)
        # This only affects immediate use, not the cached data which will always be fully interpolated
        skip_interpolation = data.get('skip_interpolation', False)
        
        if ticker:
            # Refresh specific ticker
            ticker = ticker.strip().upper()
            logger.info(f"Manually refreshing cache for {ticker} with skip_interpolation={skip_interpolation} (for immediate use only)")
            
            manager = get_data_manager()
            manager.start_fetching(ticker, skip_interpolation=skip_interpolation)
            
            return jsonify({
                'status': 'success',
                'message': f'Started refreshing cache for {ticker} (cached data will be fully interpolated)'
            })
        else:
            # Refresh all tickers
            logger.info("Manually refreshing cache for all tickers (all cached data will be fully interpolated)")
            refresh_all_tickers()
            
            return jsonify({
                'status': 'success',
                'message': 'Started refreshing cache for all tickers (all cached data will be fully interpolated)'
            })
    except Exception as e:
        logger.error(f"Error refreshing cache: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

# Start the scheduler for background tasks
scheduler = BackgroundScheduler()
scheduler.add_job(refresh_all_tickers, 'interval', minutes=30, id='refresh_all_tickers')
scheduler.start()
logger.info("Started background scheduler for cache refresh")

# Add a cleanup function to properly shutdown the thread pool when the application exits
def cleanup():
    """Cleanup function to shutdown thread pools and other resources."""
    logger.info("Shutting down thread pool...")
    thread_pool.shutdown(wait=False)
    
    # Shutdown the data manager if it exists
    if data_manager is not None:
        logger.info("Shutting down data manager...")
        if hasattr(data_manager, 'shutdown'):
            data_manager.shutdown()
    
    logger.info("Cleanup complete")

# Register the cleanup function to be called when the application exits
atexit.register(cleanup)

# Only run the app if this file is executed directly
if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Start the server
    logger.info(f"Starting backend API server on port {PORT}")
    app.run(debug=DEBUG, host=HOST, port=PORT) 