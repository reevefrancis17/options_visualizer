#!/usr/bin/env python3
import os
import sys
import time
import pickle
import logging
import json
import math
from datetime import datetime
from flask import Flask, jsonify, request, Response
import pandas as pd
import numpy as np
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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
CACHE_DURATION = 10 * 60  # 10 minutes in seconds
BATCH_SIZE = 5  # Number of tickers to refresh in a batch

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Initialize the options data manager - will be replaced with shared instance
data_manager = None

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
            from python.options_data import OptionsDataManager
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

# API endpoint to get options data for a ticker
@app.route('/api/options/<ticker>', methods=['GET'])
def get_options(ticker):
    """Get options data for a ticker."""
    try:
        ticker = ticker.upper()
        logger.info(f"API request for options data: {ticker}")
        
        # Check if we should skip interpolation (faster but less accurate)
        # Default to False (do interpolation) for better data quality
        skip_interpolation = request.args.get('skip_interpolation', 'false').lower() == 'true'
        
        # Get the data manager
        manager = get_data_manager()
        
        # Get data from cache
        processor, current_price, status, progress = manager.get_current_processor(ticker)
        
        if status == 'not_found':
            # Start fetching in background with interpolation enabled by default for cache
            # but respect skip_interpolation for immediate use
            manager.start_fetching(ticker, skip_interpolation=skip_interpolation)
            return jsonify({
                'status': 'loading',
                'message': f'Fetching data for {ticker}',
                'ticker': ticker,
                'progress': 0
            }), 202  # Accepted
        
        # Get the data frame
        df = processor.get_data_frame() if processor else None
        
        # If the processor exists but the DataFrame is None or empty, 
        # it means the data is still being processed
        if processor and (df is None or df.empty):
            # Data is being processed, return loading status
            return jsonify({
                'status': 'loading',
                'message': f'Processing data for {ticker}',
                'ticker': ticker,
                'progress': progress
            }), 202  # Accepted
        
        # If we have no data at all, return an error
        if df is None or df.empty:
            return jsonify({
                'status': 'error',
                'message': f'No options data available for {ticker}'
            }), 404
        
        # Get expiration dates
        expiry_dates = processor.get_expirations()
        expiry_dates_str = [date.strftime('%Y-%m-%d') for date in expiry_dates]
        
        # Convert datetime columns to strings
        df['expiration'] = df['expiration'].dt.strftime('%Y-%m-%d')
        
        # Replace NaN values with None using multiple approaches to ensure all NaNs are caught
        df = df.replace({np.nan: None})
        df = df.where(pd.notnull(df), None)
        
        # Additional check for object columns that might contain 'NaN' strings
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].replace('NaN', None)
        
        # Additional check for numeric columns to ensure NaN values are properly handled
        for col in df.select_dtypes(include=['float', 'int']).columns:
            df[col] = df[col].apply(lambda x: None if pd.isna(x) or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else x)
        
        # Convert to records - use a more efficient approach for large datasets
        if len(df) > 10000:
            # For very large datasets, only include essential columns
            essential_cols = [
                'strike', 'expiration', 'option_type', 'bid', 'ask', 'mid_price', 
                'price', 'intrinsic_value', 'extrinsic_value', 'impliedVolatility',
                'volume', 'openInterest', 'DTE'
            ]
            df = df[[col for col in essential_cols if col in df.columns]]
        
        data_records = df.to_dict(orient='records')
        
        # Prepare response
        response = {
            'status': status,
            'ticker': ticker,
            'current_price': current_price,
            'expiry_dates': expiry_dates_str,
            'options_data': data_records,
            'processed_dates': len(expiry_dates),
            'total_dates': len(expiry_dates),
            'progress': progress,
            'using_cached_data': status != 'loading'
        }
        
        # If status is 'partial', it means we're using cached data but also refreshing in background
        if status == 'partial':
            response['message'] = 'Using cached data while refreshing in background'
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing request for {ticker}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

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

# Only run the app if this file is executed directly
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(debug=True, host='0.0.0.0', port=port) 