import os
import sys
import json
import logging
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import pandas as pd
import math
from flask_cors import CORS
import traceback

# Import the existing modules
from options_visualizer_backend.options_data import OptionsDataManager, OptionsDataProcessor
from options_visualizer_web.config import PORT, DEBUG, HOST, BACKEND_URL, LOG_DIR

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'debug', 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Clear logs on startup
log_file = os.path.join(LOG_DIR, 'web_app.log')
if os.path.exists(log_file):
    with open(log_file, 'w') as f:
        f.write(f"=== New session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

# Configure console handler in addition to file handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Configure root logger
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

logger = logging.getLogger(__name__)
logger.info("Starting Options Visualizer Web App")

# Custom JSON encoder to handle NaN values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            logger.debug(f"Converting NaN/Inf value to None in JSON encoder")
            return None
        return super().default(obj)

# Create Flask app with explicit static folder path
static_folder = os.path.join(os.path.dirname(__file__), 'static')
template_folder = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, 
           static_folder=static_folder, 
           static_url_path='/static',
           template_folder=template_folder)
app.json_encoder = CustomJSONEncoder  # Use our custom encoder

# Enable CORS for all routes
CORS(app)

logger.info(f"Static folder: {static_folder}")
logger.info(f"Template folder: {template_folder}")

# Initialize the options data manager with a longer cache duration for web app
# This will be overridden by the shared instance from main.py
data_manager = None

# Backend API URL
BACKEND_API_URL = BACKEND_URL

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/get_options_data', methods=['POST'])
def get_options_data():
    """API endpoint to fetch options data for a given symbol with immediate cache response"""
    try:
        # Get the symbol from the request
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({'error': 'No symbol provided'}), 400
        
        symbol = data['symbol'].strip().upper()
        if not symbol or not symbol.isalnum():
            return jsonify({'error': 'Invalid symbol format'}), 400
        
        logger.info(f"Processing request for {symbol}")
        
        # Use the shared data manager from main.py if available
        global data_manager
        if data_manager is None:
            # Import the shared instance from main.py
            try:
                from main import options_data_manager
                data_manager = options_data_manager
                logger.info("Using shared options data manager from main.py")
            except ImportError:
                # Fall back to creating a new instance if not running from main.py
                data_manager = OptionsDataManager(cache_duration=600)
                logger.info("Created new options data manager (not running from main.py)")
        
        # Try to get data from backend first
        try:
            backend_response = requests.get(f"{BACKEND_API_URL}/api/options/{symbol}", timeout=5)
            if backend_response.status_code == 200:
                # Verify we got JSON response
                content_type = backend_response.headers.get('Content-Type', '')
                if 'application/json' not in content_type:
                    logger.error(f"Backend returned non-JSON response: {content_type}")
                    logger.error(f"Response preview: {backend_response.text[:100]}...")
                    raise ValueError(f"Backend returned non-JSON response: {content_type}")
                
                logger.info(f"Successfully retrieved data from backend for {symbol}")
                return jsonify(backend_response.json())
        except requests.RequestException as e:
            logger.warning(f"Failed to connect to backend: {str(e)}. Falling back to local data.")
        except ValueError as e:
            logger.error(f"Error processing backend response: {str(e)}. Falling back to local data.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse backend JSON response: {str(e)}. Falling back to local data.")
        
        # If backend request fails, fall back to local data processing
        # Get current processor status from cache
        processor, current_price, status, progress = data_manager.get_current_processor(symbol)
        
        # If no data in cache, start background fetch and return loading status
        if status == 'not_found':
            data_manager.start_fetching(symbol)
            
            # Check if data is already available after starting the fetch
            processor, current_price, status, progress = data_manager.get_current_processor(symbol)
            
            # If we still don't have data, return a loading status
            if status == 'not_found' or processor is None:
                response = {
                    'status': 'loading',
                    'symbol': symbol,
                    'progress': 0,
                    'options_data': [],
                    'current_price': None,
                    'expiry_dates': [],
                    'processed_dates': 0,
                    'total_dates': 0
                }
                logger.info(f"Started background fetch for {symbol}, no data available yet")
                return jsonify(response)
            
            # If we have partial data already, continue with that
            logger.info(f"Started background fetch for {symbol}, partial data already available")
        
        # We have data in cache (partial or complete)
        # Get the dataset as a DataFrame
        df = processor.get_data_frame()
        if df is None or df.empty:
            return jsonify({'error': f'No options data available for {symbol}'}), 404
        
        # Get all expiration dates
        expiry_dates = processor.get_expirations()
        if not expiry_dates:
            return jsonify({'error': f'No expiration dates found for {symbol}'}), 404
        
        # Convert expiry dates to string format for JSON
        expiry_dates_str = [date.strftime('%Y-%m-%d') for date in expiry_dates]
        
        # Convert datetime columns to strings
        df['expiration'] = df['expiration'].dt.strftime('%Y-%m-%d')
        
        # Replace NaN values with None (which will be converted to null in JSON)
        # Use multiple approaches to ensure all NaNs are caught
        df = df.replace({np.nan: None})
        df = df.where(pd.notnull(df), None)
        
        # Additional check for object columns that might contain 'NaN' strings
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].replace('NaN', None)
        
        # Additional check for numeric columns to ensure NaN values are properly handled
        for col in df.select_dtypes(include=['float', 'int']).columns:
            df[col] = df[col].apply(lambda x: None if pd.isna(x) or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else x)
        
        # Convert to records and check for any remaining NaN values
        data_records = df.to_dict(orient='records')
        
        # Log a sample of the data for debugging
        if data_records and len(data_records) > 0:
            sample = {k: v for k, v in list(data_records[0].items())[:5]}
            logger.info(f"Sample data record (first 5 fields): {sample}")
        
        # Prepare the response
        response = {
            'status': status,
            'symbol': symbol,
            'progress': progress,
            'current_price': current_price,
            'expiry_dates': expiry_dates_str,
            'options_data': data_records,
            'min_strike': processor.min_strike,
            'max_strike': processor.max_strike,
            'processed_dates': len(expiry_dates),
            'total_dates': data_manager._loading_state.get(symbol, {}).get('total_dates', len(expiry_dates))
        }
        
        logger.info(f"Returning {status} data for {symbol} with {len(expiry_dates)} dates")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/options/<ticker>', methods=['GET'])
def get_options_data_by_ticker(ticker):
    """
    API endpoint to get options data for a specific ticker.
    This endpoint is used by the frontend to fetch options data.
    """
    try:
        logger.info(f"Received request for options data for ticker: {ticker}")
        
        # Initialize the data manager if not already done
        if not hasattr(app, 'data_manager'):
            logger.info("Creating new OptionsDataManager instance")
            app.data_manager = OptionsDataManager()
        
        # Get the options data
        data_manager = app.data_manager
        
        # Check if we have data for this ticker
        logger.info(f"Getting current processor for {ticker}")
        processor, current_price, status, progress = data_manager.get_current_processor(ticker)
        logger.info(f"Status for {ticker}: {status}, progress: {progress}, processor: {processor is not None}, price: {current_price}")
        
        # If no data or still loading
        if status == 'loading' or not processor:
            # Start loading data in the background if not already loading
            if status != 'loading':
                logger.info(f"Starting fetch for {ticker}")
                data_manager.start_fetching(ticker)
            
            # Get loading state
            loading_state = data_manager._loading_state.get(ticker, {})
            processed_dates = loading_state.get('processed_dates', 0)
            total_dates = loading_state.get('total_dates', 0)
            logger.info(f"Loading state for {ticker}: processed_dates={processed_dates}, total_dates={total_dates}")
            
            # Return a loading status
            return jsonify({
                'status': 'loading',
                'progress': progress,
                'processed_dates': processed_dates,
                'total_dates': total_dates,
                'message': f'Loading data for {ticker}...'
            })
        
        # If we have partial data
        if status == 'partial':
            # Get the data
            logger.info(f"Getting partial data for {ticker}")
            options_data = processor.get_data()
            expiry_dates = processor.get_expirations()
            logger.info(f"Got {len(expiry_dates)} expiry dates and options data: {options_data is not None}")
            
            # Get loading state
            loading_state = data_manager._loading_state.get(ticker, {})
            processed_dates = loading_state.get('processed_dates', 0)
            total_dates = loading_state.get('total_dates', 0)
            
            return jsonify({
                'status': 'partial',
                'symbol': ticker,
                'current_price': current_price,
                'expiry_dates': expiry_dates,
                'options_data': options_data,
                'processed_dates': processed_dates,
                'total_dates': total_dates
            })
        
        # Return complete data
        logger.info(f"Getting complete data for {ticker}")
        options_data = processor.get_data()
        expiry_dates = processor.get_expirations()
        logger.info(f"Got {len(expiry_dates)} expiry dates and options data: {options_data is not None}")
        
        return jsonify({
            'status': 'complete',
            'symbol': ticker,
            'current_price': current_price,
            'expiry_dates': expiry_dates,
            'options_data': options_data
        })
    
    except Exception as e:
        logger.error(f"Error processing request for ticker {ticker}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Add health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# Only run the app if this file is executed directly
if __name__ == '__main__':
    # Start the server
    logger.info(f"Starting frontend web server on port {PORT}")
    app.run(debug=DEBUG, host=HOST, port=PORT) 