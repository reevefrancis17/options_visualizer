import os
import sys
import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime

# Fix the path modification
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Points to project root

# Import the existing modules
from python.options_data import OptionsDataManager, OptionsDataProcessor

# Configure logging
log_dir = 'debug'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Clear logs on startup
log_file = os.path.join(log_dir, 'web_app.log')
if os.path.exists(log_file):
    with open(log_file, 'w') as f:
        f.write(f"=== New session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Starting Options Visualizer Web App")

app = Flask(__name__)
# Initialize the options data manager with a longer cache duration for web app
data_manager = OptionsDataManager(cache_duration=600)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/get_options_data', methods=['POST'])
def get_options_data():
    """API endpoint to fetch options data for a given symbol"""
    try:
        # Get the symbol from the request
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({'error': 'No symbol provided'}), 400
        
        symbol = data['symbol'].strip().upper()
        if not symbol or not symbol.isalnum():
            return jsonify({'error': 'Invalid symbol format'}), 400
        
        logger.info(f"Fetching options data for {symbol}")
        
        # Fetch the options data using the data manager
        processor, current_price = data_manager.get_options_data(symbol)
        
        if processor is None or current_price is None:
            return jsonify({'error': f'Failed to fetch data for {symbol}'}), 404
        
        # Get all expiration dates
        expiry_dates = processor.get_expirations()
        if not expiry_dates:
            return jsonify({'error': f'No expiration dates found for {symbol}'}), 404
        
        # Convert expiry dates to string format for JSON
        expiry_dates_str = [date.strftime('%Y-%m-%d') for date in expiry_dates]
        
        # Get the full dataset as a DataFrame
        df = processor.get_data_frame()
        if df is None or df.empty:
            return jsonify({'error': f'No options data available for {symbol}'}), 404
        
        # Convert DataFrame to a list of dictionaries for JSON serialization
        # Convert datetime columns to strings
        df['expiration'] = df['expiration'].dt.strftime('%Y-%m-%d')
        
        # Replace NaN values with None (which will be converted to null in JSON)
        df = df.replace({np.nan: None})
        
        # Prepare the response
        response = {
            'symbol': symbol,
            'current_price': current_price,
            'expiry_dates': expiry_dates_str,
            'options_data': df.to_dict(orient='records'),
            'min_strike': processor.min_strike,
            'max_strike': processor.max_strike
        }
        
        logger.info(f"Successfully processed data for {symbol} with {len(expiry_dates)} expiration dates")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 