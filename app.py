from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import os
from datetime import datetime, timedelta
import traceback
from python.data.options_data import OptionsData

app = Flask(__name__, static_folder='frontend', static_url_path='/static')
CORS(app)

# Global options data cache
options_cache = {}

def get_this_friday():
    """Get the date of this coming Friday"""
    today = datetime.now()
    friday = today + timedelta(days=(4 - today.weekday()) % 7)  # 4 represents Friday
    return friday.strftime('%Y-%m-%d')

def log_error(error_msg):
    """Log error messages to debug/error_log.txt"""
    try:
        debug_dir = 'debug'
        if not os.path.exists(debug_dir):
            print(f"Creating debug directory at {os.path.abspath(debug_dir)}")
            os.makedirs(debug_dir)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_path = os.path.join(debug_dir, 'error_log.txt')
        
        print(f"Writing error to {os.path.abspath(log_path)}")
        with open(log_path, 'a') as f:
            f.write(f"\n[{timestamp}] {error_msg}")
            if 'Client Error' not in error_msg:  # Only add stack trace for server errors
                f.write(f"\nStack trace:\n{traceback.format_exc()}\n")
            f.write("\n" + "-"*80 + "\n")  # Add separator between log entries
        
        print(f"Error logged to {os.path.abspath(log_path)}: {error_msg}")
    except Exception as e:
        print(f"Failed to log error: {str(e)}")
        print(f"Original error was: {error_msg}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/options/<symbol>')
def get_option_chain(symbol):
    try:
        symbol = symbol.strip().upper()
        expiry_date = request.args.get('date')  # New parameter for expiry date
        
        if not expiry_date:
            expiry_date = get_this_friday()  # Use this Friday as default
            
        print(f"Fetching options for {symbol} expiring {expiry_date}")
        
        # Check if we need to fetch new data
        if symbol not in options_cache or \
           (datetime.now() - options_cache[symbol].last_updated).total_seconds() > 300:  # Cache for 5 minutes
            options_data = OptionsData()
            options_data.fetch_data(symbol)
            options_cache[symbol] = options_data
        
        # Get data for the requested expiry date
        result = options_cache[symbol].get_chain_for_date(expiry_date)
        print(f"Found {len(result['options'])} options for {symbol} expiring {expiry_date}")
        return jsonify(result)
        
    except Exception as e:
        error_msg = f'Failed to fetch data for {symbol}: {str(e)}'
        print(f"Exception occurred: {error_msg}")
        log_error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/expiry_dates/<symbol>')
def get_expiry_dates(symbol):
    try:
        symbol = symbol.strip().upper()
        
        # Check if we need to fetch new data
        if symbol not in options_cache or \
           (datetime.now() - options_cache[symbol].last_updated).total_seconds() > 300:  # Cache for 5 minutes
            options_data = OptionsData()
            options_data.fetch_data(symbol)
            options_cache[symbol] = options_data
        
        dates = options_cache[symbol].get_available_dates()
        return jsonify({'dates': dates})
        
    except Exception as e:
        error_msg = f'Failed to fetch expiry dates for {symbol}: {str(e)}'
        log_error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/log', methods=['POST'])
def log_client_error():
    try:
        log_data = request.json
        error_msg = (
            f"Client Error: {log_data['method']}\n"
            f"Error: {log_data['error']}\n"
            f"Context: {log_data['context']}"
        )
        log_error(error_msg)
        return jsonify({'status': 'logged'}), 200
    except Exception as e:
        error_msg = f'Failed to log client error: {str(e)}'
        log_error(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5001)