from flask import Flask, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import traceback

app = Flask(__name__)
CORS(app)

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
            f.write(f"\nStack trace:\n{traceback.format_exc()}\n")
        
        print(f"Error logged successfully")
    except Exception as e:
        print(f"Failed to log error: {str(e)}")
        print(f"Original error was: {error_msg}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/options/<symbol>')
def get_option_chain(symbol):
    try:
        print(f"Fetching data for symbol: {symbol}")
        symbol = symbol.strip().upper()
        
        # Get this Friday's date
        friday = get_this_friday()
        print(f"Getting options for {friday}")
        
        ticker = yf.Ticker(symbol)
        chains = ticker.option_chain(friday)
        
        # Extract just the columns we need
        calls = pd.DataFrame({
            'strike': chains.calls['strike'],
            'call_price': chains.calls['lastPrice']
        })
        
        puts = pd.DataFrame({
            'strike': chains.puts['strike'],
            'put_price': chains.puts['lastPrice']
        })
        
        # Merge calls and puts on strike price
        options = pd.merge(calls, puts, on='strike', how='outer')
        options = options.fillna(0)  # Fill NaN values with 0
        options = options.sort_values('strike')
        
        # Convert to list of dictionaries
        options_list = options.to_dict(orient='records')
        
        response_data = {
            'friday_date': friday,
            'options': options_list
        }
        
        print(f"Found {len(options_list)} options for {symbol}")
        return jsonify(response_data)
        
    except AttributeError as e:
        error_msg = f'No options data found for {symbol} expiring {friday}'
        print(f"AttributeError occurred: {error_msg}")
        log_error(f"AttributeError: {error_msg}\n{str(e)}")
        return jsonify({'error': error_msg}), 404
    except Exception as e:
        error_msg = f'Failed to fetch data for {symbol}: {str(e)}'
        print(f"Exception occurred: {error_msg}")
        log_error(f"Exception: {error_msg}")
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5001)