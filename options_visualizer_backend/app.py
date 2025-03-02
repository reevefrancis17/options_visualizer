#!/usr/bin/env python3
import os
import time
import pickle
import logging
from datetime import datetime
from flask import Flask, jsonify, request, Response
import pandas as pd
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from flask_cors import CORS

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/server.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
CACHE_DIR = 'cache'
DATA_DIR = 'data'
CSV_PATH = os.path.join(DATA_DIR, 'tickers.csv')
CACHE_DURATION = 10 * 60  # 10 minutes in seconds

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# In-memory cache (dictionary: ticker -> (data, timestamp))
cache = {}

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
    df = pd.DataFrame(list(tickers.items()), columns=['ticker', 'timestamp'])
    df.to_csv(CSV_PATH, index=False)
    logger.info("Updated tickers.csv")

# Fetch options data from yfinance
def fetch_options_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        options_dates = stock.options
        if not options_dates:
            logger.warning(f"No options data for {ticker}")
            return None
        
        options_data = {}
        try:
            current_price = stock.info.get('regularMarketPrice')
            if current_price is None:
                current_price = stock.history(period="1d")['Close'].iloc[-1]
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {str(e)}")
            try:
                current_price = stock.history(period="1d")['Close'].iloc[-1]
            except Exception as e2:
                logger.error(f"Fallback price fetch failed for {ticker}: {str(e2)}")
                current_price = None
        
        for date in options_dates:
            try:
                opt = stock.option_chain(date)
                options_data[date] = {
                    'calls': opt.calls.to_dict('records'),
                    'puts': opt.puts.to_dict('records')
                }
            except Exception as e:
                logger.error(f"Error fetching options for {ticker} date {date}: {str(e)}")
                continue
                
        logger.info(f"Fetched options data for {ticker}")
        return {'options': options_data, 'price': current_price, 'ticker': ticker}
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Load cache from disk
def load_cache(ticker):
    cache_file = os.path.join(CACHE_DIR, f"{ticker}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache for {ticker}: {str(e)}")
            return None
    return None

# Save to cache (disk and memory)
def save_cache(ticker, data):
    try:
        cache_file = os.path.join(CACHE_DIR, f"{ticker}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump((data, time.time()), f)
        cache[ticker] = (data, time.time())
        logger.info(f"Cached data for {ticker}")
    except Exception as e:
        logger.error(f"Error saving cache for {ticker}: {str(e)}")

# Get options data (from cache or fetch)
def get_options_data(ticker):
    # Check in-memory cache
    if ticker in cache:
        data, timestamp = cache[ticker]
        if time.time() - timestamp < CACHE_DURATION:
            logger.info(f"Serving {ticker} from in-memory cache")
            return data
    
    # Check disk cache
    cached = load_cache(ticker)
    if cached:
        data, timestamp = cached
        if time.time() - timestamp < CACHE_DURATION:
            cache[ticker] = (data, timestamp)
            logger.info(f"Serving {ticker} from disk cache")
            return data
    
    # Fetch fresh data if cache is old or missing
    data = fetch_options_data(ticker)
    if data:
        save_cache(ticker, data)
        # Update ticker timestamp in CSV
        tickers = load_tickers()
        tickers[ticker] = datetime.now().isoformat()
        save_tickers(tickers)
    return data

# Background task to update cache
def update_cache():
    logger.info("Starting scheduled cache update")
    tickers = load_tickers()
    for ticker in tickers.keys():
        try:
            data = fetch_options_data(ticker)
            if data:
                save_cache(ticker, data)
                tickers[ticker] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"Error updating cache for {ticker}: {str(e)}")
    save_tickers(tickers)
    logger.info("Completed scheduled cache update")

# API endpoint to get options data
@app.route('/api/options/<ticker>', methods=['GET'])
def get_ticker_options(ticker):
    ticker = ticker.upper()
    data = get_options_data(ticker)
    if data:
        return jsonify(data)
    return jsonify({'error': f"Could not fetch data for {ticker}"}), 500

# API endpoint to get available tickers
@app.route('/api/tickers', methods=['GET'])
def get_tickers():
    tickers = load_tickers()
    return jsonify(list(tickers.keys()))

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(update_cache, 'interval', minutes=10)
scheduler.start()

# Load initial cache
def initialize_cache():
    logger.info("Initializing cache...")
    tickers = load_tickers()
    for ticker in tickers.keys():
        try:
            data = get_options_data(ticker)
            if data:
                logger.info(f"Initialized cache for {ticker}")
        except Exception as e:
            logger.error(f"Error initializing cache for {ticker}: {str(e)}")
    logger.info("Cache initialization complete")

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    initialize_cache()
    # Run with Gunicorn in production: gunicorn -w 4 -b 0.0.0.0:5001 app:app
    app.run(host='0.0.0.0', port=5001, debug=False) 