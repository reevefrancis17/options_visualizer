import yfinance as yf
import logging
import time
from datetime import datetime
from typing import Callable, Dict, Any, Optional, Tuple
import pandas as pd

# Set up logger - Use existing logger without adding a duplicate handler
logger = logging.getLogger(__name__)

class YahooFinanceAPI:
    def __init__(self, cache_duration=600):
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self._cache = {}
        self.cache_duration = cache_duration  # Configurable cache duration

    def _get_from_cache(self, ticker):
        if ticker in self._cache:
            data, price, timestamp = self._cache[ticker]
            if time.time() - timestamp < self.cache_duration:
                logger.info(f"Using cached data for {ticker}")
                return data, price
        return None, None

    def get_risk_free_rate(self, ticker):
        """Get the risk-free rate."""
        treasury = yf.Ticker("^TNX")
        # Get the most recent yield (adjusted close price)
        risk_free_rate = treasury.history(period="1d")["Close"].iloc[-1] / 100
        return risk_free_rate

    def get_options_data(self, ticker, progress_callback: Optional[Callable[[Dict, float, int, int], None]] = None, max_dates=None):
        """
        Fetch options data for a ticker with progressive loading support
        
        Args:
            ticker: The stock ticker symbol
            progress_callback: Optional callback function that receives:
                - current_data: The data fetched so far
                - current_price: The stock's current price
                - processed_dates: Number of expiration dates processed so far
                - total_dates: Total number of expiration dates found
            max_dates: Maximum number of expiration dates to fetch (default: None, fetch all dates)
        
        Returns:
            Tuple of (options_data, current_price)
        """
        # Try to get data from cache first
        cached_data, cached_price = self._get_from_cache(ticker)
        if cached_data:
            logger.info(f"Using cached data for {ticker} with price {cached_price}")
            # Call the callback with complete cached data if provided
            if progress_callback:
                expiry_count = len(cached_data)
                progress_callback(cached_data, cached_price, expiry_count, expiry_count)
            return cached_data, cached_price

        logger.info(f"Fetching fresh data for ticker: {ticker}")
        
        # Initialize variables to track partial data fetching
        options_data = {}
        current_price = None
        
        for attempt in range(self.max_retries):
            try:
                stock = yf.Ticker(ticker)
                
                # Get current price
                try:
                    info = stock.info
                    if not isinstance(info, dict):
                        logger.warning(f"Unexpected info type for {ticker}: {type(info)}")
                        # Try to get price from history as a fallback
                        hist = stock.history(period="1d")
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                            logger.info(f"Got current price from history for {ticker}: {current_price}")
                        else:
                            raise ValueError("Could not get current price from history")
                    elif 'regularMarketPrice' in info:
                        current_price = info['regularMarketPrice']
                    elif 'currentPrice' in info:
                        current_price = info['currentPrice']
                    elif 'previousClose' in info:
                        # Use previous close as a last resort
                        current_price = info['previousClose']
                        logger.warning(f"Using previousClose as current price for {ticker}: {current_price}")
                    else:
                        logger.error(f"Missing price data in info for {ticker}")
                        raise ValueError("Could not get current price")
                except Exception as price_error:
                    logger.error(f"Error getting price for {ticker}: {str(price_error)}")
                    # Try to get price from history as a fallback
                    try:
                        hist = stock.history(period="1d")
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                            logger.info(f"Got current price from history for {ticker}: {current_price}")
                        else:
                            raise ValueError("Could not get current price from history")
                    except Exception as hist_error:
                        logger.error(f"Error getting price from history for {ticker}: {str(hist_error)}")
                        raise ValueError("Could not get current price")
                
                logger.info(f"Got current price for {ticker}: {current_price}")
                
                # Call the progress callback with just the price if provided
                if progress_callback and current_price:
                    progress_callback({}, current_price, 0, 0)
                
                # Get options dates
                options_dates = stock.options
                if not options_dates:
                    logger.error(f"No options dates available for {ticker}")
                    raise ValueError("No options dates available")
                logger.info(f"Found {len(options_dates)} expiration dates for {ticker}")
                
                # Sort dates to prioritize near-term expirations
                sorted_dates = sorted(options_dates, key=lambda date_str: pd.to_datetime(date_str))
                
                # Limit the number of dates to fetch if max_dates is specified
                if max_dates and len(sorted_dates) > max_dates:
                    logger.info(f"Limiting to {max_dates} expiration dates (out of {len(sorted_dates)})")
                    sorted_dates = sorted_dates[:max_dates]
                
                # Get options data
                total_dates = len(sorted_dates)
                processed_dates = 0
                
                # Fetch option chains for each date
                for date in sorted_dates:
                    try:
                        opt = stock.option_chain(date)
                        if opt is None or not hasattr(opt, 'calls') or not hasattr(opt, 'puts'):
                            logger.error(f"Invalid options data for {ticker} on {date}")
                            continue
                            
                        options_data[date] = {'calls': opt.calls, 'puts': opt.puts}
                        processed_dates += 1
                        
                        # Log less frequently for better performance
                        if processed_dates % 5 == 0 or processed_dates == 1 or processed_dates == total_dates:
                            logger.info(f"Fetched {processed_dates}/{total_dates} expiration dates for {ticker}")
                        
                        # Call the progress callback after each date is processed
                        if progress_callback and current_price:
                            # Make a copy of the data to avoid reference issues
                            callback_data = options_data.copy()
                            progress_callback(callback_data, current_price, processed_dates, total_dates)
                            
                    except Exception as e:
                        logger.error(f"Error fetching option chain for {date}: {str(e)}")
                        # Continue with other dates instead of failing completely
                        continue
                
                # Return data if we have at least one valid expiration date
                if options_data and current_price:
                    # Store both data and price in cache
                    self._cache[ticker] = (options_data, current_price, time.time())
                    logger.info(f"Successfully cached data for {ticker} with {len(options_data)} expiration dates")
                    return options_data, current_price
                else:
                    logger.error(f"No valid options data found for {ticker}")
                    raise ValueError("No valid options data found")
                
            except ValueError as e:
                logger.error(f"Data error for {ticker} on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    # Return partial data if we have any
                    if options_data and current_price:
                        logger.warning(f"Returning partial data for {ticker} after error: {str(e)}")
                        return options_data, current_price
                    return None, None
            except Exception as e:
                logger.error(f"Unexpected error for {ticker} on attempt {attempt + 1}: {str(e)}")
                if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    # Wait longer for rate limit errors
                    time.sleep(self.retry_delay * 2)
                else:
                    time.sleep(self.retry_delay)
                    
        # If we get here, all retries failed
        logger.error(f"All retries failed for {ticker}")
        return None, None