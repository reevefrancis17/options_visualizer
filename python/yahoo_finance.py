import yfinance as yf
import logging
import time
from datetime import datetime
from typing import Callable, Dict, Any, Optional, Tuple
import pandas as pd

# Set up logger - Use existing logger without adding a duplicate handler
logger = logging.getLogger(__name__)

class YahooFinanceAPI:
    def __init__(self, cache_duration=60):
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
                progress_callback(cached_data, cached_price, len(cached_data), len(cached_data))
            return cached_data, cached_price

        logger.info(f"Fetching fresh data for ticker: {ticker}")
        
        # Initialize variables to track partial data fetching
        options_data = {}
        current_price = None
        
        for attempt in range(self.max_retries):
            try:
                stock = yf.Ticker(ticker)
                
                # Get current price
                info = stock.info
                if not isinstance(info, dict) or 'regularMarketPrice' not in info:
                    if 'currentPrice' not in info:
                        logger.error(f"Missing price data in info for {ticker}")
                        raise ValueError("Could not get current price")
                    current_price = info['currentPrice']
                else:
                    current_price = info['regularMarketPrice']
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
                        
                        # Call the progress callback if provided
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