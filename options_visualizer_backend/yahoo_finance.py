#yahoo_finance.py
import yfinance as yf
import logging
import time
import concurrent.futures
from datetime import datetime
from typing import Callable, Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import requests

# Set up logger - Use existing logger without adding a duplicate handler
logger = logging.getLogger(__name__)

class YahooFinanceAPI:
    def __init__(self, cache_duration=600, max_workers=4):
        """Initialize the Yahoo Finance API wrapper.
        
        Args:
            cache_duration: Cache duration in seconds (for reference only)
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.cache_duration = cache_duration  # For reference only
        self.max_workers = max_workers  # For parallel processing
        self._session = None  # Lazy-loaded session
        
    @property
    def session(self):
        """Get or create a requests session for reuse."""
        if self._session is None:
            # Create a new session directly instead of trying to access Ticker's _session
            self._session = requests.Session()
            # Configure the session with appropriate headers
            self._session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
            })
        return self._session

    def get_risk_free_rate(self, ticker="^TNX"):
        """Get the risk-free rate from treasury yield."""
        try:
            treasury = yf.Ticker(ticker)
            # Get the most recent yield (adjusted close price)
            risk_free_rate = treasury.history(period="1d")["Close"].iloc[-1] / 100
            return risk_free_rate
        except Exception as e:
            logger.error(f"Error getting risk-free rate: {str(e)}")
            # Default to a reasonable value if we can't get the actual rate
            return 0.04  # 4% as fallback

    def _get_current_price(self, stock, ticker):
        """Helper method to get current price with fallbacks."""
        try:
            # Try to get info first (most accurate)
            info = stock.info
            if not isinstance(info, dict):
                logger.warning(f"Unexpected info type for {ticker}: {type(info)}")
                raise ValueError("Info is not a dictionary")
                
            if 'regularMarketPrice' in info:
                return info['regularMarketPrice']
            elif 'currentPrice' in info:
                return info['currentPrice']
            elif 'previousClose' in info:
                # Use previous close as a last resort
                logger.warning(f"Using previousClose as current price for {ticker}")
                return info['previousClose']
            else:
                raise ValueError("No price fields found in info")
                
        except Exception as info_error:
            logger.warning(f"Error getting price from info for {ticker}: {str(info_error)}")
            
            # Try to get price from history as a fallback
            try:
                hist = stock.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    logger.info(f"Got current price from history for {ticker}: {price}")
                    return price
                else:
                    raise ValueError("History data is empty")
            except Exception as hist_error:
                logger.error(f"Error getting price from history for {ticker}: {str(hist_error)}")
                raise ValueError(f"Failed to get price for {ticker}")

    def _process_expiry_date(self, stock, ticker, expiry, processed_dates, total_dates):
        """Process a single expiration date."""
        try:
            # Get options chain for this expiration
            opt = stock.option_chain(expiry)
            
            # Convert DataFrames to records for more efficient serialization
            # Only keep essential columns to reduce memory usage
            essential_columns = [
                'strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange', 
                'volume', 'openInterest', 'impliedVolatility', 'inTheMoney',
                'contractSymbol', 'lastTradeDate', 'contractSize', 'currency'
            ]
            
            # Filter columns for calls
            calls_df = opt.calls
            calls_columns = [col for col in essential_columns if col in calls_df.columns]
            calls = calls_df[calls_columns].to_dict('records')
            
            # Filter columns for puts
            puts_df = opt.puts
            puts_columns = [col for col in essential_columns if col in puts_df.columns]
            puts = puts_df[puts_columns].to_dict('records')
            
            logger.info(f"Processed expiry date {expiry} for {ticker}: {len(calls)} calls, {len(puts)} puts")
            
            return {
                'expiry': expiry,
                'data': {
                    'calls': calls,
                    'puts': puts
                },
                'processed': processed_dates,
                'total': total_dates
            }
        except Exception as e:
            logger.error(f"Error processing expiry date {expiry} for {ticker}: {str(e)}")
            return {
                'expiry': expiry,
                'data': {
                    'calls': [],
                    'puts': []
                },
                'error': str(e),
                'processed': processed_dates,
                'total': total_dates
            }

    def get_options_data(self, ticker, progress_callback: Optional[Callable[[Dict, float, int, int], None]] = None, max_dates=None):
        """
        Fetch options data for a ticker with progressive loading and parallel processing.
        
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
        logger.info(f"Fetching fresh data for ticker: {ticker}")
        
        # Initialize variables to track partial data fetching
        options_data = {}
        current_price = None
        
        for attempt in range(self.max_retries):
            try:
                # Create ticker object with a fresh session
                try:
                    stock = yf.Ticker(ticker)
                    logger.info(f"Successfully created Ticker object for {ticker}")
                except Exception as ticker_error:
                    logger.error(f"Error creating Ticker object for {ticker}: {str(ticker_error)}")
                    raise ValueError(f"Failed to create Ticker object for {ticker}")
                
                # Get current price with better error handling
                try:
                    current_price = self._get_current_price(stock, ticker)
                    if current_price is None or current_price <= 0:
                        logger.error(f"Invalid current price for {ticker}: {current_price}")
                        raise ValueError(f"Invalid current price for {ticker}")
                    logger.info(f"Got current price for {ticker}: {current_price}")
                except Exception as price_error:
                    logger.error(f"Error getting current price for {ticker}: {str(price_error)}")
                    raise ValueError(f"Failed to get current price for {ticker}")
                
                # Get expiration dates with better error handling
                try:
                    expiry_dates = stock.options
                    if not expiry_dates:
                        logger.error(f"No expiration dates found for {ticker}")
                        raise ValueError(f"No options data available for {ticker}")
                    logger.info(f"Found {len(expiry_dates)} expiration dates for {ticker}")
                except Exception as expiry_error:
                    logger.error(f"Error getting expiration dates for {ticker}: {str(expiry_error)}")
                    raise ValueError(f"Failed to get expiration dates for {ticker}")
                
                # Limit the number of dates if max_dates is specified
                if max_dates and len(expiry_dates) > max_dates:
                    logger.info(f"Limiting to {max_dates} expiration dates for {ticker}")
                    expiry_dates = expiry_dates[:max_dates]
                
                # Total number of dates to process
                total_dates = len(expiry_dates)
                
                # Process expiration dates in parallel for better performance
                # Use a higher number of workers for better throughput
                max_workers = min(self.max_workers * 2, total_dates)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_expiry = {
                        executor.submit(
                            self._process_expiry_date, 
                            stock, 
                            ticker, 
                            expiry, 
                            i+1, 
                            total_dates
                        ): expiry for i, expiry in enumerate(expiry_dates)
                    }
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(future_to_expiry):
                        expiry = future_to_expiry[future]
                        try:
                            result = future.result()
                            
                            # Add to options data
                            if result and 'data' in result:
                                options_data[expiry] = result['data']
                                
                                # Call progress callback if provided
                                if progress_callback:
                                    processed_dates = result['processed']
                                    progress_callback(options_data, current_price, processed_dates, total_dates)
                            
                        except Exception as e:
                            logger.error(f"Error processing future for {expiry}: {str(e)}")
                
                # Verify we have some data before returning
                if not options_data:
                    logger.error(f"No options data collected for {ticker}")
                    raise ValueError(f"Failed to collect any options data for {ticker}")
                
                logger.info(f"Successfully fetched data for {ticker} with {len(options_data)} expiration dates")
                return options_data, current_price
                
            except Exception as e:
                logger.error(f"Error in attempt {attempt + 1} for {ticker}: {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying {ticker} (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All {self.max_retries} attempts failed for {ticker}")
                    # Don't raise here, return None, None to allow caller to handle
        
        # If we get here, all attempts failed
        logger.error(f"Failed to fetch options data for {ticker} after {self.max_retries} attempts")
        return None, None
        
    def get_batch_options_data(self, tickers: List[str], max_dates=None):
        """
        Fetch options data for multiple tickers in parallel.
        
        Args:
            tickers: List of ticker symbols
            max_dates: Maximum number of expiration dates to fetch per ticker
            
        Returns:
            Dict mapping tickers to (options_data, current_price) tuples
        """
        logger.info(f"Fetching batch data for {len(tickers)} tickers")
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(tickers))) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.get_options_data, ticker, None, max_dates): ticker 
                for ticker in tickers
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    options_data, current_price = future.result()
                    results[ticker] = (options_data, current_price)
                    logger.info(f"Completed batch fetch for {ticker}")
                except Exception as e:
                    logger.error(f"Error in batch fetch for {ticker}: {str(e)}")
                    results[ticker] = (None, None)
        
        return results