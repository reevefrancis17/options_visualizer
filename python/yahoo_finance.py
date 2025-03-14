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

class TickerNotFoundError(Exception):
    """Exception raised when a ticker symbol is not found."""
    pass

class NoOptionsDataError(Exception):
    """Exception raised when a ticker has no options data."""
    pass

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
            
            # Check for 404 error which indicates ticker doesn't exist
            if "404 Client Error" in str(info_error):
                logger.error(f"Ticker {ticker} not found (404 error)")
                raise TickerNotFoundError(f"Ticker {ticker} not found")
            
            # Try to get price from history as a fallback
            try:
                hist = stock.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    logger.info(f"Got current price from history for {ticker}: {price}")
                    return price
                else:
                    # Empty history often means the ticker doesn't exist
                    logger.error(f"Empty history data for {ticker}, ticker may not exist")
                    raise TickerNotFoundError(f"Ticker {ticker} not found (empty history)")
            except Exception as hist_error:
                logger.error(f"Error getting price from history for {ticker}: {str(hist_error)}")
                
                # Check for specific error messages that indicate ticker doesn't exist
                if "possibly delisted" in str(hist_error) or "No data found" in str(hist_error):
                    raise TickerNotFoundError(f"Ticker {ticker} not found or delisted")
                
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
            
            # Filter columns that actually exist in the DataFrame
            calls_columns = [col for col in essential_columns if col in opt.calls.columns]
            puts_columns = [col for col in essential_columns if col in opt.puts.columns]
            
            # Convert to records
            calls = opt.calls[calls_columns].to_dict('records')
            puts = opt.puts[puts_columns].to_dict('records')
            
            # Add expiration date to each record
            for call in calls:
                call['expiration'] = expiry
            for put in puts:
                put['expiration'] = expiry
                
            # Update progress if callback is provided
            if processed_dates is not None and total_dates is not None:
                progress = processed_dates / total_dates
                logger.debug(f"Processed {processed_dates}/{total_dates} dates for {ticker} ({progress:.1%})")
            
            # Return the processed data
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
            # Return empty data with error message
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

    def get_options_data(self, ticker, progress_callback=None, max_dates=None):
        """Get options data for a ticker.
        
        Args:
            ticker: The ticker symbol
            progress_callback: Optional callback function to report progress
            max_dates: Maximum number of expiration dates to fetch
            
        Returns:
            Tuple of (options_data, current_price)
        """
        logger.info(f"Fetching fresh data for ticker: {ticker}")
        
        # Initialize variables to track partial data fetching
        options_data = {}
        current_price = None
        
        for attempt in range(self.max_retries):
            try:
                # Create ticker object - don't pass session parameter
                stock = yf.Ticker(ticker)
                
                # Get current price
                current_price = self._get_current_price(stock, ticker)
                
                # Get expiration dates
                expiry_dates = stock.options
                if not expiry_dates:
                    logger.error(f"No expiration dates found for {ticker}")
                    raise NoOptionsDataError(f"No options data available for {ticker}")
                
                logger.info(f"Found {len(expiry_dates)} expiration dates for {ticker}")
                
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
                    processed_dates = 0
                    for future in concurrent.futures.as_completed(future_to_expiry):
                        expiry = future_to_expiry[future]
                        try:
                            result = future.result()
                            # Store the result
                            if result['expiry'] in options_data:
                                # Merge data if we already have some for this expiry
                                options_data[result['expiry']]['calls'].extend(result['data']['calls'])
                                options_data[result['expiry']]['puts'].extend(result['data']['puts'])
                            else:
                                # Store new data
                                options_data[result['expiry']] = result['data']
                            
                            processed_dates += 1
                            
                            # Update progress if callback is provided
                            if progress_callback:
                                try:
                                    # Call the callback with the partial data, current price, processed dates, and total dates
                                    progress_callback(options_data, current_price, processed_dates, total_dates)
                                except Exception as callback_error:
                                    logger.error(f"Error in progress_callback for {ticker}: {str(callback_error)}")
                            
                        except Exception as e:
                            logger.error(f"Error processing future for {expiry}: {str(e)}")
                
                # Add metadata
                options_data['_ticker'] = ticker
                options_data['_current_price'] = current_price
                options_data['_expiration_dates'] = expiry_dates
                
                # Check if we have any valid data
                valid_data = False
                for expiry in expiry_dates:
                    if expiry in options_data and options_data[expiry]['calls'] and options_data[expiry]['puts']:
                        valid_data = True
                        break
                
                if not valid_data:
                    logger.error(f"No valid options data found for {ticker}")
                    raise NoOptionsDataError(f"No valid options data found for {ticker}")
                
                # Return the data
                return options_data, current_price
                
            except (TickerNotFoundError, NoOptionsDataError) as e:
                # Don't retry for these specific errors
                logger.error(f"{str(e)}")
                raise
                
            except Exception as e:
                logger.error(f"Error in attempt {attempt+1} for {ticker}: {str(e)}")
                if attempt < self.max_retries - 1:
                    # Wait before retrying
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed for {ticker}")
                    raise
        
        # This should never be reached due to the exception handling above
        raise ValueError(f"Failed to get options data for {ticker} after {self.max_retries} attempts")

    def get_batch_options_data(self, tickers, progress_callback=None, max_dates=None):
        """Get options data for multiple tickers in parallel.
        
        Args:
            tickers: List of ticker symbols
            progress_callback: Optional callback function to report progress
            max_dates: Maximum number of expiration dates to fetch per ticker
            
        Returns:
            Dictionary mapping tickers to options data
        """
        logger.info(f"Fetching batch data for {len(tickers)} tickers")
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.get_options_data, ticker, progress_callback, max_dates): ticker 
                for ticker in tickers
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    results[ticker] = data
                except (TickerNotFoundError, NoOptionsDataError) as e:
                    # Store the error in the results
                    results[ticker] = (None, str(e))
                except Exception as e:
                    logger.error(f"Error getting data for {ticker}: {str(e)}")
                    results[ticker] = (None, str(e))
        
        return results