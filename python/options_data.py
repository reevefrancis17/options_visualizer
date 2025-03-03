#options_data.py
import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from datetime import datetime
import logging
import traceback
import os
import time
import threading
import sys
import math
from typing import Dict, Optional, Callable, Tuple, List, Any, Union

# Import the finance API and models
from python.yahoo_finance import YahooFinanceAPI
from python.models.black_scholes import (
    call_price, put_price, delta, gamma, theta, vega, rho, implied_volatility
)
from python.cache_manager import OptionsCache

# Set up logger
logger = logging.getLogger(__name__)

# Clear error logs if they exist
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'debug', 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
error_log = os.path.join(log_dir, 'error_log.txt')
if os.path.exists(error_log):
    with open(error_log, 'w') as f:
        f.write(f"=== New session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

class OptionsDataManager:
    """Central manager for options data handling."""
    DATA_SOURCE_YAHOO = "yahoo"
    MODEL_BLACK_SCHOLES = "black_scholes"
    MODEL_MARKET = "market"

    def __init__(self, data_source=DATA_SOURCE_YAHOO, pricing_model=MODEL_MARKET, cache_duration=600):
        logger.info(f"Initializing OptionsDataManager with source={data_source}, model={pricing_model}")
        self.data_source = data_source
        self.pricing_model = pricing_model
        self.cache_duration = cache_duration
        self.api = YahooFinanceAPI(cache_duration=cache_duration) if data_source == self.DATA_SOURCE_YAHOO else None
        self._cache = OptionsCache(cache_duration=cache_duration)
        self._loading_state = {}
        self._fetch_locks = {}
        
        # Register the refresh callback with the cache manager
        self._cache.register_refresh_callback(self._refresh_ticker)
        
        # Start background polling for cache updates
        self._start_cache_polling()

    def _start_cache_polling(self):
        """Start a background thread to poll and refresh the entire cache periodically."""
        def poll_and_refresh_cache():
            while True:
                try:
                    # Sleep first to allow the application to initialize
                    time.sleep(self.cache_duration)
                    
                    logger.info("Starting cache refresh cycle")
                    # Get all tickers in the cache
                    tickers = self._cache.get_all_tickers()
                    logger.info(f"Refreshing {len(tickers)} tickers in cache")
                    
                    # Refresh each ticker
                    for ticker in tickers:
                        try:
                            # Skip if already being refreshed
                            if ticker in self._loading_state:
                                logger.info(f"Skipping {ticker} - already being refreshed")
                                continue
                            
                            logger.info(f"Refreshing cached data for {ticker}")
                            self._refresh_ticker(ticker)
                            
                            # Sleep briefly between refreshes to avoid overwhelming the API
                            time.sleep(1)
                        except Exception as e:
                            logger.error(f"Error refreshing {ticker}: {str(e)}")
                    
                    logger.info("Cache refresh cycle complete")
                except Exception as e:
                    logger.error(f"Error in cache polling: {str(e)}")
                    # Sleep before retrying
                    time.sleep(60)
        
        # Start polling thread
        polling_thread = threading.Thread(target=poll_and_refresh_cache, daemon=True)
        polling_thread.start()
        logger.info("Started cache polling thread")

    def _refresh_ticker(self, ticker: str):
        """Refresh data for a ticker in the background."""
        # Start fetching in the background
        self.start_fetching(ticker)

    def get_current_processor(self, ticker: str) -> Tuple[Optional['OptionsDataProcessor'], Optional[float], str, float]:
        """Get the current processor for a ticker with status information.
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            Tuple of (processor, price, status, progress)
            where status is one of: 'complete', 'partial', 'loading', 'not_found'
            and progress is a float between 0 and 1
        """
        # Check if we're currently loading this ticker
        is_loading = ticker in self._loading_state
        
        # Check cache
        cached_data = self._cache.get(ticker)
        
        if cached_data:
            options_data, price, timestamp, expiry, processed_dates, total_dates = cached_data
            
            # Calculate age of cache - ensure timestamp is a float
            try:
                timestamp_float = float(timestamp) if timestamp is not None else 0
                age = time.time() - timestamp_float
            except (ValueError, TypeError):
                logger.warning(f"Invalid timestamp format for {ticker}: {timestamp}, treating as expired")
                age = self.cache_duration + 1  # Treat as expired
            
            # Calculate progress
            progress = processed_dates / max(total_dates, 1) if total_dates > 0 else 1.0
            
            # Check if cache is still valid
            if age < self.cache_duration:
                # Check if the data is fully interpolated
                is_fully_interpolated = options_data.get('_is_fully_interpolated', False) if isinstance(options_data, dict) else False
                
                # Create processor from cache
                try:
                    # If we're still loading and the cache is not fully interpolated, mark as partial
                    if is_loading and not is_fully_interpolated:
                        status = 'partial'
                    else:
                        status = 'complete'
                        
                    # Create processor from cache
                    processor = OptionsDataProcessor(options_data, price, is_processed=True)
                    return processor, price, status, progress
                except Exception as e:
                    logger.error(f"Error creating processor from cache for {ticker}: {str(e)}")
                    # Fall through to loading or not found
            else:
                logger.info(f"Cache for {ticker} is expired (age: {age:.1f}s)")
                # Fall through to loading or not found
        
        # If we're loading, return loading status
        if is_loading:
            # Get progress from loading state
            last_processed = self._loading_state[ticker].get('last_processed_dates', 0)
            total = self._loading_state[ticker].get('total_dates', 0)
            progress = last_processed / max(total, 1) if total > 0 else 0.0
            
            return None, None, 'loading', progress
        
        # Not in cache and not loading
        return None, None, 'not_found', 0.0

    def start_fetching(self, ticker: str, skip_interpolation: bool = False) -> bool:
        """Start fetching data in the background if not already loading.
        
        Args:
            ticker: The stock ticker symbol
            skip_interpolation: Whether to skip interpolation for faster processing during immediate use.
                               Note: The cached data will always be fully interpolated regardless of this setting.
            
        Returns:
            bool: True if fetching started, False if already in progress
        """
        # Use a lock to prevent multiple threads from starting fetches for the same ticker
        if ticker not in self._fetch_locks:
            self._fetch_locks[ticker] = threading.Lock()
            
        with self._fetch_locks[ticker]:
            if ticker in self._loading_state:
                logger.info(f"Fetch already in progress for {ticker}")
                return False
                
            self._loading_state[ticker] = {
                'last_processed_dates': 0, 
                'total_dates': 0,
                'skip_interpolation': skip_interpolation
            }
            thread = threading.Thread(target=self._fetch_in_background, args=(ticker,))
            thread.daemon = True  # Make thread exit when main thread exits
            thread.start()
            logger.info(f"Started background fetch for {ticker} (skip_interpolation={skip_interpolation} for immediate use only, cached data will be fully interpolated)")
            return True

    def _fetch_in_background(self, ticker: str):
        """Fetch data in a background thread."""
        try:
            # Use a lock to prevent multiple threads from fetching the same ticker
            if ticker not in self._fetch_locks:
                self._fetch_locks[ticker] = threading.Lock()
                
            with self._fetch_locks[ticker]:
                # Check if another thread has already completed the fetch
                if ticker not in self._loading_state:
                    logger.info(f"Fetch for {ticker} was already completed by another thread")
                    return
                    
                # Get skip_interpolation setting from loading state
                skip_interpolation = self._loading_state.get(ticker, {}).get('skip_interpolation', False)  # Default to False to ensure interpolation
                logger.info(f"Fetching data for {ticker} with skip_interpolation={skip_interpolation}")
                
                # Collect all raw data first without processing
                raw_data_collection = {}
                final_current_price = None
                final_total_dates = 0
                
                def cache_update_callback(partial_data, current_price, processed_dates, total_dates):
                    nonlocal raw_data_collection, final_current_price, final_total_dates
                    # Just store the raw data without processing
                    if partial_data and current_price:
                        # Update our collection with the new data
                        raw_data_collection.update(partial_data)
                        final_current_price = current_price
                        final_total_dates = total_dates
                        
                        # Update loading state for progress tracking
                        if ticker in self._loading_state:  # Check if still loading
                            self._loading_state[ticker]['last_processed_dates'] = len(raw_data_collection)
                            self._loading_state[ticker]['total_dates'] = total_dates
                        
                        # Store raw data in cache for quick recovery in case of crash
                        try:
                            # Mark this as unprocessed data by not including a 'dataset' key
                            # Add a flag to indicate this is raw data (not interpolated)
                            raw_data_collection['_is_fully_interpolated'] = False
                            self._cache.set(ticker, raw_data_collection, current_price, len(raw_data_collection), total_dates)
                            logger.info(f"Cached raw data for {ticker} with {len(raw_data_collection)}/{total_dates} dates")
                        except Exception as e:
                            logger.error(f"Error caching raw data for {ticker}: {str(e)}")
                
                # Fetch data directly from the API with the callback
                logger.info(f"Fetching data for {ticker} in background thread")
                
                try:
                    options_data_dict, current_price = self.api.get_options_data(ticker, cache_update_callback)
                except Exception as fetch_error:
                    logger.error(f"Error fetching data from API for {ticker}: {str(fetch_error)}")
                    logger.error(traceback.format_exc())
                    
                    # Try to use any raw data we've collected so far
                    if raw_data_collection and final_current_price:
                        logger.info(f"Using partial data collected for {ticker} despite API error")
                        options_data_dict = raw_data_collection
                        current_price = final_current_price
                    else:
                        # Check if we have cached data we can use
                        cached_data = self._cache.get(ticker)
                        if cached_data and 'data' in cached_data and 'price' in cached_data:
                            logger.info(f"Using cached data for {ticker} due to API error")
                            options_data_dict = cached_data['data']
                            current_price = cached_data['price']
                        else:
                            # No data available
                            logger.error(f"No data available for {ticker} after API error")
                            # Clean up loading state
                            if ticker in self._loading_state:
                                del self._loading_state[ticker]
                            return
                
                # Now process all the collected data at once
                if raw_data_collection and final_current_price:
                    # Use the data from the callback which is guaranteed to be a dictionary
                    processed_dates = len(raw_data_collection)
                    total_dates = final_total_dates
                    
                    try:
                        logger.info(f"Processing complete data for {ticker} with {processed_dates} dates")
                        # Always process with interpolation for the final cached version
                        # This is the key change - we're forcing interpolation for the cached version
                        processor = OptionsDataProcessor(raw_data_collection, final_current_price, 
                                                        is_processed=False, 
                                                        skip_interpolation=False)  # Always do interpolation for cached data
                        
                        # Get the fully processed data with interpolation
                        processed_data = processor.options_data
                        
                        # Add a flag to indicate this data is fully interpolated
                        processed_data['_is_fully_interpolated'] = True
                        
                        # Final update to cache with fully processed data
                        self._cache.set(ticker, processed_data, final_current_price, processed_dates, total_dates)
                        logger.info(f"Cached fully processed data for {ticker} with {processed_dates}/{total_dates} dates (with interpolation)")
                    except Exception as proc_error:
                        logger.error(f"Error processing complete data for {ticker}: {str(proc_error)}")
                        logger.error(traceback.format_exc())
                        # Fall back to caching raw data
                        raw_data_collection['_is_fully_interpolated'] = False
                        self._cache.set(ticker, raw_data_collection, final_current_price, processed_dates, total_dates)
                    
                    logger.info(f"Completed fetching data for {ticker} with {processed_dates} dates")
                elif isinstance(options_data_dict, dict) and current_price:
                    # Handle the case where options_data is a dictionary from the API
                    logger.info(f"Received options data dictionary for {ticker}")
                    processed_dates = len(options_data_dict)
                    total_dates = processed_dates  # Assume all dates are processed
                    
                    try:
                        # Process the data
                        processor = OptionsDataProcessor(options_data_dict, current_price,
                                                        is_processed=False,
                                                        skip_interpolation=False)  # Always do interpolation for cached data
                        processed_data = processor.options_data
                        
                        # Add a flag to indicate this data is fully interpolated
                        processed_data['_is_fully_interpolated'] = True
                        
                        # Update cache with the processed data
                        self._cache.set(ticker, processed_data, current_price, processed_dates, total_dates)
                        logger.info(f"Cached processed data for {ticker} with {processed_dates}/{total_dates} dates")
                    except Exception as proc_error:
                        logger.error(f"Error processing API data for {ticker}: {str(proc_error)}")
                        logger.error(traceback.format_exc())
                        # Fall back to caching raw data
                        options_data_dict['_is_fully_interpolated'] = False
                        self._cache.set(ticker, options_data_dict, current_price, processed_dates, total_dates)
                elif hasattr(options_data_dict, 'options_data') and current_price:
                    # Handle the case where options_data is already an OptionsDataProcessor
                    logger.info(f"Received OptionsDataProcessor for {ticker}")
                    # Get the number of expiration dates from the processor
                    try:
                        processor = options_data_dict
                        processed_data = processor.options_data
                        # Get the number of expiration dates
                        expiry_dates = processor.get_expirations()
                        processed_dates = len(expiry_dates) if expiry_dates else 0
                        total_dates = processed_dates
                        
                        # Add a flag to indicate this data is fully interpolated
                        processed_data['_is_fully_interpolated'] = True
                        
                        # Update cache with the processed data
                        self._cache.set(ticker, processed_data, current_price, processed_dates, total_dates)
                        logger.info(f"Cached processed data for {ticker} with {processed_dates} dates")
                    except Exception as proc_error:
                        logger.error(f"Error processing processor data for {ticker}: {str(proc_error)}")
                        logger.error(traceback.format_exc())
                else:
                    logger.error(f"Failed to fetch data for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching data for {ticker} in background: {str(e)}")
            # Log the full traceback for debugging
            logger.error(traceback.format_exc())
        finally:
            # Clean up loading state
            if ticker in self._loading_state:
                del self._loading_state[ticker]
            logger.info(f"Background fetch for {ticker} completed")

    def get_options_data(self, ticker: str, progress_callback: Optional[Callable[[Dict, float, int, int], None]] = None,
                         max_dates: Optional[int] = None, force_reinterpolate: bool = False) -> Tuple[Optional['OptionsDataProcessor'], Optional[float]]:
        """Get options data with support for immediate cache return and background fetching.
        
        This method will:
        1. Return cached data immediately if available
        2. Start a background fetch if no cache is available
        3. Call progress_callback with the current state
        
        Args:
            ticker: The stock ticker symbol
            progress_callback: Optional callback for progress updates
            max_dates: Maximum number of expiration dates to fetch
            force_reinterpolate: If True, force reinterpolation of all values even if cached
            
        Returns:
            Tuple of (processor, price) - may be (None, None) if no cache and fetch just started
        """
        # Check cache first
        processor, price, timestamp, progress = self.get_current_processor(ticker)
        
        # If we have cached data (partial or complete), return it immediately
        if processor is not None and price is not None:
            # Force reinterpolation if requested
            if force_reinterpolate and processor:
                logger.info(f"Forcing reinterpolation for {ticker}")
                processor.force_reinterpolate()
                
                # Update cache with reinterpolated data
                # Get the number of expiration dates as processed_dates
                expiry_count = len(processor.get_expirations())
                
                # Add a flag to indicate this data is fully interpolated
                processor.options_data['_is_fully_interpolated'] = True
                
                self._cache.set(ticker, processor.options_data, price, expiry_count, expiry_count)
                logger.info(f"Updated cache for {ticker} with reinterpolated data")
                
            if progress_callback and processor:
                # Call progress callback with current state
                expiry_count = len(processor.get_expirations())
                
                # Get total count from loading state
                total_count = self._loading_state.get(ticker, {}).get('total_dates', expiry_count)
                
                progress_callback(processor.options_data, price, expiry_count, total_count)
            return processor, price
            
        # No cache available, start background fetch
        self.start_fetching(ticker)
        
        # Return None to indicate fetch has started but no data available yet
        return None, None

    def get_risk_free_rate(self):
        return YahooFinanceAPI().get_risk_free_rate("^TNX")

    def calculate_option_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        if self.pricing_model == self.MODEL_BLACK_SCHOLES:
            return call_price(S, K, T, r, sigma) if option_type == 'call' else put_price(S, K, T, r, sigma)
        raise ValueError(f"Unsupported pricing model: {self.pricing_model}")

    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
        if self.pricing_model == self.MODEL_BLACK_SCHOLES:
            return {
                'delta': delta(S, K, T, r, sigma, option_type),
                'gamma': gamma(S, K, T, r, sigma),
                'theta': theta(S, K, T, r, sigma, option_type),
                'vega': vega(S, K, T, r, sigma),
                'rho': rho(S, K, T, r, sigma, option_type)
            }
        raise ValueError(f"Unsupported pricing model: {self.pricing_model}")

    def calculate_implied_volatility(self, market_price: float, S: float, K: float, T: float, r: float, option_type: str) -> float:
        return implied_volatility(market_price, S, K, T, r, option_type)

class OptionsDataProcessor:
    """Processes raw options data into an xarray Dataset."""
    def __init__(self, options_data, current_price, is_processed=False, skip_interpolation=False):
        logger.info(f"Initializing OptionsDataProcessor with current_price: {current_price}")
        self.options_data = options_data
        self.current_price = current_price
        self.min_strike = None
        self.max_strike = None
        self.skip_interpolation = skip_interpolation
        self.ds = None  # Initialize dataset to None
        self.risk_free_rate = None
        
        if not options_data:
            logger.error("Failed to fetch options.")
            raise ValueError("Options data is None")
        
        # If data is already processed (from cache), we can skip processing steps
        if is_processed:
            logger.info("Using fully processed and interpolated data from cache")
            # Extract min/max strike from the options_data if available
            if isinstance(options_data, dict):
                if 'min_strike' in options_data and 'max_strike' in options_data:
                    self.min_strike = options_data['min_strike']
                    self.max_strike = options_data['max_strike']
                    logger.info(f"Loaded strike range from cache: {self.min_strike} to {self.max_strike}")
                    # Remove these from options_data to avoid confusion
                    processed_data = {k: v for k, v in options_data.items() if k not in ['min_strike', 'max_strike']}
                else:
                    processed_data = options_data
                
                # Get the dataset if available
                if 'dataset' in processed_data:
                    self.ds = processed_data['dataset']
                    if self.ds is not None:
                        logger.info(f"Loaded dataset from cache with dimensions: {list(self.ds.dims)}")
                        
                        # Verify that the dataset has all required dimensions
                        required_dims = ['strike', 'DTE', 'option_type']
                        missing_dims = [dim for dim in required_dims if dim not in self.ds.dims]
                        
                        if missing_dims:
                            logger.warning(f"Dataset from cache is missing dimensions: {missing_dims}")
                            
                            # Try to recover by adding missing dimensions
                            try:
                                for dim in missing_dims:
                                    if dim == 'strike':
                                        # Use min/max strike or current price if available
                                        if self.min_strike is not None and self.max_strike is not None:
                                            strikes = [self.min_strike, self.current_price, self.max_strike]
                                        else:
                                            strikes = [self.current_price]
                                        self.ds = self.ds.expand_dims({'strike': strikes})
                                    elif dim == 'DTE':
                                        # Default to 0 days to expiration
                                        self.ds = self.ds.expand_dims({'DTE': [0]})
                                    elif dim == 'option_type':
                                        # Default to both call and put
                                        self.ds = self.ds.expand_dims({'option_type': ['call', 'put']})
                                
                                logger.info(f"Recovered dataset with dimensions: {list(self.ds.dims)}")
                            except Exception as e:
                                logger.error(f"Failed to recover dataset: {str(e)}")
                                logger.error(f"Traceback: {traceback.format_exc()}")
                                # Set to None to force reprocessing
                                self.ds = None
                    else:
                        logger.warning("Dataset from cache is None")
                else:
                    logger.warning("Processed data from cache does not contain a dataset")
                
                # Get risk-free rate if available
                self.risk_free_rate = processed_data.get('risk_free_rate', self.get_risk_free_rate())
            else:
                logger.warning(f"Processed data from cache is not a dictionary: {type(options_data).__name__}")
                # Try to process the data as raw data
                logger.info("Attempting to process as raw data")
                is_processed = False
        
        # If not processed or no dataset was found, process the raw data
        if not is_processed or self.ds is None:
            # Process the raw data
            logger.info("Processing raw options data")
            start_time = time.time()
            try:
                self.ds = self.pre_process_data()
                pre_process_time = time.time() - start_time
                logger.info(f"Pre-processing completed in {pre_process_time:.2f} seconds")
                
                if self.ds is None:
                    logger.error("Failed to pre-process data, dataset is None")
                    return
                
                # Only interpolate if not skipped
                if not skip_interpolation:
                    start_time = time.time()
                    num_dates = len(self.get_expirations())
                    
                    # Try 2D interpolation first (default approach)
                    if num_dates >= 2:
                        logger.info(f"Using 2D interpolation for {num_dates} expiration dates")
                        self.interpolate_missing_values_2d()
                    # Fall back to 1D interpolation only if there's a single date
                    elif num_dates == 1:
                        logger.info("Falling back to 1D interpolation for single expiration date")
                        self.interpolate_missing_values_1d()
                    else:
                        logger.warning("No expiration dates found, skipping interpolation")
                        
                    interpolate_time = time.time() - start_time
                    logger.info(f"Interpolation completed in {interpolate_time:.2f} seconds")
                else:
                    logger.info("Skipping interpolation as requested (for immediate use only)")
                
                start_time = time.time()
                self.post_process_data()
                post_process_time = time.time() - start_time
                logger.info(f"Post-processing completed in {post_process_time:.2f} seconds")
                
                self.risk_free_rate = self.get_risk_free_rate()
                
                # Store processed data for caching
                self.options_data = {
                    'dataset': self.ds,
                    'min_strike': self.min_strike,
                    'max_strike': self.max_strike,
                    'risk_free_rate': self.risk_free_rate
                }
            except Exception as e:
                logger.error(f"Error processing options data: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

    def pre_process_data(self):
        """Pre-process the options data into a structured format."""
        try:
            logger.info(f"Processing options data with current price: {self.current_price}")
            
            # Get all expiration dates, filtering out metadata fields (those starting with underscore)
            exp_dates = [exp for exp in self.options_data.keys() if not exp.startswith('_')]
            
            if not exp_dates:
                logger.error("No expiration dates found in options data")
                return None
            
            # Sort expiration dates
            exp_dates.sort()
            
            # Process in batches to reduce memory usage
            batch_size = 10
            now = pd.Timestamp.now().normalize()
            all_dfs = []
            
            for i in range(0, len(exp_dates), batch_size):
                batch_exps = exp_dates[i:i+batch_size]
                batch_dfs = []
                
                for exp in batch_exps:
                    data = self.options_data[exp]
                    exp_date = pd.to_datetime(exp).normalize()
                    dte = max(0, (exp_date - now).days)
                    
                    for opt_type, df in [('call', data.get('calls', pd.DataFrame())), ('put', data.get('puts', pd.DataFrame()))]:
                        # Check if df is a DataFrame and not empty, or if it's a list with items
                        if (isinstance(df, pd.DataFrame) and not df.empty) or (isinstance(df, list) and df):
                            # Convert to DataFrame if it's a list
                            if isinstance(df, list):
                                df = pd.DataFrame(df)
                            
                            # Only keep necessary columns to reduce memory usage
                            keep_cols = [
                                'lastPrice', 'bid', 'ask', 'change', 'percentChange', 'volume', 
                                'openInterest', 'impliedVolatility', 'inTheMoney', 'strike',
                                'contractSymbol', 'lastTradeDate', 'contractSize', 'currency'
                            ]
                            df = df[[col for col in keep_cols if col in df.columns]].copy()
                            
                            df['option_type'] = opt_type
                            df['expiration'] = exp_date
                            df['DTE'] = dte
                            batch_dfs.append(df)
                
                if batch_dfs:
                    # Concatenate batch and append to main list
                    batch_df = pd.concat(batch_dfs, ignore_index=True)
                    all_dfs.append(batch_df)
                    
                    # Clear memory
                    del batch_dfs
            
            if not all_dfs:
                logger.error("No valid data to process")
                return None
            
            # Concatenate all batches
            df = pd.concat(all_dfs, ignore_index=True)
            
            # Define numeric columns explicitly
            numeric_cols = [
                'lastPrice', 'bid', 'ask', 'change', 'percentChange', 'volume', 
                'openInterest', 'impliedVolatility', 'inTheMoney', 'strike'
            ]
            
            # Convert to numeric, coercing errors to NaN
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info(f"Converted {col} to numeric, NaN count: {df[col].isna().sum()}")
            
            # Calculate derived values
            df['mid_price'] = (df['bid'] + df['ask']) / 2
            df['price'] = df['mid_price']
            
            # Calculate intrinsic values efficiently using vectorized operations
            mask_call = df['option_type'] == 'call'
            df.loc[mask_call, 'intrinsic_value'] = np.maximum(0, self.current_price - df.loc[mask_call, 'strike'])
            df.loc[~mask_call, 'intrinsic_value'] = np.maximum(0, df.loc[~mask_call, 'strike'] - self.current_price)
            df['extrinsic_value'] = df['price'] - df['intrinsic_value']
            
            # Store min/max strike
            self.min_strike = df['strike'].min()
            self.max_strike = df['strike'].max()
            
            # Create xarray dataset more efficiently
            strikes = sorted(df['strike'].unique())
            dtes = sorted(df['DTE'].unique())
            option_types = ['call', 'put']
            
            # Only include necessary columns in the dataset
            numeric_cols = [
                'lastPrice', 'bid', 'ask', 'change', 'percentChange', 'volume', 
                'openInterest', 'impliedVolatility', 'inTheMoney', 'strike',
                'mid_price', 'price', 'intrinsic_value', 'extrinsic_value'
            ]
            numeric_cols = [col for col in numeric_cols if col in df.columns]
            
            string_cols = ['contractSymbol', 'lastTradeDate', 'contractSize', 'currency']
            string_cols = [col for col in string_cols if col in df.columns]
            
            # Create dataset with optimized memory usage
            data_vars = {}
            
            # Ensure we have at least one DTE value
            if not dtes:
                logger.error("No DTE values found in the data")
                dtes = [0]  # Default to 0 days to expiration
            
            # Ensure we have at least one strike value
            if not strikes:
                logger.error("No strike values found in the data")
                strikes = [self.current_price]  # Default to current price
            
            for col in numeric_cols:
                # Skip 'strike' as it should only be a coordinate
                if col != 'strike':
                    data_vars[col] = (['strike', 'DTE', 'option_type'], 
                                    np.full((len(strikes), len(dtes), len(option_types)), np.nan, dtype=np.float32))
            
            for col in string_cols:
                data_vars[col] = (['strike', 'DTE', 'option_type'], 
                                 np.full((len(strikes), len(dtes), len(option_types)), None, dtype=object))
            
            ds = xr.Dataset(
                data_vars=data_vars, 
                coords={
                    'strike': strikes, 
                    'DTE': dtes, 
                    'option_type': option_types
                }
            )
            
            # Add expiration dates
            try:
                # Create a mapping of DTE to expiration date
                dte_to_exp = {}
                for dte in dtes:
                    dte_df = df[df['DTE'] == dte]
                    if not dte_df.empty:
                        dte_to_exp[dte] = dte_df['expiration'].iloc[0]
                    else:
                        # If no data for this DTE, use a default date
                        dte_to_exp[dte] = pd.Timestamp.now() + pd.Timedelta(days=dte)
                
                # Add expiration dates to coordinates
                ds.coords['expiration'] = ('DTE', np.array([dte_to_exp[dte] for dte in dtes], dtype='datetime64[ns]'))
            except Exception as e:
                logger.error(f"Error adding expiration dates to dataset: {str(e)}")
                # Add a default expiration date if we couldn't add the real ones
                default_dates = [pd.Timestamp.now() + pd.Timedelta(days=dte) for dte in dtes]
                ds.coords['expiration'] = ('DTE', np.array(default_dates, dtype='datetime64[ns]'))
            
            # Fill the dataset more efficiently
            for opt_type in option_types:
                opt_df = df[df['option_type'] == opt_type]
                for dte in dtes:
                    dte_df = opt_df[opt_df['DTE'] == dte]
                    if dte_df.empty:
                        continue
                        
                    for _, row in dte_df.iterrows():
                        strike = row['strike']
                        # Check if strike exists in the dataset coordinates
                        if strike not in ds.coords['strike'].values:
                            logger.warning(f"Strike {strike} not found in dataset coordinates, skipping")
                            continue
                        # Check if DTE exists in the dataset coordinates
                        if dte not in ds.coords['DTE'].values:
                            logger.warning(f"DTE {dte} not found in dataset coordinates, skipping")
                            continue
                        
                        try:
                            for col in numeric_cols:
                                if col != 'strike':  # Skip strike as it's a coordinate
                                    ds[col].loc[{'strike': strike, 'DTE': dte, 'option_type': opt_type}] = row[col]
                            for col in string_cols:
                                if col in row:
                                    ds[col].loc[{'strike': strike, 'DTE': dte, 'option_type': opt_type}] = str(row[col])
                        except KeyError as e:
                            logger.error(f"KeyError when setting data: {e}")
                            logger.error(f"Dataset dimensions: {list(ds.dims)}")
                            logger.error(f"Dataset coordinates: {list(ds.coords)}")
                            logger.error(f"Trying to set strike={strike}, DTE={dte}, option_type={opt_type}")
                            # Try to recover by ensuring all dimensions exist
                            if 'strike' not in ds.dims:
                                logger.warning("Adding missing 'strike' dimension")
                                ds = ds.expand_dims({'strike': strikes})
                            if 'DTE' not in ds.dims:
                                logger.warning("Adding missing 'DTE' dimension")
                                ds = ds.expand_dims({'DTE': dtes})
                            if 'option_type' not in ds.dims:
                                logger.warning("Adding missing 'option_type' dimension")
                                ds = ds.expand_dims({'option_type': option_types})
            
            logger.info("Successfully processed options data into xarray Dataset")
            return ds
        except Exception as e:
            logger.error(f"Error in pre_process_data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def interpolate_missing_values_1d(self):
        """
        Interpolate ONLY missing values (NaN, zero, or invalid) in 1D (single expiration date) 
        using a smooth IV curve and Black-Scholes model for all price fields.
        Preserves all valid market data from Yahoo Finance.
        """
        if not self.ds:
            logger.error("Cannot interpolate: No dataset available")
            return
        logger.info("Starting 1D interpolation using IV curve with Black-Scholes model")

        dte = self.ds.DTE.values[0]
        T = dte / 365.0  # Convert days to years
        S = self.current_price
        r = self.get_risk_free_rate()

        # Define fields to process
        price_fields = ['bid', 'ask', 'mid_price', 'price', 'extrinsic_value']
        # Define non-price fields that need special handling
        non_price_fields = ['volume', 'openInterest']

        for opt_type in ['call', 'put']:
            # Step 1: Create a smooth IV curve by interpolating impliedVolatility
            da_iv = self.ds['impliedVolatility'].sel(option_type=opt_type, DTE=dte)
            strikes = da_iv.strike.values
            
            # Store original values to preserve them
            original_values = da_iv.values.copy()
            
            # Check if any IV values need interpolation
            if da_iv.isnull().any() or (da_iv <= 0).any():
                values = da_iv.values
                valid = ~np.isnan(values) & (values > 0)
                
                if np.sum(valid) >= 2:  # Need at least 2 points to interpolate
                    # Try cubic spline interpolation first for smoother curve
                    try:
                        s = pd.Series(values, index=strikes)
                        s_interp = s.interpolate(method='cubic')
                        logger.info(f"Created smooth IV curve for {opt_type} using cubic interpolation")
                    except Exception as e:
                        # Fallback to linear if cubic fails
                        logger.warning(f"Cubic interpolation failed: {str(e)}. Falling back to linear.")
                        s_interp = pd.Series(values, index=strikes).interpolate(method='linear')
                        logger.info(f"Created IV curve for {opt_type} using linear interpolation")
                    
                    # Ensure no negative or zero IVs by setting a minimum
                    s_interp = s_interp.where(s_interp > 0.01, 0.01)  # Floor IV at 0.01
                    
                    # Only replace NaN or invalid values, preserve original valid data
                    mask = np.isnan(original_values) | (original_values <= 0)
                    combined_values = np.where(mask, s_interp.values, original_values)
                    
                    self.ds['impliedVolatility'].loc[{'option_type': opt_type, 'DTE': dte}] = combined_values
                else:
                    logger.warning(f"Not enough valid points to interpolate impliedVolatility for {opt_type} at DTE={dte}")
                    # Set a default IV if we can't interpolate
                    mask = np.isnan(da_iv.values) | (da_iv.values <= 0)
                    if mask.any():
                        default_iv = 0.3  # Default IV of 30%
                        logger.warning(f"Using default IV of {default_iv} for {np.sum(mask)} points")
                        self.ds['impliedVolatility'].loc[{'option_type': opt_type, 'DTE': dte}] = np.where(mask, default_iv, da_iv.values)

            # Step 2: Calculate average spread for this expiry and option type
            da_bid = self.ds['bid'].sel(option_type=opt_type, DTE=dte)
            da_ask = self.ds['ask'].sel(option_type=opt_type, DTE=dte)
            valid_mask = (~da_bid.isnull()) & (~da_ask.isnull())
            
            if valid_mask.any():
                spreads = (da_ask - da_bid).where(valid_mask)
                avg_spread = spreads.mean().item()
            else:
                avg_spread = 0.01  # Default spread if no valid bid/ask pairs
                
            # Step 3: Handle non-price fields like volume and openInterest separately
            # These should not be interpolated like continuous values
            for field in non_price_fields:
                if field in self.ds:
                    da_field = self.ds[field].sel(option_type=opt_type, DTE=dte)
                    
                    # For volume and openInterest, we should use nearest neighbor interpolation
                    # or just set missing values to 0 since they are discrete counts
                    if da_field.isnull().any():
                        # Get the values and create a mask for missing values
                        values = da_field.values
                        missing_mask = np.isnan(values)
                        
                        if missing_mask.any():
                            # For volume and openInterest, set missing values to 0
                            # This is more appropriate than interpolation for count data
                            filled_values = np.where(missing_mask, 0, values)
                            self.ds[field].loc[{'option_type': opt_type, 'DTE': dte}] = filled_values
                            logger.info(f"Set missing {field} values to 0 for {opt_type}")

            # Step 4: Use the smooth IV curve to calculate ALL price fields with Black-Scholes
            for i, strike in enumerate(strikes):
                # Get the IV from our smooth curve
                iv = self.ds['impliedVolatility'].sel(option_type=opt_type, DTE=dte, strike=strike).item()
                
                if pd.isna(iv) or iv <= 0 or T <= 0:
                    logger.warning(f"Invalid IV ({iv}) or T ({T}) for strike {strike}, {opt_type}. Setting default values.")
                    iv = 0.3  # Use a reasonable default
                
                # Calculate theoretical price using Black-Scholes
                if opt_type == 'call':
                    theo_price = call_price(S, strike, T, r, iv)
                    intrinsic = max(0, S - strike)
                else:
                    theo_price = put_price(S, strike, T, r, iv)
                    intrinsic = max(0, strike - S)
                
                # Calculate extrinsic value directly
                extrinsic = max(0, theo_price - intrinsic)
                
                # Only update values that are missing or invalid
                bid_val = self.ds['bid'].sel(option_type=opt_type, DTE=dte, strike=strike).item()
                ask_val = self.ds['ask'].sel(option_type=opt_type, DTE=dte, strike=strike).item()
                mid_val = self.ds['mid_price'].sel(option_type=opt_type, DTE=dte, strike=strike).item()
                price_val = self.ds['price'].sel(option_type=opt_type, DTE=dte, strike=strike).item()
                extrinsic_val = self.ds['extrinsic_value'].sel(option_type=opt_type, DTE=dte, strike=strike).item()
                
                # Update bid and ask if missing
                if pd.isna(bid_val):
                    self.ds['bid'].loc[{'option_type': opt_type, 'DTE': dte, 'strike': strike}] = max(0, theo_price - avg_spread / 2)
                
                if pd.isna(ask_val):
                    self.ds['ask'].loc[{'option_type': opt_type, 'DTE': dte, 'strike': strike}] = theo_price + avg_spread / 2
                
                # Update mid_price and price if missing
                if pd.isna(mid_val):
                    self.ds['mid_price'].loc[{'option_type': opt_type, 'DTE': dte, 'strike': strike}] = theo_price
                
                if pd.isna(price_val):
                    self.ds['price'].loc[{'option_type': opt_type, 'DTE': dte, 'strike': strike}] = theo_price
                
                # Update extrinsic value if missing
                if pd.isna(extrinsic_val):
                    self.ds['extrinsic_value'].loc[{'option_type': opt_type, 'DTE': dte, 'strike': strike}] = extrinsic
            
            logger.info(f"Applied Black-Scholes pricing using IV curve for {opt_type}")
            
            # Step 5: Ensure consistency by recalculating extrinsic value for all points
            intrinsic = self.ds['intrinsic_value'].sel(option_type=opt_type, DTE=dte)
            price = self.ds['price'].sel(option_type=opt_type, DTE=dte)
            extrinsic = price - intrinsic
            
            # Apply floor to ensure no negative extrinsic values
            extrinsic = xr.where(extrinsic < 0, 0, extrinsic)
            self.ds['extrinsic_value'].loc[{'option_type': opt_type, 'DTE': dte}] = extrinsic
            
            logger.info(f"Recalculated extrinsic values for {opt_type} to ensure consistency")

    def interpolate_missing_values_2d(self):
        """
        Interpolate ONLY missing values (NaN, zero, or invalid) in 2D (across strikes and expiration dates).
        First interpolates implied volatility surface, then uses it with Black-Scholes for all price fields.
        Preserves all valid market data from Yahoo Finance.
        """
        if not self.ds:
            logger.error("Cannot interpolate: No dataset available")
            return
        logger.info("Starting 2D interpolation using IV surface with Black-Scholes model")

        S = self.current_price
        r = self.get_risk_free_rate()
        dtes = self.ds.DTE.values
        if len(dtes) < 2:
            logger.info(f"Only {len(dtes)} DTE value(s) found, skipping 2D interpolation")
            return

        # Define fields to process
        price_fields = ['bid', 'ask', 'mid_price', 'price', 'extrinsic_value']
        # Define non-price fields that need special handling
        non_price_fields = ['volume', 'openInterest']

        for opt_type in ['call', 'put']:
            # Step 1: Create a smooth IV surface by interpolating impliedVolatility in 2D
            da_iv = self.ds['impliedVolatility'].sel(option_type=opt_type)
            
            # Store original IV values to preserve them
            original_iv = da_iv.values.copy()
            
            # Get grid coordinates
            strikes = da_iv.strike.values
            strike_grid, dte_grid = np.meshgrid(strikes, dtes, indexing='ij')
            points = np.column_stack([strike_grid.ravel(), dte_grid.ravel()])
            values_flat = da_iv.values.ravel()
            
            # Only consider valid IV values (not NaN and > 0)
            valid_mask = ~np.isnan(values_flat) & (values_flat > 0)
            
            if valid_mask.sum() > 3:  # Need at least 4 points for 2D interpolation
                # Perform 2D interpolation on valid points - use cubic if possible for smoother surface
                try:
                    # Try cubic interpolation first for smoother IV surface
                    interpolated_values = griddata(
                        points[valid_mask],
                        values_flat[valid_mask],
                        (strike_grid, dte_grid),
                        method='cubic'
                    )
                    
                    # Fill any NaNs from cubic with linear interpolation
                    nan_mask = np.isnan(interpolated_values)
                    if nan_mask.any():
                        linear_values = griddata(
                            points[valid_mask],
                            values_flat[valid_mask],
                            (strike_grid, dte_grid),
                            method='linear'
                        )
                        interpolated_values[nan_mask] = linear_values[nan_mask]
                    
                    # Apply additional smoothing to reduce high-frequency noise
                    # Use a Gaussian filter with adaptive sigma based on DTE
                    from scipy.ndimage import gaussian_filter
                    
                    # Reshape to match the original grid shape for proper smoothing
                    interpolated_values_reshaped = interpolated_values.reshape(len(strikes), len(dtes))
                    
                    # Apply stronger smoothing for longer-dated options to reduce oscillations
                    # Create a smoothed version for each DTE slice with appropriate sigma
                    for i, dte in enumerate(dtes):
                        # Adaptive sigma: more smoothing for longer-dated options
                        # This helps reduce oscillations in far-dated options
                        if dte <= 30:  # Short-dated options (< 1 month)
                            sigma = 0.7
                        elif dte <= 90:  # Medium-dated options (1-3 months)
                            sigma = 1.0
                        elif dte <= 365:  # Longer-dated options (3-12 months)
                            sigma = 1.5
                        else:  # LEAPS (> 1 year)
                            # Reduce excessive smoothing for LEAPS
                            sigma = 1.5 + min(1.5, (dte / 730.0))  # Cap the smoothing for very long-dated options
                        
                        # Apply 1D Gaussian smoothing to this DTE slice
                        interpolated_values_reshaped[:, i] = gaussian_filter(
                            interpolated_values_reshaped[:, i], 
                            sigma=sigma
                        )
                    
                    # Keep the reshaped version (don't flatten back)
                    interpolated_values = interpolated_values_reshaped
                        
                    logger.info(f"Created smooth IV surface for {opt_type} using cubic interpolation with linear fallback and adaptive Gaussian smoothing")
                except Exception as e:
                    # Fallback to linear if cubic fails
                    logger.warning(f"Cubic interpolation failed: {str(e)}. Falling back to linear.")
                    interpolated_values = griddata(
                        points[valid_mask],
                        values_flat[valid_mask],
                        (strike_grid, dte_grid),
                        method='linear'
                    )
                    
                    # Apply light smoothing even with linear interpolation
                    try:
                        from scipy.ndimage import gaussian_filter
                        
                        # Reshape to match the original grid shape for proper smoothing
                        interpolated_values_reshaped = interpolated_values.reshape(len(strikes), len(dtes))
                        
                        # Apply adaptive smoothing based on DTE
                        for i, dte in enumerate(dtes):
                            if dte <= 30:
                                sigma = 0.5
                            elif dte <= 90:
                                sigma = 0.8
                            elif dte <= 365:
                                sigma = 1.2
                            else:
                                # Reduce excessive smoothing for LEAPS
                                sigma = 1.2 + min(1.0, (dte / 730.0))
                            
                            interpolated_values_reshaped[:, i] = gaussian_filter(
                                interpolated_values_reshaped[:, i], 
                                sigma=sigma
                            )
                        
                        # Keep the reshaped version (don't flatten back)
                        interpolated_values = interpolated_values_reshaped
                    except Exception as e:
                        logger.warning(f"Gaussian smoothing failed: {str(e)}. Using unsmoothed values.")
                        # Make sure interpolated_values has the same shape as original_iv
                        interpolated_values = interpolated_values.reshape(original_iv.shape)
                    
                    logger.info(f"Created IV surface for {opt_type} using linear interpolation with adaptive smoothing")
                
                # Floor IV at 0.01 to ensure validity
                interpolated_values = np.where(
                    np.isnan(interpolated_values) | (interpolated_values <= 0), 
                    0.01, 
                    interpolated_values
                )
                
                # Create a combined array that preserves original valid values
                # Only replace NaN or invalid (≤ 0) values
                mask = np.isnan(original_iv) | (original_iv <= 0)
                combined_values = np.where(mask, interpolated_values, original_iv)
                
                # Update the dataset with the smooth IV surface
                self.ds['impliedVolatility'].loc[{'option_type': opt_type}] = combined_values
                logger.info(f"Created smooth IV surface for {opt_type}")
            else:
                logger.warning(f"Not enough valid points for 2D interpolation of impliedVolatility for {opt_type}")
                # Set a default IV if we can't interpolate
                mask = np.isnan(da_iv.values) | (da_iv.values <= 0)
                if mask.any():
                    default_iv = 0.3  # Default IV of 30%
                    logger.warning(f"Using default IV of {default_iv} for {np.sum(mask)} points")
                    self.ds['impliedVolatility'].loc[{'option_type': opt_type}] = np.where(mask, default_iv, da_iv.values)

            # Step 2: Calculate theoretical prices for ALL strikes and DTEs using Black-Scholes
            # This ensures consistency across all price fields
            theoretical_prices = np.zeros((len(strikes), len(dtes)))
            intrinsic_values = np.zeros((len(strikes), len(dtes)))
            extrinsic_values = np.zeros((len(strikes), len(dtes)))
            
            for i, strike in enumerate(strikes):
                for j, dte in enumerate(dtes):
                    T = dte / 365.0  # Convert days to years
                    if T <= 0:
                        continue
                    
                    # Get the IV from our smooth surface
                    iv = self.ds['impliedVolatility'].sel(strike=strike, DTE=dte, option_type=opt_type).item()
                    
                    if pd.isna(iv) or iv <= 0:
                        logger.warning(f"Invalid IV ({iv}) for strike {strike}, DTE {dte}, {opt_type}. Setting default value.")
                        iv = 0.3  # Use a reasonable default
                    
                    # Calculate theoretical price using Black-Scholes
                    if opt_type == 'call':
                        theo_price = call_price(S, strike, T, r, iv)
                        intrinsic = max(0, S - strike)
                    else:
                        theo_price = put_price(S, strike, T, r, iv)
                        intrinsic = max(0, strike - S)
                    
                    # Calculate extrinsic value directly from theoretical price
                    extrinsic = max(0, theo_price - intrinsic)
                    
                    # Store values in arrays
                    theoretical_prices[i, j] = theo_price
                    intrinsic_values[i, j] = intrinsic
                    extrinsic_values[i, j] = extrinsic
            
            # Apply additional smoothing to theoretical prices for long-dated options
            # This helps reduce oscillations in the final prices
            from scipy.ndimage import gaussian_filter
            for j, dte in enumerate(dtes):
                if dte > 365:  # Only apply to LEAPS
                    # Adaptive sigma based on time to expiry
                    price_sigma = 0.5 + (dte / 1000.0)
                    theoretical_prices[:, j] = gaussian_filter(theoretical_prices[:, j], sigma=price_sigma)
                    # Recalculate extrinsic after smoothing
                    extrinsic_values[:, j] = np.maximum(0, theoretical_prices[:, j] - intrinsic_values[:, j])
            
            # Step 3: Calculate average spread per expiration date for bid/ask
            da_bid = self.ds['bid'].sel(option_type=opt_type)
            da_ask = self.ds['ask'].sel(option_type=opt_type)
            valid_mask = (~da_bid.isnull()) & (~da_ask.isnull())
            
            if valid_mask.any():
                spreads = (da_ask - da_bid).where(valid_mask)
                avg_spread_per_dte = spreads.mean(dim='strike').fillna(0.01).to_pandas()
            else:
                # Default spread if no valid bid/ask pairs
                avg_spread_per_dte = pd.Series(0.01, index=dtes)
            
            # Step 4: Update price fields only where values are missing or invalid
            for i, strike in enumerate(strikes):
                for j, dte in enumerate(dtes):
                    theo_price = theoretical_prices[i, j]
                    intrinsic = intrinsic_values[i, j]
                    extrinsic = extrinsic_values[i, j]
                    avg_spread = avg_spread_per_dte.get(dte, 0.01)
                    
                    # Only update values that are missing or invalid
                    bid_val = self.ds['bid'].sel(strike=strike, DTE=dte, option_type=opt_type).item()
                    ask_val = self.ds['ask'].sel(strike=strike, DTE=dte, option_type=opt_type).item()
                    mid_val = self.ds['mid_price'].sel(strike=strike, DTE=dte, option_type=opt_type).item()
                    price_val = self.ds['price'].sel(strike=strike, DTE=dte, option_type=opt_type).item()
                    extrinsic_val = self.ds['extrinsic_value'].sel(strike=strike, DTE=dte, option_type=opt_type).item()
                    
                    # Set bid and ask based on theoretical price and spread
                    if pd.isna(bid_val):
                        self.ds['bid'].loc[{'strike': strike, 'DTE': dte, 'option_type': opt_type}] = max(0, theo_price - avg_spread / 2)
                    
                    if pd.isna(ask_val):
                        self.ds['ask'].loc[{'strike': strike, 'DTE': dte, 'option_type': opt_type}] = theo_price + avg_spread / 2
                    
                    # Update mid_price and price if missing
                    if pd.isna(mid_val):
                        self.ds['mid_price'].loc[{'strike': strike, 'DTE': dte, 'option_type': opt_type}] = theo_price
                    
                    if pd.isna(price_val):
                        self.ds['price'].loc[{'strike': strike, 'DTE': dte, 'option_type': opt_type}] = theo_price
                    
                    # Update extrinsic value if missing
                    if pd.isna(extrinsic_val):
                        self.ds['extrinsic_value'].loc[{'strike': strike, 'DTE': dte, 'option_type': opt_type}] = extrinsic
            
            logger.info(f"Applied Black-Scholes pricing using IV surface for {opt_type}")
            
            # Step 5: Handle non-price fields like volume and openInterest separately
            for field in non_price_fields:
                if field in self.ds:
                    da_field = self.ds[field].sel(option_type=opt_type)
                    
                    # For volume and openInterest, we should use nearest neighbor interpolation
                    # or just set missing values to 0 since they are discrete counts
                    if da_field.isnull().any():
                        # Get the values and create a mask for missing values
                        values = da_field.values
                        missing_mask = np.isnan(values)
                        
                        if missing_mask.any():
                            # For volume and openInterest, set missing values to 0
                            # This is more appropriate than interpolation for count data
                            filled_values = np.where(missing_mask, 0, values)
                            self.ds[field].loc[{'option_type': opt_type}] = filled_values
                            logger.info(f"Set missing {field} values to 0 for {opt_type} in 2D interpolation")
            
            # Step 6: Ensure consistency by recalculating extrinsic value for all points
            for j, dte in enumerate(dtes):
                intrinsic = self.ds['intrinsic_value'].sel(option_type=opt_type, DTE=dte)
                price = self.ds['price'].sel(option_type=opt_type, DTE=dte)
                extrinsic = price - intrinsic
                
                # Apply floor to ensure no negative extrinsic values
                extrinsic = xr.where(extrinsic < 0, 0, extrinsic)
                self.ds['extrinsic_value'].loc[{'option_type': opt_type, 'DTE': dte}] = extrinsic
            
            logger.info(f"Recalculated extrinsic values for {opt_type} to ensure consistency")

    def post_process_data(self):
        """
        Post-process the data after interpolation:
        1. Recalculate intrinsic values based on current price
        2. Calculate extrinsic value
        3. Calculate spread (ask - bid)
        4. Calculate greeks if dimensions are present
        5. Apply floors and rounding
        """
        logger.info("Post-processing data after interpolation")
        
        try:
            # Get current price
            current_price = self.current_price
            
            # Recalculate intrinsic values based on current price
            # For calls: max(0, S - K)
            # For puts: max(0, K - S)
            
            # Create option type masks that match the dataset dimensions
            call_indices = np.where(np.array(self.ds.option_type.values) == 'call')[0]
            put_indices = np.where(np.array(self.ds.option_type.values) == 'put')[0]
            
            # Calculate intrinsic values for each option type separately
            if 'intrinsic_value' in self.ds:
                # For calls
                if len(call_indices) > 0:
                    call_data = self.ds.isel(option_type=call_indices)
                    # Ensure proper broadcasting by explicitly creating the array with correct dimensions
                    call_intrinsic = np.maximum(0, current_price - call_data.strike.values[:, np.newaxis])
                    # Ensure the shape matches the expected dimensions
                    if call_intrinsic.shape != self.ds['intrinsic_value'].sel(option_type='call').shape:
                        # Reshape to match the expected dimensions
                        call_intrinsic = np.broadcast_to(
                            call_intrinsic, 
                            self.ds['intrinsic_value'].sel(option_type='call').shape
                        )
                    self.ds['intrinsic_value'].loc[{'option_type': 'call'}] = call_intrinsic
                
                # For puts
                if len(put_indices) > 0:
                    put_data = self.ds.isel(option_type=put_indices)
                    # Ensure proper broadcasting by explicitly creating the array with correct dimensions
                    put_intrinsic = np.maximum(0, put_data.strike.values[:, np.newaxis] - current_price)
                    # Ensure the shape matches the expected dimensions
                    if put_intrinsic.shape != self.ds['intrinsic_value'].sel(option_type='put').shape:
                        # Reshape to match the expected dimensions
                        put_intrinsic = np.broadcast_to(
                            put_intrinsic, 
                            self.ds['intrinsic_value'].sel(option_type='put').shape
                        )
                    self.ds['intrinsic_value'].loc[{'option_type': 'put'}] = put_intrinsic
            
            # Calculate extrinsic value (price - intrinsic)
            if 'extrinsic_value' in self.ds and 'price' in self.ds and 'intrinsic_value' in self.ds:
                self.ds['extrinsic_value'] = self.ds['price'] - self.ds['intrinsic_value']
                # Ensure extrinsic value is not negative
                self.ds['extrinsic_value'] = xr.where(self.ds['extrinsic_value'] < 0, 0, self.ds['extrinsic_value'])
            
            # Calculate spread (ask - bid)
            if 'ask' in self.ds and 'bid' in self.ds:
                self.ds['spread'] = self.ds['ask'] - self.ds['bid']
            
            # Calculate greeks if we have the necessary dimensions
            if 'strike' in self.ds.dims and 'DTE' in self.ds.dims and 'option_type' in self.ds.dims:
                try:
                    # Calculate greeks using Black-Scholes model
                    self.calculate_black_scholes_greeks()
                except Exception as e:
                    logger.error(f"Error computing Black-Scholes greeks: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Fallback to numerical greeks
                    try:
                        self.compute_delta()
                        self.compute_gamma()
                        self.compute_theta()
                    except Exception as e:
                        logger.error(f"Error computing numerical greeks: {str(e)}")
            else:
                logger.warning("Cannot calculate greeks: missing required dimensions")
            
            # Apply floors to ensure all values are reasonable
            self.apply_floors()
            
            logger.info("Post-processing complete")
            return self.ds
            
        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue with what we have
            return self.ds

    def calculate_black_scholes_greeks(self):
        """Calculate option greeks using Black-Scholes formulas."""
        try:
            # Get required parameters
            S = self.current_price  # Current price
            r = self.get_risk_free_rate()  # Risk-free rate
            
            # Check if we have the necessary dimensions and data
            if 'impliedVolatility' not in self.ds:
                logger.warning("Cannot calculate Black-Scholes greeks: missing implied volatility data")
                return
                
            # Get the shape of the dataset
            strikes = self.ds.strike.values
            dtes = self.ds.DTE.values
            option_types = self.ds.option_type.values
            
            # Create arrays to store the greeks
            delta_array = np.zeros((len(strikes), len(dtes), len(option_types)))
            gamma_array = np.zeros((len(strikes), len(dtes), len(option_types)))
            theta_array = np.zeros((len(strikes), len(dtes), len(option_types)))
            
            # Import the Black-Scholes functions
            from python.models.black_scholes import delta, gamma, theta
            
            # Loop through each option type, strike, and DTE
            for i, opt_type in enumerate(option_types):
                for j, strike in enumerate(strikes):
                    for k, dte in enumerate(dtes):
                        try:
                            # Get implied volatility for this option
                            sigma = self.ds['impliedVolatility'].sel(
                                strike=strike, 
                                DTE=dte, 
                                option_type=opt_type
                            ).item()
                            
                            # Skip if implied volatility is invalid
                            if np.isnan(sigma) or sigma <= 0:
                                # Use a reasonable default for long-dated options
                                if dte > 365:
                                    sigma = 0.2  # 20% IV as a reasonable default for LEAPS
                                else:
                                    continue
                            
                            # Cap extremely high IVs that can cause numerical issues
                            if sigma > 2.0:  # Cap at 200%
                                sigma = 2.0
                            
                            # Time to expiration in years
                            T = dte / 365.0
                            
                            # Skip if time to expiration is too small
                            if T <= 0.001:
                                continue
                                
                            # For very long-dated options, cap T to avoid numerical issues
                            if T > 10.0:
                                T = 10.0
                            
                            # Calculate delta
                            delta_array[j, k, i] = delta(S, strike, T, r, sigma, opt_type)
                            
                            # Calculate gamma (same for calls and puts)
                            gamma_array[j, k, i] = gamma(S, strike, T, r, sigma)
                            
                            # Calculate theta (daily)
                            theta_array[j, k, i] = theta(S, strike, T, r, sigma, opt_type) / 365.0
                        except Exception as e:
                            logger.debug(f"Error calculating greeks for {opt_type}, strike={strike}, DTE={dte}: {str(e)}")
            
            # Add or update the greeks in the dataset
            if 'delta' not in self.ds:
                self.ds['delta'] = (('strike', 'DTE', 'option_type'), delta_array)
            else:
                self.ds['delta'].values = delta_array
                
            if 'gamma' not in self.ds:
                self.ds['gamma'] = (('strike', 'DTE', 'option_type'), gamma_array)
            else:
                self.ds['gamma'].values = gamma_array
                
            if 'theta' not in self.ds:
                self.ds['theta'] = (('strike', 'DTE', 'option_type'), theta_array)
            else:
                self.ds['theta'].values = theta_array
            
            logger.info("Successfully calculated Black-Scholes greeks")
        except Exception as e:
            logger.error(f"Error in calculate_black_scholes_greeks: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def compute_delta(self):
        """Compute delta (first derivative of price with respect to strike)."""
        try:
            # Check if we have at least 2 strike values
            if len(self.ds.strike) < 2:
                logger.warning("Cannot compute delta: need at least 2 strike values")
                return
                
            delta = np.gradient(self.ds['price'].values, self.ds.strike.values, axis=0)
            self.ds['delta'] = (('strike', 'DTE', 'option_type'), delta)
        except Exception as e:
            logger.error(f"Error computing delta: {str(e)}")

    def compute_gamma(self):
        """Compute gamma (second derivative of price with respect to strike)."""
        try:
            # Check if delta exists and we have at least 2 strike values
            if 'delta' not in self.ds or len(self.ds.strike) < 2:
                logger.warning("Cannot compute gamma: need delta and at least 2 strike values")
                return
                
            gamma = np.gradient(self.ds['delta'].values, self.ds.strike.values, axis=0)
            self.ds['gamma'] = (('strike', 'DTE', 'option_type'), gamma)
        except Exception as e:
            logger.error(f"Error computing gamma: {str(e)}")

    def compute_theta(self):
        """Compute theta (derivative of price with respect to time)."""
        try:
            # Check if we have at least 2 DTE values
            if len(self.ds.DTE) < 2:
                logger.warning("Cannot compute theta: need at least 2 DTE values")
                return
                
            theta = -np.gradient(self.ds['price'].values, self.ds.DTE.values, axis=1)
            self.ds['theta'] = (('strike', 'DTE', 'option_type'), theta)
        except Exception as e:
            logger.error(f"Error computing theta: {str(e)}")

    def apply_floors(self):
        """
        Apply floors and rounding to ensure reasonable values after interpolation:
        1. Apply a floor of zero to all numeric values
        2. Round all dollar-denominated fields to the nearest $0.05
        3. Apply specific constraints to greeks and other fields
        """
        logger.info("Applying floors and rounding to ensure reasonable values after interpolation")
        
        # Define dollar-denominated fields that need rounding to nearest $0.05
        dollar_fields = ['bid', 'ask', 'mid_price', 'price', 'intrinsic_value', 'extrinsic_value', 'spread']
        
        # Apply floor of zero to all numeric fields
        for field in self.ds.data_vars:
            if np.issubdtype(self.ds[field].dtype, np.number) and field != 'impliedVolatility':
                # Apply floor of zero
                self.ds[field] = xr.where(self.ds[field] < 0, 0, self.ds[field])
                
                # Round dollar-denominated fields to nearest $0.05
                if field in dollar_fields:
                    # Round to nearest $0.05 (multiply by 20, round, divide by 20)
                    self.ds[field] = (self.ds[field] * 20).round() / 20
                    logger.debug(f"Applied zero floor and $0.05 rounding to {field}")
        
        # Special handling for implied volatility - floor at 0.01 and cap at 5.0
        iv_mask_low = (self.ds['impliedVolatility'].isnull() | (self.ds['impliedVolatility'] < 0.01))
        iv_mask_high = (self.ds['impliedVolatility'] > 5.0)
        
        if iv_mask_low.any():
            logger.info(f"Applying floor to {iv_mask_low.sum().item()} implied volatility values")
            self.ds['impliedVolatility'] = xr.where(iv_mask_low, 0.01, self.ds['impliedVolatility'])
            
        if iv_mask_high.any():
            logger.info(f"Capping {iv_mask_high.sum().item()} high implied volatility values")
            self.ds['impliedVolatility'] = xr.where(iv_mask_high, 5.0, self.ds['impliedVolatility'])
        
        # Ensure ask >= bid
        ask_lt_bid_mask = self.ds['ask'] < self.ds['bid']
        if ask_lt_bid_mask.any():
            logger.warning(f"Found {ask_lt_bid_mask.sum().item()} cases where ask < bid, fixing...")
            self.ds['ask'] = xr.where(ask_lt_bid_mask, self.ds['bid'] * 1.05, self.ds['ask'])
            # Re-round ask prices after adjustment
            self.ds['ask'] = (self.ds['ask'] * 20).round() / 20
        
        # Recalculate mid_price after floor and rounding
        self.ds['mid_price'] = (self.ds['bid'] + self.ds['ask']) / 2
        # Re-round mid_price
        self.ds['mid_price'] = (self.ds['mid_price'] * 20).round() / 20
        self.ds['price'] = self.ds['mid_price']
        
        # Calculate or update spread (ask - bid)
        self.ds['spread'] = self.ds['ask'] - self.ds['bid']
        # Round spread to nearest $0.05
        self.ds['spread'] = (self.ds['spread'] * 20).round() / 20
        
        # Special handling for greeks if they exist
        if 'delta' in self.ds:
            # Delta should be between -1 and 1
            delta_low_mask = self.ds['delta'] < -1
            delta_high_mask = self.ds['delta'] > 1
            
            if delta_low_mask.any() or delta_high_mask.any():
                logger.info(f"Constraining delta values to [-1, 1]")
                self.ds['delta'] = xr.where(delta_low_mask, -1, self.ds['delta'])
                self.ds['delta'] = xr.where(delta_high_mask, 1, self.ds['delta'])
        
        if 'gamma' in self.ds:
            # Gamma should be positive and reasonably bounded
            gamma_neg_mask = self.ds['gamma'] < 0
            gamma_high_mask = self.ds['gamma'] > 1
            
            if gamma_neg_mask.any():
                logger.info(f"Flooring {gamma_neg_mask.sum().item()} negative gamma values")
                self.ds['gamma'] = xr.where(gamma_neg_mask, 0, self.ds['gamma'])
                
            if gamma_high_mask.any():
                logger.info(f"Capping {gamma_high_mask.sum().item()} high gamma values")
                self.ds['gamma'] = xr.where(gamma_high_mask, 1, self.ds['gamma'])
        
        if 'theta' in self.ds:
            # Theta should be reasonably bounded
            theta_low_mask = self.ds['theta'] < -10
            theta_high_mask = self.ds['theta'] > 10
            
            if theta_low_mask.any() or theta_high_mask.any():
                logger.info(f"Constraining theta values to [-10, 10]")
                self.ds['theta'] = xr.where(theta_low_mask, -10, self.ds['theta'])
                self.ds['theta'] = xr.where(theta_high_mask, 10, self.ds['theta'])
        
        # Recalculate extrinsic value after all other adjustments
        self.ds['extrinsic_value'] = self.ds['price'] - self.ds['intrinsic_value']
        self.ds['extrinsic_value'] = xr.where(self.ds['extrinsic_value'] < 0, 0, self.ds['extrinsic_value'])
        # Re-round extrinsic value
        self.ds['extrinsic_value'] = (self.ds['extrinsic_value'] * 20).round() / 20
        
        logger.info("Floor application and rounding complete")

    def get_data(self):
        return self.ds if self.ds is not None else logger.error("No processed data available")

    def get_data_frame(self):
        return self.ds.to_dataframe().reset_index() if self.ds is not None else None

    def get_nearest_expiry(self):
        return pd.Timestamp(self.ds.expiration.sel(DTE=self.ds.DTE.min()).item()) if self.ds is not None else None

    def get_expirations(self):
        return [pd.Timestamp(exp) for exp in sorted(self.ds.expiration.values)] if self.ds is not None else []

    def get_strike_range(self):
        return self.min_strike, self.max_strike

    def get_data_for_expiry(self, expiry_date):
        if not self.ds:
            return None
        expiry_date = pd.Timestamp(expiry_date)
        closest_idx = np.argmin(abs(self.ds.expiration.values - np.datetime64(expiry_date)))
        closest_dte = self.ds.DTE.values[closest_idx]
        return self.ds.sel(DTE=closest_dte).to_dataframe().reset_index().dropna(subset=['price'])

    def get_risk_free_rate(self):
        return YahooFinanceAPI().get_risk_free_rate("^TNX")

    def force_reinterpolate(self):
        """Force a complete reinterpolation of all values, regardless of current state.
        This is useful when debugging interpolation issues or when the data has gaps.
        """
        logger.info("Forcing complete reinterpolation of all values")
        
        if not self.ds:
            logger.error("Cannot reinterpolate: No dataset available")
            return False
            
        try:
            # Get number of expiration dates
            num_dates = len(self.get_expirations())
            logger.info(f"Found {num_dates} expiration dates")
            
            # First try 2D interpolation if we have multiple dates
            if num_dates >= 2:
                logger.info("Performing 2D interpolation across all dates")
                self.interpolate_missing_values_2d()
                
                # Check if we still have missing values
                missing_count = self.count_missing_values()
                logger.info(f"After 2D interpolation: {missing_count} missing values")
                
                # If we still have missing values, try 1D interpolation for each date
                if missing_count > 0:
                    logger.info("Performing 1D interpolation for each date to fill remaining gaps")
                    for dte in self.ds.DTE.values:
                        # Create a temporary processor with just this date
                        temp_ds = self.ds.sel(DTE=dte)
                        # We need to manually interpolate each date
                        for opt_type in ['call', 'put']:
                            for variable in ['bid', 'ask', 'mid_price', 'price', 'impliedVolatility']:
                                if variable not in self.ds.data_vars:
                                    continue
                                    
                                da = self.ds[variable].sel(option_type=opt_type, DTE=dte)
                                if da.isnull().any():
                                    strikes = da.strike.values
                                    values = da.values
                                    valid = ~np.isnan(values)
                                    if np.sum(valid) >= 2:
                                        s = pd.Series(values, index=strikes).interpolate(method='linear')
                                        self.ds[variable].loc[{'option_type': opt_type, 'DTE': dte}] = s.values
                                        logger.info(f"1D fallback interpolated {variable} for {opt_type} at DTE={dte}")
            # If we only have one date, use 1D interpolation
            elif num_dates == 1:
                logger.info("Performing 1D interpolation for single date")
                self.interpolate_missing_values_1d()
            else:
                logger.warning("No expiration dates found, cannot interpolate")
                return False
                
            # Apply floors to ensure all values are reasonable
            self.apply_floors()
            
            # Check final missing count
            final_missing = self.count_missing_values()
            logger.info(f"Final interpolation result: {final_missing} missing values")
            
            return final_missing == 0
            
        except Exception as e:
            logger.error(f"Error during forced reinterpolation: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def count_missing_values(self):
        """Count the number of missing values in key fields."""
        if not self.ds:
            return -1
            
        count = 0
        for field in ['bid', 'ask', 'mid_price', 'price', 'impliedVolatility']:
            if field in self.ds.data_vars:
                count += self.ds[field].isnull().sum().item()
                
        return count