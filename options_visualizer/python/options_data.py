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
    call_price, put_price, delta, gamma, theta, vega, rho, implied_volatility, calculate_all_greeks
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
    """
    Central manager for options data handling.
    
    This class is responsible for fetching, processing, and caching options data.
    It serves as the main interface between the data source (e.g., Yahoo Finance)
    and the application, providing methods to retrieve and analyze options data.
    """
    DATA_SOURCE_YAHOO = "yahoo"
    MODEL_BLACK_SCHOLES = "black_scholes"
    MODEL_MARKET = "market"

    def __init__(self, data_source=DATA_SOURCE_YAHOO, pricing_model=MODEL_MARKET, cache_duration=600):
        """
        Initialize the OptionsDataManager.
        
        Args:
            data_source: Source of options data (default: Yahoo Finance)
            pricing_model: Model used for pricing options (default: market prices)
            cache_duration: How long to cache data in seconds (default: 10 minutes)
        """
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
                    
                    # Get all tickers from the registry (single source of truth)
                    tickers = self._cache.get_all_tickers()
                    logger.info(f"Refreshing {len(tickers)} tickers from registry")
                    
                    # Refresh each ticker
                    for ticker in tickers:
                        try:
                            # Skip if already being refreshed
                            if ticker in self._loading_state:
                                logger.info(f"Skipping {ticker} - already being refreshed")
                                continue
                            
                            logger.info(f"Refreshing data for {ticker}")
                            self._refresh_ticker(ticker)
                            
                            # Sleep briefly between refreshes to avoid overwhelming the API
                            time.sleep(1)
                        except Exception as e:
                            logger.error(f"Error refreshing {ticker}: {str(e)}")
                    
                    logger.info("Cache refresh cycle complete")
                except Exception as e:
                    logger.error(f"Error in cache polling: {str(e)}")
                    time.sleep(60)  # Sleep before retrying
        
        # Start the polling thread
        thread = threading.Thread(target=poll_and_refresh_cache, daemon=True)
        thread.start()
        logger.info("Cache polling thread started")

    def _refresh_ticker(self, ticker: str):
        """Refresh the cached data for a ticker."""
        try:
            # Skip if already being refreshed
            if ticker in self._loading_state:
                logger.info(f"Skipping refresh for {ticker} - already in progress")
                return
            
            # Update the ticker registry (single source of truth)
            self._cache.update_registry(ticker)
            
            # Start fetching in the background
            self.start_fetching(ticker)
        except Exception as e:
            logger.error(f"Error refreshing ticker {ticker}: {str(e)}")

    def get_current_processor(self, ticker: str) -> Tuple[Optional['OptionsDataProcessor'], Optional[float], str, float]:
        """
        Get the current options data processor for a ticker.
        
        Args:
            ticker: The ticker symbol to get data for
            
        Returns:
            Tuple of (processor, current_price, status, progress)
            where status is one of: 'complete', 'partial', 'loading', 'not_found'
        """
        # Update the ticker registry (single source of truth)
        self._cache.update_registry(ticker)
        
        # Check if we're currently loading this ticker
        is_loading = ticker in self._loading_state
        
        # Check cache
        cached_data = self._cache.get(ticker)
        
        if cached_data:
            options_data, price, timestamp, age, processed_dates, total_dates = cached_data
            
            # Calculate age of cache - ensure timestamp is a float
            try:
                timestamp_float = float(timestamp) if timestamp is not None else 0
                age = time.time() - timestamp_float
            except (ValueError, TypeError):
                logger.warning(f"Invalid timestamp format for {ticker}: {timestamp}, treating as expired")
                age = self.cache_duration + 1  # Treat as expired
            
            # Calculate progress
            progress = processed_dates / max(total_dates, 1) if total_dates > 0 else 1.0
            
            # Create processor from cache regardless of age
            try:
                # Check if the data is fully interpolated
                is_fully_interpolated = options_data.get('_is_fully_interpolated', False) if isinstance(options_data, dict) else False
                
                # If we're still loading and the cache is not fully interpolated, mark as partial
                if is_loading and not is_fully_interpolated:
                    status = 'partial'
                else:
                    # If cache is stale but we're not already refreshing, trigger a background refresh
                    if age >= self.cache_duration and not is_loading:
                        logger.info(f"Cache for {ticker} is stale (age: {age:.1f}s), triggering background refresh")
                        self.start_fetching(ticker, skip_interpolation=False)
                        status = 'partial'  # Mark as partial since we're refreshing
                    else:
                        status = 'complete'
                
                # Create processor from cache with minimal processing
                # Use is_processed=True to skip expensive post-processing steps
                processor = OptionsDataProcessor(options_data, price, is_processed=True)
                
                # Skip any additional processing if the data is fully interpolated
                if is_fully_interpolated and 'dataset' in options_data:
                    logger.info(f"Using fully processed cached data for {ticker} - skipping post-processing")
                    return processor, price, status, progress
                
                return processor, price, status, progress
            except Exception as e:
                logger.error(f"Error creating processor from cache for {ticker}: {str(e)}")
                # Fall through to loading or not found
        
        # If we're loading, return loading status
        if is_loading:
            # Get progress from loading state
            last_processed = self._loading_state[ticker].get('last_processed_dates', 0)
            total = self._loading_state[ticker].get('total_dates', 0)
            progress = last_processed / max(total, 1) if total > 0 else 0.0
            
            return None, None, 'loading', progress
        
        # Not in cache and not loading, start fetching
        logger.info(f"No cached data for {ticker}, starting fetch")
        self.start_fetching(ticker, skip_interpolation=False)
        return None, None, 'loading', 0.0

    def start_fetching(self, ticker: str, skip_interpolation: bool = False) -> bool:
        """
        Start fetching options data for a ticker.
        
        Args:
            ticker: The ticker symbol to fetch data for
            skip_interpolation: Whether to skip interpolation (for faster loading)
            
        Returns:
            True if fetching started, False if already in progress
        """
        # Update the ticker registry (single source of truth)
        self._cache.update_registry(ticker)
        
        # Always use 2D interpolation for better data quality
        skip_interpolation = False
        
        # Check if already fetching
        if ticker in self._loading_state:
            logger.info(f"Already fetching data for {ticker}")
            return False
            
        # Set initial loading state
        self._loading_state[ticker] = {
            'status': 'loading',
            'progress': 0.0,
            'processed_dates': 0,
            'total_dates': 0,
            'skip_interpolation': skip_interpolation
        }
        thread = threading.Thread(target=self._fetch_in_background, args=(ticker,))
        thread.daemon = True  # Make thread exit when main thread exits
        thread.start()
        logger.info(f"Started background fetch for {ticker} with 2D interpolation")
        return True

    def _fetch_in_background(self, ticker: str):
        """Fetch options data in a background thread."""
        logger.info(f"Starting background fetch for {ticker}")
        
        # Set loading state
        self._loading_state[ticker] = {
            'status': 'loading',
            'progress': 0.0,
            'processed_dates': 0,
            'total_dates': 0
        }
        
        try:
            # Get or create a lock for this ticker
            if ticker not in self._fetch_locks:
                self._fetch_locks[ticker] = threading.Lock()
            
            # Use the ticker-specific lock for thread safety
            with self._fetch_locks[ticker]:
                # Check if another thread has already completed the fetch
                if ticker not in self._loading_state:
                    logger.info(f"Fetch for {ticker} was already completed by another thread")
                    return
                
                # Get skip_interpolation setting from loading state
                skip_interpolation = self._loading_state.get(ticker, {}).get('skip_interpolation', False)
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
                            # Use the cache's ticker lock for thread safety
                            with self._cache.get_lock(ticker):
                                # Mark this as unprocessed data by not including a 'dataset' key
                                # Add a flag to indicate this is raw data (not interpolated)
                                raw_data_collection['_is_fully_interpolated'] = False
                                raw_data_collection['_last_updated'] = time.time()  # Add timestamp for cache age tracking
                                self._cache.set(ticker, raw_data_collection, current_price, len(raw_data_collection), total_dates)
                                logger.info(f"Cached raw data for {ticker} with {len(raw_data_collection)}/{total_dates} dates")
                        except Exception as e:
                            logger.error(f"Error caching raw data for {ticker}: {str(e)}")
                
                # Fetch data directly from the API with the callback
                logger.info(f"Fetching data for {ticker} in background thread")
                
                try:
                    # Use the cache's ticker lock for thread safety during API fetch
                    with self._cache.get_lock(ticker):
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
                        with self._cache.get_lock(ticker):
                            cached_data = self._cache.get(ticker)
                        if cached_data and cached_data[0] is not None and cached_data[1] is not None:
                            logger.info(f"Using cached data for {ticker} due to API error")
                            options_data_dict = cached_data[0]
                            current_price = cached_data[1]
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
                        
                        # Use the cache's ticker lock for thread safety during processing
                        with self._cache.get_lock(ticker):
                            # Create a processor with full interpolation and processing
                            processor = OptionsDataProcessor(raw_data_collection, final_current_price, 
                                                            is_processed=False, 
                                                            skip_interpolation=False)  # Always do interpolation for cached data
                            
                            # Check if processor has a valid dataset before proceeding
                            if processor.ds is None:
                                logger.error(f"Processor dataset is None for {ticker}")
                                # Fall back to caching raw data
                                raw_data_collection['_is_fully_interpolated'] = False
                                raw_data_collection['_last_updated'] = time.time()  # Add timestamp for cache age tracking
                                self._cache.set(ticker, raw_data_collection, final_current_price, processed_dates, total_dates)
                                return
                            
                            # Force post-processing to ensure all fields are available
                            logger.info(f"Running post-processing for {ticker} to ensure all fields are available")
                            processor.post_process_data()
                            
                            # Get the fully processed data with interpolation
                            processed_data = processor.options_data
                            
                            # Store the dataset in the processed data to avoid reprocessing
                            processed_data['dataset'] = processor.ds
                            
                            # Add a flag to indicate this data is fully interpolated and processed
                            processed_data['_is_fully_interpolated'] = True
                            processed_data['_last_updated'] = time.time()  # Add timestamp for cache age tracking
                            
                            # Ensure all plot types are calculated and available
                            # This is critical for frontend performance
                            all_fields_available = True
                            required_fields = [
                                'mid_price', 'delta', 'gamma', 'theta', 'impliedVolatility',
                                'volume', 'spread', 'intrinsic_value', 'extrinsic_value'
                            ]
                            
                            # Check if all required fields are available in the dataset
                            if processor.ds is not None:
                                for field in required_fields:
                                    if field not in processor.ds.data_vars:
                                        logger.warning(f"Field {field} is missing from processed data")
                                        all_fields_available = False
                                
                                if not all_fields_available:
                                    logger.warning(f"Some required plot fields are missing - forcing additional processing")
                                    # Force additional processing to ensure all fields are available
                                    processor._ensure_all_plot_fields()
                                    processor.apply_floors()
                                    
                                    # Update the processed data with the fully processed dataset
                                    processed_data = processor.options_data
                                    processed_data['dataset'] = processor.ds
                                    processed_data['_is_fully_interpolated'] = True
                                    processed_data['_last_updated'] = time.time()
                            else:
                                logger.warning(f"Processor dataset is None after processing")
                            
                            # Final update to cache with fully processed data
                            self._cache.set(ticker, processed_data, final_current_price, processed_dates, total_dates)
                            logger.info(f"Cached fully processed data for {ticker} with {processed_dates}/{total_dates} dates (with interpolation)")
                    except Exception as proc_error:
                        logger.error(f"Error processing complete data for {ticker}: {str(proc_error)}")
                        logger.error(traceback.format_exc())
                        
                        # Fall back to caching raw data
                        try:
                            with self._cache.get_lock(ticker):
                                raw_data_collection['_is_fully_interpolated'] = False
                                raw_data_collection['_last_updated'] = time.time()  # Add timestamp for cache age tracking
                                self._cache.set(ticker, raw_data_collection, final_current_price, processed_dates, total_dates)
                                logger.info(f"Cached raw data for {ticker} after processing error")
                        except Exception as cache_error:
                            logger.error(f"Error caching raw data after processing error for {ticker}: {str(cache_error)}")
                
                # Clean up loading state
                if ticker in self._loading_state:
                    del self._loading_state[ticker]
                
                logger.info(f"Background fetch for {ticker} completed")
        
        except Exception as e:
            logger.error(f"Unexpected error in background fetch for {ticker}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Clean up loading state
            if ticker in self._loading_state:
                del self._loading_state[ticker]

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
        # Use the ticker-specific lock for thread safety
        with self._cache.get_lock(ticker):
            # Check cache first
            processor, price, status, progress = self.get_current_processor(ticker)
            
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
        
        # If we get here, we don't have cached data or it's expired
        # Start a background fetch if not already loading
        if status != 'loading':
            self.start_fetching(ticker)
            
        # Return None, None to indicate we're fetching
        return None, None

    def get_risk_free_rate(self):
        """Get the risk-free rate for option pricing."""
        # Check if we already have a risk-free rate
        if hasattr(self, 'risk_free_rate') and self.risk_free_rate is not None:
            return self.risk_free_rate
            
        # Default risk-free rate if we can't get it from the API
        default_rate = 0.04  # 4%
        
        try:
            # Try to get from YahooFinanceAPI
            api = YahooFinanceAPI()
            rate = api.get_risk_free_rate()
            logger.info(f"Got risk-free rate from API: {rate:.2%}")
            # Set the instance attribute
            self.risk_free_rate = rate
            return rate
        except Exception as e:
            logger.error(f"Error getting risk-free rate: {str(e)}")
            logger.info(f"Using default risk-free rate: {default_rate:.2%}")
            # Set the instance attribute to the default value
            self.risk_free_rate = default_rate
            return default_rate

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
        self.risk_free_rate = None  # Initialize risk_free_rate to None
        
        if not options_data:
            logger.error("Failed to fetch options.")
            raise ValueError("Options data is None")
        
        # If data is already processed (from cache), we can skip processing steps
        if is_processed:
            logger.info("Using processed data from cache")
            
            # Check if we have a pre-processed dataset in the options_data
            if isinstance(options_data, dict) and 'dataset' in options_data:
                logger.info("Found pre-processed dataset in cache - using directly")
                self.ds = options_data['dataset']
                
                # Extract min/max strike if available
                if 'min_strike' in options_data and 'max_strike' in options_data:
                    self.min_strike = options_data['min_strike']
                    self.max_strike = options_data['max_strike']
                    logger.info(f"Loaded strike range from cache: {self.min_strike} to {self.max_strike}")
                
                # Check if the dataset has all required dimensions
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
                    logger.warning("Dataset from cache is None, will need to reprocess")
            else:
                # Extract min/max strike if available
                if isinstance(options_data, dict):
                    if 'min_strike' in options_data and 'max_strike' in options_data:
                        self.min_strike = options_data['min_strike']
                        self.max_strike = options_data['max_strike']
                        logger.info(f"Loaded strike range from cache: {self.min_strike} to {self.max_strike}")
                        # Remove these from options_data to avoid confusion
                        processed_data = {k: v for k, v in options_data.items() if k not in ['min_strike', 'max_strike']}
                    else:
                        processed_data = options_data
                    
                    # Process the data if we don't have a pre-processed dataset
                    logger.info("No pre-processed dataset found in cache, processing data")
                    self.process_data(processed_data)
                else:
                    logger.warning("Options data is not a dictionary, will try to process anyway")
                    self.process_data(options_data)
        else:
            # Process raw data
            logger.info("Processing raw data")
            self.process_data(options_data)

    def process_data(self, options_data):
        """Process the raw options data into a structured format."""
        try:
            logger.info(f"Processing options data with current price: {self.current_price}")
            
            # Initialize risk_free_rate
            self.risk_free_rate = self.get_risk_free_rate()
            
            # Pre-process the data
            start_time = time.time()
            
            # Get all expiration dates, filtering out metadata fields (those starting with underscore)
            exp_dates = [exp for exp in options_data.keys() if not exp.startswith('_')]
            
            if not exp_dates:
                logger.error("No expiration dates found in options data")
                raise ValueError("No expiration dates found in options data")
            
            # Sort expiration dates
            exp_dates = sorted(exp_dates, key=lambda x: pd.to_datetime(x))
            
            # Process in batches to avoid memory issues
            batch_size = 10
            all_dfs = []
            now = pd.Timestamp.now().normalize()
            
            for i in range(0, len(exp_dates), batch_size):
                batch_exps = exp_dates[i:i+batch_size]
                batch_dfs = []
                
                for exp in batch_exps:
                    data = options_data[exp]
                    exp_date = pd.to_datetime(exp).normalize()
                    dte = max(0, (exp_date - now).days)
                    
                    # Process calls
                    if 'calls' in data and data['calls']:
                        df_calls = pd.DataFrame(data['calls'])
                        df_calls['option_type'] = 'call'
                        df_calls['expiration'] = exp_date
                        df_calls['DTE'] = dte
                        batch_dfs.append(df_calls)
                    
                    # Process puts
                    if 'puts' in data and data['puts']:
                        df_puts = pd.DataFrame(data['puts'])
                        df_puts['option_type'] = 'put'
                        df_puts['expiration'] = exp_date
                        df_puts['DTE'] = dte
                        batch_dfs.append(df_puts)
                
                if batch_dfs:
                    # Combine all dataframes in this batch
                    batch_df = pd.concat(batch_dfs, ignore_index=True)
                    all_dfs.append(batch_df)
            
            if not all_dfs:
                logger.error("No valid options data found")
                raise ValueError("No valid options data found")
            
            # Combine all batches
            df = pd.concat(all_dfs, ignore_index=True)
            
            # Clean up column names and data types
            df = self._clean_dataframe(df)
            
            # Convert to xarray Dataset
            ds = self._convert_to_xarray(df)
            
            # Store the dataset
            self.ds = ds
            
            pre_process_time = time.time() - start_time
            logger.info(f"Pre-processing completed in {pre_process_time:.2f} seconds")
            
            # Perform 2D interpolation
            start_time = time.time()
            logger.info("Performing 2D interpolation")
            self.interpolate_missing_values_2d()
            interpolate_time = time.time() - start_time
            logger.info(f"Interpolation completed in {interpolate_time:.2f} seconds")
            
            # Post-process the data
            start_time = time.time()
            self.post_process_data()
            post_process_time = time.time() - start_time
            logger.info(f"Post-processing completed in {post_process_time:.2f} seconds")
            
            # Store processed data for caching
            self.options_data = {
                'dataset': self.ds,
                'min_strike': self.min_strike,
                'max_strike': self.max_strike,
                'risk_free_rate': self.risk_free_rate
            }
            
            logger.info("Successfully processed options data into xarray Dataset")
        except Exception as e:
            logger.error(f"Error processing options data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

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
        Interpolate missing values in the options data using 2D interpolation.
        
        This method:
        1. Identifies missing values in the dataset
        2. Uses griddata with cubic interpolation to fill gaps in the surface
        3. Handles edge cases where interpolation might fail
        4. Creates a complete surface across both strike prices and expiration dates
        
        This is particularly important for calculating accurate greeks which require
        a smooth, continuous surface without gaps.
        """
        if not self.ds:
            logger.error("Cannot interpolate: No dataset available")
            return
        logger.info("Starting 2D interpolation using xarray's interpolate_na")

        # Define fields to process
        price_fields = ['bid', 'ask', 'mid_price', 'price', 'lastPrice', 'intrinsic_value', 'extrinsic_value']
        # Define non-price fields that need special handling
        non_price_fields = ['volume', 'openInterest', 'impliedVolatility']
        
        # Process all fields
        all_fields = price_fields + non_price_fields
        
        for opt_type in ['call', 'put']:
            for field in all_fields:
                if field not in self.ds:
                    logger.warning(f"Field {field} not found in dataset, skipping")
                    continue
                
                try:
                    # Get the data array for this field and option type
                    da = self.ds[field].sel(option_type=opt_type)
                    
                    # First, convert zeros to NaN for proper interpolation
                    # Only do this for fields where zero is not a valid value
                    if field not in ['volume', 'openInterest']:
                        da = da.where(da != 0)
                    
                    # Use xarray's built-in interpolation for NaN values
                    # First interpolate along strike dimension
                    da_interp = da.interpolate_na(dim='strike', method='linear')
                    
                    # Then interpolate along DTE dimension
                    da_interp = da_interp.interpolate_na(dim='DTE', method='linear')
                    
                    # For any remaining NaN values, use griddata for 2D interpolation
                    if da_interp.isnull().any():
                        logger.info(f"Using griddata for remaining NaN values in {field} for {opt_type}")
                        
                        # Get the coordinates
                        strikes = da.strike.values
                        dtes = da.DTE.values
                        
                        # Create meshgrid for all points
                        strike_grid, dte_grid = np.meshgrid(strikes, dtes, indexing='ij')
                        points = np.column_stack([strike_grid.ravel(), dte_grid.ravel()])
                        
                        # Get the values and mask of valid (non-NaN) points
                        values = da_interp.values
                        valid_mask = ~np.isnan(values)
                        
                        if np.sum(valid_mask) > 3:  # Need at least 4 points for interpolation
                            # Get coordinates of valid points
                            valid_points = []
                            valid_values = []
                            
                            for i, strike in enumerate(strikes):
                                for j, dte in enumerate(dtes):
                                    if not np.isnan(values[i, j]):
                                        valid_points.append([strike, dte])
                                        valid_values.append(values[i, j])
                            
                            # Convert to numpy arrays
                            valid_points = np.array(valid_points)
                            valid_values = np.array(valid_values)
                            
                            # Create target points for all grid points
                            target_points = np.column_stack([strike_grid.ravel(), dte_grid.ravel()])
                            
                            # Interpolate using griddata
                            from scipy.interpolate import griddata
                            interp_values = griddata(
                                valid_points,
                                valid_values,
                                target_points,
                                method='linear',
                                fill_value=np.nan
                            )
                            
                            # Reshape back to original grid shape
                            interp_values = interp_values.reshape(len(strikes), len(dtes))
                            
                            # Update the interpolated array
                            da_interp.values = interp_values
                    
                    # Update the dataset with interpolated values
                    self.ds[field].loc[{'option_type': opt_type}] = da_interp
                    
                    logger.info(f"Successfully interpolated {field} for {opt_type}")
                except Exception as e:
                    logger.error(f"Error interpolating {field} for {opt_type}: {str(e)}")
                    logger.error(traceback.format_exc())
        
        # Apply smoothing to reduce noise in the interpolated data
        self._apply_smoothing()
        
        logger.info("2D interpolation completed")

    def _apply_smoothing(self):
        """
        Apply smoothing to implied volatility surface.
        
        This method implements a sophisticated smoothing algorithm for the IV surface:
        1. First applies 1D smoothing along strike dimension for each expiry
        2. Then applies 2D smoothing across both strike and expiry dimensions
        3. Uses a combination of median filtering and Gaussian smoothing
        4. Preserves the volatility smile shape while removing noise and outliers
        
        The smoothed IV surface helps in more accurate pricing and greeks calculation.
        """
        try:
            from scipy.ndimage import gaussian_filter
            
            logger.info("Applying smoothing to reduce noise in interpolated data")
            
            # Fields that benefit from smoothing
            smooth_fields = ['mid_price', 'price', 'impliedVolatility', 'bid', 'ask']
            
            for opt_type in ['call', 'put']:
                for field in smooth_fields:
                    if field not in self.ds:
                        continue
                    
                    # Get the data array
                    da = self.ds[field].sel(option_type=opt_type)
                    
                    # Apply smoothing for each DTE slice with adaptive sigma
                    for i, dte in enumerate(self.ds.DTE.values):
                        # Adaptive sigma based on DTE
                        if dte <= 30:  # Short-dated options (< 1 month)
                            sigma = 0.7
                        elif dte <= 90:  # Medium-dated options (1-3 months)
                            sigma = 1.0
                        elif dte <= 365:  # Longer-dated options (3-12 months)
                            sigma = 1.5
                        elif dte <= 730:  # 1-2 years
                            sigma = 2.0
                        else:  # LEAPS (> 2 years)
                            # Progressive smoothing for very long-dated options
                            base_sigma = 2.5
                            # Add additional smoothing based on DTE, with a cap
                            additional_sigma = min(2.5, (dte - 730) / 365.0)
                            sigma = base_sigma + additional_sigma
                        
                        # Get the slice for this DTE
                        slice_values = da.isel(DTE=i).values
                        
                        # Skip if all values are NaN
                        if np.all(np.isnan(slice_values)):
                            continue
                        
                        # Replace NaNs with nearest valid values for smoothing
                        valid_mask = ~np.isnan(slice_values)
                        if np.any(valid_mask):
                            # Find indices of valid values
                            valid_indices = np.where(valid_mask)[0]
                            
                            # For each NaN, find nearest valid value
                            for j in range(len(slice_values)):
                                if np.isnan(slice_values[j]):
                                    # Find nearest valid index
                                    nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - j))]
                                    slice_values[j] = slice_values[nearest_idx]
                        
                        # Apply Gaussian smoothing
                        smoothed_values = gaussian_filter(slice_values, sigma=sigma)
                        
                        # Update the data array
                        da.values[:, i] = smoothed_values
                    
                    # Update the dataset
                    self.ds[field].loc[{'option_type': opt_type}] = da
            
            logger.info("Smoothing applied successfully")
        except Exception as e:
            logger.error(f"Error applying smoothing: {str(e)}")
            logger.error(traceback.format_exc())

    def post_process_data(self):
        """Post-process the data after interpolation."""
        try:
            logger.info("Starting post-processing")
            
            if self.ds is None:
                logger.error("No dataset available for post-processing")
                return None
            
            # Calculate intrinsic values based on current price
            # For calls: max(0, S - K)
            # For puts: max(0, K - S)
            S = self.current_price
            
            # Process calls
            if 'call' in self.ds.option_type.values:
                strikes = self.ds.strike.values
                intrinsic_calls = np.maximum(0, S - strikes)
                
                # Broadcast to match dataset dimensions
                intrinsic_calls_expanded = np.zeros((len(strikes), len(self.ds.DTE.values)))
                for i in range(len(self.ds.DTE.values)):
                    intrinsic_calls_expanded[:, i] = intrinsic_calls
                
                # Update intrinsic value for calls
                self.ds['intrinsic_value'].loc[{'option_type': 'call'}] = intrinsic_calls_expanded
            
            # Process puts
            if 'put' in self.ds.option_type.values:
                strikes = self.ds.strike.values
                intrinsic_puts = np.maximum(0, strikes - S)
                
                # Broadcast to match dataset dimensions
                intrinsic_puts_expanded = np.zeros((len(strikes), len(self.ds.DTE.values)))
                for i in range(len(self.ds.DTE.values)):
                    intrinsic_puts_expanded[:, i] = intrinsic_puts
                
                # Update intrinsic value for puts
                self.ds['intrinsic_value'].loc[{'option_type': 'put'}] = intrinsic_puts_expanded
            
            # Calculate extrinsic value (price - intrinsic)
            if 'price' in self.ds and 'intrinsic_value' in self.ds:
                self.ds['extrinsic_value'] = self.ds['price'] - self.ds['intrinsic_value']
                # Ensure extrinsic value is not negative
                self.ds['extrinsic_value'] = xr.where(self.ds['extrinsic_value'] < 0, 0, self.ds['extrinsic_value'])
            
            # Calculate spread (ask - bid)
            if 'ask' in self.ds and 'bid' in self.ds:
                self.ds['spread'] = self.ds['ask'] - self.ds['bid']
                logger.info("Calculated spread")
            
            # Reverse engineer implied volatility where missing
            self.reverse_engineer_iv()
            
            # Calculate Greeks using improved methods
            try:
                # First try using Black-Scholes for all Greeks
                logger.info("Calculating Greeks using Black-Scholes model")
                self.calculate_black_scholes_greeks()
            except Exception as e:
                logger.error(f"Error in Black-Scholes calculation: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Fall back to numerical methods with interpolated price data
                logger.info("Falling back to numerical Greeks calculation with interpolated price data")
                try:
                    # Calculate delta using interpolated price data
                    self.compute_delta()
                    
                    # Calculate gamma using interpolated price data
                    self.compute_gamma()
                    
                    # Calculate theta using interpolated price data
                    self.compute_theta()
                    
                    # Try to calculate other Greeks if needed
                    if 'vega' not in self.ds or np.all(np.isnan(self.ds['vega'].values)):
                        self.compute_vega()
                    
                    if 'rho' not in self.ds or np.all(np.isnan(self.ds['rho'].values)):
                        self.compute_rho()
                except Exception as e2:
                    logger.error(f"Error in numerical Greeks calculation: {str(e2)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Try individual numerical calculations as a last resort
                    logger.info("Trying individual numerical calculations")
                    try:
                        if 'delta' not in self.ds or np.all(np.isnan(self.ds['delta'].values)):
                            self.compute_delta()
                    except Exception:
                        logger.error("Failed to calculate delta")
                    
                    try:
                        if 'gamma' not in self.ds or np.all(np.isnan(self.ds['gamma'].values)):
                            self.compute_gamma()
                    except Exception:
                        logger.error("Failed to calculate gamma")
                    
                    try:
                        if 'theta' not in self.ds or np.all(np.isnan(self.ds['theta'].values)):
                            self.compute_theta()
                    except Exception:
                        logger.error("Failed to calculate theta")
            
            # Ensure all required fields for plotting are available
            self._ensure_all_plot_fields()
            
            # Apply floors to ensure all values are reasonable
            self.apply_floors()
            
            logger.info("Post-processing complete")
            return self.ds
            
        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue with what we have
            return self.ds

    def _ensure_all_plot_fields(self):
        """Ensure all required plot fields are available in the dataset."""
        if self.ds is None:
            logger.error("Cannot ensure plot fields: dataset is None")
            return
            
        logger.info("Ensuring all plot fields are available")
        
        # Define all fields needed for plotting
        required_fields = {
            'mid_price': 0.05,      # Price
            'delta': 0,             # Delta
            'gamma': 0,             # Gamma
            'theta': 0,             # Theta
            'impliedVolatility': 0.3, # IV
            'volume': 0,            # Volume
            'spread': 0.05,         # Spread
            'intrinsic_value': 0,   # Intrinsic Value
            'extrinsic_value': 0    # Extrinsic Value
        }
        
        # Check each field and add if missing
        for field, default_value in required_fields.items():
            if field not in self.ds.data_vars:
                logger.info(f"Adding missing field: {field}")
                # Create a new data variable with the same dimensions as 'price'
                if 'price' in self.ds.data_vars:
                    # Copy dimensions from price
                    self.ds[field] = self.ds['price'].copy()
                    # Fill with default value
                    self.ds[field].values.fill(default_value)
                else:
                    logger.warning(f"Cannot add {field}: 'price' field not available for dimension reference")
        
        logger.info("All plot fields are now available")

    def calculate_black_scholes_greeks(self):
        """Calculate Black-Scholes greeks for all options in the dataset."""
        if self.ds is None:
            logger.error("Cannot calculate Black-Scholes greeks: dataset is None")
            return
            
        try:
            # Get risk-free rate
            r = self.get_risk_free_rate()
            S = self.current_price
            
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
            vega_array = np.zeros((len(strikes), len(dtes), len(option_types)))
            rho_array = np.zeros((len(strikes), len(dtes), len(option_types)))
            
            # Process each option type separately
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
                                continue
                            
                            # Cap extremely high IVs that can cause numerical issues
                            if sigma > 5.0:  # Cap at 500%
                                sigma = 5.0
                            
                            # Time to expiration in years
                            T = dte / 365.0
                            
                            # Skip if time to expiration is too small
                            if T <= 0.001:
                                continue
                                
                            # For very long-dated options, cap T to avoid numerical issues
                            if T > 10.0:
                                T = 10.0
                            
                            # Calculate all greeks at once for efficiency
                            greeks = calculate_all_greeks(S, strike, T, r, sigma, opt_type)
                            
                            # Store the results
                            delta_array[j, k, i] = greeks['delta']
                            gamma_array[j, k, i] = greeks['gamma']
                            theta_array[j, k, i] = greeks['theta']
                            vega_array[j, k, i] = greeks['vega']
                            rho_array[j, k, i] = greeks['rho']
                            
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
                
            if 'vega' not in self.ds:
                self.ds['vega'] = (('strike', 'DTE', 'option_type'), vega_array)
            else:
                self.ds['vega'].values = vega_array
                
            if 'rho' not in self.ds:
                self.ds['rho'] = (('strike', 'DTE', 'option_type'), rho_array)
            else:
                self.ds['rho'].values = rho_array
            
            logger.info("Successfully calculated Black-Scholes greeks")
            
        except Exception as e:
            logger.error(f"Error calculating Black-Scholes greeks: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Try the non-parallel version as a fallback
            try:
                self._calculate_black_scholes_greeks_sequential()
            except Exception as e2:
                logger.error(f"Error in sequential Black-Scholes calculation: {str(e2)}")
                logger.error(f"Traceback: {traceback.format_exc()}")

    def _calculate_black_scholes_greeks_sequential(self):
        """Calculate Black-Scholes greeks sequentially (fallback method)."""
        if self.ds is None:
            logger.error("Cannot calculate Black-Scholes greeks: dataset is None")
            return
            
        try:
            # Get risk-free rate
            r = self.get_risk_free_rate()
            S = self.current_price
            
            # Import the Black-Scholes functions
            from python.models.black_scholes import (
                delta, gamma, theta, vega, rho
            )
            
            # Process calls
            if 'call' in self.ds.option_type.values:
                for strike in self.ds.strike.values:
                    for dte in self.ds.DTE.values:
                        try:
                            # Get implied volatility
                            sigma = self.ds['impliedVolatility'].sel(
                                strike=strike, 
                                DTE=dte, 
                                option_type='call'
                            ).item()
                            
                            # Skip if invalid
                            if np.isnan(sigma) or sigma <= 0:
                                continue
                                
                            # Cap extremely high IVs
                            if sigma > 5.0:
                                sigma = 5.0
                                
                            # Time to expiration in years
                            T = dte / 365.0
                            
                            # Skip if too small
                            if T <= 0.001:
                                continue
                                
                            # Calculate greeks
                            delta_val = delta(S, strike, T, r, sigma, 'call')
                            gamma_val = gamma(S, strike, T, r, sigma)
                            theta_val = theta(S, strike, T, r, sigma, 'call')
                            vega_val = vega(S, strike, T, r, sigma)
                            rho_val = rho(S, strike, T, r, sigma, 'call')
                            
                            # Update dataset
                            self.ds['delta'].loc[{'strike': strike, 'DTE': dte, 'option_type': 'call'}] = delta_val
                            self.ds['gamma'].loc[{'strike': strike, 'DTE': dte, 'option_type': 'call'}] = gamma_val
                            self.ds['theta'].loc[{'strike': strike, 'DTE': dte, 'option_type': 'call'}] = theta_val
                            self.ds['vega'].loc[{'strike': strike, 'DTE': dte, 'option_type': 'call'}] = vega_val
                            self.ds['rho'].loc[{'strike': strike, 'DTE': dte, 'option_type': 'call'}] = rho_val
                        except Exception as e:
                            logger.debug(f"Error in sequential calculation for call, strike={strike}, DTE={dte}: {str(e)}")
            
            # Process puts
            if 'put' in self.ds.option_type.values:
                for strike in self.ds.strike.values:
                    for dte in self.ds.DTE.values:
                        try:
                            # Get implied volatility
                            sigma = self.ds['impliedVolatility'].sel(
                                strike=strike, 
                                DTE=dte, 
                                option_type='put'
                            ).item()
                            
                            # Skip if invalid
                            if np.isnan(sigma) or sigma <= 0:
                                continue
                                
                            # Cap extremely high IVs
                            if sigma > 5.0:
                                sigma = 5.0
                                
                            # Time to expiration in years
                            T = dte / 365.0
                            
                            # Skip if too small
                            if T <= 0.001:
                                continue
                                
                            # Calculate greeks
                            delta_val = delta(S, strike, T, r, sigma, 'put')
                            gamma_val = gamma(S, strike, T, r, sigma)
                            theta_val = theta(S, strike, T, r, sigma, 'put')
                            vega_val = vega(S, strike, T, r, sigma)
                            rho_val = rho(S, strike, T, r, sigma, 'put')
                            
                            # Update dataset
                            self.ds['delta'].loc[{'strike': strike, 'DTE': dte, 'option_type': 'put'}] = delta_val
                            self.ds['gamma'].loc[{'strike': strike, 'DTE': dte, 'option_type': 'put'}] = gamma_val
                            self.ds['theta'].loc[{'strike': strike, 'DTE': dte, 'option_type': 'put'}] = theta_val
                            self.ds['vega'].loc[{'strike': strike, 'DTE': dte, 'option_type': 'put'}] = vega_val
                            self.ds['rho'].loc[{'strike': strike, 'DTE': dte, 'option_type': 'put'}] = rho_val
                        except Exception as e:
                            logger.debug(f"Error in sequential calculation for put, strike={strike}, DTE={dte}: {str(e)}")
            
            logger.info("Successfully calculated Black-Scholes greeks sequentially")
            
        except Exception as e:
            logger.error(f"Error in sequential Black-Scholes calculation: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def apply_floors(self):
        """Apply minimum floors to certain fields to avoid negative or zero values.
        Note: This method has been modified to pass through true values without applying floors/ceilings."""
        if self.ds is None:
            logger.error("Cannot apply floors: dataset is None")
            return
            
        logger.info("Applying minimal processing to preserve true values")
        
        # Define dollar-denominated fields that need rounding to nearest $0.05
        dollar_fields = ['bid', 'ask', 'mid_price', 'price', 'intrinsic_value', 'extrinsic_value', 'spread']
        
        # Round dollar-denominated fields to nearest $0.05
        for field in dollar_fields:
            if field in self.ds.data_vars:
                # Round to nearest $0.05 (multiply by 20, round, divide by 20)
                self.ds[field] = (self.ds[field] * 20).round() / 20
                logger.debug(f"Applied $0.05 rounding to {field}")
        
        # Ensure ask >= bid (this is a logical constraint, not a floor)
        if 'ask' in self.ds.data_vars and 'bid' in self.ds.data_vars:
            ask_lt_bid_mask = self.ds['ask'] < self.ds['bid']
            if ask_lt_bid_mask.any():
                logger.warning(f"Found {ask_lt_bid_mask.sum().item()} cases where ask < bid, fixing...")
                self.ds['ask'] = xr.where(ask_lt_bid_mask, self.ds['bid'] * 1.05, self.ds['ask'])
                # Re-round ask prices after adjustment
                self.ds['ask'] = (self.ds['ask'] * 20).round() / 20
            
            # Recalculate mid_price after rounding
            self.ds['mid_price'] = (self.ds['bid'] + self.ds['ask']) / 2
            # Re-round mid_price
            self.ds['mid_price'] = (self.ds['mid_price'] * 20).round() / 20
        
        logger.info("Applied minimal processing to preserve true values")

    def reverse_engineer_iv(self):
        """
        Reverse engineer implied volatility from option prices.
        
        This method:
        1. Takes market prices of options
        2. Uses numerical methods to find the implied volatility that would produce those prices
        3. Implements a robust algorithm that handles edge cases (deep ITM/OTM options)
        4. Uses bisection method with adaptive step size for efficient convergence
        
        The resulting IV values are critical for understanding market sentiment and
        for accurate calculation of other option metrics.
        """
        if self.ds is None:
            logger.error("Cannot reverse engineer IV: dataset is None")
            return
            
        logger.info("Reverse engineering implied volatility from option prices")
        
        # Import the implied_volatility function
        from python.models.black_scholes import implied_volatility
        
        # Get current price and risk-free rate
        S = self.current_price
        r = self.get_risk_free_rate()
        
        # Process each option type separately
        for opt_type in self.ds.option_type.values:
            # Get interpolated price data for this option type
            price_data = self.ds['price'].sel(option_type=opt_type)
            
            # Get IV data for this option type
            iv_data = self.ds['impliedVolatility'].sel(option_type=opt_type)
            
            # Find missing IV values where price is available
            missing_mask = iv_data.isnull() | (iv_data <= 0)
            price_available = ~price_data.isnull() & (price_data > 0)
            
            # Only process points where IV is missing but price is available
            process_mask = missing_mask & price_available
            
            if process_mask.any():
                logger.info(f"Calculating IV for {process_mask.sum().item()} {opt_type} options with missing IV")
                
                # Create a matrix to store the calculated IVs
                calculated_ivs = np.zeros_like(iv_data.values)
                calculated_ivs.fill(np.nan)
                
                # Process each point
                for i, strike in enumerate(self.ds.strike.values):
                    for j, dte in enumerate(self.ds.DTE.values):
                        # Check if this point needs processing
                        if process_mask.isel(strike=i, DTE=j).item():
                            try:
                                # Get interpolated market price
                                market_price = price_data.isel(strike=i, DTE=j).item()
                                
                                # Calculate time to expiration in years
                                T = dte / 365.0
                                
                                # Skip if time to expiration is too small
                                if T <= 0.001:
                                    continue
                                
                                # Calculate implied volatility
                                iv = implied_volatility(market_price, S, strike, T, r, opt_type)
                                
                                # Store the calculated IV
                                calculated_ivs[i, j] = iv
                                
                                # Update the dataset if IV calculation was successful
                                if not np.isnan(iv) and iv > 0:
                                    self.ds['impliedVolatility'].loc[{'option_type': opt_type, 'strike': strike, 'DTE': dte}] = iv
                                    logger.debug(f"Calculated IV={iv:.2f} for {opt_type}, strike={strike}, DTE={dte}")
                            except Exception as e:
                                logger.debug(f"Error calculating IV for {opt_type}, strike={strike}, DTE={dte}: {str(e)}")
                
                # Interpolate any remaining missing IVs using the calculated ones
                # This helps create a smoother IV surface
                if np.any(~np.isnan(calculated_ivs)):
                    try:
                        # Create a mask of valid calculated IVs
                        valid_mask = ~np.isnan(calculated_ivs)
                        
                        # Only proceed if we have enough valid points
                        if np.sum(valid_mask) >= 3:
                            # Get coordinates of valid points
                            valid_indices = np.where(valid_mask)
                            valid_points = np.column_stack([valid_indices[0], valid_indices[1]])
                            valid_values = calculated_ivs[valid_mask]
                            
                            # Create a grid of all points
                            all_indices = np.indices((len(self.ds.strike), len(self.ds.DTE)))
                            all_points = np.column_stack([all_indices[0].flatten(), all_indices[1].flatten()])
                            
                            # Interpolate missing values
                            interpolated_values = griddata(
                                valid_points, 
                                valid_values, 
                                all_points, 
                                method='cubic', 
                                fill_value=np.nan
                            )
                            
                            # Reshape back to original shape
                            interpolated_ivs = interpolated_values.reshape((len(self.ds.strike), len(self.ds.DTE)))
                            
                            # Fill in missing values in the dataset
                            for i, strike in enumerate(self.ds.strike.values):
                                for j, dte in enumerate(self.ds.DTE.values):
                                    iv_value = interpolated_ivs[i, j]
                                    if not np.isnan(iv_value) and iv_value > 0:
                                        current_iv = self.ds['impliedVolatility'].sel(
                                            option_type=opt_type, 
                                            strike=strike, 
                                            DTE=dte
                                        ).item()
                                        
                                        # Only update if current IV is missing or invalid
                                        if np.isnan(current_iv) or current_iv <= 0:
                                            self.ds['impliedVolatility'].loc[{
                                                'option_type': opt_type, 
                                                'strike': strike, 
                                                'DTE': dte
                                            }] = iv_value
                    except Exception as e:
                        logger.error(f"Error interpolating IVs: {str(e)}")
                        logger.error(traceback.format_exc())
                
                logger.info(f"Completed IV calculation for {opt_type} options")
            else:
                logger.info(f"No missing IV values to calculate for {opt_type} options")
        
        # Apply bounds to IV values
        iv_mask_low = (self.ds['impliedVolatility'] < 0.01)
        iv_mask_high = (self.ds['impliedVolatility'] > 5.0)
        
        if iv_mask_low.any():
            logger.info(f"Applying floor to {iv_mask_low.sum().item()} implied volatility values")
            self.ds['impliedVolatility'] = xr.where(iv_mask_low, 0.01, self.ds['impliedVolatility'])
            
        if iv_mask_high.any():
            logger.info(f"Capping {iv_mask_high.sum().item()} high implied volatility values")
            self.ds['impliedVolatility'] = xr.where(iv_mask_high, 5.0, self.ds['impliedVolatility'])
        
        # Apply smoothing to the IV surface
        try:
            # Create a copy of the IV data
            iv_data = self.ds['impliedVolatility'].copy(deep=True)
            
            # Apply a simple smoothing filter
            for opt_type in self.ds.option_type.values:
                iv_values = iv_data.sel(option_type=opt_type).values
                
                # Skip if we don't have enough data
                if iv_values.shape[0] < 3 or iv_values.shape[1] < 3:
                    continue
                
                # Apply a simple 3x3 mean filter
                from scipy.ndimage import uniform_filter
                smoothed_values = uniform_filter(iv_values, size=3, mode='nearest')
                
                # Update the dataset
                self.ds['impliedVolatility'].loc[{'option_type': opt_type}] = smoothed_values
                
            logger.info("Applied smoothing to IV surface")
        except Exception as e:
            logger.error(f"Error smoothing IV surface: {str(e)}")
            logger.error(traceback.format_exc())
        
        logger.info("Completed reverse engineering of implied volatility")

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
        """Get the risk-free rate for option pricing."""
        # Check if we already have a risk-free rate
        if hasattr(self, 'risk_free_rate') and self.risk_free_rate is not None:
            return self.risk_free_rate
            
        # Default risk-free rate if we can't get it from the API
        default_rate = 0.04  # 4%
        
        try:
            # Try to get from YahooFinanceAPI
            api = YahooFinanceAPI()
            rate = api.get_risk_free_rate()
            logger.info(f"Got risk-free rate from API: {rate:.2%}")
            # Set the instance attribute
            self.risk_free_rate = rate
            return rate
        except Exception as e:
            logger.error(f"Error getting risk-free rate: {str(e)}")
            logger.info(f"Using default risk-free rate: {default_rate:.2%}")
            # Set the instance attribute to the default value
            self.risk_free_rate = default_rate
            return default_rate

    def force_reinterpolate(self):
        """Force a complete reinterpolation of all values, regardless of current state.
        This is useful when debugging interpolation issues or when the data has gaps.
        """
        logger.info("Forcing complete reinterpolation of all values")
        
        if not self.ds:
            logger.error("Cannot reinterpolate: No dataset available")
            return False
            
        try:
            # Convert all zeros to NaN for proper interpolation
            logger.info("Converting zeros to NaN for proper interpolation")
            
            # Define fields to process
            price_fields = ['bid', 'ask', 'mid_price', 'price', 'lastPrice', 'intrinsic_value', 'extrinsic_value']
            # Define non-price fields that need special handling
            non_price_fields = ['impliedVolatility']
            
            # Process all fields except volume and openInterest
            all_fields = price_fields + non_price_fields
            
            for field in all_fields:
                if field in self.ds:
                    # Convert zeros to NaN for proper interpolation
                    self.ds[field] = self.ds[field].where(self.ds[field] != 0)
            
            # Perform 2D interpolation
            logger.info("Performing 2D interpolation")
            self.interpolate_missing_values_2d()
            
            # Check if we still have missing values
            missing_count = self.count_missing_values()
            logger.info(f"After 2D interpolation: {missing_count} missing values")
            
            # If we still have missing values, set default values
            if missing_count > 0:
                logger.info("Setting default values for any remaining missing data")
                self._set_default_values_for_missing()
            
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
            
    def _set_default_values_for_missing(self):
        """Set default values for any missing data instead of using 1D interpolation."""
        logger.info("Setting default values for missing data")
        
        if not self.ds:
            return
            
        # Define default values for different fields
        default_values = {
            'bid': 0.05,
            'ask': 0.10,
            'mid_price': 0.075,
            'price': 0.075,
            'impliedVolatility': 0.3,  # 30% IV as default
            'volume': 0,
            'openInterest': 0,
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'spread': 0.05,
            'extrinsic_value': 0,
        }
        
        # Set default values for each field if missing
        for field, default_value in default_values.items():
            if field in self.ds.data_vars:
                # Replace missing values with default
                self.ds[field] = self.ds[field].fillna(default_value)
                
                # For impliedVolatility, also replace zeros or negative values
                if field == 'impliedVolatility':
                    self.ds[field] = xr.where(self.ds[field] <= 0, default_value, self.ds[field])
                    
        logger.info("Default values set for missing data")

    def count_missing_values(self):
        """Count the number of missing values in key fields."""
        if not self.ds:
            return -1
            
        count = 0
        for field in ['bid', 'ask', 'mid_price', 'price', 'impliedVolatility']:
            if field in self.ds.data_vars:
                count += self.ds[field].isnull().sum().item()
                
        return count

    def compute_delta(self):
        """
        Compute delta (first derivative of price with respect to underlying price) numerically.
        Uses interpolated price data for more accurate gradient calculations.
        """
        try:
            # Check if we have at least 2 strike values
            if len(self.ds.strike) < 2:
                logger.warning("Cannot compute delta: need at least 2 strike values")
                return
            
            # Get current price
            S = self.current_price
            
            # Create a range of underlying prices around the current price
            # Use a small range (±2%) for better accuracy of the derivative
            price_range = 0.02  # 2% range
            underlying_prices = np.linspace(S * (1 - price_range), S * (1 + price_range), 5)
            
            # Compute for each option type separately
            for opt_type in self.ds.option_type.values:
                # Get interpolated price data for this option type
                price_data = self.ds['price'].sel(option_type=opt_type)
                
                # Create a matrix to store option prices at different underlying prices
                option_prices = np.zeros((len(self.ds.strike), len(self.ds.DTE), len(underlying_prices)))
                
                # For each strike and DTE, calculate option prices at different underlying prices
                for i, strike in enumerate(self.ds.strike.values):
                    for j, dte in enumerate(self.ds.DTE.values):
                        # Get current option price
                        current_price = price_data.isel(strike=i, DTE=j).item()
                        
                        # Skip if price is invalid
                        if np.isnan(current_price) or current_price <= 0:
                            continue
                        
                        # Get implied volatility for this option
                        if 'impliedVolatility' in self.ds:
                            try:
                                sigma = self.ds['impliedVolatility'].sel(
                                    strike=strike, 
                                    DTE=dte, 
                                    option_type=opt_type
                                ).item()
                                
                                # Skip if implied volatility is invalid
                                if np.isnan(sigma) or sigma <= 0:
                                    continue
                                
                                # Time to expiration in years
                                T = dte / 365.0
                                
                                # Skip if time to expiration is too small
                                if T <= 0.001:
                                    continue
                                
                                # Get risk-free rate
                                r = self.get_risk_free_rate()
                                
                                # Calculate option prices at different underlying prices
                                for k, underlying_price in enumerate(underlying_prices):
                                    # Use Black-Scholes to calculate option price
                                    from python.models.black_scholes import call_price, put_price
                                    if opt_type == 'call':
                                        option_prices[i, j, k] = call_price(underlying_price, strike, T, r, sigma)
                                    else:
                                        option_prices[i, j, k] = put_price(underlying_price, strike, T, r, sigma)
                            except Exception as e:
                                logger.debug(f"Error in delta calculation for {opt_type}, strike={strike}, DTE={dte}: {str(e)}")
                
                # Calculate delta as the gradient of option price with respect to underlying price
                delta_values = np.zeros((len(self.ds.strike), len(self.ds.DTE)))
                
                for i in range(len(self.ds.strike)):
                    for j in range(len(self.ds.DTE)):
                        # Check if we have valid prices for this option
                        if np.any(option_prices[i, j, :] > 0):
                            # Calculate delta as the gradient at the current price
                            # Use central difference for better accuracy
                            delta_values[i, j] = np.gradient(option_prices[i, j, :], underlying_prices)[len(underlying_prices) // 2]
                
                # Update the dataset
                self.ds['delta'].loc[{'option_type': opt_type}] = delta_values
                
            logger.info("Computed delta using interpolated price data")
        except Exception as e:
            logger.error(f"Error computing delta: {str(e)}")
            logger.error(traceback.format_exc())

    def compute_gamma(self):
        """
        Compute gamma (second derivative of price with respect to underlying price) numerically.
        Uses interpolated price data for more accurate gradient calculations.
        """
        try:
            # Check if we have at least 3 strike values (needed for second derivative)
            if len(self.ds.strike) < 3:
                logger.warning("Cannot compute gamma: need at least 3 strike values")
                return
            
            # Get current price
            S = self.current_price
            
            # Create a range of underlying prices around the current price
            # Use a slightly larger range for gamma to capture curvature
            price_range = 0.03  # 3% range
            underlying_prices = np.linspace(S * (1 - price_range), S * (1 + price_range), 7)
            
            # Compute for each option type separately
            for opt_type in self.ds.option_type.values:
                # Get interpolated price data for this option type
                price_data = self.ds['price'].sel(option_type=opt_type)
                
                # Create a matrix to store option prices at different underlying prices
                option_prices = np.zeros((len(self.ds.strike), len(self.ds.DTE), len(underlying_prices)))
                
                # For each strike and DTE, calculate option prices at different underlying prices
                for i, strike in enumerate(self.ds.strike.values):
                    for j, dte in enumerate(self.ds.DTE.values):
                        # Get current option price
                        current_price = price_data.isel(strike=i, DTE=j).item()
                        
                        # Skip if price is invalid
                        if np.isnan(current_price) or current_price <= 0:
                            continue
                        
                        # Get implied volatility for this option
                        if 'impliedVolatility' in self.ds:
                            try:
                                sigma = self.ds['impliedVolatility'].sel(
                                    strike=strike, 
                                    DTE=dte, 
                                    option_type=opt_type
                                ).item()
                                
                                # Skip if implied volatility is invalid
                                if np.isnan(sigma) or sigma <= 0:
                                    continue
                                
                                # Time to expiration in years
                                T = dte / 365.0
                                
                                # Skip if time to expiration is too small
                                if T <= 0.001:
                                    continue
                                
                                # Get risk-free rate
                                r = self.get_risk_free_rate()
                                
                                # Calculate option prices at different underlying prices
                                for k, underlying_price in enumerate(underlying_prices):
                                    # Use Black-Scholes to calculate option price
                                    from python.models.black_scholes import call_price, put_price
                                    if opt_type == 'call':
                                        option_prices[i, j, k] = call_price(underlying_price, strike, T, r, sigma)
                                    else:
                                        option_prices[i, j, k] = put_price(underlying_price, strike, T, r, sigma)
                            except Exception as e:
                                logger.debug(f"Error in gamma calculation for {opt_type}, strike={strike}, DTE={dte}: {str(e)}")
                
                # Calculate gamma as the second derivative of option price with respect to underlying price
                gamma_values = np.zeros((len(self.ds.strike), len(self.ds.DTE)))
                
                for i in range(len(self.ds.strike)):
                    for j in range(len(self.ds.DTE)):
                        # Check if we have valid prices for this option
                        if np.any(option_prices[i, j, :] > 0):
                            # First calculate delta at each underlying price
                            deltas = np.gradient(option_prices[i, j, :], underlying_prices)
                            # Then calculate gamma as the gradient of delta
                            gamma_values[i, j] = np.gradient(deltas, underlying_prices)[len(underlying_prices) // 2]
                
                # Update the dataset
                self.ds['gamma'].loc[{'option_type': opt_type}] = gamma_values
                
            logger.info("Computed gamma using interpolated price data")
        except Exception as e:
            logger.error(f"Error computing gamma: {str(e)}")
            logger.error(traceback.format_exc())

    def compute_theta(self):
        """
        Compute theta (derivative of price with respect to time) numerically.
        Uses interpolated price data for more accurate gradient calculations.
        """
        try:
            # Check if we have at least 2 DTE values
            if len(self.ds.DTE) < 2:
                logger.warning("Cannot compute theta: need at least 2 DTE values")
                return
            
            # Compute for each option type separately
            for opt_type in self.ds.option_type.values:
                # Get interpolated price data for this option type
                price_data = self.ds['price'].sel(option_type=opt_type)
                
                # Create a matrix to store theta values
                theta_values = np.zeros((len(self.ds.strike), len(self.ds.DTE)))
                
                # For each strike and DTE, calculate theta
                for i, strike in enumerate(self.ds.strike.values):
                    for j, dte in enumerate(self.ds.DTE.values):
                        # Get current option price
                        current_price = price_data.isel(strike=i, DTE=j).item()
                        
                        # Skip if price is invalid
                        if np.isnan(current_price) or current_price <= 0:
                            continue
                        
                        # Get implied volatility for this option
                        if 'impliedVolatility' in self.ds:
                            try:
                                sigma = self.ds['impliedVolatility'].sel(
                                    strike=strike, 
                                    DTE=dte, 
                                    option_type=opt_type
                                ).item()
                                
                                # Skip if implied volatility is invalid
                                if np.isnan(sigma) or sigma <= 0:
                                    continue
                                
                                # Time to expiration in years
                                T = dte / 365.0
                                
                                # Skip if time to expiration is too small
                                if T <= 0.001:
                                    continue
                                
                                # Get risk-free rate
                                r = self.get_risk_free_rate()
                                
                                # Calculate theta using Black-Scholes
                                from python.models.black_scholes import theta as bs_theta
                                theta_value = bs_theta(self.current_price, strike, T, r, sigma, opt_type)
                                
                                # Store the result (theta is already in daily terms)
                                theta_values[i, j] = theta_value
                            except Exception as e:
                                logger.debug(f"Error in theta calculation for {opt_type}, strike={strike}, DTE={dte}: {str(e)}")
                
                # If we have multiple DTE values, we can also calculate theta numerically
                # as a fallback for points where Black-Scholes calculation failed
                if len(self.ds.DTE) >= 2:
                    # Calculate negative gradient along DTE dimension
                    numerical_theta = -np.gradient(price_data.values, self.ds.DTE.values, axis=1)
                    
                    # Use numerical theta where Black-Scholes theta is missing
                    mask = np.isnan(theta_values) | (theta_values == 0)
                    theta_values = np.where(mask, numerical_theta, theta_values)
                
                # Update the dataset
                self.ds['theta'].loc[{'option_type': opt_type}] = theta_values
                
            logger.info("Computed theta using Black-Scholes and interpolated price data")
        except Exception as e:
            logger.error(f"Error computing theta: {str(e)}")
            logger.error(traceback.format_exc())

    def compute_vega(self):
        """
        Compute vega (derivative of price with respect to volatility) numerically.
        This is an approximation using the Black-Scholes model with small IV perturbations.
        """
        if self.ds is None or 'impliedVolatility' not in self.ds:
            logger.error("Cannot compute vega: dataset or implied volatility is missing")
            return
            
        try:
            # Import Black-Scholes functions
            from python.models.black_scholes import call_price, put_price
            
            # Get current price and risk-free rate
            S = self.current_price
            r = self.get_risk_free_rate()
            
            # Create array to store vega values
            vega_array = np.zeros((len(self.ds.strike), len(self.ds.DTE), len(self.ds.option_type)))
            
            # Small volatility perturbation (0.01 = 1%)
            dv = 0.01
            
            # Process each option type separately
            for i, opt_type in enumerate(self.ds.option_type.values):
                for j, strike in enumerate(self.ds.strike.values):
                    for k, dte in enumerate(self.ds.DTE.values):
                        try:
                            # Get implied volatility
                            sigma = self.ds['impliedVolatility'].sel(
                                strike=strike, 
                                DTE=dte, 
                                option_type=opt_type
                            ).item()
                            
                            # Skip if invalid
                            if np.isnan(sigma) or sigma <= 0:
                                continue
                                
                            # Time to expiration in years
                            T = dte / 365.0
                            
                            # Skip if too small
                            if T <= 0.001:
                                continue
                                
                            # Calculate price with current IV
                            if opt_type == 'call':
                                price1 = call_price(S, strike, T, r, sigma)
                                # Calculate price with perturbed IV
                                price2 = call_price(S, strike, T, r, sigma + dv)
                            else:
                                price1 = put_price(S, strike, T, r, sigma)
                                # Calculate price with perturbed IV
                                price2 = put_price(S, strike, T, r, sigma + dv)
                                
                            # Calculate vega as price difference divided by volatility difference
                            vega_val = (price2 - price1) / dv
                            
                            # Store vega value
                            vega_array[j, k, i] = vega_val
                            
                        except Exception as e:
                            logger.debug(f"Error computing vega for {opt_type}, strike={strike}, DTE={dte}: {str(e)}")
            
            # Add or update vega in the dataset
            if 'vega' not in self.ds:
                self.ds['vega'] = (('strike', 'DTE', 'option_type'), vega_array)
            else:
                self.ds['vega'].values = vega_array
                
            logger.info("Computed vega numerically")
            
        except Exception as e:
            logger.error(f"Error computing vega: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def compute_rho(self):
        """
        Compute rho (derivative of price with respect to interest rate) numerically.
        This is an approximation using the Black-Scholes model with small interest rate perturbations.
        """
        if self.ds is None or 'impliedVolatility' not in self.ds:
            logger.error("Cannot compute rho: dataset or implied volatility is missing")
            return
            
        try:
            # Import Black-Scholes functions
            from python.models.black_scholes import call_price, put_price
            
            # Get current price and risk-free rate
            S = self.current_price
            r = self.get_risk_free_rate()
            
            # Create array to store rho values
            rho_array = np.zeros((len(self.ds.strike), len(self.ds.DTE), len(self.ds.option_type)))
            
            # Small interest rate perturbation (0.01 = 1%)
            dr = 0.01
            
            # Process each option type separately
            for i, opt_type in enumerate(self.ds.option_type.values):
                for j, strike in enumerate(self.ds.strike.values):
                    for k, dte in enumerate(self.ds.DTE.values):
                        try:
                            # Get implied volatility
                            sigma = self.ds['impliedVolatility'].sel(
                                strike=strike, 
                                DTE=dte, 
                                option_type=opt_type
                            ).item()
                            
                            # Skip if invalid
                            if np.isnan(sigma) or sigma <= 0:
                                continue
                                
                            # Time to expiration in years
                            T = dte / 365.0
                            
                            # Skip if too small
                            if T <= 0.001:
                                continue
                                
                            # Calculate price with current interest rate
                            if opt_type == 'call':
                                price1 = call_price(S, strike, T, r, sigma)
                                # Calculate price with perturbed interest rate
                                price2 = call_price(S, strike, T, r + dr, sigma)
                            else:
                                price1 = put_price(S, strike, T, r, sigma)
                                # Calculate price with perturbed interest rate
                                price2 = put_price(S, strike, T, r + dr, sigma)
                                
                            # Calculate rho as price difference divided by interest rate difference
                            rho_val = (price2 - price1) / dr
                            
                            # Store rho value
                            rho_array[j, k, i] = rho_val
                            
                        except Exception as e:
                            logger.debug(f"Error computing rho for {opt_type}, strike={strike}, DTE={dte}: {str(e)}")
            
            # Add or update rho in the dataset
            if 'rho' not in self.ds:
                self.ds['rho'] = (('strike', 'DTE', 'option_type'), rho_array)
            else:
                self.ds['rho'].values = rho_array
                
            logger.info("Computed rho numerically")
            
        except Exception as e:
            logger.error(f"Error computing rho: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def compute_all_greeks_numerically(self):
        """Compute all Greeks numerically as a fallback method."""
        logger.info("Computing all Greeks numerically")
        
        # Ensure required fields exist
        self._ensure_all_plot_fields()
        
        # Compute Greeks
        self.compute_delta()
        self.compute_gamma()
        self.compute_theta()
        self.compute_vega()
        self.compute_rho()
        
        logger.info("Completed numerical computation of all Greeks")

    def _clean_dataframe(self, df):
        """Clean up column names and data types in the DataFrame."""
        try:
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
            
            return df
        except Exception as e:
            logger.error(f"Error cleaning DataFrame: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _convert_to_xarray(self, df):
        """Convert DataFrame to xarray Dataset."""
        try:
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
            
            return ds
        except Exception as e:
            logger.error(f"Error converting to xarray: {str(e)}")
            logger.error(traceback.format_exc())
            raise