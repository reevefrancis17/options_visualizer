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
        
        # Override the cache's trigger_refresh method to use our refresh logic
        self._cache._trigger_refresh = self._refresh_ticker
        
        # Run cache maintenance on startup
        self._cache.maintenance()
        
        # Start background thread for cache polling
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
        """Get the current processor from cache with its status."""
        # Get data from SQLite cache
        options_data, current_price, status, progress, processed_dates, total_dates = self._cache.get(ticker)
        
        if status != 'not_found' and options_data and current_price:
            try:
                # Log the type of data we're getting from cache
                logger.info(f"Retrieved data for {ticker} from cache: type={type(options_data).__name__}, status={status}")
                
                # Check if options_data has a 'dataset' key, which indicates it's already processed
                is_processed = isinstance(options_data, dict) and 'dataset' in options_data
                
                if is_processed:
                    logger.info(f"Using fully processed and interpolated data from cache for {ticker}")
                else:
                    logger.info(f"Using raw data from cache for {ticker}, will need processing")
                
                # Create processor from cached data
                processor = OptionsDataProcessor(options_data, current_price, is_processed=is_processed)
                
                # Verify the processor has valid data
                if processor.ds is None:
                    logger.error(f"Processor for {ticker} has no dataset, recreating cache")
                    # Clear the cache entry and start a new fetch
                    self._cache.delete(ticker)
                    self.start_fetching(ticker)
                    return None, None, 'not_found', 0
                
                # If status is 'stale', we're already refreshing in the background
                # but we'll return the stale data with a 'partial' status to indicate
                # that new data is being fetched
                if status == 'stale':
                    status = 'partial'
                
                # Verify we can get data from the processor
                try:
                    df = processor.get_data_frame()
                    if df is None or df.empty:
                        logger.error(f"Processor for {ticker} returned empty DataFrame, recreating cache")
                        # Clear the cache entry and start a new fetch
                        self._cache.delete(ticker)
                        self.start_fetching(ticker)
                        return None, None, 'not_found', 0
                except Exception as df_error:
                    logger.error(f"Error getting DataFrame from processor for {ticker}: {str(df_error)}")
                    # Clear the cache entry and start a new fetch
                    self._cache.delete(ticker)
                    self.start_fetching(ticker)
                    return None, None, 'not_found', 0
                
                return processor, current_price, status, progress
            except Exception as e:
                logger.error(f"Error creating processor from cached data for {ticker}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Clear the cache entry and start a new fetch
                self._cache.delete(ticker)
                self.start_fetching(ticker)
        
        # If we get here, either cache miss or error creating processor
        return None, None, 'not_found', 0

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
                    self._loading_state[ticker]['last_processed_dates'] = len(raw_data_collection)
                    self._loading_state[ticker]['total_dates'] = total_dates
                    
                    # Store raw data in cache for quick recovery in case of crash
                    try:
                        # Mark this as unprocessed data by not including a 'dataset' key
                        self._cache.set(ticker, raw_data_collection, current_price, len(raw_data_collection), total_dates)
                        logger.info(f"Cached raw data for {ticker} with {len(raw_data_collection)}/{total_dates} dates")
                    except Exception as e:
                        logger.error(f"Error caching raw data for {ticker}: {str(e)}")
            
            # Fetch data directly from the API with the callback
            logger.info(f"Fetching data for {ticker} in background thread")
            options_data_dict, current_price = self.api.get_options_data(ticker, cache_update_callback)
            
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
                    
                    # Final update to cache with fully processed data
                    self._cache.set(ticker, processed_data, final_current_price, processed_dates, total_dates)
                    logger.info(f"Cached fully processed data for {ticker} with {processed_dates}/{total_dates} dates (with interpolation)")
                except Exception as proc_error:
                    logger.error(f"Error processing complete data for {ticker}: {str(proc_error)}")
                    # Fall back to caching raw data
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
                                                    skip_interpolation=skip_interpolation)
                    processed_data = processor.options_data
                    
                    # Update cache with the processed data
                    self._cache.set(ticker, processed_data, current_price, processed_dates, total_dates)
                    logger.info(f"Cached processed data for {ticker} with {processed_dates} dates")
                except Exception as proc_error:
                    logger.error(f"Error processing API data for {ticker}: {str(proc_error)}")
                    # Fall back to caching raw data
                    self._cache.set(ticker, options_data_dict, current_price, processed_dates, total_dates)
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
                         max_dates: Optional[int] = None) -> Tuple[Optional['OptionsDataProcessor'], Optional[float]]:
        """Get options data with support for immediate cache return and background fetching.
        
        This method will:
        1. Return cached data immediately if available
        2. Start a background fetch if no cache is available
        3. Call progress_callback with the current state
        
        Returns:
            Tuple of (processor, price) - may be (None, None) if no cache and fetch just started
        """
        # Check cache first
        processor, price, status, progress = self.get_current_processor(ticker)
        
        # If we have cached data (partial or complete), return it immediately
        if status != 'not_found':
            if progress_callback and processor:
                # Call progress callback with current state
                expiry_count = len(processor.get_expirations())
                
                # Get total count from cache or loading state
                _, _, _, _, processed_dates, total_dates = self._cache.get(ticker)
                total_count = total_dates if total_dates > 0 else expiry_count
                
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
                    if num_dates == 1:
                        self.interpolate_missing_values_1d()
                    elif num_dates >= 2:
                        self.interpolate_missing_values_2d()
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
        if not self.options_data or not self.current_price:
            logger.error("Missing required data for processing")
            return None
        logger.info(f"Processing options data with current price: {self.current_price}")
        
        # Use a more memory-efficient approach
        dfs = []
        now = pd.Timestamp.now().normalize()
        
        # Process in batches to reduce memory usage
        batch_size = 5  # Process 5 expiration dates at a time
        exp_dates = list(self.options_data.keys())
        
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
                dfs.append(batch_df)
                
                # Clear memory
                del batch_dfs
        
        if not dfs:
            logger.error("No valid data to process")
            return None
        
        # Concatenate all batches
        df = pd.concat(dfs, ignore_index=True)
        
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

    def interpolate_missing_values_1d(self):
        """Interpolate missing values in 1D (single expiration date)."""
        if not self.ds:
            logger.error("Cannot interpolate: No dataset available")
            return
        logger.info("Starting 1D interpolation of missing values")
        
        # Only interpolate essential fields to save time
        essential_fields = ['mid_price', 'price', 'bid', 'ask', 'impliedVolatility', 'intrinsic_value', 'extrinsic_value']
        numeric_vars = [var for var in self.ds.data_vars if var in essential_fields and np.issubdtype(self.ds[var].dtype, np.number)]
        
        dte = self.ds.DTE.values[0]
        for opt_type in ['call', 'put']:
            for variable in numeric_vars:
                if variable in ['volume', 'openInterest']:
                    continue
                    
                # Use vectorized operations for better performance
                da = self.ds[variable].sel(option_type=opt_type, DTE=dte)
                if da.isnull().any():
                    values = da.values
                    strikes = da.strike.values
                    valid = ~np.isnan(values)
                    
                    if np.sum(valid) >= 2:
                        # Find the range of valid indices
                        valid_indices = np.where(valid)[0]
                        start_idx, end_idx = valid_indices[0], valid_indices[-1]
                        
                        # Create a Series for interpolation
                        s = pd.Series(values[start_idx:end_idx + 1], index=strikes[start_idx:end_idx + 1])
                        
                        # Use linear interpolation for speed
                        s_interp = s.interpolate(method='linear')
                        
                        # Update the values
                        values[start_idx:end_idx + 1] = s_interp.values
                        self.ds[variable].loc[{'option_type': opt_type, 'DTE': dte}] = values
                        logger.info(f"1D interpolated {variable} for {opt_type} at DTE={dte}")
                    else:
                        logger.warning(f"Not enough valid points for 1D interpolation of {variable} for {opt_type}")

    def interpolate_missing_values_2d(self):
        """Interpolate missing values in 2D (multiple expiration dates)."""
        if not self.ds:
            logger.error("Cannot interpolate: No dataset available")
            return
        logger.info("Starting 2D interpolation of missing values")
        
        # Only interpolate essential fields to save time
        essential_fields = ['mid_price', 'price', 'bid', 'ask', 'impliedVolatility', 'intrinsic_value', 'extrinsic_value']
        numeric_vars = [var for var in self.ds.data_vars if var in essential_fields and np.issubdtype(self.ds[var].dtype, np.number)]
        
        dtes = self.ds.DTE.values
        if len(dtes) < 2:
            logger.info(f"Only {len(dtes)} DTE value(s) found, skipping 2D interpolation")
            return
            
        # Process in parallel for better performance
        for opt_type in ['call', 'put']:
            for variable in numeric_vars:
                if variable in ['volume', 'openInterest']:
                    continue
                    
                da = self.ds[variable].sel(option_type=opt_type)
                if da.isnull().any():
                    strikes = da.strike.values
                    strike_grid, dte_grid = np.meshgrid(strikes, dtes, indexing='ij')
                    points = np.column_stack([strike_grid.ravel(), dte_grid.ravel()])
                    values_flat = da.values.ravel()
                    non_nan = ~np.isnan(values_flat)
                    
                    if non_nan.sum() > 3:
                        try:
                            # Use linear interpolation for speed
                            interpolated_values = griddata(
                                points[non_nan], 
                                values_flat[non_nan], 
                                (strike_grid, dte_grid), 
                                method='linear'
                            )
                            da.values = interpolated_values
                            logger.info(f"2D interpolated {variable} for {opt_type} using linear method")
                        except Exception as e:
                            logger.warning(f"2D linear interpolation failed for {variable} {opt_type}: {str(e)}")
                            
                            # Fall back to 1D interpolation for each DTE
                            for dte in dtes:
                                slice_1d = da.sel(DTE=dte).values
                                valid = ~np.isnan(slice_1d)
                                
                                if np.sum(valid) >= 2:
                                    valid_indices = np.where(valid)[0]
                                    start_idx, end_idx = valid_indices[0], valid_indices[-1]
                                    
                                    # Create a Series for interpolation
                                    s = pd.Series(slice_1d[start_idx:end_idx + 1], index=strikes[start_idx:end_idx + 1])
                                    
                                    # Use linear interpolation for speed
                                    s_interp = s.interpolate(method='linear')
                                    
                                    # Update the values
                                    slice_1d[start_idx:end_idx + 1] = s_interp.values
                                    self.ds[variable].loc[{'option_type': opt_type, 'DTE': dte}] = slice_1d
                                    logger.info(f"1D fallback interpolated {variable} for {opt_type} at DTE={dte}")
                    else:
                        logger.warning(f"Not enough points ({non_nan.sum()}) for 2D interpolation of {variable} for {opt_type}")

    def apply_floors(self):
        bid_mask = (self.ds['bid'].isnull() | (self.ds['bid'] < 0.05))
        ask_mask = (self.ds['ask'].isnull() | (self.ds['ask'] < 0.05))
        self.ds['bid'] = xr.where(bid_mask, 0.05, self.ds['bid'])
        self.ds['ask'] = xr.where(ask_mask, 0.05, self.ds['ask'])
        self.ds['mid_price'] = (self.ds['bid'] + self.ds['ask']) / 2
        self.ds['price'] = self.ds['mid_price']
        self.ds['extrinsic_value'] = self.ds['price'] - self.ds['intrinsic_value']
        self.ds['extrinsic_value'] = xr.where(self.ds['extrinsic_value'] < 0, 0, self.ds['extrinsic_value'])

    def post_process_data(self):
        """Apply post-processing to the dataset."""
        try:
            self.apply_floors()
            self.ds['spread'] = self.ds['ask'] - self.ds['bid']
            
            # Only compute greeks if we have the necessary dimensions
            if 'DTE' in self.ds.dims and 'strike' in self.ds.dims and 'option_type' in self.ds.dims:
                try:
                    self.compute_delta()
                    self.compute_gamma()
                    self.compute_theta()
                except Exception as e:
                    logger.error(f"Error computing greeks: {str(e)}")
            else:
                logger.warning(f"Cannot compute greeks: missing dimensions. Available dims: {list(self.ds.dims)}")
        except Exception as e:
            logger.error(f"Error in post_process_data: {str(e)}")

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