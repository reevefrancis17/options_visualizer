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
from typing import Dict, Optional, Callable, Tuple, List, Any

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
        
        # Run cache maintenance on startup
        self._cache.maintenance()

    def get_current_processor(self, ticker: str) -> Tuple[Optional['OptionsDataProcessor'], Optional[float], str, float]:
        """Get the current processor from cache with its status."""
        # Get data from SQLite cache
        options_data, current_price, status, progress, processed_dates, total_dates = self._cache.get(ticker)
        
        if status != 'not_found' and options_data and current_price:
            try:
                # Create processor from cached data
                processor = OptionsDataProcessor(options_data, current_price)
                return processor, current_price, status, progress
            except Exception as e:
                logger.error(f"Error creating processor from cached data for {ticker}: {str(e)}")
        
        # If we get here, either cache miss or error creating processor
        return None, None, 'not_found', 0

    def start_fetching(self, ticker: str) -> bool:
        """Start fetching data in the background if not already loading.
        
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
                
            self._loading_state[ticker] = {'last_processed_dates': 0, 'total_dates': 0}
            thread = threading.Thread(target=self._fetch_in_background, args=(ticker,))
            thread.daemon = True  # Make thread exit when main thread exits
            thread.start()
            logger.info(f"Started background fetch for {ticker}")
            return True

    def _fetch_in_background(self, ticker: str):
        """Fetch data in a background thread."""
        try:
            def cache_update_callback(partial_data, current_price, processed_dates, total_dates):
                # Update the cache even with minimal data (just one date)
                if partial_data and current_price:
                    try:
                        self._loading_state[ticker]['last_processed_dates'] = processed_dates
                        self._loading_state[ticker]['total_dates'] = total_dates
                        
                        # Update SQLite cache with partial data
                        self._cache.set(ticker, partial_data, current_price, processed_dates, total_dates)
                        
                        logger.info(f"Updated cache with partial data for {ticker} ({processed_dates}/{total_dates} dates)")
                    except Exception as e:
                        logger.error(f"Error in background cache update: {str(e)}")

            options_data, current_price = self.api.get_options_data(ticker, progress_callback=cache_update_callback)
            if options_data and current_price:
                try:
                    # Update SQLite cache with complete data
                    processed_dates = len(options_data)
                    total_dates = processed_dates  # All dates are processed
                    self._cache.set(ticker, options_data, current_price, processed_dates, total_dates)
                    
                    logger.info(f"Completed fetching and cached data for {ticker}")
                except Exception as e:
                    logger.error(f"Error processing complete data: {str(e)}")
            else:
                logger.error(f"Failed to fetch data for {ticker}")
        except Exception as e:
            logger.error(f"Error in background fetch for {ticker}: {str(e)}")
        finally:
            # Clean up loading state when done
            if ticker in self._loading_state:
                del self._loading_state[ticker]

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
    def __init__(self, options_data, current_price):
        logger.info(f"Initializing OptionsDataProcessor with current_price: {current_price}")
        self.options_data = options_data
        self.current_price = current_price
        self.min_strike = None
        self.max_strike = None
        if not options_data:
            logger.error("Failed to fetch options.")
            raise ValueError("Options data is None")
        self.ds = self.pre_process_data()
        num_dates = len(self.get_expirations())
        if num_dates == 1:
            self.interpolate_missing_values_1d()
        elif num_dates >= 2:
            self.interpolate_missing_values_2d()
        self.post_process_data()
        self.risk_free_rate = self.get_risk_free_rate()

    def pre_process_data(self):
        if not self.options_data or not self.current_price:
            logger.error("Missing required data for processing")
            return None
        logger.info(f"Processing options data with current price: {self.current_price}")
        dfs = []
        now = pd.Timestamp.now().normalize()
        for exp, data in self.options_data.items():
            exp_date = pd.to_datetime(exp).normalize()
            dte = max(0, (exp_date - now).days)
            for opt_type, df in [('call', data.get('calls', pd.DataFrame())), ('put', data.get('puts', pd.DataFrame()))]:
                if not df.empty:
                    df = df.copy()
                    df['option_type'] = opt_type
                    df['expiration'] = exp_date
                    df['DTE'] = dte
                    dfs.append(df)
        if not dfs:
            logger.error("No valid data to process")
            return None
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
        
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['price'] = df['mid_price']
        mask_call = df['option_type'] == 'call'
        df.loc[mask_call, 'intrinsic_value'] = np.maximum(0, self.current_price - df.loc[mask_call, 'strike'])
        df.loc[~mask_call, 'intrinsic_value'] = np.maximum(0, df.loc[~mask_call, 'strike'] - self.current_price)
        df['extrinsic_value'] = df['price'] - df['intrinsic_value']
        self.min_strike = df['strike'].min()
        self.max_strike = df['strike'].max()
        strikes = sorted(df['strike'].unique())
        dtes = sorted(df['DTE'].unique())
        option_types = ['call', 'put']
        numeric_cols = [col for col in df.columns if col not in ['strike', 'DTE', 'option_type', 'expiration', 'contractSymbol', 'lastTradeDate', 'contractSize', 'currency'] and pd.api.types.is_numeric_dtype(df[col])]
        string_cols = ['contractSymbol', 'lastTradeDate', 'contractSize', 'currency']
        data_vars = {col: (['strike', 'DTE', 'option_type'], np.full((len(strikes), len(dtes), len(option_types)), np.nan if col in numeric_cols else None, dtype=float if col in numeric_cols else object)) for col in numeric_cols + string_cols}
        ds = xr.Dataset(data_vars=data_vars, coords={'strike': strikes, 'DTE': dtes, 'option_type': option_types})
        ds.coords['expiration'] = ('DTE', np.array([df[df['DTE'] == dte]['expiration'].iloc[0] for dte in dtes], dtype='datetime64[ns]'))
        for _, row in df.iterrows():
            for col in numeric_cols:
                ds[col].loc[{'strike': row['strike'], 'DTE': row['DTE'], 'option_type': row['option_type']}] = row[col]
            for col in string_cols:
                ds[col].loc[{'strike': row['strike'], 'DTE': row['DTE'], 'option_type': row['option_type']}] = str(row[col])
        logger.info("Successfully processed options data into xarray Dataset")
        return ds

    def interpolate_missing_values_1d(self):
        if not self.ds:
            logger.error("Cannot interpolate: No dataset available")
            return
        logger.info("Starting 1D interpolation of missing values")
        numeric_vars = [var for var in self.ds.data_vars if np.issubdtype(self.ds[var].dtype, np.number)]
        dte = self.ds.DTE.values[0]
        for opt_type in ['call', 'put']:
            for variable in numeric_vars:
                if variable in ['volume', 'openInterest']:
                    continue
                da = self.ds[variable].sel(option_type=opt_type, DTE=dte)
                if da.isnull().any():
                    values = da.values
                    strikes = da.strike.values
                    valid = ~np.isnan(values)
                    if np.sum(valid) >= 2:
                        valid_indices = np.where(valid)[0]
                        start_idx, end_idx = valid_indices[0], valid_indices[-1]
                        s = pd.Series(values[start_idx:end_idx + 1], index=strikes[start_idx:end_idx + 1])
                        s_interp = s.interpolate(method='linear')
                        values[start_idx:end_idx + 1] = s_interp.values
                        self.ds[variable].loc[{'option_type': opt_type, 'DTE': dte}] = values
                        logger.info(f"1D interpolated {variable} for {opt_type} at DTE={dte}")
                    else:
                        logger.warning(f"Not enough valid points for 1D interpolation of {variable} for {opt_type}")

    def interpolate_missing_values_2d(self):
        if not self.ds:
            logger.error("Cannot interpolate: No dataset available")
            return
        logger.info("Starting 2D interpolation of missing values")
        numeric_vars = [var for var in self.ds.data_vars if np.issubdtype(self.ds[var].dtype, np.number)]
        dtes = self.ds.DTE.values
        if len(dtes) < 2:
            logger.info(f"Only {len(dtes)} DTE value(s) found, skipping 2D interpolation")
            return
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
                            interpolated_values = griddata(points[non_nan], values_flat[non_nan], (strike_grid, dte_grid), method='linear')
                            da.values = interpolated_values
                            logger.info(f"2D interpolated {variable} for {opt_type} using linear method")
                        except Exception as e:
                            logger.warning(f"2D linear interpolation failed for {variable} {opt_type}: {str(e)}")
                            for dte in dtes:
                                slice_1d = da.sel(DTE=dte).values
                                valid = ~np.isnan(slice_1d)
                                if np.sum(valid) >= 2:
                                    valid_indices = np.where(valid)[0]
                                    start_idx, end_idx = valid_indices[0], valid_indices[-1]
                                    s = pd.Series(slice_1d[start_idx:end_idx + 1], index=strikes[start_idx:end_idx + 1])
                                    s_interp = s.interpolate(method='linear')
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
        self.apply_floors()
        self.ds['spread'] = self.ds['ask'] - self.ds['bid']
        self.compute_delta()
        self.compute_gamma()
        self.compute_theta()

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

    def compute_delta(self):
        delta = np.gradient(self.ds['price'].values, self.ds.strike.values, axis=0)
        self.ds['delta'] = (('strike', 'DTE', 'option_type'), delta)

    def compute_gamma(self):
        gamma = np.gradient(self.ds['delta'].values, self.ds.strike.values, axis=0)
        self.ds['gamma'] = (('strike', 'DTE', 'option_type'), gamma)

    def compute_theta(self):
        theta = -np.gradient(self.ds['price'].values, self.ds.DTE.values, axis=1)
        self.ds['theta'] = (('strike', 'DTE', 'option_type'), theta)