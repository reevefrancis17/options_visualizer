# options_data.py
import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from datetime import datetime
import logging
import traceback
import sys
import time
import os
from typing import Dict, Any, Optional, Callable, Tuple, List, Union

# Import the finance API and models
from python.yahoo_finance import YahooFinanceAPI
from python.models.black_scholes import (
    call_price, put_price, delta, gamma, theta, vega, rho, implied_volatility
)

# Set up logger
logger = logging.getLogger(__name__)

# Clear error logs if they exist
log_dir = os.path.join(os.path.dirname(__file__), 'debug')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
error_log = os.path.join(log_dir, 'error_log.txt')
if os.path.exists(error_log):
    with open(error_log, 'w') as f:
        f.write(f"=== New session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

class OptionsDataManager:
    """
    Central manager for options data handling.
    Interfaces with data sources (Yahoo Finance, etc.) and financial models (Black-Scholes, etc.).
    Provides a unified API for applications to access options data.
    """
    # Available data sources
    DATA_SOURCE_YAHOO = "yahoo"
    # Available pricing models
    MODEL_BLACK_SCHOLES = "black_scholes"
    MODEL_MARKET = "market"  # Use market prices directly
    
    def __init__(self, data_source=DATA_SOURCE_YAHOO, pricing_model=MODEL_MARKET, cache_duration=600):
        """
        Initialize the options data manager.
        
        Args:
            data_source: The data source to use (default: yahoo)
            pricing_model: The pricing model to use (default: market)
            cache_duration: Cache duration in seconds (default: 600)
        """
        logger.info(f"Initializing OptionsDataManager with source={data_source}, model={pricing_model}")
        self.data_source = data_source
        self.pricing_model = pricing_model
        self.cache_duration = cache_duration
        
        # Initialize data source APIs
        if data_source == self.DATA_SOURCE_YAHOO:
            self.api = YahooFinanceAPI(cache_duration=cache_duration)
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        # Cache for processed data
        self._processor_cache = {}
    
    def get_options_data(self, ticker: str, 
                        progress_callback: Optional[Callable[[Dict, float, int, int], None]] = None,
                        max_dates: Optional[int] = None) -> Tuple['OptionsDataProcessor', float]:
        """
        Get options data for a ticker.
        
        Args:
            ticker: The stock ticker symbol
            progress_callback: Optional callback for progress updates
            max_dates: Maximum number of expiration dates to fetch
            
        Returns:
            Tuple of (OptionsDataProcessor, current_price)
        """
        # Check processor cache first
        if ticker in self._processor_cache:
            processor, price, timestamp = self._processor_cache[ticker]
            if time.time() - timestamp < self.cache_duration:
                logger.info(f"Using cached processor for {ticker}")
                # Call the callback with complete data if provided
                if progress_callback:
                    progress_callback(processor.options_data, price, 
                                     len(processor.get_expirations()), 
                                     len(processor.get_expirations()))
                return processor, price
        
        # Fetch raw data from the selected data source
        logger.info(f"Fetching options data for {ticker} from {self.data_source}")
        
        # Create a wrapper for the progress callback to update our cache incrementally
        def cache_update_callback(partial_data, current_price, processed_dates, total_dates):
            # Only process if we have some data
            if partial_data and current_price and processed_dates > 0:
                try:
                    # Create a temporary processor with partial data
                    temp_processor = OptionsDataProcessor(partial_data, current_price)
                    # Update the cache with this partial data
                    self._processor_cache[ticker] = (temp_processor, current_price, time.time())
                except Exception as e:
                    logger.error(f"Error creating temporary processor: {str(e)}")
            
            # Forward the callback to the original progress_callback if provided
            if progress_callback:
                progress_callback(partial_data, current_price, processed_dates, total_dates)
        
        # Fetch the data
        options_data, current_price = self.api.get_options_data(
            ticker, 
            progress_callback=cache_update_callback if progress_callback else None,
            max_dates=max_dates
        )
        
        if options_data is None or current_price is None:
            logger.error(f"Failed to fetch data for {ticker}")
            return None, None
        
        # Create the processor with the fetched data
        try:
            processor = OptionsDataProcessor(options_data, current_price)
            # Cache the processor
            self._processor_cache[ticker] = (processor, current_price, time.time())
            return processor, current_price
        except Exception as e:
            logger.error(f"Error creating options data processor: {str(e)}")
            return None, None
    
    def get_risk_free_rate(self):
        """Get the risk-free rate."""
        api = YahooFinanceAPI()
        return api.get_risk_free_rate("^TNX")
    
    def calculate_option_price(self, S: float, K: float, T: float, r: float, 
                              sigma: float, option_type: str) -> float:
        """
        Calculate option price using the selected pricing model.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        if self.pricing_model == self.MODEL_BLACK_SCHOLES:
            if option_type == 'call':
                return call_price(S, K, T, r, sigma)
            elif option_type == 'put':
                return put_price(S, K, T, r, sigma)
            else:
                raise ValueError(f"Unsupported option type: {option_type}")
        else:
            raise ValueError(f"Unsupported pricing model for calculation: {self.pricing_model}")
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str) -> Dict[str, float]:
        """
        Calculate option Greeks using the selected pricing model.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary of Greeks (delta, gamma, theta, vega, rho)
        """
        if self.pricing_model == self.MODEL_BLACK_SCHOLES:
            return {
                'delta': delta(S, K, T, r, sigma, option_type),
                'gamma': gamma(S, K, T, r, sigma),
                'theta': theta(S, K, T, r, sigma, option_type),
                'vega': vega(S, K, T, r, sigma),
                'rho': rho(S, K, T, r, sigma, option_type)
            }
        else:
            raise ValueError(f"Unsupported pricing model for Greeks: {self.pricing_model}")
    
    def calculate_implied_volatility(self, market_price: float, S: float, K: float, 
                                   T: float, r: float, option_type: str) -> float:
        """
        Calculate implied volatility from market price.
        
        Args:
            market_price: Market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            option_type: 'call' or 'put'
            
        Returns:
            Implied volatility
        """
        return implied_volatility(market_price, S, K, T, r, option_type)


class OptionsDataProcessor:
    """
    Processes raw options data from data sources.
    Calculates additional metrics and stores data in an xarray Dataset for easy slicing and analysis.
    """
    def __init__(self, options_data, current_price):
        """
        Initializes the processor with raw options data and current stock price.
        """
        logger.info(f"Initializing OptionsDataProcessor with current_price: {current_price}")
        self.options_data = options_data
        self.current_price = current_price
        self.min_strike = None
        self.max_strike = None
        if self.options_data is None:
            logger.error("Failed to fetch options.")
            raise ValueError("Options data is None")
        
        try:
            self.ds = self.pre_process_data()
            
            # Try to interpolate, but continue even if it fails
            try:
                self.interpolate_missing_values_2d()
            except Exception as e:
                logger.error(f"Interpolation failed but continuing: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            self.post_process_data()
            self.risk_free_rate = self.get_risk_free_rate()
        except Exception as e:
            logger.error(f"Error in OptionsDataProcessor initialization: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def pre_process_data(self):
        """
        Processes raw options data into an xarray Dataset with calculated metrics.
        Core dimensions: strike, DTE (days to expiry), option_type (call/put)
        """
        try:
            if not self.options_data or not self.current_price:
                logger.error("Missing required data for processing")
                return None

            logger.info(f"Processing options data with current price: {self.current_price}")
            
            # First, process data into a pandas DataFrame as before
            dfs = []
            now = pd.Timestamp.now().normalize()  # Normalize to start of day to fix DTE calculation
            
            # Process each expiration date
            for exp, data in self.options_data.items():
                logger.info(f"Processing data for expiration date: {exp}")
                exp_date = pd.to_datetime(exp).normalize()  # Normalize to start of day
                # Calculate DTE, ensuring today's options show 0 DTE
                dte = max(0, (exp_date - now).days)
                
                if 'calls' in data and not data['calls'].empty:
                    # Reduce logging verbosity for better performance
                    calls_count = len(data['calls'])
                    if calls_count > 0:
                        logger.info(f"Processing {calls_count} call options for {exp}")
                        calls = data['calls'].copy()
                        calls['option_type'] = 'call'
                        calls['expiration'] = exp_date
                        calls['DTE'] = dte
                        dfs.append(calls)
                
                if 'puts' in data and not data['puts'].empty:
                    # Reduce logging verbosity for better performance
                    puts_count = len(data['puts'])
                    if puts_count > 0:
                        logger.info(f"Processing {puts_count} put options for {exp}")
                        puts = data['puts'].copy()
                        puts['option_type'] = 'put'
                        puts['expiration'] = exp_date
                        puts['DTE'] = dte
                        dfs.append(puts)

            if not dfs:
                logger.error("No valid data to process")
                return None

            # Combine data
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined data shape: {df.shape}")
            
            # Calculate additional metrics
            logger.info("Calculating additional metrics")
            
            # Calculate spot price (mean of bid and ask)
            df['spot'] = (df['bid'] + df['ask']) / 2
            
            # Use spot as the default price
            df['price'] = df['spot']
            
            # Calculate intrinsic value - vectorized for better performance
            mask_call = df['option_type'] == 'call'
            df.loc[mask_call, 'intrinsic_value'] = np.maximum(0, self.current_price - df.loc[mask_call, 'strike'])
            df.loc[~mask_call, 'intrinsic_value'] = np.maximum(0, df.loc[~mask_call, 'strike'] - self.current_price)
            
            # Calculate extrinsic value as the residue
            df['extrinsic_value'] = df['price'] - df['intrinsic_value']
            
            # Calculate min and max strike prices across all expirations
            self.min_strike = df['strike'].min()
            self.max_strike = df['strike'].max()
            
            # Log some statistics
            logger.info(f"Price range: {df['price'].min():.2f} to {df['price'].max():.2f}")
            logger.info(f"Strike range: {self.min_strike:.2f} to {self.max_strike:.2f}")
            logger.info(f"DTE range: {df['DTE'].min()} to {df['DTE'].max()}")
            
            # Convert to xarray Dataset
            logger.info("Converting to xarray Dataset")
            
            # Get unique values for each dimension
            strikes = sorted(df['strike'].unique())
            dtes = sorted(df['DTE'].unique())
            option_types = ['call', 'put']
            
            # Identify numeric and string columns
            numeric_cols = []
            string_cols = []
            
            # Log column types for debugging
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            logger.info(f"DataFrame dtypes: {df.dtypes}")
            
            # Explicitly check for known string columns
            known_string_cols = ['contractSymbol', 'lastTradeDate', 'contractSize', 'currency']
            
            for col in df.columns:
                if col not in ['strike', 'DTE', 'option_type', 'expiration']:
                    # Check if it's a known string column
                    if col in known_string_cols:
                        string_cols.append(col)
                        logger.info(f"Added known string column: {col}")
                    # Check if the column contains string data
                    elif df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col].dtype):
                        string_cols.append(col)
                        logger.info(f"Detected string column: {col}")
                    else:
                        numeric_cols.append(col)
                        logger.info(f"Detected numeric column: {col}")
            
            logger.info(f"Numeric columns: {numeric_cols}")
            logger.info(f"String columns: {string_cols}")
            
            # Create empty arrays for each variable with the right dimensions
            data_vars = {}
            
            # Handle numeric columns
            for col in numeric_cols:
                data_vars[col] = (
                    ['strike', 'DTE', 'option_type'], 
                    np.full((len(strikes), len(dtes), len(option_types)), np.nan)
                )
            
            # Handle string columns - use object arrays filled with None
            for col in string_cols:
                data_vars[col] = (
                    ['strike', 'DTE', 'option_type'], 
                    np.full((len(strikes), len(dtes), len(option_types)), None, dtype=object)
                )
            
            # Create the dataset with coordinates
            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    'strike': strikes,
                    'DTE': dtes,
                    'option_type': option_types,
                }
            )
            
            # Add expiration date as a coordinate mapped to DTE
            expiry_dates = {dte: df[df['DTE'] == dte]['expiration'].iloc[0] for dte in dtes}
            expiry_dates_array = np.array([expiry_dates[dte] for dte in dtes], dtype='datetime64[ns]')
            ds.coords['expiration'] = ('DTE', expiry_dates_array)
            
            # Fill the dataset with values from the DataFrame
            for idx, row in df.iterrows():
                strike = row['strike']
                dte = row['DTE']
                opt_type = row['option_type']
                
                # For each column, set the value at the right position
                for col in numeric_cols:
                    try:
                        ds[col].loc[{'strike': strike, 'DTE': dte, 'option_type': opt_type}] = row[col]
                    except Exception as e:
                        logger.warning(f"Could not set numeric value for {col} at strike={strike}, DTE={dte}, option_type={opt_type}: {e}")
                        continue
                
                # Handle string columns separately
                for col in string_cols:
                    try:
                        ds[col].loc[{'strike': strike, 'DTE': dte, 'option_type': opt_type}] = str(row[col])
                    except Exception as e:
                        logger.warning(f"Could not set string value for {col} at strike={strike}, DTE={dte}, option_type={opt_type}: {e}")
                        continue
            
            logger.info("Successfully processed options data into xarray Dataset")
            return ds
            
        except Exception as e:
            logger.error(f"Error processing options data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def get_data(self):
        """Returns the processed options data as an xarray Dataset."""
        if self.ds is None:
            logger.error("No processed data available")
        return self.ds
    
    def get_data_frame(self):
        """
        Returns the processed options data as a pandas DataFrame for backward compatibility.
        """
        if self.ds is None:
            logger.error("No processed data available")
            return None
        
        try:
            # Convert xarray Dataset to DataFrame
            df = self.ds.to_dataframe().reset_index()
            return df
        except Exception as e:
            logger.error(f"Error converting xarray to DataFrame: {str(e)}")
            return None

    def get_nearest_expiry(self):
        """Returns the nearest expiration date."""
        if self.ds is not None:
            try:
                min_dte = self.ds.DTE.min().item()
                nearest = self.ds.expiration.sel(DTE=min_dte).item()
                logger.info(f"Found nearest expiry: {nearest}")
                return pd.Timestamp(nearest)
            except Exception as e:
                logger.error(f"Error getting nearest expiry: {str(e)}")
        
        logger.error("Cannot get nearest expiry - no data available")
        return None

    def get_expirations(self):
        """Returns sorted list of all expiration dates."""
        if self.ds is not None:
            try:
                expirations = sorted(self.ds.expiration.values)
                logger.info(f"Found {len(expirations)} expiration dates")
                return [pd.Timestamp(exp) for exp in expirations]
            except Exception as e:
                logger.error(f"Error getting expirations: {str(e)}")
        
        logger.error("Cannot get expirations - no data available")
        return []
        
    def get_strike_range(self):
        """Returns the min and max strike prices across all expirations."""
        return self.min_strike, self.max_strike
    
    def get_data_for_expiry(self, expiry_date):
        """
        Returns data for a specific expiration date as a pandas DataFrame.
        
        Args:
            expiry_date: A pandas Timestamp or datetime object
        
        Returns:
            A pandas DataFrame with data for the specified expiration date
        """
        if self.ds is None:
            logger.error("No processed data available")
            return None
        
        try:
            # Find the DTE value that corresponds to this expiration date
            expiry_date = pd.Timestamp(expiry_date)
            
            # Find the closest matching expiration date
            time_diffs = abs(self.ds.expiration.values - np.datetime64(expiry_date))
            closest_idx = np.argmin(time_diffs)
            closest_dte = self.ds.DTE.values[closest_idx]
            
            # Select data for this DTE
            subset = self.ds.sel(DTE=closest_dte)
            
            # Convert to DataFrame and reset index
            df = subset.to_dataframe().reset_index()
            
            # Drop rows with NaN values in key fields
            df = df.dropna(subset=['price'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting data for expiry {expiry_date}: {str(e)}")
            return None
        
    def interpolate_missing_values_2d(self):
        """
        Interpolate missing values in the dataset using 2D linear interpolation.
        This fills gaps in the options chain for better visualization.
        """
        if self.ds is None:
            logger.error("Cannot interpolate: No dataset available")
            return
            
        try:
            logger.info("Starting 2D interpolation of missing values")
            
            # Identify numeric variables
            numeric_vars = [var for var in self.ds.data_vars if np.issubdtype(self.ds[var].dtype, np.number)]
            logger.info(f"Interpolating numeric variables: {numeric_vars}")
            
            # Check if we have enough DTE values for 2D interpolation
            dtes = self.ds.DTE.values
            if len(dtes) <= 1:
                logger.info(f"Only {len(dtes)} DTE value(s) found, skipping 2D interpolation")
                return
            
            # Loop over option types and variables
            for opt_type in ['call', 'put']:
                for variable in numeric_vars:
                    # Extract 2D DataArray for this option type and variable
                    da = self.ds[variable].sel(option_type=opt_type)
                    
                    # Check if there are any NaNs to interpolate
                    if da.isnull().any():
                        strikes = da.strike.values
                        
                        # Create a 2D grid of coordinates
                        strike_grid, dte_grid = np.meshgrid(strikes, dtes, indexing='ij')
                        points = np.column_stack([strike_grid.ravel(), dte_grid.ravel()])
                        
                        # Flatten the data values
                        values_flat = da.values.ravel()
                        non_nan = ~np.isnan(values_flat)
                        
                        # If there are non-NaN values to base interpolation on
                        if non_nan.sum() > 0:
                            points_known = points[non_nan]  # Coordinates of known values
                            values_known = values_flat[non_nan]  # Known values
                            
                            # Check if we have enough points for interpolation
                            if len(points_known) < 3:
                                logger.warning(f"Not enough points to interpolate {variable} for {opt_type}")
                                continue
                            
                            # Check if all points have the same value in any dimension
                            # This would cause the Qhull error
                            dim_ranges = np.ptp(points_known, axis=0)
                            if np.any(dim_ranges == 0):
                                logger.warning(f"Dimension with zero range detected for {variable} {opt_type}, using nearest neighbor")
                                try:
                                    # Use nearest neighbor interpolation directly
                                    interpolated_values = griddata(
                                        points_known, values_known, (strike_grid, dte_grid), method='nearest'
                                    )
                                    # Update the DataArray with interpolated values
                                    da.values = interpolated_values
                                    logger.info(f"Interpolated {variable} for {opt_type} using nearest neighbor")
                                    continue
                                except Exception as e:
                                    logger.error(f"Nearest interpolation failed for {variable} {opt_type}: {str(e)}")
                                    # Skip this variable if nearest method fails
                                    continue
                                
                            try:
                                # Try standard linear interpolation
                                interpolated_values = griddata(
                                    points_known, values_known, (strike_grid, dte_grid), method='linear'
                                )
                                
                                # Fill any remaining NaNs with nearest neighbor
                                if np.isnan(interpolated_values).any():
                                    nan_mask = np.isnan(interpolated_values)
                                    nearest_values = griddata(
                                        points_known, values_known, (strike_grid, dte_grid), method='nearest'
                                    )
                                    interpolated_values[nan_mask] = nearest_values[nan_mask]
                                    
                                # Update the DataArray with interpolated values
                                da.values = interpolated_values
                                logger.info(f"Interpolated {variable} for {opt_type}")
                                
                            except Exception as e:
                                logger.warning(f"Linear interpolation failed for {variable} {opt_type}: {str(e)}")
                                try:
                                    # Fall back to nearest neighbor interpolation
                                    logger.info(f"Falling back to nearest neighbor interpolation for {variable} {opt_type}")
                                    interpolated_values = griddata(
                                        points_known, values_known, (strike_grid, dte_grid), method='nearest'
                                    )
                                    # Update the DataArray with interpolated values
                                    da.values = interpolated_values
                                    logger.info(f"Interpolated {variable} for {opt_type} using nearest neighbor")
                                except Exception as e2:
                                    logger.error(f"Nearest interpolation also failed for {variable} {opt_type}: {str(e2)}")
                                    # Skip this variable if both methods fail
                                    continue
                        else:
                            logger.warning(f"No data available for {variable} {opt_type}, cannot interpolate")
            
            logger.info("Completed 2D interpolation")
            
        except Exception as e:
            logger.error(f"Error during interpolation: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue with the rest of the processing even if interpolation fails
            logger.info("Continuing with processing despite interpolation errors")

    def apply_floors(self):
        """Apply floors to bid, ask, and extrinsic value, and recompute spot and price."""
        self.ds['bid'] = self.ds['bid'].clip(min=0.05)
        self.ds['ask'] = self.ds['ask'].clip(min=0.05)
        self.ds['spot'] = (self.ds['bid'] + self.ds['ask']) / 2
        self.ds['price'] = self.ds['spot']
        self.ds['extrinsic_value'] = self.ds['price'] - self.ds['intrinsic_value']
        self.ds['extrinsic_value'] = self.ds['extrinsic_value'].clip(min=0)

    def compute_delta(self):
        """Compute delta as the first derivative of price with respect to strike."""
        strikes = self.ds.strike.values
        price = self.ds['price'].values
        delta = np.gradient(price, strikes, axis=0)
        # Specify dimensions when assigning to dataset
        self.ds['delta'] = (('strike', 'DTE', 'option_type'), delta)

    def compute_gamma(self):
        """Compute gamma as the second derivative of price with respect to strike."""
        strikes = self.ds.strike.values
        delta = self.ds['delta'].values
        gamma = np.gradient(delta, strikes, axis=0)
        # Specify dimensions when assigning to dataset
        self.ds['gamma'] = (('strike', 'DTE', 'option_type'), gamma)

    def compute_theta(self):
        """Compute theta as the first derivative of price with respect to DTE."""
        dtes = self.ds.DTE.values
        price = self.ds['price'].values
        # Convention is negative theta for long options
        theta = -np.gradient(price, dtes, axis=1) 
        # Specify dimensions when assigning to dataset
        self.ds['theta'] = (('strike', 'DTE', 'option_type'), theta)
    
    def post_process_data(self):
        """Post-process data by applying floors, computing spread, delta, and gamma."""
        self.apply_floors()
        self.ds['spread'] = self.ds['ask'] - self.ds['bid']
        self.compute_delta()
        self.compute_gamma()
        self.compute_theta()

    def get_risk_free_rate(self):
        """Get the risk-free rate."""
        api = YahooFinanceAPI()
        return api.get_risk_free_rate("^TNX")



