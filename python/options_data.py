# options_data.py
import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from datetime import datetime
import logging
import traceback

# Set up logger - Use existing logger without adding a duplicate handler
logger = logging.getLogger(__name__)

class OptionsDataProcessor:
    """
    Processes raw options data from Yahoo Finance (or other sources in the future).
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
        self.ds = self._process_data()
        if self.ds is not None:
            self.interpolate_missing_values_2d()

    def _process_data(self):
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
            
            # Loop over option types and variables
            for opt_type in ['call', 'put']:
                for variable in numeric_vars:
                    # Extract 2D DataArray for this option type and variable
                    da = self.ds[variable].sel(option_type=opt_type)
                    
                    # Check if there are any NaNs to interpolate
                    if da.isnull().any():
                        strikes = da.strike.values
                        dtes = da.DTE.values
                        
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
                            
                            # Perform 2D interpolation over the entire grid
                            interpolated_values = griddata(
                                points_known, values_known, (strike_grid, dte_grid), method='linear'
                            )
                            
                            # Update the DataArray with interpolated values
                            da.values = interpolated_values
                            logger.info(f"Interpolated {variable} for {opt_type}")
                        else:
                            logger.warning(f"No data available for {variable} {opt_type}, cannot interpolate")
            
            logger.info("Completed 2D interpolation")
            
        except Exception as e:
            logger.error(f"Error during interpolation: {str(e)}")
            logger.error(traceback.format_exc())