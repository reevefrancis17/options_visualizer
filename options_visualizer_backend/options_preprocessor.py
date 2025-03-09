#!/usr/bin/env python3
"""
Options Preprocessor Module

This module provides functionality to fetch and preprocess options data:
1. Fetch raw options data using yahoo_finance.py
2. Calculate mid-prices
3. Organize data into a 2D grid with strike prices and days to expiration
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Tuple, List, Optional, Any, Union

from python.yahoo_finance import YahooFinanceAPI

# Set up logger
logger = logging.getLogger(__name__)

class OptionsPreprocessor:
    """
    Preprocesses options data for visualization and analysis.
    
    This class fetches raw options data and organizes it into a structured 2D grid
    with strike prices on the x-axis and days to expiration on the y-axis.
    """
    
    def __init__(self, cache_duration=600):
        """
        Initialize the options preprocessor.
        
        Args:
            cache_duration: Cache duration in seconds
        """
        self.api = YahooFinanceAPI(cache_duration=cache_duration)
        
    def fetch_data(self, ticker: str, max_dates: Optional[int] = None) -> Tuple[Dict, float]:
        """
        Fetch raw options data for a ticker.
        
        Args:
            ticker: The stock ticker symbol
            max_dates: Maximum number of expiration dates to fetch
            
        Returns:
            Tuple of (options_data, current_price)
        """
        logger.info(f"Fetching options data for {ticker}")
        options_data, current_price = self.api.get_options_data(ticker, max_dates=max_dates)
        
        if not options_data:
            logger.error(f"Failed to fetch options data for {ticker}")
            raise ValueError(f"No options data available for {ticker}")
            
        logger.info(f"Successfully fetched options data for {ticker} with {len(options_data)} expiration dates")
        return options_data, current_price
    
    def calculate_mid_price(self, bid: float, ask: float, last_price: float) -> float:
        """
        Calculate the mid-price for an option.
        
        Args:
            bid: Bid price
            ask: Ask price
            last_price: Last traded price
            
        Returns:
            Mid-price if bid and ask are valid, otherwise last_price
        """
        # Check if bid and ask are valid (non-zero and not NaN)
        if (bid is not None and ask is not None and 
            not np.isnan(bid) and not np.isnan(ask) and 
            bid > 0 and ask > 0):
            return (bid + ask) / 2
        
        # Fall back to last_price if bid or ask is invalid
        if last_price is not None and not np.isnan(last_price) and last_price > 0:
            return last_price
            
        # If all values are invalid, return NaN
        return np.nan
    
    def preprocess_data(self, options_data: Dict, current_price: float) -> Dict[str, pd.DataFrame]:
        """
        Preprocess options data into a structured format.
        
        Args:
            options_data: Raw options data from yahoo_finance.py
            current_price: Current stock price
            
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames containing preprocessed data
        """
        logger.info(f"Preprocessing options data with current price: {current_price}")
        
        # Get all expiration dates, filtering out metadata fields
        exp_dates = [exp for exp in options_data.keys() if not exp.startswith('_')]
        
        if not exp_dates:
            logger.error("No expiration dates found in options data")
            raise ValueError("No expiration dates found in options data")
        
        # Sort expiration dates
        exp_dates.sort()
        
        # Current date for DTE calculation
        now = pd.Timestamp.now().normalize()
        
        # Process calls and puts separately
        calls_data = []
        puts_data = []
        
        for exp in exp_dates:
            data = options_data[exp]
            exp_date = pd.to_datetime(exp).normalize()
            dte = max(0, (exp_date - now).days)
            
            # Process calls
            if 'calls' in data and data['calls']:
                calls = pd.DataFrame(data['calls']) if isinstance(data['calls'], list) else data['calls']
                if not calls.empty:
                    # Calculate mid-price for each option
                    calls['mid_price'] = calls.apply(
                        lambda row: self.calculate_mid_price(row.get('bid'), row.get('ask'), row.get('lastPrice')),
                        axis=1
                    )
                    calls['DTE'] = dte
                    calls['expiration'] = exp_date
                    calls_data.append(calls)
            
            # Process puts
            if 'puts' in data and data['puts']:
                puts = pd.DataFrame(data['puts']) if isinstance(data['puts'], list) else data['puts']
                if not puts.empty:
                    # Calculate mid-price for each option
                    puts['mid_price'] = puts.apply(
                        lambda row: self.calculate_mid_price(row.get('bid'), row.get('ask'), row.get('lastPrice')),
                        axis=1
                    )
                    puts['DTE'] = dte
                    puts['expiration'] = exp_date
                    puts_data.append(puts)
        
        # Combine all data
        calls_df = pd.concat(calls_data, ignore_index=True) if calls_data else pd.DataFrame()
        puts_df = pd.concat(puts_data, ignore_index=True) if puts_data else pd.DataFrame()
        
        # Convert numeric columns
        numeric_cols = ['strike', 'lastPrice', 'bid', 'ask', 'mid_price', 'impliedVolatility', 'volume', 'openInterest']
        for df in [calls_df, puts_df]:
            if not df.empty:
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Preprocessed data: {len(calls_df)} calls and {len(puts_df)} puts")
        return {'calls': calls_df, 'puts': puts_df}
    
    def organize_grid_data(self, preprocessed_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Organize preprocessed data into a 2D grid format.
        
        Args:
            preprocessed_data: Dictionary with 'calls' and 'puts' DataFrames
            
        Returns:
            Dictionary with grid data for calls and puts
        """
        result = {}
        
        for option_type in ['calls', 'puts']:
            df = preprocessed_data[option_type]
            
            if df.empty:
                result[option_type] = {
                    'strikes': [],
                    'dtes': [],
                    'grid': np.array([]),
                    'expiration_dates': {}
                }
                continue
            
            # Get unique strikes and DTEs
            strikes = sorted(df['strike'].unique())
            dtes = sorted(df['DTE'].unique())
            
            # Create mapping of DTE to expiration date
            dte_to_exp = {}
            for dte in dtes:
                dte_df = df[df['DTE'] == dte]
                if not dte_df.empty:
                    # Convert NumPy int64 to Python int for JSON serialization
                    dte_key = int(dte) if isinstance(dte, np.integer) else dte
                    dte_to_exp[dte_key] = dte_df['expiration'].iloc[0].strftime('%Y-%m-%d')
            
            # Create empty grid filled with NaN
            grid = np.full((len(strikes), len(dtes)), np.nan)
            
            # Fill grid with mid-prices
            for i, strike in enumerate(strikes):
                for j, dte in enumerate(dtes):
                    # Find options with this strike and DTE
                    options = df[(df['strike'] == strike) & (df['DTE'] == dte)]
                    if not options.empty:
                        # Use mid_price if available
                        grid[i, j] = options['mid_price'].iloc[0]
            
            # Convert NumPy arrays to Python lists for JSON serialization
            result[option_type] = {
                'strikes': [float(strike) for strike in strikes],
                'dtes': [int(dte) for dte in dtes],
                'grid': grid,
                'expiration_dates': dte_to_exp
            }
        
        return result
    
    def fetch_and_preprocess(self, ticker: str, max_dates: Optional[int] = None) -> Dict:
        """
        Fetch and preprocess options data for a ticker.
        
        Args:
            ticker: The stock ticker symbol
            max_dates: Maximum number of expiration dates to fetch
            
        Returns:
            Dictionary with preprocessed grid data
        """
        # Fetch raw data
        options_data, current_price = self.fetch_data(ticker, max_dates)
        
        # Preprocess data
        preprocessed_data = self.preprocess_data(options_data, current_price)
        
        # Organize into grid format
        grid_data = self.organize_grid_data(preprocessed_data)
        
        # Add metadata
        result = {
            'ticker': ticker,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'calls': grid_data['calls'],
            'puts': grid_data['puts']
        }
        
        return result


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create preprocessor
    preprocessor = OptionsPreprocessor()
    
    # Fetch and preprocess data for a sample ticker
    ticker = "AAPL"
    result = preprocessor.fetch_and_preprocess(ticker)
    
    # Print summary
    print(f"Ticker: {result['ticker']}")
    print(f"Current Price: {result['current_price']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Calls: {len(result['calls']['strikes'])} strikes × {len(result['calls']['dtes'])} expirations")
    print(f"Puts: {len(result['puts']['strikes'])} strikes × {len(result['puts']['dtes'])} expirations") 