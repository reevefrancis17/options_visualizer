import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

class OptionsData:
    def __init__(self):
        """Initialize an empty options data structure"""
        self.data = None
        self.symbol = None
        self.last_updated = None
        self.spot_price = None

    def fetch_data(self, symbol):
        """Fetch all options data for a given symbol"""
        self.symbol = symbol.upper()
        ticker = yf.Ticker(self.symbol)
        
        # Get current stock price using history() instead of info
        try:
            hist = ticker.history(period='1d')
            if not hist.empty:
                self.spot_price = float(hist['Close'].iloc[-1])
            else:
                raise ValueError(f"Could not get spot price for {self.symbol}")
        except Exception as e:
            print(f"Error getting spot price from history: {str(e)}")
            # Fallback to info
            self.spot_price = ticker.info.get('regularMarketPrice')
            if not self.spot_price:
                raise ValueError(f"Could not get spot price for {self.symbol}")

        # Get all expiry dates
        expiry_dates = ticker.options
        if not expiry_dates:
            raise ValueError(f"No options available for {self.symbol}")

        # Initialize lists to store data
        all_data = []
        
        # Fetch data for each expiry date
        for date in expiry_dates:
            try:
                chain = ticker.option_chain(date)
                
                # Process calls
                calls_df = self._process_options_df(chain.calls, 'call', date)
                puts_df = self._process_options_df(chain.puts, 'put', date)
                
                # Combine calls and puts
                all_data.extend([calls_df, puts_df])
                
            except Exception as e:
                print(f"Error fetching data for {date}: {str(e)}")
                continue

        if not all_data:
            raise ValueError(f"Could not fetch any valid options data for {self.symbol}")

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Convert to xarray
        self.data = self._create_xarray(combined_df)
        self.last_updated = datetime.now()

    def _process_options_df(self, df, option_type, expiry_date):
        """Process options dataframe to extract relevant columns"""
        processed_df = pd.DataFrame({
            'expiry_date': expiry_date,
            'strike': df['strike'],
            'bid': df['bid'],
            'ask': df['ask'],
            'last_price': df['lastPrice'],
            'volume': df['volume'],
            'open_interest': df['openInterest'],
            'option_type': option_type
        })
        
        # Calculate days to expiry
        processed_df['days_to_expiry'] = (
            pd.to_datetime(expiry_date) - pd.Timestamp.now()
        ).days
        
        # Calculate mid price
        processed_df['mid_price'] = (processed_df['bid'] + processed_df['ask']) / 2
        
        return processed_df

    def _create_xarray(self, df):
        """Convert processed dataframe to xarray Dataset"""
        # Create unique coordinates
        dates = sorted(df['expiry_date'].unique())
        strikes = sorted(df['strike'].unique())
        option_types = ['call', 'put']
        
        # Create coordinate arrays
        coords = {
            'expiry_date': dates,
            'strike': strikes,
            'option_type': option_types
        }
        
        # Initialize data arrays with NaN
        shape = (len(dates), len(strikes), len(option_types))
        
        # Create data variables
        data_vars = {
            'bid': (('expiry_date', 'strike', 'option_type'), np.full(shape, np.nan)),
            'ask': (('expiry_date', 'strike', 'option_type'), np.full(shape, np.nan)),
            'last_price': (('expiry_date', 'strike', 'option_type'), np.full(shape, np.nan)),
            'volume': (('expiry_date', 'strike', 'option_type'), np.full(shape, np.nan)),
            'open_interest': (('expiry_date', 'strike', 'option_type'), np.full(shape, np.nan)),
            'mid_price': (('expiry_date', 'strike', 'option_type'), np.full(shape, np.nan)),
            'days_to_expiry': (('expiry_date'), np.array([
                (pd.to_datetime(date) - pd.Timestamp.now()).days 
                for date in dates
            ]))
        }
        
        # Fill in the data
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        
        # Update values from dataframe
        for _, row in df.iterrows():
            date_idx = dates.index(row['expiry_date'])
            strike_idx = strikes.index(row['strike'])
            type_idx = option_types.index(row['option_type'])
            
            for field in ['bid', 'ask', 'last_price', 'volume', 'open_interest', 'mid_price']:
                ds[field].values[date_idx, strike_idx, type_idx] = row[field]
        
        return ds

    def get_chain_for_date(self, expiry_date):
        """Get option chain for a specific expiry date"""
        if self.data is None:
            raise ValueError("No data available. Call fetch_data first.")
            
        # Convert date to string format if it's a datetime
        if isinstance(expiry_date, datetime):
            expiry_date = expiry_date.strftime('%Y-%m-%d')
            
        # Select data for the specific date
        chain = self.data.sel(expiry_date=expiry_date)
        
        # Convert to dictionary format for API response
        result = {
            'friday_date': expiry_date,
            'spot_price': self.spot_price,
            'options': []
        }
        
        def safe_float(x):
            """Safely convert value to float, handling NaN"""
            try:
                if pd.isna(x):
                    return 0.0
                return float(x)
            except:
                return 0.0
                
        def safe_int(x):
            """Safely convert value to int, handling NaN"""
            try:
                if pd.isna(x):
                    return 0
                return int(x)
            except:
                return 0
        
        # Iterate through strikes
        for strike in chain.strike.values:
            call_data = chain.sel(strike=strike, option_type='call')
            put_data = chain.sel(strike=strike, option_type='put')
            
            option_data = {
                'strike': safe_float(strike),
                'call_price': safe_float(call_data.last_price.item()),
                'put_price': safe_float(put_data.last_price.item()),
                'call_bid': safe_float(call_data.bid.item()),
                'call_ask': safe_float(call_data.ask.item()),
                'put_bid': safe_float(put_data.bid.item()),
                'put_ask': safe_float(put_data.ask.item()),
                'call_volume': safe_int(call_data.volume.item()),
                'put_volume': safe_int(put_data.volume.item()),
                'call_oi': safe_int(call_data.open_interest.item()),
                'put_oi': safe_int(put_data.open_interest.item())
            }
            result['options'].append(option_data)
            
        return result

    def get_available_dates(self):
        """Get list of available expiry dates"""
        if self.data is None:
            raise ValueError("No data available. Call fetch_data first.")
        return self.data.expiry_date.values.tolist() 