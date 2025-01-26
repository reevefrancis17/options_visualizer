import yfinance as yf
import pandas as pd
from ..utils.logging_utils import LoggerSetup

class MarketDataFetcher:
    def __init__(self):
        self.logger = LoggerSetup().logger

    def get_option_chain(self, symbol: str, expiration_date: str) -> dict:
        """Fetch option chain data for a given symbol and expiration"""
        try:
            self.logger.info(f"Fetching option chain for {symbol} expiring {expiration_date}")
            ticker = yf.Ticker(symbol)
            chains = ticker.option_chain(expiration_date)
            
            calls = pd.DataFrame({
                'strike': chains.calls['strike'],
                'call_price': chains.calls['lastPrice']
            })
            
            puts = pd.DataFrame({
                'strike': chains.puts['strike'],
                'put_price': puts.puts['lastPrice']
            })
            
            return self._merge_options(calls, puts)
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    def _merge_options(self, calls: pd.DataFrame, puts: pd.DataFrame) -> pd.DataFrame:
        """Merge calls and puts data"""
        options = pd.merge(calls, puts, on='strike', how='outer')
        return options.fillna(0).sort_values('strike') 