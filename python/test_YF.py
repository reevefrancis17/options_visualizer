import logging
from yahoo_finance import YahooFinanceAPI  # Assuming the class is in yahoo_finance.py

# Basic logging setup
logging.basicConfig(level=logging.INFO)

def test_yahoo_finance_api():
    api = YahooFinanceAPI()
    ticker = input("Enter a ticker symbol (e.g., AAPL): ").upper()
    
    options_data, current_price = api.get_options_data(ticker)
    
    if options_data and current_price:
        print(f"\nCurrent Price for {ticker}: {current_price}")
        print("\nOptions Data:")
        for date, data in options_data.items():
            print(f"\nExpiration Date: {date}")
            print("Calls:")
            print(data['calls'][['strike', 'lastPrice', 'volume']].head())  # Selected columns
            print("Puts:")
            print(data['puts'][['strike', 'lastPrice', 'volume']].head())  # Selected columns
    else:
        print(f"Failed to fetch data for {ticker}")

if __name__ == "__main__":
    test_yahoo_finance_api()