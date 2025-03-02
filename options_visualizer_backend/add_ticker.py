#!/usr/bin/env python3
import os
import sys
import pandas as pd
from datetime import datetime

def add_ticker(ticker):
    """Add a new ticker to the CSV file"""
    ticker = ticker.upper()
    csv_path = os.path.join('data', 'tickers.csv')
    
    # Create the data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Check if the CSV file exists
    if os.path.exists(csv_path):
        # Read the existing CSV file
        df = pd.read_csv(csv_path)
        
        # Check if the ticker already exists
        if ticker in df['ticker'].values:
            print(f"Ticker {ticker} already exists in the CSV file.")
            return
        
        # Add the new ticker
        new_row = pd.DataFrame({'ticker': [ticker], 'timestamp': [datetime.now().isoformat()]})
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        # Create a new CSV file
        df = pd.DataFrame({'ticker': [ticker], 'timestamp': [datetime.now().isoformat()]})
    
    # Save the CSV file
    df.to_csv(csv_path, index=False)
    print(f"Added ticker {ticker} to the CSV file.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_ticker.py <ticker>")
        sys.exit(1)
    
    ticker = sys.argv[1]
    add_ticker(ticker) 