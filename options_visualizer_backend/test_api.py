#!/usr/bin/env python3
import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_health():
    """Test the health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.status_code}")
    print(response.json())
    print()

def test_tickers():
    """Test the tickers endpoint"""
    response = requests.get(f"{BASE_URL}/api/tickers")
    print(f"Tickers: {response.status_code}")
    print(response.json())
    print()

def test_options(ticker="AAPL"):
    """Test the options endpoint for a specific ticker"""
    print(f"Fetching options data for {ticker}...")
    start_time = time.time()
    response = requests.get(f"{BASE_URL}/api/options/{ticker}")
    elapsed = time.time() - start_time
    print(f"Options for {ticker}: {response.status_code} (took {elapsed:.2f} seconds)")
    if response.status_code == 200:
        data = response.json()
        print(f"Current price: {data.get('price')}")
        print(f"Available expiration dates: {list(data.get('options', {}).keys())}")
    else:
        print(response.json())
    print()

if __name__ == "__main__":
    print("Testing API endpoints...")
    test_health()
    test_tickers()
    
    # Test a few tickers
    test_options("AAPL")
    test_options("MSFT")
    
    # Test the cache by fetching the same ticker again (should be faster)
    print("Testing cache by fetching AAPL again...")
    test_options("AAPL") 