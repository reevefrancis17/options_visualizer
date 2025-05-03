"""
Pytest configuration file with fixtures for testing.
"""
import os
import sys
import pytest
from flask import Flask

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Flask applications
from backend.app import app as backend_app


@pytest.fixture
def backend_client():
    """
    Flask test client for the backend API.
    """
    backend_app.config['TESTING'] = True
    with backend_app.test_client() as client:
        yield client


@pytest.fixture
def mock_options_data():
    """
    Mock options data for testing.
    """
    return {
        "calls": [
            {
                "strike": 100.0,
                "lastPrice": 5.0,
                "bid": 4.8,
                "ask": 5.2,
                "impliedVolatility": 0.2,
                "volume": 1000,
                "openInterest": 500,
                "expiration": "2023-12-15"
            }
        ],
        "puts": [
            {
                "strike": 100.0,
                "lastPrice": 3.0,
                "bid": 2.8,
                "ask": 3.2,
                "impliedVolatility": 0.25,
                "volume": 800,
                "openInterest": 400,
                "expiration": "2023-12-15"
            }
        ],
        "current_price": 100.0,
        "ticker": "SPY",
        "expiration_dates": ["2023-12-15"]
    } 