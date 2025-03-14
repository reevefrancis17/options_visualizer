"""
Tests for the interface between the backend and frontend.

These tests focus on how the frontend interacts with the backend API,
ensuring that data is correctly passed and processed between the two components.
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock, call

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from options_visualizer_web.app import app as frontend_app
from options_visualizer_backend.app import app as backend_app


@pytest.fixture
def mock_backend_response():
    """Mock response from the backend API."""
    return {
        "options_data": {
            "ticker": "AAPL",
            "current_price": 150.0,
            "expiration_dates": ["2023-12-15", "2024-01-19"],
            "calls": [
                {
                    "strike": 145.0,
                    "lastPrice": 10.0,
                    "bid": 9.8,
                    "ask": 10.2,
                    "impliedVolatility": 0.3,
                    "volume": 1000,
                    "openInterest": 500,
                    "expiration": "2023-12-15",
                    "delta": 0.65,
                    "gamma": 0.02,
                    "theta": -0.05,
                    "vega": 0.1
                }
            ],
            "puts": [
                {
                    "strike": 145.0,
                    "lastPrice": 5.0,
                    "bid": 4.8,
                    "ask": 5.2,
                    "impliedVolatility": 0.25,
                    "volume": 800,
                    "openInterest": 400,
                    "expiration": "2023-12-15",
                    "delta": -0.35,
                    "gamma": 0.02,
                    "theta": -0.04,
                    "vega": 0.1
                }
            ]
        }
    }


def test_frontend_calls_backend_api(frontend_client, mock_backend_response):
    """Test that the frontend correctly calls the backend API."""
    with patch('options_visualizer_web.app.requests.get') as mock_get:
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_backend_response
        mock_get.return_value = mock_response
        
        # Make a request to the frontend endpoint
        response = frontend_client.get("/api/options/AAPL")
        
        # Check that the frontend made the correct request to the backend
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "AAPL" in args[0]
        
        # Check that the frontend returned the correct response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == mock_backend_response


def test_frontend_handles_backend_error(frontend_client):
    """Test that the frontend correctly handles errors from the backend."""
    with patch('options_visualizer_web.app.requests.get') as mock_get:
        # Set up the mock response to simulate a backend error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Backend error"}
        mock_get.return_value = mock_response
        
        # Make a request to the frontend endpoint
        response = frontend_client.get("/api/options/AAPL")
        
        # Check that the frontend returned an error response
        assert response.status_code == 500
        data = json.loads(response.data)
        assert "error" in data


def test_frontend_handles_backend_timeout(frontend_client):
    """Test that the frontend correctly handles timeouts from the backend."""
    with patch('options_visualizer_web.app.requests.get') as mock_get:
        # Set up the mock to raise a timeout exception
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
        
        # Make a request to the frontend endpoint
        response = frontend_client.get("/api/options/AAPL")
        
        # Check that the frontend returned an error response
        assert response.status_code == 504  # Gateway Timeout
        data = json.loads(response.data)
        assert "error" in data
        assert "timed out" in data["error"].lower()


def test_frontend_handles_backend_connection_error(frontend_client):
    """Test that the frontend correctly handles connection errors from the backend."""
    with patch('options_visualizer_web.app.requests.get') as mock_get:
        # Set up the mock to raise a connection exception
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        # Make a request to the frontend endpoint
        response = frontend_client.get("/api/options/AAPL")
        
        # Check that the frontend returned an error response
        assert response.status_code == 503  # Service Unavailable
        data = json.loads(response.data)
        assert "error" in data
        assert "connect" in data["error"].lower()


def test_frontend_processes_backend_data(frontend_client, mock_backend_response):
    """Test that the frontend correctly processes data from the backend."""
    with patch('options_visualizer_web.app.requests.get') as mock_get:
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_backend_response
        mock_get.return_value = mock_response
        
        # Make a request to the frontend endpoint
        response = frontend_client.get("/api/options/AAPL?expiry=2023-12-15")
        
        # Check that the frontend returned the correct response
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check that the data was processed correctly
        assert data["options_data"]["ticker"] == "AAPL"
        assert data["options_data"]["current_price"] == 150.0
        assert "2023-12-15" in data["options_data"]["expiration_dates"]
        
        # Check that the options data includes the Greeks
        assert "delta" in data["options_data"]["calls"][0]
        assert "gamma" in data["options_data"]["calls"][0]
        assert "theta" in data["options_data"]["calls"][0]
        assert "vega" in data["options_data"]["calls"][0]


def test_backend_returns_correct_data_format(backend_client):
    """Test that the backend returns data in the correct format for the frontend."""
    with patch('options_visualizer_backend.app.get_data_manager') as mock_get_manager:
        # Create a mock for the OptionsDataManager
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        # Create a mock processor
        mock_processor = MagicMock()
        mock_processor.get_data.return_value = {
            "calls": [
                {
                    "strike": 145.0,
                    "lastPrice": 10.0,
                    "bid": 9.8,
                    "ask": 10.2,
                    "impliedVolatility": 0.3,
                    "volume": 1000,
                    "openInterest": 500,
                    "expiration": "2023-12-15"
                }
            ],
            "puts": [
                {
                    "strike": 145.0,
                    "lastPrice": 5.0,
                    "bid": 4.8,
                    "ask": 5.2,
                    "impliedVolatility": 0.25,
                    "volume": 800,
                    "openInterest": 400,
                    "expiration": "2023-12-15"
                }
            ]
        }
        
        # Mock the get_expirations method to return datetime objects
        from datetime import datetime
        expiry_date = datetime.strptime("2023-12-15", "%Y-%m-%d")
        mock_processor.get_expirations.return_value = [expiry_date]
        
        # Set up the mock manager to return our mock processor
        mock_manager.get_options_data.return_value = (mock_processor, 150.0)
        
        # Make a request to the backend endpoint
        response = backend_client.get("/api/options/AAPL")
        
        # Check that the response is successful
        assert response.status_code == 200
        
        # Parse the response data
        data = json.loads(response.data)
        
        # Check that the response contains the expected data structure
        assert "options_data" in data
        assert "ticker" in data["options_data"]
        assert "current_price" in data["options_data"]
        assert "expiration_dates" in data["options_data"]
        assert "calls" in data["options_data"]
        assert "puts" in data["options_data"]
        
        # Check that the data types are correct
        assert isinstance(data["options_data"]["ticker"], str)
        assert isinstance(data["options_data"]["current_price"], (int, float))
        assert isinstance(data["options_data"]["expiration_dates"], list)
        assert isinstance(data["options_data"]["calls"], list)
        assert isinstance(data["options_data"]["puts"], list)


def test_backend_handles_invalid_ticker(backend_client):
    """Test that the backend correctly handles invalid ticker symbols."""
    with patch('options_visualizer_backend.app.get_data_manager') as mock_get_manager:
        # Create a mock for the OptionsDataManager
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        # Set up the mock manager to raise an exception for an invalid ticker
        mock_manager.get_options_data.side_effect = ValueError("Invalid ticker symbol")
        
        # Make a request to the backend endpoint with an invalid ticker
        response = backend_client.get("/api/options/INVALID")
        
        # Check that the response is an error
        assert response.status_code == 400
        
        # Parse the response data
        data = json.loads(response.data)
        
        # Check that the response contains an error message
        assert "error" in data
        assert "Invalid ticker symbol" in data["error"]


def test_backend_handles_no_options_data(backend_client):
    """Test that the backend correctly handles cases where no options data is available."""
    with patch('options_visualizer_backend.app.get_data_manager') as mock_get_manager:
        # Create a mock for the OptionsDataManager
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        # Set up the mock manager to return None for no options data
        mock_manager.get_options_data.return_value = (None, 150.0)
        
        # Make a request to the backend endpoint
        response = backend_client.get("/api/options/AAPL")
        
        # Check that the response is an error
        assert response.status_code == 404
        
        # Parse the response data
        data = json.loads(response.data)
        
        # Check that the response contains an error message
        assert "error" in data
        assert "No options data available" in data["error"]


def test_backend_handles_expiry_parameter(backend_client):
    """Test that the backend correctly handles the expiry parameter."""
    with patch('options_visualizer_backend.app.get_data_manager') as mock_get_manager:
        # Create a mock for the OptionsDataManager
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        # Create a mock processor
        mock_processor = MagicMock()
        mock_processor.get_data.return_value = {
            "calls": [
                {
                    "strike": 145.0,
                    "lastPrice": 10.0,
                    "bid": 9.8,
                    "ask": 10.2,
                    "impliedVolatility": 0.3,
                    "volume": 1000,
                    "openInterest": 500,
                    "expiration": "2024-01-19"
                }
            ],
            "puts": [
                {
                    "strike": 145.0,
                    "lastPrice": 5.0,
                    "bid": 4.8,
                    "ask": 5.2,
                    "impliedVolatility": 0.25,
                    "volume": 800,
                    "openInterest": 400,
                    "expiration": "2024-01-19"
                }
            ]
        }
        
        # Mock the get_expirations method to return datetime objects
        from datetime import datetime
        expiry_dates = [
            datetime.strptime("2023-12-15", "%Y-%m-%d"),
            datetime.strptime("2024-01-19", "%Y-%m-%d")
        ]
        mock_processor.get_expirations.return_value = expiry_dates
        
        # Mock the get_data_for_expiry method
        import pandas as pd
        df = pd.DataFrame({
            'strike': [145.0, 145.0],
            'lastPrice': [10.0, 5.0],
            'bid': [9.8, 4.8],
            'ask': [10.2, 5.2],
            'impliedVolatility': [0.3, 0.25],
            'volume': [1000, 800],
            'openInterest': [500, 400],
            'expiration': [expiry_dates[1], expiry_dates[1]],
            'option_type': ['call', 'put']
        })
        mock_processor.get_data_for_expiry.return_value = df
        
        # Set up the mock manager to return our mock processor
        mock_manager.get_options_data.return_value = (mock_processor, 150.0)
        
        # Make a request to the backend endpoint with an expiry parameter
        response = backend_client.get("/api/options/AAPL?expiry=2024-01-19")
        
        # Check that the response is successful
        assert response.status_code == 200
        
        # Parse the response data
        data = json.loads(response.data)
        
        # Check that the processor's get_data_for_expiry method was called with the correct expiry
        mock_processor.get_data_for_expiry.assert_called_with(expiry_dates[1])
        
        # Check that the response contains the expected data
        assert "options_data" in data
        assert data["options_data"]["ticker"] == "AAPL"


def test_frontend_backend_data_consistency(frontend_client, backend_client):
    """Test that the data format is consistent between frontend and backend."""
    # Create a mock backend response
    backend_data = {
        'options_data': {
            'ticker': 'AAPL',
            'current_price': 150.0,
            'expiration_dates': ['2023-12-15'],
            'calls': [
                {
                    'strike': 145.0,
                    'lastPrice': 10.0,
                    'bid': 9.8,
                    'ask': 10.2,
                    'impliedVolatility': 0.3,
                    'volume': 1000,
                    'openInterest': 500,
                    'expiration': '2023-12-15',
                    'delta': 0.65,
                    'gamma': 0.02,
                    'theta': -0.05,
                    'vega': 0.1
                }
            ],
            'puts': [
                {
                    'strike': 145.0,
                    'lastPrice': 5.0,
                    'bid': 4.8,
                    'ask': 5.2,
                    'impliedVolatility': 0.25,
                    'volume': 800,
                    'openInterest': 400,
                    'expiration': '2023-12-15',
                    'delta': -0.35,
                    'gamma': 0.02,
                    'theta': -0.04,
                    'vega': 0.1
                }
            ]
        }
    }
    
    # Test the frontend with the mock backend response
    with patch('options_visualizer_web.app.requests.get') as mock_frontend_request:
        # Set up the mock frontend request to return the backend response
        mock_frontend_response = MagicMock()
        mock_frontend_response.status_code = 200
        mock_frontend_response.json.return_value = backend_data
        mock_frontend_request.return_value = mock_frontend_response
        
        # Make a request to the frontend endpoint
        frontend_response = frontend_client.get("/api/options/AAPL")
        
        # Check that the frontend response is successful
        assert frontend_response.status_code == 200
        
        # Parse the frontend response data
        frontend_data = json.loads(frontend_response.data)
        
        # Check that the data is consistent with the mock backend data
        assert frontend_data == backend_data


import requests

def test_frontend_handles_backend_request_exception(frontend_client):
    """Test that the frontend correctly handles request exceptions from the backend."""
    with patch('options_visualizer_web.app.requests.get') as mock_get:
        # Set up the mock to raise a generic request exception
        mock_get.side_effect = requests.exceptions.RequestException("Generic request error")
        
        # Make a request to the frontend endpoint
        response = frontend_client.get("/api/options/AAPL")
        
        # Check that the frontend returned an error response
        assert response.status_code == 500
        data = json.loads(response.data)
        assert "error" in data
        assert "request" in data["error"].lower() 