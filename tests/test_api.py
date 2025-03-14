"""
Integration tests for the API endpoints.
"""
import pytest
import json
from unittest.mock import patch, MagicMock


def test_options_endpoint(backend_client):
    """Test the /api/options/<ticker> endpoint."""
    with patch('options_visualizer_backend.app.get_data_manager') as mock_get_manager:
        # Create a mock for the OptionsDataManager
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        # Create a mock processor
        mock_processor = MagicMock()
        
        # Mock the get_expirations method to return datetime objects
        from datetime import datetime
        expiry_date = datetime.strptime("2023-12-15", "%Y-%m-%d")
        mock_processor.get_expirations.return_value = [expiry_date]
        
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
            'expiration': [expiry_date, expiry_date],
            'option_type': ['call', 'put']
        })
        mock_processor.get_data_for_expiry.return_value = df
        
        # Set up the mock manager to return our mock processor
        mock_manager.get_options_data.return_value = (mock_processor, 150.0)
        
        # Make a request to the endpoint
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


def test_options_endpoint_with_expiry(backend_client):
    """Test the /api/options/<ticker> endpoint with an expiry parameter."""
    with patch('options_visualizer_backend.app.get_data_manager') as mock_get_manager:
        # Create a mock for the OptionsDataManager
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        # Create a mock processor
        mock_processor = MagicMock()
        
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
        
        # Make a request to the endpoint with an expiry parameter
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


def test_health_endpoint(backend_client):
    """Test the health endpoint."""
    # Make a request to the health endpoint
    response = backend_client.get("/health")
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Parse the response data
    data = json.loads(response.data)
    
    # Check that the response contains the expected data
    assert "status" in data
    assert data["status"] == "healthy"


def test_frontend_health_endpoint(frontend_client):
    """Test the frontend health endpoint."""
    # Make a request to the health endpoint
    response = frontend_client.get("/health")
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Parse the response data
    data = json.loads(response.data)
    
    # Check that the response contains the expected data
    assert "status" in data
    assert data["status"] == "healthy"


def test_frontend_index(frontend_client):
    """Test the frontend index page."""
    # Make a request to the index page
    response = frontend_client.get("/")
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Check that the response contains HTML
    assert b"<!DOCTYPE html>" in response.data
    assert b"<html" in response.data 