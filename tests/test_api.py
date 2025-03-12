"""
Integration tests for the API endpoints.
"""
import pytest
import json
from unittest.mock import patch, MagicMock


@patch('options_visualizer_backend.app.OptionsDataManager')
def test_options_endpoint(mock_options_manager, backend_client, mock_options_data):
    """Test the options endpoint."""
    # Create a mock for the OptionsDataManager
    mock_manager = MagicMock()
    mock_options_manager.return_value = mock_manager
    
    # Create a mock processor
    mock_processor = MagicMock()
    mock_processor.get_data.return_value = mock_options_data
    mock_processor.get_expirations.return_value = ["2023-12-15"]
    
    # Set up the mock manager to return our mock processor
    mock_manager.get_options_data.return_value = (mock_processor, 100.0)
    
    # Make a request to the options endpoint
    response = backend_client.get("/api/options/SPY")
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Parse the response data
    data = json.loads(response.data)
    
    # Check that the response contains the expected data
    assert "options_data" in data
    assert data["options_data"]["ticker"] == "SPY"
    assert "current_price" in data["options_data"]
    assert "expiration_dates" in data["options_data"]


@patch('options_visualizer_backend.app.OptionsDataManager')
def test_options_endpoint_with_expiry(mock_options_manager, backend_client, mock_options_data):
    """Test the options endpoint with a specific expiry date."""
    # Create a mock for the OptionsDataManager
    mock_manager = MagicMock()
    mock_options_manager.return_value = mock_manager
    
    # Create a mock processor
    mock_processor = MagicMock()
    mock_processor.get_data.return_value = mock_options_data
    mock_processor.get_expirations.return_value = ["2023-12-15"]
    
    # Set up the mock manager to return our mock processor
    mock_manager.get_options_data.return_value = (mock_processor, 100.0)
    
    # Make a request to the options endpoint with a specific expiry
    response = backend_client.get("/api/options/SPY?expiry=2023-12-15")
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Parse the response data
    data = json.loads(response.data)
    
    # Check that the response contains the expected data
    assert "options_data" in data
    assert data["options_data"]["ticker"] == "SPY"
    assert "current_price" in data["options_data"]
    assert "expiration_dates" in data["options_data"]


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