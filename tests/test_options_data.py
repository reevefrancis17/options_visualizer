"""
Unit tests for the options data processing functionality.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from python.options_data import OptionsDataManager, OptionsDataProcessor


@pytest.fixture
def sample_options_data():
    """Create sample options data for testing."""
    # Create a simple options chain with calls and puts
    expiry_date = '2023-12-15'
    
    calls = [
        {
            'strike': 90,
            'lastPrice': 15,
            'bid': 14.5,
            'ask': 15.5,
            'impliedVolatility': 0.3,
            'volume': 500,
            'openInterest': 1000,
            'expiration': expiry_date
        },
        {
            'strike': 95,
            'lastPrice': 10,
            'bid': 9.5,
            'ask': 10.5,
            'impliedVolatility': 0.25,
            'volume': 1000,
            'openInterest': 2000,
            'expiration': expiry_date
        },
        {
            'strike': 100,
            'lastPrice': 5,
            'bid': 4.8,
            'ask': 5.2,
            'impliedVolatility': 0.2,
            'volume': 1500,
            'openInterest': 3000,
            'expiration': expiry_date
        },
        {
            'strike': 105,
            'lastPrice': 2,
            'bid': 1.8,
            'ask': 2.2,
            'impliedVolatility': 0.22,
            'volume': 1000,
            'openInterest': 2000,
            'expiration': expiry_date
        },
        {
            'strike': 110,
            'lastPrice': 1,
            'bid': 0.9,
            'ask': 1.1,
            'impliedVolatility': 0.25,
            'volume': 500,
            'openInterest': 1000,
            'expiration': expiry_date
        }
    ]
    
    puts = [
        {
            'strike': 90,
            'lastPrice': 1,
            'bid': 0.9,
            'ask': 1.1,
            'impliedVolatility': 0.25,
            'volume': 500,
            'openInterest': 1000,
            'expiration': expiry_date
        },
        {
            'strike': 95,
            'lastPrice': 2,
            'bid': 1.8,
            'ask': 2.2,
            'impliedVolatility': 0.22,
            'volume': 1000,
            'openInterest': 2000,
            'expiration': expiry_date
        },
        {
            'strike': 100,
            'lastPrice': 5,
            'bid': 4.8,
            'ask': 5.2,
            'impliedVolatility': 0.2,
            'volume': 1500,
            'openInterest': 3000,
            'expiration': expiry_date
        },
        {
            'strike': 105,
            'lastPrice': 10,
            'bid': 9.5,
            'ask': 10.5,
            'impliedVolatility': 0.25,
            'volume': 1000,
            'openInterest': 2000,
            'expiration': expiry_date
        },
        {
            'strike': 110,
            'lastPrice': 15,
            'bid': 14.5,
            'ask': 15.5,
            'impliedVolatility': 0.3,
            'volume': 500,
            'openInterest': 1000,
            'expiration': expiry_date
        }
    ]
    
    # The OptionsDataProcessor expects a dictionary with expiration dates as keys
    # Each key contains a dictionary with 'calls' and 'puts' keys
    # Metadata should be prefixed with underscore to be filtered out
    return {
        expiry_date: {
            'calls': calls,
            'puts': puts
        },
        '_current_price': 100.0,
        '_ticker': 'TEST',
        '_expiration_dates': [expiry_date]
    }


@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_manager_init(mock_yahoo_api):
    """Test initialization of OptionsDataManager."""
    # Create a mock for the YahooFinanceAPI
    mock_api = MagicMock()
    mock_yahoo_api.return_value = mock_api
    
    # Initialize the OptionsDataManager
    manager = OptionsDataManager(cache_duration=300)
    
    # Check that the API was initialized correctly
    assert manager.data_source == OptionsDataManager.DATA_SOURCE_YAHOO
    assert manager.pricing_model == OptionsDataManager.MODEL_MARKET
    assert manager.cache_duration == 300
    assert manager.api is not None


@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_init(mock_yahoo_api, sample_options_data):
    """Test initialization of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data, current_price=100.0)
    
    # Check that the data was processed correctly
    assert processor.current_price == 100.0
    # The ticker is not stored as an attribute in OptionsDataProcessor
    assert len(processor.get_expirations()) > 0
    # Expirations are returned as pandas Timestamp objects
    assert any(exp.strftime('%Y-%m-%d') == '2023-12-15' for exp in processor.get_expirations())


@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_process_data(mock_yahoo_api, sample_options_data):
    """Test the process_data method of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data, current_price=100.0)
    
    # Process the data
    processor.process_data(sample_options_data)
    
    # Check that the data was processed correctly
    assert processor.ds is not None
    assert len(processor.get_expirations()) > 0
    # Expirations are returned as pandas Timestamp objects
    assert any(exp.strftime('%Y-%m-%d') == '2023-12-15' for exp in processor.get_expirations())


@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_get_data_for_expiry(mock_yahoo_api, sample_options_data):
    """Test the get_data_for_expiry method of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data, current_price=100.0)
    
    # Get data for a specific expiry
    expiry_data = processor.get_data_for_expiry('2023-12-15')
    
    # Check that we got the correct data
    assert expiry_data is not None
    assert len(expiry_data) == 10  # 5 calls + 5 puts
    assert set(expiry_data['option_type'].unique()) == {'call', 'put'}


@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_get_strike_range(mock_yahoo_api, sample_options_data):
    """Test the get_strike_range method of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data, current_price=100.0)
    
    # Get the strike range
    min_strike, max_strike = processor.get_strike_range()
    
    # Check that we got the correct range
    assert min_strike == 90
    assert max_strike == 110


@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_compute_greeks(mock_yahoo_api, sample_options_data):
    """Test the computation of Greeks in OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data, current_price=100.0)
    
    # Compute delta
    processor.compute_delta()
    
    # Get data for the expiry
    expiry_data = processor.get_data_for_expiry('2023-12-15')
    
    # The return format might be a DataFrame, so we need to check differently
    if isinstance(expiry_data, dict) and 'calls' in expiry_data and 'puts' in expiry_data:
        # Dictionary format with 'calls' and 'puts' keys
        assert any('delta' in option for option in expiry_data['calls'])
        assert any('delta' in option for option in expiry_data['puts'])
        
        # Check that delta values are reasonable
        call_deltas = [option.get('delta', 0) for option in expiry_data['calls'] if 'delta' in option]
        put_deltas = [option.get('delta', 0) for option in expiry_data['puts'] if 'delta' in option]
        
        # Calls should have positive delta, puts should have negative delta
        if call_deltas:
            assert all(delta >= 0 for delta in call_deltas)
        if put_deltas:
            assert all(delta <= 0 for delta in put_deltas)
    else:
        # DataFrame format
        assert 'delta' in expiry_data.columns
        assert not expiry_data['delta'].isnull().all()
        
        # Check that delta values are reasonable
        call_deltas = expiry_data[expiry_data['option_type'] == 'call']['delta']
        put_deltas = expiry_data[expiry_data['option_type'] == 'put']['delta']
        
        # Calls should have positive delta, puts should have negative delta
        if not call_deltas.empty:
            assert all(call_deltas >= 0)
        if not put_deltas.empty:
            assert all(put_deltas <= 0) 