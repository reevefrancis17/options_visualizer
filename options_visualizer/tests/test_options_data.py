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
    calls = pd.DataFrame({
        'strike': [90, 95, 100, 105, 110],
        'lastPrice': [15, 10, 5, 2, 1],
        'bid': [14.5, 9.5, 4.8, 1.8, 0.9],
        'ask': [15.5, 10.5, 5.2, 2.2, 1.1],
        'impliedVolatility': [0.3, 0.25, 0.2, 0.22, 0.25],
        'volume': [500, 1000, 1500, 1000, 500],
        'openInterest': [1000, 2000, 3000, 2000, 1000],
        'expiration': ['2023-12-15'] * 5
    })
    
    puts = pd.DataFrame({
        'strike': [90, 95, 100, 105, 110],
        'lastPrice': [1, 2, 5, 10, 15],
        'bid': [0.9, 1.8, 4.8, 9.5, 14.5],
        'ask': [1.1, 2.2, 5.2, 10.5, 15.5],
        'impliedVolatility': [0.25, 0.22, 0.2, 0.25, 0.3],
        'volume': [500, 1000, 1500, 1000, 500],
        'openInterest': [1000, 2000, 3000, 2000, 1000],
        'expiration': ['2023-12-15'] * 5
    })
    
    return {
        'calls': calls.to_dict('records'),
        'puts': puts.to_dict('records'),
        'current_price': 100.0,
        'ticker': 'TEST',
        'expiration_dates': ['2023-12-15']
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
    assert processor.ticker == 'TEST'
    assert len(processor.expirations) == 1
    assert processor.expirations[0] == '2023-12-15'


@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_process_data(mock_yahoo_api, sample_options_data):
    """Test the process_data method of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data, current_price=100.0)
    
    # Process the data
    processor.process_data(sample_options_data)
    
    # Check that the data was processed correctly
    assert processor.df is not None
    assert 'strike' in processor.df.columns
    assert 'option_type' in processor.df.columns
    assert 'expiration' in processor.df.columns
    assert 'mid_price' in processor.df.columns
    
    # Check that we have both calls and puts
    assert set(processor.df['option_type'].unique()) == {'call', 'put'}
    
    # Check that we have the correct number of rows
    assert len(processor.df) == 10  # 5 calls + 5 puts


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
    
    # Check that delta was computed
    assert 'delta' in processor.df.columns
    
    # Check that delta values are reasonable
    call_deltas = processor.df[processor.df['option_type'] == 'call']['delta']
    put_deltas = processor.df[processor.df['option_type'] == 'put']['delta']
    
    # Call deltas should be between 0 and 1
    assert all(0 <= delta <= 1 for delta in call_deltas)
    
    # Put deltas should be between -1 and 0
    assert all(-1 <= delta <= 0 for delta in put_deltas) 