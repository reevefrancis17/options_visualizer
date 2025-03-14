import pytest
import pandas as pd
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock
from python.options_data import OptionsDataProcessor, OptionsDataManager

@pytest.fixture
def sample_options_data_with_multiple_expiries():
    """Create sample options data with multiple expiry dates for testing."""
    return {
        '_ticker': 'AAPL',
        '_current_price': 100.0,
        '_expiration_dates': ['2023-12-15', '2024-01-19'],
        '2023-12-15': {
            'calls': [
                {
                    'strike': 90,
                    'lastPrice': 16.0,
                    'bid': 13.3,
                    'ask': 14.2,
                    'impliedVolatility': 0.2575,
                    'volume': 600,
                    'openInterest': 1200,
                    'expiration': '2023-12-15'
                },
                {
                    'strike': 100,
                    'lastPrice': 6.0,
                    'bid': 8.0,
                    'ask': 8.5,
                    'impliedVolatility': 0.2,
                    'volume': 1200,
                    'openInterest': 2400,
                    'expiration': '2023-12-15'
                }
            ],
            'puts': [
                {
                    'strike': 90,
                    'lastPrice': 0.8,
                    'bid': 1.5,
                    'ask': 1.75,
                    'impliedVolatility': 0.22,
                    'volume': 800,
                    'openInterest': 1600,
                    'expiration': '2023-12-15'
                },
                {
                    'strike': 100,
                    'lastPrice': 4.5,
                    'bid': 3.5,
                    'ask': 3.85,
                    'impliedVolatility': 0.189,
                    'volume': 1600,
                    'openInterest': 3200,
                    'expiration': '2023-12-15'
                }
            ]
        },
        '2024-01-19': {
            'calls': [
                {
                    'strike': 90,
                    'lastPrice': 18.0,
                    'bid': 15.3,
                    'ask': 16.2,
                    'impliedVolatility': 0.2675,
                    'volume': 500,
                    'openInterest': 1000,
                    'expiration': '2024-01-19'
                },
                {
                    'strike': 100,
                    'lastPrice': 8.0,
                    'bid': 10.0,
                    'ask': 10.5,
                    'impliedVolatility': 0.21,
                    'volume': 1000,
                    'openInterest': 2000,
                    'expiration': '2024-01-19'
                }
            ],
            'puts': [
                {
                    'strike': 90,
                    'lastPrice': 1.8,
                    'bid': 2.5,
                    'ask': 2.75,
                    'impliedVolatility': 0.23,
                    'volume': 700,
                    'openInterest': 1400,
                    'expiration': '2024-01-19'
                },
                {
                    'strike': 100,
                    'lastPrice': 5.5,
                    'bid': 4.5,
                    'ask': 4.85,
                    'impliedVolatility': 0.199,
                    'volume': 1400,
                    'openInterest': 2800,
                    'expiration': '2024-01-19'
                }
            ]
        }
    }

@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_init(mock_yahoo_api, sample_options_data_with_multiple_expiries):
    """Test the initialization of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data_with_multiple_expiries, current_price=100.0)
    
    # Check that the processor was initialized correctly
    assert processor.data == sample_options_data_with_multiple_expiries
    assert processor.current_price == 100.0
    assert processor.ticker == 'AAPL'

@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_get_expirations(mock_yahoo_api, sample_options_data_with_multiple_expiries):
    """Test the get_expirations method of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data_with_multiple_expiries, current_price=100.0)
    
    # Get the expiration dates
    expiry_dates = processor.get_expirations()
    
    # Check that we got the correct dates
    assert len(expiry_dates) == 2
    assert pd.Timestamp('2023-12-15') in expiry_dates
    assert pd.Timestamp('2024-01-19') in expiry_dates

@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_get_data_frame(mock_yahoo_api, sample_options_data_with_multiple_expiries):
    """Test the get_data_frame method of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data_with_multiple_expiries, current_price=100.0)
    
    # Get the data frame
    df = processor.get_data_frame()
    
    # Check that we got a DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # Check that it has the expected columns
    expected_columns = ['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility',
                        'volume', 'openInterest', 'expiration', 'option_type']
    for col in expected_columns:
        assert col in df.columns
    
    # Check that it has the expected number of rows (4 options per expiry, 2 expiries)
    assert len(df) == 8

@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_compute_delta(mock_yahoo_api, sample_options_data_with_multiple_expiries):
    """Test the compute_delta method of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data_with_multiple_expiries, current_price=100.0)
    
    # Get the data frame
    df = processor.get_data_frame()
    
    # Compute delta
    df_with_delta = processor.compute_delta(df)
    
    # Check that delta was computed
    assert 'delta' in df_with_delta.columns
    
    # Check that call deltas are positive and put deltas are negative
    call_deltas = df_with_delta[df_with_delta['option_type'] == 'call']['delta']
    put_deltas = df_with_delta[df_with_delta['option_type'] == 'put']['delta']
    
    assert all(call_deltas >= 0)
    assert all(put_deltas <= 0)

@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_compute_gamma(mock_yahoo_api, sample_options_data_with_multiple_expiries):
    """Test the compute_gamma method of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data_with_multiple_expiries, current_price=100.0)
    
    # Get the data frame
    df = processor.get_data_frame()
    
    # Compute gamma
    df_with_gamma = processor.compute_gamma(df)
    
    # Check that gamma was computed
    assert 'gamma' in df_with_gamma.columns
    
    # Check that all gamma values are positive
    assert all(df_with_gamma['gamma'] >= 0)

@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_compute_theta(mock_yahoo_api, sample_options_data_with_multiple_expiries):
    """Test the compute_theta method of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data_with_multiple_expiries, current_price=100.0)
    
    # Get the data frame
    df = processor.get_data_frame()
    
    # Compute theta
    df_with_theta = processor.compute_theta(df)
    
    # Check that theta was computed
    assert 'theta' in df_with_theta.columns
    
    # Check that all theta values are negative
    assert all(df_with_theta['theta'] <= 0)

@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_compute_vega(mock_yahoo_api, sample_options_data_with_multiple_expiries):
    """Test the compute_vega method of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data_with_multiple_expiries, current_price=100.0)
    
    # Get the data frame
    df = processor.get_data_frame()
    
    # Compute vega
    df_with_vega = processor.compute_vega(df)
    
    # Check that vega was computed
    assert 'vega' in df_with_vega.columns
    
    # Check that all vega values are positive
    assert all(df_with_vega['vega'] >= 0)

@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_compute_rho(mock_yahoo_api, sample_options_data_with_multiple_expiries):
    """Test the compute_rho method of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data_with_multiple_expiries, current_price=100.0)
    
    # Get the data frame
    df = processor.get_data_frame()
    
    # Compute rho
    df_with_rho = processor.compute_rho(df)
    
    # Check that rho was computed
    assert 'rho' in df_with_rho.columns
    
    # Check that call rho is positive and put rho is negative
    call_rhos = df_with_rho[df_with_rho['option_type'] == 'call']['rho']
    put_rhos = df_with_rho[df_with_rho['option_type'] == 'put']['rho']
    
    assert all(call_rhos >= 0)
    assert all(put_rhos <= 0)

@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_compute_greeks(mock_yahoo_api, sample_options_data_with_multiple_expiries):
    """Test the compute_greeks method of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data_with_multiple_expiries, current_price=100.0)
    
    # Get the data frame
    df = processor.get_data_frame()
    
    # Compute all Greeks
    df_with_greeks = processor.compute_greeks(df)
    
    # Check that all Greeks were computed
    for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
        assert greek in df_with_greeks.columns

@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_get_data(mock_yahoo_api, sample_options_data_with_multiple_expiries):
    """Test the get_data method of OptionsDataProcessor."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data_with_multiple_expiries, current_price=100.0)
    
    # Get the data
    data = processor.get_data()
    
    # Check that we got an xarray Dataset
    assert isinstance(data, xr.Dataset)
    
    # Check that it has the expected dimensions
    assert 'strike' in data.dims
    assert 'expiration' in data.dims
    assert 'option_type' in data.dims

@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_processor_get_strike_range(mock_yahoo_api, sample_options_data_with_multiple_expiries):
    """Test the get_strike_range method with padding."""
    # Create a processor with sample data
    processor = OptionsDataProcessor(sample_options_data_with_multiple_expiries, current_price=100.0)
    
    # Get the strike range
    min_strike, max_strike = processor.get_strike_range()
    
    # Check that we got the correct range
    assert min_strike == 90
    assert max_strike == 100

@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_manager_get_options_data(mock_yahoo_api):
    """Test the get_options_data method of OptionsDataManager."""
    # Mock the YahooFinanceAPI
    mock_api_instance = MagicMock()
    mock_yahoo_api.return_value = mock_api_instance
    
    # Mock the get_options_data method
    mock_api_instance.get_options_data.return_value = ({
        '_current_price': 100.0,
        '_ticker': 'AAPL',
        '_expiration_dates': ['2023-12-15'],
        '2023-12-15': {
            'calls': [{'strike': 100, 'lastPrice': 5.0}],
            'puts': [{'strike': 100, 'lastPrice': 5.0}]
        }
    }, 100.0)
    
    # Create an OptionsDataManager
    manager = OptionsDataManager(cache_duration=300)
    
    # Get options data
    processor, current_price = manager.get_options_data('AAPL')
    
    # Check the results
    assert isinstance(processor, OptionsDataProcessor)
    assert current_price == 100.0
    
    # Verify the mock was called correctly
    mock_api_instance.get_options_data.assert_called_once_with('AAPL', None, None)

@patch('python.yahoo_finance.YahooFinanceAPI')
def test_options_data_manager_get_current_processor(mock_yahoo_api):
    """Test the get_current_processor method of OptionsDataManager."""
    # Mock the YahooFinanceAPI
    mock_api_instance = MagicMock()
    mock_yahoo_api.return_value = mock_api_instance
    
    # Mock the get_options_data method
    mock_api_instance.get_options_data.return_value = ({
        '_current_price': 100.0,
        '_ticker': 'AAPL',
        '_expiration_dates': ['2023-12-15'],
        '2023-12-15': {
            'calls': [{'strike': 100, 'lastPrice': 5.0}],
            'puts': [{'strike': 100, 'lastPrice': 5.0}]
        }
    }, 100.0)
    
    # Create an OptionsDataManager
    manager = OptionsDataManager(cache_duration=300)
    
    # Get the current processor (should be None initially)
    processor, current_price, status, progress = manager.get_current_processor('AAPL')
    
    # Check the results
    assert status == 'not_found'
    assert progress == 0.0
    
    # Now get the options data to populate the cache
    manager.get_options_data('AAPL')
    
    # Get the current processor again
    processor, current_price, status, progress = manager.get_current_processor('AAPL')
    
    # Check the results
    assert isinstance(processor, OptionsDataProcessor)
    assert current_price == 100.0
    assert status == 'complete'
    assert progress == 1.0 