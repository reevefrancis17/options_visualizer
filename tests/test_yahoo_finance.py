import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, PropertyMock
from python.yahoo_finance import YahooFinanceAPI

@pytest.fixture
def yahoo_api():
    """Fixture to create a YahooFinanceAPI instance."""
    return YahooFinanceAPI(cache_duration=300)

@patch('yfinance.Ticker')
def test_get_risk_free_rate(mock_ticker, yahoo_api):
    """Test getting the risk-free rate."""
    # Mock the Ticker instance
    mock_ticker_instance = MagicMock()
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock the info property
    type(mock_ticker_instance).info = PropertyMock(return_value={
        'regularMarketPrice': 100.0
    })
    
    # Mock the history method
    mock_ticker_instance.history.return_value = pd.DataFrame({
        'Close': [100.0, 101.0, 102.0]
    })
    
    # Call the method
    rate = yahoo_api.get_risk_free_rate()
    
    # Check the result
    assert isinstance(rate, float)
    assert 0.01 <= rate <= 0.05  # Reasonable range for risk-free rate
    
    # Verify the mock was called correctly
    mock_ticker.assert_called_once_with('^TNX')
    mock_ticker_instance.history.assert_called_once()

@patch('yfinance.Ticker')
def test_get_risk_free_rate_exception(mock_ticker, yahoo_api):
    """Test getting the risk-free rate when an exception occurs."""
    # Mock the Ticker instance
    mock_ticker_instance = MagicMock()
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock the info property to raise an exception
    type(mock_ticker_instance).info = PropertyMock(side_effect=Exception("Test exception"))
    
    # Call the method
    rate = yahoo_api.get_risk_free_rate()
    
    # Check that a default value is returned
    assert rate == 0.04  # Default value

@patch('yfinance.Ticker')
def test_get_current_price(mock_ticker, yahoo_api):
    """Test getting the current price."""
    # Mock the Ticker instance
    mock_ticker_instance = MagicMock()
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock the info property
    type(mock_ticker_instance).info = PropertyMock(return_value={
        'regularMarketPrice': 150.0
    })
    
    # Call the method
    price = yahoo_api._get_current_price("AAPL")
    
    # Check the result
    assert price == 150.0
    
    # Verify the mock was called correctly
    mock_ticker.assert_called_once_with("AAPL")

@patch('yfinance.Ticker')
def test_get_current_price_exception(mock_ticker, yahoo_api):
    """Test getting the current price when an exception occurs."""
    # Mock the Ticker instance
    mock_ticker_instance = MagicMock()
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock the info property to raise an exception
    type(mock_ticker_instance).info = PropertyMock(side_effect=Exception("Test exception"))
    
    # Call the method and check that it raises an exception
    with pytest.raises(ValueError, match="Failed to get price for AAPL"):
        yahoo_api._get_current_price("AAPL")

@patch('yfinance.Ticker')
def test_process_expiry_date(mock_ticker, yahoo_api):
    """Test processing an expiry date."""
    # Mock the Ticker instance
    mock_ticker_instance = MagicMock()
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock the option_chain method
    mock_option_chain = MagicMock()
    mock_option_chain.calls = pd.DataFrame({
        'strike': [100, 110],
        'lastPrice': [5.0, 2.0],
        'bid': [4.8, 1.8],
        'ask': [5.2, 2.2],
        'impliedVolatility': [0.2, 0.25],
        'volume': [1000, 500],
        'openInterest': [2000, 1000],
        'expiration': ['2023-12-15', '2023-12-15']
    })
    mock_option_chain.puts = pd.DataFrame({
        'strike': [100, 110],
        'lastPrice': [2.0, 5.0],
        'bid': [1.8, 4.8],
        'ask': [2.2, 5.2],
        'impliedVolatility': [0.2, 0.25],
        'volume': [1000, 500],
        'openInterest': [2000, 1000],
        'expiration': ['2023-12-15', '2023-12-15']
    })
    mock_ticker_instance.option_chain.return_value = mock_option_chain
    
    # Call the method
    stock = mock_ticker_instance
    expiry = '2023-12-15'
    processed_dates = {}
    total_dates = 1
    
    result = yahoo_api._process_expiry_date(stock, "AAPL", expiry, processed_dates, total_dates)
    
    # Check the result
    assert result['expiry'] == '2023-12-15'
    assert 'calls' in result['data']
    assert 'puts' in result['data']
    assert len(result['data']['calls']) == 2
    assert len(result['data']['puts']) == 2

@patch('yfinance.Ticker')
def test_process_expiry_date_exception(mock_ticker, yahoo_api):
    """Test processing an expiry date when an exception occurs."""
    # Mock the Ticker instance
    mock_ticker_instance = MagicMock()
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock the option_chain method to raise an exception
    mock_ticker_instance.option_chain.side_effect = Exception("Test exception")
    
    # Call the method
    stock = mock_ticker_instance
    expiry = '2023-12-15'
    processed_dates = {}
    total_dates = 1
    
    result = yahoo_api._process_expiry_date(stock, "AAPL", expiry, processed_dates, total_dates)
    
    # Check that an error result is returned
    assert result['expiry'] == '2023-12-15'
    assert result['data']['calls'] == []
    assert result['data']['puts'] == []
    assert result['error'] == 'Test exception'

@patch('yfinance.Ticker')
def test_get_options_data(mock_ticker, yahoo_api):
    """Test getting options data."""
    # Mock the Ticker instance
    mock_ticker_instance = MagicMock()
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock the info property
    type(mock_ticker_instance).info = PropertyMock(return_value={'regularMarketPrice': 150.0})
    
    # Mock the options property (expiry dates)
    type(mock_ticker_instance).options = PropertyMock(return_value=['2023-12-15'])
    
    # Mock the option_chain method
    mock_option_chain = MagicMock()
    mock_option_chain.calls = pd.DataFrame({
        'strike': [100, 110],
        'lastPrice': [5.0, 2.0],
        'bid': [4.8, 1.8],
        'ask': [5.2, 2.2],
        'impliedVolatility': [0.2, 0.25],
        'volume': [1000, 500],
        'openInterest': [2000, 1000],
        'expiration': ['2023-12-15', '2023-12-15']
    })
    mock_option_chain.puts = pd.DataFrame({
        'strike': [100, 110],
        'lastPrice': [2.0, 5.0],
        'bid': [1.8, 4.8],
        'ask': [2.2, 5.2],
        'impliedVolatility': [0.2, 0.25],
        'volume': [1000, 500],
        'openInterest': [2000, 1000],
        'expiration': ['2023-12-15', '2023-12-15']
    })
    mock_ticker_instance.option_chain.return_value = mock_option_chain
    
    # Call the method
    result = yahoo_api.get_options_data("AAPL")
    
    # Check the result
    assert isinstance(result, tuple)
    data, price = result
    assert isinstance(data, dict)
    assert price == 150.0
    assert '_ticker' in data
    assert '_expiration_dates' in data
    assert '2023-12-15' in data

@patch('concurrent.futures.ThreadPoolExecutor')
def test_get_batch_options_data(mock_executor, yahoo_api):
    """Test getting batch options data."""
    # Mock the executor
    mock_executor_instance = MagicMock()
    mock_executor.return_value.__enter__.return_value = mock_executor_instance
    
    # Mock the submit method to return a future with a result
    def mock_submit(func, ticker, callback=None, max_dates=None):
        mock_future = MagicMock()
        mock_future.result.return_value = ({
            '_current_price': 150.0,
            '_ticker': ticker,
            '_expiration_dates': ['2023-12-15'],
            '2023-12-15': {
                'calls': [{'strike': 100}],
                'puts': [{'strike': 100}]
            }
        }, 150.0)
        return mock_future
    
    mock_executor_instance.submit.side_effect = mock_submit
    
    # Call the method
    result = yahoo_api.get_batch_options_data(["AAPL", "MSFT"])
    
    # Check the result
    assert isinstance(result, dict)
    assert "AAPL" in result
    assert "MSFT" in result
    assert result["AAPL"][0]['_ticker'] == "AAPL"
    assert result["MSFT"][0]['_ticker'] == "MSFT"
    
    # Verify the mock was called correctly
    assert mock_executor_instance.submit.call_count == 2 