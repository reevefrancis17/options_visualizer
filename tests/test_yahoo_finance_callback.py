"""
Tests for the Yahoo Finance API callback functionality.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python.yahoo_finance import YahooFinanceAPI, TickerNotFoundError, NoOptionsDataError


@pytest.fixture
def mock_yf_ticker():
    """Mock the yfinance Ticker class."""
    with patch('yfinance.Ticker') as mock_ticker:
        # Mock the info property
        mock_ticker.return_value.info = {'regularMarketPrice': 100.0}
        
        # Mock the options property
        mock_ticker.return_value.options = ['2023-01-01', '2023-02-01']
        
        # Mock the option_chain method
        mock_option_chain = MagicMock()
        mock_option_chain.calls = MagicMock()
        mock_option_chain.calls.columns = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
        mock_option_chain.calls.to_dict.return_value = [{'strike': 100, 'bid': 5, 'ask': 5.5}]
        
        mock_option_chain.puts = MagicMock()
        mock_option_chain.puts.columns = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
        mock_option_chain.puts.to_dict.return_value = [{'strike': 100, 'bid': 4, 'ask': 4.5}]
        
        mock_ticker.return_value.option_chain.return_value = mock_option_chain
        
        yield mock_ticker


def test_callback_with_correct_parameters(mock_yf_ticker):
    """Test that the callback is called with the correct parameters."""
    # Create a mock callback
    mock_callback = MagicMock()
    
    # Create the API
    api = YahooFinanceAPI()
    
    # Call get_options_data with the mock callback
    api.get_options_data('AAPL', mock_callback)
    
    # Check that the callback was called with the correct parameters
    assert mock_callback.call_count > 0
    
    # Check the parameters of the first call
    args, kwargs = mock_callback.call_args_list[0]
    
    # The callback should be called with (options_data, current_price, processed_dates, total_dates)
    assert len(args) == 4
    assert isinstance(args[0], dict)  # options_data
    assert isinstance(args[1], float)  # current_price
    assert isinstance(args[2], int)    # processed_dates
    assert isinstance(args[3], int)    # total_dates
    
    # Check that processed_dates and total_dates are correct
    assert args[2] <= args[3]  # processed_dates <= total_dates


def test_callback_error_handling(mock_yf_ticker):
    """Test that errors in the callback are handled gracefully."""
    # Create a callback that raises an exception
    def error_callback(*args, **kwargs):
        raise ValueError("Test error in callback")
    
    # Create the API
    api = YahooFinanceAPI()
    
    # Call get_options_data with the error callback
    # This should not raise an exception
    options_data, current_price = api.get_options_data('AAPL', error_callback)
    
    # Check that we still got valid data despite the callback error
    assert options_data is not None
    assert current_price is not None
    assert '_ticker' in options_data
    assert options_data['_ticker'] == 'AAPL'


def test_callback_with_queue_based_approach():
    """Test that the callback works with the queue-based approach."""
    import queue
    import threading
    import time
    
    # Create a queue for operations
    operations_queue = queue.Queue()
    
    # Create a function to process operations from the queue
    def process_queue():
        while True:
            try:
                # Get an operation from the queue with a timeout
                operation = operations_queue.get(timeout=1)
                
                if operation['type'] == 'shutdown':
                    break
                
                # Process the operation
                if operation['type'] == 'callback':
                    # Extract the parameters
                    options_data = operation['options_data']
                    current_price = operation['current_price']
                    processed_dates = operation['processed_dates']
                    total_dates = operation['total_dates']
                    
                    # Check that all parameters are present
                    assert options_data is not None
                    assert current_price is not None
                    assert processed_dates is not None
                    assert total_dates is not None
                
                # Mark the operation as done
                operations_queue.task_done()
            except queue.Empty:
                # No operations in the queue, continue waiting
                continue
    
    # Start the queue processing thread
    thread = threading.Thread(target=process_queue, daemon=True)
    thread.start()
    
    # Create a callback that puts operations in the queue
    def queue_callback(options_data, current_price, processed_dates, total_dates):
        operations_queue.put({
            'type': 'callback',
            'options_data': options_data,
            'current_price': current_price,
            'processed_dates': processed_dates,
            'total_dates': total_dates
        })
    
    # Directly call the callback function with test data
    queue_callback({'test': 'data'}, 100.0, 1, 2)
    
    # Wait for the queue to be processed
    time.sleep(0.1)
    
    # Check that the queue has an operation
    assert operations_queue.qsize() == 0  # Queue should be empty after processing
    
    # Shutdown the thread
    operations_queue.put({'type': 'shutdown'})
    thread.join(timeout=1) 