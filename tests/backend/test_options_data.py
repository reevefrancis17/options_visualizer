import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from backend.options_data import OptionsDataManager
except ImportError:
    # If the module is not found in the backend directory, try the python directory
    from python.options_data import OptionsDataManager


class TestOptionsDataManager(unittest.TestCase):
    """Test cases for the OptionsDataManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test instance with a small max_workers value for testing
        self.data_manager = OptionsDataManager(max_workers=2)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if hasattr(self, 'data_manager') and self.data_manager:
            self.data_manager.shutdown()
    
    def test_initialization(self):
        """Test that the OptionsDataManager initializes correctly."""
        self.assertIsNotNone(self.data_manager)
        self.assertEqual(self.data_manager.max_workers, 2)
        self.assertIsNotNone(self.data_manager.thread_pool)
    
    def test_get_expiration_dates_empty(self):
        """Test that get_expiration_dates returns an empty list for invalid tickers."""
        dates = self.data_manager.get_expiration_dates("INVALID_TICKER_XYZ")
        self.assertEqual(len(dates), 0)
    
    def test_filter_by_dte(self):
        """Test the filter_by_dte method."""
        # Create a test DataFrame with expiration dates
        today = datetime.now().date()
        dates = [
            today + timedelta(days=1),  # 1 day to expiration
            today + timedelta(days=10),  # 10 days to expiration
            today + timedelta(days=30),  # 30 days to expiration
            today + timedelta(days=60),  # 60 days to expiration
        ]
        
        # Create a test DataFrame
        df = pd.DataFrame({
            'expiration': dates,
            'strike': [100, 110, 120, 130],
            'option_type': ['call', 'call', 'put', 'put'],
            'bid': [1.0, 2.0, 3.0, 4.0],
            'ask': [1.5, 2.5, 3.5, 4.5],
        })
        
        # Test filtering with min_dte only
        filtered_df = self.data_manager.filter_by_dte(df, min_dte=15)
        self.assertEqual(len(filtered_df), 2)  # Only the 30 and 60 day options
        
        # Test filtering with max_dte only
        filtered_df = self.data_manager.filter_by_dte(df, max_dte=15)
        self.assertEqual(len(filtered_df), 2)  # Only the 1 and 10 day options
        
        # Test filtering with both min_dte and max_dte
        filtered_df = self.data_manager.filter_by_dte(df, min_dte=5, max_dte=40)
        self.assertEqual(len(filtered_df), 2)  # Only the 10 and 30 day options


if __name__ == '__main__':
    unittest.main() 