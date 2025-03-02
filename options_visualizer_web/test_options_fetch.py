#!/usr/bin/env python3
"""
Simple test script to verify options data fetching functionality.
This script focuses only on testing if we can fetch and process options data.
"""
import os
import sys
import logging
import numpy as np
from datetime import datetime

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug', 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Clear logs on startup
log_file = os.path.join(log_dir, 'test_fetch.log')
with open(log_file, 'w') as f:
    f.write(f"=== New test session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,  # Use DEBUG level for more detailed logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add a console handler to see logs in real-time
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger('test_fetch')
logger.info("Starting Options Data Fetch Test")

# Fix the path modification
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger.info(f"Added to path: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

try:
    logger.info("Importing OptionsDataManager and OptionsDataProcessor")
    from python.options_data import OptionsDataManager, OptionsDataProcessor
    logger.info("Successfully imported options_data modules")
except Exception as e:
    logger.error(f"Failed to import options_data modules: {str(e)}")
    sys.exit(1)

def test_fetch_options(symbol, max_dates=None):
    """Test fetching options data for a given symbol."""
    logger.info(f"=== Testing options data fetch for {symbol} with max_dates={max_dates} ===")
    
    try:
        logger.info("Initializing OptionsDataManager")
        data_manager = OptionsDataManager(cache_duration=60)
        logger.info("OptionsDataManager initialized successfully")
        
        logger.info(f"Fetching options data for {symbol}")
        processor, current_price = data_manager.get_options_data(symbol, max_dates=max_dates)
        
        if processor is None or current_price is None:
            logger.error(f"Failed to fetch data for {symbol}")
            return False
        
        logger.info(f"Successfully fetched data for {symbol} with current price: {current_price}")
        
        # Get basic information
        logger.info("Getting expiration dates")
        expiry_dates = processor.get_expirations()
        logger.info(f"Found {len(expiry_dates)} expiration dates: {expiry_dates}")
        
        logger.info("Getting strike range")
        min_strike, max_strike = processor.get_strike_range()
        logger.info(f"Strike range: {min_strike} to {max_strike}")
        
        # Get the full dataset
        logger.info("Getting full dataset as DataFrame")
        df = processor.get_data_frame()
        if df is None or df.empty:
            logger.error("DataFrame is empty or None")
            return False
        
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        
        # Check for key columns
        required_columns = ['strike', 'expiration', 'option_type', 'price', 'bid', 'ask']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        logger.info("All required columns are present")
        
        # Check for NaN values in key columns
        nan_counts = df[required_columns].isna().sum()
        logger.info(f"NaN counts in key columns: {nan_counts.to_dict()}")
        
        # Check for data in each option type
        call_count = df[df['option_type'] == 'call'].shape[0]
        put_count = df[df['option_type'] == 'put'].shape[0]
        logger.info(f"Found {call_count} call options and {put_count} put options")
        
        # Check for computed columns
        computed_columns = ['delta', 'gamma', 'theta', 'intrinsic_value', 'extrinsic_value']
        for col in computed_columns:
            if col in df.columns:
                logger.info(f"Column {col} is present with {df[col].notna().sum()} non-NaN values")
            else:
                logger.warning(f"Computed column {col} is missing")
        
        # Check for minimum values in spot prices (which would indicate interpolation failure)
        min_value = 0.05
        spot_values = df['spot'].values
        min_count = np.sum(np.isclose(spot_values, min_value))
        total_count = len(spot_values)
        min_percentage = (min_count / total_count) * 100 if total_count > 0 else 0
        
        logger.info(f"Spot price check: {min_count}/{total_count} values ({min_percentage:.2f}%) are at minimum value {min_value}")
        
        # If more than 50% of values are at minimum, interpolation likely failed
        if min_percentage > 50:
            logger.error("Interpolation verification failed: Too many minimum values")
            return False
        
        # Check for interpolation success flag
        if hasattr(processor, 'interpolation_successful'):
            logger.info(f"Interpolation success flag: {processor.interpolation_successful}")
            if not processor.interpolation_successful and len(expiry_dates) > 1:
                logger.error("Interpolation was not successful despite having multiple expiry dates")
                return False
        
        # Test getting data for a specific expiry date
        if expiry_dates:
            logger.info(f"Testing get_data_for_expiry with date {expiry_dates[0]}")
            expiry_df = processor.get_data_for_expiry(expiry_dates[0])
            if expiry_df is None or expiry_df.empty:
                logger.error(f"Failed to get data for expiry date {expiry_dates[0]}")
                return False
            logger.info(f"Successfully got data for expiry date {expiry_dates[0]}, shape: {expiry_df.shape}")
        
        logger.info(f"=== Successfully completed test for {symbol} ===")
        return True
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_progressive_loading(symbol):
    """Test progressive loading of expiry dates and interpolation."""
    logger.info(f"=== Testing progressive loading for {symbol} ===")
    
    # Test with 1 date
    logger.info("Testing with 1 expiry date")
    success_1 = test_fetch_options(symbol, max_dates=1)
    
    # Test with 2 dates
    logger.info("Testing with 2 expiry dates")
    success_2 = test_fetch_options(symbol, max_dates=2)
    
    # Test with 3 dates
    logger.info("Testing with 3 expiry dates")
    success_3 = test_fetch_options(symbol, max_dates=3)
    
    # Test with all dates
    logger.info("Testing with all expiry dates")
    success_all = test_fetch_options(symbol)
    
    # Report results
    logger.info(f"Progressive loading test results for {symbol}:")
    logger.info(f"  1 date: {'✅ PASS' if success_1 else '❌ FAIL'}")
    logger.info(f"  2 dates: {'✅ PASS' if success_2 else '❌ FAIL'}")
    logger.info(f"  3 dates: {'✅ PASS' if success_3 else '❌ FAIL'}")
    logger.info(f"  All dates: {'✅ PASS' if success_all else '❌ FAIL'}")
    
    return success_1 and success_2 and success_3 and success_all

def main():
    """Main test function."""
    symbols_to_test = ['SPY', 'AAPL']
    
    # Test basic functionality
    for symbol in symbols_to_test:
        success = test_fetch_options(symbol)
        if success:
            logger.info(f"✅ Basic test PASSED for {symbol}")
        else:
            logger.error(f"❌ Basic test FAILED for {symbol}")
    
    # Test progressive loading
    for symbol in symbols_to_test:
        success = test_progressive_loading(symbol)
        if success:
            logger.info(f"✅ Progressive loading test PASSED for {symbol}")
        else:
            logger.error(f"❌ Progressive loading test FAILED for {symbol}")

if __name__ == "__main__":
    logger.info("Starting main test function")
    main()
    logger.info("Test completed") 