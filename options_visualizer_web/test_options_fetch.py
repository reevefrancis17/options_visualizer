#!/usr/bin/env python3
"""
Simple test script to verify options data fetching functionality.
This script focuses only on testing if we can fetch and process options data.
"""
import os
import sys
import logging
from datetime import datetime

# Configure logging
log_dir = 'debug'
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

def test_fetch_options(symbol):
    """Test fetching options data for a given symbol."""
    logger.info(f"=== Testing options data fetch for {symbol} ===")
    
    try:
        logger.info("Initializing OptionsDataManager")
        data_manager = OptionsDataManager(cache_duration=60)
        logger.info("OptionsDataManager initialized successfully")
        
        logger.info(f"Fetching options data for {symbol}")
        processor, current_price = data_manager.get_options_data(symbol)
        
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
        
        logger.info(f"=== Successfully completed test for {symbol} ===")
        return True
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function."""
    symbols_to_test = ['SPY', 'AAPL']
    
    for symbol in symbols_to_test:
        success = test_fetch_options(symbol)
        if success:
            logger.info(f"✅ Test PASSED for {symbol}")
        else:
            logger.error(f"❌ Test FAILED for {symbol}")

if __name__ == "__main__":
    logger.info("Starting main test function")
    main()
    logger.info("Test completed") 