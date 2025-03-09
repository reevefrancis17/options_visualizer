#!/usr/bin/env python3
"""
Test script to verify fixes for the tuple unpacking and datetime parsing issues.
"""
import logging
import sys
import time
from python.options_data import OptionsDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def test_get_current_processor():
    """Test the get_current_processor method."""
    logger.info("Testing get_current_processor method")
    
    # Initialize the options data manager
    manager = OptionsDataManager()
    
    # Test with SPY
    ticker = "SPY"
    logger.info(f"Testing with ticker: {ticker}")
    
    # Get the current processor
    processor, price, status, progress = manager.get_current_processor(ticker)
    
    # Log the results
    logger.info(f"Status: {status}, Progress: {progress}")
    
    # If processor is None, start fetching
    if processor is None:
        logger.info(f"Processor is None, starting fetch for {ticker}")
        manager.start_fetching(ticker)
        
        # Wait for the fetch to complete
        for _ in range(10):
            time.sleep(1)
            processor, price, status, progress = manager.get_current_processor(ticker)
            logger.info(f"Status: {status}, Progress: {progress}")
            if processor is not None:
                break
    
    # Log the results
    if processor is not None:
        logger.info(f"Processor is not None, price: {price}")
        logger.info(f"Expirations: {processor.get_expirations()}")
    else:
        logger.info(f"Processor is still None after waiting")
    
    # Test with MSFT
    ticker = "MSFT"
    logger.info(f"Testing with ticker: {ticker}")
    
    # Get the current processor
    processor, price, status, progress = manager.get_current_processor(ticker)
    
    # Log the results
    logger.info(f"Status: {status}, Progress: {progress}")
    
    # If processor is None, start fetching
    if processor is None:
        logger.info(f"Processor is None, starting fetch for {ticker}")
        manager.start_fetching(ticker)
        
        # Wait for the fetch to complete
        for _ in range(10):
            time.sleep(1)
            processor, price, status, progress = manager.get_current_processor(ticker)
            logger.info(f"Status: {status}, Progress: {progress}")
            if processor is not None:
                break
    
    # Log the results
    if processor is not None:
        logger.info(f"Processor is not None, price: {price}")
        logger.info(f"Expirations: {processor.get_expirations()}")
    else:
        logger.info(f"Processor is still None after waiting")
    
    # Shutdown the manager
    logger.info("Shutting down the manager")
    manager.shutdown()

if __name__ == "__main__":
    test_get_current_processor()