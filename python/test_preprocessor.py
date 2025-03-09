#!/usr/bin/env python3
"""
Test script for the options preprocessor module.
"""

import logging
import argparse
import json
import numpy as np
from datetime import datetime
from python.options_preprocessor import OptionsPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy arrays and NaN values
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to Python native types.
    """
    if isinstance(obj, dict):
        return {(int(k) if isinstance(k, np.integer) else k): convert_numpy_types(v) 
                for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj

def main():
    """Main function to test the options preprocessor."""
    parser = argparse.ArgumentParser(description='Test the options preprocessor.')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--max-dates', type=int, default=None, help='Maximum number of expiration dates to fetch')
    parser.add_argument('--output', type=str, default=None, help='Output file path for JSON results')
    args = parser.parse_args()
    
    logger.info(f"Testing options preprocessor for ticker: {args.ticker}")
    
    try:
        # Create preprocessor
        preprocessor = OptionsPreprocessor()
        
        # Fetch and preprocess data
        start_time = datetime.now()
        result = preprocessor.fetch_and_preprocess(args.ticker, args.max_dates)
        end_time = datetime.now()
        
        # Print summary
        logger.info(f"Ticker: {result['ticker']}")
        logger.info(f"Current Price: {result['current_price']}")
        logger.info(f"Timestamp: {result['timestamp']}")
        logger.info(f"Calls: {len(result['calls']['strikes'])} strikes × {len(result['calls']['dtes'])} expirations")
        logger.info(f"Puts: {len(result['puts']['strikes'])} strikes × {len(result['puts']['dtes'])} expirations")
        logger.info(f"Processing time: {(end_time - start_time).total_seconds():.2f} seconds")
        
        # Print sample of the grid data
        if result['calls']['grid'].size > 0:
            logger.info("Sample of calls grid data:")
            grid = result['calls']['grid']
            sample_size = min(5, grid.shape[0])
            sample_grid = grid[:sample_size, :min(5, grid.shape[1])]
            for i in range(sample_size):
                logger.info(f"Strike {result['calls']['strikes'][i]}: {sample_grid[i]}")
        
        # Save to file if output path is provided
        if args.output:
            # Convert NumPy types to Python native types before serialization
            converted_result = convert_numpy_types(result)
            with open(args.output, 'w') as f:
                json.dump(converted_result, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 