#!/usr/bin/env python3
"""
Options Visualizer - Main Entry Point for Gunicorn

This script serves as the entry point for Gunicorn to run the Options Visualizer web application.
It imports the combined Flask app from the options_visualizer_backend module.
"""
import os
import logging
from options_visualizer_backend.app import app
from python.options_data import OptionsDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the options data manager with a 10-minute cache duration
# This will be used by the combined app
options_data_manager = OptionsDataManager(cache_duration=600)

# Log startup information
logger.info("Starting Options Visualizer with Gunicorn")
logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")
logger.info(f"Debug mode: {os.environ.get('FLASK_DEBUG', 'False')}")

# This is the application that Gunicorn will use
# The app is already configured in options_visualizer_backend/app.py
if __name__ == '__main__':
    # This block will only execute when running directly with Python
    # When running with Gunicorn, only the app object is used
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=False, host='0.0.0.0', port=port) 