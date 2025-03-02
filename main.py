#!/usr/bin/env python3
"""
Options Visualizer - Main Entry Point

This script starts the Options Visualizer web application.
It uses the Flask app defined in options_visualizer_web/app.py.
"""
import os
import sys
import logging
from options_visualizer_web.app import app

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting Options Visualizer from main.py")
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5001))
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=port) 