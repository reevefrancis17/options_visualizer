#!/usr/bin/env python3
"""
Options Visualizer - Main Entry Point

This script starts the Options Visualizer web application.
It uses the Flask app defined in options_visualizer_web/app.py.
"""
import os
import sys
import logging
import threading
from options_visualizer_web.app import app as frontend_app
from options_visualizer_backend.app import app as backend_app
from python.options_data import OptionsDataManager

# Initialize the options data manager with a 10-minute cache duration
# This will be used by both the frontend and backend
options_data_manager = OptionsDataManager(cache_duration=600)

def run_backend():
    """Run the backend Flask app in a separate thread"""
    backend_port = int(os.environ.get('BACKEND_PORT', 5002))
    backend_app.run(debug=False, host='0.0.0.0', port=backend_port, use_reloader=False)

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting Options Visualizer from main.py")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    logger.info("Backend server started in background thread")
    
    # Get port from environment variable or use default for frontend
    frontend_port = int(os.environ.get('PORT', 5001))
    
    # Run the frontend app in the main thread
    logger.info(f"Starting frontend server on port {frontend_port}")
    frontend_app.run(debug=True, host='0.0.0.0', port=frontend_port) 