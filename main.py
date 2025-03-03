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
import time
import signal
import subprocess
from options_visualizer_web.app import app as frontend_app
from options_visualizer_backend.app import app as backend_app
from python.options_data import OptionsDataManager

# Initialize the options data manager with a 10-minute cache duration
# This will be used by both the frontend and backend
options_data_manager = OptionsDataManager(cache_duration=600)

# Track running processes
running_processes = []

def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals"""
    logger.info("Shutting down Options Visualizer...")
    # Clean up and exit
    sys.exit(0)

def run_backend():
    """Run the backend Flask app in a separate thread"""
    try:
        backend_port = int(os.environ.get('BACKEND_PORT', 5002))
        logger.info(f"Starting backend server on port {backend_port}")
        backend_app.run(debug=False, host='0.0.0.0', port=backend_port, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting backend server: {str(e)}")
        # Try to restart with a different port if port is in use
        if "Address already in use" in str(e):
            logger.info("Trying to restart backend with a different port")
            backend_port = backend_port + 10
            os.environ['BACKEND_PORT'] = str(backend_port)
            logger.info(f"Restarting backend server on port {backend_port}")
            backend_app.run(debug=False, host='0.0.0.0', port=backend_port, use_reloader=False)

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting Options Visualizer from main.py")
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    logger.info("Backend server started in background thread")
    
    # Get port from environment variable or use default for frontend
    frontend_port = int(os.environ.get('PORT', 5001))
    
    # Run the frontend app in the main thread
    try:
        logger.info(f"Starting frontend server on port {frontend_port}")
        frontend_app.run(debug=True, host='0.0.0.0', port=frontend_port, use_reloader=True)
    except Exception as e:
        logger.error(f"Error starting frontend server: {str(e)}")
        # Try to restart with a different port if port is in use
        if "Address already in use" in str(e):
            logger.info("Trying to restart frontend with a different port")
            frontend_port = frontend_port + 10
            os.environ['PORT'] = str(frontend_port)
            logger.info(f"Restarting frontend server on port {frontend_port}")
            frontend_app.run(debug=True, host='0.0.0.0', port=frontend_port, use_reloader=True) 