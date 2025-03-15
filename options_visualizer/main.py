#!/usr/bin/env python3
"""
Options Visualizer - Main Entry Point

This script starts the Options Visualizer web application.
It runs the combined backend and frontend web server.
"""
import os
import sys
import logging
import threading
import time
import signal
import subprocess
import webbrowser
from options_visualizer_backend.app import app as combined_app
from python.options_data import OptionsDataManager

# Initialize the options data manager with a 10-minute cache duration
# This will be used by the combined app
options_data_manager = OptionsDataManager(cache_duration=600)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals"""
    logger.info("Shutting down Options Visualizer...")
    # Clean up and exit
    sys.exit(0)

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    
    # If we get here, we couldn't find an available port
    logger.error(f"Could not find an available port after {max_attempts} attempts")
    return None

def run_server():
    """Run the combined Flask app"""
    try:
        server_port = int(os.environ.get('PORT', 5001))
        
        # Check if port is available, find another if not
        available_port = find_available_port(server_port)
        if available_port != server_port:
            logger.warning(f"Port {server_port} is in use, using port {available_port} instead")
            server_port = available_port
            os.environ['PORT'] = str(server_port)
        
        logger.info(f"Starting combined server on port {server_port}")
        combined_app.run(debug=False, host='0.0.0.0', port=server_port, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")

def open_browser():
    """Open the browser when the server is ready"""
    # Wait a bit for the server to start
    time.sleep(2)
    
    # Open browser
    server_port = os.environ.get('PORT', '5001')
    url = f"http://localhost:{server_port}"
    logger.info(f"Opening browser at {url}")
    webbrowser.open(url)

if __name__ == '__main__':
    logger.info("Starting Options Visualizer from main.py")
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    logger.info("Server started in background thread")
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        sys.exit(0) 