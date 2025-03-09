#!/usr/bin/env python3
"""
Options Visualizer - Main Entry Point

This script starts the Options Visualizer web application.
It runs both the backend API server and the frontend web server.
"""
import os
import sys
import logging
import threading
import time
import signal
import subprocess
import webbrowser
import atexit
from options_visualizer_web.app import app as frontend_app
from options_visualizer_backend.app import app as backend_app
from options_visualizer_backend.options_data import OptionsDataManager
from options_visualizer_backend.config import MAX_WORKERS, CACHE_DURATION
from options_visualizer_web.config import PORT as FRONTEND_PORT
from options_visualizer_backend.config import PORT as BACKEND_PORT

# Initialize the options data manager
options_data_manager = OptionsDataManager(cache_duration=CACHE_DURATION, max_workers=MAX_WORKERS)
logger = logging.getLogger(__name__)

# Track server status
servers_ready = {
    'backend': False,
    'frontend': False
}

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
    cleanup()
    sys.exit(0)

def cleanup():
    """Cleanup function to shutdown thread pools and other resources."""
    logger.info("Running cleanup...")
    
    # Shutdown the options data manager
    if options_data_manager is not None:
        logger.info("Shutting down options data manager...")
        if hasattr(options_data_manager, 'shutdown'):
            options_data_manager.shutdown()
    
    logger.info("Cleanup complete")

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

def run_backend():
    """Run the backend API server."""
    try:
        logger.info("Starting backend API server...")
        # Set the data manager as a global variable for the backend
        backend_app.config['OPTIONS_DATA_MANAGER'] = options_data_manager
        
        # Find an available port starting from the configured port
        port = find_available_port(BACKEND_PORT)
        if port != BACKEND_PORT:
            logger.warning(f"Port {BACKEND_PORT} is in use, using port {port} instead")
        
        # Start the backend server
        backend_thread = threading.Thread(
            target=backend_app.run,
            kwargs={'host': '0.0.0.0', 'port': port, 'debug': False, 'use_reloader': False},
            daemon=True
        )
        backend_thread.start()
        logger.info(f"Backend API server started on port {port}")
        servers_ready['backend'] = True
        return port
    except Exception as e:
        logger.error(f"Failed to start backend API server: {e}")
        return None

def run_frontend():
    """Run the frontend web server."""
    try:
        logger.info("Starting frontend web server...")
        # Set the data manager as a global variable for the frontend
        frontend_app.config['OPTIONS_DATA_MANAGER'] = options_data_manager
        
        # Find an available port starting from the configured port
        port = find_available_port(FRONTEND_PORT)
        if port != FRONTEND_PORT:
            logger.warning(f"Port {FRONTEND_PORT} is in use, using port {port} instead")
        
        # Start the frontend server
        frontend_thread = threading.Thread(
            target=frontend_app.run,
            kwargs={'host': '0.0.0.0', 'port': port, 'debug': False, 'use_reloader': False},
            daemon=True
        )
        frontend_thread.start()
        logger.info(f"Frontend web server started on port {port}")
        servers_ready['frontend'] = True
        return port
    except Exception as e:
        logger.error(f"Failed to start frontend web server: {e}")
        return None

def open_browser():
    """Open the web browser to the frontend URL."""
    # Wait a moment for the servers to start
    time.sleep(2)
    
    # Open the browser
    url = f"http://localhost:{FRONTEND_PORT}"
    logger.info(f"Opening browser at {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        logger.error(f"Failed to open browser: {e}")

def main():
    """Main entry point for the application."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function
    atexit.register(cleanup)
    
    logger.info("Starting Options Visualizer...")
    
    # Start the backend server in a separate thread
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Start the frontend server in a separate thread
    frontend_thread = threading.Thread(target=run_frontend)
    frontend_thread.daemon = True
    frontend_thread.start()
    
    # Wait for servers to start
    timeout = 10  # seconds
    start_time = time.time()
    while not (servers_ready['backend'] and servers_ready['frontend']):
        if time.time() - start_time > timeout:
            logger.error("Timeout waiting for servers to start")
            return 1
        time.sleep(0.1)
    
    # Open browser after a short delay
    browser_thread = threading.Thread(target=lambda: open_browser())
    browser_thread.daemon = True
    browser_thread.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 