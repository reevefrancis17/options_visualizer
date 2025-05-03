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
from options_visualizer_web.app import app as frontend_app
from backend.app import app as backend_app
from python.options_data import OptionsDataManager
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Initialize the options data manager with a 10-minute cache duration
# This will be used by both the frontend and backend
options_data_manager = OptionsDataManager(cache_duration=600)

# Track server status
servers_ready = {
    'backend': False,
    'frontend': False
}

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app
from backend.app import app

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

def run_backend():
    """Run the backend Flask app in a separate thread"""
    try:
        backend_port = int(os.environ.get('BACKEND_PORT', 5002))
        
        # Check if port is available, find another if not
        available_port = find_available_port(backend_port)
        if available_port != backend_port:
            logger.warning(f"Port {backend_port} is in use, using port {available_port} for backend instead")
            backend_port = available_port
            os.environ['BACKEND_PORT'] = str(backend_port)
            
            # Update the frontend config if needed
            if 'BACKEND_URL' not in os.environ:
                os.environ['BACKEND_URL'] = f"http://localhost:{backend_port}"
        
        logger.info(f"Starting backend server on port {backend_port}")
        servers_ready['backend'] = True
        backend_app.run(debug=False, host='0.0.0.0', port=backend_port, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting backend server: {str(e)}")
        servers_ready['backend'] = False

def run_frontend():
    """Run the frontend Flask app in a separate thread"""
    try:
        frontend_port = int(os.environ.get('PORT', 5001))
        
        # Check if port is available, find another if not
        available_port = find_available_port(frontend_port)
        if available_port != frontend_port:
            logger.warning(f"Port {frontend_port} is in use, using port {available_port} for frontend instead")
            frontend_port = available_port
            os.environ['PORT'] = str(frontend_port)
        
        logger.info(f"Starting frontend server on port {frontend_port}")
        servers_ready['frontend'] = True
        frontend_app.run(debug=False, host='0.0.0.0', port=frontend_port, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting frontend server: {str(e)}")
        servers_ready['frontend'] = False

def open_browser():
    """Open the browser when both servers are ready"""
    # Wait for both servers to be ready
    max_wait = 30  # Maximum wait time in seconds
    start_time = time.time()
    
    while not (servers_ready['backend'] and servers_ready['frontend']):
        if time.time() - start_time > max_wait:
            logger.warning("Timeout waiting for servers to start")
            return
        time.sleep(0.5)
    
    # Both servers are ready, open browser
    frontend_port = os.environ.get('PORT', '5001')
    url = f"http://localhost:{frontend_port}"
    logger.info(f"Opening browser at {url}")
    
    # Wait a bit more to ensure the server is fully initialized
    time.sleep(2)
    webbrowser.open(url)

if __name__ == '__main__':
    # Configure logging for uvicorn
    os.makedirs('logs', exist_ok=True)
    log_file = 'logs/uvicorn.log'
    
    # Clear log file on startup
    if os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write(f"Log file cleared on uvicorn startup: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configure rotating file handler (100KB max size, keep 3 backup files)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=100*1024,  # 100KB
        backupCount=3
    )
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Configure root logger
    logging.basicConfig(
        level=logging.WARNING,
        handlers=[file_handler, console_handler]
    )
    
    logger = logging.getLogger(__name__)
    logger.warning("Starting uvicorn server with logging level: WARNING")
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    logger.info("Backend server started in background thread")
    
    # Start frontend in a separate thread
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)
    frontend_thread.start()
    logger.info("Frontend server started in background thread")
    
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