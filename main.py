#!/usr/bin/env python3
"""
Options Visualizer - Main Entry Point

This script starts the Options Visualizer web application.
It runs both the backend API server, the frontend web server, and a dedicated cache management thread.
"""
import os
import sys
import logging
import threading
import time
import signal
import subprocess
import webbrowser
import queue
from options_visualizer_web.app import app as frontend_app
from options_visualizer_backend.app import app as backend_app
from python.options_data import OptionsDataManager

# Define fixed ports for consistency
BACKEND_PORT = 5002
FRONTEND_PORT = 5001

# Initialize the options data manager with a 10-minute cache duration
# This will be used by both the frontend and backend
options_data_manager = OptionsDataManager(cache_duration=600)

# Track server status
servers_ready = {
    'backend': False,
    'frontend': False,
    'cache_manager': False
}

# Create a thread-safe queue for cache operations
cache_queue = queue.Queue()

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

def run_cache_manager():
    """Run the cache management in a dedicated thread"""
    try:
        logger.info("Starting cache manager thread")
        
        # Set the flag to indicate the cache manager is ready
        servers_ready['cache_manager'] = True
        
        # Process cache operations from the queue
        while True:
            try:
                # Get the next operation from the queue with a timeout
                # This allows the thread to check for termination signals
                operation = cache_queue.get(timeout=1)
                
                if operation['type'] == 'refresh_ticker':
                    ticker = operation['ticker']
                    logger.info(f"Cache manager: Refreshing ticker {ticker}")
                    options_data_manager._refresh_ticker(ticker)
                elif operation['type'] == 'start_fetching':
                    ticker = operation['ticker']
                    skip_interpolation = operation.get('skip_interpolation', False)
                    logger.info(f"Cache manager: Fetching data for {ticker}")
                    options_data_manager.start_fetching(ticker, skip_interpolation)
                elif operation['type'] == 'refresh_all':
                    logger.info("Cache manager: Refreshing all tickers")
                    tickers = options_data_manager._cache.get_all_tickers()
                    for ticker in tickers:
                        options_data_manager._refresh_ticker(ticker)
                elif operation['type'] == 'shutdown':
                    logger.info("Cache manager: Shutting down")
                    break
                
                # Mark the task as done
                cache_queue.task_done()
            except queue.Empty:
                # No operations in the queue, continue waiting
                continue
            except Exception as e:
                logger.error(f"Error in cache manager: {str(e)}")
                # Mark the task as done even if it failed
                cache_queue.task_done()
    except Exception as e:
        logger.error(f"Error starting cache manager: {str(e)}")
        servers_ready['cache_manager'] = False

def run_backend():
    """Run the backend Flask app in a separate thread"""
    try:
        # Use the fixed backend port
        backend_port = BACKEND_PORT
        
        # Check if port is available, find another if not
        available_port = find_available_port(backend_port)
        if available_port is None:
            logger.error(f"Could not find an available port for backend server")
            servers_ready['backend'] = False
            return
            
        if available_port != backend_port:
            logger.warning(f"Port {backend_port} is in use, using port {available_port} for backend instead")
            backend_port = available_port
        
        # Set environment variables
        os.environ['BACKEND_PORT'] = str(backend_port)
        os.environ['BACKEND_URL'] = f"http://localhost:{backend_port}"
        
        # Override the cache operations in the backend to use the queue
        def queue_cache_operation(operation):
            cache_queue.put(operation)
        
        # Monkey patch the options_data_manager to use the queue for cache operations
        original_refresh_ticker = options_data_manager._refresh_ticker
        original_start_fetching = options_data_manager.start_fetching
        
        def patched_refresh_ticker(ticker):
            queue_cache_operation({'type': 'refresh_ticker', 'ticker': ticker})
            return True
        
        def patched_start_fetching(ticker, skip_interpolation=False):
            queue_cache_operation({'type': 'start_fetching', 'ticker': ticker, 'skip_interpolation': skip_interpolation})
            return True
        
        options_data_manager._refresh_ticker = patched_refresh_ticker
        options_data_manager.start_fetching = patched_start_fetching
        
        logger.info(f"Starting backend server on port {backend_port}")
        servers_ready['backend'] = True
        backend_app.run(debug=False, host='0.0.0.0', port=backend_port, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting backend server: {str(e)}")
        servers_ready['backend'] = False

def run_frontend():
    """Run the frontend Flask app in a separate thread"""
    try:
        # Use the fixed frontend port
        frontend_port = FRONTEND_PORT
        
        # Check if port is available, find another if not
        available_port = find_available_port(frontend_port)
        if available_port is None:
            logger.error(f"Could not find an available port for frontend server")
            servers_ready['frontend'] = False
            return
            
        if available_port != frontend_port:
            logger.warning(f"Port {frontend_port} is in use, using port {available_port} for frontend instead")
            frontend_port = available_port
        
        # Set environment variables
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
    
    while not (servers_ready['backend'] and servers_ready['frontend'] and servers_ready['cache_manager']):
        if time.time() - start_time > max_wait:
            logger.warning("Timeout waiting for servers to start")
            return
        time.sleep(0.5)
    
    # Both servers are ready, open browser
    frontend_port = os.environ.get('PORT', str(FRONTEND_PORT))
    url = f"http://localhost:{frontend_port}"
    logger.info(f"Opening browser at {url}")
    
    # Wait a bit more to ensure the server is fully initialized
    time.sleep(2)
    webbrowser.open(url)

if __name__ == '__main__':
    logger.info("Starting Options Visualizer from main.py")
    
    # Kill any existing Python processes running app.py to avoid port conflicts
    try:
        if sys.platform == 'win32':
            os.system('taskkill /f /im python.exe /fi "WINDOWTITLE eq *app.py*"')
        else:
            os.system('pkill -f "python.*app.py"')
        time.sleep(1)  # Give processes time to terminate
    except Exception as e:
        logger.warning(f"Error killing existing processes: {str(e)}")
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start cache manager in a separate thread
    cache_thread = threading.Thread(target=run_cache_manager, daemon=True)
    cache_thread.start()
    logger.info("Cache manager started in background thread")
    
    # Queue a refresh_all operation to refresh stale tickers on startup
    cache_queue.put({'type': 'refresh_all'})
    logger.info("Queued refresh_all operation to update stale tickers on startup")
    
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
        # Signal the cache manager to shut down
        cache_queue.put({'type': 'shutdown'})
        # Wait for the cache queue to be processed
        cache_queue.join()
        sys.exit(0) 