"""
Frontend Configuration

This module contains configuration settings for the Options Visualizer frontend.
"""

import os
import logging
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Web server settings
PORT = int(os.environ.get('PORT', 5001))
DEBUG = os.environ.get('DEBUG', 'True').lower() in ('true', '1', 't')
HOST = os.environ.get('HOST', '0.0.0.0')

# Backend API settings
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:5002')

# Logging settings
LOG_DIR = os.environ.get('LOG_DIR', str(PROJECT_DIR / 'debug' / 'logs'))
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'web_app.log')),
        logging.StreamHandler()
    ]
) 