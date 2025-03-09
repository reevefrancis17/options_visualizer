"""
Backend Configuration

This module contains configuration settings for the Options Visualizer backend.
"""

import os
import logging
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# API settings
PORT = int(os.environ.get('PORT', 5002))
DEBUG = os.environ.get('DEBUG', 'True').lower() in ('true', '1', 't')
HOST = os.environ.get('HOST', '0.0.0.0')

# Cache settings
CACHE_DIR = os.environ.get('CACHE_DIR', str(BASE_DIR / 'cache'))
CACHE_DURATION = int(os.environ.get('CACHE_DURATION', 600))  # 10 minutes

# Data settings
DATA_DIR = os.environ.get('DATA_DIR', str(BASE_DIR / 'data'))

# Logging settings
LOG_DIR = os.environ.get('LOG_DIR', str(BASE_DIR / 'logs'))
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# Thread pool settings
MAX_WORKERS = min(32, int(os.environ.get('MAX_WORKERS', 0)) or (os.cpu_count() or 4) * 2)
REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', 30))  # seconds

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'server.log')),
        logging.StreamHandler()
    ]
) 