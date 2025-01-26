import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

class LoggerSetup:
    def __init__(self, debug_dir='debug'):
        self.debug_dir = debug_dir
        self._ensure_debug_dir()
        self.logger = self._setup_logger()

    def _ensure_debug_dir(self):
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

    def _setup_logger(self):
        logger = logging.getLogger('OptionsVisualizer')
        if not logger.handlers:
            log_file = os.path.join(self.debug_dir, 'error_log.txt')
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1024*1024, backupCount=5)
            
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)
        
        return logger 