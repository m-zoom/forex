"""
Logging utilities for Forex Chart Pattern Recognition System
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

class ForexLogger:
    """Enhanced logger for forex pattern recognition system"""
    
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create logs directory if it doesn't exist
        log_file = self.config.get('log_file', 'logs/forex_pattern_recognition.log')
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # File handler with rotation
        if log_file:
            max_bytes = int(self.config.get('max_log_size', 10485760))
            backup_count = int(self.config.get('backup_count', 5))
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Console handler
        if self.config.get('console_logging', 'true').lower() == 'true':
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message"""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message"""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message"""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message"""
        self.logger.critical(message, *args, **kwargs)
    
    def log_api_request(self, method: str, url: str, status_code: int, response_time: float):
        """Log API request details"""
        self.info(f"API {method} {url} - Status: {status_code} - Time: {response_time:.2f}s")
    
    def log_pattern_detection(self, pattern_type: str, count: int, confidence: float):
        """Log pattern detection results"""
        self.info(f"Pattern Detection - {pattern_type}: {count} patterns found with avg confidence: {confidence:.2f}")
    
    def log_data_processing(self, operation: str, records: int, duration: float):
        """Log data processing operations"""
        self.info(f"Data Processing - {operation}: {records} records in {duration:.2f}s")
    
    def log_error_with_context(self, error: Exception, context: str):
        """Log error with additional context"""
        self.error(f"{context} - {type(error).__name__}: {str(error)}")

def setup_logger(name: str, config: Optional[dict] = None) -> ForexLogger:
    """Setup a logger with the given configuration"""
    if config is None:
        from .config import config as default_config
        config = {
            'log_level': default_config.get('LOGGING', 'log_level'),
            'log_file': default_config.get('LOGGING', 'log_file'),
            'max_log_size': default_config.get('LOGGING', 'max_log_size'),
            'backup_count': default_config.get('LOGGING', 'backup_count'),
            'console_logging': default_config.get('LOGGING', 'console_logging')
        }
    
    return ForexLogger(name, config)

def get_logger(name: str) -> ForexLogger:
    """Get or create a logger with the given name"""
    return setup_logger(name)