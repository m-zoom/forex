"""
Utilities package for Forex Chart Pattern Recognition System
Contains configuration, logging, and helper utilities
"""

__version__ = "1.0.0"
__author__ = "AI-Powered Trading Systems"

from .config import Config, config
from .logger import setup_logger, get_logger, ForexLogger
from .helpers import (
    save_results, load_results, backup_data, restore_backup,
    validate_data_file, format_number, format_percentage, 
    format_currency, get_system_info
)

__all__ = [
    'Config',
    'config',
    'setup_logger',
    'get_logger', 
    'ForexLogger',
    'save_results',
    'load_results',
    'backup_data',
    'restore_backup',
    'validate_data_file',
    'format_number',
    'format_percentage',
    'format_currency',
    'get_system_info'
]

