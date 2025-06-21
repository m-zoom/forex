"""
Data package for Forex Chart Pattern Recognition System
Contains data fetching, processing, and validation utilities
"""

__version__ = "1.0.0"
__author__ = "AI-Powered Trading Systems"

from .forex_api import ForexAPI
from .data_processor import DataProcessor

__all__ = [
    'ForexAPI',
    'DataProcessor'
]

