"""
Configuration management for Forex Chart Pattern Recognition System
"""

import os
import configparser
from typing import Dict, Any, Optional

class Config:
    """Configuration manager for the forex pattern recognition system"""
    
    def __init__(self, config_file="config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file and environment variables"""
        # Load default configuration
        self.config.read_dict(self.get_default_config())
        
        # Load from file if it exists
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        
        # Override with environment variables
        self.load_from_environment()
    
    def get_default_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default configuration values"""
        return {
            'API': {
                'financial_datasets_api_key': 'demo',
                'request_timeout': '30',
                'rate_limit_delay': '1',
                'max_retries': '3'
            },
            'PATTERNS': {
                'head_shoulders': 'true',
                'double_patterns': 'true',
                'triangles': 'true',
                'support_resistance': 'true',
                'sensitivity': '0.015',
                'min_confidence': '0.6',
                'auto_detect_patterns': 'true'
            },
            'DISPLAY': {
                'chart_type': 'candlestick',
                'show_volume': 'true',
                'show_moving_averages': 'true',
                'show_bollinger_bands': 'false',
                'chart_theme': 'default'
            },
            'REALTIME': {
                'default_interval': '60',
                'real_time_alerts': 'true',
                'sound_alerts': 'false',
                'popup_notifications': 'true',
                'max_alert_frequency': '300'
            },
            'MODELS': {
                'model_type': 'hybrid',
                'training_epochs': '100',
                'batch_size': '32',
                'sequence_length': '60',
                'validation_split': '0.2',
                'auto_save_models': 'true'
            },
            'LOGGING': {
                'log_level': 'INFO',
                'log_file': 'logs/forex_pattern_recognition.log',
                'max_log_size': '10485760',
                'backup_count': '5',
                'console_logging': 'true'
            },
            'DATA': {
                'default_symbol': 'AAPL',
                'default_timeframe': '5min',
                'cache_data': 'true',
                'cache_duration': '300',
                'data_validation': 'true'
            },
            'UI': {
                'window_width': '1400',
                'window_height': '900',
                'theme': 'default',
                'font_size': '10',
                'auto_refresh_interval': '5',
                'show_tooltips': 'true'
            }
        }
    
    def load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'FINANCIAL_DATASETS_API_KEY': ('API', 'financial_datasets_api_key'),
            'LOG_LEVEL': ('LOGGING', 'log_level'),
            'DEFAULT_SYMBOL': ('DATA', 'default_symbol'),
            'CHART_TYPE': ('DISPLAY', 'chart_type')
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self.config.set(section, key, value)
    
    def get(self, section: str, key: str, fallback: Any = None) -> str:
        """Get configuration value"""
        return self.config.get(section, key, fallback=fallback)
    
    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        """Get configuration value as integer"""
        return self.config.getint(section, key, fallback=fallback)
    
    def getfloat(self, section: str, key: str, fallback: float = 0.0) -> float:
        """Get configuration value as float"""
        return self.config.getfloat(section, key, fallback=fallback)
    
    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get configuration value as boolean"""
        return self.config.getboolean(section, key, fallback=fallback)
    
    def set(self, section: str, key: str, value: str):
        """Set configuration value"""
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, str(value))
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return {
            'api_key': self.get('API', 'financial_datasets_api_key'),
            'timeout': self.getint('API', 'request_timeout'),
            'rate_limit_delay': self.getint('API', 'rate_limit_delay'),
            'max_retries': self.getint('API', 'max_retries')
        }
    
    def get_pattern_config(self) -> Dict[str, Any]:
        """Get pattern detection configuration"""
        return {
            'head_shoulders': self.getboolean('PATTERNS', 'head_shoulders'),
            'double_patterns': self.getboolean('PATTERNS', 'double_patterns'),
            'triangles': self.getboolean('PATTERNS', 'triangles'),
            'support_resistance': self.getboolean('PATTERNS', 'support_resistance'),
            'sensitivity': self.getfloat('PATTERNS', 'sensitivity'),
            'min_confidence': self.getfloat('PATTERNS', 'min_confidence'),
            'auto_detect': self.getboolean('PATTERNS', 'auto_detect_patterns')
        }
    
    def get_display_config(self) -> Dict[str, Any]:
        """Get display configuration"""
        return {
            'chart_type': self.get('DISPLAY', 'chart_type'),
            'show_volume': self.getboolean('DISPLAY', 'show_volume'),
            'show_moving_averages': self.getboolean('DISPLAY', 'show_moving_averages'),
            'show_bollinger_bands': self.getboolean('DISPLAY', 'show_bollinger_bands'),
            'chart_theme': self.get('DISPLAY', 'chart_theme')
        }

# Global configuration instance
config = Config()