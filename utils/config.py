"""
Configuration management for the Forex Pattern Recognition System
"""

import os
import configparser
import json
from pathlib import Path

class Config:
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.default_config = self._get_default_config()
        
        # Create config directory if it doesn't exist
        config_path = Path(config_file).parent
        config_path.mkdir(parents=True, exist_ok=True)
        
        self.load()
        
    def _get_default_config(self):
        """Get default configuration values"""
        return {
            'API': {
                'alpha_vantage_api_key': os.getenv('ALPHA_VANTAGE_API_KEY', 'demo'),
                'request_timeout': '30',
                'rate_limit_delay': '12',
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
                'max_log_size': '10485760',  # 10MB
                'backup_count': '5',
                'console_logging': 'true'
            },
            'DATA': {
                'default_symbol': 'EUR/USD',
                'default_timeframe': '5min',
                'default_outputsize': 'compact',
                'cache_data': 'true',
                'cache_duration': '300',  # 5 minutes
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
    
    def load(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                self.config.read(self.config_file)
                
                # Verify all required sections exist
                for section, options in self.default_config.items():
                    if not self.config.has_section(section):
                        self.config.add_section(section)
                    
                    for option, default_value in options.items():
                        if not self.config.has_option(section, option):
                            self.config.set(section, option, default_value)
            else:
                # Create default configuration
                self._create_default_config()
                
            # Save any updates
            self.save()
            
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration file"""
        try:
            for section, options in self.default_config.items():
                self.config.add_section(section)
                for option, value in options.items():
                    self.config.set(section, option, value)
            
            self.save()
            
        except Exception as e:
            print(f"Error creating default configuration: {str(e)}")
    
    def save(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                self.config.write(f)
                
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")
    
    def get(self, key, section=None, fallback=None):
        """Get configuration value"""
        try:
            if section:
                return self.config.get(section, key, fallback=fallback)
            else:
                # Search all sections for the key
                for section_name in self.config.sections():
                    if self.config.has_option(section_name, key):
                        return self.config.get(section_name, key)
                
                return fallback
                
        except Exception:
            return fallback
    
    def get_int(self, key, section=None, fallback=0):
        """Get integer configuration value"""
        try:
            value = self.get(key, section, str(fallback))
            return int(value)
        except (ValueError, TypeError):
            return fallback
    
    def get_float(self, key, section=None, fallback=0.0):
        """Get float configuration value"""
        try:
            value = self.get(key, section, str(fallback))
            return float(value)
        except (ValueError, TypeError):
            return fallback
    
    def get_bool(self, key, section=None, fallback=False):
        """Get boolean configuration value"""
        try:
            value = self.get(key, section, str(fallback))
            return value.lower() in ('true', '1', 'yes', 'on')
        except (AttributeError, TypeError):
            return fallback
    
    def set(self, key, value, section=None):
        """Set configuration value"""
        try:
            if section:
                if not self.config.has_section(section):
                    self.config.add_section(section)
                self.config.set(section, key, str(value))
            else:
                # Find the section that contains this key
                section_found = False
                for section_name in self.config.sections():
                    if self.config.has_option(section_name, key):
                        self.config.set(section_name, key, str(value))
                        section_found = True
                        break
                
                if not section_found:
                    # Add to GENERAL section if not found
                    if not self.config.has_section('GENERAL'):
                        self.config.add_section('GENERAL')
                    self.config.set('GENERAL', key, str(value))
                    
        except Exception as e:
            print(f"Error setting configuration value: {str(e)}")
    
    def update(self, updates_dict):
        """Update multiple configuration values"""
        try:
            for key, value in updates_dict.items():
                self.set(key, value)
            self.save()
            
        except Exception as e:
            print(f"Error updating configuration: {str(e)}")
    
    def get_section(self, section):
        """Get all options from a section as dictionary"""
        try:
            if self.config.has_section(section):
                return dict(self.config.items(section))
            else:
                return {}
                
        except Exception as e:
            print(f"Error getting section {section}: {str(e)}")
            return {}
    
    def get_api_config(self):
        """Get API configuration"""
        return {
            'api_key': self.get('alpha_vantage_api_key', 'API'),
            'timeout': self.get_int('request_timeout', 'API', 30),
            'rate_limit_delay': self.get_int('rate_limit_delay', 'API', 12),
            'max_retries': self.get_int('max_retries', 'API', 3)
        }
    
    def get_pattern_config(self):
        """Get pattern detection configuration"""
        return {
            'head_shoulders': self.get_bool('head_shoulders', 'PATTERNS', True),
            'double_patterns': self.get_bool('double_patterns', 'PATTERNS', True),
            'triangles': self.get_bool('triangles', 'PATTERNS', True),
            'support_resistance': self.get_bool('support_resistance', 'PATTERNS', True),
            'sensitivity': self.get_float('sensitivity', 'PATTERNS', 0.015),
            'min_confidence': self.get_float('min_confidence', 'PATTERNS', 0.6),
            'auto_detect': self.get_bool('auto_detect_patterns', 'PATTERNS', True)
        }
    
    def get_display_config(self):
        """Get display configuration"""
        return {
            'chart_type': self.get('chart_type', 'DISPLAY', 'candlestick'),
            'show_volume': self.get_bool('show_volume', 'DISPLAY', True),
            'show_ma': self.get_bool('show_moving_averages', 'DISPLAY', True),
            'show_bb': self.get_bool('show_bollinger_bands', 'DISPLAY', False),
            'theme': self.get('chart_theme', 'DISPLAY', 'default')
        }
    
    def get_realtime_config(self):
        """Get real-time configuration"""
        return {
            'default_interval': self.get_int('default_interval', 'REALTIME', 60),
            'real_time_alerts': self.get_bool('real_time_alerts', 'REALTIME', True),
            'sound_alerts': self.get_bool('sound_alerts', 'REALTIME', False),
            'popup_notifications': self.get_bool('popup_notifications', 'REALTIME', True),
            'max_alert_frequency': self.get_int('max_alert_frequency', 'REALTIME', 300)
        }
    
    def get_model_config(self):
        """Get model configuration"""
        return {
            'model_type': self.get('model_type', 'MODELS', 'hybrid'),
            'epochs': self.get_int('training_epochs', 'MODELS', 100),
            'batch_size': self.get_int('batch_size', 'MODELS', 32),
            'sequence_length': self.get_int('sequence_length', 'MODELS', 60),
            'validation_split': self.get_float('validation_split', 'MODELS', 0.2),
            'auto_save': self.get_bool('auto_save_models', 'MODELS', True)
        }
    
    def export_config(self, filename):
        """Export configuration to JSON file"""
        try:
            config_dict = {}
            for section in self.config.sections():
                config_dict[section] = dict(self.config.items(section))
            
            with open(filename, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"Error exporting configuration: {str(e)}")
            return False
    
    def import_config(self, filename):
        """Import configuration from JSON file"""
        try:
            with open(filename, 'r') as f:
                config_dict = json.load(f)
            
            # Clear existing configuration
            for section in self.config.sections():
                self.config.remove_section(section)
            
            # Load new configuration
            for section, options in config_dict.items():
                self.config.add_section(section)
                for option, value in options.items():
                    self.config.set(section, option, str(value))
            
            self.save()
            return True
            
        except Exception as e:
            print(f"Error importing configuration: {str(e)}")
            return False
    
    def reset_to_defaults(self):
        """Reset configuration to default values"""
        try:
            # Remove existing config file
            if os.path.exists(self.config_file):
                os.remove(self.config_file)
            
            # Clear in-memory config
            for section in self.config.sections():
                self.config.remove_section(section)
            
            # Recreate default config
            self._create_default_config()
            
            return True
            
        except Exception as e:
            print(f"Error resetting configuration: {str(e)}")
            return False
    
    def validate_config(self):
        """Validate configuration values"""
        issues = []
        
        try:
            # Validate API key
            api_key = self.get('alpha_vantage_api_key', 'API')
            if not api_key or api_key == 'demo':
                issues.append("Alpha Vantage API key not configured")
            
            # Validate numeric values
            numeric_validations = [
                ('sensitivity', 'PATTERNS', 0.001, 0.1),
                ('min_confidence', 'PATTERNS', 0.1, 1.0),
                ('default_interval', 'REALTIME', 10, 3600),
                ('batch_size', 'MODELS', 1, 512),
                ('sequence_length', 'MODELS', 10, 1000)
            ]
            
            for key, section, min_val, max_val in numeric_validations:
                value = self.get_float(key, section)
                if not min_val <= value <= max_val:
                    issues.append(f"{key} value {value} is out of range [{min_val}, {max_val}]")
            
            # Validate paths
            log_file = self.get('log_file', 'LOGGING')
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception:
                    issues.append(f"Cannot create log directory: {log_dir}")
            
            return {
                'is_valid': len(issues) == 0,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'issues': [f"Configuration validation error: {str(e)}"]
            }

# Global configuration instance
config = Config()

