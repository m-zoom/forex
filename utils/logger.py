"""
Logging utilities for the Forex Pattern Recognition System
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class ForexLogger:
    """Custom logger for the Forex Pattern Recognition System"""
    
    def __init__(self, name="ForexPatternRecognition", config=None):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup logging handlers based on configuration"""
        try:
            # Get configuration values
            if self.config:
                log_level = self.config.get('log_level', 'LOGGING', 'INFO')
                log_file = self.config.get('log_file', 'LOGGING', 'logs/forex_pattern_recognition.log')
                max_log_size = self.config.get_int('max_log_size', 'LOGGING', 10485760)  # 10MB
                backup_count = self.config.get_int('backup_count', 'LOGGING', 5)
                console_logging = self.config.get_bool('console_logging', 'LOGGING', True)
            else:
                log_level = 'INFO'
                log_file = 'logs/forex_pattern_recognition.log'
                max_log_size = 10485760
                backup_count = 5
                console_logging = True
            
            # Set log level
            numeric_level = getattr(logging, log_level.upper(), logging.INFO)
            self.logger.setLevel(numeric_level)
            
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
            
            # File handler with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_log_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(numeric_level)
            self.logger.addHandler(file_handler)
            
            # Console handler
            if console_logging:
                console_handler = logging.StreamHandler(sys.stdout)
                
                # Use colored formatter for console
                console_formatter = ColoredFormatter(
                    '%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
                console_handler.setLevel(numeric_level)
                self.logger.addHandler(console_handler)
            
            # Log startup message
            self.logger.info(f"Logger initialized - Level: {log_level}, File: {log_file}")
            
        except Exception as e:
            # Fallback to basic console logging
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(console_handler)
            self.logger.error(f"Error setting up logging: {str(e)}")
    
    def get_logger(self):
        """Get the logger instance"""
        return self.logger
    
    def log_system_info(self):
        """Log system information"""
        try:
            import platform
            import psutil
            
            self.logger.info("=" * 60)
            self.logger.info("SYSTEM INFORMATION")
            self.logger.info("=" * 60)
            self.logger.info(f"Platform: {platform.platform()}")
            self.logger.info(f"Architecture: {platform.architecture()[0]}")
            self.logger.info(f"Processor: {platform.processor()}")
            self.logger.info(f"Python Version: {sys.version}")
            
            # Memory info
            memory = psutil.virtual_memory()
            self.logger.info(f"Total Memory: {memory.total / (1024**3):.2f} GB")
            self.logger.info(f"Available Memory: {memory.available / (1024**3):.2f} GB")
            self.logger.info(f"Memory Usage: {memory.percent}%")
            
            # Disk info
            disk = psutil.disk_usage('/')
            self.logger.info(f"Disk Total: {disk.total / (1024**3):.2f} GB")
            self.logger.info(f"Disk Free: {disk.free / (1024**3):.2f} GB")
            self.logger.info(f"Disk Usage: {(disk.used / disk.total) * 100:.1f}%")
            
            self.logger.info("=" * 60)
            
        except ImportError:
            self.logger.info("System info logging requires psutil package")
        except Exception as e:
            self.logger.error(f"Error logging system info: {str(e)}")
    
    def log_performance(self, operation, duration, details=None):
        """Log performance metrics"""
        try:
            message = f"PERFORMANCE - {operation}: {duration:.3f}s"
            if details:
                message += f" - {details}"
            
            if duration > 5.0:
                self.logger.warning(message)
            elif duration > 2.0:
                self.logger.info(message)
            else:
                self.logger.debug(message)
                
        except Exception as e:
            self.logger.error(f"Error logging performance: {str(e)}")
    
    def log_api_request(self, endpoint, status_code, duration, response_size=None):
        """Log API request details"""
        try:
            message = f"API REQUEST - {endpoint} - Status: {status_code} - Duration: {duration:.3f}s"
            if response_size:
                message += f" - Size: {response_size} bytes"
            
            if status_code >= 400:
                self.logger.error(message)
            elif status_code >= 300:
                self.logger.warning(message)
            else:
                self.logger.info(message)
                
        except Exception as e:
            self.logger.error(f"Error logging API request: {str(e)}")
    
    def log_pattern_detection(self, patterns_found, processing_time, data_points):
        """Log pattern detection results"""
        try:
            high_confidence = sum(1 for p in patterns_found if p.get('confidence', 0) >= 0.8)
            
            message = f"PATTERN DETECTION - Found: {len(patterns_found)} patterns "
            message += f"({high_confidence} high confidence) - "
            message += f"Time: {processing_time:.3f}s - Data Points: {data_points}"
            
            if high_confidence > 0:
                self.logger.info(message)
            else:
                self.logger.debug(message)
                
        except Exception as e:
            self.logger.error(f"Error logging pattern detection: {str(e)}")
    
    def log_model_performance(self, model_name, accuracy, training_time=None):
        """Log model performance metrics"""
        try:
            message = f"MODEL PERFORMANCE - {model_name} - Accuracy: {accuracy:.3f}"
            if training_time:
                message += f" - Training Time: {training_time:.1f}s"
            
            if accuracy >= 0.85:
                self.logger.info(message)
            elif accuracy >= 0.70:
                self.logger.warning(f"{message} - Performance below optimal")
            else:
                self.logger.error(f"{message} - Poor performance detected")
                
        except Exception as e:
            self.logger.error(f"Error logging model performance: {str(e)}")
    
    def log_error_with_context(self, error, context=None, exc_info=True):
        """Log error with additional context"""
        try:
            message = f"ERROR CONTEXT - {str(error)}"
            if context:
                message += f" - Context: {context}"
            
            self.logger.error(message, exc_info=exc_info)
            
        except Exception as e:
            self.logger.error(f"Error logging error context: {str(e)}")
    
    def log_data_quality(self, validation_result):
        """Log data quality assessment"""
        try:
            if validation_result.get('is_valid', False):
                self.logger.info(f"DATA QUALITY - Valid data: {validation_result.get('data_points', 0)} points")
            else:
                issues = validation_result.get('issues', [])
                self.logger.warning(f"DATA QUALITY - Issues found: {'; '.join(issues)}")
                
        except Exception as e:
            self.logger.error(f"Error logging data quality: {str(e)}")
    
    def log_user_action(self, action, details=None):
        """Log user actions for audit trail"""
        try:
            message = f"USER ACTION - {action}"
            if details:
                message += f" - {details}"
            
            self.logger.info(message)
            
        except Exception as e:
            self.logger.error(f"Error logging user action: {str(e)}")
    
    def create_session_log(self, session_id):
        """Create a session-specific log entry"""
        try:
            self.logger.info(f"SESSION START - ID: {session_id} - Time: {datetime.now()}")
            
        except Exception as e:
            self.logger.error(f"Error creating session log: {str(e)}")
    
    def close_session_log(self, session_id, duration=None):
        """Close session log entry"""
        try:
            message = f"SESSION END - ID: {session_id} - Time: {datetime.now()}"
            if duration:
                message += f" - Duration: {duration:.1f}s"
            
            self.logger.info(message)
            
        except Exception as e:
            self.logger.error(f"Error closing session log: {str(e)}")

def setup_logger(name="ForexPatternRecognition", config=None):
    """Setup and return a configured logger instance"""
    try:
        forex_logger = ForexLogger(name, config)
        logger = forex_logger.get_logger()
        
        # Log system info on first setup
        if config and config.get_bool('log_system_info', 'LOGGING', False):
            forex_logger.log_system_info()
        
        return logger
        
    except Exception as e:
        # Fallback logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        logger = logging.getLogger(name)
        logger.error(f"Error setting up custom logger, using fallback: {str(e)}")
        return logger

def get_logger(name="ForexPatternRecognition"):
    """Get existing logger or create new one"""
    return logging.getLogger(name)

# Performance logging decorator
def log_performance(logger_instance):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger_instance.log_performance(
                    func.__name__, 
                    duration, 
                    f"Args: {len(args)}, Kwargs: {len(kwargs)}"
                )
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger_instance.log_error_with_context(
                    e, 
                    f"Function: {func.__name__}, Duration: {duration:.3f}s"
                )
                raise
                
        return wrapper
    return decorator

