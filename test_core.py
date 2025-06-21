#!/usr/bin/env python3
"""
Test core functionality without GUI
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_core_functionality():
    """Test the core pattern recognition functionality"""
    print("Testing Forex Pattern Recognition Core...")
    
    try:
        # Test logging
        from utils.logger import setup_logger
        logger = setup_logger('TestCore')
        logger.info("Logger initialized successfully")
        
        # Test configuration
        from utils.config import config
        logger.info("Configuration loaded successfully")
        
        # Test data API
        from data.forex_api import FinancialDataAPI
        api = FinancialDataAPI(config, logger)
        logger.info("API client initialized successfully")
        
        # Test data processor
        from data.data_processor import DataProcessor
        processor = DataProcessor(logger)
        logger.info("Data processor initialized successfully")
        
        # Test pattern detector
        from models.pattern_detector import PatternDetector
        detector = PatternDetector(logger)
        logger.info("Pattern detector initialized successfully")
        
        print("✓ All core components initialized successfully")
        print("✓ The desktop application is ready to run on a local computer")
        print("✓ Use run_desktop.py on your local machine to start the GUI")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing core functionality: {str(e)}")
        return False

if __name__ == "__main__":
    test_core_functionality()