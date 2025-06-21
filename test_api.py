#!/usr/bin/env python3
"""
Test API validation functionality
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_api_validation():
    """Test the API validation fix"""
    try:
        from utils.logger import setup_logger
        from utils.config import config
        from data.forex_api import FinancialDataAPI
        
        logger = setup_logger('APITest')
        api = FinancialDataAPI(config, logger)
        
        print("Testing API validation...")
        
        # Test the fixed validation method
        is_valid, message = api.validate_api_key()
        
        print(f"API Validation Result: {is_valid}")
        print(f"Message: {message}")
        
        if is_valid:
            print("✓ API validation working correctly")
            
            # Test data fetching
            print("\nTesting data fetch...")
            data = api.get_forex_data("AAPL", "5min", "compact")
            
            if not data.empty:
                print(f"✓ Data fetched successfully: {len(data)} records")
                print(f"Date range: {data.index[0]} to {data.index[-1]}")
            else:
                print("❌ No data returned")
        else:
            print("❌ API validation failed")
            
        return is_valid
        
    except Exception as e:
        print(f"❌ Error testing API: {str(e)}")
        return False

if __name__ == "__main__":
    test_api_validation()