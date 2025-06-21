#!/usr/bin/env python3
"""
Forex Chart Pattern Recognition - Desktop Application
Lightweight version for testing core functionality
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_simple_logger(name):
    """Simple logger setup without external dependencies"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def test_imports():
    """Test all required imports"""
    logger = setup_simple_logger('DesktopApp')
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        logger.info("Basic scientific libraries imported successfully")
        
        # Test project modules
        from utils.config import config
        logger.info("Configuration module imported successfully")
        
        from data.forex_api import FinancialDataAPI
        logger.info("API module imported successfully")
        
        from data.data_processor import DataProcessor
        logger.info("Data processor imported successfully")
        
        # Initialize core components
        api = FinancialDataAPI(config, logger)
        processor = DataProcessor(logger)
        
        logger.info("All core components initialized successfully")
        
        # Test basic data fetching
        logger.info("Testing API connection...")
        symbols = api.get_supported_symbols()
        logger.info(f"Supported symbols: {symbols}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"General error: {e}")
        return False

def create_simple_gui():
    """Create a simple GUI test"""
    logger = setup_simple_logger('GUI_Test')
    
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        
        logger.info("Creating simple GUI test window...")
        
        root = tk.Tk()
        root.title("Forex Pattern Recognition - Desktop Test")
        root.geometry("600x400")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title label
        title_label = ttk.Label(main_frame, text="Forex Pattern Recognition System", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Status label
        status_label = ttk.Label(main_frame, text="Desktop application ready for use")
        status_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Test button
        def test_functionality():
            if test_imports():
                messagebox.showinfo("Test Result", "All core components working properly!")
            else:
                messagebox.showerror("Test Result", "Some components failed to initialize")
        
        test_button = ttk.Button(main_frame, text="Test Core Functionality", 
                                command=test_functionality)
        test_button.grid(row=2, column=0, pady=10)
        
        # Info text
        info_text = tk.Text(main_frame, height=15, width=70)
        info_text.grid(row=3, column=0, columnspan=2, pady=10)
        
        info_content = """
Desktop Forex Pattern Recognition System

Features:
- Real-time forex data fetching
- Chart pattern detection algorithms
- Technical indicator calculations
- Interactive chart visualization
- Pattern alerts and notifications

To use this application:
1. Install required dependencies
2. Configure API settings
3. Run the desktop application
4. Load data and detect patterns

Note: This application requires a desktop environment with display support.
For cloud environments like Replit, use the command-line interface instead.
        """
        
        info_text.insert('1.0', info_content)
        info_text.config(state='disabled')
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        logger.info("GUI created successfully")
        
        # Check if we can actually display
        if os.environ.get('DISPLAY') is None and os.name != 'nt':
            logger.warning("No display available - GUI cannot be shown")
            root.destroy()
            return False
        
        root.mainloop()
        return True
        
    except ImportError:
        logger.error("Tkinter not available")
        return False
    except Exception as e:
        logger.error(f"GUI error: {e}")
        return False

def main():
    """Main application entry point"""
    print("Forex Chart Pattern Recognition - Desktop Application")
    print("=" * 60)
    
    logger = setup_simple_logger('MainApp')
    
    # Test core functionality first
    print("Testing core functionality...")
    if test_imports():
        print("✓ Core functionality test passed")
    else:
        print("❌ Core functionality test failed")
        return
    
    # Try to create GUI
    print("\nAttempting to create desktop GUI...")
    if create_simple_gui():
        print("✓ Desktop GUI created successfully")
    else:
        print("❌ Desktop GUI cannot be displayed in this environment")
        print("   Please run this on a local computer with desktop support")
    
    print("\nDesktop application setup complete.")
    print("For full functionality, run this on your local computer.")

if __name__ == "__main__":
    main()