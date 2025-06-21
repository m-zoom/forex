#!/usr/bin/env python3
"""
Forex Chart Pattern Recognition System
Main entry point for the Tkinter desktop application

Author: AI-Powered Trading Systems
Version: 1.0.0
Description: Desktop application for automated forex chart pattern recognition
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
import threading

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
        
    try:
        import mplfinance
    except ImportError:
        missing_deps.append("mplfinance")
        
    try:
        import requests
    except ImportError:
        missing_deps.append("requests")
        
    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy")
        
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    if missing_deps:
        error_msg = f"Missing required dependencies: {', '.join(missing_deps)}\n\n"
        error_msg += "Please install them using:\n"
        error_msg += f"pip install {' '.join(missing_deps)}"
        messagebox.showerror("Missing Dependencies", error_msg)
        return False
        
    return True

def main():
    """Main application entry point"""
    try:
        # Check dependencies first
        if not check_dependencies():
            sys.exit(1)
            
        # Import application modules after dependency check
        from gui.main_window import MainWindow
        from utils.logger import setup_logger
        from utils.config import config
        
        # Setup logging
        logger = setup_logger('ForexPatternApp')
        logger.info("=" * 50)
        logger.info("Starting Forex Pattern Recognition System v1.0.0")
        logger.info("=" * 50)
        
        # Load configuration
        logger.info("Configuration loaded successfully")
        
        # Check API configuration
        logger.info("Desktop Tkinter application starting...")
        
        # Create main application window
        root = tk.Tk()
        
        # Set application icon and properties
        root.title("Forex Chart Pattern Recognition System v1.0.0")
        root.geometry("1400x900")
        root.minsize(1200, 800)
        
        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Initialize main application
        app = MainWindow(root, config, logger)
        
        # Handle window closing
        def on_closing():
            logger.info("Application shutdown requested")
            if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
                app.on_closing()
                logger.info("Application shutdown completed")
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        logger.info("GUI initialized successfully")
        logger.info("Application ready for use")
        
        # Start the GUI event loop
        root.mainloop()
        
    except ImportError as e:
        error_msg = f"Import error: {str(e)}\n\nPlease ensure all required modules are installed."
        messagebox.showerror("Import Error", error_msg)
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Failed to start application: {str(e)}\n\nPlease check the logs for more details."
        messagebox.showerror("Application Error", error_msg)
        
        # Try to log the error if logger is available
        try:
            from utils.logger import setup_logger
            logger = setup_logger('ForexPatternApp')
            logger.error(f"Application startup failed: {str(e)}", exc_info=True)
        except:
            print(f"Critical error: {str(e)}")
            
        sys.exit(1)

if __name__ == "__main__":
    main()
