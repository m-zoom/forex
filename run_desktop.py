#!/usr/bin/env python3
"""
Desktop Forex Pattern Recognition System
Run this script on your local computer to start the Tkinter GUI application
"""

import sys
import os

def check_display():
    """Check if display is available for GUI"""
    if os.name == 'nt':  # Windows
        return True
    else:  # Linux/Mac
        return 'DISPLAY' in os.environ

def main():
    """Main entry point for desktop application"""
    print("Forex Chart Pattern Recognition System - Desktop Version")
    print("=" * 60)
    
    # Check if we can run GUI
    if not check_display():
        print("❌ No display available for GUI application")
        print("   This needs to run on a local computer with a desktop environment")
        print("   Please download this project and run it locally")
        sys.exit(1)
    
    # Import and run the main application
    try:
        from main import main as run_main
        run_main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please install required dependencies:")
        print("   pip install pandas numpy matplotlib mplfinance requests scipy scikit-learn tensorflow")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()