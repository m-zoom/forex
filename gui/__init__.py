"""
GUI package for Forex Chart Pattern Recognition System
Contains all GUI components and widgets
"""

__version__ = "1.0.0"
__author__ = "AI-Powered Trading Systems"

# GUI Components
from .main_window import MainWindow
from .chart_frame import ChartFrame
from .controls_frame import ControlsFrame
from .pattern_frame import PatternFrame

__all__ = [
    'MainWindow',
    'ChartFrame', 
    'ControlsFrame',
    'PatternFrame'
]
