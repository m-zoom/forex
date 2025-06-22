# Forex Chart Pattern Recognition System

## Overview

This is an AI-powered desktop application built with Python and Tkinter that automatically detects and analyzes forex chart patterns in real-time. The system integrates with the Financial Datasets API to fetch live forex data and uses machine learning models to identify common chart patterns like Head and Shoulders, Double Tops/Bottoms, Triangles, and Support/Resistance levels.

## System Architecture

### Frontend Architecture
- **GUI Framework**: Tkinter-based desktop application
- **Chart Visualization**: Matplotlib with mplfinance for professional candlestick charts
- **Modular Design**: Separate frames for chart display, controls, and pattern results
- **Real-time Updates**: Threading for non-blocking UI updates

### Backend Architecture
- **Pattern Detection Engine**: Pre-trained machine learning models using TensorFlow/Keras
- **Data Processing Pipeline**: Pandas and NumPy for data manipulation and technical indicator calculations
- **API Integration**: Alpha Vantage API for real-time forex data
- **Configuration Management**: INI-based configuration with environment variable support

## Key Components

### Data Layer (`data/`)
- **ForexAPI**: Handles API requests with rate limiting and error handling
- **DataProcessor**: Processes raw OHLCV data and adds technical indicators using TA-Lib

### Model Layer (`models/`)
- **PatternDetector**: Core pattern detection algorithms with configurable sensitivity
- **MLModels**: CNN and LSTM models for pattern recognition
- **PretrainedWeights**: Pre-configured model architectures with trained weights

### GUI Layer (`gui/`)
- **MainWindow**: Central application window with menu system
- **ChartFrame**: Interactive chart visualization with pattern overlays
- **ControlsFrame**: Data fetching and configuration controls
- **PatternFrame**: Pattern detection results display

### Utilities (`utils/`)
- **Config**: Configuration management with defaults and validation
- **Logger**: Multi-level logging with file rotation and colored console output
- **Helpers**: File I/O, data validation, and formatting utilities

## Data Flow

1. **Data Acquisition**: Alpha Vantage API fetches real-time forex data with rate limiting
2. **Data Processing**: Raw OHLCV data is cleaned and enhanced with technical indicators
3. **Pattern Detection**: ML models analyze processed data to identify chart patterns
4. **Visualization**: Detected patterns are overlaid on interactive charts
5. **Real-time Monitoring**: Background threads continuously monitor for new patterns
6. **Export/Alerts**: Results are saved and notifications are sent based on user preferences

## External Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/mplfinance**: Chart visualization
- **tensorflow**: Machine learning models
- **scikit-learn**: Additional ML utilities
- **scipy**: Scientific computing
- **requests**: HTTP API calls

### Financial Data
- **Alpha Vantage API**: Primary data source for forex prices
- **TA-Lib**: Technical analysis indicators (RSI, MACD, Bollinger Bands)

### GUI Components
- **tkinter**: Native Python GUI framework
- **matplotlib backends**: Chart embedding in tkinter

## Deployment Strategy

### Development Environment
- **Replit Configuration**: Pre-configured with Python 3.11 and required system packages
- **Package Management**: Automatic installation via pip in startup command
- **Port Configuration**: Runs on port 5000 with automatic detection

### Production Considerations
- **Desktop Application**: Designed for local execution with persistent configuration
- **Resource Management**: Optimized for memory usage with data caching
- **Error Handling**: Comprehensive error recovery and user feedback
- **Backup System**: Automatic data backup and model persistence

### Configuration Management
- **Environment Variables**: API keys and sensitive data
- **INI Files**: User preferences and application settings
- **Logging**: Configurable log levels with file rotation

## Changelog

```
Changelog:
- June 21, 2025. Initial setup with desktop Tkinter application
- June 21, 2025. Removed web applications per user preference
- June 21, 2025. Fixed logger initialization and core functionality testing
- June 21, 2025. Fixed API validation unpacking error in validate_api_key method
- June 21, 2025. Changed forex pairs to stock symbols (AAPL, MSFT, etc.) for API compatibility
- June 21, 2025. Added Config.save() method to fix shutdown error
- June 21, 2025. Removed chart emojis from GUI to eliminate font warnings
- June 21, 2025. Fixed chart dimension mismatch error with data alignment and fallback charts
- June 22, 2025. Resolved matplotlib threading issues that caused mouse event errors
- June 22, 2025. Added thread-safe chart rendering to prevent "main thread not in main loop" errors
- June 22, 2025. Fixed pattern overlay dimension mismatch errors
- June 22, 2025. Fixed weekend API validation issue - now uses 5-day range instead of single day
- June 22, 2025. Desktop application fully functional on local computers with 29+ patterns detected
- June 22, 2025. Integrated comprehensive UX enhancements: intelligent alert system, preferences dialog, analytics dashboard
- June 22, 2025. Added smart trading suggestions with confidence-based recommendations for detected patterns
- June 22, 2025. Enhanced menu system with professional trading features and pattern history tracking
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
Application type: Desktop Tkinter GUI only - no web applications
```