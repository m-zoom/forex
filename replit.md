# Forex Chart Pattern Recognition System

## Overview

This is an AI-powered desktop application built with Python and Tkinter that automatically detects and analyzes forex chart patterns in real-time. The system integrates with the Financial Datasets API to fetch live forex data and uses machine learning models to identify common chart patterns like Head and Shoulders, Double Tops/Bottoms, Triangles, and Support/Resistance levels.

## System Architecture

### Frontend Architecture
- **GUI Framework**: Tkinter-based desktop application with enhanced UX components
- **Chart Visualization**: Matplotlib with mplfinance for professional candlestick charts
- **Modular Design**: Separate frames for chart display, controls, and pattern results
- **Real-time Updates**: Advanced threading architecture for responsive UI updates
- **Alert System**: Intelligent notifications with smart trading suggestions
- **Analytics Dashboard**: Pattern history tracking and performance metrics

### Backend Architecture
- **Advanced Real-time Monitor**: Adaptive multi-symbol monitoring with volatility-based scheduling
- **Pattern Detection Engine**: ML-enhanced pattern recognition with feedback learning
- **Data Processing Pipeline**: Sliding buffer architecture for efficient real-time processing
- **API Integration**: Financial Datasets API with resilient multi-threaded data fetching
- **Feedback Engine**: Machine learning system that improves detection accuracy over time

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

1. **Adaptive Data Acquisition**: Multi-threaded system fetches data with volatility-based intervals
2. **Sliding Buffer Management**: Maintains recent N candles per symbol/timeframe for efficient processing
3. **Real-time Processing**: Technical indicators applied to streaming data with minimal latency
4. **ML-Enhanced Detection**: Pattern recognition using rule-based algorithms and neural networks
5. **Confidence Scoring**: Advanced scoring engine with feedback-driven sensitivity adjustment
6. **Intelligent Alerts**: High-confidence patterns trigger smart notifications with trading suggestions
7. **Analytics Integration**: All patterns tracked for performance analysis and system learning

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
- June 24, 2025. Implemented advanced real-time monitoring system with adaptive fetch intervals based on market volatility
- June 24, 2025. Added multi-symbol, multi-timeframe monitoring with sliding buffer architecture
- June 24, 2025. Integrated feedback-driven pattern detection with machine learning sensitivity adjustment
- June 24, 2025. Enhanced threading architecture for improved responsiveness and error resilience
- June 24, 2025. Fixed API interval mapping inconsistency causing "Unsupported interval" errors for 1h, 4h timeframes
- June 25, 2025. Completed comprehensive analysis of entire codebase covering 250+ functions across 30+ classes
- June 25, 2025. Documented all GUI components, ML models, threading architecture, and configuration systems
- June 25, 2025. Created detailed developer guide with internal component explanations and development workflow
- June 25, 2025. Refactored threading architecture to use asyncio + bounded thread pools, reducing complexity from 4+ threads to 2 core components
- June 25, 2025. Fixed MonitoringConfig import error, async monitoring system now fully operational
- June 25, 2025. Created comprehensive user documentation (USER_GUIDE.md) and quick start guide for easy onboarding
- June 25, 2025. Fixed advanced real-time monitoring interval configuration - user settings (10-300s) now properly applied to monitoring system
- June 25, 2025. Resolved candlestick chart "zero-size array" error with enhanced data validation and minimum data requirements
- June 25, 2025. Fixed AsyncRealtimeMonitor constructor signature error causing startup failure
- June 25, 2025. Enhanced API key validation to properly use environment variables over config defaults
- June 25, 2025. Integrated Financial Datasets API key for full real-time market data access
- July 2, 2025. Integrated advanced ML pattern classifiers with real-time monitoring system
- July 2, 2025. Added 4 sophisticated machine learning models: Triple Bottom, Double Bottom, Double Top, Inverse Head & Shoulders
- July 2, 2025. Implemented real-time ML monitoring with confidence scoring and intelligent alerts
- July 2, 2025. Added ML monitoring controls to GUI with live status updates and pattern statistics
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
Application type: Desktop Tkinter GUI only - no web applications
```