# Forex Chart Pattern Recognition System - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Basic Operations](#basic-operations)
3. [Pattern Detection](#pattern-detection)
4. [Real-Time Monitoring](#real-time-monitoring)
5. [Alert System](#alert-system)
6. [Advanced Features](#advanced-features)
7. [Settings & Configuration](#settings--configuration)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### System Requirements
- Windows 10/11, macOS 10.14+, or Linux
- Python 3.9 or higher
- Internet connection for live data
- 4GB RAM minimum (8GB recommended for real-time monitoring)

### Installation
1. **Download the application** from the repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the application**:
   ```bash
   python main.py
   ```

### First Launch Setup
When you first open the application:

1. **API Key Configuration**: 
   - The app starts with a demo API key (limited functionality)
   - For full features, obtain an API key from Financial Datasets
   - Go to Tools → API Settings to enter your key

2. **Initial Interface**:
   - Main chart area (center)
   - Controls panel (left side)
   - Pattern results (right side)
   - Status bar (bottom)

## Basic Operations

### Fetching Market Data

1. **Select Symbol**:
   - Choose from the dropdown: EUR/USD, GBP/USD, AAPL, MSFT, etc.
   - The system supports major forex pairs and popular stocks

2. **Choose Timeframe**:
   - Available options: 5min, 15min, 30min, 1hour, daily
   - Daily timeframe recommended for beginners

3. **Set Date Range**:
   - Compact: Last 30 days of data
   - Full: Last 150 days of data (recommended)

4. **Fetch Data**:
   - Click "Fetch Data" button
   - Wait for the progress indicator to complete
   - Chart will automatically update with candlestick data

### Reading the Chart

**Chart Elements**:
- **Green Candles**: Price went up during that period
- **Red Candles**: Price went down during that period
- **Volume Bars**: Trading activity (if enabled)
- **Technical Indicators**: Moving averages, Bollinger Bands (optional)

**Navigation**:
- The chart automatically fits all data
- Status bar shows current price and data statistics

## Pattern Detection

### Running Pattern Analysis

1. **Ensure Data is Loaded**: Must have market data before detecting patterns

2. **Adjust Sensitivity** (optional):
   - Default: 1.5% (recommended for beginners)
   - Lower values: More patterns detected (may include false signals)
   - Higher values: Fewer, higher-confidence patterns

3. **Start Detection**:
   - Click "Detect Patterns" button
   - Analysis typically takes 2-10 seconds
   - Results appear in the patterns panel on the right

### Understanding Pattern Results

**Pattern Information Displayed**:
- **Type**: Head & Shoulders, Double Top, Triangle, etc.
- **Confidence**: Percentage indicating pattern reliability
- **Signal**: Bullish (buy) or Bearish (sell) indication
- **Price Target**: Predicted price movement
- **Detection Time**: When the pattern was found

**Confidence Levels**:
- **80-100%**: High confidence - Strong trading signal
- **60-79%**: Medium confidence - Use with other analysis
- **Below 60%**: Low confidence - Additional confirmation needed

**Visual Overlays**:
- Patterns are drawn directly on the chart
- Different colors represent different pattern types
- Click on pattern entries to highlight them on the chart

### Supported Pattern Types

1. **Head and Shoulders**: Reversal pattern with three peaks
2. **Double Top/Bottom**: Two peaks/troughs at similar levels
3. **Ascending Triangle**: Higher lows with flat resistance
4. **Descending Triangle**: Lower highs with flat support
5. **Symmetrical Triangle**: Converging trend lines
6. **Support Levels**: Price floors where buying typically occurs
7. **Resistance Levels**: Price ceilings where selling typically occurs

## Real-Time Monitoring

### Starting Real-Time Mode

1. **Basic Real-Time**:
   - Select symbol and update interval (60 seconds recommended)
   - Click "Start Real-time" button
   - System fetches new data automatically

2. **Advanced Real-Time** (Tools → Advanced Monitoring):
   - Monitor multiple symbols simultaneously
   - Adaptive fetch intervals based on market volatility
   - Enhanced pattern detection with ML algorithms

### Real-Time Features

**Automatic Updates**:
- Chart refreshes with new price data
- Patterns detected automatically on new data
- Status bar shows live price and volatility

**Performance Indicators**:
- Green dot: System active and fetching data
- Update interval adjusts based on market activity
- High volatility = faster updates (15-30 seconds)
- Low volatility = standard updates (60 seconds)

**Stopping Real-Time**:
- Click "Stop Real-time" button
- System maintains current data for analysis
- Real-time alerts are disabled

## Alert System

### Setting Up Alerts

1. **Enable Notifications**:
   - Check "Popup Alerts" for visual notifications
   - Check "Sound Alerts" for audio notifications (optional)

2. **Automatic Alerts**:
   - System creates alerts for high-confidence patterns (>75%)
   - Alerts include trading suggestions and confidence scores

### Alert Types

**Pattern Alerts**:
- New pattern detected with high confidence
- Includes suggested action (Buy/Sell/Hold)
- Shows pattern details and price targets

**Alert Information**:
- **Symbol**: Which asset triggered the alert
- **Pattern Type**: What pattern was detected
- **Confidence**: Pattern reliability percentage
- **Suggested Action**: Recommended trading decision
- **Current Price**: Market price when alert was generated

### Managing Alerts

**Alert History**:
- Tools → Analytics Dashboard → History tab
- View all past alerts with performance tracking
- Export alert history to CSV or JSON

**Alert Settings**:
- Tools → User Preferences → Alerts tab
- Adjust minimum confidence threshold
- Enable/disable specific alert types
- Set maximum alert frequency

## Advanced Features

### Analytics Dashboard

Access via Tools → Analytics Dashboard

**Pattern Performance**:
- Success rate of detected patterns
- Average confidence vs actual performance
- Best and worst performing pattern types

**Symbol Analysis**:
- Performance breakdown by trading symbol
- Volatility trends and pattern frequency
- Timeline view of pattern detections

**Export Options**:
- Pattern history as CSV/JSON
- Performance reports as PDF
- Chart images for presentations

### Model Training

**When to Retrain**:
- After collecting significant new data (100+ patterns)
- When pattern accuracy seems to decline
- To adapt to changing market conditions

**Training Process**:
1. Ensure you have substantial historical data
2. Click "Retrain Model" in controls panel
3. Wait for training completion (1-5 minutes)
4. System reports new accuracy metrics

**Training Benefits**:
- Improved pattern detection accuracy
- Better confidence scoring
- Adaptation to current market conditions

### Advanced Monitoring Setup

**Multi-Symbol Monitoring**:
1. Tools → Advanced Monitoring Setup
2. Enter symbols separated by commas (e.g., AAPL, MSFT, TSLA)
3. Select multiple timeframes
4. Set update intervals and confidence thresholds
5. Click "Start Monitoring"

**Performance Optimization**:
- Monitor 3-5 symbols maximum for best performance
- Use daily timeframe for lower resource usage
- Higher confidence thresholds reduce noise

## Settings & Configuration

### User Preferences

Access via Tools → User Preferences

**Display Settings**:
- Chart type: Candlestick, Line, or OHLC bars
- Show/hide technical indicators
- Color scheme and chart theme

**Pattern Settings**:
- Enable/disable specific pattern types
- Adjust detection sensitivity
- Set minimum confidence for display

**Alert Preferences**:
- Notification types and frequency
- Sound settings
- Alert history retention

### API Configuration

Access via Tools → API Settings

**Financial Datasets API**:
- Enter your API key for full functionality
- Test connection to verify key validity
- Monitor API usage and rate limits

**Data Preferences**:
- Default symbols and timeframes
- Historical data range
- Cache settings for faster loading

### System Configuration

**Performance Settings**:
- Real-time update intervals
- Memory usage optimization
- Thread pool sizing for pattern detection

**File Locations**:
- Pattern results save location
- Chart export directory
- Log file location

## Troubleshooting

### Common Issues

**"No Data Available"**:
- Check internet connection
- Verify API key is valid
- Try a different symbol or timeframe
- Weekend markets may be closed

**"Pattern Detection Failed"**:
- Ensure sufficient data is loaded (minimum 50 candles)
- Try adjusting sensitivity settings
- Check if data quality is sufficient

**"Real-time Not Working"**:
- Verify API key has real-time permissions
- Check rate limiting (try longer intervals)
- Restart real-time monitoring

**Slow Performance**:
- Reduce number of monitored symbols
- Use longer update intervals
- Close other resource-intensive applications
- Consider upgrading to 8GB+ RAM

### Getting Help

**Log Files**:
- Located in `logs/forex_pattern_recognition.log`
- Contains detailed error information
- Include log excerpts when reporting issues

**System Information**:
- Help → System Info shows configuration details
- Include this information when seeking support

**Best Practices**:
- Start with demo mode to learn the interface
- Use daily timeframes for initial learning
- Begin with 1-2 symbols before expanding
- Set conservative confidence thresholds (70%+)

### Data Quality Tips

**Optimal Conditions**:
- Use major currency pairs (EUR/USD, GBP/USD)
- Trade during market hours for best data
- Avoid very short timeframes (under 15 minutes) initially

**Market Hours**:
- Forex: 24/5 (Sunday 5 PM - Friday 5 PM EST)
- Stocks: 9:30 AM - 4:00 PM EST (weekdays)
- Best volatility: London/New York overlap (8 AM - 12 PM EST)

## Quick Start Checklist

1. Launch application with `python main.py`
2. Select EUR/USD and daily timeframe
3. Choose "Full" data range and click "Fetch Data"
4. Wait for chart to load with candlestick data
5. Click "Detect Patterns" and review results
6. For live monitoring, click "Start Real-time"
7. Check pattern results panel for trading signals
8. Configure alerts in User Preferences if desired

This guide covers all major features. For advanced technical details, see the developer documentation files in the project directory.