#!/usr/bin/env python3
"""
Forex Chart Pattern Recognition System - Simplified Web Interface
Flask-based web application with lightweight pattern recognition
"""

import os
import sys
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logger, config
from data import ForexAPI, DataProcessor

app = Flask(__name__)

class SimplifiedPatternDetector:
    """Simplified pattern detector without heavy ML dependencies"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def detect_simple_patterns(self, df):
        """Detect basic patterns using simple algorithms"""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        try:
            # Simple moving average crossover
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            
            # Detect crossovers
            crossover_up = (df['sma_10'] > df['sma_20']) & (df['sma_10'].shift(1) <= df['sma_20'].shift(1))
            crossover_down = (df['sma_10'] < df['sma_20']) & (df['sma_10'].shift(1) >= df['sma_20'].shift(1))
            
            # Add bullish crossovers
            bullish_crosses = df[crossover_up]
            for idx in bullish_crosses.index:
                patterns.append({
                    'type': 'MA Bullish Crossover',
                    'confidence': 0.75,
                    'start_time': str(idx),
                    'end_time': str(idx),
                    'description': '10-period MA crossed above 20-period MA',
                    'signal': 'Buy',
                    'price_target': float(df.loc[idx, 'close'] * 1.02)
                })
            
            # Add bearish crossovers
            bearish_crosses = df[crossover_down]
            for idx in bearish_crosses.index:
                patterns.append({
                    'type': 'MA Bearish Crossover',
                    'confidence': 0.75,
                    'start_time': str(idx),
                    'end_time': str(idx),
                    'description': '10-period MA crossed below 20-period MA',
                    'signal': 'Sell',
                    'price_target': float(df.loc[idx, 'close'] * 0.98)
                })
            
            # Simple support/resistance levels
            recent_data = df.tail(50)
            price_levels = []
            
            # Find potential support levels (recent lows)
            min_price = recent_data['low'].min()
            if (recent_data['low'] == min_price).sum() >= 2:
                patterns.append({
                    'type': 'Support Level',
                    'confidence': 0.65,
                    'start_time': str(recent_data.index[0]),
                    'end_time': str(recent_data.index[-1]),
                    'description': f'Support level at {min_price:.4f}',
                    'signal': 'Support',
                    'price_target': float(min_price)
                })
            
            # Find potential resistance levels (recent highs)
            max_price = recent_data['high'].max()
            if (recent_data['high'] == max_price).sum() >= 2:
                patterns.append({
                    'type': 'Resistance Level',
                    'confidence': 0.65,
                    'start_time': str(recent_data.index[0]),
                    'end_time': str(recent_data.index[-1]),
                    'description': f'Resistance level at {max_price:.4f}',
                    'signal': 'Resistance',
                    'price_target': float(max_price)
                })
            
            self.logger.info(f"Detected {len(patterns)} simple patterns")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {str(e)}")
            return patterns

class WebForexApp:
    """Simplified Web-based Forex Pattern Recognition Application"""
    
    def __init__(self):
        self.logger = setup_logger('WebForexApp')
        self.config = config
        self.api = ForexAPI(self.config, self.logger)
        self.processor = DataProcessor(self.logger)
        self.pattern_detector = SimplifiedPatternDetector(self.logger)
        
        self.current_data = pd.DataFrame()
        self.current_patterns = []
        
        self.logger.info("Simplified Web Forex Application initialized")
    
    def fetch_data(self, symbol, interval="5min", outputsize="compact"):
        """Fetch data from API"""
        try:
            self.logger.info(f"Fetching data for {symbol}")
            raw_data = self.api.get_forex_data(symbol, interval, outputsize)
            
            if raw_data.empty:
                return {"success": False, "error": "No data retrieved from API"}
            
            # Process the data
            self.current_data = self.processor.process_data(raw_data)
            
            return {
                "success": True, 
                "records": len(self.current_data),
                "date_range": {
                    "start": str(self.current_data.index.min()),
                    "end": str(self.current_data.index.max())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def detect_patterns(self):
        """Detect patterns in current data"""
        try:
            if self.current_data.empty:
                return {"success": False, "error": "No data available"}
            
            self.logger.info("Detecting patterns")
            patterns = self.pattern_detector.detect_simple_patterns(self.current_data)
            self.current_patterns = patterns
            
            return {
                "success": True,
                "patterns": patterns,
                "count": len(patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_chart(self, chart_type="line", show_patterns=True):
        """Generate chart image"""
        try:
            if self.current_data.empty:
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot price data
            if chart_type == "candlestick":
                # Simple OHLC line chart since mplfinance might be complex
                ax.plot(self.current_data.index, self.current_data['close'], label='Close Price', color='blue')
                ax.plot(self.current_data.index, self.current_data['high'], alpha=0.3, color='green')
                ax.plot(self.current_data.index, self.current_data['low'], alpha=0.3, color='red')
            else:
                ax.plot(self.current_data.index, self.current_data['close'], label='Close Price', color='blue')
            
            # Add moving averages if available
            if 'sma_10' in self.current_data.columns:
                ax.plot(self.current_data.index, self.current_data['sma_10'], label='SMA 10', alpha=0.7, color='orange')
            if 'sma_20' in self.current_data.columns:
                ax.plot(self.current_data.index, self.current_data['sma_20'], label='SMA 20', alpha=0.7, color='purple')
            
            # Add pattern annotations
            if show_patterns and self.current_patterns:
                for i, pattern in enumerate(self.current_patterns[:5]):  # Limit to 5 patterns
                    ax.text(0.02, 0.98 - i * 0.05, 
                           f"{pattern.get('type', 'Pattern')}: {pattern.get('confidence', 0):.2f}",
                           transform=ax.transAxes, fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            ax.set_title(f"Financial Data Chart - {len(self.current_data)} data points")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error generating chart: {str(e)}")
            return None

# Initialize the application
forex_app = WebForexApp()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/symbols')
def get_symbols():
    """Get supported symbols"""
    symbols = forex_app.api.get_supported_symbols()
    return jsonify({"symbols": symbols})

@app.route('/api/fetch_data', methods=['POST'])
def fetch_data():
    """Fetch data endpoint"""
    data = request.get_json()
    symbol = data.get('symbol', 'AAPL')
    interval = data.get('interval', '5min')
    outputsize = data.get('outputsize', 'compact')
    
    result = forex_app.fetch_data(symbol, interval, outputsize)
    return jsonify(result)

@app.route('/api/detect_patterns', methods=['POST'])
def detect_patterns():
    """Pattern detection endpoint"""
    result = forex_app.detect_patterns()
    return jsonify(result)

@app.route('/api/chart')
def get_chart():
    """Chart generation endpoint"""
    chart_type = request.args.get('type', 'line')
    show_patterns = request.args.get('patterns', 'true').lower() == 'true'
    
    chart_img = forex_app.generate_chart(chart_type, show_patterns)
    
    if chart_img:
        return jsonify({"success": True, "image": chart_img})
    else:
        return jsonify({"success": False, "error": "Failed to generate chart"})

@app.route('/api/status')
def get_status():
    """Application status endpoint"""
    return jsonify({
        "status": "running",
        "data_loaded": not forex_app.current_data.empty,
        "patterns_detected": len(forex_app.current_patterns),
        "api_available": True  # Simplified check
    })

if __name__ == '__main__':
    print("Starting Forex Chart Pattern Recognition Web Application...")
    print("Access the application at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)  # Disable debug for faster startup