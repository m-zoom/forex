#!/usr/bin/env python3
"""
Forex Chart Pattern Recognition System - Web Interface
Flask-based web application for pattern recognition
"""

import os
import sys
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import mplfinance as mpf
import io
import base64

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logger, config
from data import ForexAPI, DataProcessor
from models import PatternDetector, PatternRecognitionModels

app = Flask(__name__)

class WebForexApp:
    """Web-based Forex Pattern Recognition Application"""
    
    def __init__(self):
        self.logger = setup_logger('WebForexApp')
        self.config = config
        self.api = ForexAPI(self.config, self.logger)
        self.processor = DataProcessor(self.logger)
        self.pattern_detector = PatternDetector(self.logger)
        self.ml_models = PatternRecognitionModels(self.logger)
        
        self.current_data = pd.DataFrame()
        self.current_patterns = []
        
        self.logger.info("Web Forex Application initialized")
    
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
            patterns = self.pattern_detector.detect_all_patterns(self.current_data)
            self.current_patterns = patterns
            
            # Convert patterns to serializable format
            pattern_list = []
            for pattern in patterns:
                pattern_dict = {
                    "type": pattern.get("type", "Unknown"),
                    "confidence": pattern.get("confidence", 0.0),
                    "start_time": str(pattern.get("start_time", "")),
                    "end_time": str(pattern.get("end_time", "")),
                    "description": pattern.get("description", ""),
                    "signal": pattern.get("signal", ""),
                    "price_target": pattern.get("price_target", 0.0)
                }
                pattern_list.append(pattern_dict)
            
            return {
                "success": True,
                "patterns": pattern_list,
                "count": len(pattern_list)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_chart(self, chart_type="candlestick", show_patterns=True):
        """Generate chart image"""
        try:
            if self.current_data.empty:
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare data for mplfinance
            df_chart = self.current_data[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Create the chart
            if chart_type == "candlestick":
                mpf.plot(df_chart, type='candle', ax=ax, volume=False, 
                        style='charles', title=f"Price Chart - Last {len(df_chart)} periods")
            else:
                mpf.plot(df_chart, type='line', ax=ax, volume=False,
                        style='charles', title=f"Price Chart - Last {len(df_chart)} periods")
            
            # Add pattern overlays if requested
            if show_patterns and self.current_patterns:
                for pattern in self.current_patterns:
                    self._add_pattern_overlay(ax, pattern)
            
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
    
    def _add_pattern_overlay(self, ax, pattern):
        """Add pattern overlay to chart"""
        try:
            pattern_type = pattern.get("type", "").lower()
            confidence = pattern.get("confidence", 0.0)
            
            if confidence < 0.5:  # Only show high confidence patterns
                return
            
            # Add text annotation
            ax.text(0.02, 0.98 - len(ax.texts) * 0.04, 
                   f"{pattern.get('type', 'Pattern')}: {confidence:.2f}",
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                   
        except Exception as e:
            self.logger.error(f"Error adding pattern overlay: {str(e)}")

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
    chart_type = request.args.get('type', 'candlestick')
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
        "api_available": forex_app.api.validate_api_key()
    })

if __name__ == '__main__':
    print("Starting Forex Chart Pattern Recognition Web Application...")
    print("Access the application at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)