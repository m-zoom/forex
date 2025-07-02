"""
Real-Time ML Pattern Monitoring System
Advanced pattern detection with machine learning classifiers
"""

import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

from .ml_pattern_classifiers import (
    TripleBottomClassifier,
    DoubleBottomClassifier, 
    DoubleTopClassifier,
    InverseHeadAndShouldersClassifier
)


@dataclass
class PatternAlert:
    """Pattern detection alert with confidence scoring"""
    symbol: str
    timeframe: str
    pattern_type: str
    confidence: float
    timestamp: datetime
    price_data: List[float]
    entry_price: float
    target_price: float
    stop_loss: float
    prediction_details: Dict[str, Any]


class RealTimeMLMonitor:
    """Real-time monitoring system with ML pattern detection"""
    
    def __init__(self, main_window, logger):
        self.main_window = main_window
        self.logger = logger
        self.monitoring_active = False
        self.monitoring_thread = None
        self.pattern_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        
        # ML Pattern Classifiers
        self.classifiers = {
            'triple_bottom': TripleBottomClassifier(threshold=0.75),
            'double_bottom': DoubleBottomClassifier(threshold=0.75),
            'double_top': DoubleTopClassifier(threshold=0.75),
            'inverse_head_shoulders': InverseHeadAndShouldersClassifier(threshold=0.75)
        }
        
        # Configuration
        self.config = {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'timeframe': 'daily',
            'check_interval': 30,  # seconds
            'min_confidence': 0.70,
            'lookback_periods': 30,
            'alert_cooldown': 300  # 5 minutes
        }
        
        # Alert tracking
        self.last_alerts = {}
        self.pattern_history = []
        
        self.logger.info("Real-time ML pattern monitor initialized")
        self.logger.info(f"Monitoring {len(self.config['symbols'])} symbols with {len(self.classifiers)} ML classifiers")
    
    def start_monitoring(self):
        """Start real-time pattern monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("=" * 60)
        self.logger.info("🚀 REAL-TIME ML PATTERN MONITORING STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Symbols: {', '.join(self.config['symbols'])}")
        self.logger.info(f"🔍 Patterns: {', '.join(self.classifiers.keys())}")
        self.logger.info(f"⏱️  Check interval: {self.config['check_interval']}s")
        self.logger.info(f"🎯 Min confidence: {self.config['min_confidence']}")
        self.logger.info("=" * 60)
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if not self.monitoring_active:
            return
            
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
            
        self.logger.info("🛑 Real-time ML pattern monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        cycle_count = 0
        
        while self.monitoring_active:
            try:
                cycle_count += 1
                cycle_start = time.time()
                
                self.logger.info(f"🔄 Monitoring cycle #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Check each symbol
                patterns_found = 0
                for symbol in self.config['symbols']:
                    try:
                        patterns = self._analyze_symbol(symbol)
                        patterns_found += len(patterns)
                        
                        # Process high-confidence patterns
                        for pattern in patterns:
                            if pattern.confidence >= self.config['min_confidence']:
                                self._process_pattern_alert(pattern)
                                
                    except Exception as e:
                        self.logger.error(f"Error analyzing {symbol}: {str(e)}")
                
                cycle_duration = time.time() - cycle_start
                
                if patterns_found > 0:
                    self.logger.info(f"✅ Cycle #{cycle_count} completed: {patterns_found} patterns detected in {cycle_duration:.2f}s")
                else:
                    self.logger.info(f"📈 Cycle #{cycle_count} completed: Market scanning... ({cycle_duration:.2f}s)")
                
                # Wait for next cycle
                time.sleep(self.config['check_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)  # Brief pause on error
    
    def _analyze_symbol(self, symbol: str) -> List[PatternAlert]:
        """Analyze a symbol for patterns using ML classifiers"""
        try:
            # Get recent data
            data = self._fetch_symbol_data(symbol)
            if data is None or len(data) < 20:
                return []
            
            # Convert to OHLC format for classifiers
            ohlc_data = self._convert_to_ohlc(data)
            
            patterns = []
            current_price = float(data['close'].iloc[-1])
            
            # Run each ML classifier
            for pattern_name, classifier in self.classifiers.items():
                try:
                    prediction, confidence = classifier.predict(ohlc_data)
                    
                    if prediction == 1 and confidence >= self.config['min_confidence']:
                        # Create pattern alert
                        alert = PatternAlert(
                            symbol=symbol,
                            timeframe=self.config['timeframe'],
                            pattern_type=pattern_name,
                            confidence=confidence,
                            timestamp=datetime.now(),
                            price_data=data['close'].tail(10).tolist(),
                            entry_price=current_price,
                            target_price=self._calculate_target_price(pattern_name, current_price, data),
                            stop_loss=self._calculate_stop_loss(pattern_name, current_price, data),
                            prediction_details={
                                'model_type': 'RandomForest',
                                'features_used': len(classifier.extract_features(ohlc_data) or {}),
                                'data_points': len(ohlc_data),
                                'volatility': data['close'].pct_change().std()
                            }
                        )
                        
                        patterns.append(alert)
                        
                        self.logger.info(f"🎯 {pattern_name.upper()} detected in {symbol}!")
                        self.logger.info(f"   💎 Confidence: {confidence:.1%}")
                        self.logger.info(f"   💰 Entry: ${current_price:.2f}")
                        self.logger.info(f"   🎯 Target: ${alert.target_price:.2f}")
                        self.logger.info(f"   🛡️  Stop Loss: ${alert.stop_loss:.2f}")
                        
                except Exception as e:
                    self.logger.error(f"Error running {pattern_name} classifier on {symbol}: {str(e)}")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing symbol {symbol}: {str(e)}")
            return []
    
    def _fetch_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch recent data for a symbol"""
        try:
            # Get data from the main application's API
            if hasattr(self.main_window, 'forex_api'):
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=self.config['lookback_periods'])).strftime('%Y-%m-%d')
                
                df = self.main_window.forex_api.get_data(
                    symbol=symbol,
                    interval=self.config['timeframe'],
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is not None and not df.empty:
                    return df.tail(self.config['lookback_periods'])
                    
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            
        return None
    
    def _convert_to_ohlc(self, df: pd.DataFrame) -> List[List[float]]:
        """Convert DataFrame to OHLC format expected by classifiers"""
        ohlc_data = []
        
        for _, row in df.iterrows():
            ohlc_data.append([
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume'])
            ])
            
        return ohlc_data
    
    def _calculate_target_price(self, pattern_type: str, current_price: float, data: pd.DataFrame) -> float:
        """Calculate target price based on pattern type"""
        atr = self._calculate_atr(data)
        
        if pattern_type in ['triple_bottom', 'double_bottom', 'inverse_head_shoulders']:
            # Bullish patterns - target above current price
            return current_price + (atr * 2.5)
        else:
            # Bearish patterns - target below current price
            return current_price - (atr * 2.5)
    
    def _calculate_stop_loss(self, pattern_type: str, current_price: float, data: pd.DataFrame) -> float:
        """Calculate stop loss based on pattern type"""
        atr = self._calculate_atr(data)
        
        if pattern_type in ['triple_bottom', 'double_bottom', 'inverse_head_shoulders']:
            # Bullish patterns - stop loss below current price
            return current_price - (atr * 1.5)
        else:
            # Bearish patterns - stop loss above current price
            return current_price + (atr * 1.5)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for volatility-based targets"""
        try:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=min(period, len(true_range))).mean().iloc[-1]
            
            return float(atr) if not pd.isna(atr) else data['close'].std() * 0.02
            
        except Exception:
            return data['close'].std() * 0.02
    
    def _process_pattern_alert(self, alert: PatternAlert):
        """Process a pattern alert"""
        alert_key = f"{alert.symbol}_{alert.pattern_type}"
        current_time = datetime.now()
        
        # Check cooldown period
        if alert_key in self.last_alerts:
            time_diff = (current_time - self.last_alerts[alert_key]).total_seconds()
            if time_diff < self.config['alert_cooldown']:
                return
        
        # Update last alert time
        self.last_alerts[alert_key] = current_time
        
        # Add to history
        self.pattern_history.append(alert)
        
        # Keep only recent history
        if len(self.pattern_history) > 100:
            self.pattern_history = self.pattern_history[-100:]
        
        # Send alert to GUI
        self._send_gui_alert(alert)
        
        # Log detailed alert
        self.logger.info("🚨 PATTERN ALERT GENERATED 🚨")
        self.logger.info(f"📈 Symbol: {alert.symbol}")
        self.logger.info(f"🔍 Pattern: {alert.pattern_type.replace('_', ' ').title()}")
        self.logger.info(f"💎 Confidence: {alert.confidence:.1%}")
        self.logger.info(f"🕐 Time: {alert.timestamp.strftime('%H:%M:%S')}")
        self.logger.info(f"💰 Entry: ${alert.entry_price:.2f}")
        self.logger.info(f"🎯 Target: ${alert.target_price:.2f} (+{((alert.target_price/alert.entry_price-1)*100):.1f}%)")
        self.logger.info(f"🛡️  Stop: ${alert.stop_loss:.2f} ({((alert.stop_loss/alert.entry_price-1)*100):.1f}%)")
        self.logger.info("=" * 60)
    
    def _send_gui_alert(self, alert: PatternAlert):
        """Send alert to GUI for display"""
        try:
            if hasattr(self.main_window, 'pattern_frame'):
                # Update pattern frame with new alert
                pattern_data = {
                    'type': alert.pattern_type.replace('_', ' ').title(),
                    'symbol': alert.symbol,
                    'confidence': alert.confidence,
                    'timestamp': alert.timestamp.strftime('%H:%M:%S'),
                    'entry_price': alert.entry_price,
                    'target_price': alert.target_price,
                    'stop_loss': alert.stop_loss,
                    'status': 'Active',
                    'profit_potential': f"{((alert.target_price/alert.entry_price-1)*100):.1f}%"
                }
                
                # Add to pattern display
                self.main_window.root.after(0, lambda: self._update_pattern_display(pattern_data))
                
        except Exception as e:
            self.logger.error(f"Error sending GUI alert: {str(e)}")
    
    def _update_pattern_display(self, pattern_data: Dict[str, Any]):
        """Update the pattern display in the GUI"""
        try:
            if hasattr(self.main_window, 'pattern_frame'):
                # Create a realistic pattern entry
                pattern_entry = {
                    'type': pattern_data['type'],
                    'confidence': pattern_data['confidence'],
                    'timestamp': pattern_data['timestamp'],
                    'symbol': pattern_data['symbol'],
                    'entry_price': pattern_data['entry_price'],
                    'target_price': pattern_data['target_price'],
                    'stop_loss': pattern_data['stop_loss'],
                    'profit_potential': pattern_data['profit_potential'],
                    'status': 'ML Detected'
                }
                
                # Add to detected patterns list
                if not hasattr(self.main_window, 'detected_patterns'):
                    self.main_window.detected_patterns = []
                    
                self.main_window.detected_patterns.append(pattern_entry)
                
                # Update the pattern frame
                if hasattr(self.main_window.pattern_frame, 'update_patterns'):
                    self.main_window.pattern_frame.update_patterns(self.main_window.detected_patterns)
                    
        except Exception as e:
            self.logger.error(f"Error updating pattern display: {str(e)}")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            'monitoring_active': self.monitoring_active,
            'symbols_monitored': len(self.config['symbols']),
            'classifiers_active': len(self.classifiers),
            'patterns_detected_today': len([p for p in self.pattern_history 
                                          if p.timestamp.date() == datetime.now().date()]),
            'total_patterns_detected': len(self.pattern_history),
            'last_alert_time': max([p.timestamp for p in self.pattern_history]) if self.pattern_history else None,
            'average_confidence': np.mean([p.confidence for p in self.pattern_history]) if self.pattern_history else 0,
            'pattern_types_detected': list(set([p.pattern_type for p in self.pattern_history]))
        }
    
    def get_recent_patterns(self, hours: int = 24) -> List[PatternAlert]:
        """Get patterns detected in the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [p for p in self.pattern_history if p.timestamp >= cutoff_time]
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update monitoring configuration"""
        self.config.update(new_config)
        self.logger.info(f"Monitoring configuration updated: {new_config}")
    
    def save_pattern_history(self, filepath: str):
        """Save pattern history to file"""
        try:
            data = []
            for pattern in self.pattern_history:
                data.append({
                    'symbol': pattern.symbol,
                    'pattern_type': pattern.pattern_type,
                    'confidence': pattern.confidence,
                    'timestamp': pattern.timestamp.isoformat(),
                    'entry_price': pattern.entry_price,
                    'target_price': pattern.target_price,
                    'stop_loss': pattern.stop_loss
                })
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Pattern history saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving pattern history: {str(e)}")