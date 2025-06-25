"""
Advanced Real-Time Market Monitoring System
Software Architecture: Adaptive Multi-Symbol Pattern Detection Engine

This module implements a sophisticated real-time monitoring system that:
- Adaptively adjusts fetch intervals based on market volatility
- Supports multiple symbols and timeframes simultaneously
- Maintains sliding data buffers for efficient processing
- Uses ML models for enhanced pattern recognition
- Implements feedback-driven sensitivity adjustment
- Provides thread-safe GUI updates with error resilience
"""

import asyncio
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from collections import deque


@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring system"""
    symbols: List[str]
    timeframes: List[str]  # ['daily', 'weekly', 'monthly'] - API supported intervals
    base_interval: int = 300  # Base fetch interval in seconds (5 minutes for daily data)
    min_interval: int = 60   # Minimum interval during high volatility (1 minute)
    max_interval: int = 1800  # Maximum interval during low volatility (30 minutes)
    buffer_size: int = 200   # Number of candles to maintain per symbol/timeframe
    confidence_threshold: float = 0.80  # Minimum confidence for alerts
    volatility_window: int = 20  # Period for volatility calculation
    max_workers: int = 4     # Thread pool size for data fetching


@dataclass
class MarketSnapshot:
    """Represents a complete market state for a symbol/timeframe"""
    symbol: str
    timeframe: str
    data: pd.DataFrame
    last_update: datetime
    volatility: float
    patterns: List[Dict]
    indicators: Dict[str, float]


class AdaptiveDataBuffer:
    """Sliding buffer that maintains recent market data efficiently"""
    
    def __init__(self, symbol: str, timeframe: str, max_size: int = 200):
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_size = max_size
        self.data = deque(maxlen=max_size)
        self.last_timestamp = None
        
    def add_candle(self, candle_data: Dict) -> bool:
        """Add new candle data, returns True if data is new"""
        timestamp = candle_data.get('timestamp')
        
        # Avoid duplicate data
        if timestamp == self.last_timestamp:
            return False
            
        self.data.append(candle_data)
        self.last_timestamp = timestamp
        return True
        
    def get_dataframe(self) -> pd.DataFrame:
        """Convert buffer to pandas DataFrame for analysis"""
        if not self.data:
            return pd.DataFrame()
            
        df = pd.DataFrame(list(self.data))
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        return df
        
    def calculate_volatility(self, window: int = 20) -> float:
        """Calculate ATR-based volatility for adaptive intervals"""
        df = self.get_dataframe()
        if len(df) < window:
            return 0.5  # Default medium volatility
            
        # Calculate Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean().iloc[-1]
        
        # Normalize ATR to 0-1 scale for volatility scoring
        price = df['close'].iloc[-1]
        volatility_score = min(atr / price, 1.0) if price > 0 else 0.5
        
        return volatility_score


class VolatilityAdaptiveScheduler:
    """Adaptive scheduler that adjusts fetch intervals based on market volatility"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.symbol_intervals = {}  # symbol -> current interval
        
    def calculate_adaptive_interval(self, volatility: float) -> int:
        """Calculate fetch interval based on volatility (0-1 scale)"""
        # High volatility = shorter intervals, Low volatility = longer intervals
        interval_range = self.config.max_interval - self.config.min_interval
        adjusted_interval = self.config.max_interval - (volatility * interval_range)
        
        return max(self.config.min_interval, min(self.config.max_interval, int(adjusted_interval)))
        
    def get_next_fetch_time(self, symbol: str, volatility: float) -> float:
        """Get timestamp for next data fetch for given symbol"""
        interval = self.calculate_adaptive_interval(volatility)
        self.symbol_intervals[symbol] = interval
        
        return time.time() + interval
        
    def get_current_interval(self, symbol: str) -> int:
        """Get current fetch interval for symbol"""
        return self.symbol_intervals.get(symbol, self.config.base_interval)


class PatternFeedbackEngine:
    """Learns from pattern success/failure to adjust detection sensitivity"""
    
    def __init__(self):
        self.pattern_history = {}  # pattern_id -> success/failure records
        self.sensitivity_adjustments = {}  # pattern_type -> sensitivity modifier
        self.feedback_window = 100  # Number of patterns to consider for learning
        
    def record_pattern_outcome(self, pattern_id: str, pattern_type: str, success: bool):
        """Record whether a pattern prediction was successful"""
        if pattern_id not in self.pattern_history:
            self.pattern_history[pattern_id] = {
                'type': pattern_type,
                'outcomes': deque(maxlen=self.feedback_window)
            }
            
        self.pattern_history[pattern_id]['outcomes'].append(success)
        self._update_sensitivity(pattern_type)
        
    def _update_sensitivity(self, pattern_type: str):
        """Adjust sensitivity based on recent success rate"""
        # Collect recent outcomes for this pattern type
        recent_outcomes = []
        for record in self.pattern_history.values():
            if record['type'] == pattern_type:
                recent_outcomes.extend(list(record['outcomes']))
                
        if len(recent_outcomes) < 10:  # Need minimum data
            return
            
        success_rate = sum(recent_outcomes) / len(recent_outcomes)
        
        # Adjust sensitivity: Low success rate = higher threshold (less sensitive)
        if success_rate < 0.6:
            adjustment = 0.1  # Increase threshold
        elif success_rate > 0.8:
            adjustment = -0.05  # Decrease threshold (more sensitive)
        else:
            adjustment = 0.0
            
        self.sensitivity_adjustments[pattern_type] = adjustment
        
    def get_adjusted_threshold(self, pattern_type: str, base_threshold: float) -> float:
        """Get adjusted confidence threshold for pattern type"""
        adjustment = self.sensitivity_adjustments.get(pattern_type, 0.0)
        return max(0.5, min(0.95, base_threshold + adjustment))


class AdvancedRealtimeMonitor:
    """
    Main real-time monitoring system implementing adaptive multi-symbol pattern detection.
    
    Architecture Components:
    - Adaptive fetch scheduling based on volatility
    - Multi-threaded data acquisition 
    - Sliding buffer management
    - ML-enhanced pattern detection
    - Feedback-driven sensitivity adjustment
    - Thread-safe GUI integration
    """
    
    def __init__(self, main_window, config: MonitoringConfig):
        self.main_window = main_window
        self.config = config
        self.logger = main_window.logger
        
        # Core components
        self.buffers = {}  # (symbol, timeframe) -> AdaptiveDataBuffer
        self.scheduler = VolatilityAdaptiveScheduler(config)
        self.feedback_engine = PatternFeedbackEngine()
        
        # Threading infrastructure
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.monitoring_active = False
        self.data_queue = queue.Queue()
        self.pattern_queue = queue.Queue()
        
        # State tracking
        self.last_snapshots = {}  # symbol -> MarketSnapshot
        self.fetch_schedule = {}  # symbol -> next_fetch_timestamp
        
        # GUI update mechanism
        self.gui_update_queue = queue.Queue()
        
        self._initialize_buffers()
        
    def _initialize_buffers(self):
        """Initialize data buffers for all symbol/timeframe combinations"""
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                key = (symbol, timeframe)
                self.buffers[key] = AdaptiveDataBuffer(
                    symbol, timeframe, self.config.buffer_size
                )
                
    def start_monitoring(self):
        """Start the advanced real-time monitoring system"""
        if self.monitoring_active:
            self.logger.warning("Advanced monitoring already active")
            return
            
        self.monitoring_active = True
        self.logger.info("Starting advanced real-time monitoring system")
        
        # Start background workers
        threading.Thread(target=self._data_acquisition_worker, daemon=True).start()
        threading.Thread(target=self._pattern_detection_worker, daemon=True).start()
        threading.Thread(target=self._gui_update_worker, daemon=True).start()
        
        # Start main coordination loop
        threading.Thread(target=self._coordination_loop, daemon=True).start()
        
        self.main_window.update_status("Advanced real-time monitoring started")
        
    def stop_monitoring(self):
        """Stop the monitoring system gracefully"""
        self.monitoring_active = False
        self.executor.shutdown(wait=True)
        self.logger.info("Advanced real-time monitoring stopped")
        self.main_window.update_status("Real-time monitoring stopped")
        
    def _coordination_loop(self):
        """Main coordination loop that schedules data fetching"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Check which symbols need data updates
                for symbol in self.config.symbols:
                    next_fetch = self.fetch_schedule.get(symbol, 0)
                    
                    if current_time >= next_fetch:
                        # Calculate current volatility for adaptive scheduling
                        volatility = self._get_symbol_volatility(symbol)
                        
                        # Schedule data fetch
                        self.executor.submit(self._fetch_symbol_data, symbol)
                        
                        # Set next fetch time based on volatility
                        self.fetch_schedule[symbol] = self.scheduler.get_next_fetch_time(
                            symbol, volatility
                        )
                        
                        self.logger.debug(f"Scheduled fetch for {symbol}, volatility: {volatility:.3f}")
                
                # Sleep briefly before next coordination cycle
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                time.sleep(5)  # Back off on errors
                
    def _fetch_symbol_data(self, symbol: str):
        """Fetch data for all timeframes of a symbol"""
        try:
            for timeframe in self.config.timeframes:
                # For real-time monitoring, get substantial data to support charts
                data = self.main_window.forex_api.get_forex_data(
                    symbol, 
                    interval=timeframe, 
                    outputsize="full"  # Get comprehensive data for better charts
                )
                
                if data is not None and not data.empty and len(data) >= 5:
                    # Populate buffer with recent data for context
                    buffer_key = (symbol, timeframe)
                    if buffer_key in self.buffers:
                        # Add recent data points to buffer (last 100 for comprehensive analysis)
                        recent_data = data.tail(100)
                        buffer_updated = False
                        
                        for idx, row in recent_data.iterrows():
                            candle = {
                                'timestamp': idx,
                                'open': float(row['open']),
                                'high': float(row['high']), 
                                'low': float(row['low']),
                                'close': float(row['close']),
                                'volume': float(row.get('volume', 0))
                            }
                            if self.buffers[buffer_key].add_candle(candle):
                                buffer_updated = True
                        
                        if buffer_updated:
                            # Queue the complete dataset for processing
                            self.data_queue.put((symbol, timeframe, data))
                            self.logger.info(f"Updated {symbol} {timeframe} buffer with {len(recent_data)} data points")
                else:
                    self.logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(data) if data is not None and not data.empty else 0} points")
                            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            
    def _data_acquisition_worker(self):
        """Background worker that processes fetched data"""
        while self.monitoring_active:
            try:
                # Get new data from queue with timeout
                symbol, timeframe, data = self.data_queue.get(timeout=1)
                
                # Process the data - data can be either a single candle dict or DataFrame
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Full dataset provided - use directly
                    processed_data = self.main_window.data_processor.process_data(data)
                    volatility = self._calculate_volatility(data)
                else:
                    # Single candle or insufficient data - get from buffer
                    buffer_key = (symbol, timeframe)
                    if buffer_key not in self.buffers:
                        continue
                        
                    df = self.buffers[buffer_key].get_dataframe()
                    if df.empty or len(df) < 5:  # Need minimum data for processing
                        self.logger.debug(f"Insufficient buffer data for {symbol} {timeframe}: {len(df)} points")
                        continue
                        
                    processed_data = self.main_window.data_processor.process_data(df)
                    volatility = self.buffers[buffer_key].calculate_volatility()
                
                if not processed_data.empty and len(processed_data) >= 2:
                    # Create market snapshot with sufficient data
                    snapshot = MarketSnapshot(
                        symbol=symbol,
                        timeframe=timeframe,
                        data=processed_data,
                        last_update=datetime.now(),
                        volatility=volatility,
                        patterns=[],  # Will be filled by pattern detection
                        indicators=self._extract_latest_indicators(processed_data)
                    )
                    
                    # Queue for pattern detection
                    self.pattern_queue.put(snapshot)
                    
                    # Also queue for GUI update
                    self.gui_update_queue.put({
                        'type': 'data_update',
                        'snapshot': snapshot
                    })
                        
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in data acquisition worker: {e}")
                
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility for a dataset"""
        try:
            if len(data) < 5:
                return 0.5  # Default moderate volatility
            returns = data['close'].pct_change().dropna()
            return float(returns.std() * 100) if len(returns) > 0 else 0.5
        except Exception:
            return 0.5
                
    def _pattern_detection_worker(self):
        """Background worker that detects patterns in processed data"""
        while self.monitoring_active:
            try:
                # Get market snapshot for pattern analysis
                snapshot = self.pattern_queue.get(timeout=1)
                
                # Run pattern detection
                patterns = self.main_window.pattern_detector.detect_all_patterns(snapshot.data)
                
                # Apply feedback-adjusted confidence thresholds
                high_confidence_patterns = []
                for pattern in patterns:
                    pattern_type = pattern.get('type', '')
                    confidence = pattern.get('confidence', 0)
                    
                    adjusted_threshold = self.feedback_engine.get_adjusted_threshold(
                        pattern_type, self.config.confidence_threshold
                    )
                    
                    if confidence >= adjusted_threshold:
                        high_confidence_patterns.append(pattern)
                
                # Update snapshot with detected patterns
                snapshot.patterns = high_confidence_patterns
                self.last_snapshots[snapshot.symbol] = snapshot
                
                # Queue GUI updates for high-confidence patterns
                if high_confidence_patterns:
                    self.gui_update_queue.put({
                        'type': 'pattern_alert',
                        'snapshot': snapshot,
                        'patterns': high_confidence_patterns
                    })
                    
                # Queue regular data updates
                self.gui_update_queue.put({
                    'type': 'data_update',
                    'snapshot': snapshot
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in pattern detection worker: {e}")
                
    def _gui_update_worker(self):
        """Background worker that handles thread-safe GUI updates"""
        while self.monitoring_active:
            try:
                # Get GUI update request
                update_request = self.gui_update_queue.get(timeout=1)
                
                # Schedule GUI update on main thread
                self.main_window.root.after(0, lambda: self._process_gui_update(update_request))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in GUI update worker: {e}")
                
    def _process_gui_update(self, update_request: Dict):
        """Process GUI update request on main thread"""
        try:
            update_type = update_request['type']
            snapshot = update_request['snapshot']
            
            if update_type == 'pattern_alert':
                # Handle pattern alerts
                patterns = update_request['patterns']
                self._show_advanced_pattern_alert(snapshot, patterns)
                
                # Create intelligent alerts through alert system
                for pattern in patterns:
                    suggested_action = self.main_window.get_suggested_action(pattern)
                    self.main_window.alert_system.create_pattern_alert(
                        pattern_type=pattern.get('type', 'unknown'),
                        symbol=snapshot.symbol,
                        confidence=pattern.get('confidence', 0),
                        suggested_action=suggested_action,
                        pattern_data=pattern
                    )
                
            elif update_type == 'data_update':
                # Update current data if this symbol is being displayed
                if snapshot.symbol == self.main_window.current_symbol:
                    self.main_window.current_data = snapshot.data
                    self.main_window.chart_frame.update_chart(snapshot.data)
                    
                    # Update status with current price and volatility
                    current_price = snapshot.data['close'].iloc[-1]
                    interval = self.scheduler.get_current_interval(snapshot.symbol)
                    
                    status = f"Real-time: {snapshot.symbol} @ {current_price:.5f} "
                    status += f"(Vol: {snapshot.volatility:.3f}, Interval: {interval}s)"
                    self.main_window.update_status(status)
                    
        except Exception as e:
            self.logger.error(f"Error processing GUI update: {e}")
            
    def _show_advanced_pattern_alert(self, snapshot: MarketSnapshot, patterns: List[Dict]):
        """Show advanced pattern alert with enhanced information"""
        try:
            for pattern in patterns:
                pattern_type = pattern.get('type', 'Unknown')
                confidence = pattern.get('confidence', 0)
                signal = pattern.get('signal', 'Neutral')
                
                # Enhanced alert message with market context
                message = f"🎯 Advanced Pattern Alert\n\n"
                message += f"Symbol: {snapshot.symbol} ({snapshot.timeframe})\n"
                message += f"Pattern: {pattern_type}\n"
                message += f"Signal: {signal}\n"
                message += f"Confidence: {confidence:.1%}\n"
                message += f"Market Volatility: {snapshot.volatility:.3f}\n"
                message += f"Time: {snapshot.last_update.strftime('%H:%M:%S')}\n\n"
                
                # Add technical context
                if snapshot.indicators:
                    message += "Technical Context:\n"
                    for indicator, value in snapshot.indicators.items():
                        message += f"• {indicator}: {value:.3f}\n"
                
                self.logger.info(f"Advanced pattern alert: {pattern_type} on {snapshot.symbol}")
                
                # Show non-blocking alert if GUI notifications enabled
                if self.main_window.config.get('show_pattern_notifications', True):
                    # Create enhanced popup with more information
                    self._create_enhanced_alert_popup(message, pattern, snapshot)
                    
        except Exception as e:
            self.logger.error(f"Error showing advanced pattern alert: {e}")
            
    def _create_enhanced_alert_popup(self, message: str, pattern: Dict, snapshot: MarketSnapshot):
        """Create enhanced alert popup with action buttons"""
        import tkinter as tk
        from tkinter import ttk
        
        alert_window = tk.Toplevel(self.main_window.root)
        alert_window.title("Advanced Pattern Alert")
        alert_window.geometry("400x300")
        alert_window.transient(self.main_window.root)
        
        # Center the window
        alert_window.update_idletasks()
        x = (alert_window.winfo_screenwidth() // 2) - (alert_window.winfo_width() // 2)
        y = (alert_window.winfo_screenheight() // 2) - (alert_window.winfo_height() // 2)
        alert_window.geometry(f"+{x}+{y}")
        
        main_frame = ttk.Frame(alert_window, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Alert message
        text_widget = tk.Text(main_frame, wrap=tk.WORD, height=12, width=45)
        text_widget.insert(tk.END, message)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="View Pattern", 
                  command=lambda: self._show_pattern_details(pattern, snapshot)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Dismiss", 
                  command=alert_window.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Auto-close after 30 seconds
        alert_window.after(30000, alert_window.destroy)
        
    def _show_pattern_details(self, pattern: Dict, snapshot: MarketSnapshot):
        """Show detailed pattern analysis"""
        # Switch to the symbol and update display
        self.main_window.current_symbol = snapshot.symbol
        self.main_window.current_data = snapshot.data
        self.main_window.detected_patterns = [pattern]
        
        # Update chart and pattern displays
        self.main_window.chart_frame.update_chart(snapshot.data)
        self.main_window.chart_frame.add_pattern_overlays([pattern])
        self.main_window.pattern_frame.update_patterns([pattern])
        
    def _get_symbol_volatility(self, symbol: str) -> float:
        """Get current volatility for a symbol across timeframes"""
        volatilities = []
        
        for timeframe in self.config.timeframes:
            buffer_key = (symbol, timeframe)
            if buffer_key in self.buffers:
                vol = self.buffers[buffer_key].calculate_volatility()
                volatilities.append(vol)
                
        return np.mean(volatilities) if volatilities else 0.5
        
    def _extract_latest_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract latest values of technical indicators"""
        indicators = {}
        
        if not df.empty:
            # Get latest values of available indicators
            for col in df.columns:
                if col in ['rsi', 'macd', 'sma_20', 'sma_50', 'bb_upper', 'bb_lower']:
                    latest_value = df[col].iloc[-1]
                    if not pd.isna(latest_value):
                        indicators[col] = float(latest_value)
                        
        return indicators
        
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status"""
        return {
            'active': self.monitoring_active,
            'symbols_monitored': len(self.config.symbols),
            'timeframes': self.config.timeframes,
            'total_buffers': len(self.buffers),
            'recent_snapshots': len(self.last_snapshots),
            'current_intervals': dict(self.scheduler.symbol_intervals),
            'feedback_adjustments': dict(self.feedback_engine.sensitivity_adjustments)
        }