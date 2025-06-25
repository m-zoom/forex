"""
Simplified Async Real-time Monitor
Replaces complex multi-threading with asyncio + bounded thread pools
"""

import asyncio
import aiohttp
import time
import queue
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np

@dataclass
class MarketSnapshot:
    symbol: str
    timeframe: str
    data: pd.DataFrame
    last_update: datetime
    volatility: float
    patterns: List[Dict]

class AsyncRealtimeMonitor:
    """Simplified real-time monitoring with asyncio + thread pools"""
    
    def __init__(self, config, main_window):
        self.config = config
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
        
        # Asyncio components
        self.session = None
        self.monitoring_active = False
        self.data_ingestion_task = None
        
        # Thread pool for CPU-bound pattern detection
        self.pattern_executor = ThreadPoolExecutor(
            max_workers=min(4, len(config.symbols)),
            thread_name_prefix="pattern_detector"
        )
        
        # Simple data storage
        self.symbol_data = {}  # symbol -> latest DataFrame
        self.volatility_cache = {}  # symbol -> volatility
        self.fetch_intervals = {}  # symbol -> seconds
        
        # GUI update queue with listener
        self.gui_queue = queue.Queue(maxsize=100)
        self.gui_listener = None
        
    async def start_monitoring(self):
        """Start async monitoring system"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.logger.info("Starting simplified async monitoring")
        
        # Create HTTP session for API calls
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        
        # Start GUI queue listener
        self._start_gui_listener()
        
        # Start main data ingestion coroutine
        self.data_ingestion_task = asyncio.create_task(
            self._data_ingestion_coroutine()
        )
        
        self.main_window.update_status("Async monitoring started")
        
    async def stop_monitoring(self):
        """Stop monitoring gracefully"""
        self.monitoring_active = False
        
        if self.data_ingestion_task:
            self.data_ingestion_task.cancel()
            try:
                await self.data_ingestion_task
            except asyncio.CancelledError:
                pass
                
        if self.session:
            await self.session.close()
            
        self.pattern_executor.shutdown(wait=True, timeout=10)
        
        if self.gui_listener:
            self.gui_listener.stop()
            
        self.logger.info("Async monitoring stopped")
        
    async def _data_ingestion_coroutine(self):
        """Single coroutine for all data ingestion"""
        self.logger.info("Data ingestion coroutine started")
        
        while self.monitoring_active:
            try:
                # Create tasks for all symbols that need updates
                fetch_tasks = []
                current_time = time.time()
                
                for symbol in self.config.symbols:
                    # Check if symbol needs update
                    last_fetch = getattr(self, f'last_fetch_{symbol}', 0)
                    interval = self.fetch_intervals.get(symbol, 60)
                    
                    if current_time - last_fetch >= interval:
                        task = asyncio.create_task(
                            self._fetch_symbol_data_async(symbol)
                        )
                        fetch_tasks.append((symbol, task))
                        setattr(self, f'last_fetch_{symbol}', current_time)
                
                # Wait for all fetch tasks to complete
                if fetch_tasks:
                    results = await asyncio.gather(
                        *[task for _, task in fetch_tasks],
                        return_exceptions=True
                    )
                    
                    # Process results
                    for (symbol, _), result in zip(fetch_tasks, results):
                        if isinstance(result, Exception):
                            self.logger.error(f"Fetch failed for {symbol}: {result}")
                        elif result is not None:
                            # Submit to pattern detection thread pool
                            self.pattern_executor.submit(
                                self._process_symbol_data, symbol, result
                            )
                
                # Sleep before next cycle
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Data ingestion error: {e}")
                await asyncio.sleep(10)  # Back off on errors
                
        self.logger.info("Data ingestion coroutine stopped")
        
    async def _fetch_symbol_data_async(self, symbol: str) -> Optional[pd.DataFrame]:
        """Async data fetching for single symbol"""
        try:
            # Get date range for API call
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Build API request
            params = {
                'ticker': symbol,
                'interval': '5min',
                'start_date': start_date,
                'end_date': end_date,
                'api_key': self.config.get('API', 'financial_datasets_api_key')
            }
            
            # Make async HTTP request
            async with self.session.get(
                'https://api.financialdatasets.ai/financial-data',
                params=params
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return self._parse_api_response(data)
                elif response.status == 429:
                    # Rate limited - adjust interval
                    self.fetch_intervals[symbol] = self.fetch_intervals.get(symbol, 60) * 2
                    self.logger.warning(f"Rate limited for {symbol}, increasing interval")
                    return None
                else:
                    self.logger.warning(f"API error {response.status} for {symbol}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Async fetch error for {symbol}: {e}")
            return None
            
    def _parse_api_response(self, data: Dict) -> Optional[pd.DataFrame]:
        """Parse API response to DataFrame"""
        try:
            if 'prices' not in data or not data['prices']:
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame([{
                'open': float(price['open']),
                'high': float(price['high']),
                'low': float(price['low']),
                'close': float(price['close']),
                'volume': int(price['volume'])
            } for price in data['prices']])
            
            # Set datetime index
            df.index = pd.to_datetime([p['time'] for p in data['prices']])
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"API response parsing error: {e}")
            return None
            
    def _process_symbol_data(self, symbol: str, raw_data: pd.DataFrame):
        """Process data in thread pool (CPU-bound)"""
        try:
            # Apply technical indicators
            processed_data = self.main_window.data_processor.process_data(raw_data)
            
            # Calculate volatility
            volatility = self._calculate_volatility(processed_data)
            self.volatility_cache[symbol] = volatility
            
            # Adjust fetch interval based on volatility
            if volatility > 0.02:  # High volatility
                self.fetch_intervals[symbol] = 15  # 15 seconds
            elif volatility > 0.01:  # Medium volatility  
                self.fetch_intervals[symbol] = 30  # 30 seconds
            else:  # Low volatility
                self.fetch_intervals[symbol] = 60  # 1 minute
                
            # Store processed data
            self.symbol_data[symbol] = processed_data
            
            # Detect patterns
            patterns = self.main_window.pattern_detector.detect_all_patterns(processed_data)
            
            # Filter high-confidence patterns
            high_confidence_patterns = [
                p for p in patterns 
                if p.get('confidence', 0) >= self.config.get('PATTERNS', 'min_confidence', 60)
            ]
            
            # Create market snapshot
            snapshot = MarketSnapshot(
                symbol=symbol,
                timeframe='5min',
                data=processed_data,
                last_update=datetime.now(),
                volatility=volatility,
                patterns=high_confidence_patterns
            )
            
            # Queue GUI updates (non-blocking)
            self._queue_gui_update(snapshot)
            
        except Exception as e:
            self.logger.error(f"Data processing error for {symbol}: {e}")
            
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate recent volatility for adaptive scheduling"""
        if len(df) < 20:
            return 0.02  # Default volatility
            
        # 20-period price changes
        prices = df['close'].tail(20)
        price_changes = prices.pct_change().dropna()
        
        return price_changes.std()
        
    def _queue_gui_update(self, snapshot: MarketSnapshot):
        """Queue GUI update (non-blocking)"""
        try:
            # Data update
            self.gui_queue.put_nowait({
                'type': 'data_update',
                'snapshot': snapshot
            })
            
            # Pattern alerts for high-confidence patterns
            if snapshot.patterns:
                self.gui_queue.put_nowait({
                    'type': 'pattern_alert',
                    'snapshot': snapshot,
                    'patterns': snapshot.patterns
                })
                
        except queue.Full:
            # Drop updates if queue is full (prevents blocking)
            self.logger.warning("GUI queue full, dropping update")
            
    def _start_gui_listener(self):
        """Start GUI queue listener thread"""
        self.gui_listener = GUIQueueListener(
            self.gui_queue,
            self.main_window,
            self.logger
        )
        self.gui_listener.start()


class GUIQueueListener:
    """Non-blocking GUI update listener"""
    
    def __init__(self, gui_queue: queue.Queue, main_window, logger):
        self.gui_queue = gui_queue
        self.main_window = main_window
        self.logger = logger
        self.active = False
        self.thread = None
        
    def start(self):
        """Start listener thread"""
        self.active = True
        self.thread = threading.Thread(
            target=self._listener_loop,
            name="gui_listener",
            daemon=True
        )
        self.thread.start()
        
    def stop(self):
        """Stop listener thread"""
        self.active = False
        if self.thread:
            self.thread.join(timeout=5)
            
    def _listener_loop(self):
        """Main listener loop"""
        while self.active:
            try:
                # Get update with timeout
                update = self.gui_queue.get(timeout=1)
                
                # Schedule GUI update on main thread
                self.main_window.root.after(
                    0, lambda u=update: self._process_gui_update(u)
                )
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"GUI listener error: {e}")
                
    def _process_gui_update(self, update: Dict):
        """Process GUI update on main thread"""
        try:
            update_type = update['type']
            
            if update_type == 'data_update':
                snapshot = update['snapshot']
                
                # Update chart if this is the current symbol
                if snapshot.symbol == getattr(self.main_window, 'current_symbol', None):
                    self.main_window.current_data = snapshot.data
                    self.main_window.chart_frame.update_chart(snapshot.data)
                    
                    # Update status
                    current_price = snapshot.data['close'].iloc[-1]
                    status = f"Real-time: {snapshot.symbol} @ {current_price:.5f} (Vol: {snapshot.volatility:.3f})"
                    self.main_window.update_status(status)
                    
            elif update_type == 'pattern_alert':
                snapshot = update['snapshot']
                patterns = update['patterns']
                
                # Create alerts through alert system
                for pattern in patterns:
                    suggested_action = self.main_window.get_suggested_action(pattern)
                    self.main_window.alert_system.create_pattern_alert(
                        pattern_type=pattern.get('type', 'unknown'),
                        symbol=snapshot.symbol,
                        confidence=pattern.get('confidence', 0),
                        suggested_action=suggested_action,
                        pattern_data=pattern
                    )
                    
                # Update pattern display if current symbol
                if snapshot.symbol == getattr(self.main_window, 'current_symbol', None):
                    self.main_window.pattern_frame.update_patterns(patterns)
                    self.main_window.chart_frame.add_pattern_overlays(patterns)
                    
        except Exception as e:
            self.logger.error(f"GUI update processing error: {e}")


# Integration with main window
class AsyncMonitoringMixin:
    """Mixin to add async monitoring to main window"""
    
    def __init_async_monitoring__(self):
        """Initialize async monitoring system"""
        self.async_monitor = AsyncRealtimeMonitor(self.config, self)
        self.monitoring_loop = None
        
    def start_async_realtime(self):
        """Start async real-time monitoring"""
        if self.monitoring_loop and not self.monitoring_loop.is_closed():
            return
            
        # Create new event loop for monitoring
        self.monitoring_loop = asyncio.new_event_loop()
        
        def run_monitoring():
            asyncio.set_event_loop(self.monitoring_loop)
            self.monitoring_loop.run_until_complete(
                self.async_monitor.start_monitoring()
            )
            
        # Start monitoring in separate thread
        monitor_thread = threading.Thread(
            target=run_monitoring,
            name="async_monitor",
            daemon=True
        )
        monitor_thread.start()
        
        self.update_status("Async real-time monitoring started")
        
    def stop_async_realtime(self):
        """Stop async real-time monitoring"""
        if self.monitoring_loop and not self.monitoring_loop.is_closed():
            # Schedule stop in the monitoring loop
            asyncio.run_coroutine_threadsafe(
                self.async_monitor.stop_monitoring(),
                self.monitoring_loop
            )
            
            # Close the loop
            self.monitoring_loop.call_soon_threadsafe(self.monitoring_loop.stop)
            
        self.update_status("Async monitoring stopped")
        
    def get_current_data_async(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get current data for symbol"""
        return self.async_monitor.symbol_data.get(symbol)
        
    def get_current_volatility_async(self, symbol: str) -> float:
        """Get current volatility for symbol"""
        return self.async_monitor.volatility_cache.get(symbol, 0.02)