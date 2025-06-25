# Threading Architecture Guide

## Overview of Threading Strategy

The Forex Pattern Recognition System has been refactored from complex multi-threading to a simplified async + bounded thread pool architecture for better performance and maintainability.

## ⚠️ ARCHITECTURE CHANGE NOTICE
**This guide documents the OLD threading system. The NEW simplified system is implemented in `models/async_realtime_monitor.py`**

**For current implementation, see:**
- `THREADING_REFACTOR_SUMMARY.md` - New architecture details
- `models/async_realtime_monitor.py` - Implementation
- `AsyncMonitoringMixin` - Integration pattern

---

## Legacy Threading Strategy (Deprecated)

The original system used a sophisticated multi-threading architecture (now replaced):

### Core Threading Principles

1. **Main Thread Isolation**: All GUI operations occur on the main thread
2. **Worker Thread Specialization**: Dedicated threads for specific tasks
3. **Queue-Based Communication**: Thread-safe communication via queues
4. **Graceful Error Handling**: Robust error recovery in background threads

## Thread Architecture Diagram

```
Main Thread (GUI)
├── Event Loop (tkinter)
├── GUI Updates
└── User Interactions

Background Workers
├── Coordination Thread
│   ├── Schedule Management
│   ├── Volatility Monitoring
│   └── Task Distribution
├── Data Acquisition Thread Pool
│   ├── API Requests
│   ├── Data Processing
│   └── Buffer Management
├── Pattern Detection Thread
│   ├── Real-time Analysis
│   ├── ML Processing
│   └── Confidence Scoring
└── GUI Update Thread
    ├── Queue Processing
    ├── Thread-safe Updates
    └── Alert Management
```

## Implementation Details

### 1. Main Thread Management

#### GUI Thread Safety
```python
class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.update_queue = queue.Queue()
        
        # Schedule periodic queue processing
        self._schedule_queue_processing()
        
    def _schedule_queue_processing(self):
        """Process background updates on main thread"""
        try:
            # Process all queued updates
            while True:
                update = self.update_queue.get_nowait()
                self._process_update_safely(update)
        except queue.Empty:
            pass
        finally:
            # Reschedule for next processing cycle
            self.root.after(100, self._schedule_queue_processing)
            
    def _process_update_safely(self, update):
        """Handle updates with error protection"""
        try:
            update_type = update['type']
            
            if update_type == 'chart_update':
                self.chart_frame.update_chart(update['data'])
            elif update_type == 'pattern_alert':
                self.alert_system.trigger_alert(update['alert'])
            elif update_type == 'status_update':
                self.update_status(update['message'])
                
        except Exception as e:
            self.logger.error(f"GUI update error: {e}")
            # Continue processing other updates
```

#### Background Task Coordination
```python
def start_background_task(self, task_function, *args, **kwargs):
    """Start background task with proper error handling"""
    
    def task_wrapper():
        try:
            result = task_function(*args, **kwargs)
            
            # Queue result for main thread
            self.update_queue.put({
                'type': 'task_complete',
                'result': result,
                'task_name': task_function.__name__
            })
            
        except Exception as e:
            # Queue error for main thread handling
            self.update_queue.put({
                'type': 'task_error',
                'error': str(e),
                'task_name': task_function.__name__
            })
    
    # Start in background thread
    thread = threading.Thread(target=task_wrapper, daemon=True)
    thread.start()
    return thread
```

### 2. Advanced Real-Time Monitor Threading

#### Coordination Thread Implementation
```python
class AdvancedRealtimeMonitor:
    def __init__(self, config, main_window):
        self.config = config
        self.main_window = main_window
        self.monitoring_active = False
        
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix="forex_worker"
        )
        
        # Communication queues
        self.data_queue = queue.Queue(maxsize=1000)
        self.pattern_queue = queue.Queue(maxsize=500)
        self.gui_queue = queue.Queue(maxsize=200)
        
        # Thread control
        self.worker_threads = []
        self.shutdown_event = threading.Event()
        
    def start_monitoring(self):
        """Start all monitoring threads"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.shutdown_event.clear()
        
        # Start worker threads
        self.worker_threads = [
            threading.Thread(target=self._coordination_loop, 
                           name="coordinator", daemon=True),
            threading.Thread(target=self._data_acquisition_worker, 
                           name="data_worker", daemon=True),
            threading.Thread(target=self._pattern_detection_worker, 
                           name="pattern_worker", daemon=True),
            threading.Thread(target=self._gui_update_worker, 
                           name="gui_worker", daemon=True)
        ]
        
        for thread in self.worker_threads:
            thread.start()
            
        self.logger.info("All monitoring threads started")

    def _coordination_loop(self):
        """Main coordination thread - schedules all data fetching"""
        self.logger.info("Coordination thread started")
        
        while self.monitoring_active and not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                
                # Check each symbol's schedule
                for symbol in self.config.symbols:
                    next_fetch = self.fetch_schedule.get(symbol, 0)
                    
                    if current_time >= next_fetch:
                        # Submit fetch task to thread pool
                        future = self.executor.submit(
                            self._fetch_symbol_data_async, symbol
                        )
                        
                        # Calculate next fetch time based on volatility
                        volatility = self._get_symbol_volatility(symbol)
                        self.fetch_schedule[symbol] = (
                            self.scheduler.get_next_fetch_time(symbol, volatility)
                        )
                        
                        self.logger.debug(f"Scheduled {symbol} fetch, "
                                        f"volatility: {volatility:.3f}")
                
                # Coordination loop runs every second
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Coordination loop error: {e}")
                time.sleep(5)  # Back off on errors
                
        self.logger.info("Coordination thread stopped")
```

#### Data Acquisition Worker
```python
def _data_acquisition_worker(self):
    """Background thread for processing fetched data"""
    self.logger.info("Data acquisition worker started")
    
    while self.monitoring_active and not self.shutdown_event.is_set():
        try:
            # Get new data with timeout
            data_item = self.data_queue.get(timeout=1)
            
            symbol = data_item['symbol']
            timeframe = data_item['timeframe']
            raw_data = data_item['data']
            
            # Process the data
            processed_data = self._process_raw_data(raw_data)
            
            if processed_data is not None:
                # Update buffer
                buffer_key = (symbol, timeframe)
                if buffer_key in self.buffers:
                    buffer_updated = self.buffers[buffer_key].add_data(processed_data)
                    
                    if buffer_updated:
                        # Create market snapshot
                        snapshot = self._create_market_snapshot(
                            symbol, timeframe, processed_data
                        )
                        
                        # Queue for pattern detection
                        self.pattern_queue.put(snapshot)
                        
            # Mark task as done
            self.data_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            self.logger.error(f"Data acquisition error: {e}")
            
    self.logger.info("Data acquisition worker stopped")

def _process_raw_data(self, raw_data):
    """Process raw API data into analysis-ready format"""
    try:
        # Apply technical indicators
        processed = self.main_window.data_processor.process_data(raw_data)
        
        # Validate data quality
        validation = self.main_window.data_processor.validate_data_quality(processed)
        
        if not validation['valid']:
            self.logger.warning(f"Data quality issues: {validation['issues']}")
            return None
            
        return processed
        
    except Exception as e:
        self.logger.error(f"Data processing error: {e}")
        return None
```

#### Pattern Detection Worker
```python
def _pattern_detection_worker(self):
    """Dedicated thread for pattern analysis"""
    self.logger.info("Pattern detection worker started")
    
    while self.monitoring_active and not self.shutdown_event.is_set():
        try:
            # Get market snapshot for analysis
            snapshot = self.pattern_queue.get(timeout=1)
            
            # Run pattern detection algorithms
            patterns = self._detect_patterns_for_snapshot(snapshot)
            
            # Apply confidence filtering
            high_confidence_patterns = self._filter_by_confidence(patterns)
            
            # Update snapshot with patterns
            snapshot.patterns = high_confidence_patterns
            self.last_snapshots[snapshot.symbol] = snapshot
            
            # Queue GUI updates for significant patterns
            if high_confidence_patterns:
                self.gui_queue.put({
                    'type': 'pattern_alert',
                    'snapshot': snapshot,
                    'patterns': high_confidence_patterns
                })
                
            # Always queue data updates
            self.gui_queue.put({
                'type': 'data_update',
                'snapshot': snapshot
            })
            
            self.pattern_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            self.logger.error(f"Pattern detection error: {e}")
            
    self.logger.info("Pattern detection worker stopped")

def _detect_patterns_for_snapshot(self, snapshot):
    """Run all pattern detection algorithms on snapshot data"""
    try:
        # Use main pattern detector
        patterns = self.main_window.pattern_detector.detect_all_patterns(
            snapshot.data
        )
        
        # Enhance patterns with additional metadata
        for pattern in patterns:
            pattern['symbol'] = snapshot.symbol
            pattern['timeframe'] = snapshot.timeframe
            pattern['detection_time'] = snapshot.last_update
            pattern['volatility_context'] = snapshot.volatility
            
        return patterns
        
    except Exception as e:
        self.logger.error(f"Pattern detection failed for {snapshot.symbol}: {e}")
        return []
```

#### GUI Update Worker
```python
def _gui_update_worker(self):
    """Thread-safe GUI update coordination"""
    self.logger.info("GUI update worker started")
    
    while self.monitoring_active and not self.shutdown_event.is_set():
        try:
            # Get GUI update request
            update_request = self.gui_queue.get(timeout=1)
            
            # Schedule on main thread using tkinter's after method
            self.main_window.root.after(
                0, lambda req=update_request: self._process_gui_update(req)
            )
            
            self.gui_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            self.logger.error(f"GUI update worker error: {e}")
            
    self.logger.info("GUI update worker stopped")

def _process_gui_update(self, update_request):
    """Process update request on main thread"""
    try:
        update_type = update_request['type']
        
        if update_type == 'pattern_alert':
            self._handle_pattern_alert(update_request)
        elif update_type == 'data_update':
            self._handle_data_update(update_request)
        elif update_type == 'status_update':
            self._handle_status_update(update_request)
            
    except Exception as e:
        self.logger.error(f"GUI update processing error: {e}")
```

### 3. Thread Synchronization Mechanisms

#### Thread-Safe Data Buffers
```python
class ThreadSafeDataBuffer:
    """Thread-safe circular buffer for real-time data"""
    
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = collections.deque(maxlen=max_size)
        self.lock = threading.RLock()  # Reentrant lock
        self.condition = threading.Condition(self.lock)
        self.last_update = 0
        
    def add_data(self, data_point):
        """Thread-safe data addition with notifications"""
        with self.condition:
            # Add data to buffer
            self.buffer.append(data_point)
            self.last_update = time.time()
            
            # Notify waiting threads
            self.condition.notify_all()
            
            return True
            
    def get_recent_data(self, count=100):
        """Get recent data points thread-safely"""
        with self.lock:
            if len(self.buffer) == 0:
                return []
                
            start_idx = max(0, len(self.buffer) - count)
            return list(self.buffer)[start_idx:]
            
    def wait_for_new_data(self, timeout=30):
        """Wait for new data with timeout"""
        with self.condition:
            last_seen = self.last_update
            
            def new_data_available():
                return self.last_update > last_seen
                
            return self.condition.wait_for(new_data_available, timeout)
```

#### Lock-Free Queue Implementation
```python
class LockFreeQueue:
    """High-performance queue for inter-thread communication"""
    
    def __init__(self, maxsize=1000):
        self.queue = queue.Queue(maxsize=maxsize)
        self.stats = {
            'items_added': 0,
            'items_removed': 0,
            'max_size_reached': 0
        }
        
    def put_nowait_safe(self, item):
        """Non-blocking put with overflow handling"""
        try:
            self.queue.put_nowait(item)
            self.stats['items_added'] += 1
            return True
        except queue.Full:
            # Remove oldest item and try again
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(item)
                self.stats['max_size_reached'] += 1
                return True
            except queue.Empty:
                return False
                
    def get_all_available(self):
        """Get all currently available items"""
        items = []
        while True:
            try:
                item = self.queue.get_nowait()
                items.append(item)
                self.stats['items_removed'] += 1
            except queue.Empty:
                break
        return items
```

### 4. Error Handling and Recovery

#### Thread Exception Management
```python
class ThreadExceptionHandler:
    """Centralized exception handling for worker threads"""
    
    def __init__(self, logger):
        self.logger = logger
        self.error_counts = {}
        self.max_errors_per_thread = 10
        
    def handle_thread_exception(self, thread_name, exception, restart_function=None):
        """Handle exceptions in worker threads"""
        
        # Log the error
        self.logger.error(f"Thread {thread_name} error: {exception}")
        
        # Track error count
        if thread_name not in self.error_counts:
            self.error_counts[thread_name] = 0
        self.error_counts[thread_name] += 1
        
        # Check if thread should be restarted
        if self.error_counts[thread_name] < self.max_errors_per_thread:
            if restart_function:
                self.logger.info(f"Restarting thread {thread_name}")
                try:
                    restart_function()
                except Exception as e:
                    self.logger.error(f"Failed to restart {thread_name}: {e}")
        else:
            self.logger.critical(f"Thread {thread_name} exceeded error limit, stopping")
            
    def reset_error_count(self, thread_name):
        """Reset error count after successful operation"""
        if thread_name in self.error_counts:
            self.error_counts[thread_name] = 0

# Usage in worker threads
def worker_thread_with_recovery(self):
    """Worker thread template with exception handling"""
    
    while self.monitoring_active:
        try:
            # Main work logic here
            self._do_work()
            
            # Reset error count on success
            self.exception_handler.reset_error_count(threading.current_thread().name)
            
        except Exception as e:
            self.exception_handler.handle_thread_exception(
                threading.current_thread().name,
                e,
                restart_function=lambda: self._restart_worker()
            )
            
            # Back off before retrying
            time.sleep(min(30, 2 ** self.error_counts.get(threading.current_thread().name, 0)))
```

### 5. Performance Optimization

#### Thread Pool Management
```python
class AdaptiveThreadPool:
    """Dynamic thread pool that adjusts to workload"""
    
    def __init__(self, min_workers=2, max_workers=10):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=min_workers)
        self.current_workers = min_workers
        self.task_queue_size = 0
        self.last_adjustment = time.time()
        
    def submit_task(self, fn, *args, **kwargs):
        """Submit task with automatic scaling"""
        future = self.executor.submit(fn, *args, **kwargs)
        self.task_queue_size += 1
        
        # Check if scaling is needed
        self._check_scaling()
        
        return future
        
    def _check_scaling(self):
        """Adjust thread pool size based on queue"""
        now = time.time()
        
        # Only adjust every 30 seconds
        if now - self.last_adjustment < 30:
            return
            
        # Scale up if queue is backing up
        if (self.task_queue_size > self.current_workers * 2 and 
            self.current_workers < self.max_workers):
            
            self._scale_up()
            
        # Scale down if queue is empty
        elif (self.task_queue_size == 0 and 
              self.current_workers > self.min_workers):
              
            self._scale_down()
            
        self.last_adjustment = now
        
    def _scale_up(self):
        """Increase thread pool size"""
        new_size = min(self.max_workers, self.current_workers + 2)
        self.executor._max_workers = new_size
        self.current_workers = new_size
        self.logger.info(f"Scaled thread pool up to {new_size} workers")
        
    def _scale_down(self):
        """Decrease thread pool size"""
        new_size = max(self.min_workers, self.current_workers - 1)
        self.executor._max_workers = new_size
        self.current_workers = new_size
        self.logger.info(f"Scaled thread pool down to {new_size} workers")
```

### 6. Shutdown and Cleanup

#### Graceful Thread Termination
```python
def stop_monitoring(self):
    """Gracefully stop all monitoring threads"""
    self.logger.info("Initiating graceful shutdown")
    
    # Signal all threads to stop
    self.monitoring_active = False
    self.shutdown_event.set()
    
    # Wait for worker threads to complete current tasks
    for thread in self.worker_threads:
        if thread.is_alive():
            self.logger.info(f"Waiting for {thread.name} to finish")
            thread.join(timeout=10)
            
            if thread.is_alive():
                self.logger.warning(f"Thread {thread.name} did not stop gracefully")
    
    # Shutdown thread pool
    self.executor.shutdown(wait=True, timeout=30)
    
    # Clear all queues
    self._clear_all_queues()
    
    # Final cleanup
    self._cleanup_resources()
    
    self.logger.info("Monitoring system stopped gracefully")

def _clear_all_queues(self):
    """Clear all communication queues"""
    queues = [self.data_queue, self.pattern_queue, self.gui_queue]
    
    for q in queues:
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except queue.Empty:
                break

def _cleanup_resources(self):
    """Clean up any remaining resources"""
    # Close any open connections
    # Clear buffers
    # Reset state variables
    self.buffers.clear()
    self.last_snapshots.clear()
    self.fetch_schedule.clear()
```

This threading architecture guide provides developers with comprehensive understanding of how the multi-threaded system works, enabling them to debug issues, optimize performance, or extend the threading model safely.