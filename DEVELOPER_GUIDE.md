# Forex Chart Pattern Recognition System - Developer Guide

## Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [Data Flow Explanation](#data-flow-explanation)
3. [Component Deep Dive](#component-deep-dive)
4. [Threading Architecture](#threading-architecture)
5. [Machine Learning Pipeline](#machine-learning-pipeline)
6. [GUI Framework Details](#gui-framework-details)
7. [Configuration System](#configuration-system)
8. [Development Workflow](#development-workflow)
9. [Testing Strategy](#testing-strategy)
10. [Deployment Guide](#deployment-guide)

## System Architecture Overview

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GUI Layer     │    │  Model Layer    │    │   Data Layer    │
│                 │    │                 │    │                 │
│ - MainWindow    │◄──►│ - PatternDetect │◄──►│ - ForexAPI      │
│ - ChartFrame    │    │ - MLModels      │    │ - DataProcessor │
│ - ControlsFrame │    │ - RealTimeMon   │    │ - Validation    │
│ - PatternFrame  │    │ - AlertSystem   │    │ - Caching       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Utilities Layer │
                    │                 │
                    │ - Config        │
                    │ - Logger        │
                    │ - Helpers       │
                    └─────────────────┘
```

### Core Design Principles
1. **Separation of Concerns**: Each layer handles specific responsibilities
2. **Event-Driven Architecture**: Components communicate through events and callbacks
3. **Thread Safety**: All operations are designed to be thread-safe
4. **Modular Design**: Easy to extend and maintain
5. **Configuration-Driven**: Behavior controlled through config files

## Data Flow Explanation

### 1. Application Startup Flow
```python
# main.py execution flow
main() 
├── check_dependencies()              # Verify all packages installed
├── setup_logging()                   # Initialize logging system
├── load_configuration()              # Load config.ini settings
├── initialize_components()           # Create data processors, APIs
├── create_main_window()             # Initialize GUI components
└── start_application_loop()         # Begin Tkinter event loop
```

### 2. Data Fetching Flow
```python
# When user clicks "Fetch Data" button
controls_frame.fetch_data()
├── validate_inputs()                # Check symbol, timeframe validity
├── update_status("Fetching...")     # Show progress to user
├── forex_api.get_forex_data()       # API call in background thread
│   ├── _make_request()              # HTTP request with retry logic
│   ├── validate_response()          # Check API response format
│   └── parse_to_dataframe()         # Convert JSON to pandas DataFrame
├── data_processor.process_data()    # Add technical indicators
│   ├── add_technical_indicators()   # RSI, MACD, Bollinger Bands
│   ├── calculate_support_resistance() # S/R levels
│   └── validate_data_quality()      # Ensure data integrity
├── chart_frame.update_chart()       # Display new chart
└── update_status("Data loaded")     # Notify completion
```

### 3. Pattern Detection Flow
```python
# When user clicks "Detect Patterns" button
controls_frame.detect_patterns()
├── validate_current_data()          # Ensure data exists
├── pattern_detector.detect_all_patterns()
│   ├── detect_head_shoulders()      # Geometric pattern analysis
│   ├── detect_double_patterns()     # Double top/bottom detection
│   ├── detect_triangles()           # Triangle pattern recognition
│   ├── detect_support_resistance()  # S/R level identification
│   └── calculate_confidence_scores() # ML-based confidence rating
├── filter_by_confidence()           # Remove low-confidence patterns
├── pattern_frame.update_patterns()  # Display results in GUI
├── chart_frame.add_pattern_overlays() # Draw patterns on chart
└── alert_system.trigger_alerts()    # Send notifications if enabled
```

### 4. Real-Time Monitoring Flow
```python
# Advanced real-time monitoring system
AdvancedRealtimeMonitor.start_monitoring()
├── _initialize_buffers()            # Create data buffers per symbol
├── start_worker_threads()           # Launch background workers
│   ├── _coordination_loop()         # Main scheduling thread
│   ├── _data_acquisition_worker()   # Data fetching thread
│   ├── _pattern_detection_worker()  # Pattern analysis thread
│   └── _gui_update_worker()         # Thread-safe GUI updates
└── setup_volatility_scheduler()     # Adaptive fetch intervals

# Coordination Loop (runs continuously)
_coordination_loop()
├── check_symbol_schedule()          # Determine which symbols need updates
├── calculate_volatility()           # Market volatility analysis
├── schedule_data_fetch()            # Submit fetch tasks to thread pool
└── update_next_fetch_time()         # Set next fetch based on volatility
```

## Component Deep Dive

### 1. ForexAPI (data/forex_api.py)

#### How it Works Internally
```python
class FinancialDataAPI:
    def __init__(self, config, logger):
        # Initialize with rate limiting and retry configuration
        self.config = config
        self.logger = logger
        self.session = requests.Session()  # Reuse connections
        self.last_request_time = 0         # Rate limiting
        self.request_count = 0             # Track API usage
        
    def _make_request(self, params):
        """Core request method with comprehensive error handling"""
        # 1. Rate limiting check
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - time_since_last)
            
        # 2. Retry logic with exponential backoff
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                    
        return None
```

#### Data Chunk Processing
```python
def get_data_chunk(self, ticker, start_date, end_date, interval):
    """Fetch data for specific date range with error resilience"""
    try:
        # 1. Build API parameters
        params = {
            'ticker': ticker,
            'interval': self._map_interval(interval),  # Convert to API format
            'start_date': start_date,
            'end_date': end_date,
            'api_key': self.api_key
        }
        
        # 2. Make API request
        data = self._make_request(params)
        if not data or 'prices' not in data:
            return None
            
        # 3. Convert to DataFrame with proper indexing
        df = pd.DataFrame([{
            'open': float(price['open']),
            'high': float(price['high']),
            'low': float(price['low']),
            'close': float(price['close']),
            'volume': int(price['volume'])
        } for price in data['prices']])
        
        # 4. Set datetime index and sort
        df.index = pd.to_datetime([p['time'] for p in data['prices']])
        df = df.sort_index()
        
        return df
        
    except Exception as e:
        self.logger.error(f"Error fetching chunk: {e}")
        return None  # Return None to allow continuation
```

### 2. Pattern Detection Engine (models/pattern_detector.py)

#### Head and Shoulders Detection Algorithm
```python
def detect_head_shoulders(self, df):
    """Geometric head and shoulders pattern detection"""
    patterns = []
    
    # 1. Find significant peaks using sliding window
    peaks = []
    window = 5
    for i in range(window, len(df) - window):
        current_high = df['high'].iloc[i]
        # Check if current point is highest in window
        if all(current_high >= df['high'].iloc[j] for j in range(i-window, i+window+1)):
            peaks.append({
                'index': i,
                'price': current_high,
                'time': df.index[i]
            })
    
    # 2. Analyze peak sequences for H&S pattern
    for i in range(len(peaks) - 2):
        left_shoulder = peaks[i]
        head = peaks[i + 1] 
        right_shoulder = peaks[i + 2]
        
        # 3. Apply geometric rules
        if (head['price'] > left_shoulder['price'] * 1.02 and  # Head 2% higher
            head['price'] > right_shoulder['price'] * 1.02 and  # Head higher than right
            abs(left_shoulder['price'] - right_shoulder['price']) / left_shoulder['price'] < 0.05):  # Shoulders similar
            
            # 4. Calculate neckline
            neckline_level = min(
                df['low'].iloc[left_shoulder['index']:head['index']].min(),
                df['low'].iloc[head['index']:right_shoulder['index']].min()
            )
            
            # 5. Calculate confidence based on pattern quality
            price_symmetry = 1 - abs(left_shoulder['price'] - right_shoulder['price']) / left_shoulder['price']
            head_prominence = (head['price'] - max(left_shoulder['price'], right_shoulder['price'])) / head['price']
            confidence = (price_symmetry * 0.6 + head_prominence * 0.4) * 100
            
            patterns.append({
                'type': 'Head and Shoulders',
                'confidence': confidence,
                'points': [left_shoulder, head, right_shoulder],
                'neckline': neckline_level,
                'signal': 'Bearish',
                'target_price': neckline_level - (head['price'] - neckline_level),
                'timestamp': df.index[-1]
            })
    
    return patterns
```

#### Support and Resistance Detection
```python
def detect_support_resistance(self, df):
    """Dynamic support and resistance level detection"""
    levels = []
    
    # 1. Identify potential levels using price clustering
    price_points = []
    for i in range(1, len(df) - 1):
        # Find local minima (support) and maxima (resistance)
        if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
            df['low'].iloc[i] < df['low'].iloc[i+1]):
            price_points.append(('support', df['low'].iloc[i], df.index[i]))
            
        if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
            df['high'].iloc[i] > df['high'].iloc[i+1]):
            price_points.append(('resistance', df['high'].iloc[i], df.index[i]))
    
    # 2. Cluster nearby price levels
    tolerance = df['close'].iloc[-1] * 0.002  # 0.2% tolerance
    
    for level_type in ['support', 'resistance']:
        type_points = [p for p in price_points if p[0] == level_type]
        clusters = []
        
        for price_point in type_points:
            price = price_point[1]
            
            # Find existing cluster or create new one
            assigned = False
            for cluster in clusters:
                if abs(cluster['level'] - price) <= tolerance:
                    cluster['prices'].append(price)
                    cluster['touches'] += 1
                    cluster['level'] = np.mean(cluster['prices'])  # Recalculate average
                    assigned = True
                    break
                    
            if not assigned:
                clusters.append({
                    'level': price,
                    'prices': [price],
                    'touches': 1,
                    'type': level_type
                })
        
        # 3. Filter and score levels
        for cluster in clusters:
            if cluster['touches'] >= 3:  # Minimum 3 touches for validity
                strength = min(10, cluster['touches'])  # Strength score 1-10
                
                levels.append({
                    'type': f'{level_type.title()} Level',
                    'level': cluster['level'],
                    'strength': strength,
                    'touch_count': cluster['touches'],
                    'confidence': min(95, 60 + strength * 3.5),  # 60-95% confidence
                    'signal': 'Bullish' if level_type == 'support' else 'Bearish',
                    'timestamp': df.index[-1]
                })
    
    return levels
```

### 3. Real-Time Monitoring System (models/advanced_realtime_monitor.py)

#### Adaptive Data Buffer Implementation
```python
class AdaptiveDataBuffer:
    """Efficient sliding window data storage for real-time processing"""
    
    def __init__(self, symbol, timeframe, max_size=1000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_size = max_size
        self.data = deque(maxlen=max_size)  # Circular buffer
        self.last_timestamp = None
        self.lock = threading.Lock()  # Thread safety
        
    def add_candle(self, candle_data):
        """Add new candle with duplicate detection"""
        with self.lock:
            current_timestamp = candle_data['timestamp']
            
            # Prevent duplicate data
            if self.last_timestamp and current_timestamp <= self.last_timestamp:
                return False  # Not a new candle
                
            self.data.append(candle_data)
            self.last_timestamp = current_timestamp
            return True  # New data added
            
    def get_dataframe(self):
        """Convert buffer to pandas DataFrame for analysis"""
        with self.lock:
            if len(self.data) == 0:
                return pd.DataFrame()
                
            df = pd.DataFrame(list(self.data))
            df.set_index('timestamp', inplace=True)
            return df.sort_index()
            
    def calculate_volatility(self):
        """Calculate recent volatility for adaptive scheduling"""
        with self.lock:
            if len(self.data) < 20:
                return 0.02  # Default volatility
                
            # Calculate 20-period price changes
            prices = [candle['close'] for candle in list(self.data)[-20:]]
            price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] 
                           for i in range(1, len(prices))]
            
            return np.mean(price_changes)
```

#### Volatility-Adaptive Scheduler
```python
class VolatilityAdaptiveScheduler:
    """Adjusts data fetch frequency based on market volatility"""
    
    def __init__(self, config):
        self.base_interval = config.get('base_interval', 60)  # 60 seconds
        self.min_interval = config.get('min_interval', 10)    # 10 seconds minimum
        self.max_interval = config.get('max_interval', 300)   # 5 minutes maximum
        self.volatility_threshold = config.get('volatility_threshold', 0.01)
        
    def get_next_fetch_time(self, symbol, volatility):
        """Calculate next fetch time based on current volatility"""
        current_time = time.time()
        
        if volatility > self.volatility_threshold * 2:
            # High volatility: fetch more frequently
            interval = self.min_interval
        elif volatility > self.volatility_threshold:
            # Medium volatility: moderate frequency
            interval = self.base_interval * 0.5
        else:
            # Low volatility: normal frequency
            interval = self.base_interval
            
        # Apply bounds
        interval = max(self.min_interval, min(self.max_interval, interval))
        
        return current_time + interval
        
    def get_current_interval(self, symbol):
        """Get current fetch interval for display"""
        # Implementation would track actual intervals per symbol
        return self.base_interval
```

#### Multi-Threading Coordination
```python
def _coordination_loop(self):
    """Main coordination thread managing all data fetching"""
    while self.monitoring_active:
        try:
            current_time = time.time()
            
            # Check each monitored symbol
            for symbol in self.config.symbols:
                next_fetch = self.fetch_schedule.get(symbol, 0)
                
                if current_time >= next_fetch:
                    # Time to fetch new data
                    volatility = self._get_symbol_volatility(symbol)
                    
                    # Submit fetch task to thread pool
                    future = self.executor.submit(self._fetch_symbol_data, symbol)
                    
                    # Schedule next fetch based on volatility
                    self.fetch_schedule[symbol] = self.scheduler.get_next_fetch_time(
                        symbol, volatility
                    )
                    
                    self.logger.debug(f"Scheduled {symbol}, volatility: {volatility:.3f}")
            
            # Coordination loop runs every second
            time.sleep(1)
            
        except Exception as e:
            self.logger.error(f"Coordination loop error: {e}")
            time.sleep(5)  # Back off on errors
```

### 4. Machine Learning Pipeline (models/ml_models.py)

#### CNN Model Architecture
```python
def create_cnn_model(self, input_shape):
    """Create 1D CNN for pattern shape recognition"""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # First convolutional block
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.25),
        
        # Second convolutional block  
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.25),
        
        # Third convolutional block
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.5),
        
        # Dense layers
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile with appropriate optimizer and loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model
```

#### Training Data Preparation
```python
def prepare_training_data(self, df, pattern_type):
    """Convert price data to ML training sequences"""
    # 1. Feature engineering
    features = ['open', 'high', 'low', 'close', 'volume']
    technical_indicators = ['sma_5', 'sma_10', 'sma_20', 'rsi', 'macd', 'stoch_k']
    
    # Ensure all features exist
    for indicator in technical_indicators:
        if indicator in df.columns:
            features.append(indicator)
    
    # 2. Create sequences
    sequences = []
    labels = []
    
    for i in range(self.sequence_length, len(df)):
        # Extract sequence window
        sequence = df[features].iloc[i-self.sequence_length:i].values
        
        # Label: 1 if pattern occurs in next N periods, 0 otherwise
        future_window = df.iloc[i:i+10]  # Look ahead 10 periods
        has_pattern = self._check_pattern_in_window(future_window, pattern_type)
        
        sequences.append(sequence)
        labels.append(1 if has_pattern else 0)
    
    # 3. Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(labels)
    
    # 4. Normalize features
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_reshaped)
    X = X_scaled.reshape(X.shape)
    
    return X, y, scaler
```

### 5. GUI Framework Details (gui/main_window.py)

#### Thread-Safe GUI Updates
```python
def _process_gui_update(self, update_request):
    """Process GUI updates on main thread to ensure thread safety"""
    try:
        update_type = update_request['type']
        
        if update_type == 'pattern_alert':
            # Handle new pattern detection
            snapshot = update_request['snapshot']
            patterns = update_request['patterns']
            
            # Update chart if this is the current symbol
            if snapshot.symbol == self.current_symbol:
                self.chart_frame.update_chart(snapshot.data)
                self.chart_frame.add_pattern_overlays(patterns)
                self.pattern_frame.update_patterns(patterns)
            
            # Trigger alert system
            for pattern in patterns:
                self.alert_system.create_pattern_alert(
                    pattern_type=pattern.get('type', 'unknown'),
                    symbol=snapshot.symbol,
                    confidence=pattern.get('confidence', 0),
                    suggested_action=self._get_trading_suggestion(pattern),
                    pattern_data=pattern
                )
                
        elif update_type == 'data_update':
            # Handle routine data updates
            snapshot = update_request['snapshot']
            
            if snapshot.symbol == self.current_symbol:
                self.current_data = snapshot.data
                self.chart_frame.update_chart(snapshot.data)
                
                # Update status bar
                current_price = snapshot.data['close'].iloc[-1]
                status = f"Real-time: {snapshot.symbol} @ {current_price:.5f}"
                self.update_status(status)
                
    except Exception as e:
        self.logger.error(f"GUI update error: {e}")
```

#### Menu System Implementation
```python
def create_menu_bar(self):
    """Create comprehensive menu system"""
    menubar = tk.Menu(self.root)
    self.root.config(menu=menubar)
    
    # File menu with all data operations
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Load Data File...", command=self.load_data_file)
    file_menu.add_separator()
    file_menu.add_command(label="Save Results as JSON...", command=self.save_results_json)
    file_menu.add_command(label="Save Results as CSV...", command=self.save_results_csv)
    file_menu.add_separator()
    file_menu.add_command(label="Export Chart...", command=self.export_chart)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=self.on_closing)
    menubar.add_cascade(label="File", menu=file_menu)
    
    # Tools menu for advanced features
    tools_menu = tk.Menu(menubar, tearoff=0)
    tools_menu.add_command(label="Pattern Settings...", command=self.show_pattern_settings)
    tools_menu.add_command(label="User Preferences...", command=self.show_preferences)
    tools_menu.add_separator()
    tools_menu.add_command(label="Model Training...", command=self.show_training_dialog)
    tools_menu.add_command(label="Analytics Dashboard...", command=self.show_analytics)
    tools_menu.add_separator()
    tools_menu.add_command(label="API Settings...", command=self.show_api_settings)
    menubar.add_cascade(label="Tools", menu=tools_menu)
```

## Threading Architecture

### Thread Safety Strategy
```python
# Key principles for thread safety in the application:

1. **GUI Operations on Main Thread Only**
   - All tkinter operations must happen on main thread
   - Use root.after() to schedule GUI updates from background threads
   
2. **Data Structure Protection**
   - Use threading.Lock() for shared data structures
   - Implement thread-safe data buffers with proper locking
   
3. **Queue-Based Communication**
   - Use queue.Queue for inter-thread communication
   - Separate queues for different types of operations
   
4. **Background Worker Threads**
   - Data fetching happens in background threads
   - Pattern detection runs in separate threads
   - GUI updates coordinated through main thread

# Example implementation:
class ThreadSafeDataManager:
    def __init__(self):
        self.data_lock = threading.Lock()
        self.update_queue = queue.Queue()
        self.gui_callback = None
        
    def update_data(self, new_data):
        """Called from background thread"""
        with self.data_lock:
            self.current_data = new_data
            
        # Queue GUI update for main thread
        self.update_queue.put({
            'type': 'data_update',
            'data': new_data
        })
        
    def process_gui_updates(self):
        """Called from main thread periodically"""
        try:
            while True:
                update = self.update_queue.get_nowait()
                if self.gui_callback:
                    self.gui_callback(update)
        except queue.Empty:
            pass
```

## Configuration System

### How Configuration Loading Works
```python
class Config:
    """Centralized configuration management"""
    
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.defaults = self._get_defaults()
        self.load()
        
    def _get_defaults(self):
        """Define default configuration values"""
        return {
            'API': {
                'financial_datasets_api_key': 'demo',
                'request_timeout': '30',
                'rate_limit_delay': '1',
                'max_retries': '3'
            },
            'PATTERNS': {
                'head_shoulders': 'true',
                'double_patterns': 'true',
                'triangles': 'true',
                'support_resistance': 'true',
                'sensitivity': '0.015',
                'min_confidence': '0.6'
            },
            # ... more sections
        }
        
    def load(self):
        """Load configuration with fallback to defaults"""
        try:
            if os.path.exists(self.config_file):
                self.config.read(self.config_file)
            else:
                self.logger.info("Config file not found, creating with defaults")
                self._create_default_config()
                
            # Merge with defaults for missing values
            for section, options in self.defaults.items():
                if not self.config.has_section(section):
                    self.config.add_section(section)
                    
                for option, value in options.items():
                    if not self.config.has_option(section, option):
                        self.config.set(section, option, value)
                        
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self._create_default_config()
            
    def get(self, section, option, fallback=None):
        """Get configuration value with type conversion"""
        try:
            value = self.config.get(section, option)
            
            # Type conversion based on default value type
            default_value = self.defaults.get(section, {}).get(option)
            if default_value:
                if default_value.lower() in ('true', 'false'):
                    return value.lower() == 'true'
                elif '.' in default_value:
                    return float(value)
                elif default_value.isdigit():
                    return int(value)
                    
            return value
            
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
```

## Development Workflow

### Setting Up Development Environment
```bash
# 1. Clone repository
git clone <repository-url>
cd forex-pattern-recognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up configuration
cp config.ini.example config.ini
# Edit config.ini with your API keys

# 4. Run tests
python -m pytest tests/

# 5. Start application
python main.py
```

### Adding New Pattern Types
```python
# To add a new pattern type, follow these steps:

# 1. Add detection method to PatternDetector class
def detect_new_pattern(self, df):
    """Detect your new pattern type"""
    patterns = []
    
    # Your detection logic here
    # Return list of pattern dictionaries
    
    return patterns

# 2. Register in detect_all_patterns method
def detect_all_patterns(self, df):
    all_patterns = []
    
    if self.config.get('new_pattern', True):
        all_patterns.extend(self.detect_new_pattern(df))
    
    return all_patterns

# 3. Add configuration option in config.ini
[PATTERNS]
new_pattern = true

# 4. Add to GUI pattern selection
# Update preferences_dialog.py pattern_options list

# 5. Add chart overlay drawing
# Update chart_frame.py draw_pattern_overlay method
```

### Testing Strategy
```python
# Unit tests for pattern detection
import unittest
import pandas as pd
import numpy as np

class TestPatternDetection(unittest.TestCase):
    
    def setUp(self):
        """Create test data"""
        self.detector = PatternDetector()
        
        # Create synthetic price data
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.rand(100),
            'low': prices - np.random.rand(100), 
            'close': prices + np.random.randn(100) * 0.1,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
    def test_head_shoulders_detection(self):
        """Test head and shoulders pattern detection"""
        patterns = self.detector.detect_head_shoulders(self.test_data)
        
        # Verify pattern structure
        for pattern in patterns:
            self.assertIn('type', pattern)
            self.assertIn('confidence', pattern)
            self.assertIn('points', pattern)
            self.assertEqual(len(pattern['points']), 3)
            self.assertGreaterEqual(pattern['confidence'], 0)
            self.assertLessEqual(pattern['confidence'], 100)

if __name__ == '__main__':
    unittest.main()
```

## Performance Optimization

### Memory Management
```python
# Efficient data handling for real-time processing

class MemoryEfficientDataBuffer:
    """Memory-optimized circular buffer for real-time data"""
    
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = np.empty((max_size, 6))  # OHLCV + timestamp
        self.current_index = 0
        self.is_full = False
        
    def add_data_point(self, timestamp, open_price, high, low, close, volume):
        """Add data point with O(1) complexity"""
        self.buffer[self.current_index] = [timestamp, open_price, high, low, close, volume]
        self.current_index = (self.current_index + 1) % self.max_size
        
        if self.current_index == 0:
            self.is_full = True
            
    def get_recent_data(self, n_points=100):
        """Get most recent n points efficiently"""
        if not self.is_full and self.current_index < n_points:
            return self.buffer[:self.current_index]
        
        if self.is_full:
            # Data wraps around
            if n_points >= self.max_size:
                return np.roll(self.buffer, -self.current_index, axis=0)
            else:
                start_idx = (self.current_index - n_points) % self.max_size
                if start_idx + n_points <= self.max_size:
                    return self.buffer[start_idx:start_idx + n_points]
                else:
                    return np.vstack([
                        self.buffer[start_idx:],
                        self.buffer[:self.current_index]
                    ])
        
        return self.buffer[max(0, self.current_index - n_points):self.current_index]
```

### Caching Strategy
```python
class IntelligentDataCache:
    """Smart caching system for API responses"""
    
    def __init__(self, max_cache_size=100, ttl_seconds=300):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_cache_size
        self.ttl = ttl_seconds
        
    def get(self, key):
        """Get cached data with TTL check"""
        if key in self.cache:
            # Check if data is still valid
            if time.time() - self.access_times[key] < self.ttl:
                return self.cache[key]
            else:
                # Data expired
                del self.cache[key]
                del self.access_times[key]
        
        return None
        
    def set(self, key, data):
        """Cache data with LRU eviction"""
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), 
                           key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            
        self.cache[key] = data
        self.access_times[key] = time.time()
```

This developer guide provides comprehensive explanations of how each component works internally, making it easy for new developers to understand and contribute to the project. The examples show real implementation details and best practices for extending the system.