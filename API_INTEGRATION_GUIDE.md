# API Integration Guide

## Financial Datasets API Integration

### How the API Client Works

The `FinancialDataAPI` class handles all external data communication with comprehensive error handling and rate limiting.

#### Core Request Flow
```python
# Request lifecycle in forex_api.py
_make_request() 
├── Rate limiting check (1 second minimum between requests)
├── Build request parameters with API key
├── Execute HTTP request with 30-second timeout
├── Handle response codes:
│   ├── 200: Success - parse JSON data
│   ├── 429: Rate limited - exponential backoff retry
│   ├── 401: Invalid API key - log error and return None
│   └── Other errors: Retry with exponential backoff
└── Return parsed data or None on failure
```

#### Error Handling Strategy
```python
def _make_request(self, params):
    for attempt in range(self.config.max_retries):
        try:
            # Rate limiting
            time_since_last = time.time() - self.last_request_time
            if time_since_last < self.config.rate_limit_delay:
                time.sleep(self.config.rate_limit_delay - time_since_last)
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited - wait longer
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue
            elif response.status_code == 401:
                # Invalid API key
                self.logger.error("Invalid API key")
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Request failed: {e}")
            if attempt == self.config.max_retries - 1:
                return None
```

#### Data Transformation Pipeline
```python
# How raw API data becomes pandas DataFrame
get_forex_data()
├── Split date range into chunks (handles large requests)
├── For each chunk:
│   ├── get_data_chunk() - fetch raw JSON
│   ├── Parse JSON response structure:
│   │   {
│   │     "prices": [
│   │       {
│   │         "time": "2024-01-01T09:30:00Z",
│   │         "open": 1.0950,
│   │         "high": 1.0965,
│   │         "low": 1.0945,
│   │         "close": 1.0960,
│   │         "volume": 15420
│   │       }
│   │     ]
│   │   }
│   ├── Convert to DataFrame with proper types:
│   │   df['open'] = float(price['open'])
│   │   df['high'] = float(price['high']) 
│   │   df['low'] = float(price['low'])
│   │   df['close'] = float(price['close'])
│   │   df['volume'] = int(price['volume'])
│   └── Set datetime index and sort chronologically
├── Concatenate all chunks into single DataFrame
└── Return processed data ready for analysis
```

### Supported Symbols and Timeframes

#### Symbol Categories
```python
SUPPORTED_SYMBOLS = {
    'major_forex': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF'],
    'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    'crypto': ['BTC/USD', 'ETH/USD', 'ADA/USD'],
    'commodities': ['GOLD', 'SILVER', 'OIL']
}
```

#### Timeframe Mapping
```python
# Internal timeframe to API parameter mapping
TIMEFRAME_MAPPING = {
    '1min': '1minute',
    '5min': '5minute', 
    '15min': '15minute',
    '30min': '30minute',
    '1hour': '1hour',
    '4hour': '4hour',
    'daily': 'daily',
    'weekly': 'weekly'
}
```

### API Key Management

#### Configuration Setup
```python
# config.ini format
[API]
financial_datasets_api_key = your_actual_api_key_here
request_timeout = 30
rate_limit_delay = 1
max_retries = 3
```

#### Validation Process
```python
def validate_api_key(self):
    """Test API key with minimal request"""
    try:
        # Use 5-day range to avoid weekend issues
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        test_data = self.get_data_chunk('AAPL', start_date, end_date, '5min')
        
        if test_data is not None and not test_data.empty:
            return True, "API key validated successfully"
        else:
            return False, "API key validation failed - no data returned"
            
    except Exception as e:
        return False, f"API validation error: {str(e)}"
```

### Real-Time Data Handling

#### Adaptive Fetch Strategy
```python
# How the system determines when to fetch new data
class VolatilityAdaptiveScheduler:
    def calculate_fetch_interval(self, volatility):
        """Adjust fetch frequency based on market conditions"""
        base_interval = 60  # 1 minute baseline
        
        if volatility > 0.02:  # High volatility (2%+ price moves)
            return 10  # Fetch every 10 seconds
        elif volatility > 0.01:  # Medium volatility
            return 30  # Fetch every 30 seconds  
        else:  # Low volatility
            return 60  # Standard 1-minute interval
```

#### Data Quality Assurance
```python
def validate_data_quality(self, df):
    """Comprehensive data validation"""
    issues = []
    
    # Check for missing values
    if df.isnull().any().any():
        issues.append("Contains null values")
        
    # Validate OHLC relationships
    invalid_ohlc = (
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    if invalid_ohlc.any():
        issues.append("Invalid OHLC relationships detected")
        
    # Check for extreme price gaps
    price_changes = df['close'].pct_change().abs()
    if (price_changes > 0.1).any():  # 10% single-candle moves
        issues.append("Extreme price movements detected")
        
    # Validate timestamps
    if not df.index.is_monotonic_increasing:
        issues.append("Timestamps not in chronological order")
        
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'data_points': len(df),
        'date_range': f"{df.index[0]} to {df.index[-1]}"
    }
```

## Alternative Data Sources

### Adding New Data Providers

To integrate additional data sources, follow this pattern:

```python
class NewDataProvider:
    def __init__(self, config, logger):
        self.api_key = config.get('API', 'new_provider_key')
        self.base_url = "https://api.newprovider.com/v1/"
        self.logger = logger
        
    def get_forex_data(self, symbol, interval, outputsize):
        """Implement standard interface"""
        # Transform parameters to provider format
        provider_symbol = self._convert_symbol(symbol)
        provider_interval = self._convert_interval(interval)
        
        # Make API request
        data = self._fetch_data(provider_symbol, provider_interval)
        
        # Convert to standard DataFrame format
        return self._standardize_data(data)
        
    def _standardize_data(self, raw_data):
        """Convert provider data to standard OHLCV format"""
        df = pd.DataFrame()
        # Implementation depends on provider format
        return df
```

### Data Source Failover

```python
class MultiSourceDataManager:
    """Manage multiple data sources with automatic failover"""
    
    def __init__(self, config, logger):
        self.primary_source = FinancialDataAPI(config, logger)
        self.backup_sources = [
            # Additional providers as fallbacks
        ]
        self.logger = logger
        
    def get_data_with_failover(self, symbol, interval, outputsize):
        """Try primary source, fall back to alternatives"""
        
        # Try primary source first
        try:
            data = self.primary_source.get_forex_data(symbol, interval, outputsize)
            if data is not None and not data.empty:
                return data
        except Exception as e:
            self.logger.warning(f"Primary source failed: {e}")
            
        # Try backup sources
        for i, backup_source in enumerate(self.backup_sources):
            try:
                self.logger.info(f"Trying backup source {i+1}")
                data = backup_source.get_forex_data(symbol, interval, outputsize)
                if data is not None and not data.empty:
                    return data
            except Exception as e:
                self.logger.warning(f"Backup source {i+1} failed: {e}")
                
        # All sources failed
        self.logger.error("All data sources failed")
        return None
```

### WebSocket Real-Time Feeds

For ultra-low latency requirements:

```python
class WebSocketDataFeed:
    """Real-time data via WebSocket connection"""
    
    def __init__(self, config, callback):
        self.ws_url = "wss://api.provider.com/stream"
        self.api_key = config.get('API', 'websocket_key')
        self.callback = callback
        self.ws = None
        
    async def connect(self):
        """Establish WebSocket connection"""
        self.ws = await websockets.connect(
            f"{self.ws_url}?api_key={self.api_key}"
        )
        
        # Subscribe to symbols
        subscribe_msg = {
            "action": "subscribe",
            "symbols": ["EURUSD", "GBPUSD", "USDJPY"]
        }
        await self.ws.send(json.dumps(subscribe_msg))
        
        # Listen for data
        async for message in self.ws:
            data = json.loads(message)
            self.process_realtime_data(data)
            
    def process_realtime_data(self, data):
        """Process incoming real-time tick data"""
        tick = {
            'symbol': data['symbol'],
            'bid': float(data['bid']),
            'ask': float(data['ask']),
            'timestamp': pd.to_datetime(data['timestamp'])
        }
        
        # Send to callback for processing
        self.callback(tick)
```

This guide provides developers with comprehensive understanding of how data flows through the system and how to extend or modify the API integration components.