"""
Financial data API integration using Financial Datasets API
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import os

class Price(BaseModel):
    time: str
    open: float
    close: float
    high: float
    low: float
    volume: int

class PriceResponse(BaseModel):
    prices: List[Price]

class FinancialDataAPI:
    """Financial data API client using Financial Datasets API"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.api_key = config.get('API', 'financial_datasets_api_key')
        self.base_url = "https://api.financialdatasets.ai/prices/"
        self.timeout = config.getint('API', 'request_timeout', 30)
        self.rate_limit_delay = config.getint('API', 'rate_limit_delay', 1)
        self.max_retries = config.getint('API', 'max_retries', 3)
        self.last_request_time = 0
        
        # Cache for data
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Check for API key in environment
        env_key = os.getenv("FINANCIAL_DATASETS_API_KEY")
        if env_key:
            self.api_key = env_key
    
    def _make_request(self, params: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Make API request with rate limiting and retries"""
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_request)
        
        headers = {"X-API-KEY": self.api_key}
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Making API request: {params}")
                response = requests.get(self.base_url, params=params, headers=headers, timeout=self.timeout)
                self.last_request_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    return data
                else:
                    self.logger.error(f"HTTP Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def get_data_chunk(self, ticker: str, start_date: str, end_date: str, interval: str = "5min") -> Optional[pd.DataFrame]:
        """
        Fetch data for a specific date range (single chunk)
        Returns None on error to allow continuation
        """
        # Convert interval format
        interval_mapping = {
            "1min": ("minute", 1),
            "5min": ("minute", 5),
            "15min": ("minute", 15),
            "30min": ("minute", 30),
            "60min": ("minute", 60),
            "daily": ("day", 1)
        }
        
        if interval not in interval_mapping:
            self.logger.error(f"Unsupported interval: {interval}")
            return None
        
        interval_type, multiplier = interval_mapping[interval]
        
        params = {
            'ticker': ticker,
            'interval': interval_type,
            'interval_multiplier': str(multiplier),
            'start_date': start_date,
            'end_date': end_date
        }
        
        try:
            self.logger.info(f"Requesting {ticker} data from {start_date} to {end_date}...")
            data = self._make_request(params)
            
            if not data:
                self.logger.warning(f"No response for {start_date}-{end_date}")
                return None
            
            parsed = PriceResponse(**data)
            
            df = pd.DataFrame([p.dict() for p in parsed.prices])
            if not df.empty:
                # Convert to Eastern Time (US market timezone)
                df["time"] = pd.to_datetime(df["time"]).dt.tz_convert('US/Eastern')
                df.set_index("time", inplace=True)
                df.sort_index(inplace=True)
                self.logger.info(f"Retrieved {len(df)} records for {start_date} to {end_date}")
                return df
                
            self.logger.warning(f"No data for {start_date}-{end_date}")
            return None
            
        except Exception as e:
            self.logger.error(f"Exception for {start_date}-{end_date}: {str(e)}")
            return None
    
    def get_forex_data(self, symbol: str, interval: str = "5min", outputsize: str = "compact") -> pd.DataFrame:
        """
        Fetch financial data with chunking for larger date ranges
        
        Args:
            symbol: Stock symbol (e.g., "AAPL", "MSFT") 
            interval: Time interval (1min, 5min, 15min, 30min, 60min, daily)
            outputsize: compact (30 days) or full (150 days)
        
        Returns:
            pandas.DataFrame: OHLCV data
        """
        # Determine days based on outputsize
        days_back = 30 if outputsize == "compact" else 150
        
        # Check cache first
        cache_key = f"{symbol}_{interval}_{days_back}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                self.logger.debug(f"Returning cached data for {cache_key}")
                return cached_data
        
        self.logger.info(f"Fetching {symbol} data with {interval} interval for {days_back} days")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Generate monthly chunks for larger requests
        chunks = []
        current = start_date
        while current <= end_date:
            chunk_end = min(current + timedelta(days=30), end_date)
            chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
            current = chunk_end + timedelta(days=1)
        
        all_data = []
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            retry = 0
            while retry < 3:  # Max 3 retries per chunk
                df_chunk = self.get_data_chunk(symbol, chunk_start, chunk_end, interval)
                if df_chunk is not None:
                    all_data.append(df_chunk)
                    break
                retry += 1
                time.sleep(2 ** retry)  # Exponential backoff
            else:
                self.logger.warning(f"Failed to get data for {chunk_start} to {chunk_end} after 3 attempts")
            
            # Add delay between chunks to avoid rate limiting
            if i < len(chunks) - 1:
                time.sleep(1)
        
        if not all_data:
            self.logger.error("No data retrieved for the entire period")
            return pd.DataFrame()
        
        df = pd.concat(all_data).sort_index()
        
        # Remove duplicates if any
        df = df[~df.index.duplicated(keep='first')]
        
        # Ensure we return a DataFrame
        if isinstance(df, pd.DataFrame):
            # Cache the result
            self.cache[cache_key] = (df.copy(), time.time())
            
            self.logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
        else:
            self.logger.error("Data concatenation did not return a DataFrame")
            return pd.DataFrame()
    
    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol (latest available data)
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
        
        Returns:
            dict: Real-time quote data
        """
        self.logger.info(f"Fetching real-time quote for {symbol}")
        
        # Get latest day's data
        df = self.get_forex_data(symbol, interval="5min", outputsize="compact")
        
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        
        quote = {
            'symbol': symbol,
            'price': float(latest['close']),
            'open': float(latest['open']),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'volume': int(latest['volume']),
            'timestamp': str(df.index[-1]),
            'change': float(latest['close'] - latest['open']),
            'change_percent': float((latest['close'] - latest['open']) / latest['open'] * 100)
        }
        
        self.logger.info(f"Real-time quote for {symbol}: ${quote['price']:.2f}")
        return quote
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols"""
        # Common stocks and ETFs
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA",
            "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA",
            "SPY", "QQQ", "IWM", "EFA", "VTI", "BND", "GLD", "SLV",
            "XOM", "CVX", "COP", "PG", "JNJ", "UNH", "HD", "WMT"
        ]
    
    def validate_api_key(self) -> bool:
        """Validate API key by making a test request"""
        self.logger.info("Validating API key")
        
        if not self.api_key or self.api_key == "demo":
            self.logger.warning("Using demo API key - functionality may be limited")
            return False
        
        # Try to fetch a small amount of data for AAPL
        test_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        df = self.get_data_chunk("AAPL", test_date, test_date, "daily")
        
        if df is not None:
            self.logger.info("API key validation successful")
            return True
        else:
            self.logger.error("API key validation failed")
            return False

# For backward compatibility, create an alias
ForexAPI = FinancialDataAPI