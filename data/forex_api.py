"""
Forex data API integration
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

class ForexAPI:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        self.base_url = "https://www.alphavantage.co/query"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 12  # seconds (5 requests per minute limit)
        
    def get_forex_data(self, symbol, interval="5min", outputsize="compact"):
        """
        Fetch forex data from Alpha Vantage API
        
        Args:
            symbol: Currency pair (e.g., "EUR/USD")
            interval: Time interval (1min, 5min, 15min, 30min, 60min, daily)
            outputsize: compact (last 100 data points) or full (all available)
        
        Returns:
            pandas.DataFrame: OHLCV data
        """
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last_request
                self.logger.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
            
            # Prepare symbol for API (remove slash)
            api_symbol = symbol.replace("/", "")
            
            # Build API parameters
            params = {
                "function": "FX_INTRADAY" if interval != "daily" else "FX_DAILY",
                "from_symbol": api_symbol[:3],
                "to_symbol": api_symbol[3:],
                "apikey": self.api_key,
                "outputsize": outputsize
            }
            
            if interval != "daily":
                params["interval"] = interval
            
            self.logger.info(f"Fetching forex data for {symbol} with interval {interval}")
            
            # Make API request
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            self.last_request_time = time.time()
            
            # Check for API errors
            if "Error Message" in data:
                raise Exception(f"API Error: {data['Error Message']}")
            
            if "Note" in data:
                raise Exception(f"API Rate Limit: {data['Note']}")
            
            # Extract time series data
            if interval == "daily":
                time_series_key = f"Time Series FX (Daily)"
            else:
                time_series_key = f"Time Series FX ({interval})"
            
            if time_series_key not in data:
                available_keys = list(data.keys())
                self.logger.error(f"Expected key '{time_series_key}' not found. Available keys: {available_keys}")
                
                # Try to find the correct key
                for key in available_keys:
                    if "Time Series" in key:
                        time_series_key = key
                        break
                else:
                    raise Exception("No time series data found in API response")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns
            df.columns = ['open', 'high', 'low', 'close']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Sort by date (oldest first)
            df = df.sort_index()
            
            # Add volume column (forex doesn't have volume, so we'll create a placeholder)
            df['volume'] = 1000000  # Placeholder volume
            
            self.logger.info(f"Successfully fetched {len(df)} data points")
            
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error fetching forex data: {str(e)}")
            return None
        except KeyError as e:
            self.logger.error(f"Data parsing error: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching forex data: {str(e)}")
            return None
    
    def get_real_time_quote(self, symbol):
        """
        Get real-time quote for a currency pair
        
        Args:
            symbol: Currency pair (e.g., "EUR/USD")
        
        Returns:
            dict: Real-time quote data
        """
        try:
            api_symbol = symbol.replace("/", "")
            
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": api_symbol[:3],
                "to_currency": api_symbol[3:],
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if "Error Message" in data:
                raise Exception(f"API Error: {data['Error Message']}")
            
            if "Realtime Currency Exchange Rate" not in data:
                raise Exception("No real-time data available")
            
            rate_data = data["Realtime Currency Exchange Rate"]
            
            return {
                'symbol': f"{rate_data['1. From_Currency Code']}/{rate_data['3. To_Currency Code']}",
                'rate': float(rate_data['5. Exchange Rate']),
                'last_refreshed': rate_data['6. Last Refreshed'],
                'bid': float(rate_data.get('8. Bid Price', rate_data['5. Exchange Rate'])),
                'ask': float(rate_data.get('9. Ask Price', rate_data['5. Exchange Rate']))
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time quote: {str(e)}")
            return None
    
    def get_supported_symbols(self):
        """Get list of supported currency pairs"""
        # Major currency pairs
        major_pairs = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD"
        ]
        
        # Minor pairs
        minor_pairs = [
            "EUR/GBP", "EUR/JPY", "EUR/CHF", "EUR/AUD",
            "EUR/CAD", "GBP/JPY", "GBP/CHF", "AUD/JPY"
        ]
        
        # Exotic pairs
        exotic_pairs = [
            "USD/SEK", "USD/NOK", "USD/DKK", "USD/PLN",
            "USD/HUF", "USD/CZK", "USD/TRY", "USD/ZAR"
        ]
        
        return {
            "major": major_pairs,
            "minor": minor_pairs,
            "exotic": exotic_pairs
        }
    
    def validate_api_key(self):
        """Validate API key by making a test request"""
        try:
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": "USD",
                "to_currency": "EUR",
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            data = response.json()
            
            if "Error Message" in data:
                return False, data["Error Message"]
            
            if "Note" in data and "rate limit" in data["Note"].lower():
                return False, "API rate limit reached"
            
            if "Realtime Currency Exchange Rate" in data:
                return True, "API key is valid"
            
            return False, "Unexpected API response"
            
        except Exception as e:
            return False, f"API validation error: {str(e)}"
