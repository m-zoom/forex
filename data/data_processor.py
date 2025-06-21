"""
Data processing and preprocessing utilities
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler
import talib

class DataProcessor:
    def __init__(self, logger):
        self.logger = logger
        self.scaler = MinMaxScaler()
        
    def process_data(self, df):
        """
        Process raw forex data for analysis
        
        Args:
            df: Raw OHLCV DataFrame
        
        Returns:
            pandas.DataFrame: Processed data with technical indicators
        """
        try:
            # Make a copy to avoid modifying original data
            processed_df = df.copy()
            
            # Ensure proper data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            
            # Remove any rows with NaN values
            processed_df = processed_df.dropna()
            
            # Sort by index (time)
            processed_df = processed_df.sort_index()
            
            # Add technical indicators
            processed_df = self.add_technical_indicators(processed_df)
            
            # Find peaks and valleys
            processed_df = self.find_peaks_valleys(processed_df)
            
            # Add price change indicators
            processed_df = self.add_price_changes(processed_df)
            
            self.logger.info(f"Data processing completed. Shape: {processed_df.shape}")
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            return df
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the DataFrame"""
        try:
            # Moving averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_5'] = df['close'].ewm(span=5).mean()
            df['ema_10'] = df['close'].ewm(span=10).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
            bb_std_dev = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
            
            # RSI (Relative Strength Index)
            if len(df) >= 14:
                df['rsi'] = self.calculate_rsi(df['close'], period=14)
            
            # MACD
            if len(df) >= 26:
                df = self.calculate_macd(df)
            
            # Stochastic Oscillator
            if len(df) >= 14:
                df = self.calculate_stochastic(df)
            
            # Average True Range (ATR)
            if len(df) >= 14:
                df['atr'] = self.calculate_atr(df)
            
            # Volume indicators (if volume data is available)
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=10).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df
    
    def calculate_stochastic(self, df, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        return df
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def find_peaks_valleys(self, df, window=5):
        """Find local peaks and valleys in price data"""
        try:
            # Find peaks (local maxima)
            high_peaks = argrelextrema(df['high'].values, np.greater, order=window)[0]
            df['peak'] = np.nan
            if len(high_peaks) > 0:
                df.iloc[high_peaks, df.columns.get_loc('peak')] = df['high'].iloc[high_peaks]
            
            # Find valleys (local minima)
            low_valleys = argrelextrema(df['low'].values, np.less, order=window)[0]
            df['valley'] = np.nan
            if len(low_valleys) > 0:
                df.iloc[low_valleys, df.columns.get_loc('valley')] = df['low'].iloc[low_valleys]
            
            # Mark significant peaks and valleys
            df['is_peak'] = ~df['peak'].isna()
            df['is_valley'] = ~df['valley'].isna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error finding peaks and valleys: {str(e)}")
            return df
    
    def add_price_changes(self, df):
        """Add price change indicators"""
        try:
            # Price changes
            df['price_change'] = df['close'].diff()
            df['price_change_pct'] = df['close'].pct_change() * 100
            
            # High-low spread
            df['hl_spread'] = df['high'] - df['low']
            df['hl_spread_pct'] = (df['hl_spread'] / df['close']) * 100
            
            # Open-close spread
            df['oc_spread'] = df['close'] - df['open']
            df['oc_spread_pct'] = (df['oc_spread'] / df['open']) * 100
            
            # Volatility measures
            df['volatility'] = df['price_change_pct'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding price changes: {str(e)}")
            return df
    
    def normalize_data(self, df, columns=None):
        """Normalize specified columns using MinMaxScaler"""
        try:
            if columns is None:
                columns = ['open', 'high', 'low', 'close']
            
            normalized_df = df.copy()
            
            for col in columns:
                if col in df.columns:
                    # Reshape for scaler
                    values = df[col].values.reshape(-1, 1)
                    normalized_values = self.scaler.fit_transform(values)
                    normalized_df[f'{col}_normalized'] = normalized_values.flatten()
            
            return normalized_df
            
        except Exception as e:
            self.logger.error(f"Error normalizing data: {str(e)}")
            return df
    
    def create_sequences(self, df, sequence_length=60, target_column='close'):
        """Create sequences for time series analysis"""
        try:
            sequences = []
            targets = []
            
            data = df[target_column].values
            
            for i in range(sequence_length, len(data)):
                sequences.append(data[i-sequence_length:i])
                targets.append(data[i])
            
            return np.array(sequences), np.array(targets)
            
        except Exception as e:
            self.logger.error(f"Error creating sequences: {str(e)}")
            return None, None
    
    def calculate_support_resistance_levels(self, df, window=20, num_levels=3):
        """Calculate support and resistance levels"""
        try:
            levels = []
            
            # Get peaks and valleys
            peaks = df['peak'].dropna()
            valleys = df['valley'].dropna()
            
            # Combine all significant levels
            all_levels = list(peaks.values) + list(valleys.values)
            
            if len(all_levels) < num_levels:
                return []
            
            # Cluster similar levels
            all_levels.sort()
            clustered_levels = []
            
            for level in all_levels:
                if not clustered_levels:
                    clustered_levels.append([level])
                else:
                    # Check if level is close to any existing cluster
                    added_to_cluster = False
                    for cluster in clustered_levels:
                        if abs(level - np.mean(cluster)) < df['close'].mean() * 0.002:  # 0.2% threshold
                            cluster.append(level)
                            added_to_cluster = True
                            break
                    
                    if not added_to_cluster:
                        clustered_levels.append([level])
            
            # Calculate average level for each cluster and count touches
            for cluster in clustered_levels:
                avg_level = np.mean(cluster)
                strength = len(cluster)
                
                # Determine if it's support or resistance
                current_price = df['close'].iloc[-1]
                level_type = 'support' if avg_level < current_price else 'resistance'
                
                levels.append({
                    'level': avg_level,
                    'strength': strength,
                    'type': level_type,
                    'touch_count': len(cluster)
                })
            
            # Sort by strength and return top levels
            levels.sort(key=lambda x: x['strength'], reverse=True)
            return levels[:num_levels]
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance levels: {str(e)}")
            return []
    
    def validate_data_quality(self, df):
        """Validate data quality and completeness"""
        try:
            issues = []
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.sum() > 0:
                issues.append(f"Missing values found: {missing_counts.to_dict()}")
            
            # Check for duplicate timestamps
            if df.index.duplicated().sum() > 0:
                issues.append(f"Duplicate timestamps found: {df.index.duplicated().sum()}")
            
            # Check for unrealistic price movements
            price_changes = df['close'].pct_change().abs()
            extreme_changes = price_changes > 0.1  # 10% price change
            if extreme_changes.sum() > 0:
                issues.append(f"Extreme price movements detected: {extreme_changes.sum()}")
            
            # Check data consistency (high >= low, etc.)
            consistency_issues = (df['high'] < df['low']).sum()
            if consistency_issues > 0:
                issues.append(f"Data inconsistencies (high < low): {consistency_issues}")
            
            return {
                'is_valid': len(issues) == 0,
                'issues': issues,
                'data_points': len(df),
                'date_range': f"{df.index.min()} to {df.index.max()}"
            }
            
        except Exception as e:
            self.logger.error(f"Error validating data quality: {str(e)}")
            return {'is_valid': False, 'issues': [str(e)]}
