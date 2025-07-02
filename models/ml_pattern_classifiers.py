"""
Advanced Machine Learning Pattern Classifiers
Integration of sophisticated pattern detection algorithms
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TripleBottomClassifier:
    def __init__(self, threshold=0.7):
        """Initialize the Triple Bottom Pattern Classifier"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.threshold = threshold
        self.is_fitted = False
        
    def extract_features(self, ohlc_data):
        """Extract features that characterize triple bottom patterns"""
        if len(ohlc_data) < 15:
            return None
            
        ohlc_array = np.array(ohlc_data)
        opens = ohlc_array[:, 0]
        highs = ohlc_array[:, 1]
        lows = ohlc_array[:, 2]
        closes = ohlc_array[:, 3]
        volumes = ohlc_array[:, 4]
        
        features = {}
        
        # Find local minima (potential bottoms)
        low_peaks, _ = find_peaks(-lows, distance=3)
        high_peaks, _ = find_peaks(highs, distance=2)
        
        # Basic price statistics
        features['price_range'] = (np.max(highs) - np.min(lows)) / np.mean(closes)
        features['volatility'] = np.std(closes) / np.mean(closes)
        
        # Triple bottom specific features
        if len(low_peaks) >= 3:
            lowest_indices = low_peaks[np.argsort(lows[low_peaks])[:3]]
            lowest_indices = np.sort(lowest_indices)
            bottom1, bottom2, bottom3 = lowest_indices
            
            bottom_lows = lows[[bottom1, bottom2, bottom3]]
            features['bottom_similarity'] = 1 - (np.max(bottom_lows) - np.min(bottom_lows)) / np.mean(bottom_lows)
            features['bottom_distance1'] = bottom2 - bottom1
            features['bottom_distance2'] = bottom3 - bottom2
            
            # Resistance levels
            resistance1_candidates = high_peaks[(high_peaks > bottom1) & (high_peaks < bottom2)]
            resistance2_candidates = high_peaks[(high_peaks > bottom2) & (high_peaks < bottom3)]
            
            resistance1 = np.max(highs[resistance1_candidates]) if len(resistance1_candidates) > 0 else 0
            resistance2 = np.max(highs[resistance2_candidates]) if len(resistance2_candidates) > 0 else 0
            
            if resistance1 > 0 and resistance2 > 0:
                resistance_level = (resistance1 + resistance2) / 2
                features['resistance_strength'] = (resistance_level - np.max(bottom_lows)) / np.mean(closes)
                features['resistance_similarity'] = 1 - abs(resistance1 - resistance2) / np.mean([resistance1, resistance2])
            else:
                features['resistance_strength'] = 0
                features['resistance_similarity'] = 0
                
            # Volume analysis
            vol_bottom1 = volumes[bottom1]
            vol_bottom2 = volumes[bottom2]
            vol_bottom3 = volumes[bottom3]
            features['volume_decline'] = vol_bottom3 / ((vol_bottom1 + vol_bottom2) / 2) if vol_bottom1 + vol_bottom2 > 0 else 1
        else:
            features['bottom_similarity'] = 0
            features['bottom_distance1'] = 0
            features['bottom_distance2'] = 0
            features['resistance_strength'] = 0
            features['resistance_similarity'] = 0
            features['volume_decline'] = 1
            
        # Pattern shape features
        features['trend_reversal'] = self._calculate_trend_reversal(closes)
        features['breakout_strength'] = self._calculate_breakout_strength(closes, highs)
        
        # Price momentum features
        features['early_momentum'] = (closes[len(closes)//4] - closes[0]) / closes[0] if closes[0] != 0 else 0
        features['late_momentum'] = (closes[-1] - closes[len(closes)*3//4]) / closes[len(closes)*3//4] if closes[len(closes)*3//4] != 0 else 0
        
        return features
    
    def _calculate_trend_reversal(self, closes):
        """Calculate trend reversal strength"""
        if len(closes) < 8:
            return 0
        
        quarter = len(closes) // 4
        early_trend = (closes[quarter] - closes[0]) / closes[0] if closes[0] != 0 else 0
        late_trend = (closes[-1] - closes[-quarter]) / closes[-quarter] if closes[-quarter] != 0 else 0
        
        if early_trend < 0 and late_trend > 0:
            return abs(late_trend - early_trend)
        return 0
    
    def _calculate_breakout_strength(self, closes, highs):
        """Calculate breakout strength in the final portion"""
        if len(closes) < 8:
            return 0
            
        final_quarter = len(closes) * 3 // 4
        max_early = np.max(highs[:final_quarter])
        final_high = np.max(highs[final_quarter:])
        
        if final_high > max_early:
            return (final_high - max_early) / max_early
        return 0
    
    def predict(self, ohlc_data):
        """Predict if a pattern is a triple bottom"""
        if not self.is_fitted:
            # Auto-train with synthetic data if not trained
            self.train_with_synthetic_data()
            
        features = self.extract_features(ohlc_data)
        if features is None:
            return 0, 0.0
            
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        try:
            feature_vector_scaled = self.scaler.transform(feature_vector)
            probability = self.model.predict_proba(feature_vector_scaled)[0][1]
            prediction = 1 if probability >= self.threshold else 0
            return prediction, probability
        except:
            return 0, 0.0
    
    def train_with_synthetic_data(self):
        """Train the model with synthetic triple bottom patterns"""
        # Generate synthetic training data
        positive_samples = []
        negative_samples = []
        
        # Create 20 positive samples (triple bottom patterns)
        for _ in range(20):
            pattern = self._generate_synthetic_triple_bottom()
            positive_samples.append(pattern)
            
        # Create 30 negative samples (non-triple bottom patterns)  
        for _ in range(30):
            pattern = self._generate_random_pattern()
            negative_samples.append(pattern)
            
        # Prepare training data
        X = []
        y = []
        
        for pattern in positive_samples:
            features = self.extract_features(pattern)
            if features:
                X.append(list(features.values()))
                y.append(1)
                
        for pattern in negative_samples:
            features = self.extract_features(pattern)
            if features:
                X.append(list(features.values()))
                y.append(0)
                
        if len(X) > 0:
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_fitted = True
    
    def _generate_synthetic_triple_bottom(self):
        """Generate a synthetic triple bottom pattern"""
        length = 25
        start_price = 100
        bottom_price = start_price * 0.85
        resistance_price = start_price * 0.92
        
        prices = []
        
        # First decline to bottom
        for i in range(5):
            price = start_price - (start_price - bottom_price) * (i/4)
            prices.append(price)
            
        # First recovery
        for i in range(3):
            price = bottom_price + (resistance_price - bottom_price) * (i/2)
            prices.append(price)
            
        # Second decline
        for i in range(4):
            price = resistance_price - (resistance_price - bottom_price) * (i/3)
            prices.append(price)
            
        # Second recovery
        for i in range(3):
            price = bottom_price + (resistance_price - bottom_price) * (i/2)
            prices.append(price)
            
        # Third decline
        for i in range(4):
            price = resistance_price - (resistance_price - bottom_price) * (i/3)
            prices.append(price)
            
        # Final breakout
        for i in range(6):
            price = bottom_price + (start_price * 1.1 - bottom_price) * (i/5)
            prices.append(price)
            
        return self._prices_to_ohlc(prices)
    
    def _generate_random_pattern(self):
        """Generate a random non-triple bottom pattern"""
        length = 25
        start_price = 100
        prices = [start_price]
        
        for _ in range(length - 1):
            change = np.random.normal(0, 0.02) * prices[-1]
            prices.append(max(prices[-1] + change, 1))
            
        return self._prices_to_ohlc(prices)
    
    def _prices_to_ohlc(self, prices):
        """Convert price series to OHLC format"""
        ohlc = []
        for i, price in enumerate(prices):
            high = price * np.random.uniform(1.00, 1.02)
            low = price * np.random.uniform(0.98, 1.00)
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = np.random.randint(1000, 5000)
            ohlc.append([open_price, high, low, close, volume])
        return ohlc


class DoubleBottomClassifier:
    def __init__(self, threshold=0.7):
        """Initialize the Double Bottom Pattern Classifier"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.threshold = threshold
        self.is_fitted = False
        
    def extract_features(self, ohlc_data):
        """Extract features that characterize double bottom patterns"""
        if len(ohlc_data) < 10:
            return None
            
        ohlc_array = np.array(ohlc_data)
        highs = ohlc_array[:, 1]
        lows = ohlc_array[:, 2]
        closes = ohlc_array[:, 3]
        volumes = ohlc_array[:, 4]
        
        features = {}
        
        # Find local minima and maxima
        low_peaks, _ = find_peaks(-lows, distance=3)
        high_peaks, _ = find_peaks(highs, distance=2)
        
        # Basic statistics
        features['price_range'] = (np.max(highs) - np.min(lows)) / np.mean(closes)
        features['volatility'] = np.std(closes) / np.mean(closes)
        
        # Double bottom specific features
        if len(low_peaks) >= 2:
            lowest_indices = low_peaks[np.argsort(lows[low_peaks])[:2]]
            lowest_indices = np.sort(lowest_indices)
            
            features['bottom_distance'] = abs(lowest_indices[1] - lowest_indices[0])
            
            bottom1_low = lows[lowest_indices[0]]
            bottom2_low = lows[lowest_indices[1]]
            features['bottom_similarity'] = 1 - abs(bottom1_low - bottom2_low) / np.mean([bottom1_low, bottom2_low])
            
            # Resistance level
            if len(high_peaks) > 0:
                resistance_candidates = high_peaks[(high_peaks > lowest_indices[0]) & (high_peaks < lowest_indices[1])]
                if len(resistance_candidates) > 0:
                    resistance_level = np.max(highs[resistance_candidates])
                    features['resistance_strength'] = (resistance_level - np.max([bottom1_low, bottom2_low])) / np.mean(closes)
                else:
                    features['resistance_strength'] = 0
            else:
                features['resistance_strength'] = 0
                
            # Volume analysis
            vol_at_bottom1 = volumes[lowest_indices[0]]
            vol_at_bottom2 = volumes[lowest_indices[1]]
            avg_volume = np.mean(volumes)
            features['volume_increase'] = max(vol_at_bottom1, vol_at_bottom2) / avg_volume
        else:
            features['bottom_distance'] = 0
            features['bottom_similarity'] = 0
            features['resistance_strength'] = 0
            features['volume_increase'] = 1
            
        # Additional features
        features['trend_reversal'] = self._calculate_trend_reversal(closes)
        features['w_shape_score'] = self._calculate_w_shape_score(lows)
        features['breakout_strength'] = self._calculate_breakout_strength(closes, highs)
        
        return features
    
    def _calculate_trend_reversal(self, closes):
        """Calculate trend reversal strength"""
        if len(closes) < 6:
            return 0
        
        third = len(closes) // 3
        early_trend = (closes[third] - closes[0]) / closes[0] if closes[0] != 0 else 0
        late_trend = (closes[-1] - closes[-third]) / closes[-third] if closes[-third] != 0 else 0
        
        if early_trend < 0 and late_trend > 0:
            return abs(late_trend - early_trend)
        return 0
    
    def _calculate_w_shape_score(self, lows):
        """Calculate how well the pattern resembles a W shape"""
        if len(lows) < 10:
            return 0
            
        q1 = len(lows) // 4
        q2 = len(lows) // 2
        q3 = 3 * len(lows) // 4
        
        w_score = 0
        if np.mean(lows[:q1]) > np.mean(lows[q1:q2]):
            w_score += 1
        if np.mean(lows[q1:q2]) < np.mean(lows[q2:q3]):
            w_score += 1
        if np.mean(lows[q2:q3]) > np.mean(lows[q3:]):
            w_score += 1
            
        return w_score / 3
    
    def _calculate_breakout_strength(self, closes, highs):
        """Calculate breakout strength"""
        if len(closes) < 6:
            return 0
            
        final_third = len(closes) * 2 // 3
        max_early = np.max(highs[:final_third])
        final_high = np.max(highs[final_third:])
        
        if final_high > max_early:
            return (final_high - max_early) / max_early
        return 0
    
    def predict(self, ohlc_data):
        """Predict if a pattern is a double bottom"""
        if not self.is_fitted:
            self.train_with_synthetic_data()
            
        features = self.extract_features(ohlc_data)
        if features is None:
            return 0, 0.0
            
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        try:
            feature_vector_scaled = self.scaler.transform(feature_vector)
            probability = self.model.predict_proba(feature_vector_scaled)[0][1]
            prediction = 1 if probability >= self.threshold else 0
            return prediction, probability
        except:
            return 0, 0.0
    
    def train_with_synthetic_data(self):
        """Train with synthetic data"""
        # Similar training approach as triple bottom
        positive_samples = []
        negative_samples = []
        
        for _ in range(20):
            pattern = self._generate_synthetic_double_bottom()
            positive_samples.append(pattern)
            
        for _ in range(30):
            pattern = self._generate_random_pattern()
            negative_samples.append(pattern)
            
        X = []
        y = []
        
        for pattern in positive_samples:
            features = self.extract_features(pattern)
            if features:
                X.append(list(features.values()))
                y.append(1)
                
        for pattern in negative_samples:
            features = self.extract_features(pattern)
            if features:
                X.append(list(features.values()))
                y.append(0)
                
        if len(X) > 0:
            X = np.array(X)
            y = np.array(y)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_fitted = True
    
    def _generate_synthetic_double_bottom(self):
        """Generate synthetic double bottom pattern"""
        length = 18
        start_price = 100
        bottom_price = start_price * 0.85
        resistance_price = start_price * 0.92
        
        prices = []
        
        # First decline
        for i in range(4):
            price = start_price - (start_price - bottom_price) * (i/3)
            prices.append(price)
            
        # First recovery
        for i in range(3):
            price = bottom_price + (resistance_price - bottom_price) * (i/2)
            prices.append(price)
            
        # Second decline
        for i in range(4):
            price = resistance_price - (resistance_price - bottom_price) * (i/3)
            prices.append(price)
            
        # Final breakout
        for i in range(7):
            price = bottom_price + (start_price * 1.1 - bottom_price) * (i/6)
            prices.append(price)
            
        return self._prices_to_ohlc(prices)
    
    def _generate_random_pattern(self):
        """Generate random pattern"""
        length = 18
        start_price = 100
        prices = [start_price]
        
        for _ in range(length - 1):
            change = np.random.normal(0, 0.02) * prices[-1]
            prices.append(max(prices[-1] + change, 1))
            
        return self._prices_to_ohlc(prices)
    
    def _prices_to_ohlc(self, prices):
        """Convert to OHLC format"""
        ohlc = []
        for i, price in enumerate(prices):
            high = price * np.random.uniform(1.00, 1.02)
            low = price * np.random.uniform(0.98, 1.00)
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = np.random.randint(1000, 5000)
            ohlc.append([open_price, high, low, close, volume])
        return ohlc


class DoubleTopClassifier:
    def __init__(self, threshold=0.7):
        """Initialize the Double Top Pattern Classifier"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.threshold = threshold
        self.is_fitted = False
        
    def extract_features(self, ohlc_data):
        """Extract double top features"""
        if len(ohlc_data) < 10:
            return None
            
        ohlc_array = np.array(ohlc_data)
        highs = ohlc_array[:, 1]
        lows = ohlc_array[:, 2]
        closes = ohlc_array[:, 3]
        volumes = ohlc_array[:, 4]
        
        features = {}
        
        # Find peaks and valleys
        high_peaks, _ = find_peaks(highs, distance=3)
        low_valleys, _ = find_peaks(-lows, distance=2)
        
        features['price_range'] = (np.max(highs) - np.min(lows)) / np.mean(closes)
        features['volatility'] = np.std(closes) / np.mean(closes)
        
        # Double top specific features
        if len(high_peaks) >= 2:
            highest_indices = high_peaks[np.argsort(highs[high_peaks])[-2:]]
            highest_indices = np.sort(highest_indices)
            
            features['top_distance'] = abs(highest_indices[1] - highest_indices[0])
            
            top1_high = highs[highest_indices[0]]
            top2_high = highs[highest_indices[1]]
            features['top_similarity'] = 1 - abs(top1_high - top2_high) / np.mean([top1_high, top2_high])
            
            # Support level
            if len(low_valleys) > 0:
                support_candidates = low_valleys[(low_valleys > highest_indices[0]) & (low_valleys < highest_indices[1])]
                if len(support_candidates) > 0:
                    support_level = np.min(lows[support_candidates])
                    features['support_strength'] = (np.max([top1_high, top2_high]) - support_level) / np.mean(closes)
                else:
                    features['support_strength'] = 0
            else:
                features['support_strength'] = 0
                
            # Volume analysis
            vol_at_top1 = volumes[highest_indices[0]]
            vol_at_top2 = volumes[highest_indices[1]]
            avg_volume = np.mean(volumes)
            features['volume_increase'] = max(vol_at_top1, vol_at_top2) / avg_volume
        else:
            features['top_distance'] = 0
            features['top_similarity'] = 0
            features['support_strength'] = 0
            features['volume_increase'] = 1
            
        features['trend_reversal'] = self._calculate_trend_reversal(closes)
        features['m_shape_score'] = self._calculate_m_shape_score(highs)
        features['breakdown_strength'] = self._calculate_breakdown_strength(closes, lows)
        
        return features
    
    def _calculate_trend_reversal(self, closes):
        """Calculate trend reversal for double top"""
        if len(closes) < 6:
            return 0
        
        third = len(closes) // 3
        early_trend = (closes[third] - closes[0]) / closes[0] if closes[0] != 0 else 0
        late_trend = (closes[-1] - closes[-third]) / closes[-third] if closes[-third] != 0 else 0
        
        if early_trend > 0 and late_trend < 0:
            return abs(early_trend - late_trend)
        return 0
    
    def _calculate_m_shape_score(self, highs):
        """Calculate M shape score"""
        if len(highs) < 10:
            return 0
            
        q1 = len(highs) // 4
        q2 = len(highs) // 2
        q3 = 3 * len(highs) // 4
        
        m_score = 0
        if np.mean(highs[:q1]) < np.mean(highs[q1:q2]):
            m_score += 1
        if np.mean(highs[q1:q2]) > np.mean(highs[q2:q3]):
            m_score += 1
        if np.mean(highs[q2:q3]) < np.mean(highs[q3:]):
            m_score += 1
            
        return m_score / 3
    
    def _calculate_breakdown_strength(self, closes, lows):
        """Calculate breakdown strength"""
        if len(closes) < 6:
            return 0
            
        final_third = len(closes) * 2 // 3
        min_early = np.min(lows[:final_third])
        final_low = np.min(lows[final_third:])
        
        if final_low < min_early:
            return (min_early - final_low) / min_early
        return 0
    
    def predict(self, ohlc_data):
        """Predict double top pattern"""
        if not self.is_fitted:
            self.train_with_synthetic_data()
            
        features = self.extract_features(ohlc_data)
        if features is None:
            return 0, 0.0
            
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        try:
            feature_vector_scaled = self.scaler.transform(feature_vector)
            probability = self.model.predict_proba(feature_vector_scaled)[0][1]
            prediction = 1 if probability >= self.threshold else 0
            return prediction, probability
        except:
            return 0, 0.0
    
    def train_with_synthetic_data(self):
        """Train with synthetic data"""
        positive_samples = []
        negative_samples = []
        
        for _ in range(20):
            pattern = self._generate_synthetic_double_top()
            positive_samples.append(pattern)
            
        for _ in range(30):
            pattern = self._generate_random_pattern()
            negative_samples.append(pattern)
            
        X = []
        y = []
        
        for pattern in positive_samples:
            features = self.extract_features(pattern)
            if features:
                X.append(list(features.values()))
                y.append(1)
                
        for pattern in negative_samples:
            features = self.extract_features(pattern)
            if features:
                X.append(list(features.values()))
                y.append(0)
                
        if len(X) > 0:
            X = np.array(X)
            y = np.array(y)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_fitted = True
    
    def _generate_synthetic_double_top(self):
        """Generate synthetic double top"""
        length = 18
        start_price = 100
        top_price = start_price * 1.15
        support_price = start_price * 1.08
        
        prices = []
        
        # First rise
        for i in range(4):
            price = start_price + (top_price - start_price) * (i/3)
            prices.append(price)
            
        # First decline
        for i in range(3):
            price = top_price - (top_price - support_price) * (i/2)
            prices.append(price)
            
        # Second rise
        for i in range(4):
            price = support_price + (top_price - support_price) * (i/3)
            prices.append(price)
            
        # Final breakdown
        for i in range(7):
            price = top_price - (top_price - start_price * 0.9) * (i/6)
            prices.append(price)
            
        return self._prices_to_ohlc(prices)
    
    def _generate_random_pattern(self):
        """Generate random pattern"""
        length = 18
        start_price = 100
        prices = [start_price]
        
        for _ in range(length - 1):
            change = np.random.normal(0, 0.02) * prices[-1]
            prices.append(max(prices[-1] + change, 1))
            
        return self._prices_to_ohlc(prices)
    
    def _prices_to_ohlc(self, prices):
        """Convert to OHLC format"""
        ohlc = []
        for i, price in enumerate(prices):
            high = price * np.random.uniform(1.00, 1.02)
            low = price * np.random.uniform(0.98, 1.00)
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = np.random.randint(1000, 5000)
            ohlc.append([open_price, high, low, close, volume])
        return ohlc


class InverseHeadAndShouldersClassifier:
    def __init__(self, threshold=0.7):
        """Initialize the Inverse Head and Shoulders Pattern Classifier"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.threshold = threshold
        self.is_fitted = False
        
    def extract_features(self, ohlc_data):
        """Extract inverse head and shoulders features"""
        if len(ohlc_data) < 15:
            return None
            
        ohlc_array = np.array(ohlc_data)
        highs = ohlc_array[:, 1]
        lows = ohlc_array[:, 2]
        closes = ohlc_array[:, 3]
        volumes = ohlc_array[:, 4]
        
        features = {}
        
        # Find troughs and peaks
        troughs, _ = find_peaks(-lows, distance=3)
        peaks, _ = find_peaks(highs, distance=2)
        
        features['price_range'] = (np.max(highs) - np.min(lows)) / np.mean(closes)
        features['volatility'] = np.std(closes) / np.mean(closes)
        
        # Inverse head and shoulders specific features
        if len(troughs) >= 3:
            deepest_indices = troughs[np.argsort(lows[troughs])[:3]]
            sorted_indices = np.sort(deepest_indices)
            left_idx, head_idx, right_idx = sorted_indices
            
            head_low = lows[head_idx]
            left_low = lows[left_idx]
            right_low = lows[right_idx]
            
            if head_low < left_low and head_low < right_low:
                shoulder_avg = (left_low + right_low) / 2
                features['shoulder_symmetry'] = 1 - abs(left_low - right_low) / shoulder_avg
                features['head_prominence'] = (min(left_low, right_low) - head_low) / shoulder_avg
                
                # Neckline
                left_peak_candidates = peaks[(peaks > left_idx) & (peaks < head_idx)]
                right_peak_candidates = peaks[(peaks > head_idx) & (peaks < right_idx)]
                
                left_peak = np.max(highs[left_peak_candidates]) if len(left_peak_candidates) > 0 else 0
                right_peak = np.max(highs[right_peak_candidates]) if len(right_peak_candidates) > 0 else 0
                
                if left_peak > 0 and right_peak > 0:
                    neckline = (left_peak + right_peak) / 2
                    features['neckline_strength'] = (neckline - head_low) / np.mean(closes)
                    features['neckline_slope'] = (right_peak - left_peak) / (right_idx - left_idx)
                else:
                    features['neckline_strength'] = 0
                    features['neckline_slope'] = 0
                    
                # Volume analysis
                vol_left = volumes[left_idx]
                vol_head = volumes[head_idx]
                vol_right = volumes[right_idx]
                
                features['volume_head_ratio'] = vol_head / ((vol_left + vol_right) / 2) if (vol_left + vol_right) > 0 else 1
                features['volume_right_ratio'] = vol_right / vol_left if vol_left > 0 else 1
            else:
                features['shoulder_symmetry'] = 0
                features['head_prominence'] = 0
                features['neckline_strength'] = 0
                features['neckline_slope'] = 0
                features['volume_head_ratio'] = 1
                features['volume_right_ratio'] = 1
        else:
            features['shoulder_symmetry'] = 0
            features['head_prominence'] = 0
            features['neckline_strength'] = 0
            features['neckline_slope'] = 0
            features['volume_head_ratio'] = 1
            features['volume_right_ratio'] = 1
            
        features['trend_reversal'] = self._calculate_trend_reversal(closes)
        features['breakout_strength'] = self._calculate_breakout_strength(closes, highs)
        
        return features
    
    def _calculate_trend_reversal(self, closes):
        """Calculate trend reversal strength"""
        if len(closes) < 8:
            return 0
        
        quarter = len(closes) // 4
        early_trend = (closes[quarter] - closes[0]) / closes[0] if closes[0] != 0 else 0
        late_trend = (closes[-1] - closes[-quarter]) / closes[-quarter] if closes[-quarter] != 0 else 0
        
        if early_trend < 0 and late_trend > 0:
            return abs(late_trend - early_trend)
        return 0
    
    def _calculate_breakout_strength(self, closes, highs):
        """Calculate breakout strength"""
        if len(closes) < 8:
            return 0
            
        final_quarter = len(closes) * 3 // 4
        max_early = np.max(highs[:final_quarter])
        final_high = np.max(highs[final_quarter:])
        
        if final_high > max_early:
            return (final_high - max_early) / max_early
        return 0
    
    def predict(self, ohlc_data):
        """Predict inverse head and shoulders pattern"""
        if not self.is_fitted:
            self.train_with_synthetic_data()
            
        features = self.extract_features(ohlc_data)
        if features is None:
            return 0, 0.0
            
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        try:
            feature_vector_scaled = self.scaler.transform(feature_vector)
            probability = self.model.predict_proba(feature_vector_scaled)[0][1]
            prediction = 1 if probability >= self.threshold else 0
            return prediction, probability
        except:
            return 0, 0.0
    
    def train_with_synthetic_data(self):
        """Train with synthetic data"""
        positive_samples = []
        negative_samples = []
        
        for _ in range(20):
            pattern = self._generate_synthetic_inverse_head_shoulders()
            positive_samples.append(pattern)
            
        for _ in range(30):
            pattern = self._generate_random_pattern() 
            negative_samples.append(pattern)
            
        X = []
        y = []
        
        for pattern in positive_samples:
            features = self.extract_features(pattern)
            if features:
                X.append(list(features.values()))
                y.append(1)
                
        for pattern in negative_samples:
            features = self.extract_features(pattern)
            if features:
                X.append(list(features.values()))
                y.append(0)
                
        if len(X) > 0:
            X = np.array(X)
            y = np.array(y)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_fitted = True
    
    def _generate_synthetic_inverse_head_shoulders(self):
        """Generate synthetic inverse head and shoulders"""
        length = 25
        start_price = 100
        head_price = start_price * 0.8
        shoulder_price = start_price * 0.87
        neckline_price = start_price * 0.93
        
        prices = []
        
        # Left shoulder
        for i in range(4):
            price = start_price - (start_price - shoulder_price) * (i/3)
            prices.append(price)
            
        # Recovery to neckline
        for i in range(3):
            price = shoulder_price + (neckline_price - shoulder_price) * (i/2)
            prices.append(price)
            
        # Head formation
        for i in range(5):
            price = neckline_price - (neckline_price - head_price) * (i/4)
            prices.append(price)
            
        # Recovery to neckline
        for i in range(2):
            price = head_price + (neckline_price - head_price) * (i/1)
            prices.append(price)
            
        # Right shoulder
        for i in range(4):
            price = neckline_price - (neckline_price - shoulder_price) * (i/3)
            prices.append(price)
            
        # Final breakout
        for i in range(7):
            price = shoulder_price + (start_price * 1.1 - shoulder_price) * (i/6)
            prices.append(price)
            
        return self._prices_to_ohlc(prices)
    
    def _generate_random_pattern(self):
        """Generate random pattern"""
        length = 25
        start_price = 100
        prices = [start_price]
        
        for _ in range(length - 1):
            change = np.random.normal(0, 0.02) * prices[-1]
            prices.append(max(prices[-1] + change, 1))
            
        return self._prices_to_ohlc(prices)
    
    def _prices_to_ohlc(self, prices):
        """Convert to OHLC format"""
        ohlc = []
        for i, price in enumerate(prices):
            high = price * np.random.uniform(1.00, 1.02)
            low = price * np.random.uniform(0.98, 1.00)
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = np.random.randint(1000, 5000)
            ohlc.append([open_price, high, low, close, volume])
        return ohlc