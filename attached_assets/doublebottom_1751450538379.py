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

class DoubleBottomClassifier:
    def __init__(self, threshold=0.7):
        """
        Initialize the Double Bottom Pattern Classifier
        
        Args:
            threshold (float): Hard threshold for classification (default: 0.7)
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.threshold = threshold
        self.is_fitted = False
        
    def extract_features(self, ohlc_data):
        """
        Extract features that characterize double bottom patterns
        
        Args:
            ohlc_data (list): List of [Open, High, Low, Close, Volume] data points
            
        Returns:
            dict: Dictionary of extracted features
        """
        if len(ohlc_data) < 10:
            return None
            
        ohlc_array = np.array(ohlc_data)
        opens = ohlc_array[:, 0]
        highs = ohlc_array[:, 1]
        lows = ohlc_array[:, 2]
        closes = ohlc_array[:, 3]
        volumes = ohlc_array[:, 4]
        
        features = {}
        
        # 1. Find local minima (potential bottoms)
        low_peaks, _ = find_peaks(-lows, distance=3)
        
        # 2. Find local maxima (potential resistance)
        high_peaks, _ = find_peaks(highs, distance=2)
        
        # 3. Basic price statistics
        features['price_range'] = (np.max(highs) - np.min(lows)) / np.mean(closes)
        features['volatility'] = np.std(closes) / np.mean(closes)
        
        # 4. Double bottom specific features
        if len(low_peaks) >= 2:
            # Get the two lowest points
            lowest_indices = low_peaks[np.argsort(lows[low_peaks])[:2]]
            lowest_indices = np.sort(lowest_indices)
            
            # Distance between bottoms
            features['bottom_distance'] = abs(lowest_indices[1] - lowest_indices[0])
            
            # Height similarity of bottoms
            bottom1_low = lows[lowest_indices[0]]
            bottom2_low = lows[lowest_indices[1]]
            features['bottom_similarity'] = 1 - abs(bottom1_low - bottom2_low) / np.mean([bottom1_low, bottom2_low])
            
            # Resistance level between bottoms
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
            if lowest_indices[1] < len(volumes) - 1:
                vol_at_bottom1 = volumes[lowest_indices[0]]
                vol_at_bottom2 = volumes[lowest_indices[1]]
                avg_volume = np.mean(volumes)
                features['volume_increase'] = max(vol_at_bottom1, vol_at_bottom2) / avg_volume
            else:
                features['volume_increase'] = 1
                
        else:
            features['bottom_distance'] = 0
            features['bottom_similarity'] = 0
            features['resistance_strength'] = 0
            features['volume_increase'] = 1
            
        # 5. Pattern shape features
        features['trend_reversal'] = self._calculate_trend_reversal(closes)
        features['w_shape_score'] = self._calculate_w_shape_score(lows)
        
        # 6. Breakout characteristics
        features['breakout_strength'] = self._calculate_breakout_strength(closes, highs)
        
        # 7. Price momentum features
        features['early_momentum'] = (closes[len(closes)//3] - closes[0]) / closes[0]
        features['late_momentum'] = (closes[-1] - closes[len(closes)*2//3]) / closes[len(closes)*2//3]
        
        return features
    
    def _calculate_trend_reversal(self, closes):
        """Calculate trend reversal strength"""
        if len(closes) < 6:
            return 0
        
        third = len(closes) // 3
        early_trend = (closes[third] - closes[0]) / closes[0]
        late_trend = (closes[-1] - closes[-third]) / closes[-third]
        
        # Good double bottom should show decline then recovery
        if early_trend < 0 and late_trend > 0:
            return abs(late_trend - early_trend)
        return 0
    
    def _calculate_w_shape_score(self, lows):
        """Calculate how well the pattern resembles a W shape"""
        if len(lows) < 10:
            return 0
            
        # Divide into 4 quarters and check for W pattern
        q1 = len(lows) // 4
        q2 = len(lows) // 2
        q3 = 3 * len(lows) // 4
        
        # W pattern: high -> low -> high -> low -> high
        w_score = 0
        
        # First decline
        if np.mean(lows[:q1]) > np.mean(lows[q1:q2]):
            w_score += 1
            
        # Recovery
        if np.mean(lows[q1:q2]) < np.mean(lows[q2:q3]):
            w_score += 1
            
        # Second decline
        if np.mean(lows[q2:q3]) > np.mean(lows[q3:]):
            w_score += 1
            
        return w_score / 3
    
    def _calculate_breakout_strength(self, closes, highs):
        """Calculate breakout strength in the final portion"""
        if len(closes) < 6:
            return 0
            
        final_third = len(closes) * 2 // 3
        max_early = np.max(highs[:final_third])
        final_high = np.max(highs[final_third:])
        
        if final_high > max_early:
            return (final_high - max_early) / max_early
        return 0
    
    def generate_negative_samples(self, n_samples=50):
        """Generate negative samples (non-double bottom patterns)"""
        negative_samples = []
        
        for i in range(n_samples):
            # Generate random patterns that are NOT double bottoms
            pattern_length = np.random.randint(15, 25)
            
            # Random walk pattern
            if i < n_samples // 3:
                prices = self._generate_random_walk(pattern_length)
            # Trending pattern
            elif i < 2 * n_samples // 3:
                prices = self._generate_trending_pattern(pattern_length)
            # Single bottom pattern
            else:
                prices = self._generate_single_bottom_pattern(pattern_length)
                
            negative_samples.append(prices)
            
        return negative_samples
    
    def _generate_random_walk(self, length):
        """Generate random walk price pattern"""
        start_price = np.random.uniform(50, 200)
        prices = [start_price]
        
        for _ in range(length - 1):
            change = np.random.normal(0, 0.02) * prices[-1]
            prices.append(max(prices[-1] + change, 1))
            
        return self._prices_to_ohlc(prices)
    
    def _generate_trending_pattern(self, length):
        """Generate trending price pattern"""
        start_price = np.random.uniform(50, 200)
        trend = np.random.uniform(-0.02, 0.02)
        
        prices = []
        for i in range(length):
            noise = np.random.normal(0, 0.01) * start_price
            price = start_price * (1 + trend * i) + noise
            prices.append(max(price, 1))
            
        return self._prices_to_ohlc(prices)
    
    def _generate_single_bottom_pattern(self, length):
        """Generate single bottom pattern (not double bottom)"""
        start_price = np.random.uniform(50, 200)
        bottom_price = start_price * 0.8
        
        # Decline to bottom
        decline_length = length // 2
        recovery_length = length - decline_length
        
        prices = []
        
        # Decline phase
        for i in range(decline_length):
            progress = i / decline_length
            price = start_price - (start_price - bottom_price) * progress
            noise = np.random.normal(0, 0.01) * price
            prices.append(price + noise)
            
        # Recovery phase
        for i in range(recovery_length):
            progress = i / recovery_length
            price = bottom_price + (start_price - bottom_price) * progress * 1.1
            noise = np.random.normal(0, 0.01) * price
            prices.append(price + noise)
            
        return self._prices_to_ohlc(prices)
    
    def _prices_to_ohlc(self, prices):
        """Convert price series to OHLC format"""
        ohlc = []
        for i, price in enumerate(prices):
            # Add some intraday variation
            high = price * np.random.uniform(1.00, 1.03)
            low = price * np.random.uniform(0.97, 1.00)
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = np.random.randint(1000, 5000)
            
            ohlc.append([open_price, high, low, close, volume])
            
        return ohlc
    
    def prepare_training_data(self):
        """Prepare training data from double bottom patterns and negative samples"""
        # Double bottom patterns data
        double_bottom_patterns = {
            "pattern_1": [
                [100, 98, 101, 97, 1500], [98, 95, 99, 94, 1600],
                [95, 92, 96, 91, 1700], [92, 91, 93, 90, 1800],
                [91, 93, 94, 90, 1600], [93, 96, 97, 92, 1500],
                [96, 98, 99, 95, 1400], [98, 97, 99, 96, 1300],
                [97, 95, 98, 94, 1400], [95, 93, 96, 92, 1500],
                [93, 91, 94, 90, 1600], [91, 92, 93, 90, 1700],
                [92, 94, 95, 91, 1600], [94, 97, 98, 93, 1500],
                [97, 99, 100, 96, 1400], [99, 102, 103, 98, 1300],
                [102, 104, 105, 101, 1200], [104, 106, 107, 103, 1100]
            ],
            "pattern_2": [
                [85, 83, 86, 82, 2100], [83, 80, 84, 79, 2200],
                [80, 77, 81, 76, 2400], [77, 76, 78, 75, 2500],
                [76, 78, 79, 77, 2300], [78, 81, 82, 80, 2100],
                [81, 83, 84, 82, 1900], [83, 81, 84, 80, 2000],
                [81, 78, 82, 77, 2200], [78, 75, 79, 74, 2400],
                [75, 76, 77, 75, 2500], [76, 79, 80, 78, 2200],
                [79, 82, 83, 81, 2000], [82, 85, 86, 84, 1800],
                [85, 87, 88, 86, 1700]
            ],
            "pattern_3": [
                [150, 147, 152, 145, 1800],
                [147, 144, 149, 142, 1900],
                [144, 140, 146, 139, 2100],  # First bottom
                [140, 138, 142, 137, 2200],
                [138, 141, 143, 140, 2000],
                [141, 144, 146, 143, 1800],
                [144, 146, 148, 145, 1600],  # Resistance
                [146, 143, 147, 141, 1700],  # Second decline
                [143, 140, 144, 138, 1900],
                [140, 137, 141, 136, 2100],  # Second bottom
                [137, 139, 140, 138, 2200],
                [139, 142, 144, 141, 2000],  # Recovery
                [142, 145, 147, 144, 1800],
                [145, 148, 150, 147, 1600],  # Breakout
                [148, 151, 153, 150, 1500]
            ],
            
            "pattern_4": [
                [65, 62, 66, 61, 3200],
                [62, 59, 63, 58, 3400],
                [59, 56, 60, 55, 3600],     # First bottom
                [56, 55, 57, 54, 3800],
                [55, 57, 58, 56, 3500],
                [57, 60, 61, 59, 3200],
                [60, 62, 63, 61, 3000],     # Resistance
                [62, 59, 63, 58, 3100],     # Second decline
                [59, 56, 60, 55, 3300],
                [56, 54, 57, 53, 3600],     # Second bottom
                [54, 56, 57, 55, 3700],
                [56, 59, 60, 58, 3400],     # Recovery
                [59, 62, 64, 61, 3100],
                [62, 65, 67, 64, 2900],     # Breakout
                [65, 68, 69, 67, 2700]
            ],
            
            "pattern_5": [
                [45, 43, 46, 42, 4100],
                [43, 40, 44, 39, 4300],
                [40, 37, 41, 36, 4500],     # First bottom
                [37, 36, 38, 35, 4700],
                [36, 38, 39, 37, 4400],
                [38, 41, 42, 40, 4100],
                [41, 43, 44, 42, 3900],     # Resistance
                [43, 40, 44, 39, 4000],     # Second decline
                [40, 37, 41, 36, 4200],
                [37, 35, 38, 34, 4500],     # Second bottom
                [35, 37, 38, 36, 4600],
                [37, 40, 41, 39, 4300],     # Recovery
                [40, 43, 45, 42, 4000],
                [43, 46, 48, 45, 3800],     # Breakout
                [46, 49, 50, 48, 3600]
            ],
            
            "pattern_6": [
                [120, 117, 122, 115, 2800],
                [117, 114, 119, 112, 3000],
                [114, 110, 116, 109, 3200],  # First bottom
                [110, 108, 112, 107, 3400],
                [108, 111, 113, 110, 3100],
                [111, 114, 116, 113, 2900],
                [114, 116, 118, 115, 2700],  # Resistance
                [116, 113, 117, 111, 2800],  # Second decline
                [113, 110, 114, 108, 3000],
                [110, 107, 111, 106, 3200],  # Second bottom
                [107, 109, 111, 108, 3300],
                [109, 112, 114, 111, 3100],  # Recovery
                [112, 115, 117, 114, 2900],
                [115, 118, 120, 117, 2700],  # Breakout
                [118, 121, 123, 120, 2500]
            ],
            
            "pattern_7": [
                [200, 196, 202, 194, 1600],
                [196, 192, 198, 190, 1700],
                [192, 188, 194, 186, 1900],  # First bottom
                [188, 186, 190, 185, 2000],
                [186, 189, 191, 188, 1800],
                [189, 192, 194, 191, 1700],
                [192, 194, 196, 193, 1500],  # Resistance
                [194, 191, 195, 189, 1600],  # Second decline
                [191, 188, 192, 186, 1800],
                [188, 185, 189, 184, 1900],  # Second bottom
                [185, 187, 189, 186, 2000],
                [187, 190, 192, 189, 1800],  # Recovery
                [190, 193, 195, 192, 1600],
                [193, 196, 198, 195, 1400],  # Breakout
                [196, 199, 201, 198, 1300]
            ],
            
            "pattern_8": [
                [75, 72, 76, 70, 2600],
                [72, 69, 73, 67, 2800],
                [69, 65, 70, 64, 3000],     # First bottom
                [65, 63, 67, 62, 3200],
                [63, 66, 68, 65, 2900],
                [66, 69, 71, 68, 2700],
                [69, 71, 73, 70, 2500],     # Resistance
                [71, 68, 72, 66, 2600],     # Second decline
                [68, 65, 69, 63, 2800],
                [65, 62, 66, 61, 3000],     # Second bottom
                [62, 64, 66, 63, 3100],
                [64, 67, 69, 66, 2900],     # Recovery
                [67, 70, 72, 69, 2700],
                [70, 73, 75, 72, 2500],     # Breakout
                [73, 76, 78, 75, 2300]
            ],
            
            "pattern_9": [
                [55, 52, 56, 50, 3800],
                [52, 49, 53, 47, 4000],
                [49, 45, 50, 44, 4200],     # First bottom
                [45, 43, 47, 42, 4400],
                [43, 46, 48, 45, 4100],
                [46, 49, 51, 48, 3900],
                [49, 51, 53, 50, 3700],     # Resistance
                [51, 48, 52, 46, 3800],     # Second decline
                [48, 45, 49, 43, 4000],
                [45, 42, 46, 41, 4200],     # Second bottom
                [42, 44, 46, 43, 4300],
                [44, 47, 49, 46, 4100],     # Recovery
                [47, 50, 52, 49, 3900],
                [50, 53, 55, 52, 3700],     # Breakout
                [53, 56, 58, 55, 3500]
            ],
            
            "pattern_10": [
                [310, 305, 312, 302, 1200],
                [305, 300, 307, 298, 1300],
                [300, 295, 302, 293, 1500],  # First bottom
                [295, 292, 297, 291, 1600],
                [292, 296, 298, 295, 1400],
                [296, 300, 302, 299, 1300],
                [300, 303, 305, 302, 1100],  # Resistance
                [303, 299, 304, 297, 1200],  # Second decline
                [299, 295, 300, 293, 1400],
                [295, 291, 296, 290, 1500],  # Second bottom
                [291, 294, 296, 293, 1600],
                [294, 298, 300, 297, 1400],  # Recovery
                [298, 302, 304, 301, 1200],
                [302, 306, 308, 305, 1000],  # Breakout
                [306, 310, 312, 309, 900]
            ],
            
            "pattern_11": [
                [90, 87, 91, 85, 2400],
                [87, 84, 88, 82, 2600],
                [84, 80, 85, 79, 2800],     # First bottom
                [80, 78, 82, 77, 3000],
                [78, 81, 83, 80, 2700],
                [81, 84, 86, 83, 2500],
                [84, 86, 88, 85, 2300],     # Resistance
                [86, 83, 87, 81, 2400],     # Second decline
                [83, 80, 84, 78, 2600],
                [80, 77, 81, 76, 2800],     # Second bottom
                [77, 79, 81, 78, 2900],
                [79, 82, 84, 81, 2700],     # Recovery
                [82, 85, 87, 84, 2500],
                [85, 88, 90, 87, 2300],     # Breakout
                [88, 91, 93, 90, 2100]
            ],
            
            "pattern_12": [
                [175, 171, 177, 169, 1900],
                [171, 167, 173, 165, 2100],
                [167, 163, 169, 161, 2300],  # First bottom
                [163, 160, 165, 159, 2500],
                [160, 164, 166, 163, 2200],
                [164, 168, 170, 167, 2000],
                [168, 171, 173, 170, 1800],  # Resistance
                [171, 167, 172, 165, 1900],  # Second decline
                [167, 163, 168, 161, 2100],
                [163, 159, 164, 158, 2300],  # Second bottom
                [159, 162, 164, 161, 2400],
                [162, 166, 168, 165, 2200],  # Recovery
                [166, 170, 172, 169, 2000],
                [170, 174, 176, 173, 1800],  # Breakout
                [174, 178, 180, 177, 1600]
            ],
            
            "pattern_13": [
                [32, 30, 33, 29, 5200],
                [30, 27, 31, 26, 5400],
                [27, 24, 28, 23, 5600],     # First bottom
                [24, 22, 25, 21, 5800],
                [22, 25, 27, 24, 5500],
                [25, 28, 30, 27, 5200],
                [28, 30, 32, 29, 5000],     # Resistance
                [30, 27, 31, 25, 5100],     # Second decline
                [27, 24, 28, 22, 5300],
                [24, 21, 25, 20, 5500],     # Second bottom
                [21, 23, 25, 22, 5600],
                [23, 26, 28, 25, 5400],     # Recovery
                [26, 29, 31, 28, 5100],
                [29, 32, 34, 31, 4900],     # Breakout
                [32, 35, 37, 34, 4700]
            ],
            
            "pattern_14": [
                [260, 255, 262, 252, 1400],
                [255, 250, 257, 248, 1500],
                [250, 245, 252, 243, 1700],  # First bottom
                [245, 242, 247, 241, 1800],
                [242, 246, 248, 245, 1600],
                [246, 250, 252, 249, 1500],
                [250, 253, 255, 252, 1300],  # Resistance
                [253, 249, 254, 247, 1400],  # Second decline
                [249, 245, 250, 243, 1600],
                [245, 241, 246, 240, 1700],  # Second bottom
                [241, 244, 246, 243, 1800],
                [244, 248, 250, 247, 1600],  # Recovery
                [248, 252, 254, 251, 1400],
                [252, 256, 258, 255, 1200],  # Breakout
                [256, 260, 262, 259, 1100]
            ],
            
            "pattern_15": [
                [125, 122, 127, 120, 2200],
                [122, 118, 124, 116, 2400],
                [118, 114, 120, 112, 2600],  # First bottom
                [114, 111, 116, 110, 2800],
                [111, 115, 117, 114, 2500],
                [115, 119, 121, 118, 2300],
                [119, 122, 124, 121, 2100],  # Resistance
                [122, 118, 123, 116, 2200],  # Second decline
                [118, 114, 119, 112, 2400],
                [114, 110, 115, 109, 2600],  # Second bottom
                [110, 113, 115, 112, 2700],
                [113, 117, 119, 116, 2500],  # Recovery
                [117, 121, 123, 120, 2300],
                [121, 125, 127, 124, 2100],  # Breakout
                [125, 129, 131, 128, 1900]
            ],
            
            "pattern_16": [
                [48, 45, 49, 43, 4300],
                [45, 42, 46, 40, 4500],
                [42, 38, 43, 37, 4700],     # First bottom
                [38, 36, 40, 35, 4900],
                [36, 39, 41, 38, 4600],
                [39, 42, 44, 41, 4400],
                [42, 44, 46, 43, 4200],     # Resistance
                [44, 41, 45, 39, 4300],     # Second decline
                [41, 38, 42, 36, 4500],
                [38, 35, 39, 34, 4700],     # Second bottom
                [35, 37, 39, 36, 4800],
                [37, 40, 42, 39, 4600],     # Recovery
                [40, 43, 45, 42, 4400],
                [43, 46, 48, 45, 4200],     # Breakout
                [46, 49, 51, 48, 4000]
            ],
            
            "pattern_17": [
                [380, 375, 382, 372, 800],
                [375, 370, 377, 368, 900],
                [370, 365, 372, 363, 1100],  # First bottom
                [365, 362, 367, 361, 1200],
                [362, 366, 368, 365, 1000],
                [366, 370, 372, 369, 900],
                [370, 373, 375, 372, 700],   # Resistance
                [373, 369, 374, 367, 800],   # Second decline
                [369, 365, 370, 363, 1000],
                [365, 361, 366, 360, 1100],  # Second bottom
                [361, 364, 366, 363, 1200],
                [364, 368, 370, 367, 1000],  # Recovery
                [368, 372, 374, 371, 800],
                [372, 376, 378, 375, 600],   # Breakout
                [376, 380, 382, 379, 500]
            ],
            
            "pattern_18": [
                [95, 92, 96, 90, 2300],
                [92, 88, 93, 86, 2500],
                [88, 84, 89, 82, 2700],     # First bottom
                [84, 81, 86, 80, 2900],
                [81, 85, 87, 84, 2600],
                [85, 89, 91, 88, 2400],
                [89, 92, 94, 91, 2200],     # Resistance
                [92, 88, 93, 86, 2300],     # Second decline
                [88, 84, 89, 82, 2500],
                [84, 80, 85, 79, 2700],     # Second bottom
                [80, 83, 85, 82, 2800],
                [83, 87, 89, 86, 2600],     # Recovery
                [87, 91, 93, 90, 2400],
                [91, 95, 97, 94, 2200],     # Breakout
                [95, 99, 101, 98, 2000]
            ],
            
            "pattern_19": [
                [220, 216, 222, 214, 1700],
                [216, 211, 218, 209, 1800],
                [211, 206, 213, 204, 2000],  # First bottom
                [206, 203, 208, 202, 2100],
                [203, 207, 209, 206, 1900],
                [207, 211, 213, 210, 1800],
                [211, 214, 216, 213, 1600],  # Resistance
                [214, 210, 215, 208, 1700],  # Second decline
                [210, 206, 211, 204, 1900],
                [206, 202, 207, 201, 2000],  # Second bottom
                [202, 205, 207, 204, 2100],
                [205, 209, 211, 208, 1900],  # Recovery
                [209, 213, 215, 212, 1700],
                [213, 217, 219, 216, 1500],  # Breakout
                [217, 221, 223, 220, 1400]
            ],
            
            "pattern_20": [
                [68, 65, 69, 63, 3100],
                [65, 61, 66, 59, 3300],
                [61, 57, 62, 55, 3500],     # First bottom
                [57, 54, 59, 53, 3700],
                [54, 58, 60, 57, 3400],
                [58, 62, 64, 61, 3200],
                [62, 65, 67, 64, 3000],     # Resistance
                [65, 61, 66, 59, 3100],     # Second decline
                [61, 57, 62, 55, 3300],
                [57, 53, 58, 52, 3500],     # Second bottom
                [53, 56, 58, 55, 3600],
                [56, 60, 62, 59, 3400],     # Recovery
                [60, 64, 66, 63, 3200],
                [64, 68, 70, 67, 3000],     # Breakout
                [68, 72, 74, 71, 2800]
            ],
            
            "pattern_21": [
                [145, 141, 147, 139, 2000],
                [141, 137, 143, 135, 2200],
                [137, 132, 139, 131, 2400],  # First bottom
                [132, 129, 134, 128, 2600],
                [129, 133, 135, 132, 2300],
                [133, 137, 139, 136, 2100],
                [137, 140, 142, 139, 1900],  # Resistance
                [140, 136, 141, 134, 2000],  # Second decline
                [136, 132, 137, 130, 2200],
                [132, 128, 133, 127, 2400],  # Second bottom
                [128, 131, 133, 130, 2500],
                [131, 135, 137, 134, 2300],  # Recovery
                [135, 139, 141, 138, 2100],
                [139, 143, 145, 142, 1900],  # Breakout
                [143, 147, 149, 146, 1700]
            ],
            
            "pattern_22": [
                [29, 27, 30, 26, 5800],
                [27, 24, 28, 23, 6000],
                [24, 20, 25, 19, 6200],     # First bottom
                [20, 18, 22, 17, 6400],
                [18, 21, 23, 20, 6100],
                [21, 24, 26, 23, 5900],
                [24, 26, 28, 25, 5700],     # Resistance
                [26, 23, 27, 21, 5800],     # Second decline
                [23, 20, 24, 18, 6000],
                [20, 17, 21, 16, 6200],     # Second bottom
                [17, 19, 21, 18, 6300],
                [19, 22, 24, 21, 6100],     # Recovery
                [22, 25, 27, 24, 5900],
                [25, 28, 30, 27, 5700],     # Breakout
                [28, 31, 33, 30, 5500]
            ],
            
            "pattern_23": [
                [185, 181, 187, 179, 1800],
                [181, 176, 183, 174, 2000],
                [176, 171, 178, 169, 2200],  # First bottom
                [171, 168, 173, 167, 2400],
                [168, 172, 174, 171, 2100],
                [172, 176, 178, 175, 1900],
                [176, 179, 181, 178, 1700],  # Resistance
                [179, 175, 180, 173, 1800],  # Second decline
                [175, 171, 176, 169, 2000],
                [171, 167, 172, 166, 2200],  # Second bottom
                [167, 170, 172, 169, 2300],
                [170, 174, 176, 173, 2100],  # Recovery
                [174, 178, 180, 177, 1900],
                [178, 182, 184, 181, 1700],  # Breakout
                [182, 186, 188, 185, 1500]
            ],
            
            "pattern_24": [
                [112, 108, 114, 106, 2500],
                [108, 104, 110, 102, 2700],
                [104, 99, 106, 98, 2900],   # First bottom
                [99, 96, 101, 95, 3100],
                [96, 100, 102, 99, 2800],
                [100, 104, 106, 103, 2600],
                [104, 107, 109, 106, 2400],  # Resistance
                [107, 103, 108, 101, 2500],  # Second decline
                [103, 99, 104, 97, 2700],
                [99, 95, 100, 94, 2900],    # Second bottom
                [95, 98, 100, 97, 3000],
                [98, 102, 104, 101, 2800],  # Recovery
                [102, 106, 108, 105, 2600],
                [106, 110, 112, 109, 2400],  # Breakout
                [110, 114, 116, 113, 2200]
            ],
            
            "pattern_25": [
                [78, 74, 79, 72, 2900],
                [74, 70, 75, 68, 3100],
                [70, 65, 71, 64, 3300],     # First bottom
                [65, 62, 67, 61, 3500],
                [62, 66, 68, 65, 3200],
                [66, 70, 72, 69, 3000],
                [70, 73, 75, 72, 2800],     # Resistance
                [73, 69, 74, 67, 2900],     # Second decline
                [69, 65, 70, 63, 3100],
                [65, 61, 66, 60, 3300],     # Second bottom
                [61, 64, 66, 63, 3400],
                [64, 68, 70, 67, 3200],     # Recovery
                [68, 72, 74, 71, 3000],
                [72, 76, 78, 75, 2800],     # Breakout
                [76, 80, 82, 79, 2600]
            ],
            
            "pattern_26": [
                [340, 335, 342, 332, 1000],
                [335, 329, 337, 327, 1100],
                [329, 323, 331, 321, 1300],  # First bottom
                [323, 320, 325, 319, 1400],
                [320, 324, 326, 323, 1200],
                [324, 329, 331, 328, 1100],
                [329, 332, 334, 331, 900],   # Resistance
                [332, 327, 333, 325, 1000],  # Second decline
                [327, 323, 328, 321, 1200],
                [323, 319, 324, 318, 1300],  # Second bottom
                [319, 322, 324, 321, 1400],
                [322, 326, 328, 325, 1200],  # Recovery
                [326, 331, 333, 330, 1000],
                [331, 336, 338, 335, 800],   # Breakout
                [336, 341, 343, 340, 700]
            ],
            
            "pattern_27": [
                [158, 154, 160, 152, 2100],
                [154, 149, 156, 147, 2300],
                [149, 144, 151, 142, 2500],  # First bottom
                [144, 141, 146, 140, 2700],
                [141, 145, 147, 144, 2400],
                [145, 149, 151, 148, 2200],
                [149, 152, 154, 151, 2000],  # Resistance
                [152, 148, 153, 146, 2100],  # Second decline
                [148, 144, 149, 142, 2300],
                [144, 140, 145, 139, 2500],  # Second bottom
                [140, 143, 145, 142, 2600],
                [143, 147, 149, 146, 2400],  # Recovery
                [147, 151, 153, 150, 2200],
                [151, 155, 157, 154, 2000],  # Breakout
                [155, 159, 161, 158, 1800]
            ],
            
            "pattern_28": [
                [52, 49, 53, 47, 4000],
                [49, 45, 50, 43, 4200],
                [45, 40, 46, 39, 4400],     # First bottom
                [40, 37, 42, 36, 4600],
                [37, 41, 43, 40, 4300],
                [41, 45, 47, 44, 4100],
                [45, 48, 50, 47, 3900],     # Resistance
                [48, 44, 49, 42, 4000],     # Second decline
                [44, 40, 45, 38, 4200],
                [40, 36, 41, 35, 4400],     # Second bottom
                [36, 39, 41, 38, 4500],
                [39, 43, 45, 42, 4300],     # Recovery
                [43, 47, 49, 46, 4100],
                [47, 51, 53, 50, 3900],     # Breakout
                [51, 55, 57, 54, 3700]
            ],
            
            "pattern_29": [
                [88, 84, 89, 82, 2600],
                [84, 80, 85, 78, 2800],
                [80, 75, 81, 74, 3000],     # First bottom
                [75, 72, 77, 71, 3200],
                [72, 76, 78, 75, 2900],
                [76, 80, 82, 79, 2700],
                [80, 83, 85, 82, 2500],     # Resistance
                [83, 79, 84, 77, 2600],     # Second decline
                [79, 75, 80, 73, 2800],
                [75, 71, 76, 70, 3000],     # Second bottom
                [71, 74, 76, 73, 3100],
                [74, 78, 80, 77, 2900],     # Recovery
                [78, 82, 84, 81, 2700],
                [82, 86, 88, 85, 2500],     # Breakout
                [86, 90, 92, 89, 2300]
            ],
            
            "pattern_30": [
                [275, 270, 277, 268, 1300],
                [270, 264, 272, 262, 1400],
                [264, 258, 266, 256, 1600],  # First bottom
                [258, 255, 260, 254, 1700],
                [255, 259, 261, 258, 1500],
                [259, 264, 266, 263, 1400],
                [264, 267, 269, 266, 1200],  # Resistance
                [267, 262, 268, 260, 1300],  # Second decline
                [262, 258, 263, 256, 1500],
                [258, 254, 259, 253, 1600],  # Second bottom
                [254, 257, 259, 256, 1700],
                [257, 261, 263, 260, 1500],  # Recovery
                [261, 266, 268, 265, 1300],
                [266, 270, 272, 269, 1100],  # Breakout
                [270, 275, 277, 274, 1000]
            ],
            
            "pattern_31": [
                [42, 39, 43, 37, 4800],
                [39, 35, 40, 33, 5000],
                [35, 30, 36, 29, 5200],     # First bottom
                [30, 27, 32, 26, 5400],
                [27, 31, 33, 30, 5100],
                [31, 35, 37, 34, 4900],
                [35, 38, 40, 37, 4700],     # Resistance
                [38, 34, 39, 32, 4800],     # Second decline
                [34, 30, 35, 28, 5000],
                [30, 26, 31, 25, 5200],     # Second bottom
                [26, 29, 31, 28, 5300],
                [29, 33, 35, 32, 5100],     # Recovery
                [33, 37, 39, 36, 4900],
                [37, 41, 43, 40, 4700],     # Breakout
                [41, 45, 47, 44, 4500]
            ],
            
            "pattern_32": [
                [195, 191, 197, 189, 1800],
                [191, 186, 193, 184, 2000],
                [186, 181, 188, 179, 2200],  # First bottom
                [181, 178, 183, 177, 2400],
                [178, 182, 184, 181, 2100],
                [182, 186, 188, 185, 1900],
                [186, 189, 191, 188, 1700],  # Resistance
                [189, 185, 190, 183, 1800],  # Second decline
                [185, 181, 186, 179, 2000],
                [181, 177, 182, 176, 2200],  # Second bottom
                [177, 180, 182, 179, 2300],
                [180, 184, 186, 183, 2100],  # Recovery
                [184, 188, 190, 187, 1900],
                [188, 192, 194, 191, 1700],  # Breakout
                [192, 196, 198, 195, 1500]
            ],
            
            "pattern_33": [
                [137, 133, 139, 131, 2200],
                [133, 128, 135, 126, 2400],
                [128, 123, 130, 121, 2600],  # First bottom
                [123, 120, 125, 119, 2800],
                [120, 124, 126, 123, 2500],
                [124, 128, 130, 127, 2300],
                [128, 131, 133, 130, 2100],  # Resistance
                [131, 127, 132, 125, 2200],  # Second decline
                [127, 123, 128, 121, 2400],
                [123, 119, 124, 118, 2600],  # Second bottom
                [119, 122, 124, 121, 2700],
                [122, 126, 128, 125, 2500],  # Recovery
                [126, 130, 132, 129, 2300],
                [130, 134, 136, 133, 2100],  # Breakout
                [134, 138, 140, 137, 1900]
            ],
            
            "pattern_34": [
                [63, 59, 64, 57, 3400],
                [59, 55, 60, 53, 3600],
                [55, 50, 56, 49, 3800],     # First bottom
                [50, 47, 52, 46, 4000],
                [47, 51, 53, 50, 3700],
                [51, 55, 57, 54, 3500],
                [55, 58, 60, 57, 3300],     # Resistance
                [58, 54, 59, 52, 3400],     # Second decline
                [54, 50, 55, 48, 3600],
                [50, 46, 51, 45, 3800],     # Second bottom
                [46, 49, 51, 48, 3900],
                [49, 53, 55, 52, 3700],     # Recovery
                [53, 57, 59, 56, 3500],
                [57, 61, 63, 60, 3300],     # Breakout
                [61, 65, 67, 64, 3100]
            ],
            
            "pattern_35": [
                [245, 240, 247, 238, 1500],
                [240, 234, 242, 232, 1600],
                [234, 228, 236, 226, 1800],  # First bottom
                [228, 225, 230, 224, 1900],
                [225, 229, 231, 228, 1700],
                [229, 234, 236, 233, 1600],
                [234, 237, 239, 236, 1400],  # Resistance
                [237, 232, 238, 230, 1500],  # Second decline
                [232, 228, 233, 226, 1700],
                [228, 224, 229, 223, 1800],  # Second bottom
                [224, 227, 229, 226, 1900],
                [227, 231, 233, 230, 1700],  # Recovery
                [231, 236, 238, 235, 1500],
                [236, 240, 242, 239, 1300],  # Breakout
                [240, 245, 247, 244, 1200]
            ],
            
            "pattern_36": [
                [104, 100, 106, 98, 2700],
                [100, 96, 102, 94, 2900],
                [96, 91, 98, 90, 3100],     # First bottom
                [91, 88, 93, 87, 3300],
                [88, 92, 94, 91, 3000],
                [92, 96, 98, 95, 2800],
                [96, 99, 101, 98, 2600],    # Resistance
                [99, 95, 100, 93, 2700],    # Second decline
                [95, 91, 96, 89, 2900],
                [91, 87, 92, 86, 3100],     # Second bottom
                [87, 90, 92, 89, 3200],
                [90, 94, 96, 93, 3000],     # Recovery
                [94, 98, 100, 97, 2800],
                [98, 102, 104, 101, 2600],  # Breakout
                [102, 106, 108, 105, 2400]
            ],
            
            "pattern_37": [
                [72, 68, 73, 66, 3200],
                [68, 64, 69, 62, 3400],
                [64, 59, 65, 58, 3600],     # First bottom
                [59, 56, 61, 55, 3800],
                [56, 60, 62, 59, 3500],
                [60, 64, 66, 63, 3300],
                [64, 67, 69, 66, 3100],     # Resistance
                [67, 63, 68, 61, 3200],     # Second decline
                [63, 59, 64, 57, 3400],
                [59, 55, 60, 54, 3600],     # Second bottom
                [55, 58, 60, 57, 3700],
                [58, 62, 64, 61, 3500],     # Recovery
                [62, 66, 68, 65, 3300],
                [66, 70, 72, 69, 3100],     # Breakout
                [70, 74, 76, 73, 2900]
            ],
            
            "pattern_38": [
                [315, 310, 317, 308, 1100],
                [310, 304, 312, 302, 1200],
                [304, 298, 306, 296, 1400],  # First bottom
                [298, 295, 300, 294, 1500],
                [295, 299, 301, 298, 1300],
                [299, 304, 306, 303, 1200],
                [304, 307, 309, 306, 1000],  # Resistance
                [307, 302, 308, 300, 1100],  # Second decline
                [302, 298, 303, 296, 1300],
                [298, 294, 299, 293, 1400],  # Second bottom
                [294, 297, 299, 296, 1500],
                [297, 301, 303, 300, 1300],  # Recovery
                [301, 306, 308, 305, 1100],
                [306, 310, 312, 309, 900],   # Breakout
                [310, 315, 317, 314, 800]
            ],
            
            "pattern_39": [
                [86, 82, 87, 80, 2800],
                [82, 78, 83, 76, 3000],
                [78, 73, 79, 72, 3200],     # First bottom
                [73, 70, 75, 69, 3400],
                [70, 74, 76, 73, 3100],
                [74, 78, 80, 77, 2900],
                [78, 81, 83, 80, 2700],     # Resistance
                [81, 77, 82, 75, 2800],     # Second decline
                [77, 73, 78, 71, 3000],
                [73, 69, 74, 68, 3200],     # Second bottom
                [69, 72, 74, 71, 3300],
                [72, 76, 78, 75, 3100],     # Recovery
                [76, 80, 82, 79, 2900],
                [80, 84, 86, 83, 2700],     # Breakout
                [84, 88, 90, 87, 2500]
            ],
            
            "pattern_40": [
                [165, 160, 167, 158, 2000],
                [160, 155, 162, 153, 2200],
                [155, 149, 157, 148, 2400],  # First bottom
                [149, 146, 151, 145, 2600],
                [146, 150, 152, 149, 2300],
                [150, 155, 157, 154, 2100],
                [155, 158, 160, 157, 1900],  # Resistance
                [158, 153, 159, 151, 2000],  # Second decline
                [153, 149, 154, 147, 2200],
                [149, 145, 150, 144, 2400],  # Second bottom
                [145, 148, 150, 147, 2500],
                [148, 152, 154, 151, 2300],  # Recovery
                [152, 157, 159, 156, 2100],
                [157, 161, 163, 160, 1900],  # Breakout
                [161, 166, 168, 165, 1700]
            ],
            
            "pattern_41": [
                [38, 35, 39, 33, 5100],
                [35, 31, 36, 29, 5300],
                [31, 26, 32, 25, 5500],     # First bottom
                [26, 23, 28, 22, 5700],
                [23, 27, 29, 26, 5400],
                [27, 31, 33, 30, 5200],
                [31, 34, 36, 33, 5000],     # Resistance
                [34, 30, 35, 28, 5100],     # Second decline
                [30, 26, 31, 24, 5300],
                [26, 22, 27, 21, 5500],     # Second bottom
                [22, 25, 27, 24, 5600],
                [25, 29, 31, 28, 5400],     # Recovery
                [29, 33, 35, 32, 5200],
                [33, 37, 39, 36, 5000],     # Breakout
                [37, 41, 43, 40, 4800]
            ],
            
            "pattern_42": [
                [225, 220, 227, 218, 1600],
                [220, 214, 222, 212, 1700],
                [214, 208, 216, 206, 1900],  # First bottom
                [208, 205, 210, 204, 2000],
                [205, 209, 211, 208, 1800],
                [209, 214, 216, 213, 1700],
                [214, 217, 219, 216, 1500],  # Resistance
                [217, 212, 218, 210, 1600],  # Second decline
                [212, 208, 213, 206, 1800],
                [208, 204, 209, 203, 1900],  # Second bottom
                [204, 207, 209, 206, 2000],
                [207, 211, 213, 210, 1800],  # Recovery
                [211, 216, 218, 215, 1600],
                [216, 220, 222, 219, 1400],  # Breakout
                [220, 225, 227, 224, 1300]
            ],
            
            "pattern_43": [
                [118, 114, 120, 112, 2400],
                [114, 109, 116, 107, 2600],
                [109, 104, 111, 103, 2800],  # First bottom
                [104, 101, 106, 100, 3000],
                [101, 105, 107, 104, 2700],
                [105, 109, 111, 108, 2500],
                [109, 112, 114, 111, 2300],  # Resistance
                [112, 108, 113, 106, 2400],  # Second decline
                [108, 104, 109, 102, 2600],
                [104, 100, 105, 99, 2800],   # Second bottom
                [100, 103, 105, 102, 2900],
                [103, 107, 109, 106, 2700],  # Recovery
                [107, 111, 113, 110, 2500],
                [111, 115, 117, 114, 2300],  # Breakout
                [115, 119, 121, 118, 2100]
            ],
            
            "pattern_44": [
                [59, 55, 60, 53, 3800],
                [55, 51, 56, 49, 4000],
                [51, 46, 52, 45, 4200],     # First bottom
                [46, 43, 48, 42, 4400],
                [43, 47, 49, 46, 4100],
                [47, 51, 53, 50, 3900],
                [51, 54, 56, 53, 3700],     # Resistance
                [54, 50, 55, 48, 3800],     # Second decline
                [50, 46, 51, 44, 4000],
                [46, 42, 47, 41, 4200],     # Second bottom
                [42, 45, 47, 44, 4300],
                [45, 49, 51, 48, 4100],     # Recovery
                [49, 53, 55, 52, 3900],
                [53, 57, 59, 56, 3700],     # Breakout
                [57, 61, 63, 60, 3500]
            ],
            
            "pattern_45": [
                [295, 290, 297, 288, 1200],
                [290, 284, 292, 282, 1300],
                [284, 278, 286, 276, 1500],  # First bottom
                [278, 275, 280, 274, 1600],
                [275, 279, 281, 278, 1400],
                [279, 284, 286, 283, 1300],
                [284, 287, 289, 286, 1100],  # Resistance
                [287, 282, 288, 280, 1200],  # Second decline
                [282, 278, 283, 276, 1400],
                [278, 274, 279, 273, 1500],  # Second bottom
                [274, 277, 279, 276, 1600],
                [277, 281, 283, 280, 1400],  # Recovery
                [281, 286, 288, 285, 1200],
                [286, 290, 292, 289, 1000],  # Breakout
                [290, 295, 297, 294, 900]
            ],
            
            "pattern_46": [
                [81, 77, 82, 75, 2900],
                [77, 73, 78, 71, 3100],
                [73, 68, 74, 67, 3300],     # First bottom
                [68, 65, 70, 64, 3500],
                [65, 69, 71, 68, 3200],
                [69, 73, 75, 72, 3000],
                [73, 76, 78, 75, 2800],     # Resistance
                [76, 72, 77, 70, 2900],     # Second decline
                [72, 68, 73, 66, 3100],
                [68, 64, 69, 63, 3300],     # Second bottom
                [64, 67, 69, 66, 3400],
                [67, 71, 73, 70, 3200],     # Recovery
                [71, 75, 77, 74, 3000],
                [75, 79, 81, 78, 2800],     # Breakout
                [79, 83, 85, 82, 2600]
            ],
            
            "pattern_47": [
                [130, 125, 132, 123, 2300],
                [125, 120, 127, 118, 2500],
                [120, 114, 122, 113, 2700],  # First bottom
                [114, 111, 116, 110, 2900],
                [111, 115, 117, 114, 2600],
                [115, 120, 122, 119, 2400],
                [120, 123, 125, 122, 2200],  # Resistance
                [123, 118, 124, 116, 2300],  # Second decline
                [118, 114, 119, 112, 2500],
                [114, 110, 115, 109, 2700],  # Second bottom
                [110, 113, 115, 112, 2800],
                [113, 117, 119, 116, 2600],  # Recovery
                [117, 122, 124, 121, 2400],
                [122, 126, 128, 125, 2200],  # Breakout
                [126, 131, 133, 130, 2000]
            ],
            
            "pattern_48": [
                [47, 43, 48, 41, 4500],
                [43, 39, 44, 37, 4700],
                [39, 34, 40, 33, 4900],     # First bottom
                [34, 31, 36, 30, 5100],
                [31, 35, 37, 34, 4800],
                [35, 39, 41, 38, 4600],
                [39, 42, 44, 41, 4400],     # Resistance
                [42, 38, 43, 36, 4500],     # Second decline
                [38, 34, 39, 32, 4700],
                [34, 30, 35, 29, 4900],     # Second bottom
                [30, 33, 35, 32, 5000],
                [33, 37, 39, 36, 4800],     # Recovery
                [37, 41, 43, 40, 4600],
                [41, 45, 47, 44, 4400],     # Breakout
                [45, 49, 51, 48, 4200]
            ],
            
            "pattern_49": [
                [205, 200, 207, 198, 1700],
                [200, 194, 202, 192, 1800],
                [194, 188, 196, 186, 2000],  # First bottom
                [188, 185, 190, 184, 2100],
                [185, 189, 191, 188, 1900],
                [189, 194, 196, 193, 1800],
                [194, 197, 199, 196, 1600],  # Resistance
                [197, 192, 198, 190, 1700],  # Second decline
                [192, 188, 193, 186, 1900],
                [188, 184, 189, 183, 2000],  # Second bottom
                [184, 187, 189, 186, 2100],
                [187, 191, 193, 190, 1900],  # Recovery
                [191, 196, 198, 195, 1700],
                [196, 200, 202, 199, 1500],  # Breakout
                [200, 205, 207, 204, 1400]
            ],
            
            "pattern_50": [
                [92, 88, 93, 86, 2700],
                [88, 84, 89, 82, 2900],
                [84, 79, 85, 78, 3100],     # First bottom
                [79, 76, 81, 75, 3300],
                [76, 80, 82, 79, 3000],
                [80, 84, 86, 83, 2800],
                [84, 87, 89, 86, 2600],     # Resistance
                [87, 83, 88, 81, 2700],     # Second decline
                [83, 79, 84, 77, 2900],
                [79, 75, 80, 74, 3100],     # Second bottom
                [75, 78, 80, 77, 3200],
                [78, 82, 84, 81, 3000],     # Recovery
                [82, 86, 88, 85, 2800],
                [86, 90, 92, 89, 2600],     # Breakout
                [90, 94, 96, 93, 2400]
            ]
            # Adding a few more patterns for better training
        }
        
        # Add more double bottom patterns (simplified for brevity)
        for i in range(3, 21):  # Add patterns 3-20
            pattern_key = f"pattern_{i}"
            # Generate simplified double bottom pattern
            base_price = np.random.uniform(30, 300)
            pattern = self._generate_double_bottom_pattern(base_price)
            double_bottom_patterns[pattern_key] = pattern
        
        X = []
        y = []
        
        # Extract features from positive samples (double bottom patterns)
        for pattern_name, pattern_data in double_bottom_patterns.items():
            features = self.extract_features(pattern_data)
            if features:
                X.append(list(features.values()))
                y.append(1)  # Positive class
        
        # Generate and extract features from negative samples
        negative_samples = self.generate_negative_samples(len(double_bottom_patterns))
        for neg_pattern in negative_samples:
            features = self.extract_features(neg_pattern)
            if features:
                X.append(list(features.values()))
                y.append(0)  # Negative class
        
        return np.array(X), np.array(y)
    
    def _generate_double_bottom_pattern(self, base_price):
        """Generate a synthetic double bottom pattern"""
        pattern = []
        current_price = base_price
        
        # Phase 1: Initial decline
        for i in range(4):
            high = current_price * np.random.uniform(1.00, 1.02)
            low = current_price * np.random.uniform(0.95, 0.98)
            close = current_price * np.random.uniform(0.96, 0.99)
            volume = np.random.randint(1500, 2500)
            pattern.append([current_price, high, low, close, volume])
            current_price = close
        
        # Phase 2: First bottom
        bottom_price = current_price * 0.85
        pattern.append([current_price, current_price * 1.01, bottom_price, bottom_price * 1.02, np.random.randint(2000, 3000)])
        current_price = bottom_price * 1.02
        
        # Phase 3: Recovery to resistance
        resistance_price = base_price * 0.95
        for i in range(3):
            high = current_price * np.random.uniform(1.02, 1.05)
            low = current_price * np.random.uniform(0.98, 1.00)
            close = current_price * np.random.uniform(1.01, 1.04)
            volume = np.random.randint(1500, 2000)
            pattern.append([current_price, high, low, close, volume])
            current_price = close
        
        # Phase 4: Second decline
        for i in range(3):
            high = current_price * np.random.uniform(1.00, 1.02)
            low = current_price * np.random.uniform(0.95, 0.98)
            close = current_price * np.random.uniform(0.96, 0.99)
            volume = np.random.randint(1800, 2500)
            pattern.append([current_price, high, low, close, volume])
            current_price = close
        
        # Phase 5: Second bottom (similar to first)
        second_bottom = bottom_price * np.random.uniform(0.98, 1.02)
        pattern.append([current_price, current_price * 1.01, second_bottom, second_bottom * 1.02, np.random.randint(2500, 3500)])
        current_price = second_bottom * 1.02
        
        # Phase 6: Final breakout
        for i in range(4):
            high = current_price * np.random.uniform(1.03, 1.06)
            low = current_price * np.random.uniform(0.99, 1.01)
            close = current_price * np.random.uniform(1.02, 1.05)
            volume = np.random.randint(1200, 2000)
            pattern.append([current_price, high, low, close, volume])
            current_price = close
        
        return pattern
    
    def train(self):
        """Train the double bottom classifier"""
        print("Preparing training data...")
        X, y = self.prepare_training_data()
        
        print(f"Training data shape: {X.shape}")
        print(f"Positive samples: {np.sum(y == 1)}, Negative samples: {np.sum(y == 0)}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        print("Training the model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        # Make predictions with probability
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred_hard = (y_pred_proba >= self.threshold).astype(int)
        
        print(f"\nClassification Report (with threshold {self.threshold}):")
        print(classification_report(y_test, y_pred_hard))
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_hard))
        
        # Feature importance
        feature_names = ['price_range', 'volatility', 'bottom_distance', 'bottom_similarity', 
                        'resistance_strength', 'volume_increase', 'trend_reversal', 
                        'w_shape_score', 'breakout_strength', 'early_momentum', 'late_momentum']
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Most Important Features:")
        print(feature_importance.head())
        
        self.is_fitted = True
        return self.model
    
    def predict(self, ohlc_data):
        """
        Predict if a pattern is a double bottom
        
        Args:
            ohlc_data (list): OHLC data points
            
        Returns:
            int: 1 if double bottom pattern detected, 0 otherwise
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first. Call train() method.")
        
        features = self.extract_features(ohlc_data)
        if not features:
            return 0
        
        X = np.array([list(features.values())])
        X_scaled = self.scaler.transform(X)
        
        # Get probability and apply hard threshold
        probability = self.model.predict_proba(X_scaled)[0, 1]
        prediction = 1 if probability >= self.threshold else 0
        
        return prediction, probability
    
    def save_model(self, filepath=None):
        """
        Save the trained model to a pickle file
        
        Args:
            filepath (str): Path to save the model. If None, creates a timestamped filename
            
        Returns:
            str: Path where the model was saved
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first. Call train() method.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"double_bottom_classifier_{timestamp}.pkl"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"Model saved successfully to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving model: {e}")
            return None
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from a pickle file
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            DoubleBottomClassifier: Loaded classifier instance
        """
        try:
            with open(filepath, 'rb') as f:
                classifier = pickle.load(f)
            
            # Verify it's the right type
            if not isinstance(classifier, cls):
                raise ValueError("Loaded object is not a DoubleBottomClassifier instance")
            
            print(f"Model loaded successfully from: {filepath}")
            print(f"Model threshold: {classifier.threshold}")
            print(f"Model trained: {'Yes' if classifier.is_fitted else 'No'}")
            
            return classifier
        except FileNotFoundError:
            print(f"Error: Model file not found at {filepath}")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def get_model_info(self):
        """
        Get information about the current model
        
        Returns:
            dict: Model information
        """
        info = {
            'is_fitted': self.is_fitted,
            'threshold': self.threshold,
            'model_type': 'RandomForestClassifier',
            'n_estimators': self.model.n_estimators if self.is_fitted else None,
            'scaler_fitted': hasattr(self.scaler, 'mean_') if hasattr(self, 'scaler') else False
        }
        
        if self.is_fitted:
            info['feature_names'] = ['price_range', 'volatility', 'bottom_distance', 'bottom_similarity', 
                                   'resistance_strength', 'volume_increase', 'trend_reversal', 
                                   'w_shape_score', 'breakout_strength', 'early_momentum', 'late_momentum']
            info['n_features'] = len(info['feature_names'])
        
        return info
    
    def plot_pattern(self, ohlc_data, title="Price Pattern"):
        """Plot the OHLC pattern"""
        if not ohlc_data:
            return
            
        ohlc_array = np.array(ohlc_data)
        closes = ohlc_array[:, 3]
        
        plt.figure(figsize=(12, 6))
        plt.plot(closes, linewidth=2)
        plt.title(title)
        plt.xlabel('Time Period')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.show()

# Example usage and testing
if __name__ == "__main__":
    print("=== Training New Model ===")
    # Initialize the classifier
    classifier = DoubleBottomClassifier(threshold=0.7)
    
    # Train the model
    model = classifier.train()
    
    # Save the trained model
    model_path = classifier.save_model("models/double_bottom_model.pkl")
    
    # Test with a sample pattern (should be detected as double bottom)
    test_pattern = [
        [100, 102, 98, 99, 1500], [99, 101, 95, 96, 1600],
        [96, 98, 92, 93, 1700], [93, 95, 90, 91, 1800],
        [91, 94, 89, 92, 1600], [92, 96, 93, 95, 1500],
        [95, 98, 94, 97, 1400], [97, 96, 94, 95, 1300],
        [95, 96, 92, 93, 1400], [93, 95, 90, 91, 1500],
        [91, 93, 89, 90, 1600], [90, 92, 88, 91, 1700],
        [91, 94, 90, 93, 1600], [93, 97, 94, 96, 1500],
        [96, 100, 97, 99, 1400], [99, 103, 100, 102, 1300]
    ]
    
    # Make prediction with current model
    prediction, probability = classifier.predict(test_pattern)
    print(f"\nTest Pattern Prediction (Current Model):")
    print(f"Double Bottom Detected: {'Yes' if prediction == 1 else 'No'}")
    print(f"Confidence: {probability:.3f}")
    
    # Display model info
    print(f"\nModel Information:")
    model_info = classifier.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    print(f"\n" + "="*50)
    print("=== Loading Saved Model ===")
    
    # Load the saved model
    if model_path:
        loaded_classifier = DoubleBottomClassifier.load_model(model_path)
        
        if loaded_classifier:
            # Test the loaded model with the same pattern
            prediction_loaded, probability_loaded = loaded_classifier.predict(test_pattern)
            print(f"\nTest Pattern Prediction (Loaded Model):")
            print(f"Double Bottom Detected: {'Yes' if prediction_loaded == 1 else 'No'}")
            print(f"Confidence: {probability_loaded:.3f}")
            
            # Verify predictions match
            if prediction == prediction_loaded and abs(probability - probability_loaded) < 1e-6:
                print(f"✅ Model saved and loaded successfully! Predictions match.")
            else:
                print(f"⚠️  Warning: Predictions don't match between original and loaded model.")
    
    # Plot the test pattern
    classifier.plot_pattern(test_pattern, "Test Pattern - Double Bottom Check")
    
    print(f"\n" + "="*50)
    print("=== Usage Examples ===")
    print(f"""
# To save a trained model:
classifier.save_model('my_model.pkl')

# To load a saved model:
loaded_classifier = DoubleBottomClassifier.load_model('my_model.pkl')

# To use the loaded model:
prediction, confidence = loaded_classifier.predict(your_ohlc_data)

# To get model information:
info = loaded_classifier.get_model_info()
""")
    
    print(f"Model training and saving completed successfully!")
    print(f"Use classifier.predict(ohlc_data) to classify new patterns")
    print(f"Returns (prediction, probability) where prediction is 1 for double bottom, 0 otherwise")