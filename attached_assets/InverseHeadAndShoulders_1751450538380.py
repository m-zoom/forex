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

class InverseHeadAndShouldersClassifier:
    def __init__(self, threshold=0.7):
        """
        Initialize the Inverse Head and Shoulders Pattern Classifier
        
        Args:
            threshold (float): Hard threshold for classification (default: 0.7)
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.threshold = threshold
        self.is_fitted = False
        
    def extract_features(self, ohlc_data):
        """
        Extract features that characterize inverse head and shoulders patterns
        
        Args:
            ohlc_data (list): List of [Open, High, Low, Close, Volume] data points
            
        Returns:
            dict: Dictionary of extracted features
        """
        if len(ohlc_data) < 15:
            return None
            
        ohlc_array = np.array(ohlc_data)
        opens = ohlc_array[:, 0]
        highs = ohlc_array[:, 1]
        lows = ohlc_array[:, 2]
        closes = ohlc_array[:, 3]
        volumes = ohlc_array[:, 4]
        
        features = {}
        
        # 1. Find local minima (potential shoulders and head)
        troughs, _ = find_peaks(-lows, distance=3)
        
        # 2. Find local maxima (potential neckline points)
        peaks, _ = find_peaks(highs, distance=2)
        
        # 3. Basic price statistics
        features['price_range'] = (np.max(highs) - np.min(lows)) / np.mean(closes)
        features['volatility'] = np.std(closes) / np.mean(closes)
        
        # 4. Inverse head and shoulders specific features
        if len(troughs) >= 3:
            # Get the three lowest points (shoulders and head)
            deepest_indices = troughs[np.argsort(lows[troughs])[:3]]
            sorted_indices = np.sort(deepest_indices)
            left_idx, head_idx, right_idx = sorted_indices
            
            # Ensure head is deeper than shoulders
            head_low = lows[head_idx]
            left_low = lows[left_idx]
            right_low = lows[right_idx]
            
            # Head should be the deepest point
            if head_low < left_low and head_low < right_low:
                # Shoulder symmetry
                shoulder_avg = (left_low + right_low) / 2
                features['shoulder_symmetry'] = 1 - abs(left_low - right_low) / shoulder_avg
                
                # Head prominence
                features['head_prominence'] = (min(left_low, right_low) - head_low) / shoulder_avg
                
                # Neckline identification
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
                avg_volume = np.mean(volumes)
                
                features['volume_head_ratio'] = vol_head / ((vol_left + vol_right) / 2)
                features['volume_right_ratio'] = vol_right / vol_left
            else:
                # Head isn't the deepest point
                features['shoulder_symmetry'] = 0
                features['head_prominence'] = 0
                features['neckline_strength'] = 0
                features['neckline_slope'] = 0
                features['volume_head_ratio'] = 1
                features['volume_right_ratio'] = 1
        else:
            # Not enough troughs
            features['shoulder_symmetry'] = 0
            features['head_prominence'] = 0
            features['neckline_strength'] = 0
            features['neckline_slope'] = 0
            features['volume_head_ratio'] = 1
            features['volume_right_ratio'] = 1
            
        # 5. Pattern shape features
        features['trend_reversal'] = self._calculate_trend_reversal(closes)
        features['breakout_strength'] = self._calculate_breakout_strength(closes, highs)
        
        # 6. Price momentum features
        quarter_point = len(closes) // 4
        three_quarter_point = 3 * len(closes) // 4
        
        features['early_momentum'] = (closes[quarter_point] - closes[0]) / closes[0]
        features['late_momentum'] = (closes[-1] - closes[three_quarter_point]) / closes[three_quarter_point]
        
        return features
    
    def _calculate_trend_reversal(self, closes):
        """Calculate trend reversal strength"""
        if len(closes) < 8:
            return 0
        
        quarter = len(closes) // 4
        early_trend = (closes[quarter] - closes[0]) / closes[0]
        late_trend = (closes[-1] - closes[-quarter]) / closes[-quarter]
        
        # Good pattern should show decline then recovery
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
    
    def generate_negative_samples(self, n_samples=50):
        """Generate negative samples (non-inverse head and shoulders patterns)"""
        negative_samples = []
        
        for i in range(n_samples):
            # Generate random patterns that are NOT inverse head and shoulders
            pattern_length = np.random.randint(20, 30)
            
            # Random walk pattern
            if i < n_samples // 4:
                prices = self._generate_random_walk(pattern_length)
            # Head and shoulders pattern (inverted)
            elif i < n_samples // 2:
                prices = self._generate_head_and_shoulders(pattern_length)
            # Double top pattern
            elif i < 3 * n_samples // 4:
                prices = self._generate_double_top(pattern_length)
            # Trending pattern
            else:
                prices = self._generate_trending_pattern(pattern_length)
                
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
    
    def _generate_head_and_shoulders(self, length):
        """Generate head and shoulders pattern (opposite of inverse)"""
        start_price = np.random.uniform(50, 200)
        peak_price = start_price * 1.2
        
        prices = []
        
        # Left shoulder formation
        left_shoulder_length = length // 4
        for i in range(left_shoulder_length):
            progress = i / left_shoulder_length
            price = start_price + (peak_price - start_price) * progress
            noise = np.random.normal(0, 0.01) * price
            prices.append(price + noise)
            
        # Head formation
        head_length = length // 4
        for i in range(head_length):
            progress = i / head_length
            price = peak_price - (peak_price - start_price) * 0.3 * progress
            noise = np.random.normal(0, 0.01) * price
            prices.append(price + noise)
            
        # Right shoulder formation
        right_shoulder_length = length // 4
        for i in range(right_shoulder_length):
            progress = i / right_shoulder_length
            price = start_price + (peak_price - start_price) * (1 - progress)
            noise = np.random.normal(0, 0.01) * price
            prices.append(price + noise)
            
        # Decline phase
        decline_length = length - left_shoulder_length - head_length - right_shoulder_length
        for i in range(decline_length):
            progress = i / decline_length
            price = peak_price - (peak_price - start_price * 0.8) * progress
            noise = np.random.normal(0, 0.01) * price
            prices.append(price + noise)
            
        return self._prices_to_ohlc(prices)
    
    def _generate_double_top(self, length):
        """Generate double top pattern"""
        start_price = np.random.uniform(50, 200)
        resistance_price = start_price * 1.15
        
        # Initial rise
        rise_length = length // 3
        prices = []
        for i in range(rise_length):
            progress = i / rise_length
            price = start_price + (resistance_price - start_price) * progress
            noise = np.random.normal(0, 0.01) * price
            prices.append(price + noise)
            
        # First decline
        decline1_length = length // 6
        for i in range(decline1_length):
            progress = i / decline1_length
            price = resistance_price - (resistance_price - start_price) * 0.5 * progress
            noise = np.random.normal(0, 0.01) * price
            prices.append(price + noise)
            
        # Second rise (failed breakout)
        rise2_length = length // 6
        for i in range(rise2_length):
            progress = i / rise2_length
            price = start_price + (resistance_price - start_price) * progress
            noise = np.random.normal(0, 0.01) * price
            prices.append(price + noise)
            
        # Final breakdown
        breakdown_length = length - rise_length - decline1_length - rise2_length
        for i in range(breakdown_length):
            progress = i / breakdown_length
            price = resistance_price - (resistance_price - start_price * 0.8) * progress
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
    
    def prepare_training_data(self, patterns):
        """Prepare training data from patterns and negative samples"""
        X = []
        y = []
        
        # Extract features from positive samples
        for pattern_name, pattern_data in patterns.items():
            features = self.extract_features(pattern_data)
            if features:
                X.append(list(features.values()))
                y.append(1)  # Positive class
        
        # Generate and extract features from negative samples
        negative_samples = self.generate_negative_samples(len(patterns))
        for neg_pattern in negative_samples:
            features = self.extract_features(neg_pattern)
            if features:
                X.append(list(features.values()))
                y.append(0)  # Negative class
        
        return np.array(X), np.array(y)
    
    def train(self, patterns):
        """Train the inverse head and shoulders classifier"""
        print("Preparing training data...")
        X, y = self.prepare_training_data(patterns)
        
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
        feature_names = [
            'price_range', 'volatility', 'shoulder_symmetry', 
            'head_prominence', 'neckline_strength', 'neckline_slope',
            'volume_head_ratio', 'volume_right_ratio', 'trend_reversal',
            'breakout_strength', 'early_momentum', 'late_momentum'
        ]
        
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
        Predict if a pattern is an inverse head and shoulders
        
        Args:
            ohlc_data (list): OHLC data points
            
        Returns:
            tuple: (prediction, probability) 
                    prediction: 1 if pattern detected, 0 otherwise
                    probability: confidence score
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first. Call train() method.")
        
        features = self.extract_features(ohlc_data)
        if not features:
            return 0, 0.0
        
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
            filepath = f"inverse_head_and_shoulders_classifier_{timestamp}.pkl"
        
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
            InverseHeadAndShouldersClassifier: Loaded classifier instance
        """
        try:
            with open(filepath, 'rb') as f:
                classifier = pickle.load(f)
            
            # Verify it's the right type
            if not isinstance(classifier, cls):
                raise ValueError("Loaded object is not an InverseHeadAndShouldersClassifier instance")
            
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
            info['feature_names'] = [
                'price_range', 'volatility', 'shoulder_symmetry', 
                'head_prominence', 'neckline_strength', 'neckline_slope',
                'volume_head_ratio', 'volume_right_ratio', 'trend_reversal',
                'breakout_strength', 'early_momentum', 'late_momentum'
            ]
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

# Inverse Head and Shoulders Patterns - 50 manually written patterns
inverse_head_and_shoulders_patterns = {
    "pattern_1": [
        [65, 67, 60, 61, 2000], [61, 62, 57, 58, 2200],
        [58, 59, 55, 56, 2500], [56, 60, 55, 59, 2100],
        [59, 63, 58, 62, 1900], [62, 66, 61, 65, 1700],
        [65, 69, 64, 68, 1500], [68, 70, 65, 66, 1800],
        [66, 67, 62, 63, 2000], [63, 67, 62, 66, 1800],
        [66, 70, 65, 69, 1600], [69, 73, 68, 72, 1400],
        [72, 76, 71, 75, 1200], [75, 79, 74, 78, 1000],
        [78, 82, 77, 81, 800], [81, 85, 80, 84, 600],
        [84, 88, 83, 87, 400], [87, 91, 86, 90, 200]
    ],
    
    "pattern_2": [
        [125, 127, 120, 121, 900], [121, 122, 117, 118, 1000],
        [118, 119, 114, 115, 1200], [115, 119, 114, 118, 1050],
        [118, 122, 117, 121, 950], [121, 125, 120, 124, 850],
        [124, 128, 123, 127, 750], [127, 129, 124, 125, 900],
        [125, 126, 121, 122, 1000], [122, 126, 121, 125, 950],
        [125, 129, 124, 128, 850], [128, 132, 127, 131, 750],
        [131, 135, 130, 134, 650], [134, 138, 133, 137, 550],
        [137, 141, 136, 140, 450], [140, 144, 139, 143, 350],
        [143, 147, 142, 146, 250], [146, 150, 145, 149, 150]
    ],
    
    "pattern_3": [
        [42, 44, 37, 38, 2800], [38, 39, 34, 35, 3100],
        [35, 36, 31, 32, 3500], [32, 36, 31, 35, 3000],
        [35, 39, 34, 38, 2700], [38, 42, 37, 41, 2400],
        [41, 45, 40, 44, 2100], [44, 46, 41, 42, 2500],
        [42, 43, 38, 39, 2800], [39, 43, 38, 42, 2500],
        [42, 46, 41, 45, 2200], [45, 49, 44, 48, 1900],
        [48, 52, 47, 51, 1600], [51, 55, 50, 54, 1300],
        [54, 58, 53, 57, 1000], [57, 61, 56, 60, 700],
        [60, 64, 59, 63, 400], [63, 67, 62, 66, 100]
    ],
    
    "pattern_4": [
        [185, 187, 180, 181, 600], [181, 182, 177, 178, 700],
        [178, 179, 174, 175, 850], [175, 179, 174, 178, 750],
        [178, 182, 177, 181, 650], [181, 185, 180, 184, 550],
        [184, 188, 183, 187, 450], [187, 189, 184, 185, 600],
        [185, 186, 181, 182, 700], [182, 186, 181, 185, 650],
        [185, 189, 184, 188, 550], [188, 192, 187, 191, 450],
        [191, 195, 190, 194, 350], [194, 198, 193, 197, 250],
        [197, 201, 196, 200, 150], [200, 204, 199, 203, 50]
    ],
    
    "pattern_5": [
        [35, 37, 30, 31, 3200], [31, 32, 27, 28, 3600],
        [28, 29, 24, 25, 4000], [25, 29, 24, 28, 3500],
        [28, 32, 27, 31, 3100], [31, 35, 30, 34, 2700],
        [34, 38, 33, 37, 2300], [37, 39, 34, 35, 2800],
        [35, 36, 31, 32, 3200], [32, 36, 31, 35, 2800],
        [35, 39, 34, 38, 2400], [38, 42, 37, 41, 2000],
        [41, 45, 40, 44, 1600], [44, 48, 43, 47, 1200],
        [47, 51, 46, 50, 800], [50, 54, 49, 53, 400]
    ],
    
    "pattern_6": [
        [148, 150, 143, 144, 800], [144, 145, 140, 141, 900],
        [141, 142, 137, 138, 1100], [138, 142, 137, 141, 950],
        [141, 145, 140, 144, 850], [144, 148, 143, 147, 750],
        [147, 151, 146, 150, 650], [150, 152, 147, 148, 800],
        [148, 149, 144, 145, 900], [145, 149, 144, 148, 850],
        [148, 152, 147, 151, 750], [151, 155, 150, 154, 650],
        [154, 158, 153, 157, 550], [157, 161, 156, 160, 450],
        [160, 164, 159, 163, 350], [163, 167, 162, 166, 250]
    ],
    
    "pattern_7": [
        [72, 74, 67, 68, 2000], [68, 69, 64, 65, 2300],
        [65, 66, 61, 62, 2700], [62, 66, 61, 65, 2200],
        [65, 69, 64, 68, 1900], [68, 72, 67, 71, 1600],
        [71, 75, 70, 74, 1300], [74, 76, 71, 72, 1700],
        [72, 73, 68, 69, 2000], [69, 73, 68, 72, 1700],
        [72, 76, 71, 75, 1400], [75, 79, 74, 78, 1100],
        [78, 82, 77, 81, 800], [81, 85, 80, 84, 500],
        [84, 88, 83, 87, 200]
    ],
    
    "pattern_8": [
        [98, 100, 93, 94, 1300], [94, 95, 90, 91, 1500],
        [91, 92, 87, 88, 1800], [88, 92, 87, 91, 1550],
        [91, 95, 90, 94, 1350], [94, 98, 93, 97, 1150],
        [97, 101, 96, 100, 950], [100, 102, 97, 98, 1200],
        [98, 99, 94, 95, 1400], [95, 99, 94, 98, 1200],
        [98, 102, 97, 101, 1000], [101, 105, 100, 104, 800],
        [104, 108, 103, 107, 600], [107, 111, 106, 110, 400],
        [110, 114, 109, 113, 200]
    ],
    
    "pattern_9": [
        [15, 17, 10, 11, 4500], [11, 12, 7, 8, 5000],
        [8, 9, 4, 5, 5800], [5, 9, 4, 8, 5200],
        [8, 12, 7, 11, 4700], [11, 15, 10, 14, 4200],
        [14, 18, 13, 17, 3700], [17, 19, 14, 15, 4300],
        [15, 16, 11, 12, 4800], [12, 16, 11, 15, 4300],
        [15, 19, 14, 18, 3800], [18, 22, 17, 21, 3300],
        [21, 25, 20, 24, 2800], [24, 28, 23, 27, 2300],
        [27, 31, 26, 30, 1800]
    ],
    
    "pattern_10": [
        [155, 157, 150, 151, 700], [151, 152, 147, 148, 800],
        [148, 149, 144, 145, 1000], [145, 149, 144, 148, 850],
        [148, 152, 147, 151, 750], [151, 155, 150, 154, 650],
        [154, 158, 153, 157, 550], [157, 159, 154, 155, 700],
        [155, 156, 151, 152, 800], [152, 156, 151, 155, 750],
        [155, 159, 154, 158, 650], [158, 162, 157, 161, 550],
        [161, 165, 160, 164, 450], [164, 168, 163, 167, 350],
        [167, 171, 166, 170, 250]
    ],
    
    "pattern_11": [
        [78, 80, 73, 74, 1800], [74, 75, 70, 71, 2100],
        [71, 72, 67, 68, 2500], [68, 72, 67, 71, 2000],
        [71, 75, 70, 74, 1700], [74, 78, 73, 77, 1400],
        [77, 81, 76, 80, 1100], [80, 82, 77, 78, 1500],
        [78, 79, 74, 75, 1800], [75, 79, 74, 78, 1500],
        [78, 82, 77, 81, 1200], [81, 85, 80, 84, 900],
        [84, 88, 83, 87, 600], [87, 91, 86, 90, 300]
    ],
    
    "pattern_12": [
        [42, 44, 37, 38, 3000], [38, 39, 34, 35, 3400],
        [35, 36, 31, 32, 3900], [32, 36, 31, 35, 3500],
        [35, 39, 34, 38, 3100], [38, 42, 37, 41, 2700],
        [41, 45, 40, 44, 2300], [44, 46, 41, 42, 2800],
        [42, 43, 38, 39, 3200], [39, 43, 38, 42, 2800],
        [42, 46, 41, 45, 2400], [45, 49, 44, 48, 2000],
        [48, 52, 47, 51, 1600], [51, 55, 50, 54, 1200]
    ],
    
    "pattern_13": [
        [118, 120, 113, 114, 950], [114, 115, 110, 111, 1100],
        [111, 112, 107, 108, 1350], [108, 112, 107, 111, 1200],
        [111, 115, 110, 114, 1050], [114, 118, 113, 117, 900],
        [117, 121, 116, 120, 750], [120, 122, 117, 118, 950],
        [118, 119, 114, 115, 1100], [115, 119, 114, 118, 950],
        [118, 122, 117, 121, 800], [121, 125, 120, 124, 650],
        [124, 128, 123, 127, 500], [127, 131, 126, 130, 350]
    ],
    
    "pattern_14": [
        [8, 10, 3, 4, 6000], [4, 5, 0, 1, 7000],
        [1, 2, -3, -2, 8500], [-2, 2, -3, 1, 7500],
        [1, 5, 0, 4, 6800], [4, 8, 3, 7, 6100],
        [7, 11, 6, 10, 5400], [10, 12, 7, 8, 6200],
        [8, 9, 4, 5, 6800], [5, 9, 4, 8, 6200],
        [8, 12, 7, 11, 5500], [11, 15, 10, 14, 4800],
        [14, 18, 13, 17, 4100], [17, 21, 16, 20, 3400]
    ],
    
    "pattern_15": [
        [158, 160, 153, 154, 650], [154, 155, 150, 151, 750],
        [151, 152, 147, 148, 950], [148, 152, 147, 151, 800],
        [151, 155, 150, 154, 700], [154, 158, 153, 157, 600],
        [157, 161, 156, 160, 500], [160, 162, 157, 158, 650],
        [158, 159, 154, 155, 750], [155, 159, 154, 158, 650],
        [158, 162, 157, 161, 550], [161, 165, 160, 164, 450],
        [164, 168, 163, 167, 350], [167, 171, 166, 170, 250]
    ],
    
    "pattern_16": [
        [82, 84, 77, 78, 1600], [78, 79, 74, 75, 1900],
        [75, 76, 71, 72, 2300], [72, 76, 71, 75, 1950],
        [75, 79, 74, 78, 1700], [78, 82, 77, 81, 1450],
        [81, 85, 80, 84, 1200], [84, 86, 81, 82, 1550],
        [82, 83, 78, 79, 1800], [79, 83, 78, 82, 1550],
        [82, 86, 81, 85, 1300], [85, 89, 84, 88, 1050],
        [88, 92, 87, 91, 800], [91, 95, 90, 94, 550]
    ],
    
    "pattern_17": [
        [105, 107, 100, 101, 1150], [101, 102, 97, 98, 1350],
        [98, 99, 94, 95, 1650], [95, 99, 94, 98, 1400],
        [98, 102, 97, 101, 1200], [101, 105, 100, 104, 1000],
        [104, 108, 103, 107, 800], [107, 109, 104, 105, 1100],
        [105, 106, 101, 102, 1300], [102, 106, 101, 105, 1100],
        [105, 109, 104, 108, 900], [108, 112, 107, 111, 700],
        [111, 115, 110, 114, 500], [114, 118, 113, 117, 300]
    ],
    
    "pattern_18": [
        [28, 30, 23, 24, 4200], [24, 25, 20, 21, 4800],
        [21, 22, 17, 18, 5600], [18, 22, 17, 21, 5000],
        [21, 25, 20, 24, 4400], [24, 28, 23, 27, 3800],
        [27, 31, 26, 30, 3200], [30, 32, 27, 28, 3900],
        [28, 29, 24, 25, 4400], [25, 29, 24, 28, 3900],
        [28, 32, 27, 31, 3300], [31, 35, 30, 34, 2700],
        [34, 38, 33, 37, 2100], [37, 41, 36, 40, 1500]
    ],
    
    "pattern_19": [
        [192, 194, 187, 188, 450], [188, 189, 184, 185, 550],
        [185, 186, 181, 182, 750], [182, 186, 181, 185, 600],
        [185, 189, 184, 188, 500], [188, 192, 187, 191, 400],
        [191, 195, 190, 194, 300], [194, 196, 191, 192, 450],
        [192, 193, 188, 189, 550], [189, 193, 188, 192, 450],
        [192, 196, 191, 195, 350], [195, 199, 194, 198, 250],
        [198, 202, 197, 201, 150], [201, 205, 200, 204, 50]
    ],
    
    "pattern_20": [
        [68, 70, 63, 64, 2200], [64, 65, 60, 61, 2600],
        [61, 62, 57, 58, 3100], [58, 62, 57, 61, 2650],
        [61, 65, 60, 64, 2300], [64, 68, 63, 67, 1950],
        [67, 71, 66, 70, 1600], [70, 72, 67, 68, 2000],
        [68, 69, 64, 65, 2300], [65, 69, 64, 68, 2000],
        [68, 72, 67, 71, 1700], [71, 75, 70, 74, 1400],
        [74, 78, 73, 77, 1100], [77, 81, 76, 80, 800]
    ],
    
    "pattern_21": [
        [35, 37, 30, 31, 3800], [31, 32, 27, 28, 4300],
        [28, 29, 24, 25, 5000], [25, 29, 24, 28, 4400],
        [28, 32, 27, 31, 3900], [31, 35, 30, 34, 3400],
        [34, 38, 33, 37, 2900], [37, 39, 34, 35, 3500],
        [35, 36, 31, 32, 4000], [32, 36, 31, 35, 3500],
        [35, 39, 34, 38, 3000], [38, 42, 37, 41, 2500],
        [41, 45, 40, 44, 2000], [44, 48, 43, 47, 1500]
    ],
    
    "pattern_22": [
        [122, 124, 117, 118, 950], [118, 119, 114, 115, 1150],
        [115, 116, 111, 112, 1450], [112, 116, 111, 115, 1250],
        [115, 119, 114, 118, 1050], [118, 122, 117, 121, 850],
        [121, 125, 120, 124, 650], [124, 126, 121, 122, 900],
        [122, 123, 118, 119, 1100], [119, 123, 118, 122, 900],
        [122, 126, 121, 125, 700], [125, 129, 124, 128, 500],
        [128, 132, 127, 131, 300], [131, 135, 130, 134, 100]
    ],
    
    "pattern_23": [
        [172, 174, 167, 168, 580], [168, 169, 164, 165, 680],
        [165, 166, 161, 162, 880], [162, 166, 161, 165, 750],
        [165, 169, 164, 168, 650], [168, 172, 167, 171, 550],
        [171, 175, 170, 174, 450], [174, 176, 171, 172, 600],
        [172, 173, 168, 169, 700], [169, 173, 168, 172, 600],
        [172, 176, 171, 175, 500], [175, 179, 174, 178, 400],
        [178, 182, 177, 181, 300], [181, 185, 180, 184, 200]
    ],
    
    "pattern_24": [
        [52, 54, 47, 48, 2800], [48, 49, 44, 45, 3200],
        [45, 46, 41, 42, 3700], [42, 46, 41, 45, 3300],
        [45, 49, 44, 48, 2900], [48, 52, 47, 51, 2500],
        [51, 55, 50, 54, 2100], [54, 56, 51, 52, 2600],
        [52, 53, 48, 49, 3000], [49, 53, 48, 52, 2600],
        [52, 56, 51, 55, 2200], [55, 59, 54, 58, 1800],
        [58, 62, 57, 61, 1400], [61, 65, 60, 64, 1000]
    ],
    
    "pattern_25": [
        [105, 107, 100, 101, 1200], [101, 102, 97, 98, 1400],
        [98, 99, 94, 95, 1700], [95, 99, 94, 98, 1500],
        [98, 102, 97, 101, 1300], [101, 105, 100, 104, 1100],
        [104, 108, 103, 107, 900], [107, 109, 104, 105, 1200],
        [105, 106, 101, 102, 1400], [102, 106, 101, 105, 1200],
        [105, 109, 104, 108, 1000], [108, 112, 107, 111, 800],
        [111, 115, 110, 114, 600], [114, 118, 113, 117, 400]
    ],
    
    "pattern_26": [
        [18, 20, 13, 14, 5000], [14, 15, 10, 11, 5800],
        [11, 12, 7, 8, 6800], [8, 12, 7, 11, 6100],
        [11, 15, 10, 14, 5400], [14, 18, 13, 17, 4700],
        [17, 21, 16, 20, 4000], [20, 22, 17, 18, 4800],
        [18, 19, 14, 15, 5400], [15, 19, 14, 18, 4800],
        [18, 22, 17, 21, 4100], [21, 25, 20, 24, 3400],
        [24, 28, 23, 27, 2700], [27, 31, 26, 30, 2000]
    ],
    
    "pattern_27": [
        [138, 140, 133, 134, 800], [134, 135, 130, 131, 950],
        [131, 132, 127, 128, 1200], [128, 132, 127, 131, 1050],
        [131, 135, 130, 134, 900], [134, 138, 133, 137, 750],
        [137, 141, 136, 140, 600], [140, 142, 137, 138, 800],
        [138, 139, 134, 135, 950], [135, 139, 134, 138, 800],
        [138, 142, 137, 141, 650], [141, 145, 140, 144, 500],
        [144, 148, 143, 147, 350], [147, 151, 146, 150, 200]
    ],
    
    "pattern_28": [
        [75, 77, 70, 71, 2000], [71, 72, 67, 68, 2400],
        [68, 69, 64, 65, 2900], [65, 69, 64, 68, 2500],
        [68, 72, 67, 71, 2100], [71, 75, 70, 74, 1700],
        [74, 78, 73, 77, 1300], [77, 79, 74, 75, 1800],
        [75, 76, 71, 72, 2200], [72, 76, 71, 75, 1800],
        [75, 79, 74, 78, 1400], [78, 82, 77, 81, 1000],
        [81, 85, 80, 84, 600], [84, 88, 83, 87, 200]
    ],
    
    "pattern_29": [
        [158, 160, 153, 154, 650], [154, 155, 150, 151, 750],
        [151, 152, 147, 148, 950], [148, 152, 147, 151, 850],
        [151, 155, 150, 154, 750], [154, 158, 153, 157, 650],
        [157, 161, 156, 160, 550], [160, 162, 157, 158, 700],
        [158, 159, 154, 155, 800], [155, 159, 154, 158, 700],
        [158, 162, 157, 161, 600], [161, 165, 160, 164, 500],
        [164, 168, 163, 167, 400], [167, 171, 166, 170, 300]
    ],
    
    "pattern_30": [
        [88, 90, 83, 84, 1550], [84, 85, 80, 81, 1800],
        [81, 82, 77, 78, 2200], [78, 82, 77, 81, 1900],
        [81, 85, 80, 84, 1650], [84, 88, 83, 87, 1400],
        [87, 91, 86, 90, 1150], [90, 92, 87, 88, 1450],
        [88, 89, 84, 85, 1700], [85, 89, 84, 88, 1450],
        [88, 92, 87, 91, 1200], [91, 95, 90, 94, 950],
        [94, 98, 93, 97, 700], [97, 101, 96, 100, 450]
    ],
    
    "pattern_31": [
        [42, 44, 37, 38, 3200], [38, 39, 34, 35, 3700],
        [35, 36, 31, 32, 4300], [32, 36, 31, 35, 3800],
        [35, 39, 34, 38, 3300], [38, 42, 37, 41, 2800],
        [41, 45, 40, 44, 2300], [44, 46, 41, 42, 2900],
        [42, 43, 38, 39, 3400], [39, 43, 38, 42, 2900],
        [42, 46, 41, 45, 2400], [45, 49, 44, 48, 1900],
        [48, 52, 47, 51, 1400], [51, 55, 50, 54, 900]
    ],
    
    "pattern_32": [
        [165, 167, 160, 161, 600], [161, 162, 157, 158, 700],
        [158, 159, 154, 155, 900], [155, 159, 154, 158, 800],
        [158, 162, 157, 161, 700], [161, 165, 160, 164, 600],
        [164, 168, 163, 167, 500], [167, 169, 164, 165, 650],
        [165, 166, 161, 162, 750], [162, 166, 161, 165, 650],
        [165, 169, 164, 168, 550], [168, 172, 167, 171, 450],
        [171, 175, 170, 174, 350], [174, 178, 173, 177, 250]
    ],
    
    "pattern_33": [
        [115, 117, 110, 111, 1050], [111, 112, 107, 108, 1250],
        [108, 109, 104, 105, 1550], [105, 109, 104, 108, 1350],
        [108, 112, 107, 111, 1150], [111, 115, 110, 114, 950],
        [114, 118, 113, 117, 750], [117, 119, 114, 115, 1000],
        [115, 116, 111, 112, 1200], [112, 116, 111, 115, 1000],
        [115, 119, 114, 118, 850], [118, 122, 117, 121, 650],
        [121, 125, 120, 124, 450], [124, 128, 123, 127, 250]
    ],
    
    "pattern_34": [
        [28, 30, 23, 24, 4000], [24, 25, 20, 21, 4600],
        [21, 22, 17, 18, 5400], [18, 22, 17, 21, 4900],
        [21, 25, 20, 24, 4300], [24, 28, 23, 27, 3700],
        [27, 31, 26, 30, 3100], [30, 32, 27, 28, 3800],
        [28, 29, 24, 25, 4300], [25, 29, 24, 28, 3800],
        [28, 32, 27, 31, 3200], [31, 35, 30, 34, 2600],
        [34, 38, 33, 37, 2000], [37, 41, 36, 40, 1400]
    ],
    
    "pattern_35": [
        [185, 187, 180, 181, 500], [181, 182, 177, 178, 600],
        [178, 179, 174, 175, 800], [175, 179, 174, 178, 700],
        [178, 182, 177, 181, 600], [181, 185, 180, 184, 500],
        [184, 188, 183, 187, 400], [187, 189, 184, 185, 550],
        [185, 186, 181, 182, 650], [182, 186, 181, 185, 550],
        [185, 189, 184, 188, 450], [188, 192, 187, 191, 350],
        [191, 195, 190, 194, 250], [194, 198, 193, 197, 150]
    ],
    
    "pattern_36": [
        [62, 64, 57, 58, 2500], [58, 59, 54, 55, 2900],
        [55, 56, 51, 52, 3400], [52, 56, 51, 55, 3000],
        [55, 59, 54, 58, 2600], [58, 62, 57, 61, 2200],
        [61, 65, 60, 64, 1800], [64, 66, 61, 62, 2300],
        [62, 63, 58, 59, 2700], [59, 63, 58, 62, 2300],
        [62, 66, 61, 65, 1900], [65, 69, 64, 68, 1500],
        [68, 72, 67, 71, 1100], [71, 75, 70, 74, 700]
    ],
    
    "pattern_37": [
        [148, 150, 143, 144, 750], [144, 145, 140, 141, 850],
        [141, 142, 137, 138, 1050], [138, 142, 137, 141, 950],
        [141, 145, 140, 144, 850], [144, 148, 143, 147, 750],
        [147, 151, 146, 150, 650], [150, 152, 147, 148, 800],
        [148, 149, 144, 145, 900], [145, 149, 144, 148, 800],
        [148, 152, 147, 151, 700], [151, 155, 150, 154, 600],
        [154, 158, 153, 157, 500], [157, 161, 156, 160, 400]
    ],
    
    "pattern_38": [
        [78, 80, 73, 74, 1900], [74, 75, 70, 71, 2200],
        [71, 72, 67, 68, 2600], [68, 72, 67, 71, 2300],
        [71, 75, 70, 74, 2000], [74, 78, 73, 77, 1700],
        [77, 81, 76, 80, 1400], [80, 82, 77, 78, 1700],
        [78, 79, 74, 75, 2000], [75, 79, 74, 78, 1700],
        [78, 82, 77, 81, 1400], [81, 85, 80, 84, 1100],
        [84, 88, 83, 87, 800], [87, 91, 86, 90, 500]
    ],
    
    "pattern_39": [
        [98, 100, 93, 94, 1350], [94, 95, 90, 91, 1550],
        [91, 92, 87, 88, 1850], [88, 92, 87, 91, 1650],
        [91, 95, 90, 94, 1450], [94, 98, 93, 97, 1250],
        [97, 101, 96, 100, 1050], [100, 102, 97, 98, 1300],
        [98, 99, 94, 95, 1500], [95, 99, 94, 98, 1300],
        [98, 102, 97, 101, 1100], [101, 105, 100, 104, 900],
        [104, 108, 103, 107, 700], [107, 111, 106, 110, 500]
    ],
    
    "pattern_40": [
        [38, 40, 33, 34, 3600], [34, 35, 30, 31, 4100],
        [31, 32, 27, 28, 4800], [28, 32, 27, 31, 4300],
        [31, 35, 30, 34, 3800], [34, 38, 33, 37, 3300],
        [37, 41, 36, 40, 2800], [40, 42, 37, 38, 3400],
        [38, 39, 34, 35, 3900], [35, 39, 34, 38, 3400],
        [38, 42, 37, 41, 2900], [41, 45, 40, 44, 2400],
        [44, 48, 43, 47, 1900], [47, 51, 46, 50, 1400]
    ],
    
    "pattern_41": [
        [125, 127, 120, 121, 950], [121, 122, 117, 118, 1100],
        [118, 119, 114, 115, 1350], [115, 119, 114, 118, 1200],
        [118, 122, 117, 121, 1050], [121, 125, 120, 124, 900],
        [124, 128, 123, 127, 750], [127, 129, 124, 125, 950],
        [125, 126, 121, 122, 1100], [122, 126, 121, 125, 950],
        [125, 129, 124, 128, 800], [128, 132, 127, 131, 650],
        [131, 135, 130, 134, 500], [134, 138, 133, 137, 350]
    ],
    
    "pattern_42": [
        [12, 14, 7, 8, 5800], [8, 9, 4, 5, 6600],
        [5, 6, 1, 2, 7700], [2, 6, 1, 5, 7000],
        [5, 9, 4, 8, 6300], [8, 12, 7, 11, 5600],
        [11, 15, 10, 14, 4900], [14, 16, 11, 12, 5700],
        [12, 13, 8, 9, 6300], [9, 13, 8, 12, 5700],
        [12, 16, 11, 15, 5000], [15, 19, 14, 18, 4300],
        [18, 22, 17, 21, 3600], [21, 25, 20, 24, 2900]
    ],
    
    "pattern_43": [
        [158, 160, 153, 154, 700], [154, 155, 150, 151, 800],
        [151, 152, 147, 148, 1000], [148, 152, 147, 151, 900],
        [151, 155, 150, 154, 800], [154, 158, 153, 157, 700],
        [157, 161, 156, 160, 600], [160, 162, 157, 158, 750],
        [158, 159, 154, 155, 850], [155, 159, 154, 158, 750],
        [158, 162, 157, 161, 650], [161, 165, 160, 164, 550],
        [164, 168, 163, 167, 450], [167, 171, 166, 170, 350]
    ],
    
    "pattern_44": [
        [85, 87, 80, 81, 1650], [81, 82, 77, 78, 1900],
        [78, 79, 74, 75, 2300], [75, 79, 74, 78, 2000],
        [78, 82, 77, 81, 1750], [81, 85, 80, 84, 1500],
        [84, 88, 83, 87, 1250], [87, 89, 84, 85, 1550],
        [85, 86, 81, 82, 1800], [82, 86, 81, 85, 1550],
        [85, 89, 84, 88, 1300], [88, 92, 87, 91, 1050],
        [91, 95, 90, 94, 800], [94, 98, 93, 97, 550]
    ],
    
    "pattern_45": [
        [22, 24, 17, 18, 5400], [18, 19, 14, 15, 6200],
        [15, 16, 11, 12, 7300], [12, 16, 11, 15, 6600],
        [15, 19, 14, 18, 5900], [18, 22, 17, 21, 5200],
        [21, 25, 20, 24, 4500], [24, 26, 21, 22, 5300],
        [22, 23, 18, 19, 5900], [19, 23, 18, 22, 5300],
        [22, 26, 21, 25, 4600], [25, 29, 24, 28, 3900],
        [28, 32, 27, 31, 3200], [31, 35, 30, 34, 2500]
    ],
    
    "pattern_46": [
        [115, 117, 110, 111, 1100], [111, 112, 107, 108, 1300],
        [108, 109, 104, 105, 1600], [105, 109, 104, 108, 1400],
        [108, 112, 107, 111, 1200], [111, 115, 110, 114, 1000],
        [114, 118, 113, 117, 800], [117, 119, 114, 115, 1050],
        [115, 116, 111, 112, 1250], [112, 116, 111, 115, 1050],
        [115, 119, 114, 118, 850], [118, 122, 117, 121, 650],
        [121, 125, 120, 124, 450], [124, 128, 123, 127, 250]
    ],
    
    "pattern_47": [
        [68, 70, 63, 64, 2300], [64, 65, 60, 61, 2700],
        [61, 62, 57, 58, 3200], [58, 62, 57, 61, 2800],
        [61, 65, 60, 64, 2400], [64, 68, 63, 67, 2000],
        [67, 71, 66, 70, 1600], [70, 72, 67, 68, 2100],
        [68, 69, 64, 65, 2500], [65, 69, 64, 68, 2100],
        [68, 72, 67, 71, 1700], [71, 75, 70, 74, 1300],
        [74, 78, 73, 77, 900], [77, 81, 76, 80, 500]
    ],
    
    "pattern_48": [
        [175, 177, 170, 171, 550], [171, 172, 167, 168, 650],
        [168, 169, 164, 165, 850], [165, 169, 164, 168, 750],
        [168, 172, 167, 171, 650], [171, 175, 170, 174, 550],
        [174, 178, 173, 177, 450], [177, 179, 174, 175, 600],
        [175, 176, 171, 172, 700], [172, 176, 171, 175, 600],
        [175, 179, 174, 178, 500], [178, 182, 177, 181, 400],
        [181, 185, 180, 184, 300], [184, 188, 183, 187, 200]
    ],
    
    "pattern_49": [
        [102, 104, 97, 98, 1300], [98, 99, 94, 95, 1500],
        [95, 96, 91, 92, 1800], [92, 96, 91, 95, 1600],
        [95, 99, 94, 98, 1400], [98, 102, 97, 101, 1200],
        [101, 105, 100, 104, 1000], [104, 106, 101, 102, 1250],
        [102, 103, 98, 99, 1450], [99, 103, 98, 102, 1250],
        [102, 106, 101, 105, 1050], [105, 109, 104, 108, 850],
        [108, 112, 107, 111, 650], [111, 115, 110, 114, 450]
    ],
    
    "pattern_50": [
        [48, 50, 43, 44, 3100], [44, 45, 40, 41, 3600],
        [41, 42, 37, 38, 4200], [38, 42, 37, 41, 3700],
        [41, 45, 40, 44, 3200], [44, 48, 43, 47, 2700],
        [47, 51, 46, 50, 2200], [50, 52, 47, 48, 2800],
        [48, 49, 44, 45, 3300], [45, 49, 44, 48, 2800],
        [48, 52, 47, 51, 2300], [51, 55, 50, 54, 1800],
        [54, 58, 53, 57, 1300], [57, 61, 56, 60, 800]
    ],
}

# Example usage and testing
if __name__ == "__main__":
    print("=== Training New Model ===")
    # Initialize the classifier
    classifier = InverseHeadAndShouldersClassifier(threshold=0.7)
    
    # Train the model using the inverse_head_and_shoulders_patterns
    model = classifier.train(inverse_head_and_shoulders_patterns)
    
    # Save the trained model
    model_path = classifier.save_model("models/inverse_head_and_shoulders_model.pkl")
    
    # Test with a sample pattern (should be detected as inverse head and shoulders)
    test_pattern = inverse_head_and_shoulders_patterns['pattern_1']
    
    # Make prediction with current model
    prediction, probability = classifier.predict(test_pattern)
    print(f"\nTest Pattern Prediction (Current Model):")
    print(f"Inverse Head and Shoulders Detected: {'Yes' if prediction == 1 else 'No'}")
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
        loaded_classifier = InverseHeadAndShouldersClassifier.load_model(model_path)
        
        if loaded_classifier:
            # Test the loaded model with the same pattern
            prediction_loaded, probability_loaded = loaded_classifier.predict(test_pattern)
            print(f"\nTest Pattern Prediction (Loaded Model):")
            print(f"Inverse Head and Shoulders Detected: {'Yes' if prediction_loaded == 1 else 'No'}")
            print(f"Confidence: {probability_loaded:.3f}")
            
            # Verify predictions match
            if prediction == prediction_loaded and abs(probability - probability_loaded) < 1e-6:
                print(f"✅ Model saved and loaded successfully! Predictions match.")
            else:
                print(f"⚠️  Warning: Predictions don't match between original and loaded model.")
    
    # Plot the test pattern
    classifier.plot_pattern(test_pattern, "Test Pattern - Inverse Head and Shoulders Check")
    
    print(f"\n" + "="*50)
    print("=== Usage Examples ===")
    print(f"""
# To save a trained model:
classifier.save_model('my_model.pkl')

# To load a saved model:
loaded_classifier = InverseHeadAndShouldersClassifier.load_model('my_model.pkl')

# To use the loaded model:
prediction, confidence = loaded_classifier.predict(your_ohlc_data)

# To get model information:
info = loaded_classifier.get_model_info()
""")
    
    print(f"Model training and saving completed successfully!")
    print(f"Use classifier.predict(ohlc_data) to classify new patterns")
    print(f"Returns (prediction, probability) where prediction is 1 for inverse head and shoulders, 0 otherwise")