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
        """
        Initialize the Triple Bottom Pattern Classifier
        
        Args:
            threshold (float): Hard threshold for classification (default: 0.7)
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.threshold = threshold
        self.is_fitted = False
        
    def extract_features(self, ohlc_data):
        """
        Extract features that characterize triple bottom patterns
        
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
        
        # 1. Find local minima (potential bottoms)
        low_peaks, _ = find_peaks(-lows, distance=3)
        
        # 2. Find local maxima (potential resistance)
        high_peaks, _ = find_peaks(highs, distance=2)
        
        # 3. Basic price statistics
        features['price_range'] = (np.max(highs) - np.min(lows)) / np.mean(closes)
        features['volatility'] = np.std(closes) / np.mean(closes)
        
        # 4. Triple bottom specific features
        if len(low_peaks) >= 3:
            # Get the three lowest points
            lowest_indices = low_peaks[np.argsort(lows[low_peaks])[:3]]
            lowest_indices = np.sort(lowest_indices)
            bottom1, bottom2, bottom3 = lowest_indices
            
            # Bottoms similarity
            bottom_lows = lows[[bottom1, bottom2, bottom3]]
            features['bottom_similarity'] = 1 - (np.max(bottom_lows) - np.min(bottom_lows)) / np.mean(bottom_lows)
            
            # Distance between bottoms
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
            avg_volume = np.mean(volumes)
            features['volume_decline'] = vol_bottom3 / ((vol_bottom1 + vol_bottom2) / 2)
        else:
            # Set default values if not enough bottoms
            features['bottom_similarity'] = 0
            features['bottom_distance1'] = 0
            features['bottom_distance2'] = 0
            features['resistance_strength'] = 0
            features['resistance_similarity'] = 0
            features['volume_decline'] = 1
            
        # 5. Pattern shape features
        features['trend_reversal'] = self._calculate_trend_reversal(closes)
        features['breakout_strength'] = self._calculate_breakout_strength(closes, highs)
        
        # 6. Price momentum features
        features['early_momentum'] = (closes[len(closes)//4] - closes[0]) / closes[0]
        features['late_momentum'] = (closes[-1] - closes[len(closes)*3//4]) / closes[len(closes)*3//4]
        
        return features
    
    def _calculate_trend_reversal(self, closes):
        """Calculate trend reversal strength"""
        if len(closes) < 8:
            return 0
        
        quarter = len(closes) // 4
        early_trend = (closes[quarter] - closes[0]) / closes[0]
        late_trend = (closes[-1] - closes[-quarter]) / closes[-quarter]
        
        # Good triple bottom should show decline then recovery
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
        """Generate negative samples (non-triple bottom patterns)"""
        negative_samples = []
        
        for i in range(n_samples):
            # Generate random patterns that are NOT triple bottoms
            pattern_length = np.random.randint(20, 30)
            
            # Random walk pattern
            if i < n_samples // 4:
                prices = self._generate_random_walk(pattern_length)
            # Trending pattern
            elif i < n_samples // 2:
                prices = self._generate_trending_pattern(pattern_length)
            # Single bottom pattern
            elif i < 3 * n_samples // 4:
                prices = self._generate_single_bottom_pattern(pattern_length)
            # Double bottom pattern
            else:
                prices = self._generate_double_bottom_pattern(pattern_length)
                
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
        """Generate single bottom pattern (not triple bottom)"""
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
    
    def _generate_double_bottom_pattern(self, length):
        """Generate double bottom pattern (not triple bottom)"""
        start_price = np.random.uniform(50, 200)
        bottom_price = start_price * 0.8
        resistance_price = start_price * 0.9
        
        # Phase 1: Initial decline
        decline1_length = length // 4
        prices = []
        for i in range(decline1_length):
            progress = i / decline1_length
            price = start_price - (start_price - bottom_price) * progress
            noise = np.random.normal(0, 0.01) * price
            prices.append(price + noise)
            
        # Phase 2: First recovery
        recovery1_length = length // 6
        for i in range(recovery1_length):
            progress = i / recovery1_length
            price = bottom_price + (resistance_price - bottom_price) * progress
            noise = np.random.normal(0, 0.01) * price
            prices.append(price + noise)
            
        # Phase 3: Second decline
        decline2_length = length // 4
        for i in range(decline2_length):
            progress = i / decline2_length
            price = resistance_price - (resistance_price - bottom_price) * progress
            noise = np.random.normal(0, 0.01) * price
            prices.append(price + noise)
            
        # Phase 4: Final breakout
        breakout_length = length - decline1_length - recovery1_length - decline2_length
        for i in range(breakout_length):
            progress = i / breakout_length
            price = bottom_price + (start_price - bottom_price) * progress * 1.2
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
    
    def prepare_training_data(self, triple_bottom_patterns):
        """Prepare training data from triple bottom patterns and negative samples"""
        X = []
        y = []
        
        # Extract features from positive samples (triple bottom patterns)
        for pattern_name, pattern_data in triple_bottom_patterns.items():
            features = self.extract_features(pattern_data)
            if features:
                X.append(list(features.values()))
                y.append(1)  # Positive class
        
        # Generate and extract features from negative samples
        negative_samples = self.generate_negative_samples(len(triple_bottom_patterns))
        for neg_pattern in negative_samples:
            features = self.extract_features(neg_pattern)
            if features:
                X.append(list(features.values()))
                y.append(0)  # Negative class
        
        return np.array(X), np.array(y)
    
    def train(self, triple_bottom_patterns):
        """Train the triple bottom classifier"""
        print("Preparing training data...")
        X, y = self.prepare_training_data(triple_bottom_patterns)
        
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
            'price_range', 'volatility', 'bottom_similarity', 
            'bottom_distance1', 'bottom_distance2', 'resistance_strength',
            'resistance_similarity', 'volume_decline', 'trend_reversal',
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
        Predict if a pattern is a triple bottom
        
        Args:
            ohlc_data (list): OHLC data points
            
        Returns:
            tuple: (prediction, probability) 
                    prediction: 1 if triple bottom pattern detected, 0 otherwise
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
            filepath = f"triple_bottom_classifier_{timestamp}.pkl"
        
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
            TripleBottomClassifier: Loaded classifier instance
        """
        try:
            with open(filepath, 'rb') as f:
                classifier = pickle.load(f)
            
            # Verify it's the right type
            if not isinstance(classifier, cls):
                raise ValueError("Loaded object is not a TripleBottomClassifier instance")
            
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
                'price_range', 'volatility', 'bottom_similarity', 
                'bottom_distance1', 'bottom_distance2', 'resistance_strength',
                'resistance_similarity', 'volume_decline', 'trend_reversal',
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

# Triple Bottom Stock Patterns - 50 Manually Written Patterns

triple_bottom_patterns = {
    "pattern_1": [
        [50, 52, 51, 49, 2000], [49, 50, 48, 47, 2200],
        [47, 48, 46, 45, 2500], [45, 47, 45, 46, 2300],
        [46, 49, 46, 48, 2100], [48, 51, 48, 50, 1900],
        [50, 52, 49, 49, 1800], [49, 50, 47, 46, 2400],
        [46, 47, 45, 45, 2600], [45, 48, 45, 47, 2200],
        [47, 50, 47, 49, 2000], [49, 52, 49, 51, 1700],
        [51, 53, 50, 50, 1600], [50, 51, 47, 46, 2300],
        [46, 47, 45, 45, 2700], [45, 49, 45, 48, 2100],
        [48, 52, 48, 51, 1900], [51, 55, 51, 54, 1500],
        [54, 57, 54, 56, 1400]
    ],
    
    "pattern_2": [
        [85, 86, 83, 82, 1500], [82, 83, 80, 79, 1700],
        [79, 80, 78, 78, 1900], [78, 81, 78, 80, 1600],
        [80, 83, 80, 82, 1400], [82, 85, 82, 84, 1200],
        [84, 86, 83, 83, 1300], [83, 84, 80, 79, 1800],
        [79, 80, 78, 78, 2000], [78, 82, 78, 81, 1500],
        [81, 84, 81, 83, 1300], [83, 86, 83, 85, 1100],
        [85, 87, 84, 84, 1200], [84, 85, 81, 80, 1700],
        [80, 81, 78, 78, 1900], [78, 83, 78, 82, 1400],
        [82, 86, 82, 85, 1200], [85, 88, 85, 87, 1000],
        [87, 90, 87, 89, 900]
    ],
    
    "pattern_3": [
        [32, 33, 30, 29, 3200], [29, 30, 27, 26, 3500],
        [26, 27, 25, 25, 3800], [25, 28, 25, 27, 3300],
        [27, 30, 27, 29, 3000], [29, 32, 29, 31, 2700],
        [31, 33, 30, 30, 2800], [30, 31, 27, 26, 3400],
        [26, 27, 25, 25, 3700], [25, 29, 25, 28, 3100],
        [28, 31, 28, 30, 2800], [30, 33, 30, 32, 2500],
        [32, 34, 31, 31, 2600], [31, 32, 28, 27, 3300],
        [27, 28, 25, 25, 3600], [25, 30, 25, 29, 2900],
        [29, 33, 29, 32, 2600], [32, 35, 32, 34, 2300],
        [34, 36, 34, 35, 2200]
    ],
    
    "pattern_4": [
        [125, 126, 123, 122, 850], [122, 123, 120, 119, 950],
        [119, 120, 117, 115, 1100], [115, 118, 115, 117, 900],
        [117, 120, 117, 119, 800], [119, 122, 119, 121, 700],
        [121, 123, 120, 120, 750], [120, 121, 117, 116, 1000],
        [116, 117, 115, 115, 1200], [115, 119, 115, 118, 850],
        [118, 121, 118, 120, 750], [120, 123, 120, 122, 650],
        [122, 124, 121, 121, 700], [121, 122, 118, 117, 950],
        [117, 118, 115, 115, 1150], [115, 120, 115, 119, 800],
        [119, 123, 119, 122, 700], [122, 126, 122, 125, 600],
        [125, 129, 125, 128, 550]
    ],
    
    "pattern_5": [
        [68, 69, 66, 65, 1800], [65, 66, 63, 62, 2000],
        [62, 63, 60, 60, 2300], [60, 63, 60, 62, 1900],
        [62, 65, 62, 64, 1700], [64, 67, 64, 66, 1500],
        [66, 68, 65, 65, 1600], [65, 66, 62, 61, 2100],
        [61, 62, 60, 60, 2400], [60, 64, 60, 63, 1800],
        [63, 66, 63, 65, 1600], [65, 68, 65, 67, 1400],
        [67, 69, 66, 66, 1500], [66, 67, 63, 62, 2000],
        [62, 63, 60, 60, 2300], [60, 65, 60, 64, 1700],
        [64, 68, 64, 67, 1500], [67, 71, 67, 70, 1300],
        [70, 73, 70, 72, 1200]
    ],
    
    "pattern_6": [
        [42, 43, 40, 39, 2700], [39, 40, 37, 36, 3000],
        [36, 37, 35, 35, 3300], [35, 38, 35, 37, 2800],
        [37, 40, 37, 39, 2500], [39, 42, 39, 41, 2200],
        [41, 43, 40, 40, 2400], [40, 41, 37, 36, 2900],
        [36, 37, 35, 35, 3200], [35, 39, 35, 38, 2600],
        [38, 41, 38, 40, 2300], [40, 43, 40, 42, 2000],
        [42, 44, 41, 41, 2100], [41, 42, 38, 37, 2800],
        [37, 38, 35, 35, 3100], [35, 40, 35, 39, 2400],
        [39, 43, 39, 42, 2100], [42, 46, 42, 45, 1800],
        [45, 48, 45, 47, 1700]
    ],
    
    "pattern_7": [
        [95, 96, 93, 92, 1200], [92, 93, 90, 89, 1400],
        [89, 90, 87, 88, 1600], [88, 91, 88, 90, 1300],
        [90, 93, 90, 92, 1100], [92, 95, 92, 94, 900],
        [94, 96, 93, 93, 1000], [93, 94, 90, 89, 1500],
        [89, 90, 87, 88, 1700], [88, 92, 88, 91, 1200],
        [91, 94, 91, 93, 1000], [93, 96, 93, 95, 800],
        [95, 97, 94, 94, 900], [94, 95, 91, 90, 1400],
        [90, 91, 87, 88, 1600], [88, 93, 88, 92, 1100],
        [92, 96, 92, 95, 900], [95, 99, 95, 98, 700],
        [98, 102, 98, 101, 600]
    ],
    
    "pattern_8": [
        [22, 23, 20, 19, 4200], [19, 20, 17, 16, 4500],
        [16, 17, 15, 15, 4800], [15, 18, 15, 17, 4300],
        [17, 20, 17, 19, 4000], [19, 22, 19, 21, 3700],
        [21, 23, 20, 20, 3800], [20, 21, 17, 16, 4400],
        [16, 17, 15, 15, 4700], [15, 19, 15, 18, 4100],
        [18, 21, 18, 20, 3800], [20, 23, 20, 22, 3500],
        [22, 24, 21, 21, 3600], [21, 22, 18, 17, 4300],
        [17, 18, 15, 15, 4600], [15, 20, 15, 19, 3900],
        [19, 23, 19, 22, 3600], [22, 26, 22, 25, 3300],
        [25, 29, 25, 28, 3000]
    ],
    
    "pattern_9": [
        [162, 163, 160, 159, 650], [159, 160, 157, 156, 750],
        [156, 157, 154, 152, 900], [152, 155, 152, 154, 800],
        [154, 157, 154, 156, 700], [156, 159, 156, 158, 600],
        [158, 160, 157, 157, 650], [157, 158, 154, 153, 850],
        [153, 154, 152, 152, 950], [152, 156, 152, 155, 750],
        [155, 158, 155, 157, 650], [157, 160, 157, 159, 550],
        [159, 161, 158, 158, 600], [158, 159, 155, 154, 800],
        [154, 155, 152, 152, 900], [152, 157, 152, 156, 700],
        [156, 160, 156, 159, 600], [159, 163, 159, 162, 500],
        [162, 166, 162, 165, 450]
    ],
    
    "pattern_10": [
        [78, 79, 76, 75, 1600], [75, 76, 73, 72, 1800],
        [72, 73, 70, 70, 2100], [70, 73, 70, 72, 1700],
        [72, 75, 72, 74, 1500], [74, 77, 74, 76, 1300],
        [76, 78, 75, 75, 1400], [75, 76, 72, 71, 1900],
        [71, 72, 70, 70, 2200], [70, 74, 70, 73, 1600],
        [73, 76, 73, 75, 1400], [75, 78, 75, 77, 1200],
        [77, 79, 76, 76, 1300], [76, 77, 73, 72, 1800],
        [72, 73, 70, 70, 2000], [70, 75, 70, 74, 1500],
        [74, 78, 74, 77, 1300], [77, 81, 77, 80, 1100],
        [80, 84, 80, 83, 1000]
    ],
    
    "pattern_11": [
        [58, 59, 56, 55, 2200], [55, 56, 53, 52, 2500],
        [52, 53, 50, 50, 2800], [50, 53, 50, 52, 2300],
        [52, 55, 52, 54, 2000], [54, 57, 54, 56, 1700],
        [56, 58, 55, 55, 1900], [55, 56, 52, 51, 2400],
        [51, 52, 50, 50, 2700], [50, 54, 50, 53, 2100],
        [53, 56, 53, 55, 1800], [55, 58, 55, 57, 1500],
        [57, 59, 56, 56, 1700], [56, 57, 53, 52, 2300],
        [52, 53, 50, 50, 2600], [50, 55, 50, 54, 1900],
        [54, 58, 54, 57, 1600], [57, 61, 57, 60, 1300],
        [60, 64, 60, 63, 1200]
    ],
    
    "pattern_12": [
        [112, 113, 110, 109, 1050], [109, 110, 107, 106, 1200],
        [106, 107, 104, 105, 1400], [105, 108, 105, 107, 1100],
        [107, 110, 107, 109, 950], [109, 112, 109, 111, 800],
        [111, 113, 110, 110, 900], [110, 111, 107, 106, 1300],
        [106, 107, 105, 105, 1500], [105, 109, 105, 108, 1000],
        [108, 111, 108, 110, 850], [110, 113, 110, 112, 700],
        [112, 114, 111, 111, 800], [111, 112, 108, 107, 1200],
        [107, 108, 105, 105, 1400], [105, 110, 105, 109, 950],
        [109, 113, 109, 112, 800], [112, 116, 112, 115, 650],
        [115, 119, 115, 118, 600]
    ],
    
    "pattern_13": [
        [35, 36, 33, 32, 3500], [32, 33, 30, 29, 3800],
        [29, 30, 27, 28, 4200], [28, 31, 28, 30, 3700],
        [30, 33, 30, 32, 3400], [32, 35, 32, 34, 3100],
        [34, 36, 33, 33, 3200], [33, 34, 30, 29, 3700],
        [29, 30, 28, 28, 4000], [28, 32, 28, 31, 3500],
        [31, 34, 31, 33, 3200], [33, 36, 33, 35, 2900],
        [35, 37, 34, 34, 3000], [34, 35, 31, 30, 3600],
        [30, 31, 28, 28, 3900], [28, 33, 28, 32, 3300],
        [32, 36, 32, 35, 3000], [35, 39, 35, 38, 2700],
        [38, 42, 38, 41, 2500]
    ],
    
    "pattern_14": [
        [152, 153, 150, 149, 750], [149, 150, 147, 146, 850],
        [146, 147, 144, 142, 1000], [142, 145, 142, 144, 850],
        [144, 147, 144, 146, 750], [146, 149, 146, 148, 650],
        [148, 150, 147, 147, 700], [147, 148, 144, 143, 950],
        [143, 144, 142, 142, 1100], [142, 146, 142, 145, 800],
        [145, 148, 145, 147, 700], [147, 150, 147, 149, 600],
        [149, 151, 148, 148, 650], [148, 149, 145, 144, 900],
        [144, 145, 142, 142, 1000], [142, 147, 142, 146, 750],
        [146, 150, 146, 149, 650], [149, 153, 149, 152, 550],
        [152, 156, 152, 155, 500]
    ],
    
    "pattern_15": [
        [72, 73, 70, 69, 2000], [69, 70, 67, 66, 2200],
        [66, 67, 64, 65, 2500], [65, 68, 65, 67, 2100],
        [67, 70, 67, 69, 1900], [69, 72, 69, 71, 1700],
        [71, 73, 70, 70, 1800], [70, 71, 67, 66, 2300],
        [66, 67, 65, 65, 2600], [65, 69, 65, 68, 2000],
        [68, 71, 68, 70, 1800], [70, 73, 70, 72, 1600],
        [72, 74, 71, 71, 1700], [71, 72, 68, 67, 2200],
        [67, 68, 65, 65, 2400], [65, 70, 65, 69, 1900],
        [69, 73, 69, 72, 1700], [72, 76, 72, 75, 1500],
        [75, 79, 75, 78, 1400]
    ],
    
    "pattern_16": [
        [48, 49, 46, 45, 2700], [45, 46, 43, 42, 3000],
        [42, 43, 40, 40, 3300], [40, 43, 40, 42, 2800],
        [42, 45, 42, 44, 2500], [44, 47, 44, 46, 2200],
        [46, 48, 45, 45, 2400], [45, 46, 42, 41, 2900],
        [41, 42, 40, 40, 3200], [40, 44, 40, 43, 2600],
        [43, 46, 43, 45, 2300], [45, 48, 45, 47, 2000],
        [47, 49, 46, 46, 2100], [46, 47, 43, 42, 2800],
        [42, 43, 40, 40, 3100], [40, 45, 40, 44, 2400],
        [44, 48, 44, 47, 2100], [47, 51, 47, 50, 1800],
        [50, 54, 50, 53, 1700]
    ],
    
    "pattern_17": [
        [105, 106, 103, 102, 1150], [102, 103, 100, 99, 1300],
        [99, 100, 97, 95, 1500], [95, 98, 95, 97, 1250],
        [97, 100, 97, 99, 1100], [99, 102, 99, 101, 950],
        [101, 103, 100, 100, 1000], [100, 101, 97, 96, 1400],
        [96, 97, 95, 95, 1600], [95, 99, 95, 98, 1200],
        [98, 101, 98, 100, 1050], [100, 103, 100, 102, 900],
        [102, 104, 101, 101, 950], [101, 102, 98, 97, 1350],
        [97, 98, 95, 95, 1550], [95, 100, 95, 99, 1100],
        [99, 103, 99, 102, 950], [102, 106, 102, 105, 800],
        [105, 109, 105, 108, 750]
    ],
    
    "pattern_18": [
        [38, 39, 36, 35, 3200], [35, 36, 33, 32, 3500],
        [32, 33, 30, 30, 3800], [30, 33, 30, 32, 3300],
        [32, 35, 32, 34, 3000], [34, 37, 34, 36, 2700],
        [36, 38, 35, 35, 2800], [35, 36, 32, 31, 3400],
        [31, 32, 30, 30, 3700], [30, 34, 30, 33, 3100],
        [33, 36, 33, 35, 2800], [35, 38, 35, 37, 2500],
        [37, 39, 36, 36, 2600], [36, 37, 33, 32, 3300],
        [32, 33, 30, 30, 3600], [30, 35, 30, 34, 2900],
        [34, 38, 34, 37, 2600], [37, 41, 37, 40, 2300],
        [40, 44, 40, 43, 2100]
    ],
    
    "pattern_19": [
        [135, 136, 133, 132, 900], [132, 133, 130, 129, 1000],
        [129, 130, 127, 125, 1200], [125, 128, 125, 127, 1000],
        [127, 130, 127, 129, 900], [129, 132, 129, 131, 800],
        [131, 133, 130, 130, 850], [130, 131, 127, 126, 1150],
        [126, 127, 125, 125, 1300], [125, 129, 125, 128, 950],
        [128, 131, 128, 130, 850], [130, 133, 130, 132, 750],
        [132, 134, 131, 131, 800], [131, 132, 128, 127, 1100],
        [127, 128, 125, 125, 1250], [125, 130, 125, 129, 900],
        [129, 133, 129, 132, 800], [132, 136, 132, 135, 700],
        [135, 139, 135, 138, 650]
    ],
    
    "pattern_20": [
        [65, 66, 63, 62, 2150], [62, 63, 60, 59, 2400],
        [59, 60, 57, 58, 2700], [58, 61, 58, 60, 2200],
        [60, 63, 60, 62, 2000], [62, 65, 62, 64, 1800],
        [64, 66, 63, 63, 1900], [63, 64, 60, 59, 2500],
        [59, 60, 58, 58, 2800], [58, 62, 58, 61, 2100],
        [61, 64, 61, 63, 1900], [63, 66, 63, 65, 1700],
        [65, 67, 64, 64, 1800], [64, 65, 61, 60, 2400],
        [60, 61, 58, 58, 2600], [58, 63, 58, 62, 2000],
        [62, 66, 62, 65, 1800], [65, 69, 65, 68, 1600],
        [68, 72, 68, 71, 1500]
    ],
    
    "pattern_21": [
        [88, 89, 86, 85, 1450], [85, 86, 83, 82, 1650],
        [82, 83, 80, 80, 1900], [80, 83, 80, 82, 1550],
        [82, 85, 82, 84, 1350], [84, 87, 84, 86, 1150],
        [86, 88, 85, 85, 1250], [85, 86, 82, 81, 1750],
        [81, 82, 80, 80, 2000], [80, 84, 80, 83, 1450],
        [83, 86, 83, 85, 1250], [85, 88, 85, 87, 1050],
        [87, 89, 86, 86, 1150], [86, 87, 83, 82, 1650],
        [82, 83, 80, 80, 1850], [80, 85, 80, 84, 1350],
        [84, 88, 84, 87, 1150], [87, 91, 87, 90, 950],
        [90, 94, 90, 93, 850]
    ],
    
    "pattern_22": [
        [18, 19, 16, 15, 5500], [15, 16, 13, 12, 6000],
        [12, 13, 10, 11, 6500], [11, 14, 11, 13, 5800],
        [13, 16, 13, 15, 5300], [15, 18, 15, 17, 4800],
        [17, 19, 16, 16, 5000], [16, 17, 13, 12, 5900],
        [12, 13, 11, 11, 6400], [11, 15, 11, 14, 5600],
        [14, 17, 14, 16, 5100], [16, 19, 16, 18, 4600],
        [18, 20, 17, 17, 4800], [17, 18, 14, 13, 5700],
        [13, 14, 11, 11, 6200], [11, 16, 11, 15, 5400],
        [15, 19, 15, 18, 4900], [18, 22, 18, 21, 4400],
        [21, 25, 21, 24, 4000]
    ],
    
    "pattern_23": [
        [185, 186, 183, 182, 500], [182, 183, 180, 179, 600],
        [179, 180, 177, 175, 750], [175, 178, 175, 177, 650],
        [177, 180, 177, 179, 550], [179, 182, 179, 181, 450],
        [181, 183, 180, 180, 500], [180, 181, 177, 176, 700],
        [176, 177, 175, 175, 800], [175, 179, 175, 178, 600],
        [178, 181, 178, 180, 550], [180, 183, 180, 182, 450],
        [182, 184, 181, 181, 500], [181, 182, 178, 177, 650],
        [177, 178, 175, 175, 750], [175, 180, 175, 179, 550],
        [179, 183, 179, 182, 500], [182, 186, 182, 185, 400],
        [185, 189, 185, 188, 350]
    ],
    
    "pattern_24": [
        [52, 53, 50, 49, 2550], [49, 50, 47, 46, 2800],
        [46, 47, 44, 44, 3100], [44, 47, 44, 46, 2650],
        [46, 49, 46, 48, 2400], [48, 51, 48, 50, 2100],
        [50, 52, 49, 49, 2250], [49, 50, 46, 45, 2750],
        [45, 46, 44, 44, 3000], [44, 48, 44, 47, 2500],
        [47, 50, 47, 49, 2200], [49, 52, 49, 51, 1900],
        [51, 53, 50, 50, 2050], [50, 51, 47, 46, 2700],
        [46, 47, 44, 44, 2950], [44, 49, 44, 48, 2300],
        [48, 52, 48, 51, 2000], [51, 55, 51, 54, 1700],
        [54, 58, 54, 57, 1600]
    ],
    
    "pattern_25": [
        [125, 126, 123, 122, 1000], [122, 123, 120, 119, 1150],
        [119, 120, 117, 118, 1350], [118, 121, 118, 120, 1100],
        [120, 123, 120, 122, 950], [122, 125, 122, 124, 800],
        [124, 126, 123, 123, 900], [123, 124, 120, 119, 1250],
        [119, 120, 118, 118, 1450], [118, 122, 118, 121, 1050],
        [121, 124, 121, 123, 900], [123, 126, 123, 125, 750],
        [125, 127, 124, 124, 850], [124, 125, 121, 120, 1200],
        [120, 121, 118, 118, 1400], [118, 123, 118, 122, 1000],
        [122, 126, 122, 125, 850], [125, 129, 125, 128, 700],
        [128, 132, 128, 131, 650]
    ],
    
    "pattern_26": [
        [42, 43, 40, 39, 3050], [39, 40, 37, 36, 3350],
        [36, 37, 34, 35, 3700], [35, 38, 35, 37, 3200],
        [37, 40, 37, 39, 2900], [39, 42, 39, 41, 2600],
        [41, 43, 40, 40, 2750], [40, 41, 37, 36, 3300],
        [36, 37, 35, 35, 3650], [35, 39, 35, 38, 3100],
        [38, 41, 38, 40, 2800], [40, 43, 40, 42, 2500],
        [42, 44, 41, 41, 2650], [41, 42, 38, 37, 3250],
        [37, 38, 35, 35, 3600], [35, 40, 35, 39, 2950],
        [39, 43, 39, 42, 2650], [42, 46, 42, 45, 2300],
        [45, 49, 45, 48, 2100]
    ],
    
    "pattern_27": [
        [175, 176, 173, 172, 580], [172, 173, 170, 169, 650],
        [169, 170, 167, 165, 800], [165, 168, 165, 167, 700],
        [167, 170, 167, 169, 600], [169, 172, 169, 171, 500],
        [171, 173, 170, 170, 550], [170, 171, 167, 166, 750],
        [166, 167, 165, 165, 850], [165, 169, 165, 168, 650],
        [168, 171, 168, 170, 600], [170, 173, 170, 172, 500],
        [172, 174, 171, 171, 550], [171, 172, 168, 167, 700],
        [167, 168, 165, 165, 800], [165, 170, 165, 169, 600],
        [169, 173, 169, 172, 550], [172, 176, 172, 175, 450],
        [175, 179, 175, 178, 400]
    ],
    
    "pattern_28": [
        [75, 76, 73, 72, 1900], [72, 73, 70, 69, 2100],
        [69, 70, 67, 68, 2400], [68, 71, 68, 70, 2000],
        [70, 73, 70, 72, 1800], [72, 75, 72, 74, 1600],
        [74, 76, 73, 73, 1700], [73, 74, 70, 69, 2200],
        [69, 70, 68, 68, 2500], [68, 72, 68, 71, 1900],
        [71, 74, 71, 73, 1700], [73, 76, 73, 75, 1500],
        [75, 77, 74, 74, 1600], [74, 75, 71, 70, 2100],
        [70, 71, 68, 68, 2400], [68, 73, 68, 72, 1800],
        [72, 76, 72, 75, 1600], [75, 79, 75, 78, 1400],
        [78, 82, 78, 81, 1300]
    ],
    
    "pattern_29": [
        [98, 99, 96, 95, 1300], [95, 96, 93, 92, 1500],
        [92, 93, 90, 90, 1750], [90, 93, 90, 92, 1400],
        [92, 95, 92, 94, 1200], [94, 97, 94, 96, 1000],
        [96, 98, 95, 95, 1100], [95, 96, 92, 91, 1600],
        [91, 92, 90, 90, 1850], [90, 94, 90, 93, 1350],
        [93, 96, 93, 95, 1150], [95, 98, 95, 97, 950],
        [97, 99, 96, 96, 1050], [96, 97, 93, 92, 1550],
        [92, 93, 90, 90, 1800], [90, 95, 90, 94, 1250],
        [94, 98, 94, 97, 1100], [97, 101, 97, 100, 900],
        [100, 104, 100, 103, 800]
    ],
    
    "pattern_30": [
        [28, 29, 26, 25, 4200], [25, 26, 23, 22, 4600],
        [22, 23, 20, 20, 5000], [20, 23, 20, 22, 4400],
        [22, 25, 22, 24, 4000], [24, 27, 24, 26, 3600],
        [26, 28, 25, 25, 3800], [25, 26, 22, 21, 4500],
        [21, 22, 20, 20, 4900], [20, 24, 20, 23, 4200],
        [23, 26, 23, 25, 3800], [25, 28, 25, 27, 3400],
        [27, 29, 26, 26, 3600], [26, 27, 23, 22, 4400],
        [22, 23, 20, 20, 4800], [20, 25, 20, 24, 4000],
        [24, 28, 24, 27, 3600], [27, 31, 27, 30, 3200],
        [30, 34, 30, 33, 2900]
    ],
    
    "pattern_31": [
        [148, 149, 146, 145, 800], [145, 146, 143, 142, 900],
        [142, 143, 140, 138, 1100], [138, 141, 138, 140, 950],
        [140, 143, 140, 142, 850], [142, 145, 142, 144, 750],
        [144, 146, 143, 143, 800], [143, 144, 140, 139, 1050],
        [139, 140, 138, 138, 1200], [138, 142, 138, 141, 900],
        [141, 144, 141, 143, 800], [143, 146, 143, 145, 700],
        [145, 147, 144, 144, 750], [144, 145, 141, 140, 1000],
        [140, 141, 138, 138, 1150], [138, 143, 138, 142, 850],
        [142, 146, 142, 145, 750], [145, 149, 145, 148, 650],
        [148, 152, 148, 151, 600]
    ],
    
    "pattern_32": [
        [58, 59, 56, 55, 2400], [55, 56, 53, 52, 2700],
        [52, 53, 50, 50, 3000], [50, 53, 50, 52, 2550],
        [52, 55, 52, 54, 2300], [54, 57, 54, 56, 2000],
        [56, 58, 55, 55, 2150], [55, 56, 52, 51, 2650],
        [51, 52, 50, 50, 2900], [50, 54, 50, 53, 2350],
        [53, 56, 53, 55, 2100], [55, 58, 55, 57, 1850],
        [57, 59, 56, 56, 2000], [56, 57, 53, 52, 2600],
        [52, 53, 50, 50, 2850], [50, 55, 50, 54, 2200],
        [54, 58, 54, 57, 1950], [57, 61, 57, 60, 1700],
        [60, 64, 60, 63, 1600]
    ],
    
    "pattern_33": [
        [92, 93, 90, 89, 1400], [89, 90, 87, 86, 1600],
        [86, 87, 84, 85, 1900], [85, 88, 85, 87, 1500],
        [87, 90, 87, 89, 1300], [89, 92, 89, 91, 1100],
        [91, 93, 90, 90, 1200], [90, 91, 87, 86, 1700],
        [86, 87, 85, 85, 2000], [85, 89, 85, 88, 1450],
        [88, 91, 88, 90, 1250], [90, 93, 90, 92, 1050],
        [92, 94, 91, 91, 1150], [91, 92, 88, 87, 1650],
        [87, 88, 85, 85, 1900], [85, 90, 85, 89, 1350],
        [89, 93, 89, 92, 1200], [92, 96, 92, 95, 1000],
        [95, 99, 95, 98, 900]
    ],
    
    "pattern_34": [
        [15, 16, 13, 12, 6200], [12, 13, 10, 9, 6800],
        [9, 10, 7, 8, 7400], [8, 11, 8, 10, 6600],
        [10, 13, 10, 12, 6000], [12, 15, 12, 14, 5400],
        [14, 16, 13, 13, 5700], [13, 14, 10, 9, 6500],
        [9, 10, 8, 8, 7200], [8, 12, 8, 11, 6400],
        [11, 14, 11, 13, 5800], [13, 16, 13, 15, 5200],
        [15, 17, 14, 14, 5500], [14, 15, 11, 10, 6300],
        [10, 11, 8, 8, 7000], [8, 13, 8, 12, 5900],
        [12, 16, 12, 15, 5300], [15, 19, 15, 18, 4700],
        [18, 22, 18, 21, 4200]
    ],
    
    "pattern_35": [
        [168, 169, 166, 165, 620], [165, 166, 163, 162, 700],
        [162, 163, 160, 158, 850], [158, 161, 158, 160, 750],
        [160, 163, 160, 162, 650], [162, 165, 162, 164, 550],
        [164, 166, 163, 163, 600], [163, 164, 160, 159, 800],
        [159, 160, 158, 158, 900], [158, 162, 158, 161, 700],
        [161, 164, 161, 163, 650], [163, 166, 163, 165, 550],
        [165, 167, 164, 164, 600], [164, 165, 161, 160, 750],
        [160, 161, 158, 158, 850], [158, 163, 158, 162, 650],
        [162, 166, 162, 165, 600], [165, 169, 165, 168, 500],
        [168, 172, 168, 171, 450]
    ],
    
    "pattern_36": [
        [82, 83, 80, 79, 1750], [79, 80, 77, 76, 1950],
        [76, 77, 74, 75, 2200], [75, 78, 75, 77, 1850],
        [77, 80, 77, 79, 1650], [79, 82, 79, 81, 1450],
        [81, 83, 80, 80, 1550], [80, 81, 77, 76, 2000],
        [76, 77, 75, 75, 2300], [75, 79, 75, 78, 1800],
        [78, 81, 78, 80, 1600], [80, 83, 80, 82, 1400],
        [82, 84, 81, 81, 1500], [81, 82, 78, 77, 1950],
        [77, 78, 75, 75, 2200], [75, 80, 75, 79, 1700],
        [79, 83, 79, 82, 1550], [82, 86, 82, 85, 1300],
        [85, 89, 85, 88, 1200]
    ],
    
    "pattern_37": [
        [115, 116, 113, 112, 1100], [112, 113, 110, 109, 1250],
        [109, 110, 107, 105, 1450], [105, 108, 105, 107, 1200],
        [107, 110, 107, 109, 1050], [109, 112, 109, 111, 900],
        [111, 113, 110, 110, 1000], [110, 111, 107, 106, 1350],
        [106, 107, 105, 105, 1550], [105, 109, 105, 108, 1150],
        [108, 111, 108, 110, 1000], [110, 113, 110, 112, 850],
        [112, 114, 111, 111, 950], [111, 112, 108, 107, 1300],
        [107, 108, 105, 105, 1500], [105, 110, 105, 109, 1100],
        [109, 113, 109, 112, 1000], [112, 116, 112, 115, 800],
        [115, 119, 115, 118, 750]
    ],
    
    "pattern_38": [
        [45, 46, 43, 42, 2900], [42, 43, 40, 39, 3200],
        [39, 40, 37, 38, 3600], [38, 41, 38, 40, 3100],
        [40, 43, 40, 42, 2800], [42, 45, 42, 44, 2500],
        [44, 46, 43, 43, 2650], [43, 44, 40, 39, 3150],
        [39, 40, 38, 38, 3500], [38, 42, 38, 41, 2950],
        [41, 44, 41, 43, 2650], [43, 46, 43, 45, 2350],
        [45, 47, 44, 44, 2500], [44, 45, 41, 40, 3050],
        [40, 41, 38, 38, 3400], [38, 43, 38, 42, 2750],
        [42, 46, 42, 45, 2450], [45, 49, 45, 48, 2150],
        [48, 52, 48, 51, 2000]
    ],
    
    "pattern_39": [
        [138, 139, 136, 135, 850], [135, 136, 133, 132, 950],
        [132, 133, 130, 128, 1150], [128, 131, 128, 130, 1000],
        [130, 133, 130, 132, 900], [132, 135, 132, 134, 800],
        [134, 136, 133, 133, 850], [133, 134, 130, 129, 1100],
        [129, 130, 128, 128, 1250], [128, 132, 128, 131, 950],
        [131, 134, 131, 133, 850], [133, 136, 133, 135, 750],
        [135, 137, 134, 134, 800], [134, 135, 131, 130, 1050],
        [130, 131, 128, 128, 1200], [128, 133, 128, 132, 900],
        [132, 136, 132, 135, 800], [135, 139, 135, 138, 700],
        [138, 142, 138, 141, 650]
    ],
    
    "pattern_40": [
        [68, 69, 66, 65, 2100], [65, 66, 63, 62, 2350],
        [62, 63, 60, 60, 2650], [60, 63, 60, 62, 2250],
        [62, 65, 62, 64, 2000], [64, 67, 64, 66, 1750],
        [66, 68, 65, 65, 1900], [65, 66, 62, 61, 2400],
        [61, 62, 60, 60, 2700], [60, 64, 60, 63, 2150],
        [63, 66, 63, 65, 1900], [65, 68, 65, 67, 1650],
        [67, 69, 66, 66, 1800], [66, 67, 63, 62, 2300],
        [62, 63, 60, 60, 2600], [60, 65, 60, 64, 2000],
        [64, 68, 64, 67, 1800], [67, 71, 67, 70, 1550],
        [70, 74, 70, 73, 1450]
    ],
    
    "pattern_41": [
        [105, 106, 103, 102, 1200], [102, 103, 100, 99, 1350],
        [99, 100, 97, 98, 1600], [98, 101, 98, 100, 1300],
        [100, 103, 100, 102, 1150], [102, 105, 102, 104, 1000],
        [104, 106, 103, 103, 1100], [103, 104, 100, 99, 1450],
        [99, 100, 98, 98, 1650], [98, 102, 98, 101, 1250],
        [101, 104, 101, 103, 1100], [103, 106, 103, 105, 950],
        [105, 107, 104, 104, 1050], [104, 105, 101, 100, 1400],
        [100, 101, 98, 98, 1600], [98, 103, 98, 102, 1200],
        [102, 106, 102, 105, 1100], [105, 109, 105, 108, 900],
        [108, 112, 108, 111, 850]
    ],
    
    "pattern_42": [
        [32, 33, 30, 29, 3800], [29, 30, 27, 26, 4100],
        [26, 27, 24, 25, 4500], [25, 28, 25, 27, 3900],
        [27, 30, 27, 29, 3600], [29, 32, 29, 31, 3300],
        [31, 33, 30, 30, 3450], [30, 31, 27, 26, 4000],
        [26, 27, 25, 25, 4400], [25, 29, 25, 28, 3700],
        [28, 31, 28, 30, 3400], [30, 33, 30, 32, 3100],
        [32, 34, 31, 31, 3250], [31, 32, 28, 27, 3950],
        [27, 28, 25, 25, 4300], [25, 30, 25, 29, 3550],
        [29, 33, 29, 32, 3200], [32, 36, 32, 35, 2900],
        [35, 39, 35, 38, 2700]
    ],
    
    "pattern_43": [
        [158, 159, 156, 155, 700], [155, 156, 153, 152, 800],
        [152, 153, 150, 148, 950], [148, 151, 148, 150, 850],
        [150, 153, 150, 152, 750], [152, 155, 152, 154, 650],
        [154, 156, 153, 153, 700], [153, 154, 150, 149, 900],
        [149, 150, 148, 148, 1000], [148, 152, 148, 151, 800],
        [151, 154, 151, 153, 750], [153, 156, 153, 155, 650],
        [155, 157, 154, 154, 700], [154, 155, 151, 150, 850],
        [150, 151, 148, 148, 950], [148, 153, 148, 152, 750],
        [152, 156, 152, 155, 700], [155, 159, 155, 158, 600],
        [158, 162, 158, 161, 550]
    ],
    
    "pattern_44": [
        [85, 86, 83, 82, 1600], [82, 83, 80, 79, 1800],
        [79, 80, 77, 78, 2100], [78, 81, 78, 80, 1750],
        [80, 83, 80, 82, 1550], [82, 85, 82, 84, 1350],
        [84, 86, 83, 83, 1450], [83, 84, 80, 79, 1900],
        [79, 80, 78, 78, 2200], [78, 82, 78, 81, 1650],
        [81, 84, 81, 83, 1450], [83, 86, 83, 85, 1250],
        [85, 87, 84, 84, 1350], [84, 85, 81, 80, 1850],
        [80, 81, 78, 78, 2100], [78, 83, 78, 82, 1600],
        [82, 86, 82, 85, 1450], [85, 89, 85, 88, 1200],
        [88, 92, 88, 91, 1100]
    ],
    
    "pattern_45": [
        [25, 26, 23, 22, 5000], [22, 23, 20, 19, 5500],
        [19, 20, 17, 18, 6000], [18, 21, 18, 20, 5300],
        [20, 23, 20, 22, 4800], [22, 25, 22, 24, 4300],
        [24, 26, 23, 23, 4500], [23, 24, 20, 19, 5200],
        [19, 20, 18, 18, 5800], [18, 22, 18, 21, 5000],
        [21, 24, 21, 23, 4500], [23, 26, 23, 25, 4000],
        [25, 27, 24, 24, 4200], [24, 25, 21, 20, 5100],
        [20, 21, 18, 18, 5700], [18, 23, 18, 22, 4800],
        [22, 26, 22, 25, 4300], [25, 29, 25, 28, 3800],
        [28, 32, 28, 31, 3500]
    ],
    
    "pattern_46": [
        [122, 123, 120, 119, 1050], [119, 120, 117, 116, 1200],
        [116, 117, 114, 112, 1400], [112, 115, 112, 114, 1150],
        [114, 117, 114, 116, 1000], [116, 119, 116, 118, 850],
        [118, 120, 117, 117, 950], [117, 118, 114, 113, 1300],
        [113, 114, 112, 112, 1500], [112, 116, 112, 115, 1100],
        [115, 118, 115, 117, 950], [117, 120, 117, 119, 800],
        [119, 121, 118, 118, 900], [118, 119, 115, 114, 1250],
        [114, 115, 112, 112, 1450], [112, 117, 112, 116, 1050],
        [116, 120, 116, 119, 900], [119, 123, 119, 122, 750],
        [122, 126, 122, 125, 700]
    ],
    
    "pattern_47": [
        [62, 63, 60, 59, 2350], [59, 60, 57, 56, 2600],
        [56, 57, 54, 55, 2950], [55, 58, 55, 57, 2450],
        [57, 60, 57, 59, 2200], [59, 62, 59, 61, 1950],
        [61, 63, 60, 60, 2100], [60, 61, 57, 56, 2650],
        [56, 57, 55, 55, 3000], [55, 59, 55, 58, 2350],
        [58, 61, 58, 60, 2100], [60, 63, 60, 62, 1850],
        [62, 64, 61, 61, 2000], [61, 62, 58, 57, 2550],
        [57, 58, 55, 55, 2900], [55, 60, 55, 59, 2250],
        [59, 63, 59, 62, 2000], [62, 66, 62, 65, 1750],
        [65, 69, 65, 68, 1650]
    ],
    
    "pattern_48": [
        [145, 146, 143, 142, 800], [142, 143, 140, 139, 900],
        [139, 140, 137, 135, 1100], [135, 138, 135, 137, 950],
        [137, 140, 137, 139, 850], [139, 142, 139, 141, 750],
        [141, 143, 140, 140, 800], [140, 141, 137, 136, 1050],
        [136, 137, 135, 135, 1200], [135, 139, 135, 138, 900],
        [138, 141, 138, 140, 800], [140, 143, 140, 142, 700],
        [142, 144, 141, 141, 750], [141, 142, 138, 137, 1000],
        [137, 138, 135, 135, 1150], [135, 140, 135, 139, 850],
        [139, 143, 139, 142, 750], [142, 146, 142, 145, 650],
        [145, 149, 145, 148, 600]
    ],
    
    "pattern_49": [
        [78, 79, 76, 75, 1800], [75, 76, 73, 72, 2000],
        [72, 73, 70, 70, 2300], [70, 73, 70, 72, 1900],
        [72, 75, 72, 74, 1700], [74, 77, 74, 76, 1500],
        [76, 78, 75, 75, 1650], [75, 76, 72, 71, 2150],
        [71, 72, 70, 70, 2400], [70, 74, 70, 73, 1800],
        [73, 76, 73, 75, 1600], [75, 78, 75, 77, 1400],
        [77, 79, 76, 76, 1550], [76, 77, 73, 72, 2050],
        [72, 73, 70, 70, 2350], [70, 75, 70, 74, 1750],
        [74, 78, 74, 77, 1550], [77, 81, 77, 80, 1300],
        [80, 84, 80, 83, 1200]
    ],
    
    "pattern_50": [
        [52, 53, 50, 49, 2600], [49, 50, 47, 46, 2900],
        [46, 47, 44, 45, 3300], [45, 48, 45, 47, 2750],
        [47, 50, 47, 49, 2450], [49, 52, 49, 51, 2150],
        [51, 53, 50, 50, 2300], [50, 51, 47, 46, 2800],
        [46, 47, 45, 45, 3200], [45, 49, 45, 48, 2550],
        [48, 51, 48, 50, 2250], [50, 53, 50, 52, 1950],
        [52, 54, 51, 51, 2100], [51, 52, 48, 47, 2700],
        [47, 48, 45, 45, 3100], [45, 50, 45, 49, 2400],
        [49, 53, 49, 52, 2150], [52, 56, 52, 55, 1850],
        [55, 59, 55, 58, 1750]
    ],
}
# Example usage and testing
if __name__ == "__main__":
    print("=== Training New Model ===")
    # Initialize the classifier
    classifier = TripleBottomClassifier(threshold=0.7)
    
    # Train the model using the triple_bottom_patterns
    model = classifier.train(triple_bottom_patterns)
    
    # Save the trained model
    model_path = classifier.save_model("models/triple_bottom_model.pkl")
    
    # Test with a sample pattern (should be detected as triple bottom)
    test_pattern = triple_bottom_patterns['pattern_1']
    
    # Make prediction with current model
    prediction, probability = classifier.predict(test_pattern)
    print(f"\nTest Pattern Prediction (Current Model):")
    print(f"Triple Bottom Detected: {'Yes' if prediction == 1 else 'No'}")
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
        loaded_classifier = TripleBottomClassifier.load_model(model_path)
        
        if loaded_classifier:
            # Test the loaded model with the same pattern
            prediction_loaded, probability_loaded = loaded_classifier.predict(test_pattern)
            print(f"\nTest Pattern Prediction (Loaded Model):")
            print(f"Triple Bottom Detected: {'Yes' if prediction_loaded == 1 else 'No'}")
            print(f"Confidence: {probability_loaded:.3f}")
            
            # Verify predictions match
            if prediction == prediction_loaded and abs(probability - probability_loaded) < 1e-6:
                print(f"✅ Model saved and loaded successfully! Predictions match.")
            else:
                print(f"⚠️  Warning: Predictions don't match between original and loaded model.")
    
    # Plot the test pattern
    classifier.plot_pattern(test_pattern, "Test Pattern - Triple Bottom Check")
    
    print(f"\n" + "="*50)
    print("=== Usage Examples ===")
    print(f"""
# To save a trained model:
classifier.save_model('my_model.pkl')

# To load a saved model:
loaded_classifier = TripleBottomClassifier.load_model('my_model.pkl')

# To use the loaded model:
prediction, confidence = loaded_classifier.predict(your_ohlc_data)

# To get model information:
info = loaded_classifier.get_model_info()
""")
    
    print(f"Model training and saving completed successfully!")
    print(f"Use classifier.predict(ohlc_data) to classify new patterns")
    print(f"Returns (prediction, probability) where prediction is 1 for triple bottom, 0 otherwise")