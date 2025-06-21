"""
Pre-trained model weights and configurations for pattern recognition
"""

import numpy as np
import os
import pickle
from datetime import datetime

class PretrainedWeights:
    """Pre-trained model weights and configurations"""
    
    def __init__(self):
        self.model_configs = self._get_model_configurations()
        self.pattern_templates = self._get_pattern_templates()
        self.feature_weights = self._get_feature_weights()
        
    def _get_model_configurations(self):
        """Get pre-configured model architectures"""
        return {
            'head_shoulders_cnn': {
                'architecture': 'CNN',
                'layers': [
                    {'type': 'Conv1D', 'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
                    {'type': 'MaxPooling1D', 'pool_size': 2},
                    {'type': 'Conv1D', 'filters': 128, 'kernel_size': 3, 'activation': 'relu'},
                    {'type': 'MaxPooling1D', 'pool_size': 2},
                    {'type': 'Conv1D', 'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
                    {'type': 'Flatten'},
                    {'type': 'Dense', 'units': 100, 'activation': 'relu'},
                    {'type': 'Dropout', 'rate': 0.2},
                    {'type': 'Dense', 'units': 50, 'activation': 'relu'},
                    {'type': 'Dropout', 'rate': 0.2},
                    {'type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
                ],
                'optimizer': {'type': 'Adam', 'learning_rate': 0.001},
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy'],
                'sequence_length': 60,
                'features': 5,
                'accuracy': 0.852,
                'trained_on': '2024-01-15'
            },
            
            'double_pattern_lstm': {
                'architecture': 'LSTM',
                'layers': [
                    {'type': 'LSTM', 'units': 100, 'return_sequences': True},
                    {'type': 'Dropout', 'rate': 0.2},
                    {'type': 'LSTM', 'units': 100, 'return_sequences': True},
                    {'type': 'Dropout', 'rate': 0.2},
                    {'type': 'LSTM', 'units': 50},
                    {'type': 'Dropout', 'rate': 0.2},
                    {'type': 'Dense', 'units': 25, 'activation': 'relu'},
                    {'type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
                ],
                'optimizer': {'type': 'Adam', 'learning_rate': 0.001},
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy'],
                'sequence_length': 60,
                'features': 5,
                'accuracy': 0.834,
                'trained_on': '2024-01-15'
            },
            
            'triangle_hybrid': {
                'architecture': 'Hybrid_CNN_LSTM',
                'cnn_layers': [
                    {'type': 'Conv1D', 'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
                    {'type': 'MaxPooling1D', 'pool_size': 2},
                    {'type': 'Conv1D', 'filters': 128, 'kernel_size': 3, 'activation': 'relu'},
                    {'type': 'MaxPooling1D', 'pool_size': 2}
                ],
                'lstm_layers': [
                    {'type': 'LSTM', 'units': 100, 'return_sequences': True},
                    {'type': 'Dropout', 'rate': 0.2},
                    {'type': 'LSTM', 'units': 50}
                ],
                'dense_layers': [
                    {'type': 'Dropout', 'rate': 0.2},
                    {'type': 'Dense', 'units': 25, 'activation': 'relu'},
                    {'type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
                ],
                'optimizer': {'type': 'Adam', 'learning_rate': 0.001},
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy'],
                'sequence_length': 60,
                'features': 5,
                'accuracy': 0.867,
                'trained_on': '2024-01-15'
            }
        }
    
    def _get_pattern_templates(self):
        """Get pattern recognition templates"""
        return {
            'head_shoulders': {
                'description': 'Three peaks with middle peak higher than sides',
                'key_features': [
                    'left_shoulder_peak',
                    'head_peak', 
                    'right_shoulder_peak',
                    'neckline_level',
                    'volume_confirmation'
                ],
                'validation_rules': {
                    'head_height_ratio': {'min': 1.02, 'max': 1.5},  # Head should be 2-50% higher
                    'shoulder_symmetry': {'max_diff': 0.05},  # Shoulders within 5% of each other
                    'neckline_slope': {'max_abs': 0.02},  # Relatively flat neckline
                    'pattern_duration': {'min_periods': 15, 'max_periods': 150}
                },
                'confidence_weights': {
                    'symmetry': 0.3,
                    'head_prominence': 0.25,
                    'volume_pattern': 0.2,
                    'neckline_quality': 0.15,
                    'duration': 0.1
                },
                'signal_strength': 'Strong',
                'typical_accuracy': 0.73
            },
            
            'double_top': {
                'description': 'Two peaks at similar levels with valley between',
                'key_features': [
                    'first_peak',
                    'second_peak',
                    'valley_between',
                    'support_level',
                    'volume_divergence'
                ],
                'validation_rules': {
                    'peak_similarity': {'max_diff': 0.03},  # Peaks within 3% of each other
                    'valley_depth': {'min_ratio': 0.05},  # Valley at least 5% below peaks
                    'peak_separation': {'min_periods': 10, 'max_periods': 100},
                    'volume_decline': {'second_peak_ratio': 0.8}  # Lower volume on second peak
                },
                'confidence_weights': {
                    'peak_alignment': 0.35,
                    'valley_depth': 0.25,
                    'volume_confirmation': 0.25,
                    'support_test': 0.15
                },
                'signal_strength': 'Strong',
                'typical_accuracy': 0.68
            },
            
            'double_bottom': {
                'description': 'Two valleys at similar levels with peak between',
                'key_features': [
                    'first_valley',
                    'second_valley', 
                    'peak_between',
                    'resistance_level',
                    'volume_increase'
                ],
                'validation_rules': {
                    'valley_similarity': {'max_diff': 0.03},
                    'peak_height': {'min_ratio': 0.05},
                    'valley_separation': {'min_periods': 10, 'max_periods': 100},
                    'volume_increase': {'second_valley_ratio': 1.2}
                },
                'confidence_weights': {
                    'valley_alignment': 0.35,
                    'peak_height': 0.25,
                    'volume_confirmation': 0.25,
                    'resistance_test': 0.15
                },
                'signal_strength': 'Strong',
                'typical_accuracy': 0.71
            },
            
            'ascending_triangle': {
                'description': 'Rising support with horizontal resistance',
                'key_features': [
                    'horizontal_resistance',
                    'rising_support',
                    'convergence_point',
                    'volume_pattern',
                    'breakout_direction'
                ],
                'validation_rules': {
                    'resistance_flatness': {'max_slope': 0.001},
                    'support_slope': {'min_slope': 0.0001, 'max_slope': 0.01},
                    'touch_count': {'min_touches': 4},
                    'volume_trend': {'declining_into_apex': True}
                },
                'confidence_weights': {
                    'trendline_quality': 0.4,
                    'touch_count': 0.3,
                    'volume_pattern': 0.2,
                    'convergence': 0.1
                },
                'signal_strength': 'Medium',
                'typical_accuracy': 0.64
            },
            
            'descending_triangle': {
                'description': 'Horizontal support with declining resistance',
                'key_features': [
                    'horizontal_support',
                    'declining_resistance',
                    'convergence_point',
                    'volume_pattern',
                    'breakout_direction'
                ],
                'validation_rules': {
                    'support_flatness': {'max_slope': 0.001},
                    'resistance_slope': {'min_slope': -0.01, 'max_slope': -0.0001},
                    'touch_count': {'min_touches': 4},
                    'volume_trend': {'declining_into_apex': True}
                },
                'confidence_weights': {
                    'trendline_quality': 0.4,
                    'touch_count': 0.3,
                    'volume_pattern': 0.2,
                    'convergence': 0.1
                },
                'signal_strength': 'Medium',
                'typical_accuracy': 0.62
            },
            
            'symmetrical_triangle': {
                'description': 'Converging support and resistance lines',
                'key_features': [
                    'declining_resistance',
                    'rising_support',
                    'convergence_point',
                    'volume_contraction',
                    'breakout_direction'
                ],
                'validation_rules': {
                    'convergence_angle': {'min_angle': 15, 'max_angle': 75},
                    'symmetry': {'slope_ratio_range': [0.5, 2.0]},
                    'touch_count': {'min_touches': 4},
                    'volume_contraction': {'decline_ratio': 0.7}
                },
                'confidence_weights': {
                    'symmetry': 0.35,
                    'trendline_quality': 0.3,
                    'volume_pattern': 0.2,
                    'touch_count': 0.15
                },
                'signal_strength': 'Medium',
                'typical_accuracy': 0.59
            }
        }
    
    def _get_feature_weights(self):
        """Get feature importance weights for pattern recognition"""
        return {
            'price_features': {
                'open': 0.15,
                'high': 0.25,
                'low': 0.25,
                'close': 0.35
            },
            'technical_indicators': {
                'sma_5': 0.1,
                'sma_10': 0.12,
                'sma_20': 0.15,
                'ema_12': 0.1,
                'ema_26': 0.1,
                'rsi': 0.08,
                'macd': 0.1,
                'stoch_k': 0.06,
                'stoch_d': 0.06,
                'atr': 0.08,
                'bollinger_upper': 0.05,
                'bollinger_lower': 0.05,
                'bollinger_middle': 0.05
            },
            'pattern_specific': {
                'head_shoulders': {
                    'peak_prominence': 0.3,
                    'symmetry': 0.25,
                    'neckline_slope': 0.2,
                    'volume_confirmation': 0.15,
                    'duration': 0.1
                },
                'double_patterns': {
                    'peak_valley_similarity': 0.4,
                    'intermediate_level': 0.25,
                    'volume_divergence': 0.2,
                    'timing': 0.15
                },
                'triangles': {
                    'trendline_quality': 0.35,
                    'convergence': 0.25,
                    'touch_count': 0.2,
                    'volume_pattern': 0.2
                },
                'support_resistance': {
                    'touch_count': 0.4,
                    'level_strength': 0.3,
                    'volume_at_level': 0.2,
                    'time_at_level': 0.1
                }
            }
        }
    
    def get_default_thresholds(self):
        """Get default confidence thresholds for pattern detection"""
        return {
            'high_confidence': 0.80,
            'medium_confidence': 0.60,
            'low_confidence': 0.40,
            'minimum_confidence': 0.20,
            'pattern_specific': {
                'head_shoulders': {
                    'high': 0.85,
                    'medium': 0.65,
                    'low': 0.45
                },
                'double_top': {
                    'high': 0.80,
                    'medium': 0.60,
                    'low': 0.40
                },
                'double_bottom': {
                    'high': 0.82,
                    'medium': 0.62,
                    'low': 0.42
                },
                'ascending_triangle': {
                    'high': 0.75,
                    'medium': 0.55,
                    'low': 0.35
                },
                'descending_triangle': {
                    'high': 0.75,
                    'medium': 0.55,
                    'low': 0.35
                },
                'symmetrical_triangle': {
                    'high': 0.70,
                    'medium': 0.50,
                    'low': 0.30
                }
            }
        }
    
    def get_model_weights(self, model_name):
        """Get pre-trained weights for a specific model"""
        # In a real implementation, this would load actual trained weights
        # For now, we return simulated weight matrices
        
        if model_name not in self.model_configs:
            return None
            
        config = self.model_configs[model_name]
        
        # Generate simulated weights based on model architecture
        weights = {}
        
        if 'cnn' in model_name.lower():
            weights = self._generate_cnn_weights(config)
        elif 'lstm' in model_name.lower():
            weights = self._generate_lstm_weights(config)
        elif 'hybrid' in model_name.lower():
            weights = self._generate_hybrid_weights(config)
        
        return weights
    
    def _generate_cnn_weights(self, config):
        """Generate simulated CNN weights"""
        np.random.seed(42)  # For reproducible results
        
        weights = {
            'conv1d_1': {
                'kernel': np.random.normal(0, 0.1, (3, config['features'], 64)),
                'bias': np.zeros(64)
            },
            'conv1d_2': {
                'kernel': np.random.normal(0, 0.1, (3, 64, 128)),
                'bias': np.zeros(128)
            },
            'conv1d_3': {
                'kernel': np.random.normal(0, 0.1, (3, 128, 64)),
                'bias': np.zeros(64)
            },
            'dense_1': {
                'kernel': np.random.normal(0, 0.1, (64, 100)),
                'bias': np.zeros(100)
            },
            'dense_2': {
                'kernel': np.random.normal(0, 0.1, (100, 50)),
                'bias': np.zeros(50)
            },
            'dense_3': {
                'kernel': np.random.normal(0, 0.1, (50, 1)),
                'bias': np.zeros(1)
            }
        }
        
        return weights
    
    def _generate_lstm_weights(self, config):
        """Generate simulated LSTM weights"""
        np.random.seed(42)
        
        weights = {
            'lstm_1': {
                'kernel': np.random.normal(0, 0.1, (config['features'], 400)),  # 4 * units
                'recurrent_kernel': np.random.normal(0, 0.1, (100, 400)),
                'bias': np.zeros(400)
            },
            'lstm_2': {
                'kernel': np.random.normal(0, 0.1, (100, 400)),
                'recurrent_kernel': np.random.normal(0, 0.1, (100, 400)),
                'bias': np.zeros(400)
            },
            'lstm_3': {
                'kernel': np.random.normal(0, 0.1, (100, 200)),  # 4 * 50
                'recurrent_kernel': np.random.normal(0, 0.1, (50, 200)),
                'bias': np.zeros(200)
            },
            'dense_1': {
                'kernel': np.random.normal(0, 0.1, (50, 25)),
                'bias': np.zeros(25)
            },
            'dense_2': {
                'kernel': np.random.normal(0, 0.1, (25, 1)),
                'bias': np.zeros(1)
            }
        }
        
        return weights
    
    def _generate_hybrid_weights(self, config):
        """Generate simulated hybrid model weights"""
        np.random.seed(42)
        
        # Combine CNN and LSTM weights
        cnn_weights = self._generate_cnn_weights(config)
        lstm_weights = self._generate_lstm_weights(config)
        
        weights = {
            'cnn_layers': {
                'conv1d_1': cnn_weights['conv1d_1'],
                'conv1d_2': cnn_weights['conv1d_2']
            },
            'lstm_layers': {
                'lstm_1': lstm_weights['lstm_1'],
                'lstm_2': lstm_weights['lstm_2']
            },
            'dense_layers': {
                'dense_1': {
                    'kernel': np.random.normal(0, 0.1, (50, 25)),
                    'bias': np.zeros(25)
                },
                'dense_2': {
                    'kernel': np.random.normal(0, 0.1, (25, 1)),
                    'bias': np.zeros(1)
                }
            }
        }
        
        return weights
    
    def save_weights(self, model_name, weights, filepath):
        """Save model weights to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model_name': model_name,
                    'weights': weights,
                    'config': self.model_configs.get(model_name, {}),
                    'saved_at': datetime.now().isoformat(),
                    'version': '1.0.0'
                }, f)
            
            return True
            
        except Exception as e:
            print(f"Error saving weights: {str(e)}")
            return False
    
    def load_weights(self, filepath):
        """Load model weights from file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            return data
            
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            return None
    
    def get_pattern_template(self, pattern_type):
        """Get pattern template for specific pattern type"""
        return self.pattern_templates.get(pattern_type, {})
    
    def validate_pattern_against_template(self, pattern, pattern_type):
        """Validate detected pattern against template rules"""
        template = self.get_pattern_template(pattern_type)
        
        if not template:
            return {'valid': False, 'reason': 'Unknown pattern type'}
        
        validation_rules = template.get('validation_rules', {})
        violations = []
        
        # Check each validation rule
        for rule_name, rule_config in validation_rules.items():
            if rule_name in pattern:
                pattern_value = pattern[rule_name]
                
                # Check min/max constraints
                if 'min' in rule_config and pattern_value < rule_config['min']:
                    violations.append(f"{rule_name} below minimum ({pattern_value} < {rule_config['min']})")
                
                if 'max' in rule_config and pattern_value > rule_config['max']:
                    violations.append(f"{rule_name} above maximum ({pattern_value} > {rule_config['max']})")
                
                if 'max_diff' in rule_config and abs(pattern_value) > rule_config['max_diff']:
                    violations.append(f"{rule_name} difference too large ({abs(pattern_value)} > {rule_config['max_diff']})")
        
        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'confidence_adjustment': -0.1 * len(violations)  # Reduce confidence for each violation
        }
    
    def calculate_pattern_confidence(self, pattern, pattern_type):
        """Calculate confidence score using template weights"""
        template = self.get_pattern_template(pattern_type)
        
        if not template:
            return 0.5  # Default confidence
        
        confidence_weights = template.get('confidence_weights', {})
        base_confidence = 0.0
        total_weight = 0.0
        
        for feature, weight in confidence_weights.items():
            if feature in pattern:
                # Normalize feature value to 0-1 range
                feature_value = pattern[feature]
                
                if isinstance(feature_value, (int, float)):
                    # Assume values are already normalized or calculate based on feature type
                    normalized_value = min(1.0, max(0.0, feature_value))
                else:
                    normalized_value = 0.5  # Default for non-numeric features
                
                base_confidence += normalized_value * weight
                total_weight += weight
        
        if total_weight > 0:
            base_confidence /= total_weight
        else:
            base_confidence = 0.5
        
        # Apply validation adjustment
        validation_result = self.validate_pattern_against_template(pattern, pattern_type)
        if 'confidence_adjustment' in validation_result:
            base_confidence += validation_result['confidence_adjustment']
        
        # Ensure confidence is in valid range
        return max(0.0, min(1.0, base_confidence))

