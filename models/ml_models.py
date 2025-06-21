"""
Machine Learning models for pattern recognition
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class PatternRecognitionModels:
    def __init__(self, logger):
        self.logger = logger
        self.models = {}
        self.scalers = {}
        self.sequence_length = 60
        
    def create_cnn_model(self, input_shape):
        """Create CNN model for pattern recognition"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(100, activation='relu'),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_lstm_model(self, input_shape):
        """Create LSTM model for time series pattern recognition"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_hybrid_model(self, input_shape):
        """Create hybrid CNN-LSTM model"""
        inputs = Input(shape=input_shape)
        
        # CNN layers
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(pool1)
        pool2 = MaxPooling1D(pool_size=2)(conv2)
        
        # LSTM layers
        lstm1 = LSTM(100, return_sequences=True)(pool2)
        dropout1 = Dropout(0.2)(lstm1)
        lstm2 = LSTM(50)(dropout1)
        dropout2 = Dropout(0.2)(lstm2)
        
        # Dense layers
        dense1 = Dense(25, activation='relu')(dropout2)
        outputs = Dense(1, activation='sigmoid')(dense1)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_training_data(self, df, pattern_type):
        """Prepare training data for pattern recognition"""
        try:
            # Extract features
            features = ['open', 'high', 'low', 'close', 'volume']
            
            # Add technical indicators if available
            technical_indicators = ['sma_5', 'sma_10', 'sma_20', 'rsi', 'macd', 'stoch_k']
            for indicator in technical_indicators:
                if indicator in df.columns:
                    features.append(indicator)
            
            # Prepare feature data
            feature_data = df[features].fillna(method='ffill').fillna(method='bfill')
            
            # Normalize features
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # Create sequences
            X, y = self.create_sequences_for_pattern(scaled_features, df, pattern_type)
            
            # Store scaler for later use
            self.scalers[pattern_type] = scaler
            
            return X, y, scaler
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            return None, None, None
    
    def create_sequences_for_pattern(self, scaled_features, df, pattern_type):
        """Create sequences and labels for specific pattern type"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_features)):
            # Input sequence
            sequence = scaled_features[i-self.sequence_length:i]
            X.append(sequence)
            
            # Label (simplified - in real implementation, use detected patterns)
            # This is a placeholder - replace with actual pattern detection logic
            if pattern_type == 'head_shoulders':
                label = self.has_head_shoulders_pattern(df.iloc[i-self.sequence_length:i+1])
            elif pattern_type == 'double_top':
                label = self.has_double_top_pattern(df.iloc[i-self.sequence_length:i+1])
            elif pattern_type == 'triangle':
                label = self.has_triangle_pattern(df.iloc[i-self.sequence_length:i+1])
            else:
                label = 0
                
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def has_head_shoulders_pattern(self, data_segment):
        """Simplified pattern detection for training labels"""
        # This is a simplified version - replace with actual pattern detection
        try:
            peaks = data_segment.dropna(subset=['peak']) if 'peak' in data_segment.columns else []
            return 1 if len(peaks) >= 3 else 0
        except:
            return 0
    
    def has_double_top_pattern(self, data_segment):
        """Simplified double top detection for training labels"""
        try:
            peaks = data_segment.dropna(subset=['peak']) if 'peak' in data_segment.columns else []
            return 1 if len(peaks) >= 2 else 0
        except:
            return 0
    
    def has_triangle_pattern(self, data_segment):
        """Simplified triangle detection for training labels"""
        try:
            # Check for converging trendlines
            if len(data_segment) < 20:
                return 0
                
            highs = data_segment['high'].values
            lows = data_segment['low'].values
            
            # Simple trend analysis
            high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
            low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
            
            # Converging trends indicate triangle
            return 1 if (high_trend < 0 and low_trend > 0) or (abs(high_trend) < 0.0001 or abs(low_trend) < 0.0001) else 0
        except:
            return 0
    
    def train_pattern_model(self, df, pattern_type, model_type='hybrid'):
        """Train a model for specific pattern recognition"""
        try:
            self.logger.info(f"Training {model_type} model for {pattern_type} pattern...")
            
            # Prepare training data
            X, y, scaler = self.prepare_training_data(df, pattern_type)
            
            if X is None or len(X) == 0:
                raise ValueError("No training data available")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create model based on type
            if model_type == 'cnn':
                model = self.create_cnn_model((X.shape[1], X.shape[2]))
            elif model_type == 'lstm':
                model = self.create_lstm_model((X.shape[1], X.shape[2]))
            else:  # hybrid
                model = self.create_hybrid_model((X.shape[1], X.shape[2]))
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            model_checkpoint = ModelCheckpoint(
                f'models/saved_models/{pattern_type}_{model_type}_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, model_checkpoint],
                verbose=1
            )
            
            # Evaluate model
            train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
            test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
            
            # Store model
            self.models[pattern_type] = model
            
            self.logger.info(f"Model training completed:")
            self.logger.info(f"Train accuracy: {train_accuracy:.3f}")
            self.logger.info(f"Test accuracy: {test_accuracy:.3f}")
            
            return {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'history': history
            }
            
        except Exception as e:
            self.logger.error(f"Error training {pattern_type} model: {str(e)}")
            raise
    
    def predict_pattern(self, df, pattern_type):
        """Predict pattern occurrence using trained model"""
        try:
            if pattern_type not in self.models:
                self.logger.warning(f"No trained model available for {pattern_type}")
                return None
            
            model = self.models[pattern_type]
            scaler = self.scalers.get(pattern_type)
            
            if scaler is None:
                self.logger.error(f"No scaler available for {pattern_type}")
                return None
            
            # Prepare data
            features = ['open', 'high', 'low', 'close', 'volume']
            technical_indicators = ['sma_5', 'sma_10', 'sma_20', 'rsi', 'macd', 'stoch_k']
            
            for indicator in technical_indicators:
                if indicator in df.columns:
                    features.append(indicator)
            
            feature_data = df[features].fillna(method='ffill').fillna(method='bfill')
            
            if len(feature_data) < self.sequence_length:
                self.logger.warning(f"Not enough data for prediction. Need {self.sequence_length}, got {len(feature_data)}")
                return None
            
            # Scale features
            scaled_features = scaler.transform(feature_data)
            
            # Create sequence for prediction
            sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Make prediction
            prediction = model.predict(sequence)[0][0]
            
            return {
                'pattern_type': pattern_type,
                'probability': float(prediction),
                'confidence': 'High' if prediction > 0.8 else 'Medium' if prediction > 0.6 else 'Low',
                'timestamp': df.index[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting {pattern_type} pattern: {str(e)}")
            return None
    
    def save_models(self, directory='models/saved_models'):
        """Save trained models and scalers"""
        try:
            os.makedirs(directory, exist_ok=True)
            
            for pattern_type, model in self.models.items():
                # Save model
                model_path = os.path.join(directory, f"{pattern_type}_model.h5")
                model.save(model_path)
                
                # Save scaler
                if pattern_type in self.scalers:
                    scaler_path = os.path.join(directory, f"{pattern_type}_scaler.pkl")
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[pattern_type], f)
                
                self.logger.info(f"Saved {pattern_type} model and scaler")
                
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self, directory='models/saved_models'):
        """Load pre-trained models and scalers"""
        try:
            if not os.path.exists(directory):
                self.logger.warning(f"Models directory {directory} does not exist")
                return
            
            pattern_types = ['head_shoulders', 'double_top', 'triangle']
            
            for pattern_type in pattern_types:
                model_path = os.path.join(directory, f"{pattern_type}_model.h5")
                scaler_path = os.path.join(directory, f"{pattern_type}_scaler.pkl")
                
                # Load model
                if os.path.exists(model_path):
                    try:
                        model = tf.keras.models.load_model(model_path)
                        self.models[pattern_type] = model
                        self.logger.info(f"Loaded {pattern_type} model")
                    except Exception as e:
                        self.logger.warning(f"Failed to load {pattern_type} model: {str(e)}")
                
                # Load scaler
                if os.path.exists(scaler_path):
                    try:
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)
                        self.scalers[pattern_type] = scaler
                        self.logger.info(f"Loaded {pattern_type} scaler")
                    except Exception as e:
                        self.logger.warning(f"Failed to load {pattern_type} scaler: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
    
    def get_model_summary(self):
        """Get summary of loaded models"""
        summary = {}
        
        for pattern_type, model in self.models.items():
            summary[pattern_type] = {
                'model_type': type(model).__name__,
                'parameters': model.count_params() if hasattr(model, 'count_params') else 'Unknown',
                'input_shape': model.input_shape if hasattr(model, 'input_shape') else 'Unknown',
                'has_scaler': pattern_type in self.scalers
            }
        
        return summary
