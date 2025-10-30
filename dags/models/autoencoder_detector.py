"""
Autoencoder Anomaly Detector - FIXED VERSION
Neural network-based anomaly detection using reconstruction error
Fixed: Keras save/load compatibility issues
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available. Autoencoder will not work.")
    TF_AVAILABLE = False


class AutoencoderDetector:
    """Autoencoder wrapper for transaction anomaly detection"""
    
    def __init__(self,
                 encoding_dim: int = 10,
                 hidden_layers: list = [30, 20],
                 learning_rate: float = 0.001,
                 random_state: int = 42):
        """Initialize Autoencoder detector"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Autoencoder")
        
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = None
        self.threshold = None
        self.feature_names = None
        self.is_trained = False
        self.training_date = None
        self.training_samples = 0
        self.training_history = None
    
    def _build_model(self, input_dim: int):
        """Build autoencoder architecture"""
        input_layer = layers.Input(shape=(input_dim,), name='input')
        
        encoded = input_layer
        for i, units in enumerate(self.hidden_layers):
            encoded = layers.Dense(units, activation='relu', name=f'encoder_{i+1}')(encoded)
        
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='bottleneck')(encoded)
        
        decoded = encoded
        for i, units in enumerate(reversed(self.hidden_layers)):
            decoded = layers.Dense(units, activation='relu', name=f'decoder_{i+1}')(decoded)
        
        decoded = layers.Dense(input_dim, activation='linear', name='output')(decoded)
        
        autoencoder = Model(inputs=input_layer, outputs=decoded, name='autoencoder')
        
        # FIX: Use full string 'mean_squared_error' instead of 'mse'
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',  # FIXED: was 'mse'
            metrics=['mae']
        )
        
        return autoencoder
    
    def train(self, X: pd.DataFrame, validation_split: float = 0.2,
              epochs: int = 50, batch_size: int = 32,
              early_stopping_patience: int = 5, verbose: int = 1) -> Dict:
        """Train Autoencoder"""
        print(f"\n🧠 Training Autoencoder...")
        print(f"   Samples: {len(X)}, Features: {X.shape[1]}")
        
        self.feature_names = X.columns.tolist()
        self.model = self._build_model(X.shape[1])
        
        if verbose >= 2:
            self.model.summary()
        
        X_np = X.values
        callbacks = [EarlyStopping(monitor='val_loss', patience=early_stopping_patience, 
                                   restore_best_weights=True, verbose=1)]
        
        history = self.model.fit(X_np, X_np, epochs=epochs, batch_size=batch_size,
                                validation_split=validation_split, callbacks=callbacks, verbose=verbose)
        
        self.training_history = history.history
        reconstructed = self.model.predict(X_np, verbose=0)
        reconstruction_errors = np.mean(np.square(X_np - reconstructed), axis=1)
        self.threshold = np.percentile(reconstruction_errors, 95)
        
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        epochs_trained = len(history.history['loss'])
        anomaly_rate = (reconstruction_errors > self.threshold).mean()
        
        print(f"\n   ✅ Training completed!")
        print(f"   Epochs: {epochs_trained}, Threshold: {self.threshold:.6f}, Anomaly rate: {anomaly_rate:.4%}")
        
        self.is_trained = True
        self.training_date = datetime.now()
        self.training_samples = len(X)
        
        return {
            'model_type': 'autoencoder',
            'threshold': float(self.threshold),
            'train_samples': len(X),
            'epochs_trained': epochs_trained,
            'final_train_loss': float(final_train_loss),
            'final_val_loss': float(final_val_loss),
            'anomaly_rate': float(anomaly_rate),
            'n_features': X.shape[1],
            'training_date': self.training_date.isoformat(),
            'architecture': {
                'encoding_dim': self.encoding_dim,
                'hidden_layers': self.hidden_layers,
                'learning_rate': self.learning_rate
            }
        }
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_np = X.values
        reconstructed = self.model.predict(X_np, verbose=0)
        errors = np.mean(np.square(X_np - reconstructed), axis=1)
        predictions = (errors > self.threshold).astype(int)
        normalized_errors = np.clip((errors / self.threshold) * 50, 0, 100)
        
        return normalized_errors, predictions
    
    def save(self, model_path: str, metadata_path: Optional[str] = None):
        """Save model - FIXED VERSION"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # FIX: Save in Keras 3 format (not HDF5)
        # Use .keras extension instead of .h5
        if model_path.endswith('.h5'):
            model_path = model_path.replace('.h5', '.keras')
            print(f"⚠️  Converted path from .h5 to .keras format")
        
        self.model.save(model_path, save_format='keras')
        print(f"💾 Model saved: {model_path}")
        
        if metadata_path is None:
            metadata_path = model_path.replace('.keras', '_metadata.json').replace('.h5', '_metadata.json')
        
        metadata = {
            'model_type': 'autoencoder',
            'threshold': float(self.threshold),
            'feature_names': self.feature_names,
            'training_date': self.training_date.isoformat(),
            'training_samples': self.training_samples,
            'architecture': {
                'encoding_dim': self.encoding_dim,
                'hidden_layers': self.hidden_layers,
                'learning_rate': self.learning_rate
            },
            'version': '1.0'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Metadata saved: {metadata_path}")
    
    def load(self, model_path: str, metadata_path: Optional[str] = None):
        """Load model - FIXED VERSION"""
        
        # FIX: Handle both .h5 and .keras files
        if model_path.endswith('.h5'):
            print(f"⚠️  Loading .h5 file with custom_objects for compatibility")
            # Define custom objects for backward compatibility
            custom_objects = {
                'mse': keras.losses.MeanSquaredError(),
                'mean_squared_error': keras.losses.MeanSquaredError()
            }
            self.model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            
            # Recompile with proper loss
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
        else:
            # .keras format - should load fine
            self.model = keras.models.load_model(model_path)
        
        print(f"📖 Model loaded: {model_path}")
        
        if metadata_path is None:
            metadata_path = model_path.replace('.keras', '_metadata.json').replace('.h5', '_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.threshold = metadata['threshold']
            self.feature_names = metadata['feature_names']
            self.training_date = datetime.fromisoformat(metadata['training_date'])
            self.training_samples = metadata['training_samples']
            
            arch = metadata['architecture']
            self.encoding_dim = arch['encoding_dim']
            self.hidden_layers = arch['hidden_layers']
            self.learning_rate = arch['learning_rate']
            
            self.is_trained = True
            print(f"📋 Metadata loaded")
        else:
            self.is_trained = True


if __name__ == "__main__":
    print("Autoencoder Detector Module - Fixed Version")
