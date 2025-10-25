"""
Autoencoder Anomaly Detector
Neural network-based anomaly detection using reconstruction error
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
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available. Autoencoder will not work.")
    TF_AVAILABLE = False


class AutoencoderDetector:
    """
    Autoencoder wrapper for transaction anomaly detection
    
    Key Concepts:
    - Neural network trained to reconstruct normal transactions
    - Anomalies have high reconstruction error
    - Learns non-linear patterns in data
    """
    
    def __init__(self,
                 encoding_dim: int = 10,
                 hidden_layers: list = [30, 20],
                 learning_rate: float = 0.001,
                 random_state: int = 42):
        """
        Initialize Autoencoder detector
        
        Args:
            encoding_dim: Size of latent (bottleneck) layer
            hidden_layers: List of hidden layer sizes
            learning_rate: Learning rate for Adam optimizer
            random_state: Random seed for reproducibility
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Autoencoder")
        
        # Set random seeds
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
    
    def _build_model(self, input_dim: int) -> Model:
        """
        Build autoencoder architecture
        
        Architecture:
        Input → Hidden Layers → Encoding (Bottleneck) → Hidden Layers → Output
        
        Args:
            input_dim: Number of input features
        
        Returns:
            Keras Model
        """
        # Input layer
        input_layer = layers.Input(shape=(input_dim,), name='input')
        
        # Encoder
        encoded = input_layer
        for i, units in enumerate(self.hidden_layers):
            encoded = layers.Dense(
                units, 
                activation='relu',
                name=f'encoder_{i+1}'
            )(encoded)
        
        # Bottleneck (latent space)
        encoded = layers.Dense(
            self.encoding_dim,
            activation='relu',
            name='bottleneck'
        )(encoded)
        
        # Decoder (mirror of encoder)
        decoded = encoded
        for i, units in enumerate(reversed(self.hidden_layers)):
            decoded = layers.Dense(
                units,
                activation='relu',
                name=f'decoder_{i+1}'
            )(decoded)
        
        # Output layer (same size as input)
        decoded = layers.Dense(
            input_dim,
            activation='linear',  # Linear for continuous features
            name='output'
        )(decoded)
        
        # Create model
        autoencoder = Model(inputs=input_layer, outputs=decoded, name='autoencoder')
        
        # Compile
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',  # Mean Squared Error
            metrics=['mae']  # Mean Absolute Error
        )
        
        return autoencoder
    
    def train(self,
              X: pd.DataFrame,
              validation_split: float = 0.2,
              epochs: int = 50,
              batch_size: int = 32,
              early_stopping_patience: int = 5,
              verbose: int = 1) -> Dict:
        """
        Train Autoencoder on normal transaction data
        
        Args:
            X: Feature matrix (normalized)
            validation_split: Fraction of data for validation
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Epochs to wait for improvement
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        
        Returns:
            Training metrics dictionary
        """
        print(f"\n🧠 Training Autoencoder...")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Architecture: {X.shape[1]} → {self.hidden_layers} → {self.encoding_dim} → {self.hidden_layers[::-1]} → {X.shape[1]}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Build model
        self.model = self._build_model(X.shape[1])
        
        if verbose >= 2:
            self.model.summary()
        
        # Convert to numpy
        X_np = X.values
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train model (autoencoder tries to reconstruct input)
        print(f"\n   Training for up to {epochs} epochs...")
        history = self.model.fit(
            X_np, X_np,  # Input = Output (reconstruction task)
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_history = history.history
        
        # Calculate reconstruction errors on training data
        reconstructed = self.model.predict(X_np, verbose=0)
        reconstruction_errors = np.mean(np.square(X_np - reconstructed), axis=1)
        
        # Determine threshold (95th percentile)
        self.threshold = np.percentile(reconstruction_errors, 95)
        
        # Metrics
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        epochs_trained = len(history.history['loss'])
        
        anomaly_rate = (reconstruction_errors > self.threshold).mean()
        
        print(f"\n   ✅ Training completed!")
        print(f"   Epochs trained: {epochs_trained}")
        print(f"   Final train loss: {final_train_loss:.6f}")
        print(f"   Final val loss: {final_val_loss:.6f}")
        print(f"   Threshold: {self.threshold:.6f}")
        print(f"   Anomaly rate (train): {anomaly_rate:.4%}")
        
        self.is_trained = True
        self.training_date = datetime.now()
        self.training_samples = len(X)
        
        metrics = {
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
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies on new data
        
        Args:
            X: Feature matrix (normalized)
        
        Returns:
            (reconstruction_errors, predictions)
            - reconstruction_errors: Higher = more anomalous
            - predictions: 1 = anomaly, 0 = normal
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Reconstruct input
        X_np = X.values
        reconstructed = self.model.predict(X_np, verbose=0)
        
        # Calculate reconstruction error (MSE per sample)
        errors = np.mean(np.square(X_np - reconstructed), axis=1)
        
        # Convert to predictions
        predictions = (errors > self.threshold).astype(int)
        
        # Normalize errors to 0-100 range (higher = more anomalous)
        normalized_errors = np.clip((errors / self.threshold) * 50, 0, 100)
        
        return normalized_errors, predictions
    
    def get_reconstruction(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get reconstructed features for analysis
        
        Args:
            X: Feature matrix
        
        Returns:
            Reconstructed feature matrix
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        reconstructed = self.model.predict(X.values, verbose=0)
        return pd.DataFrame(reconstructed, columns=self.feature_names, index=X.index)
    
    def get_encoding(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get latent space encoding for visualization
        
        Args:
            X: Feature matrix
        
        Returns:
            Latent space encodings
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        # Create encoder-only model
        encoder = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('bottleneck').output
        )
        
        encodings = encoder.predict(X.values, verbose=0)
        return encodings
    
    def save(self, model_path: str, metadata_path: Optional[str] = None):
        """
        Save trained model and metadata
        
        Args:
            model_path: Path to save model (.h5 or .keras)
            metadata_path: Path to save metadata (.json)
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        # Save model
        self.model.save(model_path)
        print(f"💾 Model saved: {model_path}")
        
        # Save metadata
        if metadata_path is None:
            metadata_path = model_path.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
        
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
        """
        Load trained model and metadata
        
        Args:
            model_path: Path to model file (.h5 or .keras)
            metadata_path: Path to metadata file (.json)
        """
        # Load model
        self.model = keras.models.load_model(model_path)
        print(f"📖 Model loaded: {model_path}")
        
        # Load metadata
        if metadata_path is None:
            metadata_path = model_path.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
        
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
            
            print(f"📋 Metadata loaded: {metadata_path}")
            print(f"   Trained: {self.training_date}")
            print(f"   Samples: {self.training_samples}")
        else:
            print(f"⚠️  Metadata not found, using model defaults")
            self.is_trained = True


def quick_test():
    """Quick test of Autoencoder detector"""
    if not TF_AVAILABLE:
        print("❌ TensorFlow not available. Cannot run test.")
        return
    
    print("🧪 Testing Autoencoder Detector")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Normal transactions (3 features)
    normal_data = pd.DataFrame({
        'amount': np.random.normal(100, 20, 1000),
        'fee': np.random.normal(1, 0.2, 1000),
        'hour': np.random.uniform(0, 1, 1000)
    })
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normal_normalized = pd.DataFrame(
        scaler.fit_transform(normal_data),
        columns=normal_data.columns
    )
    
    # Anomalous transactions
    anomaly_data = pd.DataFrame({
        'amount': np.random.normal(500, 50, 50),
        'fee': np.random.normal(5, 1, 50),
        'hour': np.random.uniform(0, 1, 50)
    })
    anomaly_normalized = pd.DataFrame(
        scaler.transform(anomaly_data),
        columns=anomaly_data.columns
    )
    
    # Train
    detector = AutoencoderDetector(encoding_dim=2, hidden_layers=[5])
    metrics = detector.train(normal_normalized, epochs=20, verbose=0)
    
    print(f"\n📊 Training Metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    # Predict
    X_test = pd.concat([normal_normalized.sample(100), anomaly_normalized], ignore_index=True)
    errors, predictions = detector.predict(X_test)
    
    print(f"\n🔍 Predictions:")
    print(f"   Anomalies detected: {predictions.sum()} / {len(predictions)}")
    print(f"   Anomaly rate: {predictions.mean():.2%}")
    
    print(f"\n✅ Test completed!")


if __name__ == "__main__":
    quick_test()
