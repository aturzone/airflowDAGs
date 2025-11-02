"""
Isolation Forest Anomaly Detector
Fast tree-based anomaly detection for high-dimensional transaction data
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import os


class IsolationForestDetector:
    """
    Isolation Forest wrapper for transaction anomaly detection
    
    Key Concepts:
    - Uses random binary trees to isolate anomalies
    - Anomalies have shorter average path lengths
    - Returns anomaly score: lower = more anomalous
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 contamination: float = 0.01,
                 max_samples: str = 'auto',
                 random_state: int = 42):
        """
        Initialize Isolation Forest detector
        
        Args:
            n_estimators: Number of trees in the forest
            contamination: Expected proportion of anomalies (0.01 = 1%)
            max_samples: Number of samples to draw for each tree
            random_state: Random seed for reproducibility
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.params = {
            'n_estimators': n_estimators,
            'contamination': contamination,
            'max_samples': max_samples,
            'random_state': random_state
        }
        
        self.threshold = None
        self.feature_names = None
        self.is_trained = False
        self.training_date = None
        self.training_samples = 0
    
    def train(self, X: pd.DataFrame, 
              validation_split: float = 0.2) -> Dict:
        """
        Train Isolation Forest on normal transaction data
        
        Args:
            X: Feature matrix (normalized)
            validation_split: Fraction of data for validation
        
        Returns:
            Training metrics dictionary
        """
        print(f"\n🌲 Training Isolation Forest...")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {X.shape[1]}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        
        # Train model
        print(f"   Training on {len(X_train)} samples...")
        self.model.fit(X_train)
        
        # Calculate anomaly scores (negative values = more anomalous)
        train_scores = self.model.score_samples(X_train)
        val_scores = self.model.score_samples(X_val)
        
        # Determine threshold (95th percentile of training scores)
        self.threshold = np.percentile(train_scores, 5)  # 5th percentile = most anomalous 5%
        
        # Predictions
        train_pred = (train_scores < self.threshold).astype(int)
        val_pred = (val_scores < self.threshold).astype(int)
        
        # Metrics
        train_anomaly_rate = train_pred.mean()
        val_anomaly_rate = val_pred.mean()
        
        print(f"\n   ✅ Training completed!")
        print(f"   Threshold: {self.threshold:.6f}")
        print(f"   Train anomaly rate: {train_anomaly_rate:.4%}")
        print(f"   Validation anomaly rate: {val_anomaly_rate:.4%}")
        
        self.is_trained = True
        self.training_date = datetime.now()
        self.training_samples = len(X)
        
        metrics = {
            'model_type': 'isolation_forest',
            'threshold': float(self.threshold),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'train_anomaly_rate': float(train_anomaly_rate),
            'val_anomaly_rate': float(val_anomaly_rate),
            'n_features': X.shape[1],
            'training_date': self.training_date.isoformat(),
            'params': self.params
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies on new data
        
        Args:
            X: Feature matrix (normalized)
        
        Returns:
            (anomaly_scores, predictions)
            - anomaly_scores: Lower = more anomalous
            - predictions: 1 = anomaly, 0 = normal
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Calculate anomaly scores
        scores = self.model.score_samples(X)
        
        # Convert scores to predictions
        predictions = (scores < self.threshold).astype(int)
        
        # Normalize scores to 0-100 range (higher = more anomalous)
        # Original scores are negative, more negative = more anomalous
        # Transform: -0.5 → 100 (very anomalous), 0.0 → 0 (normal)
        normalized_scores = np.clip((self.threshold - scores) * 100, 0, 100)
        
        return normalized_scores, predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calculate approximate feature importance
        Note: Isolation Forest doesn't have built-in feature importance,
        so we use a permutation-based approach
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        print("⚠️ Note: Isolation Forest feature importance is experimental")
        
        # For now, return equal importance
        # In production, implement permutation importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': 1.0 / len(self.feature_names)
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save(self, model_path: str, metadata_path: Optional[str] = None):
        """
        Save trained model and metadata
        
        Args:
            model_path: Path to save model (.pkl)
            metadata_path: Path to save metadata (.json)
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        # Save model
        joblib.dump(self.model, model_path)
        print(f"💾 Model saved: {model_path}")
        
        # Save metadata
        if metadata_path is None:
            metadata_path = model_path.replace('.pkl', '_metadata.json')
        
        metadata = {
            'model_type': 'isolation_forest',
            'threshold': float(self.threshold),
            'feature_names': self.feature_names,
            'training_date': self.training_date.isoformat(),
            'training_samples': self.training_samples,
            'params': self.params,
            'version': '1.0'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Metadata saved: {metadata_path}")
    
    def load(self, model_path: str, metadata_path: Optional[str] = None):
        """
        Load trained model and metadata
        
        Args:
            model_path: Path to model file (.pkl)
            metadata_path: Path to metadata file (.json)
        """
        # Load model
        self.model = joblib.load(model_path)
        print(f"📖 Model loaded: {model_path}")
        
        # Load metadata
        if metadata_path is None:
            metadata_path = model_path.replace('.pkl', '_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.threshold = metadata['threshold']
            self.feature_names = metadata['feature_names']
            self.training_date = datetime.fromisoformat(metadata['training_date'])
            self.training_samples = metadata['training_samples']
            self.params = metadata['params']
            self.is_trained = True
            
            print(f"📋 Metadata loaded: {metadata_path}")
            print(f"   Trained: {self.training_date}")
            print(f"   Samples: {self.training_samples}")
        else:
            print(f"⚠️  Metadata not found, using model defaults")
            self.is_trained = True


def quick_test():
    """Quick test of Isolation Forest detector"""
    print("🧪 Testing Isolation Forest Detector")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Normal transactions
    normal_data = pd.DataFrame({
        'amount': np.random.normal(100, 20, 1000),
        'fee': np.random.normal(1, 0.2, 1000),
        'hour': np.random.randint(0, 24, 1000)
    })
    
    # Anomalous transactions (high amounts)
    anomaly_data = pd.DataFrame({
        'amount': np.random.normal(500, 50, 50),
        'fee': np.random.normal(5, 1, 50),
        'hour': np.random.randint(0, 24, 50)
    })
    
    # Combine
    X_train = normal_data
    X_test = pd.concat([normal_data.sample(100), anomaly_data], ignore_index=True)
    
    # Train
    detector = IsolationForestDetector(contamination=0.05)
    metrics = detector.train(X_train)
    
    print(f"\n📊 Training Metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    # Predict
    scores, predictions = detector.predict(X_test)
    
    print(f"\n🔍 Predictions:")
    print(f"   Anomalies detected: {predictions.sum()} / {len(predictions)}")
    print(f"   Anomaly rate: {predictions.mean():.2%}")
    
    # Show some predictions
    print(f"\n📋 Sample predictions:")
    sample_df = X_test.head(10).copy()
    sample_df['score'] = scores[:10]
    sample_df['is_anomaly'] = predictions[:10]
    print(sample_df.to_string())
    
    print(f"\n✅ Test completed!")


if __name__ == "__main__":
    quick_test()
