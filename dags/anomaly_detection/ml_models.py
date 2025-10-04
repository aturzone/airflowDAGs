"""
ML Models for Anomaly Detection
================================
Ensemble approach combining multiple algorithms.
Inspired by Netdata AI's multi-model strategy.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import pickle
import os

logger = logging.getLogger(__name__)


class AnomalyDetectionEnsemble:
    """Ensemble of anomaly detection models"""
    
    def __init__(self, baselines: Optional[Dict] = None):
        self.baselines = baselines or {}
        self.models = {}
        self.scaler = None
        self.feature_names = None
    
    def train(self, features_df: pd.DataFrame, contamination: float = 0.05):
        """
        Train ensemble of models.
        
        Models:
        1. Isolation Forest - multivariate anomalies
        2. Local Outlier Factor - density-based
        3. Statistical - z-score based
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        
        logger.info("🏋️ Training anomaly detection ensemble...")
        
        # Prepare data
        X = self._prepare_features(features_df)
        self.feature_names = X.columns.tolist()
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. Isolation Forest
        logger.info("  Training Isolation Forest...")
        iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X_scaled)
        self.models['isolation_forest'] = iso_forest
        
        # 2. Local Outlier Factor
        logger.info("  Training Local Outlier Factor...")
        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True,  # For prediction on new data
            n_jobs=-1
        )
        lof.fit(X_scaled)
        self.models['lof'] = lof
        
        # 3. Statistical thresholds
        logger.info("  Calculating statistical thresholds...")
        self.models['statistical'] = self._calculate_statistical_thresholds(X)
        
        logger.info(f"✅ Trained {len(self.models)} models")
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies using ensemble.
        Returns DataFrame with anomaly scores and labels.
        """
        logger.info("🔍 Predicting anomalies...")
        
        # Prepare features
        X = self._prepare_features(features_df)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        results = features_df.copy()
        
        # 1. Isolation Forest
        iso_scores = self.models['isolation_forest'].decision_function(X_scaled)
        iso_pred = self.models['isolation_forest'].predict(X_scaled)
        results['iso_forest_score'] = iso_scores
        results['iso_forest_anomaly'] = (iso_pred == -1).astype(int)
        
        # 2. Local Outlier Factor
        lof_scores = self.models['lof'].decision_function(X_scaled)
        lof_pred = self.models['lof'].predict(X_scaled)
        results['lof_score'] = lof_scores
        results['lof_anomaly'] = (lof_pred == -1).astype(int)
        
        # 3. Statistical
        stat_anomalies = self._statistical_predict(X)
        results['statistical_anomaly'] = stat_anomalies
        
        # Ensemble scoring
        results['anomaly_score'] = self._calculate_ensemble_score(results)
        
        # Final anomaly label (threshold-based)
        threshold = results['anomaly_score'].quantile(0.95)
        results['is_anomaly'] = results['anomaly_score'] > threshold
        
        # Categorize anomaly type
        results['anomaly_type'] = results.apply(self._categorize_anomaly, axis=1)
        
        logger.info(f"✅ Detected {results['is_anomaly'].sum()} anomalies")
        
        return results
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling"""
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Remove ID columns
        feature_cols = [col for col in numeric_cols 
                       if not col.endswith('_id') and col != 'row_id']
        
        X = df[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        return X
    
    def _calculate_statistical_thresholds(self, X: pd.DataFrame) -> Dict:
        """Calculate statistical thresholds for each feature"""
        thresholds = {}
        
        for col in X.columns:
            mean = X[col].mean()
            std = X[col].std()
            median = X[col].median()
            q25 = X[col].quantile(0.25)
            q75 = X[col].quantile(0.75)
            iqr = q75 - q25
            
            thresholds[col] = {
                'mean': mean,
                'std': std,
                'median': median,
                'iqr': iqr,
                'lower_bound': q25 - 1.5 * iqr,
                'upper_bound': q75 + 1.5 * iqr,
                'z_threshold': 3.0
            }
        
        return thresholds
    
    def _statistical_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using statistical methods"""
        thresholds = self.models['statistical']
        anomalies = np.zeros(len(X))
        
        for col in X.columns:
            if col in thresholds:
                t = thresholds[col]
                
                # Z-score method
                z_scores = np.abs((X[col] - t['mean']) / (t['std'] + 1e-10))
                
                # IQR method
                iqr_outliers = (X[col] < t['lower_bound']) | (X[col] > t['upper_bound'])
                
                # Combine
                col_anomalies = (z_scores > t['z_threshold']) | iqr_outliers
                anomalies += col_anomalies.astype(int)
        
        # Normalize to 0-1
        return (anomalies > 0).astype(int)
    
    def _calculate_ensemble_score(self, results: pd.DataFrame) -> pd.Series:
        """Calculate weighted ensemble score"""
        
        # Normalize scores to 0-1 range
        iso_norm = self._normalize_score(-results['iso_forest_score'])  # Invert (more negative = more anomalous)
        lof_norm = self._normalize_score(-results['lof_score'])  # Invert
        
        # Weighted average
        ensemble_score = (
            0.40 * iso_norm +
            0.35 * lof_norm +
            0.25 * results['statistical_anomaly']
        )
        
        return ensemble_score
    
    def _normalize_score(self, scores: pd.Series) -> pd.Series:
        """Normalize scores to 0-1 range"""
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score > 0:
            return (scores - min_score) / (max_score - min_score)
        else:
            return pd.Series([0.5] * len(scores))
    
    def _categorize_anomaly(self, row) -> str:
        """Categorize type of anomaly"""
        if not row['is_anomaly']:
            return 'normal'
        
        # Determine which model(s) flagged it
        flagged_by = []
        
        if row.get('iso_forest_anomaly', 0) == 1:
            flagged_by.append('isolation')
        if row.get('lof_anomaly', 0) == 1:
            flagged_by.append('density')
        if row.get('statistical_anomaly', 0) == 1:
            flagged_by.append('statistical')
        
        if len(flagged_by) >= 2:
            return 'consensus_anomaly'
        elif 'isolation' in flagged_by:
            return 'multivariate_anomaly'
        elif 'density' in flagged_by:
            return 'density_anomaly'
        elif 'statistical' in flagged_by:
            return 'statistical_outlier'
        
        return 'unknown_anomaly'
    
    def save_models(self, path: str):
        """Save trained models"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'baselines': self.baselines
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Saved models to {path}")
    
    def load_models(self, path: str):
        """Load pre-trained models"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.baselines = model_data['baselines']
        
        logger.info(f"✅ Loaded models from {path}")
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'models': list(self.models.keys()),
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'features': self.feature_names
        }


class AdaptiveBaseline:
    """
    Adaptive baseline calculation similar to Netdata AI.
    Maintains rolling statistics and auto-adjusts thresholds.
    """
    
    def __init__(self, window_size: int = 90):
        self.window_size = window_size  # Days
        self.baselines = {}
    
    def update(self, feature_name: str, values: pd.Series):
        """Update baseline for a feature"""
        
        if feature_name not in self.baselines:
            self.baselines[feature_name] = {
                'history': [],
                'stats': {}
            }
        
        baseline = self.baselines[feature_name]
        
        # Add new values
        baseline['history'].extend(values.tolist())
        
        # Keep only recent window
        if len(baseline['history']) > self.window_size * 24:  # Assuming hourly data
            baseline['history'] = baseline['history'][-self.window_size * 24:]
        
        # Recalculate statistics
        data = pd.Series(baseline['history'])
        
        baseline['stats'] = {
            'mean': data.mean(),
            'std': data.std(),
            'median': data.median(),
            'p95': data.quantile(0.95),
            'p99': data.quantile(0.99),
            'min': data.min(),
            'max': data.max()
        }
    
    def is_anomaly(self, feature_name: str, value: float, sensitivity: float = 3.0) -> bool:
        """Check if value is anomalous"""
        
        if feature_name not in self.baselines:
            return False
        
        stats = self.baselines[feature_name]['stats']
        
        # Z-score method
        if stats['std'] > 0:
            z_score = abs((value - stats['mean']) / stats['std'])
            if z_score > sensitivity:
                return True
        
        # Percentile method
        if value > stats['p99']:
            return True
        
        return False