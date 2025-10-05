"""
ML Models for Anomaly Detection - FIXED VERSION
================================================
این نسخه مشکل feature mismatch رو نداره
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import pickle
import os

logger = logging.getLogger(__name__)


class AnomalyDetectionEnsemble:
    """Ensemble of anomaly detection models - با feature consistency"""
    
    def __init__(self, baselines: Optional[Dict] = None):
        self.baselines = baselines or {}
        self.models = {}
        self.scaler = None
        self.feature_names = None
    
    def train(self, features_df: pd.DataFrame, contamination: float = 0.05):
        """
        Train ensemble of models.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        
        logger.info("🏋️ Training anomaly detection ensemble...")
        
        # Prepare data - فقط numeric columns
        X = self._prepare_features(features_df)
        
        if len(X) < 10:
            raise ValueError(f"Not enough data! Need at least 10 rows, got {len(X)}")
        
        # ذخیره feature names برای استفاده بعدی
        self.feature_names = X.columns.tolist()
        logger.info(f"📊 Training with {len(X)} samples and {len(self.feature_names)} features")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. Isolation Forest
        logger.info("  Training Isolation Forest...")
        iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            max_samples=min(256, len(X)),
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X_scaled)
        self.models['isolation_forest'] = iso_forest
        
        # 2. Local Outlier Factor
        logger.info("  Training Local Outlier Factor...")
        n_neighbors = min(20, len(X) - 1)
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1
        )
        lof.fit(X_scaled)
        self.models['lof'] = lof
        
        # 3. Statistical thresholds
        logger.info("  Calculating statistical thresholds...")
        self.models['statistical'] = self._calculate_statistical_thresholds(X)
        
        logger.info(f"✅ Trained {len(self.models)} models successfully")
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies using ensemble.
        """
        logger.info("🔍 Predicting anomalies...")
        
        # Prepare features - با همون ستون‌هایی که training داشتیم
        X = self._prepare_features(features_df, inference=True)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        results = features_df.copy()
        
        # 1. Isolation Forest
        results['iso_forest_score'] = self.models['isolation_forest'].score_samples(X_scaled)
        results['iso_forest_anomaly'] = self.models['isolation_forest'].predict(X_scaled)
        results['iso_forest_anomaly'] = (results['iso_forest_anomaly'] == -1).astype(int)
        
        # 2. LOF
        results['lof_score'] = self.models['lof'].score_samples(X_scaled)
        results['lof_anomaly'] = self.models['lof'].predict(X_scaled)
        results['lof_anomaly'] = (results['lof_anomaly'] == -1).astype(int)
        
        # 3. Statistical
        results['statistical_anomaly'] = self._statistical_predict(X)
        
        # Calculate ensemble score
        results['anomaly_score'] = self._calculate_ensemble_score(results)
        
        # Final decision (threshold at 95th percentile)
        threshold = results['anomaly_score'].quantile(0.95)
        results['is_anomaly'] = (results['anomaly_score'] > threshold).astype(int)
        
        # Categorize anomalies
        results['anomaly_type'] = results.apply(self._categorize_anomaly, axis=1)
        
        anomaly_count = results['is_anomaly'].sum()
        logger.info(f"✅ Detected {anomaly_count} anomalies")
        
        return results
    
    def _prepare_features(self, df: pd.DataFrame, inference: bool = False) -> pd.DataFrame:
        """
        CRITICAL: این متد feature consistency رو تضمین می‌کنه
        """
        # فقط numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_cols].copy()
        
        # حذف null values
        X = X.fillna(X.median())
        
        # حذف infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # حذف constant columns (فقط در training)
        if not inference:
            # Training mode: حذف ستون‌های constant
            X = X.loc[:, X.std() > 0]
        else:
            # Inference mode: استفاده از همون feature های training
            if self.feature_names is not None:
                # فقط ستون‌هایی که در training بودن رو نگه دار
                missing_features = set(self.feature_names) - set(X.columns)
                extra_features = set(X.columns) - set(self.feature_names)
                
                # اضافه کردن ستون‌های missing با مقدار 0
                for feature in missing_features:
                    X[feature] = 0
                
                # حذف ستون‌های اضافی
                X = X.drop(columns=list(extra_features), errors='ignore')
                
                # ترتیب ستون‌ها رو مثل training کن
                X = X[self.feature_names]
        
        return X
    
    def _calculate_statistical_thresholds(self, X: pd.DataFrame) -> Dict:
        """Calculate statistical thresholds for each feature"""
        thresholds = {}
        
        for col in X.columns:
            col_data = X[col]
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            
            thresholds[col] = {
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'lower_bound': float(q1 - 1.5 * iqr),
                'upper_bound': float(q3 + 1.5 * iqr),
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
        iso_norm = self._normalize_score(-results['iso_forest_score'])
        lof_norm = self._normalize_score(-results['lof_score'])
        
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