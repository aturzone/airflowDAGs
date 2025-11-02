"""
Ensemble Anomaly Detector
Combines multiple models for robust anomaly detection
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import json


class EnsembleDetector:
    """
    Ensemble detector combining:
    1. Statistical layer (fast rule-based checks)
    2. Isolation Forest (tree-based anomaly detection)
    3. Autoencoder (neural network reconstruction)
    
    Returns comprehensive risk score and decision
    """
    
    def __init__(self,
                 isolation_forest_detector=None,
                 autoencoder_detector=None,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize Ensemble detector
        
        Args:
            isolation_forest_detector: Trained IsolationForestDetector
            autoencoder_detector: Trained AutoencoderDetector
            weights: Weights for combining scores
                    {'statistical': 0.2, 'isolation': 0.4, 'autoencoder': 0.4}
        """
        self.iso_detector = isolation_forest_detector
        self.ae_detector = autoencoder_detector
        
        # Default weights
        if weights is None:
            weights = {
                'statistical': 0.2,
                'isolation': 0.4,
                'autoencoder': 0.4
            }
        
        self.weights = weights
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 30,
            'medium': 60,
            'high': 80,
            'critical': 95
        }
        
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
    
    def statistical_layer(self, df: pd.DataFrame, 
                         features: pd.DataFrame) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Fast statistical checks for obvious anomalies
        
        Args:
            df: Original transaction dataframe (with metadata)
            features: Extracted features
        
        Returns:
            (statistical_scores, flags_per_transaction)
        """
        n_samples = len(df)
        scores = np.zeros(n_samples)
        flags = [[] for _ in range(n_samples)]
        
        # Rule 1: Extremely high amount compared to user average
        if 'amount_vs_user_avg' in features.columns:
            high_amount_mask = features['amount_vs_user_avg'] > 10
            scores[high_amount_mask] += 30
            for idx in np.where(high_amount_mask)[0]:
                flags[idx].append('amount_10x_avg')
        
        # Rule 2: Unusual time (night transactions)
        if 'is_night' in features.columns:
            night_mask = features['is_night'] == 1
            scores[night_mask] += 10
            for idx in np.where(night_mask)[0]:
                flags[idx].append('night_transaction')
        
        # Rule 3: New user with high amount
        if 'user_account_age_days' in features.columns and 'amount' in df.columns:
            new_user_high_amount = (
                (features['user_account_age_days'] < 7) & 
                (df['amount'] > df['amount'].quantile(0.90))
            )
            scores[new_user_high_amount] += 25
            for idx in np.where(new_user_high_amount)[0]:
                flags[idx].append('new_user_high_amount')
        
        # Rule 4: Unusual transaction frequency
        if 'user_total_tx' in features.columns:
            # User with very few transactions doing large amount
            low_tx_user = features['user_total_tx'] < 5
            if 'amount' in df.columns:
                high_amount = df['amount'] > df['amount'].quantile(0.75)
                unusual_freq = low_tx_user & high_amount
                scores[unusual_freq] += 20
                for idx in np.where(unusual_freq)[0]:
                    flags[idx].append('low_tx_high_amount')
        
        # Rule 5: High fee ratio (suspicious)
        if 'fee_ratio' in features.columns:
            high_fee = features['fee_ratio'] > features['fee_ratio'].quantile(0.95)
            scores[high_fee] += 15
            for idx in np.where(high_fee)[0]:
                flags[idx].append('high_fee_ratio')
        
        # Normalize scores to 0-100
        scores = np.clip(scores, 0, 100)
        
        return scores, flags
    
    def predict(self, df: pd.DataFrame, 
                features: pd.DataFrame,
                return_details: bool = True) -> Dict:
        """
        Predict anomalies using ensemble approach
        
        Args:
            df: Original transaction dataframe
            features: Extracted and normalized features
            return_details: Return detailed predictions per layer
        
        Returns:
            Dictionary with scores, predictions, and metadata
        """
        if self.iso_detector is None or self.ae_detector is None:
            raise ValueError("Both detectors must be provided and trained")
        
        n_samples = len(df)
        
        print(f"\n🎯 Ensemble prediction on {n_samples} transactions...")
        
        # Layer 1: Statistical checks
        print("   Layer 1: Statistical checks...")
        stat_scores, stat_flags = self.statistical_layer(df, features)
        
        # Layer 2: Isolation Forest
        print("   Layer 2: Isolation Forest...")
        iso_scores, iso_preds = self.iso_detector.predict(features)
        
        # Layer 3: Autoencoder
        print("   Layer 3: Autoencoder...")
        ae_scores, ae_preds = self.ae_detector.predict(features)
        
        # Combine scores (weighted sum)
        total_risk = (
            self.weights['statistical'] * stat_scores +
            self.weights['isolation'] * iso_scores +
            self.weights['autoencoder'] * ae_scores
        )
        
        # Determine risk levels
        risk_levels = np.array(['low'] * n_samples, dtype=object)
        risk_levels[total_risk >= self.risk_thresholds['medium']] = 'medium'
        risk_levels[total_risk >= self.risk_thresholds['high']] = 'high'
        risk_levels[total_risk >= self.risk_thresholds['critical']] = 'critical'
        
        # Final decisions
        decisions = np.array(['approved'] * n_samples, dtype=object)
        decisions[total_risk >= self.risk_thresholds['medium']] = 'review'
        decisions[total_risk >= self.risk_thresholds['critical']] = 'blocked'
        
        # Summary
        print(f"\n   ✅ Ensemble prediction completed!")
        print(f"   Risk distribution:")
        print(f"      Low:      {(risk_levels == 'low').sum():>4} ({(risk_levels == 'low').mean():>6.1%})")
        print(f"      Medium:   {(risk_levels == 'medium').sum():>4} ({(risk_levels == 'medium').mean():>6.1%})")
        print(f"      High:     {(risk_levels == 'high').sum():>4} ({(risk_levels == 'high').mean():>6.1%})")
        print(f"      Critical: {(risk_levels == 'critical').sum():>4} ({(risk_levels == 'critical').mean():>6.1%})")
        print(f"\n   Final decisions:")
        print(f"      Approved: {(decisions == 'approved').sum():>4} ({(decisions == 'approved').mean():>6.1%})")
        print(f"      Review:   {(decisions == 'review').sum():>4} ({(decisions == 'review').mean():>6.1%})")
        print(f"      Blocked:  {(decisions == 'blocked').sum():>4} ({(decisions == 'blocked').mean():>6.1%})")
        
        # Build result dictionary
        results = {
            'total_risk_score': total_risk,
            'risk_level': risk_levels,
            'final_decision': decisions,
            'n_samples': n_samples,
            'summary': {
                'approved': int((decisions == 'approved').sum()),
                'review': int((decisions == 'review').sum()),
                'blocked': int((decisions == 'blocked').sum()),
                'avg_risk_score': float(total_risk.mean()),
                'max_risk_score': float(total_risk.max())
            }
        }
        
        if return_details:
            results['layer_scores'] = {
                'statistical': stat_scores,
                'isolation': iso_scores,
                'autoencoder': ae_scores
            }
            results['layer_predictions'] = {
                'isolation': iso_preds,
                'autoencoder': ae_preds
            }
            results['statistical_flags'] = stat_flags
        
        return results
    
    def get_high_risk_transactions(self, df: pd.DataFrame, 
                                   results: Dict,
                                   risk_threshold: str = 'high') -> pd.DataFrame:
        """
        Extract high-risk transactions for review
        
        Args:
            df: Original transaction dataframe
            results: Results from predict()
            risk_threshold: Minimum risk level ('medium', 'high', 'critical')
        
        Returns:
            DataFrame with high-risk transactions and scores
        """
        risk_order = ['low', 'medium', 'high', 'critical']
        min_level_idx = risk_order.index(risk_threshold)
        
        # Filter transactions
        mask = np.array([
            risk_order.index(level) >= min_level_idx 
            for level in results['risk_level']
        ])
        
        high_risk_df = df[mask].copy()
        high_risk_df['total_risk_score'] = results['total_risk_score'][mask]
        high_risk_df['risk_level'] = results['risk_level'][mask]
        high_risk_df['final_decision'] = results['final_decision'][mask]
        
        if 'layer_scores' in results:
            high_risk_df['statistical_score'] = results['layer_scores']['statistical'][mask]
            high_risk_df['isolation_score'] = results['layer_scores']['isolation'][mask]
            high_risk_df['autoencoder_score'] = results['layer_scores']['autoencoder'][mask]
        
        if 'statistical_flags' in results:
            high_risk_df['flags'] = [results['statistical_flags'][i] for i in np.where(mask)[0]]
        
        # Sort by risk score
        high_risk_df = high_risk_df.sort_values('total_risk_score', ascending=False)
        
        return high_risk_df
    
    def save_config(self, path: str):
        """Save ensemble configuration"""
        config = {
            'weights': self.weights,
            'risk_thresholds': self.risk_thresholds,
            'metadata': self.metadata
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Ensemble config saved: {path}")
    
    def load_config(self, path: str):
        """Load ensemble configuration"""
        with open(path, 'r') as f:
            config = json.load(f)
        
        self.weights = config['weights']
        self.risk_thresholds = config['risk_thresholds']
        self.metadata = config['metadata']
        
        print(f"📖 Ensemble config loaded: {path}")


def create_ensemble_report(df: pd.DataFrame, 
                          results: Dict,
                          top_n: int = 10) -> str:
    """
    Create a human-readable report of ensemble predictions
    
    Args:
        df: Original transaction dataframe
        results: Results from predict()
        top_n: Number of top risky transactions to show
    
    Returns:
        Formatted report string
    """
    report = []
    report.append("\n" + "=" * 80)
    report.append("ENSEMBLE ANOMALY DETECTION REPORT")
    report.append("=" * 80)
    
    # Summary
    report.append(f"\n📊 SUMMARY:")
    report.append(f"   Total transactions analyzed: {results['n_samples']}")
    report.append(f"   Average risk score: {results['summary']['avg_risk_score']:.2f}")
    report.append(f"   Maximum risk score: {results['summary']['max_risk_score']:.2f}")
    
    report.append(f"\n🎯 DECISIONS:")
    report.append(f"   ✅ Approved: {results['summary']['approved']:>6} ({results['summary']['approved']/results['n_samples']:>6.1%})")
    report.append(f"   ⚠️  Review:   {results['summary']['review']:>6} ({results['summary']['review']/results['n_samples']:>6.1%})")
    report.append(f"   🚫 Blocked:  {results['summary']['blocked']:>6} ({results['summary']['blocked']/results['n_samples']:>6.1%})")
    
    # Top risky transactions
    if results['summary']['blocked'] > 0 or results['summary']['review'] > 0:
        report.append(f"\n🔴 TOP {top_n} RISKY TRANSACTIONS:")
        report.append("-" * 80)
        
        # Sort by risk score
        risk_indices = np.argsort(results['total_risk_score'])[::-1][:top_n]
        
        for i, idx in enumerate(risk_indices, 1):
            tx = df.iloc[idx]
            risk = results['total_risk_score'][idx]
            level = results['risk_level'][idx]
            decision = results['final_decision'][idx]
            
            report.append(f"\n#{i}. Transaction {tx.get('transaction_id', 'N/A')}")
            report.append(f"   User: {tx.get('user_id', 'N/A')}")
            report.append(f"   Amount: {tx.get('amount', 0):.2f} {tx.get('currency', '')}")
            report.append(f"   Type: {tx.get('transaction_type', 'N/A')}")
            report.append(f"   Risk Score: {risk:.2f} ({level})")
            report.append(f"   Decision: {decision.upper()}")
            
            if 'layer_scores' in results:
                report.append(f"   Layer Scores:")
                report.append(f"      Statistical:  {results['layer_scores']['statistical'][idx]:.2f}")
                report.append(f"      Isolation:    {results['layer_scores']['isolation'][idx]:.2f}")
                report.append(f"      Autoencoder:  {results['layer_scores']['autoencoder'][idx]:.2f}")
            
            if 'statistical_flags' in results and results['statistical_flags'][idx]:
                report.append(f"   Flags: {', '.join(results['statistical_flags'][idx])}")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


if __name__ == "__main__":
    print("Ensemble Anomaly Detector Module")
    print("=" * 50)
    print("This module combines multiple detectors for robust anomaly detection")
    print("\nComponents:")
    print("  1. Statistical Layer: Fast rule-based checks")
    print("  2. Isolation Forest: Tree-based anomaly detection")
    print("  3. Autoencoder: Neural network reconstruction")
    print("\nUsage:")
    print("  from models.ensemble_detector import EnsembleDetector")
    print("  ensemble = EnsembleDetector(iso_detector, ae_detector)")
    print("  results = ensemble.predict(df, features)")
