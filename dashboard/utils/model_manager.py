"""
Model Manager for AnomalyGuard Dashboard
Handles ML model file operations, metadata, and versioning
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd


class ModelManager:
    """Manages model files and metadata"""

    def __init__(self, config_manager, database_manager):
        self.config_manager = config_manager
        self.database_manager = database_manager
        self.models_dir = config_manager.get('models.base_path', '/opt/airflow/models')

        # Try local path if container path doesn't exist
        if not os.path.exists(self.models_dir):
            local_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'models'
            )
            if os.path.exists(local_path):
                self.models_dir = local_path

    def list_model_files(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all model files in the models directory"""
        if not os.path.exists(self.models_dir):
            return []

        models = []

        for filename in os.listdir(self.models_dir):
            file_path = os.path.join(self.models_dir, filename)

            # Skip directories and non-model files
            if os.path.isdir(file_path):
                continue

            # Identify model type from filename and extension
            if filename.endswith('.pkl'):
                file_model_type = 'isolation_forest'
            elif filename.endswith('.keras') or filename.endswith('.h5'):
                file_model_type = 'autoencoder'
            elif filename.endswith('.json'):
                # Check if it's metadata or scaler params
                if 'metadata' in filename:
                    file_model_type = 'metadata'
                elif 'scaler' in filename:
                    file_model_type = 'scaler'
                else:
                    continue
            else:
                continue

            # Filter by type if specified
            if model_type and file_model_type != model_type and file_model_type not in ['metadata', 'scaler']:
                continue

            stat = os.stat(file_path)

            models.append({
                'filename': filename,
                'path': file_path,
                'type': file_model_type,
                'size': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime)
            })

        # Sort by modified time, newest first
        models.sort(key=lambda x: x['modified'], reverse=True)

        return models

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a model from database"""
        df = self.database_manager.get_models()

        if df.empty:
            return None

        model_row = df[df['model_id'] == model_id]

        if model_row.empty:
            return None

        return model_row.iloc[0].to_dict()

    def get_model_file_metadata(self, model_filename: str) -> Optional[Dict[str, Any]]:
        """Read metadata JSON file for a model"""
        # Construct metadata filename
        base_name = model_filename.rsplit('.', 1)[0]
        metadata_filename = f"{base_name}_metadata.json"
        metadata_path = os.path.join(self.models_dir, metadata_filename)

        if not os.path.exists(metadata_path):
            return None

        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return None

    def get_scaler_params(self) -> Optional[Dict[str, Any]]:
        """Get scaler parameters"""
        scaler_path = os.path.join(self.models_dir, 'scaler_params.json')

        if not os.path.exists(scaler_path):
            return None

        try:
            with open(scaler_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading scaler params: {e}")
            return None

    def save_scaler_params(self, params: Dict[str, Any]) -> bool:
        """Save scaler parameters"""
        scaler_path = os.path.join(self.models_dir, 'scaler_params.json')

        try:
            with open(scaler_path, 'w') as f:
                json.dump(params, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving scaler params: {e}")
            return False

    def delete_model_files(self, model_filename: str) -> tuple[bool, str]:
        """Delete model file and associated metadata"""
        try:
            # Delete main model file
            model_path = os.path.join(self.models_dir, model_filename)
            if os.path.exists(model_path):
                os.remove(model_path)

            # Delete metadata file
            base_name = model_filename.rsplit('.', 1)[0]
            metadata_filename = f"{base_name}_metadata.json"
            metadata_path = os.path.join(self.models_dir, metadata_filename)

            if os.path.exists(metadata_path):
                os.remove(metadata_path)

            return True, f"Model {model_filename} and metadata deleted successfully"

        except Exception as e:
            return False, f"Error deleting model: {str(e)}"

    def get_models_summary(self) -> Dict[str, Any]:
        """Get summary of all models"""
        # Get from database
        df_models = self.database_manager.get_models()

        # Get from filesystem
        file_models = self.list_model_files()

        # Calculate statistics
        total_models = len(df_models) if not df_models.empty else 0
        active_models = len(df_models[df_models['status'] == 'active']) if not df_models.empty else 0

        isolation_forest_count = len([f for f in file_models if f['type'] == 'isolation_forest'])
        autoencoder_count = len([f for f in file_models if f['type'] == 'autoencoder'])

        total_size_mb = sum(f['size_mb'] for f in file_models)

        return {
            'total_registered': total_models,
            'active_models': active_models,
            'inactive_models': total_models - active_models,
            'isolation_forest_files': isolation_forest_count,
            'autoencoder_files': autoencoder_count,
            'total_files': len(file_models),
            'total_size_mb': round(total_size_mb, 2),
            'models_directory': self.models_dir
        }

    def get_model_versions(self, model_type: str) -> List[Dict[str, Any]]:
        """Get all versions of a specific model type with details"""
        # Get from database
        df = self.database_manager.get_models(model_type=model_type)

        if df.empty:
            return []

        versions = []

        for _, row in df.iterrows():
            model_info = row.to_dict()

            # Add file information if available
            paths = model_info.get('paths', {})
            if isinstance(paths, str):
                try:
                    paths = json.loads(paths)
                except:
                    paths = {}

            model_path = paths.get('model_path', '')
            if model_path and os.path.exists(model_path):
                stat = os.stat(model_path)
                model_info['file_size_mb'] = round(stat.st_size / (1024 * 1024), 2)
                model_info['file_exists'] = True
            else:
                model_info['file_size_mb'] = 0
                model_info['file_exists'] = False

            versions.append(model_info)

        return versions

    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """Compare multiple models"""
        import pandas as pd

        df = self.database_manager.get_models()

        if df.empty:
            return pd.DataFrame()

        # Filter to selected models
        comparison_df = df[df['model_id'].isin(model_ids)]

        if comparison_df.empty:
            return pd.DataFrame()

        # Select relevant columns for comparison
        columns = [
            'model_id', 'model_type', 'version', 'trained_at',
            'metrics', 'threshold', 'status'
        ]

        available_columns = [col for col in columns if col in comparison_df.columns]

        return comparison_df[available_columns]

    def get_latest_active_models(self) -> Dict[str, Any]:
        """Get the latest active model for each type"""
        df = self.database_manager.get_active_models()

        if df.empty:
            return {}

        models = {}

        for model_type in ['isolation_forest', 'autoencoder', 'ensemble']:
            type_df = df[df['model_type'] == model_type]

            if not type_df.empty:
                # Sort by trained_at and get the latest
                type_df = type_df.sort_values('trained_at', ascending=False)
                models[model_type] = type_df.iloc[0].to_dict()

        return models

    def activate_model_version(self, model_id: str) -> tuple[bool, str]:
        """Activate a specific model version"""
        # Get model info
        metadata = self.get_model_metadata(model_id)

        if not metadata:
            return False, "Model not found in registry"

        model_type = metadata['model_type']

        # Use database manager to set active model
        success = self.database_manager.set_active_model(model_type, model_id)

        if success:
            return True, f"Model {model_id} activated successfully"
        return False, "Failed to activate model"

    def archive_old_models(
        self,
        model_type: str,
        keep_latest: int = 5
    ) -> tuple[int, List[str]]:
        """Archive old models, keeping only the latest N versions"""
        df = self.database_manager.get_models(model_type=model_type)

        if df.empty or len(df) <= keep_latest:
            return 0, []

        # Sort by trained_at
        df = df.sort_values('trained_at', ascending=False)

        # Get models to archive (skip the latest N)
        to_archive = df.iloc[keep_latest:]

        archived_count = 0
        archived_ids = []

        for _, row in to_archive.iterrows():
            model_id = row['model_id']
            success = self.database_manager.update_model_status(model_id, 'archived')

            if success:
                archived_count += 1
                archived_ids.append(model_id)

        return archived_count, archived_ids

    def get_model_performance_trend(
        self,
        model_type: str,
        days: int = 30
    ) -> pd.DataFrame:
        """Get performance trend for a model type"""
        return self.database_manager.get_model_performance(
            days=days,
            model_type=model_type
        )

    def export_model_metadata(self, model_id: str, export_path: str) -> bool:
        """Export model metadata to a file"""
        metadata = self.get_model_metadata(model_id)

        if not metadata:
            return False

        try:
            with open(export_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error exporting metadata: {e}")
            return False

    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get current ensemble weights from config"""
        return self.config_manager.get('ensemble.weights', {
            'statistical': 0.2,
            'isolation_forest': 0.4,
            'autoencoder': 0.4
        })

    def update_ensemble_weights(self, weights: Dict[str, float]) -> bool:
        """Update ensemble weights in config"""
        return self.config_manager.update_ensemble_weights(weights)

    def get_risk_thresholds(self) -> Dict[str, int]:
        """Get current risk thresholds from config"""
        return self.config_manager.get('ensemble.thresholds', {
            'low': 30,
            'medium': 60,
            'high': 80,
            'critical': 90
        })

    def update_risk_thresholds(self, thresholds: Dict[str, int]) -> bool:
        """Update risk thresholds in config"""
        return self.config_manager.update_risk_thresholds(thresholds)
