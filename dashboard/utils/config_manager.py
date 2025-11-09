"""
Configuration Manager for AnomalyGuard Dashboard
Handles loading, validation, and updating of system configuration
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime


class ConfigManager:
    """Centralized configuration management"""

    def __init__(self, config_path: str = "/opt/airflow/config/config.yaml"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            return self.config
        except FileNotFoundError:
            # Fall back to local path if running outside container
            local_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config',
                'config.yaml'
            )
            if os.path.exists(local_path):
                with open(local_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                return self.config
            raise FileNotFoundError(f"Config file not found at {self.config_path} or {local_path}")

    def save_config(self, backup: bool = True) -> bool:
        """Save current configuration to file with optional backup"""
        try:
            # Create backup if requested
            if backup and os.path.exists(self.config_path):
                backup_path = f"{self.config_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                os.system(f"cp {self.config_path} {backup_path}")

            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: get('databases.clickhouse.host')
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> bool:
        """
        Set configuration value using dot notation
        Example: set('databases.clickhouse.host', 'new-host')
        """
        keys = key_path.split('.')
        config = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value
        return True

    def get_clickhouse_config(self) -> Dict[str, Any]:
        """Get ClickHouse connection configuration"""
        return self.get('databases.clickhouse', {})

    def get_postgres_config(self) -> Dict[str, Any]:
        """Get PostgreSQL connection configuration"""
        return self.get('databases.postgres', {})

    def get_airflow_config(self) -> Dict[str, Any]:
        """Get Airflow configuration"""
        return self.get('airflow', {})

    def get_model_config(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Get model configuration"""
        if model_type:
            return self.get(f'models.{model_type}', {})
        return self.get('models', {})

    def get_ensemble_config(self) -> Dict[str, Any]:
        """Get ensemble configuration"""
        return self.get('ensemble', {})

    def update_ensemble_weights(self, weights: Dict[str, float]) -> bool:
        """Update ensemble model weights"""
        # Validate weights sum to 1.0
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        self.set('ensemble.weights.statistical', weights.get('statistical', 0.2))
        self.set('ensemble.weights.isolation_forest', weights.get('isolation_forest', 0.4))
        self.set('ensemble.weights.autoencoder', weights.get('autoencoder', 0.4))
        return self.save_config()

    def update_risk_thresholds(self, thresholds: Dict[str, int]) -> bool:
        """Update risk level thresholds"""
        self.set('ensemble.thresholds.low', thresholds.get('low', 30))
        self.set('ensemble.thresholds.medium', thresholds.get('medium', 60))
        self.set('ensemble.thresholds.high', thresholds.get('high', 80))
        self.set('ensemble.thresholds.critical', thresholds.get('critical', 90))
        return self.save_config()

    def update_database_connection(self, db_type: str, config: Dict[str, Any]) -> bool:
        """Update database connection configuration"""
        if db_type not in ['clickhouse', 'postgres']:
            raise ValueError(f"Invalid database type: {db_type}")

        for key, value in config.items():
            self.set(f'databases.{db_type}.{key}', value)

        return self.save_config()

    def update_dag_schedule(self, dag_name: str, schedule: str) -> bool:
        """Update DAG schedule"""
        dag_key = 'training' if 'train' in dag_name else 'detection'
        self.set(f'airflow.dags.{dag_key}.schedule', schedule)
        return self.save_config()

    def update_training_params(self, params: Dict[str, Any]) -> bool:
        """Update training DAG parameters"""
        for key, value in params.items():
            self.set(f'airflow.dags.training.params.{key}', value)
        return self.save_config()

    def export_config(self, export_path: str) -> bool:
        """Export configuration to a file"""
        try:
            with open(export_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            print(f"Error exporting config: {e}")
            return False

    def import_config(self, import_path: str, merge: bool = False) -> bool:
        """Import configuration from a file"""
        try:
            with open(import_path, 'r') as f:
                imported_config = yaml.safe_load(f)

            if merge:
                # Merge imported config with existing
                self._deep_merge(self.config, imported_config)
            else:
                # Replace entire config
                self.config = imported_config

            return self.save_config()
        except Exception as e:
            print(f"Error importing config: {e}")
            return False

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate configuration for required fields and valid values"""
        errors = []

        # Check required sections
        required_sections = ['databases', 'airflow', 'models', 'ensemble']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")

        # Validate ensemble weights
        weights = self.get('ensemble.weights', {})
        if weights:
            total = sum(weights.values())
            if not (0.99 <= total <= 1.01):
                errors.append(f"Ensemble weights must sum to 1.0, got {total}")

        # Validate thresholds are in order
        thresholds = self.get('ensemble.thresholds', {})
        if thresholds:
            low = thresholds.get('low', 0)
            medium = thresholds.get('medium', 0)
            high = thresholds.get('high', 0)
            critical = thresholds.get('critical', 0)

            if not (low < medium < high < critical):
                errors.append("Risk thresholds must be in ascending order: low < medium < high < critical")

        # Validate database connections have required fields
        for db in ['clickhouse', 'postgres']:
            db_config = self.get(f'databases.{db}', {})
            required_fields = ['host', 'port', 'database', 'user', 'password']
            for field in required_fields:
                if field not in db_config:
                    errors.append(f"Missing required field '{field}' in databases.{db}")

        return len(errors) == 0, errors

    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration"""
        return self.get('dashboard', {})

    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return self.get('features', {})

    def get_docker_services(self) -> list[Dict[str, str]]:
        """Get list of Docker services to monitor"""
        return self.get('docker.services', [])
