"""
Airflow Manager for AnomalyGuard Dashboard
Handles interactions with Airflow REST API for DAG management
"""

import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


class AirflowManager:
    """Manages Airflow DAG operations via REST API"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        config = config_manager.get_airflow_config()

        self.base_url = config.get('webserver_url', 'http://localhost:8080')
        self.username = config.get('username', 'admin')
        self.password = config.get('password', 'admin1234')
        self.session = requests.Session()
        self.session.auth = (self.username, self.password)
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> tuple[bool, Any]:
        """Make API request to Airflow"""
        url = f"{self.base_url}/api/v1/{endpoint}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return True, response.json() if response.text else {}
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_msg = e.response.json().get('detail', str(e))
                except:
                    error_msg = e.response.text or str(e)
            return False, error_msg

    # ===== Health Check =====

    def health_check(self) -> tuple[bool, Dict[str, Any]]:
        """Check Airflow health status"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return True, response.json()
        except Exception as e:
            return False, {"error": str(e)}

    def get_version(self) -> tuple[bool, str]:
        """Get Airflow version"""
        success, data = self._request('GET', 'version')
        if success:
            return True, data.get('version', 'Unknown')
        return False, str(data)

    # ===== DAG Operations =====

    def get_dags(self) -> tuple[bool, List[Dict[str, Any]]]:
        """Get list of all DAGs"""
        success, data = self._request('GET', 'dags')

        if success:
            return True, data.get('dags', [])
        return False, []

    def get_dag(self, dag_id: str) -> tuple[bool, Dict[str, Any]]:
        """Get information about a specific DAG"""
        return self._request('GET', f'dags/{dag_id}')

    def get_dag_details(self, dag_id: str) -> tuple[bool, Dict[str, Any]]:
        """Get detailed information about a DAG"""
        success, data = self._request('GET', f'dags/{dag_id}/details')
        return success, data

    def pause_dag(self, dag_id: str) -> tuple[bool, str]:
        """Pause a DAG"""
        success, data = self._request(
            'PATCH',
            f'dags/{dag_id}',
            json={'is_paused': True}
        )

        if success:
            return True, f"DAG {dag_id} paused successfully"
        return False, str(data)

    def unpause_dag(self, dag_id: str) -> tuple[bool, str]:
        """Unpause a DAG"""
        success, data = self._request(
            'PATCH',
            f'dags/{dag_id}',
            json={'is_paused': False}
        )

        if success:
            return True, f"DAG {dag_id} unpaused successfully"
        return False, str(data)

    def trigger_dag(
        self,
        dag_id: str,
        conf: Optional[Dict[str, Any]] = None,
        logical_date: Optional[str] = None
    ) -> tuple[bool, str]:
        """Trigger a DAG run"""
        payload = {}

        if conf:
            payload['conf'] = conf

        if logical_date:
            payload['logical_date'] = logical_date

        success, data = self._request(
            'POST',
            f'dags/{dag_id}/dagRuns',
            json=payload
        )

        if success:
            dag_run_id = data.get('dag_run_id', 'Unknown')
            return True, f"DAG triggered successfully. Run ID: {dag_run_id}"
        return False, str(data)

    def delete_dag(self, dag_id: str) -> tuple[bool, str]:
        """Delete a DAG and its metadata"""
        success, data = self._request('DELETE', f'dags/{dag_id}')

        if success:
            return True, f"DAG {dag_id} deleted successfully"
        return False, str(data)

    # ===== DAG Runs =====

    def get_dag_runs(
        self,
        dag_id: str,
        limit: int = 25,
        offset: int = 0,
        state: Optional[str] = None
    ) -> tuple[bool, List[Dict[str, Any]]]:
        """Get DAG runs"""
        params = {'limit': limit, 'offset': offset}

        if state:
            params['state'] = state

        success, data = self._request(
            'GET',
            f'dags/{dag_id}/dagRuns',
            params=params
        )

        if success:
            return True, data.get('dag_runs', [])
        return False, []

    def get_dag_run(
        self,
        dag_id: str,
        dag_run_id: str
    ) -> tuple[bool, Dict[str, Any]]:
        """Get specific DAG run"""
        return self._request('GET', f'dags/{dag_id}/dagRuns/{dag_run_id}')

    def clear_dag_run(
        self,
        dag_id: str,
        dag_run_id: str
    ) -> tuple[bool, str]:
        """Clear a DAG run (set tasks to None state for re-running)"""
        success, data = self._request(
            'POST',
            f'dags/{dag_id}/dagRuns/{dag_run_id}/clear',
            json={'dry_run': False}
        )

        if success:
            return True, "DAG run cleared successfully"
        return False, str(data)

    # ===== Task Instances =====

    def get_task_instances(
        self,
        dag_id: str,
        dag_run_id: str
    ) -> tuple[bool, List[Dict[str, Any]]]:
        """Get task instances for a DAG run"""
        success, data = self._request(
            'GET',
            f'dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances'
        )

        if success:
            return True, data.get('task_instances', [])
        return False, []

    def get_task_instance(
        self,
        dag_id: str,
        dag_run_id: str,
        task_id: str
    ) -> tuple[bool, Dict[str, Any]]:
        """Get specific task instance"""
        return self._request(
            'GET',
            f'dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}'
        )

    def get_task_logs(
        self,
        dag_id: str,
        dag_run_id: str,
        task_id: str,
        task_try_number: int = 1
    ) -> tuple[bool, str]:
        """Get logs for a task instance"""
        success, data = self._request(
            'GET',
            f'dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{task_try_number}'
        )

        if success:
            return True, data.get('content', '')
        return False, str(data)

    # ===== XCom =====

    def get_xcom_entries(
        self,
        dag_id: str,
        dag_run_id: str,
        task_id: Optional[str] = None
    ) -> tuple[bool, List[Dict[str, Any]]]:
        """Get XCom entries for a DAG run"""
        endpoint = f'dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances'

        if task_id:
            endpoint += f'/{task_id}'

        endpoint += '/xcomEntries'

        success, data = self._request('GET', endpoint)

        if success:
            return True, data.get('xcom_entries', [])
        return False, []

    # ===== Variables =====

    def get_variables(self) -> tuple[bool, List[Dict[str, Any]]]:
        """Get all Airflow variables"""
        success, data = self._request('GET', 'variables')

        if success:
            return True, data.get('variables', [])
        return False, []

    def get_variable(self, key: str) -> tuple[bool, str]:
        """Get specific variable value"""
        success, data = self._request('GET', f'variables/{key}')

        if success:
            return True, data.get('value', '')
        return False, str(data)

    def set_variable(
        self,
        key: str,
        value: str,
        description: Optional[str] = None
    ) -> tuple[bool, str]:
        """Set variable value"""
        payload = {'key': key, 'value': value}

        if description:
            payload['description'] = description

        success, data = self._request('POST', 'variables', json=payload)

        if success:
            return True, f"Variable {key} set successfully"
        return False, str(data)

    def delete_variable(self, key: str) -> tuple[bool, str]:
        """Delete a variable"""
        success, data = self._request('DELETE', f'variables/{key}')

        if success:
            return True, f"Variable {key} deleted successfully"
        return False, str(data)

    # ===== Connections =====

    def get_connections(self) -> tuple[bool, List[Dict[str, Any]]]:
        """Get all Airflow connections"""
        success, data = self._request('GET', 'connections')

        if success:
            return True, data.get('connections', [])
        return False, []

    def test_connection(self, connection_id: str) -> tuple[bool, str]:
        """Test a connection"""
        success, data = self._request('POST', f'connections/{connection_id}/test')

        if success:
            status = data.get('status', 'unknown')
            message = data.get('message', '')
            return status == 'success', message
        return False, str(data)

    # ===== Pools =====

    def get_pools(self) -> tuple[bool, List[Dict[str, Any]]]:
        """Get all pools"""
        success, data = self._request('GET', 'pools')

        if success:
            return True, data.get('pools', [])
        return False, []

    # ===== Statistics =====

    def get_dag_statistics(self, dag_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get statistics for DAGs"""
        stats = {
            'total_dags': 0,
            'active_dags': 0,
            'paused_dags': 0,
            'running_dag_runs': 0,
            'success_rate': 0.0
        }

        success, dags = self.get_dags()
        if not success:
            return stats

        if dag_ids:
            dags = [d for d in dags if d['dag_id'] in dag_ids]

        stats['total_dags'] = len(dags)
        stats['active_dags'] = sum(1 for d in dags if not d.get('is_paused', True))
        stats['paused_dags'] = sum(1 for d in dags if d.get('is_paused', True))

        # Get running dag runs
        total_runs = 0
        success_runs = 0
        running_runs = 0

        for dag in dags:
            dag_id = dag['dag_id']
            success, runs = self.get_dag_runs(dag_id, limit=10)

            if success:
                total_runs += len(runs)
                success_runs += sum(1 for r in runs if r.get('state') == 'success')
                running_runs += sum(1 for r in runs if r.get('state') == 'running')

        stats['running_dag_runs'] = running_runs
        stats['success_rate'] = (success_runs / total_runs * 100) if total_runs > 0 else 0.0

        return stats

    # ===== Custom DAG Operations =====

    def trigger_training_dag(
        self,
        training_days: Optional[int] = None,
        min_samples: Optional[int] = None
    ) -> tuple[bool, str]:
        """Trigger the training DAG with optional parameters"""
        conf = {}

        if training_days is not None:
            conf['training_days'] = training_days

        if min_samples is not None:
            conf['min_samples'] = min_samples

        return self.trigger_dag('train_ensemble_models', conf=conf)

    def trigger_detection_dag(
        self,
        lookback_hours: Optional[int] = None
    ) -> tuple[bool, str]:
        """Trigger the detection DAG with optional parameters"""
        conf = {}

        if lookback_hours is not None:
            conf['lookback_hours'] = lookback_hours

        return self.trigger_dag('ensemble_anomaly_detection', conf=conf)

    def get_training_dag_status(self) -> Dict[str, Any]:
        """Get current status of training DAG"""
        success, dag = self.get_dag('train_ensemble_models')

        if not success:
            return {'error': 'Failed to get DAG info'}

        success, runs = self.get_dag_runs('train_ensemble_models', limit=5)

        return {
            'is_paused': dag.get('is_paused', True),
            'last_parsed': dag.get('last_parsed_time'),
            'next_dagrun': dag.get('next_dagrun'),
            'recent_runs': runs if success else []
        }

    def get_detection_dag_status(self) -> Dict[str, Any]:
        """Get current status of detection DAG"""
        success, dag = self.get_dag('ensemble_anomaly_detection')

        if not success:
            return {'error': 'Failed to get DAG info'}

        success, runs = self.get_dag_runs('ensemble_anomaly_detection', limit=5)

        return {
            'is_paused': dag.get('is_paused', True),
            'last_parsed': dag.get('last_parsed_time'),
            'next_dagrun': dag.get('next_dagrun'),
            'recent_runs': runs if success else []
        }
