"""
Docker Manager for AnomalyGuard Dashboard
Handles Docker container monitoring and management
"""

import subprocess
import json
from typing import Dict, Any, List, Optional
from datetime import datetime


class DockerManager:
    """Manages Docker containers and services"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.services = config_manager.get_docker_services()

    def _run_command(self, command: List[str]) -> tuple[bool, str]:
        """Run a shell command and return result"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def is_docker_available(self) -> bool:
        """Check if Docker is available"""
        success, _ = self._run_command(['docker', '--version'])
        return success

    def is_compose_available(self) -> bool:
        """Check if Docker Compose is available"""
        success, _ = self._run_command(['docker', 'compose', 'version'])
        if not success:
            # Try older docker-compose command
            success, _ = self._run_command(['docker-compose', '--version'])
        return success

    def get_container_status(self, container_name: str) -> Dict[str, Any]:
        """Get status of a specific container"""
        success, output = self._run_command([
            'docker', 'ps', '-a',
            '--filter', f'name={container_name}',
            '--format', '{{json .}}'
        ])

        if not success or not output.strip():
            return {
                'name': container_name,
                'status': 'not_found',
                'state': 'unknown',
                'error': 'Container not found'
            }

        try:
            container_info = json.loads(output.strip().split('\n')[0])
            status = container_info.get('Status', 'unknown')
            state = container_info.get('State', 'unknown')

            return {
                'name': container_name,
                'id': container_info.get('ID', ''),
                'image': container_info.get('Image', ''),
                'status': status,
                'state': state,
                'created': container_info.get('CreatedAt', ''),
                'ports': container_info.get('Ports', ''),
                'is_running': state.lower() == 'running'
            }
        except json.JSONDecodeError:
            return {
                'name': container_name,
                'status': 'error',
                'state': 'unknown',
                'error': 'Failed to parse container info'
            }

    def get_all_services_status(self) -> List[Dict[str, Any]]:
        """Get status of all configured services"""
        statuses = []

        for service in self.services:
            container_name = service.get('container')
            status = self.get_container_status(container_name)
            status['service_name'] = service.get('name')
            status['health_check'] = service.get('health_check', '')
            statuses.append(status)

        return statuses

    def get_container_logs(
        self,
        container_name: str,
        tail: int = 100,
        since: Optional[str] = None
    ) -> tuple[bool, str]:
        """Get logs from a container"""
        command = ['docker', 'logs', container_name, '--tail', str(tail)]

        if since:
            command.extend(['--since', since])

        success, output = self._run_command(command)
        return success, output

    def get_container_stats(self, container_name: str) -> Dict[str, Any]:
        """Get resource usage stats for a container"""
        success, output = self._run_command([
            'docker', 'stats', container_name,
            '--no-stream', '--format', '{{json .}}'
        ])

        if not success or not output.strip():
            return {
                'name': container_name,
                'error': 'Failed to get stats'
            }

        try:
            stats = json.loads(output.strip())
            return {
                'name': stats.get('Name', container_name),
                'cpu_percent': stats.get('CPUPerc', '0%'),
                'memory_usage': stats.get('MemUsage', '0B / 0B'),
                'memory_percent': stats.get('MemPerc', '0%'),
                'network_io': stats.get('NetIO', '0B / 0B'),
                'block_io': stats.get('BlockIO', '0B / 0B'),
                'pids': stats.get('PIDs', '0')
            }
        except json.JSONDecodeError:
            return {
                'name': container_name,
                'error': 'Failed to parse stats'
            }

    def restart_container(self, container_name: str) -> tuple[bool, str]:
        """Restart a container"""
        success, output = self._run_command(['docker', 'restart', container_name])

        if success:
            return True, f"Container {container_name} restarted successfully"
        return False, output

    def stop_container(self, container_name: str) -> tuple[bool, str]:
        """Stop a container"""
        success, output = self._run_command(['docker', 'stop', container_name])

        if success:
            return True, f"Container {container_name} stopped successfully"
        return False, output

    def start_container(self, container_name: str) -> tuple[bool, str]:
        """Start a container"""
        success, output = self._run_command(['docker', 'start', container_name])

        if success:
            return True, f"Container {container_name} started successfully"
        return False, output

    def health_check_container(
        self,
        container_name: str,
        health_command: str
    ) -> tuple[bool, str]:
        """Run health check command inside container"""
        success, output = self._run_command([
            'docker', 'exec', container_name, 'sh', '-c', health_command
        ])

        return success, output

    def get_compose_status(self) -> Dict[str, Any]:
        """Get Docker Compose project status"""
        # Try to get status using docker compose ps
        success, output = self._run_command([
            'docker', 'compose', 'ps', '--format', 'json'
        ])

        if not success:
            # Try older docker-compose
            success, output = self._run_command([
                'docker-compose', 'ps', '--format', 'json'
            ])

        if not success:
            return {
                'error': 'Failed to get compose status',
                'services': []
            }

        try:
            services = []
            for line in output.strip().split('\n'):
                if line:
                    service = json.loads(line)
                    services.append({
                        'name': service.get('Service', ''),
                        'state': service.get('State', ''),
                        'status': service.get('Status', ''),
                        'ports': service.get('Publishers', [])
                    })

            return {
                'services': services,
                'total': len(services),
                'running': sum(1 for s in services if s['state'] == 'running')
            }
        except json.JSONDecodeError:
            return {
                'error': 'Failed to parse compose status',
                'services': []
            }

    def compose_up(self, detached: bool = True) -> tuple[bool, str]:
        """Start Docker Compose services"""
        command = ['docker', 'compose', 'up']
        if detached:
            command.append('-d')

        success, output = self._run_command(command)

        if not success:
            # Try older docker-compose
            command = ['docker-compose', 'up']
            if detached:
                command.append('-d')
            success, output = self._run_command(command)

        if success:
            return True, "Services started successfully"
        return False, output

    def compose_down(self, volumes: bool = False) -> tuple[bool, str]:
        """Stop Docker Compose services"""
        command = ['docker', 'compose', 'down']
        if volumes:
            command.append('-v')

        success, output = self._run_command(command)

        if not success:
            # Try older docker-compose
            command = ['docker-compose', 'down']
            if volumes:
                command.append('-v')
            success, output = self._run_command(command)

        if success:
            return True, "Services stopped successfully"
        return False, output

    def compose_restart(self) -> tuple[bool, str]:
        """Restart Docker Compose services"""
        success, output = self._run_command(['docker', 'compose', 'restart'])

        if not success:
            # Try older docker-compose
            success, output = self._run_command(['docker-compose', 'restart'])

        if success:
            return True, "Services restarted successfully"
        return False, output

    def get_volumes(self) -> List[Dict[str, Any]]:
        """Get list of Docker volumes"""
        success, output = self._run_command([
            'docker', 'volume', 'ls', '--format', '{{json .}}'
        ])

        if not success:
            return []

        volumes = []
        for line in output.strip().split('\n'):
            if line:
                try:
                    volume = json.loads(line)
                    volumes.append({
                        'name': volume.get('Name', ''),
                        'driver': volume.get('Driver', ''),
                        'size': volume.get('Size', 'N/A')
                    })
                except json.JSONDecodeError:
                    continue

        return volumes

    def get_networks(self) -> List[Dict[str, Any]]:
        """Get list of Docker networks"""
        success, output = self._run_command([
            'docker', 'network', 'ls', '--format', '{{json .}}'
        ])

        if not success:
            return []

        networks = []
        for line in output.strip().split('\n'):
            if line:
                try:
                    network = json.loads(line)
                    networks.append({
                        'name': network.get('Name', ''),
                        'driver': network.get('Driver', ''),
                        'scope': network.get('Scope', '')
                    })
                except json.JSONDecodeError:
                    continue

        return networks

    def prune_system(self, volumes: bool = False) -> tuple[bool, str]:
        """Prune unused Docker resources"""
        command = ['docker', 'system', 'prune', '-f']
        if volumes:
            command.append('--volumes')

        success, output = self._run_command(command)

        if success:
            return True, f"System pruned successfully\n{output}"
        return False, output

    def get_docker_info(self) -> Dict[str, Any]:
        """Get Docker system information"""
        success, output = self._run_command(['docker', 'info', '--format', '{{json .}}'])

        if not success:
            return {'error': 'Failed to get Docker info'}

        try:
            info = json.loads(output)
            return {
                'containers': info.get('Containers', 0),
                'containers_running': info.get('ContainersRunning', 0),
                'containers_paused': info.get('ContainersPaused', 0),
                'containers_stopped': info.get('ContainersStopped', 0),
                'images': info.get('Images', 0),
                'driver': info.get('Driver', 'unknown'),
                'memory_total': info.get('MemTotal', 0),
                'cpus': info.get('NCPU', 0),
                'server_version': info.get('ServerVersion', 'unknown')
            }
        except json.JSONDecodeError:
            return {'error': 'Failed to parse Docker info'}

    def get_service_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary of all services"""
        services = self.get_all_services_status()

        healthy = sum(1 for s in services if s.get('is_running', False))
        total = len(services)

        return {
            'total_services': total,
            'healthy_services': healthy,
            'unhealthy_services': total - healthy,
            'health_percentage': (healthy / total * 100) if total > 0 else 0,
            'services': services
        }
