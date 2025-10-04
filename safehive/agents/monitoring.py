"""
Agent Monitoring and Health Checking System

This module provides comprehensive monitoring and health checking capabilities for AI agents,
including status tracking, performance metrics, lifecycle monitoring, and alerting.
"""

import asyncio
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set, Union
import psutil
import yaml
from concurrent.futures import ThreadPoolExecutor, Future

from ..utils.logger import get_logger
from ..utils.metrics import record_metric, increment_counter, MetricType
from ..models.agent_models import AgentStatus, AgentState

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DOWN = "down"


class AlertLevel(Enum):
    """Alert level enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentHealthReport:
    """Complete health report for an agent."""
    agent_id: str
    agent_type: str
    overall_status: HealthStatus
    checks: List[HealthCheck] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_check(self, check: HealthCheck) -> None:
        """Add a health check to the report."""
        self.checks.append(check)
        self._update_overall_status()
    
    def _update_overall_status(self) -> None:
        """Update overall status based on individual checks."""
        if not self.checks:
            self.overall_status = HealthStatus.UNKNOWN
            return
        
        status_priority = {
            HealthStatus.CRITICAL: 4,
            HealthStatus.DOWN: 3,
            HealthStatus.WARNING: 2,
            HealthStatus.HEALTHY: 1,
            HealthStatus.UNKNOWN: 0
        }
        
        # Get the highest priority status
        max_priority = max(status_priority[check.status] for check in self.checks)
        self.overall_status = next(
            status for status, priority in status_priority.items() 
            if priority == max_priority
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "overall_status": self.overall_status.value,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "timestamp": check.timestamp.isoformat(),
                    "duration_ms": check.duration_ms,
                    "metadata": check.metadata
                }
                for check in self.checks
            ],
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "metadata": self.metadata
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent."""
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Response metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    
    # Memory metrics
    memory_usage_mb: float = 0.0
    memory_peak_mb: float = 0.0
    
    # CPU metrics
    cpu_usage_percent: float = 0.0
    cpu_peak_percent: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def calculate_success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": self.calculate_success_rate(),
            "average_response_time_ms": self.average_response_time_ms,
            "min_response_time_ms": self.min_response_time_ms,
            "max_response_time_ms": self.max_response_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "memory_peak_mb": self.memory_peak_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "cpu_peak_percent": self.cpu_peak_percent,
            "custom_metrics": self.custom_metrics
        }


class HealthCheckProvider(ABC):
    """Abstract base class for health check providers."""
    
    @abstractmethod
    async def check_health(self, agent: Any) -> HealthCheck:
        """Perform health check for an agent."""
        pass
    
    @abstractmethod
    def get_check_name(self) -> str:
        """Get the name of this health check."""
        pass


class BasicHealthCheckProvider(HealthCheckProvider):
    """Basic health check provider for common agent health checks."""
    
    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout_seconds = timeout_seconds
    
    async def check_health(self, agent: Any) -> HealthCheck:
        """Perform basic health check."""
        start_time = time.time()
        
        try:
            # Check if agent exists and is accessible
            if not agent:
                return HealthCheck(
                    name=self.get_check_name(),
                    status=HealthStatus.DOWN,
                    message="Agent is None or not accessible",
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Check agent status
            if hasattr(agent, 'status'):
                agent_status = agent.status
                if agent_status in [AgentState.ERROR, AgentState.STOPPED]:
                    return HealthCheck(
                        name=self.get_check_name(),
                        status=HealthStatus.CRITICAL,
                        message=f"Agent status is {agent_status.value}",
                        duration_ms=(time.time() - start_time) * 1000,
                        metadata={"agent_status": agent_status.value}
                    )
            
            # Check if agent responds to basic queries
            if hasattr(agent, 'get_agent_info'):
                try:
                    info = await asyncio.wait_for(
                        agent.get_agent_info(), 
                        timeout=self.timeout_seconds
                    )
                    if not info:
                        return HealthCheck(
                            name=self.get_check_name(),
                            status=HealthStatus.WARNING,
                            message="Agent info is empty",
                            duration_ms=(time.time() - start_time) * 1000
                        )
                except asyncio.TimeoutError:
                    return HealthCheck(
                        name=self.get_check_name(),
                        status=HealthStatus.WARNING,
                        message=f"Agent response timeout ({self.timeout_seconds}s)",
                        duration_ms=(time.time() - start_time) * 1000
                    )
            
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name=self.get_check_name(),
                status=HealthStatus.HEALTHY,
                message="Agent is healthy and responsive",
                duration_ms=duration_ms
            )
            
        except Exception as e:
            return HealthCheck(
                name=self.get_check_name(),
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )
    
    def get_check_name(self) -> str:
        """Get the name of this health check."""
        return "basic_health_check"


class PerformanceHealthCheckProvider(HealthCheckProvider):
    """Performance-based health check provider."""
    
    def __init__(self, max_response_time_ms: float = 1000.0, min_success_rate: float = 95.0):
        self.max_response_time_ms = max_response_time_ms
        self.min_success_rate = min_success_rate
    
    async def check_health(self, agent: Any) -> HealthCheck:
        """Check performance metrics."""
        start_time = time.time()
        
        try:
            if not agent or not hasattr(agent, 'get_metrics'):
                return HealthCheck(
                    name=self.get_check_name(),
                    status=HealthStatus.UNKNOWN,
                    message="Agent does not support metrics",
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            metrics = agent.get_metrics()
            if not metrics:
                return HealthCheck(
                    name=self.get_check_name(),
                    status=HealthStatus.WARNING,
                    message="No metrics available",
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Check response time
            avg_response_time = metrics.get("average_response_time_ms", 0)
            if avg_response_time > self.max_response_time_ms:
                return HealthCheck(
                    name=self.get_check_name(),
                    status=HealthStatus.WARNING,
                    message=f"Average response time ({avg_response_time:.1f}ms) exceeds threshold ({self.max_response_time_ms}ms)",
                    duration_ms=(time.time() - start_time) * 1000,
                    metadata={"avg_response_time": avg_response_time, "threshold": self.max_response_time_ms}
                )
            
            # Check success rate
            total_requests = metrics.get("total_requests", 0)
            if total_requests > 0:
                success_rate = (metrics.get("successful_requests", 0) / total_requests) * 100
                if success_rate < self.min_success_rate:
                    return HealthCheck(
                        name=self.get_check_name(),
                        status=HealthStatus.WARNING,
                        message=f"Success rate ({success_rate:.1f}%) below threshold ({self.min_success_rate}%)",
                        duration_ms=(time.time() - start_time) * 1000,
                        metadata={"success_rate": success_rate, "threshold": self.min_success_rate}
                    )
            
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name=self.get_check_name(),
                status=HealthStatus.HEALTHY,
                message="Performance metrics are within acceptable ranges",
                duration_ms=duration_ms,
                metadata=metrics
            )
            
        except Exception as e:
            return HealthCheck(
                name=self.get_check_name(),
                status=HealthStatus.CRITICAL,
                message=f"Performance check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )
    
    def get_check_name(self) -> str:
        """Get the name of this health check."""
        return "performance_health_check"


class ResourceHealthCheckProvider(HealthCheckProvider):
    """Resource-based health check provider."""
    
    def __init__(self, max_memory_mb: float = 1000.0, max_cpu_percent: float = 80.0):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
    
    async def check_health(self, agent: Any) -> HealthCheck:
        """Check resource usage."""
        start_time = time.time()
        
        try:
            # Get current process info
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            issues = []
            metadata = {
                "memory_usage_mb": memory_mb,
                "cpu_usage_percent": cpu_percent
            }
            
            # Check memory usage
            if memory_mb > self.max_memory_mb:
                issues.append(f"Memory usage ({memory_mb:.1f}MB) exceeds threshold ({self.max_memory_mb}MB)")
            
            # Check CPU usage
            if cpu_percent > self.max_cpu_percent:
                issues.append(f"CPU usage ({cpu_percent:.1f}%) exceeds threshold ({self.max_cpu_percent}%)")
            
            duration_ms = (time.time() - start_time) * 1000
            
            if issues:
                return HealthCheck(
                    name=self.get_check_name(),
                    status=HealthStatus.WARNING,
                    message="; ".join(issues),
                    duration_ms=duration_ms,
                    metadata=metadata
                )
            else:
                return HealthCheck(
                    name=self.get_check_name(),
                    status=HealthStatus.HEALTHY,
                    message="Resource usage is within acceptable ranges",
                    duration_ms=duration_ms,
                    metadata=metadata
                )
                
        except Exception as e:
            return HealthCheck(
                name=self.get_check_name(),
                status=HealthStatus.CRITICAL,
                message=f"Resource check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )
    
    def get_check_name(self) -> str:
        """Get the name of this health check."""
        return "resource_health_check"


@dataclass
class Alert:
    """Alert notification."""
    id: str
    agent_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata
        }


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        self._alerts: Dict[str, Alert] = {}
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
    
    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add alert callback."""
        with self._lock:
            self._alert_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Alert], None]) -> None:
        """Remove alert callback."""
        with self._lock:
            if callback in self._alert_callbacks:
                self._alert_callbacks.remove(callback)
    
    def create_alert(self, agent_id: str, level: AlertLevel, title: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a new alert."""
        alert_id = f"{agent_id}_{int(time.time())}"
        alert = Alert(
            id=alert_id,
            agent_id=agent_id,
            level=level,
            title=title,
            message=message,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._alerts[alert_id] = alert
        
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"Alert created: {alert_id} - {title}")
        record_metric("alert.created", 1, MetricType.COUNTER, {"level": level.value, "agent_id": agent_id})
        
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self._alerts:
                self._alerts[alert_id].resolve()
                record_metric("alert.resolved", 1, MetricType.COUNTER, {"alert_id": alert_id})
                return True
        return False
    
    def get_active_alerts(self, agent_id: Optional[str] = None) -> List[Alert]:
        """Get active (unresolved) alerts."""
        with self._lock:
            alerts = [alert for alert in self._alerts.values() if not alert.resolved]
            if agent_id:
                alerts = [alert for alert in alerts if alert.agent_id == agent_id]
            return alerts
    
    def get_alert_history(self, agent_id: Optional[str] = None, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            alerts = list(self._alerts.values())
            if agent_id:
                alerts = [alert for alert in alerts if alert.agent_id == agent_id]
            
            # Sort by timestamp descending and limit
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            return alerts[:limit]


class AgentMonitor:
    """Main agent monitoring system."""
    
    def __init__(self, check_interval_seconds: float = 30.0, max_history: int = 1000):
        self.check_interval_seconds = check_interval_seconds
        self.max_history = max_history
        
        # Agent tracking
        self._agents: Dict[str, Any] = {}
        self._agent_start_times: Dict[str, datetime] = {}
        
        # Health check providers
        self._health_providers: List[HealthCheckProvider] = [
            BasicHealthCheckProvider(),
            PerformanceHealthCheckProvider(),
            ResourceHealthCheckProvider()
        ]
        
        # Monitoring data
        self._health_history: Dict[str, List[AgentHealthReport]] = {}
        self._performance_history: Dict[str, List[PerformanceMetrics]] = {}
        
        # Alert management
        self._alert_manager = AlertManager()
        
        # Monitoring control
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Executor for async operations
        self._executor = ThreadPoolExecutor(max_workers=10)
    
    def register_agent(self, agent: Any) -> None:
        """Register an agent for monitoring."""
        if not hasattr(agent, 'agent_id'):
            raise ValueError("Agent must have an 'agent_id' attribute")
        
        agent_id = agent.agent_id
        with self._lock:
            self._agents[agent_id] = agent
            self._agent_start_times[agent_id] = datetime.now()
            self._health_history[agent_id] = []
            self._performance_history[agent_id] = []
        
        logger.info(f"Registered agent for monitoring: {agent_id}")
        record_metric("agent.registered", 1, MetricType.COUNTER, {"agent_id": agent_id})
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from monitoring."""
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                del self._agent_start_times[agent_id]
                # Keep history for analysis
                logger.info(f"Unregistered agent from monitoring: {agent_id}")
                record_metric("agent.unregistered", 1, MetricType.COUNTER, {"agent_id": agent_id})
    
    def add_health_provider(self, provider: HealthCheckProvider) -> None:
        """Add a custom health check provider."""
        with self._lock:
            self._health_providers.append(provider)
    
    def remove_health_provider(self, provider: HealthCheckProvider) -> None:
        """Remove a health check provider."""
        with self._lock:
            if provider in self._health_providers:
                self._health_providers.remove(provider)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add alert callback."""
        self._alert_manager.add_callback(callback)
    
    async def check_agent_health(self, agent_id: str) -> Optional[AgentHealthReport]:
        """Check health for a specific agent."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return None
        
        # Create health report
        agent_type = getattr(agent, 'agent_type', 'unknown')
        if hasattr(agent_type, 'value'):
            agent_type = agent_type.value
        
        report = AgentHealthReport(
            agent_id=agent_id,
            agent_type=agent_type,
            overall_status=HealthStatus.UNKNOWN
        )
        
        # Calculate uptime
        start_time = self._agent_start_times.get(agent_id)
        if start_time:
            report.uptime_seconds = (datetime.now() - start_time).total_seconds()
        
        # Run health checks
        for provider in self._health_providers:
            try:
                check = await provider.check_health(agent)
                report.add_check(check)
            except Exception as e:
                logger.error(f"Health check failed for {agent_id}: {e}")
                error_check = HealthCheck(
                    name=provider.get_check_name(),
                    status=HealthStatus.CRITICAL,
                    message=f"Health check error: {str(e)}",
                    metadata={"error": str(e)}
                )
                report.add_check(error_check)
        
        # Store report
        with self._lock:
            if agent_id in self._health_history:
                self._health_history[agent_id].append(report)
                # Keep only recent history
                if len(self._health_history[agent_id]) > self.max_history:
                    self._health_history[agent_id] = self._health_history[agent_id][-self.max_history:]
        
        # Create alerts for critical issues
        if report.overall_status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
            self._alert_manager.create_alert(
                agent_id=agent_id,
                level=AlertLevel.CRITICAL,
                title=f"Agent {agent_id} health critical",
                message=f"Agent health status: {report.overall_status.value}",
                metadata={"health_report": report.to_dict()}
            )
        elif report.overall_status == HealthStatus.WARNING:
            self._alert_manager.create_alert(
                agent_id=agent_id,
                level=AlertLevel.WARNING,
                title=f"Agent {agent_id} health warning",
                message=f"Agent health status: {report.overall_status.value}",
                metadata={"health_report": report.to_dict()}
            )
        
        record_metric("health.check.completed", 1, MetricType.COUNTER, {"agent_id": agent_id, "status": report.overall_status.value})
        return report
    
    async def collect_agent_metrics(self, agent_id: str) -> Optional[PerformanceMetrics]:
        """Collect performance metrics for a specific agent."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return None
        
        metrics = PerformanceMetrics(agent_id=agent_id)
        
        try:
            # Get agent metrics if available
            if hasattr(agent, 'get_metrics'):
                agent_metrics = agent.get_metrics()
                if agent_metrics:
                    metrics.total_requests = agent_metrics.get("total_requests", 0)
                    metrics.successful_requests = agent_metrics.get("successful_requests", 0)
                    metrics.failed_requests = agent_metrics.get("failed_requests", 0)
                    metrics.average_response_time_ms = agent_metrics.get("average_response_time_ms", 0.0)
                    metrics.min_response_time_ms = agent_metrics.get("min_response_time_ms", 0.0)
                    metrics.max_response_time_ms = agent_metrics.get("max_response_time_ms", 0.0)
                    metrics.custom_metrics = agent_metrics.get("custom_metrics", {})
            
            # Get system metrics
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
            metrics.memory_peak_mb = memory_info.rss / 1024 / 1024  # Simplified for now
            metrics.cpu_usage_percent = process.cpu_percent()
            metrics.cpu_peak_percent = process.cpu_percent()  # Simplified for now
            
        except Exception as e:
            logger.error(f"Failed to collect metrics for {agent_id}: {e}")
            return None
        
        # Store metrics
        with self._lock:
            if agent_id in self._performance_history:
                self._performance_history[agent_id].append(metrics)
                # Keep only recent history
                if len(self._performance_history[agent_id]) > self.max_history:
                    self._performance_history[agent_id] = self._performance_history[agent_id][-self.max_history:]
        
        record_metric("metrics.collected", 1, MetricType.COUNTER, {"agent_id": agent_id})
        return metrics
    
    async def monitor_all_agents(self) -> None:
        """Monitor all registered agents."""
        with self._lock:
            agent_ids = list(self._agents.keys())
        
        # Run health checks and metrics collection in parallel
        tasks = []
        for agent_id in agent_ids:
            tasks.extend([
                self.check_agent_health(agent_id),
                self.collect_agent_metrics(agent_id)
            ])
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def start_monitoring(self) -> None:
        """Start the monitoring loop."""
        if self._monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Agent monitoring started")
        record_metric("monitoring.started", 1, MetricType.COUNTER)
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        if not self._monitoring_active:
            logger.warning("Monitoring is not active")
            return
        
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        logger.info("Agent monitoring stopped")
        record_metric("monitoring.stopped", 1, MetricType.COUNTER)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Run async monitoring
                asyncio.run(self.monitor_all_agents())
                
                # Sleep for the specified interval
                time.sleep(self.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)  # Short sleep on error
    
    def get_agent_health_report(self, agent_id: str, latest_only: bool = True) -> Optional[Union[AgentHealthReport, List[AgentHealthReport]]]:
        """Get health report for an agent."""
        with self._lock:
            if agent_id not in self._health_history:
                return None
            
            reports = self._health_history[agent_id]
            if not reports:
                return None
            
            if latest_only:
                return reports[-1]
            return reports
    
    def get_agent_performance_metrics(self, agent_id: str, latest_only: bool = True) -> Optional[Union[PerformanceMetrics, List[PerformanceMetrics]]]:
        """Get performance metrics for an agent."""
        with self._lock:
            if agent_id not in self._performance_history:
                return None
            
            metrics = self._performance_history[agent_id]
            if not metrics:
                return None
            
            if latest_only:
                return metrics[-1]
            return metrics
    
    def get_all_agents_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status summary for all agents."""
        with self._lock:
            status = {}
            for agent_id in self._agents.keys():
                health_report = self.get_agent_health_report(agent_id, latest_only=True)
                performance_metrics = self.get_agent_performance_metrics(agent_id, latest_only=True)
                
                status[agent_id] = {
                    "health_status": health_report.overall_status.value if health_report else "unknown",
                    "uptime_seconds": health_report.uptime_seconds if health_report else 0.0,
                    "total_requests": performance_metrics.total_requests if performance_metrics else 0,
                    "success_rate_percent": performance_metrics.calculate_success_rate() if performance_metrics else 0.0,
                    "active_alerts": len(self._alert_manager.get_active_alerts(agent_id))
                }
            
            return status
    
    def get_active_alerts(self, agent_id: Optional[str] = None) -> List[Alert]:
        """Get active alerts."""
        return self._alert_manager.get_active_alerts(agent_id)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        return self._alert_manager.resolve_alert(alert_id)
    
    def export_monitoring_data(self, output_dir: str) -> None:
        """Export monitoring data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export health reports
        health_file = output_path / f"health_reports_{timestamp}.json"
        with open(health_file, 'w') as f:
            json.dump({
                agent_id: [report.to_dict() for report in reports]
                for agent_id, reports in self._health_history.items()
            }, f, indent=2)
        
        # Export performance metrics
        metrics_file = output_path / f"performance_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                agent_id: [metrics.to_dict() for metrics in metrics_list]
                for agent_id, metrics_list in self._performance_history.items()
            }, f, indent=2)
        
        # Export alerts
        alerts_file = output_path / f"alerts_{timestamp}.json"
        with open(alerts_file, 'w') as f:
            json.dump({
                "active_alerts": [alert.to_dict() for alert in self._alert_manager.get_active_alerts()],
                "alert_history": [alert.to_dict() for alert in self._alert_manager.get_alert_history()]
            }, f, indent=2)
        
        logger.info(f"Monitoring data exported to {output_path}")
    
    def cleanup_old_data(self, max_age_hours: int = 24) -> None:
        """Clean up old monitoring data."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._lock:
            # Clean health history
            for agent_id in self._health_history:
                self._health_history[agent_id] = [
                    report for report in self._health_history[agent_id]
                    if report.timestamp > cutoff_time
                ]
            
            # Clean performance history
            for agent_id in self._performance_history:
                self._performance_history[agent_id] = [
                    metrics for metrics in self._performance_history[agent_id]
                    if metrics.timestamp > cutoff_time
                ]
        
        logger.info(f"Cleaned up monitoring data older than {max_age_hours} hours")


# Global monitoring instance
_global_monitor: Optional[AgentMonitor] = None
_monitor_lock = threading.Lock()


def get_agent_monitor() -> AgentMonitor:
    """Get the global agent monitor instance."""
    global _global_monitor
    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = AgentMonitor()
        return _global_monitor


def create_agent_monitor(check_interval_seconds: float = 30.0, max_history: int = 1000) -> AgentMonitor:
    """Create a new agent monitor instance."""
    return AgentMonitor(check_interval_seconds, max_history)
