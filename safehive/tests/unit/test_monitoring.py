"""
Unit tests for agent monitoring and health checking system.
"""

import asyncio
import json
import tempfile
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest

from safehive.agents.monitoring import (
    HealthStatus, AlertLevel, HealthCheck, AgentHealthReport, PerformanceMetrics,
    HealthCheckProvider, BasicHealthCheckProvider, PerformanceHealthCheckProvider,
    ResourceHealthCheckProvider, Alert, AlertManager, AgentMonitor,
    get_agent_monitor, create_agent_monitor
)
from safehive.models.agent_models import AgentStatus, AgentState


class TestHealthStatus:
    """Test health status enumeration."""
    
    def test_health_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.UNKNOWN.value == "unknown"
        assert HealthStatus.DOWN.value == "down"


class TestAlertLevel:
    """Test alert level enumeration."""
    
    def test_alert_level_values(self):
        """Test alert level enum values."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"


class TestHealthCheck:
    """Test health check data class."""
    
    def test_health_check_creation(self):
        """Test health check creation."""
        check = HealthCheck(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="Test message",
            duration_ms=100.5
        )
        
        assert check.name == "test_check"
        assert check.status == HealthStatus.HEALTHY
        assert check.message == "Test message"
        assert check.duration_ms == 100.5
        assert isinstance(check.timestamp, datetime)
        assert check.metadata == {}
    
    def test_health_check_with_metadata(self):
        """Test health check with metadata."""
        metadata = {"key": "value", "number": 42}
        check = HealthCheck(
            name="test_check",
            status=HealthStatus.WARNING,
            message="Test warning",
            metadata=metadata
        )
        
        assert check.metadata == metadata


class TestAgentHealthReport:
    """Test agent health report."""
    
    def test_health_report_creation(self):
        """Test health report creation."""
        report = AgentHealthReport(
            agent_id="test_agent",
            agent_type="user_twin",
            overall_status=HealthStatus.HEALTHY,
            uptime_seconds=3600.0
        )
        
        assert report.agent_id == "test_agent"
        assert report.agent_type == "user_twin"
        assert report.overall_status == HealthStatus.HEALTHY
        assert report.uptime_seconds == 3600.0
        assert report.checks == []
        assert isinstance(report.timestamp, datetime)
        assert report.metadata == {}
    
    def test_add_health_check(self):
        """Test adding health checks to report."""
        report = AgentHealthReport(
            agent_id="test_agent",
            agent_type="user_twin",
            overall_status=HealthStatus.HEALTHY
        )
        
        # Add healthy check
        healthy_check = HealthCheck("check1", HealthStatus.HEALTHY, "OK")
        report.add_check(healthy_check)
        assert len(report.checks) == 1
        assert report.overall_status == HealthStatus.HEALTHY
        
        # Add warning check
        warning_check = HealthCheck("check2", HealthStatus.WARNING, "Warning")
        report.add_check(warning_check)
        assert len(report.checks) == 2
        assert report.overall_status == HealthStatus.WARNING
        
        # Add critical check
        critical_check = HealthCheck("check3", HealthStatus.CRITICAL, "Critical")
        report.add_check(critical_check)
        assert len(report.checks) == 3
        assert report.overall_status == HealthStatus.CRITICAL
    
    def test_health_report_serialization(self):
        """Test health report serialization."""
        report = AgentHealthReport(
            agent_id="test_agent",
            agent_type="user_twin",
            overall_status=HealthStatus.HEALTHY,
            uptime_seconds=3600.0
        )
        
        check = HealthCheck("test_check", HealthStatus.HEALTHY, "OK", duration_ms=100.0)
        report.add_check(check)
        
        data = report.to_dict()
        assert data["agent_id"] == "test_agent"
        assert data["agent_type"] == "user_twin"
        assert data["overall_status"] == "healthy"
        assert data["uptime_seconds"] == 3600.0
        assert len(data["checks"]) == 1
        assert data["checks"][0]["name"] == "test_check"


class TestPerformanceMetrics:
    """Test performance metrics."""
    
    def test_performance_metrics_creation(self):
        """Test performance metrics creation."""
        metrics = PerformanceMetrics(agent_id="test_agent")
        
        assert metrics.agent_id == "test_agent"
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.average_response_time_ms == 0.0
        assert isinstance(metrics.timestamp, datetime)
        assert metrics.custom_metrics == {}
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = PerformanceMetrics(agent_id="test_agent")
        
        # No requests
        assert metrics.calculate_success_rate() == 0.0
        
        # All successful
        metrics.total_requests = 100
        metrics.successful_requests = 100
        assert metrics.calculate_success_rate() == 100.0
        
        # Mixed results
        metrics.successful_requests = 75
        assert metrics.calculate_success_rate() == 75.0
    
    def test_performance_metrics_serialization(self):
        """Test performance metrics serialization."""
        metrics = PerformanceMetrics(
            agent_id="test_agent",
            total_requests=100,
            successful_requests=90,
            failed_requests=10,
            average_response_time_ms=250.0
        )
        
        data = metrics.to_dict()
        assert data["agent_id"] == "test_agent"
        assert data["total_requests"] == 100
        assert data["successful_requests"] == 90
        assert data["failed_requests"] == 10
        assert data["success_rate_percent"] == 90.0
        assert data["average_response_time_ms"] == 250.0


class TestBasicHealthCheckProvider:
    """Test basic health check provider."""
    
    @pytest.mark.asyncio
    async def test_healthy_agent_check(self):
        """Test health check for healthy agent."""
        provider = BasicHealthCheckProvider()
        
        # Mock healthy agent
        agent = Mock()
        agent.status = AgentState.ACTIVE
        agent.get_agent_info = AsyncMock(return_value={"status": "healthy"})
        
        check = await provider.check_health(agent)
        
        assert check.name == "basic_health_check"
        assert check.status == HealthStatus.HEALTHY
        assert "healthy" in check.message.lower()
        assert check.duration_ms > 0
    
    @pytest.mark.asyncio
    async def test_error_agent_check(self):
        """Test health check for agent with error status."""
        provider = BasicHealthCheckProvider()
        
        # Mock agent with error status
        agent = Mock()
        agent.status = AgentState.ERROR
        
        check = await provider.check_health(agent)
        
        assert check.status == HealthStatus.CRITICAL
        assert "error" in check.message.lower()
    
    @pytest.mark.asyncio
    async def test_none_agent_check(self):
        """Test health check for None agent."""
        provider = BasicHealthCheckProvider()
        
        check = await provider.check_health(None)
        
        assert check.status == HealthStatus.DOWN
        assert "none" in check.message.lower()
    
    @pytest.mark.asyncio
    async def test_timeout_agent_check(self):
        """Test health check with timeout."""
        provider = BasicHealthCheckProvider(timeout_seconds=0.1)
        
        # Mock agent that takes too long to respond
        agent = Mock()
        agent.status = AgentState.ACTIVE
        
        async def slow_get_info():
            await asyncio.sleep(0.2)
            return {"status": "healthy"}
        
        agent.get_agent_info = slow_get_info
        
        check = await provider.check_health(agent)
        
        assert check.status == HealthStatus.WARNING
        assert "timeout" in check.message.lower()
    
    def test_get_check_name(self):
        """Test get check name."""
        provider = BasicHealthCheckProvider()
        assert provider.get_check_name() == "basic_health_check"


class TestPerformanceHealthCheckProvider:
    """Test performance health check provider."""
    
    @pytest.mark.asyncio
    async def test_healthy_performance_check(self):
        """Test performance check for healthy metrics."""
        provider = PerformanceHealthCheckProvider()
        
        # Mock agent with good metrics
        agent = Mock()
        agent.get_metrics = Mock(return_value={
            "total_requests": 100,
            "successful_requests": 98,
            "average_response_time_ms": 500.0
        })
        
        check = await provider.check_health(agent)
        
        assert check.status == HealthStatus.HEALTHY
        assert "acceptable" in check.message.lower()
    
    @pytest.mark.asyncio
    async def test_slow_response_time_check(self):
        """Test performance check with slow response time."""
        provider = PerformanceHealthCheckProvider(max_response_time_ms=1000.0)
        
        # Mock agent with slow response time
        agent = Mock()
        agent.get_metrics = Mock(return_value={
            "total_requests": 100,
            "successful_requests": 98,
            "average_response_time_ms": 1500.0
        })
        
        check = await provider.check_health(agent)
        
        assert check.status == HealthStatus.WARNING
        assert "response time" in check.message.lower()
    
    @pytest.mark.asyncio
    async def test_low_success_rate_check(self):
        """Test performance check with low success rate."""
        provider = PerformanceHealthCheckProvider(min_success_rate=95.0)
        
        # Mock agent with low success rate
        agent = Mock()
        agent.get_metrics = Mock(return_value={
            "total_requests": 100,
            "successful_requests": 90,
            "average_response_time_ms": 500.0
        })
        
        check = await provider.check_health(agent)
        
        assert check.status == HealthStatus.WARNING
        assert "success rate" in check.message.lower()
    
    @pytest.mark.asyncio
    async def test_no_metrics_agent(self):
        """Test performance check for agent without metrics."""
        provider = PerformanceHealthCheckProvider()
        
        # Mock agent without metrics
        agent = Mock()
        agent.get_metrics = Mock(return_value=None)
        
        check = await provider.check_health(agent)
        
        assert check.status == HealthStatus.WARNING
        assert "no metrics" in check.message.lower()
    
    def test_get_check_name(self):
        """Test get check name."""
        provider = PerformanceHealthCheckProvider()
        assert provider.get_check_name() == "performance_health_check"


class TestResourceHealthCheckProvider:
    """Test resource health check provider."""
    
    @pytest.mark.asyncio
    async def test_healthy_resource_check(self):
        """Test resource check with healthy usage."""
        provider = ResourceHealthCheckProvider(max_memory_mb=1000.0, max_cpu_percent=80.0)
        
        # Mock agent
        agent = Mock()
        
        with patch('psutil.Process') as mock_process:
            # Mock low resource usage
            mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
            mock_process.return_value.cpu_percent.return_value = 20.0
            
            check = await provider.check_health(agent)
            
            assert check.status == HealthStatus.HEALTHY
            assert "acceptable" in check.message.lower()
    
    @pytest.mark.asyncio
    async def test_high_memory_usage_check(self):
        """Test resource check with high memory usage."""
        provider = ResourceHealthCheckProvider(max_memory_mb=100.0, max_cpu_percent=80.0)
        
        # Mock agent
        agent = Mock()
        
        with patch('psutil.Process') as mock_process:
            # Mock high memory usage
            mock_process.return_value.memory_info.return_value.rss = 200 * 1024 * 1024  # 200MB
            mock_process.return_value.cpu_percent.return_value = 20.0
            
            check = await provider.check_health(agent)
            
            assert check.status == HealthStatus.WARNING
            assert "memory" in check.message.lower()
    
    @pytest.mark.asyncio
    async def test_high_cpu_usage_check(self):
        """Test resource check with high CPU usage."""
        provider = ResourceHealthCheckProvider(max_memory_mb=1000.0, max_cpu_percent=50.0)
        
        # Mock agent
        agent = Mock()
        
        with patch('psutil.Process') as mock_process:
            # Mock high CPU usage
            mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
            mock_process.return_value.cpu_percent.return_value = 70.0
            
            check = await provider.check_health(agent)
            
            assert check.status == HealthStatus.WARNING
            assert "cpu" in check.message.lower()
    
    def test_get_check_name(self):
        """Test get check name."""
        provider = ResourceHealthCheckProvider()
        assert provider.get_check_name() == "resource_health_check"


class TestAlert:
    """Test alert data class."""
    
    def test_alert_creation(self):
        """Test alert creation."""
        alert = Alert(
            id="test_alert",
            agent_id="test_agent",
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="Test message"
        )
        
        assert alert.id == "test_alert"
        assert alert.agent_id == "test_agent"
        assert alert.level == AlertLevel.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "Test message"
        assert not alert.resolved
        assert alert.resolved_at is None
        assert alert.metadata == {}
    
    def test_alert_resolution(self):
        """Test alert resolution."""
        alert = Alert(
            id="test_alert",
            agent_id="test_agent",
            level=AlertLevel.ERROR,
            title="Test Alert",
            message="Test message"
        )
        
        assert not alert.resolved
        assert alert.resolved_at is None
        
        alert.resolve()
        
        assert alert.resolved
        assert alert.resolved_at is not None
        assert isinstance(alert.resolved_at, datetime)
    
    def test_alert_serialization(self):
        """Test alert serialization."""
        alert = Alert(
            id="test_alert",
            agent_id="test_agent",
            level=AlertLevel.CRITICAL,
            title="Test Alert",
            message="Test message",
            metadata={"key": "value"}
        )
        
        data = alert.to_dict()
        assert data["id"] == "test_alert"
        assert data["agent_id"] == "test_agent"
        assert data["level"] == "critical"
        assert data["title"] == "Test Alert"
        assert data["message"] == "Test message"
        assert data["resolved"] is False
        assert data["resolved_at"] is None
        assert data["metadata"] == {"key": "value"}


class TestAlertManager:
    """Test alert manager."""
    
    def test_alert_manager_creation(self):
        """Test alert manager creation."""
        manager = AlertManager()
        assert manager._alerts == {}
        assert manager._alert_callbacks == []
    
    def test_add_remove_callback(self):
        """Test adding and removing callbacks."""
        manager = AlertManager()
        callback = Mock()
        
        manager.add_callback(callback)
        assert callback in manager._alert_callbacks
        
        manager.remove_callback(callback)
        assert callback not in manager._alert_callbacks
    
    def test_create_alert(self):
        """Test creating alerts."""
        manager = AlertManager()
        callback = Mock()
        manager.add_callback(callback)
        
        alert = manager.create_alert(
            agent_id="test_agent",
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="Test message",
            metadata={"key": "value"}
        )
        
        assert alert.agent_id == "test_agent"
        assert alert.level == AlertLevel.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "Test message"
        assert alert.metadata == {"key": "value"}
        assert alert.id in manager._alerts
        
        # Verify callback was called
        callback.assert_called_once_with(alert)
    
    def test_resolve_alert(self):
        """Test resolving alerts."""
        manager = AlertManager()
        
        alert = manager.create_alert(
            agent_id="test_agent",
            level=AlertLevel.ERROR,
            title="Test Alert",
            message="Test message"
        )
        
        assert not alert.resolved
        
        success = manager.resolve_alert(alert.id)
        assert success
        assert alert.resolved
        assert alert.resolved_at is not None
        
        # Try to resolve non-existent alert
        success = manager.resolve_alert("non_existent")
        assert not success
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        manager = AlertManager()
        
        # Create some alerts
        alert1 = manager.create_alert("agent1", AlertLevel.WARNING, "Alert 1", "Message 1")
        alert2 = manager.create_alert("agent2", AlertLevel.ERROR, "Alert 2", "Message 2")
        
        active_alerts = manager.get_active_alerts()
        assert len(active_alerts) == 2
        
        # Resolve one alert
        manager.resolve_alert(alert1.id)
        
        active_alerts = manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].id == alert2.id
        
        # Filter by agent
        active_alerts = manager.get_active_alerts("agent2")
        assert len(active_alerts) == 1
        assert active_alerts[0].id == alert2.id
    
    def test_get_alert_history(self):
        """Test getting alert history."""
        manager = AlertManager()
        
        # Create some alerts
        alert1 = manager.create_alert("agent1", AlertLevel.WARNING, "Alert 1", "Message 1")
        alert2 = manager.create_alert("agent2", AlertLevel.ERROR, "Alert 2", "Message 2")
        
        history = manager.get_alert_history()
        assert len(history) == 2
        
        # Test limit
        history = manager.get_alert_history(limit=1)
        assert len(history) == 1
        
        # Filter by agent
        history = manager.get_alert_history("agent1")
        assert len(history) == 1
        assert history[0].id == alert1.id


class TestAgentMonitor:
    """Test agent monitor."""
    
    def test_agent_monitor_creation(self):
        """Test agent monitor creation."""
        monitor = AgentMonitor(check_interval_seconds=60.0, max_history=500)
        
        assert monitor.check_interval_seconds == 60.0
        assert monitor.max_history == 500
        assert not monitor._monitoring_active
        assert monitor._agents == {}
        assert len(monitor._health_providers) == 3  # Basic, Performance, Resource
    
    def test_register_unregister_agent(self):
        """Test agent registration and unregistration."""
        monitor = AgentMonitor()
        
        # Mock agent
        agent = Mock()
        agent.agent_id = "test_agent"
        
        # Register agent
        monitor.register_agent(agent)
        assert "test_agent" in monitor._agents
        assert "test_agent" in monitor._agent_start_times
        assert "test_agent" in monitor._health_history
        assert "test_agent" in monitor._performance_history
        
        # Unregister agent
        monitor.unregister_agent("test_agent")
        assert "test_agent" not in monitor._agents
        assert "test_agent" not in monitor._agent_start_times
        # History should be kept
    
    def test_register_agent_without_id(self):
        """Test registering agent without agent_id attribute."""
        monitor = AgentMonitor()
        
        # Mock agent without agent_id
        agent = Mock(spec=[])
        
        with pytest.raises(ValueError, match="Agent must have an 'agent_id' attribute"):
            monitor.register_agent(agent)
    
    def test_add_remove_health_provider(self):
        """Test adding and removing health providers."""
        monitor = AgentMonitor()
        
        # Create custom provider
        custom_provider = Mock(spec=HealthCheckProvider)
        custom_provider.get_check_name = Mock(return_value="custom_check")
        
        initial_count = len(monitor._health_providers)
        
        # Add provider
        monitor.add_health_provider(custom_provider)
        assert len(monitor._health_providers) == initial_count + 1
        assert custom_provider in monitor._health_providers
        
        # Remove provider
        monitor.remove_health_provider(custom_provider)
        assert len(monitor._health_providers) == initial_count
        assert custom_provider not in monitor._health_providers
    
    @pytest.mark.asyncio
    async def test_check_agent_health(self):
        """Test checking agent health."""
        monitor = AgentMonitor()
        
        # Mock agent
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.agent_type = "user_twin"
        agent.status = AgentState.ACTIVE
        agent.get_agent_info = AsyncMock(return_value={"status": "healthy"})
        
        monitor.register_agent(agent)
        
        report = await monitor.check_agent_health("test_agent")
        
        assert report is not None
        assert report.agent_id == "test_agent"
        assert report.agent_type == "user_twin"
        assert len(report.checks) == 3  # Basic, Performance, Resource
        assert report.uptime_seconds >= 0
    
    @pytest.mark.asyncio
    async def test_check_nonexistent_agent_health(self):
        """Test checking health for non-existent agent."""
        monitor = AgentMonitor()
        
        report = await monitor.check_agent_health("nonexistent")
        assert report is None
    
    @pytest.mark.asyncio
    async def test_collect_agent_metrics(self):
        """Test collecting agent metrics."""
        monitor = AgentMonitor()
        
        # Mock agent with metrics
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.get_metrics = Mock(return_value={
            "total_requests": 100,
            "successful_requests": 95,
            "failed_requests": 5,
            "average_response_time_ms": 250.0,
            "min_response_time_ms": 100.0,
            "max_response_time_ms": 500.0
        })
        
        monitor.register_agent(agent)
        
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
            mock_process.return_value.cpu_percent.return_value = 25.0
            
            metrics = await monitor.collect_agent_metrics("test_agent")
            
            assert metrics is not None
            assert metrics.agent_id == "test_agent"
            assert metrics.total_requests == 100
            assert metrics.successful_requests == 95
            assert metrics.failed_requests == 5
            assert metrics.average_response_time_ms == 250.0
            assert metrics.memory_usage_mb == 100.0
            assert metrics.cpu_usage_percent == 25.0
    
    @pytest.mark.asyncio
    async def test_collect_nonexistent_agent_metrics(self):
        """Test collecting metrics for non-existent agent."""
        monitor = AgentMonitor()
        
        metrics = await monitor.collect_agent_metrics("nonexistent")
        assert metrics is None
    
    @pytest.mark.asyncio
    async def test_monitor_all_agents(self):
        """Test monitoring all agents."""
        monitor = AgentMonitor()
        
        # Mock agents
        agent1 = Mock()
        agent1.agent_id = "agent1"
        agent1.agent_type = "user_twin"
        agent1.status = AgentState.ACTIVE
        agent1.get_agent_info = AsyncMock(return_value={"status": "healthy"})
        agent1.get_metrics = Mock(return_value={"total_requests": 10})
        
        agent2 = Mock()
        agent2.agent_id = "agent2"
        agent2.agent_type = "orchestrator"
        agent2.status = AgentState.ACTIVE
        agent2.get_agent_info = AsyncMock(return_value={"status": "healthy"})
        agent2.get_metrics = Mock(return_value={"total_requests": 20})
        
        monitor.register_agent(agent1)
        monitor.register_agent(agent2)
        
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
            mock_process.return_value.cpu_percent.return_value = 25.0
            
            await monitor.monitor_all_agents()
            
            # Check that health reports were created
            assert len(monitor._health_history["agent1"]) == 1
            assert len(monitor._health_history["agent2"]) == 1
            assert len(monitor._performance_history["agent1"]) == 1
            assert len(monitor._performance_history["agent2"]) == 1
    
    def test_get_agent_health_report(self):
        """Test getting agent health report."""
        monitor = AgentMonitor()
        
        # Mock agent
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.agent_type = "user_twin"
        
        monitor.register_agent(agent)
        
        # No reports yet
        report = monitor.get_agent_health_report("test_agent")
        assert report is None
        
        # Add a mock report
        mock_report = AgentHealthReport("test_agent", "user_twin", HealthStatus.HEALTHY)
        monitor._health_history["test_agent"].append(mock_report)
        
        # Get latest report
        report = monitor.get_agent_health_report("test_agent", latest_only=True)
        assert report is not None
        assert report.agent_id == "test_agent"
        
        # Get all reports
        reports = monitor.get_agent_health_report("test_agent", latest_only=False)
        assert isinstance(reports, list)
        assert len(reports) == 1
    
    def test_get_agent_performance_metrics(self):
        """Test getting agent performance metrics."""
        monitor = AgentMonitor()
        
        # Mock agent
        agent = Mock()
        agent.agent_id = "test_agent"
        
        monitor.register_agent(agent)
        
        # No metrics yet
        metrics = monitor.get_agent_performance_metrics("test_agent")
        assert metrics is None
        
        # Add mock metrics
        mock_metrics = PerformanceMetrics("test_agent", total_requests=100)
        monitor._performance_history["test_agent"].append(mock_metrics)
        
        # Get latest metrics
        metrics = monitor.get_agent_performance_metrics("test_agent", latest_only=True)
        assert metrics is not None
        assert metrics.agent_id == "test_agent"
        assert metrics.total_requests == 100
        
        # Get all metrics
        metrics_list = monitor.get_agent_performance_metrics("test_agent", latest_only=False)
        assert isinstance(metrics_list, list)
        assert len(metrics_list) == 1
    
    def test_get_all_agents_status(self):
        """Test getting all agents status."""
        monitor = AgentMonitor()
        
        # Mock agent with health report and metrics
        agent = Mock()
        agent.agent_id = "test_agent"
        
        monitor.register_agent(agent)
        
        # Add mock data using thread-safe approach
        with monitor._lock:
            mock_report = AgentHealthReport("test_agent", "user_twin", HealthStatus.HEALTHY, uptime_seconds=3600.0)
            monitor._health_history["test_agent"].append(mock_report)
            
            mock_metrics = PerformanceMetrics("test_agent", total_requests=100, successful_requests=95)
            monitor._performance_history["test_agent"].append(mock_metrics)
        
        status = monitor.get_all_agents_status()
        
        assert "test_agent" in status
        assert status["test_agent"]["health_status"] == "healthy"
        assert status["test_agent"]["uptime_seconds"] == 3600.0
        assert status["test_agent"]["total_requests"] == 100
        assert status["test_agent"]["success_rate_percent"] == 95.0
        assert status["test_agent"]["active_alerts"] == 0
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        monitor = AgentMonitor()
        
        # Create some alerts
        alert1 = monitor._alert_manager.create_alert("agent1", AlertLevel.WARNING, "Alert 1", "Message 1")
        alert2 = monitor._alert_manager.create_alert("agent2", AlertLevel.ERROR, "Alert 2", "Message 2")
        
        active_alerts = monitor.get_active_alerts()
        assert len(active_alerts) == 2
        
        # Filter by agent
        active_alerts = monitor.get_active_alerts("agent1")
        assert len(active_alerts) == 1
        assert active_alerts[0].id == alert1.id
    
    def test_resolve_alert(self):
        """Test resolving alerts."""
        monitor = AgentMonitor()
        
        alert = monitor._alert_manager.create_alert("test_agent", AlertLevel.WARNING, "Test Alert", "Message")
        
        success = monitor.resolve_alert(alert.id)
        assert success
        
        # Verify alert is resolved
        active_alerts = monitor.get_active_alerts("test_agent")
        assert len(active_alerts) == 0
    
    def test_export_monitoring_data(self):
        """Test exporting monitoring data."""
        monitor = AgentMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add some mock data
            mock_report = AgentHealthReport("test_agent", "user_twin", HealthStatus.HEALTHY)
            monitor._health_history["test_agent"] = [mock_report]
            
            mock_metrics = PerformanceMetrics("test_agent", total_requests=100)
            monitor._performance_history["test_agent"] = [mock_metrics]
            
            # Create an alert
            monitor._alert_manager.create_alert("test_agent", AlertLevel.WARNING, "Test Alert", "Message")
            
            # Export data
            monitor.export_monitoring_data(temp_dir)
            
            # Check files were created
            temp_path = Path(temp_dir)
            health_file = next(temp_path.glob("health_reports_*.json"))
            metrics_file = next(temp_path.glob("performance_metrics_*.json"))
            alerts_file = next(temp_path.glob("alerts_*.json"))
            
            assert health_file.exists()
            assert metrics_file.exists()
            assert alerts_file.exists()
            
            # Verify content
            with open(health_file) as f:
                health_data = json.load(f)
                assert "test_agent" in health_data
            
            with open(metrics_file) as f:
                metrics_data = json.load(f)
                assert "test_agent" in metrics_data
            
            with open(alerts_file) as f:
                alerts_data = json.load(f)
                assert "active_alerts" in alerts_data
                assert "alert_history" in alerts_data
    
    def test_cleanup_old_data(self):
        """Test cleaning up old monitoring data."""
        monitor = AgentMonitor()
        
        # Add old data
        old_time = datetime.now() - timedelta(hours=25)
        
        old_report = AgentHealthReport("test_agent", "user_twin", HealthStatus.HEALTHY)
        old_report.timestamp = old_time
        monitor._health_history["test_agent"] = [old_report]
        
        old_metrics = PerformanceMetrics("test_agent")
        old_metrics.timestamp = old_time
        monitor._performance_history["test_agent"] = [old_metrics]
        
        # Add recent data
        recent_report = AgentHealthReport("test_agent", "user_twin", HealthStatus.HEALTHY)
        monitor._health_history["test_agent"].append(recent_report)
        
        recent_metrics = PerformanceMetrics("test_agent")
        monitor._performance_history["test_agent"].append(recent_metrics)
        
        # Cleanup data older than 24 hours
        monitor.cleanup_old_data(max_age_hours=24)
        
        # Check that only recent data remains
        assert len(monitor._health_history["test_agent"]) == 1
        assert len(monitor._performance_history["test_agent"]) == 1
    
    def test_monitoring_start_stop(self):
        """Test starting and stopping monitoring."""
        monitor = AgentMonitor(check_interval_seconds=0.1)  # Short interval for testing
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor._monitoring_active
        
        # Wait a bit
        time.sleep(0.2)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor._monitoring_active


class TestGlobalMonitorFunctions:
    """Test global monitor functions."""
    
    def test_get_agent_monitor_singleton(self):
        """Test getting global agent monitor singleton."""
        monitor1 = get_agent_monitor()
        monitor2 = get_agent_monitor()
        
        assert monitor1 is monitor2
        assert isinstance(monitor1, AgentMonitor)
    
    def test_create_agent_monitor(self):
        """Test creating new agent monitor instance."""
        monitor = create_agent_monitor(check_interval_seconds=120.0, max_history=2000)
        
        assert isinstance(monitor, AgentMonitor)
        assert monitor.check_interval_seconds == 120.0
        assert monitor.max_history == 2000
        
        # Should be different from global instance
        global_monitor = get_agent_monitor()
        assert monitor is not global_monitor


class TestMonitoringIntegration:
    """Integration tests for monitoring system."""
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        monitor = AgentMonitor()
        
        # Mock agent
        agent = Mock()
        agent.agent_id = "integration_agent"
        agent.agent_type = "user_twin"
        agent.status = AgentState.ACTIVE
        agent.get_agent_info = AsyncMock(return_value={"status": "healthy"})
        agent.get_metrics = Mock(return_value={
            "total_requests": 50,
            "successful_requests": 48,
            "failed_requests": 2,
            "average_response_time_ms": 300.0
        })
        
        # Register agent
        monitor.register_agent(agent)
        
        # Add alert callback
        alert_callback = Mock()
        monitor.add_alert_callback(alert_callback)
        
        # Run health check
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 50 * 1024 * 1024
            mock_process.return_value.cpu_percent.return_value = 15.0
            
            health_report = await monitor.check_agent_health("integration_agent")
            assert health_report is not None
            assert health_report.overall_status == HealthStatus.HEALTHY
        
        # Collect metrics
        metrics = await monitor.collect_agent_metrics("integration_agent")
        assert metrics is not None
        assert metrics.total_requests == 50
        assert metrics.successful_requests == 48
        
        # Get status
        status = monitor.get_all_agents_status()
        assert "integration_agent" in status
        assert status["integration_agent"]["health_status"] == "healthy"
        assert status["integration_agent"]["total_requests"] == 50
        
        # Create an alert
        alert = monitor._alert_manager.create_alert(
            "integration_agent", 
            AlertLevel.WARNING, 
            "Test Alert", 
            "Test message"
        )
        
        # Verify alert callback was called
        alert_callback.assert_called_once_with(alert)
        
        # Check active alerts
        active_alerts = monitor.get_active_alerts("integration_agent")
        assert len(active_alerts) == 1
        
        # Resolve alert
        success = monitor.resolve_alert(alert.id)
        assert success
        
        # Verify no active alerts
        active_alerts = monitor.get_active_alerts("integration_agent")
        assert len(active_alerts) == 0
    
    def test_error_handling_in_monitoring(self):
        """Test error handling in monitoring operations."""
        monitor = AgentMonitor()
        
        # Test with agent that raises exceptions
        agent = Mock()
        agent.agent_id = "error_agent"
        agent.agent_type = "user_twin"
        agent.status = AgentState.ACTIVE
        agent.get_agent_info = AsyncMock(side_effect=Exception("Test error"))
        agent.get_metrics = Mock(side_effect=Exception("Metrics error"))
        
        monitor.register_agent(agent)
        
        # Health check should handle errors gracefully
        asyncio.run(monitor.check_agent_health("error_agent"))
        
        # Metrics collection should handle errors gracefully
        asyncio.run(monitor.collect_agent_metrics("error_agent"))
        
        # Should not crash the monitoring system
        assert "error_agent" in monitor._agents
