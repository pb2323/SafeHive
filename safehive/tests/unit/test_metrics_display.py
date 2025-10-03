"""
Unit tests for the Metrics Display Module

This module contains comprehensive tests for the MetricsDisplay class and
related functionality in the SafeHive metrics display system.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from safehive.ui.metrics_display import MetricsDisplay, display_metrics_summary, display_security_metrics, display_agent_metrics
from safehive.utils.metrics import get_metrics_collector, record_metric, MetricType


class TestMetricsDisplay:
    """Test cases for MetricsDisplay class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing metrics
        collector = get_metrics_collector()
        collector.clear_metrics()
        
        self.display = MetricsDisplay()

    def test_metrics_display_initialization(self):
        """Test MetricsDisplay initialization."""
        assert self.display.console is not None
        assert self.display.metrics_collector is not None

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_system_overview(self, mock_print):
        """Test displaying system overview."""
        # Add some test metrics
        record_metric("test.metric1", 100, MetricType.GAUGE)
        record_metric("test.metric2", 200, MetricType.COUNTER)
        
        self.display.display_system_overview()
        
        # Verify print was called
        assert mock_print.call_count > 0

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_metrics_table(self, mock_print):
        """Test displaying metrics table."""
        # Add some test metrics
        record_metric("test.metric1", 100, MetricType.GAUGE)
        record_metric("test.metric2", 200, MetricType.COUNTER)
        
        self.display.display_metrics_table()
        
        # Verify print was called
        assert mock_print.call_count > 0

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_metrics_table_with_type_filter(self, mock_print):
        """Test displaying metrics table with type filter."""
        # Add metrics of different types
        record_metric("test.gauge", 100, MetricType.GAUGE)
        record_metric("test.counter", 200, MetricType.COUNTER)
        
        self.display.display_metrics_table("gauge")
        
        # Verify print was called
        assert mock_print.call_count > 0

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_metrics_table_no_metrics(self, mock_print):
        """Test displaying metrics table with no metrics."""
        self.display.display_metrics_table()
        
        # Should print "No metrics available"
        mock_print.assert_called_with("[yellow]No metrics available[/yellow]")

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_counters(self, mock_print):
        """Test displaying counter metrics."""
        # Add some counters
        record_metric("test.counter1", 5, MetricType.COUNTER)
        record_metric("test.counter2", 15, MetricType.COUNTER)
        
        self.display.display_counters()
        
        # Verify print was called
        assert mock_print.call_count > 0

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_counters_no_counters(self, mock_print):
        """Test displaying counters with no counter metrics."""
        self.display.display_counters()
        
        # Should print "No counter metrics available"
        mock_print.assert_called_with("[yellow]No counter metrics available[/yellow]")

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_gauges(self, mock_print):
        """Test displaying gauge metrics."""
        # Add some gauges
        record_metric("test.gauge1", 25.5, MetricType.GAUGE)
        record_metric("test.gauge2", 75.0, MetricType.GAUGE)
        
        self.display.display_gauges()
        
        # Verify print was called
        assert mock_print.call_count > 0

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_gauges_no_gauges(self, mock_print):
        """Test displaying gauges with no gauge metrics."""
        self.display.display_gauges()
        
        # Should print "No gauge metrics available"
        mock_print.assert_called_with("[yellow]No gauge metrics available[/yellow]")

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_timers(self, mock_print):
        """Test displaying timer metrics."""
        # Add some timers
        record_metric("test.timer1", 1.5, MetricType.TIMER)
        record_metric("test.timer1", 2.0, MetricType.TIMER)
        record_metric("test.timer2", 0.5, MetricType.TIMER)
        
        self.display.display_timers()
        
        # Verify print was called
        assert mock_print.call_count > 0

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_timers_no_timers(self, mock_print):
        """Test displaying timers with no timer metrics."""
        self.display.display_timers()
        
        # Should print "No timer metrics available"
        mock_print.assert_called_with("[yellow]No timer metrics available[/yellow]")

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_recent_events(self, mock_print):
        """Test displaying recent events."""
        # Add some events
        record_metric("test.event1", "Event 1 occurred", MetricType.EVENT, metadata={"description": "Event 1 occurred"})
        record_metric("test.event2", "Event 2 occurred", MetricType.EVENT, metadata={"description": "Event 2 occurred"})
        
        self.display.display_recent_events()
        
        # Verify print was called
        assert mock_print.call_count > 0

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_recent_events_no_events(self, mock_print):
        """Test displaying recent events with no events."""
        self.display.display_recent_events()
        
        # Should print "No recent events available"
        mock_print.assert_called_with("[yellow]No recent events available[/yellow]")

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_security_metrics(self, mock_print):
        """Test displaying security metrics."""
        # Add some security-related metrics
        record_metric("security.attack_detected", 5, MetricType.COUNTER)
        record_metric("security.threat_level", 3, MetricType.GAUGE)
        record_metric("security.alert", "High threat detected", MetricType.EVENT, metadata={"description": "High threat detected"})
        
        self.display.display_security_metrics()
        
        # Verify print was called
        assert mock_print.call_count > 0

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_security_metrics_no_security_metrics(self, mock_print):
        """Test displaying security metrics with no security-related metrics."""
        # Add non-security metrics
        record_metric("system.cpu_usage", 75, MetricType.GAUGE)
        
        self.display.display_security_metrics()
        
        # Should print "No security metrics available"
        mock_print.assert_called_with("[yellow]No security metrics available[/yellow]")

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_agent_metrics(self, mock_print):
        """Test displaying agent metrics."""
        # Add some agent-related metrics
        record_metric("agent.orchestrator.response_time", 1.5, MetricType.TIMER)
        record_metric("agent.user_twin.conversations", 10, MetricType.COUNTER)
        record_metric("agent.vendor.memory_usage", 50.5, MetricType.GAUGE)
        
        self.display.display_agent_metrics()
        
        # Verify print was called
        assert mock_print.call_count > 0

    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_agent_metrics_no_agent_metrics(self, mock_print):
        """Test displaying agent metrics with no agent-related metrics."""
        # Add non-agent metrics
        record_metric("system.cpu_usage", 75, MetricType.GAUGE)
        
        self.display.display_agent_metrics()
        
        # Should print "No agent metrics available"
        mock_print.assert_called_with("[yellow]No agent metrics available[/yellow]")

    @patch('safehive.ui.metrics_display.Console.print')
    def test_export_metrics_display_json(self, mock_print):
        """Test exporting metrics display as JSON."""
        record_metric("test.metric", 42, MetricType.GAUGE)
        
        json_data = self.display.export_metrics_display("json")
        
        assert isinstance(json_data, str)
        # Should contain system stats
        assert "system_stats" in json_data

    @patch('safehive.ui.metrics_display.Console.print')
    def test_export_metrics_display_summary(self, mock_print):
        """Test exporting metrics display as summary."""
        record_metric("test.metric", 42, MetricType.GAUGE)
        
        summary = self.display.export_metrics_display("summary")
        
        assert isinstance(summary, str)
        # Summary format returns JSON, not text
        assert "system_stats" in summary
        assert "summaries" in summary

    @patch('safehive.ui.metrics_display.Console.print')
    def test_export_metrics_display_text(self, mock_print):
        """Test exporting metrics display as text."""
        record_metric("test.metric", 42, MetricType.GAUGE)
        
        text = self.display.export_metrics_display("text")
        
        assert isinstance(text, str)
        assert "SafeHive Metrics Summary" in text

    @patch('safehive.ui.metrics_display.Console.print')
    def test_create_dashboard_layout(self, mock_print):
        """Test creating dashboard layout."""
        record_metric("test.metric", 42, MetricType.GAUGE)
        
        layout = self.display._create_dashboard_layout()
        
        assert layout is not None
        # Rich Layout uses numeric indices, not string names
        # Just verify the layout was created successfully
        assert hasattr(layout, 'split_column')
        assert hasattr(layout, 'split_row')

    @patch('safehive.ui.metrics_display.Console.print')
    def test_create_system_metrics_panel(self, mock_print):
        """Test creating system metrics panel."""
        record_metric("test.metric", 42, MetricType.GAUGE)
        
        panel = self.display._create_system_metrics_panel()
        
        assert panel is not None

    @patch('safehive.ui.metrics_display.Console.print')
    def test_create_recent_events_panel(self, mock_print):
        """Test creating recent events panel."""
        record_metric("test.event", "Test event", MetricType.EVENT, metadata={"description": "Test event"})
        
        panel = self.display._create_recent_events_panel()
        
        assert panel is not None

    @patch('safehive.ui.metrics_display.Console.print')
    def test_create_recent_events_panel_no_events(self, mock_print):
        """Test creating recent events panel with no events."""
        panel = self.display._create_recent_events_panel()
        
        assert panel is not None

    @patch('safehive.ui.metrics_display.Live')
    @patch('safehive.ui.metrics_display.Console.print')
    def test_display_metrics_dashboard(self, mock_print, mock_live):
        """Test displaying metrics dashboard."""
        # Mock the Live context manager
        mock_live_instance = MagicMock()
        mock_live.return_value.__enter__.return_value = mock_live_instance
        
        # Add some metrics
        record_metric("test.metric", 42, MetricType.GAUGE)
        
        # This should not raise an exception
        try:
            self.display.display_metrics_dashboard(0.1)  # Very short refresh interval
        except KeyboardInterrupt:
            pass  # Expected when testing
        
        # Verify Live was called
        mock_live.assert_called()


class TestGlobalDisplayFunctions:
    """Test cases for global display functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing metrics
        collector = get_metrics_collector()
        collector.clear_metrics()

    @patch('safehive.ui.metrics_display.MetricsDisplay')
    def test_display_metrics_summary(self, mock_display_class):
        """Test display_metrics_summary function."""
        mock_display = MagicMock()
        mock_display_class.return_value = mock_display
        
        display_metrics_summary()
        
        # Verify MetricsDisplay was created and methods were called
        mock_display_class.assert_called_once()
        assert mock_display.display_system_overview.called
        assert mock_display.display_counters.called
        assert mock_display.display_gauges.called
        assert mock_display.display_timers.called
        assert mock_display.display_recent_events.called

    @patch('safehive.ui.metrics_display.MetricsDisplay')
    def test_display_security_metrics(self, mock_display_class):
        """Test display_security_metrics function."""
        mock_display = MagicMock()
        mock_display_class.return_value = mock_display
        
        display_security_metrics()
        
        # Verify MetricsDisplay was created and security metrics method was called
        mock_display_class.assert_called_once()
        mock_display.display_security_metrics.assert_called_once()

    @patch('safehive.ui.metrics_display.MetricsDisplay')
    def test_display_agent_metrics(self, mock_display_class):
        """Test display_agent_metrics function."""
        mock_display = MagicMock()
        mock_display_class.return_value = mock_display
        
        display_agent_metrics()
        
        # Verify MetricsDisplay was created and agent metrics method was called
        mock_display_class.assert_called_once()
        mock_display.display_agent_metrics.assert_called_once()

    @patch('safehive.ui.metrics_display.MetricsDisplay')
    def test_display_metrics_dashboard(self, mock_display_class):
        """Test display_metrics_dashboard function."""
        mock_display = MagicMock()
        mock_display_class.return_value = mock_display
        
        # Import the function
        from safehive.ui.metrics_display import display_metrics_dashboard
        
        try:
            display_metrics_dashboard()
        except KeyboardInterrupt:
            pass  # Expected when testing
        
        # Verify MetricsDisplay was created and dashboard method was called
        mock_display_class.assert_called_once()
        mock_display.display_metrics_dashboard.assert_called_once()


class TestMetricsDisplayIntegration:
    """Integration tests for metrics display system."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing metrics
        collector = get_metrics_collector()
        collector.clear_metrics()

    @patch('safehive.ui.metrics_display.Console.print')
    def test_complete_metrics_display_workflow(self, mock_print):
        """Test complete metrics display workflow."""
        # Add various types of metrics
        record_metric("system.cpu_usage", 75.5, MetricType.GAUGE, {"host": "server1"})
        record_metric("requests.total", 100, MetricType.COUNTER, {"endpoint": "/api/test"})
        record_metric("request.duration", 0.5, MetricType.TIMER, {"endpoint": "/api/test"})
        record_metric("security.alert", "Suspicious activity", MetricType.EVENT, 
                     metadata={"description": "Suspicious activity detected"})
        record_metric("agent.orchestrator.response_time", 1.2, MetricType.TIMER)
        
        display = MetricsDisplay()
        
        # Test all display methods
        display.display_system_overview()
        display.display_metrics_table()
        display.display_counters()
        display.display_gauges()
        display.display_timers()
        display.display_recent_events()
        display.display_security_metrics()
        display.display_agent_metrics()
        
        # Verify all methods were called (print was called multiple times)
        assert mock_print.call_count > 0

    @patch('safehive.ui.metrics_display.Console.print')
    def test_metrics_display_with_no_data(self, mock_print):
        """Test metrics display with no data."""
        display = MetricsDisplay()
        
        # Test all display methods with no data
        display.display_metrics_table()
        display.display_counters()
        display.display_gauges()
        display.display_timers()
        display.display_recent_events()
        display.display_security_metrics()
        display.display_agent_metrics()
        
        # Should print "No ... available" messages
        assert mock_print.call_count > 0

    @patch('safehive.ui.metrics_display.Console.print')
    def test_metrics_display_export_formats(self, mock_print):
        """Test metrics display export formats."""
        record_metric("test.metric", 42, MetricType.GAUGE)
        
        display = MetricsDisplay()
        
        # Test different export formats
        json_data = display.export_metrics_display("json")
        summary_data = display.export_metrics_display("summary")
        text_data = display.export_metrics_display("text")
        
        assert isinstance(json_data, str)
        assert isinstance(summary_data, str)
        assert isinstance(text_data, str)
        
        # JSON should be parseable
        import json
        json.loads(json_data)

    @patch('safehive.ui.metrics_display.Console.print')
    def test_metrics_display_performance(self, mock_print):
        """Test metrics display performance with many metrics."""
        # Add many metrics
        for i in range(100):
            record_metric(f"test.metric_{i}", i, MetricType.GAUGE)
        
        display = MetricsDisplay()
        
        # Should be able to display many metrics in reasonable time
        start_time = time.time()
        display.display_metrics_table()
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 1.0  # Less than 1 second
        
        # Verify print was called
        assert mock_print.call_count > 0
