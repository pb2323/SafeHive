"""
Metrics Display Module for SafeHive AI Security Sandbox

This module provides rich formatting and display capabilities for system metrics,
including tables, charts, and summary statistics with color coding and visual indicators.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.align import Align
from rich.syntax import Syntax
from rich.layout import Layout
from rich.live import Live

from safehive.utils.metrics import get_metrics_collector, MetricType, MetricData, get_metrics_summary
from safehive.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsDisplay:
    """
    Rich display formatter for SafeHive metrics.
    
    Provides various display formats including tables, panels, and real-time dashboards
    for system metrics, security events, and performance data.
    """
    
    def __init__(self):
        self.console = Console()
        self.metrics_collector = get_metrics_collector()
    
    def display_system_overview(self) -> None:
        """Display system overview with key metrics."""
        system_stats = self.metrics_collector.get_system_stats()
        
        # Create overview panel
        overview_text = f"""
🕒 Uptime: {system_stats['uptime_human']}
📊 Total Metrics: {system_stats['total_metrics_collected']:,}
📈 Active Metrics: {system_stats['active_metrics']}
🔢 Counters: {system_stats['active_counters']}
📏 Gauges: {system_stats['active_gauges']}
⏱️ Timers: {system_stats['active_timers']}
📝 Recent Events: {system_stats['recent_events']}
💾 Memory Usage: {system_stats['memory_usage_mb']:.2f} MB
        """.strip()
        
        self.console.print(Panel(
            Align.center(Text(overview_text, style="white")),
            title="📊 System Overview",
            border_style="blue",
            padding=(1, 2)
        ))
    
    def display_metrics_table(self, metric_type: Optional[str] = None) -> None:
        """Display metrics in a table format."""
        summaries = self.metrics_collector.get_all_summaries()
        
        if not summaries:
            self.console.print("[yellow]No metrics available[/yellow]")
            return
        
        # Filter by metric type if specified
        if metric_type:
            summaries = {
                name: summary for name, summary in summaries.items()
                if summary.metric_type.value == metric_type
            }
        
        if not summaries:
            self.console.print(f"[yellow]No {metric_type} metrics available[/yellow]")
            return
        
        # Create table
        table = Table(
            title=f"📊 Metrics Summary ({metric_type or 'All Types'})",
            show_header=True,
            header_style="bold cyan",
            border_style="blue"
        )
        
        table.add_column("Name", style="cyan", width=25)
        table.add_column("Type", style="green", width=10)
        table.add_column("Count", style="yellow", width=8)
        table.add_column("Last Value", style="white", width=12)
        table.add_column("Min", style="dim white", width=8)
        table.add_column("Max", style="dim white", width=8)
        table.add_column("Avg", style="dim white", width=8)
        table.add_column("Last Updated", style="dim white", width=16)
        
        for name, summary in sorted(summaries.items()):
            # Format last updated time
            last_updated = "Never"
            if summary.last_updated:
                time_diff = datetime.now() - summary.last_updated
                if time_diff.total_seconds() < 60:
                    last_updated = f"{int(time_diff.total_seconds())}s ago"
                elif time_diff.total_seconds() < 3600:
                    last_updated = f"{int(time_diff.total_seconds() / 60)}m ago"
                else:
                    last_updated = f"{int(time_diff.total_seconds() / 3600)}h ago"
            
            # Format values
            last_value = str(summary.last_value) if summary.last_value is not None else "N/A"
            min_value = f"{summary.min_value:.2f}" if summary.min_value is not None else "N/A"
            max_value = f"{summary.max_value:.2f}" if summary.max_value is not None else "N/A"
            avg_value = f"{summary.avg_value:.2f}" if summary.avg_value > 0 else "N/A"
            
            table.add_row(
                name,
                summary.metric_type.value,
                str(summary.count),
                last_value,
                min_value,
                max_value,
                avg_value,
                last_updated
            )
        
        self.console.print(table)
    
    def display_counters(self) -> None:
        """Display counter metrics."""
        counters = self.metrics_collector.get_counters()
        
        if not counters:
            self.console.print("[yellow]No counter metrics available[/yellow]")
            return
        
        table = Table(
            title="🔢 Counter Metrics",
            show_header=True,
            header_style="bold cyan",
            border_style="green"
        )
        
        table.add_column("Counter Name", style="cyan", width=30)
        table.add_column("Value", style="bold green", width=15)
        table.add_column("Status", style="white", width=10)
        
        for name, value in sorted(counters.items()):
            # Determine status based on value
            if value == 0:
                status = "🟡 Zero"
                status_style = "yellow"
            elif value < 10:
                status = "🟢 Low"
                status_style = "green"
            elif value < 100:
                status = "🟠 Medium"
                status_style = "orange3"
            else:
                status = "🔴 High"
                status_style = "red"
            
            table.add_row(
                name,
                f"{value:,}",
                Text(status, style=status_style)
            )
        
        self.console.print(table)
    
    def display_gauges(self) -> None:
        """Display gauge metrics."""
        gauges = self.metrics_collector.get_gauges()
        
        if not gauges:
            self.console.print("[yellow]No gauge metrics available[/yellow]")
            return
        
        table = Table(
            title="📏 Gauge Metrics",
            show_header=True,
            header_style="bold cyan",
            border_style="blue"
        )
        
        table.add_column("Gauge Name", style="cyan", width=30)
        table.add_column("Current Value", style="bold blue", width=15)
        table.add_column("Status", style="white", width=10)
        
        for name, value in sorted(gauges.items()):
            # Determine status based on value (customize thresholds as needed)
            if value == 0:
                status = "🟡 Zero"
                status_style = "yellow"
            elif value < 50:
                status = "🟢 Low"
                status_style = "green"
            elif value < 80:
                status = "🟠 Medium"
                status_style = "orange3"
            else:
                status = "🔴 High"
                status_style = "red"
            
            table.add_row(
                name,
                f"{value:.2f}",
                Text(status, style=status_style)
            )
        
        self.console.print(table)
    
    def display_timers(self) -> None:
        """Display timer metrics with statistics."""
        timer_stats = {}
        for name in self.metrics_collector._timers.keys():
            stats = self.metrics_collector.get_timer_stats(name)
            if stats:
                timer_stats[name] = stats
        
        if not timer_stats:
            self.console.print("[yellow]No timer metrics available[/yellow]")
            return
        
        table = Table(
            title="⏱️ Timer Metrics",
            show_header=True,
            header_style="bold cyan",
            border_style="magenta"
        )
        
        table.add_column("Timer Name", style="cyan", width=25)
        table.add_column("Count", style="yellow", width=8)
        table.add_column("Min (s)", style="green", width=10)
        table.add_column("Max (s)", style="red", width=10)
        table.add_column("Avg (s)", style="blue", width=10)
        table.add_column("Total (s)", style="white", width=12)
        
        for name, stats in sorted(timer_stats.items()):
            table.add_row(
                name,
                str(stats["count"]),
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
                f"{stats['avg']:.3f}",
                f"{stats['total']:.3f}"
            )
        
        self.console.print(table)
    
    def display_recent_events(self, limit: int = 20) -> None:
        """Display recent events."""
        events = self.metrics_collector.get_recent_events(limit)
        
        if not events:
            self.console.print("[yellow]No recent events available[/yellow]")
            return
        
        table = Table(
            title=f"📝 Recent Events (Last {limit})",
            show_header=True,
            header_style="bold cyan",
            border_style="yellow"
        )
        
        table.add_column("Timestamp", style="dim white", width=16)
        table.add_column("Event Name", style="cyan", width=25)
        table.add_column("Description", style="white", width=40)
        table.add_column("Tags", style="dim white", width=20)
        
        for event in events:
            # Format timestamp
            time_diff = datetime.now() - event.timestamp
            if time_diff.total_seconds() < 60:
                timestamp = f"{int(time_diff.total_seconds())}s ago"
            elif time_diff.total_seconds() < 3600:
                timestamp = f"{int(time_diff.total_seconds() / 60)}m ago"
            else:
                timestamp = event.timestamp.strftime("%H:%M:%S")
            
            # Format tags
            tags_str = ", ".join([f"{k}={v}" for k, v in event.tags.items()]) if event.tags else "None"
            
            # Get description from metadata
            description = event.metadata.get("description", str(event.value))
            
            table.add_row(
                timestamp,
                event.name,
                description[:40] + "..." if len(description) > 40 else description,
                tags_str[:20] + "..." if len(tags_str) > 20 else tags_str
            )
        
        self.console.print(table)
    
    def display_security_metrics(self) -> None:
        """Display security-specific metrics."""
        # Get security-related metrics
        summaries = self.metrics_collector.get_all_summaries()
        security_metrics = {
            name: summary for name, summary in summaries.items()
            if any(keyword in name.lower() for keyword in [
                'security', 'attack', 'threat', 'alert', 'guard', 'block', 'detect'
            ])
        }
        
        if not security_metrics:
            self.console.print("[yellow]No security metrics available[/yellow]")
            return
        
        # Create security overview
        total_security_events = sum(summary.count for summary in security_metrics.values())
        recent_security_events = len([
            event for event in self.metrics_collector.get_recent_events(100)
            if any(keyword in event.name.lower() for keyword in [
                'security', 'attack', 'threat', 'alert', 'guard', 'block', 'detect'
            ])
        ])
        
        overview_text = f"""
🛡️ Total Security Events: {total_security_events:,}
🚨 Recent Security Events: {recent_security_events}
📊 Security Metrics: {len(security_metrics)}
        """.strip()
        
        self.console.print(Panel(
            Align.center(Text(overview_text, style="white")),
            title="🛡️ Security Overview",
            border_style="red",
            padding=(1, 2)
        ))
        
        # Display security metrics table
        table = Table(
            title="🛡️ Security Metrics",
            show_header=True,
            header_style="bold red",
            border_style="red"
        )
        
        table.add_column("Metric Name", style="cyan", width=30)
        table.add_column("Type", style="green", width=10)
        table.add_column("Count", style="bold yellow", width=10)
        table.add_column("Last Value", style="white", width=15)
        table.add_column("Status", style="white", width=12)
        
        for name, summary in sorted(security_metrics.items()):
            # Determine security status
            if summary.count == 0:
                status = "🟢 Safe"
                status_style = "green"
            elif summary.count < 5:
                status = "🟡 Low Risk"
                status_style = "yellow"
            elif summary.count < 20:
                status = "🟠 Medium Risk"
                status_style = "orange3"
            else:
                status = "🔴 High Risk"
                status_style = "red"
            
            last_value = str(summary.last_value) if summary.last_value is not None else "N/A"
            
            table.add_row(
                name,
                summary.metric_type.value,
                str(summary.count),
                last_value,
                Text(status, style=status_style)
            )
        
        self.console.print(table)
    
    def display_agent_metrics(self) -> None:
        """Display agent-specific metrics."""
        # Get agent-related metrics
        summaries = self.metrics_collector.get_all_summaries()
        agent_metrics = {
            name: summary for name, summary in summaries.items()
            if any(keyword in name.lower() for keyword in [
                'agent', 'orchestrator', 'user_twin', 'vendor', 'conversation', 'memory'
            ])
        }
        
        if not agent_metrics:
            self.console.print("[yellow]No agent metrics available[/yellow]")
            return
        
        # Create agent overview
        total_agent_events = sum(summary.count for summary in agent_metrics.values())
        
        overview_text = f"""
🤖 Total Agent Events: {total_agent_events:,}
📊 Agent Metrics: {len(agent_metrics)}
        """.strip()
        
        self.console.print(Panel(
            Align.center(Text(overview_text, style="white")),
            title="🤖 Agent Overview",
            border_style="blue",
            padding=(1, 2)
        ))
        
        # Display agent metrics table
        table = Table(
            title="🤖 Agent Metrics",
            show_header=True,
            header_style="bold blue",
            border_style="blue"
        )
        
        table.add_column("Metric Name", style="cyan", width=30)
        table.add_column("Type", style="green", width=10)
        table.add_column("Count", style="bold yellow", width=10)
        table.add_column("Last Value", style="white", width=15)
        table.add_column("Performance", style="white", width=12)
        
        for name, summary in sorted(agent_metrics.items()):
            # Determine performance status
            if summary.avg_value == 0:
                performance = "🟡 No Data"
                perf_style = "yellow"
            elif summary.avg_value < 1.0:
                performance = "🟢 Excellent"
                perf_style = "green"
            elif summary.avg_value < 5.0:
                performance = "🟠 Good"
                perf_style = "orange3"
            else:
                performance = "🔴 Slow"
                perf_style = "red"
            
            last_value = str(summary.last_value) if summary.last_value is not None else "N/A"
            
            table.add_row(
                name,
                summary.metric_type.value,
                str(summary.count),
                last_value,
                Text(performance, style=perf_style)
            )
        
        self.console.print(table)
    
    def display_metrics_dashboard(self, refresh_interval: float = 2.0) -> None:
        """Display a real-time metrics dashboard."""
        try:
            with Live(self._create_dashboard_layout(), refresh_per_second=1/refresh_interval) as live:
                while True:
                    live.update(self._create_dashboard_layout())
                    time.sleep(refresh_interval)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Dashboard stopped[/yellow]")
    
    def _create_dashboard_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        # Split into sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Header
        system_stats = self.metrics_collector.get_system_stats()
        header_text = f"📊 SafeHive Metrics Dashboard | Uptime: {system_stats['uptime_human']} | Metrics: {system_stats['total_metrics_collected']:,}"
        layout["header"].update(Panel(Align.center(Text(header_text, style="bold blue")), border_style="blue"))
        
        # Left panel - System metrics
        left_content = self._create_system_metrics_panel()
        layout["left"].update(left_content)
        
        # Right panel - Recent events
        right_content = self._create_recent_events_panel()
        layout["right"].update(right_content)
        
        # Footer
        footer_text = f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Press Ctrl+C to exit"
        layout["footer"].update(Panel(Align.center(Text(footer_text, style="dim white")), border_style="dim"))
        
        return layout
    
    def _create_system_metrics_panel(self) -> Panel:
        """Create system metrics panel for dashboard."""
        system_stats = self.metrics_collector.get_system_stats()
        
        content = f"""
🕒 Uptime: {system_stats['uptime_human']}
📊 Total Metrics: {system_stats['total_metrics_collected']:,}
📈 Active Metrics: {system_stats['active_metrics']}
🔢 Counters: {system_stats['active_counters']}
📏 Gauges: {system_stats['active_gauges']}
⏱️ Timers: {system_stats['active_timers']}
📝 Events: {system_stats['recent_events']}
💾 Memory: {system_stats['memory_usage_mb']:.2f} MB
        """.strip()
        
        return Panel(
            Text(content, style="white"),
            title="📊 System Metrics",
            border_style="blue"
        )
    
    def _create_recent_events_panel(self) -> Panel:
        """Create recent events panel for dashboard."""
        events = self.metrics_collector.get_recent_events(10)
        
        if not events:
            content = "No recent events"
        else:
            content_lines = []
            for event in events:
                time_diff = datetime.now() - event.timestamp
                if time_diff.total_seconds() < 60:
                    timestamp = f"{int(time_diff.total_seconds())}s ago"
                else:
                    timestamp = event.timestamp.strftime("%H:%M:%S")
                
                description = event.metadata.get("description", str(event.value))
                content_lines.append(f"{timestamp}: {event.name} - {description[:30]}...")
            
            content = "\n".join(content_lines)
        
        return Panel(
            Text(content, style="white"),
            title="📝 Recent Events",
            border_style="yellow"
        )
    
    def export_metrics_display(self, format: str = "table") -> str:
        """Export metrics in a display-friendly format."""
        if format == "json":
            return self.metrics_collector.export_metrics("json")
        elif format == "summary":
            summary = get_metrics_summary()
            # Convert MetricSummary objects to dictionaries for JSON serialization
            serializable_summary = {
                "system_stats": summary["system_stats"],
                "summaries": {name: summary_obj.to_dict() for name, summary_obj in summary["summaries"].items()},
                "counters": summary["counters"],
                "gauges": summary["gauges"],
                "recent_events": [event.to_dict() for event in summary["recent_events"]]
            }
            return json.dumps(serializable_summary, indent=2)
        else:
            # Return a text summary
            system_stats = self.metrics_collector.get_system_stats()
            return f"""
SafeHive Metrics Summary
========================
Uptime: {system_stats['uptime_human']}
Total Metrics: {system_stats['total_metrics_collected']:,}
Active Metrics: {system_stats['active_metrics']}
Counters: {system_stats['active_counters']}
Gauges: {system_stats['active_gauges']}
Timers: {system_stats['active_timers']}
Recent Events: {system_stats['recent_events']}
Memory Usage: {system_stats['memory_usage_mb']:.2f} MB
            """.strip()


def display_metrics_summary() -> None:
    """Display a comprehensive metrics summary."""
    display = MetricsDisplay()
    
    display.display_system_overview()
    display.console.print()
    
    display.display_counters()
    display.console.print()
    
    display.display_gauges()
    display.console.print()
    
    display.display_timers()
    display.console.print()
    
    display.display_recent_events()


def display_security_metrics() -> None:
    """Display security-specific metrics."""
    display = MetricsDisplay()
    display.display_security_metrics()


def display_agent_metrics() -> None:
    """Display agent-specific metrics."""
    display = MetricsDisplay()
    display.display_agent_metrics()


def display_metrics_dashboard() -> None:
    """Display real-time metrics dashboard."""
    display = MetricsDisplay()
    display.display_metrics_dashboard()
