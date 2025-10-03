"""
Human-in-the-Loop Controls UI for SafeHive AI Security Sandbox

This module provides the CLI interface for human intervention controls,
including request display, response handling, and real-time monitoring.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.align import Align

from safehive.models.human_controls import (
    InterventionRequest, InterventionResponse, HumanControlSession,
    InterventionType, InterventionStatus, IncidentType, IncidentSeverity
)
from safehive.utils.human_controls import get_human_control_manager
from safehive.utils.logger import get_logger

logger = get_logger(__name__)


class HumanControlsUI:
    """User interface for human-in-the-loop controls."""
    
    def __init__(self):
        self.console = Console()
        self.manager = get_human_control_manager()
    
    def display_pending_requests(self, session_id: Optional[str] = None) -> None:
        """Display all pending intervention requests."""
        requests = self.manager.get_pending_requests(session_id)
        
        if not requests:
            self.console.print("✅ No pending intervention requests", style="green")
            return
        
        # Sort by priority and severity
        requests.sort(key=lambda r: (-r.priority, r.severity.value, r.requested_at))
        
        table = Table(title="🚨 Pending Intervention Requests")
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Type", style="yellow")
        table.add_column("Severity", style="red")
        table.add_column("Title", style="white")
        table.add_column("Agent", style="blue")
        table.add_column("Time Left", style="green")
        table.add_column("Priority", style="magenta")
        
        for request in requests:
            # Color severity
            severity_color = {
                "low": "green",
                "medium": "yellow", 
                "high": "red",
                "critical": "bold red"
            }.get(request.severity.value, "white")
            
            # Calculate time left
            remaining = request.get_remaining_time()
            if remaining > 60:
                time_left = f"{remaining // 60}m {remaining % 60}s"
            else:
                time_left = f"{remaining}s"
            
            # Color time left
            if remaining < 60:
                time_style = "bold red"
            elif remaining < 180:
                time_style = "yellow"
            else:
                time_style = "green"
            
            table.add_row(
                request.request_id[:8],
                request.incident_type.value.replace("_", " ").title(),
                Text(request.severity.value.upper(), style=severity_color),
                request.title[:40] + "..." if len(request.title) > 40 else request.title,
                request.agent_id,
                Text(time_left, style=time_style),
                str(request.priority)
            )
        
        self.console.print(table)
    
    def display_request_details(self, request_id: str) -> Optional[InterventionRequest]:
        """Display detailed information about a specific request."""
        requests = self.manager.get_pending_requests()
        request = next((r for r in requests if r.request_id == request_id), None)
        
        if not request:
            self.console.print(f"❌ Request {request_id} not found", style="red")
            return None
        
        # Create detailed panel
        content = f"""
[bold]Request ID:[/bold] {request.request_id}
[bold]Session:[/bold] {request.session_id}
[bold]Agent:[/bold] {request.agent_id}
[bold]Incident Type:[/bold] {request.incident_type.value.replace('_', ' ').title()}
[bold]Severity:[/bold] {request.severity.value.upper()}
[bold]Priority:[/bold] {request.priority}
[bold]Requested:[/bold] {request.requested_at.strftime('%Y-%m-%d %H:%M:%S')}
[bold]Timeout:[/bold] {request.timeout_seconds}s
[bold]Auto Action:[/bold] {request.auto_action.value if request.auto_action else 'None'}

[bold]Title:[/bold] {request.title}

[bold]Description:[/bold]
{request.description}

[bold]Context:[/bold]
{self._format_context(request.context)}

[bold]Affected Data:[/bold]
{self._format_affected_data(request.affected_data) if request.affected_data else 'None'}

[bold]Time Remaining:[/bold] {request.get_remaining_time()}s
"""
        
        panel = Panel(
            content,
            title=f"🚨 Intervention Request Details",
            border_style="red"
        )
        
        self.console.print(panel)
        return request
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context data for display."""
        if not context:
            return "None"
        
        formatted = []
        for key, value in context.items():
            if isinstance(value, dict):
                formatted.append(f"  {key}: {len(value)} items")
            elif isinstance(value, list):
                formatted.append(f"  {key}: {len(value)} items")
            else:
                formatted.append(f"  {key}: {str(value)[:50]}")
        
        return "\n".join(formatted) if formatted else "None"
    
    def _format_affected_data(self, data: Dict[str, Any]) -> str:
        """Format affected data for display."""
        if not data:
            return "None"
        
        formatted = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                formatted.append(f"  {key}: {type(value).__name__} ({len(value)} items)")
            else:
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                formatted.append(f"  {key}: {value_str}")
        
        return "\n".join(formatted) if formatted else "None"
    
    def prompt_for_intervention(self, request_id: str) -> bool:
        """Prompt user for intervention decision."""
        request = self.manager.get_pending_requests()
        request = next((r for r in request if r.request_id == request_id), None)
        
        if not request:
            self.console.print(f"❌ Request {request_id} not found", style="red")
            return False
        
        # Display request details
        self.display_request_details(request_id)
        
        # Prompt for intervention type
        self.console.print("\n[bold]Available Interventions:[/bold]")
        self.console.print("1. [green]APPROVE[/green] - Allow the activity to proceed")
        self.console.print("2. [yellow]REDACT[/yellow] - Remove sensitive information before proceeding")
        self.console.print("3. [red]QUARANTINE[/red] - Isolate or block the activity")
        self.console.print("4. [dim]IGNORE[/dim] - Dismiss the incident without action")
        
        # Get user choice
        while True:
            choice = Prompt.ask(
                "\nSelect intervention type",
                choices=["1", "2", "3", "4", "approve", "redact", "quarantine", "ignore"],
                default="4"
            )
            
            intervention_map = {
                "1": InterventionType.APPROVE,
                "2": InterventionType.REDACT,
                "3": InterventionType.QUARANTINE,
                "4": InterventionType.IGNORE,
                "approve": InterventionType.APPROVE,
                "redact": InterventionType.REDACT,
                "quarantine": InterventionType.QUARANTINE,
                "ignore": InterventionType.IGNORE
            }
            
            intervention_type = intervention_map[choice]
            break
        
        # Get reason
        reason = Prompt.ask("Enter reason for intervention (optional)", default="")
        
        # Handle intervention-specific prompts
        kwargs = {}
        
        if intervention_type == InterventionType.REDACT:
            self.console.print("\n[bold]Redaction Rules[/bold] (one per line, empty to finish):")
            rules = []
            while True:
                rule = Prompt.ask("  Rule", default="")
                if not rule:
                    break
                rules.append(rule)
            kwargs["redaction_rules"] = rules
        
        elif intervention_type == InterventionType.QUARANTINE:
            duration = Prompt.ask("Quarantine duration (minutes)", default="60")
            try:
                kwargs["quarantine_duration"] = int(duration) * 60  # Convert to seconds
            except ValueError:
                kwargs["quarantine_duration"] = 3600  # Default 1 hour
            
            kwargs["quarantine_reason"] = Prompt.ask("Quarantine reason", default="Manual quarantine")
        
        # Confirm intervention
        confirm_text = f"Apply {intervention_type.value.upper()} intervention?"
        if kwargs:
            confirm_text += f" (with options: {kwargs})"
        
        if not Confirm.ask(confirm_text):
            self.console.print("❌ Intervention cancelled", style="yellow")
            return False
        
        # Apply intervention
        success = self.manager.respond_to_intervention(
            request_id=request_id,
            intervention_type=intervention_type,
            reason=reason or None,
            **kwargs
        )
        
        if success:
            self.console.print(f"✅ Applied {intervention_type.value.upper()} intervention", style="green")
            return True
        else:
            self.console.print(f"❌ Failed to apply intervention", style="red")
            return False
    
    def display_session_statistics(self, session_id: str) -> None:
        """Display statistics for a specific session."""
        stats = self.manager.get_session_statistics(session_id)
        
        if not stats:
            self.console.print(f"❌ Session {session_id} not found", style="red")
            return
        
        # Create statistics panel
        content = f"""
[bold]Session ID:[/bold] {stats['session_id']}
[bold]Created:[/bold] {stats['created_at']}
[bold]Active:[/bold] {'Yes' if stats['is_active'] else 'No'}

[bold]Statistics:[/bold]
  Total Requests: {stats['total_requests']}
  Pending: {stats['pending_requests']}
  Completed: {stats['completed_interventions']}

[bold]Intervention Breakdown:[/bold]
  Approved: {stats['approved_count']}
  Redacted: {stats['redacted_count']}
  Quarantined: {stats['quarantined_count']}
  Ignored: {stats['ignored_count']}
  Auto Actions: {stats['auto_action_count']}
"""
        
        panel = Panel(
            content,
            title=f"📊 Human Controls Statistics - {session_id[:8]}...",
            border_style="blue"
        )
        
        self.console.print(panel)
    
    def display_global_statistics(self) -> None:
        """Display global statistics across all sessions."""
        stats = self.manager.get_global_statistics()
        
        # Create summary table
        table = Table(title="📊 Global Human Controls Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Active Sessions", str(stats["active_sessions"]))
        table.add_row("Total Requests", str(stats["total_requests"]))
        table.add_row("Pending Requests", str(stats["pending_requests"]))
        table.add_row("Completed Interventions", str(stats["completed_interventions"]))
        
        self.console.print(table)
        
        # Session breakdown
        if stats["sessions"]:
            self.console.print("\n[bold]Session Breakdown:[/bold]")
            
            session_table = Table()
            session_table.add_column("Session ID", style="cyan", width=12)
            session_table.add_column("Total", style="white")
            session_table.add_column("Pending", style="yellow")
            session_table.add_column("Approved", style="green")
            session_table.add_column("Redacted", style="yellow")
            session_table.add_column("Quarantined", style="red")
            session_table.add_column("Ignored", style="dim")
            
            for session_stats in stats["sessions"]:
                session_table.add_row(
                    session_stats["session_id"][:8] + "...",
                    str(session_stats["total_requests"]),
                    str(session_stats["pending_requests"]),
                    str(session_stats["approved_count"]),
                    str(session_stats["redacted_count"]),
                    str(session_stats["quarantined_count"]),
                    str(session_stats["ignored_count"])
                )
            
            self.console.print(session_table)
    
    def interactive_monitor(self, session_id: Optional[str] = None, refresh_interval: float = 2.0) -> None:
        """Interactive monitoring mode with live updates."""
        self.console.print(f"🔍 Starting interactive monitor for {'all sessions' if not session_id else f'session {session_id}'}")
        self.console.print("Press Ctrl+C to exit\n")
        
        try:
            with Live(self._create_monitor_layout(session_id), refresh_per_second=1/refresh_interval) as live:
                while True:
                    live.update(self._create_monitor_layout(session_id))
                    asyncio.sleep(refresh_interval)
        except KeyboardInterrupt:
            self.console.print("\n👋 Interactive monitor stopped", style="yellow")
    
    def _create_monitor_layout(self, session_id: Optional[str] = None) -> Layout:
        """Create the monitor layout for live updates."""
        layout = Layout()
        
        # Split into sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Header
        layout["header"].update(Panel(
            f"🚨 Human Controls Monitor - {datetime.now().strftime('%H:%M:%S')}",
            style="bold blue"
        ))
        
        # Main content - pending requests
        requests = self.manager.get_pending_requests(session_id)
        
        if requests:
            # Sort by priority and severity
            requests.sort(key=lambda r: (-r.priority, r.severity.value, r.requested_at))
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Time Left", style="red", width=10)
            table.add_column("Severity", style="yellow", width=10)
            table.add_column("Type", style="cyan", width=15)
            table.add_column("Title", style="white")
            table.add_column("Agent", style="blue", width=12)
            
            for request in requests[:10]:  # Show top 10
                remaining = request.get_remaining_time()
                time_str = f"{remaining // 60}m {remaining % 60}s" if remaining > 60 else f"{remaining}s"
                
                severity_color = {
                    "low": "green",
                    "medium": "yellow",
                    "high": "red",
                    "critical": "bold red"
                }.get(request.severity.value, "white")
                
                table.add_row(
                    time_str,
                    Text(request.severity.value.upper(), style=severity_color),
                    request.incident_type.value.replace("_", " ").title(),
                    request.title[:50] + "..." if len(request.title) > 50 else request.title,
                    request.agent_id
                )
            
            layout["main"].update(table)
        else:
            layout["main"].update(Panel("✅ No pending intervention requests", style="green"))
        
        # Footer - statistics
        stats = self.manager.get_global_statistics()
        footer_text = f"Active Sessions: {stats['active_sessions']} | "
        footer_text += f"Pending: {stats['pending_requests']} | "
        footer_text += f"Completed: {stats['completed_interventions']}"
        
        layout["footer"].update(Panel(footer_text, style="dim"))
        
        return layout


# Global UI instance
_human_controls_ui: Optional[HumanControlsUI] = None


def get_human_controls_ui() -> HumanControlsUI:
    """Get the global human controls UI instance."""
    global _human_controls_ui
    if _human_controls_ui is None:
        _human_controls_ui = HumanControlsUI()
    return _human_controls_ui


# Convenience functions
def display_pending_requests(session_id: Optional[str] = None) -> None:
    """Display pending intervention requests."""
    ui = get_human_controls_ui()
    ui.display_pending_requests(session_id)


def prompt_for_intervention(request_id: str) -> bool:
    """Prompt user for intervention decision."""
    ui = get_human_controls_ui()
    return ui.prompt_for_intervention(request_id)


def display_statistics(session_id: Optional[str] = None) -> None:
    """Display statistics."""
    ui = get_human_controls_ui()
    if session_id:
        ui.display_session_statistics(session_id)
    else:
        ui.display_global_statistics()


def interactive_monitor(session_id: Optional[str] = None, refresh_interval: float = 2.0) -> None:
    """Start interactive monitoring."""
    ui = get_human_controls_ui()
    ui.interactive_monitor(session_id, refresh_interval)
