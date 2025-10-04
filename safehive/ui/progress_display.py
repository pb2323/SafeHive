"""
Progress Display for SafeHive AI Security Sandbox

This module provides real-time progress indicators and status updates
for sandbox simulations and agent interactions.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
    ProgressColumn, MofNCompleteColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.align import Align

from safehive.sandbox.sandbox_manager import SandboxSession, SessionStatus, SessionPhase
from safehive.utils.logger import get_logger

logger = get_logger(__name__)


class ProgressEventType(Enum):
    """Types of progress events."""
    SESSION_START = "session_start"
    SESSION_UPDATE = "session_update"
    SESSION_COMPLETE = "session_complete"
    AGENT_UPDATE = "agent_update"
    GUARD_UPDATE = "guard_update"
    SCENARIO_UPDATE = "scenario_update"
    SECURITY_EVENT = "security_event"
    ERROR = "error"


@dataclass
class ProgressEvent:
    """Represents a progress event."""
    event_type: ProgressEventType
    timestamp: datetime
    message: str
    session_id: Optional[str] = None
    phase: Optional[SessionPhase] = None
    data: Optional[Dict[str, Any]] = None


class ProgressDisplay:
    """Real-time progress display for sandbox sessions."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        )
        self.events: List[ProgressEvent] = []
        self._live_display: Optional[Live] = None
        self._layout: Optional[Layout] = None
        self._session_tasks: Dict[str, Any] = {}
        
    def create_layout(self) -> Layout:
        """Create the layout for the progress display."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=8),
            Layout(name="status", size=6),
            Layout(name="events", ratio=1)
        )
        return layout
    
    def create_header(self) -> Panel:
        """Create the header panel."""
        header_text = Text("🛡️ SafeHive AI Security Sandbox - Real-time Progress", style="bold blue")
        return Panel(Align.center(header_text), style="blue")
    
    def create_progress_panel(self) -> Panel:
        """Create the progress panel."""
        if not self._session_tasks:
            return Panel("No active sessions", style="dim")
        
        progress_table = Table(show_header=True, header_style="bold magenta")
        progress_table.add_column("Session ID", style="cyan", width=12)
        progress_table.add_column("Status", style="green", width=12)
        progress_table.add_column("Phase", style="yellow", width=15)
        progress_table.add_column("Progress", style="blue", width=20)
        
        for session_id, task_info in self._session_tasks.items():
            session = task_info.get("session")
            if session:
                progress_bar = "█" * int(session.duration / 10) + "░" * (30 - int(session.duration / 10))
                progress_table.add_row(
                    session_id[:8] + "...",
                    session.status.value.title(),
                    session.phase.value.replace("_", " ").title(),
                    progress_bar
                )
        
        return Panel(progress_table, title="Session Progress", border_style="green")
    
    def create_status_panel(self) -> Panel:
        """Create the status panel."""
        status_table = Table(show_header=True, header_style="bold magenta")
        status_table.add_column("Component", style="cyan", width=15)
        status_table.add_column("Status", style="green", width=10)
        status_table.add_column("Details", style="white", width=25)
        
        # Add some mock status information
        status_table.add_row("AI Agents", "Active", "4 agents running")
        status_table.add_row("Security Guards", "Active", "3 guards monitoring")
        status_table.add_row("Ollama", "Connected", "Model: llama3.2")
        status_table.add_row("Memory", "Healthy", "Buffer memory active")
        
        return Panel(status_table, title="System Status", border_style="blue")
    
    def create_events_panel(self) -> Panel:
        """Create the events panel."""
        if not self.events:
            return Panel("No recent events", style="dim")
        
        events_table = Table(show_header=True, header_style="bold magenta")
        events_table.add_column("Time", style="cyan", width=8)
        events_table.add_column("Type", style="yellow", width=12)
        events_table.add_column("Message", style="white", width=40)
        
        # Show last 10 events
        recent_events = self.events[-10:]
        for event in recent_events:
            time_str = event.timestamp.strftime("%H:%M:%S")
            event_type = event.event_type.value.replace("_", " ").title()
            events_table.add_row(time_str, event_type, event.message)
        
        return Panel(events_table, title="Recent Events", border_style="yellow")
    
    def add_event(self, event: ProgressEvent) -> None:
        """Add a progress event."""
        self.events.append(event)
        # Keep only last 100 events
        if len(self.events) > 100:
            self.events = self.events[-100:]
    
    def update_session(self, session: SandboxSession) -> None:
        """Update session information."""
        if session.session_id not in self._session_tasks:
            self._session_tasks[session.session_id] = {"session": session}
        else:
            self._session_tasks[session.session_id]["session"] = session
    
    def remove_session(self, session_id: str) -> None:
        """Remove session from tracking."""
        if session_id in self._session_tasks:
            del self._session_tasks[session_id]
    
    def start_live_display(self) -> None:
        """Start the live display."""
        if self._live_display is None:
            self._layout = self.create_layout()
            self._live_display = Live(
                self._layout,
                console=self.console,
                refresh_per_second=2,
                screen=True
            )
            self._live_display.start()
    
    def stop_live_display(self) -> None:
        """Stop the live display."""
        if self._live_display:
            self._live_display.stop()
            self._live_display = None
    
    def update_display(self) -> None:
        """Update the live display."""
        if self._live_display and self._layout:
            self._layout["header"].update(self.create_header())
            self._layout["progress"].update(self.create_progress_panel())
            self._layout["status"].update(self.create_status_panel())
            self._layout["events"].update(self.create_events_panel())
    
    def show_session_progress(self, session: SandboxSession, duration: int = 0) -> None:
        """Show progress for a single session."""
        self.console.clear()
        
        with self.progress:
            task = self.progress.add_task(
                f"Running {session.scenario.name} scenario",
                total=duration or session.scenario.duration
            )
            
            # Simulate progress updates
            for i in range(duration or session.scenario.duration):
                self.progress.update(task, advance=1, description=f"Running {session.scenario.name} scenario - Step {i+1}")
                time.sleep(0.1)  # Simulate work
    
    def show_simple_progress(self, message: str, duration: int = 10) -> None:
        """Show simple progress for a single task."""
        with self.progress:
            task = self.progress.add_task(message, total=duration)
            
            for i in range(duration):
                self.progress.update(task, advance=1)
                time.sleep(0.1)


# Global progress display instance
_global_progress_display: Optional[ProgressDisplay] = None


def get_progress_display() -> ProgressDisplay:
    """Get the global progress display instance."""
    global _global_progress_display
    if _global_progress_display is None:
        _global_progress_display = ProgressDisplay()
    return _global_progress_display


def start_progress_display() -> ProgressDisplay:
    """Start the global progress display."""
    display = get_progress_display()
    display.start_live_display()
    return display


def stop_progress_display() -> None:
    """Stop the global progress display."""
    global _global_progress_display
    if _global_progress_display:
        _global_progress_display.stop_live_display()


def add_progress_event(
    event_type: ProgressEventType,
    message: str,
    session_id: Optional[str] = None,
    phase: Optional[SessionPhase] = None,
    data: Optional[Dict[str, Any]] = None
) -> None:
    """Add a progress event to the global display."""
    event = ProgressEvent(
        event_type=event_type,
        timestamp=datetime.now(),
        message=message,
        session_id=session_id,
        phase=phase,
        data=data
    )
    
    display = get_progress_display()
    display.add_event(event)
    
    # Update display if live
    if display._live_display:
        display.update_display()


def update_session_progress(session: SandboxSession) -> None:
    """Update session progress in the global display."""
    display = get_progress_display()
    display.update_session(session)
    
    # Add event
    add_progress_event(
        ProgressEventType.SESSION_UPDATE,
        f"Session {session.session_id[:8]}... updated - {session.status.value}",
        session_id=session.session_id,
        phase=session.phase
    )
    
    # Update display if live
    if display._live_display:
        display.update_display()


def remove_session_progress(session_id: str) -> None:
    """Remove session from progress tracking."""
    display = get_progress_display()
    display.remove_session(session_id)
    
    # Add event
    add_progress_event(
        ProgressEventType.SESSION_COMPLETE,
        f"Session {session_id[:8]}... completed",
        session_id=session_id
    )
    
    # Update display if live
    if display._live_display:
        display.update_display()
