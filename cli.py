#!/usr/bin/env python3
"""
SafeHive AI Security Sandbox - Main CLI Entry Point

This is the main entry point for the SafeHive AI Security Sandbox.
It provides a command-line interface for launching and controlling
the AI security sandbox environment.
"""

import typer
from typer import Context
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

from safehive import __version__
from safehive.config.config_loader import ConfigLoader
from safehive.utils.logger import setup_logging, get_logger
from safehive.utils.ai_client import check_ollama_connection, ensure_model_available
from safehive.ui.interactive_menu import InteractiveMenu
from safehive.utils.metrics import get_metrics_collector, record_metric, MetricType
from safehive.ui.metrics_display import MetricsDisplay, display_metrics_summary
from safehive.sandbox.sandbox_manager import get_sandbox_manager, SandboxSession, SessionStatus
from safehive.utils.human_controls import get_human_control_manager
from safehive.ui.human_controls_ui import get_human_controls_ui
from safehive.models.human_controls import InterventionType
from safehive.ui.progress_display import (
    get_progress_display, start_progress_display, stop_progress_display,
    add_progress_event, update_session_progress, remove_session_progress,
    ProgressEventType
)
from safehive.ui.help_system import show_help, interactive_help, get_command_suggestions

# Create the main typer app
app = typer.Typer(
    name="safehive",
    help="🛡️ SafeHive AI Security Sandbox - Advanced AI Security Testing Platform\n\n"
         "Test AI agent security, detect vulnerabilities, and validate guard systems.\n"
         "Features include real-time monitoring, human-in-the-loop controls, and comprehensive metrics.",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True
)

# Create sub-apps for different modules
sandbox_app = typer.Typer(
    name="sandbox", 
    help="🎯 Sandbox Operations - Manage security testing sessions\n\n"
         "Start, stop, and monitor sandbox sessions with various security scenarios.\n"
         "Includes interactive mode, background execution, and real-time progress tracking.",
    rich_markup_mode="rich"
)

config_app = typer.Typer(
    name="config", 
    help="⚙️ Configuration Management - View and validate system settings\n\n"
         "Display current configuration, validate settings, and manage system parameters.\n"
         "Supports custom configuration files and section-specific viewing.",
    rich_markup_mode="rich"
)

agent_app = typer.Typer(
    name="agent", 
    help="🤖 Agent Management - Control AI agents and their behavior\n\n"
         "Manage orchestrator, user twin, and vendor agents.\n"
         "Configure agent personalities, memory, and communication protocols.",
    rich_markup_mode="rich"
)

guard_app = typer.Typer(
    name="guard", 
    help="🛡️ Security Guard Operations - Manage security guards\n\n"
         "Enable, disable, and configure security guards including Privacy Sentry,\n"
         "Task Navigator, and Prompt Sanitizer.",
    rich_markup_mode="rich"
)

metrics_app = typer.Typer(
    name="metrics", 
    help="📊 Metrics and Monitoring - Track system performance\n\n"
         "View real-time metrics, export data, and launch monitoring dashboards.\n"
         "Includes system statistics, security events, and performance indicators.",
    rich_markup_mode="rich"
)

human_app = typer.Typer(
    name="human", 
    help="🎮 Human-in-the-Loop Controls - Manual intervention management\n\n"
         "Handle intervention requests, approve/redact/quarantine actions,\n"
         "and monitor human decision-making workflows.",
    rich_markup_mode="rich"
)

progress_app = typer.Typer(
    name="progress", 
    help="📈 Real-time Progress Monitoring - Live session tracking\n\n"
         "Monitor active sessions with live progress indicators, agent status,\n"
         "and guard activities. Includes demonstration mode for learning.",
    rich_markup_mode="rich"
)

# Add sub-apps to main app
app.add_typer(sandbox_app, name="sandbox")
app.add_typer(config_app, name="config")
app.add_typer(agent_app, name="agent")
app.add_typer(guard_app, name="guard")
app.add_typer(metrics_app, name="metrics")
app.add_typer(human_app, name="human")
app.add_typer(progress_app, name="progress")

console = Console()
logger = get_logger(__name__)


def check_system_requirements() -> bool:
    """Check if system requirements are met."""
    console.print("🔍 Checking system requirements...")
    
    # Check Ollama connection
    if not check_ollama_connection():
        console.print("❌ Ollama is not running. Please start Ollama first.", style="red")
        return False
    
    console.print("✅ Ollama is running", style="green")
    
    # Check if default model is available
    if not ensure_model_available("llama3.2:3b"):
        console.print("❌ Default model 'llama3.2:3b' is not available", style="red")
        return False
    
    console.print("✅ Default model is available", style="green")
    return True


@app.command()
def version():
    """Show version information."""
    version_info = f"""
    SafeHive AI Security Sandbox
    Version: {__version__}
    Python: {sys.version.split()[0]}
    Platform: {sys.platform}
    """
    
    console.print(Panel(
        Text(version_info, style="bold green"),
        title="Version Information",
        border_style="green"
    ))


@app.command()
def info():
    """Show project information."""
    info_text = """
    SafeHive AI Security Sandbox
    
    A CLI-based demonstration and testing platform that simulates a 
    food-ordering workflow where AI assistants interact with potentially 
    malicious vendors, payment services, and external APIs.
    
    Features:
    • Three AI Security Guards (Privacy Sentry, Task Navigator, Prompt Sanitizer)
    • LangChain-powered AI agents with memory and reasoning
    • Interactive CLI with human-in-the-loop controls
    • Real-time attack detection and response
    • Comprehensive logging and metrics
    """
    
    console.print(Panel(
        Text(info_text, style="white"),
        title="SafeHive AI Security Sandbox",
        border_style="blue"
    ))


@app.command()
def status(
    metrics: bool = typer.Option(False, "--metrics", "-m", help="Show detailed metrics summary")
):
    """Show system status."""
    console.print("📊 System Status", style="bold blue")

    # Check Ollama
    ollama_status = "🟢 Running" if check_ollama_connection() else "🔴 Not Running"
    console.print(f"Ollama: {ollama_status}")

    # Check configuration
    config_loader = ConfigLoader()
    config_status = "🟢 Loaded" if config_loader.load_config() else "🔴 Failed"
    console.print(f"Configuration: {config_status}")

    # Check logs directory
    logs_dir = Path("logs")
    logs_status = "🟢 Available" if logs_dir.exists() else "🔴 Not Found"
    console.print(f"Logs Directory: {logs_status}")
    
    # Record status check metrics
    record_metric("system.status_check", 1, MetricType.COUNTER, {"component": "cli"})
    record_metric("system.ollama_status", 1 if check_ollama_connection() else 0, MetricType.GAUGE)
    record_metric("system.config_status", 1 if config_loader.load_config() else 0, MetricType.GAUGE)
    
    # Show metrics if requested
    if metrics:
        console.print()
        console.print("📈 Detailed Metrics Summary", style="bold blue")
        display_metrics_summary()


@app.command()
def init(
    config_file: Optional[str] = typer.Option(
        None, 
        "--config", "-c", 
        help="📄 Path to custom configuration file\n"
             "If not provided, uses default configuration from config/default_config.yaml"
    ),
    log_level: str = typer.Option(
        "INFO", 
        "--log-level", "-l", 
        help="📝 Logging level (DEBUG, INFO, WARNING, ERROR)\n"
             "Higher levels show fewer messages but include more serious issues"
    ),
    interactive: bool = typer.Option(
        True, 
        "--interactive/--no-interactive", 
        help="🎮 Interactive mode - provides guided setup process\n"
             "Use --no-interactive for automated/silent initialization"
    )
):
    """🚀 Initialize the SafeHive AI Security Sandbox system.
    
    Sets up the complete SafeHive environment including:
    • Ollama connection verification and model downloads
    • Configuration file creation and validation
    • Logging directory setup and permissions
    • System requirements verification
    
    This command must be run before using any other SafeHive features.
    
    Examples:
        safehive init
        safehive init --log-level DEBUG
        safehive init --config custom_config.yaml --no-interactive
    """
    console.print("🚀 Initializing SafeHive AI Security Sandbox...", style="bold blue")

    # Record initialization start
    init_start_time = time.time()
    record_metric("system.initialization_start", 1, MetricType.COUNTER, {"log_level": log_level})

    # Setup logging
    setup_logging(level=log_level, log_file="logs/safehive.log")
    logger.info("SafeHive system initialized")

    # Load configuration
    config_loader = ConfigLoader()
    if config_file:
        config_loader.load_config(config_file)
    else:
        config_loader.load_config()

    # Check system requirements
    if not check_system_requirements():
        console.print("❌ System requirements not met. Please fix the issues above.", style="red")
        record_metric("system.initialization_failed", 1, MetricType.COUNTER, {"reason": "requirements_not_met"})
        raise typer.Exit(1)

    # Record successful initialization
    init_duration = time.time() - init_start_time
    record_metric("system.initialization_success", 1, MetricType.COUNTER)
    record_metric("system.initialization_duration", init_duration, MetricType.TIMER)
    record_metric("system.ollama_available", 1 if check_ollama_connection() else 0, MetricType.GAUGE)

    console.print("✅ SafeHive system initialized successfully!", style="green")

    if interactive:
        console.print("\n🎯 Ready to start sandbox operations!")
        console.print("Use 'safehive sandbox start' to begin a security testing session.")


@app.command()
def menu():
    """Launch the interactive menu system."""
    console.print("🖥️ Launching SafeHive Interactive Menu...", style="bold blue")
    
    try:
        menu = InteractiveMenu()
        menu.run()
    except Exception as e:
        console.print(f"❌ Error launching interactive menu: {e}", style="red")
        logger.error(f"Interactive menu error: {e}")
        raise typer.Exit(1)


# Sandbox Commands
@sandbox_app.command("start")
def sandbox_start(
    scenario: str = typer.Option(
        "food-ordering", 
        "--scenario", "-s", 
        help="🎯 Test scenario to run (default: food-ordering)\n"
             "Available scenarios:\n"
             "  • food-ordering: Food ordering workflow with malicious vendors"
    ),
    duration: int = typer.Option(
        None, 
        "--duration", "-d", 
        help="⏱️ Session duration in seconds (default: 300)\n"
             "Recommended: 60-600 seconds for most scenarios"
    ),
    interactive: bool = typer.Option(
        None, 
        "--interactive/--no-interactive", 
        help="🎮 Interactive mode - allows real-time user input during simulation\n"
             "Use --interactive for hands-on testing, --no-interactive for automated runs"
    ),
    background: bool = typer.Option(
        False, 
        "--background", "-b", 
        help="🔄 Run in background mode - non-blocking execution\n"
             "Session continues running while terminal is available for other commands"
    )
):
    """🚀 Start a security sandbox session with scenario execution.
    
    Launches a new sandbox session with the specified scenario, allowing you to test
    AI agent security, monitor guard behavior, and validate system responses.
    
    Examples:
        safehive sandbox start --scenario food-ordering
        safehive sandbox start --scenario food-ordering --interactive --duration 600
        safehive sandbox start --scenario food-ordering --background
    """
    console.print(f"🎯 Starting sandbox session: {scenario}", style="bold blue")

    if not check_system_requirements():
        console.print("❌ System requirements not met.", style="red")
        raise typer.Exit(1)

    try:
        # Get sandbox manager
        sandbox_manager = get_sandbox_manager()
        
        # Check if scenario exists
        available_scenarios = sandbox_manager.list_scenarios()
        if scenario not in available_scenarios:
            console.print(f"❌ Scenario '{scenario}' not found.", style="red")
            console.print("Available scenarios:", style="yellow")
            for name, scenario_obj in available_scenarios.items():
                console.print(f"  • {name}: {scenario_obj.description}")
            raise typer.Exit(1)
        
        # Create session
        session = sandbox_manager.create_session(
            scenario_name=scenario,
            duration=duration,
            interactive=interactive
        )
        
        if not session:
            console.print("❌ Failed to create sandbox session.", style="red")
            raise typer.Exit(1)
        
        console.print(f"✅ Created session: {session.session_id}", style="green")
        console.print(f"📋 Scenario: {session.scenario.description}")
        console.print(f"⏱️  Duration: {session.scenario.duration} seconds")
        console.print(f"🔄 Interactive: {'Yes' if session.interactive else 'No'}")
        
        # Record metrics
        record_metric("cli.sandbox_start", 1, MetricType.COUNTER, {"scenario": scenario})
        
        # Add progress tracking
        add_progress_event(ProgressEventType.SESSION_START, f"Starting {scenario} scenario", session.session_id)
        
        if background:
            # Start session in background
            console.print("🚀 Starting session in background...", style="blue")
            asyncio.run(sandbox_manager.start_session(session.session_id, wait_for_completion=False))
            console.print(f"✅ Session {session.session_id} started in background", style="green")
            console.print("💡 Use 'safehive progress status' to monitor progress", style="dim")
        else:
            # Start session and wait
            console.print("🚀 Starting session...", style="blue")
            asyncio.run(sandbox_manager.start_session(session.session_id, wait_for_completion=True))
            
            # Update progress
            session = sandbox_manager.get_session(session.session_id)
            if session:
                update_session_progress(session)
            
            # Show session status
            console.print("\n📊 Session Status:", style="bold blue")
            if session:
                console.print(f"  Status: {session.status.value}")
                console.print(f"  Phase: {session.phase.value}")
                console.print(f"  Duration: {session.duration}s")
                console.print(f"  Events: {len(session.events)}")
                console.print(f"  Security Events: {len([e for e in session.events if 'security' in e.get('type', '')])}")
                
                # Add completion event
                add_progress_event(ProgressEventType.SESSION_COMPLETE, f"Session {scenario} completed", session.session_id)
    
    except Exception as e:
        console.print(f"❌ Error starting sandbox session: {e}", style="red")
        logger.error(f"Sandbox start error: {e}")
        raise typer.Exit(1)


@sandbox_app.command("stop")
def sandbox_stop(
    session_id: Optional[str] = typer.Option(None, "--session-id", "-s", help="Session ID to stop (default: all active sessions)")
):
    """Stop sandbox session(s)."""
    console.print("🛑 Stopping sandbox session(s)...", style="yellow")
    
    try:
        sandbox_manager = get_sandbox_manager()
        
        if session_id:
            # Stop specific session
            session = sandbox_manager.get_session(session_id)
            if not session:
                console.print(f"❌ Session {session_id} not found.", style="red")
                raise typer.Exit(1)
            
            success = asyncio.run(sandbox_manager.stop_session(session_id))
            if success:
                console.print(f"✅ Session {session_id} stopped", style="green")
                record_metric("cli.sandbox_stop", 1, MetricType.COUNTER, {"session_id": session_id})
            else:
                console.print(f"❌ Failed to stop session {session_id}", style="red")
                raise typer.Exit(1)
        else:
            # Stop all active sessions
            active_sessions = sandbox_manager.get_active_sessions()
            if not active_sessions:
                console.print("ℹ️ No active sessions to stop", style="blue")
                return
            
            stopped_count = 0
            for sid in list(active_sessions.keys()):
                success = asyncio.run(sandbox_manager.stop_session(sid))
                if success:
                    stopped_count += 1
                    console.print(f"✅ Stopped session {sid}", style="green")
                else:
                    console.print(f"❌ Failed to stop session {sid}", style="red")
            
            console.print(f"📊 Stopped {stopped_count}/{len(active_sessions)} sessions", style="blue")
            record_metric("cli.sandbox_stop_all", stopped_count, MetricType.COUNTER)
    
    except Exception as e:
        console.print(f"❌ Error stopping sandbox session: {e}", style="red")
        logger.error(f"Sandbox stop error: {e}")
        raise typer.Exit(1)


@sandbox_app.command("list")
def sandbox_list(
    show_sessions: bool = typer.Option(False, "--sessions", help="Show active sessions instead of scenarios")
):
    """List available sandbox scenarios or active sessions."""
    
    try:
        sandbox_manager = get_sandbox_manager()
        
        if show_sessions:
            # Show active sessions
            active_sessions = sandbox_manager.get_active_sessions()
            
            if not active_sessions:
                console.print("ℹ️ No active sandbox sessions", style="blue")
                return
            
            table = Table(title="Active Sandbox Sessions")
            table.add_column("Session ID", style="cyan", width=36)
            table.add_column("Scenario", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Phase", style="blue")
            table.add_column("Duration", style="white")
            table.add_column("Events", style="dim white")
            
            for session in active_sessions.values():
                status_color = {
                    "pending": "yellow",
                    "starting": "blue",
                    "running": "green",
                    "paused": "orange3",
                    "stopping": "red",
                    "stopped": "dim",
                    "error": "red",
                    "completed": "green"
                }.get(session.status.value, "white")
                
                table.add_row(
                    session.session_id[:8] + "...",
                    session.scenario.name,
                    Text(session.status.value, style=status_color),
                    session.phase.value,
                    f"{session.duration}s",
                    str(len(session.events))
                )
            
            console.print(table)
            
        else:
            # Show available scenarios
            scenarios = sandbox_manager.list_scenarios()
            
            table = Table(title="Available Sandbox Scenarios")
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Duration", style="green")
            table.add_column("Agents", style="blue")
            table.add_column("Guards", style="yellow")
            table.add_column("Category", style="dim white")
            
            for name, scenario in scenarios.items():
                agents_count = len(scenario.agents)
                guards_count = len(scenario.guards)
                category = scenario.metadata.get("category", "general")
                
                table.add_row(
                    name,
                    scenario.description,
                    f"{scenario.duration}s",
                    str(agents_count),
                    str(guards_count),
                    category
                )
            
            console.print(table)
            
            # Show additional info
            console.print(f"\n📊 Total scenarios: {len(scenarios)}", style="dim")
            console.print("Use 'safehive sandbox list --sessions' to see active sessions", style="dim")
    
    except Exception as e:
        console.print(f"❌ Error listing sandbox data: {e}", style="red")
        logger.error(f"Sandbox list error: {e}")
        raise typer.Exit(1)


@sandbox_app.command("status")
def sandbox_status(
    session_id: Optional[str] = typer.Option(None, "--session-id", "-s", help="Session ID to check (default: all active sessions)")
):
    """Show sandbox session status."""
    
    try:
        sandbox_manager = get_sandbox_manager()
        
        if session_id:
            # Show specific session status
            session = sandbox_manager.get_session(session_id)
            if not session:
                console.print(f"❌ Session {session_id} not found.", style="red")
                raise typer.Exit(1)
            
            # Show detailed session status
            console.print(f"📊 Session Status: {session_id}", style="bold blue")
            
            # Basic info
            info_table = Table(show_header=False, box=None)
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value", style="white")
            
            info_table.add_row("Session ID", session.session_id)
            info_table.add_row("Scenario", session.scenario.name)
            info_table.add_row("Description", session.scenario.description)
            info_table.add_row("Status", session.status.value)
            info_table.add_row("Phase", session.phase.value)
            info_table.add_row("Duration", f"{session.duration}s")
            info_table.add_row("Interactive", "Yes" if session.interactive else "No")
            info_table.add_row("Created", session.created_at.strftime("%Y-%m-%d %H:%M:%S"))
            info_table.add_row("Updated", session.updated_at.strftime("%Y-%m-%d %H:%M:%S"))
            
            if session.start_time:
                info_table.add_row("Started", session.start_time.strftime("%Y-%m-%d %H:%M:%S"))
            if session.end_time:
                info_table.add_row("Ended", session.end_time.strftime("%Y-%m-%d %H:%M:%S"))
            
            console.print(info_table)
            
            # Agents status
            if session.agents:
                console.print("\n🤖 Agents:", style="bold blue")
                agents_table = Table(show_header=False, box=None)
                agents_table.add_column("Agent", style="cyan")
                agents_table.add_column("Status", style="green")
                agents_table.add_column("Initialized", style="white")
                
                for agent_name, agent_data in session.agents.items():
                    agents_table.add_row(
                        agent_name,
                        agent_data.get("status", "unknown"),
                        agent_data.get("initialized_at", "unknown")
                    )
                
                console.print(agents_table)
            
            # Guards status
            if session.guards:
                console.print("\n🛡️ Guards:", style="bold blue")
                guards_table = Table(show_header=False, box=None)
                guards_table.add_column("Guard", style="cyan")
                guards_table.add_column("Status", style="green")
                guards_table.add_column("Activated", style="white")
                
                for guard_name, guard_data in session.guards.items():
                    guards_table.add_row(
                        guard_name,
                        guard_data.get("status", "unknown"),
                        guard_data.get("activated_at", "unknown")
                    )
                
                console.print(guards_table)
            
            # Recent events
            if session.events:
                console.print(f"\n📝 Recent Events ({len(session.events)}):", style="bold blue")
                events_table = Table()
                events_table.add_column("Timestamp", style="dim white", width=16)
                events_table.add_column("Type", style="cyan")
                events_table.add_column("Message", style="white")
                
                # Show last 10 events
                recent_events = session.events[-10:]
                for event in recent_events:
                    timestamp = event.get("timestamp", "unknown")
                    if timestamp != "unknown":
                        try:
                            dt = datetime.fromisoformat(timestamp)
                            timestamp = dt.strftime("%H:%M:%S")
                        except:
                            pass
                    
                    events_table.add_row(
                        timestamp,
                        event.get("type", "unknown"),
                        event.get("message", "No message")[:50] + "..." if len(event.get("message", "")) > 50 else event.get("message", "No message")
                    )
                
                console.print(events_table)
            
            # Security events
            security_events = [e for e in session.events if 'security' in e.get('type', '') or 'suspicious' in e.get('type', '') or 'malicious' in e.get('type', '')]
            if security_events:
                console.print(f"\n🚨 Security Events ({len(security_events)}):", style="bold red")
                security_table = Table()
                security_table.add_column("Timestamp", style="dim white", width=16)
                security_table.add_column("Type", style="red")
                security_table.add_column("Severity", style="yellow")
                security_table.add_column("Description", style="white")
                
                for event in security_events[-5:]:  # Show last 5 security events
                    timestamp = event.get("timestamp", "unknown")
                    if timestamp != "unknown":
                        try:
                            dt = datetime.fromisoformat(timestamp)
                            timestamp = dt.strftime("%H:%M:%S")
                        except:
                            pass
                    
                    security_table.add_row(
                        timestamp,
                        event.get("type", "unknown"),
                        event.get("severity", "unknown"),
                        event.get("description", "No description")[:40] + "..." if len(event.get("description", "")) > 40 else event.get("description", "No description")
                    )
                
                console.print(security_table)
        
        else:
            # Show all active sessions summary
            active_sessions = sandbox_manager.get_active_sessions()
            
            if not active_sessions:
                console.print("ℹ️ No active sandbox sessions", style="blue")
                return
            
            console.print(f"📊 Active Sessions ({len(active_sessions)}):", style="bold blue")
            
            summary_table = Table()
            summary_table.add_column("Session ID", style="cyan", width=12)
            summary_table.add_column("Scenario", style="green")
            summary_table.add_column("Status", style="yellow")
            summary_table.add_column("Phase", style="blue")
            summary_table.add_column("Duration", style="white")
            summary_table.add_column("Events", style="dim white")
            summary_table.add_column("Security", style="red")
            
            for session in active_sessions.values():
                security_count = len([e for e in session.events if 'security' in e.get('type', '') or 'suspicious' in e.get('type', '') or 'malicious' in e.get('type', '')])
                
                summary_table.add_row(
                    session.session_id[:8] + "...",
                    session.scenario.name,
                    session.status.value,
                    session.phase.value,
                    f"{session.duration}s",
                    str(len(session.events)),
                    str(security_count) if security_count > 0 else "0"
                )
            
            console.print(summary_table)
    
    except Exception as e:
        console.print(f"❌ Error getting sandbox status: {e}", style="red")
        logger.error(f"Sandbox status error: {e}")
        raise typer.Exit(1)


# Configuration Commands
@config_app.command("show")
def config_show(
    section: Optional[str] = typer.Option(None, "--section", "-s", help="Show specific configuration section")
):
    """Show current configuration."""
    config_loader = ConfigLoader()
    if not config_loader.load_config():
        console.print("❌ Failed to load configuration", style="red")
        raise typer.Exit(1)
    
    if section:
        # Show specific section
        config_data = config_loader.config.__dict__
        if hasattr(config_loader.config, section):
            section_data = getattr(config_loader.config, section)
            console.print(f"📋 Configuration Section: {section}", style="bold blue")
            console.print(Syntax(str(section_data), "yaml", theme="monokai"))
        else:
            console.print(f"❌ Section '{section}' not found", style="red")
    else:
        # Show all configuration
        console.print("📋 Current Configuration", style="bold blue")
        config_data = config_loader.config.__dict__
        console.print(Syntax(str(config_data), "yaml", theme="monokai"))


@config_app.command("validate")
def config_validate(
    config_file: str = typer.Option("safehive/config/default_config.yaml", "--file", "-f", help="Configuration file to validate"),
    fix: bool = typer.Option(False, "--fix", help="Automatically fix configuration issues"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed validation results")
):
    """🔍 Validate configuration file syntax and settings with enhanced validation.
    
    Performs comprehensive validation including schema checking, value validation,
    and cross-section dependency verification.
    
    Examples:
        safehive config validate
        safehive config validate --file custom_config.yaml --detailed
        safehive config validate --fix
    """
    console.print(f"🔍 Validating configuration file: {config_file}", style="blue")
    
    try:
        from safehive.config.config_validator import validate_config_file
        from safehive.config.config_tools import get_config_tools
        
        # Perform comprehensive validation
        validation_result = validate_config_file(config_file)
        
        if validation_result.is_valid:
            console.print("✅ Configuration is valid", style="green")
            
            if validation_result.warnings:
                console.print(f"⚠️ Configuration has {len(validation_result.warnings)} warnings", style="yellow")
                if detailed:
                    for warning in validation_result.warnings:
                        console.print(f"  • {warning}")
        else:
            console.print(f"❌ Configuration validation failed", style="red")
            
            if detailed:
                # Show detailed results
                summary = validation_result.get_summary()
                console.print(f"\n📊 Validation Summary:")
                console.print(f"  • Total Issues: {summary['total_issues']}")
                console.print(f"  • Errors: {summary['errors']}")
                console.print(f"  • Warnings: {summary['warnings']}")
                
                auto_fixes = len([i for i in validation_result.issues if i.fix_available])
                console.print(f"  • Auto-fixes Available: {auto_fixes}")
                
                if validation_result.errors:
                    console.print(f"\n❌ Errors:")
                    for error in validation_result.errors:
                        console.print(f"  • {error}")
                
                if validation_result.warnings:
                    console.print(f"\n⚠️ Warnings:")
                    for warning in validation_result.warnings:
                        console.print(f"  • {warning}")
            
            if fix:
                console.print(f"\n🔧 Attempting to fix configuration...")
                tools = get_config_tools()
                success = tools.fix_configuration(config_file, backup=True)
                if success:
                    console.print("✅ Configuration fixed successfully", style="green")
                else:
                    console.print("⚠️ Some issues could not be automatically fixed", style="yellow")
                    raise typer.Exit(1)
            else:
                auto_fixes = len([i for i in validation_result.issues if i.fix_available])
                if auto_fixes > 0:
                    console.print(f"\n💡 Use --fix to automatically fix {auto_fixes} issues")
                raise typer.Exit(1)
    
    except FileNotFoundError:
        console.print(f"❌ Configuration file not found: {config_file}", style="red")
        console.print(f"💡 Use 'safehive init' to create default configuration")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Configuration validation failed: {e}", style="red")
        logger.error(f"Config validation error: {e}")
        raise typer.Exit(1)


# Agent Commands
@agent_app.command("list")
def agent_list():
    """List available agents."""
    agents = [
        ("orchestrator", "Main orchestrator agent", "🔄"),
        ("user-twin", "User twin agent", "👤"),
        ("honest-vendor", "Honest vendor agent", "✅"),
        ("malicious-vendor", "Malicious vendor agent", "⚠️")
    ]
    
    table = Table(title="Available Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Status", style="green")
    
    for name, description, status in agents:
        table.add_row(name, description, status)
    
    console.print(table)


@agent_app.command("status")
def agent_status(
    agent_id: Optional[str] = typer.Option(None, "--agent", "-a", help="Check specific agent status")
):
    """Show agent status."""
    if agent_id:
        console.print(f"📊 Agent Status: {agent_id}", style="bold blue")
        # TODO: Implement specific agent status check
        console.print("🚧 Agent status logic will be implemented in future tasks", style="yellow")
    else:
        console.print("📊 All Agents Status", style="bold blue")
        # TODO: Implement all agents status check
        console.print("🚧 Agent status logic will be implemented in future tasks", style="yellow")


# Guard Commands
@guard_app.command("list")
def guard_list():
    """List available security guards."""
    guards = [
        ("privacy-sentry", "Privacy Sentry", "🛡️", "Monitors data privacy and PII protection"),
        ("task-navigator", "Task Navigator", "🧭", "Guides AI agents through safe task execution"),
        ("prompt-sanitizer", "Prompt Sanitizer", "🧹", "Sanitizes and validates AI prompts"),
        ("mcp-server", "MCP Server", "🚀", "DoorDash integration for live ordering")
    ]
    
    table = Table(title="Available Security Guards")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Icon", style="green")
    table.add_column("Description", style="white")
    
    for guard_id, name, icon, description in guards:
        table.add_row(guard_id, name, icon, description)
    
    console.print(table)


@guard_app.command("status")
def guard_status(
    guard_id: Optional[str] = typer.Option(None, "--guard", "-g", help="Check specific guard status")
):
    """Show guard status."""
    if guard_id:
        console.print(f"📊 Guard Status: {guard_id}", style="bold blue")
        # TODO: Implement specific guard status check
        console.print("🚧 Guard status logic will be implemented in future tasks", style="yellow")
    else:
        console.print("📊 All Guards Status", style="bold blue")
        # TODO: Implement all guards status check
        console.print("🚧 Guard status logic will be implemented in future tasks", style="yellow")


# Metrics Commands
@metrics_app.command("show")
def metrics_show(
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, summary)"),
    period: str = typer.Option("1h", "--period", "-p", help="Time period for metrics"),
    category: str = typer.Option("all", "--category", "-c", help="Metrics category (all, system, security, agents)")
):
    """Show system metrics."""
    console.print(f"📊 System Metrics (Last {period})", style="bold blue")
    
    # Record metrics command usage
    record_metric("cli.metrics_show", 1, MetricType.COUNTER, {"format": format, "category": category})
    
    try:
        display = MetricsDisplay()
        
        if format == "json":
            # Export as JSON
            json_data = display.export_metrics_display("json")
            console.print(Syntax(json_data, "json", theme="monokai"))
        elif format == "summary":
            # Show summary format
            summary_text = display.export_metrics_display("summary")
            console.print(Panel(Text(summary_text, style="white"), title="Metrics Summary", border_style="blue"))
        else:
            # Show table format
            if category == "all":
                display.display_system_overview()
                console.print()
                display.display_metrics_table()
            elif category == "system":
                display.display_system_overview()
                console.print()
                display.display_counters()
                console.print()
                display.display_gauges()
                console.print()
                display.display_timers()
            elif category == "security":
                display.display_security_metrics()
            elif category == "agents":
                display.display_agent_metrics()
            else:
                console.print(f"[red]Unknown category: {category}[/red]")
                console.print("Available categories: all, system, security, agents")
    
    except Exception as e:
        console.print(f"[red]Error displaying metrics: {e}[/red]")
        logger.error(f"Metrics display error: {e}")


@metrics_app.command("export")
def metrics_export(
    output_file: str = typer.Option("metrics.json", "--output", "-o", help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Export format (json, summary)")
):
    """Export metrics to file."""
    console.print(f"📤 Exporting metrics to {output_file}", style="blue")
    
    # Record metrics export usage
    record_metric("cli.metrics_export", 1, MetricType.COUNTER, {"format": format})
    
    try:
        collector = get_metrics_collector()
        success = collector.save_metrics(output_file)
        
        if success:
            console.print(f"✅ Metrics exported successfully to {output_file}", style="green")
        else:
            console.print(f"❌ Failed to export metrics to {output_file}", style="red")
    
    except Exception as e:
        console.print(f"[red]Error exporting metrics: {e}[/red]")
        logger.error(f"Metrics export error: {e}")


@metrics_app.command("dashboard")
def metrics_dashboard(
    refresh_interval: float = typer.Option(2.0, "--refresh", "-r", help="Refresh interval in seconds")
):
    """Show real-time metrics dashboard."""
    console.print("📊 Starting real-time metrics dashboard...", style="bold blue")
    console.print("Press Ctrl+C to exit", style="dim")
    
    # Record dashboard usage
    record_metric("cli.metrics_dashboard", 1, MetricType.COUNTER, {"refresh_interval": refresh_interval})
    
    try:
        display = MetricsDisplay()
        display.display_metrics_dashboard(refresh_interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error running dashboard: {e}[/red]")
        logger.error(f"Metrics dashboard error: {e}")


@metrics_app.command("clear")
def metrics_clear():
    """Clear all metrics data."""
    if Confirm.ask("Are you sure you want to clear all metrics data?"):
        try:
            collector = get_metrics_collector()
            collector.clear_metrics()
            console.print("✅ All metrics cleared successfully", style="green")
        except Exception as e:
            console.print(f"[red]Error clearing metrics: {e}[/red]")
            logger.error(f"Metrics clear error: {e}")
    else:
        console.print("Metrics clear cancelled", style="yellow")


# Progress Commands
@progress_app.command("start")
def progress_start():
    """Start real-time progress monitoring."""
    console.print("🚀 Starting real-time progress monitoring...", style="bold blue")
    
    try:
        display = start_progress_display()
        
        # Add some initial events
        add_progress_event(ProgressEventType.SESSION_START, "Progress monitoring started")
        
        console.print("✅ Progress monitoring started. Press Ctrl+C to stop.", style="green")
        console.print("Use 'safehive progress stop' to stop monitoring.", style="dim")
        
        # Keep the display running
        try:
            while True:
                time.sleep(1)
                display.update_display()
        except KeyboardInterrupt:
            console.print("\n🛑 Stopping progress monitoring...", style="yellow")
            stop_progress_display()
            console.print("✅ Progress monitoring stopped", style="green")
            
    except Exception as e:
        console.print(f"❌ Error starting progress monitoring: {e}", style="red")
        logger.error(f"Progress start error: {e}")
        raise typer.Exit(1)


@progress_app.command("stop")
def progress_stop():
    """Stop real-time progress monitoring."""
    console.print("🛑 Stopping progress monitoring...", style="yellow")
    
    try:
        stop_progress_display()
        console.print("✅ Progress monitoring stopped", style="green")
    except Exception as e:
        console.print(f"❌ Error stopping progress monitoring: {e}", style="red")
        logger.error(f"Progress stop error: {e}")


@progress_app.command("status")
def progress_status():
    """Show current progress status."""
    console.print("📊 Current Progress Status", style="bold blue")
    
    try:
        display = get_progress_display()
        
        # Show session information
        if display._session_tasks:
            console.print("\n🔄 Active Sessions:", style="green")
            for session_id, task_info in display._session_tasks.items():
                session = task_info.get("session")
                if session:
                    console.print(f"  • {session_id[:8]}... - {session.status.value} ({session.phase.value})")
        else:
            console.print("\n💤 No active sessions", style="dim")
        
        # Show recent events
        if display.events:
            console.print(f"\n📝 Recent Events ({len(display.events)} total):", style="green")
            recent_events = display.events[-5:]
            for event in recent_events:
                time_str = event.timestamp.strftime("%H:%M:%S")
                console.print(f"  • [{time_str}] {event.event_type.value}: {event.message}")
        else:
            console.print("\n📝 No recent events", style="dim")
            
    except Exception as e:
        console.print(f"❌ Error getting progress status: {e}", style="red")
        logger.error(f"Progress status error: {e}")


@progress_app.command("demo")
def progress_demo():
    """Demonstrate progress monitoring with a simulated session."""
    console.print("🎬 Running Progress Demo...", style="bold blue")
    
    try:
        display = start_progress_display()
        
        # Simulate a session
        console.print("Creating demo session...", style="dim")
        add_progress_event(ProgressEventType.SESSION_START, "Demo session created")
        
        # Simulate session phases
        phases = [
            ("Initialization", "Setting up demo environment"),
            ("Agent Setup", "Initializing demo agents"),
            ("Guard Activation", "Activating security guards"),
            ("Scenario Execution", "Running demo scenario"),
            ("Monitoring", "Monitoring demo session"),
            ("Cleanup", "Cleaning up demo session")
        ]
        
        for i, (phase_name, description) in enumerate(phases):
            time.sleep(2)
            add_progress_event(ProgressEventType.SESSION_UPDATE, description)
            display.update_display()
            
            console.print(f"  ✓ {phase_name}", style="green")
        
        # Complete the demo
        time.sleep(1)
        add_progress_event(ProgressEventType.SESSION_COMPLETE, "Demo session completed successfully")
        
        console.print("\n✅ Progress demo completed!", style="green")
        console.print("Press Ctrl+C to stop monitoring or use 'safehive progress stop'", style="dim")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
                display.update_display()
        except KeyboardInterrupt:
            console.print("\n🛑 Stopping demo...", style="yellow")
            stop_progress_display()
            console.print("✅ Demo stopped", style="green")
            
    except Exception as e:
        console.print(f"❌ Error running progress demo: {e}", style="red")
        logger.error(f"Progress demo error: {e}")
        raise typer.Exit(1)


# Main entry point
def main():
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="yellow")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        logger.error(f"CLI error: {e}")
        raise typer.Exit(1)


# Human Controls Commands

@human_app.command("list")
def human_list(
    session_id: Optional[str] = typer.Option(None, "--session-id", "-s", help="Filter by session ID")
):
    """List pending human intervention requests."""
    console.print("🚨 Human Intervention Requests", style="bold red")
    
    try:
        ui = get_human_controls_ui()
        ui.display_pending_requests(session_id)
    except Exception as e:
        console.print(f"❌ Error listing intervention requests: {e}", style="red")
        logger.error(f"Human controls list error: {e}")
        raise typer.Exit(1)


@human_app.command("respond")
def human_respond(
    request_id: str = typer.Argument(..., help="Request ID to respond to")
):
    """Respond to a human intervention request."""
    console.print(f"🎯 Responding to intervention request: {request_id}", style="bold blue")
    
    try:
        ui = get_human_controls_ui()
        success = ui.prompt_for_intervention(request_id)
        
        if success:
            console.print("✅ Intervention response processed successfully", style="green")
        else:
            console.print("❌ Failed to process intervention response", style="red")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"❌ Error responding to intervention: {e}", style="red")
        logger.error(f"Human controls respond error: {e}")
        raise typer.Exit(1)


@human_app.command("details")
def human_details(
    request_id: str = typer.Argument(..., help="Request ID to show details for")
):
    """Show detailed information about an intervention request."""
    console.print(f"📋 Intervention Request Details: {request_id}", style="bold blue")
    
    try:
        ui = get_human_controls_ui()
        request = ui.display_request_details(request_id)
        
        if not request:
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"❌ Error showing request details: {e}", style="red")
        logger.error(f"Human controls details error: {e}")
        raise typer.Exit(1)


@human_app.command("stats")
def human_stats(
    session_id: Optional[str] = typer.Option(None, "--session-id", "-s", help="Show stats for specific session")
):
    """Show human controls statistics."""
    console.print("📊 Human Controls Statistics", style="bold blue")
    
    try:
        from safehive.ui.human_controls_ui import display_statistics
        display_statistics(session_id)
    except Exception as e:
        console.print(f"❌ Error showing statistics: {e}", style="red")
        logger.error(f"Human controls stats error: {e}")
        raise typer.Exit(1)


@human_app.command("monitor")
def human_monitor(
    session_id: Optional[str] = typer.Option(None, "--session-id", "-s", help="Monitor specific session"),
    refresh_interval: float = typer.Option(2.0, "--interval", "-i", help="Refresh interval in seconds")
):
    """Start interactive monitoring of intervention requests."""
    console.print("🔍 Starting interactive monitor...", style="bold blue")
    console.print("Press Ctrl+C to exit", style="dim")
    
    try:
        ui = get_human_controls_ui()
        ui.interactive_monitor(session_id, refresh_interval)
    except KeyboardInterrupt:
        console.print("\n👋 Monitor stopped", style="yellow")
    except Exception as e:
        console.print(f"❌ Error in monitor: {e}", style="red")
        logger.error(f"Human controls monitor error: {e}")
        raise typer.Exit(1)


@human_app.command("approve")
def human_approve(
    request_id: str = typer.Argument(..., help="Request ID to approve"),
    reason: Optional[str] = typer.Option(None, "--reason", "-r", help="Reason for approval")
):
    """Quick approve an intervention request."""
    console.print(f"✅ Approving request: {request_id}", style="green")
    
    try:
        manager = get_human_control_manager()
        success = manager.respond_to_intervention(
            request_id=request_id,
            intervention_type=InterventionType.APPROVE,
            reason=reason
        )
        
        if success:
            console.print("✅ Request approved successfully", style="green")
        else:
            console.print("❌ Failed to approve request", style="red")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"❌ Error approving request: {e}", style="red")
        logger.error(f"Human controls approve error: {e}")
        raise typer.Exit(1)


@human_app.command("quarantine")
def human_quarantine(
    request_id: str = typer.Argument(..., help="Request ID to quarantine"),
    duration: int = typer.Option(60, "--duration", "-d", help="Quarantine duration in minutes"),
    reason: Optional[str] = typer.Option(None, "--reason", "-r", help="Reason for quarantine")
):
    """Quick quarantine an intervention request."""
    console.print(f"🚫 Quarantining request: {request_id}", style="red")
    
    try:
        manager = get_human_control_manager()
        success = manager.respond_to_intervention(
            request_id=request_id,
            intervention_type=InterventionType.QUARANTINE,
            reason=reason,
            quarantine_duration=duration * 60,  # Convert to seconds
            quarantine_reason=reason or "Manual quarantine"
        )
        
        if success:
            console.print(f"✅ Request quarantined for {duration} minutes", style="green")
        else:
            console.print("❌ Failed to quarantine request", style="red")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"❌ Error quarantining request: {e}", style="red")
        logger.error(f"Human controls quarantine error: {e}")
        raise typer.Exit(1)


@human_app.command("ignore")
def human_ignore(
    request_id: str = typer.Argument(..., help="Request ID to ignore"),
    reason: Optional[str] = typer.Option(None, "--reason", "-r", help="Reason for ignoring")
):
    """Quick ignore an intervention request."""
    console.print(f"👁️ Ignoring request: {request_id}", style="dim")
    
    try:
        manager = get_human_control_manager()
        success = manager.respond_to_intervention(
            request_id=request_id,
            intervention_type=InterventionType.IGNORE,
            reason=reason
        )
        
        if success:
            console.print("✅ Request ignored successfully", style="green")
        else:
            console.print("❌ Failed to ignore request", style="red")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"❌ Error ignoring request: {e}", style="red")
        logger.error(f"Human controls ignore error: {e}")
        raise typer.Exit(1)


# Help Commands
@app.command("help", context_settings={"allow_extra_args": True})
def help_command(
    ctx: typer.Context,
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Topic to get help for"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Launch interactive help")
):
    """Show help information for commands and topics.
    
    Examples:
        safehive help
        safehive help sandbox start
        safehive help --topic scenarios
        safehive help --interactive
    """
    try:
        if interactive:
            interactive_help()
        else:
            # Handle multi-word commands from extra args
            command = None
            if ctx.args:
                command = " ".join(ctx.args)
            show_help(command, topic)
    except Exception as e:
        console.print(f"❌ Error showing help: {e}", style="red")
        logger.error(f"Help command error: {e}")
        raise typer.Exit(1)


@app.command("suggest")
def suggest_command(
    partial: str = typer.Argument(..., help="Partial command to get suggestions for")
):
    """Get command suggestions for partial input."""
    try:
        suggestions = get_command_suggestions(partial)
        if suggestions:
            console.print(f"Suggestions for '{partial}':", style="bold")
            for suggestion in suggestions:
                console.print(f"  • [cyan]{suggestion}[/cyan]")
        else:
            console.print(f"No suggestions found for '{partial}'", style="dim")
    except Exception as e:
        console.print(f"❌ Error getting suggestions: {e}", style="red")
        logger.error(f"Suggest command error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    main()
