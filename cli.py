#!/usr/bin/env python3
"""
SafeHive AI Security Sandbox - Main CLI Entry Point

This is the main entry point for the SafeHive AI Security Sandbox.
It provides a command-line interface for launching and controlling
the AI security sandbox environment.
"""

import typer
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

# Create the main typer app
app = typer.Typer(
    name="safehive",
    help="SafeHive AI Security Sandbox - Interactive AI Security Testing Platform",
    add_completion=False,
    rich_markup_mode="rich"
)

# Create sub-apps for different modules
sandbox_app = typer.Typer(name="sandbox", help="Sandbox operations")
config_app = typer.Typer(name="config", help="Configuration management")
agent_app = typer.Typer(name="agent", help="Agent management")
guard_app = typer.Typer(name="guard", help="Security guard operations")
metrics_app = typer.Typer(name="metrics", help="Metrics and monitoring")
human_app = typer.Typer(name="human", help="Human-in-the-loop controls")

# Add sub-apps to main app
app.add_typer(sandbox_app, name="sandbox")
app.add_typer(config_app, name="config")
app.add_typer(agent_app, name="agent")
app.add_typer(guard_app, name="guard")
app.add_typer(metrics_app, name="metrics")
app.add_typer(human_app, name="human")

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
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Interactive mode")
):
    """Initialize the SafeHive system."""
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
    scenario: str = typer.Option("food-ordering", "--scenario", "-s", help="Test scenario to run"),
    duration: int = typer.Option(None, "--duration", "-d", help="Session duration in seconds"),
    interactive: bool = typer.Option(None, "--interactive/--no-interactive", help="Interactive mode"),
    background: bool = typer.Option(False, "--background", "-b", help="Run in background")
):
    """Start a security sandbox session."""
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
        
        if background:
            # Start session in background
            console.print("🚀 Starting session in background...", style="blue")
            asyncio.run(sandbox_manager.start_session(session.session_id))
            console.print(f"✅ Session {session.session_id} started in background", style="green")
        else:
            # Start session and wait
            console.print("🚀 Starting session...", style="blue")
            asyncio.run(sandbox_manager.start_session(session.session_id))
            
            # Show session status
            console.print("\n📊 Session Status:", style="bold blue")
            session = sandbox_manager.get_session(session.session_id)
            if session:
                console.print(f"  Status: {session.status.value}")
                console.print(f"  Phase: {session.phase.value}")
                console.print(f"  Duration: {session.duration}s")
                console.print(f"  Events: {len(session.events)}")
                console.print(f"  Security Events: {len([e for e in session.events if 'security' in e.get('type', '')])}")
    
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
    config_file: str = typer.Option("safehive/config/default_config.yaml", "--file", "-f", help="Configuration file to validate")
):
    """Validate configuration file."""
    console.print(f"🔍 Validating configuration file: {config_file}", style="blue")
    
    try:
        config_loader = ConfigLoader()
        config_loader.load_config()
        console.print("✅ Configuration is valid", style="green")
    except Exception as e:
        console.print(f"❌ Configuration validation failed: {e}", style="red")
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


if __name__ == "__main__":
    main()
