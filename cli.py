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

# Add sub-apps to main app
app.add_typer(sandbox_app, name="sandbox")
app.add_typer(config_app, name="config")
app.add_typer(agent_app, name="agent")
app.add_typer(guard_app, name="guard")
app.add_typer(metrics_app, name="metrics")

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
    • Four AI Security Guards (Privacy Sentry, Task Navigator, Prompt Sanitizer, Honeypot Guard)
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
    duration: int = typer.Option(300, "--duration", "-d", help="Session duration in seconds"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Interactive mode")
):
    """Start a security sandbox session."""
    console.print(f"🎯 Starting sandbox session: {scenario}", style="bold blue")
    
    if not check_system_requirements():
        console.print("❌ System requirements not met.", style="red")
        raise typer.Exit(1)
    
    console.print(f"⏱️  Session duration: {duration} seconds")
    console.print(f"🔄 Interactive mode: {'Enabled' if interactive else 'Disabled'}")
    
    # TODO: Implement actual sandbox session logic
    console.print("🚧 Sandbox session logic will be implemented in future tasks", style="yellow")


@sandbox_app.command("stop")
def sandbox_stop():
    """Stop the current sandbox session."""
    console.print("🛑 Stopping sandbox session...", style="yellow")
    # TODO: Implement session stop logic
    console.print("✅ Sandbox session stopped", style="green")


@sandbox_app.command("list")
def sandbox_list():
    """List available sandbox scenarios."""
    scenarios = [
        ("food-ordering", "Food ordering workflow with malicious vendors"),
        ("payment-processing", "Payment processing with security testing"),
        ("api-integration", "API integration security testing"),
        ("data-extraction", "Data extraction and privacy testing")
    ]
    
    table = Table(title="Available Sandbox Scenarios")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    
    for name, description in scenarios:
        table.add_row(name, description)
    
    console.print(table)


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
        ("honeypot-guard", "Honeypot Guard", "🍯", "Detects and responds to malicious interactions")
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


if __name__ == "__main__":
    main()
