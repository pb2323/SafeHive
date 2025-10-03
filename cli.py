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
def status():
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


@app.command()
def init(
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Interactive mode")
):
    """Initialize the SafeHive system."""
    console.print("🚀 Initializing SafeHive AI Security Sandbox...", style="bold blue")
    
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
        raise typer.Exit(1)
    
    console.print("✅ SafeHive system initialized successfully!", style="green")
    
    if interactive:
        console.print("\n🎯 Ready to start sandbox operations!")
        console.print("Use 'safehive sandbox start' to begin a security testing session.")


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
    
    config_loader = ConfigLoader()
    if config_loader.load_config(config_file):
        console.print("✅ Configuration is valid", style="green")
    else:
        console.print("❌ Configuration validation failed", style="red")
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
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, csv)"),
    period: str = typer.Option("1h", "--period", "-p", help="Time period for metrics")
):
    """Show system metrics."""
    console.print(f"📊 System Metrics (Last {period})", style="bold blue")
    
    # TODO: Implement actual metrics collection
    console.print("🚧 Metrics collection will be implemented in future tasks", style="yellow")


@metrics_app.command("export")
def metrics_export(
    output_file: str = typer.Option("metrics.json", "--output", "-o", help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Export format (json, csv)")
):
    """Export metrics to file."""
    console.print(f"📤 Exporting metrics to {output_file}", style="blue")
    
    # TODO: Implement metrics export
    console.print("🚧 Metrics export will be implemented in future tasks", style="yellow")


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
