#!/usr/bin/env python3
"""
SafeHive AI Security Sandbox - Main CLI Entry Point

This is the main entry point for the SafeHive AI Security Sandbox.
It provides a command-line interface for launching and controlling
the AI security sandbox environment.
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

app = typer.Typer(
    name="safehive",
    help="SafeHive AI Security Sandbox - Interactive AI Security Testing Platform",
    add_completion=False
)

console = Console()

@app.command()
def version():
    """Show version information."""
    console.print(Panel(
        Text("SafeHive AI Security Sandbox\nVersion: 0.1.0", style="bold green"),
        title="Version Info",
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

if __name__ == "__main__":
    app()
