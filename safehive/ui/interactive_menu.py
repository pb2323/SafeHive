"""
Interactive Menu System for SafeHive AI Security Sandbox

This module provides an interactive menu system with rich formatting and colors
for navigating the SafeHive system. It offers a user-friendly interface for
all major operations including sandbox management, configuration, and monitoring.
"""

import asyncio
import sys
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich.markdown import Markdown
from rich.syntax import Syntax

from safehive.utils.logger import get_logger
from safehive.config.config_loader import ConfigLoader
from safehive.utils.ai_client import check_ollama_connection, ensure_model_available

logger = get_logger(__name__)


class InteractiveMenu:
    """
    Interactive menu system for SafeHive with rich formatting and colors.
    """
    
    def __init__(self):
        self.console = Console()
        self.config_loader = ConfigLoader()
        self.running = True
        self.current_session = None
        
        # Menu state
        self.current_menu = "main"
        self.menu_stack = []
        self.session_data = {}
        
        # Menu definitions
        self.menus = {
            "main": self._get_main_menu,
            "sandbox": self._get_sandbox_menu,
            "configuration": self._get_configuration_menu,
            "agents": self._get_agents_menu,
            "guards": self._get_guards_menu,
            "metrics": self._get_metrics_menu,
            "system": self._get_system_menu
        }
    
    def _get_main_menu(self) -> Dict[str, Any]:
        """Get the main menu configuration."""
        return {
            "title": "🛡️ SafeHive AI Security Sandbox",
            "subtitle": "Interactive Security Testing Platform",
            "options": [
                {
                    "key": "1",
                    "label": "🎯 Sandbox Operations",
                    "description": "Start, stop, and manage security testing sessions",
                    "action": "navigate",
                    "target": "sandbox"
                },
                {
                    "key": "2",
                    "label": "⚙️ Configuration",
                    "description": "View and modify system configuration",
                    "action": "navigate",
                    "target": "configuration"
                },
                {
                    "key": "3",
                    "label": "🤖 Agent Management",
                    "description": "Monitor and control AI agents",
                    "action": "navigate",
                    "target": "agents"
                },
                {
                    "key": "4",
                    "label": "🛡️ Security Guards",
                    "description": "View guard status and settings",
                    "action": "navigate",
                    "target": "guards"
                },
                {
                    "key": "5",
                    "label": "📊 Metrics & Monitoring",
                    "description": "View system metrics and logs",
                    "action": "navigate",
                    "target": "metrics"
                },
                {
                    "key": "6",
                    "label": "🔧 System Status",
                    "description": "Check system health and requirements",
                    "action": "navigate",
                    "target": "system"
                },
                {
                    "key": "i",
                    "label": "ℹ️ System Information",
                    "description": "Show system info and help",
                    "action": "show_info"
                },
                {
                    "key": "q",
                    "label": "🚪 Quit",
                    "description": "Exit SafeHive",
                    "action": "quit"
                }
            ]
        }
    
    def _get_sandbox_menu(self) -> Dict[str, Any]:
        """Get the sandbox menu configuration."""
        return {
            "title": "🎯 Sandbox Operations",
            "subtitle": "Security Testing Session Management",
            "options": [
                {
                    "key": "1",
                    "label": "🚀 Start New Session",
                    "description": "Launch a new security testing session",
                    "action": "start_session"
                },
                {
                    "key": "2",
                    "label": "📋 Available Scenarios",
                    "description": "View all available test scenarios",
                    "action": "list_scenarios"
                },
                {
                    "key": "3",
                    "label": "⏹️ Stop Current Session",
                    "description": "Stop the currently running session",
                    "action": "stop_session"
                },
                {
                    "key": "4",
                    "label": "📊 Session Status",
                    "description": "View current session information",
                    "action": "session_status"
                },
                {
                    "key": "5",
                    "label": "📝 Session History",
                    "description": "View previous session logs",
                    "action": "session_history"
                },
                {
                    "key": "b",
                    "label": "⬅️ Back to Main Menu",
                    "description": "Return to main menu",
                    "action": "back"
                },
                {
                    "key": "q",
                    "label": "🚪 Quit",
                    "description": "Exit SafeHive",
                    "action": "quit"
                }
            ]
        }
    
    def _get_configuration_menu(self) -> Dict[str, Any]:
        """Get the configuration menu."""
        return {
            "title": "⚙️ Configuration Management",
            "subtitle": "System Settings and Configuration",
            "options": [
                {
                    "key": "1",
                    "label": "👁️ View Configuration",
                    "description": "Display current system configuration",
                    "action": "view_config"
                },
                {
                    "key": "2",
                    "label": "🔍 Validate Configuration",
                    "description": "Check configuration for errors",
                    "action": "validate_config"
                },
                {
                    "key": "3",
                    "label": "🛡️ Guard Settings",
                    "description": "Configure security guard parameters",
                    "action": "guard_settings"
                },
                {
                    "key": "4",
                    "label": "🤖 Agent Settings",
                    "description": "Configure AI agent parameters",
                    "action": "agent_settings"
                },
                {
                    "key": "5",
                    "label": "📝 Logging Settings",
                    "description": "Configure logging and output",
                    "action": "logging_settings"
                },
                {
                    "key": "b",
                    "label": "⬅️ Back to Main Menu",
                    "description": "Return to main menu",
                    "action": "back"
                },
                {
                    "key": "q",
                    "label": "🚪 Quit",
                    "description": "Exit SafeHive",
                    "action": "quit"
                }
            ]
        }
    
    def _get_agents_menu(self) -> Dict[str, Any]:
        """Get the agents menu."""
        return {
            "title": "🤖 Agent Management",
            "subtitle": "AI Agent Monitoring and Control",
            "options": [
                {
                    "key": "1",
                    "label": "📋 List All Agents",
                    "description": "View all available AI agents",
                    "action": "list_agents"
                },
                {
                    "key": "2",
                    "label": "📊 Agent Status",
                    "description": "Check status of all agents",
                    "action": "agent_status"
                },
                {
                    "key": "3",
                    "label": "🔄 Orchestrator Agent",
                    "description": "Monitor orchestrator agent",
                    "action": "orchestrator_status"
                },
                {
                    "key": "4",
                    "label": "👤 User Twin Agent",
                    "description": "Monitor user twin agent",
                    "action": "user_twin_status"
                },
                {
                    "key": "5",
                    "label": "🏪 Vendor Agents",
                    "description": "Monitor vendor agents (honest/malicious)",
                    "action": "vendor_agents_status"
                },
                {
                    "key": "6",
                    "label": "💾 Agent Memory",
                    "description": "View agent conversation memory",
                    "action": "agent_memory"
                },
                {
                    "key": "b",
                    "label": "⬅️ Back to Main Menu",
                    "description": "Return to main menu",
                    "action": "back"
                },
                {
                    "key": "q",
                    "label": "🚪 Quit",
                    "description": "Exit SafeHive",
                    "action": "quit"
                }
            ]
        }
    
    def _get_guards_menu(self) -> Dict[str, Any]:
        """Get the guards menu."""
        return {
            "title": "🛡️ Security Guards",
            "subtitle": "Security Guard Monitoring and Configuration",
            "options": [
                {
                    "key": "1",
                    "label": "📋 List All Guards",
                    "description": "View all security guards",
                    "action": "list_guards"
                },
                {
                    "key": "2",
                    "label": "📊 Guard Status",
                    "description": "Check status of all guards",
                    "action": "guard_status"
                },
                {
                    "key": "3",
                    "label": "🛡️ Privacy Sentry",
                    "description": "Monitor privacy protection guard",
                    "action": "privacy_sentry_status"
                },
                {
                    "key": "4",
                    "label": "🧭 Task Navigator",
                    "description": "Monitor task guidance guard",
                    "action": "task_navigator_status"
                },
                {
                    "key": "5",
                    "label": "🧹 Prompt Sanitizer",
                    "description": "Monitor prompt validation guard",
                    "action": "prompt_sanitizer_status"
                },
                {
                    "key": "6",
                    "label": "🍯 Honeypot Guard",
                    "description": "Monitor honeypot detection guard",
                    "action": "honeypot_guard_status"
                },
                {
                    "key": "7",
                    "label": "🚨 Recent Alerts",
                    "description": "View recent security alerts",
                    "action": "recent_alerts"
                },
                {
                    "key": "b",
                    "label": "⬅️ Back to Main Menu",
                    "description": "Return to main menu",
                    "action": "back"
                },
                {
                    "key": "q",
                    "label": "🚪 Quit",
                    "description": "Exit SafeHive",
                    "action": "quit"
                }
            ]
        }
    
    def _get_metrics_menu(self) -> Dict[str, Any]:
        """Get the metrics menu."""
        return {
            "title": "📊 Metrics & Monitoring",
            "subtitle": "System Performance and Security Metrics",
            "options": [
                {
                    "key": "1",
                    "label": "📈 System Metrics",
                    "description": "View overall system performance",
                    "action": "system_metrics"
                },
                {
                    "key": "2",
                    "label": "🛡️ Security Metrics",
                    "description": "View security-related statistics",
                    "action": "security_metrics"
                },
                {
                    "key": "3",
                    "label": "🤖 Agent Metrics",
                    "description": "View AI agent performance metrics",
                    "action": "agent_metrics"
                },
                {
                    "key": "4",
                    "label": "📝 Log Analysis",
                    "description": "Analyze system logs",
                    "action": "log_analysis"
                },
                {
                    "key": "5",
                    "label": "📤 Export Metrics",
                    "description": "Export metrics to file",
                    "action": "export_metrics"
                },
                {
                    "key": "6",
                    "label": "📊 Real-time Dashboard",
                    "description": "Live monitoring dashboard",
                    "action": "realtime_dashboard"
                },
                {
                    "key": "b",
                    "label": "⬅️ Back to Main Menu",
                    "description": "Return to main menu",
                    "action": "back"
                },
                {
                    "key": "q",
                    "label": "🚪 Quit",
                    "description": "Exit SafeHive",
                    "action": "quit"
                }
            ]
        }
    
    def _get_system_menu(self) -> Dict[str, Any]:
        """Get the system menu."""
        return {
            "title": "🔧 System Status",
            "subtitle": "System Health and Requirements",
            "options": [
                {
                    "key": "1",
                    "label": "💚 System Health",
                    "description": "Check overall system health",
                    "action": "system_health"
                },
                {
                    "key": "2",
                    "label": "🔗 Ollama Status",
                    "description": "Check Ollama AI service status",
                    "action": "ollama_status"
                },
                {
                    "key": "3",
                    "label": "📦 Dependencies",
                    "description": "Check system dependencies",
                    "action": "dependencies_status"
                },
                {
                    "key": "4",
                    "label": "💾 Disk Space",
                    "description": "Check available disk space",
                    "action": "disk_space"
                },
                {
                    "key": "5",
                    "label": "🔄 System Logs",
                    "description": "View system logs",
                    "action": "system_logs"
                },
                {
                    "key": "6",
                    "label": "🛠️ System Info",
                    "description": "Display system information",
                    "action": "system_info"
                },
                {
                    "key": "b",
                    "label": "⬅️ Back to Main Menu",
                    "description": "Return to main menu",
                    "action": "back"
                },
                {
                    "key": "q",
                    "label": "🚪 Quit",
                    "description": "Exit SafeHive",
                    "action": "quit"
                }
            ]
        }
    
    def _display_menu(self, menu_config: Dict[str, Any]) -> None:
        """Display a menu with rich formatting."""
        # Clear screen
        self.console.clear()
        
        # Create title panel
        title_text = Text(menu_config["title"], style="bold blue")
        subtitle_text = Text(menu_config["subtitle"], style="dim white")
        
        title_panel = Panel(
            Align.center(title_text + "\n" + subtitle_text),
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(title_panel)
        self.console.print()
        
        # Create options table
        table = Table(
            title="Select an option:",
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
            title_style="bold blue"
        )
        
        table.add_column("Key", style="bold yellow", width=4)
        table.add_column("Option", style="white", width=30)
        table.add_column("Description", style="dim white", width=50)
        
        for option in menu_config["options"]:
            table.add_row(
                f"[bold yellow]{option['key']}[/bold yellow]",
                option["label"],
                option["description"]
            )
        
        self.console.print(table)
        self.console.print()
    
    def _get_user_choice(self) -> str:
        """Get user choice with validation."""
        while True:
            choice = Prompt.ask(
                "[bold cyan]Enter your choice[/bold cyan]",
                default="q"
            ).strip().lower()
            
            if choice:
                return choice
            
            self.console.print("[red]Please enter a valid choice.[/red]")
    
    def _handle_menu_action(self, action: str, target: Optional[str] = None) -> None:
        """Handle menu actions."""
        if action == "navigate":
            if target in self.menus:
                self.menu_stack.append(self.current_menu)
                self.current_menu = target
            else:
                self.console.print(f"[red]Unknown menu: {target}[/red]")
                self._wait_for_key()
        
        elif action == "back":
            if self.menu_stack:
                self.current_menu = self.menu_stack.pop()
            else:
                self.current_menu = "main"
        
        elif action == "quit":
            self.running = False
        
        elif action == "show_info":
            self._show_system_info()
        
        # Sandbox actions
        elif action == "start_session":
            self._start_sandbox_session()
        elif action == "list_scenarios":
            self._list_scenarios()
        elif action == "stop_session":
            self._stop_sandbox_session()
        elif action == "session_status":
            self._show_session_status()
        elif action == "session_history":
            self._show_session_history()
        
        # Configuration actions
        elif action == "view_config":
            self._view_configuration()
        elif action == "validate_config":
            self._validate_configuration()
        elif action == "guard_settings":
            self._show_guard_settings()
        elif action == "agent_settings":
            self._show_agent_settings()
        elif action == "logging_settings":
            self._show_logging_settings()
        
        # Agent actions
        elif action == "list_agents":
            self._list_agents()
        elif action == "agent_status":
            self._show_agent_status()
        elif action == "orchestrator_status":
            self._show_orchestrator_status()
        elif action == "user_twin_status":
            self._show_user_twin_status()
        elif action == "vendor_agents_status":
            self._show_vendor_agents_status()
        elif action == "agent_memory":
            self._show_agent_memory()
        
        # Guard actions
        elif action == "list_guards":
            self._list_guards()
        elif action == "guard_status":
            self._show_guard_status()
        elif action == "privacy_sentry_status":
            self._show_privacy_sentry_status()
        elif action == "task_navigator_status":
            self._show_task_navigator_status()
        elif action == "prompt_sanitizer_status":
            self._show_prompt_sanitizer_status()
        elif action == "honeypot_guard_status":
            self._show_honeypot_guard_status()
        elif action == "recent_alerts":
            self._show_recent_alerts()
        
        # Metrics actions
        elif action == "system_metrics":
            self._show_system_metrics()
        elif action == "security_metrics":
            self._show_security_metrics()
        elif action == "agent_metrics":
            self._show_agent_metrics()
        elif action == "log_analysis":
            self._show_log_analysis()
        elif action == "export_metrics":
            self._export_metrics()
        elif action == "realtime_dashboard":
            self._show_realtime_dashboard()
        
        # System actions
        elif action == "system_health":
            self._show_system_health()
        elif action == "ollama_status":
            self._show_ollama_status()
        elif action == "dependencies_status":
            self._show_dependencies_status()
        elif action == "disk_space":
            self._show_disk_space()
        elif action == "system_logs":
            self._show_system_logs()
        elif action == "system_info":
            self._show_system_info()
        
        else:
            self.console.print(f"[red]Unknown action: {action}[/red]")
            self._wait_for_key()
    
    def _wait_for_key(self) -> None:
        """Wait for user to press a key."""
        Prompt.ask("\n[dim]Press Enter to continue...[/dim]")
    
    def _show_system_info(self) -> None:
        """Show system information."""
        self.console.clear()
        
        info_text = """
# SafeHive AI Security Sandbox

## Overview
SafeHive is a CLI-based demonstration and testing platform that simulates a 
food-ordering workflow where AI assistants interact with potentially malicious 
vendors, payment services, and external APIs.

## Features
- **Four AI Security Guards**: Privacy Sentry, Task Navigator, Prompt Sanitizer, Honeypot Guard
- **LangChain-powered AI agents** with memory and reasoning
- **Interactive CLI** with human-in-the-loop controls
- **Real-time attack detection** and response
- **Comprehensive logging** and metrics

## System Requirements
- Python 3.8+
- Ollama AI service
- Required AI models (llama3.2:3b)
- Sufficient disk space for logs and models

## Getting Started
1. Ensure Ollama is running
2. Initialize the system with `init` command
3. Start a sandbox session
4. Monitor security guards and agents
5. Review metrics and logs

## Support
For more information, visit the project documentation or contact the development team.
        """
        
        self.console.print(Panel(
            Markdown(info_text),
            title="System Information",
            border_style="blue"
        ))
        
        self._wait_for_key()
    
    def _start_sandbox_session(self) -> None:
        """Start a new sandbox session."""
        self.console.clear()
        
        # Show available scenarios
        scenarios = [
            ("food-ordering", "Food ordering workflow with malicious vendors"),
            ("payment-processing", "Payment processing with security testing"),
            ("api-integration", "API integration security testing"),
            ("data-extraction", "Data extraction and privacy testing")
        ]
        
        table = Table(title="Available Scenarios")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Description", style="dim white")
        
        for i, (name, description) in enumerate(scenarios, 1):
            table.add_row(str(i), name, description)
        
        self.console.print(table)
        self.console.print()
        
        # Get scenario choice
        scenario_choice = IntPrompt.ask(
            "Select scenario (1-4)",
            default=1,
            choices=["1", "2", "3", "4"]
        )
        
        scenario = scenarios[scenario_choice - 1][0]
        
        # Get duration
        duration = IntPrompt.ask(
            "Session duration (seconds)",
            default=300
        )
        
        # Get interactive mode
        interactive = Confirm.ask("Enable interactive mode?", default=True)
        
        # Start session
        self.console.print(f"\n[green]Starting sandbox session: {scenario}[/green]")
        self.console.print(f"[blue]Duration: {duration} seconds[/blue]")
        self.console.print(f"[blue]Interactive: {'Yes' if interactive else 'No'}[/blue]")
        
        # TODO: Implement actual session start logic
        self.console.print("\n[yellow]🚧 Session start logic will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _list_scenarios(self) -> None:
        """List available scenarios."""
        self.console.clear()
        
        scenarios = [
            ("food-ordering", "Food ordering workflow with malicious vendors", "🛒"),
            ("payment-processing", "Payment processing with security testing", "💳"),
            ("api-integration", "API integration security testing", "🔌"),
            ("data-extraction", "Data extraction and privacy testing", "📊")
        ]
        
        table = Table(title="Available Sandbox Scenarios")
        table.add_column("Icon", style="green", width=4)
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white")
        
        for name, description, icon in scenarios:
            table.add_row(icon, name, description)
        
        self.console.print(table)
        self._wait_for_key()
    
    def _stop_sandbox_session(self) -> None:
        """Stop current sandbox session."""
        self.console.clear()
        
        if self.current_session:
            self.console.print("[yellow]Stopping current sandbox session...[/yellow]")
            # TODO: Implement actual session stop logic
            self.console.print("[green]✅ Sandbox session stopped[/green]")
            self.current_session = None
        else:
            self.console.print("[red]No active sandbox session to stop[/red]")
        
        self._wait_for_key()
    
    def _show_session_status(self) -> None:
        """Show current session status."""
        self.console.clear()
        
        if self.current_session:
            self.console.print("[green]Active Sandbox Session[/green]")
            # TODO: Show actual session details
            self.console.print("[yellow]🚧 Session status details will be implemented in future tasks[/yellow]")
        else:
            self.console.print("[red]No active sandbox session[/red]")
        
        self._wait_for_key()
    
    def _show_session_history(self) -> None:
        """Show session history."""
        self.console.clear()
        
        self.console.print("[blue]Session History[/blue]")
        # TODO: Show actual session history
        self.console.print("[yellow]🚧 Session history will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _view_configuration(self) -> None:
        """View system configuration."""
        self.console.clear()
        
        try:
            config = self.config_loader.load_config()
            
            # Show configuration sections
            sections = ["guards", "agents", "logging", "attack_simulation", "decoy_data"]
            
            table = Table(title="Configuration Sections")
            table.add_column("Section", style="cyan")
            table.add_column("Status", style="green")
            
            for section in sections:
                if hasattr(config, section):
                    table.add_row(section, "✅ Available")
                else:
                    table.add_row(section, "❌ Missing")
            
            self.console.print(table)
            
            # Ask for specific section
            section_choice = Prompt.ask(
                "\nEnter section name to view details (or 'all' for full config)",
                default="guards"
            )
            
            if section_choice.lower() == "all":
                config_data = config.__dict__
                self.console.print(Syntax(str(config_data), "yaml", theme="monokai"))
            elif hasattr(config, section_choice):
                section_data = getattr(config, section_choice)
                self.console.print(Syntax(str(section_data), "yaml", theme="monokai"))
            else:
                self.console.print(f"[red]Section '{section_choice}' not found[/red]")
                
        except Exception as e:
            self.console.print(f"[red]Error loading configuration: {e}[/red]")
        
        self._wait_for_key()
    
    def _validate_configuration(self) -> None:
        """Validate system configuration."""
        self.console.clear()
        
        self.console.print("[blue]Validating configuration...[/blue]")
        
        try:
            config = self.config_loader.load_config()
            self.console.print("[green]✅ Configuration is valid[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Configuration validation failed: {e}[/red]")
        
        self._wait_for_key()
    
    def _show_guard_settings(self) -> None:
        """Show guard settings."""
        self.console.clear()
        
        guards = [
            ("privacy-sentry", "Privacy Sentry", "🛡️", "Monitors data privacy and PII protection"),
            ("task-navigator", "Task Navigator", "🧭", "Guides AI agents through safe task execution"),
            ("prompt-sanitizer", "Prompt Sanitizer", "🧹", "Sanitizes and validates AI prompts"),
            ("honeypot-guard", "Honeypot Guard", "🍯", "Detects and responds to malicious interactions")
        ]
        
        table = Table(title="Security Guard Settings")
        table.add_column("Icon", style="green", width=4)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Description", style="dim white")
        
        for guard_id, name, icon, description in guards:
            table.add_row(icon, guard_id, name, description)
        
        self.console.print(table)
        self.console.print("[yellow]🚧 Guard settings configuration will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_agent_settings(self) -> None:
        """Show agent settings."""
        self.console.clear()
        
        agents = [
            ("orchestrator", "Main orchestrator agent", "🔄"),
            ("user-twin", "User twin agent", "👤"),
            ("honest-vendor", "Honest vendor agent", "✅"),
            ("malicious-vendor", "Malicious vendor agent", "⚠️")
        ]
        
        table = Table(title="AI Agent Settings")
        table.add_column("Icon", style="green", width=4)
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white")
        
        for name, description, icon in agents:
            table.add_row(icon, name, description)
        
        self.console.print(table)
        self.console.print("[yellow]🚧 Agent settings configuration will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_logging_settings(self) -> None:
        """Show logging settings."""
        self.console.clear()
        
        self.console.print("[blue]Logging Configuration[/blue]")
        self.console.print("[yellow]🚧 Logging settings will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _list_agents(self) -> None:
        """List all agents."""
        self.console.clear()
        
        agents = [
            ("orchestrator", "Main orchestrator agent", "🔄", "Active"),
            ("user-twin", "User twin agent", "👤", "Active"),
            ("honest-vendor", "Honest vendor agent", "✅", "Active"),
            ("malicious-vendor", "Malicious vendor agent", "⚠️", "Active")
        ]
        
        table = Table(title="Available AI Agents")
        table.add_column("Icon", style="green", width=4)
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        
        for name, description, icon, status in agents:
            table.add_row(icon, name, description, status)
        
        self.console.print(table)
        self._wait_for_key()
    
    def _show_agent_status(self) -> None:
        """Show agent status."""
        self.console.clear()
        
        self.console.print("[blue]AI Agent Status[/blue]")
        self.console.print("[yellow]🚧 Agent status monitoring will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_orchestrator_status(self) -> None:
        """Show orchestrator status."""
        self.console.clear()
        
        self.console.print("[blue]🔄 Orchestrator Agent Status[/blue]")
        self.console.print("[yellow]🚧 Orchestrator status will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_user_twin_status(self) -> None:
        """Show user twin status."""
        self.console.clear()
        
        self.console.print("[blue]👤 User Twin Agent Status[/blue]")
        self.console.print("[yellow]🚧 User twin status will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_vendor_agents_status(self) -> None:
        """Show vendor agents status."""
        self.console.clear()
        
        self.console.print("[blue]🏪 Vendor Agents Status[/blue]")
        self.console.print("[yellow]🚧 Vendor agents status will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_agent_memory(self) -> None:
        """Show agent memory."""
        self.console.clear()
        
        self.console.print("[blue]💾 Agent Memory Status[/blue]")
        self.console.print("[yellow]🚧 Agent memory monitoring will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _list_guards(self) -> None:
        """List all guards."""
        self.console.clear()
        
        guards = [
            ("privacy-sentry", "Privacy Sentry", "🛡️", "Monitors data privacy and PII protection"),
            ("task-navigator", "Task Navigator", "🧭", "Guides AI agents through safe task execution"),
            ("prompt-sanitizer", "Prompt Sanitizer", "🧹", "Sanitizes and validates AI prompts"),
            ("honeypot-guard", "Honeypot Guard", "🍯", "Detects and responds to malicious interactions")
        ]
        
        table = Table(title="Available Security Guards")
        table.add_column("Icon", style="green", width=4)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Description", style="dim white")
        
        for guard_id, name, icon, description in guards:
            table.add_row(icon, guard_id, name, description)
        
        self.console.print(table)
        self._wait_for_key()
    
    def _show_guard_status(self) -> None:
        """Show guard status."""
        self.console.clear()
        
        self.console.print("[blue]Security Guard Status[/blue]")
        self.console.print("[yellow]🚧 Guard status monitoring will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_privacy_sentry_status(self) -> None:
        """Show privacy sentry status."""
        self.console.clear()
        
        self.console.print("[blue]🛡️ Privacy Sentry Status[/blue]")
        self.console.print("[yellow]🚧 Privacy sentry status will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_task_navigator_status(self) -> None:
        """Show task navigator status."""
        self.console.clear()
        
        self.console.print("[blue]🧭 Task Navigator Status[/blue]")
        self.console.print("[yellow]🚧 Task navigator status will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_prompt_sanitizer_status(self) -> None:
        """Show prompt sanitizer status."""
        self.console.clear()
        
        self.console.print("[blue]🧹 Prompt Sanitizer Status[/blue]")
        self.console.print("[yellow]🚧 Prompt sanitizer status will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_honeypot_guard_status(self) -> None:
        """Show honeypot guard status."""
        self.console.clear()
        
        self.console.print("[blue]🍯 Honeypot Guard Status[/blue]")
        self.console.print("[yellow]🚧 Honeypot guard status will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_recent_alerts(self) -> None:
        """Show recent alerts."""
        self.console.clear()
        
        self.console.print("[blue]🚨 Recent Security Alerts[/blue]")
        self.console.print("[yellow]🚧 Recent alerts will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_system_metrics(self) -> None:
        """Show system metrics."""
        self.console.clear()
        
        self.console.print("[blue]📈 System Metrics[/blue]")
        self.console.print("[yellow]🚧 System metrics will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_security_metrics(self) -> None:
        """Show security metrics."""
        self.console.clear()
        
        self.console.print("[blue]🛡️ Security Metrics[/blue]")
        self.console.print("[yellow]🚧 Security metrics will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_agent_metrics(self) -> None:
        """Show agent metrics."""
        self.console.clear()
        
        self.console.print("[blue]🤖 Agent Metrics[/blue]")
        self.console.print("[yellow]🚧 Agent metrics will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_log_analysis(self) -> None:
        """Show log analysis."""
        self.console.clear()
        
        self.console.print("[blue]📝 Log Analysis[/blue]")
        self.console.print("[yellow]🚧 Log analysis will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _export_metrics(self) -> None:
        """Export metrics."""
        self.console.clear()
        
        self.console.print("[blue]📤 Export Metrics[/blue]")
        self.console.print("[yellow]🚧 Metrics export will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_realtime_dashboard(self) -> None:
        """Show real-time dashboard."""
        self.console.clear()
        
        self.console.print("[blue]📊 Real-time Dashboard[/blue]")
        self.console.print("[yellow]🚧 Real-time dashboard will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_system_health(self) -> None:
        """Show system health."""
        self.console.clear()
        
        # Check Ollama
        ollama_status = "🟢 Running" if check_ollama_connection() else "🔴 Not Running"
        
        # Check configuration
        try:
            self.config_loader.load_config()
            config_status = "🟢 Loaded"
        except Exception:
            config_status = "🔴 Failed"
        
        # Check logs directory
        logs_dir = Path("logs")
        logs_status = "🟢 Available" if logs_dir.exists() else "🔴 Not Found"
        
        table = Table(title="System Health Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")
        
        table.add_row("Ollama AI Service", ollama_status)
        table.add_row("Configuration", config_status)
        table.add_row("Logs Directory", logs_status)
        
        self.console.print(table)
        self._wait_for_key()
    
    def _show_ollama_status(self) -> None:
        """Show Ollama status."""
        self.console.clear()
        
        self.console.print("[blue]🔗 Ollama AI Service Status[/blue]")
        
        if check_ollama_connection():
            self.console.print("[green]✅ Ollama is running[/green]")
            
            # Check model availability
            if ensure_model_available("llama3.2:3b"):
                self.console.print("[green]✅ Default model is available[/green]")
            else:
                self.console.print("[yellow]⚠️ Default model not available[/yellow]")
        else:
            self.console.print("[red]❌ Ollama is not running[/red]")
            self.console.print("[yellow]Please start Ollama service first[/yellow]")
        
        self._wait_for_key()
    
    def _show_dependencies_status(self) -> None:
        """Show dependencies status."""
        self.console.clear()
        
        self.console.print("[blue]📦 System Dependencies[/blue]")
        self.console.print("[yellow]🚧 Dependencies check will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_disk_space(self) -> None:
        """Show disk space."""
        self.console.clear()
        
        self.console.print("[blue]💾 Disk Space Usage[/blue]")
        self.console.print("[yellow]🚧 Disk space monitoring will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_system_logs(self) -> None:
        """Show system logs."""
        self.console.clear()
        
        self.console.print("[blue]🔄 System Logs[/blue]")
        self.console.print("[yellow]🚧 System logs viewer will be implemented in future tasks[/yellow]")
        
        self._wait_for_key()
    
    def _show_system_info(self) -> None:
        """Show system information."""
        self.console.clear()
        
        import sys
        from safehive import __version__
        
        info_data = {
            "SafeHive Version": __version__,
            "Python Version": sys.version.split()[0],
            "Platform": sys.platform,
            "Current Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        table = Table(title="System Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        for key, value in info_data.items():
            table.add_row(key, str(value))
        
        self.console.print(table)
        self._wait_for_key()
    
    def run(self) -> None:
        """Run the interactive menu system."""
        try:
            while self.running:
                # Get current menu configuration
                menu_config = self.menus[self.current_menu]()
                
                # Display menu
                self._display_menu(menu_config)
                
                # Get user choice
                choice = self._get_user_choice()
                
                # Find selected option
                selected_option = None
                for option in menu_config["options"]:
                    if option["key"] == choice:
                        selected_option = option
                        break
                
                if selected_option:
                    # Handle the action
                    self._handle_menu_action(
                        selected_option["action"],
                        selected_option.get("target")
                    )
                else:
                    self.console.print(f"[red]Invalid choice: {choice}[/red]")
                    self._wait_for_key()
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]👋 Goodbye![/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Error: {e}[/red]")
            logger.error(f"Interactive menu error: {e}")


def main():
    """Main entry point for interactive menu."""
    menu = InteractiveMenu()
    menu.run()


if __name__ == "__main__":
    main()
