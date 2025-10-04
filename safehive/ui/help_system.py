"""
Help System for SafeHive AI Security Sandbox CLI.

This module provides comprehensive help documentation, examples, and interactive
help features for all CLI commands and functionality.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.columns import Columns
from rich.prompt import Prompt, Confirm
import typer
from pathlib import Path

@dataclass
class CommandHelp:
    """Represents help information for a CLI command."""
    name: str
    description: str
    usage: str
    examples: List[str]
    options: List[Dict[str, str]]
    notes: List[str] = None
    related_commands: List[str] = None

class HelpSystem:
    """Comprehensive help system for SafeHive CLI."""
    
    def __init__(self):
        self.console = Console()
        self.commands: Dict[str, CommandHelp] = {}
        self._initialize_command_help()
    
    def _initialize_command_help(self):
        """Initialize all command help information."""
        
        # Main Commands
        self.commands["init"] = CommandHelp(
            name="init",
            description="Initialize SafeHive AI Security Sandbox system",
            usage="safehive init [--force]",
            examples=[
                "safehive init",
                "safehive init --force"
            ],
            options=[
                {"--force": "Force reinitialization even if already configured"}
            ],
            notes=[
                "Checks Ollama installation and downloads required models",
                "Creates default configuration files",
                "Sets up logging directories"
            ]
        )
        
        self.commands["menu"] = CommandHelp(
            name="menu",
            description="Launch interactive menu system",
            usage="safehive menu",
            examples=[
                "safehive menu"
            ],
            options=[],
            notes=[
                "Provides guided access to all SafeHive features",
                "Includes system status and quick actions"
            ]
        )
        
        self.commands["status"] = CommandHelp(
            name="status",
            description="Show system status and health information",
            usage="safehive status [--metrics]",
            examples=[
                "safehive status",
                "safehive status --metrics"
            ],
            options=[
                {"--metrics": "Include detailed metrics summary"}
            ],
            notes=[
                "Shows Ollama connection status",
                "Displays active sessions and agents",
                "Reports guard status and configuration"
            ]
        )
        
        # Sandbox Commands
        self.commands["sandbox start"] = CommandHelp(
            name="sandbox start",
            description="Start a new sandbox session with scenario execution",
            usage="safehive sandbox start --scenario <name> [options]",
            examples=[
                "safehive sandbox start --scenario food-ordering",
                "safehive sandbox start --scenario food-ordering --duration 300",
                "safehive sandbox start --scenario food-ordering --interactive",
                "safehive sandbox start --scenario food-ordering --background"
            ],
            options=[
                {"--scenario": "Scenario name to execute (required)"},
                {"--duration": "Session duration in seconds (default: 300)"},
                {"--interactive": "Run in interactive mode with user input"},
                {"--background": "Run in background mode"},
                {"--agents": "Comma-separated list of agents to include"},
                {"--guards": "Comma-separated list of guards to enable"}
            ],
            notes=[
                "Available scenarios: food-ordering",
                "Interactive mode allows real-time user input",
                "Background mode runs without blocking terminal"
            ]
        )
        
        self.commands["sandbox stop"] = CommandHelp(
            name="sandbox stop",
            description="Stop active sandbox sessions",
            usage="safehive sandbox stop [session-id] [--all]",
            examples=[
                "safehive sandbox stop",
                "safehive sandbox stop session-123",
                "safehive sandbox stop --all"
            ],
            options=[
                {"session-id": "Specific session ID to stop"},
                {"--all": "Stop all active sessions"}
            ]
        )
        
        self.commands["sandbox list"] = CommandHelp(
            name="sandbox list",
            description="List available scenarios or active sessions",
            usage="safehive sandbox list [--sessions]",
            examples=[
                "safehive sandbox list",
                "safehive sandbox list --sessions"
            ],
            options=[
                {"--sessions": "List active sessions instead of scenarios"}
            ]
        )
        
        self.commands["sandbox status"] = CommandHelp(
            name="sandbox status",
            description="Show detailed status of sandbox sessions",
            usage="safehive sandbox status [session-id]",
            examples=[
                "safehive sandbox status",
                "safehive sandbox status session-123"
            ],
            options=[
                {"session-id": "Specific session ID to show status for"}
            ]
        )
        
        # Configuration Commands
        self.commands["config show"] = CommandHelp(
            name="config show",
            description="Display current configuration settings",
            usage="safehive config show [--section <section>]",
            examples=[
                "safehive config show",
                "safehive config show --section guards",
                "safehive config show --section agents"
            ],
            options=[
                {"--section": "Specific configuration section to display"}
            ]
        )
        
        self.commands["config validate"] = CommandHelp(
            name="config validate",
            description="Validate configuration file syntax and settings",
            usage="safehive config validate [--file <path>]",
            examples=[
                "safehive config validate",
                "safehive config validate --file custom_config.yaml"
            ],
            options=[
                {"--file": "Path to configuration file to validate"}
            ]
        )
        
        # Metrics Commands
        self.commands["metrics show"] = CommandHelp(
            name="metrics show",
            description="Display system metrics and statistics",
            usage="safehive metrics show [--format <format>]",
            examples=[
                "safehive metrics show",
                "safehive metrics show --format json",
                "safehive metrics show --format summary"
            ],
            options=[
                {"--format": "Output format: table, json, summary"}
            ]
        )
        
        self.commands["metrics export"] = CommandHelp(
            name="metrics export",
            description="Export metrics to file",
            usage="safehive metrics export --file <path> [--format <format>]",
            examples=[
                "safehive metrics export --file metrics.json",
                "safehive metrics export --file metrics.csv --format csv"
            ],
            options=[
                {"--file": "Output file path"},
                {"--format": "Export format: json, csv, yaml"}
            ]
        )
        
        self.commands["metrics dashboard"] = CommandHelp(
            name="metrics dashboard",
            description="Launch real-time metrics dashboard",
            usage="safehive metrics dashboard",
            examples=[
                "safehive metrics dashboard"
            ],
            options=[],
            notes=[
                "Shows live metrics updates",
                "Press Ctrl+C to exit dashboard"
            ]
        )
        
        # Human Controls Commands
        self.commands["human list"] = CommandHelp(
            name="human list",
            description="List pending human intervention requests",
            usage="safehive human list [--session <session-id>]",
            examples=[
                "safehive human list",
                "safehive human list --session session-123"
            ],
            options=[
                {"--session": "Filter by specific session ID"}
            ]
        )
        
        self.commands["human respond"] = CommandHelp(
            name="human respond",
            description="Respond to a human intervention request",
            usage="safehive human respond <request-id> <action> [options]",
            examples=[
                "safehive human respond req-123 approve",
                "safehive human respond req-123 redact --rules 'email,phone'",
                "safehive human respond req-123 quarantine --duration 3600"
            ],
            options=[
                {"request-id": "Request ID to respond to"},
                {"action": "Action: approve, redact, quarantine, ignore"},
                {"--rules": "Redaction rules (comma-separated)"},
                {"--duration": "Quarantine duration in seconds"},
                {"--reason": "Reason for the action"}
            ]
        )
        
        # Progress Commands
        self.commands["progress start"] = CommandHelp(
            name="progress start",
            description="Start real-time progress monitoring",
            usage="safehive progress start",
            examples=[
                "safehive progress start"
            ],
            options=[],
            notes=[
                "Shows live progress updates for active sessions",
                "Displays agent status and guard activities"
            ]
        )
        
        self.commands["progress stop"] = CommandHelp(
            name="progress stop",
            description="Stop progress monitoring",
            usage="safehive progress stop",
            examples=[
                "safehive progress stop"
            ],
            options=[]
        )
        
        self.commands["progress demo"] = CommandHelp(
            name="progress demo",
            description="Demonstrate progress monitoring with simulated session",
            usage="safehive progress demo",
            examples=[
                "safehive progress demo"
            ],
            options=[],
            notes=[
                "Creates a simulated session for demonstration",
                "Shows various progress events and updates"
            ]
        )
    
    def show_help(self, command: Optional[str] = None, topic: Optional[str] = None):
        """Display help information for a command or topic."""
        if command:
            self._show_command_help(command)
        elif topic:
            self._show_topic_help(topic)
        else:
            self._show_general_help()
    
    def _show_general_help(self):
        """Display general help information."""
        self.console.print("\n[bold blue]SafeHive AI Security Sandbox[/bold blue]")
        self.console.print("Advanced AI security testing and simulation platform\n")
        
        # Quick Start
        self.console.print(Panel(
            "[bold]Quick Start:[/bold]\n"
            "1. [cyan]safehive init[/cyan] - Initialize the system\n"
            "2. [cyan]safehive menu[/cyan] - Launch interactive interface\n"
            "3. [cyan]safehive sandbox start --scenario food-ordering[/cyan] - Start testing",
            title="🚀 Getting Started",
            border_style="green"
        ))
        
        # Command Categories
        categories = {
            "System": ["init", "status", "menu"],
            "Sandbox": ["sandbox start", "sandbox stop", "sandbox list", "sandbox status"],
            "Configuration": ["config show", "config validate"],
            "Monitoring": ["metrics show", "metrics dashboard", "progress start", "progress demo"],
            "Controls": ["human list", "human respond"]
        }
        
        for category, commands in categories.items():
            table = Table(title=f"{category} Commands")
            table.add_column("Command", style="cyan")
            table.add_column("Description", style="white")
            
            for cmd in commands:
                if cmd in self.commands:
                    help_info = self.commands[cmd]
                    table.add_row(cmd, help_info.description)
            
            self.console.print(table)
        
        # Examples
        self.console.print(Panel(
            "[bold]Common Examples:[/bold]\n"
            "• [cyan]safehive sandbox start --scenario food-ordering --interactive[/cyan]\n"
            "• [cyan]safehive metrics dashboard[/cyan]\n"
            "• [cyan]safehive human list[/cyan]\n"
            "• [cyan]safehive config show --section guards[/cyan]",
            title="💡 Examples",
            border_style="blue"
        ))
        
        self.console.print("\n[dim]Use 'safehive help <command>' for detailed information about a specific command.[/dim]")
    
    def _show_command_help(self, command: str):
        """Display detailed help for a specific command."""
        # Handle multi-word commands by joining with spaces
        if command not in self.commands:
            # Try to find a matching command
            matching_commands = [cmd for cmd in self.commands.keys() if cmd.startswith(command)]
            if matching_commands:
                command = matching_commands[0]
            else:
                self.console.print(f"[red]Error: Unknown command '{command}'[/red]")
                self.console.print("Use 'safehive help' to see available commands.")
                return
        
        help_info = self.commands[command]
        
        # Command header
        self.console.print(f"\n[bold blue]{help_info.name}[/bold blue]")
        self.console.print(f"[white]{help_info.description}[/white]\n")
        
        # Usage
        self.console.print(Panel(
            help_info.usage,
            title="📝 Usage",
            border_style="green"
        ))
        
        # Examples
        if help_info.examples:
            examples_text = "\n".join([f"• [cyan]{ex}[/cyan]" for ex in help_info.examples])
            self.console.print(Panel(
                examples_text,
                title="💡 Examples",
                border_style="blue"
            ))
        
        # Options
        if help_info.options:
            table = Table(title="⚙️ Options")
            table.add_column("Option", style="cyan")
            table.add_column("Description", style="white")
            
            for option in help_info.options:
                for opt, desc in option.items():
                    table.add_row(opt, desc)
            
            self.console.print(table)
        
        # Notes
        if help_info.notes:
            notes_text = "\n".join([f"• {note}" for note in help_info.notes])
            self.console.print(Panel(
                notes_text,
                title="📋 Notes",
                border_style="yellow"
            ))
        
        # Related commands
        if help_info.related_commands:
            related_text = " ".join([f"[cyan]{cmd}[/cyan]" for cmd in help_info.related_commands])
            self.console.print(f"\n[bold]Related commands:[/bold] {related_text}")
    
    def _show_topic_help(self, topic: str):
        """Display help for a specific topic."""
        topics = {
            "scenarios": self._show_scenarios_help,
            "guards": self._show_guards_help,
            "agents": self._show_agents_help,
            "configuration": self._show_configuration_help,
            "troubleshooting": self._show_troubleshooting_help
        }
        
        if topic in topics:
            topics[topic]()
        else:
            self.console.print(f"[red]Error: Unknown topic '{topic}'[/red]")
            self.console.print("Available topics: scenarios, guards, agents, configuration, troubleshooting")
    
    def _show_scenarios_help(self):
        """Display help about available scenarios."""
        self.console.print(Panel(
            "[bold]Available Scenarios:[/bold]\n\n"
            "[cyan]food-ordering[/cyan] - Food ordering workflow with malicious vendors\n"
            "  • Tests privacy protection and data validation\n"
            "  • Simulates honest and malicious vendor interactions\n"
            "  • Includes payment processing and order validation\n"
            "  • Duration: 5 minutes (configurable)\n\n"
            "[bold]Usage:[/bold]\n"
            "• [cyan]safehive sandbox start --scenario food-ordering[/cyan]\n"
            "• [cyan]safehive sandbox start --scenario food-ordering --interactive[/cyan]",
            title="🎯 Scenarios",
            border_style="green"
        ))
    
    def _show_guards_help(self):
        """Display help about security guards."""
        self.console.print(Panel(
            "[bold]Security Guards:[/bold]\n\n"
            "[cyan]Privacy Sentry[/cyan] - Detects and prevents PII over-sharing\n"
            "  • Monitors credit card, address, and health data\n"
            "  • Provides redaction suggestions\n\n"
            "[cyan]Task Navigator[/cyan] - Enforces task constraints\n"
            "  • Validates budget limits and dietary requirements\n"
            "  • Prevents scope creep in agent conversations\n\n"
            "[cyan]Prompt Sanitizer[/cyan] - Filters malicious inputs\n"
            "  • Detects SQL injection and XSS attempts\n"
            "  • Validates input formats and patterns",
            title="🛡️ Security Guards",
            border_style="blue"
        ))
    
    def _show_agents_help(self):
        """Display help about AI agents."""
        self.console.print(Panel(
            "[bold]AI Agents:[/bold]\n\n"
            "[cyan]Orchestrator[/cyan] - Manages food ordering workflow\n"
            "  • Coordinates with vendors and user twin\n"
            "  • Handles order validation and confirmation\n\n"
            "[cyan]User Twin[/cyan] - Represents user preferences\n"
            "  • Manages dietary restrictions and budget\n"
            "  • Makes decisions based on user profile\n\n"
            "[cyan]Vendor Agents[/cyan] - Simulate restaurant interactions\n"
            "  • Honest vendors: Normal business behavior\n"
            "  • Malicious vendors: Attempt security attacks",
            title="🤖 AI Agents",
            border_style="yellow"
        ))
    
    def _show_configuration_help(self):
        """Display help about configuration."""
        self.console.print(Panel(
            "[bold]Configuration:[/bold]\n\n"
            "[cyan]Location:[/cyan] config/default_config.yaml\n"
            "[cyan]Sections:[/cyan]\n"
            "  • guards: Security guard settings\n"
            "  • agents: AI agent configurations\n"
            "  • logging: Logging levels and files\n"
            "  • metrics: Metrics collection settings\n\n"
            "[bold]Commands:[/bold]\n"
            "• [cyan]safehive config show[/cyan] - View current config\n"
            "• [cyan]safehive config validate[/cyan] - Validate config file",
            title="⚙️ Configuration",
            border_style="green"
        ))
    
    def _show_troubleshooting_help(self):
        """Display troubleshooting help."""
        self.console.print(Panel(
            "[bold]Common Issues:[/bold]\n\n"
            "[yellow]Ollama Connection Failed[/yellow]\n"
            "• Ensure Ollama is running: [cyan]ollama serve[/cyan]\n"
            "• Check if required models are installed\n\n"
            "[yellow]Configuration Errors[/yellow]\n"
            "• Validate config: [cyan]safehive config validate[/cyan]\n"
            "• Check YAML syntax and required fields\n\n"
            "[yellow]Session Failures[/yellow]\n"
            "• Check system status: [cyan]safehive status[/cyan]\n"
            "• Review logs in logs/ directory\n\n"
            "[yellow]Permission Errors[/yellow]\n"
            "• Ensure write access to logs/ and data/ directories\n"
            "• Check file permissions on configuration files",
            title="🔧 Troubleshooting",
            border_style="red"
        ))
    
    def interactive_help(self):
        """Launch interactive help system."""
        self.console.print("[bold blue]SafeHive Interactive Help[/bold blue]\n")
        
        while True:
            self.console.print("\n[bold]What would you like help with?[/bold]")
            self.console.print("1. Command help")
            self.console.print("2. Topic help")
            self.console.print("3. General help")
            self.console.print("4. Exit")
            
            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4"], default="1")
            
            if choice == "1":
                command = Prompt.ask("Enter command name", default="sandbox start")
                self._show_command_help(command)
            elif choice == "2":
                topic = Prompt.ask("Enter topic", choices=["scenarios", "guards", "agents", "configuration", "troubleshooting"])
                self._show_topic_help(topic)
            elif choice == "3":
                self._show_general_help()
            elif choice == "4":
                break
    
    def get_command_suggestions(self, partial_command: str) -> List[str]:
        """Get command suggestions for partial input."""
        suggestions = []
        for cmd in self.commands.keys():
            if cmd.startswith(partial_command.lower()):
                suggestions.append(cmd)
        return suggestions[:5]  # Return top 5 suggestions

# Global help system instance
_help_system: Optional[HelpSystem] = None

def get_help_system() -> HelpSystem:
    """Get the global help system instance."""
    global _help_system
    if _help_system is None:
        _help_system = HelpSystem()
    return _help_system

def show_help(command: Optional[str] = None, topic: Optional[str] = None):
    """Display help information."""
    help_system = get_help_system()
    help_system.show_help(command, topic)

def interactive_help():
    """Launch interactive help system."""
    help_system = get_help_system()
    help_system.interactive_help()

def get_command_suggestions(partial_command: str) -> List[str]:
    """Get command suggestions for partial input."""
    help_system = get_help_system()
    return help_system.get_command_suggestions(partial_command)
