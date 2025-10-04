# CLI Reference - SafeHive AI Security Sandbox

Complete reference for all SafeHive CLI commands, options, and usage examples.

## 📋 Command Overview

SafeHive provides a comprehensive command-line interface with the following command categories:

- **System Commands**: Initialize and manage the SafeHive system
- **Sandbox Commands**: Control security testing sessions
- **Configuration Commands**: Manage system configuration
- **Monitoring Commands**: Track system metrics and progress
- **Control Commands**: Handle human-in-the-loop interventions
- **Help Commands**: Access documentation and assistance

## 🔧 System Commands

### `safehive init`

Initialize the SafeHive AI Security Sandbox system.

**Usage:**
```bash
safehive init [--force]
```

**Options:**
- `--force`: Force reinitialization even if already configured

**Examples:**
```bash
safehive init
safehive init --force
```

**What it does:**
- Checks Ollama installation and connection
- Downloads required AI models
- Creates default configuration files
- Sets up logging directories
- Validates system requirements

### `safehive status`

Show system status and health information.

**Usage:**
```bash
safehive status [--metrics]
```

**Options:**
- `--metrics`: Include detailed metrics summary

**Examples:**
```bash
safehive status
safehive status --metrics
```

**Output includes:**
- Ollama connection status
- Active sessions and agents
- Guard status and configuration
- System health indicators

### `safehive menu`

Launch interactive menu system.

**Usage:**
```bash
safehive menu
```

**Features:**
- Guided access to all SafeHive features
- System status dashboard
- Quick action shortcuts
- Interactive navigation

## 🎯 Sandbox Commands

### `safehive sandbox start`

Start a new sandbox session with scenario execution.

**Usage:**
```bash
safehive sandbox start --scenario <name> [options]
```

**Required Options:**
- `--scenario`: Scenario name to execute

**Optional Options:**
- `--duration <seconds>`: Session duration (default: 300)
- `--interactive`: Run in interactive mode with user input
- `--background`: Run in background mode
- `--agents <list>`: Comma-separated list of agents to include
- `--guards <list>`: Comma-separated list of guards to enable

**Examples:**
```bash
safehive sandbox start --scenario food-ordering
safehive sandbox start --scenario food-ordering --duration 300
safehive sandbox start --scenario food-ordering --interactive
safehive sandbox start --scenario food-ordering --background
```

**Available Scenarios:**
- `food-ordering`: Food ordering workflow with malicious vendors

### `safehive sandbox stop`

Stop active sandbox sessions.

**Usage:**
```bash
safehive sandbox stop [session-id] [--all]
```

**Options:**
- `session-id`: Specific session ID to stop
- `--all`: Stop all active sessions

**Examples:**
```bash
safehive sandbox stop
safehive sandbox stop session-123
safehive sandbox stop --all
```

### `safehive sandbox list`

List available scenarios or active sessions.

**Usage:**
```bash
safehive sandbox list [--sessions]
```

**Options:**
- `--sessions`: List active sessions instead of scenarios

**Examples:**
```bash
safehive sandbox list
safehive sandbox list --sessions
```

### `safehive sandbox status`

Show detailed status of sandbox sessions.

**Usage:**
```bash
safehive sandbox status [session-id]
```

**Options:**
- `session-id`: Specific session ID to show status for

**Examples:**
```bash
safehive sandbox status
safehive sandbox status session-123
```

## ⚙️ Configuration Commands

### `safehive config show`

Display current configuration settings.

**Usage:**
```bash
safehive config show [--section <section>]
```

**Options:**
- `--section`: Specific configuration section to display

**Examples:**
```bash
safehive config show
safehive config show --section guards
safehive config show --section agents
```

**Configuration Sections:**
- `guards`: Security guard settings
- `agents`: AI agent configurations
- `logging`: Logging levels and files
- `metrics`: Metrics collection settings

### `safehive config validate`

Validate configuration file syntax and settings.

**Usage:**
```bash
safehive config validate [--file <path>]
```

**Options:**
- `--file`: Path to configuration file to validate

**Examples:**
```bash
safehive config validate
safehive config validate --file custom_config.yaml
```

## 📊 Monitoring Commands

### `safehive metrics show`

Display system metrics and statistics.

**Usage:**
```bash
safehive metrics show [--format <format>]
```

**Options:**
- `--format`: Output format (table, json, summary)

**Examples:**
```bash
safehive metrics show
safehive metrics show --format json
safehive metrics show --format summary
```

### `safehive metrics export`

Export metrics to file.

**Usage:**
```bash
safehive metrics export --file <path> [--format <format>]
```

**Required Options:**
- `--file`: Output file path

**Optional Options:**
- `--format`: Export format (json, csv, yaml)

**Examples:**
```bash
safehive metrics export --file metrics.json
safehive metrics export --file metrics.csv --format csv
```

### `safehive metrics dashboard`

Launch real-time metrics dashboard.

**Usage:**
```bash
safehive metrics dashboard
```

**Features:**
- Live metrics updates
- System performance indicators
- Security event monitoring
- Press Ctrl+C to exit

### `safehive progress start`

Start real-time progress monitoring.

**Usage:**
```bash
safehive progress start
```

**Features:**
- Live progress updates for active sessions
- Agent status and guard activities
- Visual progress indicators

### `safehive progress stop`

Stop progress monitoring.

**Usage:**
```bash
safehive progress stop
```

### `safehive progress demo`

Demonstrate progress monitoring with simulated session.

**Usage:**
```bash
safehive progress demo
```

**Features:**
- Creates simulated session for demonstration
- Shows various progress events and updates
- Educational tool for understanding progress tracking

## 🎮 Control Commands

### `safehive human list`

List pending human intervention requests.

**Usage:**
```bash
safehive human list [--session <session-id>]
```

**Options:**
- `--session`: Filter by specific session ID

**Examples:**
```bash
safehive human list
safehive human list --session session-123
```

### `safehive human respond`

Respond to a human intervention request.

**Usage:**
```bash
safehive human respond <request-id> <action> [options]
```

**Required Arguments:**
- `request-id`: Request ID to respond to
- `action`: Action to take (approve, redact, quarantine, ignore)

**Optional Options:**
- `--rules`: Redaction rules (comma-separated)
- `--duration`: Quarantine duration in seconds
- `--reason`: Reason for the action

**Examples:**
```bash
safehive human respond req-123 approve
safehive human respond req-123 redact --rules 'email,phone'
safehive human respond req-123 quarantine --duration 3600
```

### `safehive human approve`

Quick approve an intervention request.

**Usage:**
```bash
safehive human approve <request-id>
```

### `safehive human quarantine`

Quick quarantine an intervention request.

**Usage:**
```bash
safehive human quarantine <request-id> [--duration <minutes>] [--reason <reason>]
```

### `safehive human ignore`

Quick ignore an intervention request.

**Usage:**
```bash
safehive human ignore <request-id> [--reason <reason>]
```

## 🆘 Help Commands

### `safehive help`

Show help information for commands and topics.

**Usage:**
```bash
safehive help [command] [--topic <topic>] [--interactive]
```

**Options:**
- `command`: Command to get help for
- `--topic`: Topic to get help for
- `--interactive`: Launch interactive help system

**Examples:**
```bash
safehive help
safehive help sandbox start
safehive help --topic scenarios
safehive help --interactive
```

**Available Topics:**
- `scenarios`: Available testing scenarios
- `guards`: Security guard functionality
- `agents`: AI agent information
- `configuration`: Configuration management
- `troubleshooting`: Common issues and solutions

### `safehive suggest`

Get command suggestions for partial input.

**Usage:**
```bash
safehive suggest <partial>
```

**Examples:**
```bash
safehive suggest sand
safehive suggest met
safehive suggest hum
```

## 🔍 Command Completion

SafeHive supports command completion for better user experience:

- **Tab Completion**: Press Tab to complete commands and options
- **Suggestions**: Use `safehive suggest <partial>` for command suggestions
- **Help Integration**: All commands include built-in help with `--help`

## 📝 Common Usage Patterns

### Initial Setup
```bash
# Initialize system
safehive init

# Check status
safehive status --metrics

# Launch interactive menu
safehive menu
```

### Running Security Tests
```bash
# Start interactive session
safehive sandbox start --scenario food-ordering --interactive

# Monitor progress
safehive progress start

# Check session status
safehive sandbox status
```

### Monitoring and Control
```bash
# View metrics dashboard
safehive metrics dashboard

# List intervention requests
safehive human list

# Respond to requests
safehive human respond req-123 approve
```

### Configuration Management
```bash
# View configuration
safehive config show

# Validate configuration
safehive config validate

# Show specific section
safehive config show --section guards
```

## 🚨 Error Handling

All commands include comprehensive error handling:

- **Validation**: Input validation with clear error messages
- **Logging**: Detailed error logging for debugging
- **Recovery**: Graceful error recovery where possible
- **Exit Codes**: Proper exit codes for automation

## 🔧 Environment Variables

SafeHive respects the following environment variables:

- `SAFEHIVE_CONFIG_PATH`: Custom configuration file path
- `SAFEHIVE_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `SAFEHIVE_DATA_DIR`: Custom data directory path
