# Quick Start Guide - SafeHive AI Security Sandbox

Get up and running with SafeHive in just a few minutes! This guide will walk you through the essential steps to start testing AI security scenarios.

## 🚀 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed on your system
- **Ollama** installed and running (for local AI models)
- **Git** (for cloning the repository)

## 📥 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/safehive.git
cd safehive
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### 4. Start Ollama Service
```bash
ollama serve
```

## 🔧 Initial Setup

### 1. Initialize SafeHive
```bash
python cli.py init
```

This command will:
- ✅ Check Ollama connection
- ✅ Download required AI models
- ✅ Create configuration files
- ✅ Set up logging directories

### 2. Verify Installation
```bash
python cli.py status
```

You should see:
- Ollama connection status: ✅ Connected
- System status: ✅ Ready
- Available scenarios: food-ordering

## 🎯 Your First Security Test

### 1. Launch Interactive Menu
```bash
python cli.py menu
```

### 2. Start a Sandbox Session
```bash
python cli.py sandbox start --scenario food-ordering --interactive
```

### 3. Monitor Progress (Optional)
In another terminal:
```bash
python cli.py progress start
```

## 📊 Understanding the Interface

### Main Commands
- `init` - Initialize the system
- `status` - Check system health
- `menu` - Launch interactive interface
- `sandbox start` - Begin security testing
- `help` - Get assistance

### Interactive Features
- **Real-time Progress**: Watch live updates during simulations
- **Human Controls**: Intervene in AI decisions when needed
- **Metrics Dashboard**: Monitor system performance
- **Configuration Management**: Customize settings

## 🛡️ Security Guards in Action

SafeHive includes three specialized security guards:

### Privacy Sentry
- **Purpose**: Detects PII over-sharing
- **Monitors**: Credit cards, addresses, health data
- **Action**: Suggests redaction or blocks requests

### Task Navigator
- **Purpose**: Enforces task constraints
- **Monitors**: Budget limits, dietary requirements
- **Action**: Prevents scope creep in conversations

### Prompt Sanitizer
- **Purpose**: Filters malicious inputs
- **Monitors**: SQL injection, XSS attempts
- **Action**: Validates and sanitizes inputs

## 🤖 AI Agents Overview

### Orchestrator Agent
- Manages the overall food ordering workflow
- Coordinates between vendors and user twin
- Handles order validation and confirmation

### User Twin Agent
- Represents your preferences and constraints
- Manages dietary restrictions and budget
- Makes decisions based on your profile

### Vendor Agents
- **Honest Vendors**: Normal business behavior
- **Malicious Vendors**: Attempt various security attacks

## 📈 Monitoring and Control

### Real-time Metrics
```bash
python cli.py metrics dashboard
```

### Progress Tracking
```bash
python cli.py progress start
```

### Human Intervention
```bash
python cli.py human list
python cli.py human respond <request-id> approve
```

## 🎮 Common Workflows

### 1. Basic Security Testing
```bash
# Start a session
python cli.py sandbox start --scenario food-ordering

# Monitor progress
python cli.py progress start

# Check results
python cli.py metrics show
```

### 2. Interactive Testing
```bash
# Start interactive session
python cli.py sandbox start --scenario food-ordering --interactive

# Respond to prompts as they appear
# System will ask for your input during the simulation
```

### 3. Configuration Management
```bash
# View current settings
python cli.py config show

# Validate configuration
python cli.py config validate

# Show specific section
python cli.py config show --section guards
```

## 🔍 Troubleshooting

### Common Issues

#### Ollama Connection Failed
```bash
# Check if Ollama is running
ollama serve

# Verify connection
python cli.py status
```

#### Configuration Errors
```bash
# Validate configuration
python cli.py config validate

# Reset to defaults
python cli.py init --force
```

#### Session Failures
```bash
# Check system status
python cli.py status

# Review logs
tail -f logs/main.log
```

### Getting Help
```bash
# General help
python cli.py help

# Command-specific help
python cli.py help sandbox start

# Interactive help
python cli.py help --interactive

# Command suggestions
python cli.py suggest sand
```

## 📚 Next Steps

Now that you have SafeHive running:

1. **Explore Scenarios**: Try different testing scenarios
2. **Customize Configuration**: Adjust guard settings and thresholds
3. **Monitor Performance**: Use metrics dashboard to track results
4. **Human Controls**: Practice intervention workflows
5. **Advanced Features**: Explore API integration and automation

## 🆘 Support

- **Documentation**: `docs/` directory
- **CLI Help**: `python cli.py help`
- **Interactive Help**: `python cli.py help --interactive`
- **Issues**: GitHub Issues page

## 🎉 Congratulations!

You've successfully set up SafeHive and run your first AI security test! The system is now ready for advanced security testing scenarios.

**Pro Tip**: Use the interactive menu (`python cli.py menu`) for guided access to all features.
