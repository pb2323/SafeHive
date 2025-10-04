# SafeHive AI Security Sandbox Documentation

Welcome to the SafeHive AI Security Sandbox documentation! This comprehensive guide will help you understand, configure, and use the SafeHive platform for AI security testing and simulation.

## 📚 Documentation Index

### Getting Started
- [Installation Guide](installation.md) - Set up SafeHive on your system
- [Quick Start Guide](quickstart.md) - Get up and running in minutes
- [Configuration Guide](configuration.md) - Configure your environment

### User Guides
- [CLI Reference](cli-reference.md) - Complete command reference
- [Scenarios Guide](scenarios.md) - Available testing scenarios
- [Security Guards](guards.md) - Understanding security guard functionality
- [AI Agents](agents.md) - Working with AI agents

### Advanced Topics
- [Architecture Overview](architecture.md) - System architecture and design
- [API Reference](api-reference.md) - Programmatic interfaces
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Development Guide](development.md) - Contributing and extending SafeHive

## 🚀 Quick Start

1. **Initialize the system:**
   ```bash
   safehive init
   ```

2. **Launch the interactive interface:**
   ```bash
   safehive menu
   ```

3. **Start a security test scenario:**
   ```bash
   safehive sandbox start --scenario food-ordering --interactive
   ```

## 🛡️ What is SafeHive?

SafeHive is an advanced AI security testing and simulation platform that allows you to:

- **Test AI Agent Security**: Simulate real-world scenarios with AI agents
- **Detect Vulnerabilities**: Identify potential security issues in AI interactions
- **Validate Guard Systems**: Test security guards and protective measures
- **Monitor Performance**: Track metrics and performance across the system
- **Human-in-the-Loop**: Intervene in AI decisions when needed

## 🎯 Key Features

### Security Guards
- **Privacy Sentry**: Detects and prevents PII over-sharing
- **Task Navigator**: Enforces task constraints and prevents scope creep
- **Prompt Sanitizer**: Filters malicious inputs and validates formats

### AI Agents
- **Orchestrator**: Manages complex workflows and coordination
- **User Twin**: Represents user preferences and decision-making
- **Vendor Agents**: Simulates honest and malicious vendor interactions

### Monitoring & Control
- **Real-time Metrics**: Live dashboard with system statistics
- **Progress Tracking**: Visual progress indicators during simulations
- **Human Controls**: Manual intervention capabilities for critical decisions

## 📖 Command Overview

| Category | Commands | Description |
|----------|----------|-------------|
| **System** | `init`, `status`, `menu` | System initialization and status |
| **Sandbox** | `sandbox start`, `sandbox stop`, `sandbox list` | Session management |
| **Configuration** | `config show`, `config validate` | Configuration management |
| **Monitoring** | `metrics show`, `progress start` | System monitoring |
| **Controls** | `human list`, `human respond` | Human intervention controls |
| **Help** | `help`, `suggest` | Documentation and assistance |

## 🔧 Configuration

SafeHive uses YAML configuration files to manage settings:

- **Location**: `config/default_config.yaml`
- **Sections**: guards, agents, logging, metrics
- **Validation**: Built-in configuration validation

## 📊 Metrics & Monitoring

Track system performance and security events:

- **Real-time Dashboard**: `safehive metrics dashboard`
- **Export Data**: `safehive metrics export --file metrics.json`
- **Progress Monitoring**: `safehive progress start`

## 🆘 Getting Help

- **General Help**: `safehive help`
- **Command Help**: `safehive help <command>`
- **Interactive Help**: `safehive help --interactive`
- **Command Suggestions**: `safehive suggest <partial>`

## 🤝 Contributing

We welcome contributions! See our [Development Guide](development.md) for:

- Setting up development environment
- Code style guidelines
- Testing requirements
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for AI agent framework
- Powered by [Ollama](https://ollama.ai/) for local AI model execution
- UI components from [Rich](https://rich.readthedocs.io/) and [Typer](https://typer.tiangolo.com/)
