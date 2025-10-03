# Task List: SafeHive AI Security Sandbox - Puneet's Tasks

## Relevant Files

- `cli.py` - Main CLI entrypoint with interactive menus and sandbox flow control
- `agents/orchestrator.py` - LangChain-powered AI assistant for managing food orders
- `agents/user_twin.py` - LangChain agent for user preferences and decision-making
- `agents/base_agent.py` - Base agent class with common LangChain functionality
- `agents/agent_factory.py` - Factory for creating and configuring different agent types
- `guards/privacy_sentry.py` - Privacy protection guard implementation
- `guards/task_navigator.py` - Task constraint enforcement guard
- `guards/prompt_sanitizer.py` - Malicious input filtering guard
- `guards/__init__.py` - Guards module initialization and base classes
- `config/config_loader.py` - YAML configuration management
- `config/default_config.yaml` - Default configuration settings
- `utils/logger.py` - Structured logging utilities
- `utils/ai_client.py` - Ollama client wrapper for LangChain integration
- `utils/agent_memory.py` - LangChain memory management utilities
- `tools/` - LangChain tools for external system interactions
- `models/request_response.py` - Data models for agent communication
- `models/agent_models.py` - Data models for agent state and memory
- `tests/test_cli.py` - Unit tests for CLI functionality
- `tests/test_orchestrator.py` - Unit tests for orchestrator agent
- `tests/test_guards.py` - Unit tests for all guard implementations
- `tests/test_config_loader.py` - Unit tests for configuration management
- `requirements.txt` - Python dependencies
- `README.md` - Project setup and usage instructions

### Notes

- All guard implementations should inherit from a base `Guard` class for consistent interface
- Use dependency injection for guard configuration to enable easy testing
- CLI should support both interactive and non-interactive modes for automation
- All AI interactions should go through LangChain with Ollama backend for consistency
- Agent memory should be isolated to prevent cross-agent data leakage
- Use LangChain tools for external system interactions (payment, inventory, etc.)
- Implement proper agent state management and conversation history

## Tasks

- [ ] 1.0 Project Setup and Infrastructure
  - [x] 1.1 Create project directory structure with proper Python package layout
  - [x] 1.2 Set up `requirements.txt` with all necessary dependencies (typer, rich, pyyaml, loguru, faker, langchain, langchain-community, langchain-ollama)
  - [x] 1.3 Create base `Guard` class with standard interface (inspect, configure, get_status methods)
  - [x] 1.4 Implement `config_loader.py` to handle YAML configuration with validation
  - [ ] 1.5 Create `default_config.yaml` with all guard settings, agent configurations, and thresholds
  - [ ] 1.6 Set up structured logging with `loguru` and create `logger.py` utility
  - [ ] 1.7 Create `models/request_response.py` with standardized data structures for agent communication
  - [ ] 1.8 Create `models/agent_models.py` for agent state, memory, and conversation data
  - [ ] 1.9 Implement `utils/ai_client.py` as LangChain wrapper for Ollama interactions
  - [ ] 1.10 Set up `utils/agent_memory.py` for LangChain memory management

- [ ] 2.0 CLI Framework and User Interface
  - [ ] 2.1 Create main `cli.py` entrypoint using typer framework
  - [ ] 2.2 Implement interactive menu system with rich formatting and colors
  - [ ] 2.3 Add `--metrics` CLI flag to display summary statistics
  - [ ] 2.4 Create sandbox launch command with scenario selection
  - [ ] 2.5 Implement human-in-the-loop controls (approve/redact/quarantine/ignore)
  - [ ] 2.6 Add real-time status updates and progress indicators during simulations
  - [ ] 2.7 Create help system and command documentation
  - [ ] 2.8 Implement configuration validation and error handling

- [ ] 3.0 AI Agent Framework and Base Implementation
  - [ ] 3.1 Create `agents/base_agent.py` with LangChain agent base class and common functionality
  - [ ] 3.2 Implement `agents/agent_factory.py` for creating and configuring different agent types
  - [ ] 3.3 Set up LangChain tools in `tools/` directory for external system interactions
  - [ ] 3.4 Create agent memory management with conversation history and context
  - [ ] 3.5 Implement agent communication protocols and message passing
  - [ ] 3.6 Add agent state persistence and recovery mechanisms
  - [ ] 3.7 Create agent configuration and personality management
  - [ ] 3.8 Implement agent monitoring and health checking

- [ ] 4.0 User Twin and Orchestrator Agents
  - [ ] 4.1 Create `agents/user_twin.py` as LangChain agent for personal preferences and decision-making
  - [ ] 4.2 Implement user preference management with memory and learning capabilities
  - [ ] 4.3 Create `agents/orchestrator.py` as LangChain-powered AI assistant for food ordering
  - [ ] 4.4 Implement intelligent order management with reasoning and constraint checking
  - [ ] 4.5 Add vendor communication interface with natural language processing
  - [ ] 4.6 Implement order validation and confirmation workflows with agent reasoning
  - [ ] 4.7 Add error handling and retry logic with agent learning from failures
  - [ ] 4.8 Create agent conversation management and context awareness

- [ ] 5.0 Core Security Guards Implementation
  - [ ] 5.1 Implement `guards/privacy_sentry.py` to detect and prevent PII over-sharing in agent communications
  - [ ] 5.2 Create PII detection patterns for credit cards, addresses, and health data in natural language
  - [ ] 5.3 Implement `guards/task_navigator.py` to enforce original task constraints with agent reasoning
  - [ ] 5.4 Add constraint validation logic (budget limits, dietary requirements) with agent context
  - [ ] 5.5 Implement `guards/prompt_sanitizer.py` to filter malicious vendor inputs and agent communications
  - [ ] 5.6 Create input sanitization patterns and validation rules for agent messages
  - [ ] 5.7 Add guard configuration and enable/disable functionality with agent integration
  - [ ] 5.8 Implement guard response formatting and logging with agent context

- [ ] 6.0 Integration and Testing
  - [ ] 6.1 Create comprehensive unit tests for all guard implementations
  - [ ] 6.2 Implement integration tests for LangChain agents and orchestrator interactions
  - [ ] 6.3 Add CLI testing with mock scenarios and user inputs
  - [ ] 6.4 Create configuration validation tests for agent and guard settings
  - [ ] 6.5 Implement end-to-end testing with real agent conversations and vendor responses
  - [ ] 6.6 Add performance testing for agent response time requirements
  - [ ] 6.7 Create agent conversation testing and validation
  - [ ] 6.8 Create documentation and usage examples for agent interactions
  - [ ] 6.9 Set up continuous integration and code quality checks

## Developer Contract with Rutuja

### Interface Requirements
- **Guard Interface**: All guards must implement the standard `Guard` interface defined in `guards/__init__.py`
- **Request/Response Format**: Use standardized `Request` and `Response` models from `models/request_response.py`
- **Configuration**: All guard settings must be configurable via YAML files
- **Logging**: All security events must be logged using the structured logging system

### Integration Points
- **Honeypot Guard Integration**: Puneet's orchestrator will call Rutuja's honeypot guard through the standard guard interface
- **Vendor Agent Integration**: Puneet's orchestrator will communicate with Rutuja's vendor agents using the defined request/response format
- **Metrics Collection**: Both developers will contribute metrics to the shared metrics system
- **Configuration Sharing**: Both will use the same YAML configuration structure

### Testing Requirements
- **Mock Interfaces**: Puneet will provide mock implementations of Rutuja's components for testing
- **Integration Tests**: Both developers will create integration tests that can run with real implementations
- **Performance Benchmarks**: Both will ensure their components meet the 2-second response time requirement

### Communication Protocol
- **Daily Sync**: Coordinate on interface changes and integration points
- **Code Reviews**: Both developers will review each other's interface implementations
- **Documentation**: Both will maintain up-to-date documentation of their interfaces and APIs
