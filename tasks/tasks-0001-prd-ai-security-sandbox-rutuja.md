# Task List: SafeHive AI Security Sandbox - Rutuja's Tasks

## Relevant Files

- `mcp/doorDash_mcp_server.py` - MCP server implementation for DoorDash API integration
- `mcp/doorDash_client.py` - DoorDash API client wrapper
- `mcp/mcp_config.yaml` - MCP server configuration settings
- `agents/vendors.py` - LangChain vendor agent implementations
- `agents/malicious_vendor.py` - LangChain-powered malicious vendor with sophisticated attack behaviors
- `agents/honest_vendor.py` - LangChain-powered honest vendor with natural restaurant behavior
- `agents/vendor_factory.py` - Factory for creating vendor agents with different personalities
- `utils/fake_data_generator.py` - Enhanced fake data generation using faker
- `utils/incident_logger.py` - Incident logging and alerting system
- `utils/agent_personality.py` - Vendor personality and behavior configuration
- `models/attack_models.py` - Data models for attack patterns and incidents
- `models/vendor_models.py` - Data models for vendor interactions and agent state
- `config/vendor_personalities.yaml` - Vendor personality and behavior configurations
- `logs/alerts.log` - Structured incident logs
- `tests/test_mcp_server.py` - Unit tests for MCP server functionality
- `tests/test_vendor_agents.py` - Unit tests for vendor agent implementations
- `tests/test_malicious_behaviors.py` - Unit tests for malicious vendor behaviors
- `tests/integration/test_mcp_integration.py` - Integration tests with orchestrator and MCP server

### Notes

- MCP server should use existing DoorDash API resources and not be built from scratch
- Vendor agents should use LangChain with Ollama for realistic AI behavior and natural conversations
- Malicious vendors should have sophisticated attack personalities and behaviors
- Honest vendors should demonstrate natural restaurant business interactions
- Agent personalities should be configurable and varied for realistic testing scenarios
- MCP server must support secure credential management for DoorDash API access
- Live ordering should be clearly separated from testing mode

## Tasks

- [ ] 1.0 MCP Server Implementation
  - [ ] 1.1 Research and integrate existing DoorDash MCP server resources
  - [ ] 1.2 Create `mcp/doorDash_mcp_server.py` with DoorDash API integration
  - [ ] 1.3 Implement `mcp/doorDash_client.py` for API communication
  - [ ] 1.4 Create `mcp/mcp_config.yaml` for server configuration
  - [ ] 1.5 Implement secure credential management for DoorDash API keys
  - [ ] 1.6 Add order validation and safety checks before placing live orders
  - [ ] 1.7 Implement error handling and retry logic for API failures
  - [ ] 1.8 Add logging and monitoring for MCP server operations

- [ ] 2.0 AI-Powered Vendor Agent Implementation
  - [ ] 2.1 Create `agents/vendors.py` with LangChain vendor agent base class
  - [ ] 2.2 Implement `agents/honest_vendor.py` as LangChain agent with natural restaurant behavior
  - [ ] 2.3 Create honest vendor personality with menu knowledge, pricing, and customer service
  - [ ] 2.4 Implement `agents/malicious_vendor.py` as LangChain agent with sophisticated attack behaviors
  - [ ] 2.5 Add malicious vendor personalities (social engineering, technical attacks, manipulation)
  - [ ] 2.6 Create `agents/vendor_factory.py` for generating vendors with different personalities
  - [ ] 2.7 Implement vendor conversation memory and context awareness
  - [ ] 2.8 Add vendor personality configuration and behavior variation
  - [ ] 2.9 Create `utils/agent_personality.py` for managing vendor behaviors and attack patterns
  - [ ] 2.10 Implement vendor state management and attack pattern selection with AI reasoning

- [ ] 3.0 Malicious Behavior Simulation
  - [ ] 3.1 Implement social engineering attack patterns (manipulation, deception)
  - [ ] 3.2 Create technical attack behaviors (prompt injection, data exfiltration attempts)
  - [ ] 3.3 Add psychological manipulation techniques (urgency, authority, scarcity)
  - [ ] 3.4 Implement conversation hijacking and redirection attempts
  - [ ] 3.5 Create realistic attack escalation patterns
  - [ ] 3.6 Add attack success/failure feedback mechanisms
  - [ ] 3.7 Implement attack pattern learning and adaptation
  - [ ] 3.8 Create attack behavior configuration and customization

- [ ] 4.0 Incident Logging and Alerting
  - [ ] 4.1 Create `utils/incident_logger.py` for structured incident logging
  - [ ] 4.2 Implement `models/attack_models.py` for attack pattern data structures
  - [ ] 4.3 Add incident logging to `logs/alerts.log` with structured format
  - [ ] 4.4 Create incident details (timestamp, source IP, attack type, response)
  - [ ] 4.5 Implement stakeholder alerting system (CLI notifications)
  - [ ] 4.6 Add incident correlation and pattern analysis
  - [ ] 4.7 Create incident reporting and summary generation
  - [ ] 4.8 Implement log rotation and retention policies

- [ ] 5.0 Integration and Advanced Features
  - [ ] 5.1 Create `config/vendor_personalities.yaml` for vendor personality configurations
  - [ ] 5.2 Implement advanced attack pattern detection (evasion techniques)
  - [ ] 5.3 Create attack pattern learning and adaptation with agent feedback
  - [ ] 5.4 Implement multi-stage attack detection and response
  - [ ] 5.5 Add vendor analytics and reporting features
  - [ ] 5.6 Create comprehensive unit tests for all components including agent interactions
  - [ ] 5.7 Implement integration tests with Puneet's orchestrator and MCP server
  - [ ] 5.8 Add agent conversation testing and validation
  - [ ] 5.9 Create MCP server integration tests with live DoorDash API (sandbox mode)
  - [ ] 5.10 Implement end-to-end testing with both vendor agents and MCP server

## Developer Contract with Puneet

### Interface Requirements
- **MCP Server Interface**: MCP server implements standard MCP protocol for DoorDash integration
- **Vendor Interface**: Vendor agents implement the standard vendor interface for orchestrator communication
- **Request/Response Format**: Uses standardized `Request` and `Response` models from `models/request_response.py`
- **Configuration**: All MCP and vendor settings configurable via YAML with Puneet's config system

### Integration Points
- **Orchestrator Integration**: Rutuja's MCP server will be called by Puneet's orchestrator for live orders
- **Vendor Communication**: Rutuja's vendor agents will communicate with Puneet's orchestrator using defined protocols
- **Metrics Integration**: MCP server and vendor metrics will be included in Puneet's `--metrics` CLI output
- **Logging Integration**: Incident logs will be accessible through Puneet's logging system
- **Live Order Flag**: MCP server integration will be triggered by Puneet's `--live-order` CLI flag

### Data Models
- **Order Request**: `{"source": "orchestrator", "restaurant": "Pizza Palace", "items": [...], "total": 25.99, "delivery_address": "..."}`
- **MCP Response**: `{"status": "success|error", "order_id": "DD-12345", "estimated_delivery": "30-45 min", "confirmation": "..."}`
- **Vendor Response**: `{"action": "allow|block|escalate", "reason": "Social engineering detected", "details": {"attack_type": "manipulation", "confidence": 0.95}}`
- **Incident Log**: Structured JSON with timestamp, source, attack_type, response_action, and details

### Testing Requirements
- **Mock Orchestrator**: Rutuja will provide mock orchestrator for testing vendor interactions
- **Integration Tests**: Both developers will create tests that validate the complete workflow
- **Performance Testing**: MCP server must respond within 5 seconds for order placement
- **Security Testing**: All vendor interactions must be clearly logged and monitored
- **Live Order Testing**: MCP server must be tested with DoorDash sandbox/development environment

### Communication Protocol
- **Interface Changes**: Notify Puneet of any changes to MCP server or vendor interfaces
- **Attack Patterns**: Share new attack patterns and detection methods with Puneet
- **Performance Issues**: Coordinate on any performance bottlenecks or optimization needs
- **Documentation**: Maintain up-to-date API documentation for all interfaces
- **Credential Management**: Coordinate secure handling of DoorDash API credentials