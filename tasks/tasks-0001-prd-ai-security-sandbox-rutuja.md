# Task List: SafeHive AI Security Sandbox - Rutuja's Tasks

## Relevant Files

- `honeypot/honeypot_guard.py` - Main honeypot guard implementation with attack detection
- `honeypot/decoy_generator.py` - Synthetic data generation for honeypot responses
- `honeypot/attack_detector.py` - OWASP attack pattern detection (SQLi, XSS, Path Traversal)
- `honeypot/threshold_manager.py` - Per-IP attempt tracking and threshold management
- `agents/vendors.py` - LangChain vendor agent implementations
- `agents/malicious_vendor.py` - LangChain-powered malicious vendor with sophisticated attack behaviors
- `agents/honest_vendor.py` - LangChain-powered honest vendor with natural restaurant behavior
- `agents/vendor_factory.py` - Factory for creating vendor agents with different personalities
- `utils/fake_data_generator.py` - Enhanced fake data generation using faker
- `utils/incident_logger.py` - Incident logging and alerting system
- `utils/agent_personality.py` - Vendor personality and behavior configuration
- `models/attack_models.py` - Data models for attack patterns and incidents
- `models/vendor_models.py` - Data models for vendor interactions and agent state
- `config/honeypot_config.yaml` - Honeypot-specific configuration settings
- `config/vendor_personalities.yaml` - Vendor personality and behavior configurations
- `logs/alerts.log` - Structured incident logs
- `tests/test_honeypot_guard.py` - Unit tests for honeypot guard functionality
- `tests/test_attack_detection.py` - Unit tests for attack pattern detection
- `tests/test_vendor_agents.py` - Unit tests for vendor agent implementations
- `tests/test_decoy_generation.py` - Unit tests for synthetic data generation
- `tests/integration/test_honeypot_integration.py` - Integration tests with orchestrator

### Notes

- All attack detection should use regex patterns initially, with extensibility for ML-based detection
- Honeypot guard must maintain state across multiple requests from the same source
- Decoy data generation must be cryptographically secure and realistic
- Incident logging must include all necessary details for forensic analysis
- Vendor agents should use LangChain with Ollama for realistic AI behavior and natural conversations
- Malicious vendors should have sophisticated attack personalities and behaviors
- Honest vendors should demonstrate natural restaurant business interactions
- Agent personalities should be configurable and varied for realistic testing scenarios

## Tasks

- [ ] 1.0 Honeypot Guard Core Implementation
  - [ ] 1.1 Create `honeypot/honeypot_guard.py` with main guard logic and standard Guard interface
  - [ ] 1.2 Implement `honeypot/attack_detector.py` with OWASP pattern detection (SQLi, XSS, Path Traversal)
  - [ ] 1.3 Create regex patterns for SQL injection detection (UNION, DROP, SELECT, etc.)
  - [ ] 1.4 Implement XSS detection patterns (script tags, event handlers, javascript:)
  - [ ] 1.5 Add Path Traversal detection (../, ..\\, directory traversal attempts)
  - [ ] 1.6 Create `honeypot/threshold_manager.py` for per-IP attempt tracking
  - [ ] 1.7 Implement threshold-based decision logic (default: 3 attempts per IP)
  - [ ] 1.8 Add attack pattern scoring and confidence levels

- [ ] 2.0 Decoy Data Generation System
  - [ ] 2.1 Create `honeypot/decoy_generator.py` for synthetic data creation
  - [ ] 2.2 Implement fake credit card number generation with realistic formats
  - [ ] 2.3 Create fake order history with realistic patterns and timestamps
  - [ ] 2.4 Generate fake customer profiles with addresses, phone numbers, preferences
  - [ ] 2.5 Implement `utils/fake_data_generator.py` using faker library
  - [ ] 2.6 Add configurable data generation (number of records, data types)
  - [ ] 2.7 Create realistic but fake restaurant menus and pricing
  - [ ] 2.8 Implement data consistency across generated records

- [ ] 3.0 AI-Powered Vendor Agent Implementation
  - [ ] 3.1 Create `agents/vendors.py` with LangChain vendor agent base class
  - [ ] 3.2 Implement `agents/honest_vendor.py` as LangChain agent with natural restaurant behavior
  - [ ] 3.3 Create honest vendor personality with menu knowledge, pricing, and customer service
  - [ ] 3.4 Implement `agents/malicious_vendor.py` as LangChain agent with sophisticated attack behaviors
  - [ ] 3.5 Add malicious vendor personalities (social engineering, technical attacks, manipulation)
  - [ ] 3.6 Create `agents/vendor_factory.py` for generating vendors with different personalities
  - [ ] 3.7 Implement vendor conversation memory and context awareness
  - [ ] 3.8 Add vendor personality configuration and behavior variation
  - [ ] 3.9 Create `utils/agent_personality.py` for managing vendor behaviors and attack patterns
  - [ ] 3.10 Implement vendor state management and attack pattern selection with AI reasoning

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
  - [ ] 5.1 Create `config/honeypot_config.yaml` for honeypot-specific settings
  - [ ] 5.2 Create `config/vendor_personalities.yaml` for vendor personality configurations
  - [ ] 5.3 Implement advanced attack pattern detection (evasion techniques)
  - [ ] 5.4 Add honeypot response customization (different decoy types)
  - [ ] 5.5 Create attack pattern learning and adaptation with agent feedback
  - [ ] 5.6 Implement multi-stage attack detection and response
  - [ ] 5.7 Add honeypot analytics and reporting features
  - [ ] 5.8 Create comprehensive unit tests for all components including agent interactions
  - [ ] 5.9 Implement integration tests with Puneet's orchestrator and LangChain agents
  - [ ] 5.10 Add agent conversation testing and validation

## Developer Contract with Puneet

### Interface Requirements
- **Guard Interface**: Honeypot guard implements the standard `Guard` interface from `guards/__init__.py`
- **Request/Response Format**: Uses standardized `Request` and `Response` models from `models/request_response.py`
- **Vendor Interface**: Vendor agents implement the standard vendor interface for orchestrator communication
- **Configuration**: All honeypot settings configurable via YAML with Puneet's config system

### Integration Points
- **Orchestrator Integration**: Rutuja's honeypot guard will be called by Puneet's orchestrator through the guard interface
- **Vendor Communication**: Rutuja's vendor agents will communicate with Puneet's orchestrator using defined protocols
- **Metrics Integration**: Honeypot metrics will be included in Puneet's `--metrics` CLI output
- **Logging Integration**: Incident logs will be accessible through Puneet's logging system

### Data Models
- **Attack Request**: `{"source": "vendor_api_23", "ip": "192.168.1.24", "payload": "SELECT * FROM...", "task": "Order veg pizza"}`
- **Guard Response**: `{"action": "allow|block|decoy|redact", "reason": "SQL injection detected", "details": {"attempts": 3, "decoy_data": "fake_orders.csv"}}`
- **Incident Log**: Structured JSON with timestamp, source, attack_type, response_action, and details

### Testing Requirements
- **Mock Orchestrator**: Rutuja will provide mock orchestrator for testing vendor interactions
- **Integration Tests**: Both developers will create tests that validate the complete workflow
- **Performance Testing**: Honeypot guard must respond within 1 second for attack detection
- **Security Testing**: All decoy data must be clearly marked as synthetic and non-functional

### Communication Protocol
- **Interface Changes**: Notify Puneet of any changes to guard or vendor interfaces
- **Attack Patterns**: Share new attack patterns and detection methods with Puneet
- **Performance Issues**: Coordinate on any performance bottlenecks or optimization needs
- **Documentation**: Maintain up-to-date API documentation for all interfaces
