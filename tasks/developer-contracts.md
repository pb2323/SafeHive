# Developer Contracts: SafeHive AI Security Sandbox

## Overview

This document defines the contracts and interfaces between Puneet and Rutuja to ensure independent development and seamless integration of the SafeHive AI Security Sandbox components.

## Core Interfaces

### 1. Guard Interface Contract

All security guards must implement the following interface:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from models.request_response import Request, Response

class Guard(ABC):
    @abstractmethod
    def inspect(self, request: Request) -> Response:
        """
        Inspect a request and return a response with action and details.
        
        Args:
            request: Standardized request object with source, payload, task info
            
        Returns:
            Response object with action (allow/block/decoy/redact), reason, and details
        """
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the guard with settings from YAML config.
        
        Args:
            config: Dictionary of configuration parameters
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status and metrics for the guard.
        
        Returns:
            Dictionary with guard status, metrics, and health info
        """
        pass
```

### 2. Request/Response Data Models

Standardized data structures for all agent communication:

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class Request:
    source: str  # e.g., "vendor_api_23"
    ip: str      # e.g., "192.168.1.24"
    payload: str # The actual request content
    task: str    # Original task context, e.g., "Order veg pizza under $20"
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

@dataclass
class Response:
    action: str  # "allow" | "block" | "decoy" | "redact"
    reason: str  # Human-readable explanation
    details: Dict[str, Any] = None  # Additional context and data
    confidence: float = 1.0  # Confidence level (0.0 to 1.0)
    timestamp: datetime = None
```

### 3. Vendor Agent Interface

Standard interface for all vendor agents (LangChain-based):

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from models.request_response import Request, Response
from langchain.agents import Agent
from langchain.memory import BaseMemory

class VendorAgent(ABC):
    def __init__(self, agent: Agent, memory: BaseMemory, personality_config: Dict[str, Any]):
        self.agent = agent
        self.memory = memory
        self.personality_config = personality_config
    
    @abstractmethod
    def process_order_request(self, request: Request) -> Response:
        """
        Process an order request using LangChain agent and return a response.
        
        Args:
            request: Order request with customer preferences and constraints
            
        Returns:
            Response with menu items, pricing, or other vendor-specific data
        """
        pass
    
    @abstractmethod
    def get_vendor_info(self) -> Dict[str, Any]:
        """
        Get vendor information and capabilities.
        
        Returns:
            Dictionary with vendor name, type, capabilities, personality, etc.
        """
        pass
    
    @abstractmethod
    def update_personality(self, new_config: Dict[str, Any]) -> None:
        """
        Update vendor personality and behavior configuration.
        
        Args:
            new_config: New personality configuration parameters
        """
        pass
    
    @abstractmethod
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history for analysis and debugging.
        
        Returns:
            List of conversation turns with timestamps and content
        """
        pass
```

## Configuration Contract

### YAML Configuration Structure

Both developers will use the same YAML configuration format:

```yaml
# config.yaml
honeypot:
  enabled: true
  threshold: 3
  attacks: ["SQLi", "XSS", "PathTraversal"]
  decoy_data_types: ["credit_cards", "order_history", "customer_profiles"]
  alert_stakeholders: true

privacy_sentry:
  enabled: true
  pii_patterns: ["credit_card", "ssn", "address", "phone"]
  redaction_method: "mask"

task_navigator:
  enabled: true
  constraint_types: ["budget", "dietary", "quantity"]
  enforcement_level: "strict"

prompt_sanitizer:
  enabled: true
  malicious_patterns: ["injection", "manipulation", "social_engineering"]
  sanitization_level: "aggressive"

agents:
  orchestrator:
    ai_model: "llama2:7b"
    max_retries: 3
    timeout_seconds: 30
    memory_type: "conversation_buffer"
    tools: ["order_management", "vendor_communication", "payment_processing"]
  
  user_twin:
    ai_model: "llama2:7b"
    memory_type: "conversation_summary"
    personality: "budget_conscious_vegetarian"
    constraints: ["budget_limit", "dietary_restrictions", "preferences"]
  
  vendors:
    honest_vendor:
      ai_model: "llama2:7b"
      personality: "friendly_restaurant_owner"
      memory_type: "conversation_buffer"
      tools: ["menu_lookup", "pricing", "inventory"]
    
    malicious_vendor:
      ai_model: "llama2:7b"
      personality: "aggressive_upseller"
      memory_type: "conversation_buffer"
      attack_behaviors: ["social_engineering", "technical_attacks", "manipulation"]
      tools: ["menu_lookup", "pricing", "inventory", "attack_patterns"]

logging:
  level: "INFO"
  file: "logs/sandbox.log"
  alerts_file: "logs/alerts.log"
  agent_conversations: "logs/agent_conversations.log"
  structured: true
```

## Integration Points

### 1. Orchestrator → Guards Communication

Puneet's orchestrator will call guards in sequence:

```python
# In orchestrator.py
def process_vendor_response(self, vendor_response: str, source: str) -> str:
    request = Request(
        source=source,
        ip=self.get_source_ip(source),
        payload=vendor_response,
        task=self.current_task
    )
    
    # Check with all enabled guards
    for guard in self.enabled_guards:
        response = guard.inspect(request)
        if response.action != "allow":
            return self.handle_guard_response(response)
    
    return vendor_response
```

### 2. Orchestrator → Vendor Agents Communication

Puneet's orchestrator will communicate with Rutuja's vendor agents using LangChain:

```python
# In orchestrator.py
def query_vendor(self, vendor_id: str, order_request: Request) -> Response:
    vendor = self.vendor_registry.get_vendor(vendor_id)
    
    # Convert request to LangChain message format
    message = self._create_agent_message(order_request)
    
    # Process through LangChain agent
    agent_response = vendor.agent.run(message)
    
    # Convert back to standard response format
    return self._convert_agent_response(agent_response, vendor)
```

### 3. Metrics Collection

Both developers will contribute to a shared metrics system:

```python
# Shared metrics interface
class MetricsCollector:
    def record_guard_action(self, guard_name: str, action: str, details: Dict[str, Any]):
        pass
    
    def record_vendor_interaction(self, vendor_id: str, interaction_type: str, success: bool):
        pass
    
    def record_agent_conversation(self, agent_id: str, conversation_turn: Dict[str, Any]):
        pass
    
    def record_attack_detection(self, attack_type: str, source: str, response: str):
        pass
    
    def get_summary_stats(self) -> Dict[str, Any]:
        pass
```

## Testing Contracts

### 1. Mock Implementations

Each developer will provide mock implementations of their components:

**Puneet's Mocks for Rutuja:**
- Mock honeypot guard that always returns "allow"
- Mock LangChain vendor agents with predictable responses
- Mock metrics collector for testing
- Mock agent memory and conversation history

**Rutuja's Mocks for Puneet:**
- Mock orchestrator that sends predefined requests
- Mock user twin with test preferences and LangChain agent behavior
- Mock guard framework for testing
- Mock agent communication protocols

### 2. Integration Test Requirements

Both developers will create integration tests that:
- Test the complete request flow from orchestrator through guards to vendors
- Validate LangChain agent communication and message passing
- Test agent memory and conversation context preservation
- Validate all data model serialization/deserialization
- Ensure performance requirements are met
- Test error handling and edge cases
- Validate agent personality and behavior consistency

### 3. Performance Benchmarks

Shared performance requirements:
- Guard inspection: < 1 second
- Agent response (LangChain): < 2 seconds
- Vendor response: < 2 seconds
- CLI response: < 2 seconds
- End-to-end scenario: < 5 minutes
- Agent memory operations: < 500ms
- Agent conversation context loading: < 1 second

## Communication Protocol

### 1. Daily Sync Points

- **Interface Changes**: Notify immediately of any changes to shared interfaces
- **Configuration Updates**: Coordinate on YAML schema changes
- **Performance Issues**: Share any performance bottlenecks or optimization needs
- **Testing Status**: Update on test coverage and integration test results

### 2. Code Review Requirements

- **Interface Code**: All interface implementations must be reviewed by both developers
- **Integration Points**: Any code that touches shared interfaces requires dual review
- **Configuration Changes**: YAML schema changes need approval from both developers
- **Performance Critical Code**: Any code affecting performance requirements needs review

### 3. Documentation Requirements

- **API Documentation**: Keep interface documentation up-to-date
- **Configuration Guide**: Maintain configuration examples and explanations
- **Integration Guide**: Document how components work together
- **Testing Guide**: Document how to run tests and validate integration

## Deployment and Integration

### 1. Integration Testing

Before final integration:
1. Both developers run their unit tests
2. Both developers run integration tests with mocks
3. Combined integration test with real implementations
4. Performance validation against benchmarks
5. End-to-end scenario testing

### 2. Configuration Validation

- All YAML configurations must be validated against the schema
- Default configurations must work out-of-the-box
- Configuration errors must provide clear, actionable error messages
- Configuration changes must be backward compatible

### 3. Error Handling

- All interfaces must handle errors gracefully
- Error messages must be clear and actionable
- Failed operations must not crash the entire system
- Error logging must include sufficient context for debugging

## Success Criteria

The integration is considered successful when:

1. **Functional Requirements**: All PRD requirements are met
2. **Performance Requirements**: All benchmarks are achieved
3. **Integration Tests**: All integration tests pass
4. **End-to-End Scenarios**: Complete attack scenarios work as designed
5. **Documentation**: All interfaces and integration points are documented
6. **Code Quality**: All code passes quality checks and reviews

## Risk Mitigation

### 1. Interface Changes

- **Versioning**: Use semantic versioning for interface changes
- **Backward Compatibility**: Maintain backward compatibility for at least one version
- **Migration Path**: Provide clear migration paths for breaking changes

### 2. Performance Issues

- **Monitoring**: Implement performance monitoring from day one
- **Profiling**: Regular performance profiling to identify bottlenecks
- **Optimization**: Continuous optimization based on performance data

### 3. Integration Failures

- **Fallback Mechanisms**: Implement fallback mechanisms for critical components
- **Graceful Degradation**: System should continue to function with reduced capabilities
- **Recovery Procedures**: Clear procedures for recovering from integration failures

---

**Document Version**: 1.0  
**Last Updated**: [Current Date]  
**Next Review**: [Date + 1 week]  
**Stakeholders**: Puneet, Rutuja, Project Lead
