"""
Unit tests for the Agent Factory and related components.

This module tests the agent factory, registry, and validation functionality
for creating and managing AI agents in the SafeHive system.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import List, Dict, Any

from safehive.agents.agent_factory import (
    AgentFactory,
    AgentRegistry,
    AgentValidator,
    AgentCreationRequest,
    AgentCreationResult,
    create_orchestrator_agent,
    create_user_twin_agent,
    create_honest_vendor_agent,
    create_malicious_vendor_agent
)
from safehive.models.agent_models import AgentType, AgentState
from safehive.agents.base_agent import BaseAgent, AgentCapabilities
from safehive.config.config_loader import AgentConfig, SystemConfig


class MockAgent:
    """Mock agent implementation for testing."""
    
    def __init__(self, config: AgentConfig, agent_type: AgentType = AgentType.ORCHESTRATOR):
        self.config = config
        self.agent_id = f"test-agent-id-{agent_type.value}-{id(self)}"
        self.agent_type = agent_type
        self.name = config.name if hasattr(config, 'name') else f"Mock {agent_type.value.title()} Agent"
        self.description = config.description if hasattr(config, 'description') else f"A mock {agent_type.value} agent for testing"
        self._status = AgentState.IDLE
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._average_response_time = 0.0
        self._last_activity = datetime.now()
        self._conversation_history = []
    
    async def process_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """Mock message processing."""
        return f"Mock response to: {message}"
    
    def get_status(self):
        """Mock status method."""
        from safehive.models.agent_models import AgentStatus
        return AgentStatus(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            current_state=self._status,
            configuration={"name": self.name, "description": self.description},
            metrics={
                "total_requests": self._total_requests,
                "successful_requests": self._successful_requests,
                "failed_requests": self._failed_requests,
                "average_response_time": self._average_response_time
            }
        )
    
    def is_healthy(self) -> bool:
        """Mock health check."""
        return self._status != AgentState.ERROR
    
    async def start(self):
        """Mock start method."""
        self._status = AgentState.ACTIVE
    
    async def stop(self):
        """Mock stop method."""
        self._status = AgentState.STOPPED


class TestAgentRegistry:
    """Test AgentRegistry functionality."""
    
    def test_agent_registry_initialization(self):
        """Test registry initialization."""
        registry = AgentRegistry()
        
        assert len(registry._agents) == 0
        assert len(registry._agent_metadata) == 0
        assert len(registry._agent_types) == 0
        assert len(registry._creation_timestamps) == 0
    
    def test_register_agent(self):
        """Test agent registration."""
        registry = AgentRegistry()
        agent = MockAgent(AgentConfig(), AgentType.ORCHESTRATOR)
        
        agent_id = registry.register_agent(agent, {"test": "metadata"})
        
        assert agent_id == agent.agent_id
        assert registry.get_agent(agent_id) == agent
        assert registry.get_agent_metadata(agent_id) == {"test": "metadata"}
        assert registry.get_agent_count() == 1
        assert registry.get_agent_count_by_type(AgentType.ORCHESTRATOR) == 1
    
    def test_unregister_agent(self):
        """Test agent unregistration."""
        registry = AgentRegistry()
        agent = MockAgent(AgentConfig(), AgentType.ORCHESTRATOR)
        
        registry.register_agent(agent)
        assert registry.get_agent_count() == 1
        
        result = registry.unregister_agent(agent.agent_id)
        assert result is True
        assert registry.get_agent_count() == 0
        assert registry.get_agent(agent.agent_id) is None
    
    def test_unregister_nonexistent_agent(self):
        """Test unregistering non-existent agent."""
        registry = AgentRegistry()
        
        result = registry.unregister_agent("nonexistent-id")
        assert result is False
    
    def test_get_agents_by_type(self):
        """Test getting agents by type."""
        registry = AgentRegistry()
        
        orchestrator = MockAgent(AgentConfig(), AgentType.ORCHESTRATOR)
        user_twin = MockAgent(AgentConfig(), AgentType.USER_TWIN)
        orchestrator2 = MockAgent(AgentConfig(), AgentType.ORCHESTRATOR)
        
        registry.register_agent(orchestrator)
        registry.register_agent(user_twin)
        registry.register_agent(orchestrator2)
        
        orchestrators = registry.get_agents_by_type(AgentType.ORCHESTRATOR)
        user_twins = registry.get_agents_by_type(AgentType.USER_TWIN)
        
        assert len(orchestrators) == 2
        assert len(user_twins) == 1
        assert orchestrator in orchestrators
        assert orchestrator2 in orchestrators
        assert user_twin in user_twins
    
    def test_get_all_agents(self):
        """Test getting all agents."""
        registry = AgentRegistry()
        
        agent1 = MockAgent(AgentConfig(), AgentType.ORCHESTRATOR)
        agent2 = MockAgent(AgentConfig(), AgentType.USER_TWIN)
        
        registry.register_agent(agent1)
        registry.register_agent(agent2)
        
        all_agents = registry.get_all_agents()
        assert len(all_agents) == 2
        assert agent1 in all_agents
        assert agent2 in all_agents
    
    def test_update_agent_metadata(self):
        """Test updating agent metadata."""
        registry = AgentRegistry()
        agent = MockAgent(AgentConfig(), AgentType.ORCHESTRATOR)
        
        registry.register_agent(agent, {"initial": "metadata"})
        
        result = registry.update_agent_metadata(agent.agent_id, {"updated": "metadata"})
        assert result is True
        assert registry.get_agent_metadata(agent.agent_id) == {"updated": "metadata"}
        
        result = registry.update_agent_metadata("nonexistent-id", {"test": "metadata"})
        assert result is False
    
    def test_clear_all_agents(self):
        """Test clearing all agents."""
        registry = AgentRegistry()
        
        agent1 = MockAgent(AgentConfig(), AgentType.ORCHESTRATOR)
        agent2 = MockAgent(AgentConfig(), AgentType.USER_TWIN)
        
        registry.register_agent(agent1)
        registry.register_agent(agent2)
        
        assert registry.get_agent_count() == 2
        
        cleared_count = registry.clear_all_agents()
        assert cleared_count == 2
        assert registry.get_agent_count() == 0
    
    def test_get_agent_status_summary(self):
        """Test getting agent status summary."""
        registry = AgentRegistry()
        
        agent = MockAgent(AgentConfig(), AgentType.ORCHESTRATOR)
        registry.register_agent(agent)
        
        summary = registry.get_agent_status_summary()
        
        assert summary["total_agents"] == 1
        assert summary["agents_by_type"][AgentType.ORCHESTRATOR.value] == 1
        assert summary["agent_states"][AgentState.IDLE.value] == 1
        assert summary["healthy_agents"] == 1
        assert summary["unhealthy_agents"] == 0


class TestAgentValidator:
    """Test AgentValidator functionality."""
    
    def test_validate_creation_request_valid(self):
        """Test validation of valid creation request."""
        request = AgentCreationRequest(
            agent_type=AgentType.ORCHESTRATOR,
            name="Test Agent",
            description="A test agent"
        )
        
        errors = AgentValidator.validate_creation_request(request)
        assert len(errors) == 0
    
    def test_validate_creation_request_missing_name(self):
        """Test validation with missing name."""
        request = AgentCreationRequest(
            agent_type=AgentType.ORCHESTRATOR,
            name="",
            description="A test agent"
        )
        
        errors = AgentValidator.validate_creation_request(request)
        assert len(errors) == 1
        assert "name is required" in errors[0]
    
    def test_validate_creation_request_missing_description(self):
        """Test validation with missing description."""
        request = AgentCreationRequest(
            agent_type=AgentType.ORCHESTRATOR,
            name="Test Agent",
            description=""
        )
        
        errors = AgentValidator.validate_creation_request(request)
        assert len(errors) == 1
        assert "description is required" in errors[0]
    
    def test_validate_creation_request_invalid_capabilities(self):
        """Test validation with invalid capabilities."""
        request = AgentCreationRequest(
            agent_type=AgentType.ORCHESTRATOR,
            name="Test Agent",
            description="A test agent",
            custom_capabilities=["invalid_capability"]
        )
        
        errors = AgentValidator.validate_creation_request(request)
        assert len(errors) == 1
        assert "Invalid capability" in errors[0]
    
    def test_validate_agent_config_valid(self):
        """Test validation of valid agent config."""
        config = AgentConfig(
            ai_model="llama2:7b",
            max_retries=3,
            timeout_seconds=30,
            memory_type="conversation_buffer"
        )
        
        errors = AgentValidator.validate_agent_config(config)
        assert len(errors) == 0
    
    def test_validate_agent_config_missing_model(self):
        """Test validation with missing AI model."""
        config = AgentConfig(
            ai_model="",
            max_retries=3,
            timeout_seconds=30,
            memory_type="conversation_buffer"
        )
        
        errors = AgentValidator.validate_agent_config(config)
        assert len(errors) == 1
        assert "AI model is required" in errors[0]
    
    def test_validate_agent_config_invalid_retries(self):
        """Test validation with invalid retry count."""
        config = AgentConfig(
            ai_model="llama2:7b",
            max_retries=0,  # Invalid
            timeout_seconds=30,
            memory_type="conversation_buffer"
        )
        
        errors = AgentValidator.validate_agent_config(config)
        assert len(errors) == 1
        assert "Max retries must be between 1 and 10" in errors[0]
    
    def test_validate_agent_config_invalid_timeout(self):
        """Test validation with invalid timeout."""
        config = AgentConfig(
            ai_model="llama2:7b",
            max_retries=3,
            timeout_seconds=2,  # Invalid
            memory_type="conversation_buffer"
        )
        
        errors = AgentValidator.validate_agent_config(config)
        assert len(errors) == 1
        assert "Timeout must be between 5 and 300 seconds" in errors[0]
    
    def test_validate_agent_config_invalid_memory_type(self):
        """Test validation with invalid memory type."""
        config = AgentConfig(
            ai_model="llama2:7b",
            max_retries=3,
            timeout_seconds=30,
            memory_type="invalid_memory_type"
        )
        
        errors = AgentValidator.validate_agent_config(config)
        assert len(errors) == 1
        assert "Invalid memory type" in errors[0]


class TestAgentFactory:
    """Test AgentFactory functionality."""
    
    def test_agent_factory_initialization(self):
        """Test factory initialization."""
        factory = AgentFactory()
        
        assert factory.registry is not None
        assert factory.validator is not None
        assert factory.system_config is None
        assert len(factory._agent_implementations) == 0
    
    def test_register_agent_implementation(self):
        """Test registering agent implementation."""
        factory = AgentFactory()
        
        factory.register_agent_implementation(AgentType.ORCHESTRATOR, MockAgent)
        
        assert AgentType.ORCHESTRATOR in factory._agent_implementations
        assert factory._agent_implementations[AgentType.ORCHESTRATOR] == MockAgent
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_create_agent_success(self, mock_memory_manager, mock_ai_client):
        """Test successful agent creation."""
        factory = AgentFactory()
        factory.register_agent_implementation(AgentType.ORCHESTRATOR, MockAgent)
        
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        request = AgentCreationRequest(
            agent_type=AgentType.ORCHESTRATOR,
            name="Test Orchestrator",
            description="A test orchestrator agent"
        )
        
        result = factory.create_agent(request)
        
        assert result.success is True
        assert result.agent is not None
        assert result.agent_id is not None
        assert result.error_message is None
        assert factory.registry.get_agent_count() == 1
    
    def test_create_agent_validation_failure(self):
        """Test agent creation with validation failure."""
        factory = AgentFactory()
        
        request = AgentCreationRequest(
            agent_type=AgentType.ORCHESTRATOR,
            name="",  # Invalid - empty name
            description="A test agent"
        )
        
        result = factory.create_agent(request)
        
        assert result.success is False
        assert result.agent is None
        assert result.error_message is not None
        assert "name is required" in result.error_message
    
    def test_create_agent_no_implementation(self):
        """Test agent creation without registered implementation."""
        factory = AgentFactory()
        # Don't register any implementations
        
        request = AgentCreationRequest(
            agent_type=AgentType.ORCHESTRATOR,
            name="Test Agent",
            description="A test agent"
        )
        
        result = factory.create_agent(request)
        
        assert result.success is False
        assert result.agent is None
        assert "No implementation available" in result.error_message
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_create_agent_by_type(self, mock_memory_manager, mock_ai_client):
        """Test creating agent by type with default settings."""
        factory = AgentFactory()
        factory.register_agent_implementation(AgentType.ORCHESTRATOR, MockAgent)
        
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        result = factory.create_agent_by_type(AgentType.ORCHESTRATOR)
        
        assert result.success is True
        assert result.agent is not None
        assert result.agent.agent_type == AgentType.ORCHESTRATOR
        assert result.agent.name == "Orchestrator"  # Default name
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_create_agent_with_custom_name(self, mock_memory_manager, mock_ai_client):
        """Test creating agent with custom name."""
        factory = AgentFactory()
        factory.register_agent_implementation(AgentType.USER_TWIN, MockAgent)
        
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        result = factory.create_agent_by_type(
            AgentType.USER_TWIN,
            name="Custom User Twin",
            description="A custom user twin agent"
        )
        
        assert result.success is True
        assert result.agent.name == "Custom User Twin"
        assert result.agent.description == "A custom user twin agent"
    
    def test_get_agent(self):
        """Test getting agent by ID."""
        factory = AgentFactory()
        factory.register_agent_implementation(AgentType.ORCHESTRATOR, MockAgent)
        
        # Create an agent
        result = factory.create_agent_by_type(AgentType.ORCHESTRATOR)
        assert result.success is True
        
        # Get the agent
        retrieved_agent = factory.get_agent(result.agent_id)
        assert retrieved_agent is not None
        assert retrieved_agent.agent_id == result.agent_id
    
    def test_get_agents_by_type(self):
        """Test getting agents by type."""
        factory = AgentFactory()
        factory.register_agent_implementation(AgentType.ORCHESTRATOR, MockAgent)
        factory.register_agent_implementation(AgentType.USER_TWIN, MockAgent)
        
        # Create agents
        factory.create_agent_by_type(AgentType.ORCHESTRATOR)
        factory.create_agent_by_type(AgentType.USER_TWIN)
        factory.create_agent_by_type(AgentType.ORCHESTRATOR)
        
        orchestrators = factory.get_agents_by_type(AgentType.ORCHESTRATOR)
        user_twins = factory.get_agents_by_type(AgentType.USER_TWIN)
        
        assert len(orchestrators) == 2
        assert len(user_twins) == 1
    
    def test_remove_agent(self):
        """Test removing an agent."""
        factory = AgentFactory()
        factory.register_agent_implementation(AgentType.ORCHESTRATOR, MockAgent)
        
        # Create an agent
        result = factory.create_agent_by_type(AgentType.ORCHESTRATOR)
        assert result.success is True
        
        # Remove the agent
        removed = factory.remove_agent(result.agent_id)
        assert removed is True
        assert factory.get_agent(result.agent_id) is None
    
    def test_get_factory_status(self):
        """Test getting factory status."""
        factory = AgentFactory()
        factory.register_agent_implementation(AgentType.ORCHESTRATOR, MockAgent)
        
        status = factory.get_factory_status()
        
        assert status["factory_initialized"] is True
        assert status["registered_implementations"] == 1
        assert AgentType.ORCHESTRATOR.value in status["available_agent_types"]
        assert status["system_config_loaded"] is False


class TestConvenienceFunctions:
    """Test convenience functions for agent creation."""
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_create_orchestrator_agent(self, mock_memory_manager, mock_ai_client):
        """Test create_orchestrator_agent convenience function."""
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        # Mock the factory to return a successful result
        with patch('safehive.agents.agent_factory.get_agent_factory') as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_agent_by_type.return_value = AgentCreationResult(
                success=True,
                agent=Mock(),
                agent_id="test-id"
            )
            mock_get_factory.return_value = mock_factory
            
            result = create_orchestrator_agent()
            
            assert result.success is True
            mock_factory.create_agent_by_type.assert_called_once_with(
                AgentType.ORCHESTRATOR,
                name="Orchestrator",
                description=None,
                config_overrides=None
            )
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_create_user_twin_agent(self, mock_memory_manager, mock_ai_client):
        """Test create_user_twin_agent convenience function."""
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        with patch('safehive.agents.agent_factory.get_agent_factory') as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_agent_by_type.return_value = AgentCreationResult(
                success=True,
                agent=Mock(),
                agent_id="test-id"
            )
            mock_get_factory.return_value = mock_factory
            
            result = create_user_twin_agent()
            
            assert result.success is True
            mock_factory.create_agent_by_type.assert_called_once_with(
                AgentType.USER_TWIN,
                name="User Twin",
                description=None,
                config_overrides=None
            )
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_create_honest_vendor_agent(self, mock_memory_manager, mock_ai_client):
        """Test create_honest_vendor_agent convenience function."""
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        with patch('safehive.agents.agent_factory.get_agent_factory') as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_agent_by_type.return_value = AgentCreationResult(
                success=True,
                agent=Mock(),
                agent_id="test-id"
            )
            mock_get_factory.return_value = mock_factory
            
            result = create_honest_vendor_agent()
            
            assert result.success is True
            mock_factory.create_agent_by_type.assert_called_once_with(
                AgentType.HONEST_VENDOR,
                name="Honest Vendor",
                description=None,
                config_overrides=None
            )
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_create_malicious_vendor_agent(self, mock_memory_manager, mock_ai_client):
        """Test create_malicious_vendor_agent convenience function."""
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        with patch('safehive.agents.agent_factory.get_agent_factory') as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_agent_by_type.return_value = AgentCreationResult(
                success=True,
                agent=Mock(),
                agent_id="test-id"
            )
            mock_get_factory.return_value = mock_factory
            
            result = create_malicious_vendor_agent()
            
            assert result.success is True
            mock_factory.create_agent_by_type.assert_called_once_with(
                AgentType.MALICIOUS_VENDOR,
                name="Malicious Vendor",
                description=None,
                config_overrides=None
            )


class TestAgentFactoryIntegration:
    """Integration tests for AgentFactory."""
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_full_agent_lifecycle(self, mock_memory_manager, mock_ai_client):
        """Test complete agent lifecycle from creation to removal."""
        factory = AgentFactory()
        factory.register_agent_implementation(AgentType.ORCHESTRATOR, MockAgent)
        
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        # Create agent
        result = factory.create_agent_by_type(AgentType.ORCHESTRATOR)
        assert result.success is True
        
        agent_id = result.agent_id
        agent = result.agent
        
        # Verify agent is registered
        assert factory.get_agent(agent_id) == agent
        assert factory.registry.get_agent_count() == 1
        
        # Test agent functionality
        assert agent.is_healthy() is True
        assert agent.get_status().agent_type == AgentType.ORCHESTRATOR
        
        # Remove agent
        removed = factory.remove_agent(agent_id)
        assert removed is True
        assert factory.get_agent(agent_id) is None
        assert factory.registry.get_agent_count() == 0
    
    def test_multiple_agent_types(self):
        """Test creating multiple agents of different types."""
        factory = AgentFactory()
        factory.register_agent_implementation(AgentType.ORCHESTRATOR, MockAgent)
        factory.register_agent_implementation(AgentType.USER_TWIN, MockAgent)
        factory.register_agent_implementation(AgentType.HONEST_VENDOR, MockAgent)
        factory.register_agent_implementation(AgentType.MALICIOUS_VENDOR, MockAgent)
        
        with patch('safehive.agents.base_agent.get_ai_client') as mock_ai_client, \
             patch('safehive.agents.base_agent.get_memory_manager') as mock_memory_manager:
            
            mock_ai_client.return_value = Mock()
            mock_memory_manager.return_value = Mock()
            
            # Create one agent of each type
            results = []
            for agent_type in AgentType:
                result = factory.create_agent_by_type(agent_type)
                assert result.success is True
                results.append(result)
            
            # Verify all agents are created
            assert factory.registry.get_agent_count() == len(AgentType)
            
            # Verify agents by type
            for agent_type in AgentType:
                agents = factory.get_agents_by_type(agent_type)
                assert len(agents) == 1
                assert agents[0].agent_type == agent_type
