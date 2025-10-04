"""
Unit tests for the base agent implementation.

This module tests the BaseAgent class and related functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

from safehive.agents.base_agent import (
    BaseAgent, AgentCapabilities, AgentConfiguration, 
    create_agent_config, get_agent_status_summary
)
from safehive.models.agent_models import AgentType, AgentState
from safehive.tools.base_tools import BaseSafeHiveTool


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent for testing."""
    
    def _get_system_prompt(self) -> str:
        return "You are a test agent for unit testing purposes."


class TestAgentCapabilities:
    """Test AgentCapabilities enum."""
    
    def test_capabilities_enum(self):
        """Test that capabilities enum has expected values."""
        assert AgentCapabilities.REASONING.value == "reasoning"
        assert AgentCapabilities.MEMORY.value == "memory"
        assert AgentCapabilities.LEARNING.value == "learning"
        assert AgentCapabilities.COMMUNICATION.value == "communication"
        assert AgentCapabilities.ORDER_MANAGEMENT.value == "order_management"
        assert AgentCapabilities.VENDOR_INTERACTION.value == "vendor_interaction"
        assert AgentCapabilities.SECURITY_ANALYSIS.value == "security_analysis"
        assert AgentCapabilities.USER_SIMULATION.value == "user_simulation"
        assert AgentCapabilities.ORCHESTRATION.value == "orchestration"


class TestAgentConfiguration:
    """Test AgentConfiguration dataclass."""
    
    def test_agent_configuration_creation(self):
        """Test creating an agent configuration."""
        config = AgentConfiguration(
            agent_type=AgentType.ORCHESTRATOR,
            name="Test Agent",
            description="A test agent",
            capabilities=[AgentCapabilities.REASONING, AgentCapabilities.MEMORY]
        )
        
        assert config.agent_type == AgentType.ORCHESTRATOR
        assert config.name == "Test Agent"
        assert config.description == "A test agent"
        assert len(config.capabilities) == 2
        assert AgentCapabilities.REASONING in config.capabilities
        assert AgentCapabilities.MEMORY in config.capabilities
    
    def test_agent_configuration_defaults(self):
        """Test agent configuration with default values."""
        config = AgentConfiguration()
        
        assert config.agent_type == AgentType.ORCHESTRATOR
        assert config.name == "SafeHive Agent"
        assert config.ai_model == "llama2:7b"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
        assert config.memory_type == "conversation_buffer"
        assert config.enable_monitoring is True
        assert config.enable_metrics is True
        assert config.enable_logging is True


class TestBaseAgent:
    """Test BaseAgent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AgentConfiguration(
            agent_type=AgentType.ORCHESTRATOR,
            name="Test Agent",
            description="A test agent for unit testing"
        )
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_base_agent_initialization(self, mock_memory_manager, mock_ai_client):
        """Test base agent initialization."""
        # Mock dependencies
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        agent = TestAgent(self.config)
        
        assert agent.agent_id == self.config.agent_id
        assert agent.agent_type == self.config.agent_type
        assert agent.name == self.config.name
        assert agent.description == self.config.description
        assert agent._status == AgentState.IDLE
        assert agent._total_requests == 0
        assert agent._successful_requests == 0
        assert agent._failed_requests == 0
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_base_agent_system_prompt(self, mock_memory_manager, mock_ai_client):
        """Test that system prompt is correctly implemented."""
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        agent = TestAgent(self.config)
        
        # Test that the abstract method is implemented
        prompt = agent._get_system_prompt()
        assert prompt == "You are a test agent for unit testing purposes."
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_base_agent_get_status(self, mock_memory_manager, mock_ai_client):
        """Test getting agent status."""
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        agent = TestAgent(self.config)
        status = agent.get_status()
        
        assert status.agent_id == agent.agent_id
        assert status.agent_type == agent.agent_type
        assert status.configuration["name"] == agent.name
        assert status.current_state == AgentState.IDLE
        assert status.metrics["total_requests"] == 0
        assert status.metrics["successful_requests"] == 0
        assert status.metrics["failed_requests"] == 0
        assert status.metrics["average_response_time"] == 0.0
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_base_agent_health_check(self, mock_memory_manager, mock_ai_client):
        """Test agent health checking."""
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        agent = TestAgent(self.config)
        
        # Agent should be healthy initially
        assert agent.is_healthy() is True
        
        # Simulate old last activity
        agent._last_activity = datetime.now() - timedelta(hours=2)
        assert agent.is_healthy() is False
        
        # Reset and test error rate
        agent._last_activity = datetime.now()
        agent._total_requests = 10
        agent._failed_requests = 6  # 60% error rate
        assert agent.is_healthy() is False
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_base_agent_metrics(self, mock_memory_manager, mock_ai_client):
        """Test agent metrics collection."""
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        agent = TestAgent(self.config)
        metrics = agent.get_metrics()
        
        assert "agent_id" in metrics
        assert "agent_type" in metrics
        assert "name" in metrics
        assert "status" in metrics
        assert "total_requests" in metrics
        assert "successful_requests" in metrics
        assert "failed_requests" in metrics
        assert "success_rate" in metrics
        assert "average_response_time" in metrics
        assert "memory_size" in metrics
        assert "last_activity" in metrics
        assert "is_healthy" in metrics
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_base_agent_conversation_history(self, mock_memory_manager, mock_ai_client):
        """Test conversation history management."""
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        agent = TestAgent(self.config)
        
        # Initially empty
        history = asyncio.run(agent.get_conversation_history())
        assert len(history) == 0
        
        # Test with limit
        history_limited = asyncio.run(agent.get_conversation_history(limit=5))
        assert len(history_limited) == 0
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_base_agent_tool_management(self, mock_memory_manager, mock_ai_client):
        """Test tool management functionality."""
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        agent = TestAgent(self.config)
        
        # Initially no tools
        tools = agent.get_available_tools()
        assert len(tools) == 0
        
        # Add a mock tool
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        agent.add_tool(mock_tool)
        
        tools = agent.get_available_tools()
        assert len(tools) == 1
        assert "test_tool" in tools
        
        # Remove the tool
        agent.remove_tool("test_tool")
        tools = agent.get_available_tools()
        assert len(tools) == 0


# class TestSafeHiveAgentCallbackHandler:
#     """Test SafeHiveAgentCallbackHandler."""
#     
#     def test_callback_handler_initialization(self):
#         """Test callback handler initialization."""
#         mock_agent = Mock()
#         handler = SafeHiveAgentCallbackHandler(mock_agent)
#         
#         assert handler.agent == mock_agent
#     
#     def test_callback_methods(self):
#         """Test callback handler methods."""
#         mock_agent = Mock()
#         mock_agent.name = "Test Agent"
#         handler = SafeHiveAgentCallbackHandler(mock_agent)
#         
#         # Test agent action callback
#         mock_action = Mock()
#         mock_action.tool = "test_tool"
#         handler.on_agent_action(mock_action)
#         
#         # Test agent finish callback
#         mock_finish = Mock()
#         mock_finish.return_values = {"result": "success"}
#         handler.on_agent_finish(mock_finish)
#         
#         # Test tool start callback
#         mock_serialized = {"name": "test_tool"}
#         handler.on_tool_start(mock_serialized, "test input")
#         
#         # Test tool end callback
#         handler.on_tool_end("test output")


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_agent_config(self):
        """Test create_agent_config utility function."""
        config = create_agent_config(
            agent_type=AgentType.USER_TWIN,
            name="Test User Twin",
            description="A test user twin agent",
            capabilities=[AgentCapabilities.USER_SIMULATION]
        )
        
        assert config.agent_type == AgentType.USER_TWIN
        assert config.name == "Test User Twin"
        assert config.description == "A test user twin agent"
        assert AgentCapabilities.USER_SIMULATION in config.capabilities
    
    def test_get_agent_status_summary(self):
        """Test get_agent_status_summary utility function."""
        # Create mock agents
        mock_agent1 = Mock()
        mock_agent1.is_healthy.return_value = True
        mock_agent1._total_requests = 10
        mock_agent1._successful_requests = 9
        mock_agent1.get_metrics.return_value = {"agent_id": "1", "health": "good"}
        
        mock_agent2 = Mock()
        mock_agent2.is_healthy.return_value = False
        mock_agent2._total_requests = 5
        mock_agent2._successful_requests = 3
        mock_agent2.get_metrics.return_value = {"agent_id": "2", "health": "poor"}
        
        agents = [mock_agent1, mock_agent2]
        summary = get_agent_status_summary(agents)
        
        assert summary["total_agents"] == 2
        assert summary["healthy_agents"] == 1
        assert summary["unhealthy_agents"] == 1
        assert summary["health_percentage"] == 50.0
        assert summary["total_requests"] == 15
        assert summary["successful_requests"] == 12
        assert summary["overall_success_rate"] == 12 / 15
        assert len(summary["agents"]) == 2


class TestBaseAgentIntegration:
    """Integration tests for BaseAgent."""
    
    @patch('safehive.agents.base_agent.get_ai_client')
    @patch('safehive.agents.base_agent.get_memory_manager')
    def test_agent_lifecycle(self, mock_memory_manager, mock_ai_client):
        """Test complete agent lifecycle."""
        # Mock dependencies
        mock_ai_client.return_value = Mock()
        mock_memory_manager.return_value = Mock()
        
        # Create agent
        config = AgentConfiguration(name="Lifecycle Test Agent")
        agent = TestAgent(config)
        
        # Check initial state
        assert agent._status == AgentState.IDLE
        assert agent.is_healthy() is True
        
        # Simulate some activity
        agent._total_requests = 5
        agent._successful_requests = 4
        agent._failed_requests = 1
        
        # Check metrics
        metrics = agent.get_metrics()
        assert metrics["total_requests"] == 5
        assert metrics["successful_requests"] == 4
        assert metrics["success_rate"] == 0.8
        
        # Test shutdown
        asyncio.run(agent.shutdown())
        assert agent._status == AgentState.STOPPED


if __name__ == "__main__":
    pytest.main([__file__])
