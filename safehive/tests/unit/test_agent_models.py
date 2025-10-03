"""
Comprehensive unit tests for Agent Models.

This module tests all functionality of the agent models including
AgentMessage, Conversation, AgentMemory, AgentState, and AgentPersonality.
"""

import pytest
import json
from datetime import datetime, timezone
from typing import Dict, Any

from safehive.models.agent_models import (
    AgentType, AgentState, MessageType,
    AgentMessage, Conversation, AgentMemory, AgentStatus, AgentPersonality,
    create_agent_message, create_conversation, create_agent_memory,
    create_agent_status, create_honest_vendor_personality, create_malicious_vendor_personality
)


class TestAgentMessage:
    """Test the AgentMessage model."""
    
    def test_agent_message_creation(self):
        """Test basic AgentMessage creation."""
        message = AgentMessage(
            content="Hello, how can I help you?",
            message_type=MessageType.REQUEST,
            sender="user_twin",
            recipient="orchestrator"
        )
        
        assert message.content == "Hello, how can I help you?"
        assert message.message_type == MessageType.REQUEST
        assert message.sender == "user_twin"
        assert message.recipient == "orchestrator"
        assert message.timestamp is not None
        assert message.metadata == {}
        assert message.message_id is None
    
    def test_agent_message_with_metadata(self):
        """Test AgentMessage creation with metadata."""
        metadata = {"priority": "high", "context": "food_ordering"}
        message = AgentMessage(
            content="Order pizza",
            message_type=MessageType.REQUEST,
            sender="user_twin",
            recipient="orchestrator",
            metadata=metadata,
            message_id="msg_123"
        )
        
        assert message.metadata == metadata
        assert message.message_id == "msg_123"
    
    def test_agent_message_serialization(self):
        """Test AgentMessage serialization to dictionary."""
        message = AgentMessage(
            content="Test message",
            message_type=MessageType.RESPONSE,
            sender="orchestrator",
            recipient="user_twin",
            metadata={"test": "data"}
        )
        
        data = message.to_dict()
        
        assert data["content"] == "Test message"
        assert data["message_type"] == "response"
        assert data["sender"] == "orchestrator"
        assert data["recipient"] == "user_twin"
        assert data["metadata"] == {"test": "data"}
        assert "timestamp" in data
        assert data["timestamp"] is not None
    
    def test_agent_message_deserialization(self):
        """Test AgentMessage creation from dictionary."""
        timestamp_str = "2024-01-01T12:00:00+00:00"
        data = {
            "content": "Test message",
            "message_type": "request",
            "sender": "user_twin",
            "recipient": "orchestrator",
            "timestamp": timestamp_str,
            "metadata": {"test": "data"},
            "message_id": "msg_123"
        }
        
        message = AgentMessage.from_dict(data)
        
        assert message.content == "Test message"
        assert message.message_type == MessageType.REQUEST
        assert message.sender == "user_twin"
        assert message.recipient == "orchestrator"
        assert message.metadata == {"test": "data"}
        assert message.message_id == "msg_123"
        assert message.timestamp.isoformat() == timestamp_str
    
    def test_agent_message_roundtrip_serialization(self):
        """Test AgentMessage serialization roundtrip."""
        original = AgentMessage(
            content="Roundtrip test",
            message_type=MessageType.SYSTEM,
            sender="system",
            recipient="orchestrator",
            metadata={"roundtrip": True}
        )
        
        data = original.to_dict()
        restored = AgentMessage.from_dict(data)
        
        assert restored.content == original.content
        assert restored.message_type == original.message_type
        assert restored.sender == original.sender
        assert restored.recipient == original.recipient
        assert restored.metadata == original.metadata


class TestConversation:
    """Test the Conversation model."""
    
    def test_conversation_creation(self):
        """Test basic Conversation creation."""
        conversation = Conversation(
            conversation_id="conv_123",
            participants=["user_twin", "orchestrator"]
        )
        
        assert conversation.conversation_id == "conv_123"
        assert conversation.participants == ["user_twin", "orchestrator"]
        assert conversation.messages == []
        assert conversation.is_active is True
        assert conversation.metadata == {}
    
    def test_conversation_add_message(self):
        """Test adding messages to a conversation."""
        conversation = Conversation(
            conversation_id="conv_123",
            participants=["user_twin", "orchestrator"]
        )
        
        message = AgentMessage(
            content="Hello",
            message_type=MessageType.REQUEST,
            sender="user_twin",
            recipient="orchestrator"
        )
        
        conversation.add_message(message)
        
        assert len(conversation.messages) == 1
        assert conversation.messages[0] == message
        assert conversation.updated_at > conversation.created_at
    
    def test_conversation_get_messages_by_type(self):
        """Test getting messages by type."""
        conversation = Conversation(
            conversation_id="conv_123",
            participants=["user_twin", "orchestrator"]
        )
        
        # Add different types of messages
        request_msg = AgentMessage(
            content="Request",
            message_type=MessageType.REQUEST,
            sender="user_twin",
            recipient="orchestrator"
        )
        response_msg = AgentMessage(
            content="Response",
            message_type=MessageType.RESPONSE,
            sender="orchestrator",
            recipient="user_twin"
        )
        
        conversation.add_message(request_msg)
        conversation.add_message(response_msg)
        
        request_messages = conversation.get_messages_by_type(MessageType.REQUEST)
        response_messages = conversation.get_messages_by_type(MessageType.RESPONSE)
        
        assert len(request_messages) == 1
        assert len(response_messages) == 1
        assert request_messages[0] == request_msg
        assert response_messages[0] == response_msg
    
    def test_conversation_get_messages_from_sender(self):
        """Test getting messages from a specific sender."""
        conversation = Conversation(
            conversation_id="conv_123",
            participants=["user_twin", "orchestrator"]
        )
        
        message1 = AgentMessage(
            content="Message 1",
            message_type=MessageType.REQUEST,
            sender="user_twin",
            recipient="orchestrator"
        )
        message2 = AgentMessage(
            content="Message 2",
            message_type=MessageType.REQUEST,
            sender="orchestrator",
            recipient="user_twin"
        )
        
        conversation.add_message(message1)
        conversation.add_message(message2)
        
        user_messages = conversation.get_messages_from("user_twin")
        orchestrator_messages = conversation.get_messages_from("orchestrator")
        
        assert len(user_messages) == 1
        assert len(orchestrator_messages) == 1
        assert user_messages[0] == message1
        assert orchestrator_messages[0] == message2
    
    def test_conversation_get_latest_message(self):
        """Test getting the latest message."""
        conversation = Conversation(
            conversation_id="conv_123",
            participants=["user_twin", "orchestrator"]
        )
        
        # Empty conversation
        assert conversation.get_latest_message() is None
        
        # Add messages
        message1 = AgentMessage(
            content="First message",
            message_type=MessageType.REQUEST,
            sender="user_twin",
            recipient="orchestrator"
        )
        message2 = AgentMessage(
            content="Second message",
            message_type=MessageType.RESPONSE,
            sender="orchestrator",
            recipient="user_twin"
        )
        
        conversation.add_message(message1)
        conversation.add_message(message2)
        
        latest = conversation.get_latest_message()
        assert latest == message2
    
    def test_conversation_serialization(self):
        """Test Conversation serialization."""
        conversation = Conversation(
            conversation_id="conv_123",
            participants=["user_twin", "orchestrator"],
            metadata={"test": "data"}
        )
        
        message = AgentMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="user_twin",
            recipient="orchestrator"
        )
        conversation.add_message(message)
        
        data = conversation.to_dict()
        
        assert data["conversation_id"] == "conv_123"
        assert data["participants"] == ["user_twin", "orchestrator"]
        assert data["metadata"] == {"test": "data"}
        assert data["is_active"] is True
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "Test message"


class TestAgentMemory:
    """Test the AgentMemory model."""
    
    def test_agent_memory_creation(self):
        """Test basic AgentMemory creation."""
        memory = AgentMemory(agent_id="agent_123")
        
        assert memory.agent_id == "agent_123"
        assert memory.conversation_history == []
        assert memory.learned_patterns == {}
        assert memory.persistent_state == {}
        assert memory.memory_size_limit == 1000
    
    def test_agent_memory_add_conversation(self):
        """Test adding conversations to memory."""
        memory = AgentMemory(agent_id="agent_123")
        
        conversation = Conversation(
            conversation_id="conv_123",
            participants=["agent_123", "user_twin"]
        )
        
        memory.add_conversation(conversation)
        
        assert len(memory.conversation_history) == 1
        assert memory.conversation_history[0] == conversation
        # Check that last_updated is recent
        assert (datetime.now() - memory.last_updated).total_seconds() < 1
    
    def test_agent_memory_size_limit(self):
        """Test memory size limit enforcement."""
        memory = AgentMemory(agent_id="agent_123", memory_size_limit=2)
        
        # Add more conversations than the limit
        for i in range(5):
            conversation = Conversation(
                conversation_id=f"conv_{i}",
                participants=["agent_123", "user_twin"]
            )
            memory.add_conversation(conversation)
        
        # Should only keep the last 2 conversations
        assert len(memory.conversation_history) == 2
        assert memory.conversation_history[0].conversation_id == "conv_3"
        assert memory.conversation_history[1].conversation_id == "conv_4"
    
    def test_agent_memory_get_conversation(self):
        """Test getting a specific conversation by ID."""
        memory = AgentMemory(agent_id="agent_123")
        
        conversation = Conversation(
            conversation_id="conv_123",
            participants=["agent_123", "user_twin"]
        )
        memory.add_conversation(conversation)
        
        found = memory.get_conversation("conv_123")
        not_found = memory.get_conversation("conv_456")
        
        assert found == conversation
        assert not_found is None
    
    def test_agent_memory_learned_patterns(self):
        """Test learned patterns management."""
        memory = AgentMemory(agent_id="agent_123")
        
        # Update learned pattern
        memory.update_learned_pattern("sql_injection", {"pattern": "SELECT * FROM"})
        
        # Get learned pattern
        pattern = memory.get_learned_pattern("sql_injection")
        assert pattern == {"pattern": "SELECT * FROM"}
        
        # Get non-existent pattern
        non_existent = memory.get_learned_pattern("non_existent")
        assert non_existent is None
    
    def test_agent_memory_persistent_state(self):
        """Test persistent state management."""
        memory = AgentMemory(agent_id="agent_123")
        
        # Update persistent state
        memory.update_persistent_state("user_preference", "vegetarian")
        
        # Get persistent state
        preference = memory.get_persistent_state("user_preference")
        assert preference == "vegetarian"
        
        # Get non-existent state
        non_existent = memory.get_persistent_state("non_existent")
        assert non_existent is None
    
    def test_agent_memory_clear(self):
        """Test clearing memory."""
        memory = AgentMemory(agent_id="agent_123")
        
        # Add some data
        conversation = Conversation(
            conversation_id="conv_123",
            participants=["agent_123", "user_twin"]
        )
        memory.add_conversation(conversation)
        memory.update_learned_pattern("test", "data")
        memory.update_persistent_state("test", "value")
        
        # Clear memory
        memory.clear_memory()
        
        assert len(memory.conversation_history) == 0
        assert memory.learned_patterns == {}
        assert memory.persistent_state == {}


class TestAgentStatus:
    """Test the AgentStatus model."""
    
    def test_agent_status_creation(self):
        """Test basic AgentStatus creation."""
        status = AgentStatus(
            agent_id="agent_123",
            agent_type=AgentType.ORCHESTRATOR,
            current_state=AgentState.IDLE
        )
        
        assert status.agent_id == "agent_123"
        assert status.agent_type == AgentType.ORCHESTRATOR
        assert status.current_state == AgentState.IDLE
        assert status.is_enabled is True
        assert status.error_count == 0
        assert status.success_count == 0
    
    def test_agent_status_update_state(self):
        """Test updating agent status."""
        status = AgentStatus(
            agent_id="agent_123",
            agent_type=AgentType.ORCHESTRATOR,
            current_state=AgentState.IDLE
        )
        
        original_activity = status.last_activity
        
        status.update_state(AgentState.ACTIVE)
        
        assert status.current_state == AgentState.ACTIVE
        assert status.last_activity > original_activity
    
    def test_agent_status_increment_counts(self):
        """Test incrementing error and success counts."""
        status = AgentStatus(
            agent_id="agent_123",
            agent_type=AgentType.ORCHESTRATOR,
            current_state=AgentState.IDLE
        )
        
        status.increment_success_count()
        status.increment_success_count()
        status.increment_error_count()
        
        assert status.success_count == 2
        assert status.error_count == 1
    
    def test_agent_status_metrics(self):
        """Test metrics management."""
        status = AgentStatus(
            agent_id="agent_123",
            agent_type=AgentType.ORCHESTRATOR,
            current_state=AgentState.IDLE
        )
        
        status.update_metric("response_time", 0.5)
        status.update_metric("accuracy", 0.95)
        
        assert status.get_metric("response_time") == 0.5
        assert status.get_metric("accuracy") == 0.95
        assert status.get_metric("non_existent") is None
    
    def test_agent_status_success_rate(self):
        """Test success rate calculation."""
        status = AgentStatus(
            agent_id="agent_123",
            agent_type=AgentType.ORCHESTRATOR,
            current_state=AgentState.IDLE
        )
        
        # No counts yet
        assert status.get_success_rate() == 0.0
        
        # Add some counts
        status.increment_success_count()
        status.increment_success_count()
        status.increment_error_count()
        
        assert status.get_success_rate() == 2/3  # 2 successes out of 3 total
    
    def test_agent_status_health_check(self):
        """Test agent health check."""
        # Healthy agent
        healthy_status = AgentStatus(
            agent_id="agent_123",
            agent_type=AgentType.ORCHESTRATOR,
            current_state=AgentState.ACTIVE
        )
        healthy_status.increment_success_count()
        healthy_status.increment_success_count()
        healthy_status.increment_error_count()  # 2/3 = 66% success rate
        
        assert healthy_status.is_healthy() is True
        
        # Unhealthy agent (disabled)
        disabled_status = AgentStatus(
            agent_id="agent_123",
            agent_type=AgentType.ORCHESTRATOR,
            current_state=AgentState.IDLE,
            is_enabled=False
        )
        assert disabled_status.is_healthy() is False
        
        # Unhealthy agent (error state)
        error_status = AgentStatus(
            agent_id="agent_123",
            agent_type=AgentType.ORCHESTRATOR,
            current_state=AgentState.ERROR
        )
        assert error_status.is_healthy() is False
        
        # Unhealthy agent (low success rate)
        low_success_status = AgentStatus(
            agent_id="agent_123",
            agent_type=AgentType.ORCHESTRATOR,
            current_state=AgentState.ACTIVE
        )
        low_success_status.increment_success_count()
        low_success_status.increment_error_count()
        low_success_status.increment_error_count()  # 1/3 = 33% success rate
        assert low_success_status.is_healthy() is False


class TestAgentPersonality:
    """Test the AgentPersonality model."""
    
    def test_agent_personality_creation(self):
        """Test basic AgentPersonality creation."""
        personality = AgentPersonality(
            agent_id="agent_123",
            personality_type="helpful"
        )
        
        assert personality.agent_id == "agent_123"
        assert personality.personality_type == "helpful"
        assert personality.communication_style == {}
        assert personality.attack_patterns == []
        assert personality.is_malicious is False
    
    def test_agent_personality_attack_patterns(self):
        """Test attack pattern management."""
        personality = AgentPersonality(
            agent_id="agent_123",
            personality_type="malicious",
            is_malicious=True
        )
        
        # Add attack patterns
        personality.add_attack_pattern("sql_injection")
        personality.add_attack_pattern("xss_attack")
        
        assert len(personality.attack_patterns) == 2
        assert "sql_injection" in personality.attack_patterns
        assert "xss_attack" in personality.attack_patterns
        
        # Check if has pattern
        assert personality.has_attack_pattern("sql_injection") is True
        assert personality.has_attack_pattern("non_existent") is False
        
        # Remove pattern
        personality.remove_attack_pattern("sql_injection")
        assert len(personality.attack_patterns) == 1
        assert "sql_injection" not in personality.attack_patterns
    
    def test_agent_personality_communication_style(self):
        """Test communication style management."""
        personality = AgentPersonality(
            agent_id="agent_123",
            personality_type="helpful"
        )
        
        personality.update_communication_style("tone", "friendly")
        personality.update_communication_style("response_time", "fast")
        
        assert personality.get_communication_style("tone") == "friendly"
        assert personality.get_communication_style("response_time") == "fast"
        assert personality.get_communication_style("non_existent") is None
    
    def test_agent_personality_knowledge_base(self):
        """Test knowledge base management."""
        personality = AgentPersonality(
            agent_id="agent_123",
            personality_type="helpful"
        )
        
        personality.add_knowledge("menu_items", ["pizza", "burger", "salad"])
        personality.add_knowledge("prices", {"pizza": 15.99, "burger": 12.99})
        
        assert personality.get_knowledge("menu_items") == ["pizza", "burger", "salad"]
        assert personality.get_knowledge("prices") == {"pizza": 15.99, "burger": 12.99}
        assert personality.get_knowledge("non_existent") is None


class TestConvenienceFunctions:
    """Test the convenience functions for creating agent models."""
    
    def test_create_agent_message(self):
        """Test create_agent_message function."""
        message = create_agent_message(
            content="Test message",
            sender="user_twin",
            recipient="orchestrator",
            message_type=MessageType.REQUEST,
            metadata={"test": "data"}
        )
        
        assert message.content == "Test message"
        assert message.sender == "user_twin"
        assert message.recipient == "orchestrator"
        assert message.message_type == MessageType.REQUEST
        assert message.metadata == {"test": "data"}
    
    def test_create_conversation(self):
        """Test create_conversation function."""
        conversation = create_conversation(
            conversation_id="conv_123",
            participants=["user_twin", "orchestrator"],
            metadata={"test": "data"}
        )
        
        assert conversation.conversation_id == "conv_123"
        assert conversation.participants == ["user_twin", "orchestrator"]
        assert conversation.metadata == {"test": "data"}
    
    def test_create_agent_memory(self):
        """Test create_agent_memory function."""
        memory = create_agent_memory("agent_123", memory_size_limit=500)
        
        assert memory.agent_id == "agent_123"
        assert memory.memory_size_limit == 500
    
    def test_create_agent_status(self):
        """Test create_agent_status function."""
        status = create_agent_status(
            agent_id="agent_123",
            agent_type=AgentType.ORCHESTRATOR,
            initial_state=AgentState.ACTIVE
        )
        
        assert status.agent_id == "agent_123"
        assert status.agent_type == AgentType.ORCHESTRATOR
        assert status.current_state == AgentState.ACTIVE
    
    def test_create_honest_vendor_personality(self):
        """Test create_honest_vendor_personality function."""
        personality = create_honest_vendor_personality("vendor_123")
        
        assert personality.agent_id == "vendor_123"
        assert personality.personality_type == "helpful"
        assert personality.is_malicious is False
        assert personality.get_communication_style("tone") == "friendly"
        assert personality.response_characteristics["honesty"] == 1.0
    
    def test_create_malicious_vendor_personality(self):
        """Test create_malicious_vendor_personality function."""
        personality = create_malicious_vendor_personality("vendor_123")
        
        assert personality.agent_id == "vendor_123"
        assert personality.personality_type == "deceptive"
        assert personality.is_malicious is True
        assert personality.get_communication_style("tone") == "manipulative"
        assert personality.response_characteristics["deceptiveness"] == 0.9
        assert "sql_injection" in personality.attack_patterns


class TestAgentModelsIntegration:
    """Test integration between different agent models."""
    
    def test_agent_workflow(self):
        """Test a complete agent workflow."""
        # Create agent state
        status = create_agent_status(
            agent_id="orchestrator_123",
            agent_type=AgentType.ORCHESTRATOR
        )
        
        # Create agent memory
        memory = create_agent_memory("orchestrator_123")
        
        # Create conversation
        conversation = create_conversation(
            conversation_id="conv_123",
            participants=["orchestrator_123", "user_twin"]
        )
        
        # Add messages to conversation
        request = create_agent_message(
            content="Order a pizza",
            sender="user_twin",
            recipient="orchestrator_123",
            message_type=MessageType.REQUEST
        )
        conversation.add_message(request)
        
        response = create_agent_message(
            content="I'll help you order a pizza",
            sender="orchestrator_123",
            recipient="user_twin",
            message_type=MessageType.RESPONSE
        )
        conversation.add_message(response)
        
        # Add conversation to memory
        memory.add_conversation(conversation)
        
        # Update agent status
        status.update_state(AgentState.ACTIVE)
        status.increment_success_count()
        
        # Verify workflow
        assert status.current_state == AgentState.ACTIVE
        assert status.success_count == 1
        assert len(memory.conversation_history) == 1
        assert conversation.get_message_count() == 2
        assert conversation.get_latest_message() == response
    
    def test_agent_models_serialization_workflow(self):
        """Test serialization workflow for agent models."""
        # Create models
        message = create_agent_message(
            content="Test message",
            sender="user_twin",
            recipient="orchestrator"
        )
        
        conversation = create_conversation(
            conversation_id="conv_123",
            participants=["user_twin", "orchestrator"]
        )
        conversation.add_message(message)
        
        memory = create_agent_memory("agent_123")
        memory.add_conversation(conversation)
        
        status = create_agent_status(
            agent_id="agent_123",
            agent_type=AgentType.ORCHESTRATOR
        )
        
        personality = create_honest_vendor_personality("agent_123")
        
        # Serialize all models
        message_data = message.to_dict()
        conversation_data = conversation.to_dict()
        memory_data = memory.to_dict()
        status_data = status.to_dict()
        personality_data = personality.to_dict()
        
        # Deserialize all models
        restored_message = AgentMessage.from_dict(message_data)
        restored_conversation = Conversation.from_dict(conversation_data)
        restored_memory = AgentMemory.from_dict(memory_data)
        restored_status = AgentStatus.from_dict(status_data)
        restored_personality = AgentPersonality.from_dict(personality_data)
        
        # Verify all models are restored correctly
        assert restored_message.content == message.content
        assert restored_conversation.conversation_id == conversation.conversation_id
        assert restored_memory.agent_id == memory.agent_id
        assert restored_status.agent_id == status.agent_id
        assert restored_personality.agent_id == personality.agent_id


if __name__ == "__main__":
    pytest.main([__file__])
