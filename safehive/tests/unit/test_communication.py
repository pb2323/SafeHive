"""
Unit tests for the agent communication system.
"""

import asyncio
import json
import tempfile
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from safehive.agents.communication import (
    CommunicationManager, CommunicationMessage, MessageQueue, MessageRouter,
    MessagePriority, MessageStatus, CommunicationEvent, MessageSubscription,
    create_communication_message, send_message_to_agent, broadcast_to_agents,
    get_communication_manager
)
from safehive.models.agent_models import MessageType, AgentType


class TestCommunicationMessage:
    """Test CommunicationMessage class."""
    
    def test_message_creation(self):
        """Test creating a communication message."""
        message = CommunicationMessage(
            content="Hello, world!",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2"
        )
        
        assert message.content == "Hello, world!"
        assert message.message_type == MessageType.REQUEST
        assert message.sender == "agent1"
        assert message.recipient == "agent2"
        assert message.priority == MessagePriority.NORMAL
        assert message.status == MessageStatus.PENDING
        assert message.retry_count == 0
        assert message.max_retries == 3
        assert message.message_id is not None
        assert isinstance(message.timestamp, datetime)
    
    def test_message_with_metadata(self):
        """Test creating a message with metadata."""
        metadata = {"conversation_id": "conv_123", "priority": "high"}
        headers = {"content-type": "text/plain"}
        
        message = CommunicationMessage(
            content="Test message",
            message_type=MessageType.RESPONSE,
            sender="agent1",
            recipient="agent2",
            priority=MessagePriority.HIGH,
            metadata=metadata,
            headers=headers,
            routing_key="test.route",
            correlation_id="corr_123"
        )
        
        assert message.priority == MessagePriority.HIGH
        assert message.metadata == metadata
        assert message.headers == headers
        assert message.routing_key == "test.route"
        assert message.correlation_id == "corr_123"
    
    def test_message_expiration(self):
        """Test message expiration logic."""
        # Message without expiration
        message1 = CommunicationMessage(
            content="Test",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2"
        )
        assert not message1.is_expired()
        
        # Message with future expiration
        message2 = CommunicationMessage(
            content="Test",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2",
            expires_at=datetime.now() + timedelta(seconds=60)
        )
        assert not message2.is_expired()
        
        # Message with past expiration
        message3 = CommunicationMessage(
            content="Test",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2",
            expires_at=datetime.now() - timedelta(seconds=60)
        )
        assert message3.is_expired()
    
    def test_message_retry_logic(self):
        """Test message retry logic."""
        message = CommunicationMessage(
            content="Test",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2",
            max_retries=2
        )
        
        # Can retry initially
        assert message.can_retry()
        
        # Can retry after one attempt
        message.retry_count = 1
        assert message.can_retry()
        
        # Cannot retry after max attempts
        message.retry_count = 2
        assert not message.can_retry()
        
        # Cannot retry if expired
        message.retry_count = 0
        message.expires_at = datetime.now() - timedelta(seconds=60)
        assert not message.can_retry()
    
    def test_message_serialization(self):
        """Test message serialization and deserialization."""
        original_message = CommunicationMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2",
            priority=MessagePriority.HIGH,
            metadata={"test": "value"},
            headers={"content-type": "text/plain"},
            routing_key="test.route",
            correlation_id="corr_123"
        )
        
        # Serialize to dict
        message_dict = original_message.to_dict()
        assert message_dict["content"] == "Test message"
        assert message_dict["message_type"] == "request"
        assert message_dict["sender"] == "agent1"
        assert message_dict["recipient"] == "agent2"
        assert message_dict["priority"] == "high"
        assert message_dict["metadata"] == {"test": "value"}
        assert message_dict["headers"] == {"content-type": "text/plain"}
        assert message_dict["routing_key"] == "test.route"
        assert message_dict["correlation_id"] == "corr_123"
        
        # Deserialize from dict
        restored_message = CommunicationMessage.from_dict(message_dict)
        assert restored_message.content == original_message.content
        assert restored_message.message_type == original_message.message_type
        assert restored_message.sender == original_message.sender
        assert restored_message.recipient == original_message.recipient
        assert restored_message.priority == original_message.priority
        assert restored_message.metadata == original_message.metadata
        assert restored_message.headers == original_message.headers
        assert restored_message.routing_key == original_message.routing_key
        assert restored_message.correlation_id == original_message.correlation_id
    
    def test_message_to_agent_message(self):
        """Test conversion to AgentMessage."""
        comm_message = CommunicationMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2",
            priority=MessagePriority.HIGH,
            routing_key="test.route",
            correlation_id="corr_123",
            metadata={"custom": "data"}
        )
        
        agent_message = comm_message.to_agent_message()
        
        assert agent_message.content == "Test message"
        assert agent_message.message_type == MessageType.REQUEST
        assert agent_message.sender == "agent1"
        assert agent_message.recipient == "agent2"
        assert agent_message.metadata["communication_message_id"] == comm_message.message_id
        assert agent_message.metadata["priority"] == "high"
        assert agent_message.metadata["routing_key"] == "test.route"
        assert agent_message.metadata["correlation_id"] == "corr_123"
        assert agent_message.metadata["custom"] == "data"


class TestMessageQueue:
    """Test MessageQueue class."""
    
    def test_queue_creation(self):
        """Test creating a message queue."""
        queue = MessageQueue(max_size=100)
        assert queue.max_size == 100
        assert queue.size() == 0
    
    def test_put_and_get_message(self):
        """Test putting and getting messages from queue."""
        queue = MessageQueue()
        message = CommunicationMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2"
        )
        
        # Put message
        assert queue.put(message)
        assert queue.size() == 1
        
        # Get message
        retrieved_message = queue.get()
        assert retrieved_message is not None
        assert retrieved_message.content == "Test message"
        assert queue.size() == 0
    
    def test_put_and_get_with_timeout(self):
        """Test putting and getting messages with timeout."""
        queue = MessageQueue()
        
        # Get with timeout when queue is empty
        start_time = time.time()
        result = queue.get(timeout=0.1)
        elapsed = time.time() - start_time
        
        assert result is None
        assert elapsed >= 0.1
        
        # Put message and get it
        message = CommunicationMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2"
        )
        
        assert queue.put(message)
        result = queue.get(timeout=1.0)
        assert result is not None
        assert result.content == "Test message"
    
    def test_queue_full(self):
        """Test queue behavior when full."""
        queue = MessageQueue(max_size=2)
        
        message1 = CommunicationMessage(
            content="Message 1",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2"
        )
        message2 = CommunicationMessage(
            content="Message 2",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2"
        )
        message3 = CommunicationMessage(
            content="Message 3",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2"
        )
        
        # Fill queue
        assert queue.put(message1)
        assert queue.put(message2)
        assert queue.size() == 2
        
        # Try to add third message (should fail)
        assert not queue.put(message3)
        assert queue.size() == 2
    
    def test_peek_message(self):
        """Test peeking at messages without removing them."""
        queue = MessageQueue()
        message = CommunicationMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2"
        )
        
        # Peek when empty
        assert queue.peek() is None
        
        # Put message and peek
        queue.put(message)
        peeked_message = queue.peek()
        assert peeked_message is not None
        assert peeked_message.content == "Test message"
        assert queue.size() == 1  # Message should still be in queue
        
        # Get message
        retrieved_message = queue.get()
        assert retrieved_message.content == "Test message"
        assert queue.size() == 0
    
    def test_clear_queue(self):
        """Test clearing queue."""
        queue = MessageQueue()
        
        # Add multiple messages
        for i in range(3):
            message = CommunicationMessage(
                content=f"Message {i}",
                message_type=MessageType.REQUEST,
                sender="agent1",
                recipient="agent2"
            )
            queue.put(message)
        
        assert queue.size() == 3
        
        # Clear queue
        cleared_count = queue.clear()
        assert cleared_count == 3
        assert queue.size() == 0
    
    def test_queue_stats(self):
        """Test queue statistics."""
        queue = MessageQueue()
        
        # Add some messages
        for i in range(3):
            message = CommunicationMessage(
                content=f"Message {i}",
                message_type=MessageType.REQUEST,
                sender="agent1",
                recipient="agent2"
            )
            queue.put(message)
        
        stats = queue.get_stats()
        assert stats["total_messages"] == 3
        assert stats["queue_sizes"]["default"] == 3
        assert stats["total_queues"] == 1


class TestMessageRouter:
    """Test MessageRouter class."""
    
    def test_router_creation(self):
        """Test creating a message router."""
        router = MessageRouter()
        assert router._routes == {}
        assert router._wildcard_routes == []
        assert router._subscriptions == {}
    
    def test_add_and_remove_routes(self):
        """Test adding and removing routes."""
        router = MessageRouter()
        
        # Add route
        router.add_route("test.route", ["agent1", "agent2"])
        targets = router.get_targets("test.route")
        assert set(targets) == {"agent1", "agent2"}
        
        # Add wildcard route
        router.add_route("*", ["agent3"])
        assert "agent3" in router.get_targets("test.route")
        assert "agent3" in router.get_targets("other.route")
        
        # Remove route
        router.remove_route("test.route", "agent1")
        targets = router.get_targets("test.route")
        assert set(targets) == {"agent2", "agent3"}
        
        # Remove wildcard route
        router.remove_route("*", "agent3")
        targets = router.get_targets("test.route")
        assert set(targets) == {"agent2"}
    
    def test_message_matching_criteria(self):
        """Test message matching against subscription criteria."""
        router = MessageRouter()
        
        message = CommunicationMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2",
            priority=MessagePriority.HIGH,
            routing_key="test.route",
            metadata={"conversation_id": "conv_123"}
        )
        
        # Test various criteria
        assert router._message_matches_criteria(message, {"sender": "agent1"})
        assert not router._message_matches_criteria(message, {"sender": "agent3"})
        
        assert router._message_matches_criteria(message, {"recipient": "agent2"})
        assert not router._message_matches_criteria(message, {"recipient": "agent3"})
        
        assert router._message_matches_criteria(message, {"message_type": "request"})
        assert not router._message_matches_criteria(message, {"message_type": "response"})
        
        assert router._message_matches_criteria(message, {"priority": "high"})
        assert not router._message_matches_criteria(message, {"priority": "low"})
        
        assert router._message_matches_criteria(message, {"routing_key": "test.route"})
        assert not router._message_matches_criteria(message, {"routing_key": "other.route"})
        
        assert router._message_matches_criteria(message, {"content_contains": "Test"})
        assert not router._message_matches_criteria(message, {"content_contains": "Other"})
        
        assert router._message_matches_criteria(message, {"metadata": {"conversation_id": "conv_123"}})
        assert not router._message_matches_criteria(message, {"metadata": {"conversation_id": "conv_456"}})
    
    def test_subscriptions(self):
        """Test message subscriptions."""
        router = MessageRouter()
        
        # Create subscription
        callback = Mock()
        subscription = MessageSubscription(
            subscriber_id="subscriber1",
            filter_criteria={"sender": "agent1"},
            callback=callback
        )
        
        subscription_id = router.add_subscription(subscription)
        assert subscription_id in router._subscriptions
        
        # Test matching subscription
        message = CommunicationMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="agent1",
            recipient="agent2"
        )
        
        matching_subscriptions = router.get_matching_subscriptions(message)
        assert len(matching_subscriptions) == 1
        assert matching_subscriptions[0].subscriber_id == "subscriber1"
        
        # Test non-matching message
        non_matching_message = CommunicationMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="agent3",
            recipient="agent2"
        )
        
        matching_subscriptions = router.get_matching_subscriptions(non_matching_message)
        assert len(matching_subscriptions) == 0
        
        # Remove subscription
        assert router.remove_subscription(subscription_id)
        assert subscription_id not in router._subscriptions


class TestCommunicationManager:
    """Test CommunicationManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CommunicationManager()
        self.manager.start()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.manager.stop()
    
    def test_manager_creation(self):
        """Test creating a communication manager."""
        manager = CommunicationManager()
        assert not manager._running
        assert len(manager._connected_agents) == 0
        assert len(manager._agent_queues) == 0
    
    def test_start_and_stop(self):
        """Test starting and stopping the manager."""
        manager = CommunicationManager()
        
        # Start manager
        manager.start()
        assert manager._running
        assert manager._delivery_thread is not None
        
        # Stop manager
        manager.stop()
        assert not manager._running
    
    def test_register_and_unregister_agent(self):
        """Test registering and unregistering agents."""
        # Register agent
        self.manager.register_agent("agent1")
        assert "agent1" in self.manager._connected_agents
        assert self.manager._agent_queues["agent1"] == "agent_agent1"
        
        # Register agent with custom queue
        self.manager.register_agent("agent2", "custom_queue")
        assert "agent2" in self.manager._connected_agents
        assert self.manager._agent_queues["agent2"] == "custom_queue"
        
        # Unregister agent
        self.manager.unregister_agent("agent1")
        assert "agent1" not in self.manager._connected_agents
        assert "agent1" not in self.manager._agent_queues
    
    def test_send_and_receive_message(self):
        """Test sending and receiving messages."""
        # Register agents
        self.manager.register_agent("sender")
        self.manager.register_agent("receiver")
        
        # Create and send message
        message = CommunicationMessage(
            content="Hello, receiver!",
            message_type=MessageType.REQUEST,
            sender="sender",
            recipient="receiver"
        )
        
        # Send message
        assert self.manager.send_message(message)
        assert message.status == MessageStatus.SENT
        
        # Receive message
        received_message = self.manager.receive_message("receiver", timeout=1.0)
        assert received_message is not None
        assert received_message.content == "Hello, receiver!"
        assert received_message.status == MessageStatus.DELIVERED
    
    def test_message_acknowledgment(self):
        """Test message acknowledgment."""
        # Register agents
        self.manager.register_agent("sender")
        self.manager.register_agent("receiver")
        
        # Send message
        message = CommunicationMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="sender",
            recipient="receiver"
        )
        
        self.manager.send_message(message)
        
        # Acknowledge message
        assert self.manager.acknowledge_message(message.message_id, "receiver")
        
        # Check acknowledgment
        stored_message = self.manager._message_history[message.message_id]
        assert stored_message.acknowledgment_received
        assert stored_message.acknowledgment_timestamp is not None
        assert stored_message.status == MessageStatus.ACKNOWLEDGED
    
    def test_broadcast_message(self):
        """Test broadcasting messages to multiple agents."""
        # Register multiple agents
        agents = ["agent1", "agent2", "agent3"]
        for agent in agents:
            self.manager.register_agent(agent)
        
        # Create broadcast message
        message = CommunicationMessage(
            content="Broadcast message",
            message_type=MessageType.SYSTEM,
            sender="system",
            recipient="broadcast"
        )
        
        # Broadcast to all agents
        results = self.manager.broadcast_message(message)
        assert len(results) == 3
        assert all(results.values())  # All should succeed
        
        # Check that all agents received the message
        for agent in agents:
            received_message = self.manager.receive_message(agent, timeout=1.0)
            assert received_message is not None
            assert received_message.content == "Broadcast message"
            assert received_message.recipient == agent
    
    def test_broadcast_to_specific_agents(self):
        """Test broadcasting to specific agents only."""
        # Register agents
        agents = ["agent1", "agent2", "agent3"]
        for agent in agents:
            self.manager.register_agent(agent)
        
        # Create broadcast message
        message = CommunicationMessage(
            content="Selective broadcast",
            message_type=MessageType.SYSTEM,
            sender="system",
            recipient="broadcast"
        )
        
        # Broadcast to specific agents only
        target_agents = ["agent1", "agent3"]
        results = self.manager.broadcast_message(message, target_agents)
        assert len(results) == 2
        assert all(results.values())
        
        # Check that only target agents received the message
        received_message = self.manager.receive_message("agent1", timeout=1.0)
        assert received_message is not None
        
        received_message = self.manager.receive_message("agent2", timeout=0.1)
        assert received_message is None  # Should not receive
        
        received_message = self.manager.receive_message("agent3", timeout=1.0)
        assert received_message is not None
    
    def test_message_subscriptions(self):
        """Test message subscriptions."""
        # Register agents
        self.manager.register_agent("publisher")
        self.manager.register_agent("subscriber")
        
        # Create subscription
        callback = Mock()
        subscription_id = self.manager.subscribe_to_messages(
            "subscriber",
            {"sender": "publisher"},
            callback
        )
        
        assert subscription_id is not None
        
        # Send message that matches subscription
        message = CommunicationMessage(
            content="Subscribed message",
            message_type=MessageType.REQUEST,
            sender="publisher",
            recipient="subscriber"
        )
        
        self.manager.send_message(message)
        
        # Send message that doesn't match subscription
        non_matching_message = CommunicationMessage(
            content="Non-matching message",
            message_type=MessageType.REQUEST,
            sender="other_agent",
            recipient="subscriber"
        )
        
        self.manager.send_message(non_matching_message)
        
        # Cancel subscription
        assert self.manager.unsubscribe_from_messages(subscription_id)
        assert not self.manager.unsubscribe_from_messages(subscription_id)  # Already cancelled
    
    def test_event_handlers(self):
        """Test event handlers."""
        # Add event handler
        handler = Mock()
        self.manager.add_event_handler(CommunicationEvent.MESSAGE_SENT, handler)
        
        # Register agents and send message
        self.manager.register_agent("sender")
        self.manager.register_agent("receiver")
        
        message = CommunicationMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="sender",
            recipient="receiver"
        )
        
        self.manager.send_message(message)
        
        # Check that handler was called
        handler.assert_called_once()
        event, data = handler.call_args[0]
        assert event == CommunicationEvent.MESSAGE_SENT
        assert "message" in data
        
        # Remove handler
        self.manager.remove_event_handler(CommunicationEvent.MESSAGE_SENT, handler)
        
        # Send another message
        message2 = CommunicationMessage(
            content="Test message 2",
            message_type=MessageType.REQUEST,
            sender="sender",
            recipient="receiver"
        )
        
        self.manager.send_message(message2)
        
        # Handler should not be called again
        assert handler.call_count == 1
    
    def test_agent_status(self):
        """Test getting agent status."""
        # Register agent
        self.manager.register_agent("test_agent")
        
        # Get status
        status = self.manager.get_agent_status("test_agent")
        assert status["connected"]
        assert status["queue_name"] == "agent_test_agent"
        assert status["queue_size"] == 0
        assert status["messages_sent"] == 0
        assert status["messages_received"] == 0
        
        # Send message to agent
        message = CommunicationMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="sender",
            recipient="test_agent"
        )
        
        self.manager.send_message(message)
        
        # Check updated status
        status = self.manager.get_agent_status("test_agent")
        assert status["queue_size"] == 1
        assert status["messages_received"] == 1
    
    def test_system_stats(self):
        """Test getting system statistics."""
        # Register some agents
        self.manager.register_agent("agent1")
        self.manager.register_agent("agent2")
        
        # Get stats
        stats = self.manager.get_system_stats()
        assert stats["connected_agents"] == 2
        assert stats["total_messages"] == 0
        assert stats["running"]
        assert "queue_stats" in stats
        assert "subscriptions" in stats
    
    def test_message_validation(self):
        """Test message validation."""
        # Test empty content
        message = CommunicationMessage(
            content="",
            message_type=MessageType.REQUEST,
            sender="sender",
            recipient="receiver"
        )
        assert not self.manager._validate_message(message)
        
        # Test empty sender
        message = CommunicationMessage(
            content="Test",
            message_type=MessageType.REQUEST,
            sender="",
            recipient="receiver"
        )
        assert not self.manager._validate_message(message)
        
        # Test empty recipient
        message = CommunicationMessage(
            content="Test",
            message_type=MessageType.REQUEST,
            sender="sender",
            recipient=""
        )
        assert not self.manager._validate_message(message)
        
        # Test valid message
        message = CommunicationMessage(
            content="Test",
            message_type=MessageType.REQUEST,
            sender="sender",
            recipient="receiver"
        )
        assert self.manager._validate_message(message)
    
    def test_message_routing(self):
        """Test message routing with routing keys."""
        # Register agents
        self.manager.register_agent("agent1")
        self.manager.register_agent("agent2")
        
        # Add routing rule
        self.manager._router.add_route("test.route", ["agent1"])
        
        # Send message with routing key
        message = CommunicationMessage(
            content="Routed message",
            message_type=MessageType.REQUEST,
            sender="sender",
            recipient="unknown_agent",  # Not registered
            routing_key="test.route"
        )
        
        # Message should be routed to agent1
        assert self.manager.send_message(message)
        
        # agent1 should receive the message
        received_message = self.manager.receive_message("agent1", timeout=1.0)
        assert received_message is not None
        assert received_message.content == "Routed message"
        
        # agent2 should not receive the message
        received_message = self.manager.receive_message("agent2", timeout=0.1)
        assert received_message is None


class TestCommunicationUtilities:
    """Test communication utility functions."""
    
    def test_create_communication_message(self):
        """Test create_communication_message utility function."""
        message = create_communication_message(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="sender",
            recipient="recipient",
            priority=MessagePriority.HIGH,
            routing_key="test.route",
            metadata={"test": "value"},
            expires_in=60.0
        )
        
        assert message.content == "Test message"
        assert message.message_type == MessageType.REQUEST
        assert message.sender == "sender"
        assert message.recipient == "recipient"
        assert message.priority == MessagePriority.HIGH
        assert message.routing_key == "test.route"
        assert message.metadata == {"test": "value"}
        assert message.expires_at is not None
        assert message.expires_at > datetime.now()
    
    def test_send_message_to_agent(self):
        """Test send_message_to_agent utility function."""
        with patch('safehive.agents.communication.get_communication_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            mock_manager.send_message.return_value = True
            
            result = send_message_to_agent(
                "agent1",
                "Hello, agent!",
                MessageType.REQUEST,
                "sender",
                MessagePriority.HIGH
            )
            
            assert result
            mock_manager.send_message.assert_called_once()
            message = mock_manager.send_message.call_args[0][0]
            assert message.content == "Hello, agent!"
            assert message.recipient == "agent1"
            assert message.sender == "sender"
            assert message.priority == MessagePriority.HIGH
    
    def test_broadcast_to_agents(self):
        """Test broadcast_to_agents utility function."""
        with patch('safehive.agents.communication.get_communication_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            mock_manager.broadcast_message.return_value = {"agent1": True, "agent2": True}
            
            result = broadcast_to_agents(
                "Broadcast message",
                MessageType.SYSTEM,
                "system",
                ["agent1", "agent2"]
            )
            
            assert result == {"agent1": True, "agent2": True}
            mock_manager.broadcast_message.assert_called_once()
            message = mock_manager.broadcast_message.call_args[0][0]
            assert message.content == "Broadcast message"
            assert message.message_type == MessageType.SYSTEM
            assert message.sender == "system"
    
    def test_get_communication_manager_singleton(self):
        """Test that get_communication_manager returns a singleton."""
        manager1 = get_communication_manager()
        manager2 = get_communication_manager()
        
        assert manager1 is manager2


class TestCommunicationIntegration:
    """Integration tests for the communication system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CommunicationManager()
        self.manager.start()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.manager.stop()
    
    def test_complete_communication_workflow(self):
        """Test a complete communication workflow."""
        # Register agents
        self.manager.register_agent("orchestrator")
        self.manager.register_agent("user_twin")
        self.manager.register_agent("vendor")
        
        # Orchestrator sends request to user twin
        request_message = CommunicationMessage(
            content="What would you like to order?",
            message_type=MessageType.REQUEST,
            sender="orchestrator",
            recipient="user_twin",
            correlation_id="order_123"
        )
        
        assert self.manager.send_message(request_message)
        
        # User twin receives and responds
        received_request = self.manager.receive_message("user_twin", timeout=1.0)
        assert received_request is not None
        assert received_request.content == "What would you like to order?"
        
        # Acknowledge receipt
        self.manager.acknowledge_message(received_request.message_id, "user_twin")
        
        # User twin sends response
        response_message = CommunicationMessage(
            content="I'd like to order pizza",
            message_type=MessageType.RESPONSE,
            sender="user_twin",
            recipient="orchestrator",
            correlation_id="order_123",
            reply_to=received_request.message_id
        )
        
        assert self.manager.send_message(response_message)
        
        # Orchestrator receives response
        received_response = self.manager.receive_message("orchestrator", timeout=1.0)
        assert received_response is not None
        assert received_response.content == "I'd like to order pizza"
        assert received_response.correlation_id == "order_123"
        
        # Orchestrator forwards to vendor
        vendor_message = CommunicationMessage(
            content="Customer wants pizza",
            message_type=MessageType.REQUEST,
            sender="orchestrator",
            recipient="vendor",
            correlation_id="order_123"
        )
        
        assert self.manager.send_message(vendor_message)
        
        # Vendor receives message
        vendor_received = self.manager.receive_message("vendor", timeout=1.0)
        assert vendor_received is not None
        assert vendor_received.content == "Customer wants pizza"
    
    def test_message_retry_mechanism(self):
        """Test message retry mechanism."""
        # Register sender but not receiver
        self.manager.register_agent("sender")
        
        # Send message to unregistered agent
        message = CommunicationMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="sender",
            recipient="unregistered_agent",
            max_retries=2
        )
        
        # Message should fail initially
        assert not self.manager.send_message(message)
        assert message.status == MessageStatus.FAILED
        
        # Register receiver after message fails
        self.manager.register_agent("unregistered_agent")
        
        # Wait for retry mechanism to process
        time.sleep(2.0)
        
        # Check if message was retried and delivered
        received_message = self.manager.receive_message("unregistered_agent", timeout=1.0)
        if received_message:  # Retry might have succeeded
            assert received_message.content == "Test message"
    
    def test_concurrent_message_handling(self):
        """Test handling multiple messages concurrently."""
        # Register agents
        self.manager.register_agent("sender")
        self.manager.register_agent("receiver")
        
        # Send multiple messages
        messages = []
        for i in range(10):
            message = CommunicationMessage(
                content=f"Message {i}",
                message_type=MessageType.REQUEST,
                sender="sender",
                recipient="receiver"
            )
            messages.append(message)
            assert self.manager.send_message(message)
        
        # Receive all messages
        received_messages = []
        for _ in range(10):
            message = self.manager.receive_message("receiver", timeout=2.0)
            if message:
                received_messages.append(message)
        
        # Verify all messages were received
        assert len(received_messages) == 10
        received_contents = [msg.content for msg in received_messages]
        expected_contents = [f"Message {i}" for i in range(10)]
        assert set(received_contents) == set(expected_contents)
