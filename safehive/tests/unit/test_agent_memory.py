"""
Unit tests for Agent Memory Management.

This module tests the agent memory management functionality for LangChain memory components.
"""

import pytest
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from safehive.utils.agent_memory import (
    SafeHiveMemoryManager, MemoryManagerFactory,
    get_memory_manager, create_memory_manager, remove_memory_manager,
    list_memory_managers, save_all_memories, load_all_memories
)
from safehive.models.agent_models import AgentMessage, MessageType, Conversation


class TestSafeHiveMemoryManager:
    """Test the SafeHiveMemoryManager class."""
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                memory_type="buffer",
                persist_directory=temp_dir
            )
            
            assert manager.agent_id == "test_agent"
            assert manager.memory_type == "buffer"
            assert manager.persist_directory == temp_dir
            assert manager.memory is not None
            assert manager.memory_stats["total_messages"] == 0
    
    def test_memory_manager_default_initialization(self):
        """Test memory manager with default parameters."""
        manager = SafeHiveMemoryManager(agent_id="test_agent")
        
        assert manager.agent_id == "test_agent"
        assert manager.memory_type == "buffer"
        assert manager.memory is not None
        assert "memory/test_agent" in manager.persist_directory
    
    def test_buffer_memory_setup(self):
        """Test buffer memory setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                memory_type="buffer",
                persist_directory=temp_dir
            )
            
            assert manager.memory is not None
            assert hasattr(manager.memory, 'chat_memory')
    
    def test_window_memory_setup(self):
        """Test window memory setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                memory_type="window",
                memory_config={"window_size": 5},
                persist_directory=temp_dir
            )
            
            assert manager.memory is not None
            assert hasattr(manager.memory, 'chat_memory')
    
    @patch('safehive.utils.agent_memory.ConversationSummaryMemory')
    def test_summary_memory_setup(self, mock_summary_memory):
        """Test summary memory setup."""
        mock_llm = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                memory_type="summary",
                memory_config={"llm": mock_llm},
                persist_directory=temp_dir
            )
            
            assert manager.memory is not None
            mock_summary_memory.assert_called_once()
    
    def test_summary_memory_setup_without_llm(self):
        """Test summary memory setup without LLM (should fallback to buffer)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                memory_type="summary",
                persist_directory=temp_dir
            )
            
            # Should fallback to buffer memory
            assert manager.memory_type == "buffer"
            assert manager.memory is not None
    
    def test_unsupported_memory_type(self):
        """Test unsupported memory type (should fallback to buffer)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                memory_type="unsupported_type",
                persist_directory=temp_dir
            )
            
            # Should fallback to buffer memory
            assert manager.memory_type == "buffer"
            assert manager.memory is not None
    
    def test_add_message(self):
        """Test adding a message to memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            message = AgentMessage(
                content="Hello, how are you?",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="test_agent"
            )
            
            manager.add_message(message)
            
            assert manager.memory_stats["total_messages"] == 1
            assert manager.memory_stats["last_accessed"] is not None
    
    def test_add_conversation(self):
        """Test adding a conversation to memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            # Create a conversation
            conversation = Conversation(
                conversation_id="conv_123",
                participants=["user", "test_agent"]
            )
            
            # Add messages to conversation
            message1 = AgentMessage(
                content="Hello",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="test_agent"
            )
            message2 = AgentMessage(
                content="Hi there!",
                message_type=MessageType.RESPONSE,
                sender="test_agent",
                recipient="user"
            )
            
            conversation.add_message(message1)
            conversation.add_message(message2)
            
            manager.add_conversation(conversation)
            
            assert manager.memory_stats["total_messages"] == 2
    
    def test_get_memory_variables(self):
        """Test getting memory variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            variables = manager.get_memory_variables()
            
            assert isinstance(variables, dict)
    
    def test_get_conversation_history(self):
        """Test getting conversation history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            # Add some messages
            message1 = AgentMessage(
                content="Hello",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="test_agent"
            )
            message2 = AgentMessage(
                content="Hi there!",
                message_type=MessageType.RESPONSE,
                sender="test_agent",
                recipient="user"
            )
            
            manager.add_message(message1)
            manager.add_message(message2)
            
            history = manager.get_conversation_history()
            
            assert len(history) == 2
            assert history[0].content == "Hello"
            assert history[1].content == "Hi there!"
    
    def test_get_conversation_history_with_limit(self):
        """Test getting conversation history with limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            # Add multiple messages
            for i in range(5):
                message = AgentMessage(
                    content=f"Message {i}",
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="test_agent"
                )
                manager.add_message(message)
            
            history = manager.get_conversation_history(limit=3)
            
            assert len(history) == 3
            assert history[-1].content == "Message 4"
    
    def test_clear_memory(self):
        """Test clearing memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            # Add some messages
            message = AgentMessage(
                content="Hello",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="test_agent"
            )
            manager.add_message(message)
            
            assert manager.memory_stats["total_messages"] == 1
            
            manager.clear_memory()
            
            assert manager.memory_stats["total_messages"] == 0
    
    def test_save_memory(self):
        """Test saving memory to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            result = manager.save_memory()
            
            assert result is True
            
            # Check if config file was created
            config_path = Path(temp_dir) / "memory_config.json"
            assert config_path.exists()
            
            # Check config content
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            assert config["agent_id"] == "test_agent"
            assert config["memory_type"] == "buffer"
    
    def test_load_memory(self):
        """Test loading memory from disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save memory
            manager1 = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            # Add some data
            message = AgentMessage(
                content="Hello",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="test_agent"
            )
            manager1.add_message(message)
            manager1.save_memory()
            
            # Create new manager and load
            manager2 = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            result = manager2.load_memory()
            
            assert result is True
            assert manager2.memory_stats["total_messages"] == 1
    
    def test_load_memory_no_file(self):
        """Test loading memory when no file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            result = manager.load_memory()
            
            assert result is False
    
    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            stats = manager.get_memory_stats()
            
            assert "total_messages" in stats
            assert "last_accessed" in stats
            assert "memory_type" in stats
            assert "persist_directory" in stats
            assert "memory_available" in stats
            assert stats["memory_type"] == "buffer"
            assert stats["memory_available"] is True


class TestMemoryManagerFactory:
    """Test the MemoryManagerFactory class."""
    
    def test_create_memory_manager(self):
        """Test creating a memory manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManagerFactory.create_memory_manager(
                agent_id="test_agent",
                memory_type="buffer",
                persist_directory=temp_dir
            )
            
            assert isinstance(manager, SafeHiveMemoryManager)
            assert manager.agent_id == "test_agent"
            assert manager.memory_type == "buffer"
    
    def test_create_buffer_memory(self):
        """Test creating a buffer memory manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManagerFactory.create_buffer_memory(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            assert manager.memory_type == "buffer"
    
    def test_create_window_memory(self):
        """Test creating a window memory manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManagerFactory.create_window_memory(
                agent_id="test_agent",
                window_size=5,
                persist_directory=temp_dir
            )
            
            assert manager.memory_type == "window"
            assert manager.memory_config["window_size"] == 5
    
    @patch('safehive.utils.agent_memory.ConversationSummaryMemory')
    def test_create_summary_memory(self, mock_summary_memory):
        """Test creating a summary memory manager."""
        mock_llm = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManagerFactory.create_summary_memory(
                agent_id="test_agent",
                llm=mock_llm,
                persist_directory=temp_dir
            )
            
            assert manager.memory_type == "summary"
            assert manager.memory_config["llm"] == mock_llm
    
    def test_create_vector_memory(self):
        """Test creating a vector memory manager (will fallback to buffer due to missing dependencies)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManagerFactory.create_vector_memory(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            # Due to complex dependencies, vector memory will fallback to buffer
            # but the factory method should still work
            assert manager.agent_id == "test_agent"
            assert manager.memory is not None
            # Check that the config was set correctly even if fallback occurred
            assert "embedding_model" in manager.memory_config
            assert "chunk_size" in manager.memory_config


class TestGlobalMemoryManagerFunctions:
    """Test global memory manager functions."""
    
    def test_get_memory_manager(self):
        """Test getting a memory manager."""
        # Initially no managers
        manager = get_memory_manager("test_agent")
        assert manager is None
        
        # Create a manager
        created_manager = create_memory_manager("test_agent")
        
        # Now should be able to get it
        manager = get_memory_manager("test_agent")
        assert manager is created_manager
    
    def test_create_memory_manager(self):
        """Test creating a memory manager."""
        manager = create_memory_manager(
            agent_id="test_agent",
            memory_type="buffer"
        )
        
        assert isinstance(manager, SafeHiveMemoryManager)
        assert manager.agent_id == "test_agent"
        
        # Should be registered
        retrieved_manager = get_memory_manager("test_agent")
        assert retrieved_manager is manager
    
    def test_remove_memory_manager(self):
        """Test removing a memory manager."""
        # Create a manager
        manager = create_memory_manager("test_agent")
        assert get_memory_manager("test_agent") is not None
        
        # Remove it
        result = remove_memory_manager("test_agent")
        assert result is True
        
        # Should be gone
        assert get_memory_manager("test_agent") is None
        
        # Try to remove non-existent
        result = remove_memory_manager("non_existent")
        assert result is False
    
    def test_list_memory_managers(self):
        """Test listing memory managers."""
        # Initially empty
        managers = list_memory_managers()
        assert len(managers) == 0
        
        # Create some managers
        create_memory_manager("agent1")
        create_memory_manager("agent2")
        
        managers = list_memory_managers()
        assert len(managers) == 2
        assert "agent1" in managers
        assert "agent2" in managers
    
    def test_save_all_memories(self):
        """Test saving all memories."""
        # Create managers
        manager1 = create_memory_manager("agent1")
        manager2 = create_memory_manager("agent2")
        
        # Mock save_memory method
        manager1.save_memory = Mock(return_value=True)
        manager2.save_memory = Mock(return_value=False)
        
        results = save_all_memories()
        
        assert results["agent1"] is True
        assert results["agent2"] is False
        manager1.save_memory.assert_called_once()
        manager2.save_memory.assert_called_once()
    
    def test_load_all_memories(self):
        """Test loading all memories."""
        # Create managers
        manager1 = create_memory_manager("agent1")
        manager2 = create_memory_manager("agent2")
        
        # Mock load_memory method
        manager1.load_memory = Mock(return_value=True)
        manager2.load_memory = Mock(return_value=False)
        
        results = load_all_memories()
        
        assert results["agent1"] is True
        assert results["agent2"] is False
        manager1.load_memory.assert_called_once()
        manager2.load_memory.assert_called_once()


class TestMemoryManagerIntegration:
    """Test memory manager integration scenarios."""
    
    def test_agent_conversation_workflow(self):
        """Test a complete agent conversation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeHiveMemoryManager(
                agent_id="assistant_agent",
                persist_directory=temp_dir
            )
            
            # Simulate a conversation
            conversation = Conversation(
                conversation_id="conv_123",
                participants=["user", "assistant_agent"]
            )
            
            # Add conversation messages
            messages = [
                AgentMessage(
                    content="Hello, can you help me?",
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="assistant_agent"
                ),
                AgentMessage(
                    content="Of course! How can I assist you?",
                    message_type=MessageType.RESPONSE,
                    sender="assistant_agent",
                    recipient="user"
                ),
                AgentMessage(
                    content="What's the weather like?",
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="assistant_agent"
                ),
                AgentMessage(
                    content="I don't have access to real-time weather data.",
                    message_type=MessageType.RESPONSE,
                    sender="assistant_agent",
                    recipient="user"
                )
            ]
            
            for message in messages:
                conversation.add_message(message)
                manager.add_message(message)
            
            # Verify memory state
            assert manager.memory_stats["total_messages"] == 4
            
            # Get conversation history
            history = manager.get_conversation_history()
            assert len(history) == 4
            
            # Get memory variables for LangChain
            variables = manager.get_memory_variables()
            assert isinstance(variables, dict)
            
            # Save and load memory
            assert manager.save_memory() is True
            assert manager.load_memory() is True
    
    def test_memory_persistence_workflow(self):
        """Test memory persistence across sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First session
            manager1 = SafeHiveMemoryManager(
                agent_id="persistent_agent",
                persist_directory=temp_dir
            )
            
            # Add some data
            message = AgentMessage(
                content="Remember this information",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="persistent_agent"
            )
            manager1.add_message(message)
            manager1.save_memory()
            
            # Second session
            manager2 = SafeHiveMemoryManager(
                agent_id="persistent_agent",
                persist_directory=temp_dir
            )
            
            # Load previous memory
            assert manager2.load_memory() is True
            assert manager2.memory_stats["total_messages"] == 1
            
            # Add more data
            message2 = AgentMessage(
                content="Additional information",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="persistent_agent"
            )
            manager2.add_message(message2)
            
            assert manager2.memory_stats["total_messages"] == 2
    
    def test_multiple_agent_memory_isolation(self):
        """Test that multiple agents have isolated memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create managers for different agents
            manager1 = SafeHiveMemoryManager(
                agent_id="agent1",
                persist_directory=temp_dir
            )
            manager2 = SafeHiveMemoryManager(
                agent_id="agent2",
                persist_directory=temp_dir
            )
            
            # Add messages to each agent
            message1 = AgentMessage(
                content="Agent 1 message",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="agent1"
            )
            message2 = AgentMessage(
                content="Agent 2 message",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="agent2"
            )
            
            manager1.add_message(message1)
            manager2.add_message(message2)
            
            # Verify isolation
            assert manager1.memory_stats["total_messages"] == 1
            assert manager2.memory_stats["total_messages"] == 1
            
            history1 = manager1.get_conversation_history()
            history2 = manager2.get_conversation_history()
            
            assert len(history1) == 1
            assert len(history2) == 1
            assert history1[0].content == "Agent 1 message"
            assert history2[0].content == "Agent 2 message"


if __name__ == "__main__":
    pytest.main([__file__])
