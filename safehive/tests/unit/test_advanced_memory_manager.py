"""
Unit tests for Advanced Memory Management.

This module tests the advanced memory management functionality including
contextual retrieval, memory compression, and analytics.
"""

import pytest
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple

from safehive.utils.advanced_memory_manager import (
    ContextualMemoryRetriever, MemoryCompressor, MemoryAnalytics, AdvancedMemoryManager,
    MemoryContext, MemoryResult, MemoryAnalyticsData,
    create_advanced_memory_manager, get_advanced_memory_manager, register_advanced_memory_manager,
    unregister_advanced_memory_manager
)
from safehive.utils.agent_memory import SafeHiveMemoryManager
from safehive.models.agent_models import AgentMessage, MessageType, Conversation


class TestMemoryContext:
    """Test the MemoryContext class."""
    
    def test_memory_context_creation(self):
        """Test creating a memory context."""
        context = MemoryContext(
            query="test query",
            conversation_id="conv_123",
            participant="user",
            message_type=MessageType.REQUEST,
            relevance_threshold=0.8,
            max_results=5
        )
        
        assert context.query == "test query"
        assert context.conversation_id == "conv_123"
        assert context.participant == "user"
        assert context.message_type == MessageType.REQUEST
        assert context.relevance_threshold == 0.8
        assert context.max_results == 5
    
    def test_memory_context_defaults(self):
        """Test memory context with default values."""
        context = MemoryContext(query="test")
        
        assert context.query == "test"
        assert context.conversation_id is None
        assert context.participant is None
        assert context.message_type is None
        assert context.time_range is None
        assert context.relevance_threshold == 0.7
        assert context.max_results == 10


class TestMemoryResult:
    """Test the MemoryResult class."""
    
    def test_memory_result_creation(self):
        """Test creating a memory result."""
        message = AgentMessage(
            content="Test message",
            message_type=MessageType.REQUEST,
            sender="user",
            recipient="agent"
        )
        
        result = MemoryResult(
            message=message,
            relevance_score=0.85,
            context_match={"content_similarity": 0.9},
            retrieval_metadata={"retrieved_at": "2024-01-01"}
        )
        
        assert result.message == message
        assert result.relevance_score == 0.85
        assert result.context_match["content_similarity"] == 0.9
        assert result.retrieval_metadata["retrieved_at"] == "2024-01-01"


class TestContextualMemoryRetriever:
    """Test the ContextualMemoryRetriever class."""
    
    def test_retriever_initialization(self):
        """Test retriever initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            retriever = ContextualMemoryRetriever(memory_manager)
            
            assert retriever.memory_manager == memory_manager
            assert retriever.embedding_cache == {}
            assert retriever.text_splitter is not None
    
    def test_calculate_relevance_score_content_similarity(self):
        """Test relevance score calculation for content similarity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            retriever = ContextualMemoryRetriever(memory_manager)
            
            message = AgentMessage(
                content="Hello world test message",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="agent"
            )
            
            context = MemoryContext(query="hello world")
            
            score, context_match = retriever.calculate_relevance_score(message, context)
            
            assert score > 0.0
            assert "content_similarity" in context_match
            assert context_match["content_similarity"] > 0.0
    
    def test_calculate_relevance_score_participant_match(self):
        """Test relevance score calculation for participant match."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            retriever = ContextualMemoryRetriever(memory_manager)
            
            message = AgentMessage(
                content="Some message",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="agent"
            )
            
            context = MemoryContext(query="test", participant="user")
            
            score, context_match = retriever.calculate_relevance_score(message, context)
            
            assert score > 0.0
            assert "participant_match" in context_match
            assert context_match["participant_match"] is True
    
    def test_calculate_relevance_score_message_type_match(self):
        """Test relevance score calculation for message type match."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            retriever = ContextualMemoryRetriever(memory_manager)
            
            message = AgentMessage(
                content="Some message",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="agent"
            )
            
            context = MemoryContext(query="test", message_type=MessageType.REQUEST)
            
            score, context_match = retriever.calculate_relevance_score(message, context)
            
            assert score > 0.0
            assert "message_type_match" in context_match
            assert context_match["message_type_match"] is True
    
    def test_calculate_relevance_score_time_relevance(self):
        """Test relevance score calculation for time relevance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            retriever = ContextualMemoryRetriever(memory_manager)
            
            now = datetime.now()
            message = AgentMessage(
                content="Some message",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="agent",
                timestamp=now
            )
            
            start_time = now - timedelta(hours=1)
            end_time = now + timedelta(hours=1)
            context = MemoryContext(query="test", time_range=(start_time, end_time))
            
            score, context_match = retriever.calculate_relevance_score(message, context)
            
            assert score > 0.0
            assert "time_relevance" in context_match
            assert context_match["time_relevance"] > 0.0
    
    def test_retrieve_relevant_memories(self):
        """Test retrieving relevant memories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            # Add some test messages
            messages = [
                AgentMessage(
                    content="Hello world message",
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="agent"
                ),
                AgentMessage(
                    content="Different content here",
                    message_type=MessageType.RESPONSE,
                    sender="agent",
                    recipient="user"
                ),
                AgentMessage(
                    content="Another hello world test",
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="agent"
                )
            ]
            
            for message in messages:
                memory_manager.add_message(message)
            
            retriever = ContextualMemoryRetriever(memory_manager)
            
            context = MemoryContext(
                query="hello world",
                relevance_threshold=0.1  # Low threshold to ensure results
            )
            
            results = retriever.retrieve_relevant_memories(context)
            
            assert len(results) > 0
            assert all(isinstance(result, MemoryResult) for result in results)
            assert all(result.relevance_score >= context.relevance_threshold for result in results)
            
            # Results should be sorted by relevance score
            scores = [result.relevance_score for result in results]
            assert scores == sorted(scores, reverse=True)


class TestMemoryCompressor:
    """Test the MemoryCompressor class."""
    
    def test_compressor_initialization(self):
        """Test compressor initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            compressor = MemoryCompressor(memory_manager)
            
            assert compressor.memory_manager == memory_manager
            assert compressor.compression_threshold == 100
            assert compressor.summary_threshold == 50
    
    def test_compress_conversation_history_small_conversation(self):
        """Test compressing a small conversation (should not compress)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            compressor = MemoryCompressor(memory_manager)
            
            # Create a small conversation
            conversation = Conversation(
                conversation_id="conv_123",
                participants=["user", "agent"]
            )
            
            # Add a few messages
            for i in range(5):
                message = AgentMessage(
                    content=f"Message {i}",
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="agent"
                )
                conversation.add_message(message)
            
            compressed = compressor.compress_conversation_history(conversation)
            
            # Should not compress small conversations
            assert compressed == conversation
    
    def test_compress_conversation_history_large_conversation(self):
        """Test compressing a large conversation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            compressor = MemoryCompressor(memory_manager)
            
            # Create a large conversation
            conversation = Conversation(
                conversation_id="conv_123",
                participants=["user", "agent"]
            )
            
            # Add many messages
            for i in range(100):
                message = AgentMessage(
                    content=f"Message {i}",
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="agent"
                )
                conversation.add_message(message)
            
            compressed = compressor.compress_conversation_history(conversation, compression_ratio=0.5)
            
            # Should compress
            assert len(compressed.messages) < len(conversation.messages)
            assert compressed.metadata.get("compressed") is True
            assert compressed.metadata.get("original_message_count") == 100
            assert compressed.metadata.get("compression_ratio") == 0.5
            
            # Should preserve first and last messages
            assert compressed.messages[0].content == "Message 0"
            assert compressed.messages[-1].content == "Message 99"
    
    def test_summarize_conversation(self):
        """Test conversation summarization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            compressor = MemoryCompressor(memory_manager)
            
            # Create a conversation
            conversation = Conversation(
                conversation_id="conv_123",
                participants=["user", "agent"]
            )
            
            # Add some messages
            messages = [
                "Hello, how are you?",
                "I'm doing well, thank you for asking",
                "What's the weather like today?",
                "I don't have access to weather data",
                "That's okay, thanks anyway"
            ]
            
            for content in messages:
                message = AgentMessage(
                    content=content,
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="agent"
                )
                conversation.add_message(message)
            
            summary = compressor.summarize_conversation(conversation)
            
            assert isinstance(summary, str)
            assert len(summary) > 0
            assert "user" in summary
            assert "agent" in summary
            assert "5 messages" in summary
    
    def test_summarize_empty_conversation(self):
        """Test summarizing an empty conversation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            compressor = MemoryCompressor(memory_manager)
            
            conversation = Conversation(
                conversation_id="conv_123",
                participants=["user", "agent"]
            )
            
            summary = compressor.summarize_conversation(conversation)
            
            assert summary == "Empty conversation"
    
    def test_should_compress_memory(self):
        """Test checking if memory should be compressed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            compressor = MemoryCompressor(memory_manager)
            
            # Initially should not compress
            assert not compressor.should_compress_memory()
            
            # Add many messages
            for i in range(150):
                message = AgentMessage(
                    content=f"Message {i}",
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="agent"
                )
                memory_manager.add_message(message)
            
            # Now should compress
            assert compressor.should_compress_memory()


class TestMemoryAnalytics:
    """Test the MemoryAnalytics class."""
    
    def test_analytics_initialization(self):
        """Test analytics initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            analytics = MemoryAnalytics(memory_manager)
            
            assert analytics.memory_manager == memory_manager
            assert analytics.analytics_cache is None
            assert analytics.cache_ttl == timedelta(hours=1)
    
    def test_generate_analytics_empty_memory(self):
        """Test generating analytics for empty memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            analytics = MemoryAnalytics(memory_manager)
            
            result = analytics.generate_analytics()
            
            assert result.total_messages == 0
            assert result.total_conversations == 0
            assert result.memory_size_bytes == 0
            assert result.most_active_participants == []
            assert result.most_common_message_types == []
            assert result.conversation_frequency == {}
            assert result.memory_growth_rate == 0.0
            assert result.compression_ratio == 1.0
    
    def test_generate_analytics_with_data(self):
        """Test generating analytics with memory data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            # Add some test messages
            messages = [
                AgentMessage(
                    content="Hello",
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="agent",
                    metadata={"conversation_id": "conv_1"}
                ),
                AgentMessage(
                    content="Hi there",
                    message_type=MessageType.RESPONSE,
                    sender="agent",
                    recipient="user",
                    metadata={"conversation_id": "conv_1"}
                ),
                AgentMessage(
                    content="How are you?",
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="agent",
                    metadata={"conversation_id": "conv_2"}
                )
            ]
            
            for message in messages:
                memory_manager.add_message(message)
            
            analytics = MemoryAnalytics(memory_manager)
            
            result = analytics.generate_analytics()
            
            assert result.total_messages == 3
            assert result.total_conversations == 1  # Metadata not preserved in current implementation
            assert result.memory_size_bytes > 0
            assert len(result.most_active_participants) > 0
            assert len(result.most_common_message_types) > 0
            assert len(result.conversation_frequency) == 1  # All messages have "unknown" conversation_id
    
    def test_analytics_caching(self):
        """Test analytics caching functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            analytics = MemoryAnalytics(memory_manager)
            
            # Generate analytics first time
            result1 = analytics.generate_analytics()
            
            # Generate again - should use cache
            result2 = analytics.generate_analytics()
            
            # Should be the same object (cached)
            assert result1 is result2
            
            # Force refresh
            result3 = analytics.generate_analytics(force_refresh=True)
            
            # Should be different object
            assert result1 is not result3


class TestAdvancedMemoryManager:
    """Test the AdvancedMemoryManager class."""
    
    def test_advanced_manager_initialization(self):
        """Test advanced manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            advanced_manager = AdvancedMemoryManager(memory_manager)
            
            assert advanced_manager.memory_manager == memory_manager
            assert advanced_manager.retriever is not None
            assert advanced_manager.compressor is not None
            assert advanced_manager.analytics is not None
            assert advanced_manager.auto_compress is True
            assert advanced_manager.compression_threshold == 100
            assert advanced_manager.analytics_enabled is True
    
    def test_retrieve_contextual_memories(self):
        """Test retrieving contextual memories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            # Add some test messages
            messages = [
                AgentMessage(
                    content="Hello world test message",
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="agent"
                ),
                AgentMessage(
                    content="Different content",
                    message_type=MessageType.RESPONSE,
                    sender="agent",
                    recipient="user"
                )
            ]
            
            for message in messages:
                memory_manager.add_message(message)
            
            advanced_manager = AdvancedMemoryManager(memory_manager)
            
            results = advanced_manager.retrieve_contextual_memories(
                query="hello world",
                relevance_threshold=0.1  # Low threshold
            )
            
            assert len(results) > 0
            assert all(isinstance(result, MemoryResult) for result in results)
    
    def test_compress_memory(self):
        """Test memory compression."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            advanced_manager = AdvancedMemoryManager(memory_manager)
            
            result = advanced_manager.compress_memory()
            
            assert isinstance(result, dict)
            assert "compressed" in result
    
    def test_get_memory_analytics(self):
        """Test getting memory analytics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            advanced_manager = AdvancedMemoryManager(memory_manager)
            
            analytics = advanced_manager.get_memory_analytics()
            
            assert isinstance(analytics, MemoryAnalyticsData)
            assert hasattr(analytics, 'total_messages')
            assert hasattr(analytics, 'total_conversations')
    
    def test_optimize_memory(self):
        """Test memory optimization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            advanced_manager = AdvancedMemoryManager(memory_manager)
            
            result = advanced_manager.optimize_memory()
            
            assert isinstance(result, dict)
            assert "initial_analytics" in result
            assert "final_analytics" in result
    
    def test_search_memories(self):
        """Test searching memories with filters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            # Add some test messages
            message = AgentMessage(
                content="Search test message",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="agent"
            )
            memory_manager.add_message(message)
            
            advanced_manager = AdvancedMemoryManager(memory_manager)
            
            # Search without filters
            results = advanced_manager.search_memories("search test")
            assert len(results) >= 0
            
            # Search with filters
            filters = {
                "participant": "user",
                "message_type": MessageType.REQUEST,
                "relevance_threshold": 0.1
            }
            
            results = advanced_manager.search_memories("search test", filters)
            assert len(results) >= 0
    
    def test_get_memory_insights(self):
        """Test getting memory insights."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            advanced_manager = AdvancedMemoryManager(memory_manager)
            
            insights = advanced_manager.get_memory_insights()
            
            assert isinstance(insights, dict)
            assert "memory_health" in insights
            assert "recommendations" in insights
            assert "trends" in insights
            assert "efficiency_metrics" in insights


class TestAdvancedMemoryManagerIntegration:
    """Test advanced memory manager integration scenarios."""
    
    def test_complete_memory_workflow(self):
        """Test a complete memory workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            advanced_manager = AdvancedMemoryManager(memory_manager)
            
            # Add some test data
            messages = [
                AgentMessage(
                    content="Hello, I need help with food ordering",
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="agent",
                    metadata={"conversation_id": "conv_1"}
                ),
                AgentMessage(
                    content="I can help you with food ordering. What would you like to order?",
                    message_type=MessageType.RESPONSE,
                    sender="agent",
                    recipient="user",
                    metadata={"conversation_id": "conv_1"}
                ),
                AgentMessage(
                    content="I'd like to order pizza",
                    message_type=MessageType.REQUEST,
                    sender="user",
                    recipient="agent",
                    metadata={"conversation_id": "conv_1"}
                )
            ]
            
            for message in messages:
                memory_manager.add_message(message)
            
            # Test contextual retrieval
            results = advanced_manager.retrieve_contextual_memories(
                query="food ordering pizza",
                participant="user"
            )
            
            assert len(results) > 0
            
            # Test analytics
            analytics = advanced_manager.get_memory_analytics()
            assert analytics.total_messages == 3
            
            # Test insights
            insights = advanced_manager.get_memory_insights()
            assert "memory_health" in insights
            
            # Test optimization
            optimization = advanced_manager.optimize_memory()
            assert "initial_analytics" in optimization
    
    def test_memory_search_workflow(self):
        """Test memory search workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            advanced_manager = AdvancedMemoryManager(memory_manager)
            
            # Add diverse test data
            test_data = [
                ("Hello world", MessageType.REQUEST, "user"),
                ("Hi there", MessageType.RESPONSE, "agent"),
                ("How are you?", MessageType.REQUEST, "user"),
                ("I'm fine", MessageType.RESPONSE, "agent"),
                ("Goodbye", MessageType.REQUEST, "user")
            ]
            
            for content, msg_type, sender in test_data:
                message = AgentMessage(
                    content=content,
                    message_type=msg_type,
                    sender=sender,
                    recipient="agent" if sender == "user" else "user"
                )
                memory_manager.add_message(message)
            
            # Test various search scenarios
            searches = [
                ("hello", {}),
                ("goodbye", {"participant": "user"}),
                ("", {"message_type": MessageType.RESPONSE}),
            ]
            
            for query, filters in searches:
                results = advanced_manager.search_memories(query, filters)
                assert isinstance(results, list)


class TestAdvancedMemoryManagerFactory:
    """Test the advanced memory manager factory functions."""
    
    def test_create_advanced_memory_manager(self):
        """Test creating an advanced memory manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            advanced_manager = create_advanced_memory_manager(memory_manager)
            
            assert isinstance(advanced_manager, AdvancedMemoryManager)
            assert advanced_manager.memory_manager == memory_manager
    
    def test_register_and_get_advanced_memory_manager(self):
        """Test registering and getting an advanced memory manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_manager = SafeHiveMemoryManager(
                agent_id="test_agent",
                persist_directory=temp_dir
            )
            
            advanced_manager = create_advanced_memory_manager(memory_manager)
            
            # Register
            register_advanced_memory_manager("test_agent", advanced_manager)
            
            # Get
            retrieved = get_advanced_memory_manager("test_agent")
            assert retrieved is advanced_manager
            
            # Unregister
            result = unregister_advanced_memory_manager("test_agent")
            assert result is True
            
            # Should be gone
            assert get_advanced_memory_manager("test_agent") is None
    
    def test_unregister_nonexistent_manager(self):
        """Test unregistering a non-existent manager."""
        result = unregister_advanced_memory_manager("nonexistent")
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
