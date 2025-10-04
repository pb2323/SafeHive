"""
Unit tests for Conversation Management and Context Awareness.
"""

import asyncio
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from safehive.agents.conversation_management import (
    ConversationState, ConversationType, TurnType, ContextType,
    ConversationTurn, ConversationContext, ConversationSession, ConversationFlow, ConversationManager
)


class TestConversationState:
    """Test ConversationState enum."""
    
    def test_conversation_state_values(self):
        """Test ConversationState enum values."""
        assert ConversationState.ACTIVE.value == "active"
        assert ConversationState.PAUSED.value == "paused"
        assert ConversationState.COMPLETED.value == "completed"
        assert ConversationState.ABANDONED.value == "abandoned"
        assert ConversationState.ERROR.value == "error"


class TestConversationType:
    """Test ConversationType enum."""
    
    def test_conversation_type_values(self):
        """Test ConversationType enum values."""
        assert ConversationType.ORDER_PLACEMENT.value == "order_placement"
        assert ConversationType.ORDER_INQUIRY.value == "order_inquiry"
        assert ConversationType.VENDOR_SEARCH.value == "vendor_search"
        assert ConversationType.PREFERENCE_UPDATE.value == "preference_update"
        assert ConversationType.SUPPORT.value == "support"
        assert ConversationType.GENERAL.value == "general"


class TestTurnType:
    """Test TurnType enum."""
    
    def test_turn_type_values(self):
        """Test TurnType enum values."""
        assert TurnType.USER_INPUT.value == "user_input"
        assert TurnType.AGENT_RESPONSE.value == "agent_response"
        assert TurnType.SYSTEM_MESSAGE.value == "system_message"
        assert TurnType.ERROR_MESSAGE.value == "error_message"
        assert TurnType.CONFIRMATION.value == "confirmation"
        assert TurnType.CLARIFICATION.value == "clarification"


class TestContextType:
    """Test ContextType enum."""
    
    def test_context_type_values(self):
        """Test ContextType enum values."""
        assert ContextType.ORDER_CONTEXT.value == "order_context"
        assert ContextType.USER_PREFERENCE.value == "user_preference"
        assert ContextType.VENDOR_CONTEXT.value == "vendor_context"
        assert ContextType.SYSTEM_STATE.value == "system_state"
        assert ContextType.CONVERSATION_HISTORY.value == "conversation_history"
        assert ContextType.EXTERNAL_DATA.value == "external_data"


class TestConversationTurn:
    """Test ConversationTurn functionality."""
    
    def test_conversation_turn_creation(self):
        """Test ConversationTurn creation."""
        turn = ConversationTurn(
            turn_id="turn_001",
            conversation_id="conv_001",
            turn_type=TurnType.USER_INPUT,
            content="I want to order pizza",
            intent="order_request",
            entities={"food_type": "pizza"},
            confidence=0.9
        )
        
        assert turn.turn_id == "turn_001"
        assert turn.conversation_id == "conv_001"
        assert turn.turn_type == TurnType.USER_INPUT
        assert turn.content == "I want to order pizza"
        assert turn.intent == "order_request"
        assert turn.entities["food_type"] == "pizza"
        assert turn.confidence == 0.9
        assert isinstance(turn.timestamp, datetime)
    
    def test_conversation_turn_serialization(self):
        """Test ConversationTurn serialization."""
        turn = ConversationTurn(
            turn_id="turn_001",
            conversation_id="conv_001",
            turn_type=TurnType.USER_INPUT,
            content="I want to order pizza",
            intent="order_request",
            entities={"food_type": "pizza"},
            confidence=0.9
        )
        
        data = turn.to_dict()
        
        assert data["turn_id"] == "turn_001"
        assert data["conversation_id"] == "conv_001"
        assert data["turn_type"] == "user_input"
        assert data["content"] == "I want to order pizza"
        assert data["intent"] == "order_request"
        assert data["entities"]["food_type"] == "pizza"
        assert data["confidence"] == 0.9
        assert "timestamp" in data


class TestConversationContext:
    """Test ConversationContext functionality."""
    
    def test_conversation_context_creation(self):
        """Test ConversationContext creation."""
        context = ConversationContext(
            context_type=ContextType.ORDER_CONTEXT,
            key="order_items",
            value=["pizza", "burger"],
            confidence=0.8,
            source="user_input",
            metadata={"count": 2}
        )
        
        assert context.context_type == ContextType.ORDER_CONTEXT
        assert context.key == "order_items"
        assert context.value == ["pizza", "burger"]
        assert context.confidence == 0.8
        assert context.source == "user_input"
        assert context.metadata["count"] == 2
        assert isinstance(context.timestamp, datetime)
    
    def test_conversation_context_serialization(self):
        """Test ConversationContext serialization."""
        context = ConversationContext(
            context_type=ContextType.ORDER_CONTEXT,
            key="order_items",
            value=["pizza", "burger"],
            confidence=0.8,
            source="user_input"
        )
        
        data = context.to_dict()
        
        assert data["context_type"] == "order_context"
        assert data["key"] == "order_items"
        assert data["value"] == ["pizza", "burger"]
        assert data["confidence"] == 0.8
        assert data["source"] == "user_input"
        assert "timestamp" in data


class TestConversationSession:
    """Test ConversationSession functionality."""
    
    def test_conversation_session_creation(self):
        """Test ConversationSession creation."""
        session = ConversationSession(
            session_id="session_001",
            user_id="user_001",
            conversation_type=ConversationType.ORDER_PLACEMENT
        )
        
        assert session.session_id == "session_001"
        assert session.user_id == "user_001"
        assert session.conversation_type == ConversationType.ORDER_PLACEMENT
        assert session.state == ConversationState.ACTIVE
        assert len(session.turns) == 0
        assert len(session.context) == 0
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)
        assert isinstance(session.last_activity, datetime)
    
    def test_add_turn(self):
        """Test adding turns to conversation session."""
        session = ConversationSession(
            session_id="session_001",
            user_id="user_001",
            conversation_type=ConversationType.ORDER_PLACEMENT
        )
        
        turn = ConversationTurn(
            turn_id="turn_001",
            conversation_id="session_001",
            turn_type=TurnType.USER_INPUT,
            content="I want to order pizza"
        )
        
        session.add_turn(turn)
        
        assert len(session.turns) == 1
        assert session.turns[0] == turn
        assert session.updated_at > session.created_at
        assert session.last_activity > session.created_at
    
    def test_update_context(self):
        """Test updating conversation context."""
        session = ConversationSession(
            session_id="session_001",
            user_id="user_001",
            conversation_type=ConversationType.ORDER_PLACEMENT
        )
        
        session.update_context("order_items", ["pizza", "burger"], "user_input")
        
        assert len(session.context) == 1
        assert "order_items" in session.context
        context_item = session.context["order_items"]
        assert context_item.key == "order_items"
        assert context_item.value == ["pizza", "burger"]
        assert context_item.source == "user_input"
    
    def test_get_context(self):
        """Test getting context value."""
        session = ConversationSession(
            session_id="session_001",
            user_id="user_001",
            conversation_type=ConversationType.ORDER_PLACEMENT
        )
        
        session.update_context("order_items", ["pizza", "burger"], "user_input")
        
        value = session.get_context("order_items")
        assert value == ["pizza", "burger"]
        
        # Test non-existent context
        non_existent = session.get_context("non_existent")
        assert non_existent is None
    
    def test_get_contexts_by_type(self):
        """Test getting contexts by type."""
        session = ConversationSession(
            session_id="session_001",
            user_id="user_001",
            conversation_type=ConversationType.ORDER_PLACEMENT
        )
        
        session.update_context("order_items", ["pizza"], "user_input", ContextType.ORDER_CONTEXT)
        session.update_context("user_pref", "vegetarian", "user_input", ContextType.USER_PREFERENCE)
        session.update_context("vendor", "pizza_place", "user_input", ContextType.VENDOR_CONTEXT)
        
        order_contexts = session.get_contexts_by_type(ContextType.ORDER_CONTEXT)
        assert len(order_contexts) == 1
        assert order_contexts["order_items"] == ["pizza"]
        
        user_contexts = session.get_contexts_by_type(ContextType.USER_PREFERENCE)
        assert len(user_contexts) == 1
        assert user_contexts["user_pref"] == "vegetarian"
    
    def test_get_recent_turns(self):
        """Test getting recent turns."""
        session = ConversationSession(
            session_id="session_001",
            user_id="user_001",
            conversation_type=ConversationType.ORDER_PLACEMENT
        )
        
        # Add multiple turns
        for i in range(10):
            turn = ConversationTurn(
                turn_id=f"turn_{i:03d}",
                conversation_id="session_001",
                turn_type=TurnType.USER_INPUT,
                content=f"Turn {i}"
            )
            session.add_turn(turn)
        
        recent_turns = session.get_recent_turns(5)
        assert len(recent_turns) == 5
        assert recent_turns[-1].content == "Turn 9"  # Most recent
        assert recent_turns[0].content == "Turn 5"   # Oldest of recent
    
    def test_get_turns_by_type(self):
        """Test getting turns by type."""
        session = ConversationSession(
            session_id="session_001",
            user_id="user_001",
            conversation_type=ConversationType.ORDER_PLACEMENT
        )
        
        # Add turns of different types
        user_turn = ConversationTurn(
            turn_id="turn_001",
            conversation_id="session_001",
            turn_type=TurnType.USER_INPUT,
            content="I want pizza"
        )
        agent_turn = ConversationTurn(
            turn_id="turn_002",
            conversation_id="session_001",
            turn_type=TurnType.AGENT_RESPONSE,
            content="What type of pizza?"
        )
        
        session.add_turn(user_turn)
        session.add_turn(agent_turn)
        
        user_turns = session.get_turns_by_type(TurnType.USER_INPUT)
        assert len(user_turns) == 1
        assert user_turns[0].content == "I want pizza"
        
        agent_turns = session.get_turns_by_type(TurnType.AGENT_RESPONSE)
        assert len(agent_turns) == 1
        assert agent_turns[0].content == "What type of pizza?"
    
    def test_conversation_session_serialization(self):
        """Test ConversationSession serialization."""
        session = ConversationSession(
            session_id="session_001",
            user_id="user_001",
            conversation_type=ConversationType.ORDER_PLACEMENT
        )
        
        # Add some data
        session.update_context("order_items", ["pizza"], "user_input")
        turn = ConversationTurn(
            turn_id="turn_001",
            conversation_id="session_001",
            turn_type=TurnType.USER_INPUT,
            content="I want pizza"
        )
        session.add_turn(turn)
        
        data = session.to_dict()
        
        assert data["session_id"] == "session_001"
        assert data["user_id"] == "user_001"
        assert data["conversation_type"] == "order_placement"
        assert data["state"] == "active"
        assert len(data["turns"]) == 1
        assert len(data["context"]) == 1
        assert "created_at" in data
        assert "updated_at" in data


class TestConversationFlow:
    """Test ConversationFlow functionality."""
    
    def test_conversation_flow_creation(self):
        """Test ConversationFlow creation."""
        flow = ConversationFlow(
            flow_id="test_flow",
            name="Test Flow",
            description="A test conversation flow",
            conversation_type=ConversationType.ORDER_PLACEMENT,
            expected_turns=[TurnType.USER_INPUT, TurnType.AGENT_RESPONSE],
            required_context=["order_items"],
            optional_context=["delivery_address"],
            completion_criteria={"order_created": True}
        )
        
        assert flow.flow_id == "test_flow"
        assert flow.name == "Test Flow"
        assert flow.description == "A test conversation flow"
        assert flow.conversation_type == ConversationType.ORDER_PLACEMENT
        assert len(flow.expected_turns) == 2
        assert flow.required_context == ["order_items"]
        assert flow.optional_context == ["delivery_address"]
        assert flow.completion_criteria["order_created"] is True
    
    def test_conversation_flow_serialization(self):
        """Test ConversationFlow serialization."""
        flow = ConversationFlow(
            flow_id="test_flow",
            name="Test Flow",
            description="A test conversation flow",
            conversation_type=ConversationType.ORDER_PLACEMENT,
            expected_turns=[TurnType.USER_INPUT, TurnType.AGENT_RESPONSE],
            required_context=["order_items"],
            completion_criteria={"order_created": True}
        )
        
        data = flow.to_dict()
        
        assert data["flow_id"] == "test_flow"
        assert data["name"] == "Test Flow"
        assert data["conversation_type"] == "order_placement"
        assert data["expected_turns"] == ["user_input", "agent_response"]
        assert data["required_context"] == ["order_items"]
        assert data["completion_criteria"]["order_created"] is True


class TestConversationManager:
    """Test ConversationManager functionality."""
    
    def test_conversation_manager_creation(self):
        """Test ConversationManager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            assert manager.storage_path == Path(temp_dir)
            assert len(manager.active_sessions) == 0
            assert len(manager.conversation_history) == 0
            assert len(manager.conversation_flows) > 0  # Should have default flows
    
    def test_create_conversation_session(self):
        """Test creating conversation sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            session = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            
            assert session.user_id == "user_001"
            assert session.conversation_type == ConversationType.ORDER_PLACEMENT
            assert session.state == ConversationState.ACTIVE
            assert session.session_id in manager.active_sessions
    
    def test_create_conversation_session_with_initial_context(self):
        """Test creating conversation session with initial context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            initial_context = {"user_preferences": "vegetarian", "location": "downtown"}
            session = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT, initial_context
            )
            
            assert session.get_context("user_preferences") == "vegetarian"
            assert session.get_context("location") == "downtown"
    
    def test_add_turn_to_conversation(self):
        """Test adding turns to conversation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            session = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            
            turn = manager.add_turn_to_conversation(
                session.session_id, TurnType.USER_INPUT, "I want to order pizza",
                intent="order_request", entities={"food_type": "pizza"}
            )
            
            assert turn is not None
            assert turn.content == "I want to order pizza"
            assert turn.intent == "order_request"
            assert turn.entities["food_type"] == "pizza"
            assert len(session.turns) == 1
    
    def test_get_conversation_session(self):
        """Test getting conversation session."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            session = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            
            retrieved_session = manager.get_conversation_session(session.session_id)
            assert retrieved_session is not None
            assert retrieved_session.session_id == session.session_id
            
            # Test non-existent session
            non_existent = manager.get_conversation_session("non_existent")
            assert non_existent is None
    
    def test_get_user_conversations(self):
        """Test getting user conversations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            # Create multiple sessions for same user
            session1 = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            session2 = manager.create_conversation_session(
                "user_001", ConversationType.VENDOR_SEARCH
            )
            
            # Create session for different user
            session3 = manager.create_conversation_session(
                "user_002", ConversationType.ORDER_PLACEMENT
            )
            
            user_conversations = manager.get_user_conversations("user_001")
            assert len(user_conversations) == 2
            
            user_conversations = manager.get_user_conversations("user_002")
            assert len(user_conversations) == 1
            assert user_conversations[0].session_id == session3.session_id
    
    def test_update_conversation_state(self):
        """Test updating conversation state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            session = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            
            # Update state to completed
            success = manager.update_conversation_state(
                session.session_id, ConversationState.COMPLETED
            )
            
            assert success is True
            assert session.state == ConversationState.COMPLETED
            # Should be moved to history
            assert session.session_id not in manager.active_sessions
            assert len(manager.conversation_history) == 1
    
    def test_analyze_conversation_context(self):
        """Test analyzing conversation context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            session = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            
            # Add some context and turns
            session.update_context("order_items", ["pizza"], "user_input")
            manager.add_turn_to_conversation(
                session.session_id, TurnType.USER_INPUT, "I want pizza",
                intent="order_request", entities={"food_type": "pizza"}
            )
            
            analysis = manager.analyze_conversation_context(session.session_id)
            
            assert analysis["session_id"] == session.session_id
            assert analysis["conversation_type"] == "order_placement"
            assert analysis["turn_count"] == 1
            assert analysis["context_items"] == 1
            assert "context_summary" in analysis
            assert "intent_analysis" in analysis
            assert "entity_analysis" in analysis
    
    def test_get_conversation_flow(self):
        """Test getting conversation flow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            flow = manager.get_conversation_flow(ConversationType.ORDER_PLACEMENT)
            assert flow is not None
            assert flow.conversation_type == ConversationType.ORDER_PLACEMENT
            
            # Test non-existent flow
            non_existent = manager.get_conversation_flow(ConversationType.GENERAL)
            assert non_existent is None
    
    def test_suggest_next_turn(self):
        """Test suggesting next turn."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            session = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            
            # First turn should be user input
            next_turn = manager.suggest_next_turn(session.session_id)
            assert next_turn == TurnType.USER_INPUT
            
            # Add a turn
            manager.add_turn_to_conversation(
                session.session_id, TurnType.USER_INPUT, "I want pizza"
            )
            
            # Next turn should be agent response
            next_turn = manager.suggest_next_turn(session.session_id)
            assert next_turn == TurnType.AGENT_RESPONSE
    
    def test_check_conversation_completion(self):
        """Test checking conversation completion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            session = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            
            # Initially not complete
            completion = manager.check_conversation_completion(session.session_id)
            assert completion["completed"] is False
            assert completion["completion_percentage"] < 100
            
            # Add required context
            session.update_context("order_created", True, "system")
            session.update_context("user_confirmed", True, "system")
            
            # Now should be complete
            completion = manager.check_conversation_completion(session.session_id)
            assert completion["completed"] is True
            assert completion["completion_percentage"] == 100.0
    
    def test_generate_conversation_summary(self):
        """Test generating conversation summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            session = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            
            # Add some context
            session.update_context("order_items", "pizza, burger", "user_input")
            session.update_context("vendor_selection", "pizza_place", "user_input")
            
            summary = manager.generate_conversation_summary(session.session_id)
            
            assert summary is not None
            assert "Conversation type: order_placement" in summary
            assert "Order items: pizza, burger" in summary
            assert "Selected vendor: pizza_place" in summary
            assert session.summary == summary
    
    def test_get_conversation_statistics(self):
        """Test getting conversation statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            # Test with no sessions
            stats = manager.get_conversation_statistics()
            assert stats["total_sessions"] == 0
            assert stats["active_sessions"] == 0
            assert stats["completed_sessions"] == 0
            
            # Create some sessions
            session1 = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            session2 = manager.create_conversation_session(
                "user_002", ConversationType.VENDOR_SEARCH
            )
            
            # Complete one session
            manager.update_conversation_state(
                session1.session_id, ConversationState.COMPLETED
            )
            
            stats = manager.get_conversation_statistics()
            assert stats["total_sessions"] == 2
            assert stats["active_sessions"] == 1
            assert stats["completed_sessions"] == 1
            assert "conversation_types" in stats
            assert "state_counts" in stats
            assert stats["flows_available"] > 0
    
    def test_cleanup_inactive_sessions(self):
        """Test cleaning up inactive sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            session = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            
            # Manually set last activity to be old
            session.last_activity = datetime.now() - timedelta(minutes=70)
            
            # Cleanup sessions inactive for more than 60 minutes
            cleaned_count = manager.cleanup_inactive_sessions(60)
            
            assert cleaned_count == 1
            assert session.session_id not in manager.active_sessions
            assert len(manager.conversation_history) == 1
            assert session.state == ConversationState.ABANDONED


class TestConversationManagerIntegration:
    """Integration tests for ConversationManager."""
    
    def test_persistence_and_recovery(self):
        """Test persistence and recovery of conversation data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first manager instance
            manager1 = ConversationManager(temp_dir)
            
            # Create session and add data
            session = manager1.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            session.update_context("order_items", ["pizza"], "user_input")
            manager1.add_turn_to_conversation(
                session.session_id, TurnType.USER_INPUT, "I want pizza"
            )
            
            # Complete session
            manager1.update_conversation_state(
                session.session_id, ConversationState.COMPLETED
            )
            
            # Create second manager instance (should load data)
            manager2 = ConversationManager(temp_dir)
            
            # Verify data was loaded
            assert len(manager2.conversation_history) == 1
            loaded_session = manager2.conversation_history[0]
            assert loaded_session.user_id == "user_001"
            assert loaded_session.get_context("order_items") == ["pizza"]
            assert len(loaded_session.turns) == 1
            assert loaded_session.state == ConversationState.COMPLETED
    
    def test_complete_conversation_workflow(self):
        """Test complete conversation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            # Create order placement conversation
            session = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            
            # Simulate conversation flow
            # Turn 1: User input
            manager.add_turn_to_conversation(
                session.session_id, TurnType.USER_INPUT, "I want to order pizza",
                intent="order_request", entities={"food_type": "pizza"}
            )
            
            # Turn 2: Agent response
            manager.add_turn_to_conversation(
                session.session_id, TurnType.AGENT_RESPONSE, "What type of pizza would you like?",
                intent="clarification_request"
            )
            
            # Turn 3: User input with context update
            manager.add_turn_to_conversation(
                session.session_id, TurnType.USER_INPUT, "Margherita pizza",
                intent="item_selection", entities={"pizza_type": "margherita"},
                context_updates={"order_items": "margherita pizza"}
            )
            
            # Check conversation state
            analysis = manager.analyze_conversation_context(session.session_id)
            assert analysis["turn_count"] == 3
            assert analysis["context_items"] == 1
            
            # Check completion
            completion = manager.check_conversation_completion(session.session_id)
            assert completion["completed"] is False  # Missing required context
            
            # Add required context
            session.update_context("order_created", True, "system")
            session.update_context("user_confirmed", True, "system")
            
            # Now should be complete
            completion = manager.check_conversation_completion(session.session_id)
            assert completion["completed"] is True
            
            # Generate summary
            summary = manager.generate_conversation_summary(session.session_id)
            assert "margherita pizza" in summary
            assert "Total turns: 3" in summary
    
    def test_multiple_conversation_types(self):
        """Test managing multiple conversation types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConversationManager(temp_dir)
            
            # Create different types of conversations
            order_session = manager.create_conversation_session(
                "user_001", ConversationType.ORDER_PLACEMENT
            )
            vendor_session = manager.create_conversation_session(
                "user_001", ConversationType.VENDOR_SEARCH
            )
            preference_session = manager.create_conversation_session(
                "user_002", ConversationType.PREFERENCE_UPDATE
            )
            
            # Add specific context for each type
            order_session.update_context("order_items", ["pizza"], "user_input")
            vendor_session.update_context("search_criteria", "italian", "user_input")
            preference_session.update_context("preference_type", "dietary", "user_input")
            
            # Get user conversations
            user1_conversations = manager.get_user_conversations("user_001")
            assert len(user1_conversations) == 2
            
            user2_conversations = manager.get_user_conversations("user_002")
            assert len(user2_conversations) == 1
            
            # Check statistics
            stats = manager.get_conversation_statistics()
            assert stats["total_sessions"] == 3
            assert stats["active_sessions"] == 3
            assert "order_placement" in stats["conversation_types"]
            assert "vendor_search" in stats["conversation_types"]
            assert "preference_update" in stats["conversation_types"]
