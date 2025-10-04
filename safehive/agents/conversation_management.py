"""
Agent Conversation Management and Context Awareness

This module implements intelligent conversation management, context tracking,
and awareness capabilities for the SafeHive AI Security Sandbox, providing
coherent multi-turn conversations with persistent context and memory integration.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.metrics import record_metric, MetricType

logger = get_logger(__name__)


class ConversationState(Enum):
    """States of a conversation."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ERROR = "error"


class ConversationType(Enum):
    """Types of conversations."""
    ORDER_PLACEMENT = "order_placement"
    ORDER_INQUIRY = "order_inquiry"
    VENDOR_SEARCH = "vendor_search"
    PREFERENCE_UPDATE = "preference_update"
    SUPPORT = "support"
    GENERAL = "general"


class TurnType(Enum):
    """Types of conversation turns."""
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_MESSAGE = "system_message"
    ERROR_MESSAGE = "error_message"
    CONFIRMATION = "confirmation"
    CLARIFICATION = "clarification"


class ContextType(Enum):
    """Types of conversation context."""
    ORDER_CONTEXT = "order_context"
    USER_PREFERENCE = "user_preference"
    VENDOR_CONTEXT = "vendor_context"
    SYSTEM_STATE = "system_state"
    CONVERSATION_HISTORY = "conversation_history"
    EXTERNAL_DATA = "external_data"


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    turn_id: str
    conversation_id: str
    turn_type: TurnType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_updates: Dict[str, Any] = field(default_factory=dict)
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "turn_type": self.turn_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "context_updates": self.context_updates,
            "intent": self.intent,
            "entities": self.entities,
            "confidence": self.confidence
        }


@dataclass
class ConversationContext:
    """Represents the context of a conversation."""
    context_type: ContextType
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    source: str = "user_input"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context_type": self.context_type.value,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata
        }


@dataclass
class ConversationSession:
    """Represents a conversation session."""
    session_id: str
    user_id: str
    conversation_type: ConversationType
    state: ConversationState = ConversationState.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    turns: List[ConversationTurn] = field(default_factory=list)
    context: Dict[str, ConversationContext] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn to the conversation."""
        self.turns.append(turn)
        self.updated_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Update context based on turn
        for key, value in turn.context_updates.items():
            self.update_context(key, value, turn.turn_type.value)
    
    def update_context(self, key: str, value: Any, source: str = "system", 
                      context_type: ContextType = ContextType.CONVERSATION_HISTORY,
                      confidence: float = 1.0) -> None:
        """Update conversation context."""
        context_item = ConversationContext(
            context_type=context_type,
            key=key,
            value=value,
            source=source,
            confidence=confidence
        )
        self.context[key] = context_item
        self.updated_at = datetime.now()
    
    def get_context(self, key: str) -> Optional[Any]:
        """Get context value by key."""
        context_item = self.context.get(key)
        return context_item.value if context_item else None
    
    def get_contexts_by_type(self, context_type: ContextType) -> Dict[str, Any]:
        """Get all contexts of a specific type."""
        return {
            key: ctx.value for key, ctx in self.context.items()
            if ctx.context_type == context_type
        }
    
    def get_recent_turns(self, count: int = 5) -> List[ConversationTurn]:
        """Get recent conversation turns."""
        return self.turns[-count:] if self.turns else []
    
    def get_turns_by_type(self, turn_type: TurnType) -> List[ConversationTurn]:
        """Get turns by type."""
        return [turn for turn in self.turns if turn.turn_type == turn_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "conversation_type": self.conversation_type.value,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "turns": [turn.to_dict() for turn in self.turns],
            "context": {key: ctx.to_dict() for key, ctx in self.context.items()},
            "metadata": self.metadata,
            "summary": self.summary
        }


@dataclass
class ConversationFlow:
    """Represents a conversation flow pattern."""
    flow_id: str
    name: str
    description: str
    conversation_type: ConversationType
    expected_turns: List[TurnType] = field(default_factory=list)
    required_context: List[str] = field(default_factory=list)
    optional_context: List[str] = field(default_factory=list)
    completion_criteria: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "flow_id": self.flow_id,
            "name": self.name,
            "description": self.description,
            "conversation_type": self.conversation_type.value,
            "expected_turns": [turn.value for turn in self.expected_turns],
            "required_context": self.required_context,
            "optional_context": self.optional_context,
            "completion_criteria": self.completion_criteria,
            "metadata": self.metadata
        }


class ConversationManager:
    """Manager for conversation sessions and context awareness."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_conversations"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Conversation management
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.conversation_history: List[ConversationSession] = []
        self.conversation_flows: Dict[str, ConversationFlow] = {}
        
        # Context awareness
        self.context_patterns: Dict[str, List[str]] = {}
        self.intent_patterns: Dict[str, List[str]] = {}
        
        # Initialize default flows
        self._initialize_default_flows()
        
        # Load conversation history
        self._load_conversation_history()
        
        logger.info("Conversation Manager initialized")
    
    def _initialize_default_flows(self) -> None:
        """Initialize default conversation flows."""
        # Order placement flow
        order_placement_flow = ConversationFlow(
            flow_id="order_placement_flow",
            name="Order Placement Flow",
            description="Standard flow for placing food orders",
            conversation_type=ConversationType.ORDER_PLACEMENT,
            expected_turns=[
                TurnType.USER_INPUT,  # User requests to place order
                TurnType.AGENT_RESPONSE,  # Agent asks for preferences
                TurnType.USER_INPUT,  # User provides preferences
                TurnType.AGENT_RESPONSE,  # Agent suggests vendors/items
                TurnType.USER_INPUT,  # User selects items
                TurnType.CONFIRMATION,  # Agent confirms order
                TurnType.USER_INPUT,  # User confirms
                TurnType.AGENT_RESPONSE  # Agent processes order
            ],
            required_context=["user_preferences", "order_items", "vendor_selection"],
            optional_context=["delivery_address", "special_instructions", "payment_method"],
            completion_criteria={
                "order_created": True,
                "user_confirmed": True
            }
        )
        
        # Vendor search flow
        vendor_search_flow = ConversationFlow(
            flow_id="vendor_search_flow",
            name="Vendor Search Flow",
            description="Flow for searching and selecting vendors",
            conversation_type=ConversationType.VENDOR_SEARCH,
            expected_turns=[
                TurnType.USER_INPUT,  # User requests vendor search
                TurnType.AGENT_RESPONSE,  # Agent asks for criteria
                TurnType.USER_INPUT,  # User provides search criteria
                TurnType.AGENT_RESPONSE,  # Agent shows results
                TurnType.USER_INPUT,  # User selects vendor
                TurnType.AGENT_RESPONSE  # Agent confirms selection
            ],
            required_context=["search_criteria", "selected_vendor"],
            optional_context=["cuisine_preference", "location", "price_range"],
            completion_criteria={
                "vendor_selected": True
            }
        )
        
        # Preference update flow
        preference_update_flow = ConversationFlow(
            flow_id="preference_update_flow",
            name="Preference Update Flow",
            description="Flow for updating user preferences",
            conversation_type=ConversationType.PREFERENCE_UPDATE,
            expected_turns=[
                TurnType.USER_INPUT,  # User wants to update preferences
                TurnType.AGENT_RESPONSE,  # Agent asks what to update
                TurnType.USER_INPUT,  # User specifies preferences
                TurnType.CONFIRMATION,  # Agent confirms changes
                TurnType.USER_INPUT,  # User confirms
                TurnType.AGENT_RESPONSE  # Agent updates preferences
            ],
            required_context=["preference_type", "new_preference_value"],
            optional_context=["preference_category", "preference_priority"],
            completion_criteria={
                "preferences_updated": True
            }
        )
        
        self.conversation_flows["order_placement_flow"] = order_placement_flow
        self.conversation_flows["vendor_search_flow"] = vendor_search_flow
        self.conversation_flows["preference_update_flow"] = preference_update_flow
    
    def _load_conversation_history(self) -> None:
        """Load conversation history from storage."""
        history_file = self.storage_path / "conversation_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for session_data in data:
                        session = self._reconstruct_conversation_session(session_data)
                        if session:
                            self.conversation_history.append(session)
                logger.info(f"Loaded {len(self.conversation_history)} conversation sessions")
            except Exception as e:
                logger.error(f"Failed to load conversation history: {e}")
    
    def _save_conversation_history(self) -> None:
        """Save conversation history to storage."""
        history_file = self.storage_path / "conversation_history.json"
        try:
            # Combine active sessions and history
            all_sessions = list(self.active_sessions.values()) + self.conversation_history
            data = [session.to_dict() for session in all_sessions]
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved conversation history")
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")
    
    def _reconstruct_conversation_session(self, data: Dict[str, Any]) -> Optional[ConversationSession]:
        """Reconstruct ConversationSession from stored data."""
        try:
            # Reconstruct turns
            turns = []
            for turn_data in data.get("turns", []):
                turn = ConversationTurn(
                    turn_id=turn_data["turn_id"],
                    conversation_id=turn_data["conversation_id"],
                    turn_type=TurnType(turn_data["turn_type"]),
                    content=turn_data["content"],
                    timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                    metadata=turn_data.get("metadata", {}),
                    context_updates=turn_data.get("context_updates", {}),
                    intent=turn_data.get("intent"),
                    entities=turn_data.get("entities", {}),
                    confidence=turn_data.get("confidence", 1.0)
                )
                turns.append(turn)
            
            # Reconstruct context
            context = {}
            for key, ctx_data in data.get("context", {}).items():
                context_item = ConversationContext(
                    context_type=ContextType(ctx_data["context_type"]),
                    key=ctx_data["key"],
                    value=ctx_data["value"],
                    timestamp=datetime.fromisoformat(ctx_data["timestamp"]),
                    confidence=ctx_data.get("confidence", 1.0),
                    source=ctx_data.get("source", "user_input"),
                    metadata=ctx_data.get("metadata", {})
                )
                context[key] = context_item
            
            session = ConversationSession(
                session_id=data["session_id"],
                user_id=data["user_id"],
                conversation_type=ConversationType(data["conversation_type"]),
                state=ConversationState(data["state"]),
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                last_activity=datetime.fromisoformat(data["last_activity"]),
                turns=turns,
                context=context,
                metadata=data.get("metadata", {}),
                summary=data.get("summary")
            )
            
            return session
        except Exception as e:
            logger.error(f"Failed to reconstruct conversation session: {e}")
            return None
    
    def create_conversation_session(self, user_id: str, conversation_type: ConversationType,
                                  initial_context: Optional[Dict[str, Any]] = None) -> ConversationSession:
        """Create a new conversation session."""
        session_id = f"conv_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            conversation_type=conversation_type
        )
        
        # Add initial context if provided
        if initial_context:
            for key, value in initial_context.items():
                session.update_context(key, value, "system")
        
        # Add to active sessions
        self.active_sessions[session_id] = session
        
        logger.info(f"Created conversation session {session_id} for user {user_id}")
        
        # Record metrics
        record_metric("conversation.session_created", 1, MetricType.COUNTER, {
            "conversation_type": conversation_type.value,
            "user_id": user_id
        })
        
        return session
    
    def add_turn_to_conversation(self, session_id: str, turn_type: TurnType, content: str,
                               intent: Optional[str] = None, entities: Optional[Dict[str, Any]] = None,
                               context_updates: Optional[Dict[str, Any]] = None,
                               confidence: float = 1.0) -> Optional[ConversationTurn]:
        """Add a turn to a conversation session."""
        if session_id not in self.active_sessions:
            logger.error(f"Conversation session {session_id} not found")
            return None
        
        session = self.active_sessions[session_id]
        
        turn_id = f"turn_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        turn = ConversationTurn(
            turn_id=turn_id,
            conversation_id=session_id,
            turn_type=turn_type,
            content=content,
            intent=intent,
            entities=entities or {},
            context_updates=context_updates or {},
            confidence=confidence
        )
        
        session.add_turn(turn)
        
        logger.debug(f"Added turn {turn_id} to conversation {session_id}")
        
        # Record metrics
        record_metric("conversation.turn_added", 1, MetricType.COUNTER, {
            "turn_type": turn_type.value,
            "conversation_type": session.conversation_type.value,
            "session_id": session_id
        })
        
        return turn
    
    def get_conversation_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a conversation session by ID."""
        return self.active_sessions.get(session_id)
    
    def get_user_conversations(self, user_id: str, limit: int = 10) -> List[ConversationSession]:
        """Get recent conversations for a user."""
        user_sessions = []
        
        # Get from active sessions
        for session in self.active_sessions.values():
            if session.user_id == user_id:
                user_sessions.append(session)
        
        # Get from history
        for session in self.conversation_history:
            if session.user_id == user_id:
                user_sessions.append(session)
        
        # Sort by last activity and limit
        user_sessions.sort(key=lambda x: x.last_activity, reverse=True)
        return user_sessions[:limit]
    
    def update_conversation_state(self, session_id: str, state: ConversationState) -> bool:
        """Update the state of a conversation session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        old_state = session.state
        session.state = state
        session.updated_at = datetime.now()
        
        # If conversation is completed or abandoned, move to history
        if state in [ConversationState.COMPLETED, ConversationState.ABANDONED]:
            self.conversation_history.append(session)
            del self.active_sessions[session_id]
            self._save_conversation_history()
        
        logger.info(f"Updated conversation {session_id} state from {old_state.value} to {state.value}")
        
        # Record metrics
        record_metric("conversation.state_changed", 1, MetricType.COUNTER, {
            "old_state": old_state.value,
            "new_state": state.value,
            "session_id": session_id
        })
        
        return True
    
    def analyze_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """Analyze the context of a conversation session."""
        session = self.get_conversation_session(session_id)
        if not session:
            return {}
        
        analysis = {
            "session_id": session_id,
            "conversation_type": session.conversation_type.value,
            "state": session.state.value,
            "turn_count": len(session.turns),
            "context_items": len(session.context),
            "duration_minutes": (session.updated_at - session.created_at).total_seconds() / 60,
            "last_activity_minutes_ago": (datetime.now() - session.last_activity).total_seconds() / 60,
            "context_summary": {},
            "intent_analysis": {},
            "entity_analysis": {}
        }
        
        # Analyze context by type
        for context_type in ContextType:
            contexts = session.get_contexts_by_type(context_type)
            if contexts:
                analysis["context_summary"][context_type.value] = {
                    "count": len(contexts),
                    "keys": list(contexts.keys())
                }
        
        # Analyze intents
        intents = [turn.intent for turn in session.turns if turn.intent]
        if intents:
            intent_counts = {}
            for intent in intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            analysis["intent_analysis"] = intent_counts
        
        # Analyze entities
        all_entities = {}
        for turn in session.turns:
            for entity_type, entity_value in turn.entities.items():
                if entity_type not in all_entities:
                    all_entities[entity_type] = []
                all_entities[entity_type].append(entity_value)
        
        if all_entities:
            analysis["entity_analysis"] = {
                entity_type: list(set(values))  # Remove duplicates
                for entity_type, values in all_entities.items()
            }
        
        return analysis
    
    def get_conversation_flow(self, conversation_type: ConversationType) -> Optional[ConversationFlow]:
        """Get the conversation flow for a specific type."""
        for flow in self.conversation_flows.values():
            if flow.conversation_type == conversation_type:
                return flow
        return None
    
    def suggest_next_turn(self, session_id: str) -> Optional[TurnType]:
        """Suggest the next turn type based on conversation flow."""
        session = self.get_conversation_session(session_id)
        if not session:
            return None
        
        flow = self.get_conversation_flow(session.conversation_type)
        if not flow or not flow.expected_turns:
            return None
        
        # Find the next expected turn type
        current_turn_count = len(session.turns)
        if current_turn_count < len(flow.expected_turns):
            return flow.expected_turns[current_turn_count]
        
        # If we've exceeded expected turns, suggest based on context
        if session.get_context("order_created"):
            return TurnType.CONFIRMATION
        elif session.get_context("vendor_selected"):
            return TurnType.AGENT_RESPONSE
        else:
            return TurnType.CLARIFICATION
    
    def check_conversation_completion(self, session_id: str) -> Dict[str, Any]:
        """Check if a conversation meets completion criteria."""
        session = self.get_conversation_session(session_id)
        if not session:
            return {"completed": False, "reason": "Session not found"}
        
        flow = self.get_conversation_flow(session.conversation_type)
        if not flow:
            return {"completed": False, "reason": "No flow defined"}
        
        completion_status = {
            "completed": True,
            "criteria_met": {},
            "missing_criteria": [],
            "completion_percentage": 0.0
        }
        
        total_criteria = len(flow.completion_criteria)
        met_criteria = 0
        
        for criterion, expected_value in flow.completion_criteria.items():
            actual_value = session.get_context(criterion)
            if actual_value == expected_value:
                completion_status["criteria_met"][criterion] = actual_value
                met_criteria += 1
            else:
                completion_status["missing_criteria"].append(criterion)
        
        completion_status["completion_percentage"] = (met_criteria / total_criteria) * 100 if total_criteria > 0 else 0
        
        if completion_status["missing_criteria"]:
            completion_status["completed"] = False
        
        return completion_status
    
    def generate_conversation_summary(self, session_id: str) -> Optional[str]:
        """Generate a summary of the conversation."""
        session = self.get_conversation_session(session_id)
        if not session:
            return None
        
        # Basic summary based on conversation type and key events
        summary_parts = [f"Conversation type: {session.conversation_type.value}"]
        
        if session.conversation_type == ConversationType.ORDER_PLACEMENT:
            order_items = session.get_context("order_items")
            vendor = session.get_context("vendor_selection")
            if order_items:
                summary_parts.append(f"Order items: {order_items}")
            if vendor:
                summary_parts.append(f"Selected vendor: {vendor}")
        
        elif session.conversation_type == ConversationType.VENDOR_SEARCH:
            search_criteria = session.get_context("search_criteria")
            selected_vendor = session.get_context("selected_vendor")
            if search_criteria:
                summary_parts.append(f"Search criteria: {search_criteria}")
            if selected_vendor:
                summary_parts.append(f"Selected vendor: {selected_vendor}")
        
        elif session.conversation_type == ConversationType.PREFERENCE_UPDATE:
            preference_type = session.get_context("preference_type")
            new_value = session.get_context("new_preference_value")
            if preference_type and new_value:
                summary_parts.append(f"Updated {preference_type}: {new_value}")
        
        # Add turn count and duration
        summary_parts.append(f"Total turns: {len(session.turns)}")
        duration = (session.updated_at - session.created_at).total_seconds() / 60
        summary_parts.append(f"Duration: {duration:.1f} minutes")
        
        summary = ". ".join(summary_parts) + "."
        session.summary = summary
        
        return summary
    
    def get_conversation_statistics(self) -> Dict[str, Any]:
        """Get conversation management statistics."""
        total_sessions = len(self.active_sessions) + len(self.conversation_history)
        
        if total_sessions == 0:
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "completed_sessions": 0,
                "conversation_types": {},
                "average_turns_per_session": 0,
                "average_duration_minutes": 0
            }
        
        # Count by conversation type
        type_counts = {}
        for session in list(self.active_sessions.values()) + self.conversation_history:
            conv_type = session.conversation_type.value
            type_counts[conv_type] = type_counts.get(conv_type, 0) + 1
        
        # Count by state
        state_counts = {}
        for session in list(self.active_sessions.values()) + self.conversation_history:
            state = session.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Calculate averages
        total_turns = sum(len(session.turns) for session in list(self.active_sessions.values()) + self.conversation_history)
        average_turns = total_turns / total_sessions if total_sessions > 0 else 0
        
        total_duration = sum(
            (session.updated_at - session.created_at).total_seconds() / 60
            for session in list(self.active_sessions.values()) + self.conversation_history
        )
        average_duration = total_duration / total_sessions if total_sessions > 0 else 0
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.conversation_history),
            "conversation_types": type_counts,
            "state_counts": state_counts,
            "average_turns_per_session": average_turns,
            "average_duration_minutes": average_duration,
            "flows_available": len(self.conversation_flows)
        }
    
    def cleanup_inactive_sessions(self, max_idle_minutes: int = 60) -> int:
        """Clean up inactive conversation sessions."""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session in self.active_sessions.items():
            idle_minutes = (current_time - session.last_activity).total_seconds() / 60
            if idle_minutes > max_idle_minutes:
                sessions_to_remove.append(session_id)
        
        # Move inactive sessions to history
        for session_id in sessions_to_remove:
            session = self.active_sessions[session_id]
            session.state = ConversationState.ABANDONED
            self.conversation_history.append(session)
            del self.active_sessions[session_id]
        
        if sessions_to_remove:
            self._save_conversation_history()
            logger.info(f"Cleaned up {len(sessions_to_remove)} inactive conversation sessions")
        
        return len(sessions_to_remove)
    
    def get_active_sessions(self) -> List[ConversationSession]:
        """Get all active conversation sessions."""
        return list(self.active_sessions.values())
    
    def get_conversation_history(self, limit: int = 100) -> List[ConversationSession]:
        """Get conversation history."""
        return sorted(self.conversation_history, key=lambda x: x.updated_at, reverse=True)[:limit]
