"""
Agent Models for State, Memory, and Conversation Data

This module defines the data structures used for managing AI agent state,
memory, conversations, and related metadata in the SafeHive system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import json


class AgentType(Enum):
    """Types of agents in the system."""
    ORCHESTRATOR = "orchestrator"
    USER_TWIN = "user_twin"
    HONEST_VENDOR = "honest_vendor"
    MALICIOUS_VENDOR = "malicious_vendor"


class AgentState(Enum):
    """Possible states for an agent."""
    IDLE = "idle"
    ACTIVE = "active"
    THINKING = "thinking"
    RESPONDING = "responding"
    ERROR = "error"
    DISABLED = "disabled"


class MessageType(Enum):
    """Types of messages in agent conversations."""
    REQUEST = "request"
    RESPONSE = "response"
    SYSTEM = "system"
    ERROR = "error"
    THINKING = "thinking"


@dataclass
class AgentMessage:
    """
    Represents a single message in an agent conversation.
    
    This class encapsulates all information about a message including
    content, metadata, and conversation context.
    """
    content: str
    message_type: MessageType
    sender: str
    recipient: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "content": self.content,
            "message_type": self.message_type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "message_id": self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(
            content=data["content"],
            message_type=MessageType(data["message_type"]),
            sender=data["sender"],
            recipient=data["recipient"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            message_id=data.get("message_id")
        )


@dataclass
class Conversation:
    """
    Represents a conversation between agents.
    
    This class manages a sequence of messages between agents,
    including metadata about the conversation context.
    """
    conversation_id: str
    participants: List[str]
    messages: List[AgentMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def add_message(self, message: AgentMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_messages_by_type(self, message_type: MessageType) -> List[AgentMessage]:
        """Get all messages of a specific type."""
        return [msg for msg in self.messages if msg.message_type == message_type]
    
    def get_messages_from(self, sender: str) -> List[AgentMessage]:
        """Get all messages from a specific sender."""
        return [msg for msg in self.messages if msg.sender == sender]
    
    def get_messages_to(self, recipient: str) -> List[AgentMessage]:
        """Get all messages to a specific recipient."""
        return [msg for msg in self.messages if msg.recipient == recipient]
    
    def get_latest_message(self) -> Optional[AgentMessage]:
        """Get the most recent message in the conversation."""
        return self.messages[-1] if self.messages else None
    
    def get_message_count(self) -> int:
        """Get the total number of messages in the conversation."""
        return len(self.messages)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "participants": self.participants,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create conversation from dictionary."""
        messages = [AgentMessage.from_dict(msg_data) for msg_data in data.get("messages", [])]
        return cls(
            conversation_id=data["conversation_id"],
            participants=data["participants"],
            messages=messages,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            is_active=data.get("is_active", True)
        )


@dataclass
class AgentMemory:
    """
    Represents the memory state of an AI agent.
    
    This class manages the agent's memory including conversation history,
    learned patterns, and persistent state information.
    """
    agent_id: str
    conversation_history: List[Conversation] = field(default_factory=list)
    learned_patterns: Dict[str, Any] = field(default_factory=dict)
    persistent_state: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    memory_size_limit: int = 1000  # Maximum number of conversations to keep
    
    def add_conversation(self, conversation: Conversation) -> None:
        """Add a conversation to the agent's memory."""
        self.conversation_history.append(conversation)
        self.last_updated = datetime.now()
        
        # Enforce memory size limit
        if len(self.conversation_history) > self.memory_size_limit:
            # Remove oldest conversations
            self.conversation_history = self.conversation_history[-self.memory_size_limit:]
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a specific conversation by ID."""
        for conv in self.conversation_history:
            if conv.conversation_id == conversation_id:
                return conv
        return None
    
    def get_recent_conversations(self, count: int = 10) -> List[Conversation]:
        """Get the most recent conversations."""
        return self.conversation_history[-count:] if self.conversation_history else []
    
    def update_learned_pattern(self, pattern_name: str, pattern_data: Any) -> None:
        """Update a learned pattern in the agent's memory."""
        self.learned_patterns[pattern_name] = pattern_data
        self.last_updated = datetime.now()
    
    def get_learned_pattern(self, pattern_name: str) -> Optional[Any]:
        """Get a learned pattern by name."""
        return self.learned_patterns.get(pattern_name)
    
    def update_persistent_state(self, key: str, value: Any) -> None:
        """Update persistent state information."""
        self.persistent_state[key] = value
        self.last_updated = datetime.now()
    
    def get_persistent_state(self, key: str) -> Optional[Any]:
        """Get persistent state information by key."""
        return self.persistent_state.get(key)
    
    def clear_memory(self) -> None:
        """Clear all memory data."""
        self.conversation_history.clear()
        self.learned_patterns.clear()
        self.persistent_state.clear()
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "conversation_history": [conv.to_dict() for conv in self.conversation_history],
            "learned_patterns": self.learned_patterns,
            "persistent_state": self.persistent_state,
            "last_updated": self.last_updated.isoformat(),
            "memory_size_limit": self.memory_size_limit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMemory":
        """Create memory from dictionary."""
        conversations = [Conversation.from_dict(conv_data) for conv_data in data.get("conversation_history", [])]
        return cls(
            agent_id=data["agent_id"],
            conversation_history=conversations,
            learned_patterns=data.get("learned_patterns", {}),
            persistent_state=data.get("persistent_state", {}),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            memory_size_limit=data.get("memory_size_limit", 1000)
        )


@dataclass
class AgentStatus:
    """
    Represents the current state of an AI agent.
    
    This class tracks the agent's current status, configuration,
    and operational metrics.
    """
    agent_id: str
    agent_type: AgentType
    current_state: AgentState
    is_enabled: bool = True
    configuration: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    success_count: int = 0
    
    def update_state(self, new_state: AgentState) -> None:
        """Update the agent's current state."""
        self.current_state = new_state
        self.last_activity = datetime.now()
    
    def increment_error_count(self) -> None:
        """Increment the error count."""
        self.error_count += 1
        self.last_activity = datetime.now()
    
    def increment_success_count(self) -> None:
        """Increment the success count."""
        self.success_count += 1
        self.last_activity = datetime.now()
    
    def update_metric(self, metric_name: str, value: Any) -> None:
        """Update a specific metric."""
        self.metrics[metric_name] = value
        self.last_activity = datetime.now()
    
    def get_metric(self, metric_name: str) -> Optional[Any]:
        """Get a specific metric value."""
        return self.metrics.get(metric_name)
    
    def get_success_rate(self) -> float:
        """Calculate the success rate."""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.0
    
    def is_healthy(self) -> bool:
        """Check if the agent is in a healthy state."""
        return (
            self.is_enabled and
            self.current_state != AgentState.ERROR and
            self.get_success_rate() > 0.5  # At least 50% success rate
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "current_state": self.current_state.value,
            "is_enabled": self.is_enabled,
            "configuration": self.configuration,
            "metrics": self.metrics,
            "last_activity": self.last_activity.isoformat(),
            "error_count": self.error_count,
            "success_count": self.success_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentStatus":
        """Create agent state from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            agent_type=AgentType(data["agent_type"]),
            current_state=AgentState(data["current_state"]),
            is_enabled=data.get("is_enabled", True),
            configuration=data.get("configuration", {}),
            metrics=data.get("metrics", {}),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            error_count=data.get("error_count", 0),
            success_count=data.get("success_count", 0)
        )


@dataclass
class AgentPersonality:
    """
    Represents the personality and behavior configuration of an agent.
    
    This class defines how an agent should behave, including its
    communication style, attack patterns (for malicious agents),
    and response characteristics.
    """
    agent_id: str
    personality_type: str  # e.g., "helpful", "aggressive", "deceptive"
    communication_style: Dict[str, Any] = field(default_factory=dict)
    attack_patterns: List[str] = field(default_factory=list)
    response_characteristics: Dict[str, Any] = field(default_factory=dict)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    is_malicious: bool = False
    
    def add_attack_pattern(self, pattern: str) -> None:
        """Add an attack pattern to the agent's repertoire."""
        if pattern not in self.attack_patterns:
            self.attack_patterns.append(pattern)
    
    def remove_attack_pattern(self, pattern: str) -> None:
        """Remove an attack pattern from the agent's repertoire."""
        if pattern in self.attack_patterns:
            self.attack_patterns.remove(pattern)
    
    def has_attack_pattern(self, pattern: str) -> bool:
        """Check if the agent has a specific attack pattern."""
        return pattern in self.attack_patterns
    
    def update_communication_style(self, style_key: str, style_value: Any) -> None:
        """Update a communication style parameter."""
        self.communication_style[style_key] = style_value
    
    def get_communication_style(self, style_key: str) -> Optional[Any]:
        """Get a communication style parameter."""
        return self.communication_style.get(style_key)
    
    def add_knowledge(self, knowledge_key: str, knowledge_value: Any) -> None:
        """Add knowledge to the agent's knowledge base."""
        self.knowledge_base[knowledge_key] = knowledge_value
    
    def get_knowledge(self, knowledge_key: str) -> Optional[Any]:
        """Get knowledge from the agent's knowledge base."""
        return self.knowledge_base.get(knowledge_key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert personality to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "personality_type": self.personality_type,
            "communication_style": self.communication_style,
            "attack_patterns": self.attack_patterns,
            "response_characteristics": self.response_characteristics,
            "knowledge_base": self.knowledge_base,
            "is_malicious": self.is_malicious
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPersonality":
        """Create personality from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            personality_type=data["personality_type"],
            communication_style=data.get("communication_style", {}),
            attack_patterns=data.get("attack_patterns", []),
            response_characteristics=data.get("response_characteristics", {}),
            knowledge_base=data.get("knowledge_base", {}),
            is_malicious=data.get("is_malicious", False)
        )


# Convenience functions for creating common agent models
def create_agent_message(
    content: str,
    sender: str,
    recipient: str,
    message_type: MessageType = MessageType.REQUEST,
    metadata: Optional[Dict[str, Any]] = None
) -> AgentMessage:
    """Create an agent message with default values."""
    return AgentMessage(
        content=content,
        message_type=message_type,
        sender=sender,
        recipient=recipient,
        metadata=metadata or {}
    )


def create_conversation(
    conversation_id: str,
    participants: List[str],
    metadata: Optional[Dict[str, Any]] = None
) -> Conversation:
    """Create a new conversation."""
    return Conversation(
        conversation_id=conversation_id,
        participants=participants,
        metadata=metadata or {}
    )


def create_agent_memory(agent_id: str, memory_size_limit: int = 1000) -> AgentMemory:
    """Create a new agent memory instance."""
    return AgentMemory(
        agent_id=agent_id,
        memory_size_limit=memory_size_limit
    )


def create_agent_status(
    agent_id: str,
    agent_type: AgentType,
    initial_state: AgentState = AgentState.IDLE
) -> AgentStatus:
    """Create a new agent status instance."""
    return AgentStatus(
        agent_id=agent_id,
        agent_type=agent_type,
        current_state=initial_state
    )


def create_honest_vendor_personality(agent_id: str) -> AgentPersonality:
    """Create a personality for an honest vendor agent."""
    return AgentPersonality(
        agent_id=agent_id,
        personality_type="helpful",
        communication_style={
            "tone": "friendly",
            "response_time": "fast",
            "detail_level": "comprehensive"
        },
        response_characteristics={
            "honesty": 1.0,
            "helpfulness": 1.0,
            "aggression": 0.0
        },
        is_malicious=False
    )


def create_malicious_vendor_personality(agent_id: str) -> AgentPersonality:
    """Create a personality for a malicious vendor agent."""
    return AgentPersonality(
        agent_id=agent_id,
        personality_type="deceptive",
        communication_style={
            "tone": "manipulative",
            "response_time": "variable",
            "detail_level": "selective"
        },
        attack_patterns=[
            "sql_injection",
            "xss_attack",
            "data_exfiltration",
            "social_engineering"
        ],
        response_characteristics={
            "honesty": 0.2,
            "helpfulness": 0.3,
            "aggression": 0.8,
            "deceptiveness": 0.9
        },
        is_malicious=True
    )
