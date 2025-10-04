"""
Agent Communication System

This module implements the communication protocols and message passing system
for inter-agent communication in the SafeHive AI Security Sandbox.
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from collections import defaultdict, deque
import threading
import time

from ..utils.logger import get_logger
from ..utils.metrics import record_metric, increment_counter, MetricType
from ..models.agent_models import AgentMessage, MessageType, AgentType

logger = get_logger(__name__)


class MessagePriority(Enum):
    """Priority levels for messages."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageStatus(Enum):
    """Status of a message in the communication system."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"


class CommunicationEvent(Enum):
    """Events in the communication system."""
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_DELIVERED = "message_delivered"
    MESSAGE_ACKNOWLEDGED = "message_acknowledged"
    MESSAGE_FAILED = "message_failed"
    AGENT_CONNECTED = "agent_connected"
    AGENT_DISCONNECTED = "agent_disconnected"
    SUBSCRIPTION_CREATED = "subscription_created"
    SUBSCRIPTION_CANCELLED = "subscription_cancelled"


@dataclass
class CommunicationMessage:
    """
    Enhanced message structure for inter-agent communication.
    
    This extends the basic AgentMessage with communication-specific
    metadata and routing information.
    """
    # Core message data
    content: str
    message_type: MessageType
    sender: str
    recipient: str
    
    # Communication metadata
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Routing and delivery
    routing_key: Optional[str] = None
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Message metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Delivery tracking
    delivery_attempts: List[datetime] = field(default_factory=list)
    acknowledgment_received: bool = False
    acknowledgment_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "content": self.content,
            "message_type": self.message_type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "priority": self.priority.value,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "routing_key": self.routing_key,
            "reply_to": self.reply_to,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
            "headers": self.headers,
            "delivery_attempts": [dt.isoformat() for dt in self.delivery_attempts],
            "acknowledgment_received": self.acknowledgment_received,
            "acknowledgment_timestamp": self.acknowledgment_timestamp.isoformat() if self.acknowledgment_timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommunicationMessage":
        """Create message from dictionary."""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            content=data["content"],
            message_type=MessageType(data["message_type"]),
            sender=data["sender"],
            recipient=data["recipient"],
            priority=MessagePriority(data.get("priority", "normal")),
            status=MessageStatus(data.get("status", "pending")),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            routing_key=data.get("routing_key"),
            reply_to=data.get("reply_to"),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
            headers=data.get("headers", {}),
            delivery_attempts=[datetime.fromisoformat(dt) for dt in data.get("delivery_attempts", [])],
            acknowledgment_received=data.get("acknowledgment_received", False),
            acknowledgment_timestamp=datetime.fromisoformat(data["acknowledgment_timestamp"]) if data.get("acknowledgment_timestamp") else None
        )
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries and not self.is_expired()
    
    def to_agent_message(self) -> AgentMessage:
        """Convert to basic AgentMessage."""
        return AgentMessage(
            content=self.content,
            message_type=self.message_type,
            sender=self.sender,
            recipient=self.recipient,
            timestamp=self.timestamp,
            metadata={
                **self.metadata,
                "communication_message_id": self.message_id,
                "priority": self.priority.value,
                "routing_key": self.routing_key,
                "correlation_id": self.correlation_id
            },
            message_id=self.message_id
        )


@dataclass
class MessageSubscription:
    """Represents a message subscription for filtering."""
    subscriber_id: str
    filter_criteria: Dict[str, Any]
    callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True


class MessageQueue:
    """Thread-safe message queue for agent communication."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queues: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stats = {
            "total_messages": 0,
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "queue_sizes": defaultdict(int)
        }
    
    def put(self, message: CommunicationMessage, queue_name: str = "default") -> bool:
        """Add message to queue."""
        with self._condition:
            if len(self._queues[queue_name]) >= self.max_size:
                logger.warning(f"Queue {queue_name} is full, dropping message {message.message_id}")
                return False
            
            self._queues[queue_name].append(message)
            self._stats["total_messages"] += 1
            self._stats["queue_sizes"][queue_name] = len(self._queues[queue_name])
            self._condition.notify()
            return True
    
    def get(self, queue_name: str = "default", timeout: Optional[float] = None) -> Optional[CommunicationMessage]:
        """Get message from queue."""
        with self._condition:
            if timeout is None:
                while not self._queues[queue_name]:
                    self._condition.wait()
            else:
                if not self._condition.wait_for(
                    lambda: bool(self._queues[queue_name]), 
                    timeout=timeout
                ):
                    return None
            
            message = self._queues[queue_name].popleft()
            self._stats["queue_sizes"][queue_name] = len(self._queues[queue_name])
            return message
    
    def peek(self, queue_name: str = "default") -> Optional[CommunicationMessage]:
        """Peek at next message without removing it."""
        with self._lock:
            return self._queues[queue_name][0] if self._queues[queue_name] else None
    
    def size(self, queue_name: str = "default") -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queues[queue_name])
    
    def clear(self, queue_name: str = "default") -> int:
        """Clear queue and return number of messages removed."""
        with self._lock:
            count = len(self._queues[queue_name])
            self._queues[queue_name].clear()
            self._stats["queue_sizes"][queue_name] = 0
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                **self._stats,
                "queue_sizes": dict(self._stats["queue_sizes"]),
                "total_queues": len(self._queues)
            }


class MessageRouter:
    """Routes messages between agents based on routing rules."""
    
    def __init__(self):
        self._routes: Dict[str, List[str]] = defaultdict(list)
        self._wildcard_routes: List[str] = []
        self._subscriptions: Dict[str, MessageSubscription] = {}
        self._lock = threading.RLock()
    
    def add_route(self, routing_key: str, target_agents: List[str]) -> None:
        """Add routing rule."""
        with self._lock:
            if routing_key == "*":
                self._wildcard_routes.extend(target_agents)
            else:
                self._routes[routing_key].extend(target_agents)
    
    def remove_route(self, routing_key: str, target_agent: str) -> None:
        """Remove routing rule."""
        with self._lock:
            if routing_key == "*":
                if target_agent in self._wildcard_routes:
                    self._wildcard_routes.remove(target_agent)
            else:
                if target_agent in self._routes[routing_key]:
                    self._routes[routing_key].remove(target_agent)
    
    def get_targets(self, routing_key: str) -> List[str]:
        """Get target agents for routing key."""
        with self._lock:
            targets = self._routes.get(routing_key, []).copy()
            targets.extend(self._wildcard_routes)
            return list(set(targets))  # Remove duplicates
    
    def add_subscription(self, subscription: MessageSubscription) -> str:
        """Add message subscription."""
        with self._lock:
            subscription_id = str(uuid.uuid4())
            self._subscriptions[subscription_id] = subscription
            return subscription_id
    
    def remove_subscription(self, subscription_id: str) -> bool:
        """Remove message subscription."""
        with self._lock:
            return self._subscriptions.pop(subscription_id, None) is not None
    
    def get_matching_subscriptions(self, message: CommunicationMessage) -> List[MessageSubscription]:
        """Get subscriptions that match the message."""
        with self._lock:
            matching = []
            for subscription in self._subscriptions.values():
                if not subscription.active:
                    continue
                
                if self._message_matches_criteria(message, subscription.filter_criteria):
                    matching.append(subscription)
            
            return matching
    
    def _message_matches_criteria(self, message: CommunicationMessage, criteria: Dict[str, Any]) -> bool:
        """Check if message matches subscription criteria."""
        for key, expected_value in criteria.items():
            if key == "sender" and message.sender != expected_value:
                return False
            elif key == "recipient" and message.recipient != expected_value:
                return False
            elif key == "message_type" and message.message_type.value != expected_value:
                return False
            elif key == "priority" and message.priority.value != expected_value:
                return False
            elif key == "routing_key" and message.routing_key != expected_value:
                return False
            elif key == "content_contains" and expected_value not in message.content:
                return False
            elif key == "metadata":
                if isinstance(expected_value, dict):
                    for meta_key, meta_value in expected_value.items():
                        if message.metadata.get(meta_key) != meta_value:
                            return False
        
        return True


class CommunicationManager:
    """
    Central communication manager for inter-agent messaging.
    
    This class manages message routing, delivery, acknowledgments,
    and provides a unified interface for agent communication.
    """
    
    def __init__(self):
        self._message_queue = MessageQueue()
        self._router = MessageRouter()
        self._connected_agents: Set[str] = set()
        self._agent_queues: Dict[str, str] = {}
        self._message_history: Dict[str, CommunicationMessage] = {}
        self._event_handlers: Dict[CommunicationEvent, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self._running = False
        self._delivery_thread: Optional[threading.Thread] = None
        
        # Configuration
        self.delivery_timeout = 30.0  # seconds
        self.retry_delay = 5.0  # seconds
        self.cleanup_interval = 300.0  # seconds (5 minutes)
        
        logger.info("CommunicationManager initialized")
    
    def start(self) -> None:
        """Start the communication manager."""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._delivery_thread = threading.Thread(target=self._delivery_worker, daemon=True)
            self._delivery_thread.start()
            
            # Start cleanup thread
            cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            cleanup_thread.start()
            
            logger.info("CommunicationManager started")
    
    def stop(self) -> None:
        """Stop the communication manager."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._delivery_thread:
                self._delivery_thread.join(timeout=5.0)
            
            logger.info("CommunicationManager stopped")
    
    def register_agent(self, agent_id: str, queue_name: Optional[str] = None) -> None:
        """Register an agent with the communication system."""
        with self._lock:
            self._connected_agents.add(agent_id)
            if queue_name:
                self._agent_queues[agent_id] = queue_name
            else:
                self._agent_queues[agent_id] = f"agent_{agent_id}"
            
            self._emit_event(CommunicationEvent.AGENT_CONNECTED, {"agent_id": agent_id})
            logger.info(f"Agent {agent_id} registered with queue {self._agent_queues[agent_id]}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the communication system."""
        with self._lock:
            self._connected_agents.discard(agent_id)
            self._agent_queues.pop(agent_id, None)
            
            # Clear agent's queue
            queue_name = f"agent_{agent_id}"
            cleared_count = self._message_queue.clear(queue_name)
            
            self._emit_event(CommunicationEvent.AGENT_DISCONNECTED, {"agent_id": agent_id})
            logger.info(f"Agent {agent_id} unregistered, cleared {cleared_count} messages")
    
    def send_message(
        self,
        message: CommunicationMessage,
        wait_for_ack: bool = False,
        timeout: Optional[float] = None
    ) -> bool:
        """Send a message to an agent."""
        with self._lock:
            if not self._running:
                logger.error("CommunicationManager not running")
                return False
            
            # Validate message
            if not self._validate_message(message):
                return False
            
            # Set expiration if not set
            if message.expires_at is None:
                message.expires_at = datetime.now() + timedelta(seconds=self.delivery_timeout)
            
            # Store message in history
            self._message_history[message.message_id] = message
            
            # Determine target queue
            target_queue = self._get_target_queue(message)
            if not target_queue:
                logger.error(f"No target queue found for message {message.message_id}")
                message.status = MessageStatus.FAILED
                return False
            
            # Add to queue
            if not self._message_queue.put(message, target_queue):
                logger.error(f"Failed to queue message {message.message_id}")
                message.status = MessageStatus.FAILED
                return False
            
            message.status = MessageStatus.SENT
            self._emit_event(CommunicationEvent.MESSAGE_SENT, {"message": message})
            
            # Wait for acknowledgment if requested
            if wait_for_ack:
                return self._wait_for_acknowledgment(message, timeout)
            
            return True
    
    def receive_message(self, agent_id: str, timeout: Optional[float] = None) -> Optional[CommunicationMessage]:
        """Receive a message for an agent."""
        queue_name = self._agent_queues.get(agent_id)
        if not queue_name:
            logger.error(f"No queue found for agent {agent_id}")
            return None
        
        message = self._message_queue.get(queue_name, timeout)
        if message:
            message.status = MessageStatus.DELIVERED
            message.delivery_attempts.append(datetime.now())
            self._emit_event(CommunicationEvent.MESSAGE_RECEIVED, {"message": message, "agent_id": agent_id})
        
        return message
    
    def acknowledge_message(self, message_id: str, agent_id: str) -> bool:
        """Acknowledge receipt of a message."""
        with self._lock:
            message = self._message_history.get(message_id)
            if not message:
                logger.warning(f"Message {message_id} not found in history")
                return False
            
            message.acknowledgment_received = True
            message.acknowledgment_timestamp = datetime.now()
            message.status = MessageStatus.ACKNOWLEDGED
            
            self._emit_event(CommunicationEvent.MESSAGE_ACKNOWLEDGED, {"message": message, "agent_id": agent_id})
            logger.debug(f"Message {message_id} acknowledged by agent {agent_id}")
            return True
    
    def broadcast_message(
        self,
        message: CommunicationMessage,
        target_agents: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Broadcast a message to multiple agents."""
        results = {}
        
        if target_agents is None:
            target_agents = list(self._connected_agents)
        
        for agent_id in target_agents:
            if agent_id in self._connected_agents:
                # Create a copy of the message for each recipient
                broadcast_msg = CommunicationMessage(
                    content=message.content,
                    message_type=message.message_type,
                    sender=message.sender,
                    recipient=agent_id,
                    priority=message.priority,
                    routing_key=message.routing_key,
                    reply_to=message.reply_to,
                    correlation_id=message.correlation_id,
                    metadata=message.metadata.copy(),
                    headers=message.headers.copy(),
                    max_retries=message.max_retries
                )
                
                results[agent_id] = self.send_message(broadcast_msg)
            else:
                results[agent_id] = False
                logger.warning(f"Agent {agent_id} not connected for broadcast")
        
        return results
    
    def subscribe_to_messages(
        self,
        subscriber_id: str,
        filter_criteria: Dict[str, Any],
        callback: Optional[Callable] = None
    ) -> str:
        """Subscribe to messages matching criteria."""
        subscription = MessageSubscription(
            subscriber_id=subscriber_id,
            filter_criteria=filter_criteria,
            callback=callback
        )
        
        subscription_id = self._router.add_subscription(subscription)
        self._emit_event(CommunicationEvent.SUBSCRIPTION_CREATED, {"subscription_id": subscription_id})
        logger.info(f"Subscription created for {subscriber_id}: {filter_criteria}")
        
        return subscription_id
    
    def unsubscribe_from_messages(self, subscription_id: str) -> bool:
        """Cancel a message subscription."""
        success = self._router.remove_subscription(subscription_id)
        if success:
            self._emit_event(CommunicationEvent.SUBSCRIPTION_CANCELLED, {"subscription_id": subscription_id})
            logger.info(f"Subscription {subscription_id} cancelled")
        
        return success
    
    def add_event_handler(self, event: CommunicationEvent, handler: Callable) -> None:
        """Add event handler for communication events."""
        self._event_handlers[event].append(handler)
    
    def remove_event_handler(self, event: CommunicationEvent, handler: Callable) -> None:
        """Remove event handler."""
        if handler in self._event_handlers[event]:
            self._event_handlers[event].remove(handler)
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get communication status for an agent."""
        with self._lock:
            queue_name = self._agent_queues.get(agent_id)
            queue_size = self._message_queue.size(queue_name) if queue_name else 0
            
            return {
                "connected": agent_id in self._connected_agents,
                "queue_name": queue_name,
                "queue_size": queue_size,
                "messages_sent": sum(1 for msg in self._message_history.values() if msg.sender == agent_id),
                "messages_received": sum(1 for msg in self._message_history.values() if msg.recipient == agent_id)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall communication system statistics."""
        with self._lock:
            queue_stats = self._message_queue.get_stats()
            
            return {
                "connected_agents": len(self._connected_agents),
                "total_messages": len(self._message_history),
                "queue_stats": queue_stats,
                "subscriptions": len(self._router._subscriptions),
                "running": self._running
            }
    
    def _validate_message(self, message: CommunicationMessage) -> bool:
        """Validate message before sending."""
        if not message.content:
            logger.error("Message content cannot be empty")
            return False
        
        if not message.sender:
            logger.error("Message sender cannot be empty")
            return False
        
        if not message.recipient:
            logger.error("Message recipient cannot be empty")
            return False
        
        if message.sender not in self._connected_agents:
            logger.warning(f"Sender {message.sender} not registered")
        
        return True
    
    def _get_target_queue(self, message: CommunicationMessage) -> Optional[str]:
        """Get target queue for message."""
        # Direct recipient
        if message.recipient in self._agent_queues:
            return self._agent_queues[message.recipient]
        
        # Routing-based delivery
        if message.routing_key:
            targets = self._router.get_targets(message.routing_key)
            if targets:
                # For now, send to first target (could be enhanced for load balancing)
                target = targets[0]
                return self._agent_queues.get(target)
        
        return None
    
    def _wait_for_acknowledgment(self, message: CommunicationMessage, timeout: Optional[float]) -> bool:
        """Wait for message acknowledgment."""
        start_time = time.time()
        timeout = timeout or self.delivery_timeout
        
        while time.time() - start_time < timeout:
            if message.acknowledgment_received:
                return True
            
            time.sleep(0.1)  # Small delay to avoid busy waiting
        
        logger.warning(f"Message {message.message_id} acknowledgment timeout")
        return False
    
    def _delivery_worker(self) -> None:
        """Background worker for message delivery and retry logic."""
        while self._running:
            try:
                # Process failed messages for retry
                self._process_failed_messages()
                
                # Process subscriptions
                self._process_subscriptions()
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in delivery worker: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def _process_failed_messages(self) -> None:
        """Process failed messages for retry."""
        with self._lock:
            for message in list(self._message_history.values()):
                if (message.status == MessageStatus.FAILED and 
                    message.can_retry() and 
                    not message.is_expired()):
                    
                    # Check if enough time has passed since last attempt
                    if message.delivery_attempts:
                        last_attempt = message.delivery_attempts[-1]
                        if datetime.now() - last_attempt < timedelta(seconds=self.retry_delay):
                            continue
                    
                    # Retry message
                    message.retry_count += 1
                    message.status = MessageStatus.PENDING
                    
                    target_queue = self._get_target_queue(message)
                    if target_queue:
                        self._message_queue.put(message, target_queue)
                        logger.info(f"Retrying message {message.message_id} (attempt {message.retry_count})")
    
    def _process_subscriptions(self) -> None:
        """Process message subscriptions."""
        # This would be implemented to handle subscription-based message delivery
        # For now, it's a placeholder for future enhancement
        pass
    
    def _cleanup_worker(self) -> None:
        """Background worker for cleanup tasks."""
        while self._running:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired_messages()
                self._cleanup_old_history()
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    def _cleanup_expired_messages(self) -> None:
        """Remove expired messages from queues."""
        # This would be implemented to clean up expired messages
        # For now, it's a placeholder
        pass
    
    def _cleanup_old_history(self) -> None:
        """Clean up old message history."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours of history
            expired_messages = [
                msg_id for msg_id, msg in self._message_history.items()
                if msg.timestamp < cutoff_time
            ]
            
            for msg_id in expired_messages:
                del self._message_history[msg_id]
            
            if expired_messages:
                logger.info(f"Cleaned up {len(expired_messages)} old messages from history")
    
    def _emit_event(self, event: CommunicationEvent, data: Dict[str, Any]) -> None:
        """Emit communication event to handlers."""
        for handler in self._event_handlers[event]:
            try:
                handler(event, data)
            except Exception as e:
                logger.error(f"Error in event handler for {event}: {e}")


# Global communication manager instance
_communication_manager: Optional[CommunicationManager] = None


def get_communication_manager() -> CommunicationManager:
    """Get the global communication manager instance."""
    global _communication_manager
    if _communication_manager is None:
        _communication_manager = CommunicationManager()
        _communication_manager.start()
    return _communication_manager


def create_communication_message(
    content: str,
    message_type: MessageType,
    sender: str,
    recipient: str,
    priority: MessagePriority = MessagePriority.NORMAL,
    routing_key: Optional[str] = None,
    reply_to: Optional[str] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    expires_in: Optional[float] = None
) -> CommunicationMessage:
    """Create a new communication message."""
    expires_at = None
    if expires_in:
        expires_at = datetime.now() + timedelta(seconds=expires_in)
    
    return CommunicationMessage(
        content=content,
        message_type=message_type,
        sender=sender,
        recipient=recipient,
        priority=priority,
        routing_key=routing_key,
        reply_to=reply_to,
        correlation_id=correlation_id,
        metadata=metadata or {},
        headers=headers or {},
        expires_at=expires_at
    )


def send_message_to_agent(
    agent_id: str,
    content: str,
    message_type: MessageType = MessageType.REQUEST,
    sender: str = "system",
    priority: MessagePriority = MessagePriority.NORMAL,
    **kwargs
) -> bool:
    """Convenience function to send a message to an agent."""
    message = create_communication_message(
        content=content,
        message_type=message_type,
        sender=sender,
        recipient=agent_id,
        priority=priority,
        **kwargs
    )
    
    comm_manager = get_communication_manager()
    return comm_manager.send_message(message)


def broadcast_to_agents(
    content: str,
    message_type: MessageType = MessageType.SYSTEM,
    sender: str = "system",
    target_agents: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, bool]:
    """Convenience function to broadcast a message to multiple agents."""
    message = create_communication_message(
        content=content,
        message_type=message_type,
        sender=sender,
        recipient="broadcast",  # Will be overridden in broadcast
        **kwargs
    )
    
    comm_manager = get_communication_manager()
    return comm_manager.broadcast_message(message, target_agents)
