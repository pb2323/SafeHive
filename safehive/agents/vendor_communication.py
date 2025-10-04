"""
Vendor Communication Interface with Natural Language Processing

This module implements natural language communication capabilities between
the orchestrator and vendor agents in the SafeHive AI Security Sandbox.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from .order_models import Order, OrderItem, Vendor, OrderStatus, OrderType
from .user_twin import UserTwinAgent, PreferenceCategory
from ..utils.logger import get_logger
from ..utils.metrics import record_metric, MetricType

logger = get_logger(__name__)


class CommunicationIntent(Enum):
    """Types of communication intents."""
    ORDER_INQUIRY = "order_inquiry"
    ORDER_PLACEMENT = "order_placement"
    ORDER_CONFIRMATION = "order_confirmation"
    ORDER_MODIFICATION = "order_modification"
    ORDER_CANCELLATION = "order_cancellation"
    PRICE_NEGOTIATION = "price_negotiation"
    AVAILABILITY_CHECK = "availability_check"
    DELIVERY_INQUIRY = "delivery_inquiry"
    PAYMENT_INQUIRY = "payment_inquiry"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    GENERAL_INQUIRY = "general_inquiry"
    MENU_REQUEST = "menu_request"
    SPECIAL_REQUEST = "special_request"
    UNKNOWN = "unknown"


class MessageType(Enum):
    """Types of messages in communication."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    CONFIRMATION = "confirmation"
    ERROR = "error"
    GREETING = "greeting"
    GOODBYE = "goodbye"


class CommunicationStatus(Enum):
    """Status of communication interactions."""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class CommunicationMessage:
    """Represents a message in vendor communication."""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    intent: CommunicationIntent
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_required: bool = False
    priority: int = 5  # 1-10 scale, 10 being highest
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "intent": self.intent.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "response_required": self.response_required,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommunicationMessage":
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            message_type=MessageType(data["message_type"]),
            intent=CommunicationIntent(data["intent"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            response_required=data.get("response_required", False),
            priority=data.get("priority", 5)
        )


@dataclass
class CommunicationSession:
    """Represents a communication session between orchestrator and vendor."""
    session_id: str
    orchestrator_id: str
    vendor_id: str
    status: CommunicationStatus
    messages: List[CommunicationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: CommunicationMessage) -> None:
        """Add a message to the session."""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_messages_by_intent(self, intent: CommunicationIntent) -> List[CommunicationMessage]:
        """Get messages by intent."""
        return [msg for msg in self.messages if msg.intent == intent]
    
    def get_last_message(self) -> Optional[CommunicationMessage]:
        """Get the last message in the session."""
        return self.messages[-1] if self.messages else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "orchestrator_id": self.orchestrator_id,
            "vendor_id": self.vendor_id,
            "status": self.status.value,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "context": self.context,
            "metadata": self.metadata
        }


@dataclass
class IntentClassification:
    """Result of intent classification."""
    intent: CommunicationIntent
    confidence: float
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "extracted_entities": self.extracted_entities,
            "reasoning": self.reasoning
        }


class MessageTemplate:
    """Template for generating consistent messages."""
    
    def __init__(self, template: str, variables: List[str]):
        self.template = template
        self.variables = variables
    
    def generate(self, **kwargs) -> str:
        """Generate message from template."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing variable {e} in template: {self.template}")
            return self.template


class VendorCommunicationInterface:
    """Natural language communication interface for vendor interactions."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_vendor_communications"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Communication components
        self.active_sessions: Dict[str, CommunicationSession] = {}
        self.message_templates: Dict[CommunicationIntent, MessageTemplate] = {}
        self.conversation_flows: Dict[CommunicationIntent, List[str]] = {}
        
        # Initialize templates and flows
        self._initialize_message_templates()
        self._initialize_conversation_flows()
        
        # Communication history
        self.communication_history: List[CommunicationSession] = []
        self._load_communication_history()
        
        logger.info("Vendor Communication Interface initialized")
    
    def _initialize_message_templates(self) -> None:
        """Initialize message templates for different intents."""
        self.message_templates = {
            CommunicationIntent.ORDER_INQUIRY: MessageTemplate(
                "Hello {vendor_name}! I'd like to inquire about placing an order. Do you have {item_name} available?",
                ["vendor_name", "item_name"]
            ),
            CommunicationIntent.ORDER_PLACEMENT: MessageTemplate(
                "Hello {vendor_name}! I'd like to place an order for {item_name} x{quantity}. The total should be ${total_amount:.2f}. Can you confirm this order?",
                ["vendor_name", "item_name", "quantity", "total_amount"]
            ),
            CommunicationIntent.ORDER_CONFIRMATION: MessageTemplate(
                "Great! I confirm the order for {item_name} x{quantity} for ${total_amount:.2f}. When can I expect delivery?",
                ["item_name", "quantity", "total_amount"]
            ),
            CommunicationIntent.ORDER_MODIFICATION: MessageTemplate(
                "I need to modify my order. Can you change {item_name} quantity from {old_quantity} to {new_quantity}?",
                ["item_name", "old_quantity", "new_quantity"]
            ),
            CommunicationIntent.ORDER_CANCELLATION: MessageTemplate(
                "I need to cancel my order for {item_name}. What's your cancellation policy?",
                ["item_name"]
            ),
            CommunicationIntent.PRICE_NEGOTIATION: MessageTemplate(
                "The total for my order is ${total_amount:.2f}. Do you have any discounts or special offers available?",
                ["total_amount"]
            ),
            CommunicationIntent.AVAILABILITY_CHECK: MessageTemplate(
                "Hi {vendor_name}! Are you currently accepting orders? What's your estimated delivery time?",
                ["vendor_name"]
            ),
            CommunicationIntent.DELIVERY_INQUIRY: MessageTemplate(
                "What's the status of my order? When can I expect delivery to {delivery_address}?",
                ["delivery_address"]
            ),
            CommunicationIntent.PAYMENT_INQUIRY: MessageTemplate(
                "What payment methods do you accept? Is payment required upfront or upon delivery?",
                []
            ),
            CommunicationIntent.COMPLAINT: MessageTemplate(
                "I have a concern about my recent order. {complaint_description}. Can you help resolve this?",
                ["complaint_description"]
            ),
            CommunicationIntent.COMPLIMENT: MessageTemplate(
                "Thank you for the excellent service! The {item_name} was delicious and delivered on time.",
                ["item_name"]
            ),
            CommunicationIntent.GENERAL_INQUIRY: MessageTemplate(
                "Hello {vendor_name}! I have a question about your services. {inquiry_description}",
                ["vendor_name", "inquiry_description"]
            ),
            CommunicationIntent.MENU_REQUEST: MessageTemplate(
                "Could you please share your current menu and prices? I'm looking for {cuisine_type} options.",
                ["cuisine_type"]
            ),
            CommunicationIntent.SPECIAL_REQUEST: MessageTemplate(
                "I have a special dietary requirement: {dietary_requirement}. Can you accommodate this in my order?",
                ["dietary_requirement"]
            )
        }
    
    def _initialize_conversation_flows(self) -> None:
        """Initialize conversation flows for different intents."""
        self.conversation_flows = {
            CommunicationIntent.ORDER_PLACEMENT: [
                "greeting",
                "order_inquiry",
                "availability_check",
                "price_confirmation",
                "order_confirmation",
                "payment_arrangement",
                "delivery_confirmation"
            ],
            CommunicationIntent.ORDER_MODIFICATION: [
                "order_identification",
                "modification_request",
                "availability_check",
                "price_adjustment",
                "confirmation",
                "delivery_update"
            ],
            CommunicationIntent.ORDER_CANCELLATION: [
                "order_identification",
                "cancellation_request",
                "policy_explanation",
                "refund_arrangement",
                "confirmation"
            ],
            CommunicationIntent.COMPLAINT: [
                "greeting",
                "complaint_description",
                "apology",
                "resolution_offer",
                "follow_up_arrangement"
            ]
        }
    
    def _load_communication_history(self) -> None:
        """Load communication history from storage."""
        history_file = self.storage_path / "communication_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for session_data in data:
                        session = self._reconstruct_session(session_data)
                        if session:
                            self.communication_history.append(session)
                logger.info(f"Loaded {len(self.communication_history)} communication sessions")
            except Exception as e:
                logger.error(f"Failed to load communication history: {e}")
    
    def _save_communication_history(self) -> None:
        """Save communication history to storage."""
        history_file = self.storage_path / "communication_history.json"
        try:
            data = [session.to_dict() for session in self.communication_history]
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved communication history")
        except Exception as e:
            logger.error(f"Failed to save communication history: {e}")
    
    def _reconstruct_session(self, data: Dict[str, Any]) -> Optional[CommunicationSession]:
        """Reconstruct CommunicationSession from stored data."""
        try:
            messages = [CommunicationMessage.from_dict(msg) for msg in data.get("messages", [])]
            return CommunicationSession(
                session_id=data["session_id"],
                orchestrator_id=data["orchestrator_id"],
                vendor_id=data["vendor_id"],
                status=CommunicationStatus(data["status"]),
                messages=messages,
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                context=data.get("context", {}),
                metadata=data.get("metadata", {})
            )
        except Exception as e:
            logger.error(f"Failed to reconstruct session: {e}")
            return None
    
    async def classify_intent(self, message: str, context: Optional[Dict[str, Any]] = None) -> IntentClassification:
        """Classify the intent of a natural language message."""
        message_lower = message.lower()
        reasoning = []
        confidence = 0.0
        intent = CommunicationIntent.UNKNOWN
        extracted_entities = {}
        
        # Simple keyword-based classification (in a real system, this would use NLP/ML)
        
        # Check for specific phrases first (more specific)
        if "accepting orders" in message_lower:
            intent = CommunicationIntent.AVAILABILITY_CHECK
            confidence = 0.9
            reasoning.append("Detected availability check phrase")
        
        # Order-related intents
        elif any(word in message_lower for word in ["cancel", "stop"]) and any(word in message_lower for word in ["order", "order"]):
            intent = CommunicationIntent.ORDER_CANCELLATION
            confidence = 0.8
            reasoning.append("Detected order cancellation keywords")
        elif any(word in message_lower for word in ["modify", "change", "update", "edit"]) and any(word in message_lower for word in ["order", "order"]):
            intent = CommunicationIntent.ORDER_MODIFICATION
            confidence = 0.8
            reasoning.append("Detected order modification keywords")
        elif any(word in message_lower for word in ["confirm", "confirm"]) and any(word in message_lower for word in ["order", "order"]):
            intent = CommunicationIntent.ORDER_CONFIRMATION
            confidence = 0.9
            reasoning.append("Detected order confirmation keywords")
        elif any(word in message_lower for word in ["order", "place", "buy", "purchase"]):
            intent = CommunicationIntent.ORDER_PLACEMENT
            confidence = 0.7
            reasoning.append("Detected order placement keywords")
        
        # Price and negotiation
        elif any(word in message_lower for word in ["price", "cost", "discount", "offer", "deal"]):
            intent = CommunicationIntent.PRICE_NEGOTIATION
            confidence = 0.8
            reasoning.append("Detected price negotiation keywords")
        
        # Availability and delivery
        elif any(word in message_lower for word in ["available", "open", "accepting", "taking"]):
            intent = CommunicationIntent.AVAILABILITY_CHECK
            confidence = 0.8
            reasoning.append("Detected availability check keywords")
        elif any(word in message_lower for word in ["delivery", "deliver", "arrive", "when"]):
            intent = CommunicationIntent.DELIVERY_INQUIRY
            confidence = 0.7
            reasoning.append("Detected delivery inquiry keywords")
        
        # Payment
        elif any(word in message_lower for word in ["payment", "pay", "card", "cash", "method"]):
            intent = CommunicationIntent.PAYMENT_INQUIRY
            confidence = 0.8
            reasoning.append("Detected payment inquiry keywords")
        
        # Feedback
        elif any(word in message_lower for word in ["complaint", "problem", "issue", "wrong", "bad"]):
            intent = CommunicationIntent.COMPLAINT
            confidence = 0.8
            reasoning.append("Detected complaint keywords")
        elif any(word in message_lower for word in ["thank", "great", "excellent", "good", "love"]):
            intent = CommunicationIntent.COMPLIMENT
            confidence = 0.7
            reasoning.append("Detected compliment keywords")
        
        # Menu and special requests
        elif any(word in message_lower for word in ["menu", "list", "options", "items"]):
            intent = CommunicationIntent.MENU_REQUEST
            confidence = 0.8
            reasoning.append("Detected menu request keywords")
        elif any(word in message_lower for word in ["special", "dietary", "allergy", "vegetarian", "vegan"]):
            intent = CommunicationIntent.SPECIAL_REQUEST
            confidence = 0.7
            reasoning.append("Detected special request keywords")
        
        # General inquiry
        elif any(word in message_lower for word in ["hello", "hi", "help", "question", "ask"]):
            intent = CommunicationIntent.GENERAL_INQUIRY
            confidence = 0.6
            reasoning.append("Detected general inquiry keywords")
        
        # Extract entities based on intent
        extracted_entities = self._extract_entities(message, intent)
        
        return IntentClassification(
            intent=intent,
            confidence=confidence,
            extracted_entities=extracted_entities,
            reasoning=reasoning
        )
    
    def _extract_entities(self, message: str, intent: CommunicationIntent) -> Dict[str, Any]:
        """Extract entities from message based on intent."""
        entities = {}
        message_lower = message.lower()
        
        if intent == CommunicationIntent.ORDER_PLACEMENT:
            # Extract quantities
            import re
            quantity_matches = re.findall(r'x(\d+)', message_lower)
            if quantity_matches:
                entities["quantity"] = int(quantity_matches[0])
            
            # Extract item names (simple approach)
            if "pizza" in message_lower:
                entities["item_name"] = "pizza"
            elif "burger" in message_lower:
                entities["item_name"] = "burger"
            elif "salad" in message_lower:
                entities["item_name"] = "salad"
            elif "pasta" in message_lower:
                entities["item_name"] = "pasta"
        
        elif intent == CommunicationIntent.PRICE_NEGOTIATION:
            # Extract price amounts
            import re
            price_matches = re.findall(r'\$(\d+\.?\d*)', message)
            if price_matches:
                entities["total_amount"] = float(price_matches[0])
        
        elif intent == CommunicationIntent.DELIVERY_INQUIRY:
            # Extract addresses (simple approach)
            if "123" in message or "main" in message_lower or "street" in message_lower:
                entities["delivery_address"] = "123 Main Street"
        
        elif intent == CommunicationIntent.SPECIAL_REQUEST:
            if "vegetarian" in message_lower:
                entities["dietary_requirement"] = "vegetarian"
            elif "vegan" in message_lower:
                entities["dietary_requirement"] = "vegan"
            elif "gluten" in message_lower:
                entities["dietary_requirement"] = "gluten-free"
        
        return entities
    
    async def generate_message(self, intent: CommunicationIntent, 
                             vendor: Vendor, order: Optional[Order] = None,
                             **kwargs) -> str:
        """Generate a natural language message for the given intent."""
        if intent not in self.message_templates:
            return f"Hello {vendor.name}! I have a {intent.value} request."
        
        template = self.message_templates[intent]
        
        # Prepare template variables
        template_vars = {
            "vendor_name": vendor.name,
            "vendor_id": vendor.vendor_id,
            "cuisine_type": vendor.cuisine_type,
            **kwargs
        }
        
        # Add order-specific variables if order is provided
        if order:
            template_vars.update({
                "order_id": order.order_id,
                "total_amount": order.total_amount,
                "delivery_address": order.delivery_address or "your address",
                "special_instructions": order.special_instructions
            })
            
            # Add item information if available
            if order.items:
                first_item = order.items[0]
                template_vars.update({
                    "item_name": first_item.name,
                    "quantity": first_item.quantity,
                    "unit_price": first_item.unit_price
                })
        
        return template.generate(**template_vars)
    
    async def create_communication_session(self, orchestrator_id: str, vendor_id: str,
                                         initial_message: Optional[str] = None) -> CommunicationSession:
        """Create a new communication session."""
        session_id = f"session_{int(time.time())}_{orchestrator_id}_{vendor_id}"
        
        session = CommunicationSession(
            session_id=session_id,
            orchestrator_id=orchestrator_id,
            vendor_id=vendor_id,
            status=CommunicationStatus.INITIATED
        )
        
        # Add initial message if provided
        if initial_message:
            message = CommunicationMessage(
                message_id=f"msg_{int(time.time())}",
                sender_id=orchestrator_id,
                recipient_id=vendor_id,
                message_type=MessageType.GREETING,
                intent=CommunicationIntent.GENERAL_INQUIRY,
                content=initial_message
            )
            session.add_message(message)
        
        self.active_sessions[session_id] = session
        
        logger.info(f"Created communication session: {session_id}")
        return session
    
    async def send_message(self, session_id: str, sender_id: str, recipient_id: str,
                          content: str, intent: Optional[CommunicationIntent] = None,
                          message_type: MessageType = MessageType.REQUEST) -> CommunicationMessage:
        """Send a message in an existing communication session."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Classify intent if not provided
        if intent is None:
            classification = await self.classify_intent(content)
            intent = classification.intent
        
        # Create message
        message = CommunicationMessage(
            message_id=f"msg_{int(time.time())}_{sender_id}",
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            intent=intent,
            content=content
        )
        
        # Add to session
        session.add_message(message)
        session.status = CommunicationStatus.IN_PROGRESS
        
        # Record metrics
        record_metric("vendor_communication.message_sent", 1, MetricType.COUNTER, {
            "session_id": session_id,
            "intent": intent.value,
            "sender_id": sender_id,
            "recipient_id": recipient_id
        })
        
        logger.info(f"Sent message in session {session_id}: {intent.value}")
        return message
    
    async def process_vendor_response(self, session_id: str, vendor_id: str,
                                    response_content: str) -> Dict[str, Any]:
        """Process a response from a vendor."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Classify the vendor's response intent
        classification = await self.classify_intent(response_content)
        
        # Create vendor response message
        response_message = CommunicationMessage(
            message_id=f"msg_{int(time.time())}_{vendor_id}",
            sender_id=vendor_id,
            recipient_id=session.orchestrator_id,
            message_type=MessageType.RESPONSE,
            intent=classification.intent,
            content=response_content
        )
        
        session.add_message(response_message)
        
        # Determine next action based on response
        next_action = self._determine_next_action(session, response_message)
        
        # Update session status
        if next_action.get("session_complete", False):
            session.status = CommunicationStatus.COMPLETED
            self.communication_history.append(session)
            del self.active_sessions[session_id]
            self._save_communication_history()
        
        # Record metrics
        record_metric("vendor_communication.response_received", 1, MetricType.COUNTER, {
            "session_id": session_id,
            "vendor_id": vendor_id,
            "intent": classification.intent.value,
            "confidence": classification.confidence
        })
        
        return {
            "response_message": response_message,
            "classification": classification,
            "next_action": next_action,
            "session_status": session.status
        }
    
    def _determine_next_action(self, session: CommunicationSession, 
                             last_message: CommunicationMessage) -> Dict[str, Any]:
        """Determine the next action based on the conversation flow."""
        # Get the conversation flow for the primary intent
        primary_intent = session.messages[0].intent if session.messages else CommunicationIntent.GENERAL_INQUIRY
        flow = self.conversation_flows.get(primary_intent, [])
        
        # Simple flow management (in a real system, this would be more sophisticated)
        if last_message.intent == CommunicationIntent.ORDER_CONFIRMATION:
            return {
                "action": "complete_order",
                "session_complete": True,
                "next_message_type": None
            }
        elif last_message.intent == CommunicationIntent.ORDER_CANCELLATION:
            return {
                "action": "process_cancellation",
                "session_complete": True,
                "next_message_type": None
            }
        elif "problem" in last_message.content.lower() or "issue" in last_message.content.lower():
            return {
                "action": "escalate_issue",
                "session_complete": False,
                "next_message_type": MessageType.REQUEST
            }
        else:
            return {
                "action": "continue_conversation",
                "session_complete": False,
                "next_message_type": MessageType.RESPONSE
            }
    
    async def simulate_vendor_response(self, session_id: str, vendor: Vendor,
                                     last_message: CommunicationMessage) -> str:
        """Simulate a vendor response for testing purposes."""
        intent = last_message.intent
        content = last_message.content.lower()
        
        # Generate appropriate vendor response based on intent
        if intent == CommunicationIntent.ORDER_INQUIRY:
            return f"Hello! Yes, we have that available. Our current delivery time is {vendor.delivery_time_minutes} minutes."
        
        elif intent == CommunicationIntent.ORDER_PLACEMENT:
            return f"Perfect! I can confirm your order. The total is correct at ${last_message.metadata.get('total_amount', 'N/A')}. We'll have it ready in {vendor.delivery_time_minutes} minutes."
        
        elif intent == CommunicationIntent.PRICE_NEGOTIATION:
            return f"Thank you for asking! We don't have discounts right now, but we offer free delivery on orders over $30."
        
        elif intent == CommunicationIntent.AVAILABILITY_CHECK:
            status = "open and accepting orders" if vendor.is_available else "currently closed"
            return f"We are {status}. Our delivery time is {vendor.delivery_time_minutes} minutes."
        
        elif intent == CommunicationIntent.DELIVERY_INQUIRY:
            return f"Your order is being prepared and should arrive within {vendor.delivery_time_minutes} minutes."
        
        elif intent == CommunicationIntent.PAYMENT_INQUIRY:
            return f"We accept cash, credit cards, and digital payments. Payment is due upon delivery."
        
        elif intent == CommunicationIntent.COMPLAINT:
            return f"I apologize for the inconvenience. Let me help resolve this issue for you. Could you provide more details?"
        
        elif intent == CommunicationIntent.COMPLIMENT:
            return f"Thank you so much for your kind words! We appreciate your business and look forward to serving you again."
        
        elif intent == CommunicationIntent.MENU_REQUEST:
            return f"Our current {vendor.cuisine_type} menu includes our specialties: {', '.join(vendor.specialties)}. Would you like to see our full menu?"
        
        elif intent == CommunicationIntent.SPECIAL_REQUEST:
            return f"We can accommodate special dietary requirements. Please let us know the specific needs for your order."
        
        else:
            return f"Thank you for your message. How can I help you today?"
    
    def get_session(self, session_id: str) -> Optional[CommunicationSession]:
        """Get a communication session by ID."""
        return self.active_sessions.get(session_id)
    
    def get_active_sessions(self) -> List[CommunicationSession]:
        """Get all active communication sessions."""
        return list(self.active_sessions.values())
    
    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        total_sessions = len(self.communication_history) + len(self.active_sessions)
        
        if total_sessions == 0:
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "completed_sessions": 0,
                "status_counts": {},
                "intent_counts": {}
            }
        
        # Count sessions by status
        status_counts = {}
        for session in self.communication_history:
            status = session.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for session in self.active_sessions.values():
            status = session.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count messages by intent
        intent_counts = {}
        for session in self.communication_history + list(self.active_sessions.values()):
            for message in session.messages:
                intent = message.intent.value
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.communication_history),
            "status_counts": status_counts,
            "intent_counts": intent_counts
        }
