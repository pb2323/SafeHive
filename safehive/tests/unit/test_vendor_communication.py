"""
Unit tests for the Vendor Communication Interface.
"""

import asyncio
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from safehive.agents.vendor_communication import (
    CommunicationIntent, MessageType, CommunicationStatus, CommunicationMessage,
    CommunicationSession, IntentClassification, VendorCommunicationInterface
)
from safehive.agents.order_models import Order, OrderItem, Vendor, OrderStatus, OrderType, PaymentStatus


class TestCommunicationMessage:
    """Test CommunicationMessage functionality."""
    
    def test_communication_message_creation(self):
        """Test CommunicationMessage creation."""
        message = CommunicationMessage(
            message_id="msg_001",
            sender_id="orchestrator_001",
            recipient_id="vendor_001",
            message_type=MessageType.REQUEST,
            intent=CommunicationIntent.ORDER_PLACEMENT,
            content="I'd like to place an order for pizza",
            response_required=True,
            priority=8
        )
        
        assert message.message_id == "msg_001"
        assert message.sender_id == "orchestrator_001"
        assert message.recipient_id == "vendor_001"
        assert message.message_type == MessageType.REQUEST
        assert message.intent == CommunicationIntent.ORDER_PLACEMENT
        assert message.content == "I'd like to place an order for pizza"
        assert message.response_required is True
        assert message.priority == 8
        assert isinstance(message.timestamp, datetime)
    
    def test_communication_message_serialization(self):
        """Test CommunicationMessage serialization."""
        message = CommunicationMessage(
            message_id="msg_001",
            sender_id="orchestrator_001",
            recipient_id="vendor_001",
            message_type=MessageType.REQUEST,
            intent=CommunicationIntent.ORDER_PLACEMENT,
            content="I'd like to place an order for pizza",
            metadata={"order_id": "order_001"},
            response_required=True,
            priority=8
        )
        
        data = message.to_dict()
        
        assert data["message_id"] == "msg_001"
        assert data["sender_id"] == "orchestrator_001"
        assert data["recipient_id"] == "vendor_001"
        assert data["message_type"] == "request"
        assert data["intent"] == "order_placement"
        assert data["content"] == "I'd like to place an order for pizza"
        assert data["metadata"]["order_id"] == "order_001"
        assert data["response_required"] is True
        assert data["priority"] == 8
        assert "timestamp" in data
    
    def test_communication_message_deserialization(self):
        """Test CommunicationMessage deserialization."""
        data = {
            "message_id": "msg_001",
            "sender_id": "orchestrator_001",
            "recipient_id": "vendor_001",
            "message_type": "request",
            "intent": "order_placement",
            "content": "I'd like to place an order for pizza",
            "timestamp": "2023-01-01T12:00:00",
            "metadata": {"order_id": "order_001"},
            "response_required": True,
            "priority": 8
        }
        
        message = CommunicationMessage.from_dict(data)
        
        assert message.message_id == "msg_001"
        assert message.sender_id == "orchestrator_001"
        assert message.recipient_id == "vendor_001"
        assert message.message_type == MessageType.REQUEST
        assert message.intent == CommunicationIntent.ORDER_PLACEMENT
        assert message.content == "I'd like to place an order for pizza"
        assert message.metadata["order_id"] == "order_001"
        assert message.response_required is True
        assert message.priority == 8


class TestCommunicationSession:
    """Test CommunicationSession functionality."""
    
    def test_communication_session_creation(self):
        """Test CommunicationSession creation."""
        session = CommunicationSession(
            session_id="session_001",
            orchestrator_id="orchestrator_001",
            vendor_id="vendor_001",
            status=CommunicationStatus.INITIATED,
            context={"order_id": "order_001"}
        )
        
        assert session.session_id == "session_001"
        assert session.orchestrator_id == "orchestrator_001"
        assert session.vendor_id == "vendor_001"
        assert session.status == CommunicationStatus.INITIATED
        assert session.context["order_id"] == "order_001"
        assert len(session.messages) == 0
        assert isinstance(session.created_at, datetime)
    
    def test_add_message_to_session(self):
        """Test adding messages to session."""
        session = CommunicationSession(
            session_id="session_001",
            orchestrator_id="orchestrator_001",
            vendor_id="vendor_001",
            status=CommunicationStatus.INITIATED
        )
        
        message = CommunicationMessage(
            message_id="msg_001",
            sender_id="orchestrator_001",
            recipient_id="vendor_001",
            message_type=MessageType.REQUEST,
            intent=CommunicationIntent.ORDER_PLACEMENT,
            content="I'd like to place an order"
        )
        
        session.add_message(message)
        
        assert len(session.messages) == 1
        assert session.messages[0] == message
        assert session.updated_at > session.created_at
    
    def test_get_messages_by_intent(self):
        """Test getting messages by intent."""
        session = CommunicationSession(
            session_id="session_001",
            orchestrator_id="orchestrator_001",
            vendor_id="vendor_001",
            status=CommunicationStatus.INITIATED
        )
        
        # Add messages with different intents
        message1 = CommunicationMessage(
            message_id="msg_001",
            sender_id="orchestrator_001",
            recipient_id="vendor_001",
            message_type=MessageType.REQUEST,
            intent=CommunicationIntent.ORDER_PLACEMENT,
            content="Place order"
        )
        
        message2 = CommunicationMessage(
            message_id="msg_002",
            sender_id="vendor_001",
            recipient_id="orchestrator_001",
            message_type=MessageType.RESPONSE,
            intent=CommunicationIntent.ORDER_CONFIRMATION,
            content="Order confirmed"
        )
        
        message3 = CommunicationMessage(
            message_id="msg_003",
            sender_id="orchestrator_001",
            recipient_id="vendor_001",
            message_type=MessageType.REQUEST,
            intent=CommunicationIntent.ORDER_PLACEMENT,
            content="Another order"
        )
        
        session.add_message(message1)
        session.add_message(message2)
        session.add_message(message3)
        
        # Test filtering by intent
        order_placement_messages = session.get_messages_by_intent(CommunicationIntent.ORDER_PLACEMENT)
        assert len(order_placement_messages) == 2
        assert message1 in order_placement_messages
        assert message3 in order_placement_messages
        
        order_confirmation_messages = session.get_messages_by_intent(CommunicationIntent.ORDER_CONFIRMATION)
        assert len(order_confirmation_messages) == 1
        assert message2 in order_confirmation_messages
    
    def test_get_last_message(self):
        """Test getting the last message."""
        session = CommunicationSession(
            session_id="session_001",
            orchestrator_id="orchestrator_001",
            vendor_id="vendor_001",
            status=CommunicationStatus.INITIATED
        )
        
        # Test with no messages
        assert session.get_last_message() is None
        
        # Add messages
        message1 = CommunicationMessage(
            message_id="msg_001",
            sender_id="orchestrator_001",
            recipient_id="vendor_001",
            message_type=MessageType.REQUEST,
            intent=CommunicationIntent.ORDER_PLACEMENT,
            content="First message"
        )
        
        message2 = CommunicationMessage(
            message_id="msg_002",
            sender_id="vendor_001",
            recipient_id="orchestrator_001",
            message_type=MessageType.RESPONSE,
            intent=CommunicationIntent.ORDER_CONFIRMATION,
            content="Second message"
        )
        
        session.add_message(message1)
        session.add_message(message2)
        
        assert session.get_last_message() == message2
    
    def test_communication_session_serialization(self):
        """Test CommunicationSession serialization."""
        session = CommunicationSession(
            session_id="session_001",
            orchestrator_id="orchestrator_001",
            vendor_id="vendor_001",
            status=CommunicationStatus.INITIATED,
            context={"order_id": "order_001"}
        )
        
        message = CommunicationMessage(
            message_id="msg_001",
            sender_id="orchestrator_001",
            recipient_id="vendor_001",
            message_type=MessageType.REQUEST,
            intent=CommunicationIntent.ORDER_PLACEMENT,
            content="Test message"
        )
        
        session.add_message(message)
        
        data = session.to_dict()
        
        assert data["session_id"] == "session_001"
        assert data["orchestrator_id"] == "orchestrator_001"
        assert data["vendor_id"] == "vendor_001"
        assert data["status"] == "initiated"
        assert data["context"]["order_id"] == "order_001"
        assert len(data["messages"]) == 1
        assert data["messages"][0]["message_id"] == "msg_001"
        assert "created_at" in data
        assert "updated_at" in data


class TestIntentClassification:
    """Test IntentClassification functionality."""
    
    def test_intent_classification_creation(self):
        """Test IntentClassification creation."""
        classification = IntentClassification(
            intent=CommunicationIntent.ORDER_PLACEMENT,
            confidence=0.85,
            extracted_entities={"item_name": "pizza", "quantity": 2},
            reasoning=["Detected order placement keywords", "Found quantity entity"]
        )
        
        assert classification.intent == CommunicationIntent.ORDER_PLACEMENT
        assert classification.confidence == 0.85
        assert classification.extracted_entities["item_name"] == "pizza"
        assert classification.extracted_entities["quantity"] == 2
        assert len(classification.reasoning) == 2
    
    def test_intent_classification_serialization(self):
        """Test IntentClassification serialization."""
        classification = IntentClassification(
            intent=CommunicationIntent.ORDER_PLACEMENT,
            confidence=0.85,
            extracted_entities={"item_name": "pizza", "quantity": 2},
            reasoning=["Detected order placement keywords"]
        )
        
        data = classification.to_dict()
        
        assert data["intent"] == "order_placement"
        assert data["confidence"] == 0.85
        assert data["extracted_entities"]["item_name"] == "pizza"
        assert data["extracted_entities"]["quantity"] == 2
        assert data["reasoning"][0] == "Detected order placement keywords"


class TestVendorCommunicationInterface:
    """Test VendorCommunicationInterface functionality."""
    
    def test_vendor_communication_interface_creation(self):
        """Test VendorCommunicationInterface creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            assert interface.storage_path == Path(temp_dir)
            assert len(interface.message_templates) > 0
            assert len(interface.conversation_flows) > 0
            assert len(interface.active_sessions) == 0
            assert len(interface.communication_history) == 0
    
    @pytest.mark.asyncio
    async def test_classify_intent(self):
        """Test intent classification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            # Test order placement intent
            classification = await interface.classify_intent("I'd like to place an order for pizza")
            assert classification.intent == CommunicationIntent.ORDER_PLACEMENT
            assert classification.confidence > 0.0
            
            # Test order cancellation intent
            classification = await interface.classify_intent("I need to cancel my order")
            assert classification.intent == CommunicationIntent.ORDER_CANCELLATION
            assert classification.confidence > 0.0
            
            # Test price negotiation intent
            classification = await interface.classify_intent("Do you have any discounts available?")
            assert classification.intent == CommunicationIntent.PRICE_NEGOTIATION
            assert classification.confidence > 0.0
            
            # Test availability check intent
            classification = await interface.classify_intent("Are you currently accepting orders?")
            assert classification.intent == CommunicationIntent.AVAILABILITY_CHECK
            assert classification.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_classify_intent_with_context(self):
        """Test intent classification with context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            context = {"order_id": "order_001", "vendor_id": "vendor_001"}
            classification = await interface.classify_intent(
                "I'd like to place an order", context
            )
            
            assert classification.intent == CommunicationIntent.ORDER_PLACEMENT
            assert len(classification.reasoning) > 0
    
    def test_extract_entities(self):
        """Test entity extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            # Test order placement entity extraction
            entities = interface._extract_entities(
                "I'd like to order x2 pizza for $25.50", 
                CommunicationIntent.ORDER_PLACEMENT
            )
            assert entities.get("quantity") == 2
            assert entities.get("item_name") == "pizza"
            
            # Test price negotiation entity extraction
            entities = interface._extract_entities(
                "The total is $45.00, do you have discounts?", 
                CommunicationIntent.PRICE_NEGOTIATION
            )
            assert entities.get("total_amount") == 45.0
            
            # Test special request entity extraction
            entities = interface._extract_entities(
                "I need vegetarian options", 
                CommunicationIntent.SPECIAL_REQUEST
            )
            assert entities.get("dietary_requirement") == "vegetarian"
    
    @pytest.mark.asyncio
    async def test_generate_message(self):
        """Test message generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="italian",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            # Test order placement message generation
            message = await interface.generate_message(
                CommunicationIntent.ORDER_PLACEMENT,
                vendor,
                item_name="pizza",
                quantity=2,
                total_amount=25.50
            )
            
            assert "pizza" in message.lower()
            assert "2" in message
            assert "25.50" in message
            assert vendor.name in message
    
    @pytest.mark.asyncio
    async def test_generate_message_with_order(self):
        """Test message generation with order context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="italian",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            items = [
                OrderItem(
                    item_id="item_001",
                    name="Margherita Pizza",
                    quantity=1,
                    unit_price=20.0,
                    total_price=20.0
                )
            ]
            
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=items,
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=23.0,
                delivery_address="123 Main St"
            )
            
            # Test order placement message generation with order
            message = await interface.generate_message(
                CommunicationIntent.ORDER_PLACEMENT,
                vendor,
                order=order
            )
            
            assert "Margherita Pizza" in message
            assert "23.0" in message
            assert vendor.name in message
    
    @pytest.mark.asyncio
    async def test_create_communication_session(self):
        """Test communication session creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            session = await interface.create_communication_session(
                "orchestrator_001",
                "vendor_001",
                "Hello, I'd like to place an order"
            )
            
            assert session.orchestrator_id == "orchestrator_001"
            assert session.vendor_id == "vendor_001"
            assert session.status == CommunicationStatus.INITIATED
            assert len(session.messages) == 1
            assert session.messages[0].content == "Hello, I'd like to place an order"
            assert session.session_id in interface.active_sessions
    
    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending messages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            # Create session
            session = await interface.create_communication_session(
                "orchestrator_001",
                "vendor_001"
            )
            
            # Send message
            message = await interface.send_message(
                session.session_id,
                "orchestrator_001",
                "vendor_001",
                "I'd like to place an order for pizza",
                CommunicationIntent.ORDER_PLACEMENT
            )
            
            assert message.sender_id == "orchestrator_001"
            assert message.recipient_id == "vendor_001"
            assert message.content == "I'd like to place an order for pizza"
            assert message.intent == CommunicationIntent.ORDER_PLACEMENT
            assert len(session.messages) == 1
            assert session.status == CommunicationStatus.IN_PROGRESS
    
    @pytest.mark.asyncio
    async def test_process_vendor_response(self):
        """Test processing vendor responses."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            # Create session and send initial message
            session = await interface.create_communication_session(
                "orchestrator_001",
                "vendor_001"
            )
            
            await interface.send_message(
                session.session_id,
                "orchestrator_001",
                "vendor_001",
                "I'd like to place an order",
                CommunicationIntent.ORDER_PLACEMENT
            )
            
            # Process vendor response
            response_data = await interface.process_vendor_response(
                session.session_id,
                "vendor_001",
                "Yes, I can confirm your order. It will be ready in 25 minutes."
            )
            
            assert "response_message" in response_data
            assert "classification" in response_data
            assert "next_action" in response_data
            assert response_data["response_message"].sender_id == "vendor_001"
            assert len(session.messages) == 2
    
    @pytest.mark.asyncio
    async def test_simulate_vendor_response(self):
        """Test vendor response simulation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="italian",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            message = CommunicationMessage(
                message_id="msg_001",
                sender_id="orchestrator_001",
                recipient_id="vendor_001",
                message_type=MessageType.REQUEST,
                intent=CommunicationIntent.ORDER_PLACEMENT,
                content="I'd like to place an order for pizza"
            )
            
            session = await interface.create_communication_session(
                "orchestrator_001",
                "vendor_001"
            )
            
            response = await interface.simulate_vendor_response(
                session.session_id,
                vendor,
                message
            )
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert "25" in response  # Should mention delivery time
    
    def test_get_session(self):
        """Test getting session by ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            # Test with non-existent session
            session = interface.get_session("non_existent")
            assert session is None
    
    def test_get_active_sessions(self):
        """Test getting active sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            # Test with no active sessions
            sessions = interface.get_active_sessions()
            assert len(sessions) == 0
    
    def test_get_communication_statistics(self):
        """Test getting communication statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            stats = interface.get_communication_statistics()
            
            assert "total_sessions" in stats
            assert "active_sessions" in stats
            assert "completed_sessions" in stats
            assert stats["total_sessions"] == 0
            assert stats["active_sessions"] == 0
            assert stats["completed_sessions"] == 0


class TestVendorCommunicationIntegration:
    """Integration tests for VendorCommunicationInterface."""
    
    @pytest.mark.asyncio
    async def test_complete_communication_workflow(self):
        """Test complete communication workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            interface = VendorCommunicationInterface(temp_dir)
            
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="italian",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            # Step 1: Create communication session
            session = await interface.create_communication_session(
                "orchestrator_001",
                "vendor_001",
                "Hello, I'd like to place an order"
            )
            
            assert session.status == CommunicationStatus.INITIATED
            
            # Step 2: Send order inquiry
            inquiry_message = await interface.send_message(
                session.session_id,
                "orchestrator_001",
                "vendor_001",
                "Do you have pizza available?",
                CommunicationIntent.ORDER_INQUIRY
            )
            
            assert inquiry_message.intent == CommunicationIntent.ORDER_INQUIRY
            
            # Step 3: Simulate vendor response
            vendor_response = await interface.simulate_vendor_response(
                session.session_id,
                vendor,
                inquiry_message
            )
            
            assert len(vendor_response) > 0
            
            # Step 4: Process vendor response
            response_data = await interface.process_vendor_response(
                session.session_id,
                "vendor_001",
                vendor_response
            )
            
            assert "response_message" in response_data
            assert "classification" in response_data
            assert "next_action" in response_data
            
            # Step 5: Check session status
            updated_session = interface.get_session(session.session_id)
            assert updated_session is not None
            assert len(updated_session.messages) == 3  # Initial + inquiry + response
    
    @pytest.mark.asyncio
    async def test_persistence_and_recovery(self):
        """Test persistence and recovery of communication sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first interface instance
            interface1 = VendorCommunicationInterface(temp_dir)
            
            # Create and complete a session
            session = await interface1.create_communication_session(
                "orchestrator_001",
                "vendor_001",
                "Test message"
            )
            
            await interface1.send_message(
                session.session_id,
                "orchestrator_001",
                "vendor_001",
                "Test content",
                CommunicationIntent.GENERAL_INQUIRY
            )
            
            # Complete the session
            session.status = CommunicationStatus.COMPLETED
            interface1.communication_history.append(session)
            interface1._save_communication_history()
            
            # Create second interface instance (should load history)
            interface2 = VendorCommunicationInterface(temp_dir)
            
            assert len(interface2.communication_history) == 1
            assert interface2.communication_history[0].orchestrator_id == "orchestrator_001"
            assert interface2.communication_history[0].vendor_id == "vendor_001"
            assert len(interface2.communication_history[0].messages) == 2
