"""
Unit tests for the OrchestratorAgent.
"""

import asyncio
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import pytest

from safehive.agents.orchestrator import (
    OrderStatus, OrderType, PaymentStatus, OrderItem, Vendor, Order,
    OrderManager, VendorManager, OrchestratorAgent, create_orchestrator_agent
)
from safehive.agents.user_twin import UserTwinAgent, PreferenceCategory
from safehive.agents.configuration import AgentConfiguration, PersonalityProfile, PersonalityTrait
from safehive.models.agent_models import AgentType, AgentState


class TestOrderItem:
    """Test OrderItem functionality."""
    
    def test_order_item_creation(self):
        """Test OrderItem creation."""
        item = OrderItem(
            item_id="item_001",
            name="Margherita Pizza",
            quantity=2,
            unit_price=15.99,
            total_price=31.98,
            special_instructions="Extra cheese",
            dietary_requirements=["vegetarian"],
            allergens=["gluten", "dairy"]
        )
        
        assert item.item_id == "item_001"
        assert item.name == "Margherita Pizza"
        assert item.quantity == 2
        assert item.unit_price == 15.99
        assert item.total_price == 31.98
        assert item.special_instructions == "Extra cheese"
        assert "vegetarian" in item.dietary_requirements
        assert "gluten" in item.allergens
    
    def test_order_item_serialization(self):
        """Test OrderItem serialization."""
        item = OrderItem(
            item_id="item_001",
            name="Margherita Pizza",
            quantity=2,
            unit_price=15.99,
            total_price=31.98,
            special_instructions="Extra cheese",
            dietary_requirements=["vegetarian"],
            allergens=["gluten", "dairy"]
        )
        
        data = item.to_dict()
        
        assert data["item_id"] == "item_001"
        assert data["name"] == "Margherita Pizza"
        assert data["quantity"] == 2
        assert data["unit_price"] == 15.99
        assert data["total_price"] == 31.98
        assert data["special_instructions"] == "Extra cheese"
        assert "vegetarian" in data["dietary_requirements"]
        assert "gluten" in data["allergens"]
    
    def test_order_item_deserialization(self):
        """Test OrderItem deserialization."""
        data = {
            "item_id": "item_001",
            "name": "Margherita Pizza",
            "quantity": 2,
            "unit_price": 15.99,
            "total_price": 31.98,
            "special_instructions": "Extra cheese",
            "dietary_requirements": ["vegetarian"],
            "allergens": ["gluten", "dairy"]
        }
        
        item = OrderItem.from_dict(data)
        
        assert item.item_id == "item_001"
        assert item.name == "Margherita Pizza"
        assert item.quantity == 2
        assert item.unit_price == 15.99
        assert item.total_price == 31.98
        assert item.special_instructions == "Extra cheese"
        assert "vegetarian" in item.dietary_requirements
        assert "gluten" in item.allergens


class TestVendor:
    """Test Vendor functionality."""
    
    def test_vendor_creation(self):
        """Test Vendor creation."""
        vendor = Vendor(
            vendor_id="vendor_001",
            name="Mario's Italian Bistro",
            cuisine_type="italian",
            rating=4.8,
            delivery_time_minutes=25,
            minimum_order=15.0,
            delivery_fee=3.0,
            is_available=True,
            specialties=["pizza", "pasta"],
            contact_info={"phone": "+1-555-0123", "email": "orders@marios.com"}
        )
        
        assert vendor.vendor_id == "vendor_001"
        assert vendor.name == "Mario's Italian Bistro"
        assert vendor.cuisine_type == "italian"
        assert vendor.rating == 4.8
        assert vendor.delivery_time_minutes == 25
        assert vendor.minimum_order == 15.0
        assert vendor.delivery_fee == 3.0
        assert vendor.is_available is True
        assert "pizza" in vendor.specialties
        assert vendor.contact_info["phone"] == "+1-555-0123"
    
    def test_vendor_serialization(self):
        """Test Vendor serialization."""
        vendor = Vendor(
            vendor_id="vendor_001",
            name="Mario's Italian Bistro",
            cuisine_type="italian",
            rating=4.8,
            delivery_time_minutes=25,
            minimum_order=15.0,
            delivery_fee=3.0,
            specialties=["pizza", "pasta"],
            contact_info={"phone": "+1-555-0123", "email": "orders@marios.com"}
        )
        
        data = vendor.to_dict()
        
        assert data["vendor_id"] == "vendor_001"
        assert data["name"] == "Mario's Italian Bistro"
        assert data["cuisine_type"] == "italian"
        assert data["rating"] == 4.8
        assert data["delivery_time_minutes"] == 25
        assert data["minimum_order"] == 15.0
        assert data["delivery_fee"] == 3.0
        assert data["is_available"] is True
        assert "pizza" in data["specialties"]
        assert data["contact_info"]["phone"] == "+1-555-0123"
    
    def test_vendor_deserialization(self):
        """Test Vendor deserialization."""
        data = {
            "vendor_id": "vendor_001",
            "name": "Mario's Italian Bistro",
            "cuisine_type": "italian",
            "rating": 4.8,
            "delivery_time_minutes": 25,
            "minimum_order": 15.0,
            "delivery_fee": 3.0,
            "is_available": True,
            "specialties": ["pizza", "pasta"],
            "contact_info": {"phone": "+1-555-0123", "email": "orders@marios.com"}
        }
        
        vendor = Vendor.from_dict(data)
        
        assert vendor.vendor_id == "vendor_001"
        assert vendor.name == "Mario's Italian Bistro"
        assert vendor.cuisine_type == "italian"
        assert vendor.rating == 4.8
        assert vendor.delivery_time_minutes == 25
        assert vendor.minimum_order == 15.0
        assert vendor.delivery_fee == 3.0
        assert vendor.is_available is True
        assert "pizza" in vendor.specialties
        assert vendor.contact_info["phone"] == "+1-555-0123"


class TestOrder:
    """Test Order functionality."""
    
    def test_order_creation(self):
        """Test Order creation."""
        vendor = Vendor(
            vendor_id="vendor_001",
            name="Mario's Italian Bistro",
            cuisine_type="italian",
            rating=4.8,
            delivery_time_minutes=25,
            minimum_order=15.0,
            delivery_fee=3.0
        )
        
        item = OrderItem(
            item_id="item_001",
            name="Margherita Pizza",
            quantity=1,
            unit_price=15.99,
            total_price=15.99
        )
        
        order = Order(
            order_id="order_001",
            user_id="user_001",
            vendor=vendor,
            items=[item],
            order_type=OrderType.DELIVERY,
            status=OrderStatus.PENDING,
            payment_status=PaymentStatus.PENDING,
            total_amount=18.99,
            delivery_address="123 Main St"
        )
        
        assert order.order_id == "order_001"
        assert order.user_id == "user_001"
        assert order.vendor.name == "Mario's Italian Bistro"
        assert len(order.items) == 1
        assert order.items[0].name == "Margherita Pizza"
        assert order.order_type == OrderType.DELIVERY
        assert order.status == OrderStatus.PENDING
        assert order.payment_status == PaymentStatus.PENDING
        assert order.total_amount == 18.99
        assert order.delivery_address == "123 Main St"
    
    def test_order_serialization(self):
        """Test Order serialization."""
        vendor = Vendor(
            vendor_id="vendor_001",
            name="Mario's Italian Bistro",
            cuisine_type="italian",
            rating=4.8,
            delivery_time_minutes=25,
            minimum_order=15.0,
            delivery_fee=3.0
        )
        
        item = OrderItem(
            item_id="item_001",
            name="Margherita Pizza",
            quantity=1,
            unit_price=15.99,
            total_price=15.99
        )
        
        order = Order(
            order_id="order_001",
            user_id="user_001",
            vendor=vendor,
            items=[item],
            order_type=OrderType.DELIVERY,
            status=OrderStatus.PENDING,
            payment_status=PaymentStatus.PENDING,
            total_amount=18.99
        )
        
        data = order.to_dict()
        
        assert data["order_id"] == "order_001"
        assert data["user_id"] == "user_001"
        assert data["vendor"]["name"] == "Mario's Italian Bistro"
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "Margherita Pizza"
        assert data["order_type"] == "delivery"
        assert data["status"] == "pending"
        assert data["payment_status"] == "pending"
        assert data["total_amount"] == 18.99


class TestOrderManager:
    """Test OrderManager functionality."""
    
    def test_order_manager_creation(self):
        """Test OrderManager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OrderManager(temp_dir)
            assert manager.storage_path == Path(temp_dir)
            assert len(manager._orders) == 0
            assert len(manager._order_history) == 0
    
    def test_create_order(self):
        """Test order creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OrderManager(temp_dir)
            
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.0,
                delivery_time_minutes=20,
                minimum_order=10.0,
                delivery_fee=2.0
            )
            
            item = OrderItem(
                item_id="item_001",
                name="Test Item",
                quantity=1,
                unit_price=12.0,
                total_price=12.0
            )
            
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=[item],
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=14.0
            )
            
            success = manager.create_order(order)
            assert success is True
            assert "order_001" in manager._orders
            assert len(manager._order_history) == 1
            assert manager._order_history[0]["action"] == "create"
    
    def test_get_order(self):
        """Test getting order by ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OrderManager(temp_dir)
            
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.0,
                delivery_time_minutes=20,
                minimum_order=10.0,
                delivery_fee=2.0
            )
            
            item = OrderItem(
                item_id="item_001",
                name="Test Item",
                quantity=1,
                unit_price=12.0,
                total_price=12.0
            )
            
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=[item],
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=14.0
            )
            
            manager.create_order(order)
            
            retrieved_order = manager.get_order("order_001")
            assert retrieved_order is not None
            assert retrieved_order.order_id == "order_001"
            assert retrieved_order.user_id == "user_001"
            
            # Test non-existent order
            non_existent = manager.get_order("non_existent")
            assert non_existent is None
    
    def test_update_order_status(self):
        """Test updating order status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OrderManager(temp_dir)
            
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.0,
                delivery_time_minutes=20,
                minimum_order=10.0,
                delivery_fee=2.0
            )
            
            item = OrderItem(
                item_id="item_001",
                name="Test Item",
                quantity=1,
                unit_price=12.0,
                total_price=12.0
            )
            
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=[item],
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=14.0
            )
            
            manager.create_order(order)
            
            # Update status
            success = manager.update_order_status("order_001", OrderStatus.CONFIRMED)
            assert success is True
            
            # Check updated order
            updated_order = manager.get_order("order_001")
            assert updated_order.status == OrderStatus.CONFIRMED
            
            # Check history
            assert len(manager._order_history) == 2
            assert manager._order_history[1]["action"] == "status_update"
            assert manager._order_history[1]["new_status"] == "confirmed"
    
    def test_get_orders_by_user(self):
        """Test getting orders by user."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OrderManager(temp_dir)
            
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.0,
                delivery_time_minutes=20,
                minimum_order=10.0,
                delivery_fee=2.0
            )
            
            item = OrderItem(
                item_id="item_001",
                name="Test Item",
                quantity=1,
                unit_price=12.0,
                total_price=12.0
            )
            
            # Create orders for different users
            order1 = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=[item],
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=14.0
            )
            
            order2 = Order(
                order_id="order_002",
                user_id="user_002",
                vendor=vendor,
                items=[item],
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=14.0
            )
            
            manager.create_order(order1)
            manager.create_order(order2)
            
            # Get orders for user_001
            user_orders = manager.get_orders_by_user("user_001")
            assert len(user_orders) == 1
            assert user_orders[0].user_id == "user_001"
            
            # Get orders for user_002
            user2_orders = manager.get_orders_by_user("user_002")
            assert len(user2_orders) == 1
            assert user2_orders[0].user_id == "user_002"
    
    def test_get_order_statistics(self):
        """Test getting order statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OrderManager(temp_dir)
            
            # Test with no orders
            stats = manager.get_order_statistics()
            assert stats["total_orders"] == 0
            
            # Add some orders
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.0,
                delivery_time_minutes=20,
                minimum_order=10.0,
                delivery_fee=2.0
            )
            
            item = OrderItem(
                item_id="item_001",
                name="Test Item",
                quantity=1,
                unit_price=12.0,
                total_price=12.0
            )
            
            order1 = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=[item],
                order_type=OrderType.DELIVERY,
                status=OrderStatus.COMPLETED,
                payment_status=PaymentStatus.COMPLETED,
                total_amount=14.0
            )
            
            order2 = Order(
                order_id="order_002",
                user_id="user_002",
                vendor=vendor,
                items=[item],
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=16.0
            )
            
            manager.create_order(order1)
            manager.create_order(order2)
            
            stats = manager.get_order_statistics()
            assert stats["total_orders"] == 2
            assert stats["status_counts"]["completed"] == 1
            assert stats["status_counts"]["pending"] == 1
            assert stats["payment_status_counts"]["completed"] == 1
            assert stats["payment_status_counts"]["pending"] == 1
            assert stats["total_revenue"] == 14.0  # Only completed orders
            assert stats["average_order_value"] == 14.0


class TestVendorManager:
    """Test VendorManager functionality."""
    
    def test_vendor_manager_creation(self):
        """Test VendorManager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VendorManager(temp_dir)
            assert manager.storage_path == Path(temp_dir)
            # Should initialize default vendors
            assert len(manager._vendors) > 0
    
    def test_get_vendor(self):
        """Test getting vendor by ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VendorManager(temp_dir)
            
            # Get default vendor
            vendor = manager.get_vendor("vendor_001")
            assert vendor is not None
            assert vendor.vendor_id == "vendor_001"
            
            # Test non-existent vendor
            non_existent = manager.get_vendor("non_existent")
            assert non_existent is None
    
    def test_get_vendors_by_cuisine(self):
        """Test getting vendors by cuisine."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VendorManager(temp_dir)
            
            # Get Italian vendors
            italian_vendors = manager.get_vendors_by_cuisine("italian")
            assert len(italian_vendors) > 0
            for vendor in italian_vendors:
                assert vendor.cuisine_type == "italian"
                assert vendor.is_available is True
            
            # Test non-existent cuisine
            french_vendors = manager.get_vendors_by_cuisine("french")
            assert len(french_vendors) == 0
    
    def test_get_all_vendors(self):
        """Test getting all available vendors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VendorManager(temp_dir)
            
            vendors = manager.get_all_vendors()
            assert len(vendors) > 0
            for vendor in vendors:
                assert vendor.is_available is True
    
    def test_search_vendors(self):
        """Test vendor search functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VendorManager(temp_dir)
            
            # Search by name
            results = manager.search_vendors("Mario")
            assert len(results) > 0
            
            # Search by cuisine
            results = manager.search_vendors("italian")
            assert len(results) > 0
            
            # Search by specialty
            results = manager.search_vendors("pizza")
            assert len(results) > 0
            
            # Test case insensitive
            results = manager.search_vendors("MARIO")
            assert len(results) > 0
    
    def test_update_vendor_availability(self):
        """Test updating vendor availability."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VendorManager(temp_dir)
            
            # Update availability
            success = manager.update_vendor_availability("vendor_001", False)
            assert success is True
            
            # Check updated vendor
            vendor = manager.get_vendor("vendor_001")
            assert vendor.is_available is False
            
            # Check it's not in available vendors list
            available_vendors = manager.get_all_vendors()
            vendor_ids = [v.vendor_id for v in available_vendors]
            assert "vendor_001" not in vendor_ids


class TestOrchestratorAgent:
    """Test OrchestratorAgent functionality."""
    
    def test_orchestrator_agent_creation(self):
        """Test OrchestratorAgent creation."""
        agent = OrchestratorAgent("test_orchestrator")
        
        assert agent.agent_id == "test_orchestrator"
        assert agent.agent_type == AgentType.ORCHESTRATOR
        assert agent.order_manager is not None
        assert agent.vendor_manager is not None
        assert agent.user_twin_agent is None
        assert len(agent.current_orders) == 0
        assert len(agent.order_workflows) == 0
    
    def test_orchestrator_agent_with_config(self):
        """Test OrchestratorAgent creation with configuration."""
        config = AgentConfiguration(
            agent_id="test_orchestrator",
            agent_type=AgentType.ORCHESTRATOR,
            name="Test Orchestrator",
            description="Test orchestrator agent",
            personality=PersonalityProfile(
                primary_traits=[PersonalityTrait.SYSTEMATIC, PersonalityTrait.HELPFUL],
                response_style="professional",
                verbosity_level=5,
                formality_level=6,
                cooperation_tendency=0.8,
                honesty_tendency=0.9
            )
        )
        
        agent = OrchestratorAgent("test_orchestrator", config)
        
        assert agent.agent_id == "test_orchestrator"
        assert agent.name == "Test Orchestrator"
        assert agent.description == "Test orchestrator agent"
    
    def test_set_user_twin_agent(self):
        """Test setting user twin agent."""
        orchestrator = OrchestratorAgent("test_orchestrator")
        user_twin = UserTwinAgent("test_user_twin")
        
        orchestrator.set_user_twin_agent(user_twin)
        
        assert orchestrator.user_twin_agent is not None
        assert orchestrator.user_twin_agent.agent_id == "test_user_twin"
    
    @pytest.mark.asyncio
    async def test_search_vendors(self):
        """Test vendor search functionality."""
        orchestrator = OrchestratorAgent("test_orchestrator")
        
        # Search without preferences
        vendors = await orchestrator.search_vendors("italian")
        assert len(vendors) > 0
        
        # Search with preferences
        user_preferences = {"cuisine": "italian", "budget": "moderate"}
        vendors = await orchestrator.search_vendors("pizza", user_preferences)
        assert len(vendors) > 0
    
    @pytest.mark.asyncio
    async def test_create_order(self):
        """Test order creation."""
        orchestrator = OrchestratorAgent("test_orchestrator")
        
        # Create order items
        items = [
            {
                "item_id": "item_001",
                "name": "Margherita Pizza",
                "quantity": 1,
                "unit_price": 15.99,
                "special_instructions": "Extra cheese"
            }
        ]
        
        # Create order
        order = await orchestrator.create_order(
            user_id="user_001",
            vendor_id="vendor_001",
            items=items,
            order_type=OrderType.DELIVERY,
            delivery_address="123 Main St"
        )
        
        assert order is not None
        assert order.user_id == "user_001"
        assert order.vendor.vendor_id == "vendor_001"
        assert len(order.items) == 1
        assert order.items[0].name == "Margherita Pizza"
        assert order.order_type == OrderType.DELIVERY
        assert order.status == OrderStatus.PENDING
        assert order.delivery_address == "123 Main St"
        
        # Check that order is in current orders
        assert order.order_id in orchestrator.current_orders
    
    @pytest.mark.asyncio
    async def test_create_order_invalid_vendor(self):
        """Test order creation with invalid vendor."""
        orchestrator = OrchestratorAgent("test_orchestrator")
        
        items = [
            {
                "item_id": "item_001",
                "name": "Test Item",
                "quantity": 1,
                "unit_price": 10.0
            }
        ]
        
        # Try to create order with non-existent vendor
        order = await orchestrator.create_order(
            user_id="user_001",
            vendor_id="non_existent_vendor",
            items=items
        )
        
        assert order is None
    
    @pytest.mark.asyncio
    async def test_create_order_below_minimum(self):
        """Test order creation below minimum order amount."""
        orchestrator = OrchestratorAgent("test_orchestrator")
        
        # Create order below minimum (vendor_001 has minimum 15.0)
        items = [
            {
                "item_id": "item_001",
                "name": "Small Item",
                "quantity": 1,
                "unit_price": 5.0  # Below minimum
            }
        ]
        
        order = await orchestrator.create_order(
            user_id="user_001",
            vendor_id="vendor_001",
            items=items
        )
        
        assert order is None
    
    @pytest.mark.asyncio
    async def test_get_order_status(self):
        """Test getting order status."""
        orchestrator = OrchestratorAgent("test_orchestrator")
        
        # Create an order first
        items = [
            {
                "item_id": "item_001",
                "name": "Test Item",
                "quantity": 1,
                "unit_price": 16.0
            }
        ]
        
        order = await orchestrator.create_order(
            user_id="user_001",
            vendor_id="vendor_001",
            items=items
        )
        
        assert order is not None
        
        # Get order status
        status = await orchestrator.get_order_status(order.order_id)
        assert status is not None
        assert status["order_id"] == order.order_id
        assert status["status"] == "pending"
        assert status["payment_status"] == "pending"
        assert status["vendor_name"] == order.vendor.name
        assert status["total_amount"] == order.total_amount
        
        # Test non-existent order
        non_existent_status = await orchestrator.get_order_status("non_existent")
        assert non_existent_status is None
    
    @pytest.mark.asyncio
    async def test_update_order_status(self):
        """Test updating order status."""
        orchestrator = OrchestratorAgent("test_orchestrator")
        
        # Create an order first
        items = [
            {
                "item_id": "item_001",
                "name": "Test Item",
                "quantity": 1,
                "unit_price": 16.0
            }
        ]
        
        order = await orchestrator.create_order(
            user_id="user_001",
            vendor_id="vendor_001",
            items=items
        )
        
        assert order is not None
        
        # Update status
        success = await orchestrator.update_order_status(order.order_id, OrderStatus.CONFIRMED)
        assert success is True
        
        # Check updated status
        status = await orchestrator.get_order_status(order.order_id)
        assert status["status"] == "confirmed"
        
        # Test non-existent order
        non_existent_success = await orchestrator.update_order_status("non_existent", OrderStatus.CONFIRMED)
        assert non_existent_success is False
    
    @pytest.mark.asyncio
    async def test_process_message(self):
        """Test message processing."""
        orchestrator = OrchestratorAgent("test_orchestrator")
        
        # Test search message
        response = await orchestrator.process_message("I'm looking for Italian food")
        assert response is not None
        assert len(response) > 0
        
        # Test order message
        response = await orchestrator.process_message("I want to place an order")
        assert response is not None
        assert len(response) > 0
        
        # Test status message
        response = await orchestrator.process_message("What's the status of my order?")
        assert response is not None
        assert len(response) > 0
        
        # Test help message
        response = await orchestrator.process_message("Help me")
        assert response is not None
        assert len(response) > 0
    
    def test_get_order_statistics(self):
        """Test getting order statistics."""
        orchestrator = OrchestratorAgent("test_orchestrator")
        
        stats = orchestrator.get_order_statistics()
        assert "total_orders" in stats
        assert "status_counts" in stats
        assert "payment_status_counts" in stats
        assert "total_revenue" in stats
        assert "average_order_value" in stats
    
    def test_get_vendor_statistics(self):
        """Test getting vendor statistics."""
        orchestrator = OrchestratorAgent("test_orchestrator")
        
        stats = orchestrator.get_vendor_statistics()
        assert "total_vendors" in stats
        assert "cuisine_counts" in stats
        assert "average_rating" in stats
        assert "available_vendors" in stats
        assert stats["total_vendors"] > 0
    
    def test_get_metrics(self):
        """Test getting agent metrics."""
        orchestrator = OrchestratorAgent("test_orchestrator")
        
        metrics = orchestrator.get_metrics()
        assert "total_orders" in metrics
        assert "active_orders" in metrics
        assert "total_vendors" in metrics
        assert "available_vendors" in metrics
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test agent shutdown."""
        orchestrator = OrchestratorAgent("test_orchestrator")
        
        await orchestrator.shutdown()
        
        assert orchestrator._status == AgentState.STOPPED


class TestOrchestratorFactory:
    """Test OrchestratorAgent factory function."""
    
    def test_create_orchestrator_agent(self):
        """Test factory function."""
        agent = create_orchestrator_agent("test_orchestrator")
        
        assert isinstance(agent, OrchestratorAgent)
        assert agent.agent_id == "test_orchestrator"
        assert agent.agent_type == AgentType.ORCHESTRATOR


class TestOrchestratorIntegration:
    """Integration tests for OrchestratorAgent."""
    
    @pytest.mark.asyncio
    async def test_complete_order_workflow(self):
        """Test complete order workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = OrchestratorAgent("test_orchestrator")
            # Create fresh managers with temporary storage
            orchestrator.order_manager = OrderManager(temp_dir)
            orchestrator.vendor_manager = VendorManager(temp_dir)
            
            # Step 1: Search for vendors
            vendors = await orchestrator.search_vendors("italian")
            assert len(vendors) > 0
            
            # Step 2: Create order
            items = [
                {
                    "item_id": "item_001",
                    "name": "Margherita Pizza",
                    "quantity": 1,
                    "unit_price": 16.0,
                    "special_instructions": "Extra cheese"
                }
            ]
            
            order = await orchestrator.create_order(
                user_id="user_001",
                vendor_id="vendor_001",
                items=items,
                order_type=OrderType.DELIVERY,
                delivery_address="123 Main St"
            )
            
            assert order is not None
            
            # Step 3: Check initial status
            status = await orchestrator.get_order_status(order.order_id)
            assert status["status"] == "pending"
            
            # Step 4: Update status to confirmed
            success = await orchestrator.update_order_status(order.order_id, OrderStatus.CONFIRMED)
            assert success is True
            
            # Step 5: Check updated status
            updated_status = await orchestrator.get_order_status(order.order_id)
            assert updated_status["status"] == "confirmed"
            
            # Step 6: Check statistics
            stats = orchestrator.get_order_statistics()
            assert stats["total_orders"] == 1
    
    @pytest.mark.asyncio
    async def test_orchestrator_with_user_twin_integration(self):
        """Test orchestrator integration with UserTwin agent."""
        orchestrator = OrchestratorAgent("test_orchestrator")
        user_twin = UserTwinAgent("test_user_twin")
        
        # Set up user preferences
        user_twin.add_preference(PreferenceCategory.FOOD, "cuisine", "italian", 0.9)
        user_twin.add_preference(PreferenceCategory.COST, "budget", "moderate", 0.7)
        
        # Connect user twin to orchestrator
        orchestrator.set_user_twin_agent(user_twin)
        
        # Search with user preferences
        vendors = await orchestrator.search_vendors("pizza")
        assert len(vendors) > 0
        
        # The search should be influenced by user preferences
        # (In a real implementation, this would be more sophisticated)
        
        # Create order
        items = [
            {
                "item_id": "item_001",
                "name": "Margherita Pizza",
                "quantity": 1,
                "unit_price": 16.0
            }
        ]
        
        order = await orchestrator.create_order(
            user_id="user_001",
            vendor_id="vendor_001",
            items=items
        )
        
        assert order is not None
        assert order.vendor.cuisine_type == "italian"  # Should match user preference
