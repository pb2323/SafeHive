"""
Unit tests for food ordering tools.
"""

import pytest
import json
from unittest.mock import patch, Mock

from safehive.tools.food_ordering_tools import (
    lookup_menu, place_order, process_payment, check_order_status, check_inventory,
    MenuLookupTool, OrderPlacementTool, PaymentProcessingTool, OrderStatusTool, InventoryCheckTool,
    MenuLookupInput, OrderPlacementInput, PaymentProcessingInput, OrderStatusInput, InventoryCheckInput
)


class TestFoodOrderingTools:
    """Test food ordering tool functions."""

    def test_lookup_menu_success(self):
        """Test successful menu lookup."""
        result = lookup_menu("restaurant_1")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "Menu lookup successful" in result_data["message"]
        assert result_data["data"]["restaurant_name"] == "Pizza Palace"
        assert "menu" in result_data["data"]

    def test_lookup_menu_with_category_filter(self):
        """Test menu lookup with category filter."""
        result = lookup_menu("restaurant_1", category="pizza")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["category_filter"] == "pizza"
        assert "pizza" in result_data["data"]["menu"]

    def test_lookup_menu_with_dietary_restrictions(self):
        """Test menu lookup with dietary restrictions."""
        result = lookup_menu("restaurant_1", dietary_restrictions=["vegetarian"])
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "vegetarian" in result_data["data"]["dietary_restrictions_applied"]

    def test_lookup_menu_restaurant_not_found(self):
        """Test menu lookup for non-existent restaurant."""
        result = lookup_menu("nonexistent_restaurant")
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "not found" in result_data["message"]

    def test_place_order_success(self):
        """Test successful order placement."""
        items = [{"id": "pizza_1", "quantity": 2}]
        customer_info = {"name": "Test Customer", "phone": "555-1234"}
        delivery_address = {"street": "123 Main St", "city": "Test City"}
        
        result = place_order("restaurant_1", items, customer_info, delivery_address)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "Order placed successfully" in result_data["message"]
        assert "order_id" in result_data["data"]
        assert result_data["data"]["restaurant_name"] == "Pizza Palace"
        assert result_data["data"]["total_price"] > 0

    def test_place_order_item_not_found(self):
        """Test order placement with non-existent item."""
        items = [{"id": "nonexistent_item", "quantity": 1}]
        customer_info = {"name": "Test Customer"}
        delivery_address = {"street": "123 Main St"}
        
        result = place_order("restaurant_1", items, customer_info, delivery_address)
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "not found in restaurant menu" in result_data["message"]

    def test_process_payment_success(self):
        """Test successful payment processing."""
        # First place an order
        items = [{"id": "pizza_1", "quantity": 1}]
        customer_info = {"name": "Test Customer"}
        delivery_address = {"street": "123 Main St"}
        order_result = place_order("restaurant_1", items, customer_info, delivery_address)
        order_data = json.loads(order_result)
        order_id = order_data["data"]["order_id"]
        
        # Process payment
        payment_details = {"card_number": "****1234", "expiry": "12/25"}
        result = process_payment(order_id, "credit_card", payment_details, order_data["data"]["total_price"])
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "Payment processed successfully" in result_data["message"]
        assert "payment_id" in result_data["data"]

    def test_process_payment_amount_mismatch(self):
        """Test payment processing with amount mismatch."""
        # First place an order
        items = [{"id": "pizza_1", "quantity": 1}]
        customer_info = {"name": "Test Customer"}
        delivery_address = {"street": "123 Main St"}
        order_result = place_order("restaurant_1", items, customer_info, delivery_address)
        order_data = json.loads(order_result)
        order_id = order_data["data"]["order_id"]
        
        # Process payment with wrong amount
        payment_details = {"card_number": "****1234"}
        result = process_payment(order_id, "credit_card", payment_details, 999.99)
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "does not match order total" in result_data["message"]

    def test_check_order_status_success(self):
        """Test successful order status check."""
        # First place an order
        items = [{"id": "pizza_1", "quantity": 1}]
        customer_info = {"name": "Test Customer"}
        delivery_address = {"street": "123 Main St"}
        order_result = place_order("restaurant_1", items, customer_info, delivery_address)
        order_data = json.loads(order_result)
        order_id = order_data["data"]["order_id"]
        
        result = check_order_status(order_id)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "Order status retrieved" in result_data["message"]
        assert "status" in result_data["data"]

    def test_check_order_status_not_found(self):
        """Test order status check for non-existent order."""
        result = check_order_status("nonexistent_order_id")
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "not found" in result_data["message"]

    def test_check_inventory_success(self):
        """Test successful inventory check."""
        result = check_inventory("restaurant_1", ["pizza_1", "pizza_2"])
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "Inventory check completed" in result_data["message"]
        assert "items" in result_data["data"]
        assert "pizza_1" in result_data["data"]["items"]

    def test_check_inventory_restaurant_not_found(self):
        """Test inventory check for non-existent restaurant."""
        result = check_inventory("nonexistent_restaurant", ["pizza_1"])
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "not found" in result_data["message"]


class TestFoodOrderingToolClasses:
    """Test food ordering tool classes."""

    def test_menu_lookup_tool(self):
        """Test MenuLookupTool class."""
        tool = MenuLookupTool()
        input_data = MenuLookupInput(restaurant_id="restaurant_1")
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["restaurant_name"] == "Pizza Palace"

    def test_order_placement_tool(self):
        """Test OrderPlacementTool class."""
        tool = OrderPlacementTool()
        input_data = OrderPlacementInput(
            restaurant_id="restaurant_1",
            items=[{"id": "pizza_1", "quantity": 1}],
            customer_info={"name": "Test Customer"},
            delivery_address={"street": "123 Main St"}
        )
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "order_id" in result_data["data"]

    def test_payment_processing_tool(self):
        """Test PaymentProcessingTool class."""
        # First create an order
        items = [{"id": "pizza_1", "quantity": 1}]
        customer_info = {"name": "Test Customer"}
        delivery_address = {"street": "123 Main St"}
        order_result = place_order("restaurant_1", items, customer_info, delivery_address)
        order_data = json.loads(order_result)
        order_id = order_data["data"]["order_id"]
        
        tool = PaymentProcessingTool()
        input_data = PaymentProcessingInput(
            order_id=order_id,
            payment_method="credit_card",
            payment_details={"card_number": "****1234"},
            amount=order_data["data"]["total_price"]
        )
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "payment_id" in result_data["data"]

    def test_order_status_tool(self):
        """Test OrderStatusTool class."""
        # First create an order
        items = [{"id": "pizza_1", "quantity": 1}]
        customer_info = {"name": "Test Customer"}
        delivery_address = {"street": "123 Main St"}
        order_result = place_order("restaurant_1", items, customer_info, delivery_address)
        order_data = json.loads(order_result)
        order_id = order_data["data"]["order_id"]
        
        tool = OrderStatusTool()
        input_data = OrderStatusInput(order_id=order_id)
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "status" in result_data["data"]

    def test_inventory_check_tool(self):
        """Test InventoryCheckTool class."""
        tool = InventoryCheckTool()
        input_data = InventoryCheckInput(restaurant_id="restaurant_1", items=["pizza_1"])
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "items" in result_data["data"]


class TestFoodOrderingIntegration:
    """Test integration between food ordering tools."""

    def test_complete_order_workflow(self):
        """Test complete order workflow from menu lookup to payment."""
        # 1. Look up menu
        menu_result = lookup_menu("restaurant_1")
        menu_data = json.loads(menu_result)
        assert menu_data["success"] is True
        
        # 2. Place order
        items = [{"id": "pizza_1", "quantity": 1}]
        customer_info = {"name": "Integration Test Customer", "phone": "555-5678"}
        delivery_address = {"street": "456 Test Ave", "city": "Test City"}
        
        order_result = place_order("restaurant_1", items, customer_info, delivery_address)
        order_data = json.loads(order_result)
        assert order_data["success"] is True
        order_id = order_data["data"]["order_id"]
        
        # 3. Check order status
        status_result = check_order_status(order_id)
        status_data = json.loads(status_result)
        assert status_data["success"] is True
        
        # 4. Process payment
        payment_details = {"card_number": "****5678", "expiry": "12/25"}
        payment_result = process_payment(order_id, "credit_card", payment_details, order_data["data"]["total_price"])
        payment_data = json.loads(payment_result)
        assert payment_data["success"] is True
        
        # 5. Check final order status
        final_status_result = check_order_status(order_id)
        final_status_data = json.loads(final_status_result)
        assert final_status_data["success"] is True
        assert final_status_data["data"]["status"] == "paid"
