"""
Food ordering specific tools for the SafeHive AI Security Sandbox.

This module provides tools for menu lookup, order placement, payment processing,
and other food ordering related operations that agents can use.
"""

from typing import Any, Dict, List, Optional
import json
import uuid
from datetime import datetime, timedelta

from pydantic import Field, BaseModel
from ..utils.logger import get_logger
from .base_tools import BaseSafeHiveTool, ToolInput, ToolOutput, create_tool_output

logger = get_logger(__name__)


# Pydantic models for tool input validation
class MenuLookupInput(ToolInput):
    """Input for menu lookup requests."""
    restaurant_id: str = Field(description="The restaurant or vendor ID to lookup menu for")
    category: Optional[str] = Field(default=None, description="Specific menu category to filter by")
    dietary_restrictions: Optional[List[str]] = Field(default=None, description="List of dietary restrictions to filter by")


class OrderPlacementInput(ToolInput):
    """Input for order placement requests."""
    restaurant_id: str = Field(description="The restaurant ID to place order with")
    items: List[Dict[str, Any]] = Field(description="List of items to order with quantities")
    customer_info: Dict[str, Any] = Field(description="Customer information for the order")
    delivery_address: Dict[str, Any] = Field(description="Delivery address information")


class PaymentProcessingInput(ToolInput):
    """Input for payment processing requests."""
    order_id: str = Field(description="The order ID to process payment for")
    payment_method: str = Field(description="Payment method (credit_card, debit_card, digital_wallet)")
    payment_details: Dict[str, Any] = Field(description="Payment details (encrypted)")
    amount: float = Field(description="Payment amount in dollars")


class OrderStatusInput(ToolInput):
    """Input for order status requests."""
    order_id: str = Field(description="The order ID to check status for")


class InventoryCheckInput(ToolInput):
    """Input for inventory check requests."""
    restaurant_id: str = Field(description="The restaurant ID to check inventory for")
    items: List[str] = Field(description="List of item IDs to check availability for")


# Mock data for demonstration
MOCK_RESTAURANTS = {
    "restaurant_1": {
        "name": "Pizza Palace",
        "menu": {
            "pizza": [
                {"id": "pizza_1", "name": "Margherita Pizza", "price": 12.99, "description": "Classic tomato and mozzarella"},
                {"id": "pizza_2", "name": "Pepperoni Pizza", "price": 14.99, "description": "Pepperoni with mozzarella"},
                {"id": "pizza_3", "name": "Veggie Delight", "price": 13.99, "description": "Mixed vegetables with cheese"}
            ],
            "sides": [
                {"id": "side_1", "name": "Garlic Bread", "price": 4.99, "description": "Fresh baked with garlic butter"},
                {"id": "side_2", "name": "Caesar Salad", "price": 6.99, "description": "Fresh romaine with caesar dressing"}
            ]
        },
        "dietary_info": {
            "vegetarian": ["pizza_1", "pizza_3", "side_2"],
            "vegan": ["side_2"],
            "gluten_free": []
        }
    },
    "restaurant_2": {
        "name": "Burger Barn",
        "menu": {
            "burgers": [
                {"id": "burger_1", "name": "Classic Burger", "price": 9.99, "description": "Beef patty with lettuce and tomato"},
                {"id": "burger_2", "name": "Veggie Burger", "price": 8.99, "description": "Plant-based patty with all the fixings"},
                {"id": "burger_3", "name": "Chicken Burger", "price": 10.99, "description": "Grilled chicken breast with mayo"}
            ],
            "sides": [
                {"id": "side_3", "name": "French Fries", "price": 3.99, "description": "Crispy golden fries"},
                {"id": "side_4", "name": "Onion Rings", "price": 4.99, "description": "Beer battered onion rings"}
            ]
        },
        "dietary_info": {
            "vegetarian": ["burger_2"],
            "vegan": ["burger_2"],
            "gluten_free": []
        }
    }
}

MOCK_ORDERS = {}
MOCK_INVENTORY = {
    "restaurant_1": {"pizza_1": 10, "pizza_2": 8, "pizza_3": 12, "side_1": 20, "side_2": 15},
    "restaurant_2": {"burger_1": 15, "burger_2": 10, "burger_3": 12, "side_3": 30, "side_4": 18}
}


def lookup_menu(restaurant_id: str, category: str = None, dietary_restrictions: List[str] = None) -> str:
    """Look up menu items for a specific restaurant.
    
    Args:
        restaurant_id: The restaurant ID to lookup menu for
        category: Optional category to filter by
        dietary_restrictions: Optional list of dietary restrictions
        
    Returns:
        A JSON string containing the menu information
    """
    try:
        if restaurant_id not in MOCK_RESTAURANTS:
            return create_tool_output(
                success=False,
                message=f"Restaurant {restaurant_id} not found",
                data={"restaurant_id": restaurant_id}
            ).to_json()
        
        restaurant = MOCK_RESTAURANTS[restaurant_id]
        menu_data = restaurant["menu"]
        
        # Filter by category if specified
        if category and category in menu_data:
            filtered_menu = {category: menu_data[category]}
        else:
            filtered_menu = menu_data
        
        # Filter by dietary restrictions if specified
        if dietary_restrictions:
            dietary_info = restaurant.get("dietary_info", {})
            filtered_items = {}
            
            for cat, items in filtered_menu.items():
                filtered_category_items = []
                for item in items:
                    item_id = item["id"]
                    meets_requirements = True
                    
                    for restriction in dietary_restrictions:
                        if restriction in dietary_info:
                            if item_id not in dietary_info[restriction]:
                                meets_requirements = False
                                break
                    
                    if meets_requirements:
                        filtered_category_items.append(item)
                
                if filtered_category_items:
                    filtered_items[cat] = filtered_category_items
            
            filtered_menu = filtered_items
        
        result_data = {
            "restaurant_id": restaurant_id,
            "restaurant_name": restaurant["name"],
            "menu": filtered_menu,
            "dietary_restrictions_applied": dietary_restrictions or [],
            "category_filter": category
        }
        
        return create_tool_output(
            success=True,
            message=f"Menu lookup successful for {restaurant['name']}",
            data=result_data
        ).to_json()
        
    except Exception as e:
        logger.error(f"Failed to lookup menu for {restaurant_id}: {e}")
        return create_tool_output(
            success=False,
            message=f"Failed to lookup menu: {str(e)}",
            data={"restaurant_id": restaurant_id}
        ).to_json()


def place_order(restaurant_id: str, items: List[Dict[str, Any]], customer_info: Dict[str, Any], delivery_address: Dict[str, Any]) -> str:
    """Place an order with a restaurant.
    
    Args:
        restaurant_id: The restaurant ID to place order with
        items: List of items with quantities
        customer_info: Customer information
        delivery_address: Delivery address information
        
    Returns:
        A JSON string containing the order confirmation
    """
    try:
        if restaurant_id not in MOCK_RESTAURANTS:
            return create_tool_output(
                success=False,
                message=f"Restaurant {restaurant_id} not found",
                data={"restaurant_id": restaurant_id}
            ).to_json()
        
        restaurant = MOCK_RESTAURANTS[restaurant_id]
        order_id = str(uuid.uuid4())
        
        # Calculate total price
        total_price = 0.0
        validated_items = []
        
        for item in items:
            item_id = item.get("id")
            quantity = item.get("quantity", 1)
            
            # Find item in menu
            item_found = False
            for category, menu_items in restaurant["menu"].items():
                for menu_item in menu_items:
                    if menu_item["id"] == item_id:
                        item_total = menu_item["price"] * quantity
                        total_price += item_total
                        validated_items.append({
                            **menu_item,
                            "quantity": quantity,
                            "item_total": item_total
                        })
                        item_found = True
                        break
                if item_found:
                    break
            
            if not item_found:
                return create_tool_output(
                    success=False,
                    message=f"Item {item_id} not found in restaurant menu",
                    data={"item_id": item_id, "restaurant_id": restaurant_id}
                ).to_json()
        
        # Create order record
        order_data = {
            "order_id": order_id,
            "restaurant_id": restaurant_id,
            "restaurant_name": restaurant["name"],
            "items": validated_items,
            "total_price": round(total_price, 2),
            "customer_info": customer_info,
            "delivery_address": delivery_address,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "estimated_delivery": (datetime.now() + timedelta(minutes=30)).isoformat()
        }
        
        MOCK_ORDERS[order_id] = order_data
        
        logger.info(f"Order placed successfully: {order_id} for {restaurant['name']}")
        
        return create_tool_output(
            success=True,
            message=f"Order placed successfully with {restaurant['name']}",
            data=order_data
        ).to_json()
        
    except Exception as e:
        logger.error(f"Failed to place order: {e}")
        return create_tool_output(
            success=False,
            message=f"Failed to place order: {str(e)}",
            data={"restaurant_id": restaurant_id}
        ).to_json()


def process_payment(order_id: str, payment_method: str, payment_details: Dict[str, Any], amount: float) -> str:
    """Process payment for an order.
    
    Args:
        order_id: The order ID to process payment for
        payment_method: Payment method type
        payment_details: Payment details (should be encrypted in real implementation)
        amount: Payment amount
        
    Returns:
        A JSON string containing the payment processing result
    """
    try:
        if order_id not in MOCK_ORDERS:
            return create_tool_output(
                success=False,
                message=f"Order {order_id} not found",
                data={"order_id": order_id}
            ).to_json()
        
        order = MOCK_ORDERS[order_id]
        
        # Validate amount matches order total
        if abs(amount - order["total_price"]) > 0.01:  # Allow for small floating point differences
            return create_tool_output(
                success=False,
                message=f"Payment amount {amount} does not match order total {order['total_price']}",
                data={"order_id": order_id, "payment_amount": amount, "order_total": order["total_price"]}
            ).to_json()
        
        # Simulate payment processing
        payment_id = str(uuid.uuid4())
        
        # Update order status
        order["status"] = "paid"
        order["payment_info"] = {
            "payment_id": payment_id,
            "payment_method": payment_method,
            "amount": amount,
            "processed_at": datetime.now().isoformat(),
            "status": "successful"
        }
        
        logger.info(f"Payment processed successfully: {payment_id} for order {order_id}")
        
        return create_tool_output(
            success=True,
            message=f"Payment processed successfully for order {order_id}",
            data={
                "payment_id": payment_id,
                "order_id": order_id,
                "amount": amount,
                "status": "successful",
                "restaurant_name": order["restaurant_name"]
            }
        ).to_json()
        
    except Exception as e:
        logger.error(f"Failed to process payment for order {order_id}: {e}")
        return create_tool_output(
            success=False,
            message=f"Failed to process payment: {str(e)}",
            data={"order_id": order_id}
        ).to_json()


def check_order_status(order_id: str) -> str:
    """Check the status of an order.
    
    Args:
        order_id: The order ID to check status for
        
    Returns:
        A JSON string containing the order status
    """
    try:
        if order_id not in MOCK_ORDERS:
            return create_tool_output(
                success=False,
                message=f"Order {order_id} not found",
                data={"order_id": order_id}
            ).to_json()
        
        order = MOCK_ORDERS[order_id]
        
        # Simulate order progression
        created_at = datetime.fromisoformat(order["created_at"])
        elapsed_minutes = (datetime.now() - created_at).total_seconds() / 60
        
        # Update status based on elapsed time (for demo purposes)
        if order["status"] == "pending" and elapsed_minutes > 5:
            order["status"] = "preparing"
        elif order["status"] == "preparing" and elapsed_minutes > 15:
            order["status"] = "out_for_delivery"
        elif order["status"] == "out_for_delivery" and elapsed_minutes > 25:
            order["status"] = "delivered"
        
        return create_tool_output(
            success=True,
            message=f"Order status retrieved for {order_id}",
            data={
                "order_id": order_id,
                "status": order["status"],
                "restaurant_name": order["restaurant_name"],
                "total_price": order["total_price"],
                "estimated_delivery": order["estimated_delivery"],
                "elapsed_minutes": round(elapsed_minutes, 1)
            }
        ).to_json()
        
    except Exception as e:
        logger.error(f"Failed to check order status for {order_id}: {e}")
        return create_tool_output(
            success=False,
            message=f"Failed to check order status: {str(e)}",
            data={"order_id": order_id}
        ).to_json()


def check_inventory(restaurant_id: str, items: List[str]) -> str:
    """Check inventory availability for specific items.
    
    Args:
        restaurant_id: The restaurant ID to check inventory for
        items: List of item IDs to check availability for
        
    Returns:
        A JSON string containing inventory information
    """
    try:
        if restaurant_id not in MOCK_RESTAURANTS:
            return create_tool_output(
                success=False,
                message=f"Restaurant {restaurant_id} not found",
                data={"restaurant_id": restaurant_id}
            ).to_json()
        
        if restaurant_id not in MOCK_INVENTORY:
            return create_tool_output(
                success=False,
                message=f"Inventory data not available for restaurant {restaurant_id}",
                data={"restaurant_id": restaurant_id}
            ).to_json()
        
        inventory = MOCK_INVENTORY[restaurant_id]
        availability = {}
        
        for item_id in items:
            if item_id in inventory:
                quantity = inventory[item_id]
                availability[item_id] = {
                    "available": quantity > 0,
                    "quantity": quantity,
                    "status": "in_stock" if quantity > 5 else "low_stock" if quantity > 0 else "out_of_stock"
                }
            else:
                availability[item_id] = {
                    "available": False,
                    "quantity": 0,
                    "status": "not_found"
                }
        
        return create_tool_output(
            success=True,
            message=f"Inventory check completed for {restaurant_id}",
            data={
                "restaurant_id": restaurant_id,
                "restaurant_name": MOCK_RESTAURANTS[restaurant_id]["name"],
                "items": availability,
                "checked_at": datetime.now().isoformat()
            }
        ).to_json()
        
    except Exception as e:
        logger.error(f"Failed to check inventory for {restaurant_id}: {e}")
        return create_tool_output(
            success=False,
            message=f"Failed to check inventory: {str(e)}",
            data={"restaurant_id": restaurant_id}
        ).to_json()


# Tool classes
class MenuLookupTool(BaseSafeHiveTool):
    name: str = "lookup_menu"
    description: str = "Look up menu items for a specific restaurant with optional filtering by category and dietary restrictions."
    args_schema: type[MenuLookupInput] = MenuLookupInput

    def _execute(self, input_data: MenuLookupInput) -> str:
        return lookup_menu(
            input_data.restaurant_id,
            input_data.category,
            input_data.dietary_restrictions
        )


class OrderPlacementTool(BaseSafeHiveTool):
    name: str = "place_order"
    description: str = "Place an order with a restaurant including items, customer info, and delivery address."
    args_schema: type[OrderPlacementInput] = OrderPlacementInput

    def _execute(self, input_data: OrderPlacementInput) -> str:
        return place_order(
            input_data.restaurant_id,
            input_data.items,
            input_data.customer_info,
            input_data.delivery_address
        )


class PaymentProcessingTool(BaseSafeHiveTool):
    name: str = "process_payment"
    description: str = "Process payment for an order using specified payment method and details."
    args_schema: type[PaymentProcessingInput] = PaymentProcessingInput

    def _execute(self, input_data: PaymentProcessingInput) -> str:
        return process_payment(
            input_data.order_id,
            input_data.payment_method,
            input_data.payment_details,
            input_data.amount
        )


class OrderStatusTool(BaseSafeHiveTool):
    name: str = "check_order_status"
    description: str = "Check the current status of an order including delivery progress."
    args_schema: type[OrderStatusInput] = OrderStatusInput

    def _execute(self, input_data: OrderStatusInput) -> str:
        return check_order_status(input_data.order_id)


class InventoryCheckTool(BaseSafeHiveTool):
    name: str = "check_inventory"
    description: str = "Check inventory availability for specific items at a restaurant."
    args_schema: type[InventoryCheckInput] = InventoryCheckInput

    def _execute(self, input_data: InventoryCheckInput) -> str:
        return check_inventory(input_data.restaurant_id, input_data.items)


# Convenience function to get all food ordering tools
def get_food_ordering_tools() -> List[BaseSafeHiveTool]:
    """Get all food ordering tools for agent configuration."""
    return [
        MenuLookupTool(),
        OrderPlacementTool(),
        PaymentProcessingTool(),
        OrderStatusTool(),
        InventoryCheckTool()
    ]
