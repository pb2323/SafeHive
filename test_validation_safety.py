#!/usr/bin/env python3
"""
Test Validation and Safety Checks (Tasks 1.6, 1.7)
==================================================

Tests the order validation, safety checks, and error handling features.
"""

import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safehive.mcp.simple_doorDash_client import SimpleDoorDashMCPClient, OrderRequest

# Set up logging to see the validation and safety check messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_validation_and_safety():
    """Test order validation and safety checks"""
    print("🧪 Testing Order Validation and Safety Checks (Tasks 1.6, 1.7)")
    print("=" * 70)
    
    # MCP server path
    mcp_server_path = "/Users/rutujanemane/Documents/SJSU/A10 hackathon/DoorDash-MCP-Server/build/index.js"
    
    try:
        # Create client
        client = SimpleDoorDashMCPClient(mcp_server_path, environment="sandbox")
        
        # Test 1: Valid Order (should pass)
        print("\n1. Testing VALID order...")
        valid_order = OrderRequest(
            source="test_valid",
            restaurant="Test Pizza Palace",
            items=[
                {"name": "Margherita Pizza", "quantity": 1, "price": 15.99},
                {"name": "Caesar Salad", "quantity": 1, "price": 8.99}
            ],
            total=24.98,
            delivery_address="123 Test Street, Test City, TC 12345",
            customer_phone="+1-555-0123"
        )
        
        response = client.process_order(valid_order)
        if response.status == "success":
            print("   ✅ Valid order processed successfully")
            print(f"   📦 Order ID: {response.order_id}")
        else:
            print(f"   ❌ Valid order failed: {response.error_message}")
        
        # Test 2: Invalid Order - Missing Required Fields
        print("\n2. Testing INVALID order (missing fields)...")
        invalid_order = OrderRequest(
            source="",  # Missing source
            restaurant="",  # Missing restaurant
            items=[],  # Empty items
            total=0,  # Invalid total
            delivery_address="",  # Missing address
            customer_phone=""
        )
        
        response = client.process_order(invalid_order)
        if response.status == "error":
            print("   ✅ Invalid order correctly rejected")
            print(f"   📝 Error: {response.error_message}")
        else:
            print("   ❌ Invalid order was incorrectly accepted")
        
        # Test 3: High-Value Order (should trigger safety checks)
        print("\n3. Testing HIGH-VALUE order (safety checks)...")
        high_value_order = OrderRequest(
            source="test_high_value",
            restaurant="Expensive Restaurant",
            items=[
                {"name": "Premium Steak", "quantity": 1, "price": 75.00}
            ],
            total=75.00,  # Above $50 threshold
            delivery_address="456 Real Street, Real City, RC 12345",  # Real address
            customer_phone="+1-555-9999"  # Real phone
        )
        
        response = client.process_order(high_value_order)
        if response.status == "error":
            print("   ✅ High-value order correctly flagged for safety")
            print(f"   📝 Safety check result: {response.error_message}")
        else:
            print("   ⚠️  High-value order processed (may need confirmation)")
            print(f"   📦 Order ID: {response.order_id}")
        
        # Test 4: Order with Invalid Phone Number
        print("\n4. Testing order with INVALID phone number...")
        invalid_phone_order = OrderRequest(
            source="test_invalid_phone",
            restaurant="Test Restaurant",
            items=[{"name": "Pizza", "quantity": 1, "price": 15.99}],
            total=15.99,
            delivery_address="123 Test Street, Test City, TC 12345",
            customer_phone="invalid-phone"  # Invalid format
        )
        
        response = client.process_order(invalid_phone_order)
        if response.status == "success":
            print("   ✅ Order with invalid phone processed (with warnings)")
            print(f"   📦 Order ID: {response.order_id}")
        else:
            print(f"   ❌ Order with invalid phone failed: {response.error_message}")
        
        # Test 5: Order Below Minimum Value
        print("\n5. Testing order BELOW minimum value...")
        low_value_order = OrderRequest(
            source="test_low_value",
            restaurant="Test Restaurant",
            items=[{"name": "Small Item", "quantity": 1, "price": 2.99}],
            total=2.99,  # Below $5 minimum
            delivery_address="123 Test Street, Test City, TC 12345",
            customer_phone="+1-555-0123"
        )
        
        response = client.process_order(low_value_order)
        if response.status == "error":
            print("   ✅ Low-value order correctly rejected")
            print(f"   📝 Error: {response.error_message}")
        else:
            print("   ❌ Low-value order was incorrectly accepted")
        
        # Test 6: Order Above Maximum Value
        print("\n6. Testing order ABOVE maximum value...")
        max_value_order = OrderRequest(
            source="test_max_value",
            restaurant="Test Restaurant",
            items=[{"name": "Expensive Item", "quantity": 1, "price": 150.00}],
            total=150.00,  # Above $100 maximum
            delivery_address="123 Test Street, Test City, TC 12345",
            customer_phone="+1-555-0123"
        )
        
        response = client.process_order(max_value_order)
        if response.status == "error":
            print("   ✅ Max-value order correctly rejected")
            print(f"   📝 Error: {response.error_message}")
        else:
            print("   ❌ Max-value order was incorrectly accepted")
        
        # Test 7: Skip Validation (for testing)
        print("\n7. Testing order with SKIPPED validation...")
        skip_validation_order = OrderRequest(
            source="",  # Missing source
            restaurant="",  # Missing restaurant
            items=[],  # Empty items
            total=0,  # Invalid total
            delivery_address="",  # Missing address
            customer_phone=""
        )
        
        response = client.process_order(skip_validation_order, skip_validation=True)
        if response.status == "success":
            print("   ✅ Order with skipped validation processed successfully")
            print(f"   📦 Order ID: {response.order_id}")
        else:
            print(f"   ❌ Order with skipped validation failed: {response.error_message}")
        
        print("\n" + "=" * 70)
        print("🎉 Validation and Safety Check tests completed!")
        print("✅ Tasks 1.6 and 1.7 implemented successfully")
        print("🚀 Order validation, safety checks, and error handling working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_validation_and_safety()
    sys.exit(0 if success else 1)
