#!/usr/bin/env python3
"""
Quick test for MCP client integration
=====================================

Tests the DoorDash MCP client wrapper.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safehive.mcp.doorDash_client import DoorDashMCPClient, OrderRequest

def test_mcp_client():
    """Test the MCP client functionality"""
    print("🧪 Testing DoorDash MCP Client Integration")
    print("=" * 50)
    
    # MCP server path
    mcp_server_path = "/Users/rutujanemane/Documents/SJSU/A10 hackathon/DoorDash-MCP-Server/build/index.js"
    
    try:
        # Create client
        client = DoorDashMCPClient(mcp_server_path, environment="sandbox")
        
        # Test initialization
        print("1. Testing MCP server initialization...")
        if client.initialize():
            print("   ✅ MCP server initialized successfully")
        else:
            print("   ❌ MCP server initialization failed")
            return False
        
        # Test business listing
        print("\n2. Testing business listing...")
        businesses = client.list_businesses()
        if "error" not in businesses:
            print("   ✅ Business listing successful")
            print(f"   📊 Response: {businesses}")
        else:
            print(f"   ❌ Business listing failed: {businesses['error']}")
        
        # Test store listing
        print("\n3. Testing store listing...")
        stores = client.list_stores("default")
        if "error" not in stores:
            print("   ✅ Store listing successful")
            print(f"   📊 Response: {stores}")
        else:
            print(f"   ❌ Store listing failed: {stores['error']}")
        
        # Test order creation
        print("\n4. Testing order creation...")
        order_request = OrderRequest(
            source="test",
            restaurant="Test Pizza Palace",
            items=[{"name": "Margherita Pizza", "quantity": 1, "price": 15.99}],
            total=15.99,
            delivery_address="123 Test Street, Test City, TC 12345",
            customer_phone="+1-555-0123"
        )
        
        order_response = client.process_order(order_request)
        if order_response.status == "success":
            print("   ✅ Order creation successful")
            print(f"   📦 Order ID: {order_response.order_id}")
            print(f"   ⏰ Delivery: {order_response.estimated_delivery}")
        else:
            print(f"   ❌ Order creation failed: {order_response.error_message}")
        
        # Stop server
        client.stop_server()
        
        print("\n🎉 MCP client integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_mcp_client()
    sys.exit(0 if success else 1)
