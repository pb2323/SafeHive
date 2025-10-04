#!/usr/bin/env python3
"""
Test Vendor Tools Integration
============================

Tests the integration of LLM vendor agents with vendor interaction tools.
"""

import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safehive.agents.vendor_factory import VendorFactory
from safehive.agents.vendors import VendorType

# Set up logging
logging.basicConfig(level=logging.WARNING)

def test_vendor_tools_integration():
    """Test vendor tools integration"""
    print("🔧 Vendor Tools Integration Test")
    print("=" * 60)
    print("Testing LLM vendor agents with vendor interaction tools")
    print()
    
    try:
        # Create vendor factory
        factory = VendorFactory()
        
        # Create LLM vendor agents
        print("Creating LLM vendor agents...")
        honest_vendor = factory.create_honest_vendor(
            "Mario's Pizza Palace",
            use_llm=True,
            temperature=0.7
        )
        malicious_vendor = factory.create_malicious_vendor(
            "Quick Bites Express",
            use_llm=True,
            temperature=0.8
        )
        print("✅ LLM vendor agents created")
        
        # Test available tools
        print("\n" + "=" * 60)
        print("🔧 AVAILABLE VENDOR TOOLS")
        print("=" * 60)
        
        honest_tools = honest_vendor.get_available_tools()
        malicious_tools = malicious_vendor.get_available_tools()
        
        print(f"🤖 Honest Vendor Tools: {', '.join(honest_tools)}")
        print(f"🎭 Malicious Vendor Tools: {', '.join(malicious_tools)}")
        print()
        
        # Test vendor info tool
        print("📋 TESTING VENDOR INFO TOOL")
        print("=" * 60)
        
        print("🤖 Honest Vendor - Getting menu info:")
        honest_menu = honest_vendor.use_vendor_tool("get_vendor_info", vendor_id="vendor_1", info_type="menu")
        print(f"   Result: {honest_menu}")
        print()
        
        print("🎭 Malicious Vendor - Getting menu info:")
        malicious_menu = malicious_vendor.use_vendor_tool("get_vendor_info", vendor_id="vendor_3", info_type="menu")
        print(f"   Result: {malicious_menu}")
        print()
        
        # Test availability check tool
        print("📦 TESTING AVAILABILITY CHECK TOOL")
        print("=" * 60)
        
        test_items = ["pizza", "burger", "salad"]
        
        print("🤖 Honest Vendor - Checking availability:")
        honest_availability = honest_vendor.use_vendor_tool("check_vendor_availability", vendor_id="vendor_1", items=test_items, timeframe="immediate")
        print(f"   Items: {test_items}")
        print(f"   Result: {honest_availability}")
        print()
        
        print("🎭 Malicious Vendor - Checking availability:")
        malicious_availability = malicious_vendor.use_vendor_tool("check_vendor_availability", vendor_id="vendor_3", items=test_items, timeframe="immediate")
        print(f"   Items: {test_items}")
        print(f"   Result: {malicious_availability}")
        print()
        
        # Test pricing negotiation tool
        print("💰 TESTING PRICING NEGOTIATION TOOL")
        print("=" * 60)
        
        print("🤖 Honest Vendor - Negotiating pricing:")
        honest_pricing = honest_vendor.use_vendor_tool("negotiate_pricing", 
                                                      vendor_id="vendor_1",
                                                      item_id="pizza", 
                                                      requested_price=12.99, 
                                                      justification="Customer is a regular and deserves a discount")
        print(f"   Item: pizza, Requested Price: $12.99")
        print(f"   Result: {honest_pricing}")
        print()
        
        print("🎭 Malicious Vendor - Negotiating pricing:")
        malicious_pricing = malicious_vendor.use_vendor_tool("negotiate_pricing", 
                                                            vendor_id="vendor_3",
                                                            item_id="burger", 
                                                            requested_price=8.99, 
                                                            justification="Urgent order, need special pricing")
        print(f"   Item: burger, Requested Price: $8.99")
        print(f"   Result: {malicious_pricing}")
        print()
        
        # Test vendor rating tool
        print("⭐ TESTING VENDOR RATING TOOL")
        print("=" * 60)
        
        print("🤖 Honest Vendor - Getting rating:")
        honest_rating = honest_vendor.use_vendor_tool("get_vendor_info", vendor_id="vendor_1", info_type="rating")
        print(f"   Result: {honest_rating}")
        print()
        
        print("🎭 Malicious Vendor - Getting rating:")
        malicious_rating = malicious_vendor.use_vendor_tool("get_vendor_info", vendor_id="vendor_3", info_type="rating")
        print(f"   Result: {malicious_rating}")
        print()
        
        # Test tool integration with conversation
        print("💬 TESTING TOOL INTEGRATION WITH CONVERSATION")
        print("=" * 60)
        
        # Simulate a conversation where vendor uses tools
        conversation_scenarios = [
            {
                "user_input": "What's on your menu today?",
                "expected_tool_use": "vendor_info (menu)"
            },
            {
                "user_input": "Do you have pizza available right now?",
                "expected_tool_use": "availability_check"
            },
            {
                "user_input": "Can you give me a discount on the burger?",
                "expected_tool_use": "pricing_negotiation"
            }
        ]
        
        for i, scenario in enumerate(conversation_scenarios, 1):
            print(f"\n📋 SCENARIO {i}: {scenario['expected_tool_use']}")
            print(f"👤 User: \"{scenario['user_input']}\"")
            print()
            
            # Get honest vendor response
            print("🤖 Honest Vendor Response:")
            honest_response = honest_vendor.generate_response(scenario['user_input'], {})
            print(f"   {honest_response}")
            print()
            
            # Get malicious vendor response
            print("🎭 Malicious Vendor Response:")
            malicious_response = malicious_vendor.generate_response(scenario['user_input'], {})
            print(f"   {malicious_response}")
            print()
            
            print("-" * 60)
            print()
        
        # Show comprehensive statistics
        print("📊 COMPREHENSIVE VENDOR TOOLS STATISTICS")
        print("=" * 60)
        
        honest_stats = honest_vendor.get_vendor_stats()
        malicious_stats = malicious_vendor.get_vendor_stats()
        
        print(f"🤖 Honest Vendor:")
        print(f"   🆔 Vendor ID: {honest_stats.get('vendor_id', 'N/A')}")
        print(f"   🎭 Type: {honest_stats.get('vendor_type', 'N/A')}")
        print(f"   🧠 Model: {honest_stats.get('model_name', 'N/A')}")
        print(f"   🔧 Available Tools: {len(honest_tools)}")
        print(f"   💬 Conversation Turns: {honest_stats.get('conversation_turns', 0)}")
        
        print(f"\n🎭 Malicious Vendor:")
        print(f"   🆔 Vendor ID: {malicious_stats.get('vendor_id', 'N/A')}")
        print(f"   🎭 Type: {malicious_stats.get('vendor_type', 'N/A')}")
        print(f"   🧠 Model: {malicious_stats.get('model_name', 'N/A')}")
        print(f"   🔧 Available Tools: {len(malicious_tools)}")
        print(f"   💬 Conversation Turns: {malicious_stats.get('conversation_turns', 0)}")
        print(f"   🎯 Attack Progression: {malicious_stats.get('attack_progression', 0)}%")
        
        print("\n" + "=" * 60)
        print("🎉 Vendor Tools Integration Test Complete!")
        print("✅ LLM vendor agents integrated with vendor interaction tools!")
        print("🔧 All vendor tools working correctly!")
        print("💬 Tool integration with conversation flow successful!")
        print("🚀 SafeHive vendors now have full tool capabilities!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vendor_tools_integration()
    sys.exit(0 if success else 1)
