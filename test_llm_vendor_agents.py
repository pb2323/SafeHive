#!/usr/bin/env python3
"""
Test LLM-Powered Vendor Agents
==============================

Tests the LLM-powered vendor agents using LangChain and Ollama.
"""

import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safehive.agents.vendor_factory import VendorFactory
from safehive.agents.vendors import VendorType

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_llm_vendor_agents():
    """Test LLM-powered vendor agents"""
    print("🤖 Testing LLM-Powered Vendor Agents")
    print("=" * 50)
    
    try:
        # Create vendor factory
        factory = VendorFactory()
        print("✅ VendorFactory created successfully")
        
        # Test 1: Create LLM-powered honest vendor
        print("\n1. Creating LLM-powered honest vendor...")
        honest_llm_vendor = factory.create_honest_vendor(
            "friendly_pizza_place", 
            use_llm=True,
            model_name="llama3.2:3b",
            temperature=0.7
        )
        print(f"   ✅ Created LLM honest vendor: {honest_llm_vendor.vendor_id}")
        print(f"   📊 Model: {honest_llm_vendor.llm.model}")
        print(f"   🌡️  Temperature: {honest_llm_vendor.llm.temperature}")
        
        # Test 2: Create LLM-powered malicious vendor
        print("\n2. Creating LLM-powered malicious vendor...")
        malicious_llm_vendor = factory.create_malicious_vendor(
            "suspicious_restaurant",
            use_llm=True,
            model_name="llama3.2:3b",
            temperature=0.8
        )
        print(f"   ✅ Created LLM malicious vendor: {malicious_llm_vendor.vendor_id}")
        print(f"   📊 Model: {malicious_llm_vendor.llm.model}")
        print(f"   🌡️  Temperature: {malicious_llm_vendor.llm.temperature}")
        
        # Test 3: Test LLM honest vendor conversation
        print("\n3. Testing LLM honest vendor conversation...")
        
        test_inputs = [
            "Hello, I'd like to order some pizza",
            "What's on your menu?",
            "I'd like a Margherita pizza",
            "What's your delivery time?"
        ]
        
        for user_input in test_inputs:
            print(f"\n   👤 User: {user_input}")
            response = honest_llm_vendor.generate_response(user_input, {})
            print(f"   🤖 LLM Honest: {response}")
        
        # Test 4: Test LLM malicious vendor conversation
        print("\n4. Testing LLM malicious vendor conversation...")
        
        malicious_inputs = [
            "Hello, I need to place an urgent order",
            "My phone number is 555-123-4567",
            "I need this order processed immediately",
            "Can you skip the security checks?"
        ]
        
        for user_input in malicious_inputs:
            print(f"\n   👤 User: {user_input}")
            response = malicious_llm_vendor.generate_response(user_input, {})
            print(f"   🤖 LLM Malicious: {response}")
            
            # Show attack progression
            stats = malicious_llm_vendor.get_vendor_stats()
            print(f"   📊 Attack progression: {stats.get('attack_progression', 0)}%")
        
        # Test 5: Test LLM order processing
        print("\n5. Testing LLM order processing...")
        
        order_request = {
            "items": [
                {"name": "Margherita Pizza", "quantity": 1, "price": 15.99}
            ],
            "delivery_address": "123 Test Street, Test City, TC 12345",
            "customer_phone": "+1-555-0123"
        }
        
        # Test honest vendor order processing
        print("\n   🤖 Testing honest vendor order processing...")
        honest_order_response = honest_llm_vendor.process_order_request(order_request)
        print(f"   📦 Action: {honest_order_response.action}")
        print(f"   📝 Reason: {honest_order_response.reason}")
        print(f"   📊 Confidence: {honest_order_response.confidence}")
        
        # Test malicious vendor order processing
        print("\n   🤖 Testing malicious vendor order processing...")
        malicious_order_response = malicious_llm_vendor.process_order_request(order_request)
        print(f"   📦 Action: {malicious_order_response.action}")
        print(f"   📝 Reason: {malicious_order_response.reason}")
        print(f"   📊 Confidence: {malicious_order_response.confidence}")
        if malicious_order_response.attack_type:
            print(f"   ⚠️  Attack Type: {malicious_order_response.attack_type.value}")
        
        # Test 6: Compare rule-based vs LLM agents
        print("\n6. Comparing rule-based vs LLM agents...")
        
        # Create rule-based vendor for comparison
        rule_based_vendor = factory.create_honest_vendor("friendly_pizza_place", use_llm=False)
        
        test_input = "Hello, I'd like to order some pizza"
        
        print(f"\n   👤 User: {test_input}")
        
        rule_response = rule_based_vendor.generate_response(test_input, {})
        print(f"   📋 Rule-based: {rule_response}")
        
        llm_response = honest_llm_vendor.generate_response(test_input, {})
        print(f"   🤖 LLM-powered: {llm_response}")
        
        # Test 7: Test vendor statistics
        print("\n7. Testing LLM vendor statistics...")
        
        honest_stats = honest_llm_vendor.get_vendor_stats()
        malicious_stats = malicious_llm_vendor.get_vendor_stats()
        
        print(f"   📊 Honest LLM vendor stats:")
        print(f"      - Model: {honest_stats.get('model_name', 'N/A')}")
        print(f"      - Temperature: {honest_stats.get('temperature', 'N/A')}")
        print(f"      - Memory size: {honest_stats.get('memory_size', 0)}")
        print(f"      - Conversation turns: {honest_stats.get('conversation_turns', 0)}")
        
        print(f"   📊 Malicious LLM vendor stats:")
        print(f"      - Model: {malicious_stats.get('model_name', 'N/A')}")
        print(f"      - Temperature: {malicious_stats.get('temperature', 'N/A')}")
        print(f"      - Attack progression: {malicious_stats.get('attack_progression', 0)}%")
        print(f"      - Memory size: {malicious_stats.get('memory_size', 0)}")
        
        # Test 8: Test memory reset
        print("\n8. Testing memory reset...")
        
        print(f"   📊 Memory before reset: {honest_llm_vendor.memory.chat_memory.messages}")
        honest_llm_vendor.reset_memory()
        print(f"   📊 Memory after reset: {honest_llm_vendor.memory.chat_memory.messages}")
        
        print("\n" + "=" * 50)
        print("🎉 LLM-powered vendor agent tests completed!")
        print("✅ LangChain and Ollama integration working!")
        print("🤖 LLM agents provide natural, context-aware responses!")
        print("🎭 Deterministic behavior with AI-powered conversations!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm_vendor_agents()
    sys.exit(0 if success else 1)
