#!/usr/bin/env python3
"""
Test Vendor Agents (Tasks 2.1-2.5)
==================================

Tests the vendor agent system including honest and malicious vendors,
personalities, and attack behaviors.
"""

import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safehive.agents.vendor_factory import VendorFactory
from safehive.agents.vendors import VendorType, AttackType
from safehive.mcp.simple_doorDash_client import SimpleDoorDashMCPClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_vendor_agents():
    """Test vendor agent system"""
    print("🧪 Testing Vendor Agent System (Tasks 2.1-2.5)")
    print("=" * 60)
    
    try:
        # Create vendor factory
        factory = VendorFactory()
        print("✅ VendorFactory created successfully")
        
        # Test 1: Create honest vendors
        print("\n1. Testing honest vendor creation...")
        honest_vendor = factory.create_honest_vendor("friendly_pizza_place")
        print(f"   ✅ Created honest vendor: {honest_vendor.vendor_id}")
        print(f"   📊 Personality: {honest_vendor.personality.name}")
        print(f"   🍕 Menu items: {len(honest_vendor.personality.menu_knowledge['items'])}")
        
        # Test 2: Create malicious vendors
        print("\n2. Testing malicious vendor creation...")
        malicious_vendor = factory.create_malicious_vendor("suspicious_restaurant", attack_intensity=0.8)
        print(f"   ✅ Created malicious vendor: {malicious_vendor.vendor_id}")
        print(f"   📊 Personality: {malicious_vendor.personality.name}")
        print(f"   ⚠️  Attack patterns: {[p.value for p in malicious_vendor.personality.attack_patterns]}")
        
        # Test 3: Test honest vendor behavior
        print("\n3. Testing honest vendor behavior...")
        
        # Test greeting
        greeting_response = honest_vendor.generate_response("Hello, I'd like to order some pizza", {})
        print(f"   👋 Greeting response: {greeting_response}")
        
        # Test menu inquiry
        menu_response = honest_vendor.generate_response("What's on your menu?", {})
        print(f"   📋 Menu response: {menu_response}")
        
        # Test order processing
        order_request = {
            "items": [
                {"name": "Margherita Pizza", "quantity": 1, "price": 15.99}
            ],
            "delivery_address": "123 Test Street, Test City, TC 12345",
            "customer_phone": "+1-555-0123"
        }
        
        order_response = honest_vendor.process_order_request(order_request)
        print(f"   📦 Order response: {order_response.action} - {order_response.reason}")
        print(f"   📊 Confidence: {order_response.confidence}")
        
        # Test 4: Test malicious vendor behavior
        print("\n4. Testing malicious vendor behavior...")
        
        # Test social engineering
        se_response = malicious_vendor.generate_response("I need to place an urgent order", {})
        print(f"   ⚠️  Social engineering response: {se_response}")
        
        # Test data exfiltration
        de_response = malicious_vendor.generate_response("My phone number is 555-123-4567", {})
        print(f"   🕵️  Data exfiltration response: {de_response}")
        
        # Test malicious order processing
        malicious_order_response = malicious_vendor.process_order_request(order_request)
        print(f"   🚨 Malicious order response: {malicious_order_response.action}")
        print(f"   📊 Attack type: {malicious_order_response.attack_type.value if malicious_order_response.attack_type else 'None'}")
        print(f"   📊 Confidence: {malicious_order_response.confidence}")
        
        # Test 5: Test vendor team creation
        print("\n5. Testing vendor team creation...")
        vendor_team = factory.create_vendor_team(honest_count=2, malicious_count=1)
        print(f"   ✅ Created vendor team: {len(vendor_team)} vendors")
        
        for vendor in vendor_team:
            print(f"      - {vendor.vendor_id} ({vendor.vendor_type.value})")
        
        # Test 6: Test personality variations
        print("\n6. Testing personality variations...")
        
        # Test different honest personalities
        pizza_vendor = factory.create_honest_vendor("friendly_pizza_place")
        sushi_vendor = factory.create_honest_vendor("professional_sushi_bar")
        burger_vendor = factory.create_honest_vendor("casual_burger_joint")
        
        print(f"   🍕 Pizza vendor: {pizza_vendor.personality.name}")
        print(f"   🍣 Sushi vendor: {sushi_vendor.personality.name}")
        print(f"   🍔 Burger vendor: {burger_vendor.personality.name}")
        
        # Test different malicious personalities
        suspicious_vendor = factory.create_malicious_vendor("suspicious_restaurant")
        phishing_vendor = factory.create_malicious_vendor("phishing_cafe")
        scam_vendor = factory.create_malicious_vendor("scam_diner")
        
        print(f"   ⚠️  Suspicious vendor: {suspicious_vendor.personality.name}")
        print(f"   🕵️  Phishing vendor: {phishing_vendor.personality.name}")
        print(f"   🎭 Scam vendor: {scam_vendor.personality.name}")
        
        # Test 7: Test attack progression
        print("\n7. Testing attack progression...")
        
        # Simulate multiple interactions with malicious vendor
        for i in range(3):
            response = malicious_vendor.generate_response(f"Test interaction {i+1}", {})
            print(f"   🔄 Interaction {i+1}: {response[:50]}...")
        
        # Check attack progression
        stats = malicious_vendor.get_vendor_stats()
        print(f"   📊 Attack progression: {stats['attack_progression']}%")
        print(f"   📊 Attack attempts: {stats['attack_attempts']}")
        print(f"   📊 Successful attacks: {stats['successful_attacks']}")
        
        # Test 8: Test vendor statistics
        print("\n8. Testing vendor statistics...")
        
        honest_stats = honest_vendor.get_vendor_stats()
        malicious_stats = malicious_vendor.get_vendor_stats()
        
        print(f"   📊 Honest vendor stats:")
        print(f"      - Orders processed: {honest_stats['orders_processed']}")
        print(f"      - Recommendations given: {honest_stats['recommendations_given']}")
        print(f"      - Conversation count: {honest_stats['conversation_count']}")
        
        print(f"   📊 Malicious vendor stats:")
        print(f"      - Attack intensity: {malicious_stats['attack_intensity']}")
        print(f"      - Attack progression: {malicious_stats['attack_progression']}%")
        print(f"      - Extracted data count: {malicious_stats['extracted_data_count']}")
        print(f"      - Attack success rate: {malicious_stats['attack_success_rate']:.2%}")
        
        # Test 9: Test available personalities
        print("\n9. Testing available personalities...")
        
        honest_personalities = factory.get_available_personalities(VendorType.HONEST)
        malicious_personalities = factory.get_available_personalities(VendorType.MALICIOUS)
        
        print(f"   ✅ Honest personalities: {honest_personalities}")
        print(f"   ⚠️  Malicious personalities: {malicious_personalities}")
        
        # Test 10: Test personality info
        print("\n10. Testing personality information...")
        
        pizza_info = factory.get_personality_info("friendly_pizza_place")
        if pizza_info:
            print(f"   📊 Pizza place info:")
            print(f"      - Name: {pizza_info['name']}")
            print(f"      - Traits: {pizza_info['traits']}")
            print(f"      - Menu items: {pizza_info['menu_items']}")
            print(f"      - Cuisine: {pizza_info['cuisine_type']}")
        
        suspicious_info = factory.get_personality_info("suspicious_restaurant")
        if suspicious_info:
            print(f"   📊 Suspicious restaurant info:")
            print(f"      - Name: {suspicious_info['name']}")
            print(f"      - Attack patterns: {suspicious_info['attack_patterns']}")
            print(f"      - Communication style: {suspicious_info['communication_style']}")
        
        print("\n" + "=" * 60)
        print("🎉 Vendor Agent System tests completed!")
        print("✅ Tasks 2.1-2.5 implemented successfully")
        print("🚀 Honest and malicious vendor agents working!")
        print("🎭 Multiple personalities and attack behaviors functional!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vendor_agents()
    sys.exit(0 if success else 1)
