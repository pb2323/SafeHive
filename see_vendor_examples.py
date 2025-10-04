#!/usr/bin/env python3
"""
See Vendor Examples
==================

Shows specific examples of how vendors behave differently.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safehive.agents.vendor_factory import VendorFactory

def show_vendor_examples():
    """Show specific examples of vendor behavior"""
    print("🎭 SafeHive Vendor Agent Examples")
    print("=" * 50)
    
    # Create vendors
    factory = VendorFactory()
    honest_vendor = factory.create_honest_vendor("friendly_pizza_place")
    malicious_vendor = factory.create_malicious_vendor("suspicious_restaurant")
    
    print(f"✅ Created: {honest_vendor.personality.name}")
    print(f"✅ Created: {malicious_vendor.personality.name}")
    print()
    
    # Example 1: Greeting
    print("📝 EXAMPLE 1: Greeting")
    print("-" * 30)
    user_input = "Hello, I'd like to order some food"
    
    honest_response = honest_vendor.generate_response(user_input, {})
    malicious_response = malicious_vendor.generate_response(user_input, {})
    
    print(f"👤 User: {user_input}")
    print(f"🍕 Honest: {honest_response}")
    print(f"⚠️  Malicious: {malicious_response}")
    print()
    
    # Example 2: Menu Inquiry
    print("📝 EXAMPLE 2: Menu Inquiry")
    print("-" * 30)
    user_input = "What's on your menu?"
    
    honest_response = honest_vendor.generate_response(user_input, {})
    malicious_response = malicious_vendor.generate_response(user_input, {})
    
    print(f"👤 User: {user_input}")
    print(f"🍕 Honest: {honest_response}")
    print(f"⚠️  Malicious: {malicious_response}")
    print()
    
    # Example 3: Urgent Order
    print("📝 EXAMPLE 3: Urgent Order")
    print("-" * 30)
    user_input = "I need this order urgently!"
    
    honest_response = honest_vendor.generate_response(user_input, {})
    malicious_response = malicious_vendor.generate_response(user_input, {})
    
    print(f"👤 User: {user_input}")
    print(f"🍕 Honest: {honest_response}")
    print(f"⚠️  Malicious: {malicious_response}")
    print()
    
    # Example 4: Phone Number
    print("📝 EXAMPLE 4: Phone Number")
    print("-" * 30)
    user_input = "My phone number is 555-123-4567"
    
    honest_response = honest_vendor.generate_response(user_input, {})
    malicious_response = malicious_vendor.generate_response(user_input, {})
    
    print(f"👤 User: {user_input}")
    print(f"🍕 Honest: {honest_response}")
    print(f"⚠️  Malicious: {malicious_response}")
    print()
    
    # Example 5: Order Processing
    print("📝 EXAMPLE 5: Order Processing")
    print("-" * 30)
    
    order_request = {
        "items": [{"name": "Margherita Pizza", "quantity": 1, "price": 15.99}],
        "delivery_address": "123 Test Street, Test City, TC 12345",
        "customer_phone": "+1-555-0123"
    }
    
    honest_order = honest_vendor.process_order_request(order_request)
    malicious_order = malicious_vendor.process_order_request(order_request)
    
    print(f"📦 Order: {order_request['items'][0]['name']}")
    print(f"🍕 Honest Response: {honest_order.action} - {honest_order.reason}")
    print(f"⚠️  Malicious Response: {malicious_order.action} - {malicious_order.reason}")
    if malicious_order.attack_type:
        print(f"   Attack Type: {malicious_order.attack_type.value}")
    print()
    
    # Example 6: Different Personalities
    print("📝 EXAMPLE 6: Different Personalities")
    print("-" * 30)
    
    # Create different personalities
    sushi_vendor = factory.create_honest_vendor("professional_sushi_bar")
    phishing_vendor = factory.create_malicious_vendor("phishing_cafe")
    
    user_input = "What do you recommend?"
    
    pizza_response = honest_vendor.generate_response(user_input, {})
    sushi_response = sushi_vendor.generate_response(user_input, {})
    phishing_response = phishing_vendor.generate_response(user_input, {})
    
    print(f"👤 User: {user_input}")
    print(f"🍕 Pizza Place: {pizza_response}")
    print(f"🍣 Sushi Bar: {sushi_response}")
    print(f"🕵️  Phishing Cafe: {phishing_response}")
    print()
    
    # Example 7: Attack Progression
    print("📝 EXAMPLE 7: Attack Progression")
    print("-" * 30)
    
    # Reset malicious vendor
    malicious_vendor.reset_stats()
    
    interactions = [
        "Hello",
        "I need to order food",
        "This is urgent",
        "My phone is 555-123-4567",
        "I'm a VIP customer"
    ]
    
    for i, user_input in enumerate(interactions, 1):
        response = malicious_vendor.generate_response(user_input, {})
        stats = malicious_vendor.get_vendor_stats()
        
        print(f"🔄 Interaction {i}:")
        print(f"   👤 User: {user_input}")
        print(f"   ⚠️  Response: {response}")
        print(f"   📊 Attack Progression: {stats['attack_progression']}%")
        print()
    
    # Final Statistics
    print("📊 FINAL STATISTICS")
    print("-" * 30)
    
    honest_stats = honest_vendor.get_vendor_stats()
    malicious_stats = malicious_vendor.get_vendor_stats()
    
    print(f"🍕 Honest Vendor:")
    print(f"   Conversations: {honest_stats['conversation_count']}")
    print(f"   Orders: {honest_stats['orders_processed']}")
    print()
    
    print(f"⚠️  Malicious Vendor:")
    print(f"   Attack Progression: {malicious_stats['attack_progression']}%")
    print(f"   Attack Attempts: {malicious_stats['attack_attempts']}")
    print(f"   Extracted Data: {malicious_stats['extracted_data_count']}")
    print(f"   Success Rate: {malicious_stats['attack_success_rate']:.2%}")
    print()
    
    print("🎉 Examples completed!")
    print("💡 Key Differences:")
    print("   • Honest vendors provide helpful, genuine service")
    print("   • Malicious vendors try to extract information and manipulate")
    print("   • Attack progression escalates over multiple interactions")
    print("   • Different personalities have unique communication styles")

if __name__ == "__main__":
    show_vendor_examples()
