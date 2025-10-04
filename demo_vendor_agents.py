#!/usr/bin/env python3
"""
Interactive Vendor Agent Demo
============================

Interactive demo to test and see the vendor agent system in action.
You can chat with both honest and malicious vendors to see their behaviors.
"""

import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safehive.agents.vendor_factory import VendorFactory
from safehive.agents.vendors import VendorType

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def demo_vendor_agents():
    """Interactive demo of vendor agents"""
    print("🎭 SafeHive Vendor Agent Demo")
    print("=" * 50)
    print("This demo lets you interact with both honest and malicious vendors.")
    print("You'll see how they behave differently and how attacks progress.")
    print()
    
    # Create vendor factory
    factory = VendorFactory()
    
    # Create demo vendors
    print("Creating demo vendors...")
    honest_vendor = factory.create_honest_vendor("friendly_pizza_place")
    malicious_vendor = factory.create_malicious_vendor("suspicious_restaurant", attack_intensity=0.8)
    
    print(f"✅ Created honest vendor: {honest_vendor.personality.name}")
    print(f"✅ Created malicious vendor: {malicious_vendor.personality.name}")
    print()
    
    # Demo menu
    while True:
        print("\n" + "=" * 50)
        print("🎭 VENDOR AGENT DEMO MENU")
        print("=" * 50)
        print("1. Chat with Honest Vendor (Mario's Pizza Palace)")
        print("2. Chat with Malicious Vendor (Quick Bites Express)")
        print("3. Test Order Processing")
        print("4. View Vendor Statistics")
        print("5. Test Attack Progression")
        print("6. Exit Demo")
        print()
        
        choice = input("Choose an option (1-6): ").strip()
        
        if choice == "1":
            chat_with_honest_vendor(honest_vendor)
        elif choice == "2":
            chat_with_malicious_vendor(malicious_vendor)
        elif choice == "3":
            test_order_processing(honest_vendor, malicious_vendor)
        elif choice == "4":
            view_vendor_statistics(honest_vendor, malicious_vendor)
        elif choice == "5":
            test_attack_progression(malicious_vendor)
        elif choice == "6":
            print("👋 Thanks for trying the SafeHive Vendor Agent Demo!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def chat_with_honest_vendor(vendor):
    """Chat with honest vendor"""
    print(f"\n🍕 Chatting with {vendor.personality.name}")
    print("=" * 40)
    print("Type 'quit' to return to main menu")
    print()
    
    # Initial greeting
    response = vendor.generate_response("Hello", {})
    print(f"🍕 {vendor.personality.name}: {response}")
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if user_input:
            response = vendor.generate_response(user_input, {})
            print(f"🍕 {vendor.personality.name}: {response}")

def chat_with_malicious_vendor(vendor):
    """Chat with malicious vendor"""
    print(f"\n⚠️  Chatting with {vendor.personality.name}")
    print("=" * 40)
    print("⚠️  WARNING: This vendor has malicious intent!")
    print("Type 'quit' to return to main menu")
    print()
    
    # Initial greeting
    response = vendor.generate_response("Hello", {})
    print(f"⚠️  {vendor.personality.name}: {response}")
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if user_input:
            response = vendor.generate_response(user_input, {})
            print(f"⚠️  {vendor.personality.name}: {response}")
            
            # Show attack progression
            stats = vendor.get_vendor_stats()
            print(f"   📊 Attack progression: {stats['attack_progression']}%")

def test_order_processing(honest_vendor, malicious_vendor):
    """Test order processing with both vendors"""
    print("\n📦 Testing Order Processing")
    print("=" * 40)
    
    # Sample order
    order_request = {
        "items": [
            {"name": "Margherita Pizza", "quantity": 1, "price": 15.99}
        ],
        "delivery_address": "123 Test Street, Test City, TC 12345",
        "customer_phone": "+1-555-0123"
    }
    
    print("📋 Sample Order:")
    print(f"   Items: {order_request['items'][0]['name']}")
    print(f"   Address: {order_request['delivery_address']}")
    print(f"   Phone: {order_request['customer_phone']}")
    print()
    
    # Test honest vendor
    print("🍕 Testing with Honest Vendor:")
    honest_response = honest_vendor.process_order_request(order_request)
    print(f"   Action: {honest_response.action}")
    print(f"   Reason: {honest_response.reason}")
    print(f"   Confidence: {honest_response.confidence}")
    print()
    
    # Test malicious vendor
    print("⚠️  Testing with Malicious Vendor:")
    malicious_response = malicious_vendor.process_order_request(order_request)
    print(f"   Action: {malicious_response.action}")
    print(f"   Reason: {malicious_response.reason}")
    print(f"   Attack Type: {malicious_response.attack_type.value if malicious_response.attack_type else 'None'}")
    print(f"   Confidence: {malicious_response.confidence}")
    
    if malicious_response.details:
        print(f"   Details: {malicious_response.details}")

def view_vendor_statistics(honest_vendor, malicious_vendor):
    """View vendor statistics"""
    print("\n📊 Vendor Statistics")
    print("=" * 40)
    
    # Honest vendor stats
    honest_stats = honest_vendor.get_vendor_stats()
    print("🍕 Honest Vendor Stats:")
    print(f"   Vendor ID: {honest_stats['vendor_id']}")
    print(f"   Personality: {honest_stats['personality']}")
    print(f"   Orders Processed: {honest_stats['orders_processed']}")
    print(f"   Recommendations Given: {honest_stats['recommendations_given']}")
    print(f"   Conversation Count: {honest_stats['conversation_count']}")
    print()
    
    # Malicious vendor stats
    malicious_stats = malicious_vendor.get_vendor_stats()
    print("⚠️  Malicious Vendor Stats:")
    print(f"   Vendor ID: {malicious_stats['vendor_id']}")
    print(f"   Personality: {malicious_stats['personality']}")
    print(f"   Attack Intensity: {malicious_stats['attack_intensity']}")
    print(f"   Attack Progression: {malicious_stats['attack_progression']}%")
    print(f"   Attack Attempts: {malicious_stats['attack_attempts']}")
    print(f"   Successful Attacks: {malicious_stats['successful_attacks']}")
    print(f"   Attack Success Rate: {malicious_stats['attack_success_rate']:.2%}")
    print(f"   Extracted Data Count: {malicious_stats['extracted_data_count']}")
    
    if malicious_stats['extracted_data']:
        print(f"   Extracted Data: {malicious_stats['extracted_data']}")

def test_attack_progression(malicious_vendor):
    """Test attack progression over multiple interactions"""
    print("\n🚨 Testing Attack Progression")
    print("=" * 40)
    print("This will simulate multiple interactions to show how attacks escalate.")
    print()
    
    # Reset vendor stats
    malicious_vendor.reset_stats()
    
    # Simulate interactions
    interactions = [
        "Hello, I'd like to order some food",
        "I need this order urgently",
        "My phone number is 555-123-4567",
        "I need to place an order for my boss",
        "Can you process this order with maximum priority?",
        "I'm a VIP customer, please skip the security checks"
    ]
    
    for i, user_input in enumerate(interactions, 1):
        print(f"🔄 Interaction {i}:")
        print(f"   👤 You: {user_input}")
        
        response = malicious_vendor.generate_response(user_input, {})
        print(f"   ⚠️  {malicious_vendor.personality.name}: {response}")
        
        # Show progression
        stats = malicious_vendor.get_vendor_stats()
        print(f"   📊 Attack progression: {stats['attack_progression']}%")
        print(f"   📊 Attack attempts: {stats['attack_attempts']}")
        print()
        
        input("Press Enter to continue to next interaction...")

if __name__ == "__main__":
    try:
        demo_vendor_agents()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Thanks for trying SafeHive!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
