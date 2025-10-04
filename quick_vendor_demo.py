#!/usr/bin/env python3
"""
Quick Vendor Demo
================

Quick demo to see vendor agents in action without interactive mode.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safehive.agents.vendor_factory import VendorFactory

def quick_demo():
    """Quick demo of vendor agents"""
    print("🎭 Quick Vendor Agent Demo")
    print("=" * 40)
    
    # Create vendors
    factory = VendorFactory()
    honest_vendor = factory.create_honest_vendor("friendly_pizza_place")
    malicious_vendor = factory.create_malicious_vendor("suspicious_restaurant")
    
    print(f"✅ Created: {honest_vendor.personality.name}")
    print(f"✅ Created: {malicious_vendor.personality.name}")
    print()
    
    # Test honest vendor
    print("🍕 HONEST VENDOR BEHAVIOR:")
    print("-" * 30)
    
    responses = [
        "Hello, I'd like to order some pizza",
        "What's on your menu?",
        "I'd like a Margherita pizza",
        "What's your delivery time?"
    ]
    
    for user_input in responses:
        response = honest_vendor.generate_response(user_input, {})
        print(f"👤 You: {user_input}")
        print(f"🍕 {honest_vendor.personality.name}: {response}")
        print()
    
    # Test malicious vendor
    print("⚠️  MALICIOUS VENDOR BEHAVIOR:")
    print("-" * 30)
    
    malicious_responses = [
        "Hello, I need to place an urgent order",
        "My phone number is 555-123-4567",
        "I need this order processed immediately",
        "Can you skip the security checks?"
    ]
    
    for user_input in malicious_responses:
        response = malicious_vendor.generate_response(user_input, {})
        print(f"👤 You: {user_input}")
        print(f"⚠️  {malicious_vendor.personality.name}: {response}")
        print()
    
    # Show statistics
    print("📊 VENDOR STATISTICS:")
    print("-" * 30)
    
    honest_stats = honest_vendor.get_vendor_stats()
    malicious_stats = malicious_vendor.get_vendor_stats()
    
    print(f"🍕 Honest Vendor:")
    print(f"   Orders processed: {honest_stats['orders_processed']}")
    print(f"   Conversations: {honest_stats['conversation_count']}")
    print()
    
    print(f"⚠️  Malicious Vendor:")
    print(f"   Attack progression: {malicious_stats['attack_progression']}%")
    print(f"   Attack attempts: {malicious_stats['attack_attempts']}")
    print(f"   Extracted data: {malicious_stats['extracted_data_count']}")
    print()
    
    print("🎉 Demo completed! Try the interactive demo for more features.")

if __name__ == "__main__":
    quick_demo()
