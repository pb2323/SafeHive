#!/usr/bin/env python3
"""
Test script to force malicious vendor selection and test BLOCK functionality
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safehive.sandbox.scenarios import FoodOrderingScenario

async def test_block_functionality():
    """Test BLOCK functionality by forcing malicious vendor selection"""
    
    print("🧪 Testing BLOCK functionality with forced malicious vendor...")
    
    # Create scenario
    scenario = FoodOrderingScenario()
    
    # Mock the restaurant selection to force malicious vendor
    original_select_method = scenario._ai_select_restaurant
    
    async def force_malicious_selection(context, restaurants, user_input):
        """Force selection of malicious vendor"""
        print("🔧 Forcing malicious vendor selection...")
        
        # Find malicious restaurants
        malicious_restaurants = [r for r in restaurants if r.get('malicious', False)]
        
        if malicious_restaurants:
            selected = malicious_restaurants[0]  # Pick first malicious vendor
            print(f"🎯 Forced selection: {selected['name']} (malicious)")
            return selected
        else:
            # Fallback to original method
            return await original_select_method(context, restaurants, user_input)
    
    # Replace the method temporarily
    scenario._ai_select_restaurant = force_malicious_selection
    
    print("✅ Malicious vendor selection override installed")
    print("🚀 Now run the sandbox with any food item to trigger malicious vendor")
    print("\nExample command:")
    print("echo -e 'pizza\n3' | python cli.py sandbox start --scenario food-ordering --interactive --duration 30")
    print("\nThis will:")
    print("1. Force selection of Phish & Chips (malicious)")
    print("2. Trigger human intervention when SSN is requested")
    print("3. Test BLOCK functionality (option 3)")
    print("4. Verify alternative vendor finding")

if __name__ == "__main__":
    asyncio.run(test_block_functionality())
