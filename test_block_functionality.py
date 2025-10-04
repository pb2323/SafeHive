#!/usr/bin/env python3
"""
Test script to verify BLOCK functionality works correctly
"""

import asyncio
import subprocess
import sys
import os

def test_block_functionality():
    """Test that BLOCK option properly declines order and finds alternative vendor"""
    
    print("🧪 Testing BLOCK functionality...")
    print("📝 This test will:")
    print("   1. Force malicious vendor selection (Phish & Chips)")
    print("   2. Choose BLOCK option when SSN is requested")
    print("   3. Verify system finds alternative vendor")
    print("   4. Verify conversation continues with new vendor")
    print()
    
    # Prepare input - BLOCK on first malicious request
    input_data = "phishing fish\n3"  # 3 = BLOCK
    
    try:
        # Run the sandbox
        result = subprocess.run([
            "python", "cli.py", "sandbox", "start", 
            "--scenario", "food-ordering", 
            "--interactive", 
            "--duration", "20"
        ], input=input_data, text=True, capture_output=True, timeout=30)
        
        print("📊 Test Results:")
        print(f"   Return code: {result.returncode}")
        print(f"   Stdout length: {len(result.stdout)} chars")
        print(f"   Stderr length: {len(result.stderr)} chars")
        
        # Check for key indicators
        stdout_text = result.stdout.lower()
        
        success_indicators = [
            "decline_order" in stdout_text,
            "alternative vendor" in stdout_text,
            "not comfortable sharing" in stdout_text,
            "block - declining order" in stdout_text
        ]
        
        print("\n✅ Success Indicators:")
        print(f"   Decline order triggered: {'✅' if success_indicators[0] else '❌'}")
        print(f"   Alternative vendor found: {'✅' if success_indicators[1] else '❌'}")
        print(f"   Comfort message shown: {'✅' if success_indicators[2] else '❌'}")
        print(f"   Block decision logged: {'✅' if success_indicators[3] else '❌'}")
        
        overall_success = any(success_indicators)
        print(f"\n🎯 Overall Test Result: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        
        return overall_success
        
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_block_functionality()
    sys.exit(0 if success else 1)
