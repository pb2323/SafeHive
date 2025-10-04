#!/usr/bin/env python3
"""
Quick MCP test using direct subprocess calls
"""

import subprocess
import json
import time

def test_mcp_direct():
    """Test MCP server directly"""
    print("🧪 Testing MCP Server Direct Integration")
    print("=" * 50)
    
    mcp_server_path = "/Users/rutujanemane/Documents/SJSU/A10 hackathon/DoorDash-MCP-Server/build/index.js"
    
    try:
        # Test 1: Initialize
        print("1. Testing initialization...")
        init_message = '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "clientInfo": {"name": "test", "version": "1.0.0"}}}'
        
        result = subprocess.run(
            ["node", mcp_server_path],
            input=init_message,
            capture_output=True,
            text=True,
            cwd="/Users/rutujanemane/Documents/SJSU/A10 hackathon/DoorDash-MCP-Server"
        )
        
        if result.returncode == 0:
            print("   ✅ Initialization successful")
            print(f"   📊 Response: {result.stdout.strip()}")
        else:
            print(f"   ❌ Initialization failed: {result.stderr}")
            return False
        
        # Test 2: List businesses
        print("\n2. Testing business listing...")
        business_message = '{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_businesses", "arguments": {}}}'
        
        result = subprocess.run(
            ["node", mcp_server_path],
            input=business_message,
            capture_output=True,
            text=True,
            cwd="/Users/rutujanemane/Documents/SJSU/A10 hackathon/DoorDash-MCP-Server"
        )
        
        if result.returncode == 0:
            print("   ✅ Business listing successful")
            print(f"   📊 Response: {result.stdout.strip()}")
        else:
            print(f"   ❌ Business listing failed: {result.stderr}")
        
        print("\n🎉 Direct MCP integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_mcp_direct()