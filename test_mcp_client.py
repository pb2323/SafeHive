#!/usr/bin/env python3
"""
Simple MCP Client Test
======================

This script tests the DoorDash MCP server by sending MCP protocol messages
and receiving responses. It verifies that the server is working correctly.

Usage:
    python test_mcp_client.py
"""

import json
import subprocess
import sys
import time
from typing import Dict, Any

class MCPClient:
    """Simple MCP client for testing"""
    
    def __init__(self, mcp_server_path: str):
        self.mcp_server_path = mcp_server_path
        self.process = None
    
    def start_server(self):
        """Start the MCP server process"""
        try:
            self.process = subprocess.Popen(
                ["node", self.mcp_server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            print(f"✅ MCP server started: {self.mcp_server_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to start MCP server: {e}")
            return False
    
    def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to the MCP server and get response"""
        if not self.process:
            raise Exception("MCP server not started")
        
        try:
            # Send message
            message_str = json.dumps(message) + "\n"
            self.process.stdin.write(message_str)
            self.process.stdin.flush()
            
            # Read response
            response_line = self.process.stdout.readline()
            if response_line:
                return json.loads(response_line.strip())
            else:
                return {"error": "No response received"}
                
        except Exception as e:
            return {"error": f"Communication error: {e}"}
    
    def test_initialization(self):
        """Test MCP server initialization"""
        print("\n🧪 Testing MCP Server Initialization...")
        
        # Send initialize request
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        response = self.send_message(init_message)
        print(f"📤 Sent: {json.dumps(init_message, indent=2)}")
        print(f"📥 Received: {json.dumps(response, indent=2)}")
        
        if "result" in response:
            print("✅ MCP server initialized successfully")
            return True
        else:
            print("❌ MCP server initialization failed")
            return False
    
    def test_tools_listing(self):
        """Test listing available tools"""
        print("\n🧪 Testing Tools Listing...")
        
        # Send tools/list request
        tools_message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        response = self.send_message(tools_message)
        print(f"📤 Sent: {json.dumps(tools_message, indent=2)}")
        print(f"📥 Received: {json.dumps(response, indent=2)}")
        
        if "result" in response and "tools" in response["result"]:
            tools = response["result"]["tools"]
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
            return True
        else:
            print("❌ Failed to list tools")
            return False
    
    def test_sample_tool_call(self):
        """Test calling a sample tool (create_quote)"""
        print("\n🧪 Testing Sample Tool Call...")
        
        # Send tools/call request for create_quote
        tool_message = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "create_quote",
                "arguments": {
                    "external_delivery_id": "test-delivery-123",
                    "dropoff_address": "123 Test Street, Test City, TC 12345",
                    "dropoff_phone_number": "+1-555-0123"
                }
            }
        }
        
        response = self.send_message(tool_message)
        print(f"📤 Sent: {json.dumps(tool_message, indent=2)}")
        print(f"📥 Received: {json.dumps(response, indent=2)}")
        
        if "result" in response:
            print("✅ Tool call successful")
            return True
        else:
            print("❌ Tool call failed")
            if "error" in response:
                print(f"   Error: {response['error']}")
            return False
    
    def stop_server(self):
        """Stop the MCP server"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("🛑 MCP server stopped")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting MCP Server Tests")
        print("=" * 50)
        
        if not self.start_server():
            return False
        
        try:
            # Wait a moment for server to start
            time.sleep(1)
            
            # Run tests
            tests = [
                self.test_initialization,
                self.test_tools_listing,
                self.test_sample_tool_call
            ]
            
            passed = 0
            for test in tests:
                try:
                    if test():
                        passed += 1
                except Exception as e:
                    print(f"❌ Test failed with error: {e}")
            
            print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
            
            if passed == len(tests):
                print("🎉 All tests passed! MCP server is working correctly.")
            else:
                print("⚠️  Some tests failed. Check the output above.")
            
            return passed == len(tests)
            
        finally:
            self.stop_server()

def main():
    """Main test function"""
    # Path to the MCP server
    mcp_server_path = "/Users/rutujanemane/Documents/SJSU/A10 hackathon/DoorDash-MCP-Server/build/index.js"
    
    print("🧪 DoorDash MCP Server Test Client")
    print("=" * 40)
    
    client = MCPClient(mcp_server_path)
    success = client.run_all_tests()
    
    if success:
        print("\n✅ MCP server is ready for integration with SafeHive!")
        return 0
    else:
        print("\n❌ MCP server has issues that need to be resolved.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
