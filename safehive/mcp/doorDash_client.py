#!/usr/bin/env python3
"""
DoorDash MCP Client
===================

Python wrapper for communicating with the DoorDash MCP server.
Handles JSON-RPC protocol communication and order management.
"""

import json
import os
import subprocess
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class OrderRequest:
    """Order request data structure"""
    source: str
    restaurant: str
    items: List[Dict[str, Any]]
    total: float
    delivery_address: str
    customer_phone: Optional[str] = None
    pickup_address: Optional[str] = None
    pickup_business_name: Optional[str] = None
    pickup_phone_number: Optional[str] = None

@dataclass
class OrderResponse:
    """Order response data structure"""
    status: str  # "success" or "error"
    order_id: Optional[str] = None
    estimated_delivery: Optional[str] = None
    confirmation: Optional[str] = None
    error_message: Optional[str] = None

class DoorDashMCPClient:
    """Python client for DoorDash MCP server integration"""
    
    def __init__(self, mcp_server_path: str, environment: str = "sandbox"):
        self.mcp_server_path = mcp_server_path
        self.environment = environment
        self.process = None
        self.message_id = 0
        
        # Validate MCP server path
        if not Path(mcp_server_path).exists():
            raise FileNotFoundError(f"MCP server not found at: {mcp_server_path}")
    
    def _get_next_id(self) -> int:
        """Get next message ID for JSON-RPC"""
        self.message_id += 1
        return self.message_id
    
    def _start_server(self) -> bool:
        """Start the MCP server process"""
        try:
            if self.process and self.process.poll() is None:
                logger.info("MCP server already running")
                return True
            
            # Start server in a way that it stays running
            self.process = subprocess.Popen(
                ["node", self.mcp_server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                cwd=os.path.dirname(self.mcp_server_path)
            )
            
            # Wait for server to start
            time.sleep(2)
            
            if self.process.poll() is None:
                logger.info(f"MCP server started: {self.mcp_server_path}")
                return True
            else:
                # Check stderr for error details
                stderr_output = self.process.stderr.read() if self.process.stderr else "No error details"
                logger.error(f"MCP server failed to start. Error: {stderr_output}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False
    
    def _send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC message to MCP server"""
        try:
            # Use direct subprocess call for each message
            message_str = json.dumps(message)
            
            # Set up environment with DoorDash credentials
            env = os.environ.copy()
            # .env file is in the root directory, not in build/
            env_path = os.path.join(os.path.dirname(os.path.dirname(self.mcp_server_path)), '.env')
            if os.path.exists(env_path):
                # Load .env file if it exists
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            env[key] = value
                            logger.debug(f"Set env var: {key}")
            
            # Also set the working directory to the MCP server root
            mcp_root = os.path.dirname(os.path.dirname(self.mcp_server_path))
            
            result = subprocess.run(
                ["node", self.mcp_server_path],
                input=message_str,
                capture_output=True,
                text=True,
                cwd=mcp_root,
                env=env
            )
            
            if result.returncode == 0:
                # Parse the response (remove dotenv messages)
                output_lines = result.stdout.strip().split('\n')
                logger.debug(f"MCP server output: {result.stdout}")
                
                for line in output_lines:
                    if line.startswith('{'):
                        try:
                            response = json.loads(line)
                            logger.debug(f"Parsed response: {response}")
                            return response
                        except json.JSONDecodeError as e:
                            logger.debug(f"JSON decode error for line '{line}': {e}")
                            continue
                
                return {"error": f"No valid JSON response found. Output: {result.stdout}"}
            else:
                return {"error": f"MCP server error: {result.stderr}"}
                
        except Exception as e:
            logger.error(f"Communication error: {e}")
            return {"error": f"Communication error: {e}"}
    
    def initialize(self) -> bool:
        """Initialize MCP server connection"""
        try:
            init_message = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {
                        "name": "safehive-mcp-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            response = self._send_message(init_message)
            
            if "result" in response:
                logger.info("MCP server initialized successfully")
                return True
            else:
                logger.error(f"MCP initialization failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    def list_businesses(self) -> Dict[str, Any]:
        """List available businesses"""
        try:
            message = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/call",
                "params": {
                    "name": "list_businesses",
                    "arguments": {}
                }
            }
            
            response = self._send_message(message)
            return response
            
        except Exception as e:
            logger.error(f"List businesses error: {e}")
            return {"error": f"List businesses error: {e}"}
    
    def get_business(self, external_business_id: str) -> Dict[str, Any]:
        """Get business details"""
        try:
            message = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/call",
                "params": {
                    "name": "get_business",
                    "arguments": {
                        "external_business_id": external_business_id
                    }
                }
            }
            
            response = self._send_message(message)
            return response
            
        except Exception as e:
            logger.error(f"Get business error: {e}")
            return {"error": f"Get business error: {e}"}
    
    def list_stores(self, external_business_id: str) -> Dict[str, Any]:
        """List stores for a business"""
        try:
            message = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/call",
                "params": {
                    "name": "list_stores",
                    "arguments": {
                        "external_business_id": external_business_id
                    }
                }
            }
            
            response = self._send_message(message)
            return response
            
        except Exception as e:
            logger.error(f"List stores error: {e}")
            return {"error": f"List stores error: {e}"}
    
    def create_quote(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Create a delivery quote"""
        try:
            # Convert OrderRequest to MCP server format
            quote_args = {
                "external_delivery_id": f"safehive-{int(time.time())}",
                "pickup_address": order_request.pickup_address or "116 New Montgomery St, San Francisco CA 94105-3607, United States",
                "pickup_business_name": order_request.pickup_business_name or "Test Restaurant",
                "pickup_phone_number": order_request.pickup_phone_number or "6505555555",
                "dropoff_address": order_request.delivery_address,
                "dropoff_phone_number": order_request.customer_phone or "+1-555-0123"
            }
            
            message = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/call",
                "params": {
                    "name": "create_quote",
                    "arguments": quote_args
                }
            }
            
            response = self._send_message(message)
            return response
            
        except Exception as e:
            logger.error(f"Create quote error: {e}")
            return {"error": f"Create quote error: {e}"}
    
    def accept_quote(self, external_delivery_id: str, tip: Optional[float] = None) -> Dict[str, Any]:
        """Accept a delivery quote"""
        try:
            args = {"external_delivery_id": external_delivery_id}
            if tip:
                args["tip"] = tip
            
            message = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/call",
                "params": {
                    "name": "accept_quote",
                    "arguments": args
                }
            }
            
            response = self._send_message(message)
            return response
            
        except Exception as e:
            logger.error(f"Accept quote error: {e}")
            return {"error": f"Accept quote error: {e}"}
    
    def create_delivery(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Create a delivery order"""
        try:
            # Convert OrderRequest to MCP server format
            delivery_args = {
                "external_delivery_id": f"safehive-{int(time.time())}",
                "pickup_address": order_request.pickup_address or "116 New Montgomery St, San Francisco CA 94105-3607, United States",
                "pickup_business_name": order_request.pickup_business_name or "Test Restaurant",
                "pickup_phone_number": order_request.pickup_phone_number or "6505555555",
                "dropoff_address": order_request.delivery_address,
                "dropoff_phone_number": order_request.customer_phone or "+1-555-0123"
            }
            
            message = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/call",
                "params": {
                    "name": "create_delivery",
                    "arguments": delivery_args
                }
            }
            
            response = self._send_message(message)
            return response
            
        except Exception as e:
            logger.error(f"Create delivery error: {e}")
            return {"error": f"Create delivery error: {e}"}
    
    def get_delivery(self, external_delivery_id: str) -> Dict[str, Any]:
        """Get delivery status"""
        try:
            message = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/call",
                "params": {
                    "name": "get_delivery",
                    "arguments": {
                        "external_delivery_id": external_delivery_id
                    }
                }
            }
            
            response = self._send_message(message)
            return response
            
        except Exception as e:
            logger.error(f"Get delivery error: {e}")
            return {"error": f"Get delivery error: {e}"}
    
    def cancel_delivery(self, external_delivery_id: str) -> Dict[str, Any]:
        """Cancel a delivery"""
        try:
            message = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/call",
                "params": {
                    "name": "cancel_delivery",
                    "arguments": {
                        "external_delivery_id": external_delivery_id
                    }
                }
            }
            
            response = self._send_message(message)
            return response
            
        except Exception as e:
            logger.error(f"Cancel delivery error: {e}")
            return {"error": f"Cancel delivery error: {e}"}
    
    def process_order(self, order_request: OrderRequest) -> OrderResponse:
        """Process a complete order (quote + accept + create)"""
        try:
            logger.info(f"Processing order: {order_request.restaurant}")
            
            # Step 1: Create quote
            quote_response = self.create_quote(order_request)
            if "error" in quote_response:
                return OrderResponse(
                    status="error",
                    error_message=f"Quote creation failed: {quote_response['error']}"
                )
            
            # Extract delivery ID from quote response
            quote_data = quote_response.get("result", {}).get("content", [{}])[0].get("text", "{}")
            try:
                quote_info = json.loads(quote_data)
                delivery_id = quote_info.get("external_delivery_id")
            except:
                delivery_id = f"safehive-{int(time.time())}"
            
            # Step 2: Accept quote
            accept_response = self.accept_quote(delivery_id)
            if "error" in accept_response:
                return OrderResponse(
                    status="error",
                    error_message=f"Quote acceptance failed: {accept_response['error']}"
                )
            
            # Step 3: Create delivery
            delivery_response = self.create_delivery(order_request)
            if "error" in delivery_response:
                return OrderResponse(
                    status="error",
                    error_message=f"Delivery creation failed: {delivery_response['error']}"
                )
            
            # Parse successful response
            delivery_data = delivery_response.get("result", {}).get("content", [{}])[0].get("text", "{}")
            try:
                delivery_info = json.loads(delivery_data)
                return OrderResponse(
                    status="success",
                    order_id=delivery_id,
                    estimated_delivery="30-45 min",
                    confirmation=f"Order placed successfully: {delivery_id}"
                )
            except:
                return OrderResponse(
                    status="success",
                    order_id=delivery_id,
                    estimated_delivery="30-45 min",
                    confirmation=f"Order placed successfully: {delivery_id}"
                )
                
        except Exception as e:
            logger.error(f"Order processing error: {e}")
            return OrderResponse(
                status="error",
                error_message=f"Order processing failed: {e}"
            )
    
    def stop_server(self):
        """Stop the MCP server process"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            logger.info("MCP server stopped")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_server()
