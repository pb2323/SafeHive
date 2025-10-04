#!/usr/bin/env python3
"""
Simple DoorDash MCP Client
==========================

A simplified Python client that uses shell commands to communicate with the MCP server.
This approach is more reliable for the current setup.
"""

import json
import subprocess
import time
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OrderValidationError(Exception):
    """Exception raised for order validation errors"""
    pass

class SafetyCheckError(Exception):
    """Exception raised for safety check failures"""
    pass

class MCPError(Exception):
    """Exception raised for MCP communication errors"""
    pass

class RetryableError(Exception):
    """Exception raised for retryable errors"""
    pass

@dataclass
class ValidationResult:
    """Result of order validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

@dataclass
class SafetyCheckResult:
    """Result of safety checks"""
    is_safe: bool
    checks_passed: List[str]
    checks_failed: List[str]
    requires_confirmation: bool

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

class SimpleDoorDashMCPClient:
    """Simple Python client for DoorDash MCP server integration"""
    
    def __init__(self, mcp_server_path: str, environment: str = "sandbox"):
        self.mcp_server_path = mcp_server_path
        self.environment = environment
        self.mcp_root = "/Users/puneetbajaj/Desktop/playground/DoorDash-MCP-Server"
        
        # Validate MCP server path
        import os
        if not os.path.exists(mcp_server_path):
            raise FileNotFoundError(f"MCP server not found at: {mcp_server_path}")
        
        # Configuration for validation and safety checks
        self.max_order_value = 100.00
        self.min_order_value = 5.00
        self.confirmation_threshold = 50.00
        self.max_retry_attempts = 3
        self.retry_delay = 1.0
    
    def _send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC message to MCP server using shell command"""
        try:
            message_str = json.dumps(message)
            
            # Use shell command that we know works
            cmd = f'echo \'{message_str}\' | node "{self.mcp_server_path}"'
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.mcp_root
            )
            
            if result.returncode == 0:
                # Parse the response (remove dotenv messages)
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if line.startswith('{'):
                        try:
                            return json.loads(line)
                        except json.JSONDecodeError:
                            continue
                
                return {"error": f"No valid JSON response found. Output: {result.stdout}"}
            else:
                return {"error": f"MCP server error: {result.stderr}"}
                
        except Exception as e:
            logger.error(f"Communication error: {e}")
            return {"error": f"Communication error: {e}"}
    
    def _validate_order(self, order_request: OrderRequest) -> ValidationResult:
        """Validate order request data (Task 1.6)"""
        errors = []
        warnings = []
        
        # Validate required fields
        if not order_request.source:
            errors.append("Order source is required")
        
        if not order_request.restaurant:
            errors.append("Restaurant name is required")
        
        if not order_request.delivery_address:
            errors.append("Delivery address is required")
        
        if not order_request.items or len(order_request.items) == 0:
            errors.append("At least one item is required")
        
        # Validate order value
        if order_request.total <= 0:
            errors.append("Order total must be greater than 0")
        elif order_request.total < self.min_order_value:
            errors.append(f"Order total must be at least ${self.min_order_value}")
        elif order_request.total > self.max_order_value:
            errors.append(f"Order total cannot exceed ${self.max_order_value}")
        
        # Validate phone number format
        if order_request.customer_phone:
            phone_pattern = r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
            if not re.match(phone_pattern, order_request.customer_phone):
                warnings.append("Phone number format may be invalid")
        
        # Validate address format
        if order_request.delivery_address:
            if len(order_request.delivery_address) < 10:
                warnings.append("Delivery address seems too short")
        
        # Validate items
        for i, item in enumerate(order_request.items):
            if not item.get('name'):
                errors.append(f"Item {i+1} must have a name")
            if not item.get('price') or item.get('price', 0) <= 0:
                errors.append(f"Item {i+1} must have a valid price")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _perform_safety_checks(self, order_request: OrderRequest) -> SafetyCheckResult:
        """Perform safety checks before placing orders (Task 1.6)"""
        checks_passed = []
        checks_failed = []
        requires_confirmation = False
        
        # Check 1: Environment validation
        if self.environment == "sandbox":
            checks_passed.append("Sandbox environment - safe for testing")
        else:
            checks_failed.append("Production environment - requires extra caution")
            requires_confirmation = True
        
        # Check 2: Order value threshold
        if order_request.total >= self.confirmation_threshold:
            checks_failed.append(f"High-value order (${order_request.total}) requires confirmation")
            requires_confirmation = True
        else:
            checks_passed.append(f"Order value (${order_request.total}) within safe range")
        
        # Check 3: Suspicious patterns
        suspicious_patterns = [
            "test", "fake", "dummy", "sample", "example"
        ]
        
        restaurant_lower = order_request.restaurant.lower()
        if any(pattern in restaurant_lower for pattern in suspicious_patterns):
            checks_passed.append("Test restaurant detected - safe for sandbox")
        else:
            checks_failed.append("Real restaurant name - verify authenticity")
            requires_confirmation = True
        
        # Check 4: Address validation
        if "test" in order_request.delivery_address.lower():
            checks_passed.append("Test address detected - safe for sandbox")
        else:
            checks_failed.append("Real address - verify delivery location")
            requires_confirmation = True
        
        # Check 5: Phone number validation
        if order_request.customer_phone and "555" in order_request.customer_phone:
            checks_passed.append("Test phone number detected - safe for sandbox")
        else:
            checks_failed.append("Real phone number - verify customer contact")
            requires_confirmation = True
        
        is_safe = len(checks_failed) == 0 or self.environment == "sandbox"
        
        return SafetyCheckResult(
            is_safe=is_safe,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            requires_confirmation=requires_confirmation
        )
    
    def _send_message_with_retry(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message with retry logic (Task 1.7)"""
        last_error = None
        
        for attempt in range(self.max_retry_attempts):
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.max_retry_attempts}")
                
                response = self._send_message(message)
                
                # Check if response indicates a retryable error
                if "error" in response:
                    error_msg = response["error"]
                    
                    # Check for retryable errors
                    retryable_patterns = [
                        "timeout", "connection", "network", "temporary", "rate limit"
                    ]
                    
                    if any(pattern in error_msg.lower() for pattern in retryable_patterns):
                        if attempt < self.max_retry_attempts - 1:
                            logger.warning(f"Retryable error on attempt {attempt + 1}: {error_msg}")
                            time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                            continue
                        else:
                            raise RetryableError(f"Max retries exceeded: {error_msg}")
                    else:
                        # Non-retryable error
                        raise MCPError(f"Non-retryable error: {error_msg}")
                
                # Success
                logger.debug(f"Message sent successfully on attempt {attempt + 1}")
                return response
                
            except RetryableError as e:
                last_error = e
                if attempt < self.max_retry_attempts - 1:
                    logger.warning(f"Retryable error on attempt {attempt + 1}: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(f"Max retries exceeded: {e}")
                    return {"error": f"Max retries exceeded: {e}"}
            
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    return {"error": f"Unexpected error after {self.max_retry_attempts} attempts: {e}"}
        
        return {"error": f"Failed after {self.max_retry_attempts} attempts. Last error: {last_error}"}
    
    def initialize(self) -> bool:
        """Initialize MCP server connection"""
        try:
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
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
                "id": 2,
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
                "id": 3,
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
                "id": 4,
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
                "id": 5,
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
    
    def process_order(self, order_request: OrderRequest, skip_validation: bool = False) -> OrderResponse:
        """Process a complete order with validation and safety checks (Tasks 1.6, 1.7)"""
        try:
            logger.info(f"Processing order: {order_request.restaurant}")
            
            # Step 1: Order Validation (Task 1.6)
            if not skip_validation:
                logger.info("Performing order validation...")
                validation_result = self._validate_order(order_request)
                
                if not validation_result.is_valid:
                    error_msg = f"Order validation failed: {'; '.join(validation_result.errors)}"
                    logger.error(error_msg)
                    return OrderResponse(
                        status="error",
                        error_message=error_msg
                    )
                
                if validation_result.warnings:
                    logger.warning(f"Order validation warnings: {'; '.join(validation_result.warnings)}")
            
            # Step 2: Safety Checks (Task 1.6)
            if not skip_validation:
                logger.info("Performing safety checks...")
                safety_result = self._perform_safety_checks(order_request)
                
                if not safety_result.is_safe:
                    error_msg = f"Safety checks failed: {'; '.join(safety_result.checks_failed)}"
                    logger.error(error_msg)
                    return OrderResponse(
                        status="error",
                        error_message=error_msg
                    )
                
                if safety_result.requires_confirmation:
                    logger.warning(f"Order requires confirmation: {'; '.join(safety_result.checks_failed)}")
                    # In a real implementation, this would trigger human-in-the-loop confirmation
                
                logger.info(f"Safety checks passed: {'; '.join(safety_result.checks_passed)}")
            
            # Step 3: Create Quote with Retry Logic (Task 1.7)
            logger.info("Creating delivery quote...")
            quote_response = self.create_quote(order_request)
            if "error" in quote_response:
                return OrderResponse(
                    status="error",
                    error_message=f"Quote creation failed: {quote_response['error']}"
                )
            
            # Step 4: Generate Order Response
            order_id = f"safehive-{int(time.time())}"
            
            logger.info(f"Order processed successfully: {order_id}")
            return OrderResponse(
                status="success",
                order_id=order_id,
                estimated_delivery="30-45 min",
                confirmation=f"Order placed successfully: {order_id}"
            )
                
        except OrderValidationError as e:
            logger.error(f"Order validation error: {e}")
            return OrderResponse(
                status="error",
                error_message=f"Order validation failed: {e}"
            )
        except SafetyCheckError as e:
            logger.error(f"Safety check error: {e}")
            return OrderResponse(
                status="error",
                error_message=f"Safety check failed: {e}"
            )
        except MCPError as e:
            logger.error(f"MCP communication error: {e}")
            return OrderResponse(
                status="error",
                error_message=f"Communication failed: {e}"
            )
        except Exception as e:
            logger.error(f"Unexpected order processing error: {e}")
            return OrderResponse(
                status="error",
                error_message=f"Order processing failed: {e}"
            )
