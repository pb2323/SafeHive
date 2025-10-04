#!/usr/bin/env python3
"""
Honest Vendor Agent
==================

LangChain-powered honest vendor agent that simulates natural restaurant behavior.
Provides genuine customer service, menu knowledge, and order processing.
"""

import logging
import random
from typing import Dict, Any, List
from .vendors import BaseVendorAgent, VendorResponse, VendorType, VendorPersonality

logger = logging.getLogger(__name__)

class HonestVendorAgent(BaseVendorAgent):
    """Honest vendor agent with natural restaurant behavior"""
    
    def __init__(self, 
                 vendor_id: str,
                 personality: VendorPersonality,
                 mcp_client=None):
        super().__init__(vendor_id, VendorType.HONEST, personality, mcp_client)
        
        # Honest vendor specific attributes
        self.customer_satisfaction_score = 0.0
        self.orders_processed = 0
        self.recommendations_given = 0
        
        logger.info(f"Initialized honest vendor: {vendor_id} with personality: {personality.name}")
    
    def process_order_request(self, order_request: Dict[str, Any]) -> VendorResponse:
        """Process an order request with honest behavior"""
        try:
            logger.info(f"Processing order request for {self.vendor_id}")
            
            # Validate order request
            if not self._validate_order_request(order_request):
                return VendorResponse(
                    action="block",
                    reason="Invalid order request format",
                    details={"error": "Order request missing required fields"},
                    confidence=1.0,
                    vendor_type=VendorType.HONEST
                )
            
            # Check if items are available
            unavailable_items = self._check_item_availability(order_request.get("items", []))
            if unavailable_items:
                return VendorResponse(
                    action="block",
                    reason="Some items are currently unavailable",
                    details={"unavailable_items": unavailable_items},
                    confidence=0.9,
                    vendor_type=VendorType.HONEST
                )
            
            # Calculate pricing
            pricing = self.calculate_pricing(order_request.get("items", []))
            
            # Process order through MCP client if available
            if self.mcp_client:
                try:
                    # Create order request for MCP client
                    from ..mcp.simple_doorDash_client import OrderRequest
                    mcp_order = OrderRequest(
                        source=f"honest_vendor_{self.vendor_id}",
                        restaurant=self.personality.name,
                        items=order_request.get("items", []),
                        total=pricing["total"],
                        delivery_address=order_request.get("delivery_address", ""),
                        customer_phone=order_request.get("customer_phone", "")
                    )
                    
                    mcp_response = self.mcp_client.process_order(mcp_order)
                    
                    if mcp_response.status == "success":
                        self.orders_processed += 1
                        return VendorResponse(
                            action="allow",
                            reason="Order processed successfully",
                            details={
                                "order_id": mcp_response.order_id,
                                "estimated_delivery": mcp_response.estimated_delivery,
                                "pricing": pricing,
                                "vendor_response": mcp_response.confirmation
                            },
                            confidence=0.95,
                            vendor_type=VendorType.HONEST
                        )
                    else:
                        return VendorResponse(
                            action="block",
                            reason="Order processing failed",
                            details={"error": mcp_response.error_message},
                            confidence=0.9,
                            vendor_type=VendorType.HONEST
                        )
                        
                except Exception as e:
                    logger.error(f"MCP client error: {e}")
                    return VendorResponse(
                        action="escalate",
                        reason="Technical error in order processing",
                        details={"error": str(e)},
                        confidence=0.8,
                        vendor_type=VendorType.HONEST
                    )
            else:
                # Mock successful order processing
                self.orders_processed += 1
                return VendorResponse(
                    action="allow",
                    reason="Order processed successfully (mock mode)",
                    details={
                        "order_id": f"honest-{self.vendor_id}-{self.orders_processed}",
                        "estimated_delivery": "30-45 minutes",
                        "pricing": pricing,
                        "vendor_response": "Thank you for your order! We'll have it ready soon."
                    },
                    confidence=0.9,
                    vendor_type=VendorType.HONEST
                )
                
        except Exception as e:
            logger.error(f"Error processing order request: {e}")
            return VendorResponse(
                action="escalate",
                reason="Unexpected error in order processing",
                details={"error": str(e)},
                confidence=0.7,
                vendor_type=VendorType.HONEST
            )
    
    def generate_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate natural language response with honest behavior"""
        try:
            # Get conversation context
            conversation_context = self.get_conversation_context()
            
            # Determine response type based on user input
            response_type = self._classify_user_input(user_input)
            
            # Generate appropriate response
            if response_type == "greeting":
                response = self._generate_greeting_response()
            elif response_type == "menu_inquiry":
                response = self._generate_menu_response(user_input, context)
            elif response_type == "order_request":
                response = self._generate_order_response(user_input, context)
            elif response_type == "complaint":
                response = self._generate_complaint_response(user_input, context)
            elif response_type == "recommendation_request":
                response = self._generate_recommendation_response(user_input, context)
            else:
                response = self._generate_general_response(user_input, context)
            
            # Add personality traits to response
            response = self._add_personality_traits(response)
            
            # Update conversation history
            self.add_to_conversation_history(user_input, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
    
    def _validate_order_request(self, order_request: Dict[str, Any]) -> bool:
        """Validate order request format"""
        required_fields = ["items", "delivery_address"]
        return all(field in order_request for field in required_fields)
    
    def _check_item_availability(self, items: List[Dict[str, Any]]) -> List[str]:
        """Check if items are available (mock implementation)"""
        unavailable = []
        menu_items = self.personality.menu_knowledge.get("items", [])
        menu_names = [item.get("name", "").lower() for item in menu_items]
        
        for item in items:
            item_name = item.get("name", "").lower()
            if item_name not in menu_names:
                unavailable.append(item.get("name", "Unknown item"))
        
        return unavailable
    
    def _classify_user_input(self, user_input: str) -> str:
        """Classify user input to determine response type"""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return "greeting"
        elif any(word in user_input_lower for word in ["menu", "what do you have", "available", "options"]):
            return "menu_inquiry"
        elif any(word in user_input_lower for word in ["order", "buy", "purchase", "get", "want"]):
            return "order_request"
        elif any(word in user_input_lower for word in ["complaint", "problem", "issue", "wrong", "bad"]):
            return "complaint"
        elif any(word in user_input_lower for word in ["recommend", "suggest", "best", "popular"]):
            return "recommendation_request"
        else:
            return "general"
    
    def _generate_greeting_response(self) -> str:
        """Generate greeting response"""
        greetings = [
            f"Hello! Welcome to {self.personality.name}! How can I help you today?",
            f"Hi there! Thanks for choosing {self.personality.name}. What can I get for you?",
            f"Good day! I'm here to help you with your order at {self.personality.name}.",
            f"Welcome to {self.personality.name}! I'd be happy to assist you with your order."
        ]
        return random.choice(greetings)
    
    def _generate_menu_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate menu inquiry response"""
        menu_items = self.personality.menu_knowledge.get("items", [])
        
        if not menu_items:
            return "I apologize, but I don't have access to our current menu. Please contact us directly for menu information."
        
        # Get top 5 items
        top_items = menu_items[:5]
        response = f"Here are some of our popular items at {self.personality.name}:\n\n"
        
        for item in top_items:
            response += f"• {item.get('name', 'Unknown')} - ${item.get('price', 0):.2f}\n"
            if item.get('description'):
                response += f"  {item.get('description')}\n"
        
        response += "\nWould you like to see more items or place an order?"
        return response
    
    def _generate_order_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate order response"""
        return "I'd be happy to help you place an order! Please let me know what items you'd like and your delivery address."
    
    def _generate_complaint_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate complaint response"""
        responses = [
            "I'm sorry to hear about your experience. Let me help resolve this issue for you.",
            "I apologize for any inconvenience. Please tell me more about the problem so I can assist you.",
            "Thank you for bringing this to our attention. We take customer feedback seriously and want to make this right.",
            "I'm here to help resolve this issue. Can you provide more details about what went wrong?"
        ]
        return random.choice(responses)
    
    def _generate_recommendation_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate recommendation response"""
        self.recommendations_given += 1
        
        # Get user preferences from context
        preferences = context.get("user_preferences", {})
        suggestions = self.get_menu_suggestions(preferences)
        
        if suggestions:
            response = f"Based on your preferences, I'd recommend:\n\n"
            for item in suggestions[:3]:
                response += f"• {item.get('name', 'Unknown')} - ${item.get('price', 0):.2f}\n"
                if item.get('description'):
                    response += f"  {item.get('description')}\n"
        else:
            # Fallback to popular items
            menu_items = self.personality.menu_knowledge.get("items", [])
            popular_items = menu_items[:3]
            response = f"Here are some of our most popular items:\n\n"
            for item in popular_items:
                response += f"• {item.get('name', 'Unknown')} - ${item.get('price', 0):.2f}\n"
        
        return response
    
    def _generate_general_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate general response"""
        responses = [
            "I'm here to help! How can I assist you with your order?",
            "Is there anything specific you'd like to know about our menu or services?",
            "I'd be happy to help you with any questions you have about ordering.",
            "What can I do to make your experience better today?"
        ]
        return random.choice(responses)
    
    def _add_personality_traits(self, response: str) -> str:
        """Add personality traits to response"""
        # Add communication style elements
        if "friendly" in self.personality.communication_style:
            response = response.replace("I'm", "I'm so")
            response = response.replace("Thank you", "Thank you so much")
        
        if "professional" in self.personality.communication_style:
            response = response.replace("Hi", "Hello")
            response = response.replace("Hey", "Good day")
        
        if "enthusiastic" in self.personality.communication_style:
            response = response.replace(".", "!")
            response = response.replace("?", "?")
        
        return response
    
    def get_vendor_stats(self) -> Dict[str, Any]:
        """Get honest vendor specific statistics"""
        base_stats = super().get_vendor_stats()
        base_stats.update({
            "customer_satisfaction_score": self.customer_satisfaction_score,
            "orders_processed": self.orders_processed,
            "recommendations_given": self.recommendations_given,
            "average_order_value": self.orders_processed / max(self.orders_processed, 1)
        })
        return base_stats
