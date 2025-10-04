#!/usr/bin/env python3
"""
Vendor Agent Base Class
======================

Base class for all vendor agents (honest and malicious) using LangChain.
Implements the standard vendor interface for orchestrator communication.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VendorType(Enum):
    """Types of vendor agents"""
    HONEST = "honest"
    MALICIOUS = "malicious"

class AttackType(Enum):
    """Types of malicious attacks"""
    SOCIAL_ENGINEERING = "social_engineering"
    DATA_EXFILTRATION = "data_exfiltration"
    PROMPT_INJECTION = "prompt_injection"
    TASK_DRIFT = "task_drift"
    PSYCHOLOGICAL_MANIPULATION = "psychological_manipulation"
    CONVERSATION_HIJACKING = "conversation_hijacking"

@dataclass
class VendorResponse:
    """Standard vendor response format for orchestrator communication"""
    action: str  # "allow", "block", "escalate"
    reason: str
    details: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    vendor_type: VendorType
    attack_type: Optional[AttackType] = None

@dataclass
class VendorPersonality:
    """Vendor personality configuration"""
    name: str
    description: str
    personality_traits: List[str]
    communication_style: str
    menu_knowledge: Dict[str, Any]
    pricing_strategy: str
    customer_service_approach: str
    attack_patterns: Optional[List[AttackType]] = None

class BaseVendorAgent(ABC):
    """Base class for all vendor agents"""
    
    def __init__(self, 
                 vendor_id: str,
                 vendor_type: VendorType,
                 personality: VendorPersonality,
                 mcp_client=None):
        self.vendor_id = vendor_id
        self.vendor_type = vendor_type
        self.personality = personality
        self.mcp_client = mcp_client
        self.conversation_history = []
        self.attack_attempts = 0
        self.successful_attacks = 0
        
        logger.info(f"Initialized {vendor_type.value} vendor: {vendor_id}")
    
    @abstractmethod
    def process_order_request(self, order_request: Dict[str, Any]) -> VendorResponse:
        """Process an order request and return vendor response"""
        pass
    
    @abstractmethod
    def generate_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate a natural language response to user input"""
        pass
    
    def add_to_conversation_history(self, user_input: str, vendor_response: str):
        """Add interaction to conversation history"""
        self.conversation_history.append({
            "user_input": user_input,
            "vendor_response": vendor_response,
            "timestamp": self._get_timestamp()
        })
        
        # Keep only last 10 interactions to manage memory
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_context(self) -> str:
        """Get recent conversation context for AI reasoning"""
        if not self.conversation_history:
            return "No previous conversation."
        
        context = "Recent conversation:\n"
        for interaction in self.conversation_history[-3:]:  # Last 3 interactions
            context += f"User: {interaction['user_input']}\n"
            context += f"Vendor: {interaction['vendor_response']}\n\n"
        
        return context
    
    def get_menu_suggestions(self, user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get menu suggestions based on user preferences"""
        menu_items = self.personality.menu_knowledge.get("items", [])
        suggestions = []
        
        # Filter based on preferences
        for item in menu_items:
            if self._matches_preferences(item, user_preferences):
                suggestions.append(item)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _matches_preferences(self, item: Dict[str, Any], preferences: Dict[str, Any]) -> bool:
        """Check if menu item matches user preferences"""
        # Check dietary restrictions
        if preferences.get("dietary_restrictions"):
            item_dietary = item.get("dietary_info", [])
            for restriction in preferences["dietary_restrictions"]:
                if restriction not in item_dietary:
                    return False
        
        # Check price range
        if preferences.get("max_price"):
            if item.get("price", 0) > preferences["max_price"]:
                return False
        
        # Check cuisine type
        if preferences.get("cuisine_type"):
            if item.get("cuisine_type") != preferences["cuisine_type"]:
                return False
        
        return True
    
    def calculate_pricing(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate pricing based on vendor's pricing strategy"""
        total = 0
        item_prices = []
        
        for item in items:
            base_price = item.get("price", 0)
            
            # Apply pricing strategy
            if self.personality.pricing_strategy == "premium":
                final_price = base_price * 1.2  # 20% markup
            elif self.personality.pricing_strategy == "competitive":
                final_price = base_price * 0.95  # 5% discount
            else:  # standard
                final_price = base_price
            
            item_prices.append({
                "name": item.get("name"),
                "base_price": base_price,
                "final_price": final_price
            })
            total += final_price
        
        return {
            "items": item_prices,
            "subtotal": total,
            "tax": total * 0.08,  # 8% tax
            "total": total * 1.08,
            "pricing_strategy": self.personality.pricing_strategy
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def get_vendor_stats(self) -> Dict[str, Any]:
        """Get vendor statistics for monitoring"""
        return {
            "vendor_id": self.vendor_id,
            "vendor_type": self.vendor_type.value,
            "conversation_count": len(self.conversation_history),
            "attack_attempts": self.attack_attempts,
            "successful_attacks": self.successful_attacks,
            "success_rate": self.successful_attacks / max(self.attack_attempts, 1),
            "personality": self.personality.name
        }
    
    def reset_stats(self):
        """Reset vendor statistics"""
        self.attack_attempts = 0
        self.successful_attacks = 0
        self.conversation_history = []
        logger.info(f"Reset stats for vendor: {self.vendor_id}")
    
    def __str__(self) -> str:
        return f"{self.vendor_type.value.title()}Vendor({self.vendor_id})"
    
    def __repr__(self) -> str:
        return f"BaseVendorAgent(vendor_id='{self.vendor_id}', type={self.vendor_type.value})"
