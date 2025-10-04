"""
SafeHive Tools Package

This package contains LangChain tools for external system interactions
and specialized functionality for SafeHive agents.
"""

from .base_tools import BaseSafeHiveTool, ToolInput, ToolOutput, create_tool_output
from .communication_tools import MessageTool, LoggingTool
from .system_tools import SystemInfoTool, HealthCheckTool, AgentStatusTool
from .food_ordering_tools import (
    MenuLookupTool, OrderPlacementTool, PaymentProcessingTool, 
    OrderStatusTool, InventoryCheckTool, get_food_ordering_tools
)
from .vendor_interaction_tools import (
    VendorCommunicationTool, PricingNegotiationTool, AvailabilityCheckTool,
    VendorRatingTool, VendorInfoTool, get_vendor_interaction_tools
)
from .security_analysis_tools import (
    PIIDetectionTool, AttackPatternAnalysisTool, SecurityAssessmentTool,
    DataSanitizationTool, get_security_analysis_tools
)

__all__ = [
    # Base classes
    "BaseSafeHiveTool",
    "ToolInput", 
    "ToolOutput",
    "create_tool_output",
    
    # Communication tools
    "MessageTool",
    "LoggingTool",
    
    # System tools
    "SystemInfoTool",
    "HealthCheckTool", 
    "AgentStatusTool",
    
    # Food ordering tools
    "MenuLookupTool",
    "OrderPlacementTool",
    "PaymentProcessingTool",
    "OrderStatusTool",
    "InventoryCheckTool",
    "get_food_ordering_tools",
    
    # Vendor interaction tools
    "VendorCommunicationTool",
    "PricingNegotiationTool",
    "AvailabilityCheckTool",
    "VendorRatingTool",
    "VendorInfoTool",
    "get_vendor_interaction_tools",
    
    # Security analysis tools
    "PIIDetectionTool",
    "AttackPatternAnalysisTool",
    "SecurityAssessmentTool",
    "DataSanitizationTool",
    "get_security_analysis_tools"
]


def get_all_tools() -> list[BaseSafeHiveTool]:
    """Get all available tools for agent configuration."""
    return (
        get_food_ordering_tools() +
        get_vendor_interaction_tools() +
        get_security_analysis_tools() +
        [
            MessageTool(),
            LoggingTool(),
            SystemInfoTool(),
            HealthCheckTool(),
            AgentStatusTool()
        ]
    )
