"""
SafeHive AI Agents Package

This package contains the AI agent implementations for the SafeHive AI Security Sandbox.
All agents are built on top of LangChain and provide common functionality for
orchestration, user simulation, and vendor interactions.
"""

from .base_agent import BaseAgent, AgentCapabilities
from .state_manager import StateManager, AgentStateSnapshot, StateMetadata, create_state_manager, get_state_manager
from .configuration import (
    ConfigurationManager, AgentConfiguration, PersonalityProfile, ConfigurationTemplate,
    PersonalityTrait, ResponseStyle, ConfigurationScope, get_configuration_manager, create_configuration_manager
)
from .monitoring import (
    HealthStatus, AlertLevel, HealthCheck, AgentHealthReport, PerformanceMetrics,
    HealthCheckProvider, BasicHealthCheckProvider, PerformanceHealthCheckProvider,
    ResourceHealthCheckProvider, Alert, AlertManager, AgentMonitor,
    get_agent_monitor, create_agent_monitor
)
from .user_twin import (
    DecisionStyle, PreferenceCategory, UserPreference, DecisionContext, Decision,
    PreferenceConflict, PreferenceManager, DecisionEngine, UserTwinAgent, create_user_twin_agent
)
from .order_models import (
    OrderStatus, OrderType, PaymentStatus, OrderItem, Vendor, Order
)
from .orchestrator import (
    OrderManager, VendorManager, OrchestratorAgent, create_orchestrator_agent
)
from .intelligent_order_manager import (
    ConstraintType, ReasoningType, OrderConstraint, OrderReasoning,
    OrderOptimizationResult, IntelligentOrderManager
)
from .vendor_communication import (
    CommunicationIntent, MessageType, CommunicationStatus, CommunicationMessage,
    CommunicationSession, IntentClassification, VendorCommunicationInterface
)
from .order_validation import (
    ValidationStatus, ValidationSeverity, ValidationType, ValidationRule,
    ValidationResult, ValidationReport, OrderValidationEngine
)
from .order_confirmation import (
    ConfirmationStatus, ApprovalType, ApprovalResult, ConfirmationStep,
    ApprovalRequirement, ConfirmationWorkflow, ConfirmationSession, OrderConfirmationManager
)
from ..models.agent_models import AgentStatus

__all__ = [
    "BaseAgent",
    "AgentCapabilities", 
    "AgentStatus",
    "StateManager",
    "AgentStateSnapshot",
    "StateMetadata",
    "create_state_manager",
    "get_state_manager",
    "ConfigurationManager",
    "AgentConfiguration",
    "PersonalityProfile",
    "ConfigurationTemplate",
    "PersonalityTrait",
    "ResponseStyle",
    "ConfigurationScope",
    "get_configuration_manager",
    "create_configuration_manager",
    "HealthStatus",
    "AlertLevel",
    "HealthCheck",
    "AgentHealthReport",
    "PerformanceMetrics",
    "HealthCheckProvider",
    "BasicHealthCheckProvider",
    "PerformanceHealthCheckProvider",
    "ResourceHealthCheckProvider",
    "Alert",
    "AlertManager",
    "AgentMonitor",
    "get_agent_monitor",
    "create_agent_monitor",
    "DecisionStyle",
    "PreferenceCategory",
    "UserPreference",
    "DecisionContext",
    "Decision",
    "PreferenceConflict",
    "PreferenceManager",
    "DecisionEngine",
    "UserTwinAgent",
    "create_user_twin_agent",
    "OrderStatus",
    "OrderType", 
    "PaymentStatus",
    "OrderItem",
    "Vendor",
    "Order",
    "OrderManager",
    "VendorManager",
    "OrchestratorAgent",
    "create_orchestrator_agent",
    "ConstraintType",
    "ReasoningType",
    "OrderConstraint",
    "OrderReasoning",
    "OrderOptimizationResult",
    "IntelligentOrderManager",
    "CommunicationIntent",
    "MessageType",
    "CommunicationStatus",
    "CommunicationMessage",
    "CommunicationSession",
    "IntentClassification",
    "VendorCommunicationInterface",
    "ValidationStatus",
    "ValidationSeverity",
    "ValidationType",
    "ValidationRule",
    "ValidationResult",
    "ValidationReport",
    "OrderValidationEngine",
    "ConfirmationStatus",
    "ApprovalType",
    "ApprovalResult",
    "ConfirmationStep",
    "ApprovalRequirement",
    "ConfirmationWorkflow",
    "ConfirmationSession",
    "OrderConfirmationManager"
]
