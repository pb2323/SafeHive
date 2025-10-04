"""
Order Validation and Confirmation Workflows with Agent Reasoning

This module implements intelligent order validation and confirmation workflows
for the SafeHive AI Security Sandbox, providing multi-step validation with
agent reasoning and business logic enforcement.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path

from .order_models import Order, OrderItem, Vendor, OrderStatus, OrderType, PaymentStatus
from .user_twin import UserTwinAgent, PreferenceCategory
from .intelligent_order_manager import OrderConstraint, ConstraintType
from .vendor_communication import CommunicationSession, CommunicationIntent
from ..utils.logger import get_logger
from ..utils.metrics import record_metric, MetricType

logger = get_logger(__name__)


class ValidationStatus(Enum):
    """Status of validation operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    OVERRIDDEN = "overridden"
    SKIPPED = "skipped"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Types of validation operations."""
    BUSINESS_LOGIC = "business_logic"
    USER_PREFERENCE = "user_preference"
    VENDOR_CONSTRAINT = "vendor_constraint"
    PAYMENT_VALIDATION = "payment_validation"
    INVENTORY_CHECK = "inventory_check"
    DELIVERY_VALIDATION = "delivery_validation"
    SECURITY_CHECK = "security_check"
    CUSTOM = "custom"


@dataclass
class ValidationRule:
    """Represents a validation rule."""
    rule_id: str
    name: str
    description: str
    validation_type: ValidationType
    severity: ValidationSeverity
    is_required: bool = True
    is_enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "validation_type": self.validation_type.value,
            "severity": self.severity.value,
            "is_required": self.is_required,
            "is_enabled": self.is_enabled,
            "parameters": self.parameters
        }


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    rule_id: str
    validation_type: ValidationType
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    agent_reasoning: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_id": self.rule_id,
            "validation_type": self.validation_type.value,
            "status": self.status.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "agent_reasoning": self.agent_reasoning,
            "suggestions": self.suggestions
        }


@dataclass
class ValidationReport:
    """Complete validation report for an order."""
    order_id: str
    validation_session_id: str
    overall_status: ValidationStatus
    validation_results: List[ValidationResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    agent_reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result to the report."""
        self.validation_results.append(result)
        
        # Update overall status based on results
        if result.status == ValidationStatus.FAILED and result.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]:
            self.overall_status = ValidationStatus.FAILED
        elif result.status == ValidationStatus.WARNING and self.overall_status == ValidationStatus.PENDING:
            self.overall_status = ValidationStatus.WARNING
        elif result.status == ValidationStatus.PASSED and self.overall_status == ValidationStatus.PENDING:
            self.overall_status = ValidationStatus.PASSED
    
    def get_results_by_type(self, validation_type: ValidationType) -> List[ValidationResult]:
        """Get validation results by type."""
        return [result for result in self.validation_results if result.validation_type == validation_type]
    
    def get_failed_results(self) -> List[ValidationResult]:
        """Get all failed validation results."""
        return [result for result in self.validation_results if result.status == ValidationStatus.FAILED]
    
    def get_warning_results(self) -> List[ValidationResult]:
        """Get all warning validation results."""
        return [result for result in self.validation_results if result.status == ValidationStatus.WARNING]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "order_id": self.order_id,
            "validation_session_id": self.validation_session_id,
            "overall_status": self.overall_status.value,
            "validation_results": [result.to_dict() for result in self.validation_results],
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "agent_reasoning": self.agent_reasoning,
            "metadata": self.metadata
        }


@dataclass
class ConfirmationWorkflow:
    """Represents a confirmation workflow."""
    workflow_id: str
    name: str
    description: str
    steps: List[str] = field(default_factory=list)
    required_approvals: List[str] = field(default_factory=list)
    auto_approval_conditions: List[Dict[str, Any]] = field(default_factory=list)
    timeout_minutes: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "required_approvals": self.required_approvals,
            "auto_approval_conditions": self.auto_approval_conditions,
            "timeout_minutes": self.timeout_minutes
        }


class OrderValidationEngine:
    """Engine for validating orders with agent reasoning."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_order_validation"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Validation components
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.validation_history: List[ValidationReport] = []
        
        # Initialize default validation rules
        self._initialize_default_rules()
        
        # Load validation history
        self._load_validation_history()
        
        logger.info("Order Validation Engine initialized")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default validation rules."""
        default_rules = [
            ValidationRule(
                rule_id="vendor_availability",
                name="Vendor Availability Check",
                description="Verify that the vendor is available and accepting orders",
                validation_type=ValidationType.VENDOR_CONSTRAINT,
                severity=ValidationSeverity.CRITICAL,
                is_required=True,
                parameters={"check_hours": True, "check_capacity": True}
            ),
            ValidationRule(
                rule_id="minimum_order_amount",
                name="Minimum Order Amount",
                description="Ensure order meets vendor's minimum order requirement",
                validation_type=ValidationType.BUSINESS_LOGIC,
                severity=ValidationSeverity.HIGH,
                is_required=True,
                parameters={"tolerance_percentage": 0.05}
            ),
            ValidationRule(
                rule_id="payment_method_validation",
                name="Payment Method Validation",
                description="Validate that payment method is accepted and valid",
                validation_type=ValidationType.PAYMENT_VALIDATION,
                severity=ValidationSeverity.HIGH,
                is_required=True,
                parameters={"check_balance": True, "check_expiry": True}
            ),
            ValidationRule(
                rule_id="delivery_address_validation",
                name="Delivery Address Validation",
                description="Validate delivery address is within vendor's delivery area",
                validation_type=ValidationType.DELIVERY_VALIDATION,
                severity=ValidationSeverity.MEDIUM,
                is_required=True,
                parameters={"max_distance_miles": 10}
            ),
            ValidationRule(
                rule_id="inventory_check",
                name="Inventory Availability",
                description="Check if requested items are available in vendor's inventory",
                validation_type=ValidationType.INVENTORY_CHECK,
                severity=ValidationSeverity.HIGH,
                is_required=True,
                parameters={"check_real_time": False}
            ),
            ValidationRule(
                rule_id="user_preference_compatibility",
                name="User Preference Compatibility",
                description="Check if order aligns with user's dietary and preference constraints",
                validation_type=ValidationType.USER_PREFERENCE,
                severity=ValidationSeverity.MEDIUM,
                is_required=False,
                parameters={"strict_mode": False}
            ),
            ValidationRule(
                rule_id="order_size_reasonableness",
                name="Order Size Reasonableness",
                description="Validate that order size is reasonable for the number of people",
                validation_type=ValidationType.BUSINESS_LOGIC,
                severity=ValidationSeverity.LOW,
                is_required=False,
                parameters={"max_items_per_person": 3, "max_total_amount": 200.0}
            ),
            ValidationRule(
                rule_id="security_screening",
                name="Security Screening",
                description="Perform security checks on order data",
                validation_type=ValidationType.SECURITY_CHECK,
                severity=ValidationSeverity.CRITICAL,
                is_required=True,
                parameters={"check_sql_injection": True, "check_xss": True}
            )
        ]
        
        for rule in default_rules:
            self.validation_rules[rule.rule_id] = rule
    
    def _load_validation_history(self) -> None:
        """Load validation history from storage."""
        history_file = self.storage_path / "validation_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for report_data in data:
                        report = self._reconstruct_validation_report(report_data)
                        if report:
                            self.validation_history.append(report)
                logger.info(f"Loaded {len(self.validation_history)} validation reports")
            except Exception as e:
                logger.error(f"Failed to load validation history: {e}")
    
    def _save_validation_history(self) -> None:
        """Save validation history to storage."""
        history_file = self.storage_path / "validation_history.json"
        try:
            data = [report.to_dict() for report in self.validation_history]
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved validation history")
        except Exception as e:
            logger.error(f"Failed to save validation history: {e}")
    
    def _reconstruct_validation_report(self, data: Dict[str, Any]) -> Optional[ValidationReport]:
        """Reconstruct ValidationReport from stored data."""
        try:
            results = []
            for result_data in data.get("validation_results", []):
                result = ValidationResult(
                    rule_id=result_data["rule_id"],
                    validation_type=ValidationType(result_data["validation_type"]),
                    status=ValidationStatus(result_data["status"]),
                    severity=ValidationSeverity(result_data["severity"]),
                    message=result_data["message"],
                    details=result_data.get("details", {}),
                    timestamp=datetime.fromisoformat(result_data["timestamp"]),
                    agent_reasoning=result_data.get("agent_reasoning", []),
                    suggestions=result_data.get("suggestions", [])
                )
                results.append(result)
            
            report = ValidationReport(
                order_id=data["order_id"],
                validation_session_id=data["validation_session_id"],
                overall_status=ValidationStatus(data["overall_status"]),
                validation_results=results,
                created_at=datetime.fromisoformat(data["created_at"]),
                completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
                agent_reasoning=data.get("agent_reasoning", []),
                metadata=data.get("metadata", {})
            )
            
            return report
        except Exception as e:
            logger.error(f"Failed to reconstruct validation report: {e}")
            return None
    
    def add_validation_rule(self, rule: ValidationRule) -> None:
        """Add a new validation rule."""
        self.validation_rules[rule.rule_id] = rule
        logger.info(f"Added validation rule: {rule.name}")
    
    def update_validation_rule(self, rule_id: str, **updates) -> bool:
        """Update an existing validation rule."""
        if rule_id in self.validation_rules:
            rule = self.validation_rules[rule_id]
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            logger.info(f"Updated validation rule: {rule_id}")
            return True
        return False
    
    async def validate_order(self, order: Order, user_twin_agent: Optional[UserTwinAgent] = None,
                           vendor_communication_session: Optional[CommunicationSession] = None) -> ValidationReport:
        """Perform comprehensive order validation with agent reasoning."""
        validation_session_id = f"validation_{int(time.time())}_{order.order_id}"
        
        logger.info(f"Starting order validation for order: {order.order_id}")
        
        # Create validation report
        report = ValidationReport(
            order_id=order.order_id,
            validation_session_id=validation_session_id,
            overall_status=ValidationStatus.PENDING
        )
        
        # Add initial agent reasoning
        report.agent_reasoning.append(f"Starting validation for order {order.order_id}")
        report.agent_reasoning.append(f"Order contains {len(order.items)} items with total amount ${order.total_amount:.2f}")
        
        # Run validation rules
        for rule_id, rule in self.validation_rules.items():
            if not rule.is_enabled:
                continue
            
            logger.debug(f"Running validation rule: {rule.name}")
            result = await self._run_validation_rule(rule, order, user_twin_agent, vendor_communication_session)
            report.add_result(result)
            
            # Add agent reasoning based on result
            if result.status == ValidationStatus.PASSED:
                report.agent_reasoning.append(f"✓ {rule.name}: {result.message}")
            elif result.status == ValidationStatus.FAILED:
                report.agent_reasoning.append(f"✗ {rule.name}: {result.message}")
                if result.severity == ValidationSeverity.CRITICAL:
                    report.agent_reasoning.append(f"CRITICAL: {rule.name} failed - order may need to be rejected")
            elif result.status == ValidationStatus.WARNING:
                report.agent_reasoning.append(f"⚠ {rule.name}: {result.message}")
        
        # Finalize validation report
        report.completed_at = datetime.now()
        
        # Determine final status
        if not report.get_failed_results():
            if report.get_warning_results():
                report.overall_status = ValidationStatus.WARNING
                report.agent_reasoning.append("Order validation completed with warnings")
            else:
                report.overall_status = ValidationStatus.PASSED
                report.agent_reasoning.append("Order validation completed successfully")
        else:
            critical_failures = [r for r in report.get_failed_results() 
                               if r.severity == ValidationSeverity.CRITICAL]
            if critical_failures:
                report.overall_status = ValidationStatus.FAILED
                report.agent_reasoning.append("Order validation failed due to critical issues")
            else:
                report.overall_status = ValidationStatus.WARNING
                report.agent_reasoning.append("Order validation completed with non-critical failures")
        
        # Save validation report
        self.validation_history.append(report)
        self._save_validation_history()
        
        # Record metrics
        record_metric("order_validation.completed", 1, MetricType.COUNTER, {
            "order_id": order.order_id,
            "validation_status": report.overall_status.value,
            "rules_checked": len(report.validation_results),
            "failures": len(report.get_failed_results()),
            "warnings": len(report.get_warning_results())
        })
        
        logger.info(f"Order validation completed with status: {report.overall_status.value}")
        return report
    
    async def _run_validation_rule(self, rule: ValidationRule, order: Order,
                                 user_twin_agent: Optional[UserTwinAgent] = None,
                                 vendor_communication_session: Optional[CommunicationSession] = None) -> ValidationResult:
        """Run a specific validation rule."""
        reasoning = []
        suggestions = []
        
        try:
            if rule.rule_id == "vendor_availability":
                result = await self._validate_vendor_availability(rule, order, reasoning, suggestions)
            elif rule.rule_id == "minimum_order_amount":
                result = await self._validate_minimum_order_amount(rule, order, reasoning, suggestions)
            elif rule.rule_id == "payment_method_validation":
                result = await self._validate_payment_method(rule, order, reasoning, suggestions)
            elif rule.rule_id == "delivery_address_validation":
                result = await self._validate_delivery_address(rule, order, reasoning, suggestions)
            elif rule.rule_id == "inventory_check":
                result = await self._validate_inventory(rule, order, reasoning, suggestions)
            elif rule.rule_id == "user_preference_compatibility":
                result = await self._validate_user_preferences(rule, order, user_twin_agent, reasoning, suggestions)
            elif rule.rule_id == "order_size_reasonableness":
                result = await self._validate_order_size(rule, order, reasoning, suggestions)
            elif rule.rule_id == "security_screening":
                result = await self._validate_security(rule, order, reasoning, suggestions)
            else:
                result = ValidationResult(
                    rule_id=rule.rule_id,
                    validation_type=rule.validation_type,
                    status=ValidationStatus.SKIPPED,
                    severity=rule.severity,
                    message=f"Unknown validation rule: {rule.rule_id}",
                    agent_reasoning=reasoning,
                    suggestions=suggestions
                )
            
            result.agent_reasoning.extend(reasoning)
            result.suggestions.extend(suggestions)
            return result
            
        except Exception as e:
            logger.error(f"Error running validation rule {rule.rule_id}: {e}")
            return ValidationResult(
                rule_id=rule.rule_id,
                validation_type=rule.validation_type,
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation rule failed with error: {str(e)}",
                agent_reasoning=reasoning + [f"Error during validation: {str(e)}"],
                suggestions=suggestions
            )
    
    async def _validate_vendor_availability(self, rule: ValidationRule, order: Order,
                                          reasoning: List[str], suggestions: List[str]) -> ValidationResult:
        """Validate vendor availability."""
        reasoning.append(f"Checking vendor availability for {order.vendor.name}")
        
        if not order.vendor.is_available:
            reasoning.append("Vendor is marked as unavailable")
            suggestions.append("Consider selecting an alternative vendor")
            return ValidationResult(
                rule_id=rule.rule_id,
                validation_type=rule.validation_type,
                status=ValidationStatus.FAILED,
                severity=rule.severity,
                message="Vendor is currently unavailable",
                agent_reasoning=reasoning,
                suggestions=suggestions
            )
        
        # Check if vendor is within business hours (simplified)
        current_hour = datetime.now().hour
        if current_hour < 8 or current_hour > 22:
            reasoning.append(f"Current time is {current_hour}:00, checking if vendor operates 24/7")
            suggestions.append("Confirm vendor operating hours")
            return ValidationResult(
                rule_id=rule.rule_id,
                validation_type=rule.validation_type,
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.LOW,
                message=f"Order placed outside typical business hours ({current_hour}:00)",
                agent_reasoning=reasoning,
                suggestions=suggestions
            )
        
        reasoning.append("Vendor availability check passed")
        return ValidationResult(
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            status=ValidationStatus.PASSED,
            severity=rule.severity,
            message="Vendor is available and accepting orders",
            agent_reasoning=reasoning,
            suggestions=suggestions
        )
    
    async def _validate_minimum_order_amount(self, rule: ValidationRule, order: Order,
                                           reasoning: List[str], suggestions: List[str]) -> ValidationResult:
        """Validate minimum order amount."""
        reasoning.append(f"Checking minimum order amount: ${order.vendor.minimum_order:.2f}")
        reasoning.append(f"Order total amount: ${order.total_amount:.2f}")
        
        tolerance = rule.parameters.get("tolerance_percentage", 0.05)
        min_with_tolerance = order.vendor.minimum_order * (1 - tolerance)
        
        if order.total_amount < min_with_tolerance:
            reasoning.append(f"Order amount ${order.total_amount:.2f} is below minimum ${order.vendor.minimum_order:.2f}")
            suggestions.append(f"Add items to reach minimum order of ${order.vendor.minimum_order:.2f}")
            suggestions.append("Consider selecting items with higher value")
            
            return ValidationResult(
                rule_id=rule.rule_id,
                validation_type=rule.validation_type,
                status=ValidationStatus.FAILED,
                severity=rule.severity,
                message=f"Order amount ${order.total_amount:.2f} is below vendor minimum ${order.vendor.minimum_order:.2f}",
                agent_reasoning=reasoning,
                suggestions=suggestions
            )
        
        reasoning.append("Minimum order amount validation passed")
        return ValidationResult(
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            status=ValidationStatus.PASSED,
            severity=rule.severity,
            message="Order meets vendor's minimum order requirement",
            agent_reasoning=reasoning,
            suggestions=suggestions
        )
    
    async def _validate_payment_method(self, rule: ValidationRule, order: Order,
                                     reasoning: List[str], suggestions: List[str]) -> ValidationResult:
        """Validate payment method."""
        reasoning.append("Validating payment method and processing capability")
        
        # Simulate payment validation
        if order.total_amount > 100.0:
            reasoning.append(f"Large order amount ${order.total_amount:.2f} requires additional payment verification")
            suggestions.append("Consider splitting large orders into smaller transactions")
        
        if order.payment_status == PaymentStatus.FAILED:
            reasoning.append("Payment status indicates previous failure")
            suggestions.append("Update payment information")
            suggestions.append("Try alternative payment method")
            
            return ValidationResult(
                rule_id=rule.rule_id,
                validation_type=rule.validation_type,
                status=ValidationStatus.FAILED,
                severity=rule.severity,
                message="Payment method validation failed",
                agent_reasoning=reasoning,
                suggestions=suggestions
            )
        
        reasoning.append("Payment method validation passed")
        return ValidationResult(
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            status=ValidationStatus.PASSED,
            severity=rule.severity,
            message="Payment method is valid and accepted",
            agent_reasoning=reasoning,
            suggestions=suggestions
        )
    
    async def _validate_delivery_address(self, rule: ValidationRule, order: Order,
                                       reasoning: List[str], suggestions: List[str]) -> ValidationResult:
        """Validate delivery address."""
        if order.order_type != OrderType.DELIVERY:
            reasoning.append("Order is not for delivery, skipping address validation")
            return ValidationResult(
                rule_id=rule.rule_id,
                validation_type=rule.validation_type,
                status=ValidationStatus.SKIPPED,
                severity=rule.severity,
                message="Order is not for delivery",
                agent_reasoning=reasoning,
                suggestions=suggestions
            )
        
        reasoning.append(f"Validating delivery address: {order.delivery_address}")
        
        if not order.delivery_address:
            reasoning.append("No delivery address provided for delivery order")
            suggestions.append("Provide a valid delivery address")
            
            return ValidationResult(
                rule_id=rule.rule_id,
                validation_type=rule.validation_type,
                status=ValidationStatus.FAILED,
                severity=rule.severity,
                message="Delivery address is required for delivery orders",
                agent_reasoning=reasoning,
                suggestions=suggestions
            )
        
        # Simulate address validation
        max_distance = rule.parameters.get("max_distance_miles", 10)
        reasoning.append(f"Checking if address is within {max_distance} miles of vendor")
        
        # Simple validation - in real system would check actual distance
        if "remote" in order.delivery_address.lower() or "rural" in order.delivery_address.lower():
            reasoning.append("Address appears to be in remote area")
            suggestions.append("Confirm delivery area coverage with vendor")
            
            return ValidationResult(
                rule_id=rule.rule_id,
                validation_type=rule.validation_type,
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.MEDIUM,
                message="Delivery address may be outside normal delivery area",
                agent_reasoning=reasoning,
                suggestions=suggestions
            )
        
        reasoning.append("Delivery address validation passed")
        return ValidationResult(
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            status=ValidationStatus.PASSED,
            severity=rule.severity,
            message="Delivery address is valid and within delivery area",
            agent_reasoning=reasoning,
            suggestions=suggestions
        )
    
    async def _validate_inventory(self, rule: ValidationRule, order: Order,
                                reasoning: List[str], suggestions: List[str]) -> ValidationResult:
        """Validate inventory availability."""
        reasoning.append("Checking inventory availability for all items")
        
        unavailable_items = []
        for item in order.items:
            reasoning.append(f"Checking availability for {item.name}")
            # Simulate inventory check
            if "out of stock" in item.name.lower() or "unavailable" in item.name.lower():
                unavailable_items.append(item.name)
                reasoning.append(f"{item.name} appears to be unavailable")
        
        if unavailable_items:
            suggestions.append("Select alternative items")
            suggestions.append("Contact vendor for availability confirmation")
            
            return ValidationResult(
                rule_id=rule.rule_id,
                validation_type=rule.validation_type,
                status=ValidationStatus.FAILED,
                severity=rule.severity,
                message=f"Items unavailable: {', '.join(unavailable_items)}",
                agent_reasoning=reasoning,
                suggestions=suggestions
            )
        
        reasoning.append("All items are available in inventory")
        return ValidationResult(
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            status=ValidationStatus.PASSED,
            severity=rule.severity,
            message="All requested items are available",
            agent_reasoning=reasoning,
            suggestions=suggestions
        )
    
    async def _validate_user_preferences(self, rule: ValidationRule, order: Order,
                                       user_twin_agent: Optional[UserTwinAgent],
                                       reasoning: List[str], suggestions: List[str]) -> ValidationResult:
        """Validate user preference compatibility."""
        if not user_twin_agent:
            reasoning.append("No user twin agent provided, skipping preference validation")
            return ValidationResult(
                rule_id=rule.rule_id,
                validation_type=rule.validation_type,
                status=ValidationStatus.SKIPPED,
                severity=rule.severity,
                message="User preferences not available",
                agent_reasoning=reasoning,
                suggestions=suggestions
            )
        
        reasoning.append("Checking user preference compatibility")
        
        conflicts = []
        
        # Check dietary preferences
        dietary_prefs = user_twin_agent.get_preferences_by_category(PreferenceCategory.FOOD)
        for pref in dietary_prefs:
            if pref.key == "dietary":
                user_dietary = pref.value.lower()
                for item in order.items:
                    item_dietary = [d.lower() for d in item.dietary_requirements]
                    
                    if user_dietary == "vegetarian" and any(word in item.name.lower() 
                                                         for word in ["meat", "beef", "chicken", "pork"]):
                        conflicts.append(f"{item.name} contains meat but user prefers vegetarian")
                        reasoning.append(f"Dietary conflict: {item.name} vs vegetarian preference")
        
        # Check cost preferences
        cost_prefs = user_twin_agent.get_preferences_by_category(PreferenceCategory.COST)
        for pref in cost_prefs:
            if pref.key == "budget":
                if pref.value == "cheap" and order.total_amount > 25.0:
                    conflicts.append(f"Order total ${order.total_amount:.2f} exceeds cheap budget preference")
                    reasoning.append("Budget conflict: order exceeds cheap preference threshold")
        
        if conflicts:
            suggestions.append("Review items for dietary compatibility")
            suggestions.append("Consider budget-friendly alternatives")
            
            return ValidationResult(
                rule_id=rule.rule_id,
                validation_type=rule.validation_type,
                status=ValidationStatus.WARNING,
                severity=rule.severity,
                message=f"User preference conflicts: {'; '.join(conflicts)}",
                agent_reasoning=reasoning,
                suggestions=suggestions
            )
        
        reasoning.append("User preference validation passed")
        return ValidationResult(
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            status=ValidationStatus.PASSED,
            severity=rule.severity,
            message="Order is compatible with user preferences",
            agent_reasoning=reasoning,
            suggestions=suggestions
        )
    
    async def _validate_order_size(self, rule: ValidationRule, order: Order,
                                 reasoning: List[str], suggestions: List[str]) -> ValidationResult:
        """Validate order size reasonableness."""
        reasoning.append("Checking order size reasonableness")
        
        max_items_per_person = rule.parameters.get("max_items_per_person", 3)
        max_total_amount = rule.parameters.get("max_total_amount", 200.0)
        
        total_items = sum(item.quantity for item in order.items)
        reasoning.append(f"Total items: {total_items}, Total amount: ${order.total_amount:.2f}")
        
        issues = []
        
        if total_items > max_items_per_person * 4:  # Assume 4 people max
            issues.append(f"Large number of items ({total_items}) for typical order")
            reasoning.append("Order contains unusually high number of items")
        
        if order.total_amount > max_total_amount:
            issues.append(f"High order value (${order.total_amount:.2f})")
            reasoning.append("Order amount is unusually high")
        
        if issues:
            suggestions.append("Confirm order size with customer")
            suggestions.append("Consider splitting large orders")
            
            return ValidationResult(
                rule_id=rule.rule_id,
                validation_type=rule.validation_type,
                status=ValidationStatus.WARNING,
                severity=rule.severity,
                message=f"Order size concerns: {'; '.join(issues)}",
                agent_reasoning=reasoning,
                suggestions=suggestions
            )
        
        reasoning.append("Order size validation passed")
        return ValidationResult(
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            status=ValidationStatus.PASSED,
            severity=rule.severity,
            message="Order size is reasonable",
            agent_reasoning=reasoning,
            suggestions=suggestions
        )
    
    async def _validate_security(self, rule: ValidationRule, order: Order,
                               reasoning: List[str], suggestions: List[str]) -> ValidationResult:
        """Perform security screening."""
        reasoning.append("Performing security screening on order data")
        
        # Check for potential SQL injection in special instructions
        if order.special_instructions:
            suspicious_patterns = ["';", "--", "/*", "*/", "xp_", "sp_"]
            for pattern in suspicious_patterns:
                if pattern in order.special_instructions.lower():
                    reasoning.append(f"Potential SQL injection pattern detected: {pattern}")
                    suggestions.append("Sanitize special instructions input")
                    
                    return ValidationResult(
                        rule_id=rule.rule_id,
                        validation_type=rule.validation_type,
                        status=ValidationStatus.FAILED,
                        severity=rule.severity,
                        message=f"Potential security threat detected in special instructions",
                        agent_reasoning=reasoning,
                        suggestions=suggestions
                    )
        
        # Check for XSS patterns
        xss_patterns = ["<script", "javascript:", "onload=", "onerror="]
        for item in order.items:
            for pattern in xss_patterns:
                if pattern in item.name.lower() or pattern in item.special_instructions.lower():
                    reasoning.append(f"Potential XSS pattern detected: {pattern}")
                    suggestions.append("Sanitize item names and special instructions")
                    
                    return ValidationResult(
                        rule_id=rule.rule_id,
                        validation_type=rule.validation_type,
                        status=ValidationStatus.FAILED,
                        severity=rule.severity,
                        message="Potential XSS threat detected in item data",
                        agent_reasoning=reasoning,
                        suggestions=suggestions
                    )
        
        reasoning.append("Security screening passed")
        return ValidationResult(
            rule_id=rule.rule_id,
            validation_type=rule.validation_type,
            status=ValidationStatus.PASSED,
            severity=rule.severity,
            message="Security screening completed successfully",
            agent_reasoning=reasoning,
            suggestions=suggestions
        )
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self.validation_history:
            return {"total_validations": 0}
        
        total_validations = len(self.validation_history)
        passed_validations = len([r for r in self.validation_history if r.overall_status == ValidationStatus.PASSED])
        failed_validations = len([r for r in self.validation_history if r.overall_status == ValidationStatus.FAILED])
        warning_validations = len([r for r in self.validation_history if r.overall_status == ValidationStatus.WARNING])
        
        # Rule performance statistics
        rule_stats = {}
        for rule_id in self.validation_rules.keys():
            rule_failures = 0
            rule_total = 0
            
            for report in self.validation_history:
                for result in report.validation_results:
                    if result.rule_id == rule_id:
                        rule_total += 1
                        if result.status == ValidationStatus.FAILED:
                            rule_failures += 1
            
            if rule_total > 0:
                rule_stats[rule_id] = {
                    "total_runs": rule_total,
                    "failures": rule_failures,
                    "success_rate": (rule_total - rule_failures) / rule_total
                }
        
        return {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "failed_validations": failed_validations,
            "warning_validations": warning_validations,
            "success_rate": passed_validations / total_validations if total_validations > 0 else 0,
            "rule_performance": rule_stats
        }
