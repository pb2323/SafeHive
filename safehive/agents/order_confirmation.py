"""
Order Confirmation Workflows with Agent Reasoning

This module implements intelligent order confirmation workflows for the SafeHive AI Security Sandbox,
providing multi-step confirmation processes with agent reasoning and approval management.
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
from .order_validation import ValidationReport, ValidationStatus, ValidationSeverity
from .user_twin import UserTwinAgent, PreferenceCategory
from .vendor_communication import CommunicationSession, CommunicationIntent
from ..utils.logger import get_logger
from ..utils.metrics import record_metric, MetricType

logger = get_logger(__name__)


class ConfirmationStatus(Enum):
    """Status of confirmation operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ESCALATED = "escalated"


class ApprovalType(Enum):
    """Types of approvals required."""
    AUTOMATIC = "automatic"
    USER_CONFIRMATION = "user_confirmation"
    VENDOR_CONFIRMATION = "vendor_confirmation"
    PAYMENT_APPROVAL = "payment_approval"
    MANAGER_APPROVAL = "manager_approval"
    SECURITY_REVIEW = "security_review"
    CUSTOM = "custom"


class ConfirmationStep(Enum):
    """Steps in the confirmation process."""
    VALIDATION_REVIEW = "validation_review"
    USER_CONFIRMATION = "user_confirmation"
    VENDOR_CONFIRMATION = "vendor_confirmation"
    PAYMENT_PROCESSING = "payment_processing"
    INVENTORY_RESERVATION = "inventory_reservation"
    SECURITY_REVIEW = "security_review"
    FINAL_APPROVAL = "final_approval"


@dataclass
class ApprovalRequirement:
    """Represents an approval requirement."""
    approval_id: str
    approval_type: ApprovalType
    description: str
    required_by: str  # Who needs to approve
    timeout_minutes: int = 30
    is_required: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "approval_id": self.approval_id,
            "approval_type": self.approval_type.value,
            "description": self.description,
            "required_by": self.required_by,
            "timeout_minutes": self.timeout_minutes,
            "is_required": self.is_required,
            "conditions": self.conditions
        }


@dataclass
class ApprovalResult:
    """Result of an approval operation."""
    approval_id: str
    approval_type: ApprovalType
    status: ConfirmationStatus
    approver: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "approval_id": self.approval_id,
            "approval_type": self.approval_type.value,
            "status": self.status.value,
            "approver": self.approver,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }


@dataclass
class ConfirmationWorkflow:
    """Represents a confirmation workflow."""
    workflow_id: str
    name: str
    description: str
    steps: List[ConfirmationStep] = field(default_factory=list)
    approval_requirements: List[ApprovalRequirement] = field(default_factory=list)
    auto_approval_conditions: List[Dict[str, Any]] = field(default_factory=list)
    timeout_minutes: int = 30
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "steps": [step.value for step in self.steps],
            "approval_requirements": [req.to_dict() for req in self.approval_requirements],
            "auto_approval_conditions": self.auto_approval_conditions,
            "timeout_minutes": self.timeout_minutes,
            "is_active": self.is_active
        }


@dataclass
class ConfirmationSession:
    """Represents a confirmation session for an order."""
    session_id: str
    order_id: str
    workflow_id: str
    status: ConfirmationStatus
    current_step: Optional[ConfirmationStep] = None
    approval_results: List[ApprovalResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    agent_reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_approval_result(self, result: ApprovalResult) -> None:
        """Add an approval result to the session."""
        self.approval_results.append(result)
        self.updated_at = datetime.now()
    
    def get_approval_by_type(self, approval_type: ApprovalType) -> Optional[ApprovalResult]:
        """Get approval result by type."""
        for result in self.approval_results:
            if result.approval_type == approval_type:
                return result
        return None
    
    def is_all_required_approvals_complete(self, workflow: ConfirmationWorkflow) -> bool:
        """Check if all required approvals are complete."""
        required_approvals = [req for req in workflow.approval_requirements if req.is_required]
        
        for req in required_approvals:
            result = self.get_approval_by_type(req.approval_type)
            if not result or result.status not in [ConfirmationStatus.APPROVED]:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "order_id": self.order_id,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "current_step": self.current_step.value if self.current_step else None,
            "approval_results": [result.to_dict() for result in self.approval_results],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "agent_reasoning": self.agent_reasoning,
            "metadata": self.metadata
        }


class OrderConfirmationManager:
    """Manager for order confirmation workflows with agent reasoning."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_order_confirmation"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Confirmation components
        self.confirmation_workflows: Dict[str, ConfirmationWorkflow] = {}
        self.active_sessions: Dict[str, ConfirmationSession] = {}
        self.confirmation_history: List[ConfirmationSession] = []
        
        # Initialize default workflows
        self._initialize_default_workflows()
        
        # Load confirmation history
        self._load_confirmation_history()
        
        logger.info("Order Confirmation Manager initialized")
    
    def _initialize_default_workflows(self) -> None:
        """Initialize default confirmation workflows."""
        # Standard order confirmation workflow
        standard_workflow = ConfirmationWorkflow(
            workflow_id="standard_order_confirmation",
            name="Standard Order Confirmation",
            description="Standard workflow for confirming food orders",
            steps=[
                ConfirmationStep.VALIDATION_REVIEW,
                ConfirmationStep.USER_CONFIRMATION,
                ConfirmationStep.VENDOR_CONFIRMATION,
                ConfirmationStep.PAYMENT_PROCESSING,
                ConfirmationStep.FINAL_APPROVAL
            ],
            approval_requirements=[
                ApprovalRequirement(
                    approval_id="validation_approval",
                    approval_type=ApprovalType.AUTOMATIC,
                    description="Automatic approval based on validation results",
                    required_by="system",
                    timeout_minutes=5,
                    conditions={"validation_status": ["passed", "warning"]}
                ),
                ApprovalRequirement(
                    approval_id="user_confirmation",
                    approval_type=ApprovalType.USER_CONFIRMATION,
                    description="User confirmation of order details",
                    required_by="user",
                    timeout_minutes=10,
                    is_required=True
                ),
                ApprovalRequirement(
                    approval_id="vendor_confirmation",
                    approval_type=ApprovalType.VENDOR_CONFIRMATION,
                    description="Vendor confirmation of order acceptance",
                    required_by="vendor",
                    timeout_minutes=15,
                    is_required=True
                ),
                ApprovalRequirement(
                    approval_id="payment_approval",
                    approval_type=ApprovalType.PAYMENT_APPROVAL,
                    description="Payment processing approval",
                    required_by="payment_system",
                    timeout_minutes=10,
                    is_required=True
                )
            ],
            auto_approval_conditions=[
                {"condition": "low_risk_order", "criteria": {"total_amount": {"max": 50.0}}},
                {"condition": "trusted_user", "criteria": {"user_trust_score": {"min": 0.8}}}
            ],
            timeout_minutes=30
        )
        
        # High-value order confirmation workflow
        high_value_workflow = ConfirmationWorkflow(
            workflow_id="high_value_order_confirmation",
            name="High Value Order Confirmation",
            description="Enhanced workflow for high-value orders",
            steps=[
                ConfirmationStep.VALIDATION_REVIEW,
                ConfirmationStep.SECURITY_REVIEW,
                ConfirmationStep.USER_CONFIRMATION,
                ConfirmationStep.VENDOR_CONFIRMATION,
                ConfirmationStep.PAYMENT_PROCESSING,
                ConfirmationStep.FINAL_APPROVAL
            ],
            approval_requirements=[
                ApprovalRequirement(
                    approval_id="validation_approval",
                    approval_type=ApprovalType.AUTOMATIC,
                    description="Automatic approval based on validation results",
                    required_by="system",
                    timeout_minutes=5
                ),
                ApprovalRequirement(
                    approval_id="security_review",
                    approval_type=ApprovalType.SECURITY_REVIEW,
                    description="Security review for high-value order",
                    required_by="security_team",
                    timeout_minutes=20,
                    is_required=True
                ),
                ApprovalRequirement(
                    approval_id="manager_approval",
                    approval_type=ApprovalType.MANAGER_APPROVAL,
                    description="Manager approval for high-value order",
                    required_by="manager",
                    timeout_minutes=30,
                    is_required=True
                ),
                ApprovalRequirement(
                    approval_id="user_confirmation",
                    approval_type=ApprovalType.USER_CONFIRMATION,
                    description="User confirmation with additional verification",
                    required_by="user",
                    timeout_minutes=15,
                    is_required=True
                ),
                ApprovalRequirement(
                    approval_id="vendor_confirmation",
                    approval_type=ApprovalType.VENDOR_CONFIRMATION,
                    description="Vendor confirmation with inventory check",
                    required_by="vendor",
                    timeout_minutes=20,
                    is_required=True
                ),
                ApprovalRequirement(
                    approval_id="payment_approval",
                    approval_type=ApprovalType.PAYMENT_APPROVAL,
                    description="Enhanced payment processing approval",
                    required_by="payment_system",
                    timeout_minutes=15,
                    is_required=True
                )
            ],
            timeout_minutes=60
        )
        
        self.confirmation_workflows["standard_order_confirmation"] = standard_workflow
        self.confirmation_workflows["high_value_order_confirmation"] = high_value_workflow
    
    def _load_confirmation_history(self) -> None:
        """Load confirmation history from storage."""
        history_file = self.storage_path / "confirmation_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for session_data in data:
                        session = self._reconstruct_confirmation_session(session_data)
                        if session:
                            self.confirmation_history.append(session)
                logger.info(f"Loaded {len(self.confirmation_history)} confirmation sessions")
            except Exception as e:
                logger.error(f"Failed to load confirmation history: {e}")
    
    def _save_confirmation_history(self) -> None:
        """Save confirmation history to storage."""
        history_file = self.storage_path / "confirmation_history.json"
        try:
            data = [session.to_dict() for session in self.confirmation_history]
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved confirmation history")
        except Exception as e:
            logger.error(f"Failed to save confirmation history: {e}")
    
    def _reconstruct_confirmation_session(self, data: Dict[str, Any]) -> Optional[ConfirmationSession]:
        """Reconstruct ConfirmationSession from stored data."""
        try:
            approval_results = []
            for result_data in data.get("approval_results", []):
                result = ApprovalResult(
                    approval_id=result_data["approval_id"],
                    approval_type=ApprovalType(result_data["approval_type"]),
                    status=ConfirmationStatus(result_data["status"]),
                    approver=result_data["approver"],
                    message=result_data["message"],
                    timestamp=datetime.fromisoformat(result_data["timestamp"]),
                    reasoning=result_data.get("reasoning", []),
                    metadata=result_data.get("metadata", {})
                )
                approval_results.append(result)
            
            session = ConfirmationSession(
                session_id=data["session_id"],
                order_id=data["order_id"],
                workflow_id=data["workflow_id"],
                status=ConfirmationStatus(data["status"]),
                current_step=ConfirmationStep(data["current_step"]) if data.get("current_step") else None,
                approval_results=approval_results,
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
                agent_reasoning=data.get("agent_reasoning", []),
                metadata=data.get("metadata", {})
            )
            
            return session
        except Exception as e:
            logger.error(f"Failed to reconstruct confirmation session: {e}")
            return None
    
    def add_confirmation_workflow(self, workflow: ConfirmationWorkflow) -> None:
        """Add a new confirmation workflow."""
        self.confirmation_workflows[workflow.workflow_id] = workflow
        logger.info(f"Added confirmation workflow: {workflow.name}")
    
    def get_workflow_for_order(self, order: Order, validation_report: ValidationReport) -> ConfirmationWorkflow:
        """Determine the appropriate workflow for an order."""
        # Check for high-value order conditions
        if order.total_amount > 100.0:
            return self.confirmation_workflows["high_value_order_confirmation"]
        
        # Check for validation issues that require enhanced workflow
        critical_failures = [r for r in validation_report.get_failed_results() 
                           if r.severity == ValidationSeverity.CRITICAL]
        if critical_failures:
            return self.confirmation_workflows["high_value_order_confirmation"]
        
        # Default to standard workflow
        return self.confirmation_workflows["standard_order_confirmation"]
    
    async def start_confirmation_workflow(self, order: Order, validation_report: ValidationReport,
                                        user_twin_agent: Optional[UserTwinAgent] = None,
                                        vendor_communication_session: Optional[CommunicationSession] = None) -> ConfirmationSession:
        """Start a confirmation workflow for an order."""
        session_id = f"confirmation_{int(time.time())}_{order.order_id}"
        
        logger.info(f"Starting confirmation workflow for order: {order.order_id}")
        
        # Determine appropriate workflow
        workflow = self.get_workflow_for_order(order, validation_report)
        
        # Create confirmation session
        session = ConfirmationSession(
            session_id=session_id,
            order_id=order.order_id,
            workflow_id=workflow.workflow_id,
            status=ConfirmationStatus.PENDING,
            current_step=workflow.steps[0] if workflow.steps else None,
            metadata={
                "order_total": order.total_amount,
                "vendor_id": order.vendor.vendor_id,
                "validation_report_id": validation_report.validation_session_id
            }
        )
        
        session.agent_reasoning.append(f"Starting confirmation workflow: {workflow.name}")
        session.agent_reasoning.append(f"Order total: ${order.total_amount:.2f}, Vendor: {order.vendor.name}")
        
        # Add to active sessions
        self.active_sessions[session_id] = session
        
        # Start the workflow execution
        await self._execute_workflow_step(session, workflow, order, validation_report, 
                                        user_twin_agent, vendor_communication_session)
        
        return session
    
    async def _execute_workflow_step(self, session: ConfirmationSession, workflow: ConfirmationWorkflow,
                                   order: Order, validation_report: ValidationReport,
                                   user_twin_agent: Optional[UserTwinAgent] = None,
                                   vendor_communication_session: Optional[CommunicationSession] = None) -> None:
        """Execute the current step of the confirmation workflow."""
        if not session.current_step:
            logger.warning(f"No current step for session {session.session_id}")
            return
        
        logger.info(f"Executing workflow step: {session.current_step.value}")
        session.agent_reasoning.append(f"Executing step: {session.current_step.value}")
        
        # Execute step-specific logic
        if session.current_step == ConfirmationStep.VALIDATION_REVIEW:
            await self._execute_validation_review(session, workflow, order, validation_report)
        elif session.current_step == ConfirmationStep.USER_CONFIRMATION:
            await self._execute_user_confirmation(session, workflow, order, user_twin_agent)
        elif session.current_step == ConfirmationStep.VENDOR_CONFIRMATION:
            await self._execute_vendor_confirmation(session, workflow, order, vendor_communication_session)
        elif session.current_step == ConfirmationStep.PAYMENT_PROCESSING:
            await self._execute_payment_processing(session, workflow, order)
        elif session.current_step == ConfirmationStep.FINAL_APPROVAL:
            await self._execute_final_approval(session, workflow, order)
        
        # Check if workflow is complete
        if session.is_all_required_approvals_complete(workflow):
            session.status = ConfirmationStatus.APPROVED
            session.completed_at = datetime.now()
            session.agent_reasoning.append("All required approvals received - order confirmed")
            
            # Move to history
            self.confirmation_history.append(session)
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
            
            self._save_confirmation_history()
            
            # Record metrics
            record_metric("order_confirmation.completed", 1, MetricType.COUNTER, {
                "order_id": order.order_id,
                "workflow_id": workflow.workflow_id,
                "approval_time_minutes": (session.completed_at - session.created_at).total_seconds() / 60
            })
        else:
            # Move to next step
            await self._move_to_next_step(session, workflow)
    
    async def _execute_validation_review(self, session: ConfirmationSession, workflow: ConfirmationWorkflow,
                                       order: Order, validation_report: ValidationReport) -> None:
        """Execute validation review step."""
        session.agent_reasoning.append("Reviewing validation results")
        
        # Check if validation passed
        if validation_report.overall_status in [ValidationStatus.PASSED, ValidationStatus.WARNING]:
            approval_result = ApprovalResult(
                approval_id="validation_approval",
                approval_type=ApprovalType.AUTOMATIC,
                status=ConfirmationStatus.APPROVED,
                approver="system",
                message="Validation review passed",
                reasoning=["Validation status is acceptable", "No critical issues found"]
            )
            session.agent_reasoning.append("✓ Validation review passed")
        else:
            approval_result = ApprovalResult(
                approval_id="validation_approval",
                approval_type=ApprovalType.AUTOMATIC,
                status=ConfirmationStatus.REJECTED,
                approver="system",
                message="Validation review failed",
                reasoning=["Validation status is unacceptable", "Critical issues found"]
            )
            session.agent_reasoning.append("✗ Validation review failed")
            session.status = ConfirmationStatus.REJECTED
        
        session.add_approval_result(approval_result)
    
    async def _execute_user_confirmation(self, session: ConfirmationSession, workflow: ConfirmationWorkflow,
                                       order: Order, user_twin_agent: Optional[UserTwinAgent]) -> None:
        """Execute user confirmation step."""
        session.agent_reasoning.append("Requesting user confirmation")
        
        # Simulate user confirmation (in real system, this would be actual user interaction)
        # For now, we'll auto-approve based on user preferences if available
        if user_twin_agent:
            # Check user's confirmation preference
            speed_prefs = user_twin_agent.get_preferences_by_category(PreferenceCategory.SPEED)
            auto_confirm = any(pref.key == "auto_confirm" and pref.value == "enabled" for pref in speed_prefs)
            
            if auto_confirm:
                approval_result = ApprovalResult(
                    approval_id="user_confirmation",
                    approval_type=ApprovalType.USER_CONFIRMATION,
                    status=ConfirmationStatus.APPROVED,
                    approver="user",
                    message="User auto-confirmation based on preferences",
                    reasoning=["User has auto-confirmation enabled", "Order matches user preferences"]
                )
                session.agent_reasoning.append("✓ User auto-confirmation approved")
            else:
                # Simulate user approval for demo
                approval_result = ApprovalResult(
                    approval_id="user_confirmation",
                    approval_type=ApprovalType.USER_CONFIRMATION,
                    status=ConfirmationStatus.APPROVED,
                    approver="user",
                    message="User manually confirmed order",
                    reasoning=["User reviewed order details", "User approved the order"]
                )
                session.agent_reasoning.append("✓ User confirmation received")
        else:
            # Simulate user approval for demo
            approval_result = ApprovalResult(
                approval_id="user_confirmation",
                approval_type=ApprovalType.USER_CONFIRMATION,
                status=ConfirmationStatus.APPROVED,
                approver="user",
                message="User confirmed order",
                reasoning=["User reviewed and approved order"]
            )
            session.agent_reasoning.append("✓ User confirmation received")
        
        session.add_approval_result(approval_result)
    
    async def _execute_vendor_confirmation(self, session: ConfirmationSession, workflow: ConfirmationWorkflow,
                                         order: Order, vendor_communication_session: Optional[CommunicationSession]) -> None:
        """Execute vendor confirmation step."""
        session.agent_reasoning.append("Requesting vendor confirmation")
        
        # Check if we have vendor communication session
        if vendor_communication_session:
            # Look for vendor confirmation in communication
            confirmation_messages = vendor_communication_session.get_messages_by_intent(CommunicationIntent.ORDER_CONFIRMATION)
            if confirmation_messages:
                last_confirmation = confirmation_messages[-1]
                if "confirm" in last_confirmation.content.lower():
                    approval_result = ApprovalResult(
                        approval_id="vendor_confirmation",
                        approval_type=ApprovalType.VENDOR_CONFIRMATION,
                        status=ConfirmationStatus.APPROVED,
                        approver="vendor",
                        message="Vendor confirmed order via communication",
                        reasoning=["Vendor responded positively", "Order accepted by vendor"]
                    )
                    session.agent_reasoning.append("✓ Vendor confirmation received via communication")
                else:
                    approval_result = ApprovalResult(
                        approval_id="vendor_confirmation",
                        approval_type=ApprovalType.VENDOR_CONFIRMATION,
                        status=ConfirmationStatus.REJECTED,
                        approver="vendor",
                        message="Vendor declined order via communication",
                        reasoning=["Vendor responded negatively", "Order declined by vendor"]
                    )
                    session.agent_reasoning.append("✗ Vendor declined order")
                    session.status = ConfirmationStatus.REJECTED
            else:
                # Simulate vendor confirmation for demo
                approval_result = ApprovalResult(
                    approval_id="vendor_confirmation",
                    approval_type=ApprovalType.VENDOR_CONFIRMATION,
                    status=ConfirmationStatus.APPROVED,
                    approver="vendor",
                    message="Vendor confirmed order",
                    reasoning=["Vendor accepted the order", "Inventory confirmed available"]
                )
                session.agent_reasoning.append("✓ Vendor confirmation received")
        else:
            # Simulate vendor confirmation for demo
            approval_result = ApprovalResult(
                approval_id="vendor_confirmation",
                approval_type=ApprovalType.VENDOR_CONFIRMATION,
                status=ConfirmationStatus.APPROVED,
                approver="vendor",
                message="Vendor confirmed order",
                reasoning=["Vendor accepted the order"]
            )
            session.agent_reasoning.append("✓ Vendor confirmation received")
        
        session.add_approval_result(approval_result)
    
    async def _execute_payment_processing(self, session: ConfirmationSession, workflow: ConfirmationWorkflow,
                                        order: Order) -> None:
        """Execute payment processing step."""
        session.agent_reasoning.append("Processing payment")
        
        # Simulate payment processing
        if order.total_amount > 0:
            approval_result = ApprovalResult(
                approval_id="payment_approval",
                approval_type=ApprovalType.PAYMENT_APPROVAL,
                status=ConfirmationStatus.APPROVED,
                approver="payment_system",
                message="Payment processed successfully",
                reasoning=["Payment method validated", "Funds available", "Transaction authorized"]
            )
            session.agent_reasoning.append("✓ Payment processed successfully")
        else:
            approval_result = ApprovalResult(
                approval_id="payment_approval",
                approval_type=ApprovalType.PAYMENT_APPROVAL,
                status=ConfirmationStatus.REJECTED,
                approver="payment_system",
                message="Payment processing failed",
                reasoning=["No payment amount specified", "Payment validation failed"]
            )
            session.agent_reasoning.append("✗ Payment processing failed")
            session.status = ConfirmationStatus.REJECTED
        
        session.add_approval_result(approval_result)
    
    async def _execute_final_approval(self, session: ConfirmationSession, workflow: ConfirmationWorkflow,
                                    order: Order) -> None:
        """Execute final approval step."""
        session.agent_reasoning.append("Final approval review")
        
        # Check all previous approvals
        if session.is_all_required_approvals_complete(workflow):
            approval_result = ApprovalResult(
                approval_id="final_approval",
                approval_type=ApprovalType.AUTOMATIC,
                status=ConfirmationStatus.APPROVED,
                approver="system",
                message="Final approval granted",
                reasoning=["All required approvals received", "Order ready for processing"]
            )
            session.agent_reasoning.append("✓ Final approval granted")
        else:
            approval_result = ApprovalResult(
                approval_id="final_approval",
                approval_type=ApprovalType.AUTOMATIC,
                status=ConfirmationStatus.REJECTED,
                approver="system",
                message="Final approval denied",
                reasoning=["Not all required approvals received", "Order cannot proceed"]
            )
            session.agent_reasoning.append("✗ Final approval denied")
            session.status = ConfirmationStatus.REJECTED
        
        session.add_approval_result(approval_result)
    
    async def _move_to_next_step(self, session: ConfirmationSession, workflow: ConfirmationWorkflow) -> None:
        """Move to the next step in the workflow."""
        if not session.current_step:
            return
        
        current_index = workflow.steps.index(session.current_step)
        if current_index < len(workflow.steps) - 1:
            session.current_step = workflow.steps[current_index + 1]
            session.agent_reasoning.append(f"Moving to next step: {session.current_step.value}")
        else:
            session.current_step = None
            session.agent_reasoning.append("Workflow completed")
    
    def get_confirmation_session(self, session_id: str) -> Optional[ConfirmationSession]:
        """Get a confirmation session by ID."""
        return self.active_sessions.get(session_id)
    
    def get_active_sessions(self) -> List[ConfirmationSession]:
        """Get all active confirmation sessions."""
        return list(self.active_sessions.values())
    
    def get_confirmation_statistics(self) -> Dict[str, Any]:
        """Get confirmation statistics."""
        total_sessions = len(self.confirmation_history) + len(self.active_sessions)
        
        if total_sessions == 0:
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "completed_sessions": 0,
                "status_counts": {},
                "average_approval_time_minutes": 0,
                "workflows_available": len(self.confirmation_workflows)
            }
        
        # Count sessions by status
        status_counts = {}
        for session in self.confirmation_history:
            status = session.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for session in self.active_sessions.values():
            status = session.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate average approval time
        completed_sessions = [s for s in self.confirmation_history if s.completed_at]
        if completed_sessions:
            total_time = sum((s.completed_at - s.created_at).total_seconds() for s in completed_sessions)
            avg_approval_time = total_time / len(completed_sessions) / 60  # in minutes
        else:
            avg_approval_time = 0
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.confirmation_history),
            "status_counts": status_counts,
            "average_approval_time_minutes": avg_approval_time,
            "workflows_available": len(self.confirmation_workflows)
        }
