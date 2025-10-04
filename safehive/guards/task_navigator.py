"""
Task Navigator - Task Constraint Enforcement and Agent Reasoning

This module implements a comprehensive task navigation system that enforces
original task constraints and ensures agents stay focused on their intended
purpose through intelligent reasoning and boundary monitoring.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.metrics import record_metric, MetricType

logger = get_logger(__name__)


class TaskType(Enum):
    """Types of tasks that can be navigated."""
    FOOD_ORDERING = "food_ordering"
    VENDOR_SEARCH = "vendor_search"
    PREFERENCE_MANAGEMENT = "preference_management"
    ORDER_TRACKING = "order_tracking"
    PAYMENT_PROCESSING = "payment_processing"
    CUSTOMER_SUPPORT = "customer_support"
    DATA_ANALYSIS = "data_analysis"
    SYSTEM_MONITORING = "system_monitoring"
    SECURITY_ANALYSIS = "security_analysis"
    GENERAL_ASSISTANCE = "general_assistance"


class TaskStatus(Enum):
    """Status of task execution."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    DEVIATED = "deviated"
    BLOCKED = "blocked"
    ERROR = "error"


class ConstraintType(Enum):
    """Types of task constraints."""
    SCOPE_LIMITATION = "scope_limitation"
    BUDGET_LIMIT = "budget_limit"
    TIME_LIMIT = "time_limit"
    RESOURCE_LIMIT = "resource_limit"
    DOMAIN_RESTRICTION = "domain_restriction"
    FUNCTIONALITY_LIMIT = "functionality_limit"
    DATA_ACCESS_LIMIT = "data_access_limit"
    INTERACTION_LIMIT = "interaction_limit"


class DeviationSeverity(Enum):
    """Severity levels for task deviations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NavigationAction(Enum):
    """Actions to take when task deviation is detected."""
    ALLOW = "allow"
    WARN = "warn"
    REDIRECT = "redirect"
    BLOCK = "block"
    ESCALATE = "escalate"
    QUARANTINE = "quarantine"


@dataclass
class TaskConstraint:
    """Represents a constraint on task execution."""
    constraint_id: str
    constraint_type: ConstraintType
    name: str
    description: str
    value: Any
    unit: Optional[str] = None
    strict: bool = True
    warning_threshold: Optional[float] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type.value,
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "unit": self.unit,
            "strict": self.strict,
            "warning_threshold": self.warning_threshold,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class TaskDefinition:
    """Represents a task definition with constraints and boundaries."""
    task_id: str
    task_type: TaskType
    name: str
    description: str
    objectives: List[str]
    constraints: List[TaskConstraint]
    allowed_actions: List[str]
    forbidden_actions: List[str]
    success_criteria: List[str]
    failure_criteria: List[str]
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "name": self.name,
            "description": self.description,
            "objectives": self.objectives,
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "allowed_actions": self.allowed_actions,
            "forbidden_actions": self.forbidden_actions,
            "success_criteria": self.success_criteria,
            "failure_criteria": self.failure_criteria,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class TaskExecution:
    """Represents an active task execution."""
    execution_id: str
    task_definition: TaskDefinition
    agent_id: str
    status: TaskStatus = TaskStatus.ACTIVE
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    progress: float = 0.0
    actions_taken: List[str] = field(default_factory=list)
    deviations: List['TaskDeviation'] = field(default_factory=list)
    context_updates: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "execution_id": self.execution_id,
            "task_definition": self.task_definition.to_dict(),
            "agent_id": self.agent_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "progress": self.progress,
            "actions_taken": self.actions_taken,
            "deviations": [deviation.to_dict() for deviation in self.deviations],
            "context_updates": self.context_updates,
            "metrics": self.metrics
        }


@dataclass
class TaskDeviation:
    """Represents a deviation from the original task."""
    deviation_id: str
    execution_id: str
    deviation_type: str
    description: str
    severity: DeviationSeverity
    detected_action: str
    original_intent: str
    deviation_reason: str
    suggested_correction: str
    action_taken: NavigationAction
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "deviation_id": self.deviation_id,
            "execution_id": self.execution_id,
            "deviation_type": self.deviation_type,
            "description": self.description,
            "severity": self.severity.value,
            "detected_action": self.detected_action,
            "original_intent": self.original_intent,
            "deviation_reason": self.deviation_reason,
            "suggested_correction": self.suggested_correction,
            "action_taken": self.action_taken.value,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes
        }


@dataclass
class NavigationResult:
    """Result of task navigation analysis."""
    allowed: bool
    action: NavigationAction
    reason: str
    suggestions: List[str]
    warnings: List[str]
    deviations: List[TaskDeviation]
    confidence: float
    context_updates: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "allowed": self.allowed,
            "action": self.action.value,
            "reason": self.reason,
            "suggestions": self.suggestions,
            "warnings": self.warnings,
            "deviations": [deviation.to_dict() for deviation in self.deviations],
            "confidence": self.confidence,
            "context_updates": self.context_updates
        }


class TaskNavigator:
    """Task navigation system for enforcing constraints and monitoring agent behavior."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_task_navigation"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Task management
        self.task_definitions: Dict[str, TaskDefinition] = {}
        self.active_executions: Dict[str, TaskExecution] = {}
        self.execution_history: List[TaskExecution] = []
        
        # Deviation tracking
        self.deviations: List[TaskDeviation] = []
        self.navigation_logs: List[NavigationResult] = []
        
        # Statistics and metrics
        self.navigation_stats: Dict[str, int] = {}
        self.deviation_stats: Dict[str, int] = {}
        
        # Initialize default task definitions
        self._initialize_default_tasks()
        
        # Load historical data
        self._load_navigation_data()
        
        logger.info("Task Navigator initialized")
    
    def _initialize_default_tasks(self) -> None:
        """Initialize default task definitions."""
        # Food ordering task
        food_ordering_constraints = [
            TaskConstraint(
                constraint_id="food_ordering_scope",
                constraint_type=ConstraintType.SCOPE_LIMITATION,
                name="Food Ordering Scope",
                description="Limit to food ordering and related activities",
                value=["search_vendors", "browse_menu", "place_order", "track_order", "process_payment", "update_preferences", "get_recommendations"],
                strict=True
            ),
            TaskConstraint(
                constraint_id="food_ordering_budget",
                constraint_type=ConstraintType.BUDGET_LIMIT,
                name="Order Budget Limit",
                description="Maximum order amount",
                value=100.0,
                unit="USD",
                strict=True,
                warning_threshold=80.0
            ),
            TaskConstraint(
                constraint_id="food_ordering_time",
                constraint_type=ConstraintType.TIME_LIMIT,
                name="Order Time Limit",
                description="Maximum time for order completion",
                value=30,
                unit="minutes",
                strict=False,
                warning_threshold=20
            )
        ]
        
        food_ordering_task = TaskDefinition(
            task_id="food_ordering_task",
            task_type=TaskType.FOOD_ORDERING,
            name="Food Ordering Assistant",
            description="Assist users with food ordering, vendor selection, and order management",
            objectives=[
                "Help users find suitable food vendors",
                "Assist with menu selection and customization",
                "Process food orders efficiently",
                "Handle payment and delivery coordination"
            ],
            constraints=food_ordering_constraints,
            allowed_actions=[
                "search_vendors", "browse_menu", "place_order", "track_order",
                "process_payment", "update_preferences", "get_recommendations"
            ],
            forbidden_actions=[
                "access_personal_data", "modify_system_settings", "execute_code",
                "access_external_apis", "modify_user_accounts"
            ],
            success_criteria=[
                "Order placed successfully",
                "User satisfied with selection",
                "Payment processed",
                "Delivery confirmed"
            ],
            failure_criteria=[
                "Order placement failed",
                "Payment processing error",
                "Vendor unavailable",
                "User cancellation"
            ]
        )
        
        # Vendor search task
        vendor_search_constraints = [
            TaskConstraint(
                constraint_id="vendor_search_scope",
                constraint_type=ConstraintType.SCOPE_LIMITATION,
                name="Vendor Search Scope",
                description="Limit to vendor search and selection",
                value=["search_vendors", "filter_vendors", "compare_vendors", "get_vendor_details", "select_vendor"],
                strict=True
            ),
            TaskConstraint(
                constraint_id="vendor_search_results",
                constraint_type=ConstraintType.RESOURCE_LIMIT,
                name="Search Results Limit",
                description="Maximum number of vendors to return",
                value=10,
                strict=True
            )
        ]
        
        vendor_search_task = TaskDefinition(
            task_id="vendor_search_task",
            task_type=TaskType.VENDOR_SEARCH,
            name="Vendor Search Assistant",
            description="Help users find and select food vendors based on preferences",
            objectives=[
                "Search for vendors based on criteria",
                "Compare vendor options",
                "Provide vendor recommendations",
                "Assist with vendor selection"
            ],
            constraints=vendor_search_constraints,
            allowed_actions=[
                "search_vendors", "filter_vendors", "compare_vendors",
                "get_vendor_details", "select_vendor"
            ],
            forbidden_actions=[
                "place_orders", "process_payments", "modify_vendor_data",
                "access_vendor_credentials", "modify_system_settings"
            ],
            success_criteria=[
                "Relevant vendors found",
                "User selects a vendor",
                "Vendor information provided",
                "Search criteria met"
            ],
            failure_criteria=[
                "No vendors found",
                "Search criteria too restrictive",
                "Vendor data unavailable",
                "User abandons search"
            ]
        )
        
        # Preference management task
        preference_constraints = [
            TaskConstraint(
                constraint_id="preference_scope",
                constraint_type=ConstraintType.SCOPE_LIMITATION,
                name="Preference Management Scope",
                description="Limit to user preference management",
                value=["update_preferences", "get_preferences", "analyze_preferences", "recommend_based_on_preferences", "preference_history"],
                strict=True
            ),
            TaskConstraint(
                constraint_id="preference_data_access",
                constraint_type=ConstraintType.DATA_ACCESS_LIMIT,
                name="Preference Data Access",
                description="Limit access to user preference data only",
                value=["dietary_preferences", "cuisine_preferences", "price_preferences"],
                strict=True
            )
        ]
        
        preference_task = TaskDefinition(
            task_id="preference_management_task",
            task_type=TaskType.PREFERENCE_MANAGEMENT,
            name="Preference Management Assistant",
            description="Manage user preferences for food ordering and recommendations",
            objectives=[
                "Update user preferences",
                "Analyze preference patterns",
                "Provide preference-based recommendations",
                "Maintain preference history"
            ],
            constraints=preference_constraints,
            allowed_actions=[
                "update_preferences", "get_preferences", "analyze_preferences",
                "recommend_based_on_preferences", "preference_history"
            ],
            forbidden_actions=[
                "access_payment_info", "modify_orders", "access_personal_data",
                "modify_system_settings", "execute_external_commands"
            ],
            success_criteria=[
                "Preferences updated successfully",
                "Recommendations provided",
                "User satisfied with changes",
                "Preference data maintained"
            ],
            failure_criteria=[
                "Preference update failed",
                "Invalid preference data",
                "User rejects changes",
                "Data corruption"
            ]
        )
        
        self.task_definitions["food_ordering_task"] = food_ordering_task
        self.task_definitions["vendor_search_task"] = vendor_search_task
        self.task_definitions["preference_management_task"] = preference_task
    
    def _load_navigation_data(self) -> None:
        """Load navigation data from storage."""
        # Load execution history
        history_file = self.storage_path / "execution_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for execution_data in data:
                        execution = self._reconstruct_task_execution(execution_data)
                        if execution:
                            self.execution_history.append(execution)
                logger.info(f"Loaded {len(self.execution_history)} task executions")
            except Exception as e:
                logger.error(f"Failed to load execution history: {e}")
        
        # Load deviations
        deviations_file = self.storage_path / "deviations.json"
        if deviations_file.exists():
            try:
                with open(deviations_file, 'r') as f:
                    data = json.load(f)
                    for deviation_data in data:
                        deviation = self._reconstruct_task_deviation(deviation_data)
                        if deviation:
                            self.deviations.append(deviation)
                logger.info(f"Loaded {len(self.deviations)} task deviations")
            except Exception as e:
                logger.error(f"Failed to load deviations: {e}")
    
    def _save_navigation_data(self) -> None:
        """Save navigation data to storage."""
        # Save execution history
        history_file = self.storage_path / "execution_history.json"
        try:
            data = [execution.to_dict() for execution in self.execution_history[-1000:]]  # Keep last 1000
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved execution history")
        except Exception as e:
            logger.error(f"Failed to save execution history: {e}")
        
        # Save deviations
        deviations_file = self.storage_path / "deviations.json"
        try:
            data = [deviation.to_dict() for deviation in self.deviations]
            with open(deviations_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved deviations")
        except Exception as e:
            logger.error(f"Failed to save deviations: {e}")
    
    def _reconstruct_task_execution(self, data: Dict[str, Any]) -> Optional[TaskExecution]:
        """Reconstruct TaskExecution from stored data."""
        try:
            # Reconstruct task definition
            task_def_data = data["task_definition"]
            constraints = []
            for constraint_data in task_def_data.get("constraints", []):
                constraint = TaskConstraint(
                    constraint_id=constraint_data["constraint_id"],
                    constraint_type=ConstraintType(constraint_data["constraint_type"]),
                    name=constraint_data["name"],
                    description=constraint_data["description"],
                    value=constraint_data["value"],
                    unit=constraint_data.get("unit"),
                    strict=constraint_data.get("strict", True),
                    warning_threshold=constraint_data.get("warning_threshold"),
                    enabled=constraint_data.get("enabled", True),
                    created_at=datetime.fromisoformat(constraint_data["created_at"])
                )
                constraints.append(constraint)
            
            task_definition = TaskDefinition(
                task_id=task_def_data["task_id"],
                task_type=TaskType(task_def_data["task_type"]),
                name=task_def_data["name"],
                description=task_def_data["description"],
                objectives=task_def_data["objectives"],
                constraints=constraints,
                allowed_actions=task_def_data["allowed_actions"],
                forbidden_actions=task_def_data["forbidden_actions"],
                success_criteria=task_def_data["success_criteria"],
                failure_criteria=task_def_data["failure_criteria"],
                context=task_def_data.get("context", {}),
                created_at=datetime.fromisoformat(task_def_data["created_at"]),
                updated_at=datetime.fromisoformat(task_def_data["updated_at"])
            )
            
            # Reconstruct deviations
            deviations = []
            for deviation_data in data.get("deviations", []):
                deviation = TaskDeviation(
                    deviation_id=deviation_data["deviation_id"],
                    execution_id=deviation_data["execution_id"],
                    deviation_type=deviation_data["deviation_type"],
                    description=deviation_data["description"],
                    severity=DeviationSeverity(deviation_data["severity"]),
                    detected_action=deviation_data["detected_action"],
                    original_intent=deviation_data["original_intent"],
                    deviation_reason=deviation_data["deviation_reason"],
                    suggested_correction=deviation_data["suggested_correction"],
                    action_taken=NavigationAction(deviation_data["action_taken"]),
                    timestamp=datetime.fromisoformat(deviation_data["timestamp"]),
                    resolved=deviation_data.get("resolved", False),
                    resolution_notes=deviation_data.get("resolution_notes")
                )
                deviations.append(deviation)
            
            execution = TaskExecution(
                execution_id=data["execution_id"],
                task_definition=task_definition,
                agent_id=data["agent_id"],
                status=TaskStatus(data["status"]),
                started_at=datetime.fromisoformat(data["started_at"]),
                last_activity=datetime.fromisoformat(data["last_activity"]),
                progress=data["progress"],
                actions_taken=data["actions_taken"],
                deviations=deviations,
                context_updates=data.get("context_updates", {}),
                metrics=data.get("metrics", {})
            )
            
            return execution
        except Exception as e:
            logger.error(f"Failed to reconstruct task execution: {e}")
            return None
    
    def _reconstruct_task_deviation(self, data: Dict[str, Any]) -> Optional[TaskDeviation]:
        """Reconstruct TaskDeviation from stored data."""
        try:
            deviation = TaskDeviation(
                deviation_id=data["deviation_id"],
                execution_id=data["execution_id"],
                deviation_type=data["deviation_type"],
                description=data["description"],
                severity=DeviationSeverity(data["severity"]),
                detected_action=data["detected_action"],
                original_intent=data["original_intent"],
                deviation_reason=data["deviation_reason"],
                suggested_correction=data["suggested_correction"],
                action_taken=NavigationAction(data["action_taken"]),
                timestamp=datetime.fromisoformat(data["timestamp"]),
                resolved=data.get("resolved", False),
                resolution_notes=data.get("resolution_notes")
            )
            return deviation
        except Exception as e:
            logger.error(f"Failed to reconstruct task deviation: {e}")
            return None
    
    def create_task_execution(self, task_id: str, agent_id: str, 
                            context: Optional[Dict[str, Any]] = None) -> Optional[TaskExecution]:
        """Create a new task execution."""
        if task_id not in self.task_definitions:
            logger.error(f"Task definition {task_id} not found")
            return None
        
        task_definition = self.task_definitions[task_id]
        execution_id = f"exec_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        execution = TaskExecution(
            execution_id=execution_id,
            task_definition=task_definition,
            agent_id=agent_id,
            context_updates=context or {}
        )
        
        self.active_executions[execution_id] = execution
        
        logger.info(f"Created task execution {execution_id} for task {task_id}")
        
        # Record metrics
        record_metric("task_navigator.execution_created", 1, MetricType.COUNTER, {
            "task_type": task_definition.task_type.value,
            "agent_id": agent_id
        })
        
        return execution
    
    def navigate_action(self, execution_id: str, action: str, context: Dict[str, Any]) -> NavigationResult:
        """Navigate and validate an agent action against task constraints."""
        if execution_id not in self.active_executions:
            return NavigationResult(
                allowed=False,
                action=NavigationAction.BLOCK,
                reason="Task execution not found",
                suggestions=["Create a new task execution"],
                warnings=[],
                deviations=[],
                confidence=1.0
            )
        
        execution = self.active_executions[execution_id]
        task_definition = execution.task_definition
        
        # Update last activity
        execution.last_activity = datetime.now()
        
        # Analyze the action
        analysis_result = self._analyze_action(action, context, task_definition, execution)
        
        # Create navigation result
        result = NavigationResult(
            allowed=analysis_result["allowed"],
            action=analysis_result["action"],
            reason=analysis_result["reason"],
            suggestions=analysis_result["suggestions"],
            warnings=analysis_result["warnings"],
            deviations=analysis_result["deviations"],
            confidence=analysis_result["confidence"],
            context_updates=analysis_result["context_updates"]
        )
        
        # Update execution
        execution.actions_taken.append(action)
        execution.deviations.extend(analysis_result["deviations"])
        execution.context_updates.update(analysis_result["context_updates"])
        
        # Log navigation result
        self.navigation_logs.append(result)
        
        # Record metrics
        record_metric("task_navigator.action_navigated", 1, MetricType.COUNTER, {
            "action": action,
            "allowed": result.allowed,
            "task_type": task_definition.task_type.value,
            "agent_id": execution.agent_id
        })
        
        return result
    
    def _analyze_action(self, action: str, context: Dict[str, Any], 
                       task_definition: TaskDefinition, execution: TaskExecution) -> Dict[str, Any]:
        """Analyze an action against task constraints."""
        deviations = []
        warnings = []
        suggestions = []
        context_updates = {}
        
        # Check if action is allowed
        if action in task_definition.forbidden_actions:
            deviation = self._create_deviation(
                execution.execution_id,
                "forbidden_action",
                f"Action '{action}' is forbidden for this task",
                DeviationSeverity.HIGH,
                action,
                "Execute allowed task actions only",
                "Action is explicitly forbidden in task definition",
                f"Use one of the allowed actions: {', '.join(task_definition.allowed_actions)}",
                NavigationAction.BLOCK
            )
            deviations.append(deviation)
            
            return {
                "allowed": False,
                "action": NavigationAction.BLOCK,
                "reason": f"Action '{action}' is forbidden for task '{task_definition.name}'",
                "suggestions": [f"Use allowed actions: {', '.join(task_definition.allowed_actions)}"],
                "warnings": [],
                "deviations": deviations,
                "confidence": 1.0,
                "context_updates": context_updates
            }
        
        # Check if action is in allowed actions
        if action not in task_definition.allowed_actions:
            deviation = self._create_deviation(
                execution.execution_id,
                "unauthorized_action",
                f"Action '{action}' is not authorized for this task",
                DeviationSeverity.MEDIUM,
                action,
                "Execute only authorized task actions",
                "Action is not in the allowed actions list",
                f"Consider using: {', '.join(task_definition.allowed_actions)}",
                NavigationAction.WARN
            )
            deviations.append(deviation)
            warnings.append(f"Action '{action}' is not explicitly allowed")
            suggestions.append(f"Consider using authorized actions: {', '.join(task_definition.allowed_actions)}")
        
        # Check constraints
        constraint_violations = self._check_constraints(action, context, task_definition, execution)
        deviations.extend(constraint_violations["deviations"])
        warnings.extend(constraint_violations["warnings"])
        suggestions.extend(constraint_violations["suggestions"])
        context_updates.update(constraint_violations["context_updates"])
        
        # Determine overall result
        if any(d.severity == DeviationSeverity.CRITICAL for d in deviations):
            action_taken = NavigationAction.BLOCK
            allowed = False
            reason = "Critical constraint violation detected"
        elif any(d.severity == DeviationSeverity.HIGH for d in deviations):
            action_taken = NavigationAction.REDIRECT
            allowed = False
            reason = "High severity deviation detected"
        elif any(d.severity == DeviationSeverity.MEDIUM for d in deviations):
            action_taken = NavigationAction.WARN
            allowed = True
            reason = "Medium severity deviation detected"
        elif warnings:
            action_taken = NavigationAction.WARN
            allowed = True
            reason = "Warning conditions detected"
        else:
            action_taken = NavigationAction.ALLOW
            allowed = True
            reason = "Action is compliant with task constraints"
        
        return {
            "allowed": allowed,
            "action": action_taken,
            "reason": reason,
            "suggestions": suggestions,
            "warnings": warnings,
            "deviations": deviations,
            "confidence": 0.9 if allowed else 0.7,
            "context_updates": context_updates
        }
    
    def _check_constraints(self, action: str, context: Dict[str, Any], 
                          task_definition: TaskDefinition, execution: TaskExecution) -> Dict[str, Any]:
        """Check action against task constraints."""
        deviations = []
        warnings = []
        suggestions = []
        context_updates = {}
        
        for constraint in task_definition.constraints:
            if not constraint.enabled:
                continue
            
            violation_result = self._check_constraint_violation(
                constraint, action, context, execution
            )
            
            if violation_result["violated"]:
                deviation = self._create_deviation(
                    execution.execution_id,
                    "constraint_violation",
                    f"Constraint '{constraint.name}' violated",
                    DeviationSeverity.HIGH if constraint.strict else DeviationSeverity.MEDIUM,
                    action,
                    f"Respect constraint: {constraint.description}",
                    violation_result["reason"],
                    violation_result["suggestion"],
                    NavigationAction.BLOCK if constraint.strict else NavigationAction.WARN
                )
                deviations.append(deviation)
                
                if constraint.strict:
                    warnings.append(f"Strict constraint violated: {constraint.name}")
                else:
                    warnings.append(f"Constraint warning: {constraint.name}")
                
                suggestions.append(violation_result["suggestion"])
            
            elif violation_result["warning"]:
                warnings.append(f"Approaching constraint limit: {constraint.name}")
                suggestions.append(violation_result["suggestion"])
        
        return {
            "deviations": deviations,
            "warnings": warnings,
            "suggestions": suggestions,
            "context_updates": context_updates
        }
    
    def _check_constraint_violation(self, constraint: TaskConstraint, action: str, 
                                   context: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Check if a specific constraint is violated."""
        if constraint.constraint_type == ConstraintType.SCOPE_LIMITATION:
            return self._check_scope_constraint(constraint, action, context)
        elif constraint.constraint_type == ConstraintType.BUDGET_LIMIT:
            return self._check_budget_constraint(constraint, action, context, execution)
        elif constraint.constraint_type == ConstraintType.TIME_LIMIT:
            return self._check_time_constraint(constraint, action, context, execution)
        elif constraint.constraint_type == ConstraintType.RESOURCE_LIMIT:
            return self._check_resource_constraint(constraint, action, context, execution)
        elif constraint.constraint_type == ConstraintType.DOMAIN_RESTRICTION:
            return self._check_domain_constraint(constraint, action, context)
        else:
            return {"violated": False, "warning": False, "reason": "", "suggestion": ""}
    
    def _check_scope_constraint(self, constraint: TaskConstraint, action: str, 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Check scope limitation constraint."""
        allowed_scope = constraint.value
        if isinstance(allowed_scope, list) and action not in allowed_scope:
            return {
                "violated": True,
                "warning": False,
                "reason": f"Action '{action}' is outside allowed scope",
                "suggestion": f"Use actions within scope: {', '.join(allowed_scope)}"
            }
        return {"violated": False, "warning": False, "reason": "", "suggestion": ""}
    
    def _check_budget_constraint(self, constraint: TaskConstraint, action: str, 
                                context: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Check budget limit constraint."""
        if action in ["place_order", "process_payment"]:
            order_amount = context.get("order_amount", 0)
            budget_limit = constraint.value
            
            if order_amount > budget_limit:
                return {
                    "violated": True,
                    "warning": False,
                    "reason": f"Order amount ${order_amount} exceeds budget limit ${budget_limit}",
                    "suggestion": f"Reduce order amount to under ${budget_limit}"
                }
            elif constraint.warning_threshold and order_amount > constraint.warning_threshold:
                return {
                    "violated": False,
                    "warning": True,
                    "reason": f"Order amount ${order_amount} approaching budget limit ${budget_limit}",
                    "suggestion": f"Consider reducing order amount (current: ${order_amount}, limit: ${budget_limit})"
                }
        
        return {"violated": False, "warning": False, "reason": "", "suggestion": ""}
    
    def _check_time_constraint(self, constraint: TaskConstraint, action: str, 
                              context: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Check time limit constraint."""
        time_limit_minutes = constraint.value
        elapsed_minutes = (datetime.now() - execution.started_at).total_seconds() / 60
        
        if elapsed_minutes > time_limit_minutes:
            return {
                "violated": True,
                "warning": False,
                "reason": f"Task execution time {elapsed_minutes:.1f} minutes exceeds limit {time_limit_minutes} minutes",
                "suggestion": "Complete task more quickly or request time extension"
            }
        elif constraint.warning_threshold and elapsed_minutes > constraint.warning_threshold:
            return {
                "violated": False,
                "warning": True,
                "reason": f"Task execution time {elapsed_minutes:.1f} minutes approaching limit {time_limit_minutes} minutes",
                "suggestion": f"Complete task within {time_limit_minutes - elapsed_minutes:.1f} minutes"
            }
        
        return {"violated": False, "warning": False, "reason": "", "suggestion": ""}
    
    def _check_resource_constraint(self, constraint: TaskConstraint, action: str, 
                                  context: Dict[str, Any], execution: TaskExecution) -> Dict[str, Any]:
        """Check resource limit constraint."""
        resource_limit = constraint.value
        current_usage = execution.metrics.get("resource_usage", 0)
        
        if current_usage >= resource_limit:
            return {
                "violated": True,
                "warning": False,
                "reason": f"Resource usage {current_usage} exceeds limit {resource_limit}",
                "suggestion": "Reduce resource usage or request limit increase"
            }
        elif constraint.warning_threshold and current_usage >= constraint.warning_threshold:
            return {
                "violated": False,
                "warning": True,
                "reason": f"Resource usage {current_usage} approaching limit {resource_limit}",
                "suggestion": f"Monitor resource usage (current: {current_usage}, limit: {resource_limit})"
            }
        
        return {"violated": False, "warning": False, "reason": "", "suggestion": ""}
    
    def _check_domain_constraint(self, constraint: TaskConstraint, action: str, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Check domain restriction constraint."""
        allowed_domains = constraint.value
        if isinstance(allowed_domains, list):
            action_domain = self._extract_action_domain(action, context)
            if action_domain and action_domain not in allowed_domains:
                return {
                    "violated": True,
                    "warning": False,
                    "reason": f"Action domain '{action_domain}' is not allowed",
                    "suggestion": f"Use actions within allowed domains: {', '.join(allowed_domains)}"
                }
        return {"violated": False, "warning": False, "reason": "", "suggestion": ""}
    
    def _extract_action_domain(self, action: str, context: Dict[str, Any]) -> Optional[str]:
        """Extract domain from action and context."""
        # Simple domain extraction based on action patterns
        if "vendor" in action.lower():
            return "vendor_management"
        elif "order" in action.lower():
            return "order_management"
        elif "payment" in action.lower():
            return "payment_processing"
        elif "preference" in action.lower():
            return "preference_management"
        else:
            return "general"
    
    def _create_deviation(self, execution_id: str, deviation_type: str, description: str,
                         severity: DeviationSeverity, detected_action: str, original_intent: str,
                         deviation_reason: str, suggested_correction: str, 
                         action_taken: NavigationAction) -> TaskDeviation:
        """Create a task deviation record."""
        deviation_id = f"dev_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        deviation = TaskDeviation(
            deviation_id=deviation_id,
            execution_id=execution_id,
            deviation_type=deviation_type,
            description=description,
            severity=severity,
            detected_action=detected_action,
            original_intent=original_intent,
            deviation_reason=deviation_reason,
            suggested_correction=suggested_correction,
            action_taken=action_taken
        )
        
        self.deviations.append(deviation)
        
        # Update deviation statistics
        severity_key = severity.value
        self.deviation_stats[severity_key] = self.deviation_stats.get(severity_key, 0) + 1
        
        logger.warning(f"Task deviation created: {deviation_id} - {severity.value} severity")
        
        return deviation
    
    def complete_task_execution(self, execution_id: str, success: bool = True, 
                              completion_notes: Optional[str] = None) -> bool:
        """Complete a task execution."""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        execution.status = TaskStatus.COMPLETED if success else TaskStatus.ERROR
        execution.progress = 100.0 if success else execution.progress
        
        # Move to history
        self.execution_history.append(execution)
        del self.active_executions[execution_id]
        
        # Save data
        self._save_navigation_data()
        
        logger.info(f"Completed task execution {execution_id} - Success: {success}")
        
        # Record metrics
        record_metric("task_navigator.execution_completed", 1, MetricType.COUNTER, {
            "success": success,
            "task_type": execution.task_definition.task_type.value,
            "agent_id": execution.agent_id
        })
        
        return True
    
    def get_task_execution(self, execution_id: str) -> Optional[TaskExecution]:
        """Get a task execution by ID."""
        return self.active_executions.get(execution_id)
    
    def get_agent_executions(self, agent_id: str) -> List[TaskExecution]:
        """Get all executions for an agent."""
        agent_executions = []
        
        # Get from active executions
        for execution in self.active_executions.values():
            if execution.agent_id == agent_id:
                agent_executions.append(execution)
        
        # Get from history
        for execution in self.execution_history:
            if execution.agent_id == agent_id:
                agent_executions.append(execution)
        
        return sorted(agent_executions, key=lambda x: x.started_at, reverse=True)
    
    def get_navigation_statistics(self) -> Dict[str, Any]:
        """Get task navigation statistics."""
        total_executions = len(self.active_executions) + len(self.execution_history)
        total_deviations = len(self.deviations)
        
        # Count by task type
        task_type_counts = {}
        for execution in list(self.active_executions.values()) + self.execution_history:
            task_type = execution.task_definition.task_type.value
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        
        # Count by status
        status_counts = {}
        for execution in list(self.active_executions.values()) + self.execution_history:
            status = execution.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count by deviation severity
        deviation_severity_counts = {}
        for deviation in self.deviations:
            severity = deviation.severity.value
            deviation_severity_counts[severity] = deviation_severity_counts.get(severity, 0) + 1
        
        return {
            "total_executions": total_executions,
            "active_executions": len(self.active_executions),
            "completed_executions": len(self.execution_history),
            "total_deviations": total_deviations,
            "task_type_counts": task_type_counts,
            "status_counts": status_counts,
            "deviation_severity_counts": deviation_severity_counts,
            "navigation_logs_count": len(self.navigation_logs),
            "task_definitions_count": len(self.task_definitions)
        }
    
    def get_recent_deviations(self, limit: int = 50) -> List[TaskDeviation]:
        """Get recent task deviations."""
        return sorted(self.deviations, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_navigation_logs(self, limit: int = 100) -> List[NavigationResult]:
        """Get recent navigation logs."""
        return sorted(self.navigation_logs, key=lambda x: x.confidence, reverse=True)[:limit]
    
    def add_task_definition(self, task_definition: TaskDefinition) -> None:
        """Add a new task definition."""
        self.task_definitions[task_definition.task_id] = task_definition
        logger.info(f"Added task definition: {task_definition.name}")
    
    def update_task_definition(self, task_id: str, **updates) -> bool:
        """Update an existing task definition."""
        if task_id not in self.task_definitions:
            return False
        
        task_definition = self.task_definitions[task_id]
        for key, value in updates.items():
            if hasattr(task_definition, key):
                setattr(task_definition, key, value)
        
        task_definition.updated_at = datetime.now()
        logger.info(f"Updated task definition: {task_definition.name}")
        return True
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Clean up old navigation data."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up execution history
        old_executions = [exec for exec in self.execution_history if exec.started_at < cutoff_date]
        self.execution_history = [exec for exec in self.execution_history if exec.started_at >= cutoff_date]
        
        # Clean up deviations
        old_deviations = [dev for dev in self.deviations if dev.timestamp < cutoff_date]
        self.deviations = [dev for dev in self.deviations if dev.timestamp >= cutoff_date]
        
        # Clean up navigation logs
        old_logs = [log for log in self.navigation_logs if hasattr(log, 'timestamp') and log.timestamp < cutoff_date]
        self.navigation_logs = [log for log in self.navigation_logs if not (hasattr(log, 'timestamp') and log.timestamp < cutoff_date)]
        
        total_cleaned = len(old_executions) + len(old_deviations) + len(old_logs)
        
        if total_cleaned > 0:
            self._save_navigation_data()
            logger.info(f"Cleaned up {total_cleaned} old navigation records")
        
        return total_cleaned
