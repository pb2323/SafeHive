"""
Unit tests for Task Navigator - Task Constraint Enforcement and Agent Reasoning.
"""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from safehive.guards.task_navigator import (
    TaskNavigator, TaskType, TaskStatus, ConstraintType, DeviationSeverity,
    NavigationAction, TaskConstraint, TaskDefinition, TaskExecution,
    TaskDeviation, NavigationResult
)


class TestTaskType:
    """Test TaskType enum."""
    
    def test_task_type_values(self):
        """Test TaskType enum values."""
        assert TaskType.FOOD_ORDERING.value == "food_ordering"
        assert TaskType.VENDOR_SEARCH.value == "vendor_search"
        assert TaskType.PREFERENCE_MANAGEMENT.value == "preference_management"
        assert TaskType.ORDER_TRACKING.value == "order_tracking"
        assert TaskType.PAYMENT_PROCESSING.value == "payment_processing"
        assert TaskType.CUSTOMER_SUPPORT.value == "customer_support"
        assert TaskType.DATA_ANALYSIS.value == "data_analysis"
        assert TaskType.SYSTEM_MONITORING.value == "system_monitoring"
        assert TaskType.SECURITY_ANALYSIS.value == "security_analysis"
        assert TaskType.GENERAL_ASSISTANCE.value == "general_assistance"


class TestTaskStatus:
    """Test TaskStatus enum."""
    
    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.ACTIVE.value == "active"
        assert TaskStatus.PAUSED.value == "paused"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.DEVIATED.value == "deviated"
        assert TaskStatus.BLOCKED.value == "blocked"
        assert TaskStatus.ERROR.value == "error"


class TestConstraintType:
    """Test ConstraintType enum."""
    
    def test_constraint_type_values(self):
        """Test ConstraintType enum values."""
        assert ConstraintType.SCOPE_LIMITATION.value == "scope_limitation"
        assert ConstraintType.BUDGET_LIMIT.value == "budget_limit"
        assert ConstraintType.TIME_LIMIT.value == "time_limit"
        assert ConstraintType.RESOURCE_LIMIT.value == "resource_limit"
        assert ConstraintType.DOMAIN_RESTRICTION.value == "domain_restriction"
        assert ConstraintType.FUNCTIONALITY_LIMIT.value == "functionality_limit"
        assert ConstraintType.DATA_ACCESS_LIMIT.value == "data_access_limit"
        assert ConstraintType.INTERACTION_LIMIT.value == "interaction_limit"


class TestDeviationSeverity:
    """Test DeviationSeverity enum."""
    
    def test_deviation_severity_values(self):
        """Test DeviationSeverity enum values."""
        assert DeviationSeverity.LOW.value == "low"
        assert DeviationSeverity.MEDIUM.value == "medium"
        assert DeviationSeverity.HIGH.value == "high"
        assert DeviationSeverity.CRITICAL.value == "critical"


class TestNavigationAction:
    """Test NavigationAction enum."""
    
    def test_navigation_action_values(self):
        """Test NavigationAction enum values."""
        assert NavigationAction.ALLOW.value == "allow"
        assert NavigationAction.WARN.value == "warn"
        assert NavigationAction.REDIRECT.value == "redirect"
        assert NavigationAction.BLOCK.value == "block"
        assert NavigationAction.ESCALATE.value == "escalate"
        assert NavigationAction.QUARANTINE.value == "quarantine"


class TestTaskConstraint:
    """Test TaskConstraint functionality."""
    
    def test_task_constraint_creation(self):
        """Test TaskConstraint creation."""
        constraint = TaskConstraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.BUDGET_LIMIT,
            name="Budget Limit",
            description="Maximum budget for orders",
            value=100.0,
            unit="USD",
            strict=True,
            warning_threshold=80.0,
            enabled=True
        )
        
        assert constraint.constraint_id == "test_constraint"
        assert constraint.constraint_type == ConstraintType.BUDGET_LIMIT
        assert constraint.name == "Budget Limit"
        assert constraint.description == "Maximum budget for orders"
        assert constraint.value == 100.0
        assert constraint.unit == "USD"
        assert constraint.strict is True
        assert constraint.warning_threshold == 80.0
        assert constraint.enabled is True
        assert isinstance(constraint.created_at, datetime)
    
    def test_task_constraint_serialization(self):
        """Test TaskConstraint serialization."""
        constraint = TaskConstraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.BUDGET_LIMIT,
            name="Budget Limit",
            description="Maximum budget for orders",
            value=100.0,
            unit="USD",
            strict=True,
            warning_threshold=80.0
        )
        
        data = constraint.to_dict()
        
        assert data["constraint_id"] == "test_constraint"
        assert data["constraint_type"] == "budget_limit"
        assert data["name"] == "Budget Limit"
        assert data["description"] == "Maximum budget for orders"
        assert data["value"] == 100.0
        assert data["unit"] == "USD"
        assert data["strict"] is True
        assert data["warning_threshold"] == 80.0
        assert data["enabled"] is True
        assert "created_at" in data


class TestTaskDefinition:
    """Test TaskDefinition functionality."""
    
    def test_task_definition_creation(self):
        """Test TaskDefinition creation."""
        constraint = TaskConstraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.BUDGET_LIMIT,
            name="Budget Limit",
            description="Maximum budget for orders",
            value=100.0
        )
        
        task_def = TaskDefinition(
            task_id="test_task",
            task_type=TaskType.FOOD_ORDERING,
            name="Test Task",
            description="A test task definition",
            objectives=["Objective 1", "Objective 2"],
            constraints=[constraint],
            allowed_actions=["action1", "action2"],
            forbidden_actions=["forbidden1", "forbidden2"],
            success_criteria=["Success 1", "Success 2"],
            failure_criteria=["Failure 1", "Failure 2"],
            context={"key": "value"}
        )
        
        assert task_def.task_id == "test_task"
        assert task_def.task_type == TaskType.FOOD_ORDERING
        assert task_def.name == "Test Task"
        assert task_def.description == "A test task definition"
        assert task_def.objectives == ["Objective 1", "Objective 2"]
        assert len(task_def.constraints) == 1
        assert task_def.constraints[0] == constraint
        assert task_def.allowed_actions == ["action1", "action2"]
        assert task_def.forbidden_actions == ["forbidden1", "forbidden2"]
        assert task_def.success_criteria == ["Success 1", "Success 2"]
        assert task_def.failure_criteria == ["Failure 1", "Failure 2"]
        assert task_def.context == {"key": "value"}
        assert isinstance(task_def.created_at, datetime)
        assert isinstance(task_def.updated_at, datetime)
    
    def test_task_definition_serialization(self):
        """Test TaskDefinition serialization."""
        constraint = TaskConstraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.BUDGET_LIMIT,
            name="Budget Limit",
            description="Maximum budget for orders",
            value=100.0
        )
        
        task_def = TaskDefinition(
            task_id="test_task",
            task_type=TaskType.FOOD_ORDERING,
            name="Test Task",
            description="A test task definition",
            objectives=["Objective 1"],
            constraints=[constraint],
            allowed_actions=["action1"],
            forbidden_actions=["forbidden1"],
            success_criteria=["Success 1"],
            failure_criteria=["Failure 1"]
        )
        
        data = task_def.to_dict()
        
        assert data["task_id"] == "test_task"
        assert data["task_type"] == "food_ordering"
        assert data["name"] == "Test Task"
        assert data["description"] == "A test task definition"
        assert data["objectives"] == ["Objective 1"]
        assert len(data["constraints"]) == 1
        assert data["allowed_actions"] == ["action1"]
        assert data["forbidden_actions"] == ["forbidden1"]
        assert data["success_criteria"] == ["Success 1"]
        assert data["failure_criteria"] == ["Failure 1"]
        assert data["context"] == {}
        assert "created_at" in data
        assert "updated_at" in data


class TestTaskExecution:
    """Test TaskExecution functionality."""
    
    def test_task_execution_creation(self):
        """Test TaskExecution creation."""
        constraint = TaskConstraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.BUDGET_LIMIT,
            name="Budget Limit",
            description="Maximum budget for orders",
            value=100.0
        )
        
        task_def = TaskDefinition(
            task_id="test_task",
            task_type=TaskType.FOOD_ORDERING,
            name="Test Task",
            description="A test task definition",
            objectives=["Objective 1"],
            constraints=[constraint],
            allowed_actions=["action1"],
            forbidden_actions=["forbidden1"],
            success_criteria=["Success 1"],
            failure_criteria=["Failure 1"]
        )
        
        execution = TaskExecution(
            execution_id="exec_001",
            task_definition=task_def,
            agent_id="agent_001",
            status=TaskStatus.ACTIVE,
            progress=25.0,
            actions_taken=["action1"],
            context_updates={"key": "value"},
            metrics={"metric1": 10}
        )
        
        assert execution.execution_id == "exec_001"
        assert execution.task_definition == task_def
        assert execution.agent_id == "agent_001"
        assert execution.status == TaskStatus.ACTIVE
        assert execution.progress == 25.0
        assert execution.actions_taken == ["action1"]
        assert execution.context_updates == {"key": "value"}
        assert execution.metrics == {"metric1": 10}
        assert len(execution.deviations) == 0
        assert isinstance(execution.started_at, datetime)
        assert isinstance(execution.last_activity, datetime)
    
    def test_task_execution_serialization(self):
        """Test TaskExecution serialization."""
        constraint = TaskConstraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.BUDGET_LIMIT,
            name="Budget Limit",
            description="Maximum budget for orders",
            value=100.0
        )
        
        task_def = TaskDefinition(
            task_id="test_task",
            task_type=TaskType.FOOD_ORDERING,
            name="Test Task",
            description="A test task definition",
            objectives=["Objective 1"],
            constraints=[constraint],
            allowed_actions=["action1"],
            forbidden_actions=["forbidden1"],
            success_criteria=["Success 1"],
            failure_criteria=["Failure 1"]
        )
        
        execution = TaskExecution(
            execution_id="exec_001",
            task_definition=task_def,
            agent_id="agent_001"
        )
        
        data = execution.to_dict()
        
        assert data["execution_id"] == "exec_001"
        assert data["agent_id"] == "agent_001"
        assert data["status"] == "active"
        assert data["progress"] == 0.0
        assert data["actions_taken"] == []
        assert data["deviations"] == []
        assert data["context_updates"] == {}
        assert data["metrics"] == {}
        assert "started_at" in data
        assert "last_activity" in data


class TestTaskDeviation:
    """Test TaskDeviation functionality."""
    
    def test_task_deviation_creation(self):
        """Test TaskDeviation creation."""
        deviation = TaskDeviation(
            deviation_id="dev_001",
            execution_id="exec_001",
            deviation_type="constraint_violation",
            description="Budget limit exceeded",
            severity=DeviationSeverity.HIGH,
            detected_action="place_order",
            original_intent="Place order within budget",
            deviation_reason="Order amount exceeds budget limit",
            suggested_correction="Reduce order amount",
            action_taken=NavigationAction.BLOCK,
            resolved=False,
            resolution_notes="Under review"
        )
        
        assert deviation.deviation_id == "dev_001"
        assert deviation.execution_id == "exec_001"
        assert deviation.deviation_type == "constraint_violation"
        assert deviation.description == "Budget limit exceeded"
        assert deviation.severity == DeviationSeverity.HIGH
        assert deviation.detected_action == "place_order"
        assert deviation.original_intent == "Place order within budget"
        assert deviation.deviation_reason == "Order amount exceeds budget limit"
        assert deviation.suggested_correction == "Reduce order amount"
        assert deviation.action_taken == NavigationAction.BLOCK
        assert deviation.resolved is False
        assert deviation.resolution_notes == "Under review"
        assert isinstance(deviation.timestamp, datetime)
    
    def test_task_deviation_serialization(self):
        """Test TaskDeviation serialization."""
        deviation = TaskDeviation(
            deviation_id="dev_001",
            execution_id="exec_001",
            deviation_type="constraint_violation",
            description="Budget limit exceeded",
            severity=DeviationSeverity.HIGH,
            detected_action="place_order",
            original_intent="Place order within budget",
            deviation_reason="Order amount exceeds budget limit",
            suggested_correction="Reduce order amount",
            action_taken=NavigationAction.BLOCK
        )
        
        data = deviation.to_dict()
        
        assert data["deviation_id"] == "dev_001"
        assert data["execution_id"] == "exec_001"
        assert data["deviation_type"] == "constraint_violation"
        assert data["description"] == "Budget limit exceeded"
        assert data["severity"] == "high"
        assert data["detected_action"] == "place_order"
        assert data["original_intent"] == "Place order within budget"
        assert data["deviation_reason"] == "Order amount exceeds budget limit"
        assert data["suggested_correction"] == "Reduce order amount"
        assert data["action_taken"] == "block"
        assert data["resolved"] is False
        assert data["resolution_notes"] is None
        assert "timestamp" in data


class TestNavigationResult:
    """Test NavigationResult functionality."""
    
    def test_navigation_result_creation(self):
        """Test NavigationResult creation."""
        deviation = TaskDeviation(
            deviation_id="dev_001",
            execution_id="exec_001",
            deviation_type="constraint_violation",
            description="Budget limit exceeded",
            severity=DeviationSeverity.HIGH,
            detected_action="place_order",
            original_intent="Place order within budget",
            deviation_reason="Order amount exceeds budget limit",
            suggested_correction="Reduce order amount",
            action_taken=NavigationAction.BLOCK
        )
        
        result = NavigationResult(
            allowed=False,
            action=NavigationAction.BLOCK,
            reason="Budget limit exceeded",
            suggestions=["Reduce order amount"],
            warnings=["Approaching budget limit"],
            deviations=[deviation],
            confidence=0.9,
            context_updates={"budget_used": 120.0}
        )
        
        assert result.allowed is False
        assert result.action == NavigationAction.BLOCK
        assert result.reason == "Budget limit exceeded"
        assert result.suggestions == ["Reduce order amount"]
        assert result.warnings == ["Approaching budget limit"]
        assert len(result.deviations) == 1
        assert result.deviations[0] == deviation
        assert result.confidence == 0.9
        assert result.context_updates == {"budget_used": 120.0}
    
    def test_navigation_result_serialization(self):
        """Test NavigationResult serialization."""
        deviation = TaskDeviation(
            deviation_id="dev_001",
            execution_id="exec_001",
            deviation_type="constraint_violation",
            description="Budget limit exceeded",
            severity=DeviationSeverity.HIGH,
            detected_action="place_order",
            original_intent="Place order within budget",
            deviation_reason="Order amount exceeds budget limit",
            suggested_correction="Reduce order amount",
            action_taken=NavigationAction.BLOCK
        )
        
        result = NavigationResult(
            allowed=False,
            action=NavigationAction.BLOCK,
            reason="Budget limit exceeded",
            suggestions=["Reduce order amount"],
            warnings=["Approaching budget limit"],
            deviations=[deviation],
            confidence=0.9,
            context_updates={"budget_used": 120.0}
        )
        
        data = result.to_dict()
        
        assert data["allowed"] is False
        assert data["action"] == "block"
        assert data["reason"] == "Budget limit exceeded"
        assert data["suggestions"] == ["Reduce order amount"]
        assert data["warnings"] == ["Approaching budget limit"]
        assert len(data["deviations"]) == 1
        assert data["confidence"] == 0.9
        assert data["context_updates"] == {"budget_used": 120.0}


class TestTaskNavigator:
    """Test TaskNavigator functionality."""
    
    def test_task_navigator_creation(self):
        """Test TaskNavigator creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            assert navigator.storage_path == Path(temp_dir)
            assert len(navigator.task_definitions) > 0  # Should have default tasks
            assert len(navigator.active_executions) == 0
            assert len(navigator.execution_history) == 0
            assert len(navigator.deviations) == 0
            assert len(navigator.navigation_logs) == 0
    
    def test_create_task_execution(self):
        """Test creating task executions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            
            assert execution is not None
            assert execution.task_definition.task_id == "food_ordering_task"
            assert execution.agent_id == "agent_001"
            assert execution.status == TaskStatus.ACTIVE
            assert execution.execution_id in navigator.active_executions
    
    def test_create_task_execution_invalid_task(self):
        """Test creating task execution with invalid task ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            execution = navigator.create_task_execution("invalid_task", "agent_001")
            
            assert execution is None
    
    def test_navigate_action_allowed(self):
        """Test navigating an allowed action."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            result = navigator.navigate_action(
                execution.execution_id,
                "search_vendors",
                {"query": "pizza"}
            )
            
            assert result.allowed is True
            assert result.action == NavigationAction.ALLOW
            assert len(result.deviations) == 0
            assert len(result.warnings) == 0
    
    def test_navigate_action_forbidden(self):
        """Test navigating a forbidden action."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            result = navigator.navigate_action(
                execution.execution_id,
                "access_personal_data",
                {}
            )
            
            assert result.allowed is False
            assert result.action == NavigationAction.BLOCK
            assert len(result.deviations) >= 1
            assert result.deviations[0].severity == DeviationSeverity.HIGH
    
    def test_navigate_action_unauthorized(self):
        """Test navigating an unauthorized action."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            result = navigator.navigate_action(
                execution.execution_id,
                "unauthorized_action",
                {}
            )
            
            # Unauthorized action should be blocked due to scope constraint
            assert result.allowed is False
            assert result.action == NavigationAction.REDIRECT
            assert len(result.deviations) >= 1
            assert any(dev.severity == DeviationSeverity.HIGH for dev in result.deviations)
            assert len(result.warnings) >= 1
    
    def test_navigate_action_budget_constraint_violation(self):
        """Test navigating action with budget constraint violation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            result = navigator.navigate_action(
                execution.execution_id,
                "place_order",
                {"order_amount": 150.0}  # Exceeds budget limit of 100
            )
            
            assert result.allowed is False
            assert result.action == NavigationAction.REDIRECT  # High severity deviation
            assert len(result.deviations) >= 1
            assert any("budget" in dev.description.lower() for dev in result.deviations)
    
    def test_navigate_action_budget_constraint_warning(self):
        """Test navigating action with budget constraint warning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            result = navigator.navigate_action(
                execution.execution_id,
                "place_order",
                {"order_amount": 85.0}  # Above warning threshold of 80
            )
            
            assert result.allowed is True
            assert result.action == NavigationAction.WARN
            assert len(result.warnings) >= 1
            assert "budget" in result.warnings[0].lower()
    
    def test_navigate_action_time_constraint_violation(self):
        """Test navigating action with time constraint violation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            # Manually set start time to be old
            execution.started_at = datetime.now() - timedelta(minutes=35)  # Exceeds 30 min limit
            
            result = navigator.navigate_action(
                execution.execution_id,
                "place_order",
                {}
            )
            
            assert result.allowed is True  # Time constraint is not strict
            assert result.action == NavigationAction.WARN  # Medium severity deviation
            assert len(result.deviations) >= 1
            assert any("time" in dev.description.lower() for dev in result.deviations)
    
    def test_navigate_action_invalid_execution(self):
        """Test navigating action with invalid execution ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            result = navigator.navigate_action(
                "invalid_execution_id",
                "search_vendors",
                {}
            )
            
            assert result.allowed is False
            assert result.action == NavigationAction.BLOCK
            assert "not found" in result.reason.lower()
    
    def test_complete_task_execution(self):
        """Test completing a task execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            success = navigator.complete_task_execution(execution.execution_id, True)
            
            assert success is True
            assert execution.execution_id not in navigator.active_executions
            assert len(navigator.execution_history) == 1
            assert navigator.execution_history[0].status == TaskStatus.COMPLETED
    
    def test_complete_task_execution_invalid(self):
        """Test completing an invalid task execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            success = navigator.complete_task_execution("invalid_execution_id", True)
            
            assert success is False
    
    def test_get_task_execution(self):
        """Test getting a task execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            retrieved = navigator.get_task_execution(execution.execution_id)
            
            assert retrieved is not None
            assert retrieved.execution_id == execution.execution_id
            
            # Test invalid execution ID
            invalid = navigator.get_task_execution("invalid_id")
            assert invalid is None
    
    def test_get_agent_executions(self):
        """Test getting executions for an agent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            # Create executions for different agents
            execution1 = navigator.create_task_execution("food_ordering_task", "agent_001")
            execution2 = navigator.create_task_execution("vendor_search_task", "agent_001")
            execution3 = navigator.create_task_execution("food_ordering_task", "agent_002")
            
            assert execution1 is not None
            assert execution2 is not None
            assert execution3 is not None
            
            # Get executions for agent_001
            agent_executions = navigator.get_agent_executions("agent_001")
            
            assert len(agent_executions) == 2
            assert all(exec.agent_id == "agent_001" for exec in agent_executions)
            
            # Get executions for agent_002
            agent2_executions = navigator.get_agent_executions("agent_002")
            
            assert len(agent2_executions) == 1
            assert agent2_executions[0].agent_id == "agent_002"
    
    def test_get_navigation_statistics(self):
        """Test getting navigation statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            # Create some executions and navigate actions
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            navigator.navigate_action(execution.execution_id, "search_vendors", {})
            navigator.navigate_action(execution.execution_id, "forbidden_action", {})
            
            stats = navigator.get_navigation_statistics()
            
            assert "total_executions" in stats
            assert "active_executions" in stats
            assert "total_deviations" in stats
            assert "task_type_counts" in stats
            assert "status_counts" in stats
            assert "deviation_severity_counts" in stats
            assert stats["active_executions"] == 1
            assert stats["total_deviations"] >= 1
    
    def test_get_recent_deviations(self):
        """Test getting recent deviations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            # Create some deviations
            navigator.navigate_action(execution.execution_id, "forbidden_action", {})
            navigator.navigate_action(execution.execution_id, "place_order", {"order_amount": 150.0})
            
            deviations = navigator.get_recent_deviations(10)
            
            assert len(deviations) >= 2
            assert all(isinstance(dev, TaskDeviation) for dev in deviations)
    
    def test_get_navigation_logs(self):
        """Test getting navigation logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            # Create some navigation results
            navigator.navigate_action(execution.execution_id, "search_vendors", {})
            navigator.navigate_action(execution.execution_id, "place_order", {})
            
            logs = navigator.get_navigation_logs(10)
            
            assert len(logs) >= 2
            assert all(isinstance(log, NavigationResult) for log in logs)
    
    def test_add_task_definition(self):
        """Test adding a task definition."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            constraint = TaskConstraint(
                constraint_id="custom_constraint",
                constraint_type=ConstraintType.RESOURCE_LIMIT,
                name="Custom Resource Limit",
                description="Custom resource constraint",
                value=50
            )
            
            task_def = TaskDefinition(
                task_id="custom_task",
                task_type=TaskType.GENERAL_ASSISTANCE,
                name="Custom Task",
                description="A custom task definition",
                objectives=["Custom objective"],
                constraints=[constraint],
                allowed_actions=["custom_action"],
                forbidden_actions=["forbidden_custom"],
                success_criteria=["Custom success"],
                failure_criteria=["Custom failure"]
            )
            
            navigator.add_task_definition(task_def)
            
            assert "custom_task" in navigator.task_definitions
            assert navigator.task_definitions["custom_task"] == task_def
    
    def test_update_task_definition(self):
        """Test updating a task definition."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            # Update existing task definition
            success = navigator.update_task_definition(
                "food_ordering_task",
                name="Updated Food Ordering Task"
            )
            
            assert success is True
            assert navigator.task_definitions["food_ordering_task"].name == "Updated Food Ordering Task"
            
            # Try to update non-existent task
            success = navigator.update_task_definition(
                "non_existent_task",
                name="Updated"
            )
            
            assert success is False
    
    def test_cleanup_old_data(self):
        """Test cleaning up old data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            # Create old execution
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            # Manually set old start time
            execution.started_at = datetime.now() - timedelta(days=35)
            
            # Complete execution to move to history
            navigator.complete_task_execution(execution.execution_id, True)
            
            # Create old deviation
            old_deviation = TaskDeviation(
                deviation_id="old_dev",
                execution_id="old_exec",
                deviation_type="old_violation",
                description="Old deviation",
                severity=DeviationSeverity.LOW,
                detected_action="old_action",
                original_intent="Old intent",
                deviation_reason="Old reason",
                suggested_correction="Old correction",
                action_taken=NavigationAction.WARN
            )
            old_deviation.timestamp = datetime.now() - timedelta(days=35)
            navigator.deviations.append(old_deviation)
            
            # Clean up data older than 30 days
            cleaned_count = navigator.cleanup_old_data(30)
            
            assert cleaned_count >= 1
            assert len(navigator.execution_history) == 0
            assert len(navigator.deviations) == 0


class TestTaskNavigatorIntegration:
    """Integration tests for TaskNavigator."""
    
    def test_complete_task_workflow(self):
        """Test complete task workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            # Create task execution
            execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            # Navigate through allowed actions
            result1 = navigator.navigate_action(
                execution.execution_id,
                "search_vendors",
                {"query": "pizza"}
            )
            assert result1.allowed is True
            
            result2 = navigator.navigate_action(
                execution.execution_id,
                "browse_menu",
                {"vendor_id": "pizza_place"}
            )
            assert result2.allowed is True
            
            # Try forbidden action
            result3 = navigator.navigate_action(
                execution.execution_id,
                "access_personal_data",
                {}
            )
            assert result3.allowed is False
            assert result3.action == NavigationAction.BLOCK
            
            # Complete task
            success = navigator.complete_task_execution(execution.execution_id, True)
            assert success is True
            
            # Check statistics
            stats = navigator.get_navigation_statistics()
            assert stats["total_executions"] == 1
            assert stats["completed_executions"] == 1
            assert stats["total_deviations"] >= 1
    
    def test_persistence_and_recovery(self):
        """Test persistence and recovery of navigation data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first navigator instance
            navigator1 = TaskNavigator(temp_dir)
            
            # Create execution and navigate actions
            execution = navigator1.create_task_execution("food_ordering_task", "agent_001")
            assert execution is not None
            
            navigator1.navigate_action(execution.execution_id, "search_vendors", {})
            navigator1.navigate_action(execution.execution_id, "forbidden_action", {})
            
            # Complete execution
            navigator1.complete_task_execution(execution.execution_id, True)
            
            # Create second navigator instance (should load data)
            navigator2 = TaskNavigator(temp_dir)
            
            # Verify data was loaded
            assert len(navigator2.execution_history) == 1
            assert len(navigator2.deviations) >= 1
            assert navigator2.execution_history[0].agent_id == "agent_001"
    
    def test_multiple_task_types(self):
        """Test managing multiple task types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            navigator = TaskNavigator(temp_dir)
            
            # Create executions for different task types
            food_execution = navigator.create_task_execution("food_ordering_task", "agent_001")
            vendor_execution = navigator.create_task_execution("vendor_search_task", "agent_001")
            preference_execution = navigator.create_task_execution("preference_management_task", "agent_002")
            
            assert food_execution is not None
            assert vendor_execution is not None
            assert preference_execution is not None
            
            # Navigate actions for each task type
            navigator.navigate_action(food_execution.execution_id, "search_vendors", {})
            navigator.navigate_action(vendor_execution.execution_id, "search_vendors", {})
            navigator.navigate_action(preference_execution.execution_id, "update_preferences", {})
            
            # Check statistics
            stats = navigator.get_navigation_statistics()
            assert stats["active_executions"] == 3
            assert "food_ordering" in stats["task_type_counts"]
            assert "vendor_search" in stats["task_type_counts"]
            assert "preference_management" in stats["task_type_counts"]
