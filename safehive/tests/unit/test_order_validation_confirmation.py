"""
Unit tests for Order Validation and Confirmation Systems.
"""

import asyncio
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from safehive.agents.order_validation import (
    ValidationStatus, ValidationSeverity, ValidationType, ValidationRule,
    ValidationResult, ValidationReport, OrderValidationEngine
)
from safehive.agents.order_confirmation import (
    ConfirmationStatus, ApprovalType, ApprovalResult, ConfirmationStep,
    ApprovalRequirement, ConfirmationWorkflow, ConfirmationSession, OrderConfirmationManager
)
from safehive.agents.order_models import Order, OrderItem, Vendor, OrderStatus, OrderType, PaymentStatus
from safehive.agents.user_twin import UserTwinAgent, PreferenceCategory


class TestValidationRule:
    """Test ValidationRule functionality."""
    
    def test_validation_rule_creation(self):
        """Test ValidationRule creation."""
        rule = ValidationRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test validation rule",
            validation_type=ValidationType.BUSINESS_LOGIC,
            severity=ValidationSeverity.HIGH,
            is_required=True,
            parameters={"test_param": "test_value"}
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.description == "A test validation rule"
        assert rule.validation_type == ValidationType.BUSINESS_LOGIC
        assert rule.severity == ValidationSeverity.HIGH
        assert rule.is_required is True
        assert rule.parameters["test_param"] == "test_value"
    
    def test_validation_rule_serialization(self):
        """Test ValidationRule serialization."""
        rule = ValidationRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test validation rule",
            validation_type=ValidationType.BUSINESS_LOGIC,
            severity=ValidationSeverity.HIGH
        )
        
        data = rule.to_dict()
        
        assert data["rule_id"] == "test_rule"
        assert data["name"] == "Test Rule"
        assert data["validation_type"] == "business_logic"
        assert data["severity"] == "high"
        assert data["is_required"] is True


class TestValidationResult:
    """Test ValidationResult functionality."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            rule_id="test_rule",
            validation_type=ValidationType.BUSINESS_LOGIC,
            status=ValidationStatus.PASSED,
            severity=ValidationSeverity.HIGH,
            message="Validation passed",
            agent_reasoning=["Reason 1", "Reason 2"],
            suggestions=["Suggestion 1"]
        )
        
        assert result.rule_id == "test_rule"
        assert result.validation_type == ValidationType.BUSINESS_LOGIC
        assert result.status == ValidationStatus.PASSED
        assert result.severity == ValidationSeverity.HIGH
        assert result.message == "Validation passed"
        assert len(result.agent_reasoning) == 2
        assert len(result.suggestions) == 1
    
    def test_validation_result_serialization(self):
        """Test ValidationResult serialization."""
        result = ValidationResult(
            rule_id="test_rule",
            validation_type=ValidationType.BUSINESS_LOGIC,
            status=ValidationStatus.PASSED,
            severity=ValidationSeverity.HIGH,
            message="Validation passed"
        )
        
        data = result.to_dict()
        
        assert data["rule_id"] == "test_rule"
        assert data["validation_type"] == "business_logic"
        assert data["status"] == "passed"
        assert data["severity"] == "high"
        assert data["message"] == "Validation passed"
        assert "timestamp" in data


class TestValidationReport:
    """Test ValidationReport functionality."""
    
    def test_validation_report_creation(self):
        """Test ValidationReport creation."""
        report = ValidationReport(
            order_id="order_001",
            validation_session_id="validation_001",
            overall_status=ValidationStatus.PENDING
        )
        
        assert report.order_id == "order_001"
        assert report.validation_session_id == "validation_001"
        assert report.overall_status == ValidationStatus.PENDING
        assert len(report.validation_results) == 0
    
    def test_add_validation_result(self):
        """Test adding validation results to report."""
        report = ValidationReport(
            order_id="order_001",
            validation_session_id="validation_001",
            overall_status=ValidationStatus.PENDING
        )
        
        result = ValidationResult(
            rule_id="test_rule",
            validation_type=ValidationType.BUSINESS_LOGIC,
            status=ValidationStatus.PASSED,
            severity=ValidationSeverity.HIGH,
            message="Test passed"
        )
        
        report.add_result(result)
        
        assert len(report.validation_results) == 1
        assert report.overall_status == ValidationStatus.PASSED
    
    def test_add_failed_validation_result(self):
        """Test adding failed validation result."""
        report = ValidationReport(
            order_id="order_001",
            validation_session_id="validation_001",
            overall_status=ValidationStatus.PENDING
        )
        
        result = ValidationResult(
            rule_id="test_rule",
            validation_type=ValidationType.BUSINESS_LOGIC,
            status=ValidationStatus.FAILED,
            severity=ValidationSeverity.CRITICAL,
            message="Test failed"
        )
        
        report.add_result(result)
        
        assert len(report.validation_results) == 1
        assert report.overall_status == ValidationStatus.FAILED
    
    def test_get_results_by_type(self):
        """Test getting results by validation type."""
        report = ValidationReport(
            order_id="order_001",
            validation_session_id="validation_001",
            overall_status=ValidationStatus.PENDING
        )
        
        # Add results of different types
        result1 = ValidationResult(
            rule_id="rule1",
            validation_type=ValidationType.BUSINESS_LOGIC,
            status=ValidationStatus.PASSED,
            severity=ValidationSeverity.HIGH,
            message="Business logic passed"
        )
        
        result2 = ValidationResult(
            rule_id="rule2",
            validation_type=ValidationType.USER_PREFERENCE,
            status=ValidationStatus.WARNING,
            severity=ValidationSeverity.MEDIUM,
            message="User preference warning"
        )
        
        report.add_result(result1)
        report.add_result(result2)
        
        business_results = report.get_results_by_type(ValidationType.BUSINESS_LOGIC)
        assert len(business_results) == 1
        assert business_results[0].rule_id == "rule1"
        
        user_results = report.get_results_by_type(ValidationType.USER_PREFERENCE)
        assert len(user_results) == 1
        assert user_results[0].rule_id == "rule2"
    
    def test_get_failed_results(self):
        """Test getting failed validation results."""
        report = ValidationReport(
            order_id="order_001",
            validation_session_id="validation_001",
            overall_status=ValidationStatus.PENDING
        )
        
        # Add mixed results
        result1 = ValidationResult(
            rule_id="rule1",
            validation_type=ValidationType.BUSINESS_LOGIC,
            status=ValidationStatus.PASSED,
            severity=ValidationSeverity.HIGH,
            message="Passed"
        )
        
        result2 = ValidationResult(
            rule_id="rule2",
            validation_type=ValidationType.USER_PREFERENCE,
            status=ValidationStatus.FAILED,
            severity=ValidationSeverity.HIGH,
            message="Failed"
        )
        
        report.add_result(result1)
        report.add_result(result2)
        
        failed_results = report.get_failed_results()
        assert len(failed_results) == 1
        assert failed_results[0].rule_id == "rule2"


class TestOrderValidationEngine:
    """Test OrderValidationEngine functionality."""
    
    def test_order_validation_engine_creation(self):
        """Test OrderValidationEngine creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = OrderValidationEngine(temp_dir)
            
            assert engine.storage_path == Path(temp_dir)
            assert len(engine.validation_rules) > 0
            assert len(engine.validation_history) == 0
    
    def test_add_validation_rule(self):
        """Test adding validation rules."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = OrderValidationEngine(temp_dir)
            
            rule = ValidationRule(
                rule_id="custom_rule",
                name="Custom Rule",
                description="A custom validation rule",
                validation_type=ValidationType.CUSTOM,
                severity=ValidationSeverity.MEDIUM
            )
            
            engine.add_validation_rule(rule)
            assert "custom_rule" in engine.validation_rules
    
    def test_update_validation_rule(self):
        """Test updating validation rules."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = OrderValidationEngine(temp_dir)
            
            # Update existing rule
            success = engine.update_validation_rule("vendor_availability", is_enabled=False)
            assert success is True
            assert engine.validation_rules["vendor_availability"].is_enabled is False
            
            # Try to update non-existent rule
            success = engine.update_validation_rule("non_existent", is_enabled=False)
            assert success is False
    
    @pytest.mark.asyncio
    async def test_validate_order_basic(self):
        """Test basic order validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = OrderValidationEngine(temp_dir)
            
            # Create test order
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            items = [
                OrderItem(
                    item_id="item_001",
                    name="Test Item",
                    quantity=1,
                    unit_price=20.0,
                    total_price=20.0
                )
            ]
            
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=items,
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=23.0,
                delivery_address="123 Test St"
            )
            
            # Validate order
            report = await engine.validate_order(order)
            
            assert report is not None
            assert report.order_id == "order_001"
            assert len(report.validation_results) > 0
            assert report.overall_status in [ValidationStatus.PASSED, ValidationStatus.WARNING, ValidationStatus.FAILED]
    
    @pytest.mark.asyncio
    async def test_validate_order_with_user_preferences(self):
        """Test order validation with user preferences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = OrderValidationEngine(temp_dir)
            
            # Create user twin agent
            user_twin = UserTwinAgent("test_user")
            user_twin.add_preference(PreferenceCategory.FOOD, "dietary", "vegetarian", 0.9)
            user_twin.add_preference(PreferenceCategory.COST, "budget", "cheap", 0.8)
            
            # Create test order
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            items = [
                OrderItem(
                    item_id="item_001",
                    name="Vegetarian Pizza",
                    quantity=1,
                    unit_price=18.0,
                    total_price=18.0,
                    dietary_requirements=["vegetarian"]
                )
            ]
            
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=items,
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=21.0,
                delivery_address="123 Test St"
            )
            
            # Validate order with user preferences
            report = await engine.validate_order(order, user_twin)
            
            assert report is not None
            assert len(report.validation_results) > 0
            
            # Check for user preference validation
            user_pref_results = report.get_results_by_type(ValidationType.USER_PREFERENCE)
            assert len(user_pref_results) > 0
    
    def test_get_validation_statistics(self):
        """Test getting validation statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = OrderValidationEngine(temp_dir)
            
            # Test with no history
            stats = engine.get_validation_statistics()
            assert stats["total_validations"] == 0
            
            # Add some mock validation reports
            report1 = ValidationReport(
                order_id="order_001",
                validation_session_id="validation_001",
                overall_status=ValidationStatus.PASSED
            )
            
            report2 = ValidationReport(
                order_id="order_002",
                validation_session_id="validation_002",
                overall_status=ValidationStatus.FAILED
            )
            
            engine.validation_history.extend([report1, report2])
            
            # Test statistics with history
            stats = engine.get_validation_statistics()
            assert stats["total_validations"] == 2
            assert stats["passed_validations"] == 1
            assert stats["failed_validations"] == 1
            assert "rule_performance" in stats


class TestApprovalRequirement:
    """Test ApprovalRequirement functionality."""
    
    def test_approval_requirement_creation(self):
        """Test ApprovalRequirement creation."""
        requirement = ApprovalRequirement(
            approval_id="test_approval",
            approval_type=ApprovalType.USER_CONFIRMATION,
            description="Test approval requirement",
            required_by="user",
            timeout_minutes=15,
            is_required=True,
            conditions={"test_condition": "test_value"}
        )
        
        assert requirement.approval_id == "test_approval"
        assert requirement.approval_type == ApprovalType.USER_CONFIRMATION
        assert requirement.description == "Test approval requirement"
        assert requirement.required_by == "user"
        assert requirement.timeout_minutes == 15
        assert requirement.is_required is True
        assert requirement.conditions["test_condition"] == "test_value"
    
    def test_approval_requirement_serialization(self):
        """Test ApprovalRequirement serialization."""
        requirement = ApprovalRequirement(
            approval_id="test_approval",
            approval_type=ApprovalType.USER_CONFIRMATION,
            description="Test approval requirement",
            required_by="user"
        )
        
        data = requirement.to_dict()
        
        assert data["approval_id"] == "test_approval"
        assert data["approval_type"] == "user_confirmation"
        assert data["description"] == "Test approval requirement"
        assert data["required_by"] == "user"
        assert data["timeout_minutes"] == 30  # default value


class TestApprovalResult:
    """Test ApprovalResult functionality."""
    
    def test_approval_result_creation(self):
        """Test ApprovalResult creation."""
        result = ApprovalResult(
            approval_id="test_approval",
            approval_type=ApprovalType.USER_CONFIRMATION,
            status=ConfirmationStatus.APPROVED,
            approver="user",
            message="Approved by user",
            reasoning=["User reviewed order", "User confirmed details"],
            metadata={"confidence": 0.9}
        )
        
        assert result.approval_id == "test_approval"
        assert result.approval_type == ApprovalType.USER_CONFIRMATION
        assert result.status == ConfirmationStatus.APPROVED
        assert result.approver == "user"
        assert result.message == "Approved by user"
        assert len(result.reasoning) == 2
        assert result.metadata["confidence"] == 0.9
    
    def test_approval_result_serialization(self):
        """Test ApprovalResult serialization."""
        result = ApprovalResult(
            approval_id="test_approval",
            approval_type=ApprovalType.USER_CONFIRMATION,
            status=ConfirmationStatus.APPROVED,
            approver="user",
            message="Approved by user"
        )
        
        data = result.to_dict()
        
        assert data["approval_id"] == "test_approval"
        assert data["approval_type"] == "user_confirmation"
        assert data["status"] == "approved"
        assert data["approver"] == "user"
        assert data["message"] == "Approved by user"
        assert "timestamp" in data


class TestConfirmationWorkflow:
    """Test ConfirmationWorkflow functionality."""
    
    def test_confirmation_workflow_creation(self):
        """Test ConfirmationWorkflow creation."""
        workflow = ConfirmationWorkflow(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="A test confirmation workflow",
            steps=[ConfirmationStep.USER_CONFIRMATION, ConfirmationStep.FINAL_APPROVAL],
            timeout_minutes=45
        )
        
        assert workflow.workflow_id == "test_workflow"
        assert workflow.name == "Test Workflow"
        assert workflow.description == "A test confirmation workflow"
        assert len(workflow.steps) == 2
        assert workflow.timeout_minutes == 45
    
    def test_confirmation_workflow_serialization(self):
        """Test ConfirmationWorkflow serialization."""
        workflow = ConfirmationWorkflow(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="A test confirmation workflow",
            steps=[ConfirmationStep.USER_CONFIRMATION]
        )
        
        data = workflow.to_dict()
        
        assert data["workflow_id"] == "test_workflow"
        assert data["name"] == "Test Workflow"
        assert data["steps"] == ["user_confirmation"]
        assert data["is_active"] is True


class TestConfirmationSession:
    """Test ConfirmationSession functionality."""
    
    def test_confirmation_session_creation(self):
        """Test ConfirmationSession creation."""
        session = ConfirmationSession(
            session_id="session_001",
            order_id="order_001",
            workflow_id="workflow_001",
            status=ConfirmationStatus.PENDING,
            current_step=ConfirmationStep.USER_CONFIRMATION
        )
        
        assert session.session_id == "session_001"
        assert session.order_id == "order_001"
        assert session.workflow_id == "workflow_001"
        assert session.status == ConfirmationStatus.PENDING
        assert session.current_step == ConfirmationStep.USER_CONFIRMATION
        assert len(session.approval_results) == 0
    
    def test_add_approval_result(self):
        """Test adding approval results to session."""
        session = ConfirmationSession(
            session_id="session_001",
            order_id="order_001",
            workflow_id="workflow_001",
            status=ConfirmationStatus.PENDING
        )
        
        result = ApprovalResult(
            approval_id="test_approval",
            approval_type=ApprovalType.USER_CONFIRMATION,
            status=ConfirmationStatus.APPROVED,
            approver="user",
            message="Approved"
        )
        
        session.add_approval_result(result)
        
        assert len(session.approval_results) == 1
        assert session.approval_results[0].approval_id == "test_approval"
        assert session.updated_at > session.created_at
    
    def test_get_approval_by_type(self):
        """Test getting approval result by type."""
        session = ConfirmationSession(
            session_id="session_001",
            order_id="order_001",
            workflow_id="workflow_001",
            status=ConfirmationStatus.PENDING
        )
        
        # Add approval results of different types
        result1 = ApprovalResult(
            approval_id="approval1",
            approval_type=ApprovalType.USER_CONFIRMATION,
            status=ConfirmationStatus.APPROVED,
            approver="user",
            message="User approved"
        )
        
        result2 = ApprovalResult(
            approval_id="approval2",
            approval_type=ApprovalType.VENDOR_CONFIRMATION,
            status=ConfirmationStatus.APPROVED,
            approver="vendor",
            message="Vendor approved"
        )
        
        session.add_approval_result(result1)
        session.add_approval_result(result2)
        
        user_approval = session.get_approval_by_type(ApprovalType.USER_CONFIRMATION)
        assert user_approval is not None
        assert user_approval.approval_id == "approval1"
        
        vendor_approval = session.get_approval_by_type(ApprovalType.VENDOR_CONFIRMATION)
        assert vendor_approval is not None
        assert vendor_approval.approval_id == "approval2"
        
        non_existent_approval = session.get_approval_by_type(ApprovalType.MANAGER_APPROVAL)
        assert non_existent_approval is None
    
    def test_is_all_required_approvals_complete(self):
        """Test checking if all required approvals are complete."""
        # Create workflow with required approvals
        workflow = ConfirmationWorkflow(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="Test workflow",
            approval_requirements=[
                ApprovalRequirement(
                    approval_id="user_approval",
                    approval_type=ApprovalType.USER_CONFIRMATION,
                    description="User approval",
                    required_by="user",
                    is_required=True
                ),
                ApprovalRequirement(
                    approval_id="vendor_approval",
                    approval_type=ApprovalType.VENDOR_CONFIRMATION,
                    description="Vendor approval",
                    required_by="vendor",
                    is_required=True
                ),
                ApprovalRequirement(
                    approval_id="optional_approval",
                    approval_type=ApprovalType.MANAGER_APPROVAL,
                    description="Optional manager approval",
                    required_by="manager",
                    is_required=False
                )
            ]
        )
        
        session = ConfirmationSession(
            session_id="session_001",
            order_id="order_001",
            workflow_id="test_workflow",
            status=ConfirmationStatus.PENDING
        )
        
        # Test with no approvals
        assert not session.is_all_required_approvals_complete(workflow)
        
        # Add user approval only
        user_result = ApprovalResult(
            approval_id="user_approval",
            approval_type=ApprovalType.USER_CONFIRMATION,
            status=ConfirmationStatus.APPROVED,
            approver="user",
            message="User approved"
        )
        session.add_approval_result(user_result)
        assert not session.is_all_required_approvals_complete(workflow)
        
        # Add vendor approval
        vendor_result = ApprovalResult(
            approval_id="vendor_approval",
            approval_type=ApprovalType.VENDOR_CONFIRMATION,
            status=ConfirmationStatus.APPROVED,
            approver="vendor",
            message="Vendor approved"
        )
        session.add_approval_result(vendor_result)
        assert session.is_all_required_approvals_complete(workflow)


class TestOrderConfirmationManager:
    """Test OrderConfirmationManager functionality."""
    
    def test_order_confirmation_manager_creation(self):
        """Test OrderConfirmationManager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OrderConfirmationManager(temp_dir)
            
            assert manager.storage_path == Path(temp_dir)
            assert len(manager.confirmation_workflows) > 0
            assert len(manager.active_sessions) == 0
            assert len(manager.confirmation_history) == 0
    
    def test_add_confirmation_workflow(self):
        """Test adding confirmation workflows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OrderConfirmationManager(temp_dir)
            
            workflow = ConfirmationWorkflow(
                workflow_id="custom_workflow",
                name="Custom Workflow",
                description="A custom confirmation workflow"
            )
            
            manager.add_confirmation_workflow(workflow)
            assert "custom_workflow" in manager.confirmation_workflows
    
    def test_get_workflow_for_order(self):
        """Test determining workflow for order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OrderConfirmationManager(temp_dir)
            
            # Create validation report
            validation_report = ValidationReport(
                order_id="order_001",
                validation_session_id="validation_001",
                overall_status=ValidationStatus.PASSED
            )
            
            # Test with low-value order
            low_value_order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=Vendor("vendor_001", "Test Vendor", "test", 4.5, 25, 15.0, 3.0),
                items=[],
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=25.0
            )
            
            workflow = manager.get_workflow_for_order(low_value_order, validation_report)
            assert workflow.workflow_id == "standard_order_confirmation"
            
            # Test with high-value order
            high_value_order = Order(
                order_id="order_002",
                user_id="user_001",
                vendor=Vendor("vendor_001", "Test Vendor", "test", 4.5, 25, 15.0, 3.0),
                items=[],
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=150.0
            )
            
            workflow = manager.get_workflow_for_order(high_value_order, validation_report)
            assert workflow.workflow_id == "high_value_order_confirmation"
    
    @pytest.mark.asyncio
    async def test_start_confirmation_workflow(self):
        """Test starting confirmation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OrderConfirmationManager(temp_dir)
            
            # Create test order
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=Vendor("vendor_001", "Test Vendor", "test", 4.5, 25, 15.0, 3.0),
                items=[],
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=25.0
            )
            
            # Create validation report
            validation_report = ValidationReport(
                order_id="order_001",
                validation_session_id="validation_001",
                overall_status=ValidationStatus.PASSED
            )
            
            # Start confirmation workflow
            session = await manager.start_confirmation_workflow(order, validation_report)
            
            assert session is not None
            assert session.order_id == "order_001"
            assert session.status in [ConfirmationStatus.PENDING, ConfirmationStatus.APPROVED, ConfirmationStatus.REJECTED]
            assert session.session_id in manager.active_sessions
    
    def test_get_confirmation_session(self):
        """Test getting confirmation session by ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OrderConfirmationManager(temp_dir)
            
            # Test with non-existent session
            session = manager.get_confirmation_session("non_existent")
            assert session is None
    
    def test_get_active_sessions(self):
        """Test getting active confirmation sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OrderConfirmationManager(temp_dir)
            
            # Test with no active sessions
            sessions = manager.get_active_sessions()
            assert len(sessions) == 0
    
    def test_get_confirmation_statistics(self):
        """Test getting confirmation statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OrderConfirmationManager(temp_dir)
            
            # Test with no history
            stats = manager.get_confirmation_statistics()
            assert stats["total_sessions"] == 0
            assert stats["workflows_available"] > 0


class TestOrderValidationConfirmationIntegration:
    """Integration tests for Order Validation and Confirmation systems."""
    
    @pytest.mark.asyncio
    async def test_complete_validation_and_confirmation_workflow(self):
        """Test complete workflow from validation to confirmation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize systems
            validation_engine = OrderValidationEngine(temp_dir)
            confirmation_manager = OrderConfirmationManager(temp_dir)
            
            # Create test order
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            items = [
                OrderItem(
                    item_id="item_001",
                    name="Test Item",
                    quantity=1,
                    unit_price=20.0,
                    total_price=20.0
                )
            ]
            
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=items,
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=23.0,
                delivery_address="123 Test St"
            )
            
            # Step 1: Validate order
            validation_report = await validation_engine.validate_order(order)
            assert validation_report is not None
            assert validation_report.order_id == "order_001"
            
            # Step 2: Start confirmation workflow
            confirmation_session = await confirmation_manager.start_confirmation_workflow(
                order, validation_report
            )
            assert confirmation_session is not None
            assert confirmation_session.order_id == "order_001"
            
            # Step 3: Verify workflow completion
            assert confirmation_session.status in [
                ConfirmationStatus.PENDING, 
                ConfirmationStatus.APPROVED, 
                ConfirmationStatus.REJECTED
            ]
    
    @pytest.mark.asyncio
    async def test_persistence_and_recovery(self):
        """Test persistence and recovery of validation and confirmation data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first instances
            validation_engine1 = OrderValidationEngine(temp_dir)
            confirmation_manager1 = OrderConfirmationManager(temp_dir)
            
            # Create test order
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=Vendor("vendor_001", "Test Vendor", "test", 4.5, 25, 15.0, 3.0),
                items=[],
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=25.0
            )
            
            # Perform validation
            validation_report = await validation_engine1.validate_order(order)
            
            # Start confirmation workflow
            confirmation_session = await confirmation_manager1.start_confirmation_workflow(
                order, validation_report
            )
            
            # Complete the session
            confirmation_session.status = ConfirmationStatus.APPROVED
            confirmation_session.completed_at = datetime.now()
            confirmation_manager1.confirmation_history.append(confirmation_session)
            confirmation_manager1._save_confirmation_history()
            
            # Create second instances (should load history)
            validation_engine2 = OrderValidationEngine(temp_dir)
            confirmation_manager2 = OrderConfirmationManager(temp_dir)
            
            # Verify history was loaded
            assert len(validation_engine2.validation_history) == 1
            assert len(confirmation_manager2.confirmation_history) == 1
            
            # Verify data integrity
            loaded_report = validation_engine2.validation_history[0]
            assert loaded_report.order_id == "order_001"
            
            loaded_session = confirmation_manager2.confirmation_history[0]
            assert loaded_session.order_id == "order_001"
            assert loaded_session.status == ConfirmationStatus.APPROVED
