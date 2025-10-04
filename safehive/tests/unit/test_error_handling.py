"""
Unit tests for Error Handling and Retry Logic with Agent Learning.
"""

import asyncio
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from safehive.agents.error_handling import (
    ErrorSeverity, ErrorCategory, RetryStrategy, RecoveryAction,
    ErrorContext, ErrorRecord, RetryConfig, LearningInsight, ErrorHandler, with_error_handling
)


class TestErrorSeverity:
    """Test ErrorSeverity enum."""
    
    def test_error_severity_values(self):
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestErrorCategory:
    """Test ErrorCategory enum."""
    
    def test_error_category_values(self):
        """Test ErrorCategory enum values."""
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.AUTHENTICATION.value == "authentication"
        assert ErrorCategory.AUTHORIZATION.value == "authorization"
        assert ErrorCategory.RESOURCE.value == "resource"
        assert ErrorCategory.BUSINESS_LOGIC.value == "business_logic"
        assert ErrorCategory.SYSTEM.value == "system"
        assert ErrorCategory.USER_INPUT.value == "user_input"
        assert ErrorCategory.EXTERNAL_SERVICE.value == "external_service"
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.UNKNOWN.value == "unknown"


class TestRetryStrategy:
    """Test RetryStrategy enum."""
    
    def test_retry_strategy_values(self):
        """Test RetryStrategy enum values."""
        assert RetryStrategy.IMMEDIATE.value == "immediate"
        assert RetryStrategy.EXPONENTIAL_BACKOFF.value == "exponential_backoff"
        assert RetryStrategy.LINEAR_BACKOFF.value == "linear_backoff"
        assert RetryStrategy.FIXED_DELAY.value == "fixed_delay"
        assert RetryStrategy.CUSTOM.value == "custom"
        assert RetryStrategy.NO_RETRY.value == "no_retry"


class TestRecoveryAction:
    """Test RecoveryAction enum."""
    
    def test_recovery_action_values(self):
        """Test RecoveryAction enum values."""
        assert RecoveryAction.RETRY.value == "retry"
        assert RecoveryAction.FALLBACK.value == "fallback"
        assert RecoveryAction.SKIP.value == "skip"
        assert RecoveryAction.ESCALATE.value == "escalate"
        assert RecoveryAction.TERMINATE.value == "terminate"
        assert RecoveryAction.USER_INTERVENTION.value == "user_intervention"


class TestErrorContext:
    """Test ErrorContext functionality."""
    
    def test_error_context_creation(self):
        """Test ErrorContext creation."""
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            user_id="user_001",
            session_id="session_001",
            order_id="order_001",
            vendor_id="vendor_001",
            metadata={"test_key": "test_value"}
        )
        
        assert context.operation == "test_operation"
        assert context.component == "test_component"
        assert context.user_id == "user_001"
        assert context.session_id == "session_001"
        assert context.order_id == "order_001"
        assert context.vendor_id == "vendor_001"
        assert context.metadata["test_key"] == "test_value"
        assert isinstance(context.timestamp, datetime)
    
    def test_error_context_serialization(self):
        """Test ErrorContext serialization."""
        context = ErrorContext(
            operation="test_operation",
            component="test_component",
            user_id="user_001",
            metadata={"test_key": "test_value"}
        )
        
        data = context.to_dict()
        
        assert data["operation"] == "test_operation"
        assert data["component"] == "test_component"
        assert data["user_id"] == "user_001"
        assert data["metadata"]["test_key"] == "test_value"
        assert "timestamp" in data


class TestErrorRecord:
    """Test ErrorRecord functionality."""
    
    def test_error_record_creation(self):
        """Test ErrorRecord creation."""
        context = ErrorContext(
            operation="test_operation",
            component="test_component"
        )
        
        record = ErrorRecord(
            error_id="error_001",
            error_type="TestError",
            error_message="Test error message",
            error_category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            stack_trace="test stack trace",
            retry_count=1,
            max_retries=3,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            recovery_action=RecoveryAction.RETRY,
            resolved=True,
            resolution_notes="Test resolution"
        )
        
        assert record.error_id == "error_001"
        assert record.error_type == "TestError"
        assert record.error_message == "Test error message"
        assert record.error_category == ErrorCategory.VALIDATION
        assert record.severity == ErrorSeverity.MEDIUM
        assert record.context == context
        assert record.stack_trace == "test stack trace"
        assert record.retry_count == 1
        assert record.max_retries == 3
        assert record.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert record.recovery_action == RecoveryAction.RETRY
        assert record.resolved is True
        assert record.resolution_notes == "Test resolution"
    
    def test_error_record_serialization(self):
        """Test ErrorRecord serialization."""
        context = ErrorContext(
            operation="test_operation",
            component="test_component"
        )
        
        record = ErrorRecord(
            error_id="error_001",
            error_type="TestError",
            error_message="Test error message",
            error_category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )
        
        data = record.to_dict()
        
        assert data["error_id"] == "error_001"
        assert data["error_type"] == "TestError"
        assert data["error_message"] == "Test error message"
        assert data["error_category"] == "validation"
        assert data["severity"] == "medium"
        assert data["context"]["operation"] == "test_operation"
        assert data["retry_count"] == 0
        assert data["max_retries"] == 3


class TestRetryConfig:
    """Test RetryConfig functionality."""
    
    def test_retry_config_creation(self):
        """Test RetryConfig creation."""
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=30.0,
            backoff_multiplier=1.5,
            jitter=True,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0
        assert config.backoff_multiplier == 1.5
        assert config.jitter is True
        assert config.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF
    
    def test_retry_config_serialization(self):
        """Test RetryConfig serialization."""
        config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        data = config.to_dict()
        
        assert data["max_retries"] == 3
        assert data["base_delay"] == 1.0
        assert data["max_delay"] == 60.0  # default
        assert data["backoff_multiplier"] == 2.0  # default
        assert data["jitter"] is True  # default
        assert data["retry_strategy"] == "exponential_backoff"


class TestLearningInsight:
    """Test LearningInsight functionality."""
    
    def test_learning_insight_creation(self):
        """Test LearningInsight creation."""
        insight = LearningInsight(
            insight_id="insight_001",
            error_pattern="network|timeout",
            insight_type="failure_pattern",
            description="Network timeout pattern",
            confidence=0.8,
            occurrence_count=5,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            suggested_action="Implement exponential backoff",
            metadata={"pattern_type": "network"}
        )
        
        assert insight.insight_id == "insight_001"
        assert insight.error_pattern == "network|timeout"
        assert insight.insight_type == "failure_pattern"
        assert insight.description == "Network timeout pattern"
        assert insight.confidence == 0.8
        assert insight.occurrence_count == 5
        assert insight.suggested_action == "Implement exponential backoff"
        assert insight.metadata["pattern_type"] == "network"
    
    def test_learning_insight_serialization(self):
        """Test LearningInsight serialization."""
        now = datetime.now()
        insight = LearningInsight(
            insight_id="insight_001",
            error_pattern="network|timeout",
            insight_type="failure_pattern",
            description="Network timeout pattern",
            confidence=0.8,
            occurrence_count=5,
            first_seen=now,
            last_seen=now,
            suggested_action="Implement exponential backoff"
        )
        
        data = insight.to_dict()
        
        assert data["insight_id"] == "insight_001"
        assert data["error_pattern"] == "network|timeout"
        assert data["insight_type"] == "failure_pattern"
        assert data["description"] == "Network timeout pattern"
        assert data["confidence"] == 0.8
        assert data["occurrence_count"] == 5
        assert data["suggested_action"] == "Implement exponential backoff"
        assert data["first_seen"] == now.isoformat()
        assert data["last_seen"] == now.isoformat()


class TestErrorHandler:
    """Test ErrorHandler functionality."""
    
    def test_error_handler_creation(self):
        """Test ErrorHandler creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            assert handler.storage_path == Path(temp_dir)
            assert len(handler.error_records) == 0
            assert len(handler.active_retries) == 0
            assert len(handler.learning_insights) == 0
            assert len(handler.retry_configs) > 0
    
    def test_classify_error_network(self):
        """Test error classification for network errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="test_operation",
                component="test_component"
            )
            
            # Test network error classification
            network_error = Exception("Connection timeout")
            category, severity = handler.classify_error(network_error, context)
            
            assert category == ErrorCategory.NETWORK
            assert severity == ErrorSeverity.HIGH
    
    def test_classify_error_validation(self):
        """Test error classification for validation errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="test_operation",
                component="test_component"
            )
            
            # Test validation error classification
            validation_error = Exception("Invalid input validation failed")
            category, severity = handler.classify_error(validation_error, context)
            
            assert category == ErrorCategory.VALIDATION
            assert severity == ErrorSeverity.MEDIUM
    
    def test_classify_error_authentication(self):
        """Test error classification for authentication errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="test_operation",
                component="test_component"
            )
            
            # Test authentication error classification
            auth_error = Exception("Authentication failed")
            category, severity = handler.classify_error(auth_error, context)
            
            assert category == ErrorCategory.AUTHENTICATION
            assert severity == ErrorSeverity.HIGH
    
    def test_classify_error_unknown(self):
        """Test error classification for unknown errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="test_operation",
                component="test_component"
            )
            
            # Test unknown error classification
            unknown_error = Exception("Some random error")
            category, severity = handler.classify_error(unknown_error, context)
            
            assert category == ErrorCategory.UNKNOWN
            assert severity == ErrorSeverity.MEDIUM
    
    def test_create_error_record(self):
        """Test creating error record."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="test_operation",
                component="test_component"
            )
            
            error = Exception("Test error")
            record = handler.create_error_record(error, context)
            
            assert record.error_type == "Exception"
            assert record.error_message == "Test error"
            assert record.context == context
            assert record.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF  # default for unknown
            assert record.recovery_action == RecoveryAction.RETRY
    
    def test_calculate_retry_delay_immediate(self):
        """Test retry delay calculation for immediate strategy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            config = RetryConfig(retry_strategy=RetryStrategy.IMMEDIATE)
            delay = handler._calculate_retry_delay(2, config)
            
            assert delay == 0.0
    
    def test_calculate_retry_delay_fixed(self):
        """Test retry delay calculation for fixed delay strategy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            config = RetryConfig(
                base_delay=2.0,
                retry_strategy=RetryStrategy.FIXED_DELAY,
                jitter=False  # Disable jitter for predictable testing
            )
            delay = handler._calculate_retry_delay(3, config)
            
            assert delay == 2.0
    
    def test_calculate_retry_delay_linear(self):
        """Test retry delay calculation for linear backoff strategy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            config = RetryConfig(
                base_delay=1.0,
                retry_strategy=RetryStrategy.LINEAR_BACKOFF,
                jitter=False  # Disable jitter for predictable testing
            )
            
            delay1 = handler._calculate_retry_delay(0, config)
            delay2 = handler._calculate_retry_delay(1, config)
            delay3 = handler._calculate_retry_delay(2, config)
            
            assert delay1 == 1.0  # base_delay * (0 + 1)
            assert delay2 == 2.0  # base_delay * (1 + 1)
            assert delay3 == 3.0  # base_delay * (2 + 1)
    
    def test_calculate_retry_delay_exponential(self):
        """Test retry delay calculation for exponential backoff strategy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            config = RetryConfig(
                base_delay=1.0,
                backoff_multiplier=2.0,
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                jitter=False  # Disable jitter for predictable testing
            )
            
            delay1 = handler._calculate_retry_delay(0, config)
            delay2 = handler._calculate_retry_delay(1, config)
            delay3 = handler._calculate_retry_delay(2, config)
            
            assert delay1 == 1.0  # base_delay * (2.0^0)
            assert delay2 == 2.0  # base_delay * (2.0^1)
            assert delay3 == 4.0  # base_delay * (2.0^2)
    
    def test_calculate_retry_delay_max_limit(self):
        """Test retry delay calculation respects max delay limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            config = RetryConfig(
                base_delay=1.0,
                max_delay=5.0,
                backoff_multiplier=10.0,
                retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                jitter=False
            )
            
            delay = handler._calculate_retry_delay(1, config)
            
            # Should be capped at max_delay (5.0) even though 1.0 * 10.0^1 = 10.0
            assert delay == 5.0
    
    def test_matches_error_pattern(self):
        """Test error pattern matching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="test_operation",
                component="test_component"
            )
            
            record = ErrorRecord(
                error_id="error_001",
                error_type="NetworkError",
                error_message="Connection timeout occurred",
                error_category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                context=context
            )
            
            # Test pattern matching
            assert handler._matches_error_pattern(record, "timeout")
            assert handler._matches_error_pattern(record, "NetworkError")
            assert handler._matches_error_pattern(record, "connection")
            assert not handler._matches_error_pattern(record, "validation")
    
    def test_extract_error_pattern(self):
        """Test error pattern extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="create_order",
                component="orchestrator"
            )
            
            record = ErrorRecord(
                error_id="error_001",
                error_type="ValidationError",
                error_message="Invalid input",
                error_category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                context=context
            )
            
            pattern = handler._extract_error_pattern(record)
            
            expected_pattern = "ValidationError|validation|create_order|orchestrator"
            assert pattern == expected_pattern
    
    def test_generate_suggested_action(self):
        """Test suggested action generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="test_operation",
                component="test_component"
            )
            
            # Test network error suggestion
            network_record = ErrorRecord(
                error_id="error_001",
                error_type="NetworkError",
                error_message="Connection failed",
                error_category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                context=context
            )
            
            action = handler._generate_suggested_action(network_record)
            assert "network connectivity" in action.lower()
            
            # Test validation error suggestion
            validation_record = ErrorRecord(
                error_id="error_002",
                error_type="ValidationError",
                error_message="Invalid input",
                error_category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                context=context
            )
            
            action = handler._generate_suggested_action(validation_record)
            assert "validate input" in action.lower()
    
    def test_get_error_statistics_empty(self):
        """Test error statistics with no errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            stats = handler.get_error_statistics()
            
            assert stats["total_errors"] == 0
            assert "resolved_errors" in stats
            assert "unresolved_errors" in stats
            assert "resolution_rate" in stats
            assert "retry_attempts" in stats
            assert "successful_retries" in stats
            assert "retry_success_rate" in stats
            assert "learning_insights_count" in stats
    
    def test_get_error_statistics_with_data(self):
        """Test error statistics with error data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            # Add some mock error records
            context1 = ErrorContext(operation="op1", component="comp1")
            record1 = ErrorRecord(
                error_id="error_001",
                error_type="TestError",
                error_message="Test error 1",
                error_category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                context=context1,
                retry_count=2,
                resolved=True
            )
            
            context2 = ErrorContext(operation="op2", component="comp2")
            record2 = ErrorRecord(
                error_id="error_002",
                error_type="TestError",
                error_message="Test error 2",
                error_category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                context=context2,
                retry_count=0,
                resolved=False
            )
            
            handler.error_records.extend([record1, record2])
            
            stats = handler.get_error_statistics()
            
            assert stats["total_errors"] == 2
            assert stats["resolved_errors"] == 1
            assert stats["unresolved_errors"] == 1
            assert stats["resolution_rate"] == 0.5
            assert stats["retry_attempts"] == 1  # Only record1 has retry_count > 0
            assert stats["successful_retries"] == 1  # Only record1 is resolved with retries
            assert stats["retry_success_rate"] == 1.0  # 1 successful / 1 attempt
            assert "category_counts" in stats
            assert "severity_counts" in stats


class TestErrorHandlerIntegration:
    """Integration tests for ErrorHandler."""
    
    @pytest.mark.asyncio
    async def test_handle_error_with_retry_success(self):
        """Test error handling with successful retry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="test_operation",
                component="test_component"
            )
            
            # Mock operation that fails first time, then succeeds
            call_count = 0
            
            async def mock_operation():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Temporary failure")
                return "success"
            
            # Handle error with retry
            result = await handler.handle_error(
                Exception("Temporary failure"),
                context,
                mock_operation
            )
            
            assert result == "success"
            assert call_count == 2  # Should have retried once
            assert len(handler.error_records) == 1
            
            error_record = handler.error_records[0]
            assert error_record.retry_count == 1
            assert error_record.resolved is True
    
    @pytest.mark.asyncio
    async def test_handle_error_with_retry_exhausted(self):
        """Test error handling with exhausted retries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="test_operation",
                component="test_component"
            )
            
            # Mock operation that always fails
            async def mock_operation():
                raise Exception("Persistent failure")
            
            # Handle error with retry (should exhaust retries)
            with pytest.raises(Exception):
                await handler.handle_error(
                    Exception("Persistent failure"),
                    context,
                    mock_operation
                )
            
            assert len(handler.error_records) == 1
            
            error_record = handler.error_records[0]
            assert error_record.retry_count == 2  # Should have exhausted retries (max_retries=2 for unknown category)
            assert error_record.resolved is False
    
    @pytest.mark.asyncio
    async def test_handle_error_fallback(self):
        """Test error handling with fallback mechanism."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="test_operation",
                component="test_component"
            )
            
            # Create error record and add to handler
            error_record = handler.create_error_record(Exception("Test error"), context)
            handler.error_records.append(error_record)
            
            # Handle error with fallback
            result = await handler._handle_fallback(error_record, None)
            
            assert result is not None
            assert result["status"] == "fallback"
            assert len(handler.error_records) == 1
    
    @pytest.mark.asyncio
    async def test_handle_error_escalation(self):
        """Test error handling with escalation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="test_operation",
                component="test_component"
            )
            
            # Create error record and add to handler
            error_record = handler.create_error_record(Exception("Critical error"), context)
            handler.error_records.append(error_record)
            
            # Handle error with escalation
            with pytest.raises(Exception):
                await handler._handle_escalation(error_record)
            
            assert len(handler.error_records) == 1
    
    def test_learn_from_failure(self):
        """Test learning from failure patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="test_operation",
                component="test_component"
            )
            
            error_record = ErrorRecord(
                error_id="error_001",
                error_type="NetworkError",
                error_message="Connection timeout",
                error_category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                context=context,
                retry_count=3,
                resolved=False
            )
            
            # Learn from failure
            handler._learn_from_failure(error_record)
            
            assert len(handler.learning_insights) == 1
            
            insight = handler.learning_insights[0]
            assert insight.error_pattern == "NetworkError|network|test_operation|test_component"
            assert insight.insight_type == "failure_pattern"
            assert insight.occurrence_count == 1
            assert "network connectivity" in insight.suggested_action.lower()
    
    def test_learn_from_failure_existing_pattern(self):
        """Test learning from existing failure pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            context = ErrorContext(
                operation="test_operation",
                component="test_component"
            )
            
            # Create existing insight
            existing_insight = LearningInsight(
                insight_id="insight_001",
                error_pattern="NetworkError|network|test_operation|test_component",
                insight_type="failure_pattern",
                description="Existing pattern",
                confidence=0.5,
                occurrence_count=1,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                suggested_action="Existing action"
            )
            handler.learning_insights.append(existing_insight)
            
            error_record = ErrorRecord(
                error_id="error_001",
                error_type="NetworkError",
                error_message="Connection timeout",
                error_category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                context=context,
                retry_count=3,
                resolved=False
            )
            
            # Learn from failure (should update existing insight)
            handler._learn_from_failure(error_record)
            
            assert len(handler.learning_insights) == 1  # Should not create new insight
            
            insight = handler.learning_insights[0]
            assert insight.occurrence_count == 2  # Should increment count
            assert insight.confidence > 0.5  # Should increase confidence
    
    def test_persistence_and_recovery(self):
        """Test persistence and recovery of error data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first handler instance
            handler1 = ErrorHandler(temp_dir)
            
            # Add some error records
            context = ErrorContext(operation="test_op", component="test_comp")
            record = handler1.create_error_record(Exception("Test error"), context)
            record.resolved = True
            record.resolution_time = datetime.now()
            handler1.error_records.append(record)
            
            # Add some learning insights
            insight = LearningInsight(
                insight_id="insight_001",
                error_pattern="TestError|unknown|test_op|test_comp",
                insight_type="failure_pattern",
                description="Test insight",
                confidence=0.8,
                occurrence_count=1,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                suggested_action="Test action"
            )
            handler1.learning_insights.append(insight)
            
            # Save data
            handler1._save_error_history()
            handler1._save_learning_insights()
            
            # Create second handler instance (should load data)
            handler2 = ErrorHandler(temp_dir)
            
            # Verify data was loaded
            assert len(handler2.error_records) == 1
            assert handler2.error_records[0].error_id == record.error_id
            assert handler2.error_records[0].resolved is True
            
            assert len(handler2.learning_insights) == 1
            assert handler2.learning_insights[0].insight_id == insight.insight_id


class TestWithErrorHandlingDecorator:
    """Test the with_error_handling decorator."""
    
    @pytest.mark.asyncio
    async def test_with_error_handling_async_success(self):
        """Test decorator with successful async operation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            @with_error_handling(handler, "test_operation", "test_component")
            async def async_operation():
                return "success"
            
            result = await async_operation()
            assert result == "success"
            assert len(handler.error_records) == 0  # No errors recorded
    
    @pytest.mark.asyncio
    async def test_with_error_handling_async_error(self):
        """Test decorator with async operation that raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            @with_error_handling(handler, "test_operation", "test_component")
            async def async_operation():
                raise Exception("Test error")
            
            # Should handle error and potentially retry
            try:
                await async_operation()
            except Exception:
                pass  # Expected to raise exception after retries
            
            assert len(handler.error_records) == 1
            assert handler.error_records[0].error_message == "Test error"
    
    def test_with_error_handling_sync_success(self):
        """Test decorator with successful sync operation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            @with_error_handling(handler, "test_operation", "test_component")
            def sync_operation():
                return "success"
            
            result = sync_operation()
            assert result == "success"
            assert len(handler.error_records) == 0  # No errors recorded
    
    def test_with_error_handling_sync_error(self):
        """Test decorator with sync operation that raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ErrorHandler(temp_dir)
            
            @with_error_handling(handler, "test_operation", "test_component")
            def sync_operation():
                raise Exception("Test error")
            
            # Should handle error and potentially retry
            try:
                sync_operation()
            except Exception:
                pass  # Expected to raise exception after retries
            
            assert len(handler.error_records) == 1
            assert handler.error_records[0].error_message == "Test error"
