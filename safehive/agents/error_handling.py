"""
Error Handling and Retry Logic with Agent Learning

This module implements intelligent error handling, retry mechanisms, and learning
capabilities for the SafeHive AI Security Sandbox, providing resilience and
continuous improvement through failure analysis.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type
import traceback
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.metrics import record_metric, MetricType

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors."""
    NETWORK = "network"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategies for different error types."""
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    CUSTOM = "custom"
    NO_RETRY = "no_retry"


class RecoveryAction(Enum):
    """Recovery actions for errors."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ESCALATE = "escalate"
    TERMINATE = "terminate"
    USER_INTERVENTION = "user_intervention"


@dataclass
class ErrorContext:
    """Context information for an error."""
    operation: str
    component: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    order_id: Optional[str] = None
    vendor_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "component": self.component,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "order_id": self.order_id,
            "vendor_id": self.vendor_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_id: str
    error_type: str
    error_message: str
    error_category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    stack_trace: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    recovery_action: RecoveryAction = RecoveryAction.RETRY
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    learning_insights: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_category": self.error_category.value,
            "severity": self.severity.value,
            "context": self.context.to_dict(),
            "stack_trace": self.stack_trace,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "retry_strategy": self.retry_strategy.value,
            "recovery_action": self.recovery_action.value,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "resolution_notes": self.resolution_notes,
            "learning_insights": self.learning_insights
        }


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "backoff_multiplier": self.backoff_multiplier,
            "jitter": self.jitter,
            "retry_strategy": self.retry_strategy.value
        }


@dataclass
class LearningInsight:
    """Insight learned from error analysis."""
    insight_id: str
    error_pattern: str
    insight_type: str
    description: str
    confidence: float
    occurrence_count: int
    first_seen: datetime
    last_seen: datetime
    suggested_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "insight_id": self.insight_id,
            "error_pattern": self.error_pattern,
            "insight_type": self.insight_type,
            "description": self.description,
            "confidence": self.confidence,
            "occurrence_count": self.occurrence_count,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "suggested_action": self.suggested_action,
            "metadata": self.metadata
        }


class ErrorHandler:
    """Intelligent error handler with learning capabilities."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_error_handling"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Error tracking
        self.error_records: List[ErrorRecord] = []
        self.active_retries: Dict[str, ErrorRecord] = {}
        self.learning_insights: List[LearningInsight] = []
        
        # Error patterns and configurations
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        self.retry_configs: Dict[ErrorCategory, RetryConfig] = {}
        
        # Initialize default configurations
        self._initialize_default_configs()
        
        # Load historical data
        self._load_error_history()
        self._load_learning_insights()
        
        logger.info("Error Handler initialized")
    
    def _initialize_default_configs(self) -> None:
        """Initialize default error handling configurations."""
        # Network errors - aggressive retry with exponential backoff
        self.retry_configs[ErrorCategory.NETWORK] = RetryConfig(
            max_retries=5,
            base_delay=1.0,
            max_delay=30.0,
            backoff_multiplier=2.0,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        # Timeout errors - moderate retry
        self.retry_configs[ErrorCategory.TIMEOUT] = RetryConfig(
            max_retries=3,
            base_delay=2.0,
            max_delay=20.0,
            backoff_multiplier=1.5,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        # Validation errors - no retry, immediate fallback
        self.retry_configs[ErrorCategory.VALIDATION] = RetryConfig(
            max_retries=0,
            retry_strategy=RetryStrategy.NO_RETRY
        )
        
        # Authentication errors - limited retry
        self.retry_configs[ErrorCategory.AUTHENTICATION] = RetryConfig(
            max_retries=2,
            base_delay=5.0,
            max_delay=15.0,
            retry_strategy=RetryStrategy.FIXED_DELAY
        )
        
        # Authorization errors - no retry
        self.retry_configs[ErrorCategory.AUTHORIZATION] = RetryConfig(
            max_retries=0,
            retry_strategy=RetryStrategy.NO_RETRY
        )
        
        # Resource errors - moderate retry
        self.retry_configs[ErrorCategory.RESOURCE] = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            retry_strategy=RetryStrategy.LINEAR_BACKOFF
        )
        
        # System errors - limited retry with escalation
        self.retry_configs[ErrorCategory.SYSTEM] = RetryConfig(
            max_retries=2,
            base_delay=3.0,
            max_delay=15.0,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        # External service errors - aggressive retry
        self.retry_configs[ErrorCategory.EXTERNAL_SERVICE] = RetryConfig(
            max_retries=4,
            base_delay=2.0,
            max_delay=60.0,
            backoff_multiplier=2.5,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        # Default configuration
        self.retry_configs[ErrorCategory.UNKNOWN] = RetryConfig(
            max_retries=2,
            base_delay=1.0,
            max_delay=10.0,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
    
    def _load_error_history(self) -> None:
        """Load error history from storage."""
        history_file = self.storage_path / "error_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for error_data in data:
                        error_record = self._reconstruct_error_record(error_data)
                        if error_record:
                            self.error_records.append(error_record)
                logger.info(f"Loaded {len(self.error_records)} error records")
            except Exception as e:
                logger.error(f"Failed to load error history: {e}")
    
    def _save_error_history(self) -> None:
        """Save error history to storage."""
        history_file = self.storage_path / "error_history.json"
        try:
            data = [record.to_dict() for record in self.error_records]
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved error history")
        except Exception as e:
            logger.error(f"Failed to save error history: {e}")
    
    def _load_learning_insights(self) -> None:
        """Load learning insights from storage."""
        insights_file = self.storage_path / "learning_insights.json"
        if insights_file.exists():
            try:
                with open(insights_file, 'r') as f:
                    data = json.load(f)
                    for insight_data in data:
                        insight = self._reconstruct_learning_insight(insight_data)
                        if insight:
                            self.learning_insights.append(insight)
                logger.info(f"Loaded {len(self.learning_insights)} learning insights")
            except Exception as e:
                logger.error(f"Failed to load learning insights: {e}")
    
    def _save_learning_insights(self) -> None:
        """Save learning insights to storage."""
        insights_file = self.storage_path / "learning_insights.json"
        try:
            data = [insight.to_dict() for insight in self.learning_insights]
            with open(insights_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved learning insights")
        except Exception as e:
            logger.error(f"Failed to save learning insights: {e}")
    
    def _reconstruct_error_record(self, data: Dict[str, Any]) -> Optional[ErrorRecord]:
        """Reconstruct ErrorRecord from stored data."""
        try:
            context_data = data["context"]
            context = ErrorContext(
                operation=context_data["operation"],
                component=context_data["component"],
                user_id=context_data.get("user_id"),
                session_id=context_data.get("session_id"),
                order_id=context_data.get("order_id"),
                vendor_id=context_data.get("vendor_id"),
                metadata=context_data.get("metadata", {}),
                timestamp=datetime.fromisoformat(context_data["timestamp"])
            )
            
            record = ErrorRecord(
                error_id=data["error_id"],
                error_type=data["error_type"],
                error_message=data["error_message"],
                error_category=ErrorCategory(data["error_category"]),
                severity=ErrorSeverity(data["severity"]),
                context=context,
                stack_trace=data.get("stack_trace"),
                retry_count=data.get("retry_count", 0),
                max_retries=data.get("max_retries", 3),
                retry_strategy=RetryStrategy(data["retry_strategy"]),
                recovery_action=RecoveryAction(data["recovery_action"]),
                resolved=data.get("resolved", False),
                resolution_time=datetime.fromisoformat(data["resolution_time"]) if data.get("resolution_time") else None,
                resolution_notes=data.get("resolution_notes"),
                learning_insights=data.get("learning_insights", [])
            )
            
            return record
        except Exception as e:
            logger.error(f"Failed to reconstruct error record: {e}")
            return None
    
    def _reconstruct_learning_insight(self, data: Dict[str, Any]) -> Optional[LearningInsight]:
        """Reconstruct LearningInsight from stored data."""
        try:
            insight = LearningInsight(
                insight_id=data["insight_id"],
                error_pattern=data["error_pattern"],
                insight_type=data["insight_type"],
                description=data["description"],
                confidence=data["confidence"],
                occurrence_count=data["occurrence_count"],
                first_seen=datetime.fromisoformat(data["first_seen"]),
                last_seen=datetime.fromisoformat(data["last_seen"]),
                suggested_action=data["suggested_action"],
                metadata=data.get("metadata", {})
            )
            
            return insight
        except Exception as e:
            logger.error(f"Failed to reconstruct learning insight: {e}")
            return None
    
    def classify_error(self, exception: Exception, context: ErrorContext) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify an error based on exception type and context."""
        error_type = type(exception).__name__
        error_message = str(exception)
        
        # Network-related errors
        if any(keyword in error_message.lower() for keyword in ["connection", "network", "timeout", "unreachable"]):
            return ErrorCategory.NETWORK, ErrorSeverity.HIGH
        
        # Timeout errors
        if "timeout" in error_message.lower() or "timed out" in error_message.lower():
            return ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM
        
        # Validation errors
        if any(keyword in error_message.lower() for keyword in ["validation", "invalid", "required", "missing"]):
            return ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM
        
        # Authentication errors
        if any(keyword in error_message.lower() for keyword in ["authentication", "unauthorized", "login", "credentials"]):
            return ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH
        
        # Authorization errors
        if any(keyword in error_message.lower() for keyword in ["forbidden", "access denied", "permission"]):
            return ErrorCategory.AUTHORIZATION, ErrorSeverity.HIGH
        
        # Resource errors
        if any(keyword in error_message.lower() for keyword in ["not found", "resource", "unavailable", "out of"]):
            return ErrorCategory.RESOURCE, ErrorSeverity.MEDIUM
        
        # Business logic errors
        if any(keyword in error_message.lower() for keyword in ["business", "constraint", "rule", "policy"]):
            return ErrorCategory.BUSINESS_LOGIC, ErrorSeverity.MEDIUM
        
        # System errors
        if any(keyword in error_message.lower() for keyword in ["system", "internal", "server", "crash"]):
            return ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL
        
        # External service errors
        if any(keyword in error_message.lower() for keyword in ["service", "api", "external", "third-party"]):
            return ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.HIGH
        
        # Default classification
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    def create_error_record(self, exception: Exception, context: ErrorContext) -> ErrorRecord:
        """Create an error record for an exception."""
        error_id = f"error_{int(time.time())}_{context.operation}"
        error_category, severity = self.classify_error(exception, context)
        
        # Get retry configuration for this error category
        retry_config = self.retry_configs.get(error_category, self.retry_configs[ErrorCategory.UNKNOWN])
        
        # Determine recovery action based on error category
        recovery_action = self._determine_recovery_action(error_category, severity)
        
        record = ErrorRecord(
            error_id=error_id,
            error_type=type(exception).__name__,
            error_message=str(exception),
            error_category=error_category,
            severity=severity,
            context=context,
            stack_trace=traceback.format_exc(),
            max_retries=retry_config.max_retries,
            retry_strategy=retry_config.retry_strategy,
            recovery_action=recovery_action
        )
        
        return record
    
    def _determine_recovery_action(self, category: ErrorCategory, severity: ErrorSeverity) -> RecoveryAction:
        """Determine the appropriate recovery action for an error."""
        if category in [ErrorCategory.VALIDATION, ErrorCategory.AUTHORIZATION]:
            return RecoveryAction.USER_INTERVENTION
        elif category == ErrorCategory.SYSTEM and severity == ErrorSeverity.CRITICAL:
            return RecoveryAction.ESCALATE
        elif severity == ErrorSeverity.CRITICAL:
            return RecoveryAction.TERMINATE
        else:
            return RecoveryAction.RETRY
    
    async def handle_error(self, exception: Exception, context: ErrorContext, 
                          operation_func: Optional[Callable] = None) -> Any:
        """Handle an error with intelligent retry and recovery."""
        # Create error record
        error_record = self.create_error_record(exception, context)
        self.error_records.append(error_record)
        
        # Record metrics
        record_metric("error_handler.error_occurred", 1, MetricType.COUNTER, {
            "error_category": error_record.error_category.value,
            "severity": error_record.severity.value,
            "component": context.component,
            "operation": context.operation
        })
        
        logger.error(f"Error occurred: {error_record.error_type} in {context.operation} - {error_record.error_message}")
        
        # Apply learning insights
        insights = self._apply_learning_insights(error_record)
        error_record.learning_insights.extend(insights)
        
        # Handle based on recovery action
        if error_record.recovery_action == RecoveryAction.RETRY:
            return await self._handle_retry(error_record, operation_func)
        elif error_record.recovery_action == RecoveryAction.FALLBACK:
            return await self._handle_fallback(error_record, operation_func)
        elif error_record.recovery_action == RecoveryAction.ESCALATE:
            return await self._handle_escalation(error_record)
        elif error_record.recovery_action == RecoveryAction.USER_INTERVENTION:
            return await self._handle_user_intervention(error_record)
        else:
            return await self._handle_termination(error_record)
    
    async def _handle_retry(self, error_record: ErrorRecord, operation_func: Optional[Callable]) -> Any:
        """Handle error with retry logic."""
        if not operation_func:
            logger.warning("No operation function provided for retry")
            return None
        
        retry_config = self.retry_configs.get(error_record.error_category, self.retry_configs[ErrorCategory.UNKNOWN])
        
        while error_record.retry_count < error_record.max_retries:
            try:
                # Calculate delay
                delay = self._calculate_retry_delay(error_record.retry_count, retry_config)
                
                logger.info(f"Retrying {error_record.context.operation} after {delay:.2f}s (attempt {error_record.retry_count + 1}/{error_record.max_retries})")
                
                # Wait before retry
                if delay > 0:
                    await asyncio.sleep(delay)
                
                # Execute operation
                result = await operation_func() if asyncio.iscoroutinefunction(operation_func) else operation_func()
                
                # Success - mark as resolved
                error_record.resolved = True
                error_record.resolution_time = datetime.now()
                error_record.resolution_notes = f"Resolved after {error_record.retry_count + 1} retries"
                
                logger.info(f"Operation {error_record.context.operation} succeeded after {error_record.retry_count + 1} attempts")
                
                # Record success metrics
                record_metric("error_handler.retry_success", 1, MetricType.COUNTER, {
                    "error_category": error_record.error_category.value,
                    "retry_count": error_record.retry_count + 1
                })
                
                return result
                
            except Exception as retry_exception:
                error_record.retry_count += 1
                
                # Check if this is the same type of error
                if type(retry_exception).__name__ != error_record.error_type:
                    logger.warning(f"Different error type on retry: {type(retry_exception).__name__} vs {error_record.error_type}")
                    # Create new error record for different error type
                    new_context = ErrorContext(
                        operation=f"{error_record.context.operation}_retry_{error_record.retry_count}",
                        component=error_record.context.component,
                        user_id=error_record.context.user_id,
                        session_id=error_record.context.session_id,
                        order_id=error_record.context.order_id,
                        vendor_id=error_record.context.vendor_id
                    )
                    new_error = self.create_error_record(retry_exception, new_context)
                    self.error_records.append(new_error)
                
                logger.warning(f"Retry {error_record.retry_count} failed for {error_record.context.operation}: {str(retry_exception)}")
                
                # Record retry failure metrics
                record_metric("error_handler.retry_failed", 1, MetricType.COUNTER, {
                    "error_category": error_record.error_category.value,
                    "retry_count": error_record.retry_count
                })
        
        # All retries exhausted
        error_record.resolved = False
        error_record.resolution_notes = f"Failed after {error_record.max_retries} retries"
        
        logger.error(f"All retries exhausted for {error_record.context.operation}")
        
        # Record final failure metrics
        record_metric("error_handler.all_retries_exhausted", 1, MetricType.COUNTER, {
            "error_category": error_record.error_category.value,
            "max_retries": error_record.max_retries
        })
        
        # Learn from this failure pattern
        self._learn_from_failure(error_record)
        
        raise error_record
    
    def _calculate_retry_delay(self, retry_count: int, config: RetryConfig) -> float:
        """Calculate delay for retry based on strategy."""
        if config.retry_strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif config.retry_strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
        elif config.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * (retry_count + 1)
        elif config.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_multiplier ** retry_count)
        else:
            delay = config.base_delay
        
        # Apply jitter
        if config.jitter:
            import random
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor
        
        # Cap at max delay
        delay = min(delay, config.max_delay)
        
        return delay
    
    async def _handle_fallback(self, error_record: ErrorRecord, operation_func: Optional[Callable]) -> Any:
        """Handle error with fallback mechanism."""
        logger.info(f"Attempting fallback for {error_record.context.operation}")
        
        # Implement fallback logic based on error category and context
        fallback_result = None
        
        if error_record.error_category == ErrorCategory.EXTERNAL_SERVICE:
            # Try alternative service or cached data
            fallback_result = await self._fallback_external_service(error_record)
        elif error_record.error_category == ErrorCategory.RESOURCE:
            # Try alternative resource or default values
            fallback_result = await self._fallback_resource(error_record)
        elif error_record.error_category == ErrorCategory.NETWORK:
            # Try offline mode or cached data
            fallback_result = await self._fallback_network(error_record)
        else:
            # Generic fallback
            fallback_result = await self._fallback_generic(error_record)
        
        if fallback_result is not None:
            error_record.resolved = True
            error_record.resolution_time = datetime.now()
            error_record.resolution_notes = "Resolved using fallback mechanism"
            
            logger.info(f"Fallback successful for {error_record.context.operation}")
            return fallback_result
        else:
            logger.error(f"Fallback failed for {error_record.context.operation}")
            raise error_record
    
    async def _fallback_external_service(self, error_record: ErrorRecord) -> Any:
        """Fallback for external service errors."""
        # Implement service-specific fallback logic
        logger.info("Using external service fallback")
        return {"status": "fallback", "message": "Using cached or alternative data"}
    
    async def _fallback_resource(self, error_record: ErrorRecord) -> Any:
        """Fallback for resource errors."""
        # Implement resource-specific fallback logic
        logger.info("Using resource fallback")
        return {"status": "fallback", "message": "Using default or alternative resource"}
    
    async def _fallback_network(self, error_record: ErrorRecord) -> Any:
        """Fallback for network errors."""
        # Implement network-specific fallback logic
        logger.info("Using network fallback")
        return {"status": "fallback", "message": "Operating in offline mode"}
    
    async def _fallback_generic(self, error_record: ErrorRecord) -> Any:
        """Generic fallback mechanism."""
        # Implement generic fallback logic
        logger.info("Using generic fallback")
        return {"status": "fallback", "message": "Using default behavior"}
    
    async def _handle_escalation(self, error_record: ErrorRecord) -> Any:
        """Handle error escalation."""
        logger.critical(f"Escalating error: {error_record.error_id}")
        
        # Implement escalation logic (e.g., notify administrators, create tickets)
        escalation_data = {
            "error_id": error_record.error_id,
            "severity": error_record.severity.value,
            "category": error_record.error_category.value,
            "context": error_record.context.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Record escalation metrics
        record_metric("error_handler.error_escalated", 1, MetricType.COUNTER, {
            "error_category": error_record.error_category.value,
            "severity": error_record.severity.value
        })
        
        # In a real system, this would send alerts, create tickets, etc.
        logger.critical(f"Error escalated: {escalation_data}")
        
        raise error_record
    
    async def _handle_user_intervention(self, error_record: ErrorRecord) -> Any:
        """Handle error requiring user intervention."""
        logger.warning(f"User intervention required for: {error_record.error_id}")
        
        # Implement user intervention logic
        intervention_data = {
            "error_id": error_record.error_id,
            "message": error_record.error_message,
            "suggested_action": "Please check your input and try again",
            "context": error_record.context.to_dict()
        }
        
        # Record user intervention metrics
        record_metric("error_handler.user_intervention_required", 1, MetricType.COUNTER, {
            "error_category": error_record.error_category.value
        })
        
        raise error_record
    
    async def _handle_termination(self, error_record: ErrorRecord) -> Any:
        """Handle error termination."""
        logger.critical(f"Terminating due to error: {error_record.error_id}")
        
        # Implement termination logic
        termination_data = {
            "error_id": error_record.error_id,
            "reason": "Critical error requiring termination",
            "context": error_record.context.to_dict()
        }
        
        # Record termination metrics
        record_metric("error_handler.operation_terminated", 1, MetricType.COUNTER, {
            "error_category": error_record.error_category.value,
            "severity": error_record.severity.value
        })
        
        raise error_record
    
    def _apply_learning_insights(self, error_record: ErrorRecord) -> List[str]:
        """Apply learning insights to error handling."""
        insights = []
        
        # Look for similar error patterns
        for insight in self.learning_insights:
            if self._matches_error_pattern(error_record, insight.error_pattern):
                insights.append(insight.suggested_action)
                logger.info(f"Applied learning insight: {insight.description}")
        
        return insights
    
    def _matches_error_pattern(self, error_record: ErrorRecord, pattern: str) -> bool:
        """Check if error matches a learned pattern."""
        # Simple pattern matching - in a real system, this would be more sophisticated
        pattern_lower = pattern.lower()
        error_message_lower = error_record.error_message.lower()
        
        return (pattern_lower in error_message_lower or 
                error_record.error_type.lower() in pattern_lower)
    
    def _learn_from_failure(self, error_record: ErrorRecord) -> None:
        """Learn from failure patterns."""
        # Analyze the failure pattern
        pattern = self._extract_error_pattern(error_record)
        
        # Check if we already have this insight
        existing_insight = None
        for insight in self.learning_insights:
            if insight.error_pattern == pattern:
                existing_insight = insight
                break
        
        if existing_insight:
            # Update existing insight
            existing_insight.occurrence_count += 1
            existing_insight.last_seen = datetime.now()
            existing_insight.confidence = min(1.0, existing_insight.confidence + 0.1)
        else:
            # Create new insight
            insight = LearningInsight(
                insight_id=f"insight_{int(time.time())}_{pattern[:20]}",
                error_pattern=pattern,
                insight_type="failure_pattern",
                description=f"Pattern observed in {error_record.context.operation}",
                confidence=0.5,
                occurrence_count=1,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                suggested_action=self._generate_suggested_action(error_record)
            )
            self.learning_insights.append(insight)
        
        # Save insights
        self._save_learning_insights()
        
        logger.info(f"Learned from failure pattern: {pattern}")
    
    def _extract_error_pattern(self, error_record: ErrorRecord) -> str:
        """Extract error pattern for learning."""
        # Extract key components of the error
        components = [
            error_record.error_type,
            error_record.error_category.value,
            error_record.context.operation,
            error_record.context.component
        ]
        
        return "|".join(components)
    
    def _generate_suggested_action(self, error_record: ErrorRecord) -> str:
        """Generate suggested action based on error analysis."""
        if error_record.error_category == ErrorCategory.NETWORK:
            return "Check network connectivity and retry with exponential backoff"
        elif error_record.error_category == ErrorCategory.VALIDATION:
            return "Validate input data before processing"
        elif error_record.error_category == ErrorCategory.RESOURCE:
            return "Implement resource availability checks and fallback mechanisms"
        elif error_record.error_category == ErrorCategory.EXTERNAL_SERVICE:
            return "Implement service health checks and alternative endpoints"
        else:
            return "Review error logs and implement appropriate error handling"
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        if not self.error_records:
            return {
                "total_errors": 0,
                "resolved_errors": 0,
                "unresolved_errors": 0,
                "resolution_rate": 0,
                "category_counts": {},
                "severity_counts": {},
                "retry_attempts": 0,
                "successful_retries": 0,
                "retry_success_rate": 0,
                "learning_insights_count": len(self.learning_insights)
            }
        
        total_errors = len(self.error_records)
        resolved_errors = len([r for r in self.error_records if r.resolved])
        unresolved_errors = total_errors - resolved_errors
        
        # Count by category
        category_counts = {}
        for record in self.error_records:
            category = record.error_category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for record in self.error_records:
            severity = record.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Calculate retry success rate
        retry_attempts = len([r for r in self.error_records if r.retry_count > 0])
        successful_retries = len([r for r in self.error_records if r.resolved and r.retry_count > 0])
        retry_success_rate = successful_retries / retry_attempts if retry_attempts > 0 else 0
        
        return {
            "total_errors": total_errors,
            "resolved_errors": resolved_errors,
            "unresolved_errors": unresolved_errors,
            "resolution_rate": resolved_errors / total_errors if total_errors > 0 else 0,
            "category_counts": category_counts,
            "severity_counts": severity_counts,
            "retry_attempts": retry_attempts,
            "successful_retries": successful_retries,
            "retry_success_rate": retry_success_rate,
            "learning_insights_count": len(self.learning_insights)
        }
    
    def get_learning_insights(self) -> List[LearningInsight]:
        """Get all learning insights."""
        return self.learning_insights
    
    def get_error_records(self, limit: int = 100) -> List[ErrorRecord]:
        """Get recent error records."""
        return sorted(self.error_records, key=lambda x: x.context.timestamp, reverse=True)[:limit]


# Decorator for automatic error handling
def with_error_handling(error_handler: ErrorHandler, operation: str, component: str, **context_kwargs):
    """Decorator for automatic error handling."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation,
                component=component,
                **context_kwargs
            )
            
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                return await error_handler.handle_error(e, context, lambda: func(*args, **kwargs))
        
        def sync_wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation,
                component=component,
                **context_kwargs
            )
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, we need to handle this differently
                import asyncio
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(
                    error_handler.handle_error(e, context, lambda: func(*args, **kwargs))
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
