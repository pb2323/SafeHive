"""
Guard Manager - Configuration and Management System for Security Guards

This module implements a comprehensive guard management system that handles
configuration, enable/disable functionality, and agent integration for all
security guards in the SafeHive system.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Callable, Set
from abc import ABC, abstractmethod

from ..utils.logger import get_logger
from ..utils.metrics import record_metric, MetricType

logger = get_logger(__name__)


class GuardType(Enum):
    """Types of security guards available."""
    PRIVACY_SENTRY = "privacy_sentry"
    TASK_NAVIGATOR = "task_navigator"
    PROMPT_SANITIZER = "prompt_sanitizer"


class GuardStatus(Enum):
    """Status of guard instances."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class GuardPriority(Enum):
    """Priority levels for guard execution."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GuardConfiguration:
    """Configuration for a guard instance."""
    guard_id: str
    guard_type: GuardType
    name: str
    description: str
    enabled: bool = True
    priority: GuardPriority = GuardPriority.MEDIUM
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['guard_type'] = self.guard_type.value
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GuardConfiguration':
        """Create from dictionary."""
        data['guard_type'] = GuardType(data['guard_type'])
        data['priority'] = GuardPriority(data['priority'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class GuardInstance:
    """Instance of a guard with runtime information."""
    config: GuardConfiguration
    instance: Any  # The actual guard instance
    status: GuardStatus = GuardStatus.ENABLED
    last_execution: Optional[datetime] = None
    execution_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'config': self.config.to_dict(),
            'status': self.status.value,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'execution_count': self.execution_count,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'performance_metrics': self.performance_metrics
        }


class BaseGuard(ABC):
    """Base class for all security guards."""
    
    def __init__(self, config: GuardConfiguration):
        self.config = config
        self.guard_id = config.guard_id
        self.guard_type = config.guard_type
        self.name = config.name
        self.description = config.description
        self.enabled = config.enabled
        self.priority = config.priority
        self.logger = get_logger(f"guard.{self.guard_type.value}.{self.guard_id}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the guard. Return True if successful."""
        pass
    
    @abstractmethod
    def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Process input data through the guard. Return processed result."""
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """Cleanup guard resources. Return True if successful."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get guard status information."""
        return {
            'guard_id': self.guard_id,
            'guard_type': self.guard_type.value,
            'name': self.name,
            'enabled': self.enabled,
            'priority': self.priority.value,
            'config': self.config.config
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update guard configuration."""
        try:
            self.config.config.update(new_config)
            self.config.updated_at = datetime.now()
            return True
        except Exception as e:
            self.logger.error(f"Failed to update config: {e}")
            return False


class GuardRegistry:
    """Registry for managing guard types and factories."""
    
    def __init__(self):
        self._guard_types: Dict[GuardType, Type[BaseGuard]] = {}
        self._guard_factories: Dict[GuardType, Callable[[GuardConfiguration], BaseGuard]] = {}
    
    def register_guard_type(self, guard_type: GuardType, guard_class: Type[BaseGuard]) -> None:
        """Register a guard type with its class."""
        self._guard_types[guard_type] = guard_class
        self._guard_factories[guard_type] = lambda config: guard_class(config)
        logger.info(f"Registered guard type: {guard_type.value}")
    
    def register_guard_factory(self, guard_type: GuardType, factory: Callable[[GuardConfiguration], BaseGuard]) -> None:
        """Register a custom factory for a guard type."""
        self._guard_factories[guard_type] = factory
        logger.info(f"Registered custom factory for guard type: {guard_type.value}")
    
    def create_guard(self, config: GuardConfiguration) -> Optional[BaseGuard]:
        """Create a guard instance from configuration."""
        if config.guard_type not in self._guard_factories:
            logger.error(f"No factory registered for guard type: {config.guard_type.value}")
            return None
        
        try:
            factory = self._guard_factories[config.guard_type]
            guard = factory(config)
            
            if guard.initialize():
                logger.info(f"Created guard: {config.guard_id} ({config.guard_type.value})")
                return guard
            else:
                logger.error(f"Failed to initialize guard: {config.guard_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create guard {config.guard_id}: {e}")
            return None
    
    def get_available_guard_types(self) -> List[GuardType]:
        """Get list of available guard types."""
        return list(self._guard_types.keys())


class GuardManager:
    """Main manager for all security guards."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_guards"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.registry = GuardRegistry()
        self.guards: Dict[str, GuardInstance] = {}
        self.configurations: Dict[str, GuardConfiguration] = {}
        
        # Runtime state
        self.initialized = False
        self.enabled_guards: Set[str] = set()
        self.disabled_guards: Set[str] = set()
        
        # Statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        
        logger.info("Guard Manager initialized")
    
    def initialize(self) -> bool:
        """Initialize the guard manager."""
        try:
            # Register default guard types
            self._register_default_guards()
            
            # Load configurations
            self._load_configurations()
            
            # Create guard instances
            self._create_guard_instances()
            
            self.initialized = True
            logger.info("Guard Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Guard Manager: {e}")
            return False
    
    def _register_default_guards(self) -> None:
        """Register default guard types."""
        try:
            # Import guard classes
            from .privacy_sentry import PrivacySentry
            from .task_navigator import TaskNavigator
            from .prompt_sanitizer import PromptSanitizer
            
            # Create wrapper classes for base guard compatibility
            class PrivacySentryGuard(BaseGuard):
                def __init__(self, config: GuardConfiguration):
                    super().__init__(config)
                    self._guard = None
                
                def initialize(self) -> bool:
                    try:
                        self._guard = PrivacySentry(self.config.config.get('storage_path', '/tmp/safehive_privacy'))
                        return True
                    except Exception as e:
                        self.logger.error(f"Failed to initialize PrivacySentry: {e}")
                        return False
                
                def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
                    if not self._guard:
                        return None
                    
                    if isinstance(input_data, str):
                        return self._guard.process_message(input_data, context)
                    return None
                
                def cleanup(self) -> bool:
                    try:
                        if self._guard:
                            self._guard.cleanup_old_data()
                        return True
                    except Exception as e:
                        self.logger.error(f"Failed to cleanup PrivacySentry: {e}")
                        return False
            
            class TaskNavigatorGuard(BaseGuard):
                def __init__(self, config: GuardConfiguration):
                    super().__init__(config)
                    self._guard = None
                
                def initialize(self) -> bool:
                    try:
                        self._guard = TaskNavigator(self.config.config.get('storage_path', '/tmp/safehive_tasks'))
                        return True
                    except Exception as e:
                        self.logger.error(f"Failed to initialize TaskNavigator: {e}")
                        return False
                
                def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
                    if not self._guard:
                        return None
                    
                    if isinstance(input_data, dict) and 'action' in input_data:
                        return self._guard.navigate_action(
                            input_data['action'],
                            input_data.get('task_id'),
                            input_data.get('context', {})
                        )
                    return None
                
                def cleanup(self) -> bool:
                    return True  # TaskNavigator doesn't need cleanup
            
            class PromptSanitizerGuard(BaseGuard):
                def __init__(self, config: GuardConfiguration):
                    super().__init__(config)
                    self._guard = None
                
                def initialize(self) -> bool:
                    try:
                        self._guard = PromptSanitizer(self.config.config.get('storage_path', '/tmp/safehive_sanitizer'))
                        return True
                    except Exception as e:
                        self.logger.error(f"Failed to initialize PromptSanitizer: {e}")
                        return False
                
                def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
                    if not self._guard:
                        return None
                    
                    if isinstance(input_data, str):
                        return self._guard.sanitize_prompt(input_data, context)
                    return None
                
                def cleanup(self) -> bool:
                    try:
                        if self._guard:
                            self._guard.cleanup_old_data()
                        return True
                    except Exception as e:
                        self.logger.error(f"Failed to cleanup PromptSanitizer: {e}")
                        return False
            
            # Register guard types
            self.registry.register_guard_type(GuardType.PRIVACY_SENTRY, PrivacySentryGuard)
            self.registry.register_guard_type(GuardType.TASK_NAVIGATOR, TaskNavigatorGuard)
            self.registry.register_guard_type(GuardType.PROMPT_SANITIZER, PromptSanitizerGuard)
            
        except ImportError as e:
            logger.warning(f"Could not import some guard modules: {e}")
    
    def _load_configurations(self) -> None:
        """Load guard configurations from storage."""
        config_file = self.storage_path / "guard_configurations.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                for config_dict in config_data:
                    config = GuardConfiguration.from_dict(config_dict)
                    self.configurations[config.guard_id] = config
                    
                logger.info(f"Loaded {len(self.configurations)} guard configurations")
                
            except Exception as e:
                logger.error(f"Failed to load configurations: {e}")
        
        # Create default configurations if none exist
        if not self.configurations:
            self._create_default_configurations()
    
    def _create_default_configurations(self) -> None:
        """Create default guard configurations."""
        default_configs = [
            GuardConfiguration(
                guard_id="privacy_sentry_default",
                guard_type=GuardType.PRIVACY_SENTRY,
                name="Default Privacy Sentry",
                description="Default privacy protection guard",
                enabled=True,
                priority=GuardPriority.HIGH,
                config={
                    "storage_path": "/tmp/safehive_privacy",
                    "auto_mask": True,
                    "strict_mode": False
                }
            ),
            GuardConfiguration(
                guard_id="task_navigator_default",
                guard_type=GuardType.TASK_NAVIGATOR,
                name="Default Task Navigator",
                description="Default task constraint enforcement guard",
                enabled=True,
                priority=GuardPriority.HIGH,
                config={
                    "storage_path": "/tmp/safehive_tasks",
                    "strict_mode": True,
                    "auto_redirect": True
                }
            ),
            GuardConfiguration(
                guard_id="prompt_sanitizer_default",
                guard_type=GuardType.PROMPT_SANITIZER,
                name="Default Prompt Sanitizer",
                description="Default prompt sanitization guard",
                enabled=True,
                priority=GuardPriority.CRITICAL,
                config={
                    "storage_path": "/tmp/safehive_sanitizer",
                    "auto_block": True,
                    "strict_filtering": True
                }
            )
        ]
        
        for config in default_configs:
            self.configurations[config.guard_id] = config
        
        logger.info(f"Created {len(default_configs)} default guard configurations")
    
    def _create_guard_instances(self) -> None:
        """Create guard instances from configurations."""
        for config in self.configurations.values():
            if config.enabled:
                guard = self.registry.create_guard(config)
                if guard:
                    instance = GuardInstance(
                        config=config,
                        instance=guard,
                        status=GuardStatus.ENABLED
                    )
                    self.guards[config.guard_id] = instance
                    self.enabled_guards.add(config.guard_id)
                    logger.info(f"Created enabled guard: {config.guard_id}")
                else:
                    logger.error(f"Failed to create guard: {config.guard_id}")
            else:
                self.disabled_guards.add(config.guard_id)
                logger.info(f"Guard disabled: {config.guard_id}")
    
    def add_guard(self, config: GuardConfiguration) -> bool:
        """Add a new guard configuration."""
        try:
            self.configurations[config.guard_id] = config
            
            if config.enabled:
                guard = self.registry.create_guard(config)
                if guard:
                    instance = GuardInstance(
                        config=config,
                        instance=guard,
                        status=GuardStatus.ENABLED
                    )
                    self.guards[config.guard_id] = instance
                    self.enabled_guards.add(config.guard_id)
                    logger.info(f"Added enabled guard: {config.guard_id}")
                    return True
                else:
                    logger.error(f"Failed to create guard instance: {config.guard_id}")
                    return False
            else:
                self.disabled_guards.add(config.guard_id)
                logger.info(f"Added disabled guard: {config.guard_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add guard {config.guard_id}: {e}")
            return False
    
    def remove_guard(self, guard_id: str) -> bool:
        """Remove a guard configuration and instance."""
        try:
            if guard_id in self.guards:
                instance = self.guards[guard_id]
                instance.instance.cleanup()
                del self.guards[guard_id]
                self.enabled_guards.discard(guard_id)
            
            if guard_id in self.configurations:
                del self.configurations[guard_id]
            
            self.disabled_guards.discard(guard_id)
            
            logger.info(f"Removed guard: {guard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove guard {guard_id}: {e}")
            return False
    
    def enable_guard(self, guard_id: str) -> bool:
        """Enable a guard."""
        try:
            if guard_id not in self.configurations:
                logger.error(f"Guard configuration not found: {guard_id}")
                return False
            
            config = self.configurations[guard_id]
            config.enabled = True
            config.updated_at = datetime.now()
            
            if guard_id not in self.guards:
                guard = self.registry.create_guard(config)
                if guard:
                    instance = GuardInstance(
                        config=config,
                        instance=guard,
                        status=GuardStatus.ENABLED
                    )
                    self.guards[guard_id] = instance
                else:
                    logger.error(f"Failed to create guard instance: {guard_id}")
                    return False
            
            self.enabled_guards.add(guard_id)
            self.disabled_guards.discard(guard_id)
            
            logger.info(f"Enabled guard: {guard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable guard {guard_id}: {e}")
            return False
    
    def disable_guard(self, guard_id: str) -> bool:
        """Disable a guard."""
        try:
            if guard_id not in self.configurations:
                logger.error(f"Guard configuration not found: {guard_id}")
                return False
            
            config = self.configurations[guard_id]
            config.enabled = False
            config.updated_at = datetime.now()
            
            if guard_id in self.guards:
                instance = self.guards[guard_id]
                instance.instance.cleanup()
                instance.status = GuardStatus.DISABLED
                del self.guards[guard_id]
            
            self.enabled_guards.discard(guard_id)
            self.disabled_guards.add(guard_id)
            
            logger.info(f"Disabled guard: {guard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable guard {guard_id}: {e}")
            return False
    
    def process_through_guards(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input data through all enabled guards."""
        if not self.initialized:
            logger.error("Guard Manager not initialized")
            return {'error': 'Guard Manager not initialized'}
        
        results = {}
        
        # Sort guards by priority
        priority_order = {GuardPriority.LOW: 1, GuardPriority.MEDIUM: 2, 
                         GuardPriority.HIGH: 3, GuardPriority.CRITICAL: 4}
        
        sorted_guards = sorted(
            self.guards.values(),
            key=lambda g: priority_order[g.config.priority],
            reverse=True  # Critical guards first
        )
        
        for guard_instance in sorted_guards:
            if guard_instance.status != GuardStatus.ENABLED:
                continue
            
            try:
                start_time = time.time()
                
                result = guard_instance.instance.process(input_data, context)
                
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Update metrics
                guard_instance.last_execution = datetime.now()
                guard_instance.execution_count += 1
                guard_instance.performance_metrics['last_processing_time_ms'] = processing_time
                guard_instance.performance_metrics['avg_processing_time_ms'] = (
                    guard_instance.performance_metrics.get('avg_processing_time_ms', 0) * 
                    (guard_instance.execution_count - 1) + processing_time
                ) / guard_instance.execution_count
                
                results[guard_instance.config.guard_id] = {
                    'result': result,
                    'processing_time_ms': processing_time,
                    'guard_type': guard_instance.config.guard_type.value,
                    'priority': guard_instance.config.priority.value
                }
                
                self.total_executions += 1
                self.successful_executions += 1
                
                # Record metrics
                record_metric("guard_manager.guard_execution", 1, MetricType.COUNTER, {
                    "guard_id": guard_instance.config.guard_id,
                    "guard_type": guard_instance.config.guard_type.value,
                    "status": "success"
                })
                
            except Exception as e:
                guard_instance.error_count += 1
                guard_instance.last_error = str(e)
                guard_instance.status = GuardStatus.ERROR
                
                results[guard_instance.config.guard_id] = {
                    'error': str(e),
                    'guard_type': guard_instance.config.guard_type.value,
                    'priority': guard_instance.config.priority.value
                }
                
                self.total_executions += 1
                self.failed_executions += 1
                
                logger.error(f"Guard {guard_instance.config.guard_id} failed: {e}")
                
                # Record metrics
                record_metric("guard_manager.guard_execution", 1, MetricType.COUNTER, {
                    "guard_id": guard_instance.config.guard_id,
                    "guard_type": guard_instance.config.guard_type.value,
                    "status": "error"
                })
        
        return results
    
    def get_guard_status(self, guard_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of guards."""
        if guard_id:
            if guard_id in self.guards:
                return self.guards[guard_id].to_dict()
            elif guard_id in self.configurations:
                config = self.configurations[guard_id]
                return {
                    'config': config.to_dict(),
                    'status': GuardStatus.DISABLED.value,
                    'enabled': False
                }
            else:
                return {'error': f'Guard not found: {guard_id}'}
        
        # Return all guard statuses
        statuses = {}
        for guard_id, instance in self.guards.items():
            statuses[guard_id] = instance.to_dict()
        
        for guard_id in self.disabled_guards:
            if guard_id in self.configurations:
                config = self.configurations[guard_id]
                statuses[guard_id] = {
                    'config': config.to_dict(),
                    'status': GuardStatus.DISABLED.value,
                    'enabled': False
                }
        
        return statuses
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get guard manager statistics."""
        return {
            'total_guards': len(self.configurations),
            'enabled_guards': len(self.enabled_guards),
            'disabled_guards': len(self.disabled_guards),
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'failed_executions': self.failed_executions,
            'success_rate': (
                self.successful_executions / self.total_executions 
                if self.total_executions > 0 else 0
            ),
            'available_guard_types': [gt.value for gt in self.registry.get_available_guard_types()]
        }
    
    def save_configurations(self) -> bool:
        """Save guard configurations to storage."""
        try:
            config_file = self.storage_path / "guard_configurations.json"
            
            config_data = [config.to_dict() for config in self.configurations.values()]
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved {len(config_data)} guard configurations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configurations: {e}")
            return False
    
    def cleanup(self) -> bool:
        """Cleanup all guard resources."""
        try:
            for instance in self.guards.values():
                instance.instance.cleanup()
            
            self.save_configurations()
            
            logger.info("Guard Manager cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup Guard Manager: {e}")
            return False
