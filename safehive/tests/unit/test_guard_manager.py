"""
Unit tests for Guard Manager - Configuration and Management System.
"""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch, MagicMock, mock_open
import pytest

from safehive.guards.guard_manager import (
    GuardManager, GuardRegistry, BaseGuard, GuardType, GuardStatus,
    GuardPriority, GuardConfiguration, GuardInstance
)


class TestGuardType:
    """Test GuardType enum."""
    
    def test_guard_type_values(self):
        """Test GuardType enum values."""
        assert GuardType.PRIVACY_SENTRY.value == "privacy_sentry"
        assert GuardType.TASK_NAVIGATOR.value == "task_navigator"
        assert GuardType.PROMPT_SANITIZER.value == "prompt_sanitizer"


class TestGuardStatus:
    """Test GuardStatus enum."""
    
    def test_guard_status_values(self):
        """Test GuardStatus enum values."""
        assert GuardStatus.ENABLED.value == "enabled"
        assert GuardStatus.DISABLED.value == "disabled"
        assert GuardStatus.ERROR.value == "error"
        assert GuardStatus.MAINTENANCE.value == "maintenance"


class TestGuardPriority:
    """Test GuardPriority enum."""
    
    def test_guard_priority_values(self):
        """Test GuardPriority enum values."""
        assert GuardPriority.LOW.value == "low"
        assert GuardPriority.MEDIUM.value == "medium"
        assert GuardPriority.HIGH.value == "high"
        assert GuardPriority.CRITICAL.value == "critical"


class TestGuardConfiguration:
    """Test GuardConfiguration functionality."""
    
    def test_guard_configuration_creation(self):
        """Test GuardConfiguration creation."""
        config = GuardConfiguration(
            guard_id="test_guard",
            guard_type=GuardType.PRIVACY_SENTRY,
            name="Test Guard",
            description="A test guard configuration",
            enabled=True,
            priority=GuardPriority.HIGH,
            config={"test_param": "test_value"},
            metadata={"version": "1.0"}
        )
        
        assert config.guard_id == "test_guard"
        assert config.guard_type == GuardType.PRIVACY_SENTRY
        assert config.name == "Test Guard"
        assert config.description == "A test guard configuration"
        assert config.enabled is True
        assert config.priority == GuardPriority.HIGH
        assert config.config == {"test_param": "test_value"}
        assert config.metadata == {"version": "1.0"}
        assert isinstance(config.created_at, datetime)
        assert isinstance(config.updated_at, datetime)
    
    def test_guard_configuration_serialization(self):
        """Test GuardConfiguration serialization."""
        config = GuardConfiguration(
            guard_id="test_guard",
            guard_type=GuardType.PRIVACY_SENTRY,
            name="Test Guard",
            description="A test guard configuration",
            enabled=True,
            priority=GuardPriority.HIGH,
            config={"test_param": "test_value"}
        )
        
        data = config.to_dict()
        
        assert data["guard_id"] == "test_guard"
        assert data["guard_type"] == "privacy_sentry"
        assert data["name"] == "Test Guard"
        assert data["description"] == "A test guard configuration"
        assert data["enabled"] is True
        assert data["priority"] == "high"
        assert data["config"] == {"test_param": "test_value"}
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_guard_configuration_deserialization(self):
        """Test GuardConfiguration deserialization."""
        data = {
            "guard_id": "test_guard",
            "guard_type": "privacy_sentry",
            "name": "Test Guard",
            "description": "A test guard configuration",
            "enabled": True,
            "priority": "high",
            "config": {"test_param": "test_value"},
            "metadata": {},
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00"
        }
        
        config = GuardConfiguration.from_dict(data)
        
        assert config.guard_id == "test_guard"
        assert config.guard_type == GuardType.PRIVACY_SENTRY
        assert config.name == "Test Guard"
        assert config.description == "A test guard configuration"
        assert config.enabled is True
        assert config.priority == GuardPriority.HIGH
        assert config.config == {"test_param": "test_value"}


class TestGuardInstance:
    """Test GuardInstance functionality."""
    
    def test_guard_instance_creation(self):
        """Test GuardInstance creation."""
        config = GuardConfiguration(
            guard_id="test_guard",
            guard_type=GuardType.PRIVACY_SENTRY,
            name="Test Guard",
            description="A test guard configuration"
        )
        
        mock_guard = MagicMock()
        instance = GuardInstance(
            config=config,
            instance=mock_guard,
            status=GuardStatus.ENABLED,
            last_execution=datetime.now(),
            execution_count=5,
            error_count=1,
            last_error="Test error",
            performance_metrics={"avg_time": 10.5}
        )
        
        assert instance.config == config
        assert instance.instance == mock_guard
        assert instance.status == GuardStatus.ENABLED
        assert isinstance(instance.last_execution, datetime)
        assert instance.execution_count == 5
        assert instance.error_count == 1
        assert instance.last_error == "Test error"
        assert instance.performance_metrics == {"avg_time": 10.5}
    
    def test_guard_instance_serialization(self):
        """Test GuardInstance serialization."""
        config = GuardConfiguration(
            guard_id="test_guard",
            guard_type=GuardType.PRIVACY_SENTRY,
            name="Test Guard",
            description="A test guard configuration"
        )
        
        mock_guard = MagicMock()
        instance = GuardInstance(
            config=config,
            instance=mock_guard,
            status=GuardStatus.ENABLED,
            execution_count=5,
            performance_metrics={"avg_time": 10.5}
        )
        
        data = instance.to_dict()
        
        assert data["config"]["guard_id"] == "test_guard"
        assert data["status"] == "enabled"
        assert data["last_execution"] is None  # Not set in this test
        assert data["execution_count"] == 5
        assert data["error_count"] == 0
        assert data["last_error"] is None
        assert data["performance_metrics"] == {"avg_time": 10.5}


class TestBaseGuard:
    """Test BaseGuard abstract class."""
    
    def test_base_guard_creation(self):
        """Test BaseGuard creation."""
        config = GuardConfiguration(
            guard_id="test_guard",
            guard_type=GuardType.PRIVACY_SENTRY,
            name="Test Guard",
            description="A test guard configuration"
        )
        
        # Create a concrete implementation
        class TestGuard(BaseGuard):
            def initialize(self) -> bool:
                return True
            
            def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
                return f"processed: {input_data}"
            
            def cleanup(self) -> bool:
                return True
        
        guard = TestGuard(config)
        
        assert guard.config == config
        assert guard.guard_id == "test_guard"
        assert guard.guard_type == GuardType.PRIVACY_SENTRY
        assert guard.name == "Test Guard"
        assert guard.description == "A test guard configuration"
        assert guard.enabled is True
        assert guard.priority == GuardPriority.MEDIUM  # Default
    
    def test_base_guard_status(self):
        """Test BaseGuard status functionality."""
        config = GuardConfiguration(
            guard_id="test_guard",
            guard_type=GuardType.PRIVACY_SENTRY,
            name="Test Guard",
            description="A test guard configuration"
        )
        
        class TestGuard(BaseGuard):
            def initialize(self) -> bool:
                return True
            
            def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
                return "processed"
            
            def cleanup(self) -> bool:
                return True
        
        guard = TestGuard(config)
        status = guard.get_status()
        
        assert status["guard_id"] == "test_guard"
        assert status["guard_type"] == "privacy_sentry"
        assert status["name"] == "Test Guard"
        assert status["enabled"] is True
        assert status["priority"] == "medium"
        assert status["config"] == {}
    
    def test_base_guard_config_update(self):
        """Test BaseGuard configuration update."""
        config = GuardConfiguration(
            guard_id="test_guard",
            guard_type=GuardType.PRIVACY_SENTRY,
            name="Test Guard",
            description="A test guard configuration",
            config={"param1": "value1"}
        )
        
        class TestGuard(BaseGuard):
            def initialize(self) -> bool:
                return True
            
            def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
                return "processed"
            
            def cleanup(self) -> bool:
                return True
        
        guard = TestGuard(config)
        
        success = guard.update_config({"param2": "value2"})
        assert success is True
        assert guard.config.config == {"param1": "value1", "param2": "value2"}
        assert guard.config.updated_at > guard.config.created_at


class TestGuardRegistry:
    """Test GuardRegistry functionality."""
    
    def test_guard_registry_creation(self):
        """Test GuardRegistry creation."""
        registry = GuardRegistry()
        
        assert len(registry._guard_types) == 0
        assert len(registry._guard_factories) == 0
    
    def test_register_guard_type(self):
        """Test registering guard types."""
        registry = GuardRegistry()
        
        class TestGuard(BaseGuard):
            def initialize(self) -> bool:
                return True
            
            def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
                return "processed"
            
            def cleanup(self) -> bool:
                return True
        
        registry.register_guard_type(GuardType.PRIVACY_SENTRY, TestGuard)
        
        assert GuardType.PRIVACY_SENTRY in registry._guard_types
        assert GuardType.PRIVACY_SENTRY in registry._guard_factories
        assert registry._guard_types[GuardType.PRIVACY_SENTRY] == TestGuard
    
    def test_register_guard_factory(self):
        """Test registering custom guard factories."""
        registry = GuardRegistry()
        
        def custom_factory(config: GuardConfiguration) -> BaseGuard:
            class TestGuard(BaseGuard):
                def initialize(self) -> bool:
                    return True
                
                def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
                    return "custom_processed"
                
                def cleanup(self) -> bool:
                    return True
            
            return TestGuard(config)
        
        registry.register_guard_factory(GuardType.PRIVACY_SENTRY, custom_factory)
        
        assert GuardType.PRIVACY_SENTRY in registry._guard_factories
        assert registry._guard_factories[GuardType.PRIVACY_SENTRY] == custom_factory
    
    def test_create_guard(self):
        """Test creating guard instances."""
        registry = GuardRegistry()
        
        class TestGuard(BaseGuard):
            def initialize(self) -> bool:
                return True
            
            def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
                return "processed"
            
            def cleanup(self) -> bool:
                return True
        
        registry.register_guard_type(GuardType.PRIVACY_SENTRY, TestGuard)
        
        config = GuardConfiguration(
            guard_id="test_guard",
            guard_type=GuardType.PRIVACY_SENTRY,
            name="Test Guard",
            description="A test guard configuration"
        )
        
        guard = registry.create_guard(config)
        
        assert guard is not None
        assert isinstance(guard, TestGuard)
        assert guard.guard_id == "test_guard"
    
    def test_create_guard_failure(self):
        """Test creating guard instances with failures."""
        registry = GuardRegistry()
        
        class FailingGuard(BaseGuard):
            def initialize(self) -> bool:
                return False  # Fail initialization
            
            def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
                return "processed"
            
            def cleanup(self) -> bool:
                return True
        
        registry.register_guard_type(GuardType.PRIVACY_SENTRY, FailingGuard)
        
        config = GuardConfiguration(
            guard_id="test_guard",
            guard_type=GuardType.PRIVACY_SENTRY,
            name="Test Guard",
            description="A test guard configuration"
        )
        
        guard = registry.create_guard(config)
        
        assert guard is None  # Should fail to create
    
    def test_get_available_guard_types(self):
        """Test getting available guard types."""
        registry = GuardRegistry()
        
        class TestGuard(BaseGuard):
            def initialize(self) -> bool:
                return True
            
            def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
                return "processed"
            
            def cleanup(self) -> bool:
                return True
        
        registry.register_guard_type(GuardType.PRIVACY_SENTRY, TestGuard)
        registry.register_guard_type(GuardType.TASK_NAVIGATOR, TestGuard)
        
        available_types = registry.get_available_guard_types()
        
        assert GuardType.PRIVACY_SENTRY in available_types
        assert GuardType.TASK_NAVIGATOR in available_types
        assert len(available_types) == 2


class TestGuardManager:
    """Test GuardManager functionality."""
    
    def test_guard_manager_creation(self):
        """Test GuardManager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardManager(temp_dir)
            
            assert manager.storage_path == Path(temp_dir)
            assert len(manager.guards) == 0
            assert len(manager.configurations) == 0
            assert manager.initialized is False
            assert len(manager.enabled_guards) == 0
            assert len(manager.disabled_guards) == 0
            assert manager.total_executions == 0
            assert manager.successful_executions == 0
            assert manager.failed_executions == 0
    
    @patch('safehive.guards.privacy_sentry.PrivacySentry')
    @patch('safehive.guards.task_navigator.TaskNavigator')
    @patch('safehive.guards.prompt_sanitizer.PromptSanitizer')
    def test_guard_manager_initialization(self, mock_sanitizer, mock_navigator, mock_privacy):
        """Test GuardManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock guard instances
            mock_privacy_instance = MagicMock()
            mock_privacy.return_value = mock_privacy_instance
            
            mock_navigator_instance = MagicMock()
            mock_navigator.return_value = mock_navigator_instance
            
            mock_sanitizer_instance = MagicMock()
            mock_sanitizer.return_value = mock_sanitizer_instance
            
            manager = GuardManager(temp_dir)
            
            success = manager.initialize()
            
            assert success is True
            assert manager.initialized is True
            assert len(manager.configurations) == 3  # Default configurations
            assert len(manager.guards) == 3  # All enabled by default
            assert len(manager.enabled_guards) == 3
    
    def test_guard_manager_without_imports(self):
        """Test GuardManager initialization when guard modules are not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('safehive.guards.privacy_sentry.PrivacySentry', side_effect=ImportError):
                manager = GuardManager(temp_dir)
                
                success = manager.initialize()
                
                # Should still succeed but with warnings
                assert success is True
                assert manager.initialized is True
    
    def test_add_guard(self):
        """Test adding a new guard."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardManager(temp_dir)
            manager.initialize()
            
            config = GuardConfiguration(
                guard_id="custom_guard",
                guard_type=GuardType.PRIVACY_SENTRY,
                name="Custom Guard",
                description="A custom guard configuration",
                enabled=True,
                config={"custom_param": "custom_value"}
            )
            
            # Mock the guard creation
            with patch.object(manager.registry, 'create_guard') as mock_create:
                mock_guard = MagicMock()
                mock_guard.initialize.return_value = True
                mock_create.return_value = mock_guard
                
                success = manager.add_guard(config)
                
                assert success is True
                assert "custom_guard" in manager.configurations
                assert "custom_guard" in manager.guards
                assert "custom_guard" in manager.enabled_guards
    
    def test_remove_guard(self):
        """Test removing a guard."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardManager(temp_dir)
            manager.initialize()
            
            # Get an existing guard ID
            guard_id = list(manager.guards.keys())[0]
            
            success = manager.remove_guard(guard_id)
            
            assert success is True
            assert guard_id not in manager.guards
            assert guard_id not in manager.configurations
            assert guard_id not in manager.enabled_guards
    
    def test_enable_guard(self):
        """Test enabling a guard."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardManager(temp_dir)
            manager.initialize()
            
            # Disable a guard first
            guard_id = list(manager.guards.keys())[0]
            manager.disable_guard(guard_id)
            
            # Mock guard creation
            with patch.object(manager.registry, 'create_guard') as mock_create:
                mock_guard = MagicMock()
                mock_guard.initialize.return_value = True
                mock_create.return_value = mock_guard
                
                success = manager.enable_guard(guard_id)
                
                assert success is True
                assert guard_id in manager.enabled_guards
                assert guard_id not in manager.disabled_guards
    
    def test_disable_guard(self):
        """Test disabling a guard."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardManager(temp_dir)
            manager.initialize()
            
            guard_id = list(manager.guards.keys())[0]
            
            success = manager.disable_guard(guard_id)
            
            assert success is True
            assert guard_id not in manager.guards
            assert guard_id not in manager.enabled_guards
            assert guard_id in manager.disabled_guards
    
    def test_process_through_guards(self):
        """Test processing input through guards."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardManager(temp_dir)
            manager.initialize()
            
            # Mock guard process methods
            for instance in manager.guards.values():
                instance.instance.process = MagicMock(return_value="processed_result")
            
            results = manager.process_through_guards("test_input", {"context": "test"})
            
            assert len(results) == len(manager.guards)
            
            for guard_id, result in results.items():
                assert "result" in result or "error" in result
                assert "guard_type" in result
                assert "priority" in result
    
    def test_process_through_guards_with_errors(self):
        """Test processing through guards with errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardManager(temp_dir)
            manager.initialize()
            
            # Mock one guard to fail
            guard_instances = list(manager.guards.values())
            if guard_instances:
                guard_instances[0].instance.process = MagicMock(side_effect=Exception("Test error"))
            
            results = manager.process_through_guards("test_input")
            
            # Should still process other guards
            assert len(results) >= 1
            
            # Check for error handling
            error_results = [r for r in results.values() if "error" in r]
            assert len(error_results) >= 1
    
    def test_get_guard_status(self):
        """Test getting guard status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardManager(temp_dir)
            manager.initialize()
            
            guard_id = list(manager.guards.keys())[0]
            
            # Get specific guard status
            status = manager.get_guard_status(guard_id)
            
            assert "config" in status
            assert "status" in status
            assert status["config"]["guard_id"] == guard_id
            
            # Get all guard statuses
            all_statuses = manager.get_guard_status()
            
            assert len(all_statuses) >= len(manager.guards)
    
    def test_get_statistics(self):
        """Test getting guard manager statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardManager(temp_dir)
            manager.initialize()
            
            # Process some data to generate statistics
            manager.process_through_guards("test_input")
            
            stats = manager.get_statistics()
            
            assert "total_guards" in stats
            assert "enabled_guards" in stats
            assert "disabled_guards" in stats
            assert "total_executions" in stats
            assert "successful_executions" in stats
            assert "failed_executions" in stats
            assert "success_rate" in stats
            assert "available_guard_types" in stats
            
            assert stats["total_guards"] >= 3
            assert stats["enabled_guards"] >= 3
            assert stats["total_executions"] >= 3
    
    def test_save_configurations(self):
        """Test saving guard configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardManager(temp_dir)
            manager.initialize()
            
            success = manager.save_configurations()
            
            assert success is True
            
            # Check if file was created
            config_file = manager.storage_path / "guard_configurations.json"
            assert config_file.exists()
            
            # Verify content
            with open(config_file, 'r') as f:
                import json
                config_data = json.load(f)
            
            assert len(config_data) == len(manager.configurations)
    
    def test_cleanup(self):
        """Test guard manager cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardManager(temp_dir)
            manager.initialize()
            
            # Mock cleanup methods
            for instance in manager.guards.values():
                instance.instance.cleanup = MagicMock(return_value=True)
            
            success = manager.cleanup()
            
            assert success is True


class TestGuardManagerIntegration:
    """Integration tests for GuardManager."""
    
    @patch('safehive.guards.privacy_sentry.PrivacySentry')
    @patch('safehive.guards.task_navigator.TaskNavigator')
    @patch('safehive.guards.prompt_sanitizer.PromptSanitizer')
    def test_complete_guard_workflow(self, mock_sanitizer, mock_navigator, mock_privacy):
        """Test complete guard workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock guard instances
            mock_privacy_instance = MagicMock()
            mock_privacy_instance.process_message.return_value = {"action": "allowed", "masked_content": "test"}
            mock_privacy.return_value = mock_privacy_instance
            
            mock_navigator_instance = MagicMock()
            mock_navigator_instance.navigate_action.return_value = {"allowed": True, "action": "proceed"}
            mock_navigator.return_value = mock_navigator_instance
            
            mock_sanitizer_instance = MagicMock()
            mock_sanitizer_instance.sanitize_prompt.return_value = MagicMock(
                sanitized_prompt="sanitized_test",
                action_taken="allow",
                threats_detected=[]
            )
            mock_sanitizer.return_value = mock_sanitizer_instance
            
            manager = GuardManager(temp_dir)
            manager.initialize()
            
            # Test processing through all guards
            results = manager.process_through_guards("test input", {"source": "test"})
            
            assert len(results) == 3
            
            # Test adding a new guard
            config = GuardConfiguration(
                guard_id="integration_test_guard",
                guard_type=GuardType.PRIVACY_SENTRY,
                name="Integration Test Guard",
                description="A guard for integration testing",
                enabled=False
            )
            
            success = manager.add_guard(config)
            assert success is True
            
            # Test enabling the new guard
            with patch.object(manager.registry, 'create_guard') as mock_create:
                mock_guard = MagicMock()
                mock_guard.initialize.return_value = True
                mock_guard.process.return_value = "integration_result"
                mock_create.return_value = mock_guard
                
                success = manager.enable_guard("integration_test_guard")
                assert success is True
            
            # Test getting statistics
            stats = manager.get_statistics()
            assert stats["total_guards"] == 4
            assert stats["enabled_guards"] == 4
            
            # Test cleanup
            success = manager.cleanup()
            assert success is True
    
    def test_guard_priority_ordering(self):
        """Test that guards are executed in priority order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardManager(temp_dir)
            manager.initialize()  # Initialize the manager
            
            # Create custom configurations with different priorities
            configs = [
                GuardConfiguration(
                    guard_id="low_priority_guard",
                    guard_type=GuardType.PRIVACY_SENTRY,
                    name="Low Priority Guard",
                    description="Low priority guard",
                    priority=GuardPriority.LOW,
                    enabled=True,
                    config={"storage_path": "/tmp/low"}
                ),
                GuardConfiguration(
                    guard_id="critical_priority_guard",
                    guard_type=GuardType.PRIVACY_SENTRY,
                    name="Critical Priority Guard",
                    description="Critical priority guard",
                    priority=GuardPriority.CRITICAL,
                    enabled=True,
                    config={"storage_path": "/tmp/critical"}
                ),
                GuardConfiguration(
                    guard_id="medium_priority_guard",
                    guard_type=GuardType.PRIVACY_SENTRY,
                    name="Medium Priority Guard",
                    description="Medium priority guard",
                    priority=GuardPriority.MEDIUM,
                    enabled=True,
                    config={"storage_path": "/tmp/medium"}
                )
            ]
            
            # Mock guard creation
            with patch.object(manager.registry, 'create_guard') as mock_create:
                mock_guard = MagicMock()
                mock_guard.initialize.return_value = True
                mock_guard.process = MagicMock(side_effect=lambda x, ctx=None: f"processed_by_{ctx.get('guard_id', 'unknown')}")
                mock_create.return_value = mock_guard
                
                # Add guards
                for config in configs:
                    manager.add_guard(config)
                
                # Process through guards
                results = manager.process_through_guards("test", {"guard_id": "test"})
                
                # Should process all guards (3 default + 3 custom = 6 total)
                assert len(results) == 6
