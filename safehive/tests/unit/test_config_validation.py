"""
Unit tests for configuration validation system.

This module tests the configuration validation, error handling, and tools functionality.
"""

import pytest
import tempfile
import yaml
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from typing import Dict, Any

from safehive.config.config_validator import (
    ConfigurationValidator, ValidationResult, ValidationIssue, ValidationSeverity,
    get_config_validator, validate_config_file, validate_config_data
)
from safehive.config.config_error_handler import (
    ConfigurationErrorHandler, ConfigurationError, ConfigurationErrorType,
    get_config_error_handler, handle_configuration_error
)
from safehive.config.config_tools import (
    ConfigurationTools, ConfigAnalysisResult, get_config_tools,
    display_analysis_result, display_comparison_result
)


class TestValidationIssue:
    """Test ValidationIssue dataclass."""
    
    def test_validation_issue_creation(self):
        """Test creating a ValidationIssue."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field_path="test.field",
            message="Test validation issue",
            current_value="invalid_value",
            suggested_value="valid_value",
            fix_available=True
        )
        
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.field_path == "test.field"
        assert issue.message == "Test validation issue"
        assert issue.current_value == "invalid_value"
        assert issue.suggested_value == "valid_value"
        assert issue.fix_available is True
    
    def test_validation_issue_str(self):
        """Test ValidationIssue string representation."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            field_path="test.field",
            message="Test warning"
        )
        
        str_repr = str(issue)
        assert "⚠️" in str_repr
        assert "test.field" in str_repr
        assert "Test warning" in str_repr


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(is_valid=True)
        
        assert result.is_valid is True
        assert len(result.issues) == 0
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert len(result.fixes_applied) == 0
    
    def test_add_issue(self):
        """Test adding issues to ValidationResult."""
        result = ValidationResult(is_valid=True)
        
        error_issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field_path="test.error",
            message="Test error"
        )
        
        warning_issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            field_path="test.warning",
            message="Test warning"
        )
        
        result.add_issue(error_issue)
        result.add_issue(warning_issue)
        
        assert len(result.issues) == 2
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert result.is_valid is False  # Should be False due to error
    
    def test_get_summary(self):
        """Test getting validation summary."""
        result = ValidationResult(is_valid=False)
        
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field_path="test.error",
            message="Test error"
        ))
        
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            field_path="test.warning",
            message="Test warning"
        ))
        
        result.fixes_applied = ["fix1", "fix2"]
        
        summary = result.get_summary()
        
        assert summary["total_issues"] == 2
        assert summary["errors"] == 1
        assert summary["warnings"] == 1
        assert summary["info"] == 0
        assert summary["fixes_applied"] == 2


class TestConfigurationValidator:
    """Test ConfigurationValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConfigurationValidator()
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        assert len(self.validator.required_sections) > 0
        assert len(self.validator.valid_log_levels) > 0
        assert len(self.validator.valid_metric_types) > 0
        assert len(self.validator.valid_memory_types) > 0
    
    def test_validate_configuration_valid(self):
        """Test validation of valid configuration."""
        valid_config = {
            "guards": {
                "privacy_sentry": {"enabled": True, "threshold": 3},
                "task_navigator": {"enabled": True, "threshold": 3},
                "prompt_sanitizer": {"enabled": True, "threshold": 3}
            },
            "agents": {
                "orchestrator": {"ai_model": "llama2:7b", "timeout_seconds": 30},
                "user_twin": {"ai_model": "llama2:7b", "timeout_seconds": 30}
            },
            "logging": {"level": "INFO", "file": "logs/sandbox.log"},
            "metrics": {"enabled": True, "retention_hours": 24},
            "notifications": {"enabled": True}
        }
        
        result = self.validator.validate_configuration(valid_config)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_configuration_missing_sections(self):
        """Test validation with missing required sections."""
        invalid_config = {
            "guards": {"privacy_sentry": {"enabled": True}}
        }
        
        result = self.validator.validate_configuration(invalid_config)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        
        # Check for missing sections error
        missing_sections = [issue.field_path for issue in result.errors 
                          if "missing" in issue.message.lower()]
        assert len(missing_sections) > 0
    
    def test_validate_guards_section(self):
        """Test guards section validation."""
        guards_config = {
            "privacy_sentry": {"enabled": True, "threshold": 3},
            "invalid_guard": {"enabled": True}  # Invalid guard type
        }
        
        result = ValidationResult(is_valid=True)
        self.validator._validate_guards_section(guards_config, result)
        
        # Should have warning for invalid guard
        warnings = [issue for issue in result.issues 
                   if issue.severity == ValidationSeverity.WARNING]
        assert len(warnings) > 0
    
    def test_validate_agents_section(self):
        """Test agents section validation."""
        agents_config = {
            "orchestrator": {
                "ai_model": "llama2:7b",
                "timeout_seconds": 30,
                "memory_type": "conversation_buffer"
            },
            "invalid_agent": {"ai_model": "test"}  # Invalid agent type
        }
        
        result = ValidationResult(is_valid=True)
        self.validator._validate_agents_section(agents_config, result)
        
        # Should have warning for invalid agent
        warnings = [issue for issue in result.issues 
                   if issue.severity == ValidationSeverity.WARNING]
        assert len(warnings) > 0
    
    def test_validate_logging_section(self):
        """Test logging section validation."""
        logging_config = {
            "level": "INVALID_LEVEL",  # Invalid log level
            "file": "logs/test.log"
        }
        
        result = ValidationResult(is_valid=True)
        self.validator._validate_logging_section(logging_config, result)
        
        # Should have error for invalid log level
        errors = [issue for issue in result.issues 
                 if issue.severity == ValidationSeverity.ERROR]
        assert len(errors) > 0
    
    def test_fix_configuration(self):
        """Test automatic configuration fixing."""
        config_data = {
            "guards": {
                "privacy_sentry": {"enabled": "invalid_bool", "threshold": -1}
            }
        }
        
        result = ValidationResult(is_valid=False)
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field_path="guards.privacy_sentry.enabled",
            message="Invalid boolean value",
            current_value="invalid_bool",
            suggested_value=True,
            fix_available=True
        ))
        
        fixed_config = self.validator.fix_configuration(config_data, result)
        
        assert fixed_config["guards"]["privacy_sentry"]["enabled"] is True
        assert len(result.fixes_applied) > 0
    
    def test_set_nested_value(self):
        """Test setting nested dictionary values."""
        config = {"level1": {"level2": {"level3": "old_value"}}}
        
        self.validator._set_nested_value(config, ["level1", "level2", "level3"], "new_value")
        
        assert config["level1"]["level2"]["level3"] == "new_value"


class TestConfigurationErrorHandler:
    """Test ConfigurationErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ConfigurationErrorHandler()
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        assert len(self.error_handler.fallback_configs) > 0
        assert "minimal" in self.error_handler.fallback_configs
        assert "safe" in self.error_handler.fallback_configs
    
    def test_handle_file_not_found(self):
        """Test handling file not found errors."""
        file_path = "/nonexistent/config.yaml"
        
        fallback_config, errors = self.error_handler.handle_file_not_found(file_path)
        
        assert fallback_config is not None
        assert len(errors) > 0
        assert any(error.error_type == ConfigurationErrorType.FILE_NOT_FOUND for error in errors)
    
    def test_handle_invalid_yaml(self):
        """Test handling invalid YAML errors."""
        file_path = "/test/invalid.yaml"
        yaml_error = yaml.YAMLError("Invalid YAML syntax")
        
        fallback_config, errors = self.error_handler.handle_invalid_yaml(file_path, yaml_error)
        
        assert fallback_config is not None
        assert len(errors) > 0
        assert any(error.error_type == ConfigurationErrorType.INVALID_YAML for error in errors)
    
    def test_handle_validation_errors(self):
        """Test handling validation errors."""
        file_path = "/test/config.yaml"
        
        # Create validation result with errors
        validation_result = ValidationResult(is_valid=False)
        validation_result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field_path="test.field",
            message="Test validation error"
        ))
        
        fallback_config, errors = self.error_handler.handle_validation_errors(file_path, validation_result)
        
        # Should return None for config (indicating config is usable) and convert issues to errors
        assert fallback_config is None
        assert len(errors) > 0
    
    def test_handle_permission_error(self):
        """Test handling permission errors."""
        file_path = "/protected/config.yaml"
        permission_error = PermissionError("Permission denied")
        
        fallback_config, errors = self.error_handler.handle_permission_error(file_path, permission_error)
        
        assert fallback_config is not None
        assert len(errors) > 0
        assert any(error.error_type == ConfigurationErrorType.PERMISSION_ERROR for error in errors)
    
    def test_create_recovery_report(self):
        """Test creating recovery report."""
        errors = [
            ConfigurationError(
                error_type=ConfigurationErrorType.FILE_NOT_FOUND,
                message="File not found",
                suggested_fix="Create file"
            ),
            ConfigurationError(
                error_type=ConfigurationErrorType.VALIDATION_ERROR,
                message="Validation failed",
                severity="warning"
            )
        ]
        
        report = self.error_handler.create_recovery_report(errors)
        
        assert "Configuration Recovery Report" in report
        assert "File not found" in report
        assert "Validation failed" in report
        assert "Total Issues: 2" in report


class TestConfigurationTools:
    """Test ConfigurationTools class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tools = ConfigurationTools()
    
    def test_tools_initialization(self):
        """Test tools initialization."""
        assert self.tools.validator is not None
        assert self.tools.error_handler is not None
    
    def test_analyze_configuration(self):
        """Test configuration analysis."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                "guards": {"privacy_sentry": {"enabled": True}},
                "agents": {"orchestrator": {"ai_model": "llama2:7b"}},
                "logging": {"level": "INFO"},
                "metrics": {"enabled": True},
                "notifications": {"enabled": True}
            }
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            result = self.tools.analyze_configuration(temp_path)
            
            assert result.file_path == temp_path
            assert isinstance(result.is_valid, bool)
            assert isinstance(result.validation_result, ValidationResult)
            assert isinstance(result.errors, list)
            assert isinstance(result.suggestions, list)
        finally:
            Path(temp_path).unlink()
    
    def test_fix_configuration(self):
        """Test configuration fixing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                "guards": {"privacy_sentry": {"enabled": True}},
                "agents": {"orchestrator": {"ai_model": "llama2:7b"}},
                "logging": {"level": "INVALID_LEVEL"},  # Invalid level
                "metrics": {"enabled": True},
                "notifications": {"enabled": True}
            }
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            success = self.tools.fix_configuration(temp_path, backup=False)
            
            # Should succeed in fixing the invalid log level
            assert success is True
            
            # Verify the file was modified
            with open(temp_path, 'r') as f:
                fixed_config = yaml.safe_load(f)
            
            assert fixed_config["logging"]["level"] == "INFO"  # Should be fixed to valid level
        finally:
            Path(temp_path).unlink()
    
    def test_create_configuration_template(self):
        """Test creating configuration templates."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            success = self.tools.create_configuration_template(temp_path, "minimal")
            
            assert success is True
            
            # Verify template was created
            with open(temp_path, 'r') as f:
                template_config = yaml.safe_load(f)
            
            assert "guards" in template_config
            assert "agents" in template_config
            assert "logging" in template_config
        finally:
            Path(temp_path).unlink()
    
    def test_compare_configurations(self):
        """Test configuration comparison."""
        config1 = {
            "guards": {"privacy_sentry": {"enabled": True}},
            "logging": {"level": "INFO"}
        }
        
        config2 = {
            "guards": {"privacy_sentry": {"enabled": False}},  # Different value
            "logging": {"level": "INFO"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f1, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
            
            yaml.dump(config1, f1)
            yaml.dump(config2, f2)
            temp_path1 = f1.name
            temp_path2 = f2.name
        
        try:
            comparison = self.tools.compare_configurations(temp_path1, temp_path2)
            
            assert not comparison["identical"]
            assert len(comparison["differences"]) > 0
            
            # Check that the difference is found
            differences = comparison["differences"]
            privacy_sentry_diff = [d for d in differences if "privacy_sentry" in d["path"]]
            assert len(privacy_sentry_diff) > 0
        finally:
            Path(temp_path1).unlink()
            Path(temp_path2).unlink()


class TestGlobalFunctions:
    """Test global helper functions."""
    
    def test_get_config_validator(self):
        """Test getting global validator instance."""
        validator1 = get_config_validator()
        validator2 = get_config_validator()
        
        assert validator1 is validator2
        assert isinstance(validator1, ConfigurationValidator)
    
    def test_get_config_error_handler(self):
        """Test getting global error handler instance."""
        handler1 = get_config_error_handler()
        handler2 = get_config_error_handler()
        
        assert handler1 is handler2
        assert isinstance(handler1, ConfigurationErrorHandler)
    
    def test_get_config_tools(self):
        """Test getting global tools instance."""
        tools1 = get_config_tools()
        tools2 = get_config_tools()
        
        assert tools1 is tools2
        assert isinstance(tools1, ConfigurationTools)
    
    def test_validate_config_data(self):
        """Test validating configuration data."""
        config_data = {
            "guards": {"privacy_sentry": {"enabled": True}},
            "agents": {"orchestrator": {"ai_model": "llama2:7b"}},
            "logging": {"level": "INFO"},
            "metrics": {"enabled": True},
            "notifications": {"enabled": True}
        }
        
        result = validate_config_data(config_data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
    
    def test_handle_configuration_error(self):
        """Test handling configuration errors."""
        file_path = "/test/config.yaml"
        error = FileNotFoundError("File not found")
        
        fallback_config, errors = handle_configuration_error(file_path, error)
        
        assert fallback_config is not None
        assert len(errors) > 0
        assert any(e.error_type == ConfigurationErrorType.FILE_NOT_FOUND for e in errors)


class TestIntegration:
    """Integration tests for configuration validation system."""
    
    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        # Create a test configuration with issues
        config_data = {
            "guards": {
                "privacy_sentry": {"enabled": True, "threshold": 3},
                "invalid_guard": {"enabled": True}  # Invalid guard
            },
            "agents": {
                "orchestrator": {"ai_model": "llama2:7b", "timeout_seconds": -1},  # Invalid timeout
                "invalid_agent": {"ai_model": "test"}  # Invalid agent
            },
            "logging": {"level": "INVALID_LEVEL"},  # Invalid log level
            "metrics": {"enabled": True},
            "notifications": {"enabled": True}
        }
        
        # Validate configuration
        validator = get_config_validator()
        result = validator.validate_configuration(config_data)
        
        # Should have errors and warnings
        assert not result.is_valid
        assert len(result.errors) > 0
        assert len(result.warnings) > 0
        
        # Try to fix configuration
        fixed_config = validator.fix_configuration(config_data, result)
        
        # Verify some fixes were applied
        assert len(result.fixes_applied) > 0
        
        # Validate fixed configuration
        fixed_result = validator.validate_configuration(fixed_config)
        
        # Should be better (fewer errors)
        assert len(fixed_result.errors) <= len(result.errors)
    
    def test_error_handling_workflow(self):
        """Test complete error handling workflow."""
        error_handler = get_config_error_handler()
        
        # Test file not found
        fallback_config, errors = error_handler.handle_file_not_found("/nonexistent/config.yaml")
        assert fallback_config is not None
        assert len(errors) > 0
        
        # Test invalid YAML
        yaml_error = yaml.YAMLError("Invalid YAML")
        fallback_config, errors = error_handler.handle_invalid_yaml("/test/invalid.yaml", yaml_error)
        assert fallback_config is not None
        assert len(errors) > 0
        
        # Create recovery report
        report = error_handler.create_recovery_report(errors)
        assert "Configuration Recovery Report" in report
    
    def test_tools_workflow(self):
        """Test complete tools workflow."""
        tools = get_config_tools()
        
        # Create temporary configuration files
        config1 = {
            "guards": {"privacy_sentry": {"enabled": True}},
            "agents": {"orchestrator": {"ai_model": "llama2:7b"}},
            "logging": {"level": "INFO"},
            "metrics": {"enabled": True},
            "notifications": {"enabled": True}
        }
        
        config2 = {
            "guards": {"privacy_sentry": {"enabled": False}},  # Different
            "agents": {"orchestrator": {"ai_model": "llama2:7b"}},
            "logging": {"level": "INFO"},
            "metrics": {"enabled": True},
            "notifications": {"enabled": True}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f1, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
            
            yaml.dump(config1, f1)
            yaml.dump(config2, f2)
            temp_path1 = f1.name
            temp_path2 = f2.name
        
        try:
            # Analyze configuration
            analysis = tools.analyze_configuration(temp_path1)
            assert analysis.file_path == temp_path1
            
            # Compare configurations
            comparison = tools.compare_configurations(temp_path1, temp_path2)
            assert not comparison["identical"]
            
            # Create template
            template_path = temp_path1 + ".template"
            success = tools.create_configuration_template(template_path, "minimal")
            assert success is True
            
            # Clean up template
            Path(template_path).unlink()
        finally:
            Path(temp_path1).unlink()
            Path(temp_path2).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
