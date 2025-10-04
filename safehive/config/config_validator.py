"""
Configuration Validation System for SafeHive AI Security Sandbox.

This module provides comprehensive validation for configuration files,
including schema validation, value validation, and error reporting.
"""

import os
import yaml
import re
from typing import Dict, Any, List, Optional, Union, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .config_loader import GuardConfig, AgentConfig, LoggingConfig, SystemConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a configuration validation issue."""
    severity: ValidationSeverity
    field_path: str
    message: str
    current_value: Any = None
    suggested_value: Any = None
    fix_available: bool = False
    
    def __str__(self) -> str:
        severity_icon = {
            ValidationSeverity.ERROR: "❌",
            ValidationSeverity.WARNING: "⚠️",
            ValidationSeverity.INFO: "ℹ️"
        }
        return f"{severity_icon[self.severity]} {self.field_path}: {self.message}"


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.now)
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue to the result."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.errors.append(issue)
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)
        
        # Update validity based on errors
        if issue.severity == ValidationSeverity.ERROR:
            self.is_valid = False
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of validation results."""
        return {
            "total_issues": len(self.issues),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "info": len(self.issues) - len(self.errors) - len(self.warnings),
            "fixes_applied": len(self.fixes_applied)
        }


class ConfigurationValidator:
    """Comprehensive configuration validator."""
    
    def __init__(self):
        self.required_sections = {
            "guards", "agents", "logging", "metrics", "notifications"
        }
        
        self.valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        self.valid_metric_types = {"counter", "gauge", "timer", "event"}
        self.valid_memory_types = {
            "conversation_buffer", "conversation_window", 
            "conversation_summary", "vector", "combined"
        }
        self.valid_agent_types = {
            "orchestrator", "user_twin", "honest_vendor", "malicious_vendor"
        }
        self.valid_guard_types = {
            "privacy_sentry", "task_navigator", "prompt_sanitizer", "mcp_server"
        }
        
        # File path patterns
        self.file_path_pattern = re.compile(r'^[a-zA-Z0-9_/.-]+$')
        self.url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    def validate_configuration(self, config_data: Dict[str, Any], 
                             config_path: Optional[str] = None) -> ValidationResult:
        """Validate a complete configuration."""
        result = ValidationResult(is_valid=True)
        
        logger.info(f"Starting configuration validation for: {config_path or 'in-memory config'}")
        
        # Top-level validation
        self._validate_top_level_structure(config_data, result)
        
        if not result.is_valid:
            return result
        
        # Section-specific validation
        if "guards" in config_data:
            self._validate_guards_section(config_data["guards"], result)
        
        if "agents" in config_data:
            self._validate_agents_section(config_data["agents"], result)
        
        if "logging" in config_data:
            self._validate_logging_section(config_data["logging"], result)
        
        if "metrics" in config_data:
            self._validate_metrics_section(config_data["metrics"], result)
        
        if "notifications" in config_data:
            self._validate_notifications_section(config_data["notifications"], result)
        
        if "mcp_server" in config_data:
            self._validate_mcp_server_section(config_data["mcp_server"], result)
        
        # Cross-section validation
        self._validate_cross_section_dependencies(config_data, result)
        
        logger.info(f"Configuration validation completed: {result.get_summary()}")
        return result
    
    def _validate_top_level_structure(self, config_data: Dict[str, Any], 
                                    result: ValidationResult):
        """Validate top-level configuration structure."""
        # Check for required sections
        missing_sections = self.required_sections - set(config_data.keys())
        for section in missing_sections:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field_path=f"root.{section}",
                message=f"Required section '{section}' is missing",
                suggested_value="Add the missing section with appropriate configuration"
            ))
        
        # Check for unknown sections
        known_sections = self.required_sections | {"mcp_server", "version", "metadata"}
        unknown_sections = set(config_data.keys()) - known_sections
        for section in unknown_sections:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field_path=f"root.{section}",
                message=f"Unknown section '{section}' - this may be ignored",
                current_value=config_data[section]
            ))
    
    def _validate_guards_section(self, guards_config: Dict[str, Any], 
                               result: ValidationResult):
        """Validate guards configuration section."""
        if not isinstance(guards_config, dict):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field_path="guards",
                message="Guards section must be a dictionary",
                current_value=type(guards_config).__name__
            ))
            return
        
        for guard_name, guard_config in guards_config.items():
            self._validate_individual_guard(guard_name, guard_config, result)
    
    def _validate_individual_guard(self, guard_name: str, guard_config: Any,
                                 result: ValidationResult):
        """Validate individual guard configuration."""
        field_prefix = f"guards.{guard_name}"
        
        if not isinstance(guard_config, dict):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field_path=field_prefix,
                message=f"Guard configuration must be a dictionary",
                current_value=type(guard_config).__name__
            ))
            return
        
        # Validate guard name
        if guard_name not in self.valid_guard_types:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field_path=field_prefix,
                message=f"Unknown guard type '{guard_name}'",
                current_value=guard_name,
                suggested_value="Use one of: " + ", ".join(self.valid_guard_types)
            ))
        
        # Validate enabled field
        if "enabled" in guard_config:
            if not isinstance(guard_config["enabled"], bool):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path=f"{field_prefix}.enabled",
                    message="Enabled field must be a boolean",
                    current_value=guard_config["enabled"],
                    suggested_value=True
                ))
        
        # Validate threshold field
        if "threshold" in guard_config:
            threshold = guard_config["threshold"]
            if not isinstance(threshold, (int, float)):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path=f"{field_prefix}.threshold",
                    message="Threshold must be a number",
                    current_value=threshold,
                    suggested_value=3
                ))
            elif threshold < 0:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path=f"{field_prefix}.threshold",
                    message="Threshold must be non-negative",
                    current_value=threshold,
                    suggested_value=3
                ))
        
        # Guard-specific validation
        self._validate_guard_specific_config(guard_name, guard_config, result)
    
    def _validate_guard_specific_config(self, guard_name: str, guard_config: Dict[str, Any],
                                      result: ValidationResult):
        """Validate guard-specific configuration."""
        field_prefix = f"guards.{guard_name}"
        
        if guard_name == "privacy_sentry":
            self._validate_privacy_sentry_config(guard_config, field_prefix, result)
        elif guard_name == "task_navigator":
            self._validate_task_navigator_config(guard_config, field_prefix, result)
        elif guard_name == "prompt_sanitizer":
            self._validate_prompt_sanitizer_config(guard_config, field_prefix, result)
        elif guard_name == "mcp_server":
            self._validate_mcp_server_guard_config(guard_config, field_prefix, result)
    
    def _validate_privacy_sentry_config(self, config: Dict[str, Any], 
                                      field_prefix: str, result: ValidationResult):
        """Validate privacy sentry specific configuration."""
        # Validate PII patterns
        if "patterns" in config:
            patterns = config["patterns"]
            if not isinstance(patterns, list):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path=f"{field_prefix}.patterns",
                    message="Patterns must be a list",
                    current_value=type(patterns).__name__
                ))
            else:
                for i, pattern in enumerate(patterns):
                    if not isinstance(pattern, str):
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            field_path=f"{field_prefix}.patterns[{i}]",
                            message="Pattern must be a string",
                            current_value=pattern
                        ))
    
    def _validate_task_navigator_config(self, config: Dict[str, Any],
                                      field_prefix: str, result: ValidationResult):
        """Validate task navigator specific configuration."""
        # Validate constraints
        if "constraints" in config:
            constraints = config["constraints"]
            if not isinstance(constraints, dict):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path=f"{field_prefix}.constraints",
                    message="Constraints must be a dictionary",
                    current_value=type(constraints).__name__
                ))
    
    def _validate_prompt_sanitizer_config(self, config: Dict[str, Any],
                                        field_prefix: str, result: ValidationResult):
        """Validate prompt sanitizer specific configuration."""
        # Validate sanitization rules
        if "rules" in config:
            rules = config["rules"]
            if not isinstance(rules, list):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path=f"{field_prefix}.rules",
                    message="Rules must be a list",
                    current_value=type(rules).__name__
                ))
    
    def _validate_mcp_server_guard_config(self, config: Dict[str, Any],
                                        field_prefix: str, result: ValidationResult):
        """Validate MCP server guard specific configuration."""
        # Validate API URL
        if "doordash_api_url" in config:
            url = config["doordash_api_url"]
            if not isinstance(url, str) or not self.url_pattern.match(url):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path=f"{field_prefix}.doordash_api_url",
                    message="Invalid API URL format",
                    current_value=url,
                    suggested_value="https://api.doordash.com/v1"
                ))
        
        # Validate timeout settings
        for timeout_field in ["timeout_seconds"]:
            if timeout_field in config:
                timeout = config[timeout_field]
                if not isinstance(timeout, (int, float)) or timeout <= 0:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field_path=f"{field_prefix}.{timeout_field}",
                        message=f"{timeout_field} must be a positive number",
                        current_value=timeout,
                        suggested_value=30
                    ))
    
    def _validate_agents_section(self, agents_config: Dict[str, Any],
                               result: ValidationResult):
        """Validate agents configuration section."""
        if not isinstance(agents_config, dict):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field_path="agents",
                message="Agents section must be a dictionary",
                current_value=type(agents_config).__name__
            ))
            return
        
        for agent_name, agent_config in agents_config.items():
            self._validate_individual_agent(agent_name, agent_config, result)
    
    def _validate_individual_agent(self, agent_name: str, agent_config: Any,
                                 result: ValidationResult):
        """Validate individual agent configuration."""
        field_prefix = f"agents.{agent_name}"
        
        if not isinstance(agent_config, dict):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field_path=field_prefix,
                message="Agent configuration must be a dictionary",
                current_value=type(agent_config).__name__
            ))
            return
        
        # Validate agent name
        if agent_name not in self.valid_agent_types:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field_path=field_prefix,
                message=f"Unknown agent type '{agent_name}'",
                current_value=agent_name,
                suggested_value="Use one of: " + ", ".join(self.valid_agent_types)
            ))
        
        # Validate AI model
        if "ai_model" in agent_config:
            ai_model = agent_config["ai_model"]
            if not isinstance(ai_model, str) or not ai_model.strip():
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path=f"{field_prefix}.ai_model",
                    message="AI model must be a non-empty string",
                    current_value=ai_model,
                    suggested_value="llama2:7b"
                ))
        
        # Validate memory type
        if "memory_type" in agent_config:
            memory_type = agent_config["memory_type"]
            if memory_type not in self.valid_memory_types:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path=f"{field_prefix}.memory_type",
                    message=f"Invalid memory type '{memory_type}'",
                    current_value=memory_type,
                    suggested_value="conversation_buffer"
                ))
        
        # Validate timeout
        if "timeout_seconds" in agent_config:
            timeout = agent_config["timeout_seconds"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path=f"{field_prefix}.timeout_seconds",
                    message="Timeout must be a positive number",
                    current_value=timeout,
                    suggested_value=30,
                    fix_available=True
                ))
        
        # Validate max retries
        if "max_retries" in agent_config:
            max_retries = agent_config["max_retries"]
            if not isinstance(max_retries, int) or max_retries < 0:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path=f"{field_prefix}.max_retries",
                    message="Max retries must be a non-negative integer",
                    current_value=max_retries,
                    suggested_value=3
                ))
    
    def _validate_logging_section(self, logging_config: Dict[str, Any],
                                result: ValidationResult):
        """Validate logging configuration section."""
        if not isinstance(logging_config, dict):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field_path="logging",
                message="Logging section must be a dictionary",
                current_value=type(logging_config).__name__
            ))
            return
        
        # Validate log level
        if "level" in logging_config:
            level = logging_config["level"]
            if level not in self.valid_log_levels:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path="logging.level",
                    message=f"Invalid log level '{level}'",
                    current_value=level,
                    suggested_value="INFO",
                    fix_available=True
                ))
        
        # Validate file paths
        file_fields = ["file", "alerts_file", "agent_conversations"]
        for field in file_fields:
            if field in logging_config:
                file_path = logging_config[field]
                if not isinstance(file_path, str):
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field_path=f"logging.{field}",
                        message="File path must be a string",
                        current_value=file_path
                    ))
                elif not self.file_path_pattern.match(file_path):
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        field_path=f"logging.{field}",
                        message="File path contains invalid characters",
                        current_value=file_path
                    ))
    
    def _validate_metrics_section(self, metrics_config: Dict[str, Any],
                                result: ValidationResult):
        """Validate metrics configuration section."""
        if not isinstance(metrics_config, dict):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field_path="metrics",
                message="Metrics section must be a dictionary",
                current_value=type(metrics_config).__name__
            ))
            return
        
        # Validate enabled flag
        if "enabled" in metrics_config:
            enabled = metrics_config["enabled"]
            if not isinstance(enabled, bool):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path="metrics.enabled",
                    message="Enabled must be a boolean",
                    current_value=enabled,
                    suggested_value=True
                ))
        
        # Validate retention settings
        if "retention_hours" in metrics_config:
            retention = metrics_config["retention_hours"]
            if not isinstance(retention, (int, float)) or retention <= 0:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path="metrics.retention_hours",
                    message="Retention hours must be a positive number",
                    current_value=retention,
                    suggested_value=24
                ))
    
    def _validate_notifications_section(self, notifications_config: Dict[str, Any],
                                      result: ValidationResult):
        """Validate notifications configuration section."""
        if not isinstance(notifications_config, dict):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field_path="notifications",
                message="Notifications section must be a dictionary",
                current_value=type(notifications_config).__name__
            ))
            return
        
        # Validate enabled flag
        if "enabled" in notifications_config:
            enabled = notifications_config["enabled"]
            if not isinstance(enabled, bool):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path="notifications.enabled",
                    message="Enabled must be a boolean",
                    current_value=enabled,
                    suggested_value=True
                ))
    
    def _validate_mcp_server_section(self, mcp_config: Dict[str, Any],
                                   result: ValidationResult):
        """Validate MCP server configuration section."""
        if not isinstance(mcp_config, dict):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field_path="mcp_server",
                message="MCP server section must be a dictionary",
                current_value=type(mcp_config).__name__
            ))
            return
        
        # Validate enabled flag
        if "enabled" in mcp_config:
            enabled = mcp_config["enabled"]
            if not isinstance(enabled, bool):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field_path="mcp_server.enabled",
                    message="Enabled must be a boolean",
                    current_value=enabled,
                    suggested_value=False
                ))
    
    def _validate_cross_section_dependencies(self, config_data: Dict[str, Any],
                                           result: ValidationResult):
        """Validate dependencies between different configuration sections."""
        # Check if agents reference valid guards
        if "agents" in config_data and "guards" in config_data:
            available_guards = set(config_data["guards"].keys())
            
            for agent_name, agent_config in config_data["agents"].items():
                if isinstance(agent_config, dict) and "guards" in agent_config:
                    agent_guards = agent_config["guards"]
                    if isinstance(agent_guards, list):
                        for guard in agent_guards:
                            if guard not in available_guards:
                                result.add_issue(ValidationIssue(
                                    severity=ValidationSeverity.WARNING,
                                    field_path=f"agents.{agent_name}.guards",
                                    message=f"Agent references non-existent guard '{guard}'",
                                    current_value=guard,
                                    suggested_value=list(available_guards)
                                ))
    
    def fix_configuration(self, config_data: Dict[str, Any], 
                         validation_result: ValidationResult) -> Dict[str, Any]:
        """Apply automatic fixes to configuration based on validation results."""
        fixed_config = config_data.copy()
        
        for issue in validation_result.issues:
            if issue.fix_available and issue.suggested_value is not None:
                # Apply the fix
                field_path = issue.field_path.split('.')
                self._set_nested_value(fixed_config, field_path, issue.suggested_value)
                validation_result.fixes_applied.append(f"Fixed {issue.field_path}: {issue.message}")
        
        return fixed_config
    
    def _set_nested_value(self, config: Dict[str, Any], field_path: List[str], value: Any):
        """Set a value in a nested dictionary structure."""
        current = config
        
        # Navigate to the parent of the target field
        for field in field_path[:-1]:
            if field not in current:
                current[field] = {}
            current = current[field]
        
        # Set the final value
        current[field_path[-1]] = value


# Global validator instance
_config_validator: Optional[ConfigurationValidator] = None


def get_config_validator() -> ConfigurationValidator:
    """Get the global configuration validator instance."""
    global _config_validator
    if _config_validator is None:
        _config_validator = ConfigurationValidator()
    return _config_validator


def validate_config_file(file_path: str) -> ValidationResult:
    """Validate a configuration file."""
    validator = get_config_validator()
    
    try:
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return validator.validate_configuration(config_data, file_path)
    except FileNotFoundError:
        result = ValidationResult(is_valid=False)
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field_path="file",
            message=f"Configuration file not found: {file_path}"
        ))
        return result
    except yaml.YAMLError as e:
        result = ValidationResult(is_valid=False)
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field_path="file",
            message=f"Invalid YAML syntax: {str(e)}"
        ))
        return result
    except Exception as e:
        result = ValidationResult(is_valid=False)
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field_path="file",
            message=f"Unexpected error during validation: {str(e)}"
        ))
        return result


def validate_config_data(config_data: Dict[str, Any]) -> ValidationResult:
    """Validate configuration data."""
    validator = get_config_validator()
    return validator.validate_configuration(config_data)
