"""
Configuration Error Handler for SafeHive AI Security Sandbox.

This module provides robust error handling for configuration-related issues,
including recovery strategies, user-friendly error messages, and fallback mechanisms.
"""

import os
import yaml
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .config_validator import ValidationResult, ValidationIssue, ValidationSeverity
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ConfigurationErrorType(Enum):
    """Types of configuration errors."""
    FILE_NOT_FOUND = "file_not_found"
    INVALID_YAML = "invalid_yaml"
    MISSING_SECTION = "missing_section"
    INVALID_VALUE = "invalid_value"
    VALIDATION_ERROR = "validation_error"
    PERMISSION_ERROR = "permission_error"
    CORRUPTED_FILE = "corrupted_file"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ConfigurationError:
    """Represents a configuration error with context and recovery options."""
    error_type: ConfigurationErrorType
    message: str
    field_path: Optional[str] = None
    original_value: Any = None
    suggested_fix: Optional[str] = None
    recovery_options: List[str] = field(default_factory=list)
    severity: str = "error"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        severity_icon = {
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️"
        }
        return f"{severity_icon.get(self.severity, '❓')} {self.message}"


class ConfigurationErrorHandler:
    """Handles configuration errors with recovery strategies."""
    
    def __init__(self):
        self.error_history: List[ConfigurationError] = []
        self.fallback_configs: Dict[str, Dict[str, Any]] = {}
        self._setup_fallback_configs()
    
    def _setup_fallback_configs(self):
        """Set up fallback configurations for different scenarios."""
        # Minimal fallback configuration
        self.fallback_configs["minimal"] = {
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
        
        # Safe fallback configuration (all features disabled)
        self.fallback_configs["safe"] = {
            "guards": {
                "privacy_sentry": {"enabled": False, "threshold": 3},
                "task_navigator": {"enabled": False, "threshold": 3},
                "prompt_sanitizer": {"enabled": False, "threshold": 3}
            },
            "agents": {
                "orchestrator": {"ai_model": "llama2:7b", "timeout_seconds": 30},
                "user_twin": {"ai_model": "llama2:7b", "timeout_seconds": 30}
            },
            "logging": {"level": "WARNING", "file": "logs/sandbox.log"},
            "metrics": {"enabled": False, "retention_hours": 24},
            "notifications": {"enabled": False}
        }
    
    def handle_file_not_found(self, file_path: str) -> Tuple[Optional[Dict[str, Any]], List[ConfigurationError]]:
        """Handle file not found errors."""
        errors = []
        
        # Check if it's the default config file
        if "default_config.yaml" in file_path:
            error = ConfigurationError(
                error_type=ConfigurationErrorType.FILE_NOT_FOUND,
                message=f"Default configuration file not found: {file_path}",
                suggested_fix="Run 'safehive init' to create default configuration",
                recovery_options=[
                    "Use minimal fallback configuration",
                    "Create default configuration file",
                    "Specify alternative config file"
                ]
            )
            errors.append(error)
            logger.error(f"Default config file not found: {file_path}")
            return self.fallback_configs["minimal"], errors
        
        # Check for alternative locations
        alternative_paths = self._find_alternative_config_files(file_path)
        if alternative_paths:
            error = ConfigurationError(
                error_type=ConfigurationErrorType.FILE_NOT_FOUND,
                message=f"Configuration file not found: {file_path}",
                suggested_fix=f"Found alternatives: {', '.join(alternative_paths)}",
                recovery_options=alternative_paths + ["Use fallback configuration"]
            )
            errors.append(error)
            logger.warning(f"Config file not found, alternatives available: {alternative_paths}")
        else:
            error = ConfigurationError(
                error_type=ConfigurationErrorType.FILE_NOT_FOUND,
                message=f"Configuration file not found: {file_path}",
                suggested_fix="Create configuration file or use fallback",
                recovery_options=["Use fallback configuration", "Create new configuration"]
            )
            errors.append(error)
            logger.error(f"Config file not found, no alternatives: {file_path}")
        
        return self.fallback_configs["safe"], errors
    
    def _find_alternative_config_files(self, original_path: str) -> List[str]:
        """Find alternative configuration files."""
        alternatives = []
        original_file = Path(original_path)
        
        # Check common alternative locations
        search_paths = [
            original_file.parent.parent / "config" / original_file.name,
            original_file.parent / "default_config.yaml",
            Path.cwd() / "config" / original_file.name,
            Path.cwd() / original_file.name,
            Path.home() / ".safehive" / original_file.name
        ]
        
        for path in search_paths:
            if path.exists() and path != original_file:
                alternatives.append(str(path))
        
        return alternatives
    
    def handle_invalid_yaml(self, file_path: str, yaml_error: Exception) -> Tuple[Optional[Dict[str, Any]], List[ConfigurationError]]:
        """Handle invalid YAML syntax errors."""
        errors = []
        
        error = ConfigurationError(
            error_type=ConfigurationErrorType.INVALID_YAML,
            message=f"Invalid YAML syntax in {file_path}: {str(yaml_error)}",
            suggested_fix="Fix YAML syntax errors or use backup configuration",
            recovery_options=[
                "Use backup configuration file",
                "Use fallback configuration",
                "Fix YAML syntax manually"
            ]
        )
        errors.append(error)
        
        logger.error(f"Invalid YAML in {file_path}: {yaml_error}")
        
        # Try to find a backup file
        backup_path = self._find_backup_config(file_path)
        if backup_path:
            try:
                with open(backup_path, 'r') as f:
                    backup_config = yaml.safe_load(f)
                
                backup_error = ConfigurationError(
                    error_type=ConfigurationErrorType.INVALID_YAML,
                    message=f"Using backup configuration from {backup_path}",
                    severity="info",
                    suggested_fix="Backup configuration loaded successfully"
                )
                errors.append(backup_error)
                
                return backup_config, errors
            except Exception as e:
                logger.warning(f"Failed to load backup config {backup_path}: {e}")
        
        return self.fallback_configs["safe"], errors
    
    def _find_backup_config(self, original_path: str) -> Optional[str]:
        """Find backup configuration files."""
        original_file = Path(original_path)
        
        # Check for common backup patterns
        backup_patterns = [
            f"{original_file.stem}.backup.yaml",
            f"{original_file.stem}.bak.yaml",
            f"{original_file.stem}.old.yaml",
            f"{original_file.stem}.orig.yaml"
        ]
        
        for pattern in backup_patterns:
            backup_path = original_file.parent / pattern
            if backup_path.exists():
                return str(backup_path)
        
        return None
    
    def handle_validation_errors(self, file_path: str, validation_result: ValidationResult) -> Tuple[Optional[Dict[str, Any]], List[ConfigurationError]]:
        """Handle configuration validation errors."""
        errors = []
        
        # Convert validation issues to configuration errors
        for issue in validation_result.issues:
            error_type = ConfigurationErrorType.VALIDATION_ERROR
            severity = "error" if issue.severity == ValidationSeverity.ERROR else "warning"
            
            error = ConfigurationError(
                error_type=error_type,
                message=issue.message,
                field_path=issue.field_path,
                original_value=issue.current_value,
                suggested_fix=issue.suggested_value,
                severity=severity
            )
            errors.append(error)
        
        # Determine recovery strategy based on error severity
        if validation_result.is_valid or len(validation_result.errors) == 0:
            # Only warnings, configuration is usable
            logger.warning(f"Configuration has warnings but is usable: {file_path}")
            return None, errors  # Return None to indicate config is usable despite warnings
        
        elif len(validation_result.errors) <= 3:
            # Few errors, try to fix them automatically
            logger.warning(f"Configuration has {len(validation_result.errors)} errors, attempting auto-fix: {file_path}")
            return None, errors  # Let the caller handle auto-fix
        
        else:
            # Too many errors, use fallback
            error = ConfigurationError(
                error_type=ConfigurationErrorType.VALIDATION_ERROR,
                message=f"Configuration has {len(validation_result.errors)} errors, using fallback",
                suggested_fix="Review and fix configuration file",
                recovery_options=[
                    "Use minimal fallback configuration",
                    "Use safe fallback configuration",
                    "Fix configuration manually"
                ]
            )
            errors.append(error)
            logger.error(f"Too many config errors ({len(validation_result.errors)}), using fallback")
            return self.fallback_configs["safe"], errors
    
    def handle_permission_error(self, file_path: str, permission_error: Exception) -> Tuple[Optional[Dict[str, Any]], List[ConfigurationError]]:
        """Handle file permission errors."""
        errors = []
        
        error = ConfigurationError(
            error_type=ConfigurationErrorType.PERMISSION_ERROR,
            message=f"Permission denied accessing {file_path}: {str(permission_error)}",
            suggested_fix="Check file permissions or use alternative location",
            recovery_options=[
                "Use fallback configuration",
                "Fix file permissions",
                "Use alternative config location"
            ]
        )
        errors.append(error)
        
        logger.error(f"Permission error accessing {file_path}: {permission_error}")
        return self.fallback_configs["safe"], errors
    
    def handle_corrupted_file(self, file_path: str, corruption_error: Exception) -> Tuple[Optional[Dict[str, Any]], List[ConfigurationError]]:
        """Handle corrupted configuration files."""
        errors = []
        
        error = ConfigurationError(
            error_type=ConfigurationErrorType.CORRUPTED_FILE,
            message=f"Configuration file appears corrupted: {file_path}",
            suggested_fix="Use backup or fallback configuration",
            recovery_options=[
                "Use backup configuration",
                "Use fallback configuration",
                "Recreate configuration file"
            ]
        )
        errors.append(error)
        
        logger.error(f"Corrupted config file {file_path}: {corruption_error}")
        
        # Try to recover partial data
        try:
            partial_config = self._attempt_partial_recovery(file_path)
            if partial_config:
                recovery_error = ConfigurationError(
                    error_type=ConfigurationErrorType.CORRUPTED_FILE,
                    message="Partial configuration recovered",
                    severity="info",
                    suggested_fix="Partial config loaded, some features may be disabled"
                )
                errors.append(recovery_error)
                return partial_config, errors
        except Exception:
            pass
        
        return self.fallback_configs["safe"], errors
    
    def _attempt_partial_recovery(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Attempt to recover partial configuration from corrupted file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Try to extract valid YAML blocks
            lines = content.split('\n')
            valid_lines = []
            
            for line in lines:
                # Skip lines that look corrupted
                if line.strip() and not any(char in line for char in ['\x00', '\xff', '\xfe']):
                    valid_lines.append(line)
            
            if valid_lines:
                partial_content = '\n'.join(valid_lines)
                partial_config = yaml.safe_load(partial_content)
                
                if partial_config and isinstance(partial_config, dict):
                    # Merge with fallback to ensure all required sections exist
                    fallback = self.fallback_configs["minimal"]
                    recovered_config = {**fallback, **partial_config}
                    return recovered_config
        
        except Exception:
            pass
        
        return None
    
    def create_recovery_report(self, errors: List[ConfigurationError]) -> str:
        """Create a detailed recovery report."""
        if not errors:
            return "✅ Configuration loaded successfully with no issues."
        
        report_lines = [
            "🔧 Configuration Recovery Report",
            "=" * 50,
            f"Timestamp: {datetime.now().isoformat()}",
            f"Total Issues: {len(errors)}",
            ""
        ]
        
        # Group errors by type
        error_groups = {}
        for error in errors:
            error_type = error.error_type.value
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(error)
        
        for error_type, type_errors in error_groups.items():
            report_lines.append(f"📋 {error_type.replace('_', ' ').title()} ({len(type_errors)} issues):")
            
            for error in type_errors:
                report_lines.append(f"  {error}")
                if error.suggested_fix:
                    report_lines.append(f"    💡 Suggestion: {error.suggested_fix}")
                if error.recovery_options:
                    report_lines.append(f"    🔧 Options: {', '.join(error.recovery_options)}")
                report_lines.append("")
        
        # Add recovery recommendations
        report_lines.extend([
            "🎯 Recovery Recommendations:",
            "- Review the issues above and apply suggested fixes",
            "- Use 'safehive config validate' to check configuration",
            "- Run 'safehive init' to recreate default configuration if needed",
            "- Check file permissions and disk space",
            ""
        ])
        
        return "\n".join(report_lines)
    
    def log_error_history(self):
        """Log the error history for debugging."""
        if not self.error_history:
            return
        
        logger.info(f"Configuration error history ({len(self.error_history)} errors):")
        for error in self.error_history[-10:]:  # Log last 10 errors
            logger.info(f"  {error.timestamp.isoformat()}: {error}")
    
    def clear_error_history(self):
        """Clear the error history."""
        self.error_history.clear()


# Global error handler instance
_config_error_handler: Optional[ConfigurationErrorHandler] = None


def get_config_error_handler() -> ConfigurationErrorHandler:
    """Get the global configuration error handler instance."""
    global _config_error_handler
    if _config_error_handler is None:
        _config_error_handler = ConfigurationErrorHandler()
    return _config_error_handler


def handle_configuration_error(file_path: str, error: Exception) -> Tuple[Optional[Dict[str, Any]], List[ConfigurationError]]:
    """Handle configuration errors with appropriate recovery strategies."""
    error_handler = get_config_error_handler()
    
    # Determine error type and handle accordingly
    if isinstance(error, FileNotFoundError):
        return error_handler.handle_file_not_found(file_path)
    elif isinstance(error, yaml.YAMLError):
        return error_handler.handle_invalid_yaml(file_path, error)
    elif isinstance(error, PermissionError):
        return error_handler.handle_permission_error(file_path, error)
    elif isinstance(error, (UnicodeDecodeError, json.JSONDecodeError)):
        return error_handler.handle_corrupted_file(file_path, error)
    else:
        # Unknown error
        config_error = ConfigurationError(
            error_type=ConfigurationErrorType.UNKNOWN_ERROR,
            message=f"Unknown configuration error: {str(error)}",
            suggested_fix="Use fallback configuration",
            recovery_options=["Use fallback configuration", "Check system logs"]
        )
        error_handler.error_history.append(config_error)
        logger.error(f"Unknown config error for {file_path}: {error}")
        return error_handler.fallback_configs["safe"], [config_error]
