"""
Configuration Tools and Utilities for SafeHive AI Security Sandbox.

This module provides command-line tools and utilities for configuration management,
including validation, fixing, and migration tools.
"""

import os
import yaml
import json
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

from .config_validator import (
    get_config_validator, validate_config_file, validate_config_data,
    ValidationResult, ValidationIssue, ValidationSeverity
)
from .config_error_handler import (
    get_config_error_handler, handle_configuration_error,
    ConfigurationError, ConfigurationErrorType
)
from ..utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass
class ConfigAnalysisResult:
    """Result of configuration analysis."""
    file_path: str
    is_valid: bool
    validation_result: ValidationResult
    errors: List[ConfigurationError]
    suggestions: List[str] = field(default_factory=list)
    auto_fixes_available: int = 0
    analysis_timestamp: datetime = field(default_factory=datetime.now)


class ConfigurationTools:
    """Command-line tools for configuration management."""
    
    def __init__(self):
        self.validator = get_config_validator()
        self.error_handler = get_config_error_handler()
    
    def analyze_configuration(self, file_path: str) -> ConfigAnalysisResult:
        """Perform comprehensive configuration analysis."""
        console.print(f"🔍 Analyzing configuration: {file_path}")
        
        # Validate configuration
        validation_result = validate_config_file(file_path)
        
        # Check for errors during loading
        errors = []
        try:
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            _, errors = handle_configuration_error(file_path, e)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(validation_result, errors)
        
        # Count auto-fixes available
        auto_fixes = sum(1 for issue in validation_result.issues if issue.fix_available)
        
        return ConfigAnalysisResult(
            file_path=file_path,
            is_valid=validation_result.is_valid and len(errors) == 0,
            validation_result=validation_result,
            errors=errors,
            suggestions=suggestions,
            auto_fixes_available=auto_fixes
        )
    
    def _generate_suggestions(self, validation_result: ValidationResult, 
                            errors: List[ConfigurationError]) -> List[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        # Suggestions based on validation issues
        if validation_result.errors:
            suggestions.append("Fix validation errors to ensure proper functionality")
        
        if validation_result.warnings:
            suggestions.append("Address warnings to optimize configuration")
        
        # Performance suggestions
        if self._has_performance_issues(validation_result):
            suggestions.append("Consider optimizing configuration for better performance")
        
        # Security suggestions
        if self._has_security_issues(validation_result):
            suggestions.append("Review security settings and enable appropriate guards")
        
        # Feature suggestions
        if self._has_missing_features(validation_result):
            suggestions.append("Consider enabling additional features for enhanced functionality")
        
        return suggestions
    
    def _has_performance_issues(self, validation_result: ValidationResult) -> bool:
        """Check if configuration has performance-related issues."""
        # Check for high timeout values, excessive logging, etc.
        for issue in validation_result.issues:
            if "timeout" in issue.field_path.lower() and isinstance(issue.current_value, (int, float)):
                if issue.current_value > 60:  # High timeout
                    return True
        return False
    
    def _has_security_issues(self, validation_result: ValidationResult) -> bool:
        """Check if configuration has security-related issues."""
        # Check if security guards are disabled
        for issue in validation_result.issues:
            if "guards" in issue.field_path and "enabled" in issue.field_path:
                if issue.current_value is False:
                    return True
        return False
    
    def _has_missing_features(self, validation_result: ValidationResult) -> bool:
        """Check if configuration is missing useful features."""
        # Check for missing optional features
        for issue in validation_result.issues:
            if issue.severity == ValidationSeverity.WARNING and "missing" in issue.message.lower():
                return True
        return False
    
    def fix_configuration(self, file_path: str, backup: bool = True) -> bool:
        """Fix configuration issues automatically."""
        console.print(f"🔧 Fixing configuration: {file_path}")
        
        try:
            # Load current configuration
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Validate and get issues
            validation_result = validate_config_data(config_data)
            
            if validation_result.is_valid:
                console.print("✅ Configuration is already valid")
                return True
            
            # Create backup if requested
            if backup:
                backup_path = self._create_backup(file_path)
                console.print(f"📋 Backup created: {backup_path}")
            
            # Apply fixes
            fixed_config = self.validator.fix_configuration(config_data, validation_result)
            
            # Write fixed configuration
            with open(file_path, 'w') as f:
                yaml.dump(fixed_config, f, default_flow_style=False, indent=2)
            
            # Verify fixes
            verification_result = validate_config_file(file_path)
            
            if verification_result.is_valid:
                console.print("✅ Configuration fixed successfully")
                console.print(f"📊 Applied {len(validation_result.fixes_applied)} fixes")
                return True
            else:
                console.print("⚠️ Some issues could not be automatically fixed")
                console.print(f"📊 Applied {len(validation_result.fixes_applied)} fixes")
                return False
        
        except Exception as e:
            console.print(f"❌ Failed to fix configuration: {e}")
            logger.error(f"Config fix failed for {file_path}: {e}")
            return False
    
    def _create_backup(self, file_path: str) -> str:
        """Create a backup of the configuration file."""
        original_path = Path(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = original_path.parent / f"{original_path.stem}.backup.{timestamp}.yaml"
        
        shutil.copy2(file_path, backup_path)
        return str(backup_path)
    
    def migrate_configuration(self, old_file: str, new_file: str, version: str = "latest") -> bool:
        """Migrate configuration to a new format."""
        console.print(f"🔄 Migrating configuration: {old_file} -> {new_file}")
        
        try:
            # Load old configuration
            with open(old_file, 'r') as f:
                old_config = yaml.safe_load(f)
            
            # Perform migration
            migrated_config = self._perform_migration(old_config, version)
            
            # Save migrated configuration
            with open(new_file, 'w') as f:
                yaml.dump(migrated_config, f, default_flow_style=False, indent=2)
            
            # Validate migrated configuration
            validation_result = validate_config_file(new_file)
            
            if validation_result.is_valid:
                console.print("✅ Configuration migrated successfully")
                return True
            else:
                console.print("⚠️ Migrated configuration has issues")
                console.print("Run 'safehive config fix' to resolve them")
                return False
        
        except Exception as e:
            console.print(f"❌ Migration failed: {e}")
            logger.error(f"Config migration failed: {e}")
            return False
    
    def _perform_migration(self, old_config: Dict[str, Any], version: str) -> Dict[str, Any]:
        """Perform configuration migration."""
        # This would contain migration logic for different versions
        # For now, just ensure the config has all required sections
        
        migrated_config = old_config.copy()
        
        # Ensure all required sections exist
        required_sections = {
            "guards", "agents", "logging", "metrics", "notifications"
        }
        
        for section in required_sections:
            if section not in migrated_config:
                migrated_config[section] = {}
        
        return migrated_config
    
    def compare_configurations(self, file1: str, file2: str) -> Dict[str, Any]:
        """Compare two configuration files."""
        console.print(f"📊 Comparing configurations: {file1} vs {file2}")
        
        try:
            # Load both configurations
            with open(file1, 'r') as f:
                config1 = yaml.safe_load(f)
            
            with open(file2, 'r') as f:
                config2 = yaml.safe_load(f)
            
            # Perform comparison
            differences = self._find_differences(config1, config2, "root")
            
            return {
                "file1": file1,
                "file2": file2,
                "differences": differences,
                "identical": len(differences) == 0
            }
        
        except Exception as e:
            console.print(f"❌ Comparison failed: {e}")
            logger.error(f"Config comparison failed: {e}")
            return {"error": str(e)}
    
    def _find_differences(self, config1: Dict[str, Any], config2: Dict[str, Any], 
                         path: str) -> List[Dict[str, Any]]:
        """Find differences between two configurations."""
        differences = []
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path != "root" else key
            
            if key not in config1:
                differences.append({
                    "path": current_path,
                    "type": "missing_in_first",
                    "value": config2[key]
                })
            elif key not in config2:
                differences.append({
                    "path": current_path,
                    "type": "missing_in_second",
                    "value": config1[key]
                })
            elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
                differences.extend(self._find_differences(config1[key], config2[key], current_path))
            elif config1[key] != config2[key]:
                differences.append({
                    "path": current_path,
                    "type": "different_value",
                    "value1": config1[key],
                    "value2": config2[key]
                })
        
        return differences
    
    def export_configuration(self, file_path: str, output_format: str = "json") -> bool:
        """Export configuration to different formats."""
        console.print(f"📤 Exporting configuration: {file_path} -> {output_format}")
        
        try:
            # Load configuration
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Generate output filename
            output_file = Path(file_path).with_suffix(f".{output_format}")
            
            # Export based on format
            if output_format == "json":
                with open(output_file, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
            elif output_format == "yaml":
                with open(output_file, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            else:
                console.print(f"❌ Unsupported format: {output_format}")
                return False
            
            console.print(f"✅ Configuration exported to: {output_file}")
            return True
        
        except Exception as e:
            console.print(f"❌ Export failed: {e}")
            logger.error(f"Config export failed: {e}")
            return False
    
    def create_configuration_template(self, output_path: str, template_type: str = "default") -> bool:
        """Create a configuration template."""
        console.print(f"📝 Creating configuration template: {output_path}")
        
        try:
            templates = {
                "default": self._get_default_template(),
                "minimal": self._get_minimal_template(),
                "secure": self._get_secure_template(),
                "development": self._get_development_template()
            }
            
            if template_type not in templates:
                console.print(f"❌ Unknown template type: {template_type}")
                console.print(f"Available types: {', '.join(templates.keys())}")
                return False
            
            template_config = templates[template_type]
            
            with open(output_path, 'w') as f:
                yaml.dump(template_config, f, default_flow_style=False, indent=2)
            
            console.print(f"✅ Template created: {output_path}")
            console.print(f"📋 Template type: {template_type}")
            return True
        
        except Exception as e:
            console.print(f"❌ Template creation failed: {e}")
            logger.error(f"Template creation failed: {e}")
            return False
    
    def _get_default_template(self) -> Dict[str, Any]:
        """Get default configuration template."""
        return {
            "guards": {
                "privacy_sentry": {
                    "enabled": True,
                    "threshold": 3,
                    "patterns": ["credit_card", "ssn", "email", "phone"]
                },
                "task_navigator": {
                    "enabled": True,
                    "threshold": 3,
                    "constraints": {
                        "max_budget": 100.0,
                        "dietary_restrictions": []
                    }
                },
                "prompt_sanitizer": {
                    "enabled": True,
                    "threshold": 3,
                    "rules": ["sql_injection", "xss", "path_traversal"]
                }
            },
            "agents": {
                "orchestrator": {
                    "ai_model": "llama2:7b",
                    "timeout_seconds": 30,
                    "max_retries": 3,
                    "memory_type": "conversation_buffer"
                },
                "user_twin": {
                    "ai_model": "llama2:7b",
                    "timeout_seconds": 30,
                    "max_retries": 3,
                    "memory_type": "conversation_buffer"
                }
            },
            "logging": {
                "level": "INFO",
                "file": "logs/sandbox.log",
                "alerts_file": "logs/alerts.log",
                "agent_conversations": "logs/agent_conversations.log",
                "structured": True
            },
            "metrics": {
                "enabled": True,
                "retention_hours": 24,
                "collection_interval": 60
            },
            "notifications": {
                "enabled": True,
                "email": False,
                "webhook": False
            }
        }
    
    def _get_minimal_template(self) -> Dict[str, Any]:
        """Get minimal configuration template."""
        return {
            "guards": {
                "privacy_sentry": {"enabled": True, "threshold": 3},
                "task_navigator": {"enabled": True, "threshold": 3},
                "prompt_sanitizer": {"enabled": True, "threshold": 3}
            },
            "agents": {
                "orchestrator": {"ai_model": "llama2:7b"},
                "user_twin": {"ai_model": "llama2:7b"}
            },
            "logging": {"level": "WARNING", "file": "logs/sandbox.log"},
            "metrics": {"enabled": False},
            "notifications": {"enabled": False}
        }
    
    def _get_secure_template(self) -> Dict[str, Any]:
        """Get security-focused configuration template."""
        template = self._get_default_template()
        template["guards"]["privacy_sentry"]["threshold"] = 1  # More sensitive
        template["guards"]["task_navigator"]["threshold"] = 1
        template["guards"]["prompt_sanitizer"]["threshold"] = 1
        template["logging"]["level"] = "DEBUG"  # More verbose logging
        template["notifications"]["enabled"] = True
        return template
    
    def _get_development_template(self) -> Dict[str, Any]:
        """Get development-focused configuration template."""
        template = self._get_default_template()
        template["logging"]["level"] = "DEBUG"
        template["metrics"]["enabled"] = True
        template["metrics"]["retention_hours"] = 168  # 1 week
        return template


def display_analysis_result(result: ConfigAnalysisResult):
    """Display configuration analysis results in a formatted way."""
    console.print(f"\n📊 Configuration Analysis Results")
    console.print(f"File: {result.file_path}")
    console.print(f"Valid: {'✅' if result.is_valid else '❌'}")
    console.print(f"Timestamp: {result.analysis_timestamp.isoformat()}")
    
    # Summary
    summary = result.validation_result.get_summary()
    table = Table(title="Validation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="white")
    
    table.add_row("Total Issues", str(summary["total_issues"]))
    table.add_row("Errors", str(summary["errors"]))
    table.add_row("Warnings", str(summary["warnings"]))
    table.add_row("Auto-fixes Available", str(result.auto_fixes_available))
    
    console.print(table)
    
    # Issues
    if result.validation_result.issues:
        console.print("\n🔍 Issues Found:")
        for issue in result.validation_result.issues:
            console.print(f"  {issue}")
    
    # Errors
    if result.errors:
        console.print("\n❌ Configuration Errors:")
        for error in result.errors:
            console.print(f"  {error}")
    
    # Suggestions
    if result.suggestions:
        console.print("\n💡 Suggestions:")
        for suggestion in result.suggestions:
            console.print(f"  • {suggestion}")


def display_comparison_result(comparison: Dict[str, Any]):
    """Display configuration comparison results."""
    if "error" in comparison:
        console.print(f"❌ Comparison failed: {comparison['error']}")
        return
    
    console.print(f"\n📊 Configuration Comparison")
    console.print(f"File 1: {comparison['file1']}")
    console.print(f"File 2: {comparison['file2']}")
    console.print(f"Identical: {'✅' if comparison['identical'] else '❌'}")
    
    if not comparison["identical"]:
        console.print(f"\n🔍 Differences Found ({len(comparison['differences'])}):")
        
        table = Table(title="Configuration Differences")
        table.add_column("Path", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Value 1", style="red")
        table.add_column("Value 2", style="green")
        
        for diff in comparison["differences"]:
            value1 = str(diff.get("value", diff.get("value1", "N/A")))
            value2 = str(diff.get("value", diff.get("value2", "N/A")))
            
            table.add_row(
                diff["path"],
                diff["type"],
                value1,
                value2
            )
        
        console.print(table)


# Global tools instance
_config_tools: Optional[ConfigurationTools] = None


def get_config_tools() -> ConfigurationTools:
    """Get the global configuration tools instance."""
    global _config_tools
    if _config_tools is None:
        _config_tools = ConfigurationTools()
    return _config_tools
