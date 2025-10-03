"""
Unit tests for the CLI module.

This module tests the command-line interface functionality for the SafeHive system.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
from pathlib import Path

# Import the CLI app
from cli import app, check_system_requirements

runner = CliRunner()


class TestCLICommands:
    """Test CLI commands."""
    
    def test_version_command(self):
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "SafeHive AI Security Sandbox" in result.output
        assert "Version:" in result.output
    
    def test_info_command(self):
        """Test info command."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "SafeHive AI Security Sandbox" in result.output
        assert "Features:" in result.output
        assert "AI Security Guards" in result.output
    
    def test_status_command(self):
        """Test status command."""
        with patch('cli.check_ollama_connection') as mock_ollama, \
             patch('cli.ConfigLoader') as mock_config_loader, \
             patch('pathlib.Path.exists') as mock_path_exists:
            
            mock_ollama.return_value = True
            mock_config_loader.return_value.load_config.return_value = True
            mock_path_exists.return_value = True
            
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0
            assert "System Status" in result.output
            assert "Ollama:" in result.output
            assert "Configuration:" in result.output
            assert "Logs Directory:" in result.output
    
    def test_init_command(self):
        """Test init command."""
        with patch('cli.setup_logging') as mock_setup_logging, \
             patch('cli.ConfigLoader') as mock_config_loader, \
             patch('cli.check_system_requirements') as mock_check_requirements:
            
            mock_config_loader.return_value.load_config.return_value = True
            mock_check_requirements.return_value = True
            
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0
            assert "Initializing SafeHive" in result.output
            assert "system initialized successfully" in result.output
    
    def test_init_command_with_config_file(self):
        """Test init command with custom config file."""
        with patch('cli.setup_logging') as mock_setup_logging, \
             patch('cli.ConfigLoader') as mock_config_loader, \
             patch('cli.check_system_requirements') as mock_check_requirements:
            
            mock_config_loader.return_value.load_config.return_value = True
            mock_check_requirements.return_value = True
            
            result = runner.invoke(app, ["init", "--config", "custom_config.yaml"])
            assert result.exit_code == 0
            assert "Initializing SafeHive" in result.output
    
    def test_init_command_system_requirements_failed(self):
        """Test init command when system requirements fail."""
        with patch('cli.setup_logging') as mock_setup_logging, \
             patch('cli.ConfigLoader') as mock_config_loader, \
             patch('cli.check_system_requirements') as mock_check_requirements:
            
            mock_config_loader.return_value.load_config.return_value = True
            mock_check_requirements.return_value = False
            
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 1
            assert "System requirements not met" in result.output


class TestSandboxCommands:
    """Test sandbox commands."""
    
    def test_sandbox_start_command(self):
        """Test sandbox start command."""
        with patch('cli.check_system_requirements') as mock_check_requirements:
            mock_check_requirements.return_value = True
            
            result = runner.invoke(app, ["sandbox", "start"])
            assert result.exit_code == 0
            assert "Starting sandbox session" in result.output
            assert "food-ordering" in result.output
    
    def test_sandbox_start_command_with_options(self):
        """Test sandbox start command with options."""
        with patch('cli.check_system_requirements') as mock_check_requirements:
            mock_check_requirements.return_value = True
            
            result = runner.invoke(app, [
                "sandbox", "start", 
                "--scenario", "payment-processing",
                "--duration", "600",
                "--no-interactive"
            ])
            assert result.exit_code == 0
            assert "payment-processing" in result.output
            assert "600 seconds" in result.output
            assert "Disabled" in result.output
    
    def test_sandbox_start_command_requirements_failed(self):
        """Test sandbox start command when requirements fail."""
        with patch('cli.check_system_requirements') as mock_check_requirements:
            mock_check_requirements.return_value = False
            
            result = runner.invoke(app, ["sandbox", "start"])
            assert result.exit_code == 1
            assert "System requirements not met" in result.output
    
    def test_sandbox_stop_command(self):
        """Test sandbox stop command."""
        result = runner.invoke(app, ["sandbox", "stop"])
        assert result.exit_code == 0
        assert "Stopping sandbox session" in result.output
        assert "Sandbox session stopped" in result.output
    
    def test_sandbox_list_command(self):
        """Test sandbox list command."""
        result = runner.invoke(app, ["sandbox", "list"])
        assert result.exit_code == 0
        assert "Available Sandbox Scenarios" in result.output
        assert "food-ordering" in result.output
        assert "payment-processing" in result.output


class TestConfigCommands:
    """Test configuration commands."""
    
    def test_config_show_command(self):
        """Test config show command."""
        with patch('cli.ConfigLoader') as mock_config_loader:
            mock_config = MagicMock()
            mock_config.__dict__ = {"test": "value"}
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_config.return_value = True
            mock_loader_instance.config = mock_config
            mock_config_loader.return_value = mock_loader_instance
            
            result = runner.invoke(app, ["config", "show"])
            assert result.exit_code == 0
            assert "Current Configuration" in result.output
    
    def test_config_show_command_with_section(self):
        """Test config show command with specific section."""
        with patch('cli.ConfigLoader') as mock_config_loader:
            mock_config = MagicMock()
            mock_config.__dict__ = {"guards": {"test": "value"}}
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_config.return_value = True
            mock_loader_instance.config = mock_config
            mock_config_loader.return_value = mock_loader_instance
            
            result = runner.invoke(app, ["config", "show", "--section", "guards"])
            assert result.exit_code == 0
            assert "Configuration Section: guards" in result.output
    
    def test_config_show_command_section_not_found(self):
        """Test config show command with non-existent section."""
        with patch('cli.ConfigLoader') as mock_config_loader:
            mock_config = MagicMock()
            mock_config.__dict__ = {"test": "value"}
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_config.return_value = True
            mock_loader_instance.config = mock_config
            mock_config_loader.return_value = mock_loader_instance
            
            result = runner.invoke(app, ["config", "show", "--section", "nonexistent"])
            assert result.exit_code == 0
            assert "Section 'nonexistent' not found" in result.output
    
    def test_config_show_command_load_failed(self):
        """Test config show command when config load fails."""
        with patch('cli.ConfigLoader') as mock_config_loader:
            mock_config_loader.return_value.load_config.return_value = False
            
            result = runner.invoke(app, ["config", "show"])
            assert result.exit_code == 1
            assert "Failed to load configuration" in result.output
    
    def test_config_validate_command(self):
        """Test config validate command."""
        with patch('cli.ConfigLoader') as mock_config_loader:
            mock_config_loader.return_value.load_config.return_value = True
            
            result = runner.invoke(app, ["config", "validate"])
            assert result.exit_code == 0
            assert "Validating configuration file" in result.output
            assert "Configuration is valid" in result.output
    
    def test_config_validate_command_failed(self):
        """Test config validate command when validation fails."""
        with patch('cli.ConfigLoader') as mock_config_loader:
            mock_config_loader.return_value.load_config.return_value = False
            
            result = runner.invoke(app, ["config", "validate"])
            assert result.exit_code == 1
            assert "Configuration validation failed" in result.output


class TestAgentCommands:
    """Test agent commands."""
    
    def test_agent_list_command(self):
        """Test agent list command."""
        result = runner.invoke(app, ["agent", "list"])
        assert result.exit_code == 0
        assert "Available Agents" in result.output
        assert "orchestrator" in result.output
        assert "user-twin" in result.output
        assert "honest-vendor" in result.output
        assert "malicious-vendor" in result.output
    
    def test_agent_status_command(self):
        """Test agent status command."""
        result = runner.invoke(app, ["agent", "status"])
        assert result.exit_code == 0
        assert "All Agents Status" in result.output
    
    def test_agent_status_command_specific_agent(self):
        """Test agent status command for specific agent."""
        result = runner.invoke(app, ["agent", "status", "--agent", "orchestrator"])
        assert result.exit_code == 0
        assert "Agent Status: orchestrator" in result.output


class TestGuardCommands:
    """Test guard commands."""
    
    def test_guard_list_command(self):
        """Test guard list command."""
        result = runner.invoke(app, ["guard", "list"])
        assert result.exit_code == 0
        assert "Available Security Guards" in result.output
        assert "privacy-sentry" in result.output
        assert "task-navigator" in result.output
        assert "prompt-sanitizer" in result.output
        assert "honeypot-guard" in result.output
    
    def test_guard_status_command(self):
        """Test guard status command."""
        result = runner.invoke(app, ["guard", "status"])
        assert result.exit_code == 0
        assert "All Guards Status" in result.output
    
    def test_guard_status_command_specific_guard(self):
        """Test guard status command for specific guard."""
        result = runner.invoke(app, ["guard", "status", "--guard", "privacy-sentry"])
        assert result.exit_code == 0
        assert "Guard Status: privacy-sentry" in result.output


class TestMetricsCommands:
    """Test metrics commands."""
    
    def test_metrics_show_command(self):
        """Test metrics show command."""
        result = runner.invoke(app, ["metrics", "show"])
        assert result.exit_code == 0
        assert "System Metrics" in result.output
    
    def test_metrics_show_command_with_options(self):
        """Test metrics show command with options."""
        result = runner.invoke(app, [
            "metrics", "show", 
            "--format", "json",
            "--period", "2h"
        ])
        assert result.exit_code == 0
        assert "System Metrics (Last 2h)" in result.output
    
    def test_metrics_export_command(self):
        """Test metrics export command."""
        result = runner.invoke(app, ["metrics", "export"])
        assert result.exit_code == 0
        assert "Exporting metrics to metrics.json" in result.output
    
    def test_metrics_export_command_with_options(self):
        """Test metrics export command with options."""
        result = runner.invoke(app, [
            "metrics", "export",
            "--output", "custom_metrics.csv",
            "--format", "csv"
        ])
        assert result.exit_code == 0
        assert "Exporting metrics to custom_metrics.csv" in result.output


class TestSystemRequirements:
    """Test system requirements checking."""
    
    @patch('cli.check_ollama_connection')
    @patch('cli.ensure_model_available')
    def test_check_system_requirements_success(self, mock_ensure_model, mock_check_ollama):
        """Test successful system requirements check."""
        mock_check_ollama.return_value = True
        mock_ensure_model.return_value = True
        
        result = check_system_requirements()
        assert result is True
    
    @patch('cli.check_ollama_connection')
    def test_check_system_requirements_ollama_failed(self, mock_check_ollama):
        """Test system requirements check when Ollama fails."""
        mock_check_ollama.return_value = False
        
        result = check_system_requirements()
        assert result is False
    
    @patch('cli.check_ollama_connection')
    @patch('cli.ensure_model_available')
    def test_check_system_requirements_model_failed(self, mock_ensure_model, mock_check_ollama):
        """Test system requirements check when model is not available."""
        mock_check_ollama.return_value = True
        mock_ensure_model.return_value = False
        
        result = check_system_requirements()
        assert result is False


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def test_help_command(self):
        """Test help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "SafeHive AI Security Sandbox" in result.output
    
    def test_invalid_command(self):
        """Test invalid command."""
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code == 2  # Typer returns 2 for invalid commands
    
    def test_sandbox_help(self):
        """Test sandbox help."""
        result = runner.invoke(app, ["sandbox", "--help"])
        assert result.exit_code == 0
        assert "Sandbox operations" in result.output
    
    def test_config_help(self):
        """Test config help."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "Configuration management" in result.output


class TestCLIIntegration:
    """Test CLI integration scenarios."""
    
    def test_full_workflow_simulation(self):
        """Test a full workflow simulation."""
        with patch('cli.check_ollama_connection') as mock_ollama, \
             patch('cli.ensure_model_available') as mock_model, \
             patch('cli.setup_logging') as mock_logging, \
             patch('cli.ConfigLoader') as mock_config:
            
            # Setup mocks
            mock_ollama.return_value = True
            mock_model.return_value = True
            mock_config.return_value.load_config.return_value = True
            
            # Test initialization
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0
            
            # Test status check
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0
            
            # Test sandbox list
            result = runner.invoke(app, ["sandbox", "list"])
            assert result.exit_code == 0
            
            # Test agent list
            result = runner.invoke(app, ["agent", "list"])
            assert result.exit_code == 0
            
            # Test guard list
            result = runner.invoke(app, ["guard", "list"])
            assert result.exit_code == 0
    
    def test_configuration_workflow(self):
        """Test configuration workflow."""
        with patch('cli.ConfigLoader') as mock_config_loader:
            mock_config = MagicMock()
            mock_config.__dict__ = {
                "guards": {"privacy_sentry": {"enabled": True}},
                "agents": {"orchestrator": {"model": "llama3.2:3b"}}
            }
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_config.return_value = True
            mock_loader_instance.config = mock_config
            mock_config_loader.return_value = mock_loader_instance
            
            # Test config validation
            result = runner.invoke(app, ["config", "validate"])
            assert result.exit_code == 0
            
            # Test config show
            result = runner.invoke(app, ["config", "show"])
            assert result.exit_code == 0
            
            # Test config show with section
            result = runner.invoke(app, ["config", "show", "--section", "guards"])
            assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__])
