"""
Unit tests for the Interactive Menu System

This module contains comprehensive tests for the InteractiveMenu class and
related functionality in the SafeHive UI package.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

from safehive.ui.interactive_menu import InteractiveMenu


class TestInteractiveMenu:
    """Test cases for InteractiveMenu class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.menu = InteractiveMenu()

    def test_menu_initialization(self):
        """Test menu initialization."""
        assert self.menu.console is not None
        assert self.menu.config_loader is not None
        assert self.menu.running is True
        assert self.menu.current_menu == "main"
        assert self.menu.menu_stack == []
        assert self.menu.session_data == {}
        assert "main" in self.menu.menus

    def test_get_main_menu(self):
        """Test main menu configuration."""
        menu_config = self.menu._get_main_menu()
        
        assert menu_config["title"] == "🛡️ SafeHive AI Security Sandbox"
        assert menu_config["subtitle"] == "Interactive Security Testing Platform"
        assert len(menu_config["options"]) > 0
        
        # Check for key options
        option_keys = [opt["key"] for opt in menu_config["options"]]
        assert "1" in option_keys  # Sandbox Operations
        assert "q" in option_keys  # Quit
        assert "i" in option_keys  # System Information

    def test_get_sandbox_menu(self):
        """Test sandbox menu configuration."""
        menu_config = self.menu._get_sandbox_menu()
        
        assert menu_config["title"] == "🎯 Sandbox Operations"
        assert menu_config["subtitle"] == "Security Testing Session Management"
        assert len(menu_config["options"]) > 0
        
        # Check for key options
        option_keys = [opt["key"] for opt in menu_config["options"]]
        assert "1" in option_keys  # Start New Session
        assert "b" in option_keys  # Back to Main Menu
        assert "q" in option_keys  # Quit

    def test_get_configuration_menu(self):
        """Test configuration menu."""
        menu_config = self.menu._get_configuration_menu()
        
        assert menu_config["title"] == "⚙️ Configuration Management"
        assert menu_config["subtitle"] == "System Settings and Configuration"
        assert len(menu_config["options"]) > 0

    def test_get_agents_menu(self):
        """Test agents menu."""
        menu_config = self.menu._get_agents_menu()
        
        assert menu_config["title"] == "🤖 Agent Management"
        assert menu_config["subtitle"] == "AI Agent Monitoring and Control"
        assert len(menu_config["options"]) > 0

    def test_get_guards_menu(self):
        """Test guards menu."""
        menu_config = self.menu._get_guards_menu()
        
        assert menu_config["title"] == "🛡️ Security Guards"
        assert menu_config["subtitle"] == "Security Guard Monitoring and Configuration"
        assert len(menu_config["options"]) > 0

    def test_get_metrics_menu(self):
        """Test metrics menu."""
        menu_config = self.menu._get_metrics_menu()
        
        assert menu_config["title"] == "📊 Metrics & Monitoring"
        assert menu_config["subtitle"] == "System Performance and Security Metrics"
        assert len(menu_config["options"]) > 0

    def test_get_system_menu(self):
        """Test system menu."""
        menu_config = self.menu._get_system_menu()
        
        assert menu_config["title"] == "🔧 System Status"
        assert menu_config["subtitle"] == "System Health and Requirements"
        assert len(menu_config["options"]) > 0

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    def test_display_menu(self, mock_print, mock_clear):
        """Test menu display functionality."""
        menu_config = self.menu._get_main_menu()
        
        self.menu._display_menu(menu_config)
        
        # Verify clear was called
        mock_clear.assert_called_once()
        
        # Verify print was called multiple times
        assert mock_print.call_count > 0

    @patch('safehive.ui.interactive_menu.Prompt.ask')
    def test_get_user_choice(self, mock_prompt):
        """Test user choice input."""
        mock_prompt.return_value = "1"
        
        choice = self.menu._get_user_choice()
        
        assert choice == "1"
        mock_prompt.assert_called_once()

    @patch('safehive.ui.interactive_menu.Prompt.ask')
    def test_get_user_choice_empty_input(self, mock_prompt):
        """Test user choice with empty input."""
        mock_prompt.side_effect = ["", "1"]  # First empty, then valid
        
        choice = self.menu._get_user_choice()
        
        assert choice == "1"
        assert mock_prompt.call_count == 2

    def test_handle_menu_action_navigate(self):
        """Test navigation action."""
        self.menu._handle_menu_action("navigate", "sandbox")
        
        assert self.menu.current_menu == "sandbox"
        assert "main" in self.menu.menu_stack

    def test_handle_menu_action_back(self):
        """Test back action."""
        # Set up menu stack
        self.menu.menu_stack = ["main"]
        self.menu.current_menu = "sandbox"
        
        self.menu._handle_menu_action("back")
        
        assert self.menu.current_menu == "main"
        assert self.menu.menu_stack == []

    def test_handle_menu_action_back_empty_stack(self):
        """Test back action with empty stack."""
        self.menu.current_menu = "sandbox"
        
        self.menu._handle_menu_action("back")
        
        assert self.menu.current_menu == "main"

    def test_handle_menu_action_quit(self):
        """Test quit action."""
        self.menu._handle_menu_action("quit")
        
        assert self.menu.running is False

    def test_handle_menu_action_unknown(self):
        """Test unknown action."""
        with patch.object(self.menu, '_wait_for_key') as mock_wait:
            self.menu._handle_menu_action("unknown_action")
            mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Prompt.ask')
    def test_wait_for_key(self, mock_prompt):
        """Test wait for key functionality."""
        mock_prompt.return_value = ""
        
        self.menu._wait_for_key()
        
        mock_prompt.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_system_info(self, mock_wait, mock_print, mock_clear):
        """Test system info display."""
        self.menu._show_system_info()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch('safehive.ui.interactive_menu.IntPrompt.ask')
    @patch('safehive.ui.interactive_menu.Confirm.ask')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_start_sandbox_session(self, mock_wait, mock_confirm, mock_int_prompt, mock_print, mock_clear):
        """Test sandbox session start."""
        mock_int_prompt.side_effect = [1, 300]  # scenario choice, duration
        mock_confirm.return_value = True
        
        self.menu._start_sandbox_session()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_list_scenarios(self, mock_wait, mock_print, mock_clear):
        """Test scenario listing."""
        self.menu._list_scenarios()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_stop_sandbox_session_no_session(self, mock_wait, mock_print, mock_clear):
        """Test stopping sandbox session when no session is active."""
        self.menu.current_session = None
        
        self.menu._stop_sandbox_session()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_stop_sandbox_session_with_session(self, mock_wait, mock_print, mock_clear):
        """Test stopping sandbox session when session is active."""
        self.menu.current_session = {"id": "test_session"}
        
        self.menu._stop_sandbox_session()
        
        assert self.menu.current_session is None
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_session_status_no_session(self, mock_wait, mock_print, mock_clear):
        """Test showing session status when no session is active."""
        self.menu.current_session = None
        
        self.menu._show_session_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_session_status_with_session(self, mock_wait, mock_print, mock_clear):
        """Test showing session status when session is active."""
        self.menu.current_session = {"id": "test_session"}
        
        self.menu._show_session_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_session_history(self, mock_wait, mock_print, mock_clear):
        """Test showing session history."""
        self.menu._show_session_history()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch('safehive.ui.interactive_menu.Prompt.ask')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_view_configuration_success(self, mock_wait, mock_prompt, mock_print, mock_clear):
        """Test viewing configuration successfully."""
        # Create a simple mock config object
        class MockConfig:
            def __init__(self):
                self.guards = "test_guards"
                self.agents = "test_agents"
            
            def __dict__(self):
                return {"guards": "test_guards", "agents": "test_agents"}
        
        mock_config = MockConfig()
        
        with patch.object(self.menu.config_loader, 'load_config', return_value=mock_config):
            mock_prompt.return_value = "guards"
            
            self.menu._view_configuration()
            
            mock_clear.assert_called_once()
            assert mock_print.call_count > 0
            mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_view_configuration_error(self, mock_wait, mock_print, mock_clear):
        """Test viewing configuration with error."""
        with patch.object(self.menu.config_loader, 'load_config', side_effect=Exception("Config error")):
            self.menu._view_configuration()
            
            mock_clear.assert_called_once()
            assert mock_print.call_count > 0
            mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_validate_configuration_success(self, mock_wait, mock_print, mock_clear):
        """Test configuration validation success."""
        mock_config = Mock()
        
        with patch.object(self.menu.config_loader, 'load_config', return_value=mock_config):
            self.menu._validate_configuration()
            
            mock_clear.assert_called_once()
            assert mock_print.call_count > 0
            mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_validate_configuration_error(self, mock_wait, mock_print, mock_clear):
        """Test configuration validation error."""
        with patch.object(self.menu.config_loader, 'load_config', side_effect=Exception("Config error")):
            self.menu._validate_configuration()
            
            mock_clear.assert_called_once()
            assert mock_print.call_count > 0
            mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_guard_settings(self, mock_wait, mock_print, mock_clear):
        """Test showing guard settings."""
        self.menu._show_guard_settings()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_agent_settings(self, mock_wait, mock_print, mock_clear):
        """Test showing agent settings."""
        self.menu._show_agent_settings()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_logging_settings(self, mock_wait, mock_print, mock_clear):
        """Test showing logging settings."""
        self.menu._show_logging_settings()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_list_agents(self, mock_wait, mock_print, mock_clear):
        """Test listing agents."""
        self.menu._list_agents()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_agent_status(self, mock_wait, mock_print, mock_clear):
        """Test showing agent status."""
        self.menu._show_agent_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_orchestrator_status(self, mock_wait, mock_print, mock_clear):
        """Test showing orchestrator status."""
        self.menu._show_orchestrator_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_user_twin_status(self, mock_wait, mock_print, mock_clear):
        """Test showing user twin status."""
        self.menu._show_user_twin_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_vendor_agents_status(self, mock_wait, mock_print, mock_clear):
        """Test showing vendor agents status."""
        self.menu._show_vendor_agents_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_agent_memory(self, mock_wait, mock_print, mock_clear):
        """Test showing agent memory."""
        self.menu._show_agent_memory()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_list_guards(self, mock_wait, mock_print, mock_clear):
        """Test listing guards."""
        self.menu._list_guards()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_guard_status(self, mock_wait, mock_print, mock_clear):
        """Test showing guard status."""
        self.menu._show_guard_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_privacy_sentry_status(self, mock_wait, mock_print, mock_clear):
        """Test showing privacy sentry status."""
        self.menu._show_privacy_sentry_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_task_navigator_status(self, mock_wait, mock_print, mock_clear):
        """Test showing task navigator status."""
        self.menu._show_task_navigator_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_prompt_sanitizer_status(self, mock_wait, mock_print, mock_clear):
        """Test showing prompt sanitizer status."""
        self.menu._show_prompt_sanitizer_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_mcp_server_status(self, mock_wait, mock_print, mock_clear):
        """Test showing MCP server status."""
        self.menu._show_mcp_server_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_recent_alerts(self, mock_wait, mock_print, mock_clear):
        """Test showing recent alerts."""
        self.menu._show_recent_alerts()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_system_metrics(self, mock_wait, mock_print, mock_clear):
        """Test showing system metrics."""
        self.menu._show_system_metrics()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_security_metrics(self, mock_wait, mock_print, mock_clear):
        """Test showing security metrics."""
        self.menu._show_security_metrics()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_agent_metrics(self, mock_wait, mock_print, mock_clear):
        """Test showing agent metrics."""
        self.menu._show_agent_metrics()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_log_analysis(self, mock_wait, mock_print, mock_clear):
        """Test showing log analysis."""
        self.menu._show_log_analysis()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_export_metrics(self, mock_wait, mock_print, mock_clear):
        """Test exporting metrics."""
        self.menu._export_metrics()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_realtime_dashboard(self, mock_wait, mock_print, mock_clear):
        """Test showing real-time dashboard."""
        self.menu._show_realtime_dashboard()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch('safehive.ui.interactive_menu.check_ollama_connection')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_system_health(self, mock_wait, mock_ollama_check, mock_print, mock_clear):
        """Test showing system health."""
        mock_ollama_check.return_value = True
        
        with patch.object(self.menu.config_loader, 'load_config', return_value=Mock()):
            self.menu._show_system_health()
            
            mock_clear.assert_called_once()
            assert mock_print.call_count > 0
            mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch('safehive.ui.interactive_menu.check_ollama_connection')
    @patch('safehive.ui.interactive_menu.ensure_model_available')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_ollama_status_running(self, mock_wait, mock_model_check, mock_ollama_check, mock_print, mock_clear):
        """Test showing Ollama status when running."""
        mock_ollama_check.return_value = True
        mock_model_check.return_value = True
        
        self.menu._show_ollama_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch('safehive.ui.interactive_menu.check_ollama_connection')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_ollama_status_not_running(self, mock_wait, mock_ollama_check, mock_print, mock_clear):
        """Test showing Ollama status when not running."""
        mock_ollama_check.return_value = False
        
        self.menu._show_ollama_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_dependencies_status(self, mock_wait, mock_print, mock_clear):
        """Test showing dependencies status."""
        self.menu._show_dependencies_status()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_disk_space(self, mock_wait, mock_print, mock_clear):
        """Test showing disk space."""
        self.menu._show_disk_space()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_system_logs(self, mock_wait, mock_print, mock_clear):
        """Test showing system logs."""
        self.menu._show_system_logs()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_show_system_info(self, mock_wait, mock_print, mock_clear):
        """Test showing system information."""
        self.menu._show_system_info()
        
        mock_clear.assert_called_once()
        assert mock_print.call_count > 0
        mock_wait.assert_called_once()

    @patch.object(InteractiveMenu, '_display_menu')
    @patch.object(InteractiveMenu, '_get_user_choice')
    @patch.object(InteractiveMenu, '_handle_menu_action')
    def test_run_main_loop(self, mock_handle_action, mock_get_choice, mock_display_menu):
        """Test main menu loop."""
        # Set up mocks to exit after first iteration
        mock_get_choice.return_value = "q"
        mock_handle_action.side_effect = lambda action, target=None: setattr(self.menu, 'running', False)
        
        self.menu.run()
        
        mock_display_menu.assert_called()
        mock_get_choice.assert_called()
        mock_handle_action.assert_called()

    @patch.object(InteractiveMenu, '_display_menu')
    @patch.object(InteractiveMenu, '_get_user_choice')
    @patch.object(InteractiveMenu, '_wait_for_key')
    def test_run_invalid_choice(self, mock_wait, mock_get_choice, mock_display_menu):
        """Test handling invalid choice."""
        # Set up mocks to exit after invalid choice
        mock_get_choice.return_value = "invalid"
        mock_wait.side_effect = lambda: setattr(self.menu, 'running', False)
        
        self.menu.run()
        
        mock_display_menu.assert_called()
        mock_get_choice.assert_called()
        mock_wait.assert_called()

    @patch.object(InteractiveMenu, '_display_menu')
    @patch.object(InteractiveMenu, '_get_user_choice')
    def test_run_keyboard_interrupt(self, mock_get_choice, mock_display_menu):
        """Test handling keyboard interrupt."""
        mock_get_choice.side_effect = KeyboardInterrupt()
        
        with patch.object(self.menu.console, 'print') as mock_print:
            self.menu.run()
            
            mock_print.assert_called_with("\n[yellow]👋 Goodbye![/yellow]")

    @patch.object(InteractiveMenu, '_display_menu')
    @patch.object(InteractiveMenu, '_get_user_choice')
    def test_run_exception(self, mock_get_choice, mock_display_menu):
        """Test handling general exception."""
        mock_get_choice.side_effect = Exception("Test error")
        
        with patch.object(self.menu.console, 'print') as mock_print:
            with patch('safehive.ui.interactive_menu.logger') as mock_logger:
                self.menu.run()
                
                mock_print.assert_called()
                mock_logger.error.assert_called()


class TestInteractiveMenuIntegration:
    """Integration tests for InteractiveMenu."""

    def setup_method(self):
        """Set up test fixtures."""
        self.menu = InteractiveMenu()

    @patch('safehive.ui.interactive_menu.Console.clear')
    @patch('safehive.ui.interactive_menu.Console.print')
    @patch('safehive.ui.interactive_menu.Prompt.ask')
    def test_menu_navigation_flow(self, mock_prompt, mock_print, mock_clear):
        """Test complete menu navigation flow."""
        # Mock user choices: go to sandbox, then back, then quit
        mock_prompt.side_effect = ["1", "b", "q"]
        
        # Mock menu actions to exit after first call
        with patch.object(self.menu, '_handle_menu_action') as mock_handle:
            def mock_action(action, target=None):
                if action == "navigate" and target == "sandbox":
                    self.menu.current_menu = "sandbox"
                elif action == "back":
                    self.menu.current_menu = "main"
                elif action == "quit":
                    self.menu.running = False
            
            mock_handle.side_effect = mock_action
            
            self.menu.run()
            
            # Verify navigation occurred
            assert mock_handle.call_count >= 1

    def test_menu_configuration_consistency(self):
        """Test that all menu configurations are consistent."""
        menu_names = ["main", "sandbox", "configuration", "agents", "guards", "metrics", "system"]
        
        for menu_name in menu_names:
            menu_config = self.menu.menus[menu_name]()
            
            # Check required fields
            assert "title" in menu_config
            assert "subtitle" in menu_config
            assert "options" in menu_config
            assert isinstance(menu_config["options"], list)
            assert len(menu_config["options"]) > 0
            
            # Check that all options have required fields
            for option in menu_config["options"]:
                assert "key" in option
                assert "label" in option
                assert "description" in option
                assert "action" in option
            
            # Check that back and quit options exist
            option_keys = [opt["key"] for opt in menu_config["options"]]
            if menu_name != "main":
                assert "b" in option_keys  # Back option
            assert "q" in option_keys  # Quit option

    def test_menu_action_handlers_exist(self):
        """Test that all menu actions have corresponding handlers."""
        all_actions = set()
        
        # Collect all actions from all menus
        for menu_name in self.menu.menus:
            menu_config = self.menu.menus[menu_name]()
            for option in menu_config["options"]:
                all_actions.add(option["action"])
        
        # Check that all actions have handlers
        for action in all_actions:
            if action in ["navigate", "back", "quit", "show_info"]:
                continue  # These are handled in _handle_menu_action directly
            
            # Check if method exists - handle special cases
            if action == "start_session":
                method_name = "_start_sandbox_session"
            elif action == "stop_session":
                method_name = "_stop_sandbox_session"
            elif action == "session_status":
                method_name = "_show_session_status"
            elif action == "session_history":
                method_name = "_show_session_history"
            elif action == "list_scenarios":
                method_name = "_list_scenarios"
            elif action == "view_config":
                method_name = "_view_configuration"
            elif action == "validate_config":
                method_name = "_validate_configuration"
            elif action == "guard_settings":
                method_name = "_show_guard_settings"
            elif action == "agent_settings":
                method_name = "_show_agent_settings"
            elif action == "logging_settings":
                method_name = "_show_logging_settings"
            elif action == "list_agents":
                method_name = "_list_agents"
            elif action == "agent_status":
                method_name = "_show_agent_status"
            elif action == "orchestrator_status":
                method_name = "_show_orchestrator_status"
            elif action == "user_twin_status":
                method_name = "_show_user_twin_status"
            elif action == "vendor_agents_status":
                method_name = "_show_vendor_agents_status"
            elif action == "agent_memory":
                method_name = "_show_agent_memory"
            elif action == "list_guards":
                method_name = "_list_guards"
            elif action == "guard_status":
                method_name = "_show_guard_status"
            elif action == "privacy_sentry_status":
                method_name = "_show_privacy_sentry_status"
            elif action == "task_navigator_status":
                method_name = "_show_task_navigator_status"
            elif action == "prompt_sanitizer_status":
                method_name = "_show_prompt_sanitizer_status"
            elif action == "mcp_server_status":
                method_name = "_show_mcp_server_status"
            elif action == "recent_alerts":
                method_name = "_show_recent_alerts"
            elif action == "system_metrics":
                method_name = "_show_system_metrics"
            elif action == "security_metrics":
                method_name = "_show_security_metrics"
            elif action == "agent_metrics":
                method_name = "_show_agent_metrics"
            elif action == "log_analysis":
                method_name = "_show_log_analysis"
            elif action == "export_metrics":
                method_name = "_export_metrics"
            elif action == "realtime_dashboard":
                method_name = "_show_realtime_dashboard"
            elif action == "system_health":
                method_name = "_show_system_health"
            elif action == "ollama_status":
                method_name = "_show_ollama_status"
            elif action == "dependencies_status":
                method_name = "_show_dependencies_status"
            elif action == "disk_space":
                method_name = "_show_disk_space"
            elif action == "system_logs":
                method_name = "_show_system_logs"
            elif action == "system_info":
                method_name = "_show_system_info"
            else:
                # Default pattern
                method_name = f"_show_{action}" if action.startswith("show_") else f"_{action}"
            
            assert hasattr(self.menu, method_name), f"Missing handler for action: {action} -> {method_name}"
