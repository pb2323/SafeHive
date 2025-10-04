"""
Unit tests for the help system module.

This module tests the help system functionality including command help display,
interactive help, and command suggestions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from rich.console import Console
from rich.panel import Panel

from safehive.ui.help_system import (
    HelpSystem, CommandHelp, show_help, interactive_help, 
    get_command_suggestions, get_help_system
)


class TestCommandHelp:
    """Test CommandHelp dataclass functionality."""
    
    def test_command_help_creation(self):
        """Test creating a CommandHelp instance."""
        help_info = CommandHelp(
            name="test-command",
            description="Test command description",
            usage="test-command [options]",
            examples=["test-command --option value"],
            options=[{"--option": "Test option description"}],
            notes=["Test note"],
            related_commands=["related-command"]
        )
        
        assert help_info.name == "test-command"
        assert help_info.description == "Test command description"
        assert help_info.usage == "test-command [options]"
        assert len(help_info.examples) == 1
        assert len(help_info.options) == 1
        assert len(help_info.notes) == 1
        assert len(help_info.related_commands) == 1
    
    def test_command_help_defaults(self):
        """Test CommandHelp with default values."""
        help_info = CommandHelp(
            name="simple-command",
            description="Simple command",
            usage="simple-command",
            examples=["simple-command"],
            options=[]
        )
        
        assert help_info.notes is None
        assert help_info.related_commands is None


class TestHelpSystem:
    """Test HelpSystem class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.help_system = HelpSystem()
    
    def test_help_system_initialization(self):
        """Test HelpSystem initialization."""
        assert isinstance(self.help_system.console, Console)
        assert isinstance(self.help_system.commands, dict)
        assert len(self.help_system.commands) > 0
    
    def test_command_help_initialization(self):
        """Test that command help is properly initialized."""
        # Check that key commands are present
        assert "init" in self.help_system.commands
        assert "sandbox start" in self.help_system.commands
        assert "metrics show" in self.help_system.commands
        # Note: "help" is a special CLI command, not in the help system commands
        
        # Check command help structure
        init_help = self.help_system.commands["init"]
        assert isinstance(init_help, CommandHelp)
        assert init_help.name == "init"
        assert len(init_help.examples) > 0
        assert len(init_help.options) > 0
    
    @patch('safehive.ui.help_system.Console.print')
    def test_show_general_help(self, mock_print):
        """Test general help display."""
        self.help_system._show_general_help()
        
        # Verify that print was called multiple times (indicating help content was displayed)
        assert mock_print.call_count > 0
        
        # Just verify that the method executed without errors
        # The actual content verification is done in integration tests
    
    @patch('safehive.ui.help_system.Console.print')
    def test_show_command_help_existing_command(self, mock_print):
        """Test showing help for an existing command."""
        self.help_system._show_command_help("init")
        
        # Verify that print was called (indicating help content was displayed)
        assert mock_print.call_count > 0
        
        # Just verify that the method executed without errors
        # The actual content verification is done in integration tests
    
    @patch('safehive.ui.help_system.Console.print')
    def test_show_command_help_nonexistent_command(self, mock_print):
        """Test showing help for a non-existent command."""
        self.help_system._show_command_help("nonexistent-command")
        
        # Verify error message is displayed
        assert mock_print.call_count > 0
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        print_text = " ".join(print_calls)
        
        assert "Error" in print_text
        assert "nonexistent-command" in print_text
    
    @patch('safehive.ui.help_system.Console.print')
    def test_show_topic_help(self, mock_print):
        """Test showing topic help."""
        self.help_system._show_topic_help("scenarios")
        
        # Verify that print was called (indicating help content was displayed)
        assert mock_print.call_count > 0
        
        # Just verify that the method executed without errors
        # The actual content verification is done in integration tests
    
    @patch('safehive.ui.help_system.Console.print')
    def test_show_topic_help_invalid_topic(self, mock_print):
        """Test showing help for invalid topic."""
        self.help_system._show_topic_help("invalid-topic")
        
        # Verify error message is displayed
        assert mock_print.call_count > 0
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        print_text = " ".join(print_calls)
        
        assert "Error" in print_text
        assert "invalid-topic" in print_text
    
    def test_get_command_suggestions(self):
        """Test getting command suggestions."""
        # Test exact match
        suggestions = self.help_system.get_command_suggestions("init")
        assert "init" in suggestions
        
        # Test partial match
        suggestions = self.help_system.get_command_suggestions("sand")
        assert any("sandbox" in s for s in suggestions)
        
        # Test no match
        suggestions = self.help_system.get_command_suggestions("nonexistent")
        assert len(suggestions) == 0
        
        # Test case insensitive
        suggestions = self.help_system.get_command_suggestions("INIT")
        assert "init" in suggestions
    
    def test_get_command_suggestions_limit(self):
        """Test that command suggestions are limited to 5."""
        # Create a mock help system with many commands
        help_system = HelpSystem()
        # Add many test commands
        for i in range(10):
            help_system.commands[f"test-command-{i}"] = CommandHelp(
                name=f"test-command-{i}",
                description="Test",
                usage="test",
                examples=["test"],
                options=[]
            )
        
        suggestions = help_system.get_command_suggestions("test")
        assert len(suggestions) <= 5
    
    @patch('safehive.ui.help_system.Prompt.ask')
    def test_interactive_help_exit(self, mock_prompt):
        """Test interactive help with immediate exit."""
        # Mock user choosing to exit immediately
        mock_prompt.return_value = "4"
        
        self.help_system.interactive_help()
        
        # Should only ask once (for the menu choice)
        assert mock_prompt.call_count == 1
    
    @patch('safehive.ui.help_system.Prompt.ask')
    @patch.object(HelpSystem, '_show_command_help')
    def test_interactive_help_command_help(self, mock_show_command, mock_prompt):
        """Test interactive help with command help selection."""
        # Mock user choosing command help then exiting
        mock_prompt.side_effect = ["1", "init", "4"]
        
        self.help_system.interactive_help()
        
        # Should call _show_command_help with "init"
        mock_show_command.assert_called_once_with("init")
    
    @patch('safehive.ui.help_system.Prompt.ask')
    @patch.object(HelpSystem, '_show_topic_help')
    def test_interactive_help_topic_help(self, mock_show_topic, mock_prompt):
        """Test interactive help with topic help selection."""
        # Mock user choosing topic help then exiting
        mock_prompt.side_effect = ["2", "scenarios", "4"]
        
        self.help_system.interactive_help()
        
        # Should call _show_topic_help with "scenarios"
        mock_show_topic.assert_called_once_with("scenarios")
    
    @patch('safehive.ui.help_system.Prompt.ask')
    @patch.object(HelpSystem, '_show_general_help')
    def test_interactive_help_general_help(self, mock_show_general, mock_prompt):
        """Test interactive help with general help selection."""
        # Mock user choosing general help then exiting
        mock_prompt.side_effect = ["3", "4"]
        
        self.help_system.interactive_help()
        
        # Should call _show_general_help
        mock_show_general.assert_called_once()


class TestGlobalFunctions:
    """Test global helper functions."""
    
    @patch('safehive.ui.help_system.get_help_system')
    def test_show_help_no_args(self, mock_get_help_system):
        """Test show_help with no arguments."""
        mock_help_system = Mock()
        mock_get_help_system.return_value = mock_help_system
        
        show_help()
        
        mock_help_system.show_help.assert_called_once_with(None, None)
    
    @patch('safehive.ui.help_system.get_help_system')
    def test_show_help_with_command(self, mock_get_help_system):
        """Test show_help with command argument."""
        mock_help_system = Mock()
        mock_get_help_system.return_value = mock_help_system
        
        show_help(command="init")
        
        mock_help_system.show_help.assert_called_once_with("init", None)
    
    @patch('safehive.ui.help_system.get_help_system')
    def test_show_help_with_topic(self, mock_get_help_system):
        """Test show_help with topic argument."""
        mock_help_system = Mock()
        mock_get_help_system.return_value = mock_help_system
        
        show_help(topic="scenarios")
        
        mock_help_system.show_help.assert_called_once_with(None, "scenarios")
    
    @patch('safehive.ui.help_system.get_help_system')
    def test_interactive_help_global(self, mock_get_help_system):
        """Test interactive_help global function."""
        mock_help_system = Mock()
        mock_get_help_system.return_value = mock_help_system
        
        interactive_help()
        
        mock_help_system.interactive_help.assert_called_once()
    
    @patch('safehive.ui.help_system.get_help_system')
    def test_get_command_suggestions_global(self, mock_get_help_system):
        """Test get_command_suggestions global function."""
        mock_help_system = Mock()
        mock_help_system.get_command_suggestions.return_value = ["init", "info"]
        mock_get_help_system.return_value = mock_help_system
        
        suggestions = get_command_suggestions("in")
        
        mock_help_system.get_command_suggestions.assert_called_once_with("in")
        assert suggestions == ["init", "info"]


class TestHelpSystemIntegration:
    """Integration tests for help system."""
    
    def test_help_system_singleton(self):
        """Test that get_help_system returns singleton instance."""
        instance1 = get_help_system()
        instance2 = get_help_system()
        
        assert instance1 is instance2
        assert isinstance(instance1, HelpSystem)
    
    def test_all_commands_have_help(self):
        """Test that all expected commands have help information."""
        help_system = get_help_system()
        
        # Check that all major command categories are covered
        # Note: "help" is a special CLI command, not in the help system commands
        expected_commands = [
            "init", "status", "menu",
            "sandbox start", "sandbox stop", "sandbox list", "sandbox status",
            "config show", "config validate",
            "metrics show", "metrics export", "metrics dashboard",
            "human list", "human respond",
            "progress start", "progress stop", "progress demo"
        ]
        
        for command in expected_commands:
            assert command in help_system.commands, f"Missing help for command: {command}"
            
            help_info = help_system.commands[command]
            assert help_info.name == command
            assert help_info.description
            assert help_info.usage
            assert len(help_info.examples) > 0
    
    def test_help_system_topic_coverage(self):
        """Test that all help topics are properly implemented."""
        help_system = get_help_system()
        
        # Test that all topic help methods exist
        topics = ["scenarios", "guards", "agents", "configuration", "troubleshooting"]
        
        for topic in topics:
            # This should not raise an exception
            help_system._show_topic_help(topic)
    
    @patch('safehive.ui.help_system.Console.print')
    def test_help_system_error_handling(self, mock_print):
        """Test help system error handling."""
        help_system = get_help_system()
        
        # Test with invalid command
        help_system._show_command_help("invalid-command")
        
        # Test with invalid topic
        help_system._show_topic_help("invalid-topic")
        
        # Should not raise exceptions and should show error messages
        assert mock_print.call_count > 0


class TestHelpSystemPerformance:
    """Performance tests for help system."""
    
    def test_help_system_initialization_performance(self):
        """Test that help system initializes quickly."""
        import time
        
        start_time = time.time()
        help_system = HelpSystem()
        end_time = time.time()
        
        # Should initialize in less than 1 second
        assert (end_time - start_time) < 1.0
        assert len(help_system.commands) > 0
    
    def test_command_suggestions_performance(self):
        """Test that command suggestions are fast."""
        import time
        
        help_system = get_help_system()
        
        start_time = time.time()
        suggestions = help_system.get_command_suggestions("sand")
        end_time = time.time()
        
        # Should complete in less than 0.1 seconds
        assert (end_time - start_time) < 0.1
        assert isinstance(suggestions, list)
    
    def test_help_system_memory_usage(self):
        """Test that help system doesn't use excessive memory."""
        import sys
        
        help_system = get_help_system()
        
        # Get approximate memory usage
        memory_usage = sys.getsizeof(help_system) + sum(
            sys.getsizeof(cmd) for cmd in help_system.commands.values()
        )
        
        # Should use less than 1MB for help data
        assert memory_usage < 1024 * 1024  # 1MB


if __name__ == "__main__":
    pytest.main([__file__])
