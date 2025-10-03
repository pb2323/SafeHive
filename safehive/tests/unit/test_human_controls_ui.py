"""
Unit tests for human-in-the-loop controls UI.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from safehive.models.human_controls import (
    InterventionRequest, InterventionResponse, HumanControlSession,
    InterventionType, InterventionStatus, IncidentType, IncidentSeverity
)
from safehive.ui.human_controls_ui import (
    HumanControlsUI, get_human_controls_ui,
    display_pending_requests, prompt_for_intervention,
    display_statistics, interactive_monitor
)


class TestHumanControlsUI:
    """Test HumanControlsUI class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ui = HumanControlsUI()
        self.mock_console = MagicMock()
        self.ui.console = self.mock_console
    
    def test_ui_initialization(self):
        """Test UI initialization."""
        ui = HumanControlsUI()
        
        assert ui.console is not None
        assert ui.manager is not None
    
    @patch('safehive.ui.human_controls_ui.get_human_control_manager')
    def test_display_pending_requests_empty(self, mock_manager):
        """Test displaying empty pending requests."""
        mock_manager.return_value.get_pending_requests.return_value = []
        
        ui = HumanControlsUI()
        ui.console = MagicMock()
        
        ui.display_pending_requests()
        
        ui.console.print.assert_called_with("✅ No pending intervention requests", style="green")
    
    @patch('safehive.ui.human_controls_ui.get_human_control_manager')
    def test_display_pending_requests_with_data(self, mock_manager):
        """Test displaying pending requests with data."""
        # Create mock requests
        request1 = InterventionRequest(
            request_id="req-1",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            session_id="session-1",
            agent_id="agent-1",
            title="Test Request 1",
            description="Test description 1",
            timeout_seconds=300
        )
        
        request2 = InterventionRequest(
            request_id="req-2",
            incident_type=IncidentType.POTENTIAL_ATTACK,
            severity=IncidentSeverity.HIGH,
            session_id="session-2",
            agent_id="agent-2",
            title="Test Request 2",
            description="Test description 2",
            timeout_seconds=180
        )
        
        mock_manager.return_value.get_pending_requests.return_value = [request1, request2]
        
        ui = HumanControlsUI()
        ui.console = MagicMock()
        
        ui.display_pending_requests()
        
        # Should call print with a table
        assert ui.console.print.called
        call_args = ui.console.print.call_args
        assert call_args is not None
    
    @patch('safehive.ui.human_controls_ui.get_human_control_manager')
    def test_display_request_details_not_found(self, mock_manager):
        """Test displaying details for non-existent request."""
        mock_manager.return_value.get_pending_requests.return_value = []
        
        ui = HumanControlsUI()
        ui.console = MagicMock()
        
        result = ui.display_request_details("non-existent")
        
        assert result is None
        ui.console.print.assert_called_with("❌ Request non-existent not found", style="red")
    
    @patch('safehive.ui.human_controls_ui.get_human_control_manager')
    def test_display_request_details_found(self, mock_manager):
        """Test displaying details for existing request."""
        request = InterventionRequest(
            request_id="req-1",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            session_id="session-1",
            agent_id="agent-1",
            title="Test Request",
            description="Test description",
            context={"key": "value"},
            affected_data={"data": "sensitive"},
            timeout_seconds=300
        )
        
        mock_manager.return_value.get_pending_requests.return_value = [request]
        
        ui = HumanControlsUI()
        ui.console = MagicMock()
        
        result = ui.display_request_details("req-1")
        
        assert result == request
        assert ui.console.print.called
    
    @patch('safehive.ui.human_controls_ui.get_human_control_manager')
    @patch('safehive.ui.human_controls_ui.Prompt')
    @patch('safehive.ui.human_controls_ui.Confirm')
    def test_prompt_for_intervention_success(self, mock_confirm, mock_prompt, mock_manager):
        """Test successful intervention prompt."""
        request = InterventionRequest(
            request_id="req-1",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            session_id="session-1",
            agent_id="agent-1",
            title="Test Request",
            description="Test description",
            timeout_seconds=300
        )
        
        mock_manager.return_value.get_pending_requests.return_value = [request]
        mock_manager.return_value.respond_to_intervention.return_value = True
        
        # Mock user input
        mock_prompt.ask.side_effect = ["approve", "Test approval"]
        mock_confirm.ask.return_value = True
        
        ui = HumanControlsUI()
        ui.console = MagicMock()
        
        result = ui.prompt_for_intervention("req-1")
        
        assert result is True
        mock_manager.return_value.respond_to_intervention.assert_called_once_with(
            request_id="req-1",
            intervention_type=InterventionType.APPROVE,
            reason="Test approval"
        )
    
    @patch('safehive.ui.human_controls_ui.get_human_control_manager')
    @patch('safehive.ui.human_controls_ui.Prompt')
    @patch('safehive.ui.human_controls_ui.Confirm')
    def test_prompt_for_intervention_cancelled(self, mock_confirm, mock_prompt, mock_manager):
        """Test cancelled intervention prompt."""
        request = InterventionRequest(
            request_id="req-1",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            session_id="session-1",
            agent_id="agent-1",
            title="Test Request",
            description="Test description",
            timeout_seconds=300
        )
        
        mock_manager.return_value.get_pending_requests.return_value = [request]
        
        # Mock user input
        mock_prompt.ask.side_effect = ["approve", "Test approval"]
        mock_confirm.ask.return_value = False  # User cancels
        
        ui = HumanControlsUI()
        ui.console = MagicMock()
        
        result = ui.prompt_for_intervention("req-1")
        
        assert result is False
        mock_manager.return_value.respond_to_intervention.assert_not_called()
    
    @patch('safehive.ui.human_controls_ui.get_human_control_manager')
    def test_prompt_for_intervention_not_found(self, mock_manager):
        """Test intervention prompt for non-existent request."""
        mock_manager.return_value.get_pending_requests.return_value = []
        
        ui = HumanControlsUI()
        ui.console = MagicMock()
        
        result = ui.prompt_for_intervention("non-existent")
        
        assert result is False
        ui.console.print.assert_called_with("❌ Request non-existent not found", style="red")
    
    @patch('safehive.ui.human_controls_ui.get_human_control_manager')
    def test_display_session_statistics_not_found(self, mock_manager):
        """Test displaying statistics for non-existent session."""
        mock_manager.return_value.get_session_statistics.return_value = None
        
        ui = HumanControlsUI()
        ui.console = MagicMock()
        
        ui.display_session_statistics("non-existent")
        
        ui.console.print.assert_called_with("❌ Session non-existent not found", style="red")
    
    @patch('safehive.ui.human_controls_ui.get_human_control_manager')
    def test_display_session_statistics_found(self, mock_manager):
        """Test displaying statistics for existing session."""
        stats = {
            "session_id": "test-session",
            "created_at": "2023-01-01T00:00:00",
            "is_active": True,
            "total_requests": 5,
            "pending_requests": 1,
            "completed_interventions": 4,
            "approved_count": 2,
            "redacted_count": 1,
            "quarantined_count": 1,
            "ignored_count": 0,
            "auto_action_count": 0
        }
        
        mock_manager.return_value.get_session_statistics.return_value = stats
        
        ui = HumanControlsUI()
        ui.console = MagicMock()
        
        ui.display_session_statistics("test-session")
        
        assert ui.console.print.called
    
    @patch('safehive.ui.human_controls_ui.get_human_control_manager')
    def test_display_global_statistics(self, mock_manager):
        """Test displaying global statistics."""
        stats = {
            "active_sessions": 2,
            "total_requests": 10,
            "pending_requests": 3,
            "completed_interventions": 7,
            "sessions": [
                {
                    "session_id": "session-1",
                    "total_requests": 5,
                    "pending_requests": 1,
                    "approved_count": 2,
                    "redacted_count": 1,
                    "quarantined_count": 1,
                    "ignored_count": 0
                },
                {
                    "session_id": "session-2",
                    "total_requests": 5,
                    "pending_requests": 2,
                    "approved_count": 1,
                    "redacted_count": 0,
                    "quarantined_count": 1,
                    "ignored_count": 1
                }
            ]
        }
        
        mock_manager.return_value.get_global_statistics.return_value = stats
        
        ui = HumanControlsUI()
        ui.console = MagicMock()
        
        ui.display_global_statistics()
        
        assert ui.console.print.called
    
    def test_format_context(self):
        """Test context formatting."""
        ui = HumanControlsUI()
        
        # Test empty context
        result = ui._format_context({})
        assert result == "None"
        
        # Test context with various data types
        context = {
            "string": "test",
            "number": 42,
            "dict": {"key": "value"},
            "list": [1, 2, 3],
            "long_string": "x" * 100
        }
        
        result = ui._format_context(context)
        assert "string: test" in result
        assert "number: 42" in result
        assert "dict: 1 items" in result
        assert "list: 3 items" in result
        assert "long_string: xxxxx" in result  # Truncated
    
    def test_format_affected_data(self):
        """Test affected data formatting."""
        ui = HumanControlsUI()
        
        # Test empty data
        result = ui._format_affected_data({})
        assert result == "None"
        
        # Test data with various types
        data = {
            "personal_info": "John Doe",
            "email": "john@example.com",
            "metadata": {"age": 30, "city": "NYC"},
            "tags": ["user", "premium"],
            "long_text": "x" * 150
        }
        
        result = ui._format_affected_data(data)
        assert "personal_info: John Doe" in result
        assert "email: john@example.com" in result
        assert "metadata: dict (2 items)" in result
        assert "tags: list (2 items)" in result
        assert "long_text: xxxxx" in result  # Truncated


class TestGlobalUIFunctions:
    """Test global UI convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global UI instance
        global _human_controls_ui
        _human_controls_ui = None
    
    @patch('safehive.ui.human_controls_ui.get_human_controls_ui')
    def test_display_pending_requests_global(self, mock_get_ui):
        """Test global display_pending_requests function."""
        mock_ui = MagicMock()
        mock_get_ui.return_value = mock_ui
        
        display_pending_requests("test-session")
        
        mock_ui.display_pending_requests.assert_called_once_with("test-session")
    
    @patch('safehive.ui.human_controls_ui.get_human_controls_ui')
    def test_prompt_for_intervention_global(self, mock_get_ui):
        """Test global prompt_for_intervention function."""
        mock_ui = MagicMock()
        mock_ui.prompt_for_intervention.return_value = True
        mock_get_ui.return_value = mock_ui
        
        result = prompt_for_intervention("req-123")
        
        assert result is True
        mock_ui.prompt_for_intervention.assert_called_once_with("req-123")
    
    @patch('safehive.ui.human_controls_ui.get_human_controls_ui')
    def test_display_statistics_global(self, mock_get_ui):
        """Test global display_statistics function."""
        mock_ui = MagicMock()
        mock_get_ui.return_value = mock_ui
        
        # Test with session ID
        display_statistics("test-session")
        mock_ui.display_session_statistics.assert_called_once_with("test-session")
        
        # Test without session ID
        mock_ui.reset_mock()
        display_statistics()
        mock_ui.display_global_statistics.assert_called_once()
    
    @patch('safehive.ui.human_controls_ui.get_human_controls_ui')
    def test_interactive_monitor_global(self, mock_get_ui):
        """Test global interactive_monitor function."""
        mock_ui = MagicMock()
        mock_get_ui.return_value = mock_ui
        
        interactive_monitor("test-session", 5.0)
        
        mock_ui.interactive_monitor.assert_called_once_with("test-session", 5.0)


class TestHumanControlsUIIntegration:
    """Integration tests for human controls UI."""
    
    @patch('safehive.ui.human_controls_ui.get_human_control_manager')
    def test_complete_ui_workflow(self, mock_manager):
        """Test complete UI workflow."""
        # Setup mock manager
        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance
        
        # Create mock requests
        request1 = InterventionRequest(
            request_id="req-1",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            session_id="session-1",
            agent_id="agent-1",
            title="Test Request 1",
            description="Test description 1",
            timeout_seconds=300
        )
        
        request2 = InterventionRequest(
            request_id="req-2",
            incident_type=IncidentType.POTENTIAL_ATTACK,
            severity=IncidentSeverity.HIGH,
            session_id="session-2",
            agent_id="agent-2",
            title="Test Request 2",
            description="Test description 2",
            timeout_seconds=180
        )
        
        mock_manager_instance.get_pending_requests.return_value = [request1, request2]
        mock_manager_instance.respond_to_intervention.return_value = True
        
        # Setup session statistics
        session_stats = {
            "session_id": "session-1",
            "created_at": "2023-01-01T00:00:00",
            "is_active": True,
            "total_requests": 5,
            "pending_requests": 1,
            "completed_interventions": 4,
            "approved_count": 2,
            "redacted_count": 1,
            "quarantined_count": 1,
            "ignored_count": 0,
            "auto_action_count": 0
        }
        
        mock_manager_instance.get_session_statistics.return_value = session_stats
        
        # Setup global statistics
        global_stats = {
            "active_sessions": 2,
            "total_requests": 10,
            "pending_requests": 2,
            "completed_interventions": 8,
            "sessions": [session_stats]
        }
        
        mock_manager_instance.get_global_statistics.return_value = global_stats
        
        # Test UI workflow
        ui = HumanControlsUI()
        ui.console = MagicMock()
        
        # 1. Display pending requests
        ui.display_pending_requests()
        assert ui.console.print.called
        
        # 2. Display request details
        ui.display_request_details("req-1")
        assert ui.console.print.called
        
        # 3. Display session statistics
        ui.display_session_statistics("session-1")
        assert ui.console.print.called
        
        # 4. Display global statistics
        ui.display_global_statistics()
        assert ui.console.print.called
        
        # 5. Test context formatting
        context = {"key": "value", "nested": {"inner": "data"}}
        formatted = ui._format_context(context)
        assert "key: value" in formatted
        assert "nested: 1 items" in formatted
        
        # 6. Test affected data formatting
        data = {"personal": "John Doe", "metadata": {"age": 30}}
        formatted = ui._format_affected_data(data)
        assert "personal: John Doe" in formatted
        assert "metadata: dict (1 items)" in formatted
