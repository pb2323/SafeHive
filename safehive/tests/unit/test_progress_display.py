"""
Unit tests for progress display functionality.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from rich.console import Console

from safehive.ui.progress_display import (
    ProgressDisplay, ProgressEvent, ProgressEventType,
    get_progress_display, start_progress_display, stop_progress_display,
    add_progress_event, update_session_progress, remove_session_progress
)
from safehive.sandbox.sandbox_manager import SandboxSession, SandboxScenario, SessionStatus, SessionPhase


class TestProgressEvent:
    """Test ProgressEvent dataclass."""
    
    def test_progress_event_creation(self):
        """Test creating a progress event."""
        event = ProgressEvent(
            event_type=ProgressEventType.SESSION_START,
            timestamp=datetime.now(),
            message="Test message",
            session_id="test-session",
            phase=SessionPhase.INITIALIZATION
        )
        
        assert event.event_type == ProgressEventType.SESSION_START
        assert event.message == "Test message"
        assert event.session_id == "test-session"
        assert event.phase == SessionPhase.INITIALIZATION
        assert event.data is None
    
    def test_progress_event_with_data(self):
        """Test creating a progress event with data."""
        data = {"key": "value"}
        event = ProgressEvent(
            event_type=ProgressEventType.AGENT_UPDATE,
            timestamp=datetime.now(),
            message="Agent updated",
            data=data
        )
        
        assert event.data == data


class TestProgressDisplay:
    """Test ProgressDisplay class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.console = Mock()
        self.console.get_time = Mock(return_value=time.time())
        self.display = ProgressDisplay(console=self.console)
    
    def test_progress_display_initialization(self):
        """Test ProgressDisplay initialization."""
        assert self.display.console == self.console
        assert self.display.events == []
        assert self.display._session_tasks == {}
        assert self.display._live_display is None
        assert self.display._layout is None
    
    def test_add_event(self):
        """Test adding progress events."""
        event = ProgressEvent(
            event_type=ProgressEventType.SESSION_START,
            timestamp=datetime.now(),
            message="Test event"
        )
        
        self.display.add_event(event)
        
        assert len(self.display.events) == 1
        assert self.display.events[0] == event
    
    def test_add_event_limits_events(self):
        """Test that events are limited to 100."""
        # Add 101 events
        for i in range(101):
            event = ProgressEvent(
                event_type=ProgressEventType.SESSION_UPDATE,
                timestamp=datetime.now(),
                message=f"Event {i}"
            )
            self.display.add_event(event)
        
        assert len(self.display.events) == 100
        assert self.display.events[0].message == "Event 1"  # First event should be removed
        assert self.display.events[-1].message == "Event 100"  # Last event should be kept
    
    def test_update_session(self):
        """Test updating session information."""
        scenario = SandboxScenario(
            name="test-scenario",
            description="Test scenario",
            duration=60,
            interactive=True
        )
        session = SandboxSession(
            session_id="test-session",
            scenario=scenario
        )
        
        self.display.update_session(session)
        
        assert "test-session" in self.display._session_tasks
        assert self.display._session_tasks["test-session"]["session"] == session
    
    def test_remove_session(self):
        """Test removing session from tracking."""
        scenario = SandboxScenario(
            name="test-scenario",
            description="Test scenario",
            duration=60,
            interactive=True
        )
        session = SandboxSession(
            session_id="test-session",
            scenario=scenario
        )
        
        self.display.update_session(session)
        assert "test-session" in self.display._session_tasks
        
        self.display.remove_session("test-session")
        assert "test-session" not in self.display._session_tasks
    
    def test_create_layout(self):
        """Test creating the layout."""
        layout = self.display.create_layout()
        
        assert layout is not None
        assert hasattr(layout, 'split_column')
    
    def test_create_header(self):
        """Test creating the header panel."""
        header = self.display.create_header()
        
        assert header is not None
        assert hasattr(header, 'render')
    
    def test_create_progress_panel_empty(self):
        """Test creating progress panel with no sessions."""
        panel = self.display.create_progress_panel()
        
        assert panel is not None
        assert hasattr(panel, 'render')
    
    def test_create_progress_panel_with_sessions(self):
        """Test creating progress panel with sessions."""
        scenario = SandboxScenario(
            name="test-scenario",
            description="Test scenario",
            duration=60,
            interactive=True
        )
        session = SandboxSession(
            session_id="test-session",
            scenario=scenario,
            status=SessionStatus.RUNNING,
            phase=SessionPhase.SCENARIO_EXECUTION
        )
        
        self.display.update_session(session)
        panel = self.display.create_progress_panel()
        
        assert panel is not None
        assert hasattr(panel, 'render')
    
    def test_create_status_panel(self):
        """Test creating the status panel."""
        panel = self.display.create_status_panel()
        
        assert panel is not None
        assert hasattr(panel, 'render')
    
    def test_create_events_panel_empty(self):
        """Test creating events panel with no events."""
        panel = self.display.create_events_panel()
        
        assert panel is not None
        assert hasattr(panel, 'render')
    
    def test_create_events_panel_with_events(self):
        """Test creating events panel with events."""
        event = ProgressEvent(
            event_type=ProgressEventType.SESSION_START,
            timestamp=datetime.now(),
            message="Test event"
        )
        self.display.add_event(event)
        
        panel = self.display.create_events_panel()
        
        assert panel is not None
        assert hasattr(panel, 'render')
    
    @patch('safehive.ui.progress_display.Live')
    def test_start_live_display(self, mock_live):
        """Test starting live display."""
        mock_live_instance = Mock()
        mock_live.return_value = mock_live_instance
        
        self.display.start_live_display()
        
        assert self.display._live_display == mock_live_instance
        mock_live_instance.start.assert_called_once()
    
    def test_stop_live_display(self):
        """Test stopping live display."""
        mock_live = Mock()
        self.display._live_display = mock_live
        
        self.display.stop_live_display()
        
        mock_live.stop.assert_called_once()
        assert self.display._live_display is None
    
    @patch('safehive.ui.progress_display.time.sleep')
    def test_show_session_progress(self, mock_sleep):
        """Test showing session progress."""
        scenario = SandboxScenario(
            name="test-scenario",
            description="Test scenario",
            duration=10,
            interactive=True
        )
        session = SandboxSession(
            session_id="test-session",
            scenario=scenario
        )
        
        # Mock the progress context
        with patch.object(self.display, 'progress') as mock_progress:
            mock_task = Mock()
            mock_progress.add_task.return_value = mock_task
            mock_progress.__enter__ = Mock(return_value=mock_progress)
            mock_progress.__exit__ = Mock(return_value=None)
            
            self.display.show_session_progress(session, duration=5)
            
            mock_progress.add_task.assert_called_once()
            mock_progress.update.assert_called()
            mock_sleep.assert_called()


class TestGlobalProgressFunctions:
    """Test global progress functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global state
        import safehive.ui.progress_display
        safehive.ui.progress_display._global_progress_display = None
    
    def test_get_progress_display(self):
        """Test getting global progress display."""
        display1 = get_progress_display()
        display2 = get_progress_display()
        
        assert display1 is display2  # Should be the same instance
        assert isinstance(display1, ProgressDisplay)
    
    @patch('safehive.ui.progress_display.Live')
    def test_start_progress_display(self, mock_live):
        """Test starting global progress display."""
        mock_live_instance = Mock()
        mock_live.return_value = mock_live_instance
        
        display = start_progress_display()
        
        assert isinstance(display, ProgressDisplay)
        assert display._live_display == mock_live_instance
        mock_live_instance.start.assert_called_once()
    
    def test_stop_progress_display(self):
        """Test stopping global progress display."""
        display = get_progress_display()
        mock_live = Mock()
        display._live_display = mock_live
        
        stop_progress_display()
        
        mock_live.stop.assert_called_once()
    
    def test_add_progress_event(self):
        """Test adding progress event globally."""
        add_progress_event(
            ProgressEventType.SESSION_START,
            "Test message",
            session_id="test-session"
        )
        
        display = get_progress_display()
        assert len(display.events) == 1
        assert display.events[0].event_type == ProgressEventType.SESSION_START
        assert display.events[0].message == "Test message"
        assert display.events[0].session_id == "test-session"
    
    def test_update_session_progress(self):
        """Test updating session progress globally."""
        scenario = SandboxScenario(
            name="test-scenario",
            description="Test scenario",
            duration=60,
            interactive=True
        )
        session = SandboxSession(
            session_id="test-session",
            scenario=scenario,
            status=SessionStatus.RUNNING,
            phase=SessionPhase.SCENARIO_EXECUTION
        )
        
        update_session_progress(session)
        
        display = get_progress_display()
        assert "test-session" in display._session_tasks
        assert len(display.events) >= 1  # Should have added an event
    
    def test_remove_session_progress(self):
        """Test removing session progress globally."""
        scenario = SandboxScenario(
            name="test-scenario",
            description="Test scenario",
            duration=60,
            interactive=True
        )
        session = SandboxSession(
            session_id="test-session",
            scenario=scenario
        )
        
        # Add session first
        display = get_progress_display()
        display.update_session(session)
        assert "test-session" in display._session_tasks
        
        # Remove session
        remove_session_progress("test-session")
        
        assert "test-session" not in display._session_tasks
        assert len(display.events) >= 1  # Should have added an event


class TestProgressEventTypes:
    """Test ProgressEventType enum."""
    
    def test_progress_event_types(self):
        """Test that all progress event types are defined."""
        expected_types = [
            "session_start",
            "session_update", 
            "session_complete",
            "agent_update",
            "guard_update",
            "scenario_update",
            "security_event",
            "error"
        ]
        
        for event_type in expected_types:
            assert hasattr(ProgressEventType, event_type.upper())
            assert getattr(ProgressEventType, event_type.upper()).value == event_type


class TestProgressDisplayIntegration:
    """Integration tests for progress display."""
    
    def setup_method(self):
        """Set up test fixtures."""
        import safehive.ui.progress_display
        safehive.ui.progress_display._global_progress_display = None
    
    def test_full_session_lifecycle(self):
        """Test complete session lifecycle with progress tracking."""
        scenario = SandboxScenario(
            name="test-scenario",
            description="Test scenario",
            duration=60,
            interactive=True
        )
        
        # Create session
        session = SandboxSession(
            session_id="test-session",
            scenario=scenario,
            status=SessionStatus.PENDING
        )
        
        # Start session
        add_progress_event(ProgressEventType.SESSION_START, "Session started", session.session_id)
        update_session_progress(session)
        
        # Update session
        session.status = SessionStatus.RUNNING
        session.phase = SessionPhase.AGENT_SETUP
        update_session_progress(session)
        
        # Complete session
        session.status = SessionStatus.COMPLETED
        session.phase = SessionPhase.CLEANUP
        remove_session_progress(session.session_id)
        
        # Verify
        display = get_progress_display()
        assert len(display.events) >= 3  # Should have multiple events
        assert "test-session" not in display._session_tasks  # Should be removed
        
        # Check event types
        event_types = [event.event_type for event in display.events]
        assert ProgressEventType.SESSION_START in event_types
        assert ProgressEventType.SESSION_UPDATE in event_types
        assert ProgressEventType.SESSION_COMPLETE in event_types
