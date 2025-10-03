"""
Unit tests for human-in-the-loop controls system.
"""

import pytest
import asyncio
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from safehive.models.human_controls import (
    InterventionRequest, InterventionResponse, HumanControlSession,
    InterventionType, InterventionStatus, IncidentType, IncidentSeverity,
    create_suspicious_request_incident, create_potential_attack_incident,
    create_privacy_violation_incident
)
from safehive.utils.human_controls import (
    HumanControlManager, get_human_control_manager,
    request_human_intervention, respond_to_intervention
)


class TestInterventionRequest:
    """Test InterventionRequest model."""
    
    def test_intervention_request_creation(self):
        """Test creating an intervention request."""
        request = InterventionRequest(
            request_id="test-123",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            session_id="session-456",
            agent_id="agent-789",
            title="Test Request",
            description="Test description"
        )
        
        assert request.request_id == "test-123"
        assert request.incident_type == IncidentType.SUSPICIOUS_REQUEST
        assert request.severity == IncidentSeverity.MEDIUM
        assert request.session_id == "session-456"
        assert request.agent_id == "agent-789"
        assert request.title == "Test Request"
        assert request.description == "Test description"
        assert request.status == InterventionStatus.PENDING
    
    def test_intervention_request_serialization(self):
        """Test serialization and deserialization."""
        request = InterventionRequest(
            request_id="test-123",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            session_id="session-456",
            agent_id="agent-789",
            title="Test Request",
            description="Test description",
            context={"key": "value"},
            affected_data={"data": "sensitive"}
        )
        
        # Test to_dict
        data = request.to_dict()
        assert data["request_id"] == "test-123"
        assert data["incident_type"] == "suspicious_request"
        assert data["severity"] == "medium"
        assert data["context"] == {"key": "value"}
        assert data["affected_data"] == {"data": "sensitive"}
        
        # Test from_dict
        restored = InterventionRequest.from_dict(data)
        assert restored.request_id == request.request_id
        assert restored.incident_type == request.incident_type
        assert restored.severity == request.severity
        assert restored.context == request.context
        assert restored.affected_data == request.affected_data
    
    def test_intervention_request_expiry(self):
        """Test request expiry functionality."""
        request = InterventionRequest(
            request_id="test-123",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            session_id="session-456",
            agent_id="agent-789",
            title="Test Request",
            description="Test description",
            timeout_seconds=10
        )
        
        # Should not be expired initially
        assert not request.is_expired()
        # Add a small delay to ensure some time has passed
        time.sleep(0.01)
        remaining = request.get_remaining_time()
        assert remaining > 0
        
        # Wait for expiry
        time.sleep(10.1)
        assert request.is_expired()
        assert request.get_remaining_time() == 0


class TestInterventionResponse:
    """Test InterventionResponse model."""
    
    def test_intervention_response_creation(self):
        """Test creating an intervention response."""
        response = InterventionResponse(
            request_id="test-123",
            intervention_type=InterventionType.APPROVE,
            reason="Test approval"
        )
        
        assert response.request_id == "test-123"
        assert response.intervention_type == InterventionType.APPROVE
        assert response.reason == "Test approval"
    
    def test_intervention_response_serialization(self):
        """Test serialization and deserialization."""
        response = InterventionResponse(
            request_id="test-123",
            intervention_type=InterventionType.REDACT,
            reason="Test redaction",
            redaction_rules=["rule1", "rule2"],
            quarantine_duration=3600,
            quarantine_reason="Test quarantine"
        )
        
        # Test to_dict
        data = response.to_dict()
        assert data["request_id"] == "test-123"
        assert data["intervention_type"] == "redact"
        assert data["reason"] == "Test redaction"
        assert data["redaction_rules"] == ["rule1", "rule2"]
        assert data["quarantine_duration"] == 3600
        assert data["quarantine_reason"] == "Test quarantine"
        
        # Test from_dict
        restored = InterventionResponse.from_dict(data)
        assert restored.request_id == response.request_id
        assert restored.intervention_type == response.intervention_type
        assert restored.reason == response.reason
        assert restored.redaction_rules == response.redaction_rules
        assert restored.quarantine_duration == response.quarantine_duration
        assert restored.quarantine_reason == response.quarantine_reason


class TestHumanControlSession:
    """Test HumanControlSession model."""
    
    def test_session_creation(self):
        """Test creating a control session."""
        session = HumanControlSession(session_id="test-session")
        
        assert session.session_id == "test-session"
        assert session.is_active
        assert len(session.pending_requests) == 0
        assert len(session.completed_interventions) == 0
        assert session.total_requests == 0
    
    def test_session_add_request(self):
        """Test adding requests to session."""
        session = HumanControlSession(session_id="test-session")
        
        request = InterventionRequest(
            request_id="req-1",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            session_id="test-session",
            agent_id="agent-1",
            title="Test Request",
            description="Test description"
        )
        
        session.add_request(request)
        
        assert len(session.pending_requests) == 1
        assert session.total_requests == 1
        assert "req-1" in session.pending_requests
    
    def test_session_respond_to_request(self):
        """Test responding to requests."""
        session = HumanControlSession(session_id="test-session")
        
        request = InterventionRequest(
            request_id="req-1",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            session_id="test-session",
            agent_id="agent-1",
            title="Test Request",
            description="Test description"
        )
        
        session.add_request(request)
        
        response = InterventionResponse(
            request_id="req-1",
            intervention_type=InterventionType.APPROVE,
            reason="Test approval"
        )
        
        result = session.respond_to_request("req-1", response)
        
        assert result is not None
        assert len(session.pending_requests) == 0
        assert len(session.completed_interventions) == 1
        assert session.approved_count == 1
    
    def test_session_auto_respond_expired(self):
        """Test auto-responding to expired requests."""
        session = HumanControlSession(session_id="test-session")
        
        # Create expired request
        request = InterventionRequest(
            request_id="req-1",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            session_id="test-session",
            agent_id="agent-1",
            title="Test Request",
            description="Test description",
            timeout_seconds=0.1  # Very short timeout
        )
        
        session.add_request(request)
        
        # Wait for expiry
        time.sleep(0.2)
        
        # Auto-respond
        expired_requests = session.auto_respond_to_expired()
        
        assert len(expired_requests) == 1
        assert len(session.pending_requests) == 0
        assert len(session.completed_interventions) == 1
        assert session.ignored_count >= 1  # At least one ignore (may be more due to session closure)


class TestHumanControlManager:
    """Test HumanControlManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global manager for each test
        from safehive.utils.human_controls import _human_control_manager
        import safehive.utils.human_controls
        safehive.utils.human_controls._human_control_manager = None
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = HumanControlManager(default_timeout=300)
        
        assert manager.default_timeout == 300
        assert len(manager.active_sessions) == 0
        assert len(manager.pending_requests) == 0
    
    def test_create_session(self):
        """Test creating a session."""
        manager = HumanControlManager()
        
        session = manager.create_session("test-session")
        
        assert session.session_id == "test-session"
        assert session.is_active
        assert "test-session" in manager.active_sessions
    
    def test_request_intervention(self):
        """Test requesting intervention."""
        manager = HumanControlManager()
        
        request = manager.request_intervention(
            session_id="test-session",
            agent_id="test-agent",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            title="Test Request",
            description="Test description"
        )
        
        assert request.session_id == "test-session"
        assert request.agent_id == "test-agent"
        assert request.incident_type == IncidentType.SUSPICIOUS_REQUEST
        assert request.severity == IncidentSeverity.MEDIUM
        assert request.title == "Test Request"
        assert request.description == "Test description"
        assert request.request_id in manager.pending_requests
        
        # Session should be created automatically
        assert "test-session" in manager.active_sessions
    
    def test_respond_to_intervention(self):
        """Test responding to intervention."""
        manager = HumanControlManager()
        
        # Create a request
        request = manager.request_intervention(
            session_id="test-session",
            agent_id="test-agent",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            title="Test Request",
            description="Test description"
        )
        
        # Respond to it
        success = manager.respond_to_intervention(
            request_id=request.request_id,
            intervention_type=InterventionType.APPROVE,
            reason="Test approval"
        )
        
        assert success
        assert request.request_id not in manager.pending_requests
        assert request.status == InterventionStatus.COMPLETED
        assert request.human_response == InterventionType.APPROVE
        assert request.human_reason == "Test approval"
    
    def test_get_pending_requests(self):
        """Test getting pending requests."""
        manager = HumanControlManager()
        
        # Create requests for different sessions
        request1 = manager.request_intervention(
            session_id="session-1",
            agent_id="agent-1",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            title="Request 1",
            description="Description 1"
        )
        
        request2 = manager.request_intervention(
            session_id="session-2",
            agent_id="agent-2",
            incident_type=IncidentType.POTENTIAL_ATTACK,
            severity=IncidentSeverity.HIGH,
            title="Request 2",
            description="Description 2"
        )
        
        # Get all pending requests
        all_pending = manager.get_pending_requests()
        assert len(all_pending) == 2
        
        # Get pending requests for specific session
        session1_pending = manager.get_pending_requests("session-1")
        assert len(session1_pending) == 1
        assert session1_pending[0].request_id == request1.request_id
        
        session2_pending = manager.get_pending_requests("session-2")
        assert len(session2_pending) == 1
        assert session2_pending[0].request_id == request2.request_id
    
    def test_session_statistics(self):
        """Test session statistics."""
        manager = HumanControlManager()
        
        # Create session and requests
        session = manager.create_session("test-session")
        
        request1 = manager.request_intervention(
            session_id="test-session",
            agent_id="agent-1",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            title="Request 1",
            description="Description 1"
        )
        
        request2 = manager.request_intervention(
            session_id="test-session",
            agent_id="agent-2",
            incident_type=IncidentType.POTENTIAL_ATTACK,
            severity=IncidentSeverity.HIGH,
            title="Request 2",
            description="Description 2"
        )
        
        # Respond to one request
        manager.respond_to_intervention(
            request_id=request1.request_id,
            intervention_type=InterventionType.APPROVE,
            reason="Approved"
        )
        
        # Get statistics
        stats = manager.get_session_statistics("test-session")
        
        assert stats is not None
        assert stats["session_id"] == "test-session"
        assert stats["total_requests"] == 2
        assert stats["pending_requests"] == 1
        assert stats["completed_interventions"] == 1
        assert stats["approved_count"] == 1
    
    def test_global_statistics(self):
        """Test global statistics."""
        manager = HumanControlManager()
        
        # Create multiple sessions with requests
        manager.request_intervention(
            session_id="session-1",
            agent_id="agent-1",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            title="Request 1",
            description="Description 1"
        )
        
        manager.request_intervention(
            session_id="session-2",
            agent_id="agent-2",
            incident_type=IncidentType.POTENTIAL_ATTACK,
            severity=IncidentSeverity.HIGH,
            title="Request 2",
            description="Description 2"
        )
        
        # Get global statistics
        stats = manager.get_global_statistics()
        
        assert stats["active_sessions"] == 2
        assert stats["total_requests"] == 2
        assert stats["pending_requests"] == 2
        assert stats["completed_interventions"] == 0
        assert len(stats["sessions"]) == 2
    
    def test_close_session(self):
        """Test closing a session."""
        manager = HumanControlManager()
        
        # Create session with pending request
        request = manager.request_intervention(
            session_id="test-session",
            agent_id="test-agent",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            title="Test Request",
            description="Test description"
        )
        
        # Close session
        success = manager.close_session("test-session")
        
        assert success
        assert "test-session" not in manager.active_sessions
        assert request.request_id not in manager.pending_requests
        
        # Pending request should be auto-ignored
        session_stats = manager.get_session_statistics("test-session")
        assert session_stats is None  # Session no longer exists
    
    def test_callback_registration(self):
        """Test callback registration and execution."""
        manager = HumanControlManager()
        
        # Track callback calls
        callback_calls = []
        
        def test_callback(request, response):
            callback_calls.append((request.request_id, response.intervention_type))
        
        # Register callback
        manager.register_callback(InterventionType.APPROVE, test_callback)
        
        # Create and respond to request
        request = manager.request_intervention(
            session_id="test-session",
            agent_id="test-agent",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            title="Test Request",
            description="Test description"
        )
        
        manager.respond_to_intervention(
            request_id=request.request_id,
            intervention_type=InterventionType.APPROVE,
            reason="Test approval"
        )
        
        # Check callback was called
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == request.request_id
        assert callback_calls[0][1] == InterventionType.APPROVE


class TestConvenienceFunctions:
    """Test convenience functions for creating incidents."""
    
    def test_create_suspicious_request_incident(self):
        """Test creating suspicious request incident."""
        incident = create_suspicious_request_incident(
            session_id="test-session",
            agent_id="test-agent",
            title="Suspicious Request",
            description="A suspicious request was detected",
            severity=IncidentSeverity.HIGH,
            context={"ip": "192.168.1.1"},
            timeout_seconds=180
        )
        
        assert incident.incident_type == IncidentType.SUSPICIOUS_REQUEST
        assert incident.severity == IncidentSeverity.HIGH
        assert incident.session_id == "test-session"
        assert incident.agent_id == "test-agent"
        assert incident.title == "Suspicious Request"
        assert incident.description == "A suspicious request was detected"
        assert incident.context == {"ip": "192.168.1.1"}
        assert incident.timeout_seconds == 180
        assert incident.auto_action is None  # High severity has no default auto action
    
    def test_create_potential_attack_incident(self):
        """Test creating potential attack incident."""
        incident = create_potential_attack_incident(
            session_id="test-session",
            agent_id="test-agent",
            title="Potential Attack",
            description="A potential attack was detected",
            severity=IncidentSeverity.CRITICAL,
            context={"attack_type": "sql_injection"},
            timeout_seconds=60
        )
        
        assert incident.incident_type == IncidentType.POTENTIAL_ATTACK
        assert incident.severity == IncidentSeverity.CRITICAL
        assert incident.session_id == "test-session"
        assert incident.agent_id == "test-agent"
        assert incident.title == "Potential Attack"
        assert incident.description == "A potential attack was detected"
        assert incident.context == {"attack_type": "sql_injection"}
        assert incident.timeout_seconds == 60
        assert incident.auto_action == InterventionType.QUARANTINE  # Critical severity default
    
    def test_create_privacy_violation_incident(self):
        """Test creating privacy violation incident."""
        affected_data = {"personal_info": "John Doe", "email": "john@example.com"}
        
        incident = create_privacy_violation_incident(
            session_id="test-session",
            agent_id="test-agent",
            title="Privacy Violation",
            description="Personal information exposed",
            affected_data=affected_data,
            severity=IncidentSeverity.MEDIUM,
            context={"data_type": "personal"},
            timeout_seconds=120
        )
        
        assert incident.incident_type == IncidentType.PRIVACY_VIOLATION
        assert incident.severity == IncidentSeverity.MEDIUM
        assert incident.session_id == "test-session"
        assert incident.agent_id == "test-agent"
        assert incident.title == "Privacy Violation"
        assert incident.description == "Personal information exposed"
        assert incident.affected_data == affected_data
        assert incident.context == {"data_type": "personal"}
        assert incident.timeout_seconds == 120
        assert incident.auto_action == InterventionType.REDACT  # Medium severity default


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        import safehive.utils.human_controls
        safehive.utils.human_controls._human_control_manager = None
    
    def test_get_human_control_manager(self):
        """Test getting global manager instance."""
        manager1 = get_human_control_manager()
        manager2 = get_human_control_manager()
        
        assert manager1 is manager2  # Should be singleton
    
    def test_request_human_intervention_global(self):
        """Test global request_human_intervention function."""
        request = request_human_intervention(
            session_id="test-session",
            agent_id="test-agent",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            title="Global Request",
            description="Global description"
        )
        
        assert request.session_id == "test-session"
        assert request.agent_id == "test-agent"
        assert request.title == "Global Request"
        assert request.description == "Global description"
        
        # Should be in global manager
        manager = get_human_control_manager()
        assert request.request_id in manager.pending_requests
    
    def test_respond_to_intervention_global(self):
        """Test global respond_to_intervention function."""
        # Create request
        request = request_human_intervention(
            session_id="test-session",
            agent_id="test-agent",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            title="Global Request",
            description="Global description"
        )
        
        # Respond to it
        success = respond_to_intervention(
            request_id=request.request_id,
            intervention_type=InterventionType.APPROVE,
            reason="Global approval"
        )
        
        assert success
        
        # Check request was processed
        manager = get_human_control_manager()
        assert request.request_id not in manager.pending_requests
        assert request.status == InterventionStatus.COMPLETED
        assert request.human_response == InterventionType.APPROVE
        assert request.human_reason == "Global approval"


class TestHumanControlIntegration:
    """Integration tests for human controls system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        import safehive.utils.human_controls
        safehive.utils.human_controls._human_control_manager = None
    
    def test_complete_intervention_workflow(self):
        """Test complete intervention workflow."""
        manager = get_human_control_manager()
        
        # 1. Create session
        session = manager.create_session("integration-test")
        
        # 2. Request intervention
        request = manager.request_intervention(
            session_id="integration-test",
            agent_id="test-agent",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.HIGH,
            title="Integration Test",
            description="Testing complete workflow",
            context={"test": True},
            timeout_seconds=300
        )
        
        # 3. Verify request is pending
        pending = manager.get_pending_requests("integration-test")
        assert len(pending) == 1
        assert pending[0].request_id == request.request_id
        
        # 4. Respond to intervention
        success = manager.respond_to_intervention(
            request_id=request.request_id,
            intervention_type=InterventionType.QUARANTINE,
            reason="Integration test quarantine",
            quarantine_duration=3600,
            quarantine_reason="Test quarantine"
        )
        
        assert success
        
        # 5. Verify request is completed
        assert request.request_id not in manager.pending_requests
        assert request.status == InterventionStatus.COMPLETED
        assert request.human_response == InterventionType.QUARANTINE
        assert request.quarantine_duration == 3600
        
        # 6. Verify session statistics
        stats = manager.get_session_statistics("integration-test")
        assert stats["total_requests"] == 1
        assert stats["pending_requests"] == 0
        assert stats["completed_interventions"] == 1
        assert stats["quarantined_count"] == 1
        
        # 7. Close session
        manager.close_session("integration-test")
        assert "integration-test" not in manager.active_sessions
    
    def test_multiple_sessions_workflow(self):
        """Test workflow with multiple sessions."""
        manager = get_human_control_manager()
        
        # Create multiple sessions
        session1 = manager.create_session("session-1")
        session2 = manager.create_session("session-2")
        
        # Create requests in different sessions
        request1 = manager.request_intervention(
            session_id="session-1",
            agent_id="agent-1",
            incident_type=IncidentType.SUSPICIOUS_REQUEST,
            severity=IncidentSeverity.MEDIUM,
            title="Session 1 Request",
            description="First session request"
        )
        
        request2 = manager.request_intervention(
            session_id="session-2",
            agent_id="agent-2",
            incident_type=IncidentType.POTENTIAL_ATTACK,
            severity=IncidentSeverity.HIGH,
            title="Session 2 Request",
            description="Second session request"
        )
        
        # Verify both requests are pending
        all_pending = manager.get_pending_requests()
        assert len(all_pending) == 2
        
        session1_pending = manager.get_pending_requests("session-1")
        assert len(session1_pending) == 1
        
        session2_pending = manager.get_pending_requests("session-2")
        assert len(session2_pending) == 1
        
        # Respond to one request
        manager.respond_to_intervention(
            request_id=request1.request_id,
            intervention_type=InterventionType.APPROVE,
            reason="Approved session 1"
        )
        
        # Verify only one request remains pending
        all_pending = manager.get_pending_requests()
        assert len(all_pending) == 1
        assert all_pending[0].request_id == request2.request_id
        
        # Verify global statistics
        global_stats = manager.get_global_statistics()
        assert global_stats["active_sessions"] == 2
        assert global_stats["total_requests"] == 2
        assert global_stats["pending_requests"] == 1
        assert global_stats["completed_interventions"] == 1
