"""
Unit tests for the base Guard class and Request/Response models.

This module tests the core functionality of the Guard base class
and the Request/Response data models.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from safehive.guards import Guard
from safehive.models.request_response import (
    Request, Response, 
    create_allow_response, create_block_response, 
    create_decoy_response, create_redact_response
)


class MockGuard(Guard):
    """Test implementation of Guard for testing purposes."""
    
    def inspect(self, request: Request) -> Response:
        """Simple test implementation that always allows requests."""
        return self._create_response("allow", "Test guard allows all requests")
    
    def configure(self, config: dict) -> None:
        """Test configuration implementation."""
        self.config = config
    
    def get_status(self) -> dict:
        """Test status implementation."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "config": self.config,
            "metrics": self.metrics
        }


class TestRequestResponse:
    """Test Request and Response models."""
    
    def test_request_creation(self):
        """Test Request object creation and serialization."""
        request = Request(
            source="test_source",
            ip="192.168.1.1",
            payload="test payload",
            task="test task"
        )
        
        assert request.source == "test_source"
        assert request.ip == "192.168.1.1"
        assert request.payload == "test payload"
        assert request.task == "test task"
        assert isinstance(request.timestamp, datetime)
        assert request.metadata == {}
    
    def test_request_serialization(self):
        """Test Request serialization to/from dictionary."""
        request = Request(
            source="test_source",
            ip="192.168.1.1", 
            payload="test payload",
            task="test task",
            metadata={"key": "value"}
        )
        
        # Test to_dict
        request_dict = request.to_dict()
        assert request_dict["source"] == "test_source"
        assert request_dict["ip"] == "192.168.1.1"
        assert request_dict["payload"] == "test payload"
        assert request_dict["task"] == "test task"
        assert request_dict["metadata"] == {"key": "value"}
        assert "timestamp" in request_dict
        
        # Test from_dict
        restored_request = Request.from_dict(request_dict)
        assert restored_request.source == request.source
        assert restored_request.ip == request.ip
        assert restored_request.payload == request.payload
        assert restored_request.task == request.task
        assert restored_request.metadata == request.metadata
    
    def test_response_creation(self):
        """Test Response object creation."""
        response = Response(
            action="allow",
            reason="Test reason",
            details={"key": "value"},
            confidence=0.9
        )
        
        assert response.action == "allow"
        assert response.reason == "Test reason"
        assert response.details == {"key": "value"}
        assert response.confidence == 0.9
        assert isinstance(response.timestamp, datetime)
    
    def test_response_validation(self):
        """Test Response validation."""
        # Test invalid action
        with pytest.raises(ValueError, match="Invalid action"):
            Response(action="invalid", reason="test")
        
        # Test invalid confidence
        with pytest.raises(ValueError, match="Invalid confidence"):
            Response(action="allow", reason="test", confidence=1.5)
        
        with pytest.raises(ValueError, match="Invalid confidence"):
            Response(action="allow", reason="test", confidence=-0.1)
    
    def test_response_convenience_methods(self):
        """Test Response convenience methods."""
        allow_response = Response(action="allow", reason="test")
        block_response = Response(action="block", reason="test")
        decoy_response = Response(action="decoy", reason="test")
        redact_response = Response(action="redact", reason="test")
        
        assert allow_response.is_allowed()
        assert not allow_response.is_blocked()
        assert not allow_response.is_decoy()
        assert not allow_response.is_redacted()
        
        assert block_response.is_blocked()
        assert not block_response.is_allowed()
        
        assert decoy_response.is_decoy()
        assert not decoy_response.is_allowed()
        
        assert redact_response.is_redacted()
        assert not redact_response.is_allowed()
    
    def test_convenience_functions(self):
        """Test convenience response creation functions."""
        # Test allow response
        allow_resp = create_allow_response("Test allow")
        assert allow_resp.action == "allow"
        assert allow_resp.reason == "Test allow"
        
        # Test block response
        block_resp = create_block_response("Test block")
        assert block_resp.action == "block"
        assert block_resp.reason == "Test block"
        
        # Test decoy response
        decoy_data = {"fake_data": "test"}
        decoy_resp = create_decoy_response("Test decoy", decoy_data)
        assert decoy_resp.action == "decoy"
        assert decoy_resp.reason == "Test decoy"
        assert decoy_resp.details["decoy_data"] == decoy_data
        
        # Test redact response
        redact_resp = create_redact_response("Test redact", "REDACTED")
        assert redact_resp.action == "redact"
        assert redact_resp.reason == "Test redact"
        assert redact_resp.details["redacted_content"] == "REDACTED"


class TestGuardBase:
    """Test the base Guard class functionality."""
    
    def test_guard_initialization(self):
        """Test Guard initialization."""
        guard = MockGuard("test_guard")
        
        assert guard.name == "test_guard"
        assert guard.enabled is True
        assert guard.config == {}
        assert guard.metrics["total_requests"] == 0
        assert guard.metrics["blocked_requests"] == 0
        assert guard.metrics["allowed_requests"] == 0
    
    def test_guard_enable_disable(self):
        """Test guard enable/disable functionality."""
        guard = MockGuard("test_guard")
        
        assert guard.is_enabled()
        
        guard.disable()
        assert not guard.is_enabled()
        
        guard.enable()
        assert guard.is_enabled()
    
    def test_guard_inspect(self):
        """Test guard inspect method."""
        guard = MockGuard("test_guard")
        request = Request(
            source="test_source",
            ip="192.168.1.1",
            payload="test payload",
            task="test task"
        )
        
        response = guard.inspect(request)
        
        assert isinstance(response, Response)
        assert response.action == "allow"
        assert response.reason == "Test guard allows all requests"
        assert guard.metrics["total_requests"] == 1
        assert guard.metrics["allowed_requests"] == 1
    
    def test_guard_configure(self):
        """Test guard configuration."""
        guard = MockGuard("test_guard")
        config = {"threshold": 3, "enabled": True}
        
        guard.configure(config)
        
        assert guard.config == config
    
    def test_guard_status(self):
        """Test guard status retrieval."""
        guard = MockGuard("test_guard")
        config = {"test": "config"}
        guard.configure(config)
        
        status = guard.get_status()
        
        assert status["name"] == "test_guard"
        assert status["enabled"] is True
        assert status["config"] == config
        assert "metrics" in status
    
    def test_metrics_update(self):
        """Test metrics updating functionality."""
        guard = MockGuard("test_guard")
        
        # Test allow action
        guard._update_metrics("allow")
        assert guard.metrics["total_requests"] == 1
        assert guard.metrics["allowed_requests"] == 1
        
        # Test block action
        guard._update_metrics("block")
        assert guard.metrics["total_requests"] == 2
        assert guard.metrics["blocked_requests"] == 1
        
        # Test decoy action
        guard._update_metrics("decoy")
        assert guard.metrics["total_requests"] == 3
        assert guard.metrics["decoy_responses"] == 1
        
        # Test redact action
        guard._update_metrics("redact")
        assert guard.metrics["total_requests"] == 4
        assert guard.metrics["redacted_responses"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
