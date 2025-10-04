"""
Comprehensive unit tests for Request and Response models.

This module tests all functionality of the Request and Response data models
used for agent communication in the SafeHive system.
"""

import pytest
import json
from datetime import datetime, timezone
from typing import Dict, Any

from safehive.models.request_response import (
    Request, Response,
    create_allow_response, create_block_response,
    create_decoy_response, create_redact_response
)


class TestRequest:
    """Test the Request model."""
    
    def test_request_creation_basic(self):
        """Test basic Request object creation."""
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
        assert request.timestamp is not None
        assert request.metadata == {}
    
    def test_request_creation_with_metadata(self):
        """Test Request creation with metadata."""
        metadata = {"user_id": "123", "session_id": "abc"}
        request = Request(
            source="vendor_api",
            ip="10.0.0.1",
            payload="order pizza",
            task="food ordering",
            metadata=metadata
        )
        
        assert request.metadata == metadata
        assert request.metadata["user_id"] == "123"
        assert request.metadata["session_id"] == "abc"
    
    def test_request_creation_with_timestamp(self):
        """Test Request creation with custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        request = Request(
            source="test_source",
            ip="192.168.1.1",
            payload="test payload",
            task="test task",
            timestamp=custom_time
        )
        
        assert request.timestamp == custom_time
    
    def test_request_to_dict(self):
        """Test Request serialization to dictionary."""
        request = Request(
            source="test_source",
            ip="192.168.1.1",
            payload="test payload",
            task="test task",
            metadata={"key": "value"}
        )
        
        data = request.to_dict()
        
        assert data["source"] == "test_source"
        assert data["ip"] == "192.168.1.1"
        assert data["payload"] == "test payload"
        assert data["task"] == "test task"
        assert data["metadata"] == {"key": "value"}
        assert "timestamp" in data
        assert data["timestamp"] is not None
    
    def test_request_from_dict(self):
        """Test Request creation from dictionary."""
        timestamp_str = "2024-01-01T12:00:00+00:00"
        data = {
            "source": "test_source",
            "ip": "192.168.1.1",
            "payload": "test payload",
            "task": "test task",
            "timestamp": timestamp_str,
            "metadata": {"key": "value"}
        }
        
        request = Request.from_dict(data)
        
        assert request.source == "test_source"
        assert request.ip == "192.168.1.1"
        assert request.payload == "test payload"
        assert request.task == "test task"
        assert request.metadata == {"key": "value"}
        assert request.timestamp is not None
        assert request.timestamp.isoformat() == timestamp_str
    
    def test_request_from_dict_no_timestamp(self):
        """Test Request creation from dictionary without timestamp."""
        data = {
            "source": "test_source",
            "ip": "192.168.1.1",
            "payload": "test payload",
            "task": "test task",
            "metadata": {"key": "value"}
        }
        
        request = Request.from_dict(data)
        
        assert request.timestamp is None
    
    def test_request_from_dict_no_metadata(self):
        """Test Request creation from dictionary without metadata."""
        data = {
            "source": "test_source",
            "ip": "192.168.1.1",
            "payload": "test payload",
            "task": "test task"
        }
        
        request = Request.from_dict(data)
        
        assert request.metadata == {}
    
    def test_request_roundtrip_serialization(self):
        """Test Request serialization roundtrip."""
        original = Request(
            source="test_source",
            ip="192.168.1.1",
            payload="test payload",
            task="test task",
            metadata={"key": "value"}
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = Request.from_dict(data)
        
        assert restored.source == original.source
        assert restored.ip == original.ip
        assert restored.payload == original.payload
        assert restored.task == original.task
        assert restored.metadata == original.metadata
        # Timestamps might differ slightly due to serialization
        assert abs((restored.timestamp - original.timestamp).total_seconds()) < 1
    
    def test_request_json_serialization(self):
        """Test Request JSON serialization."""
        request = Request(
            source="test_source",
            ip="192.168.1.1",
            payload="test payload",
            task="test task",
            metadata={"key": "value"}
        )
        
        # Convert to dict, then to JSON, then back
        data = request.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = Request.from_dict(restored_data)
        
        assert restored.source == request.source
        assert restored.ip == request.ip
        assert restored.payload == request.payload
        assert restored.task == request.task
        assert restored.metadata == request.metadata


class TestResponse:
    """Test the Response model."""
    
    def test_response_creation_allow(self):
        """Test Response creation with allow action."""
        response = Response(
            action="allow",
            reason="Request is safe",
            details={"confidence_score": 0.95},
            confidence=0.95
        )
        
        assert response.action == "allow"
        assert response.reason == "Request is safe"
        assert response.details == {"confidence_score": 0.95}
        assert response.confidence == 0.95
        assert response.timestamp is not None
    
    def test_response_creation_block(self):
        """Test Response creation with block action."""
        response = Response(
            action="block",
            reason="SQL injection detected",
            details={"pattern": "SELECT * FROM users", "severity": "high"},
            confidence=0.99
        )
        
        assert response.action == "block"
        assert response.reason == "SQL injection detected"
        assert response.details["pattern"] == "SELECT * FROM users"
        assert response.details["severity"] == "high"
        assert response.confidence == 0.99
    
    def test_response_creation_decoy(self):
        """Test Response creation with decoy action."""
        decoy_data = {"fake_credit_cards": ["4111-1111-1111-1111"]}
        response = Response(
            action="decoy",
            reason="Decoy response triggered",
            details={"decoy_data": decoy_data},
            confidence=1.0
        )
        
        assert response.action == "decoy"
        assert response.reason == "Decoy response triggered"
        assert response.details["decoy_data"] == decoy_data
        assert response.confidence == 1.0
    
    def test_response_creation_redact(self):
        """Test Response creation with redact action."""
        response = Response(
            action="redact",
            reason="PII detected",
            details={"redacted_content": "***@example.com"},
            confidence=0.9
        )
        
        assert response.action == "redact"
        assert response.reason == "PII detected"
        assert response.details["redacted_content"] == "***@example.com"
        assert response.confidence == 0.9
    
    def test_response_invalid_action(self):
        """Test Response creation with invalid action."""
        with pytest.raises(ValueError, match="Invalid action"):
            Response(
                action="invalid_action",
                reason="Test reason"
            )
    
    def test_response_invalid_confidence_low(self):
        """Test Response creation with confidence below 0.0."""
        with pytest.raises(ValueError, match="Invalid confidence"):
            Response(
                action="allow",
                reason="Test reason",
                confidence=-0.1
            )
    
    def test_response_invalid_confidence_high(self):
        """Test Response creation with confidence above 1.0."""
        with pytest.raises(ValueError, match="Invalid confidence"):
            Response(
                action="allow",
                reason="Test reason",
                confidence=1.1
            )
    
    def test_response_confidence_boundaries(self):
        """Test Response creation with confidence at boundaries."""
        # Test minimum confidence
        response_min = Response(
            action="allow",
            reason="Test reason",
            confidence=0.0
        )
        assert response_min.confidence == 0.0
        
        # Test maximum confidence
        response_max = Response(
            action="allow",
            reason="Test reason",
            confidence=1.0
        )
        assert response_max.confidence == 1.0
    
    def test_response_to_dict(self):
        """Test Response serialization to dictionary."""
        response = Response(
            action="block",
            reason="Test reason",
            details={"key": "value"},
            confidence=0.8
        )
        
        data = response.to_dict()
        
        assert data["action"] == "block"
        assert data["reason"] == "Test reason"
        assert data["details"] == {"key": "value"}
        assert data["confidence"] == 0.8
        assert "timestamp" in data
        assert data["timestamp"] is not None
    
    def test_response_from_dict(self):
        """Test Response creation from dictionary."""
        timestamp_str = "2024-01-01T12:00:00+00:00"
        data = {
            "action": "allow",
            "reason": "Test reason",
            "details": {"key": "value"},
            "confidence": 0.8,
            "timestamp": timestamp_str
        }
        
        response = Response.from_dict(data)
        
        assert response.action == "allow"
        assert response.reason == "Test reason"
        assert response.details == {"key": "value"}
        assert response.confidence == 0.8
        assert response.timestamp is not None
        assert response.timestamp.isoformat() == timestamp_str
    
    def test_response_from_dict_defaults(self):
        """Test Response creation from dictionary with defaults."""
        data = {
            "action": "allow",
            "reason": "Test reason"
        }
        
        response = Response.from_dict(data)
        
        assert response.action == "allow"
        assert response.reason == "Test reason"
        assert response.details == {}
        assert response.confidence == 1.0
        assert response.timestamp is not None
    
    def test_response_roundtrip_serialization(self):
        """Test Response serialization roundtrip."""
        original = Response(
            action="block",
            reason="Test reason",
            details={"key": "value"},
            confidence=0.8
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = Response.from_dict(data)
        
        assert restored.action == original.action
        assert restored.reason == original.reason
        assert restored.details == original.details
        assert restored.confidence == original.confidence
        # Timestamps might differ slightly due to serialization
        assert abs((restored.timestamp - original.timestamp).total_seconds()) < 1
    
    def test_response_action_methods(self):
        """Test Response action checking methods."""
        # Test allow response
        allow_response = Response(action="allow", reason="Test")
        assert allow_response.is_allowed() is True
        assert allow_response.is_blocked() is False
        assert allow_response.is_decoy() is False
        assert allow_response.is_redacted() is False
        
        # Test block response
        block_response = Response(action="block", reason="Test")
        assert block_response.is_allowed() is False
        assert block_response.is_blocked() is True
        assert block_response.is_decoy() is False
        assert block_response.is_redacted() is False
        
        # Test decoy response
        decoy_response = Response(action="decoy", reason="Test")
        assert decoy_response.is_allowed() is False
        assert decoy_response.is_blocked() is False
        assert decoy_response.is_decoy() is True
        assert decoy_response.is_redacted() is False
        
        # Test redact response
        redact_response = Response(action="redact", reason="Test")
        assert redact_response.is_allowed() is False
        assert redact_response.is_blocked() is False
        assert redact_response.is_decoy() is False
        assert redact_response.is_redacted() is True


class TestConvenienceFunctions:
    """Test the convenience functions for creating responses."""
    
    def test_create_allow_response(self):
        """Test create_allow_response function."""
        response = create_allow_response("Request is safe")
        
        assert response.action == "allow"
        assert response.reason == "Request is safe"
        assert response.details == {}
        assert response.confidence == 1.0
    
    def test_create_allow_response_with_details(self):
        """Test create_allow_response function with details."""
        details = {"score": 0.95}
        response = create_allow_response("Request is safe", details)
        
        assert response.action == "allow"
        assert response.reason == "Request is safe"
        assert response.details == details
        assert response.confidence == 1.0
    
    def test_create_block_response(self):
        """Test create_block_response function."""
        response = create_block_response("SQL injection detected")
        
        assert response.action == "block"
        assert response.reason == "SQL injection detected"
        assert response.details == {}
        assert response.confidence == 1.0
    
    def test_create_block_response_with_details(self):
        """Test create_block_response function with details."""
        details = {"pattern": "SELECT * FROM users"}
        response = create_block_response("SQL injection detected", details)
        
        assert response.action == "block"
        assert response.reason == "SQL injection detected"
        assert response.details == details
        assert response.confidence == 1.0
    
    def test_create_decoy_response(self):
        """Test create_decoy_response function."""
        decoy_data = {"fake_credit_cards": ["4111-1111-1111-1111"]}
        response = create_decoy_response("Decoy response triggered", decoy_data)
        
        assert response.action == "decoy"
        assert response.reason == "Decoy response triggered"
        assert response.details["decoy_data"] == decoy_data
        assert response.confidence == 1.0
    
    def test_create_redact_response(self):
        """Test create_redact_response function."""
        redacted_content = "***@example.com"
        response = create_redact_response("PII detected", redacted_content)
        
        assert response.action == "redact"
        assert response.reason == "PII detected"
        assert response.details["redacted_content"] == redacted_content
        assert response.confidence == 1.0


class TestRequestResponseIntegration:
    """Test integration between Request and Response models."""
    
    def test_request_response_workflow(self):
        """Test a complete request-response workflow."""
        # Create a request
        request = Request(
            source="vendor_api_23",
            ip="192.168.1.24",
            payload="SELECT * FROM users WHERE id = 1",
            task="Order veg pizza under $20",
            metadata={"user_id": "123", "session_id": "abc"}
        )
        
        # Simulate guard processing
        if "SELECT" in request.payload:
            response = create_block_response(
                "SQL injection detected",
                {"pattern": "SELECT", "severity": "high"}
            )
        else:
            response = create_allow_response("Request is safe")
        
        # Verify the workflow
        assert request.source == "vendor_api_23"
        assert request.payload == "SELECT * FROM users WHERE id = 1"
        assert response.is_blocked() is True
        assert response.reason == "SQL injection detected"
        assert response.details["pattern"] == "SELECT"
    
    def test_request_response_serialization_workflow(self):
        """Test request-response workflow with serialization."""
        # Create request
        request = Request(
            source="test_source",
            ip="192.168.1.1",
            payload="test payload",
            task="test task"
        )
        
        # Create response
        response = create_allow_response("Request allowed")
        
        # Serialize both
        request_data = request.to_dict()
        response_data = response.to_dict()
        
        # Deserialize both
        restored_request = Request.from_dict(request_data)
        restored_response = Response.from_dict(response_data)
        
        # Verify workflow still works
        assert restored_request.source == request.source
        assert restored_response.is_allowed() is True
        assert restored_response.reason == "Request allowed"
    
    def test_multiple_responses_for_request(self):
        """Test handling multiple responses for a single request."""
        request = Request(
            source="vendor_api",
            ip="192.168.1.1",
            payload="test payload with PII: john@example.com",
            task="test task"
        )
        
        # Simulate multiple guards
        privacy_response = create_redact_response(
            "PII detected",
            "test payload with PII: ***@example.com"
        )
        
        security_response = create_allow_response("No security threats")
        
        # Verify both responses
        assert privacy_response.is_redacted() is True
        assert security_response.is_allowed() is True
        
        # In a real system, these would be combined or prioritized
        final_action = "redact" if privacy_response.is_redacted() else security_response.action
        assert final_action == "redact"


if __name__ == "__main__":
    pytest.main([__file__])
