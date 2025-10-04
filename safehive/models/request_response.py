"""
Request and Response Models for Agent Communication

This module defines the standardized data structures used for communication
between agents, guards, and other components in the SafeHive system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class Request:
    """
    Standardized request object for agent communication.
    
    This class represents a request that flows through the system,
    containing all necessary information for processing and security analysis.
    """
    source: str  # e.g., "vendor_api_23", "orchestrator", "user_twin"
    ip: str      # e.g., "192.168.1.24" or "localhost"
    payload: str # The actual request content/message
    task: str    # Original task context, e.g., "Order veg pizza under $20"
    timestamp: Optional[datetime] = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary for serialization."""
        return {
            "source": self.source,
            "ip": self.ip,
            "payload": self.payload,
            "task": self.task,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Request":
        """Create request from dictionary."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
        
        return cls(
            source=data["source"],
            ip=data["ip"],
            payload=data["payload"],
            task=data["task"],
            timestamp=timestamp,
            metadata=data.get("metadata", {})
        )


@dataclass
class Response:
    """
    Standardized response object for guard and agent communication.
    
    This class represents a response from a guard or agent, containing
    the action to take and supporting information.
    """
    action: str  # "allow" | "block" | "decoy" | "redact"
    reason: str  # Human-readable explanation
    details: Optional[Dict[str, Any]] = field(default_factory=dict)
    confidence: float = 1.0  # Confidence level (0.0 to 1.0)
    timestamp: Optional[datetime] = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate response data after initialization."""
        if self.action not in ["allow", "block", "decoy", "redact"]:
            raise ValueError(f"Invalid action: {self.action}. Must be one of: allow, block, decoy, redact")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {self.confidence}. Must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for serialization."""
        return {
            "action": self.action,
            "reason": self.reason,
            "details": self.details,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Response":
        """Create response from dictionary."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
        else:
            # If no timestamp provided, create a new one
            timestamp = datetime.now()
        
        return cls(
            action=data["action"],
            reason=data["reason"],
            details=data.get("details", {}),
            confidence=data.get("confidence", 1.0),
            timestamp=timestamp
        )
    
    def is_allowed(self) -> bool:
        """Check if the response allows the request to proceed."""
        return self.action == "allow"
    
    def is_blocked(self) -> bool:
        """Check if the response blocks the request."""
        return self.action == "block"
    
    def is_decoy(self) -> bool:
        """Check if the response is a decoy/synthetic response."""
        return self.action == "decoy"
    
    def is_redacted(self) -> bool:
        """Check if the response requires redaction."""
        return self.action == "redact"


# Convenience functions for creating common responses
def create_allow_response(reason: str = "Request allowed", details: Optional[Dict[str, Any]] = None) -> Response:
    """Create an allow response."""
    return Response(action="allow", reason=reason, details=details or {})


def create_block_response(reason: str, details: Optional[Dict[str, Any]] = None) -> Response:
    """Create a block response."""
    return Response(action="block", reason=reason, details=details or {})


def create_decoy_response(reason: str, decoy_data: Dict[str, Any]) -> Response:
    """Create a decoy response with synthetic data."""
    return Response(
        action="decoy", 
        reason=reason, 
        details={"decoy_data": decoy_data}
    )


def create_redact_response(reason: str, redacted_content: str) -> Response:
    """Create a redact response with redacted content."""
    return Response(
        action="redact", 
        reason=reason, 
        details={"redacted_content": redacted_content}
    )
