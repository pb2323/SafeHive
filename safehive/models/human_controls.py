"""
Human-in-the-Loop Control Models for SafeHive AI Security Sandbox

This module defines the data models and enums for human intervention controls
that allow users to approve, redact, quarantine, or ignore suspicious activities
during security testing scenarios.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json


class InterventionType(Enum):
    """Types of human intervention available."""
    APPROVE = "approve"          # Allow the activity to proceed
    REDACT = "redact"           # Remove sensitive information before proceeding
    QUARANTINE = "quarantine"   # Isolate or block the activity
    IGNORE = "ignore"           # Dismiss the incident without action


class InterventionStatus(Enum):
    """Status of human intervention."""
    PENDING = "pending"         # Awaiting human decision
    PROCESSING = "processing"   # Human decision received, being processed
    COMPLETED = "completed"     # Intervention completed successfully
    FAILED = "failed"          # Intervention failed
    EXPIRED = "expired"        # Intervention request expired


class IncidentSeverity(Enum):
    """Severity levels for incidents requiring human intervention."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentType(Enum):
    """Types of incidents that may require human intervention."""
    SUSPICIOUS_REQUEST = "suspicious_request"
    POTENTIAL_ATTACK = "potential_attack"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVACY_VIOLATION = "privacy_violation"
    MALICIOUS_INPUT = "malicious_input"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PAYMENT_FRAUD = "payment_fraud"
    SOCIAL_ENGINEERING = "social_engineering"


@dataclass
class InterventionRequest:
    """Request for human intervention in a security incident."""
    
    request_id: str
    incident_type: IncidentType
    severity: IncidentSeverity
    session_id: str
    agent_id: str
    title: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Incident details
    context: Dict[str, Any] = field(default_factory=dict)
    affected_data: Optional[Dict[str, Any]] = None
    
    # Request details
    timeout_seconds: int = 300  # 5 minutes default
    auto_action: Optional[InterventionType] = None  # Action if timeout
    priority: int = 0  # Higher number = higher priority
    
    # Status tracking
    status: InterventionStatus = InterventionStatus.PENDING
    requested_at: datetime = field(default_factory=datetime.now)
    responded_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Response details
    human_response: Optional[InterventionType] = None
    human_reason: Optional[str] = None
    redaction_rules: Optional[List[str]] = None  # For redact actions
    quarantine_duration: Optional[int] = None    # For quarantine actions
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "incident_type": self.incident_type.value,
            "severity": self.severity.value,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "title": self.title,
            "description": self.description,
            "context": self.context,
            "affected_data": self.affected_data,
            "timeout_seconds": self.timeout_seconds,
            "auto_action": self.auto_action.value if self.auto_action else None,
            "priority": self.priority,
            "status": self.status.value,
            "requested_at": self.requested_at.isoformat(),
            "responded_at": self.responded_at.isoformat() if self.responded_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "human_response": self.human_response.value if self.human_response else None,
            "human_reason": self.human_reason,
            "redaction_rules": self.redaction_rules,
            "quarantine_duration": self.quarantine_duration,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterventionRequest':
        """Create from dictionary."""
        return cls(
            request_id=data["request_id"],
            incident_type=IncidentType(data["incident_type"]),
            severity=IncidentSeverity(data["severity"]),
            session_id=data["session_id"],
            agent_id=data["agent_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            title=data["title"],
            description=data["description"],
            context=data.get("context", {}),
            affected_data=data.get("affected_data"),
            timeout_seconds=data.get("timeout_seconds", 300),
            auto_action=InterventionType(data["auto_action"]) if data.get("auto_action") else None,
            priority=data.get("priority", 0),
            status=InterventionStatus(data["status"]),
            requested_at=datetime.fromisoformat(data["requested_at"]),
            responded_at=datetime.fromisoformat(data["responded_at"]) if data.get("responded_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            human_response=InterventionType(data["human_response"]) if data.get("human_response") else None,
            human_reason=data.get("human_reason"),
            redaction_rules=data.get("redaction_rules"),
            quarantine_duration=data.get("quarantine_duration"),
            metadata=data.get("metadata", {})
        )
    
    def is_expired(self) -> bool:
        """Check if the intervention request has expired."""
        if self.status != InterventionStatus.PENDING:
            return False
        
        elapsed = (datetime.now() - self.requested_at).total_seconds()
        return elapsed > self.timeout_seconds
    
    def get_remaining_time(self) -> int:
        """Get remaining time in seconds before auto-action."""
        if self.status != InterventionStatus.PENDING:
            return 0
        
        elapsed = (datetime.now() - self.requested_at).total_seconds()
        remaining = self.timeout_seconds - elapsed
        return max(0, int(remaining)) if remaining > 0 else 0


@dataclass
class InterventionResponse:
    """Response to an intervention request."""
    
    request_id: str
    intervention_type: InterventionType
    reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Action-specific parameters
    redaction_rules: Optional[List[str]] = None
    quarantine_duration: Optional[int] = None
    quarantine_reason: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "intervention_type": self.intervention_type.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "redaction_rules": self.redaction_rules,
            "quarantine_duration": self.quarantine_duration,
            "quarantine_reason": self.quarantine_reason,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterventionResponse':
        """Create from dictionary."""
        return cls(
            request_id=data["request_id"],
            intervention_type=InterventionType(data["intervention_type"]),
            reason=data.get("reason"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            redaction_rules=data.get("redaction_rules"),
            quarantine_duration=data.get("quarantine_duration"),
            quarantine_reason=data.get("quarantine_reason"),
            metadata=data.get("metadata", {})
        )


@dataclass
class HumanControlSession:
    """Session for managing human-in-the-loop controls."""
    
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    # Pending requests
    pending_requests: Dict[str, InterventionRequest] = field(default_factory=dict)
    
    # Completed interventions
    completed_interventions: List[InterventionRequest] = field(default_factory=list)
    
    # Session statistics
    total_requests: int = 0
    approved_count: int = 0
    redacted_count: int = 0
    quarantined_count: int = 0
    ignored_count: int = 0
    auto_action_count: int = 0
    
    def add_request(self, request: InterventionRequest) -> None:
        """Add a new intervention request."""
        self.pending_requests[request.request_id] = request
        self.total_requests += 1
    
    def respond_to_request(self, request_id: str, response: InterventionResponse) -> Optional[InterventionRequest]:
        """Respond to an intervention request."""
        if request_id not in self.pending_requests:
            return None
        
        request = self.pending_requests[request_id]
        request.human_response = response.intervention_type
        request.human_reason = response.reason
        request.responded_at = response.timestamp
        
        # Move to completed and update statistics
        del self.pending_requests[request_id]
        self.completed_interventions.append(request)
        
        # Update statistics
        if response.intervention_type == InterventionType.APPROVE:
            self.approved_count += 1
        elif response.intervention_type == InterventionType.REDACT:
            self.redacted_count += 1
        elif response.intervention_type == InterventionType.QUARANTINE:
            self.quarantined_count += 1
        elif response.intervention_type == InterventionType.IGNORE:
            self.ignored_count += 1
        
        return request
    
    def auto_respond_to_expired(self) -> List[InterventionRequest]:
        """Handle expired requests with auto-actions."""
        expired_requests = []
        
        for request_id, request in list(self.pending_requests.items()):
            if request.is_expired():
                # Apply auto-action if specified
                if request.auto_action:
                    response = InterventionResponse(
                        request_id=request_id,
                        intervention_type=request.auto_action,
                        reason="Auto-action due to timeout"
                    )
                    self.respond_to_request(request_id, response)
                    self.auto_action_count += 1
                    expired_requests.append(request)
                else:
                    # Default to ignore if no auto-action specified
                    response = InterventionResponse(
                        request_id=request_id,
                        intervention_type=InterventionType.IGNORE,
                        reason="Auto-ignore due to timeout"
                    )
                    self.respond_to_request(request_id, response)
                    self.ignored_count += 1
                    expired_requests.append(request)
        
        return expired_requests
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
            "total_requests": self.total_requests,
            "pending_requests": len(self.pending_requests),
            "completed_interventions": len(self.completed_interventions),
            "approved_count": self.approved_count,
            "redacted_count": self.redacted_count,
            "quarantined_count": self.quarantined_count,
            "ignored_count": self.ignored_count,
            "auto_action_count": self.auto_action_count
        }


# Convenience functions for creating intervention requests

def create_suspicious_request_incident(
    session_id: str,
    agent_id: str,
    title: str,
    description: str,
    severity: IncidentSeverity = IncidentSeverity.MEDIUM,
    context: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 300
) -> InterventionRequest:
    """Create a suspicious request intervention."""
    import uuid
    
    return InterventionRequest(
        request_id=str(uuid.uuid4()),
        incident_type=IncidentType.SUSPICIOUS_REQUEST,
        severity=severity,
        session_id=session_id,
        agent_id=agent_id,
        title=title,
        description=description,
        context=context or {},
        timeout_seconds=timeout_seconds,
        auto_action=InterventionType.IGNORE if severity == IncidentSeverity.LOW else None
    )


def create_potential_attack_incident(
    session_id: str,
    agent_id: str,
    title: str,
    description: str,
    severity: IncidentSeverity = IncidentSeverity.HIGH,
    context: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 180  # Shorter timeout for potential attacks
) -> InterventionRequest:
    """Create a potential attack intervention."""
    import uuid
    
    return InterventionRequest(
        request_id=str(uuid.uuid4()),
        incident_type=IncidentType.POTENTIAL_ATTACK,
        severity=severity,
        session_id=session_id,
        agent_id=agent_id,
        title=title,
        description=description,
        context=context or {},
        timeout_seconds=timeout_seconds,
        auto_action=InterventionType.QUARANTINE if severity == IncidentSeverity.CRITICAL else None
    )


def create_privacy_violation_incident(
    session_id: str,
    agent_id: str,
    title: str,
    description: str,
    affected_data: Dict[str, Any],
    severity: IncidentSeverity = IncidentSeverity.MEDIUM,
    context: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 240
) -> InterventionRequest:
    """Create a privacy violation intervention."""
    import uuid
    
    return InterventionRequest(
        request_id=str(uuid.uuid4()),
        incident_type=IncidentType.PRIVACY_VIOLATION,
        severity=severity,
        session_id=session_id,
        agent_id=agent_id,
        title=title,
        description=description,
        affected_data=affected_data,
        context=context or {},
        timeout_seconds=timeout_seconds,
        auto_action=InterventionType.REDACT if severity in [IncidentSeverity.MEDIUM, IncidentSeverity.HIGH] else None
    )
