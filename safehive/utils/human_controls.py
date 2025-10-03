"""
Human-in-the-Loop Control Manager for SafeHive AI Security Sandbox

This module provides the core functionality for managing human intervention
requests during security testing scenarios. It handles request queuing,
timeout management, and response processing.
"""

import asyncio
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import asdict

from safehive.models.human_controls import (
    InterventionRequest, InterventionResponse, HumanControlSession,
    InterventionType, InterventionStatus, IncidentType, IncidentSeverity
)
from safehive.utils.logger import get_logger
from safehive.utils.metrics import record_metric, record_event, MetricType

logger = get_logger(__name__)


class HumanControlManager:
    """Manager for human-in-the-loop controls."""
    
    def __init__(self, default_timeout: int = 300, auto_cleanup_interval: int = 60):
        """
        Initialize the human control manager.
        
        Args:
            default_timeout: Default timeout in seconds for intervention requests
            auto_cleanup_interval: Interval in seconds for auto-cleanup of expired requests
        """
        self.default_timeout = default_timeout
        self.auto_cleanup_interval = auto_cleanup_interval
        
        # Active sessions
        self.active_sessions: Dict[str, HumanControlSession] = {}
        
        # Pending requests across all sessions
        self.pending_requests: Dict[str, InterventionRequest] = {}
        
        # Callbacks for different intervention types
        self.intervention_callbacks: Dict[InterventionType, List[Callable]] = {
            InterventionType.APPROVE: [],
            InterventionType.REDACT: [],
            InterventionType.QUARANTINE: [],
            InterventionType.IGNORE: []
        }
        
        # Auto-cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()
        
        logger.info(f"HumanControlManager initialized with default_timeout={default_timeout}s")
    
    def create_session(self, session_id: str) -> HumanControlSession:
        """Create a new human control session."""
        if session_id in self.active_sessions:
            logger.warning(f"Session {session_id} already exists, returning existing session")
            return self.active_sessions[session_id]
        
        session = HumanControlSession(session_id=session_id)
        self.active_sessions[session_id] = session
        
        logger.info(f"Created human control session: {session_id}")
        record_metric("human_controls.session_created", 1, MetricType.COUNTER, {"session_id": session_id})
        
        return session
    
    def get_session(self, session_id: str) -> Optional[HumanControlSession]:
        """Get an existing session."""
        return self.active_sessions.get(session_id)
    
    def close_session(self, session_id: str) -> bool:
        """Close a session and handle remaining pending requests."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.is_active = False
        
        # Handle remaining pending requests with auto-ignore
        pending_requests = list(session.pending_requests.values())
        for request in pending_requests:
            response = InterventionResponse(
                request_id=request.request_id,
                intervention_type=InterventionType.IGNORE,
                reason="Session closed - auto-ignore remaining requests"
            )
            self._process_intervention_response(request, response)
        
        del self.active_sessions[session_id]
        
        logger.info(f"Closed human control session: {session_id}")
        record_metric("human_controls.session_closed", 1, MetricType.COUNTER, {"session_id": session_id})
        
        return True
    
    def request_intervention(
        self,
        session_id: str,
        agent_id: str,
        incident_type: IncidentType,
        severity: IncidentSeverity,
        title: str,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        affected_data: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
        auto_action: Optional[InterventionType] = None,
        priority: int = 0
    ) -> InterventionRequest:
        """
        Request human intervention for a security incident.
        
        Args:
            session_id: ID of the sandbox session
            agent_id: ID of the agent that triggered the incident
            incident_type: Type of incident
            severity: Severity level
            title: Human-readable title
            description: Detailed description
            context: Additional context information
            affected_data: Data affected by the incident
            timeout_seconds: Timeout for the request (default: manager default)
            auto_action: Action to take if timeout occurs
            priority: Priority level (higher = more important)
        
        Returns:
            InterventionRequest object
        """
        # Get or create session
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
        
        # Create intervention request
        request = InterventionRequest(
            request_id=str(uuid.uuid4()),
            incident_type=incident_type,
            severity=severity,
            session_id=session_id,
            agent_id=agent_id,
            title=title,
            description=description,
            context=context or {},
            affected_data=affected_data,
            timeout_seconds=timeout_seconds or self.default_timeout,
            auto_action=auto_action,
            priority=priority
        )
        
        # Add to session and manager
        session.add_request(request)
        self.pending_requests[request.request_id] = request
        
        logger.info(f"Created intervention request {request.request_id} for {incident_type.value} "
                   f"(severity: {severity.value}, session: {session_id})")
        
        record_metric("human_controls.intervention_requested", 1, MetricType.COUNTER, {
            "session_id": session_id,
            "agent_id": agent_id,
            "incident_type": incident_type.value,
            "severity": severity.value
        })
        
        record_event("human_controls.intervention_requested", 
                    f"Intervention requested: {title}", {
                        "request_id": request.request_id,
                        "session_id": session_id,
                        "agent_id": agent_id,
                        "incident_type": incident_type.value,
                        "severity": severity.value
                    })
        
        return request
    
    def respond_to_intervention(
        self,
        request_id: str,
        intervention_type: InterventionType,
        reason: Optional[str] = None,
        redaction_rules: Optional[List[str]] = None,
        quarantine_duration: Optional[int] = None,
        quarantine_reason: Optional[str] = None
    ) -> bool:
        """
        Respond to an intervention request.
        
        Args:
            request_id: ID of the intervention request
            intervention_type: Type of intervention to apply
            reason: Human-readable reason for the intervention
            redaction_rules: Rules for redaction (if intervention_type is REDACT)
            quarantine_duration: Duration for quarantine (if intervention_type is QUARANTINE)
            quarantine_reason: Reason for quarantine
        
        Returns:
            True if response was processed successfully, False otherwise
        """
        if request_id not in self.pending_requests:
            logger.error(f"Intervention request {request_id} not found")
            return False
        
        request = self.pending_requests[request_id]
        
        # Check if request has expired
        if request.is_expired():
            logger.warning(f"Intervention request {request_id} has expired")
            return False
        
        # Create response
        response = InterventionResponse(
            request_id=request_id,
            intervention_type=intervention_type,
            reason=reason,
            redaction_rules=redaction_rules,
            quarantine_duration=quarantine_duration,
            quarantine_reason=quarantine_reason
        )
        
        # Process response
        return self._process_intervention_response(request, response)
    
    def _process_intervention_response(
        self,
        request: InterventionRequest,
        response: InterventionResponse
    ) -> bool:
        """Process an intervention response."""
        try:
            # Update request with response
            request.human_response = response.intervention_type
            request.human_reason = response.reason
            request.responded_at = response.timestamp
            request.status = InterventionStatus.PROCESSING
            request.redaction_rules = response.redaction_rules
            request.quarantine_duration = response.quarantine_duration
            request.completed_at = datetime.now()
            request.status = InterventionStatus.COMPLETED
            
            # Remove from pending
            if request.request_id in self.pending_requests:
                del self.pending_requests[request.request_id]
            
            # Update session
            session = self.get_session(request.session_id)
            if session:
                session.respond_to_request(request.request_id, response)
            
            # Execute callbacks
            self._execute_callbacks(request, response)
            
            logger.info(f"Processed intervention response {request.request_id}: {response.intervention_type.value}")
            
            record_metric("human_controls.intervention_completed", 1, MetricType.COUNTER, {
                "session_id": request.session_id,
                "agent_id": request.agent_id,
                "intervention_type": response.intervention_type.value,
                "incident_type": request.incident_type.value
            })
            
            record_event("human_controls.intervention_completed",
                        f"Intervention completed: {request.title}", {
                            "request_id": request.request_id,
                            "intervention_type": response.intervention_type.value,
                            "reason": response.reason
                        })
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing intervention response {request.request_id}: {e}")
            request.status = InterventionStatus.FAILED
            return False
    
    def _execute_callbacks(self, request: InterventionRequest, response: InterventionResponse):
        """Execute registered callbacks for the intervention type."""
        callbacks = self.intervention_callbacks.get(response.intervention_type, [])
        
        for callback in callbacks:
            try:
                callback(request, response)
            except Exception as e:
                logger.error(f"Error executing callback for {response.intervention_type.value}: {e}")
    
    def register_callback(self, intervention_type: InterventionType, callback: Callable):
        """Register a callback for a specific intervention type."""
        if intervention_type not in self.intervention_callbacks:
            self.intervention_callbacks[intervention_type] = []
        
        self.intervention_callbacks[intervention_type].append(callback)
        logger.info(f"Registered callback for {intervention_type.value}")
    
    def get_pending_requests(self, session_id: Optional[str] = None) -> List[InterventionRequest]:
        """Get pending intervention requests, optionally filtered by session."""
        if session_id:
            return [req for req in self.pending_requests.values() if req.session_id == session_id]
        return list(self.pending_requests.values())
    
    def get_session_statistics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific session."""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return session.get_statistics()
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global statistics across all sessions."""
        total_requests = sum(session.total_requests for session in self.active_sessions.values())
        total_pending = len(self.pending_requests)
        total_completed = sum(len(session.completed_interventions) for session in self.active_sessions.values())
        
        return {
            "active_sessions": len(self.active_sessions),
            "total_requests": total_requests,
            "pending_requests": total_pending,
            "completed_interventions": total_completed,
            "sessions": [session.get_statistics() for session in self.active_sessions.values()]
        }
    
    def _start_cleanup_thread(self):
        """Start the auto-cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="HumanControlCleanup"
        )
        self._cleanup_thread.start()
        logger.info("Started human control cleanup thread")
    
    def _cleanup_loop(self):
        """Main cleanup loop for expired requests."""
        while not self._stop_cleanup.is_set():
            try:
                self._cleanup_expired_requests()
                self._stop_cleanup.wait(self.auto_cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                self._stop_cleanup.wait(self.auto_cleanup_interval)
    
    def _cleanup_expired_requests(self):
        """Clean up expired requests."""
        expired_count = 0
        
        for session in self.active_sessions.values():
            if session.is_active:
                expired_requests = session.auto_respond_to_expired()
                for request in expired_requests:
                    # Create auto-response
                    auto_action = request.auto_action or InterventionType.IGNORE
                    response = InterventionResponse(
                        request_id=request.request_id,
                        intervention_type=auto_action,
                        reason=f"Auto-{auto_action.value} due to timeout"
                    )
                    self._process_intervention_response(request, response)
                    expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Auto-processed {expired_count} expired intervention requests")
            record_metric("human_controls.auto_processed_expired", expired_count, MetricType.COUNTER)
    
    def shutdown(self):
        """Shutdown the manager and cleanup resources."""
        logger.info("Shutting down HumanControlManager")
        
        # Stop cleanup thread
        self._stop_cleanup.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        # Close all sessions
        for session_id in list(self.active_sessions.keys()):
            self.close_session(session_id)
        
        logger.info("HumanControlManager shutdown complete")


# Global instance
_human_control_manager: Optional[HumanControlManager] = None


def get_human_control_manager() -> HumanControlManager:
    """Get the global human control manager instance."""
    global _human_control_manager
    if _human_control_manager is None:
        _human_control_manager = HumanControlManager()
    return _human_control_manager


def request_human_intervention(
    session_id: str,
    agent_id: str,
    incident_type: IncidentType,
    severity: IncidentSeverity,
    title: str,
    description: str,
    context: Optional[Dict[str, Any]] = None,
    affected_data: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[int] = None,
    auto_action: Optional[InterventionType] = None,
    priority: int = 0
) -> InterventionRequest:
    """Convenience function to request human intervention."""
    manager = get_human_control_manager()
    return manager.request_intervention(
        session_id=session_id,
        agent_id=agent_id,
        incident_type=incident_type,
        severity=severity,
        title=title,
        description=description,
        context=context,
        affected_data=affected_data,
        timeout_seconds=timeout_seconds,
        auto_action=auto_action,
        priority=priority
    )


def respond_to_intervention(
    request_id: str,
    intervention_type: InterventionType,
    reason: Optional[str] = None,
    **kwargs
) -> bool:
    """Convenience function to respond to an intervention."""
    manager = get_human_control_manager()
    return manager.respond_to_intervention(
        request_id=request_id,
        intervention_type=intervention_type,
        reason=reason,
        **kwargs
    )
