"""
SafeHive Security Guards Module

This module contains the base Guard class and all security guard implementations.
Security guards are responsible for intercepting and analyzing agent communications
to detect and prevent malicious activities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

from ..models.request_response import Request, Response


class Guard(ABC):
    """
    Abstract base class for all security guards.
    
    All guards must implement the standard interface to ensure consistent
    behavior and integration with the orchestrator.
    """
    
    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize the guard.
        
        Args:
            name: Human-readable name of the guard
            enabled: Whether the guard is currently enabled
        """
        self.name = name
        self.enabled = enabled
        self.config = {}
        self.metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "allowed_requests": 0,
            "decoy_responses": 0,
            "redacted_responses": 0,
            "last_activity": None
        }
    
    @abstractmethod
    def inspect(self, request: Request) -> Response:
        """
        Inspect a request and return a response with action and details.
        
        Args:
            request: Standardized request object with source, payload, task info
            
        Returns:
            Response object with action (allow/block/decoy/redact), reason, and details
        """
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the guard with settings from YAML config.
        
        Args:
            config: Dictionary of configuration parameters
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status and metrics for the guard.
        
        Returns:
            Dictionary with guard status, metrics, and health info
        """
        pass
    
    def is_enabled(self) -> bool:
        """
        Check if the guard is currently enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return self.enabled
    
    def enable(self) -> None:
        """Enable the guard."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the guard."""
        self.enabled = False
    
    def _update_metrics(self, action: str) -> None:
        """
        Update guard metrics based on the action taken.
        
        Args:
            action: The action taken (allow, block, decoy, redact)
        """
        self.metrics["total_requests"] += 1
        self.metrics["last_activity"] = datetime.now().isoformat()
        
        if action == "allow":
            self.metrics["allowed_requests"] += 1
        elif action == "block":
            self.metrics["blocked_requests"] += 1
        elif action == "decoy":
            self.metrics["decoy_responses"] += 1
        elif action == "redact":
            self.metrics["redacted_responses"] += 1
    
    def _create_response(self, action: str, reason: str, details: Optional[Dict[str, Any]] = None) -> Response:
        """
        Create a standardized response object.
        
        Args:
            action: The action to take (allow/block/decoy/redact)
            reason: Human-readable explanation
            details: Additional context and data
            
        Returns:
            Response object
        """
        self._update_metrics(action)
        
        return Response(
            action=action,
            reason=reason,
            details=details or {},
            timestamp=datetime.now()
        )


# Import all guard implementations when they are created
# from .privacy_sentry import PrivacySentry
# from .task_navigator import TaskNavigator  
# from .prompt_sanitizer import PromptSanitizer

__all__ = [
    "Guard",
    # "PrivacySentry",
    # "TaskNavigator", 
    # "PromptSanitizer"
]
