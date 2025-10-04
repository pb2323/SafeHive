"""
Communication Tools for SafeHive Agents

This module provides tools for agent communication, messaging, and logging.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from langchain.tools import tool
from pydantic import BaseModel, Field

from .base_tools import BaseSafeHiveTool, ToolOutput


class MessageInput(BaseModel):
    """Input for message sending."""
    
    recipient: str = Field(description="Recipient agent ID or name")
    message: str = Field(description="Message content to send")
    priority: str = Field(default="normal", description="Message priority (low, normal, high)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class LoggingInput(BaseModel):
    """Input for logging operations."""
    
    level: str = Field(description="Log level (DEBUG, INFO, WARNING, ERROR)")
    message: str = Field(description="Log message")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


def send_message(recipient: str, message: str, priority: str = "normal") -> str:
    """Send a message to another agent or system component.
    
    Args:
        recipient: The ID or name of the recipient
        message: The message content to send
        priority: Message priority level (low, normal, high)
        
    Returns:
        Confirmation of message sent or error details
    """
    try:
        # In a real implementation, this would integrate with the agent communication system
        timestamp = datetime.now().isoformat()
        
        result = {
            "success": True,
            "message": f"Message sent to {recipient}",
            "details": {
                "recipient": recipient,
                "message": message,
                "priority": priority,
                "timestamp": timestamp,
                "message_id": f"msg_{hash(message + recipient + timestamp) % 1000000:06d}"
            }
        }
        
        return f"✅ Message sent successfully to {recipient} at {timestamp}"
        
    except Exception as e:
        return f"❌ Failed to send message to {recipient}: {str(e)}"


def log_event(level: str, message: str, context: str = "") -> str:
    """Log an event or message with specified level.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        message: The message to log
        context: Additional context information (JSON string)
        
    Returns:
        Confirmation of logging or error details
    """
    try:
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if level.upper() not in valid_levels:
            return f"❌ Invalid log level: {level}. Must be one of {valid_levels}"
        
        # Parse context if provided
        context_data = {}
        if context:
            try:
                import json
                context_data = json.loads(context)
            except json.JSONDecodeError:
                context_data = {"raw_context": context}
        
        # In a real implementation, this would integrate with the logging system
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "level": level.upper(),
            "message": message,
            "context": context_data,
            "timestamp": timestamp,
            "source": "agent_tool"
        }
        
        return f"📝 Logged {level.upper()} event: {message[:50]}{'...' if len(message) > 50 else ''}"
        
    except Exception as e:
        return f"❌ Failed to log event: {str(e)}"


class MessageTool(BaseSafeHiveTool):
    """Tool for sending messages between agents."""
    
    name: str = "send_message"
    description: str = "Send a message to another agent or system component"
    
    def _execute(self, recipient: str, message: str, priority: str = "normal") -> str:
        """Execute message sending."""
        return send_message(recipient, message, priority)


class LoggingTool(BaseSafeHiveTool):
    """Tool for logging events and messages."""
    
    name: str = "log_event"
    description: str = "Log an event or message with specified level"
    
    def _execute(self, level: str, message: str, context: str = "") -> str:
        """Execute logging."""
        return log_event(level, message, context)
