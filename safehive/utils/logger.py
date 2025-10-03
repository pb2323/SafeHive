"""
Logging utilities for SafeHive AI Security Sandbox

This module provides structured logging functionality using loguru
for consistent logging across the entire system.
"""

import sys
import os
from typing import Optional, Dict, Any
from loguru import logger as loguru_logger


def get_logger(name: str) -> Any:
    """
    Get a logger instance for the given module name.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance
    """
    return loguru_logger.bind(module=name)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True,
    **kwargs
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        structured: Whether to use structured logging format
        **kwargs: Additional logging configuration options
    """
    # Remove default handler
    loguru_logger.remove()
    
    # Console handler
    if structured:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan> | "
            "<level>{message}</level>"
        )
    else:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        )
    
    loguru_logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        if structured:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{module} | "
                "{message}"
            )
        else:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{message}"
            )
        
        loguru_logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation=kwargs.get("max_file_size", "10MB"),
            retention=kwargs.get("backup_count", 5),
            compression="zip",
            backtrace=True,
            diagnose=True
        )


def setup_alert_logging(
    alerts_file: str = "logs/alerts.log",
    level: str = "WARNING"
) -> None:
    """
    Set up alert-specific logging for security events.
    
    Args:
        alerts_file: Path to alerts log file
        level: Minimum level for alerts
    """
    # Ensure alerts directory exists
    os.makedirs(os.path.dirname(alerts_file), exist_ok=True)
    
    # Alert-specific format
    alert_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "ALERT | "
        "{level: <8} | "
        "{module} | "
        "{message}"
    )
    
    loguru_logger.add(
        alerts_file,
        format=alert_format,
        level=level,
        rotation="10MB",
        retention=10,
        compression="zip",
        filter=lambda record: record["level"].name in ["WARNING", "ERROR", "CRITICAL"]
    )


def setup_agent_logging(
    conversations_file: str = "logs/agent_conversations.log",
    level: str = "INFO"
) -> None:
    """
    Set up agent conversation logging.
    
    Args:
        conversations_file: Path to conversations log file
        level: Logging level for conversations
    """
    # Ensure conversations directory exists
    os.makedirs(os.path.dirname(conversations_file), exist_ok=True)
    
    # Agent conversation format
    conversation_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "AGENT | "
        "{level: <8} | "
        "{module} | "
        "{message}"
    )
    
    loguru_logger.add(
        conversations_file,
        format=conversation_format,
        level=level,
        rotation="50MB",
        retention=7,
        compression="zip",
        filter=lambda record: "agent" in record["module"].lower() or "conversation" in record["message"].lower()
    )


def log_security_event(
    event_type: str,
    source: str,
    details: Dict[str, Any],
    severity: str = "WARNING"
) -> None:
    """
    Log a security event with structured data.
    
    Args:
        event_type: Type of security event
        source: Source of the event
        details: Additional event details
        severity: Event severity level
    """
    logger = get_logger("security")
    
    message = f"Security Event: {event_type} from {source}"
    
    if severity.upper() == "CRITICAL":
        logger.critical(message, extra={"event_type": event_type, "source": source, "details": details})
    elif severity.upper() == "ERROR":
        logger.error(message, extra={"event_type": event_type, "source": source, "details": details})
    elif severity.upper() == "WARNING":
        logger.warning(message, extra={"event_type": event_type, "source": source, "details": details})
    else:
        logger.info(message, extra={"event_type": event_type, "source": source, "details": details})


def log_agent_interaction(
    agent_from: str,
    agent_to: str,
    message_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log agent interaction for debugging and analysis.
    
    Args:
        agent_from: Source agent name
        agent_to: Destination agent name
        message_type: Type of message/interaction
        content: Message content
        metadata: Additional metadata
    """
    logger = get_logger("agent_interaction")
    
    message = f"Agent Interaction: {agent_from} -> {agent_to} ({message_type})"
    
    logger.info(
        message,
        extra={
            "agent_from": agent_from,
            "agent_to": agent_to,
            "message_type": message_type,
            "content": content,
            "metadata": metadata or {}
        }
    )


def log_guard_action(
    guard_name: str,
    action: str,
    request_source: str,
    reason: str,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log guard action for security monitoring.
    
    Args:
        guard_name: Name of the guard
        action: Action taken (allow, block, decoy, redact)
        request_source: Source of the request
        reason: Reason for the action
        details: Additional action details
    """
    logger = get_logger("guard_action")
    
    message = f"Guard Action: {guard_name} {action} request from {request_source}"
    
    if action in ["block", "decoy"]:
        logger.warning(message, extra={
            "guard_name": guard_name,
            "action": action,
            "request_source": request_source,
            "reason": reason,
            "details": details or {}
        })
    else:
        logger.info(message, extra={
            "guard_name": guard_name,
            "action": action,
            "request_source": request_source,
            "reason": reason,
            "details": details or {}
        })


# Initialize default logging
setup_logging()
