"""
Guard Response Formatter - Response Formatting and Logging System

This module implements a comprehensive system for formatting guard responses
and logging them with proper agent context for monitoring, debugging, and
compliance purposes.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
from abc import ABC, abstractmethod

from ..utils.logger import get_logger
from ..utils.metrics import record_metric, MetricType

logger = get_logger(__name__)


class ResponseFormat(Enum):
    """Available response formats."""
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"


class LogLevel(Enum):
    """Log levels for guard responses."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ResponseType(Enum):
    """Types of guard responses."""
    ALLOW = "allow"
    BLOCK = "block"
    FILTER = "filter"
    SANITIZE = "sanitize"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"
    REDIRECT = "redirect"
    WARN = "warn"
    AUDIT = "audit"


class ContextType(Enum):
    """Types of agent context."""
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    VENDOR_COMMUNICATION = "vendor_communication"
    SYSTEM_EVENT = "system_event"
    SECURITY_SCAN = "security_scan"
    TASK_EXECUTION = "task_execution"


@dataclass
class AgentContext:
    """Context information about the agent and environment."""
    agent_id: str
    agent_type: str
    session_id: str
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    conversation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class GuardResponse:
    """Formatted guard response with context."""
    response_id: str
    guard_id: str
    guard_type: str
    response_type: ResponseType
    original_input: str
    processed_output: str
    confidence: float
    processing_time_ms: float
    agent_context: AgentContext
    threats_detected: List[Dict[str, Any]] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['response_type'] = self.response_type.value
        data['agent_context'] = self.agent_context.to_dict()
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class LogEntry:
    """Structured log entry for guard responses."""
    log_id: str
    guard_response: GuardResponse
    log_level: LogLevel
    message: str
    context_type: ContextType
    tags: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['guard_response'] = self.guard_response.to_dict()
        data['log_level'] = self.log_level.value
        data['context_type'] = self.context_type.value
        data['tags'] = list(self.tags)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class ResponseFormatter(ABC):
    """Abstract base class for response formatters."""
    
    @abstractmethod
    def format_response(self, response: GuardResponse) -> str:
        """Format a guard response."""
        pass
    
    @abstractmethod
    def get_content_type(self) -> str:
        """Get the content type for the formatted response."""
        pass


class JSONResponseFormatter(ResponseFormatter):
    """JSON response formatter."""
    
    def format_response(self, response: GuardResponse) -> str:
        """Format response as JSON."""
        return json.dumps(response.to_dict(), indent=2, ensure_ascii=False)
    
    def get_content_type(self) -> str:
        """Get JSON content type."""
        return "application/json"


class XMLResponseFormatter(ResponseFormatter):
    """XML response formatter."""
    
    def format_response(self, response: GuardResponse) -> str:
        """Format response as XML."""
        data = response.to_dict()
        
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_parts.append('<GuardResponse>')
        
        # Basic fields
        xml_parts.append(f'  <ResponseId>{data["response_id"]}</ResponseId>')
        xml_parts.append(f'  <GuardId>{data["guard_id"]}</GuardId>')
        xml_parts.append(f'  <GuardType>{data["guard_type"]}</GuardType>')
        xml_parts.append(f'  <ResponseType>{data["response_type"]}</ResponseType>')
        xml_parts.append(f'  <Confidence>{data["confidence"]}</Confidence>')
        xml_parts.append(f'  <ProcessingTimeMs>{data["processing_time_ms"]}</ProcessingTimeMs>')
        xml_parts.append(f'  <Timestamp>{data["timestamp"]}</Timestamp>')
        
        # Input/Output
        xml_parts.append('  <OriginalInput><![CDATA[' + data["original_input"] + ']]></OriginalInput>')
        xml_parts.append('  <ProcessedOutput><![CDATA[' + data["processed_output"] + ']]></ProcessedOutput>')
        
        # Agent Context
        context = data["agent_context"]
        xml_parts.append('  <AgentContext>')
        xml_parts.append(f'    <AgentId>{context["agent_id"]}</AgentId>')
        xml_parts.append(f'    <AgentType>{context["agent_type"]}</AgentType>')
        xml_parts.append(f'    <SessionId>{context["session_id"]}</SessionId>')
        if context.get("user_id"):
            xml_parts.append(f'    <UserId>{context["user_id"]}</UserId>')
        if context.get("task_id"):
            xml_parts.append(f'    <TaskId>{context["task_id"]}</TaskId>')
        xml_parts.append(f'    <ContextTimestamp>{context["timestamp"]}</ContextTimestamp>')
        xml_parts.append('  </AgentContext>')
        
        # Threats detected
        if data["threats_detected"]:
            xml_parts.append('  <ThreatsDetected>')
            for threat in data["threats_detected"]:
                xml_parts.append('    <Threat>')
                xml_parts.append(f'      <Type>{threat.get("type", "")}</Type>')
                xml_parts.append(f'      <Severity>{threat.get("severity", "")}</Severity>')
                xml_parts.append(f'      <Confidence>{threat.get("confidence", 0)}</Confidence>')
                xml_parts.append('    </Threat>')
            xml_parts.append('  </ThreatsDetected>')
        
        # Actions taken
        if data["actions_taken"]:
            xml_parts.append('  <ActionsTaken>')
            for action in data["actions_taken"]:
                xml_parts.append(f'    <Action>{action}</Action>')
            xml_parts.append('  </ActionsTaken>')
        
        xml_parts.append('</GuardResponse>')
        
        return '\n'.join(xml_parts)
    
    def get_content_type(self) -> str:
        """Get XML content type."""
        return "application/xml"


class TextResponseFormatter(ResponseFormatter):
    """Text response formatter."""
    
    def format_response(self, response: GuardResponse) -> str:
        """Format response as human-readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append("GUARD RESPONSE")
        lines.append("=" * 60)
        lines.append(f"Response ID: {response.response_id}")
        lines.append(f"Guard: {response.guard_id} ({response.guard_type})")
        lines.append(f"Type: {response.response_type.value.upper()}")
        lines.append(f"Confidence: {response.confidence:.2f}")
        lines.append(f"Processing Time: {response.processing_time_ms:.2f}ms")
        lines.append(f"Timestamp: {response.timestamp.isoformat()}")
        lines.append("")
        
        lines.append("AGENT CONTEXT:")
        lines.append("-" * 20)
        lines.append(f"Agent ID: {response.agent_context.agent_id}")
        lines.append(f"Agent Type: {response.agent_context.agent_type}")
        lines.append(f"Session ID: {response.agent_context.session_id}")
        if response.agent_context.user_id:
            lines.append(f"User ID: {response.agent_context.user_id}")
        if response.agent_context.task_id:
            lines.append(f"Task ID: {response.agent_context.task_id}")
        lines.append("")
        
        lines.append("INPUT/OUTPUT:")
        lines.append("-" * 20)
        lines.append("Original Input:")
        lines.append(f"  {response.original_input}")
        lines.append("")
        lines.append("Processed Output:")
        lines.append(f"  {response.processed_output}")
        lines.append("")
        
        if response.threats_detected:
            lines.append("THREATS DETECTED:")
            lines.append("-" * 20)
            for i, threat in enumerate(response.threats_detected, 1):
                lines.append(f"{i}. Type: {threat.get('type', 'Unknown')}")
                lines.append(f"   Severity: {threat.get('severity', 'Unknown')}")
                lines.append(f"   Confidence: {threat.get('confidence', 0):.2f}")
                lines.append("")
        
        if response.actions_taken:
            lines.append("ACTIONS TAKEN:")
            lines.append("-" * 20)
            for i, action in enumerate(response.actions_taken, 1):
                lines.append(f"{i}. {action}")
            lines.append("")
        
        if response.recommendations:
            lines.append("RECOMMENDATIONS:")
            lines.append("-" * 20)
            for i, rec in enumerate(response.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)
    
    def get_content_type(self) -> str:
        """Get text content type."""
        return "text/plain"


class MarkdownResponseFormatter(ResponseFormatter):
    """Markdown response formatter."""
    
    def format_response(self, response: GuardResponse) -> str:
        """Format response as Markdown."""
        lines = []
        lines.append("# Guard Response")
        lines.append("")
        lines.append(f"**Response ID:** `{response.response_id}`")
        lines.append(f"**Guard:** {response.guard_id} ({response.guard_type})")
        lines.append(f"**Type:** `{response.response_type.value.upper()}`")
        lines.append(f"**Confidence:** {response.confidence:.2f}")
        lines.append(f"**Processing Time:** {response.processing_time_ms:.2f}ms")
        lines.append(f"**Timestamp:** {response.timestamp.isoformat()}")
        lines.append("")
        
        lines.append("## Agent Context")
        lines.append("")
        lines.append(f"- **Agent ID:** `{response.agent_context.agent_id}`")
        lines.append(f"- **Agent Type:** {response.agent_context.agent_type}")
        lines.append(f"- **Session ID:** `{response.agent_context.session_id}`")
        if response.agent_context.user_id:
            lines.append(f"- **User ID:** `{response.agent_context.user_id}`")
        if response.agent_context.task_id:
            lines.append(f"- **Task ID:** `{response.agent_context.task_id}`")
        lines.append("")
        
        lines.append("## Input/Output")
        lines.append("")
        lines.append("### Original Input")
        lines.append("```")
        lines.append(response.original_input)
        lines.append("```")
        lines.append("")
        lines.append("### Processed Output")
        lines.append("```")
        lines.append(response.processed_output)
        lines.append("```")
        lines.append("")
        
        if response.threats_detected:
            lines.append("## Threats Detected")
            lines.append("")
            for i, threat in enumerate(response.threats_detected, 1):
                lines.append(f"### Threat {i}")
                lines.append(f"- **Type:** {threat.get('type', 'Unknown')}")
                lines.append(f"- **Severity:** {threat.get('severity', 'Unknown')}")
                lines.append(f"- **Confidence:** {threat.get('confidence', 0):.2f}")
                lines.append("")
        
        if response.actions_taken:
            lines.append("## Actions Taken")
            lines.append("")
            for i, action in enumerate(response.actions_taken, 1):
                lines.append(f"{i}. {action}")
            lines.append("")
        
        if response.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for i, rec in enumerate(response.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def get_content_type(self) -> str:
        """Get Markdown content type."""
        return "text/markdown"


class ResponseLogger:
    """Logger for guard responses with structured logging."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_guard_logs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Logging configuration
        self.log_levels: Set[LogLevel] = {LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR}
        self.context_types: Set[ContextType] = set(ContextType)
        self.enabled_tags: Set[str] = set()
        
        # Statistics
        self.total_logs = 0
        self.logs_by_level: Dict[LogLevel, int] = {level: 0 for level in LogLevel}
        self.logs_by_context: Dict[ContextType, int] = {context: 0 for context in ContextType}
        
        logger.info("Response Logger initialized")
    
    def log_response(self, response: GuardResponse, log_level: LogLevel, 
                    context_type: ContextType, message: str, 
                    tags: Optional[Set[str]] = None) -> LogEntry:
        """Log a guard response with context."""
        # Check if logging is enabled for this level and context
        if log_level not in self.log_levels or context_type not in self.context_types:
            return None
        
        # Filter by tags if specified
        if self.enabled_tags and tags:
            if not tags.intersection(self.enabled_tags):
                return None
        
        # Create log entry
        log_entry = LogEntry(
            log_id=str(uuid.uuid4()),
            guard_response=response,
            log_level=log_level,
            message=message,
            context_type=context_type,
            tags=tags or set()
        )
        
        # Log to file
        self._write_log_entry(log_entry)
        
        # Update statistics
        self.total_logs += 1
        self.logs_by_level[log_level] += 1
        self.logs_by_context[context_type] += 1
        
        # Record metrics
        record_metric("response_logger.logs_created", 1, MetricType.COUNTER, {
            "log_level": log_level.value,
            "context_type": context_type.value,
            "guard_type": response.guard_type
        })
        
        logger.info(f"Logged guard response: {response.response_id} ({log_level.value})")
        
        return log_entry
    
    def _write_log_entry(self, log_entry: LogEntry) -> None:
        """Write log entry to file."""
        try:
            # Create daily log file
            date_str = log_entry.timestamp.strftime("%Y-%m-%d")
            log_file = self.storage_path / f"guard_responses_{date_str}.jsonl"
            
            # Append log entry as JSONL
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry.to_dict(), ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write log entry: {e}")
    
    def get_logs(self, start_time: Optional[datetime] = None, 
                end_time: Optional[datetime] = None,
                log_levels: Optional[Set[LogLevel]] = None,
                context_types: Optional[Set[ContextType]] = None,
                guard_types: Optional[Set[str]] = None,
                limit: int = 100) -> List[LogEntry]:
        """Retrieve logs with filtering."""
        logs = []
        
        try:
            # Determine date range
            if start_time is None:
                start_time = datetime.now() - timedelta(days=7)  # Default to last week
            if end_time is None:
                end_time = datetime.now()
            
            # Get all log files in date range
            current_date = start_time.date()
            end_date = end_time.date()
            
            while current_date <= end_date:
                log_file = self.storage_path / f"guard_responses_{current_date.strftime('%Y-%m-%d')}.jsonl"
                
                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                log_data = json.loads(line.strip())
                                log_entry = self._parse_log_entry(log_data)
                                
                                # Apply filters
                                if self._matches_filters(log_entry, log_levels, context_types, guard_types, start_time, end_time):
                                    logs.append(log_entry)
                                    
                                    if len(logs) >= limit:
                                        break
                                        
                            except json.JSONDecodeError:
                                continue
                
                if len(logs) >= limit:
                    break
                    
                current_date += timedelta(days=1)
            
            # Sort by timestamp (newest first)
            logs.sort(key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to retrieve logs: {e}")
        
        return logs[:limit]
    
    def _parse_log_entry(self, log_data: Dict[str, Any]) -> LogEntry:
        """Parse log entry from dictionary."""
        # Parse guard response
        response_data = log_data['guard_response']
        agent_context_data = response_data['agent_context']
        
        agent_context = AgentContext(
            agent_id=agent_context_data['agent_id'],
            agent_type=agent_context_data['agent_type'],
            session_id=agent_context_data['session_id'],
            user_id=agent_context_data.get('user_id'),
            task_id=agent_context_data.get('task_id'),
            conversation_id=agent_context_data.get('conversation_id'),
            timestamp=datetime.fromisoformat(agent_context_data['timestamp']),
            metadata=agent_context_data.get('metadata', {})
        )
        
        guard_response = GuardResponse(
            response_id=response_data['response_id'],
            guard_id=response_data['guard_id'],
            guard_type=response_data['guard_type'],
            response_type=ResponseType(response_data['response_type']),
            original_input=response_data['original_input'],
            processed_output=response_data['processed_output'],
            confidence=response_data['confidence'],
            processing_time_ms=response_data['processing_time_ms'],
            agent_context=agent_context,
            threats_detected=response_data.get('threats_detected', []),
            actions_taken=response_data.get('actions_taken', []),
            recommendations=response_data.get('recommendations', []),
            metadata=response_data.get('metadata', {}),
            timestamp=datetime.fromisoformat(response_data['timestamp'])
        )
        
        return LogEntry(
            log_id=log_data['log_id'],
            guard_response=guard_response,
            log_level=LogLevel(log_data['log_level']),
            message=log_data['message'],
            context_type=ContextType(log_data['context_type']),
            tags=set(log_data.get('tags', [])),
            timestamp=datetime.fromisoformat(log_data['timestamp'])
        )
    
    def _matches_filters(self, log_entry: LogEntry, log_levels: Optional[Set[LogLevel]],
                        context_types: Optional[Set[ContextType]], guard_types: Optional[Set[str]],
                        start_time: datetime, end_time: datetime) -> bool:
        """Check if log entry matches filters."""
        # Time filter
        if not (start_time <= log_entry.timestamp <= end_time):
            return False
        
        # Log level filter
        if log_levels and log_entry.log_level not in log_levels:
            return False
        
        # Context type filter
        if context_types and log_entry.context_type not in context_types:
            return False
        
        # Guard type filter
        if guard_types and log_entry.guard_response.guard_type not in guard_types:
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            'total_logs': self.total_logs,
            'logs_by_level': {level.value: count for level, count in self.logs_by_level.items()},
            'logs_by_context': {context.value: count for context, count in self.logs_by_context.items()},
            'enabled_log_levels': [level.value for level in self.log_levels],
            'enabled_context_types': [context.value for context in self.context_types],
            'enabled_tags': list(self.enabled_tags)
        }
    
    def configure_logging(self, log_levels: Optional[Set[LogLevel]] = None,
                         context_types: Optional[Set[ContextType]] = None,
                         enabled_tags: Optional[Set[str]] = None) -> None:
        """Configure logging settings."""
        if log_levels is not None:
            self.log_levels = log_levels
        
        if context_types is not None:
            self.context_types = context_types
        
        if enabled_tags is not None:
            self.enabled_tags = enabled_tags
        
        logger.info("Response logging configuration updated")


class GuardResponseManager:
    """Main manager for guard response formatting and logging."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_guard_responses"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Formatters
        self.formatters: Dict[ResponseFormat, ResponseFormatter] = {
            ResponseFormat.JSON: JSONResponseFormatter(),
            ResponseFormat.XML: XMLResponseFormatter(),
            ResponseFormat.TEXT: TextResponseFormatter(),
            ResponseFormat.MARKDOWN: MarkdownResponseFormatter()
        }
        
        # Logger
        self.logger = ResponseLogger(str(self.storage_path / "logs"))
        
        # Statistics
        self.total_responses = 0
        self.responses_by_type: Dict[ResponseType, int] = {rt: 0 for rt in ResponseType}
        self.responses_by_guard: Dict[str, int] = {}
        
        logger.info("Guard Response Manager initialized")
    
    def create_response(self, guard_id: str, guard_type: str, response_type: ResponseType,
                       original_input: str, processed_output: str, confidence: float,
                       processing_time_ms: float, agent_context: AgentContext,
                       threats_detected: Optional[List[Dict[str, Any]]] = None,
                       actions_taken: Optional[List[str]] = None,
                       recommendations: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> GuardResponse:
        """Create a formatted guard response."""
        response = GuardResponse(
            response_id=str(uuid.uuid4()),
            guard_id=guard_id,
            guard_type=guard_type,
            response_type=response_type,
            original_input=original_input,
            processed_output=processed_output,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            agent_context=agent_context,
            threats_detected=threats_detected or [],
            actions_taken=actions_taken or [],
            recommendations=recommendations or [],
            metadata=metadata or {}
        )
        
        # Update statistics
        self.total_responses += 1
        self.responses_by_type[response_type] += 1
        self.responses_by_guard[guard_id] = self.responses_by_guard.get(guard_id, 0) + 1
        
        # Record metrics
        record_metric("guard_response_manager.responses_created", 1, MetricType.COUNTER, {
            "guard_id": guard_id,
            "guard_type": guard_type,
            "response_type": response_type.value
        })
        
        return response
    
    def format_response(self, response: GuardResponse, format_type: ResponseFormat) -> str:
        """Format a guard response in the specified format."""
        if format_type not in self.formatters:
            raise ValueError(f"Unsupported format: {format_type}")
        
        formatter = self.formatters[format_type]
        return formatter.format_response(response)
    
    def log_response(self, response: GuardResponse, log_level: LogLevel,
                    context_type: ContextType, message: str,
                    tags: Optional[Set[str]] = None) -> Optional[LogEntry]:
        """Log a guard response."""
        return self.logger.log_response(response, log_level, context_type, message, tags)
    
    def get_response_content_type(self, format_type: ResponseFormat) -> str:
        """Get content type for a response format."""
        if format_type not in self.formatters:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return self.formatters[format_type].get_content_type()
    
    def get_logs(self, **kwargs) -> List[LogEntry]:
        """Get logs with filtering options."""
        return self.logger.get_logs(**kwargs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get response manager statistics."""
        return {
            'total_responses': self.total_responses,
            'responses_by_type': {rt.value: count for rt, count in self.responses_by_type.items()},
            'responses_by_guard': self.responses_by_guard,
            'available_formats': [fmt.value for fmt in self.formatters.keys()],
            'logging_stats': self.logger.get_statistics()
        }
    
    def configure_logging(self, **kwargs) -> None:
        """Configure logging settings."""
        self.logger.configure_logging(**kwargs)
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> int:
        """Clean up old log files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            for log_file in self.storage_path.glob("logs/guard_responses_*.jsonl"):
                try:
                    # Extract date from filename
                    date_str = log_file.stem.split('_')[-1]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    
                    if file_date < cutoff_date.date():
                        log_file.unlink()
                        deleted_count += 1
                        
                except (ValueError, IndexError):
                    continue
            
            logger.info(f"Cleaned up {deleted_count} old log files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")
            return 0
