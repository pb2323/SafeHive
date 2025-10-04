"""
Unit tests for Guard Response Formatter - Response Formatting and Logging System.
"""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Set
from unittest.mock import patch, MagicMock
import pytest

from safehive.guards.response_formatter import (
    GuardResponseManager, ResponseLogger, ResponseFormatter, JSONResponseFormatter,
    XMLResponseFormatter, TextResponseFormatter, MarkdownResponseFormatter,
    ResponseFormat, LogLevel, ResponseType, ContextType, AgentContext,
    GuardResponse, LogEntry
)


class TestResponseFormat:
    """Test ResponseFormat enum."""
    
    def test_response_format_values(self):
        """Test ResponseFormat enum values."""
        assert ResponseFormat.JSON.value == "json"
        assert ResponseFormat.XML.value == "xml"
        assert ResponseFormat.YAML.value == "yaml"
        assert ResponseFormat.TEXT.value == "text"
        assert ResponseFormat.HTML.value == "html"
        assert ResponseFormat.MARKDOWN.value == "markdown"


class TestLogLevel:
    """Test LogLevel enum."""
    
    def test_log_level_values(self):
        """Test LogLevel enum values."""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"


class TestResponseType:
    """Test ResponseType enum."""
    
    def test_response_type_values(self):
        """Test ResponseType enum values."""
        assert ResponseType.ALLOW.value == "allow"
        assert ResponseType.BLOCK.value == "block"
        assert ResponseType.FILTER.value == "filter"
        assert ResponseType.SANITIZE.value == "sanitize"
        assert ResponseType.QUARANTINE.value == "quarantine"
        assert ResponseType.ESCALATE.value == "escalate"
        assert ResponseType.REDIRECT.value == "redirect"
        assert ResponseType.WARN.value == "warn"
        assert ResponseType.AUDIT.value == "audit"


class TestContextType:
    """Test ContextType enum."""
    
    def test_context_type_values(self):
        """Test ContextType enum values."""
        assert ContextType.USER_INPUT.value == "user_input"
        assert ContextType.AGENT_RESPONSE.value == "agent_response"
        assert ContextType.VENDOR_COMMUNICATION.value == "vendor_communication"
        assert ContextType.SYSTEM_EVENT.value == "system_event"
        assert ContextType.SECURITY_SCAN.value == "security_scan"
        assert ContextType.TASK_EXECUTION.value == "task_execution"


class TestAgentContext:
    """Test AgentContext functionality."""
    
    def test_agent_context_creation(self):
        """Test AgentContext creation."""
        context = AgentContext(
            agent_id="test_agent",
            agent_type="user_twin",
            session_id="session_123",
            user_id="user_456",
            task_id="task_789",
            conversation_id="conv_101",
            metadata={"version": "1.0"}
        )
        
        assert context.agent_id == "test_agent"
        assert context.agent_type == "user_twin"
        assert context.session_id == "session_123"
        assert context.user_id == "user_456"
        assert context.task_id == "task_789"
        assert context.conversation_id == "conv_101"
        assert context.metadata == {"version": "1.0"}
        assert isinstance(context.timestamp, datetime)
    
    def test_agent_context_serialization(self):
        """Test AgentContext serialization."""
        context = AgentContext(
            agent_id="test_agent",
            agent_type="user_twin",
            session_id="session_123",
            user_id="user_456"
        )
        
        data = context.to_dict()
        
        assert data["agent_id"] == "test_agent"
        assert data["agent_type"] == "user_twin"
        assert data["session_id"] == "session_123"
        assert data["user_id"] == "user_456"
        assert data["task_id"] is None
        assert data["conversation_id"] is None
        assert "timestamp" in data


class TestGuardResponse:
    """Test GuardResponse functionality."""
    
    def test_guard_response_creation(self):
        """Test GuardResponse creation."""
        agent_context = AgentContext(
            agent_id="test_agent",
            agent_type="user_twin",
            session_id="session_123"
        )
        
        response = GuardResponse(
            response_id="response_001",
            guard_id="privacy_sentry",
            guard_type="privacy_sentry",
            response_type=ResponseType.FILTER,
            original_input="test input",
            processed_output="filtered output",
            confidence=0.95,
            processing_time_ms=5.2,
            agent_context=agent_context,
            threats_detected=[{"type": "pii", "severity": "high"}],
            actions_taken=["mask_pii"],
            recommendations=["review_input"]
        )
        
        assert response.response_id == "response_001"
        assert response.guard_id == "privacy_sentry"
        assert response.guard_type == "privacy_sentry"
        assert response.response_type == ResponseType.FILTER
        assert response.original_input == "test input"
        assert response.processed_output == "filtered output"
        assert response.confidence == 0.95
        assert response.processing_time_ms == 5.2
        assert response.agent_context == agent_context
        assert len(response.threats_detected) == 1
        assert len(response.actions_taken) == 1
        assert len(response.recommendations) == 1
        assert isinstance(response.timestamp, datetime)
    
    def test_guard_response_serialization(self):
        """Test GuardResponse serialization."""
        agent_context = AgentContext(
            agent_id="test_agent",
            agent_type="user_twin",
            session_id="session_123"
        )
        
        response = GuardResponse(
            response_id="response_001",
            guard_id="privacy_sentry",
            guard_type="privacy_sentry",
            response_type=ResponseType.FILTER,
            original_input="test input",
            processed_output="filtered output",
            confidence=0.95,
            processing_time_ms=5.2,
            agent_context=agent_context
        )
        
        data = response.to_dict()
        
        assert data["response_id"] == "response_001"
        assert data["guard_id"] == "privacy_sentry"
        assert data["guard_type"] == "privacy_sentry"
        assert data["response_type"] == "filter"
        assert data["original_input"] == "test input"
        assert data["processed_output"] == "filtered output"
        assert data["confidence"] == 0.95
        assert data["processing_time_ms"] == 5.2
        assert "agent_context" in data
        assert "timestamp" in data


class TestLogEntry:
    """Test LogEntry functionality."""
    
    def test_log_entry_creation(self):
        """Test LogEntry creation."""
        agent_context = AgentContext(
            agent_id="test_agent",
            agent_type="user_twin",
            session_id="session_123"
        )
        
        guard_response = GuardResponse(
            response_id="response_001",
            guard_id="privacy_sentry",
            guard_type="privacy_sentry",
            response_type=ResponseType.FILTER,
            original_input="test input",
            processed_output="filtered output",
            confidence=0.95,
            processing_time_ms=5.2,
            agent_context=agent_context
        )
        
        log_entry = LogEntry(
            log_id="log_001",
            guard_response=guard_response,
            log_level=LogLevel.INFO,
            message="Guard response processed",
            context_type=ContextType.USER_INPUT,
            tags={"security", "pii"}
        )
        
        assert log_entry.log_id == "log_001"
        assert log_entry.guard_response == guard_response
        assert log_entry.log_level == LogLevel.INFO
        assert log_entry.message == "Guard response processed"
        assert log_entry.context_type == ContextType.USER_INPUT
        assert log_entry.tags == {"security", "pii"}
        assert isinstance(log_entry.timestamp, datetime)
    
    def test_log_entry_serialization(self):
        """Test LogEntry serialization."""
        agent_context = AgentContext(
            agent_id="test_agent",
            agent_type="user_twin",
            session_id="session_123"
        )
        
        guard_response = GuardResponse(
            response_id="response_001",
            guard_id="privacy_sentry",
            guard_type="privacy_sentry",
            response_type=ResponseType.FILTER,
            original_input="test input",
            processed_output="filtered output",
            confidence=0.95,
            processing_time_ms=5.2,
            agent_context=agent_context
        )
        
        log_entry = LogEntry(
            log_id="log_001",
            guard_response=guard_response,
            log_level=LogLevel.INFO,
            message="Guard response processed",
            context_type=ContextType.USER_INPUT,
            tags={"security"}
        )
        
        data = log_entry.to_dict()
        
        assert data["log_id"] == "log_001"
        assert data["log_level"] == "info"
        assert data["message"] == "Guard response processed"
        assert data["context_type"] == "user_input"
        assert data["tags"] == ["security"]
        assert "guard_response" in data
        assert "timestamp" in data


class TestJSONResponseFormatter:
    """Test JSONResponseFormatter functionality."""
    
    def test_json_formatter_creation(self):
        """Test JSONResponseFormatter creation."""
        formatter = JSONResponseFormatter()
        assert isinstance(formatter, ResponseFormatter)
    
    def test_json_format_response(self):
        """Test JSON response formatting."""
        formatter = JSONResponseFormatter()
        
        agent_context = AgentContext(
            agent_id="test_agent",
            agent_type="user_twin",
            session_id="session_123"
        )
        
        response = GuardResponse(
            response_id="response_001",
            guard_id="privacy_sentry",
            guard_type="privacy_sentry",
            response_type=ResponseType.FILTER,
            original_input="test input",
            processed_output="filtered output",
            confidence=0.95,
            processing_time_ms=5.2,
            agent_context=agent_context
        )
        
        formatted = formatter.format_response(response)
        
        assert isinstance(formatted, str)
        # Should be valid JSON
        import json
        parsed = json.loads(formatted)
        assert parsed["response_id"] == "response_001"
        assert parsed["guard_id"] == "privacy_sentry"
        assert parsed["response_type"] == "filter"
    
    def test_json_content_type(self):
        """Test JSON content type."""
        formatter = JSONResponseFormatter()
        assert formatter.get_content_type() == "application/json"


class TestXMLResponseFormatter:
    """Test XMLResponseFormatter functionality."""
    
    def test_xml_formatter_creation(self):
        """Test XMLResponseFormatter creation."""
        formatter = XMLResponseFormatter()
        assert isinstance(formatter, ResponseFormatter)
    
    def test_xml_format_response(self):
        """Test XML response formatting."""
        formatter = XMLResponseFormatter()
        
        agent_context = AgentContext(
            agent_id="test_agent",
            agent_type="user_twin",
            session_id="session_123"
        )
        
        response = GuardResponse(
            response_id="response_001",
            guard_id="privacy_sentry",
            guard_type="privacy_sentry",
            response_type=ResponseType.FILTER,
            original_input="test input",
            processed_output="filtered output",
            confidence=0.95,
            processing_time_ms=5.2,
            agent_context=agent_context
        )
        
        formatted = formatter.format_response(response)
        
        assert isinstance(formatted, str)
        assert "<?xml version=" in formatted
        assert "<GuardResponse>" in formatted
        assert "<ResponseId>response_001</ResponseId>" in formatted
        assert "<GuardId>privacy_sentry</GuardId>" in formatted
    
    def test_xml_content_type(self):
        """Test XML content type."""
        formatter = XMLResponseFormatter()
        assert formatter.get_content_type() == "application/xml"


class TestTextResponseFormatter:
    """Test TextResponseFormatter functionality."""
    
    def test_text_formatter_creation(self):
        """Test TextResponseFormatter creation."""
        formatter = TextResponseFormatter()
        assert isinstance(formatter, ResponseFormatter)
    
    def test_text_format_response(self):
        """Test text response formatting."""
        formatter = TextResponseFormatter()
        
        agent_context = AgentContext(
            agent_id="test_agent",
            agent_type="user_twin",
            session_id="session_123"
        )
        
        response = GuardResponse(
            response_id="response_001",
            guard_id="privacy_sentry",
            guard_type="privacy_sentry",
            response_type=ResponseType.FILTER,
            original_input="test input",
            processed_output="filtered output",
            confidence=0.95,
            processing_time_ms=5.2,
            agent_context=agent_context
        )
        
        formatted = formatter.format_response(response)
        
        assert isinstance(formatted, str)
        assert "GUARD RESPONSE" in formatted
        assert "Response ID: response_001" in formatted
        assert "Guard: privacy_sentry" in formatted
        assert "Type: FILTER" in formatted
        assert "AGENT CONTEXT:" in formatted
        assert "INPUT/OUTPUT:" in formatted
    
    def test_text_content_type(self):
        """Test text content type."""
        formatter = TextResponseFormatter()
        assert formatter.get_content_type() == "text/plain"


class TestMarkdownResponseFormatter:
    """Test MarkdownResponseFormatter functionality."""
    
    def test_markdown_formatter_creation(self):
        """Test MarkdownResponseFormatter creation."""
        formatter = MarkdownResponseFormatter()
        assert isinstance(formatter, ResponseFormatter)
    
    def test_markdown_format_response(self):
        """Test Markdown response formatting."""
        formatter = MarkdownResponseFormatter()
        
        agent_context = AgentContext(
            agent_id="test_agent",
            agent_type="user_twin",
            session_id="session_123"
        )
        
        response = GuardResponse(
            response_id="response_001",
            guard_id="privacy_sentry",
            guard_type="privacy_sentry",
            response_type=ResponseType.FILTER,
            original_input="test input",
            processed_output="filtered output",
            confidence=0.95,
            processing_time_ms=5.2,
            agent_context=agent_context
        )
        
        formatted = formatter.format_response(response)
        
        assert isinstance(formatted, str)
        assert "# Guard Response" in formatted
        assert "**Response ID:** `response_001`" in formatted
        assert "**Guard:** privacy_sentry" in formatted
        assert "**Type:** `FILTER`" in formatted
        assert "## Agent Context" in formatted
        assert "## Input/Output" in formatted
    
    def test_markdown_content_type(self):
        """Test Markdown content type."""
        formatter = MarkdownResponseFormatter()
        assert formatter.get_content_type() == "text/markdown"


class TestResponseLogger:
    """Test ResponseLogger functionality."""
    
    def test_response_logger_creation(self):
        """Test ResponseLogger creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ResponseLogger(temp_dir)
            
            assert logger.storage_path == Path(temp_dir)
            assert len(logger.log_levels) > 0
            assert len(logger.context_types) > 0
            assert logger.total_logs == 0
    
    def test_log_response(self):
        """Test logging a response."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ResponseLogger(temp_dir)
            
            agent_context = AgentContext(
                agent_id="test_agent",
                agent_type="user_twin",
                session_id="session_123"
            )
            
            response = GuardResponse(
                response_id="response_001",
                guard_id="privacy_sentry",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="test input",
                processed_output="filtered output",
                confidence=0.95,
                processing_time_ms=5.2,
                agent_context=agent_context
            )
            
            log_entry = logger.log_response(
                response=response,
                log_level=LogLevel.INFO,
                context_type=ContextType.USER_INPUT,
                message="Test log message",
                tags={"test", "security"}
            )
            
            assert log_entry is not None
            assert log_entry.guard_response == response
            assert log_entry.log_level == LogLevel.INFO
            assert log_entry.message == "Test log message"
            assert log_entry.context_type == ContextType.USER_INPUT
            assert log_entry.tags == {"test", "security"}
            assert logger.total_logs == 1
    
    def test_log_response_filtered(self):
        """Test logging a response that gets filtered."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ResponseLogger(temp_dir)
            
            # Configure to only log ERROR level
            logger.configure_logging(log_levels={LogLevel.ERROR})
            
            agent_context = AgentContext(
                agent_id="test_agent",
                agent_type="user_twin",
                session_id="session_123"
            )
            
            response = GuardResponse(
                response_id="response_001",
                guard_id="privacy_sentry",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="test input",
                processed_output="filtered output",
                confidence=0.95,
                processing_time_ms=5.2,
                agent_context=agent_context
            )
            
            # Try to log at INFO level (should be filtered)
            log_entry = logger.log_response(
                response=response,
                log_level=LogLevel.INFO,
                context_type=ContextType.USER_INPUT,
                message="Test log message"
            )
            
            assert log_entry is None
            assert logger.total_logs == 0
    
    def test_get_logs(self):
        """Test retrieving logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ResponseLogger(temp_dir)
            
            agent_context = AgentContext(
                agent_id="test_agent",
                agent_type="user_twin",
                session_id="session_123"
            )
            
            response = GuardResponse(
                response_id="response_001",
                guard_id="privacy_sentry",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="test input",
                processed_output="filtered output",
                confidence=0.95,
                processing_time_ms=5.2,
                agent_context=agent_context
            )
            
            # Log a response
            logger.log_response(
                response=response,
                log_level=LogLevel.INFO,
                context_type=ContextType.USER_INPUT,
                message="Test log message"
            )
            
            # Retrieve logs
            logs = logger.get_logs(limit=10)
            
            assert len(logs) >= 1
            assert logs[0].guard_response.response_id == "response_001"
            assert logs[0].message == "Test log message"
    
    def test_get_statistics(self):
        """Test getting logging statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ResponseLogger(temp_dir)
            
            agent_context = AgentContext(
                agent_id="test_agent",
                agent_type="user_twin",
                session_id="session_123"
            )
            
            response = GuardResponse(
                response_id="response_001",
                guard_id="privacy_sentry",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="test input",
                processed_output="filtered output",
                confidence=0.95,
                processing_time_ms=5.2,
                agent_context=agent_context
            )
            
            # Log a response
            logger.log_response(
                response=response,
                log_level=LogLevel.INFO,
                context_type=ContextType.USER_INPUT,
                message="Test log message"
            )
            
            stats = logger.get_statistics()
            
            assert "total_logs" in stats
            assert "logs_by_level" in stats
            assert "logs_by_context" in stats
            assert "enabled_log_levels" in stats
            assert "enabled_context_types" in stats
            assert "enabled_tags" in stats
            assert stats["total_logs"] == 1
    
    def test_configure_logging(self):
        """Test configuring logging settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ResponseLogger(temp_dir)
            
            # Configure logging
            logger.configure_logging(
                log_levels={LogLevel.ERROR, LogLevel.CRITICAL},
                context_types={ContextType.SECURITY_SCAN},
                enabled_tags={"security", "critical"}
            )
            
            assert logger.log_levels == {LogLevel.ERROR, LogLevel.CRITICAL}
            assert logger.context_types == {ContextType.SECURITY_SCAN}
            assert logger.enabled_tags == {"security", "critical"}


class TestGuardResponseManager:
    """Test GuardResponseManager functionality."""
    
    def test_guard_response_manager_creation(self):
        """Test GuardResponseManager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            assert manager.storage_path == Path(temp_dir)
            assert len(manager.formatters) == 4  # JSON, XML, TEXT, MARKDOWN
            assert isinstance(manager.logger, ResponseLogger)
            assert manager.total_responses == 0
    
    def test_create_response(self):
        """Test creating a guard response."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            agent_context = AgentContext(
                agent_id="test_agent",
                agent_type="user_twin",
                session_id="session_123"
            )
            
            response = manager.create_response(
                guard_id="privacy_sentry",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="test input",
                processed_output="filtered output",
                confidence=0.95,
                processing_time_ms=5.2,
                agent_context=agent_context,
                threats_detected=[{"type": "pii", "severity": "high"}],
                actions_taken=["mask_pii"],
                recommendations=["review_input"]
            )
            
            assert response.guard_id == "privacy_sentry"
            assert response.guard_type == "privacy_sentry"
            assert response.response_type == ResponseType.FILTER
            assert response.original_input == "test input"
            assert response.processed_output == "filtered output"
            assert response.confidence == 0.95
            assert response.processing_time_ms == 5.2
            assert response.agent_context == agent_context
            assert len(response.threats_detected) == 1
            assert len(response.actions_taken) == 1
            assert len(response.recommendations) == 1
            assert manager.total_responses == 1
    
    def test_format_response_json(self):
        """Test formatting response as JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            agent_context = AgentContext(
                agent_id="test_agent",
                agent_type="user_twin",
                session_id="session_123"
            )
            
            response = manager.create_response(
                guard_id="privacy_sentry",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="test input",
                processed_output="filtered output",
                confidence=0.95,
                processing_time_ms=5.2,
                agent_context=agent_context
            )
            
            formatted = manager.format_response(response, ResponseFormat.JSON)
            
            assert isinstance(formatted, str)
            import json
            parsed = json.loads(formatted)
            assert parsed["guard_id"] == "privacy_sentry"
            assert parsed["response_type"] == "filter"
    
    def test_format_response_xml(self):
        """Test formatting response as XML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            agent_context = AgentContext(
                agent_id="test_agent",
                agent_type="user_twin",
                session_id="session_123"
            )
            
            response = manager.create_response(
                guard_id="privacy_sentry",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="test input",
                processed_output="filtered output",
                confidence=0.95,
                processing_time_ms=5.2,
                agent_context=agent_context
            )
            
            formatted = manager.format_response(response, ResponseFormat.XML)
            
            assert isinstance(formatted, str)
            assert "<?xml version=" in formatted
            assert "<GuardResponse>" in formatted
            assert "<GuardId>privacy_sentry</GuardId>" in formatted
    
    def test_format_response_text(self):
        """Test formatting response as text."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            agent_context = AgentContext(
                agent_id="test_agent",
                agent_type="user_twin",
                session_id="session_123"
            )
            
            response = manager.create_response(
                guard_id="privacy_sentry",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="test input",
                processed_output="filtered output",
                confidence=0.95,
                processing_time_ms=5.2,
                agent_context=agent_context
            )
            
            formatted = manager.format_response(response, ResponseFormat.TEXT)
            
            assert isinstance(formatted, str)
            assert "GUARD RESPONSE" in formatted
            assert "Guard: privacy_sentry" in formatted
            assert "Type: FILTER" in formatted
    
    def test_format_response_markdown(self):
        """Test formatting response as Markdown."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            agent_context = AgentContext(
                agent_id="test_agent",
                agent_type="user_twin",
                session_id="session_123"
            )
            
            response = manager.create_response(
                guard_id="privacy_sentry",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="test input",
                processed_output="filtered output",
                confidence=0.95,
                processing_time_ms=5.2,
                agent_context=agent_context
            )
            
            formatted = manager.format_response(response, ResponseFormat.MARKDOWN)
            
            assert isinstance(formatted, str)
            assert "# Guard Response" in formatted
            assert "**Guard:** privacy_sentry" in formatted
            assert "**Type:** `FILTER`" in formatted
    
    def test_log_response(self):
        """Test logging a response."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            agent_context = AgentContext(
                agent_id="test_agent",
                agent_type="user_twin",
                session_id="session_123"
            )
            
            response = manager.create_response(
                guard_id="privacy_sentry",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="test input",
                processed_output="filtered output",
                confidence=0.95,
                processing_time_ms=5.2,
                agent_context=agent_context
            )
            
            log_entry = manager.log_response(
                response=response,
                log_level=LogLevel.INFO,
                context_type=ContextType.USER_INPUT,
                message="Test log message",
                tags={"test", "security"}
            )
            
            assert log_entry is not None
            assert log_entry.guard_response == response
            assert log_entry.log_level == LogLevel.INFO
            assert log_entry.message == "Test log message"
    
    def test_get_response_content_type(self):
        """Test getting response content type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            assert manager.get_response_content_type(ResponseFormat.JSON) == "application/json"
            assert manager.get_response_content_type(ResponseFormat.XML) == "application/xml"
            assert manager.get_response_content_type(ResponseFormat.TEXT) == "text/plain"
            assert manager.get_response_content_type(ResponseFormat.MARKDOWN) == "text/markdown"
    
    def test_get_logs(self):
        """Test getting logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            agent_context = AgentContext(
                agent_id="test_agent",
                agent_type="user_twin",
                session_id="session_123"
            )
            
            response = manager.create_response(
                guard_id="privacy_sentry",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="test input",
                processed_output="filtered output",
                confidence=0.95,
                processing_time_ms=5.2,
                agent_context=agent_context
            )
            
            # Log a response
            manager.log_response(
                response=response,
                log_level=LogLevel.INFO,
                context_type=ContextType.USER_INPUT,
                message="Test log message"
            )
            
            # Get logs
            logs = manager.get_logs(limit=10)
            
            assert len(logs) >= 1
            assert logs[0].guard_response.response_id == response.response_id
    
    def test_get_statistics(self):
        """Test getting statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            agent_context = AgentContext(
                agent_id="test_agent",
                agent_type="user_twin",
                session_id="session_123"
            )
            
            # Create a response
            manager.create_response(
                guard_id="privacy_sentry",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="test input",
                processed_output="filtered output",
                confidence=0.95,
                processing_time_ms=5.2,
                agent_context=agent_context
            )
            
            stats = manager.get_statistics()
            
            assert "total_responses" in stats
            assert "responses_by_type" in stats
            assert "responses_by_guard" in stats
            assert "available_formats" in stats
            assert "logging_stats" in stats
            assert stats["total_responses"] == 1
            assert stats["responses_by_type"]["filter"] == 1
            assert stats["responses_by_guard"]["privacy_sentry"] == 1
    
    def test_configure_logging(self):
        """Test configuring logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            # Configure logging
            manager.configure_logging(
                log_levels={LogLevel.ERROR, LogLevel.CRITICAL},
                context_types={ContextType.SECURITY_SCAN}
            )
            
            assert manager.logger.log_levels == {LogLevel.ERROR, LogLevel.CRITICAL}
            assert manager.logger.context_types == {ContextType.SECURITY_SCAN}
    
    def test_cleanup_old_logs(self):
        """Test cleaning up old logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            # Create some old log files
            old_date = (datetime.now() - timedelta(days=35)).strftime("%Y-%m-%d")
            old_log_file = manager.storage_path / "logs" / f"guard_responses_{old_date}.jsonl"
            old_log_file.parent.mkdir(exist_ok=True)
            old_log_file.write_text("old log content")
            
            # Cleanup old logs
            deleted_count = manager.cleanup_old_logs(days_to_keep=30)
            
            assert deleted_count >= 0  # Should not fail


class TestResponseFormatterIntegration:
    """Integration tests for Response Formatter system."""
    
    def test_complete_response_workflow(self):
        """Test complete response formatting and logging workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            # Create agent context
            agent_context = AgentContext(
                agent_id="user_twin_001",
                agent_type="user_twin",
                session_id="session_123",
                user_id="user_456",
                task_id="task_789",
                conversation_id="conv_101",
                metadata={"version": "1.0", "environment": "test"}
            )
            
            # Create guard response
            response = manager.create_response(
                guard_id="privacy_sentry_001",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="My email is john@example.com and my phone is 555-1234",
                processed_output="My email is jo****@example.com and my phone is ***-****",
                confidence=0.95,
                processing_time_ms=12.5,
                agent_context=agent_context,
                threats_detected=[
                    {"type": "email", "severity": "medium", "confidence": 0.9},
                    {"type": "phone", "severity": "medium", "confidence": 0.8}
                ],
                actions_taken=["mask_email", "mask_phone"],
                recommendations=["review_privacy_settings", "consider_data_retention"]
            )
            
            # Test all formatters
            json_formatted = manager.format_response(response, ResponseFormat.JSON)
            xml_formatted = manager.format_response(response, ResponseFormat.XML)
            text_formatted = manager.format_response(response, ResponseFormat.TEXT)
            markdown_formatted = manager.format_response(response, ResponseFormat.MARKDOWN)
            
            # Verify all formats are valid
            assert isinstance(json_formatted, str)
            assert isinstance(xml_formatted, str)
            assert isinstance(text_formatted, str)
            assert isinstance(markdown_formatted, str)
            
            # Verify JSON format
            import json
            json_data = json.loads(json_formatted)
            assert json_data["guard_id"] == "privacy_sentry_001"
            assert json_data["response_type"] == "filter"
            assert len(json_data["threats_detected"]) == 2
            
            # Verify XML format
            assert "<?xml version=" in xml_formatted
            assert "<GuardId>privacy_sentry_001</GuardId>" in xml_formatted
            assert "<ThreatsDetected>" in xml_formatted
            
            # Verify text format
            assert "GUARD RESPONSE" in text_formatted
            assert "privacy_sentry_001" in text_formatted
            assert "THREATS DETECTED:" in text_formatted
            
            # Verify Markdown format
            assert "# Guard Response" in markdown_formatted
            assert "**Guard:** privacy_sentry_001" in markdown_formatted
            assert "## Threats Detected" in markdown_formatted
            
            # Log the response
            log_entry = manager.log_response(
                response=response,
                log_level=LogLevel.INFO,
                context_type=ContextType.USER_INPUT,
                message="PII detected and filtered in user input",
                tags={"pii", "filtering", "privacy"}
            )
            
            assert log_entry is not None
            assert log_entry.guard_response == response
            
            # Retrieve logs
            logs = manager.get_logs(limit=10)
            assert len(logs) >= 1
            assert logs[0].guard_response.response_id == response.response_id
            
            # Get statistics
            stats = manager.get_statistics()
            assert stats["total_responses"] == 1
            assert stats["responses_by_type"]["filter"] == 1
            assert stats["responses_by_guard"]["privacy_sentry_001"] == 1
    
    def test_response_formatting_performance(self):
        """Test response formatting performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = GuardResponseManager(temp_dir)
            
            agent_context = AgentContext(
                agent_id="test_agent",
                agent_type="user_twin",
                session_id="session_123"
            )
            
            response = manager.create_response(
                guard_id="privacy_sentry",
                guard_type="privacy_sentry",
                response_type=ResponseType.FILTER,
                original_input="test input",
                processed_output="filtered output",
                confidence=0.95,
                processing_time_ms=5.2,
                agent_context=agent_context
            )
            
            # Test formatting performance
            start_time = time.time()
            
            for _ in range(100):
                manager.format_response(response, ResponseFormat.JSON)
                manager.format_response(response, ResponseFormat.TEXT)
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Should format 200 responses quickly (under 100ms)
            assert processing_time < 100
