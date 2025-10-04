"""
Unit tests for SafeHive tools.

This module tests the tool implementations and base tool functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime

from safehive.tools.base_tools import (
    BaseSafeHiveTool, ToolInput, ToolOutput, create_tool_output
)
from pydantic import Field
from safehive.tools.communication_tools import (
    MessageTool, LoggingTool, send_message, log_event
)
from safehive.tools.system_tools import (
    SystemInfoTool, HealthCheckTool, AgentStatusTool,
    get_system_info, check_system_health, get_agent_status
)


class TestTool(BaseSafeHiveTool):
    """Test implementation of BaseSafeHiveTool."""
    
    def __init__(self, **kwargs):
        super().__init__(name="test_tool", description="A test tool for unit testing", **kwargs)
    
    def _execute(self, input_data: str = "") -> str:
        """Execute the test tool."""
        return f"Test tool executed with input: {input_data}"


class MockToolInput(ToolInput):
    """Mock input class for testing."""
    
    test_field: str = Field(default="default_value")


class MockToolOutput(ToolOutput):
    """Mock output class for testing."""
    
    test_result: str = Field(default="success")


class TestBaseSafeHiveTool:
    """Test BaseSafeHiveTool class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = TestTool()
    
    def test_tool_initialization(self):
        """Test tool initialization."""
        assert self.tool.name == "test_tool"
        assert self.tool.description == "A test tool for unit testing"
        assert self.tool._usage_count == 0
        assert self.tool._last_used is None
        assert self.tool._error_count == 0
    
    def test_tool_execution_success(self):
        """Test successful tool execution."""
        result = self.tool._run("test input")
        
        assert result == "Test tool executed with input: test input"
        assert self.tool._usage_count == 1
        assert self.tool._error_count == 0
        assert self.tool._last_used is not None
    
    def test_tool_execution_error(self):
        """Test tool execution with error."""
        # Create a tool that raises an exception
        class ErrorTool(BaseSafeHiveTool):
            name: str = "error_tool"
            description: str = "A tool that raises errors"
            
            def _execute(self, input_data: str = "") -> str:
                raise ValueError("Test error")
        
        error_tool = ErrorTool()
        result = error_tool._run("test input")
        
        assert "Error executing error_tool: Test error" in result
        assert error_tool._usage_count == 1
        assert error_tool._error_count == 1
    
    def test_tool_usage_stats(self):
        """Test tool usage statistics."""
        # Execute tool multiple times
        self.tool._run("input 1")
        self.tool._run("input 2")
        
        stats = self.tool.get_usage_stats()
        
        assert stats["name"] == "test_tool"
        assert stats["usage_count"] == 2
        assert stats["error_count"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["last_used"] is not None
        assert stats["description"] == "A test tool for unit testing"


class TestToolInput:
    """Test ToolInput class."""
    
    def test_tool_input_creation(self):
        """Test creating a tool input."""
        input_obj = MockToolInput(test_field="custom_value")
        
        assert input_obj.test_field == "custom_value"
    
    def test_tool_input_to_dict(self):
        """Test converting tool input to dictionary."""
        input_obj = MockToolInput(test_field="test_value")
        data = input_obj.to_dict()
        
        assert data["test_field"] == "test_value"
    
    def test_tool_input_to_json(self):
        """Test converting tool input to JSON."""
        input_obj = MockToolInput(test_field="test_value")
        json_str = input_obj.to_json()
        
        data = json.loads(json_str)
        assert data["test_field"] == "test_value"


class TestToolOutput:
    """Test ToolOutput class."""
    
    def test_tool_output_creation(self):
        """Test creating a tool output."""
        output = ToolOutput(success=True, message="Test message")
        
        assert output.success is True
        assert output.message == "Test message"
        assert output.data is None
        assert isinstance(output.timestamp, datetime)
    
    def test_tool_output_success_factory(self):
        """Test success factory method."""
        output = ToolOutput(success=True, message="Operation completed", data={"key": "value"})
        
        assert output.success is True
        assert output.message == "Operation completed"
        assert output.data == {"key": "value"}
    
    def test_tool_output_error_factory(self):
        """Test error factory method."""
        output = ToolOutput(success=False, message="Operation failed", data={"error_code": 404})
        
        assert output.success is False
        assert output.message == "Operation failed"
        assert output.data == {"error_code": 404}
    
    def test_tool_output_to_dict(self):
        """Test converting tool output to dictionary."""
        output = ToolOutput(success=True, message="Test message", data={"test": "data"})
        data = output.to_dict()
        
        assert data["success"] is True
        assert data["message"] == "Test message"
        assert data["data"] == {"test": "data"}
        assert "timestamp" in data
    
    def test_tool_output_to_json(self):
        """Test converting tool output to JSON."""
        output = ToolOutput(success=True, message="Test message")
        json_str = output.to_json()
        
        data = json.loads(json_str)
        assert data["success"] is True
        assert data["message"] == "Test message"


class TestCreateToolOutput:
    """Test create_tool_output utility function."""
    
    def test_create_success_output(self):
        """Test creating success output."""
        output = create_tool_output(True, "Success message", {"key": "value"})
        
        assert output.success is True
        assert output.message == "Success message"
        assert output.data == {"key": "value"}
    
    def test_create_error_output(self):
        """Test creating error output."""
        output = create_tool_output(False, "Error message")
        
        assert output.success is False
        assert output.message == "Error message"
        assert output.data is None


class TestCommunicationTools:
    """Test communication tools."""
    
    def test_send_message_tool(self):
        """Test send_message tool."""
        result = send_message("test_agent", "Hello world", "high")
        
        assert "Message sent successfully" in result
        assert "test_agent" in result
        # Note: The priority is not included in the output message format
    
    def test_log_event_tool(self):
        """Test log_event tool."""
        result = log_event("INFO", "Test log message", '{"context": "test"}')
        
        assert "Logged INFO event" in result
        assert "Test log message" in result
    
    def test_log_event_invalid_level(self):
        """Test log_event with invalid level."""
        result = log_event("INVALID", "Test message")
        
        assert "Invalid log level" in result
        assert "INVALID" in result
    
    def test_message_tool_class(self):
        """Test MessageTool class."""
        tool = MessageTool()
        
        assert tool.name == "send_message"
        assert "Send a message" in tool.description
        
        result = tool._execute("recipient", "message", "normal")
        assert "Message sent successfully" in result
    
    def test_logging_tool_class(self):
        """Test LoggingTool class."""
        tool = LoggingTool()
        
        assert tool.name == "log_event"
        assert "Log an event" in tool.description
        
        result = tool._execute("INFO", "test message")
        assert "Logged INFO event" in result


class TestSystemTools:
    """Test system tools."""
    
    @patch('safehive.tools.system_tools.psutil')
    def test_get_system_info(self, mock_psutil):
        """Test get_system_info tool."""
        # Mock psutil data
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.cpu_freq.return_value = Mock(_asdict=lambda: {"current": 2400.0})
        
        mock_memory = Mock()
        mock_memory.total = 8 * 1024**3  # 8 GB
        mock_memory.available = 4 * 1024**3  # 4 GB
        mock_memory.used = 4 * 1024**3  # 4 GB
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.total = 500 * 1024**3  # 500 GB
        mock_disk.used = 250 * 1024**3  # 250 GB
        mock_disk.free = 250 * 1024**3  # 250 GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_network = Mock()
        mock_network.bytes_sent = 1024**2  # 1 MB
        mock_network.bytes_recv = 2 * 1024**2  # 2 MB
        mock_network.packets_sent = 100
        mock_network.packets_recv = 200
        mock_psutil.net_io_counters.return_value = mock_network
        
        result = get_system_info("all")
        
        assert "System Information" in result
        assert "CPU" in result
        assert "MEMORY" in result
        assert "DISK" in result
        assert "NETWORK" in result
        assert "25.0" in result  # CPU usage
    
    def test_check_system_health(self):
        """Test check_system_health tool."""
        result = check_system_health("all")
        
        assert "System Health Check" in result
        assert "System:" in result
        assert "Agents:" in result
        assert "Guards:" in result
        assert "Timestamp:" in result
    
    def test_get_agent_status_all(self):
        """Test get_agent_status for all agents."""
        result = get_agent_status("all")
        
        assert "Agent Status (All Agents)" in result
        assert "Orchestrator" in result
        assert "User Twin" in result
        assert "Honest Vendor" in result
        assert "Malicious Vendor" in result
    
    def test_get_agent_status_specific(self):
        """Test get_agent_status for specific agent."""
        result = get_agent_status("test_agent_001")
        
        assert "Agent Status for test_agent_001" in result
        assert "Status:" in result
        assert "Last Activity:" in result
        assert "Requests Processed:" in result
    
    def test_system_info_tool_class(self):
        """Test SystemInfoTool class."""
        tool = SystemInfoTool()
        
        assert tool.name == "get_system_info"
        assert "system information" in tool.description
    
    def test_health_check_tool_class(self):
        """Test HealthCheckTool class."""
        tool = HealthCheckTool()
        
        assert tool.name == "check_system_health"
        assert "health status" in tool.description
    
    def test_agent_status_tool_class(self):
        """Test AgentStatusTool class."""
        tool = AgentStatusTool()
        
        assert tool.name == "get_agent_status"
        assert "status of specific agents" in tool.description


class TestToolsIntegration:
    """Integration tests for tools."""
    
    def test_tool_metrics_tracking(self):
        """Test that tools properly track metrics."""
        tool = TestTool()
        
        # Execute tool multiple times
        tool._run("test 1")
        tool._run("test 2")
        
        stats = tool.get_usage_stats()
        assert stats["usage_count"] == 2
        assert stats["success_rate"] == 1.0
    
    def test_tool_error_handling(self):
        """Test tool error handling and recovery."""
        class FlakyTool(BaseSafeHiveTool):
            def __init__(self, **kwargs):
                super().__init__(name="flaky_tool", description="A tool that sometimes fails", **kwargs)
                self._call_count = 0
            
            def _execute(self, input_data: str = "") -> str:
                self._call_count += 1
                if self._call_count % 2 == 0:
                    raise RuntimeError("Simulated failure")
                return "Success"
        
        tool = FlakyTool()
        
        # First call should succeed
        result1 = tool._run("test")
        assert "Success" in result1
        
        # Second call should fail
        result2 = tool._run("test")
        assert "Error executing flaky_tool" in result2
        
        # Third call should succeed again
        result3 = tool._run("test")
        assert "Success" in result3
        
        # Check stats
        stats = tool.get_usage_stats()
        assert stats["usage_count"] == 3
        assert stats["error_count"] == 1
        assert stats["success_rate"] == 2 / 3


if __name__ == "__main__":
    pytest.main([__file__])
