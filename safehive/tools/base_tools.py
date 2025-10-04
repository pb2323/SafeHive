"""
Base Tools for SafeHive Agents

This module provides base tool classes and utilities for creating
LangChain tools that integrate with the SafeHive system.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from datetime import datetime

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..utils.logger import get_logger
from ..utils.metrics import record_metric, MetricType

logger = get_logger(__name__)


class BaseSafeHiveTool(BaseTool, ABC):
    """
    Base class for all SafeHive tools.
    
    Provides common functionality including:
    - Structured input/output handling
    - Metrics tracking
    - Error handling
    - Logging integration
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._usage_count = 0
        self._last_used = None
        self._error_count = 0
    
    def _run(self, *args, **kwargs) -> str:
        """Execute the tool with error handling and metrics."""
        start_time = datetime.now()
        self._usage_count += 1
        
        try:
            logger.debug(f"Executing tool: {self.name}")
            
            # Execute the tool
            result = self._execute(*args, **kwargs)
            
            # Update metrics
            self._last_used = datetime.now()
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            record_metric(
                f"tool.{self.name}.execution",
                1,
                MetricType.COUNTER,
                {"success": True, "execution_time": execution_time}
            )
            
            logger.debug(f"Tool {self.name} executed successfully")
            return result
            
        except Exception as e:
            self._error_count += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Tool {self.name} failed: {e}")
            
            # Record error metrics
            record_metric(
                f"tool.{self.name}.execution",
                1,
                MetricType.COUNTER,
                {"success": False, "execution_time": execution_time}
            )
            
            # Return error information
            return f"Error executing {self.name}: {str(e)}"
    
    @abstractmethod
    def _execute(self, *args, **kwargs) -> str:
        """Execute the tool logic. Must be implemented by subclasses."""
        pass
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "name": self.name,
            "usage_count": self._usage_count,
            "error_count": self._error_count,
            "success_rate": (self._usage_count - self._error_count) / max(1, self._usage_count),
            "last_used": self._last_used.isoformat() if self._last_used else None,
            "description": self.description
        }


class ToolInput(BaseModel):
    """Base class for tool input validation."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ToolOutput(BaseModel):
    """Base class for tool output formatting."""
    
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Result message or error description")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Additional data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the operation")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = self.model_dump()
        result["timestamp"] = self.timestamp.isoformat()
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def success(cls, message: str, data: Optional[Dict[str, Any]] = None) -> "ToolOutput":
        """Create a successful output."""
        return cls(success=True, message=message, data=data)
    
    @classmethod
    def error(cls, message: str, data: Optional[Dict[str, Any]] = None) -> "ToolOutput":
        """Create an error output."""
        return cls(success=False, message=message, data=data)


def create_tool_output(success: bool, message: str, data: Optional[Dict[str, Any]] = None) -> ToolOutput:
    """Create a tool output."""
    return ToolOutput(success=success, message=message, data=data)
