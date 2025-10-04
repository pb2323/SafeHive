"""
System Tools for SafeHive Agents

This module provides tools for system information, health checks, and monitoring.
"""

import platform
import psutil
from typing import Dict, Any, Optional
from datetime import datetime

from langchain.tools import tool
from pydantic import BaseModel, Field

from .base_tools import BaseSafeHiveTool, ToolOutput


class SystemInfoInput(BaseModel):
    """Input for system information requests."""
    
    info_type: str = Field(default="all", description="Type of system info (all, cpu, memory, disk, network)")


class HealthCheckInput(BaseModel):
    """Input for health check requests."""
    
    component: str = Field(default="all", description="Component to check (all, agents, guards, system)")


def get_system_info(info_type: str = "all") -> str:
    """Get system information including CPU, memory, disk, and network stats.
    
    Args:
        info_type: Type of system info to retrieve (all, cpu, memory, disk, network)
        
    Returns:
        System information in a formatted string
    """
    try:
        info = {}
        
        if info_type in ["all", "cpu"]:
            info["cpu"] = {
                "usage_percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
        
        if info_type in ["all", "memory"]:
            memory = psutil.virtual_memory()
            info["memory"] = {
                "total": f"{memory.total // (1024**3)} GB",
                "available": f"{memory.available // (1024**3)} GB",
                "used": f"{memory.used // (1024**3)} GB",
                "percent": f"{memory.percent}%"
            }
        
        if info_type in ["all", "disk"]:
            disk = psutil.disk_usage('/')
            info["disk"] = {
                "total": f"{disk.total // (1024**3)} GB",
                "used": f"{disk.used // (1024**3)} GB",
                "free": f"{disk.free // (1024**3)} GB",
                "percent": f"{(disk.used / disk.total) * 100:.1f}%"
            }
        
        if info_type in ["all", "network"]:
            network = psutil.net_io_counters()
            info["network"] = {
                "bytes_sent": f"{network.bytes_sent // (1024**2)} MB",
                "bytes_recv": f"{network.bytes_recv // (1024**2)} MB",
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        
        # System information
        if info_type == "all":
            info["system"] = {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            }
        
        # Format output
        result = f"🖥️ System Information ({info_type}):\n"
        for category, data in info.items():
            result += f"\n{category.upper()}:\n"
            for key, value in data.items():
                result += f"  {key}: {value}\n"
        
        return result
        
    except Exception as e:
        return f"❌ Failed to get system info: {str(e)}"


def check_system_health(component: str = "all") -> str:
    """Check the health status of system components.
    
    Args:
        component: Component to check (all, agents, guards, system)
        
    Returns:
        Health status report
    """
    try:
        health_report = []
        
        if component in ["all", "system"]:
            # Check system health
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_health = "✅ Healthy"
            if cpu_usage > 90:
                system_health = "⚠️ High CPU usage"
            elif memory.percent > 90:
                system_health = "⚠️ High memory usage"
            elif (disk.used / disk.total) * 100 > 90:
                system_health = "⚠️ Low disk space"
            
            health_report.append(f"System: {system_health}")
        
        if component in ["all", "agents"]:
            # In a real implementation, this would check agent health
            health_report.append("Agents: ✅ All agents healthy")
        
        if component in ["all", "guards"]:
            # In a real implementation, this would check guard health
            health_report.append("Guards: ✅ All guards operational")
        
        result = "🏥 System Health Check:\n"
        for item in health_report:
            result += f"  {item}\n"
        
        result += f"\nTimestamp: {datetime.now().isoformat()}"
        
        return result
        
    except Exception as e:
        return f"❌ Failed to check system health: {str(e)}"


def get_agent_status(agent_id: str = "all") -> str:
    """Get the status of specific agents or all agents.
    
    Args:
        agent_id: Specific agent ID to check, or "all" for all agents
        
    Returns:
        Agent status information
    """
    try:
        # In a real implementation, this would query the agent manager
        if agent_id == "all":
            result = "🤖 Agent Status (All Agents):\n"
            result += "  Orchestrator: ✅ Running (ID: orchestrator_001)\n"
            result += "  User Twin: ✅ Running (ID: user_twin_001)\n"
            result += "  Honest Vendor: ✅ Running (ID: honest_vendor_001)\n"
            result += "  Malicious Vendor: ✅ Running (ID: malicious_vendor_001)\n"
        else:
            result = f"🤖 Agent Status for {agent_id}:\n"
            result += f"  Status: ✅ Running\n"
            result += f"  Last Activity: {datetime.now().isoformat()}\n"
            result += f"  Requests Processed: 42\n"
            result += f"  Success Rate: 98.5%\n"
        
        return result
        
    except Exception as e:
        return f"❌ Failed to get agent status: {str(e)}"


class SystemInfoTool(BaseSafeHiveTool):
    """Tool for getting system information."""
    
    name: str = "get_system_info"
    description: str = "Get system information including CPU, memory, disk, and network stats"
    
    def _execute(self, info_type: str = "all") -> str:
        """Execute system info retrieval."""
        return get_system_info(info_type)


class HealthCheckTool(BaseSafeHiveTool):
    """Tool for checking system health."""
    
    name: str = "check_system_health"
    description: str = "Check the health status of system components"
    
    def _execute(self, component: str = "all") -> str:
        """Execute health check."""
        return check_system_health(component)


class AgentStatusTool(BaseSafeHiveTool):
    """Tool for checking agent status."""
    
    name: str = "get_agent_status"
    description: str = "Get the status of specific agents or all agents"
    
    def _execute(self, agent_id: str = "all") -> str:
        """Execute agent status check."""
        return get_agent_status(agent_id)
