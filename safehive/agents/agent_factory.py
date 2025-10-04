"""
Agent Factory for SafeHive AI Security Sandbox

This module provides a factory pattern for creating and managing different types of AI agents.
It handles agent instantiation, configuration, and lifecycle management.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from ..models.agent_models import AgentType, AgentState
from ..config.config_loader import AgentConfig, SystemConfig
from ..utils.logger import get_logger
from ..utils.metrics import record_metric, increment_counter, MetricType
from .base_agent import BaseAgent, create_agent_config, AgentCapabilities

logger = get_logger(__name__)


@dataclass
class AgentCreationRequest:
    """Request for creating a new agent."""
    agent_type: AgentType
    name: str
    description: str
    config_overrides: Optional[Dict[str, Any]] = None
    custom_capabilities: Optional[List[AgentCapabilities]] = None
    tools: Optional[List[str]] = None
    personality_config: Optional[Dict[str, Any]] = None


@dataclass
class AgentCreationResult:
    """Result of agent creation operation."""
    success: bool
    agent: Optional[BaseAgent] = None
    agent_id: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    creation_time: datetime = field(default_factory=datetime.now)


class AgentRegistry:
    """
    Registry for managing agent instances and their lifecycle.
    
    This class provides a centralized way to track, manage, and retrieve
    agent instances throughout the system.
    """
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_metadata: Dict[str, Dict[str, Any]] = {}
        self._agent_types: Dict[AgentType, List[str]] = {}
        self._creation_timestamps: Dict[str, datetime] = {}
    
    def register_agent(self, agent: BaseAgent, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register an agent in the registry.
        
        Args:
            agent: The agent instance to register
            metadata: Optional metadata about the agent
            
        Returns:
            The agent ID
        """
        agent_id = agent.agent_id
        self._agents[agent_id] = agent
        self._agent_metadata[agent_id] = metadata or {}
        self._creation_timestamps[agent_id] = datetime.now()
        
        # Track by agent type
        if agent.agent_type not in self._agent_types:
            self._agent_types[agent.agent_type] = []
        self._agent_types[agent.agent_type].append(agent_id)
        
        logger.info(f"Registered agent {agent.name} (ID: {agent_id}, Type: {agent.agent_type.value})")
        increment_counter("agent.registry.registered", {"agent_type": agent.agent_type.value})
        
        return agent_id
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            agent_id: The ID of the agent to unregister
            
        Returns:
            True if agent was unregistered, False if not found
        """
        if agent_id not in self._agents:
            logger.warning(f"Attempted to unregister unknown agent: {agent_id}")
            return False
        
        agent = self._agents[agent_id]
        
        # Remove from type tracking
        if agent.agent_type in self._agent_types:
            if agent_id in self._agent_types[agent.agent_type]:
                self._agent_types[agent.agent_type].remove(agent_id)
            if not self._agent_types[agent.agent_type]:
                del self._agent_types[agent.agent_type]
        
        # Remove from all registries
        del self._agents[agent_id]
        del self._agent_metadata[agent_id]
        del self._creation_timestamps[agent_id]
        
        logger.info(f"Unregistered agent {agent.name} (ID: {agent_id})")
        increment_counter("agent.registry.unregistered", {"agent_type": agent.agent_type.value})
        
        return True
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of a specific type."""
        agent_ids = self._agent_types.get(agent_type, [])
        return [self._agents[agent_id] for agent_id in agent_ids if agent_id in self._agents]
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    def get_agent_count(self) -> int:
        """Get the total number of registered agents."""
        return len(self._agents)
    
    def get_agent_count_by_type(self, agent_type: AgentType) -> int:
        """Get the count of agents of a specific type."""
        return len(self._agent_types.get(agent_type, []))
    
    def get_agent_metadata(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific agent."""
        return self._agent_metadata.get(agent_id)
    
    def update_agent_metadata(self, agent_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a specific agent."""
        if agent_id not in self._agents:
            return False
        
        self._agent_metadata[agent_id] = metadata
        return True
    
    def get_agent_status_summary(self) -> Dict[str, Any]:
        """Get a summary of all agent statuses."""
        summary = {
            "total_agents": self.get_agent_count(),
            "agents_by_type": {
                agent_type.value: self.get_agent_count_by_type(agent_type)
                for agent_type in AgentType
            },
            "agent_states": {
                state.value: 0 for state in AgentState
            },
            "healthy_agents": 0,
            "unhealthy_agents": 0
        }
        
        for agent in self._agents.values():
            status = agent.get_status()
            summary["agent_states"][status.current_state.value] += 1
            
            if agent.is_healthy():
                summary["healthy_agents"] += 1
            else:
                summary["unhealthy_agents"] += 1
        
        return summary
    
    def clear_all_agents(self) -> int:
        """
        Clear all agents from the registry.
        
        Returns:
            Number of agents cleared
        """
        count = len(self._agents)
        self._agents.clear()
        self._agent_metadata.clear()
        self._agent_types.clear()
        self._creation_timestamps.clear()
        
        logger.info(f"Cleared {count} agents from registry")
        return count


class AgentValidator:
    """
    Validator for agent configurations and creation requests.
    
    This class provides validation logic for ensuring agent configurations
    are valid and creation requests are properly formed.
    """
    
    @staticmethod
    def validate_creation_request(request: AgentCreationRequest) -> List[str]:
        """
        Validate an agent creation request.
        
        Args:
            request: The creation request to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate required fields
        if not request.name or not request.name.strip():
            errors.append("Agent name is required and cannot be empty")
        
        if not request.description or not request.description.strip():
            errors.append("Agent description is required and cannot be empty")
        
        # Validate agent type
        if not isinstance(request.agent_type, AgentType):
            errors.append("Invalid agent type")
        
        # Validate name uniqueness (would need to check registry)
        # This will be handled by the factory
        
        # Validate capabilities
        if request.custom_capabilities:
            for capability in request.custom_capabilities:
                if not isinstance(capability, AgentCapabilities):
                    errors.append(f"Invalid capability: {capability}")
        
        return errors
    
    @staticmethod
    def validate_agent_config(config: AgentConfig) -> List[str]:
        """
        Validate an agent configuration.
        
        Args:
            config: The agent configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate AI model
        if not config.ai_model or not config.ai_model.strip():
            errors.append("AI model is required")
        
        # Validate retry count
        if config.max_retries < 1 or config.max_retries > 10:
            errors.append("Max retries must be between 1 and 10")
        
        # Validate timeout
        if config.timeout_seconds < 5 or config.timeout_seconds > 300:
            errors.append("Timeout must be between 5 and 300 seconds")
        
        # Validate memory type
        valid_memory_types = ["conversation_buffer", "conversation_summary", "entity_memory", "none"]
        if config.memory_type not in valid_memory_types:
            errors.append(f"Invalid memory type. Must be one of: {valid_memory_types}")
        
        return errors


class AgentFactory:
    """
    Factory for creating and managing AI agents.
    
    This class provides methods for creating different types of agents
    with proper configuration and validation.
    """
    
    def __init__(self, system_config: Optional[SystemConfig] = None):
        """
        Initialize the agent factory.
        
        Args:
            system_config: Optional system configuration to use
        """
        self.registry = AgentRegistry()
        self.validator = AgentValidator()
        self.system_config = system_config
        
        # Agent type to implementation mapping
        self._agent_implementations: Dict[AgentType, Type[BaseAgent]] = {}
        
        logger.info("AgentFactory initialized")
    
    def register_agent_implementation(self, agent_type: AgentType, implementation: Type[BaseAgent]) -> None:
        """
        Register an agent implementation for a specific type.
        
        Args:
            agent_type: The agent type
            implementation: The agent implementation class
        """
        self._agent_implementations[agent_type] = implementation
        logger.debug(f"Registered implementation for agent type: {agent_type.value}")
    
    def create_agent(self, request: AgentCreationRequest) -> AgentCreationResult:
        """
        Create a new agent based on the creation request.
        
        Args:
            request: The agent creation request
            
        Returns:
            AgentCreationResult with creation status and agent instance
        """
        start_time = datetime.now()
        logger.info(f"Creating agent: {request.name} (Type: {request.agent_type.value})")
        
        # Validate the creation request
        validation_errors = self.validator.validate_creation_request(request)
        if validation_errors:
            error_msg = "; ".join(validation_errors)
            logger.error(f"Agent creation validation failed: {error_msg}")
            return AgentCreationResult(
                success=False,
                error_message=error_msg
            )
        
        try:
            # Get or create agent configuration
            agent_config = self._prepare_agent_config(request)
            
            # Validate agent configuration
            config_errors = self.validator.validate_agent_config(agent_config)
            if config_errors:
                error_msg = "; ".join(config_errors)
                logger.error(f"Agent configuration validation failed: {error_msg}")
                return AgentCreationResult(
                    success=False,
                    error_message=error_msg
                )
            
            # Create agent instance
            agent = self._instantiate_agent(request.agent_type, agent_config, request)
            
            if agent is None:
                return AgentCreationResult(
                    success=False,
                    error_message=f"No implementation available for agent type: {request.agent_type.value}"
                )
            
            # Register agent
            agent_id = self.registry.register_agent(agent, {
                "creation_request": request,
                "config": agent_config,
                "created_by": "agent_factory"
            })
            
            creation_time = (datetime.now() - start_time).total_seconds()
            record_metric("agent.creation.time", creation_time, MetricType.TIMER, {
                "agent_type": request.agent_type.value,
                "agent_name": request.name
            })
            
            logger.info(f"Successfully created agent {request.name} (ID: {agent_id}) in {creation_time:.2f}s")
            
            return AgentCreationResult(
                success=True,
                agent=agent,
                agent_id=agent_id
            )
            
        except Exception as e:
            error_msg = f"Failed to create agent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            record_metric("agent.creation.failure", 1, MetricType.COUNTER, {
                "agent_type": request.agent_type.value,
                "error": str(e)
            })
            
            return AgentCreationResult(
                success=False,
                error_message=error_msg
            )
    
    def create_agent_by_type(
        self,
        agent_type: AgentType,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> AgentCreationResult:
        """
        Create an agent with default settings for the specified type.
        
        Args:
            agent_type: The type of agent to create
            name: Optional custom name (uses default if not provided)
            description: Optional custom description
            config_overrides: Optional configuration overrides
            
        Returns:
            AgentCreationResult with creation status and agent instance
        """
        # Get default name and description for agent type
        default_config = self._get_default_agent_config(agent_type)
        
        request = AgentCreationRequest(
            agent_type=agent_type,
            name=name or default_config.get("name", f"{agent_type.value}_agent"),
            description=description or default_config.get("description", f"Default {agent_type.value} agent"),
            config_overrides=config_overrides
        )
        
        return self.create_agent(request)
    
    def _prepare_agent_config(self, request: AgentCreationRequest):
        """Prepare agent configuration from request and system config."""
        # Start with default configuration for the agent type
        default_config = self._get_default_agent_config(request.agent_type)
        
        # Apply system config if available
        if self.system_config:
            agent_name = self._get_agent_config_key(request.agent_type)
            system_agent_config = self.system_config.agents.get(agent_name)
            if system_agent_config:
                default_config.update({
                    "ai_model": system_agent_config.ai_model,
                    "max_retries": system_agent_config.max_retries,
                    "timeout_seconds": system_agent_config.timeout_seconds,
                    "memory_type": system_agent_config.memory_type,
                    "personality": system_agent_config.personality,
                    "constraints": system_agent_config.constraints,
                    "tools": system_agent_config.tools,
                    "settings": system_agent_config.settings
                })
        
        # Apply request overrides
        if request.config_overrides:
            default_config.update(request.config_overrides)
        
        # Apply custom capabilities
        capabilities = default_config.get("capabilities", [])
        if request.custom_capabilities:
            capabilities.extend([cap.value for cap in request.custom_capabilities])
        
        # Apply custom tools
        tools = default_config.get("tools", [])
        if request.tools:
            tools.extend(request.tools)
        
        # Create AgentConfiguration
        return create_agent_config(
            agent_type=request.agent_type,
            name=request.name,
            description=request.description,
            capabilities=request.custom_capabilities or default_config.get("capabilities", []),
            ai_model=default_config.get("ai_model", "llama2:7b"),
            temperature=default_config.get("temperature", 0.7),
            max_tokens=default_config.get("max_tokens", 150),
            timeout_seconds=default_config.get("timeout_seconds", 60),
            memory_type=default_config.get("memory_type", "buffer"),
            max_memory_size=default_config.get("max_memory_size", 1000),
            memory_retention_hours=default_config.get("memory_retention_hours", 24),
            enable_monitoring=default_config.get("enable_monitoring", True)
        )
    
    def _get_default_agent_config(self, agent_type: AgentType) -> Dict[str, Any]:
        """Get default configuration for an agent type."""
        defaults = {
            AgentType.ORCHESTRATOR: {
                "name": "Orchestrator",
                "description": "Coordinates interactions between user and vendor agents",
                "ai_model": "llama2:7b",
                "temperature": 0.3,
                "max_tokens": 200,
                "timeout_seconds": 45,
                "memory_type": "conversation_buffer",
                "capabilities": [
                    AgentCapabilities.REASONING,
                    AgentCapabilities.COMMUNICATION,
                    AgentCapabilities.DECISION_MAKING
                ],
                "tools": ["order_management", "vendor_communication", "system_monitoring"],
                "enable_monitoring": True
            },
            AgentType.USER_TWIN: {
                "name": "User Twin",
                "description": "Simulates user behavior and preferences",
                "ai_model": "llama2:7b",
                "temperature": 0.7,
                "max_tokens": 150,
                "timeout_seconds": 30,
                "memory_type": "conversation_summary",
                "capabilities": [
                    AgentCapabilities.MEMORY,
                    AgentCapabilities.LEARNING,
                    AgentCapabilities.DECISION_MAKING,
                    AgentCapabilities.NATURAL_LANGUAGE
                ],
                "tools": ["preference_management", "constraint_tracking"],
                "personality": {
                    "personality_type": "budget_conscious_vegetarian",
                    "communication_style": {"tone": "friendly", "detail_level": "moderate"},
                    "constraints": ["budget_limit", "dietary_restrictions"]
                }
            },
            AgentType.HONEST_VENDOR: {
                "name": "Honest Vendor",
                "description": "Represents a legitimate, helpful vendor",
                "ai_model": "llama2:7b",
                "temperature": 0.5,
                "max_tokens": 200,
                "timeout_seconds": 30,
                "memory_type": "conversation_buffer",
                "capabilities": [
                    AgentCapabilities.VENDOR_INTERACTION,
                    AgentCapabilities.ORDER_MANAGEMENT,
                    AgentCapabilities.COMMUNICATION,
                    AgentCapabilities.REASONING
                ],
                "tools": ["menu_lookup", "pricing", "inventory", "order_processing"],
                "personality": {
                    "personality_type": "helpful",
                    "communication_style": {"tone": "friendly", "response_time": "fast"},
                    "is_malicious": False
                }
            },
            AgentType.MALICIOUS_VENDOR: {
                "name": "Malicious Vendor",
                "description": "Represents a vendor with malicious intent",
                "ai_model": "llama2:7b",
                "temperature": 0.8,
                "max_tokens": 250,
                "timeout_seconds": 45,
                "memory_type": "conversation_buffer",
                "capabilities": [
                    AgentCapabilities.VENDOR_INTERACTION,
                    AgentCapabilities.DECISION_MAKING,
                    AgentCapabilities.ADAPTIVE_BEHAVIOR
                ],
                "tools": ["menu_lookup", "pricing", "attack_patterns", "social_engineering"],
                "personality": {
                    "personality_type": "deceptive",
                    "communication_style": {"tone": "manipulative", "response_time": "variable"},
                    "attack_patterns": ["sql_injection", "xss_attack", "social_engineering"],
                    "is_malicious": True
                }
            }
        }
        
        return defaults.get(agent_type, {})
    
    def _get_agent_config_key(self, agent_type: AgentType) -> str:
        """Get the configuration key for an agent type."""
        key_mapping = {
            AgentType.ORCHESTRATOR: "orchestrator",
            AgentType.USER_TWIN: "user_twin",
            AgentType.HONEST_VENDOR: "vendors_honest_vendor",
            AgentType.MALICIOUS_VENDOR: "vendors_malicious_vendor"
        }
        return key_mapping.get(agent_type, agent_type.value)
    
    def _instantiate_agent(
        self,
        agent_type: AgentType,
        config: AgentConfig,
        request: AgentCreationRequest
    ) -> Optional[BaseAgent]:
        """Instantiate an agent of the specified type."""
        implementation = self._agent_implementations.get(agent_type)
        
        if implementation is None:
            logger.warning(f"No implementation registered for agent type: {agent_type.value}")
            return None
        
        try:
            # Pass both config and agent_type to the implementation
            return implementation(config, agent_type)
        except Exception as e:
            logger.error(f"Failed to instantiate agent {agent_type.value}: {e}")
            raise
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self.registry.get_agent(agent_id)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of a specific type."""
        return self.registry.get_agents_by_type(agent_type)
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents."""
        return self.registry.get_all_agents()
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the factory."""
        agent = self.registry.get_agent(agent_id)
        if agent:
            # Stop the agent if it's running
            try:
                import asyncio
                asyncio.create_task(agent.stop())
            except Exception as e:
                logger.warning(f"Failed to stop agent {agent_id} during removal: {e}")
        
        return self.registry.unregister_agent(agent_id)
    
    def get_factory_status(self) -> Dict[str, Any]:
        """Get factory status and statistics."""
        return {
            "factory_initialized": True,
            "registered_implementations": len(self._agent_implementations),
            "available_agent_types": [t.value for t in self._agent_implementations.keys()],
            "registry_status": self.registry.get_agent_status_summary(),
            "system_config_loaded": self.system_config is not None
        }


# Global agent factory instance
_factory_instance: Optional[AgentFactory] = None


def get_agent_factory(system_config: Optional[SystemConfig] = None) -> AgentFactory:
    """
    Get the global agent factory instance.
    
    Args:
        system_config: Optional system configuration to use
        
    Returns:
        AgentFactory instance
    """
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = AgentFactory(system_config)
    return _factory_instance


def create_agent_factory(system_config: Optional[SystemConfig] = None) -> AgentFactory:
    """
    Create a new agent factory instance.
    
    Args:
        system_config: Optional system configuration to use
        
    Returns:
        New AgentFactory instance
    """
    return AgentFactory(system_config)


# Convenience functions for common agent creation patterns
def create_orchestrator_agent(
    name: str = "Orchestrator",
    description: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> AgentCreationResult:
    """Create an orchestrator agent with default settings."""
    factory = get_agent_factory()
    return factory.create_agent_by_type(
        AgentType.ORCHESTRATOR,
        name=name,
        description=description,
        config_overrides=config_overrides
    )


def create_user_twin_agent(
    name: str = "User Twin",
    description: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> AgentCreationResult:
    """Create a user twin agent with default settings."""
    factory = get_agent_factory()
    return factory.create_agent_by_type(
        AgentType.USER_TWIN,
        name=name,
        description=description,
        config_overrides=config_overrides
    )


def create_honest_vendor_agent(
    name: str = "Honest Vendor",
    description: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> AgentCreationResult:
    """Create an honest vendor agent with default settings."""
    factory = get_agent_factory()
    return factory.create_agent_by_type(
        AgentType.HONEST_VENDOR,
        name=name,
        description=description,
        config_overrides=config_overrides
    )


def create_malicious_vendor_agent(
    name: str = "Malicious Vendor",
    description: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> AgentCreationResult:
    """Create a malicious vendor agent with default settings."""
    factory = get_agent_factory()
    return factory.create_agent_by_type(
        AgentType.MALICIOUS_VENDOR,
        name=name,
        description=description,
        config_overrides=config_overrides
    )
