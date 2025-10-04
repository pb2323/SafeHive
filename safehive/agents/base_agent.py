"""
Base Agent Implementation for SafeHive AI Security Sandbox

This module provides the foundational agent class that all SafeHive agents inherit from.
It integrates with LangChain and provides common functionality for AI agent operations.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import json

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import BaseTool
# from langchain.callbacks import BaseCallbackHandler  # Not available in current version

from ..utils.ai_client import get_ai_client, AIClient
from ..utils.agent_memory import get_memory_manager, SafeHiveMemoryManager
from ..models.agent_models import (
    AgentType, AgentState, AgentMessage, Conversation, 
    AgentMemory, AgentStatus as AgentStatusModel, AgentPersonality
)
from ..utils.logger import get_logger
from ..utils.metrics import record_metric, MetricType

logger = get_logger(__name__)


class AgentCapabilities(Enum):
    """Defines the capabilities that an agent can have."""
    
    # Core capabilities
    REASONING = "reasoning"
    MEMORY = "memory"
    LEARNING = "learning"
    COMMUNICATION = "communication"
    
    # Specialized capabilities
    ORDER_MANAGEMENT = "order_management"
    VENDOR_INTERACTION = "vendor_interaction"
    SECURITY_ANALYSIS = "security_analysis"
    USER_SIMULATION = "user_simulation"
    ORCHESTRATION = "orchestration"
    
    # Advanced capabilities
    NATURAL_LANGUAGE = "natural_language"
    DECISION_MAKING = "decision_making"
    CONTEXT_AWARENESS = "context_awareness"
    ADAPTIVE_BEHAVIOR = "adaptive_behavior"


@dataclass
class AgentConfiguration:
    """Configuration for agent initialization and behavior."""
    
    # Core settings
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType = AgentType.ORCHESTRATOR
    name: str = "SafeHive Agent"
    description: str = "A SafeHive AI agent"
    
    # AI model settings
    ai_model: str = "llama2:7b"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout_seconds: int = 30
    max_retries: int = 3
    
    # Memory settings
    memory_type: str = "conversation_buffer"
    max_memory_size: int = 1000
    memory_retention_hours: int = 24
    
    # Capabilities
    capabilities: List[AgentCapabilities] = field(default_factory=list)
    
    # Behavior settings
    personality: Optional[AgentPersonality] = None
    
    # Tools and resources
    tools: List[BaseTool] = field(default_factory=list)
    
    # Advanced settings
    enable_monitoring: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all SafeHive AI agents.
    
    This class provides common functionality for all agents including:
    - LangChain integration
    - Memory management
    - Communication protocols
    - State management
    - Monitoring and metrics
    """
    
    def __init__(self, config: AgentConfiguration):
        """Initialize the base agent with configuration."""
        self.config = config
        self.agent_id = config.agent_id
        self.agent_type = config.agent_type
        self.name = config.name
        self.description = config.description
        
        # Initialize components
        self._ai_client: Optional[AIClient] = None
        self._memory_manager: Optional[SafeHiveMemoryManager] = None
        self._agent_executor: Optional[AgentExecutor] = None
        
        # State management
        self._status = AgentState.IDLE
        self._last_activity = datetime.now()
        self._conversation_history: List[AgentMessage] = []
        
        # Performance tracking
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._average_response_time = 0.0
        
        # Callbacks (disabled for now due to LangChain version compatibility)
        self._callbacks: List[Any] = []
        
        logger.info(f"Initializing {self.name} (ID: {self.agent_id})")
        
        # Initialize agent components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all agent components."""
        try:
            # Initialize AI client
            self._ai_client = get_ai_client()
            
            # Initialize memory manager
            self._memory_manager = get_memory_manager(
                agent_id=self.agent_id,
                memory_type=self.config.memory_type,
                max_size=self.config.max_memory_size
            )
            
            # Initialize LangChain agent
            self._initialize_langchain_agent()
            
            # Set up callbacks
            self._setup_callbacks()
            
            logger.info(f"Successfully initialized {self.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            raise
    
    def _initialize_langchain_agent(self):
        """Initialize the LangChain agent with tools and prompt."""
        try:
            # For now, we'll create a simplified agent executor
            # This will be enhanced when we implement specific agent types
            if self._ai_client and self._ai_client.llm:
                # Get the agent's system prompt
                system_prompt = self._get_system_prompt()
                
                # Create the prompt template
                prompt_template = self._create_prompt_template(system_prompt)
                
                # Create the LangChain agent
                langchain_agent = create_react_agent(
                    llm=self._ai_client.llm,
                    tools=self.config.tools,
                    prompt=prompt_template
                )
                
                # Create the agent executor
                self._agent_executor = AgentExecutor(
                    agent=langchain_agent,
                    tools=self.config.tools,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=5
                    # callbacks=self._callbacks  # Disabled for compatibility
                )
                
                logger.debug(f"Initialized LangChain agent for {self.name}")
            else:
                logger.warning(f"No AI client available for {self.name}, using mock executor")
                self._agent_executor = None
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain agent for {self.name}: {e}")
            # Don't raise, just log the error and continue with mock executor
            self._agent_executor = None
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent. Must be implemented by subclasses."""
        pass
    
    def _create_prompt_template(self, system_prompt: str) -> PromptTemplate:
        """Create the prompt template for the agent."""
        template = f"""You are {self.name}, {self.description}.

{system_prompt}

You have access to the following tools:
{{tools}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought: {{agent_scratchpad}}"""
        
        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
    
    def _setup_callbacks(self):
        """Set up callbacks for monitoring and metrics."""
        # Callbacks disabled for now due to LangChain version compatibility
        # if self.config.enable_monitoring:
        #     self._callbacks.append(SafeHiveAgentCallbackHandler(self))
        pass
    
    # Public interface methods
    
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a message and return a response.
        
        Args:
            message: The input message to process
            context: Optional context information
            
        Returns:
            The agent's response
        """
        start_time = datetime.now()
        self._total_requests += 1
        self._status = AgentState.PROCESSING
        
        try:
            logger.info(f"{self.name} processing message: {message[:100]}...")
            
            # Add context to message if provided
            if context:
                message = self._enhance_message_with_context(message, context)
            
            # Process with LangChain agent
            response = await self._process_with_langchain(message)
            
            # Store in memory
            await self._store_conversation(message, response)
            
            # Update metrics
            self._successful_requests += 1
            self._update_response_time(start_time)
            
            # Record metrics
            if self.config.enable_metrics:
                record_metric(
                    f"agent.{self.agent_type.value}.message_processed",
                    1,
                    MetricType.COUNTER,
                    {"agent_id": self.agent_id, "success": True}
                )
            
            self._status = AgentState.IDLE
            self._last_activity = datetime.now()
            
            logger.info(f"{self.name} successfully processed message")
            return response
            
        except Exception as e:
            self._failed_requests += 1
            self._status = AgentState.ERROR
            
            logger.error(f"{self.name} failed to process message: {e}")
            
            if self.config.enable_metrics:
                record_metric(
                    f"agent.{self.agent_type.value}.message_processed",
                    1,
                    MetricType.COUNTER,
                    {"agent_id": self.agent_id, "success": False}
                )
            
            raise
    
    async def _process_with_langchain(self, message: str) -> str:
        """Process message using LangChain agent executor."""
        if not self._agent_executor:
            # Fallback to simple AI client response
            if self._ai_client:
                try:
                    response = await self._ai_client.generate_response(
                        message, 
                        system_prompt=self._get_system_prompt()
                    )
                    return response
                except Exception as e:
                    logger.error(f"Failed to generate response with AI client: {e}")
                    return f"I'm sorry, I encountered an error processing your message: {str(e)}"
            else:
                return f"I'm sorry, I'm not properly configured to respond to messages right now."
        
        # Run the agent executor
        try:
            result = await self._agent_executor.arun(message)
            return str(result)
        except Exception as e:
            logger.error(f"Agent executor failed: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def _enhance_message_with_context(self, message: str, context: Dict[str, Any]) -> str:
        """Enhance message with context information."""
        if not context:
            return message
        
        context_str = json.dumps(context, indent=2)
        enhanced_message = f"""Context Information:
{context_str}

Message: {message}"""
        
        return enhanced_message
    
    async def _store_conversation(self, input_message: str, response: str):
        """Store conversation in memory."""
        if not self._memory_manager:
            return
        
        try:
            # Create agent messages
            input_msg = AgentMessage(
                content=input_message,
                message_type="human",
                timestamp=datetime.now()
            )
            
            response_msg = AgentMessage(
                content=response,
                message_type="ai",
                timestamp=datetime.now()
            )
            
            # Store in memory manager
            await self._memory_manager.add_message(input_msg)
            await self._memory_manager.add_message(response_msg)
            
            # Store in local history
            self._conversation_history.extend([input_msg, response_msg])
            
        except Exception as e:
            logger.warning(f"Failed to store conversation for {self.name}: {e}")
    
    def _update_response_time(self, start_time: datetime):
        """Update average response time."""
        response_time = (datetime.now() - start_time).total_seconds()
        
        if self._average_response_time == 0:
            self._average_response_time = response_time
        else:
            # Simple moving average
            self._average_response_time = (self._average_response_time + response_time) / 2
    
    # Memory and context methods
    
    async def get_conversation_history(self, limit: Optional[int] = None) -> List[AgentMessage]:
        """Get conversation history."""
        if limit:
            return self._conversation_history[-limit:]
        return self._conversation_history.copy()
    
    async def clear_memory(self):
        """Clear agent memory."""
        if self._memory_manager:
            await self._memory_manager.clear_memory()
        self._conversation_history.clear()
        logger.info(f"Cleared memory for {self.name}")
    
    async def save_memory(self, file_path: str):
        """Save agent memory to file."""
        if self._memory_manager:
            await self._memory_manager.save_memory(file_path)
            logger.info(f"Saved memory for {self.name} to {file_path}")
    
    async def load_memory(self, file_path: str):
        """Load agent memory from file."""
        if self._memory_manager:
            await self._memory_manager.load_memory(file_path)
            logger.info(f"Loaded memory for {self.name} from {file_path}")
    
    # State and status methods
    
    def get_status(self) -> AgentStatusModel:
        """Get current agent status."""
        return AgentStatusModel(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            current_state=self._status,
            is_enabled=True,
            configuration={
                "name": self.name,
                "description": self.description,
                "capabilities": [cap.value for cap in self.config.capabilities],
                "memory_type": self.config.memory_type,
                "ai_model": self.config.ai_model
            },
            metrics={
                "total_requests": self._total_requests,
                "successful_requests": self._successful_requests,
                "failed_requests": self._failed_requests,
                "average_response_time": self._average_response_time,
                "memory_size": len(self._conversation_history)
            },
            last_activity=self._last_activity,
            error_count=self._failed_requests,
            success_count=self._successful_requests
        )
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy."""
        # Check if agent has been active recently
        time_since_activity = datetime.now() - self._last_activity
        if time_since_activity > timedelta(hours=1):
            return False
        
        # Check error rate
        if self._total_requests > 0:
            error_rate = self._failed_requests / self._total_requests
            if error_rate > 0.5:  # 50% error rate threshold
                return False
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "name": self.name,
            "status": self._status.value,
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": self._successful_requests / max(1, self._total_requests),
            "average_response_time": self._average_response_time,
            "memory_size": len(self._conversation_history),
            "last_activity": self._last_activity.isoformat(),
            "is_healthy": self.is_healthy()
        }
    
    # Tool management
    
    def add_tool(self, tool: BaseTool):
        """Add a tool to the agent."""
        self.config.tools.append(tool)
        # Reinitialize agent with new tools
        self._initialize_langchain_agent()
        logger.info(f"Added tool {tool.name} to {self.name}")
    
    def remove_tool(self, tool_name: str):
        """Remove a tool from the agent."""
        self.config.tools = [tool for tool in self.config.tools if tool.name != tool_name]
        # Reinitialize agent without the tool
        self._initialize_langchain_agent()
        logger.info(f"Removed tool {tool_name} from {self.name}")
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.config.tools]
    
    # Configuration and personality
    
    def update_personality(self, personality: AgentPersonality):
        """Update agent personality."""
        self.config.personality = personality
        # Reinitialize agent with new personality
        self._initialize_langchain_agent()
        logger.info(f"Updated personality for {self.name}")
    
    def get_personality(self) -> Optional[AgentPersonality]:
        """Get current agent personality."""
        return self.config.personality
    
    # Cleanup
    
    async def shutdown(self):
        """Shutdown the agent and clean up resources."""
        logger.info(f"Shutting down {self.name}")
        
        # Save memory if configured
        if self.config.enable_logging and self._memory_manager:
            try:
                await self.save_memory(f"logs/{self.agent_id}_memory.json")
            except Exception as e:
                logger.warning(f"Failed to save memory during shutdown: {e}")
        
        # Clear resources
        self._conversation_history.clear()
        self._callbacks.clear()
        
        self._status = AgentState.STOPPED
        logger.info(f"Successfully shut down {self.name}")


# class SafeHiveAgentCallbackHandler(BaseCallbackHandler):
#     """Custom callback handler for SafeHive agents."""
#     
#     def __init__(self, agent: BaseAgent):
#         super().__init__()
#         self.agent = agent
#     
#     def on_agent_action(self, action, **kwargs):
#         """Called when agent takes an action."""
#         logger.debug(f"{self.agent.name} took action: {action.tool}")
#     
#     def on_agent_finish(self, finish, **kwargs):
#         """Called when agent finishes."""
#         logger.debug(f"{self.agent.name} finished: {finish.return_values}")
#     
#     def on_tool_start(self, serialized, input_str, **kwargs):
#         """Called when tool starts."""
#         logger.debug(f"{self.agent.name} started tool: {serialized.get('name', 'unknown')}")
#     
#     def on_tool_end(self, output, **kwargs):
#         """Called when tool ends."""
#         logger.debug(f"{self.agent.name} ended tool with output: {output[:100]}...")


# Utility functions

def create_agent_config(
    agent_type: AgentType,
    name: str,
    description: str = "",
    capabilities: Optional[List[AgentCapabilities]] = None,
    **kwargs
) -> AgentConfiguration:
    """Create an agent configuration with sensible defaults."""
    
    if capabilities is None:
        capabilities = []
    
    return AgentConfiguration(
        agent_type=agent_type,
        name=name,
        description=description,
        capabilities=capabilities,
        **kwargs
    )


def get_agent_status_summary(agents: List[BaseAgent]) -> Dict[str, Any]:
    """Get a summary of multiple agents' status."""
    total_agents = len(agents)
    healthy_agents = sum(1 for agent in agents if agent.is_healthy())
    
    total_requests = sum(agent._total_requests for agent in agents)
    successful_requests = sum(agent._successful_requests for agent in agents)
    
    return {
        "total_agents": total_agents,
        "healthy_agents": healthy_agents,
        "unhealthy_agents": total_agents - healthy_agents,
        "health_percentage": (healthy_agents / max(1, total_agents)) * 100,
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "overall_success_rate": successful_requests / max(1, total_requests),
        "agents": [agent.get_metrics() for agent in agents]
    }
