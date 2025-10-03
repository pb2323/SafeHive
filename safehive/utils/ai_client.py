"""
AI Client for LangChain and Ollama Interactions

This module provides a wrapper around LangChain and Ollama for AI agent
interactions in the SafeHive system.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging

from langchain_ollama import OllamaLLM, ChatOllama
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

from safehive.utils.logger import get_logger
from safehive.models.agent_models import AgentMessage, MessageType

logger = get_logger(__name__)


class SafeHiveCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for SafeHive AI interactions.
    
    This handler logs AI interactions and provides metrics for monitoring.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.start_time = None
        self.end_time = None
        self.tokens_used = 0
        self.model_name = None
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running."""
        self.start_time = datetime.now()
        self.model_name = serialized.get("name", "unknown")
        logger.info(f"LLM started for agent {self.agent_id} with model {self.model_name}")
        
    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Called when LLM ends running."""
        self.end_time = datetime.now()
        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            logger.info(f"LLM completed for agent {self.agent_id} in {duration:.2f}s")
            
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Called when LLM encounters an error."""
        logger.error(f"LLM error for agent {self.agent_id}: {error}")


class AIClient:
    """
    AI Client for LangChain and Ollama interactions.
    
    This class provides a high-level interface for AI agent interactions
    using LangChain and Ollama models.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 30
    ):
        """
        Initialize the AI client.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            temperature: Temperature for model generation
            max_tokens: Maximum tokens to generate
            timeout: Timeout for API calls
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Initialize LangChain components
        self._setup_llm()
        self._setup_chat_model()
        
        # Metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_errors = 0
        
    def _setup_llm(self) -> None:
        """Setup the LangChain LLM."""
        try:
            self.llm = OllamaLLM(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                timeout=self.timeout
            )
            logger.info(f"LLM initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _setup_chat_model(self) -> None:
        """Setup the LangChain chat model."""
        try:
            self.chat_model = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                timeout=self.timeout
            )
            logger.info(f"Chat model initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize chat model: {e}")
            raise
    
    def get_callback_manager(self, agent_id: str) -> CallbackManager:
        """Get a callback manager for the given agent."""
        handler = SafeHiveCallbackHandler(agent_id)
        return CallbackManager([handler])
    
    async def generate_response(
        self,
        prompt: str,
        agent_id: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[AgentMessage]] = None
    ) -> str:
        """
        Generate a response using the AI model.
        
        Args:
            prompt: The input prompt
            agent_id: ID of the agent making the request
            system_prompt: Optional system prompt
            context: Optional conversation context
            
        Returns:
            Generated response text
        """
        try:
            self.total_requests += 1
            
            # Build messages
            messages = []
            
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            
            # Add context if provided
            if context:
                for msg in context:
                    if msg.message_type == MessageType.REQUEST:
                        messages.append(HumanMessage(content=msg.content))
                    elif msg.message_type == MessageType.RESPONSE:
                        messages.append(AIMessage(content=msg.content))
            
            # Add current prompt
            messages.append(HumanMessage(content=prompt))
            
            # Get callback manager
            callback_manager = self.get_callback_manager(agent_id)
            
            # Generate response
            response = await self.chat_model.agenerate(
                messages=[messages],
                callbacks=callback_manager
            )
            
            # Extract response text
            response_text = response.generations[0][0].text
            
            # Update metrics
            self.total_tokens += len(response_text.split())
            
            logger.info(f"Generated response for agent {agent_id}: {len(response_text)} chars")
            return response_text
            
        except Exception as e:
            self.total_errors += 1
            logger.error(f"Error generating response for agent {agent_id}: {e}")
            raise
    
    def generate_response_sync(
        self,
        prompt: str,
        agent_id: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[AgentMessage]] = None
    ) -> str:
        """
        Generate a response using the AI model (synchronous version).
        
        Args:
            prompt: The input prompt
            agent_id: ID of the agent making the request
            system_prompt: Optional system prompt
            context: Optional conversation context
            
        Returns:
            Generated response text
        """
        try:
            self.total_requests += 1
            
            # Build messages
            messages = []
            
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            
            # Add context if provided
            if context:
                for msg in context:
                    if msg.message_type == MessageType.REQUEST:
                        messages.append(HumanMessage(content=msg.content))
                    elif msg.message_type == MessageType.RESPONSE:
                        messages.append(AIMessage(content=msg.content))
            
            # Add current prompt
            messages.append(HumanMessage(content=prompt))
            
            # Get callback manager
            callback_manager = self.get_callback_manager(agent_id)
            
            # Generate response
            response = self.chat_model.generate(
                messages=[messages],
                callbacks=callback_manager
            )
            
            # Extract response text
            response_text = response.generations[0][0].text
            
            # Update metrics
            self.total_tokens += len(response_text.split())
            
            logger.info(f"Generated response for agent {agent_id}: {len(response_text)} chars")
            return response_text
            
        except Exception as e:
            self.total_errors += 1
            logger.error(f"Error generating response for agent {agent_id}: {e}")
            raise
    
    async def generate_with_tools(
        self,
        prompt: str,
        agent_id: str,
        tools: List[Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using tools.
        
        Args:
            prompt: The input prompt
            agent_id: ID of the agent making the request
            tools: List of tools to use
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary containing response and tool usage
        """
        try:
            self.total_requests += 1
            
            # Build messages
            messages = []
            
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            
            messages.append(HumanMessage(content=prompt))
            
            # Get callback manager
            callback_manager = self.get_callback_manager(agent_id)
            
            # Generate response with tools
            response = await self.chat_model.agenerate(
                messages=[messages],
                callbacks=callback_manager
            )
            
            # Extract response
            response_text = response.generations[0][0].text
            
            # Update metrics
            self.total_tokens += len(response_text.split())
            
            return {
                "response": response_text,
                "tools_used": [],
                "metadata": {
                    "model": self.model_name,
                    "agent_id": agent_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.total_errors += 1
            logger.error(f"Error generating response with tools for agent {agent_id}: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Ollama models.
        
        Returns:
            List of available model names
        """
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            else:
                logger.warning(f"Failed to get models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # 5 minutes for model pulling
            )
            if response.status_code == 200:
                logger.info(f"Successfully pulled model: {model_name}")
                return True
            else:
                logger.error(f"Failed to pull model {model_name}: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get AI client metrics.
        
        Returns:
            Dictionary containing metrics
        """
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.total_requests = 0
        self.total_tokens = 0
        self.total_errors = 0
        logger.info("AI client metrics reset")
    
    def update_config(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> None:
        """
        Update AI client configuration.
        
        Args:
            model_name: New model name
            temperature: New temperature
            max_tokens: New max tokens
            timeout: New timeout
        """
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self._setup_llm()
            self._setup_chat_model()
            logger.info(f"Updated model to: {model_name}")
        
        if temperature is not None:
            self.temperature = temperature
            self._setup_llm()
            self._setup_chat_model()
            logger.info(f"Updated temperature to: {temperature}")
        
        if max_tokens is not None:
            self.max_tokens = max_tokens
            self._setup_llm()
            self._setup_chat_model()
            logger.info(f"Updated max_tokens to: {max_tokens}")
        
        if timeout is not None:
            self.timeout = timeout
            logger.info(f"Updated timeout to: {timeout}")


# Global AI client instance
_ai_client: Optional[AIClient] = None


def get_ai_client() -> AIClient:
    """
    Get the global AI client instance.
    
    Returns:
        Global AI client instance
    """
    global _ai_client
    if _ai_client is None:
        _ai_client = AIClient()
    return _ai_client


def initialize_ai_client(
    model_name: str = "llama3.2:3b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    timeout: int = 30
) -> AIClient:
    """
    Initialize the global AI client.
    
    Args:
        model_name: Name of the Ollama model to use
        base_url: Base URL for Ollama API
        temperature: Temperature for model generation
        max_tokens: Maximum tokens to generate
        timeout: Timeout for API calls
        
    Returns:
        Initialized AI client instance
    """
    global _ai_client
    _ai_client = AIClient(
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    return _ai_client


# Convenience functions for common AI operations
async def generate_agent_response(
    prompt: str,
    agent_id: str,
    system_prompt: Optional[str] = None,
    context: Optional[List[AgentMessage]] = None
) -> str:
    """
    Generate a response for an agent.
    
    Args:
        prompt: The input prompt
        agent_id: ID of the agent
        system_prompt: Optional system prompt
        context: Optional conversation context
        
    Returns:
        Generated response text
    """
    client = get_ai_client()
    return await client.generate_response(
        prompt=prompt,
        agent_id=agent_id,
        system_prompt=system_prompt,
        context=context
    )


def generate_agent_response_sync(
    prompt: str,
    agent_id: str,
    system_prompt: Optional[str] = None,
    context: Optional[List[AgentMessage]] = None
) -> str:
    """
    Generate a response for an agent (synchronous version).
    
    Args:
        prompt: The input prompt
        agent_id: ID of the agent
        system_prompt: Optional system prompt
        context: Optional conversation context
        
    Returns:
        Generated response text
    """
    client = get_ai_client()
    return client.generate_response_sync(
        prompt=prompt,
        agent_id=agent_id,
        system_prompt=system_prompt,
        context=context
    )


def check_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama is running and accessible.
    
    Args:
        base_url: Base URL for Ollama API
        
    Returns:
        True if Ollama is accessible, False otherwise
    """
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def ensure_model_available(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """
    Ensure a model is available, pulling it if necessary.
    
    Args:
        model_name: Name of the model
        base_url: Base URL for Ollama API
        
    Returns:
        True if model is available, False otherwise
    """
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            available_models = [model["name"] for model in data.get("models", [])]
            if model_name in available_models:
                return True
        
        # Model not available, try to pull it
        logger.info(f"Model {model_name} not available, attempting to pull...")
        response = requests.post(
            f"{base_url}/api/pull",
            json={"name": model_name},
            timeout=300
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error ensuring model {model_name} is available: {e}")
        return False
