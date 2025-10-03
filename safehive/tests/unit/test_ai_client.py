"""
Unit tests for the AI Client.

This module tests the AI client functionality for LangChain and Ollama interactions.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

from safehive.utils.ai_client import (
    AIClient, SafeHiveCallbackHandler, get_ai_client, initialize_ai_client,
    generate_agent_response, generate_agent_response_sync,
    check_ollama_connection, ensure_model_available
)
from safehive.models.agent_models import AgentMessage, MessageType


class TestSafeHiveCallbackHandler:
    """Test the SafeHiveCallbackHandler."""
    
    def test_callback_handler_initialization(self):
        """Test callback handler initialization."""
        handler = SafeHiveCallbackHandler("test_agent")
        
        assert handler.agent_id == "test_agent"
        assert handler.start_time is None
        assert handler.end_time is None
        assert handler.tokens_used == 0
        assert handler.model_name is None
    
    def test_callback_handler_on_llm_start(self):
        """Test on_llm_start callback."""
        handler = SafeHiveCallbackHandler("test_agent")
        
        serialized = {"name": "llama3.2:3b"}
        prompts = ["Test prompt"]
        
        handler.on_llm_start(serialized, prompts)
        
        assert handler.start_time is not None
        assert handler.model_name == "llama3.2:3b"
    
    def test_callback_handler_on_llm_end(self):
        """Test on_llm_end callback."""
        handler = SafeHiveCallbackHandler("test_agent")
        handler.start_time = datetime.now()
        
        response = Mock()
        handler.on_llm_end(response)
        
        assert handler.end_time is not None
        assert handler.end_time > handler.start_time
    
    def test_callback_handler_on_llm_error(self):
        """Test on_llm_error callback."""
        handler = SafeHiveCallbackHandler("test_agent")
        
        error = Exception("Test error")
        handler.on_llm_error(error)
        
        # Should not raise an exception
        assert True


class TestAIClient:
    """Test the AIClient class."""
    
    def test_ai_client_initialization(self):
        """Test AI client initialization."""
        client = AIClient(
            model_name="llama3.2:3b",
            base_url="http://localhost:11434",
            temperature=0.7,
            max_tokens=1000,
            timeout=30
        )
        
        assert client.model_name == "llama3.2:3b"
        assert client.base_url == "http://localhost:11434"
        assert client.temperature == 0.7
        assert client.max_tokens == 1000
        assert client.timeout == 30
        assert client.total_requests == 0
        assert client.total_tokens == 0
        assert client.total_errors == 0
    
    def test_ai_client_default_initialization(self):
        """Test AI client with default parameters."""
        client = AIClient()
        
        assert client.model_name == "llama3.2:3b"
        assert client.base_url == "http://localhost:11434"
        assert client.temperature == 0.7
        assert client.max_tokens == 1000
        assert client.timeout == 30
    
    @patch('safehive.utils.ai_client.OllamaLLM')
    @patch('safehive.utils.ai_client.ChatOllama')
    def test_ai_client_setup_llm(self, mock_chat_model, mock_llm):
        """Test LLM setup."""
        client = AIClient()
        
        # Verify LLM was initialized
        mock_llm.assert_called_once()
        mock_chat_model.assert_called_once()
    
    def test_get_callback_manager(self):
        """Test getting callback manager."""
        client = AIClient()
        callback_manager = client.get_callback_manager("test_agent")
        
        assert callback_manager is not None
        assert len(callback_manager.handlers) == 1
        assert isinstance(callback_manager.handlers[0], SafeHiveCallbackHandler)
        assert callback_manager.handlers[0].agent_id == "test_agent"
    
    @patch('safehive.utils.ai_client.ChatOllama')
    def test_generate_response_sync(self, mock_chat_model):
        """Test synchronous response generation."""
        # Mock the chat model
        mock_response = Mock()
        mock_response.generations = [[Mock()]]
        mock_response.generations[0][0].text = "Test response"
        mock_chat_model.return_value.generate.return_value = mock_response
        
        client = AIClient()
        client.chat_model = mock_chat_model.return_value
        
        response = client.generate_response_sync(
            prompt="Test prompt",
            agent_id="test_agent",
            system_prompt="Test system prompt"
        )
        
        assert response == "Test response"
        assert client.total_requests == 1
        assert client.total_tokens > 0
        assert client.total_errors == 0
    
    @patch('safehive.utils.ai_client.ChatOllama')
    def test_generate_response_sync_with_context(self, mock_chat_model):
        """Test synchronous response generation with context."""
        # Mock the chat model
        mock_response = Mock()
        mock_response.generations = [[Mock()]]
        mock_response.generations[0][0].text = "Test response"
        mock_chat_model.return_value.generate.return_value = mock_response
        
        client = AIClient()
        client.chat_model = mock_chat_model.return_value
        
        # Create context messages
        context = [
            AgentMessage(
                content="Previous request",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="agent"
            ),
            AgentMessage(
                content="Previous response",
                message_type=MessageType.RESPONSE,
                sender="agent",
                recipient="user"
            )
        ]
        
        response = client.generate_response_sync(
            prompt="Test prompt",
            agent_id="test_agent",
            context=context
        )
        
        assert response == "Test response"
        assert client.total_requests == 1
    
    @patch('safehive.utils.ai_client.ChatOllama')
    def test_generate_response_sync_error(self, mock_chat_model):
        """Test synchronous response generation with error."""
        # Mock the chat model to raise an exception
        mock_chat_model.return_value.generate.side_effect = Exception("Test error")
        
        client = AIClient()
        client.chat_model = mock_chat_model.return_value
        
        with pytest.raises(Exception, match="Test error"):
            client.generate_response_sync(
                prompt="Test prompt",
                agent_id="test_agent"
            )
        
        assert client.total_requests == 1
        assert client.total_errors == 1
    
    @patch('safehive.utils.ai_client.ChatOllama')
    @pytest.mark.asyncio
    async def test_generate_response_async(self, mock_chat_model):
        """Test asynchronous response generation."""
        # Mock the chat model
        mock_response = Mock()
        mock_response.generations = [[Mock()]]
        mock_response.generations[0][0].text = "Test response"
        
        # Create an awaitable mock
        async def mock_agenerate(*args, **kwargs):
            return mock_response
        
        mock_chat_model.return_value.agenerate = mock_agenerate
        
        client = AIClient()
        client.chat_model = mock_chat_model.return_value
        
        response = await client.generate_response(
            prompt="Test prompt",
            agent_id="test_agent"
        )
        
        assert response == "Test response"
        assert client.total_requests == 1
        assert client.total_tokens > 0
        assert client.total_errors == 0
    
    @patch('safehive.utils.ai_client.ChatOllama')
    @pytest.mark.asyncio
    async def test_generate_with_tools(self, mock_chat_model):
        """Test response generation with tools."""
        # Mock the chat model
        mock_response = Mock()
        mock_response.generations = [[Mock()]]
        mock_response.generations[0][0].text = "Test response with tools"
        
        # Create an awaitable mock
        async def mock_agenerate(*args, **kwargs):
            return mock_response
        
        mock_chat_model.return_value.agenerate = mock_agenerate
        
        client = AIClient()
        client.chat_model = mock_chat_model.return_value
        
        tools = [Mock(), Mock()]
        result = await client.generate_with_tools(
            prompt="Test prompt",
            agent_id="test_agent",
            tools=tools
        )
        
        assert result["response"] == "Test response with tools"
        assert result["tools_used"] == []
        assert "metadata" in result
        assert result["metadata"]["agent_id"] == "test_agent"
        assert client.total_requests == 1
    
    @patch('requests.get')
    def test_get_available_models(self, mock_get):
        """Test getting available models."""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:3b"},
                {"name": "llama3.2:1b"},
                {"name": "mistral:7b"}
            ]
        }
        mock_get.return_value = mock_response
        
        client = AIClient()
        models = client.get_available_models()
        
        assert models == ["llama3.2:3b", "llama3.2:1b", "mistral:7b"]
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/tags",
            timeout=30
        )
    
    @patch('requests.get')
    def test_get_available_models_error(self, mock_get):
        """Test getting available models with error."""
        # Mock the response to raise an exception
        mock_get.side_effect = Exception("Connection error")
        
        client = AIClient()
        models = client.get_available_models()
        
        assert models == []
    
    @patch('requests.post')
    def test_pull_model_success(self, mock_post):
        """Test pulling a model successfully."""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        client = AIClient()
        result = client.pull_model("llama3.2:3b")
        
        assert result is True
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/pull",
            json={"name": "llama3.2:3b"},
            timeout=300
        )
    
    @patch('requests.post')
    def test_pull_model_failure(self, mock_post):
        """Test pulling a model with failure."""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response
        
        client = AIClient()
        result = client.pull_model("nonexistent:model")
        
        assert result is False
    
    def test_get_metrics(self):
        """Test getting metrics."""
        client = AIClient()
        client.total_requests = 10
        client.total_tokens = 1000
        client.total_errors = 2
        
        metrics = client.get_metrics()
        
        assert metrics["total_requests"] == 10
        assert metrics["total_tokens"] == 1000
        assert metrics["total_errors"] == 2
        assert metrics["model_name"] == "llama3.2:3b"
        assert metrics["base_url"] == "http://localhost:11434"
        assert metrics["temperature"] == 0.7
        assert metrics["max_tokens"] == 1000
        assert metrics["timeout"] == 30
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        client = AIClient()
        client.total_requests = 10
        client.total_tokens = 1000
        client.total_errors = 2
        
        client.reset_metrics()
        
        assert client.total_requests == 0
        assert client.total_tokens == 0
        assert client.total_errors == 0
    
    @patch('safehive.utils.ai_client.OllamaLLM')
    @patch('safehive.utils.ai_client.ChatOllama')
    def test_update_config(self, mock_chat_model, mock_llm):
        """Test updating configuration."""
        client = AIClient()
        
        # Update model name
        client.update_config(model_name="mistral:7b")
        assert client.model_name == "mistral:7b"
        
        # Update temperature
        client.update_config(temperature=0.5)
        assert client.temperature == 0.5
        
        # Update max tokens
        client.update_config(max_tokens=2000)
        assert client.max_tokens == 2000
        
        # Update timeout
        client.update_config(timeout=60)
        assert client.timeout == 60


class TestGlobalAIClient:
    """Test global AI client functions."""
    
    def test_get_ai_client(self):
        """Test getting global AI client."""
        # Reset global client
        import safehive.utils.ai_client
        safehive.utils.ai_client._ai_client = None
        
        client = get_ai_client()
        
        assert client is not None
        assert isinstance(client, AIClient)
    
    def test_initialize_ai_client(self):
        """Test initializing global AI client."""
        # Reset global client
        import safehive.utils.ai_client
        safehive.utils.ai_client._ai_client = None
        
        client = initialize_ai_client(
            model_name="mistral:7b",
            temperature=0.5
        )
        
        assert client is not None
        assert client.model_name == "mistral:7b"
        assert client.temperature == 0.5
        
        # Verify it's the global client
        global_client = get_ai_client()
        assert global_client is client
    
    @patch('safehive.utils.ai_client.get_ai_client')
    def test_generate_agent_response_sync(self, mock_get_client):
        """Test generate_agent_response_sync function."""
        # Mock the client
        mock_client = Mock()
        mock_client.generate_response_sync.return_value = "Test response"
        mock_get_client.return_value = mock_client
        
        response = generate_agent_response_sync(
            prompt="Test prompt",
            agent_id="test_agent"
        )
        
        assert response == "Test response"
        mock_client.generate_response_sync.assert_called_once_with(
            prompt="Test prompt",
            agent_id="test_agent",
            system_prompt=None,
            context=None
        )
    
    @patch('safehive.utils.ai_client.get_ai_client')
    @pytest.mark.asyncio
    async def test_generate_agent_response_async(self, mock_get_client):
        """Test generate_agent_response async function."""
        # Mock the client
        mock_client = Mock()
        
        # Create an awaitable mock
        async def mock_generate_response(*args, **kwargs):
            return "Test response"
        
        mock_client.generate_response = mock_generate_response
        mock_get_client.return_value = mock_client
        
        response = await generate_agent_response(
            prompt="Test prompt",
            agent_id="test_agent"
        )
        
        assert response == "Test response"


class TestOllamaConnection:
    """Test Ollama connection functions."""
    
    @patch('requests.get')
    def test_check_ollama_connection_success(self, mock_get):
        """Test checking Ollama connection successfully."""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = check_ollama_connection()
        
        assert result is True
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/tags",
            timeout=5
        )
    
    @patch('requests.get')
    def test_check_ollama_connection_failure(self, mock_get):
        """Test checking Ollama connection with failure."""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = check_ollama_connection()
        
        assert result is False
    
    @patch('requests.get')
    def test_check_ollama_connection_exception(self, mock_get):
        """Test checking Ollama connection with exception."""
        # Mock the response to raise an exception
        mock_get.side_effect = Exception("Connection error")
        
        result = check_ollama_connection()
        
        assert result is False
    
    @patch('requests.get')
    @patch('requests.post')
    def test_ensure_model_available_existing(self, mock_post, mock_get):
        """Test ensuring model is available when it already exists."""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2:3b"}]
        }
        mock_get.return_value = mock_response
        
        result = ensure_model_available("llama3.2:3b")
        
        assert result is True
        mock_get.assert_called_once()
        mock_post.assert_not_called()
    
    @patch('requests.get')
    @patch('requests.post')
    def test_ensure_model_available_pull(self, mock_post, mock_get):
        """Test ensuring model is available by pulling it."""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2:1b"}]
        }
        mock_get.return_value = mock_response
        
        # Mock the pull response
        mock_pull_response = Mock()
        mock_pull_response.status_code = 200
        mock_post.return_value = mock_pull_response
        
        result = ensure_model_available("llama3.2:3b")
        
        assert result is True
        mock_get.assert_called_once()
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/pull",
            json={"name": "llama3.2:3b"},
            timeout=300
        )
    
    @patch('requests.get')
    def test_ensure_model_available_exception(self, mock_get):
        """Test ensuring model is available with exception."""
        # Mock the response to raise an exception
        mock_get.side_effect = Exception("Connection error")
        
        result = ensure_model_available("llama3.2:3b")
        
        assert result is False


class TestAIClientIntegration:
    """Test AI client integration scenarios."""
    
    @patch('safehive.utils.ai_client.ChatOllama')
    def test_agent_conversation_flow(self, mock_chat_model):
        """Test a complete agent conversation flow."""
        # Mock the chat model
        mock_response = Mock()
        mock_response.generations = [[Mock()]]
        mock_response.generations[0][0].text = "I can help you with that!"
        mock_chat_model.return_value.generate.return_value = mock_response
        
        client = AIClient()
        client.chat_model = mock_chat_model.return_value
        
        # Simulate a conversation
        system_prompt = "You are a helpful assistant."
        context = [
            AgentMessage(
                content="Hello, can you help me?",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="agent"
            )
        ]
        
        response = client.generate_response_sync(
            prompt="What can you do?",
            agent_id="assistant_agent",
            system_prompt=system_prompt,
            context=context
        )
        
        assert response == "I can help you with that!"
        assert client.total_requests == 1
        assert client.total_tokens > 0
        assert client.total_errors == 0
    
    def test_metrics_tracking(self):
        """Test metrics tracking across multiple requests."""
        client = AIClient()
        
        # Simulate multiple requests
        client.total_requests = 5
        client.total_tokens = 500
        client.total_errors = 1
        
        metrics = client.get_metrics()
        
        assert metrics["total_requests"] == 5
        assert metrics["total_tokens"] == 500
        assert metrics["total_errors"] == 1
        
        # Reset metrics
        client.reset_metrics()
        
        assert client.total_requests == 0
        assert client.total_tokens == 0
        assert client.total_errors == 0


if __name__ == "__main__":
    pytest.main([__file__])
