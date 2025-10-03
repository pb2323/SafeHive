"""
Agent Memory Management for LangChain

This module provides memory management utilities for AI agents using LangChain
memory components in the SafeHive system.
"""

import json
import pickle
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging

from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory,
    VectorStoreRetrieverMemory,
    CombinedMemory
)
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from safehive.utils.logger import get_logger
from safehive.models.agent_models import AgentMessage, MessageType, Conversation, AgentMemory

logger = get_logger(__name__)


class SafeHiveMemoryManager:
    """
    Memory manager for SafeHive AI agents.
    
    This class provides a unified interface for managing different types of
    LangChain memory components for AI agents.
    """
    
    def __init__(
        self,
        agent_id: str,
        memory_type: str = "buffer",
        memory_config: Optional[Dict[str, Any]] = None,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the memory manager.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory to use (buffer, window, summary, etc.)
            memory_config: Configuration for the memory type
            persist_directory: Directory to persist memory data
        """
        self.agent_id = agent_id
        self.memory_type = memory_type
        self.memory_config = memory_config or {}
        self.persist_directory = persist_directory or f"./memory/{agent_id}"
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize memory components
        self.memory = None
        self.vectorstore = None
        self.embeddings = None
        self.text_splitter = None
        
        # Initialize the memory system
        self._setup_memory()
        
        # Track memory usage
        self.memory_stats = {
            "total_messages": 0,
            "last_accessed": datetime.now(),
            "memory_size_bytes": 0
        }
    
    def _setup_memory(self) -> None:
        """Setup the memory system based on configuration."""
        try:
            if self.memory_type == "buffer":
                self.memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="chat_history",
                    **self.memory_config
                )
            elif self.memory_type == "window":
                window_size = self.memory_config.get("window_size", 10)
                self.memory = ConversationBufferWindowMemory(
                    k=window_size,
                    return_messages=True,
                    memory_key="chat_history",
                    **{k: v for k, v in self.memory_config.items() if k != "window_size"}
                )
            elif self.memory_type == "summary":
                llm = self.memory_config.get("llm")
                if not llm:
                    raise ValueError("LLM required for summary memory")
                
                self.memory = ConversationSummaryMemory(
                    llm=llm,
                    return_messages=True,
                    memory_key="chat_history",
                    **{k: v for k, v in self.memory_config.items() if k != "llm"}
                )
            elif self.memory_type == "summary_buffer":
                llm = self.memory_config.get("llm")
                max_token_limit = self.memory_config.get("max_token_limit", 2000)
                
                if not llm:
                    raise ValueError("LLM required for summary buffer memory")
                
                self.memory = ConversationSummaryBufferMemory(
                    llm=llm,
                    max_token_limit=max_token_limit,
                    return_messages=True,
                    memory_key="chat_history",
                    **{k: v for k, v in self.memory_config.items() 
                       if k not in ["llm", "max_token_limit"]}
                )
            elif self.memory_type == "token_buffer":
                llm = self.memory_config.get("llm")
                max_token_limit = self.memory_config.get("max_token_limit", 2000)
                
                if not llm:
                    raise ValueError("LLM required for token buffer memory")
                
                self.memory = ConversationTokenBufferMemory(
                    llm=llm,
                    max_token_limit=max_token_limit,
                    return_messages=True,
                    memory_key="chat_history",
                    **{k: v for k, v in self.memory_config.items() 
                       if k not in ["llm", "max_token_limit"]}
                )
            elif self.memory_type == "vector":
                self._setup_vector_memory()
            elif self.memory_type == "combined":
                self._setup_combined_memory()
            else:
                raise ValueError(f"Unsupported memory type: {self.memory_type}")
            
            logger.info(f"Memory system initialized for agent {self.agent_id} with type {self.memory_type}")
            
        except Exception as e:
            logger.error(f"Failed to setup memory for agent {self.agent_id}: {e}")
            # Fallback to buffer memory
            self.memory_type = "buffer"
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
            logger.info(f"Fallback to buffer memory for agent {self.agent_id}")
    
    def _setup_vector_memory(self) -> None:
        """Setup vector-based memory system."""
        try:
            # Initialize embeddings
            model_name = self.memory_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            
            # Initialize text splitter
            chunk_size = self.memory_config.get("chunk_size", 1000)
            chunk_overlap = self.memory_config.get("chunk_overlap", 200)
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Try to load existing vectorstore
            vectorstore_path = Path(self.persist_directory) / "vectorstore"
            if vectorstore_path.exists():
                self.vectorstore = FAISS.load_local(
                    str(vectorstore_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                # Create new vectorstore
                self.vectorstore = FAISS.from_texts(
                    ["Initial memory"],
                    self.embeddings
                )
            
            # Create retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.memory_config.get("retriever_k", 5)}
            )
            
            self.memory = VectorStoreRetrieverMemory(
                retriever=retriever,
                memory_key="chat_history",
                return_messages=True
            )
            
        except Exception as e:
            logger.error(f"Failed to setup vector memory: {e}")
            raise
    
    def _setup_combined_memory(self) -> None:
        """Setup combined memory system."""
        try:
            memories = []
            
            # Buffer memory
            buffer_memory = ConversationBufferMemory(
                memory_key="buffer_history",
                return_messages=True
            )
            memories.append(buffer_memory)
            
            # Summary memory (if LLM provided)
            if "llm" in self.memory_config:
                summary_memory = ConversationSummaryMemory(
                    llm=self.memory_config["llm"],
                    memory_key="summary_history",
                    return_messages=True
                )
                memories.append(summary_memory)
            
            self.memory = CombinedMemory(memories=memories)
            
        except Exception as e:
            logger.error(f"Failed to setup combined memory: {e}")
            raise
    
    def add_message(self, message: AgentMessage) -> None:
        """
        Add a message to the agent's memory.
        
        Args:
            message: The message to add
        """
        try:
            # Convert AgentMessage to LangChain message
            if message.message_type == MessageType.REQUEST:
                langchain_message = HumanMessage(content=message.content)
            elif message.message_type == MessageType.RESPONSE:
                langchain_message = AIMessage(content=message.content)
            elif message.message_type == MessageType.SYSTEM:
                langchain_message = SystemMessage(content=message.content)
            else:
                # For other message types, treat as human message
                langchain_message = HumanMessage(content=message.content)
            
            # Add to memory
            if hasattr(self.memory, 'chat_memory'):
                self.memory.chat_memory.add_message(langchain_message)
            elif hasattr(self.memory, 'save_context'):
                # For some memory types, we need to save context
                self.memory.save_context(
                    {"input": message.content},
                    {"output": ""}  # Empty output for now
                )
            
            # Update stats
            self.memory_stats["total_messages"] += 1
            self.memory_stats["last_accessed"] = datetime.now()
            
            # Update vectorstore if using vector memory
            if self.memory_type == "vector" and self.vectorstore:
                self._update_vectorstore(message)
            
            logger.debug(f"Added message to memory for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to add message to memory for agent {self.agent_id}: {e}")
    
    def add_conversation(self, conversation: Conversation) -> None:
        """
        Add an entire conversation to memory.
        
        Args:
            conversation: The conversation to add
        """
        try:
            for message in conversation.messages:
                self.add_message(message)
            
            logger.info(f"Added conversation {conversation.conversation_id} to memory for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to add conversation to memory for agent {self.agent_id}: {e}")
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """
        Get memory variables for use in LangChain chains.
        
        Returns:
            Dictionary of memory variables
        """
        try:
            if hasattr(self.memory, 'load_memory_variables'):
                return self.memory.load_memory_variables({})
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to get memory variables for agent {self.agent_id}: {e}")
            return {}
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[AgentMessage]:
        """
        Get conversation history as AgentMessage objects.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of AgentMessage objects
        """
        try:
            messages = []
            
            if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                langchain_messages = self.memory.chat_memory.messages
                
                for msg in langchain_messages:
                    if isinstance(msg, HumanMessage):
                        message_type = MessageType.REQUEST
                    elif isinstance(msg, AIMessage):
                        message_type = MessageType.RESPONSE
                    elif isinstance(msg, SystemMessage):
                        message_type = MessageType.SYSTEM
                    else:
                        message_type = MessageType.REQUEST
                    
                    agent_message = AgentMessage(
                        content=msg.content,
                        message_type=message_type,
                        sender=self.agent_id,
                        recipient="system",
                        timestamp=datetime.now()
                    )
                    messages.append(agent_message)
            
            if limit:
                messages = messages[-limit:]
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get conversation history for agent {self.agent_id}: {e}")
            return []
    
    def clear_memory(self) -> None:
        """Clear all memory for the agent."""
        try:
            if hasattr(self.memory, 'clear'):
                self.memory.clear()
            elif hasattr(self.memory, 'chat_memory'):
                self.memory.chat_memory.clear()
            
            # Reset stats
            self.memory_stats = {
                "total_messages": 0,
                "last_accessed": datetime.now(),
                "memory_size_bytes": 0
            }
            
            logger.info(f"Cleared memory for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to clear memory for agent {self.agent_id}: {e}")
    
    def save_memory(self) -> bool:
        """
        Save memory to persistent storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save vectorstore if using vector memory
            if self.memory_type == "vector" and self.vectorstore:
                vectorstore_path = Path(self.persist_directory) / "vectorstore"
                self.vectorstore.save_local(str(vectorstore_path))
            
            # Save memory configuration and stats
            # Convert datetime objects to ISO format strings for JSON serialization
            memory_stats_serializable = self.memory_stats.copy()
            if "last_accessed" in memory_stats_serializable:
                memory_stats_serializable["last_accessed"] = memory_stats_serializable["last_accessed"].isoformat()
            
            memory_data = {
                "agent_id": self.agent_id,
                "memory_type": self.memory_type,
                "memory_config": self.memory_config,
                "memory_stats": memory_stats_serializable,
                "timestamp": datetime.now().isoformat()
            }
            
            config_path = Path(self.persist_directory) / "memory_config.json"
            with open(config_path, 'w') as f:
                json.dump(memory_data, f, indent=2)
            
            # Save memory state if possible
            if hasattr(self.memory, 'save_context'):
                state_path = Path(self.persist_directory) / "memory_state.pkl"
                with open(state_path, 'wb') as f:
                    pickle.dump(self.memory, f)
            
            logger.info(f"Saved memory for agent {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save memory for agent {self.agent_id}: {e}")
            return False
    
    def load_memory(self) -> bool:
        """
        Load memory from persistent storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            config_path = Path(self.persist_directory) / "memory_config.json"
            if not config_path.exists():
                logger.info(f"No saved memory found for agent {self.agent_id}")
                return False
            
            with open(config_path, 'r') as f:
                memory_data = json.load(f)
            
            # Update stats and convert datetime strings back to datetime objects
            loaded_stats = memory_data.get("memory_stats", self.memory_stats)
            if "last_accessed" in loaded_stats and isinstance(loaded_stats["last_accessed"], str):
                loaded_stats["last_accessed"] = datetime.fromisoformat(loaded_stats["last_accessed"])
            self.memory_stats = loaded_stats
            
            # Load memory state if available
            state_path = Path(self.persist_directory) / "memory_state.pkl"
            if state_path.exists():
                with open(state_path, 'rb') as f:
                    self.memory = pickle.load(f)
            
            logger.info(f"Loaded memory for agent {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load memory for agent {self.agent_id}: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        stats = self.memory_stats.copy()
        stats.update({
            "memory_type": self.memory_type,
            "persist_directory": self.persist_directory,
            "memory_available": self.memory is not None
        })
        
        # Add memory-specific stats
        if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
            stats["current_messages"] = len(self.memory.chat_memory.messages)
        
        return stats
    
    def _update_vectorstore(self, message: AgentMessage) -> None:
        """Update vectorstore with new message."""
        try:
            if not self.vectorstore or not self.text_splitter:
                return
            
            # Split message content into chunks
            chunks = self.text_splitter.split_text(message.content)
            
            # Add chunks to vectorstore
            for chunk in chunks:
                self.vectorstore.add_texts([chunk])
            
            # Save updated vectorstore
            vectorstore_path = Path(self.persist_directory) / "vectorstore"
            self.vectorstore.save_local(str(vectorstore_path))
            
        except Exception as e:
            logger.error(f"Failed to update vectorstore for agent {self.agent_id}: {e}")


class MemoryManagerFactory:
    """
    Factory class for creating memory managers.
    """
    
    @staticmethod
    def create_memory_manager(
        agent_id: str,
        memory_type: str = "buffer",
        memory_config: Optional[Dict[str, Any]] = None,
        persist_directory: Optional[str] = None
    ) -> SafeHiveMemoryManager:
        """
        Create a memory manager for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_type: Type of memory to use
            memory_config: Configuration for the memory type
            persist_directory: Directory to persist memory data
            
        Returns:
            Configured memory manager instance
        """
        return SafeHiveMemoryManager(
            agent_id=agent_id,
            memory_type=memory_type,
            memory_config=memory_config,
            persist_directory=persist_directory
        )
    
    @staticmethod
    def create_buffer_memory(
        agent_id: str,
        persist_directory: Optional[str] = None
    ) -> SafeHiveMemoryManager:
        """Create a buffer memory manager."""
        return MemoryManagerFactory.create_memory_manager(
            agent_id=agent_id,
            memory_type="buffer",
            persist_directory=persist_directory
        )
    
    @staticmethod
    def create_window_memory(
        agent_id: str,
        window_size: int = 10,
        persist_directory: Optional[str] = None
    ) -> SafeHiveMemoryManager:
        """Create a window memory manager."""
        return MemoryManagerFactory.create_memory_manager(
            agent_id=agent_id,
            memory_type="window",
            memory_config={"window_size": window_size},
            persist_directory=persist_directory
        )
    
    @staticmethod
    def create_summary_memory(
        agent_id: str,
        llm: Any,
        persist_directory: Optional[str] = None
    ) -> SafeHiveMemoryManager:
        """Create a summary memory manager."""
        return MemoryManagerFactory.create_memory_manager(
            agent_id=agent_id,
            memory_type="summary",
            memory_config={"llm": llm},
            persist_directory=persist_directory
        )
    
    @staticmethod
    def create_vector_memory(
        agent_id: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        retriever_k: int = 5,
        persist_directory: Optional[str] = None
    ) -> SafeHiveMemoryManager:
        """Create a vector memory manager."""
        return MemoryManagerFactory.create_memory_manager(
            agent_id=agent_id,
            memory_type="vector",
            memory_config={
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "retriever_k": retriever_k
            },
            persist_directory=persist_directory
        )


# Global memory manager registry
_memory_managers: Dict[str, SafeHiveMemoryManager] = {}


def get_memory_manager(agent_id: str) -> Optional[SafeHiveMemoryManager]:
    """
    Get a memory manager for an agent.
    
    Args:
        agent_id: Unique identifier for the agent
        
    Returns:
        Memory manager instance or None if not found
    """
    return _memory_managers.get(agent_id)


def create_memory_manager(
    agent_id: str,
    memory_type: str = "buffer",
    memory_config: Optional[Dict[str, Any]] = None,
    persist_directory: Optional[str] = None
) -> SafeHiveMemoryManager:
    """
    Create and register a memory manager for an agent.
    
    Args:
        agent_id: Unique identifier for the agent
        memory_type: Type of memory to use
        memory_config: Configuration for the memory type
        persist_directory: Directory to persist memory data
        
    Returns:
        Created memory manager instance
    """
    memory_manager = MemoryManagerFactory.create_memory_manager(
        agent_id=agent_id,
        memory_type=memory_type,
        memory_config=memory_config,
        persist_directory=persist_directory
    )
    
    _memory_managers[agent_id] = memory_manager
    return memory_manager


def remove_memory_manager(agent_id: str) -> bool:
    """
    Remove a memory manager for an agent.
    
    Args:
        agent_id: Unique identifier for the agent
        
    Returns:
        True if removed, False if not found
    """
    if agent_id in _memory_managers:
        del _memory_managers[agent_id]
        return True
    return False


def list_memory_managers() -> List[str]:
    """
    List all registered memory managers.
    
    Returns:
        List of agent IDs with memory managers
    """
    return list(_memory_managers.keys())


def save_all_memories() -> Dict[str, bool]:
    """
    Save all registered memory managers.
    
    Returns:
        Dictionary mapping agent IDs to save success status
    """
    results = {}
    for agent_id, memory_manager in _memory_managers.items():
        results[agent_id] = memory_manager.save_memory()
    return results


def load_all_memories() -> Dict[str, bool]:
    """
    Load all registered memory managers.
    
    Returns:
        Dictionary mapping agent IDs to load success status
    """
    results = {}
    for agent_id, memory_manager in _memory_managers.items():
        results[agent_id] = memory_manager.load_memory()
    return results
