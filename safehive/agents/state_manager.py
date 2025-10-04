"""
Agent State Persistence and Recovery System

This module provides comprehensive state management for AI agents including
persistence, recovery, versioning, and backup/restore functionality.
"""

import asyncio
import json
import os
import pickle
import shutil
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
import threading
import time
import gzip
import hashlib

from ..utils.logger import get_logger
from ..utils.metrics import record_metric, increment_counter, MetricType
from ..models.agent_models import AgentState, AgentType, AgentStatus, AgentMessage, MessageType
from ..agents.base_agent import BaseAgent

logger = get_logger(__name__)

T = TypeVar('T')


class PersistenceFormat(Enum):
    """Supported persistence formats."""
    JSON = "json"
    PICKLE = "pickle"
    COMPRESSED_JSON = "compressed_json"
    COMPRESSED_PICKLE = "compressed_pickle"


class StateVersion(Enum):
    """State version for migration support."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V1_2 = "1.2"
    CURRENT = V1_2


@dataclass
class AgentStateSnapshot:
    """
    Complete snapshot of an agent's state for persistence.
    
    This includes all necessary data to restore an agent to its
    previous state including configuration, memory, and metrics.
    """
    # Agent identification
    agent_id: str
    agent_type: AgentType
    agent_name: str
    agent_description: str
    
    # Current state
    current_state: AgentState
    is_enabled: bool
    
    # Configuration and capabilities
    configuration: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    
    # Memory and conversation data
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    memory_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics and performance data
    metrics: Dict[str, Any] = field(default_factory=dict)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    snapshot_timestamp: datetime = field(default_factory=datetime.now)
    
    # State version and metadata
    state_version: str = StateVersion.CURRENT.value
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key in ['created_at', 'last_activity', 'snapshot_timestamp']:
            if key in data and data[key]:
                data[key] = data[key].isoformat()
        # Convert enums to values
        data['agent_type'] = self.agent_type.value
        data['current_state'] = self.current_state.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentStateSnapshot":
        """Create snapshot from dictionary."""
        # Convert ISO strings back to datetime objects
        for key in ['created_at', 'last_activity', 'snapshot_timestamp']:
            if key in data and data[key]:
                data[key] = datetime.fromisoformat(data[key])
        # Convert values back to enums
        data['agent_type'] = AgentType(data['agent_type'])
        data['current_state'] = AgentState(data['current_state'])
        return cls(**data)
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for state integrity verification."""
        data = self.to_dict().copy()
        # Exclude timestamp fields and checksum from checksum calculation
        exclude_fields = ['created_at', 'last_activity', 'snapshot_timestamp', 'checksum']
        for field in exclude_fields:
            data.pop(field, None)
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_checksum(self) -> bool:
        """Verify the state checksum."""
        if not self.checksum:
            return False
        return self.checksum == self.calculate_checksum()


@dataclass
class StateMetadata:
    """Metadata for state persistence operations."""
    agent_id: str
    timestamp: datetime
    format: PersistenceFormat
    size_bytes: int
    version: str
    checksum: str
    backup_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "format": self.format.value,
            "size_bytes": self.size_bytes,
            "version": self.version,
            "checksum": self.checksum,
            "backup_id": self.backup_id
        }


class StateSerializer(ABC):
    """Abstract base class for state serializers."""
    
    @abstractmethod
    def serialize(self, snapshot: AgentStateSnapshot) -> bytes:
        """Serialize agent state snapshot to bytes."""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> AgentStateSnapshot:
        """Deserialize bytes to agent state snapshot."""
        pass


class JSONSerializer(StateSerializer):
    """JSON-based state serializer."""
    
    def serialize(self, snapshot: AgentStateSnapshot) -> bytes:
        """Serialize to JSON bytes."""
        data = snapshot.to_dict()
        # Calculate and set checksum
        snapshot.checksum = snapshot.calculate_checksum()
        data['checksum'] = snapshot.checksum
        
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8')
    
    def deserialize(self, data: bytes) -> AgentStateSnapshot:
        """Deserialize from JSON bytes."""
        json_str = data.decode('utf-8')
        data_dict = json.loads(json_str)
        
        snapshot = AgentStateSnapshot.from_dict(data_dict)
        
        # Verify checksum
        if not snapshot.verify_checksum():
            logger.warning(f"Checksum verification failed for agent {snapshot.agent_id}")
        
        return snapshot


class PickleSerializer(StateSerializer):
    """Pickle-based state serializer."""
    
    def serialize(self, snapshot: AgentStateSnapshot) -> bytes:
        """Serialize to pickle bytes."""
        # Calculate and set checksum
        snapshot.checksum = snapshot.calculate_checksum()
        return pickle.dumps(snapshot)
    
    def deserialize(self, data: bytes) -> AgentStateSnapshot:
        """Deserialize from pickle bytes."""
        snapshot = pickle.loads(data)
        
        # Verify checksum
        if not snapshot.verify_checksum():
            logger.warning(f"Checksum verification failed for agent {snapshot.agent_id}")
        
        return snapshot


class CompressedJSONSerializer(StateSerializer):
    """Compressed JSON-based state serializer."""
    
    def serialize(self, snapshot: AgentStateSnapshot) -> bytes:
        """Serialize to compressed JSON bytes."""
        json_serializer = JSONSerializer()
        json_data = json_serializer.serialize(snapshot)
        return gzip.compress(json_data)
    
    def deserialize(self, data: bytes) -> AgentStateSnapshot:
        """Deserialize from compressed JSON bytes."""
        json_data = gzip.decompress(data)
        json_serializer = JSONSerializer()
        return json_serializer.deserialize(json_data)


class CompressedPickleSerializer(StateSerializer):
    """Compressed pickle-based state serializer."""
    
    def serialize(self, snapshot: AgentStateSnapshot) -> bytes:
        """Serialize to compressed pickle bytes."""
        pickle_serializer = PickleSerializer()
        pickle_data = pickle_serializer.serialize(snapshot)
        return gzip.compress(pickle_data)
    
    def deserialize(self, data: bytes) -> AgentStateSnapshot:
        """Deserialize from compressed pickle bytes."""
        pickle_data = gzip.decompress(data)
        pickle_serializer = PickleSerializer()
        return pickle_serializer.deserialize(pickle_data)


class StateStorage(ABC):
    """Abstract base class for state storage backends."""
    
    @abstractmethod
    async def save_state(self, snapshot: AgentStateSnapshot, metadata: StateMetadata) -> bool:
        """Save agent state to storage."""
        pass
    
    @abstractmethod
    async def load_state(self, agent_id: str, version: Optional[str] = None) -> Optional[AgentStateSnapshot]:
        """Load agent state from storage."""
        pass
    
    @abstractmethod
    async def list_states(self, agent_id: Optional[str] = None) -> List[StateMetadata]:
        """List available states."""
        pass
    
    @abstractmethod
    async def delete_state(self, agent_id: str, version: Optional[str] = None) -> bool:
        """Delete agent state from storage."""
        pass


class FileSystemStorage(StateStorage):
    """File system-based state storage."""
    
    def __init__(self, base_path: str, format: PersistenceFormat = PersistenceFormat.JSON):
        self.base_path = Path(base_path)
        self.format = format
        self._serializers = {
            PersistenceFormat.JSON: JSONSerializer(),
            PersistenceFormat.PICKLE: PickleSerializer(),
            PersistenceFormat.COMPRESSED_JSON: CompressedJSONSerializer(),
            PersistenceFormat.COMPRESSED_PICKLE: CompressedPickleSerializer()
        }
        self._lock = threading.RLock()
        
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FileSystemStorage initialized at {self.base_path}")
    
    def _get_agent_path(self, agent_id: str) -> Path:
        """Get the path for an agent's state directory."""
        return self.base_path / agent_id
    
    def _get_state_file_path(self, agent_id: str, version: Optional[str] = None) -> Path:
        """Get the file path for agent state."""
        agent_path = self._get_agent_path(agent_id)
        if version:
            filename = f"state_{version}.{self._get_file_extension()}"
        else:
            filename = f"state_latest.{self._get_file_extension()}"
        return agent_path / filename
    
    def _get_metadata_file_path(self, agent_id: str, version: Optional[str] = None) -> Path:
        """Get the metadata file path."""
        agent_path = self._get_agent_path(agent_id)
        if version:
            filename = f"metadata_{version}.json"
        else:
            filename = "metadata_latest.json"
        return agent_path / filename
    
    def _get_file_extension(self) -> str:
        """Get file extension based on format."""
        if self.format in [PersistenceFormat.JSON, PersistenceFormat.COMPRESSED_JSON]:
            return "json" if self.format == PersistenceFormat.JSON else "json.gz"
        else:
            return "pkl" if self.format == PersistenceFormat.PICKLE else "pkl.gz"
    
    async def save_state(self, snapshot: AgentStateSnapshot, metadata: StateMetadata) -> bool:
        """Save agent state to file system."""
        try:
            with self._lock:
                agent_path = self._get_agent_path(snapshot.agent_id)
                agent_path.mkdir(parents=True, exist_ok=True)
                
                # Serialize state
                serializer = self._serializers[self.format]
                state_data = serializer.serialize(snapshot)
                
                # Save state file
                state_file = self._get_state_file_path(snapshot.agent_id, metadata.version)
                with open(state_file, 'wb') as f:
                    f.write(state_data)
                
                # Save metadata file
                metadata_file = self._get_metadata_file_path(snapshot.agent_id, metadata.version)
                with open(metadata_file, 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
                
                # Update latest symlink
                latest_state_file = self._get_state_file_path(snapshot.agent_id)
                latest_metadata_file = self._get_metadata_file_path(snapshot.agent_id)
                
                if latest_state_file.exists():
                    latest_state_file.unlink()
                if latest_metadata_file.exists():
                    latest_metadata_file.unlink()
                
                latest_state_file.symlink_to(state_file.name)
                latest_metadata_file.symlink_to(metadata_file.name)
                
                logger.info(f"Saved state for agent {snapshot.agent_id} to {state_file}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save state for agent {snapshot.agent_id}: {e}")
            return False
    
    async def load_state(self, agent_id: str, version: Optional[str] = None) -> Optional[AgentStateSnapshot]:
        """Load agent state from file system."""
        try:
            with self._lock:
                state_file = self._get_state_file_path(agent_id, version)
                
                if not state_file.exists():
                    logger.warning(f"State file not found for agent {agent_id}")
                    return None
                
                # Load state data
                with open(state_file, 'rb') as f:
                    state_data = f.read()
                
                # Deserialize state
                serializer = self._serializers[self.format]
                snapshot = serializer.deserialize(state_data)
                
                logger.info(f"Loaded state for agent {snapshot.agent_id}")
                return snapshot
                
        except Exception as e:
            logger.error(f"Failed to load state for agent {agent_id}: {e}")
            return None
    
    async def list_states(self, agent_id: Optional[str] = None) -> List[StateMetadata]:
        """List available states."""
        try:
            with self._lock:
                metadata_list = []
                
                if agent_id:
                    # List states for specific agent
                    agent_path = self._get_agent_path(agent_id)
                    if agent_path.exists():
                        # Only look for version-specific metadata files, not latest
                        for metadata_file in agent_path.glob("metadata_*.json"):
                            if metadata_file.name != "metadata_latest.json":
                                try:
                                    with open(metadata_file, 'r') as f:
                                        metadata_dict = json.load(f)
                                    metadata = StateMetadata(
                                        agent_id=metadata_dict["agent_id"],
                                        timestamp=datetime.fromisoformat(metadata_dict["timestamp"]),
                                        format=PersistenceFormat(metadata_dict["format"]),
                                        size_bytes=metadata_dict["size_bytes"],
                                        version=metadata_dict["version"],
                                        checksum=metadata_dict["checksum"],
                                        backup_id=metadata_dict.get("backup_id")
                                    )
                                    metadata_list.append(metadata)
                                except Exception as e:
                                    logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
                else:
                    # List states for all agents
                    for agent_dir in self.base_path.iterdir():
                        if agent_dir.is_dir():
                            agent_states = await self.list_states(agent_dir.name)
                            metadata_list.extend(agent_states)
                
                return metadata_list
                
        except Exception as e:
            logger.error(f"Failed to list states: {e}")
            return []
    
    async def delete_state(self, agent_id: str, version: Optional[str] = None) -> bool:
        """Delete agent state from file system."""
        try:
            with self._lock:
                if version:
                    # Delete specific version
                    state_file = self._get_state_file_path(agent_id, version)
                    metadata_file = self._get_metadata_file_path(agent_id, version)
                else:
                    # Delete all versions for agent
                    agent_path = self._get_agent_path(agent_id)
                    if agent_path.exists():
                        shutil.rmtree(agent_path)
                        logger.info(f"Deleted all states for agent {agent_id}")
                        return True
                    return False
                
                if state_file.exists():
                    state_file.unlink()
                if metadata_file.exists():
                    metadata_file.unlink()
                
                logger.info(f"Deleted state for agent {agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete state for agent {agent_id}: {e}")
            return False


class StateManager:
    """
    Central manager for agent state persistence and recovery.
    
    This class provides a unified interface for saving, loading, and managing
    agent states with support for versioning, backup, and recovery.
    """
    
    def __init__(self, storage: StateStorage, auto_save_interval: int = 300):
        self.storage = storage
        self.auto_save_interval = auto_save_interval  # seconds
        self._agents: Dict[str, BaseAgent] = {}
        self._auto_save_tasks: Dict[str, asyncio.Task] = {}
        self._lock = threading.RLock()
        self._running = False
        
        logger.info("StateManager initialized")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for state management."""
        with self._lock:
            self._agents[agent.agent_id] = agent
            logger.info(f"Registered agent {agent.agent_id} for state management")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from state management."""
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
            
            # Cancel auto-save task if running
            if agent_id in self._auto_save_tasks:
                self._auto_save_tasks[agent_id].cancel()
                del self._auto_save_tasks[agent_id]
            
            logger.info(f"Unregistered agent {agent_id} from state management")
    
    async def save_agent_state(self, agent: BaseAgent, version: Optional[str] = None) -> bool:
        """Save the current state of an agent."""
        try:
            # Create state snapshot
            snapshot = await self._create_state_snapshot(agent)
            
            # Create metadata
            version = version or StateVersion.CURRENT.value
            metadata = StateMetadata(
                agent_id=agent.agent_id,
                timestamp=datetime.now(),
                format=PersistenceFormat.JSON,  # Default format
                size_bytes=0,  # Will be calculated
                version=version,
                checksum=snapshot.calculate_checksum()
            )
            
            # Save to storage
            success = await self.storage.save_state(snapshot, metadata)
            
            if success:
                record_metric("state.save.success", 1, MetricType.COUNTER, {"agent_id": agent.agent_id, "version": version})
                logger.info(f"Successfully saved state for agent {agent.agent_id}")
            else:
                record_metric("state.save.failure", 1, MetricType.COUNTER, {"agent_id": agent.agent_id, "version": version})
                logger.error(f"Failed to save state for agent {agent.agent_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving state for agent {agent.agent_id}: {e}")
            record_metric("state.save.error", 1, MetricType.COUNTER, {"agent_id": agent.agent_id, "error": str(e)})
            return False
    
    async def load_agent_state(self, agent_id: str, version: Optional[str] = None) -> Optional[AgentStateSnapshot]:
        """Load agent state from storage."""
        try:
            snapshot = await self.storage.load_state(agent_id, version)
            
            if snapshot:
                record_metric("state.load.success", 1, MetricType.COUNTER, {"agent_id": agent_id, "version": version or "latest"})
                logger.info(f"Successfully loaded state for agent {agent_id}")
            else:
                record_metric("state.load.not_found", 1, MetricType.COUNTER, {"agent_id": agent_id, "version": version or "latest"})
                logger.warning(f"State not found for agent {agent_id}")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error loading state for agent {agent_id}: {e}")
            record_metric("state.load.error", 1, MetricType.COUNTER, {"agent_id": agent_id, "error": str(e)})
            return None
    
    async def restore_agent(self, agent: BaseAgent, snapshot: AgentStateSnapshot) -> bool:
        """Restore an agent to a previous state."""
        try:
            # Restore agent configuration
            agent.name = snapshot.agent_name
            agent.description = snapshot.agent_description
            
            # Restore agent state
            agent._status = snapshot.current_state
            
            # Restore metrics
            if snapshot.metrics:
                agent._total_requests = snapshot.metrics.get("total_requests", 0)
                agent._successful_requests = snapshot.metrics.get("successful_requests", 0)
                agent._failed_requests = snapshot.metrics.get("failed_requests", 0)
                agent._average_response_time = snapshot.metrics.get("average_response_time", 0.0)
            
            # Restore conversation history
            if snapshot.conversation_history:
                agent._conversation_history = [
                    AgentMessage.from_dict(msg_data) for msg_data in snapshot.conversation_history
                ]
            
            # Restore memory if available
            if snapshot.memory_data and agent._memory_manager:
                try:
                    # Save memory data to temporary file and load it
                    temp_file = f"/tmp/memory_restore_{agent.agent_id}.json"
                    with open(temp_file, 'w') as f:
                        json.dump(snapshot.memory_data, f)
                    await agent.load_memory(temp_file)
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to restore memory for agent {agent.agent_id}: {e}")
            
            record_metric("state.restore.success", 1, MetricType.COUNTER, {"agent_id": agent.agent_id})
            logger.info(f"Successfully restored agent {agent.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring agent {agent.agent_id}: {e}")
            record_metric("state.restore.error", 1, MetricType.COUNTER, {"agent_id": agent.agent_id, "error": str(e)})
            return False
    
    async def create_backup(self, agent_id: str, backup_name: Optional[str] = None) -> Optional[str]:
        """Create a backup of agent state."""
        try:
            if agent_id not in self._agents:
                logger.error(f"Agent {agent_id} not registered")
                return None
            
            agent = self._agents[agent_id]
            backup_id = backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save state with backup version
            success = await self.save_agent_state(agent, backup_id)
            
            if success:
                record_metric("state.backup.success", 1, MetricType.COUNTER, {"agent_id": agent_id, "backup_id": backup_id})
                logger.info(f"Created backup {backup_id} for agent {agent_id}")
                return backup_id
            else:
                record_metric("state.backup.failure", 1, MetricType.COUNTER, {"agent_id": agent_id, "backup_id": backup_id})
                logger.error(f"Failed to create backup {backup_id} for agent {agent_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating backup for agent {agent_id}: {e}")
            record_metric("state.backup.error", 1, MetricType.COUNTER, {"agent_id": agent_id, "error": str(e)})
            return None
    
    async def restore_from_backup(self, agent_id: str, backup_id: str) -> bool:
        """Restore agent from a backup."""
        try:
            if agent_id not in self._agents:
                logger.error(f"Agent {agent_id} not registered")
                return False
            
            agent = self._agents[agent_id]
            
            # Load backup state
            snapshot = await self.load_agent_state(agent_id, backup_id)
            if not snapshot:
                logger.error(f"Backup {backup_id} not found for agent {agent_id}")
                return False
            
            # Restore agent
            success = await self.restore_agent(agent, snapshot)
            
            if success:
                record_metric("state.restore_backup.success", 1, MetricType.COUNTER, {"agent_id": agent_id, "backup_id": backup_id})
                logger.info(f"Successfully restored agent {agent_id} from backup {backup_id}")
            else:
                record_metric("state.restore_backup.failure", 1, MetricType.COUNTER, {"agent_id": agent_id, "backup_id": backup_id})
                logger.error(f"Failed to restore agent {agent_id} from backup {backup_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error restoring agent {agent_id} from backup {backup_id}: {e}")
            record_metric("state.restore_backup.error", 1, MetricType.COUNTER, {"agent_id": agent_id, "error": str(e)})
            return False
    
    async def list_agent_states(self, agent_id: Optional[str] = None) -> List[StateMetadata]:
        """List available states for agents."""
        return await self.storage.list_states(agent_id)
    
    async def delete_agent_state(self, agent_id: str, version: Optional[str] = None) -> bool:
        """Delete agent state from storage."""
        try:
            success = await self.storage.delete_state(agent_id, version)
            
            if success:
                record_metric("state.delete.success", 1, MetricType.COUNTER, {"agent_id": agent_id, "version": version or "all"})
                logger.info(f"Deleted state for agent {agent_id}")
            else:
                record_metric("state.delete.failure", 1, MetricType.COUNTER, {"agent_id": agent_id, "version": version or "all"})
                logger.error(f"Failed to delete state for agent {agent_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting state for agent {agent_id}: {e}")
            record_metric("state.delete.error", 1, MetricType.COUNTER, {"agent_id": agent_id, "error": str(e)})
            return False
    
    async def start_auto_save(self, agent_id: str) -> None:
        """Start automatic state saving for an agent."""
        if agent_id not in self._agents:
            logger.error(f"Agent {agent_id} not registered for auto-save")
            return
        
        if agent_id in self._auto_save_tasks:
            logger.warning(f"Auto-save already running for agent {agent_id}")
            return
        
        async def auto_save_worker():
            while True:
                try:
                    await asyncio.sleep(self.auto_save_interval)
                    agent = self._agents.get(agent_id)
                    if agent:
                        await self.save_agent_state(agent)
                    else:
                        break
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in auto-save for agent {agent_id}: {e}")
        
        self._auto_save_tasks[agent_id] = asyncio.create_task(auto_save_worker())
        logger.info(f"Started auto-save for agent {agent_id}")
    
    async def stop_auto_save(self, agent_id: str) -> None:
        """Stop automatic state saving for an agent."""
        if agent_id in self._auto_save_tasks:
            self._auto_save_tasks[agent_id].cancel()
            del self._auto_save_tasks[agent_id]
            logger.info(f"Stopped auto-save for agent {agent_id}")
    
    async def _create_state_snapshot(self, agent: BaseAgent) -> AgentStateSnapshot:
        """Create a state snapshot from an agent."""
        # Get conversation history
        conversation_history = [msg.to_dict() for msg in agent._conversation_history]
        
        # Get memory data if available
        memory_data = {}
        if agent._memory_manager:
            try:
                # Get memory stats and recent conversations
                memory_data = {
                    "memory_type": getattr(agent.config, 'memory_type', None),
                    "conversation_history": agent._memory_manager.get_conversation_history()
                }
            except Exception as e:
                logger.warning(f"Failed to get memory data for agent {agent.agent_id}: {e}")
        
        # Get performance metrics
        performance_stats = agent.get_metrics()
        
        return AgentStateSnapshot(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            agent_name=agent.name,
            agent_description=agent.description,
            current_state=agent._status,
            is_enabled=True,  # Assume enabled if agent is running
            configuration=agent.config.__dict__ if hasattr(agent.config, '__dict__') else {},
            capabilities=[cap.value if hasattr(cap, 'value') else str(cap) for cap in agent.config.capabilities] if hasattr(agent.config, 'capabilities') and agent.config.capabilities else [],
            conversation_history=conversation_history,
            memory_data=memory_data,
            metrics=performance_stats,
            performance_stats=performance_stats,
            last_activity=agent._last_activity
        )
    
    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent IDs."""
        with self._lock:
            return list(self._agents.keys())
    
    def is_agent_registered(self, agent_id: str) -> bool:
        """Check if an agent is registered."""
        with self._lock:
            return agent_id in self._agents


# Global state manager instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        # Create default file system storage
        storage = FileSystemStorage("/tmp/safehive_states")
        _state_manager = StateManager(storage)
    return _state_manager


def create_state_manager(
    storage_backend: str = "filesystem",
    storage_config: Optional[Dict[str, Any]] = None
) -> StateManager:
    """Create a new state manager with specified configuration."""
    if storage_backend == "filesystem":
        base_path = storage_config.get("base_path", "/tmp/safehive_states") if storage_config else "/tmp/safehive_states"
        format_str = storage_config.get("format", "json") if storage_config else "json"
        format_enum = PersistenceFormat(format_str)
        storage = FileSystemStorage(base_path, format_enum)
    else:
        raise ValueError(f"Unsupported storage backend: {storage_backend}")
    
    return StateManager(storage)
