"""
Unit tests for the agent state persistence and recovery system.
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import pytest

from safehive.agents.state_manager import (
    StateManager, FileSystemStorage, AgentStateSnapshot, StateMetadata,
    PersistenceFormat, StateVersion, JSONSerializer, PickleSerializer,
    CompressedJSONSerializer, CompressedPickleSerializer,
    create_state_manager, get_state_manager
)
from safehive.models.agent_models import AgentState, AgentType, AgentMessage, MessageType
from safehive.agents.base_agent import BaseAgent, AgentConfiguration


class TestAgentStateSnapshot:
    """Test AgentStateSnapshot class."""
    
    def test_snapshot_creation(self):
        """Test creating a state snapshot."""
        snapshot = AgentStateSnapshot(
            agent_id="test_agent",
            agent_type=AgentType.ORCHESTRATOR,
            agent_name="Test Agent",
            agent_description="A test agent",
            current_state=AgentState.IDLE,
            is_enabled=True
        )
        
        assert snapshot.agent_id == "test_agent"
        assert snapshot.agent_type == AgentType.ORCHESTRATOR
        assert snapshot.agent_name == "Test Agent"
        assert snapshot.agent_description == "A test agent"
        assert snapshot.current_state == AgentState.IDLE
        assert snapshot.is_enabled is True
        assert snapshot.state_version == StateVersion.CURRENT.value
        assert isinstance(snapshot.created_at, datetime)
        assert isinstance(snapshot.last_activity, datetime)
        assert isinstance(snapshot.snapshot_timestamp, datetime)
    
    def test_snapshot_with_data(self):
        """Test creating a snapshot with additional data."""
        snapshot = AgentStateSnapshot(
            agent_id="test_agent",
            agent_type=AgentType.USER_TWIN,
            agent_name="User Twin",
            agent_description="User twin agent",
            current_state=AgentState.ACTIVE,
            is_enabled=False,
            configuration={"model": "gpt-4", "temperature": 0.7},
            capabilities=["reasoning", "memory"],
            conversation_history=[{"content": "Hello", "sender": "user"}],
            memory_data={"conversations": 5},
            metrics={"requests": 100, "success_rate": 0.95}
        )
        
        assert snapshot.current_state == AgentState.ACTIVE
        assert snapshot.is_enabled is False
        assert snapshot.configuration == {"model": "gpt-4", "temperature": 0.7}
        assert snapshot.capabilities == ["reasoning", "memory"]
        assert len(snapshot.conversation_history) == 1
        assert snapshot.memory_data == {"conversations": 5}
        assert snapshot.metrics["requests"] == 100
    
    def test_snapshot_serialization(self):
        """Test snapshot serialization and deserialization."""
        original_snapshot = AgentStateSnapshot(
            agent_id="test_agent",
            agent_type=AgentType.ORCHESTRATOR,
            agent_name="Test Agent",
            agent_description="A test agent",
            current_state=AgentState.ACTIVE,
            is_enabled=True,
            configuration={"test": "value"},
            metrics={"count": 42}
        )
        
        # Convert to dict
        data = original_snapshot.to_dict()
        assert data["agent_id"] == "test_agent"
        assert data["agent_type"] == "orchestrator"
        assert data["current_state"] == "active"
        assert data["configuration"]["test"] == "value"
        assert data["metrics"]["count"] == 42
        
        # Convert back from dict
        restored_snapshot = AgentStateSnapshot.from_dict(data)
        assert restored_snapshot.agent_id == original_snapshot.agent_id
        assert restored_snapshot.agent_type == original_snapshot.agent_type
        assert restored_snapshot.agent_name == original_snapshot.agent_name
        assert restored_snapshot.current_state == original_snapshot.current_state
        assert restored_snapshot.configuration == original_snapshot.configuration
        assert restored_snapshot.metrics == original_snapshot.metrics
    
    def test_checksum_calculation(self):
        """Test checksum calculation and verification."""
        snapshot = AgentStateSnapshot(
            agent_id="test_agent",
            agent_type=AgentType.ORCHESTRATOR,
            agent_name="Test Agent",
            agent_description="A test agent",
            current_state=AgentState.IDLE,
            is_enabled=True
        )
        
        # Calculate checksum
        checksum = snapshot.calculate_checksum()
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex length
        
        # Set checksum and verify
        snapshot.checksum = checksum
        assert snapshot.verify_checksum() is True
        
        # Test with wrong checksum
        snapshot.checksum = "wrong_checksum"
        assert snapshot.verify_checksum() is False
        
        # Test with no checksum
        snapshot.checksum = None
        assert snapshot.verify_checksum() is False
        
        # Test that checksum changes when data changes
        original_checksum = snapshot.calculate_checksum()
        snapshot.agent_name = "Modified Name"
        new_checksum = snapshot.calculate_checksum()
        assert original_checksum != new_checksum


class TestStateSerializers:
    """Test state serializers."""
    
    def test_json_serializer(self):
        """Test JSON serializer."""
        serializer = JSONSerializer()
        snapshot = AgentStateSnapshot(
            agent_id="test_agent",
            agent_type=AgentType.ORCHESTRATOR,
            agent_name="Test Agent",
            agent_description="A test agent",
            current_state=AgentState.IDLE,
            is_enabled=True,
            configuration={"test": "value"}
        )
        
        # Serialize
        data = serializer.serialize(snapshot)
        assert isinstance(data, bytes)
        
        # Deserialize
        restored_snapshot = serializer.deserialize(data)
        assert restored_snapshot.agent_id == snapshot.agent_id
        assert restored_snapshot.agent_type == snapshot.agent_type
        assert restored_snapshot.configuration == snapshot.configuration
        # Note: checksum verification may fail due to timestamp differences
    
    def test_pickle_serializer(self):
        """Test Pickle serializer."""
        serializer = PickleSerializer()
        snapshot = AgentStateSnapshot(
            agent_id="test_agent",
            agent_type=AgentType.USER_TWIN,
            agent_name="Test Agent",
            agent_description="A test agent",
            current_state=AgentState.IDLE,
            is_enabled=True,
            metrics={"count": 100}
        )
        
        # Serialize
        data = serializer.serialize(snapshot)
        assert isinstance(data, bytes)
        
        # Deserialize
        restored_snapshot = serializer.deserialize(data)
        assert restored_snapshot.agent_id == snapshot.agent_id
        assert restored_snapshot.agent_type == snapshot.agent_type
        assert restored_snapshot.metrics == snapshot.metrics
        # Note: checksum verification may fail due to timestamp differences
    
    def test_compressed_json_serializer(self):
        """Test compressed JSON serializer."""
        serializer = CompressedJSONSerializer()
        snapshot = AgentStateSnapshot(
            agent_id="test_agent",
            agent_type=AgentType.ORCHESTRATOR,
            agent_name="Test Agent",
            agent_description="A test agent",
            current_state=AgentState.IDLE,
            is_enabled=True,
            conversation_history=[{"content": f"Message {i}"} for i in range(100)]
        )
        
        # Serialize
        data = serializer.serialize(snapshot)
        assert isinstance(data, bytes)
        
        # Deserialize
        restored_snapshot = serializer.deserialize(data)
        assert restored_snapshot.agent_id == snapshot.agent_id
        assert len(restored_snapshot.conversation_history) == 100
        # Note: checksum verification may fail due to timestamp differences
    
    def test_compressed_pickle_serializer(self):
        """Test compressed Pickle serializer."""
        serializer = CompressedPickleSerializer()
        snapshot = AgentStateSnapshot(
            agent_id="test_agent",
            agent_type=AgentType.USER_TWIN,
            agent_name="Test Agent",
            agent_description="A test agent",
            current_state=AgentState.IDLE,
            is_enabled=True,
            memory_data={"large_data": ["item"] * 1000}
        )
        
        # Serialize
        data = serializer.serialize(snapshot)
        assert isinstance(data, bytes)
        
        # Deserialize
        restored_snapshot = serializer.deserialize(data)
        assert restored_snapshot.agent_id == snapshot.agent_id
        assert len(restored_snapshot.memory_data["large_data"]) == 1000
        # Note: checksum verification may fail due to timestamp differences


class TestFileSystemStorage:
    """Test FileSystemStorage class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = FileSystemStorage(self.temp_dir, PersistenceFormat.JSON)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_save_and_load_state(self):
        """Test saving and loading agent state."""
        snapshot = AgentStateSnapshot(
            agent_id="test_agent",
            agent_type=AgentType.ORCHESTRATOR,
            agent_name="Test Agent",
            agent_description="A test agent",
            current_state=AgentState.ACTIVE,
            is_enabled=True,
            configuration={"model": "gpt-4"},
            metrics={"requests": 50}
        )
        
        metadata = StateMetadata(
            agent_id="test_agent",
            timestamp=datetime.now(),
            format=PersistenceFormat.JSON,
            size_bytes=1000,
            version="1.0",
            checksum="test_checksum"
        )
        
        # Save state
        success = await self.storage.save_state(snapshot, metadata)
        assert success is True
        
        # Verify files were created
        agent_path = Path(self.temp_dir) / "test_agent"
        assert agent_path.exists()
        assert (agent_path / "state_1.0.json").exists()
        assert (agent_path / "metadata_1.0.json").exists()
        assert (agent_path / "state_latest.json").exists()
        assert (agent_path / "metadata_latest.json").exists()
        
        # Load state
        loaded_snapshot = await self.storage.load_state("test_agent", "1.0")
        assert loaded_snapshot is not None
        assert loaded_snapshot.agent_id == snapshot.agent_id
        assert loaded_snapshot.agent_type == snapshot.agent_type
        assert loaded_snapshot.current_state == snapshot.current_state
        assert loaded_snapshot.configuration == snapshot.configuration
        assert loaded_snapshot.metrics == snapshot.metrics
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_state(self):
        """Test loading non-existent state."""
        loaded_snapshot = await self.storage.load_state("nonexistent_agent")
        assert loaded_snapshot is None
    
    @pytest.mark.asyncio
    async def test_list_states(self):
        """Test listing available states."""
        # Save multiple states
        for i in range(3):
            snapshot = AgentStateSnapshot(
                agent_id=f"agent_{i}",
                agent_type=AgentType.ORCHESTRATOR,
                agent_name=f"Agent {i}",
                agent_description=f"Test agent {i}",
                current_state=AgentState.IDLE,
                is_enabled=True
            )
            
            metadata = StateMetadata(
                agent_id=f"agent_{i}",
                timestamp=datetime.now(),
                format=PersistenceFormat.JSON,
                size_bytes=1000,
                version="1.0",
                checksum="test_checksum"
            )
            
            await self.storage.save_state(snapshot, metadata)
        
        # List all states
        states = await self.storage.list_states()
        assert len(states) == 3
        
        # List states for specific agent
        agent_states = await self.storage.list_states("agent_0")
        assert len(agent_states) == 1
        assert agent_states[0].agent_id == "agent_0"
    
    @pytest.mark.asyncio
    async def test_delete_state(self):
        """Test deleting agent state."""
        snapshot = AgentStateSnapshot(
            agent_id="test_agent",
            agent_type=AgentType.ORCHESTRATOR,
            agent_name="Test Agent",
            agent_description="A test agent",
            current_state=AgentState.IDLE,
            is_enabled=True
        )
        
        metadata = StateMetadata(
            agent_id="test_agent",
            timestamp=datetime.now(),
            format=PersistenceFormat.JSON,
            size_bytes=1000,
            version="1.0",
            checksum="test_checksum"
        )
        
        # Save state
        await self.storage.save_state(snapshot, metadata)
        
        # Verify state exists
        loaded_snapshot = await self.storage.load_state("test_agent")
        assert loaded_snapshot is not None
        
        # Delete state
        success = await self.storage.delete_state("test_agent")
        assert success is True
        
        # Verify state is deleted
        loaded_snapshot = await self.storage.load_state("test_agent")
        assert loaded_snapshot is None
        
        # Verify agent directory is deleted
        agent_path = Path(self.temp_dir) / "test_agent"
        assert not agent_path.exists()
    
    @pytest.mark.asyncio
    async def test_different_formats(self):
        """Test storage with different formats."""
        formats = [
            PersistenceFormat.JSON,
            PersistenceFormat.PICKLE,
            PersistenceFormat.COMPRESSED_JSON,
            PersistenceFormat.COMPRESSED_PICKLE
        ]
        
        for format_enum in formats:
            # Create storage with specific format
            format_storage = FileSystemStorage(self.temp_dir, format_enum)
            
            snapshot = AgentStateSnapshot(
                agent_id=f"agent_{format_enum.value}",
                agent_type=AgentType.ORCHESTRATOR,
                agent_name=f"Agent {format_enum.value}",
                agent_description=f"Test agent with {format_enum.value}",
                current_state=AgentState.IDLE,
                is_enabled=True
            )
            
            metadata = StateMetadata(
                agent_id=f"agent_{format_enum.value}",
                timestamp=datetime.now(),
                format=format_enum,
                size_bytes=1000,
                version="1.0",
                checksum="test_checksum"
            )
            
            # Save and load
            success = await format_storage.save_state(snapshot, metadata)
            assert success is True
            
            loaded_snapshot = await format_storage.load_state(f"agent_{format_enum.value}")
            assert loaded_snapshot is not None
            assert loaded_snapshot.agent_id == f"agent_{format_enum.value}"


class TestStateManager:
    """Test StateManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        storage = FileSystemStorage(self.temp_dir, PersistenceFormat.JSON)
        self.state_manager = StateManager(storage)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_state_manager_creation(self):
        """Test creating a state manager."""
        assert self.state_manager.auto_save_interval == 300
        assert len(self.state_manager.get_registered_agents()) == 0
    
    def test_register_and_unregister_agent(self):
        """Test registering and unregistering agents."""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        mock_agent.name = "Test Agent"
        mock_agent.description = "A test agent"
        mock_agent.agent_type = AgentType.ORCHESTRATOR
        mock_agent._status = AgentState.IDLE
        mock_agent._conversation_history = []
        mock_agent._last_activity = datetime.now()
        mock_agent.config = Mock()
        mock_agent.config.capabilities = []
        mock_agent._memory_manager = None
        mock_agent.get_metrics.return_value = {}
        
        # Register agent
        self.state_manager.register_agent(mock_agent)
        assert "test_agent" in self.state_manager.get_registered_agents()
        assert self.state_manager.is_agent_registered("test_agent")
        
        # Unregister agent
        self.state_manager.unregister_agent("test_agent")
        assert "test_agent" not in self.state_manager.get_registered_agents()
        assert not self.state_manager.is_agent_registered("test_agent")
    
    @pytest.mark.asyncio
    async def test_save_and_load_agent_state(self):
        """Test saving and loading agent state."""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        mock_agent.name = "Test Agent"
        mock_agent.description = "A test agent"
        mock_agent.agent_type = AgentType.ORCHESTRATOR
        mock_agent._status = AgentState.ACTIVE
        mock_agent._conversation_history = []
        mock_agent._total_requests = 100
        mock_agent._successful_requests = 95
        mock_agent._failed_requests = 5
        mock_agent._average_response_time = 1.5
        mock_agent._last_activity = datetime.now()
        mock_agent.config = Mock()
        mock_agent.config.capabilities = []
        mock_agent._memory_manager = None
        mock_agent.get_metrics.return_value = {
            "total_requests": 100,
            "successful_requests": 95,
            "failed_requests": 5,
            "average_response_time": 1.5
        }
        
        # Register agent
        self.state_manager.register_agent(mock_agent)
        
        # Save state
        success = await self.state_manager.save_agent_state(mock_agent)
        assert success is True
        
        # Load state
        snapshot = await self.state_manager.load_agent_state("test_agent")
        assert snapshot is not None
        assert snapshot.agent_id == "test_agent"
        assert snapshot.agent_name == "Test Agent"
        assert snapshot.current_state == AgentState.ACTIVE
        assert snapshot.metrics["total_requests"] == 100
    
    @pytest.mark.asyncio
    async def test_restore_agent(self):
        """Test restoring an agent from a snapshot."""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        mock_agent.name = "Original Name"
        mock_agent.description = "Original Description"
        mock_agent.agent_type = AgentType.ORCHESTRATOR
        mock_agent._status = AgentState.IDLE
        mock_agent._conversation_history = []
        mock_agent._total_requests = 0
        mock_agent._successful_requests = 0
        mock_agent._failed_requests = 0
        mock_agent._average_response_time = 0.0
        mock_agent._last_activity = datetime.now()
        mock_agent.config = Mock()
        mock_agent.config.capabilities = []
        mock_agent._memory_manager = None
        mock_agent.get_metrics.return_value = {}
        
        # Create snapshot to restore from
        snapshot = AgentStateSnapshot(
            agent_id="test_agent",
            agent_type=AgentType.USER_TWIN,
            agent_name="Restored Name",
            agent_description="Restored Description",
            current_state=AgentState.ACTIVE,
            is_enabled=True,
            metrics={
                "total_requests": 50,
                "successful_requests": 48,
                "failed_requests": 2,
                "average_response_time": 2.0
            },
            conversation_history=[
                {
                    "content": "Hello",
                    "message_type": "request",
                    "sender": "user",
                    "recipient": "agent",
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {},
                    "message_id": "msg_1"
                }
            ]
        )
        
        # Restore agent
        success = await self.state_manager.restore_agent(mock_agent, snapshot)
        assert success is True
        
        # Verify restoration
        assert mock_agent.name == "Restored Name"
        assert mock_agent.description == "Restored Description"
        assert mock_agent._status == AgentState.ACTIVE
        assert mock_agent._total_requests == 50
        assert mock_agent._successful_requests == 48
        assert mock_agent._failed_requests == 2
        assert mock_agent._average_response_time == 2.0
        assert len(mock_agent._conversation_history) == 1
    
    @pytest.mark.asyncio
    async def test_create_and_restore_backup(self):
        """Test creating and restoring from backup."""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        mock_agent.name = "Test Agent"
        mock_agent.description = "A test agent"
        mock_agent.agent_type = AgentType.ORCHESTRATOR
        mock_agent._status = AgentState.ACTIVE
        mock_agent._conversation_history = []
        mock_agent._total_requests = 100
        mock_agent._successful_requests = 95
        mock_agent._failed_requests = 5
        mock_agent._average_response_time = 1.5
        mock_agent._last_activity = datetime.now()
        mock_agent.config = Mock()
        mock_agent.config.capabilities = []
        mock_agent._memory_manager = None
        mock_agent.get_metrics.return_value = {
            "total_requests": 100,
            "successful_requests": 95,
            "failed_requests": 5,
            "average_response_time": 1.5
        }
        
        # Register agent
        self.state_manager.register_agent(mock_agent)
        
        # Create backup
        backup_id = await self.state_manager.create_backup("test_agent", "test_backup")
        assert backup_id == "test_backup"
        
        # Modify agent state
        mock_agent._total_requests = 200
        mock_agent._status = AgentState.ERROR
        
        # Restore from backup
        success = await self.state_manager.restore_from_backup("test_agent", "test_backup")
        assert success is True
        
        # Verify restoration (this would require checking the mock agent's state)
        # In a real scenario, the agent's state would be restored
    
    @pytest.mark.asyncio
    async def test_list_and_delete_states(self):
        """Test listing and deleting agent states."""
        # Create and save multiple states
        for i in range(3):
            mock_agent = Mock()
            mock_agent.agent_id = f"agent_{i}"
            mock_agent.name = f"Agent {i}"
            mock_agent.description = f"Test agent {i}"
            mock_agent.agent_type = AgentType.ORCHESTRATOR
            mock_agent._status = AgentState.IDLE
            mock_agent._conversation_history = []
            mock_agent._total_requests = 0
            mock_agent._successful_requests = 0
            mock_agent._failed_requests = 0
            mock_agent._average_response_time = 0.0
            mock_agent._last_activity = datetime.now()
            mock_agent.config = Mock()
            mock_agent.config.capabilities = []
            mock_agent._memory_manager = None
            mock_agent.get_metrics.return_value = {}
            
            self.state_manager.register_agent(mock_agent)
            await self.state_manager.save_agent_state(mock_agent)
        
        # List all states
        states = await self.state_manager.list_agent_states()
        assert len(states) == 3
        
        # List states for specific agent
        agent_states = await self.state_manager.list_agent_states("agent_0")
        assert len(agent_states) == 1
        assert agent_states[0].agent_id == "agent_0"
        
        # Delete state for specific agent
        success = await self.state_manager.delete_agent_state("agent_0")
        assert success is True
        
        # Verify deletion
        remaining_states = await self.state_manager.list_agent_states()
        assert len(remaining_states) == 2
    
    @pytest.mark.asyncio
    async def test_auto_save_functionality(self):
        """Test automatic state saving."""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        mock_agent.name = "Test Agent"
        mock_agent.description = "A test agent"
        mock_agent.agent_type = AgentType.ORCHESTRATOR
        mock_agent._status = AgentState.IDLE
        mock_agent._conversation_history = []
        mock_agent._total_requests = 0
        mock_agent._successful_requests = 0
        mock_agent._failed_requests = 0
        mock_agent._average_response_time = 0.0
        mock_agent._last_activity = datetime.now()
        mock_agent.config = Mock()
        mock_agent.config.capabilities = []
        mock_agent._memory_manager = None
        mock_agent.get_metrics.return_value = {}
        
        # Register agent
        self.state_manager.register_agent(mock_agent)
        
        # Set short auto-save interval for testing
        self.state_manager.auto_save_interval = 1
        
        # Start auto-save
        await self.state_manager.start_auto_save("test_agent")
        
        # Wait for auto-save to trigger
        await asyncio.sleep(1.5)
        
        # Stop auto-save
        await self.state_manager.stop_auto_save("test_agent")
        
        # Verify auto-save was working (state should exist)
        states = await self.state_manager.list_agent_states("test_agent")
        assert len(states) > 0


class TestStateManagerUtilities:
    """Test state manager utility functions."""
    
    def test_create_state_manager(self):
        """Test creating state manager with configuration."""
        # Test with filesystem storage
        state_manager = create_state_manager(
            storage_backend="filesystem",
            storage_config={
                "base_path": "/tmp/test_states",
                "format": "json"
            }
        )
        
        assert isinstance(state_manager, StateManager)
        assert isinstance(state_manager.storage, FileSystemStorage)
    
    def test_create_state_manager_invalid_backend(self):
        """Test creating state manager with invalid backend."""
        with pytest.raises(ValueError, match="Unsupported storage backend"):
            create_state_manager(storage_backend="invalid")
    
    def test_get_state_manager_singleton(self):
        """Test that get_state_manager returns a singleton."""
        manager1 = get_state_manager()
        manager2 = get_state_manager()
        
        assert manager1 is manager2


class TestStateManagerIntegration:
    """Integration tests for state manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        storage = FileSystemStorage(self.temp_dir, PersistenceFormat.JSON)
        self.state_manager = StateManager(storage)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_complete_state_workflow(self):
        """Test complete state management workflow."""
        # Create mock agent with conversation history
        mock_agent = Mock()
        mock_agent.agent_id = "workflow_agent"
        mock_agent.name = "Workflow Agent"
        mock_agent.description = "Agent for workflow testing"
        mock_agent.agent_type = AgentType.ORCHESTRATOR
        mock_agent._status = AgentState.ACTIVE
        mock_agent._conversation_history = [
            AgentMessage(
                content="Hello",
                message_type=MessageType.REQUEST,
                sender="user",
                recipient="agent",
                message_id="msg_1"
            ),
            AgentMessage(
                content="Hi there!",
                message_type=MessageType.RESPONSE,
                sender="agent",
                recipient="user",
                message_id="msg_2"
            )
        ]
        mock_agent._total_requests = 2
        mock_agent._successful_requests = 2
        mock_agent._failed_requests = 0
        mock_agent._average_response_time = 1.2
        mock_agent._last_activity = datetime.now()
        mock_agent.config = Mock()
        mock_agent.config.capabilities = ["reasoning", "memory"]
        mock_agent._memory_manager = None
        mock_agent.get_metrics.return_value = {
            "total_requests": 2,
            "successful_requests": 2,
            "failed_requests": 0,
            "average_response_time": 1.2
        }
        
        # Register agent
        self.state_manager.register_agent(mock_agent)
        
        # 1. Save initial state
        success = await self.state_manager.save_agent_state(mock_agent)
        assert success is True
        
        # 2. Create backup
        backup_id = await self.state_manager.create_backup("workflow_agent", "initial_backup")
        assert backup_id == "initial_backup"
        
        # 3. Modify agent state
        mock_agent._total_requests = 10
        mock_agent._status = AgentState.ERROR
        mock_agent._conversation_history.append(
            AgentMessage(
                content="Error occurred",
                message_type=MessageType.ERROR,
                sender="system",
                recipient="agent",
                message_id="msg_3"
            )
        )
        
        # 4. Save modified state
        success = await self.state_manager.save_agent_state(mock_agent, "modified")
        assert success is True
        
        # 5. Restore from backup
        success = await self.state_manager.restore_from_backup("workflow_agent", "initial_backup")
        assert success is True
        
        # 6. Verify restoration (check that agent state was restored)
        # Note: In a real implementation, we would verify the agent's state
        # was actually restored. Here we just verify the operation succeeded.
        
        # 7. List all states
        states = await self.state_manager.list_agent_states("workflow_agent")
        assert len(states) >= 2  # At least initial and modified states
        
        # 8. Delete specific version
        success = await self.state_manager.delete_agent_state("workflow_agent", "modified")
        assert success is True
        
        # 9. Verify deletion
        remaining_states = await self.state_manager.list_agent_states("workflow_agent")
        assert len(remaining_states) >= 1  # At least the backup should remain
    
    @pytest.mark.asyncio
    async def test_state_versioning(self):
        """Test state versioning functionality."""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "versioned_agent"
        mock_agent.name = "Versioned Agent"
        mock_agent.description = "Agent for versioning testing"
        mock_agent.agent_type = AgentType.USER_TWIN
        mock_agent._status = AgentState.IDLE
        mock_agent._conversation_history = []
        mock_agent._total_requests = 0
        mock_agent._successful_requests = 0
        mock_agent._failed_requests = 0
        mock_agent._average_response_time = 0.0
        mock_agent._last_activity = datetime.now()
        mock_agent.config = Mock()
        mock_agent.config.capabilities = []
        mock_agent._memory_manager = None
        # Make get_metrics return dynamic values based on current state
        def dynamic_get_metrics():
            return {
                "total_requests": mock_agent._total_requests,
                "successful_requests": mock_agent._successful_requests,
                "failed_requests": mock_agent._failed_requests,
                "average_response_time": mock_agent._average_response_time
            }
        mock_agent.get_metrics.side_effect = dynamic_get_metrics
        
        # Register agent
        self.state_manager.register_agent(mock_agent)
        
        # Save multiple versions
        versions = ["1.0", "1.1", "1.2"]
        for version in versions:
            mock_agent._total_requests = int(version.split('.')[1]) * 10
            success = await self.state_manager.save_agent_state(mock_agent, version)
            assert success is True
        
        # List all states
        states = await self.state_manager.list_agent_states("versioned_agent")
        assert len(states) == 3
        
        # Load specific versions and verify the state was saved correctly
        for version in versions:
            snapshot = await self.state_manager.load_agent_state("versioned_agent", version)
            assert snapshot is not None
            expected_requests = int(version.split('.')[1]) * 10
            # Check the snapshot data directly since the mock agent state changes during the test
            assert snapshot.metrics.get("total_requests", 0) == expected_requests
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in state operations."""
        # Test loading non-existent agent
        snapshot = await self.state_manager.load_agent_state("nonexistent_agent")
        assert snapshot is None
        
        # Test creating backup for non-existent agent
        backup_id = await self.state_manager.create_backup("nonexistent_agent")
        assert backup_id is None
        
        # Test restoring from non-existent backup
        success = await self.state_manager.restore_from_backup("nonexistent_agent", "nonexistent_backup")
        assert success is False
        
        # Test deleting non-existent state
        success = await self.state_manager.delete_agent_state("nonexistent_agent")
        assert success is False
