"""
Unit tests for the Sandbox Manager System

This module contains comprehensive tests for the SandboxManager, SandboxSession,
SandboxScenario, and related functionality in the SafeHive sandbox system.
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from safehive.sandbox.sandbox_manager import (
    SandboxManager, SandboxSession, SandboxScenario, SessionStatus, SessionPhase
)
from safehive.config.config_loader import ConfigLoader


class TestSandboxScenario:
    """Test cases for SandboxScenario class."""

    def test_sandbox_scenario_creation(self):
        """Test SandboxScenario creation."""
        scenario = SandboxScenario(
            name="test-scenario",
            description="Test scenario description",
            duration=300,
            interactive=True,
            agents=["agent1", "agent2"],
            guards=["guard1", "guard2"],
            parameters={"param1": "value1"},
            metadata={"category": "test"}
        )
        
        assert scenario.name == "test-scenario"
        assert scenario.description == "Test scenario description"
        assert scenario.duration == 300
        assert scenario.interactive is True
        assert scenario.agents == ["agent1", "agent2"]
        assert scenario.guards == ["guard1", "guard2"]
        assert scenario.parameters == {"param1": "value1"}
        assert scenario.metadata == {"category": "test"}


class TestSandboxSession:
    """Test cases for SandboxSession class."""

    def test_sandbox_session_creation(self):
        """Test SandboxSession creation."""
        scenario = SandboxScenario(
            name="test-scenario",
            description="Test scenario",
            duration=300,
            interactive=True
        )
        
        session = SandboxSession(
            session_id="test-session-id",
            scenario=scenario,
            interactive=True
        )
        
        assert session.session_id == "test-session-id"
        assert session.scenario == scenario
        assert session.status == SessionStatus.PENDING
        assert session.phase == SessionPhase.INITIALIZATION
        assert session.start_time is None
        assert session.end_time is None
        assert session.duration == 0
        assert session.interactive is True
        assert session.metrics == {}
        assert session.logs == []
        assert session.agents == {}
        assert session.guards == {}
        assert session.events == []
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)

    def test_sandbox_session_to_dict(self):
        """Test SandboxSession serialization."""
        scenario = SandboxScenario(
            name="test-scenario",
            description="Test scenario",
            duration=300,
            interactive=True
        )
        
        session = SandboxSession(
            session_id="test-session-id",
            scenario=scenario,
            interactive=True
        )
        
        data = session.to_dict()
        
        assert data["session_id"] == "test-session-id"
        assert data["scenario"]["name"] == "test-scenario"
        assert data["status"] == "pending"
        assert data["phase"] == "initialization"
        assert data["interactive"] is True
        assert "created_at" in data
        assert "updated_at" in data

    def test_sandbox_session_from_dict(self):
        """Test SandboxSession deserialization."""
        data = {
            "session_id": "test-session-id",
            "scenario": {
                "name": "test-scenario",
                "description": "Test scenario",
                "duration": 300,
                "interactive": True,
                "agents": [],
                "guards": [],
                "parameters": {},
                "metadata": {}
            },
            "status": "running",
            "phase": "agent_setup",
            "start_time": "2023-01-01T12:00:00",
            "end_time": None,
            "duration": 60,
            "interactive": True,
            "metrics": {},
            "logs": [],
            "agents": {},
            "guards": {},
            "events": [],
            "created_at": "2023-01-01T11:00:00",
            "updated_at": "2023-01-01T12:00:00"
        }
        
        session = SandboxSession.from_dict(data)
        
        assert session.session_id == "test-session-id"
        assert session.scenario.name == "test-scenario"
        assert session.status == SessionStatus.RUNNING
        assert session.phase == SessionPhase.AGENT_SETUP
        assert session.duration == 60
        assert session.interactive is True
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)


class TestSandboxManager:
    """Test cases for SandboxManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('safehive.sandbox.sandbox_manager.ConfigLoader'):
            self.manager = SandboxManager()

    def test_sandbox_manager_initialization(self):
        """Test SandboxManager initialization."""
        assert len(self.manager.available_scenarios) == 4
        assert "food-ordering" in self.manager.available_scenarios
        assert "payment-processing" in self.manager.available_scenarios
        assert "api-integration" in self.manager.available_scenarios
        assert "data-extraction" in self.manager.available_scenarios
        assert len(self.manager.active_sessions) == 0
        assert len(self.manager.session_history) == 0

    def test_list_scenarios(self):
        """Test listing available scenarios."""
        scenarios = self.manager.list_scenarios()
        
        assert len(scenarios) == 4
        assert "food-ordering" in scenarios
        assert scenarios["food-ordering"].name == "food-ordering"
        assert scenarios["food-ordering"].description == "Food ordering workflow with malicious vendors"

    def test_get_scenario(self):
        """Test getting a specific scenario."""
        scenario = self.manager.get_scenario("food-ordering")
        
        assert scenario is not None
        assert scenario.name == "food-ordering"
        assert scenario.duration == 300
        assert len(scenario.agents) == 4
        assert len(scenario.guards) == 4

    def test_get_scenario_nonexistent(self):
        """Test getting a nonexistent scenario."""
        scenario = self.manager.get_scenario("nonexistent")
        
        assert scenario is None

    def test_create_session(self):
        """Test creating a sandbox session."""
        session = self.manager.create_session("food-ordering")
        
        assert session is not None
        assert session.session_id is not None
        assert session.scenario.name == "food-ordering"
        assert session.status == SessionStatus.PENDING
        assert session.session_id in self.manager.active_sessions

    def test_create_session_with_overrides(self):
        """Test creating a session with parameter overrides."""
        session = self.manager.create_session(
            "food-ordering",
            duration=600,
            interactive=False,
            parameters={"custom_param": "custom_value"}
        )
        
        assert session is not None
        assert session.scenario.duration == 600
        assert session.interactive is False
        assert session.scenario.parameters["custom_param"] == "custom_value"

    def test_create_session_nonexistent_scenario(self):
        """Test creating a session with nonexistent scenario."""
        session = self.manager.create_session("nonexistent")
        
        assert session is None

    @pytest.mark.asyncio
    async def test_start_session(self):
        """Test starting a sandbox session."""
        session = self.manager.create_session("food-ordering")
        assert session is not None
        
        success = await self.manager.start_session(session.session_id, wait_for_completion=True)
        
        assert success is True
        # Session should be moved to history after completion
        assert session.session_id not in self.manager.active_sessions

    @pytest.mark.asyncio
    async def test_start_session_nonexistent(self):
        """Test starting a nonexistent session."""
        success = await self.manager.start_session("nonexistent-session-id")
        
        assert success is False

    @pytest.mark.asyncio
    async def test_stop_session(self):
        """Test stopping a sandbox session."""
        session = self.manager.create_session("food-ordering")
        assert session is not None
        
        # Start session first (without waiting for completion)
        success = await self.manager.start_session(session.session_id, wait_for_completion=False)
        assert success is True
        
        # Wait a bit for session to start
        await asyncio.sleep(0.1)
        
        # Stop session
        success = await self.manager.stop_session(session.session_id)
        
        assert success is True
        assert session.status == SessionStatus.STOPPED
        assert session.end_time is not None

    @pytest.mark.asyncio
    async def test_pause_resume_session(self):
        """Test pausing and resuming a session."""
        session = self.manager.create_session("food-ordering")
        assert session is not None
        
        # Start session (without waiting for completion)
        success = await self.manager.start_session(session.session_id, wait_for_completion=False)
        assert success is True
        
        # Wait a bit for session to start
        await asyncio.sleep(0.1)
        
        # Pause session
        success = await self.manager.pause_session(session.session_id)
        assert success is True
        assert session.status == SessionStatus.PAUSED
        
        # Resume session
        success = await self.manager.resume_session(session.session_id)
        assert success is True
        assert session.status == SessionStatus.RUNNING

    def test_get_session(self):
        """Test getting a session by ID."""
        session = self.manager.create_session("food-ordering")
        assert session is not None
        
        retrieved_session = self.manager.get_session(session.session_id)
        
        assert retrieved_session is not None
        assert retrieved_session.session_id == session.session_id

    def test_get_session_nonexistent(self):
        """Test getting a nonexistent session."""
        session = self.manager.get_session("nonexistent-session-id")
        
        assert session is None

    def test_get_active_sessions(self):
        """Test getting all active sessions."""
        session1 = self.manager.create_session("food-ordering")
        session2 = self.manager.create_session("payment-processing")
        
        active_sessions = self.manager.get_active_sessions()
        
        assert len(active_sessions) == 2
        assert session1.session_id in active_sessions
        assert session2.session_id in active_sessions

    def test_get_session_history(self):
        """Test getting session history."""
        # Create and complete a session
        session = self.manager.create_session("food-ordering")
        self.manager.session_history.append(session)
        
        history = self.manager.get_session_history()
        
        assert len(history) == 1
        assert history[0].session_id == session.session_id

    def test_save_session_history(self, tmp_path):
        """Test saving session history to file."""
        # Create a test session
        session = self.manager.create_session("food-ordering")
        self.manager.session_history.append(session)
        
        filepath = tmp_path / "test_history.json"
        success = self.manager.save_session_history(str(filepath))
        
        assert success is True
        assert filepath.exists()
        
        # Verify file content
        with open(filepath, 'r') as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["session_id"] == session.session_id

    def test_load_session_history(self, tmp_path):
        """Test loading session history from file."""
        # Create test data
        session_data = {
            "session_id": "test-session-id",
            "scenario": {
                "name": "test-scenario",
                "description": "Test scenario",
                "duration": 300,
                "interactive": True,
                "agents": [],
                "guards": [],
                "parameters": {},
                "metadata": {}
            },
            "status": "completed",
            "phase": "cleanup",
            "start_time": None,
            "end_time": None,
            "duration": 0,
            "interactive": True,
            "metrics": {},
            "logs": [],
            "agents": {},
            "guards": {},
            "events": [],
            "created_at": "2023-01-01T11:00:00",
            "updated_at": "2023-01-01T12:00:00"
        }
        
        filepath = tmp_path / "test_history.json"
        with open(filepath, 'w') as f:
            json.dump([session_data], f)
        
        success = self.manager.load_session_history(str(filepath))
        
        assert success is True
        assert len(self.manager.session_history) == 1
        assert self.manager.session_history[0].session_id == "test-session-id"

    def test_load_session_history_nonexistent_file(self, tmp_path):
        """Test loading session history from nonexistent file."""
        filepath = tmp_path / "nonexistent.json"
        success = self.manager.load_session_history(str(filepath))
        
        assert success is False

    @pytest.mark.asyncio
    async def test_session_execution_phases(self):
        """Test that session goes through all execution phases."""
        session = self.manager.create_session("food-ordering")
        assert session is not None
        
        # Start session and let it complete
        await self.manager.start_session(session.session_id, wait_for_completion=True)
        
        # Check that session went through phases
        assert session.status == SessionStatus.COMPLETED
        assert session.end_time is not None
        assert session.duration > 0
        assert len(session.events) > 0
        
        # Check that events were recorded for each phase
        event_types = [event["type"] for event in session.events]
        assert "initialization" in event_types
        assert "agent_setup" in event_types
        assert "guard_activation" in event_types
        assert "scenario_execution" in event_types
        assert "monitoring" in event_types
        assert "cleanup" in event_types

    @pytest.mark.asyncio
    async def test_session_cancellation(self):
        """Test session cancellation."""
        session = self.manager.create_session("food-ordering")
        assert session is not None
        
        # Start session
        start_task = asyncio.create_task(self.manager.start_session(session.session_id))
        
        # Wait a bit then cancel
        await asyncio.sleep(0.1)
        await self.manager.stop_session(session.session_id)
        
        # Wait for start task to complete
        await start_task
        
        assert session.status == SessionStatus.STOPPED

    def test_thread_safety(self):
        """Test thread safety of sandbox manager."""
        import threading
        
        def create_sessions():
            for _ in range(10):
                session = self.manager.create_session("food-ordering")
                assert session is not None
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_sessions)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have 50 sessions total
        assert len(self.manager.active_sessions) == 50


class TestSandboxManagerIntegration:
    """Integration tests for sandbox manager."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('safehive.sandbox.sandbox_manager.ConfigLoader'):
            self.manager = SandboxManager()

    @pytest.mark.asyncio
    async def test_complete_session_workflow(self):
        """Test complete session workflow from creation to completion."""
        # Create session
        session = self.manager.create_session("food-ordering", duration=10)
        assert session is not None
        assert session.status == SessionStatus.PENDING
        
        # Start session
        success = await self.manager.start_session(session.session_id, wait_for_completion=True)
        assert success is True
        
        # Wait for completion
        await asyncio.sleep(0.1)  # Let it run briefly
        
        # Check final state
        assert session.status == SessionStatus.COMPLETED
        assert session.end_time is not None
        assert session.duration > 0
        assert len(session.events) > 0
        
        # Check that session is in history
        history = self.manager.get_session_history()
        assert len(history) == 1
        assert history[0].session_id == session.session_id

    @pytest.mark.asyncio
    async def test_multiple_concurrent_sessions(self):
        """Test multiple concurrent sessions."""
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = self.manager.create_session("food-ordering", duration=5)
            sessions.append(session)
        
        # Start all sessions concurrently
        start_tasks = []
        for session in sessions:
            task = asyncio.create_task(self.manager.start_session(session.session_id, wait_for_completion=True))
            start_tasks.append(task)
        
        # Wait for all to complete
        await asyncio.gather(*start_tasks)
        
        # Check that all sessions completed
        for session in sessions:
            assert session.status == SessionStatus.COMPLETED
        
        # Check history
        history = self.manager.get_session_history()
        assert len(history) == 3

    def test_scenario_metadata(self):
        """Test scenario metadata and parameters."""
        scenarios = self.manager.list_scenarios()
        
        # Check food-ordering scenario
        food_scenario = scenarios["food-ordering"]
        assert food_scenario.metadata["category"] == "e-commerce"
        assert food_scenario.metadata["difficulty"] == "medium"
        assert "attack_types" in food_scenario.metadata
        
        # Check payment-processing scenario
        payment_scenario = scenarios["payment-processing"]
        assert payment_scenario.metadata["category"] == "fintech"
        assert payment_scenario.metadata["difficulty"] == "high"
        
        # Check API integration scenario
        api_scenario = scenarios["api-integration"]
        assert api_scenario.metadata["category"] == "api_security"
        
        # Check data extraction scenario
        data_scenario = scenarios["data-extraction"]
        assert data_scenario.metadata["category"] == "data_privacy"

    def test_session_metrics_tracking(self):
        """Test that session metrics are properly tracked."""
        session = self.manager.create_session("food-ordering")
        
        # Add some test metrics
        session.metrics["test_metric"] = {"value": 42, "timestamp": datetime.now().isoformat()}
        session.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "test_event",
            "message": "Test event"
        })
        
        # Verify metrics are preserved
        assert "test_metric" in session.metrics
        assert len(session.events) == 1
        assert session.events[0]["type"] == "test_event"

    def test_session_serialization_roundtrip(self):
        """Test session serialization and deserialization."""
        # Create original session
        original_session = self.manager.create_session("food-ordering")
        original_session.metrics["test"] = "value"
        original_session.events.append({"type": "test", "message": "test"})
        
        # Serialize to dict
        data = original_session.to_dict()
        
        # Deserialize from dict
        restored_session = SandboxSession.from_dict(data)
        
        # Verify they match
        assert restored_session.session_id == original_session.session_id
        assert restored_session.scenario.name == original_session.scenario.name
        assert restored_session.metrics == original_session.metrics
        assert len(restored_session.events) == len(original_session.events)
