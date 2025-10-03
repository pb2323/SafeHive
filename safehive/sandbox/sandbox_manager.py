"""
Sandbox Manager for SafeHive AI Security Sandbox

This module provides the core sandbox management functionality for launching,
controlling, and monitoring security testing sessions with AI agents and guards.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import json

from safehive.utils.logger import get_logger
from safehive.utils.metrics import record_metric, MetricType, record_event
from safehive.config.config_loader import ConfigLoader

logger = get_logger(__name__)


class SessionStatus(Enum):
    """Status of a sandbox session."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETED = "completed"


class SessionPhase(Enum):
    """Phase of a sandbox session."""
    INITIALIZATION = "initialization"
    AGENT_SETUP = "agent_setup"
    GUARD_ACTIVATION = "guard_activation"
    SCENARIO_EXECUTION = "scenario_execution"
    MONITORING = "monitoring"
    CLEANUP = "cleanup"


@dataclass
class SandboxScenario:
    """Configuration for a sandbox scenario."""
    name: str
    description: str
    duration: int  # seconds
    interactive: bool
    agents: List[str] = field(default_factory=list)
    guards: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxSession:
    """Represents an active sandbox session."""
    session_id: str
    scenario: SandboxScenario
    status: SessionStatus = SessionStatus.PENDING
    phase: SessionPhase = SessionPhase.INITIALIZATION
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: int = 0
    interactive: bool = True
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    agents: Dict[str, Any] = field(default_factory=dict)
    guards: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "scenario": {
                "name": self.scenario.name,
                "description": self.scenario.description,
                "duration": self.scenario.duration,
                "interactive": self.scenario.interactive,
                "agents": self.scenario.agents,
                "guards": self.scenario.guards,
                "parameters": self.scenario.parameters,
                "metadata": self.scenario.metadata
            },
            "status": self.status.value,
            "phase": self.phase.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "interactive": self.interactive,
            "metrics": self.metrics,
            "logs": self.logs,
            "agents": self.agents,
            "guards": self.guards,
            "events": self.events,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SandboxSession':
        """Create session from dictionary."""
        scenario_data = data["scenario"]
        scenario = SandboxScenario(
            name=scenario_data["name"],
            description=scenario_data["description"],
            duration=scenario_data["duration"],
            interactive=scenario_data["interactive"],
            agents=scenario_data.get("agents", []),
            guards=scenario_data.get("guards", []),
            parameters=scenario_data.get("parameters", {}),
            metadata=scenario_data.get("metadata", {})
        )
        
        session = cls(
            session_id=data["session_id"],
            scenario=scenario,
            status=SessionStatus(data["status"]),
            phase=SessionPhase(data["phase"]),
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            duration=data["duration"],
            interactive=data["interactive"],
            metrics=data.get("metrics", {}),
            logs=data.get("logs", []),
            agents=data.get("agents", {}),
            guards=data.get("guards", {}),
            events=data.get("events", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
        
        return session


class SandboxManager:
    """
    Manages sandbox sessions and scenarios.
    
    Provides functionality to launch, control, and monitor security testing
    sessions with AI agents and security guards.
    """
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """
        Initialize the sandbox manager.
        
        Args:
            config_loader: Optional configuration loader instance
        """
        self.config_loader = config_loader or ConfigLoader()
        self.active_sessions: Dict[str, SandboxSession] = {}
        self.session_history: List[SandboxSession] = []
        self._lock = threading.RLock()
        self._session_tasks: Dict[str, asyncio.Task] = {}
        
        # Load configuration
        self.config = self.config_loader.load_config()
        
        # Available scenarios
        self.available_scenarios = {
            "food-ordering": SandboxScenario(
                name="food-ordering",
                description="Food ordering workflow with malicious vendors",
                duration=300,  # 5 minutes
                interactive=True,
                agents=["orchestrator", "user-twin", "honest-vendor", "malicious-vendor"],
                guards=["privacy-sentry", "task-navigator", "prompt-sanitizer", "honeypot-guard"],
                parameters={
                    "max_orders": 10,
                    "vendor_types": ["restaurant", "fast_food", "delivery"],
                    "payment_methods": ["credit_card", "paypal", "crypto"]
                },
                metadata={
                    "category": "e-commerce",
                    "difficulty": "medium",
                    "attack_types": ["data_exfiltration", "payment_fraud", "social_engineering"]
                }
            ),
            "payment-processing": SandboxScenario(
                name="payment-processing",
                description="Payment processing with security testing",
                duration=180,  # 3 minutes
                interactive=True,
                agents=["orchestrator", "user-twin", "payment-processor", "fraud-detector"],
                guards=["privacy-sentry", "honeypot-guard", "prompt-sanitizer"],
                parameters={
                    "transaction_types": ["purchase", "refund", "chargeback"],
                    "amount_range": [1, 1000],
                    "currency_types": ["USD", "EUR", "BTC"]
                },
                metadata={
                    "category": "fintech",
                    "difficulty": "high",
                    "attack_types": ["payment_fraud", "card_testing", "chargeback_fraud"]
                }
            ),
            "api-integration": SandboxScenario(
                name="api-integration",
                description="API integration security testing",
                duration=240,  # 4 minutes
                interactive=True,
                agents=["orchestrator", "user-twin", "api-client", "api-server"],
                guards=["prompt-sanitizer", "task-navigator", "honeypot-guard"],
                parameters={
                    "api_endpoints": ["/users", "/orders", "/payments", "/admin"],
                    "http_methods": ["GET", "POST", "PUT", "DELETE"],
                    "auth_types": ["bearer", "api_key", "oauth"]
                },
                metadata={
                    "category": "api_security",
                    "difficulty": "medium",
                    "attack_types": ["injection", "authentication_bypass", "rate_limiting"]
                }
            ),
            "data-extraction": SandboxScenario(
                name="data-extraction",
                description="Data extraction and privacy testing",
                duration=360,  # 6 minutes
                interactive=True,
                agents=["orchestrator", "user-twin", "data-analyst", "privacy-auditor"],
                guards=["privacy-sentry", "task-navigator", "prompt-sanitizer"],
                parameters={
                    "data_types": ["personal", "financial", "health", "behavioral"],
                    "extraction_methods": ["query", "scraping", "api", "export"],
                    "privacy_levels": ["public", "internal", "confidential", "restricted"]
                },
                metadata={
                    "category": "data_privacy",
                    "difficulty": "high",
                    "attack_types": ["data_exfiltration", "privacy_violation", "unauthorized_access"]
                }
            )
        }
        
        logger.info(f"SandboxManager initialized with {len(self.available_scenarios)} scenarios")

    def list_scenarios(self) -> Dict[str, SandboxScenario]:
        """List all available scenarios."""
        return self.available_scenarios.copy()

    def get_scenario(self, name: str) -> Optional[SandboxScenario]:
        """Get a specific scenario by name."""
        return self.available_scenarios.get(name)

    def create_session(
        self,
        scenario_name: str,
        duration: Optional[int] = None,
        interactive: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[SandboxSession]:
        """
        Create a new sandbox session.
        
        Args:
            scenario_name: Name of the scenario to run
            duration: Optional custom duration in seconds
            interactive: Optional interactive mode override
            parameters: Optional scenario parameters override
        
        Returns:
            Created session or None if scenario not found
        """
        scenario = self.get_scenario(scenario_name)
        if not scenario:
            logger.error(f"Scenario '{scenario_name}' not found")
            return None
        
        # Create session with overrides
        session_scenario = SandboxScenario(
            name=scenario.name,
            description=scenario.description,
            duration=duration or scenario.duration,
            interactive=interactive if interactive is not None else scenario.interactive,
            agents=scenario.agents.copy(),
            guards=scenario.guards.copy(),
            parameters={**scenario.parameters, **(parameters or {})},
            metadata=scenario.metadata.copy()
        )
        
        session_id = str(uuid.uuid4())
        session = SandboxSession(
            session_id=session_id,
            scenario=session_scenario,
            interactive=session_scenario.interactive
        )
        
        with self._lock:
            self.active_sessions[session_id] = session
        
        # Record metrics
        record_metric("sandbox.session_created", 1, MetricType.COUNTER, {"scenario": scenario_name})
        record_event("sandbox.session_created", f"Session {session_id} created for scenario {scenario_name}")
        
        logger.info(f"Created sandbox session {session_id} for scenario '{scenario_name}'")
        return session

    async def start_session(self, session_id: str, wait_for_completion: bool = False) -> bool:
        """
        Start a sandbox session.
        
        Args:
            session_id: ID of the session to start
            wait_for_completion: If True, wait for session to complete before returning
        
        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            session = self.active_sessions.get(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return False
            
            if session.status != SessionStatus.PENDING:
                logger.error(f"Session {session_id} is not in pending status: {session.status}")
                return False
        
        try:
            # Update session status
            session.status = SessionStatus.STARTING
            session.start_time = datetime.now()
            session.updated_at = datetime.now()
            
            # Record metrics
            record_metric("sandbox.session_started", 1, MetricType.COUNTER, {"scenario": session.scenario.name})
            record_event("sandbox.session_started", f"Session {session_id} started")
            
            logger.info(f"Starting sandbox session {session_id}")
            
            # Create and start session task
            task = asyncio.create_task(self._run_session(session))
            self._session_tasks[session_id] = task
            
            if wait_for_completion:
                # Wait for the session to complete
                try:
                    await task
                    return True
                except Exception as e:
                    logger.error(f"Session {session_id} failed: {e}")
                    return False
            else:
                return True
            
        except Exception as e:
            logger.error(f"Failed to start session {session_id}: {e}")
            session.status = SessionStatus.ERROR
            session.updated_at = datetime.now()
            return False

    async def stop_session(self, session_id: str) -> bool:
        """
        Stop a sandbox session.
        
        Args:
            session_id: ID of the session to stop
        
        Returns:
            True if stopped successfully, False otherwise
        """
        with self._lock:
            session = self.active_sessions.get(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return False
        
        try:
            # Update session status
            session.status = SessionStatus.STOPPING
            session.updated_at = datetime.now()
            
            # Cancel session task if running
            if session_id in self._session_tasks:
                task = self._session_tasks[session_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self._session_tasks[session_id]
            
            # Finalize session
            session.status = SessionStatus.STOPPED
            session.end_time = datetime.now()
            if session.start_time:
                session.duration = int((session.end_time - session.start_time).total_seconds())
            session.updated_at = datetime.now()
            
            # Move to history
            with self._lock:
                self.session_history.append(session)
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
            
            # Record metrics
            record_metric("sandbox.session_stopped", 1, MetricType.COUNTER, {"scenario": session.scenario.name})
            record_metric("sandbox.session_duration", session.duration, MetricType.TIMER, {"scenario": session.scenario.name})
            record_event("sandbox.session_stopped", f"Session {session_id} stopped after {session.duration}s")
            
            logger.info(f"Stopped sandbox session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop session {session_id}: {e}")
            session.status = SessionStatus.ERROR
            session.updated_at = datetime.now()
            return False

    async def pause_session(self, session_id: str) -> bool:
        """Pause a sandbox session."""
        with self._lock:
            session = self.active_sessions.get(session_id)
            if not session:
                return False
            
            if session.status != SessionStatus.RUNNING:
                return False
            
            session.status = SessionStatus.PAUSED
            session.updated_at = datetime.now()
            
            record_metric("sandbox.session_paused", 1, MetricType.COUNTER)
            logger.info(f"Paused sandbox session {session_id}")
            return True

    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused sandbox session."""
        with self._lock:
            session = self.active_sessions.get(session_id)
            if not session:
                return False
            
            if session.status != SessionStatus.PAUSED:
                return False
            
            session.status = SessionStatus.RUNNING
            session.updated_at = datetime.now()
            
            record_metric("sandbox.session_resumed", 1, MetricType.COUNTER)
            logger.info(f"Resumed sandbox session {session_id}")
            return True

    def get_session(self, session_id: str) -> Optional[SandboxSession]:
        """Get a session by ID."""
        with self._lock:
            return self.active_sessions.get(session_id)

    def get_active_sessions(self) -> Dict[str, SandboxSession]:
        """Get all active sessions."""
        with self._lock:
            return self.active_sessions.copy()

    def get_session_history(self, limit: int = 10) -> List[SandboxSession]:
        """Get session history."""
        with self._lock:
            return self.session_history[-limit:] if self.session_history else []

    async def _run_session(self, session: SandboxSession) -> None:
        """Run a sandbox session."""
        try:
            # Phase 1: Initialization
            session.phase = SessionPhase.INITIALIZATION
            session.status = SessionStatus.RUNNING
            session.updated_at = datetime.now()
            
            await self._initialize_session(session)
            
            # Phase 2: Agent Setup
            session.phase = SessionPhase.AGENT_SETUP
            session.updated_at = datetime.now()
            
            await self._setup_agents(session)
            
            # Phase 3: Guard Activation
            session.phase = SessionPhase.GUARD_ACTIVATION
            session.updated_at = datetime.now()
            
            await self._activate_guards(session)
            
            # Phase 4: Scenario Execution
            session.phase = SessionPhase.SCENARIO_EXECUTION
            session.updated_at = datetime.now()
            
            await self._execute_scenario(session)
            
            # Phase 5: Monitoring
            session.phase = SessionPhase.MONITORING
            session.updated_at = datetime.now()
            
            await self._monitor_session(session)
            
            # Phase 6: Cleanup
            session.phase = SessionPhase.CLEANUP
            session.updated_at = datetime.now()
            
            await self._cleanup_session(session)
            
            # Mark as completed
            session.status = SessionStatus.COMPLETED
            session.end_time = datetime.now()
            if session.start_time:
                session.duration = int((session.end_time - session.start_time).total_seconds())
            session.updated_at = datetime.now()
            
            # Move to history
            with self._lock:
                self.session_history.append(session)
                if session.session_id in self.active_sessions:
                    del self.active_sessions[session.session_id]
            
            record_metric("sandbox.session_completed", 1, MetricType.COUNTER, {"scenario": session.scenario.name})
            record_event("sandbox.session_completed", f"Session {session.session_id} completed successfully")
            
            logger.info(f"Sandbox session {session.session_id} completed successfully")
            
        except asyncio.CancelledError:
            logger.info(f"Sandbox session {session.session_id} was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in sandbox session {session.session_id}: {e}")
            session.status = SessionStatus.ERROR
            session.updated_at = datetime.now()
            record_metric("sandbox.session_error", 1, MetricType.COUNTER, {"scenario": session.scenario.name})
            record_event("sandbox.session_error", f"Session {session.session_id} failed: {str(e)}")

    async def _initialize_session(self, session: SandboxSession) -> None:
        """Initialize a sandbox session."""
        logger.info(f"Initializing session {session.session_id}")
        
        # Add initialization event
        session.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "initialization",
            "message": "Session initialization started",
            "phase": session.phase.value
        })
        
        # Simulate initialization time
        await asyncio.sleep(1)
        
        session.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "initialization",
            "message": "Session initialization completed",
            "phase": session.phase.value
        })

    async def _setup_agents(self, session: SandboxSession) -> None:
        """Setup AI agents for the session."""
        logger.info(f"Setting up agents for session {session.session_id}")
        
        session.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "agent_setup",
            "message": f"Setting up {len(session.scenario.agents)} agents",
            "phase": session.phase.value
        })
        
        # Simulate agent setup
        for agent in session.scenario.agents:
            await asyncio.sleep(0.5)  # Simulate setup time
            session.agents[agent] = {
                "status": "active",
                "initialized_at": datetime.now().isoformat(),
                "type": agent
            }
            
            session.events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "agent_setup",
                "message": f"Agent {agent} initialized",
                "phase": session.phase.value
            })
        
        session.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "agent_setup",
            "message": "All agents setup completed",
            "phase": session.phase.value
        })

    async def _activate_guards(self, session: SandboxSession) -> None:
        """Activate security guards for the session."""
        logger.info(f"Activating guards for session {session.session_id}")
        
        session.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "guard_activation",
            "message": f"Activating {len(session.scenario.guards)} guards",
            "phase": session.phase.value
        })
        
        # Simulate guard activation
        for guard in session.scenario.guards:
            await asyncio.sleep(0.3)  # Simulate activation time
            session.guards[guard] = {
                "status": "active",
                "activated_at": datetime.now().isoformat(),
                "type": guard
            }
            
            session.events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "guard_activation",
                "message": f"Guard {guard} activated",
                "phase": session.phase.value
            })
        
        session.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "guard_activation",
            "message": "All guards activated",
            "phase": session.phase.value
        })

    async def _execute_scenario(self, session: SandboxSession) -> None:
        """Execute the main scenario logic."""
        logger.info(f"Executing scenario for session {session.session_id}")
        
        session.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "scenario_execution",
            "message": f"Starting scenario: {session.scenario.name}",
            "phase": session.phase.value
        })
        
        try:
            # Import scenario execution logic
            from safehive.sandbox.scenarios import create_scenario
            
            # Create scenario instance
            scenario = create_scenario(session.scenario.name)
            if scenario:
                # Create scenario context
                from safehive.sandbox.scenarios import ScenarioContext, ScenarioStep
                
                context = ScenarioContext(
                    session=session,
                    step=ScenarioStep.START,
                    data={},
                    interactions=[],
                    security_events=[],
                    metrics={}
                )
                
                # Execute scenario
                success = await scenario.execute(context)
                
                if success:
                    session.events.append({
                        "timestamp": datetime.now().isoformat(),
                        "type": "scenario_execution",
                        "message": "Scenario execution completed successfully",
                        "phase": session.phase.value
                    })
                    
                    # Add scenario data to session
                    session.metrics.update(context.metrics)
                    session.events.extend([
                        {
                            "timestamp": datetime.now().isoformat(),
                            "type": "scenario_interaction",
                            "message": f"Interaction: {interaction['type']}",
                            "data": interaction
                        }
                        for interaction in context.interactions
                    ])
                    session.events.extend([
                        {
                            "timestamp": datetime.now().isoformat(),
                            "type": "security_event",
                            "message": f"Security event: {event['type']}",
                            "data": event
                        }
                        for event in context.security_events
                    ])
                else:
                    session.events.append({
                        "timestamp": datetime.now().isoformat(),
                        "type": "scenario_execution",
                        "message": "Scenario execution failed",
                        "phase": session.phase.value
                    })
                    raise Exception("Scenario execution failed")
            else:
                # Fallback to simulation if scenario not found
                logger.warning(f"Scenario {session.scenario.name} not found, using simulation")
                await self._simulate_scenario_execution(session)
                
        except Exception as e:
            logger.error(f"Error executing scenario: {e}")
            session.events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "scenario_execution",
                "message": f"Scenario execution error: {str(e)}",
                "phase": session.phase.value
            })
            raise

    async def _simulate_scenario_execution(self, session: SandboxSession) -> None:
        """Simulate scenario execution as fallback."""
        execution_time = min(session.scenario.duration - 30, 60)  # Leave time for monitoring and cleanup
        steps = 10
        step_duration = execution_time / steps
        
        for step in range(steps):
            if session.status == SessionStatus.STOPPING:
                break
                
            await asyncio.sleep(step_duration)
            
            # Simulate scenario activity
            session.events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "scenario_execution",
                "message": f"Scenario step {step + 1}/{steps} completed",
                "phase": session.phase.value
            })
            
            # Update metrics
            session.metrics[f"step_{step + 1}"] = {
                "completed_at": datetime.now().isoformat(),
                "duration": step_duration
            }
        
        session.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "scenario_execution",
            "message": "Scenario simulation completed",
            "phase": session.phase.value
        })

    async def _monitor_session(self, session: SandboxSession) -> None:
        """Monitor the session for security events."""
        logger.info(f"Monitoring session {session.session_id}")
        
        session.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "monitoring",
            "message": "Starting session monitoring",
            "phase": session.phase.value
        })
        
        # Simulate monitoring
        await asyncio.sleep(10)
        
        session.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "monitoring",
            "message": "Session monitoring completed",
            "phase": session.phase.value
        })

    async def _cleanup_session(self, session: SandboxSession) -> None:
        """Cleanup session resources."""
        logger.info(f"Cleaning up session {session.session_id}")
        
        session.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "cleanup",
            "message": "Starting session cleanup",
            "phase": session.phase.value
        })
        
        # Simulate cleanup
        await asyncio.sleep(2)
        
        session.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "cleanup",
            "message": "Session cleanup completed",
            "phase": session.phase.value
        })

    def save_session_history(self, filepath: str) -> bool:
        """Save session history to file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with self._lock:
                history_data = [session.to_dict() for session in self.session_history]
            
            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            logger.info(f"Session history saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session history: {e}")
            return False

    def load_session_history(self, filepath: str) -> bool:
        """Load session history from file."""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return False
            
            with open(filepath, 'r') as f:
                history_data = json.load(f)
            
            with self._lock:
                self.session_history = [SandboxSession.from_dict(data) for data in history_data]
            
            logger.info(f"Session history loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load session history: {e}")
            return False


# Global sandbox manager instance
_global_sandbox_manager: Optional[SandboxManager] = None


def get_sandbox_manager() -> SandboxManager:
    """Get the global sandbox manager instance."""
    global _global_sandbox_manager
    if _global_sandbox_manager is None:
        _global_sandbox_manager = SandboxManager()
    return _global_sandbox_manager
