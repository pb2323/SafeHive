"""
Agent Configuration and Personality Management System

This module provides comprehensive configuration management for AI agents including
personality traits, behavior patterns, configuration validation, and dynamic updates.
"""

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Type, TypeVar
import yaml
import jsonschema
from jsonschema import validate, ValidationError

from ..utils.logger import get_logger
from ..utils.metrics import record_metric, increment_counter, MetricType
from ..models.agent_models import AgentType, AgentPersonality

logger = get_logger(__name__)

T = TypeVar('T')


class ConfigurationScope(Enum):
    """Scope of configuration changes."""
    GLOBAL = "global"
    AGENT_TYPE = "agent_type"
    AGENT_INSTANCE = "agent_instance"


class PersonalityTrait(Enum):
    """Available personality traits for agents."""
    # Communication traits
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CASUAL = "casual"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    
    # Behavior traits
    HELPFUL = "helpful"
    DECEPTIVE = "deceptive"
    HONEST = "honest"
    MANIPULATIVE = "manipulative"
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    
    # Intelligence traits
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    SYSTEMATIC = "systematic"
    INTUITIVE = "intuitive"
    
    # Risk traits
    CAUTIOUS = "cautious"
    BOLD = "bold"
    RECKLESS = "reckless"
    CALCULATED = "calculated"


class ResponseStyle(Enum):
    """Response style preferences."""
    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    SIMPLE = "simple"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"


@dataclass
class PersonalityProfile:
    """
    Comprehensive personality profile for an agent.
    
    This class defines the complete personality characteristics that
    influence an agent's behavior and responses.
    """
    # Core traits
    primary_traits: List[PersonalityTrait] = field(default_factory=list)
    secondary_traits: List[PersonalityTrait] = field(default_factory=list)
    
    # Communication preferences
    response_style: ResponseStyle = ResponseStyle.CONVERSATIONAL
    verbosity_level: int = 5  # 1-10 scale
    formality_level: int = 5  # 1-10 scale
    humor_level: int = 3  # 1-10 scale
    
    # Behavioral parameters
    cooperation_tendency: float = 0.7  # 0.0-1.0
    honesty_tendency: float = 0.8  # 0.0-1.0
    aggression_level: float = 0.2  # 0.0-1.0
    risk_tolerance: float = 0.5  # 0.0-1.0
    
    # Knowledge and expertise
    expertise_level: Dict[str, float] = field(default_factory=dict)  # domain -> level
    knowledge_breadth: float = 0.6  # 0.0-1.0
    learning_rate: float = 0.7  # 0.0-1.0
    
    # Response characteristics
    response_time_preference: float = 1.0  # seconds
    detail_preference: float = 0.6  # 0.0-1.0
    creativity_level: float = 0.5  # 0.0-1.0
    
    # Specialized behaviors
    attack_patterns: List[str] = field(default_factory=list)
    defense_strategies: List[str] = field(default_factory=list)
    negotiation_style: str = "collaborative"
    
    def get_trait_strength(self, trait: PersonalityTrait) -> float:
        """Get the strength of a specific trait (0.0-1.0)."""
        if trait in self.primary_traits:
            return 0.8
        elif trait in self.secondary_traits:
            return 0.5
        else:
            return 0.0
    
    def has_trait(self, trait: PersonalityTrait) -> bool:
        """Check if the agent has a specific trait."""
        return trait in self.primary_traits or trait in self.secondary_traits
    
    def add_trait(self, trait: PersonalityTrait, is_primary: bool = False) -> None:
        """Add a trait to the personality profile."""
        if is_primary:
            if trait not in self.primary_traits:
                self.primary_traits.append(trait)
            # Remove from secondary if present
            if trait in self.secondary_traits:
                self.secondary_traits.remove(trait)
        else:
            if trait not in self.secondary_traits and trait not in self.primary_traits:
                self.secondary_traits.append(trait)
    
    def remove_trait(self, trait: PersonalityTrait) -> None:
        """Remove a trait from the personality profile."""
        if trait in self.primary_traits:
            self.primary_traits.remove(trait)
        if trait in self.secondary_traits:
            self.secondary_traits.remove(trait)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert personality profile to dictionary."""
        return {
            "primary_traits": [trait.value for trait in self.primary_traits],
            "secondary_traits": [trait.value for trait in self.secondary_traits],
            "response_style": self.response_style.value,
            "verbosity_level": self.verbosity_level,
            "formality_level": self.formality_level,
            "humor_level": self.humor_level,
            "cooperation_tendency": self.cooperation_tendency,
            "honesty_tendency": self.honesty_tendency,
            "aggression_level": self.aggression_level,
            "risk_tolerance": self.risk_tolerance,
            "expertise_level": self.expertise_level,
            "knowledge_breadth": self.knowledge_breadth,
            "learning_rate": self.learning_rate,
            "response_time_preference": self.response_time_preference,
            "detail_preference": self.detail_preference,
            "creativity_level": self.creativity_level,
            "attack_patterns": self.attack_patterns,
            "defense_strategies": self.defense_strategies,
            "negotiation_style": self.negotiation_style
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalityProfile":
        """Create personality profile from dictionary."""
        return cls(
            primary_traits=[PersonalityTrait(t) for t in data.get("primary_traits", [])],
            secondary_traits=[PersonalityTrait(t) for t in data.get("secondary_traits", [])],
            response_style=ResponseStyle(data.get("response_style", "conversational")),
            verbosity_level=data.get("verbosity_level", 5),
            formality_level=data.get("formality_level", 5),
            humor_level=data.get("humor_level", 3),
            cooperation_tendency=data.get("cooperation_tendency", 0.7),
            honesty_tendency=data.get("honesty_tendency", 0.8),
            aggression_level=data.get("aggression_level", 0.2),
            risk_tolerance=data.get("risk_tolerance", 0.5),
            expertise_level=data.get("expertise_level", {}),
            knowledge_breadth=data.get("knowledge_breadth", 0.6),
            learning_rate=data.get("learning_rate", 0.7),
            response_time_preference=data.get("response_time_preference", 1.0),
            detail_preference=data.get("detail_preference", 0.6),
            creativity_level=data.get("creativity_level", 0.5),
            attack_patterns=data.get("attack_patterns", []),
            defense_strategies=data.get("defense_strategies", []),
            negotiation_style=data.get("negotiation_style", "collaborative")
        )


@dataclass
class AgentConfiguration:
    """
    Complete configuration for an AI agent.
    
    This class contains all configuration parameters needed to
    initialize and configure an agent instance.
    """
    # Basic identification
    agent_id: str
    agent_type: AgentType
    name: str
    description: str
    
    # Personality and behavior
    personality: PersonalityProfile
    
    # Model and AI settings
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # System settings
    max_retries: int = 3
    timeout_seconds: int = 30
    enable_memory: bool = True
    memory_type: str = "conversation_buffer"
    max_memory_size: int = 1000
    
    # Communication settings
    enable_communication: bool = True
    communication_timeout: int = 10
    max_message_queue_size: int = 100
    
    # Security and safety
    enable_safety_checks: bool = True
    safety_threshold: float = 0.8
    enable_content_filtering: bool = True
    
    # Performance settings
    enable_metrics: bool = True
    metrics_collection_interval: int = 60
    enable_profiling: bool = False
    response_time_preference: str = "balanced"  # fast, balanced, thorough
    
    # Tools and capabilities
    tools: List[Any] = field(default_factory=list)
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    def update_setting(self, key: str, value: Any) -> None:
        """Update a configuration setting."""
        if hasattr(self, key):
            setattr(self, key, value)
            self.updated_at = datetime.now()
        else:
            self.custom_settings[key] = value
            self.updated_at = datetime.now()
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a configuration setting."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.custom_settings.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        # Convert personality profile
        data["personality"] = self.personality.to_dict()
        # Convert enum values
        data["agent_type"] = self.agent_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfiguration":
        """Create configuration from dictionary."""
        # Separate known fields from custom settings
        known_fields = {
            'agent_id', 'agent_type', 'name', 'description', 'personality',
            'model_name', 'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty',
            'max_retries', 'timeout_seconds', 'enable_memory', 'memory_type', 'max_memory_size',
            'enable_communication', 'communication_timeout', 'max_message_queue_size',
            'enable_safety_checks', 'safety_threshold', 'enable_content_filtering',
            'enable_metrics', 'metrics_collection_interval', 'enable_profiling', 'response_time_preference',
            'custom_settings', 'created_at', 'updated_at', 'version'
        }
        
        # Extract custom settings
        custom_settings = data.get("custom_settings", {})
        for key, value in data.items():
            if key not in known_fields:
                custom_settings[key] = value
        
        # Prepare data for constructor
        config_data = data.copy()
        config_data["custom_settings"] = custom_settings
        
        # Remove custom fields from config_data to avoid passing them to constructor
        for key in list(config_data.keys()):
            if key not in known_fields:
                config_data.pop(key, None)
        
        # Convert datetime strings back to datetime objects
        config_data["created_at"] = datetime.fromisoformat(config_data.get("created_at", datetime.now().isoformat()))
        config_data["updated_at"] = datetime.fromisoformat(config_data.get("updated_at", datetime.now().isoformat()))
        
        # Convert personality profile
        personality_data = config_data.get("personality", {})
        if isinstance(personality_data, PersonalityProfile):
            config_data["personality"] = personality_data
        else:
            config_data["personality"] = PersonalityProfile.from_dict(personality_data)
        
        # Convert enum values
        config_data["agent_type"] = AgentType(config_data["agent_type"])
        
        return cls(**config_data)


class ConfigurationValidator(ABC):
    """Abstract base class for configuration validators."""
    
    @abstractmethod
    def validate(self, config: AgentConfiguration) -> List[str]:
        """
        Validate a configuration and return list of errors.
        
        Returns empty list if configuration is valid.
        """
        pass


class SchemaValidator(ConfigurationValidator):
    """JSON Schema-based configuration validator."""
    
    def __init__(self, schema_path: Optional[str] = None):
        self.schema_path = schema_path
        self.schema = self._load_schema()
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load validation schema."""
        if self.schema_path and os.path.exists(self.schema_path):
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        return self._get_default_schema()
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """Get default validation schema."""
        return {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "minLength": 1},
                "agent_type": {"type": "string"},
                "name": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "model_name": {"type": "string"},
                "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                "max_tokens": {"type": "integer", "minimum": 1},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "personality": {"type": "object"}
            },
            "required": ["agent_id", "agent_type", "name", "description"]
        }
    
    def validate(self, config: AgentConfiguration) -> List[str]:
        """Validate configuration using JSON schema."""
        errors = []
        try:
            config_dict = config.to_dict()
            validate(instance=config_dict, schema=self.schema)
        except ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        return errors


class BusinessRuleValidator(ConfigurationValidator):
    """Business rule-based configuration validator."""
    
    def validate(self, config: AgentConfiguration) -> List[str]:
        """Validate configuration using business rules."""
        errors = []
        
        # Check personality consistency
        if config.personality.aggression_level > 0.8 and config.personality.cooperation_tendency > 0.7:
            errors.append("High aggression and cooperation levels are contradictory")
        
        # Check model settings
        if config.temperature > 1.5 and config.personality.creativity_level < 0.3:
            errors.append("High temperature without creativity may lead to inconsistent responses")
        
        # Check timeout settings based on response time preference
        timeouts = {"fast": 5, "balanced": 15, "thorough": 30}
        preferred_timeout = timeouts.get(config.response_time_preference, 15)
        if config.timeout_seconds < preferred_timeout:
            errors.append(f"Timeout ({config.timeout_seconds}s) should be >= {preferred_timeout}s for {config.response_time_preference} preference")
        
        # Check memory settings
        if config.enable_memory and config.max_memory_size < 100:
            errors.append("Memory size too small for effective operation")
        
        return errors


class ConfigurationTemplate:
    """Template for creating agent configurations."""
    
    def __init__(self, name: str, description: str, config_data: Dict[str, Any]):
        self.name = name
        self.description = description
        self.config_data = config_data
    
    def create_configuration(self, agent_id: str, **overrides) -> AgentConfiguration:
        """Create a configuration from this template."""
        config_data = self.config_data.copy()
        config_data.update(overrides)
        config_data["agent_id"] = agent_id
        
        # Create personality profile if not present
        if "personality" not in config_data:
            config_data["personality"] = PersonalityProfile()
        elif isinstance(config_data["personality"], dict):
            config_data["personality"] = PersonalityProfile.from_dict(config_data["personality"])
        elif isinstance(config_data["personality"], PersonalityProfile):
            # Already a PersonalityProfile object, convert to dict for serialization
            config_data["personality"] = config_data["personality"].to_dict()
        
        return AgentConfiguration.from_dict(config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "config_data": self.config_data
        }


class ConfigurationManager:
    """
    Central manager for agent configurations and personalities.
    
    This class provides comprehensive configuration management including
    validation, templates, dynamic updates, and persistence.
    """
    
    def __init__(self, config_dir: str = "/tmp/safehive_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._configurations: Dict[str, AgentConfiguration] = {}
        self._templates: Dict[str, ConfigurationTemplate] = {}
        self._validators: List[ConfigurationValidator] = []
        self._update_callbacks: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        
        # Add default validators
        self._validators.append(SchemaValidator())
        self._validators.append(BusinessRuleValidator())
        
        # Load default templates
        self._load_default_templates()
        
        logger.info(f"ConfigurationManager initialized with config_dir: {self.config_dir}")
    
    def add_validator(self, validator: ConfigurationValidator) -> None:
        """Add a configuration validator."""
        with self._lock:
            self._validators.append(validator)
            logger.info(f"Added validator: {validator.__class__.__name__}")
    
    def remove_validator(self, validator: ConfigurationValidator) -> None:
        """Remove a configuration validator."""
        with self._lock:
            if validator in self._validators:
                self._validators.remove(validator)
                logger.info(f"Removed validator: {validator.__class__.__name__}")
    
    def validate_configuration(self, config: AgentConfiguration) -> List[str]:
        """Validate a configuration using all validators."""
        errors = []
        with self._lock:
            for validator in self._validators:
                try:
                    validator_errors = validator.validate(config)
                    errors.extend(validator_errors)
                except Exception as e:
                    errors.append(f"Validator {validator.__class__.__name__} error: {str(e)}")
        
        if errors:
            logger.warning(f"Configuration validation failed for {config.agent_id}: {errors}")
            record_metric("config.validation.failed", len(errors), MetricType.COUNTER, {"agent_id": config.agent_id})
        else:
            record_metric("config.validation.success", 1, MetricType.COUNTER, {"agent_id": config.agent_id})
        
        return errors
    
    def save_configuration(self, config: AgentConfiguration, validate: bool = True) -> bool:
        """Save a configuration to disk."""
        try:
            with self._lock:
                # Validate if requested
                if validate:
                    errors = self.validate_configuration(config)
                    if errors:
                        logger.error(f"Configuration validation failed for {config.agent_id}: {errors}")
                        return False
                
                # Save to memory
                self._configurations[config.agent_id] = config
                
                # Save to disk
                config_file = self.config_dir / f"{config.agent_id}.json"
                with open(config_file, 'w') as f:
                    json.dump(config.to_dict(), f, indent=2)
                
                logger.info(f"Saved configuration for agent {config.agent_id}")
                record_metric("config.save.success", 1, MetricType.COUNTER, {"agent_id": config.agent_id})
                return True
                
        except Exception as e:
            logger.error(f"Failed to save configuration for agent {config.agent_id}: {e}")
            record_metric("config.save.error", 1, MetricType.COUNTER, {"agent_id": config.agent_id, "error": str(e)})
            return False
    
    def load_configuration(self, agent_id: str) -> Optional[AgentConfiguration]:
        """Load a configuration from disk."""
        try:
            with self._lock:
                # Check memory first
                if agent_id in self._configurations:
                    return self._configurations[agent_id]
                
                # Load from disk
                config_file = self.config_dir / f"{agent_id}.json"
                if not config_file.exists():
                    logger.warning(f"Configuration file not found for agent {agent_id}")
                    return None
                
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                config = AgentConfiguration.from_dict(config_data)
                self._configurations[agent_id] = config
                
                logger.info(f"Loaded configuration for agent {agent_id}")
                record_metric("config.load.success", 1, MetricType.COUNTER, {"agent_id": agent_id})
                return config
                
        except Exception as e:
            logger.error(f"Failed to load configuration for agent {agent_id}: {e}")
            record_metric("config.load.error", 1, MetricType.COUNTER, {"agent_id": agent_id, "error": str(e)})
            return None
    
    def delete_configuration(self, agent_id: str) -> bool:
        """Delete a configuration."""
        try:
            with self._lock:
                config_existed = False
                
                # Remove from memory
                if agent_id in self._configurations:
                    del self._configurations[agent_id]
                    config_existed = True
                
                # Remove from disk
                config_file = self.config_dir / f"{agent_id}.json"
                if config_file.exists():
                    config_file.unlink()
                    config_existed = True
                
                if config_existed:
                    logger.info(f"Deleted configuration for agent {agent_id}")
                    record_metric("config.delete.success", 1, MetricType.COUNTER, {"agent_id": agent_id})
                    return True
                else:
                    logger.error(f"Configuration not found for agent {agent_id}")
                    record_metric("config.delete.not_found", 1, MetricType.COUNTER, {"agent_id": agent_id})
                    return False
                
        except Exception as e:
            logger.error(f"Failed to delete configuration for agent {agent_id}: {e}")
            record_metric("config.delete.error", 1, MetricType.COUNTER, {"agent_id": agent_id, "error": str(e)})
            return False
    
    def list_configurations(self) -> List[str]:
        """List all available configuration IDs."""
        with self._lock:
            return list(self._configurations.keys())
    
    def update_configuration(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """Update a configuration with new values."""
        try:
            with self._lock:
                if agent_id not in self._configurations:
                    logger.error(f"Configuration not found for agent {agent_id}")
                    return False
                
                config = self._configurations[agent_id]
                
                # Apply updates
                for key, value in updates.items():
                    if key == "personality" and isinstance(value, dict):
                        # Update personality profile
                        personality_updates = value
                        for pkey, pvalue in personality_updates.items():
                            if hasattr(config.personality, pkey):
                                setattr(config.personality, pkey, pvalue)
                    else:
                        config.update_setting(key, value)
                
                # Validate updated configuration
                errors = self.validate_configuration(config)
                if errors:
                    logger.error(f"Configuration update validation failed for {agent_id}: {errors}")
                    return False
                
                # Save updated configuration
                success = self.save_configuration(config, validate=False)
                
                if success:
                    # Notify callbacks
                    self._notify_update_callbacks(agent_id, updates)
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to update configuration for agent {agent_id}: {e}")
            record_metric("config.update.error", 1, MetricType.COUNTER, {"agent_id": agent_id, "error": str(e)})
            return False
    
    def add_update_callback(self, agent_id: str, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add a callback for configuration updates."""
        with self._lock:
            if agent_id not in self._update_callbacks:
                self._update_callbacks[agent_id] = []
            self._update_callbacks[agent_id].append(callback)
    
    def remove_update_callback(self, agent_id: str, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Remove a configuration update callback."""
        with self._lock:
            if agent_id in self._update_callbacks:
                if callback in self._update_callbacks[agent_id]:
                    self._update_callbacks[agent_id].remove(callback)
    
    def _notify_update_callbacks(self, agent_id: str, updates: Dict[str, Any]) -> None:
        """Notify all callbacks about configuration updates."""
        if agent_id in self._update_callbacks:
            for callback in self._update_callbacks[agent_id]:
                try:
                    callback(agent_id, updates)
                except Exception as e:
                    logger.error(f"Error in configuration update callback for {agent_id}: {e}")
    
    def add_template(self, template: ConfigurationTemplate) -> None:
        """Add a configuration template."""
        with self._lock:
            self._templates[template.name] = template
            logger.info(f"Added configuration template: {template.name}")
    
    def get_template(self, name: str) -> Optional[ConfigurationTemplate]:
        """Get a configuration template by name."""
        with self._lock:
            return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        with self._lock:
            return list(self._templates.keys())
    
    def create_configuration_from_template(self, template_name: str, agent_id: str, **overrides) -> Optional[AgentConfiguration]:
        """Create a configuration from a template."""
        template = self.get_template(template_name)
        if not template:
            logger.error(f"Template not found: {template_name}")
            return None
        
        try:
            config = template.create_configuration(agent_id, **overrides)
            logger.info(f"Created configuration from template {template_name} for agent {agent_id}")
            return config
        except Exception as e:
            logger.error(f"Failed to create configuration from template {template_name}: {e}")
            return None
    
    def _load_default_templates(self) -> None:
        """Load default configuration templates."""
        # Friendly helper template
        friendly_template = ConfigurationTemplate(
            name="friendly_helper",
            description="A friendly and helpful agent",
            config_data={
                "agent_type": "user_twin",
                "name": "Friendly Helper",
                "description": "A helpful and friendly AI assistant",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.7,
                "personality": {
                    "primary_traits": ["helpful", "friendly", "cooperative"],
                    "secondary_traits": ["analytical"],
                    "response_style": "conversational",
                    "verbosity_level": 6,
                    "formality_level": 4,
                    "cooperation_tendency": 0.9,
                    "honesty_tendency": 0.9
                }
            }
        )
        self.add_template(friendly_template)
        
        # Aggressive attacker template
        aggressive_template = ConfigurationTemplate(
            name="aggressive_attacker",
            description="An aggressive and deceptive agent",
            config_data={
                "agent_type": "malicious_vendor",
                "name": "Aggressive Attacker",
                "description": "An aggressive and deceptive AI agent",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.8,
                "personality": {
                    "primary_traits": ["aggressive", "deceptive", "manipulative"],
                    "secondary_traits": ["competitive"],
                    "response_style": "casual",
                    "verbosity_level": 7,
                    "formality_level": 2,
                    "aggression_level": 0.8,
                    "honesty_tendency": 0.3,
                    "attack_patterns": ["social_engineering", "phishing", "manipulation"]
                }
            }
        )
        self.add_template(aggressive_template)
        
        # Orchestrator template
        orchestrator_template = ConfigurationTemplate(
            name="orchestrator",
            description="A system orchestrator agent",
            config_data={
                "agent_type": "orchestrator",
                "name": "System Orchestrator",
                "description": "Coordinates system operations and agent interactions",
                "model_name": "gpt-4",
                "temperature": 0.3,
                "personality": {
                    "primary_traits": ["analytical", "systematic", "cooperative"],
                    "secondary_traits": ["formal"],
                    "response_style": "professional",
                    "verbosity_level": 5,
                    "formality_level": 8,
                    "cooperation_tendency": 0.8,
                    "honesty_tendency": 0.9
                }
            }
        )
        self.add_template(orchestrator_template)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration manager statistics."""
        with self._lock:
            return {
                "total_configurations": len(self._configurations),
                "total_templates": len(self._templates),
                "total_validators": len(self._validators),
                "config_directory": str(self.config_dir)
            }


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_configuration_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def create_configuration_manager(config_dir: str = "/tmp/safehive_configs") -> ConfigurationManager:
    """Create a new configuration manager instance."""
    return ConfigurationManager(config_dir)
