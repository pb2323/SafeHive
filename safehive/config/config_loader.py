"""
Configuration Loader for SafeHive AI Security Sandbox

This module handles loading, validation, and management of YAML configuration files
for the SafeHive system. It provides a centralized way to manage all system settings
including guard configurations, agent settings, and runtime parameters.
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GuardConfig:
    """Configuration for individual security guards."""
    enabled: bool = True
    threshold: int = 3
    patterns: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Configuration for AI agents."""
    ai_model: str = "llama2:7b"
    max_retries: int = 3
    timeout_seconds: int = 30
    memory_type: str = "conversation_buffer"
    personality: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    level: str = "INFO"
    file: str = "logs/sandbox.log"
    alerts_file: str = "logs/alerts.log"
    agent_conversations: str = "logs/agent_conversations.log"
    structured: bool = True
    max_file_size: str = "10MB"
    backup_count: int = 5


@dataclass
class SystemConfig:
    """Main system configuration container."""
    guards: Dict[str, GuardConfig] = field(default_factory=dict)
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigLoader:
    """
    Configuration loader and validator for SafeHive system.
    
    This class handles loading YAML configuration files, validating their structure,
    and providing access to configuration data throughout the system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Optional[SystemConfig] = None
        self._validation_schema = self._create_validation_schema()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        # Look for config in current directory, then in safehive/config/
        possible_paths = [
            "config.yaml",
            "safehive/config/default_config.yaml",
            "safehive/config/config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return default path if none found
        return "safehive/config/default_config.yaml"
    
    def _create_validation_schema(self) -> Dict[str, Any]:
        """Create validation schema for configuration validation."""
        return {
            "guards": {
                "type": "dict",
                "required": False,
                "keys": {
                    "honeypot": {
                        "type": "dict",
                        "required": False,
                        "keys": {
                            "enabled": {"type": "bool", "required": False, "default": True},
                            "threshold": {"type": "int", "required": False, "default": 3, "min": 1, "max": 100},
                            "attacks": {"type": "list", "required": False, "default": ["SQLi", "XSS", "PathTraversal"]},
                            "decoy_data_types": {"type": "list", "required": False, "default": ["credit_cards", "order_history"]},
                            "alert_stakeholders": {"type": "bool", "required": False, "default": True}
                        }
                    },
                    "privacy_sentry": {
                        "type": "dict",
                        "required": False,
                        "keys": {
                            "enabled": {"type": "bool", "required": False, "default": True},
                            "pii_patterns": {"type": "list", "required": False, "default": ["credit_card", "ssn", "address", "phone"]},
                            "redaction_method": {"type": "str", "required": False, "default": "mask", "choices": ["mask", "remove", "replace"]}
                        }
                    },
                    "task_navigator": {
                        "type": "dict",
                        "required": False,
                        "keys": {
                            "enabled": {"type": "bool", "required": False, "default": True},
                            "constraint_types": {"type": "list", "required": False, "default": ["budget", "dietary", "quantity"]},
                            "enforcement_level": {"type": "str", "required": False, "default": "strict", "choices": ["strict", "moderate", "lenient"]}
                        }
                    },
                    "prompt_sanitizer": {
                        "type": "dict",
                        "required": False,
                        "keys": {
                            "enabled": {"type": "bool", "required": False, "default": True},
                            "malicious_patterns": {"type": "list", "required": False, "default": ["injection", "manipulation", "social_engineering"]},
                            "sanitization_level": {"type": "str", "required": False, "default": "aggressive", "choices": ["aggressive", "moderate", "conservative"]}
                        }
                    }
                }
            },
            "agents": {
                "type": "dict",
                "required": False,
                "keys": {
                    "orchestrator": {
                        "type": "dict",
                        "required": False,
                        "keys": {
                            "ai_model": {"type": "str", "required": False, "default": "llama2:7b"},
                            "max_retries": {"type": "int", "required": False, "default": 3, "min": 1, "max": 10},
                            "timeout_seconds": {"type": "int", "required": False, "default": 30, "min": 5, "max": 300},
                            "memory_type": {"type": "str", "required": False, "default": "conversation_buffer"},
                            "tools": {"type": "list", "required": False, "default": ["order_management", "vendor_communication"]}
                        }
                    },
                    "user_twin": {
                        "type": "dict",
                        "required": False,
                        "keys": {
                            "ai_model": {"type": "str", "required": False, "default": "llama2:7b"},
                            "memory_type": {"type": "str", "required": False, "default": "conversation_summary"},
                            "personality": {"type": "str", "required": False, "default": "budget_conscious_vegetarian"},
                            "constraints": {"type": "list", "required": False, "default": ["budget_limit", "dietary_restrictions"]}
                        }
                    },
                    "vendors": {
                        "type": "dict",
                        "required": False,
                        "keys": {
                            "honest_vendor": {
                                "type": "dict",
                                "required": False,
                                "keys": {
                                    "ai_model": {"type": "str", "required": False, "default": "llama2:7b"},
                                    "personality": {"type": "str", "required": False, "default": "friendly_restaurant_owner"},
                                    "memory_type": {"type": "str", "required": False, "default": "conversation_buffer"},
                                    "tools": {"type": "list", "required": False, "default": ["menu_lookup", "pricing", "inventory"]}
                                }
                            },
                            "malicious_vendor": {
                                "type": "dict",
                                "required": False,
                                "keys": {
                                    "ai_model": {"type": "str", "required": False, "default": "llama2:7b"},
                                    "personality": {"type": "str", "required": False, "default": "aggressive_upseller"},
                                    "memory_type": {"type": "str", "required": False, "default": "conversation_buffer"},
                                    "attack_behaviors": {"type": "list", "required": False, "default": ["social_engineering", "technical_attacks"]},
                                    "tools": {"type": "list", "required": False, "default": ["menu_lookup", "pricing", "inventory", "attack_patterns"]}
                                }
                            }
                        }
                    }
                }
            },
            "logging": {
                "type": "dict",
                "required": False,
                "keys": {
                    "level": {"type": "str", "required": False, "default": "INFO", "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                    "file": {"type": "str", "required": False, "default": "logs/sandbox.log"},
                    "alerts_file": {"type": "str", "required": False, "default": "logs/alerts.log"},
                    "agent_conversations": {"type": "str", "required": False, "default": "logs/agent_conversations.log"},
                    "structured": {"type": "bool", "required": False, "default": True},
                    "max_file_size": {"type": "str", "required": False, "default": "10MB"},
                    "backup_count": {"type": "int", "required": False, "default": 5, "min": 1, "max": 20}
                }
            }
        }
    
    def load_config(self) -> SystemConfig:
        """
        Load and validate configuration from YAML file.
        
        Returns:
            SystemConfig object with loaded configuration
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration validation fails
        """
        logger.info(f"Loading configuration from: {self.config_path}")
        
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found: {self.config_path}")
            logger.info("Using default configuration")
            self.config = self._create_default_config()
            return self.config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
            
            if config_data is None:
                logger.warning("Configuration file is empty, using defaults")
                self.config = self._create_default_config()
                return self.config
            
            # Validate configuration
            self._validate_config(config_data)
            
            # Convert to SystemConfig object
            self.config = self._convert_to_system_config(config_data)
            
            logger.info("Configuration loaded successfully")
            return self.config
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Configuration loading error: {e}")
            raise
    
    def _validate_config(self, config_data: Dict[str, Any]) -> None:
        """
        Validate configuration data against schema.
        
        Args:
            config_data: Configuration data to validate
            
        Raises:
            ValueError: If validation fails
        """
        logger.debug("Validating configuration data")
        
        # Validate top-level structure
        for section_name, section_schema in self._validation_schema.items():
            if section_name in config_data:
                self._validate_section(config_data[section_name], section_schema, section_name)
        
        logger.debug("Configuration validation completed successfully")
    
    def _validate_section(self, data: Any, schema: Dict[str, Any], path: str) -> None:
        """
        Validate a configuration section.
        
        Args:
            data: Data to validate
            schema: Schema to validate against
            path: Current path in configuration (for error messages)
        """
        if schema.get("type") == "dict":
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict at {path}, got {type(data).__name__}")
            
            # Validate required keys
            required_keys = [k for k, v in schema.get("keys", {}).items() if v.get("required", False)]
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Required key '{key}' missing at {path}")
            
            # Validate each key
            for key, value in data.items():
                if key in schema.get("keys", {}):
                    key_schema = schema["keys"][key]
                    self._validate_section(value, key_schema, f"{path}.{key}")
        
        elif schema.get("type") == "list":
            if not isinstance(data, list):
                raise ValueError(f"Expected list at {path}, got {type(data).__name__}")
        
        elif schema.get("type") == "str":
            if not isinstance(data, str):
                raise ValueError(f"Expected string at {path}, got {type(data).__name__}")
            
            # Validate choices
            choices = schema.get("choices")
            if choices and data not in choices:
                raise ValueError(f"Invalid value '{data}' at {path}. Must be one of: {choices}")
        
        elif schema.get("type") == "int":
            if not isinstance(data, int):
                raise ValueError(f"Expected integer at {path}, got {type(data).__name__}")
            
            # Validate min/max
            min_val = schema.get("min")
            max_val = schema.get("max")
            if min_val is not None and data < min_val:
                raise ValueError(f"Value {data} at {path} is below minimum {min_val}")
            if max_val is not None and data > max_val:
                raise ValueError(f"Value {data} at {path} is above maximum {max_val}")
        
        elif schema.get("type") == "bool":
            if not isinstance(data, bool):
                raise ValueError(f"Expected boolean at {path}, got {type(data).__name__}")
    
    def _convert_to_system_config(self, config_data: Dict[str, Any]) -> SystemConfig:
        """
        Convert validated configuration data to SystemConfig object.
        
        Args:
            config_data: Validated configuration data
            
        Returns:
            SystemConfig object
        """
        # Convert guards configuration
        guards = {}
        guards_data = config_data.get("guards", {})
        for guard_name, guard_data in guards_data.items():
            guards[guard_name] = GuardConfig(
                enabled=guard_data.get("enabled", True),
                threshold=guard_data.get("threshold", 3),
                patterns=guard_data.get("attacks", guard_data.get("patterns", [])),
                settings=guard_data
            )
        
        # Convert agents configuration
        agents = {}
        agents_data = config_data.get("agents", {})
        for agent_name, agent_data in agents_data.items():
            if agent_name == "vendors":
                # Handle vendor agents specially
                for vendor_name, vendor_data in agent_data.items():
                    agents[f"{agent_name}_{vendor_name}"] = AgentConfig(
                        ai_model=vendor_data.get("ai_model", "llama2:7b"),
                        max_retries=vendor_data.get("max_retries", 3),
                        timeout_seconds=vendor_data.get("timeout_seconds", 30),
                        memory_type=vendor_data.get("memory_type", "conversation_buffer"),
                        personality=vendor_data.get("personality"),
                        constraints=vendor_data.get("constraints", []),
                        tools=vendor_data.get("tools", []),
                        settings=vendor_data
                    )
            else:
                agents[agent_name] = AgentConfig(
                    ai_model=agent_data.get("ai_model", "llama2:7b"),
                    max_retries=agent_data.get("max_retries", 3),
                    timeout_seconds=agent_data.get("timeout_seconds", 30),
                    memory_type=agent_data.get("memory_type", "conversation_buffer"),
                    personality=agent_data.get("personality"),
                    constraints=agent_data.get("constraints", []),
                    tools=agent_data.get("tools", []),
                    settings=agent_data
                )
        
        # Convert logging configuration
        logging_data = config_data.get("logging", {})
        logging_config = LoggingConfig(
            level=logging_data.get("level", "INFO"),
            file=logging_data.get("file", "logs/sandbox.log"),
            alerts_file=logging_data.get("alerts_file", "logs/alerts.log"),
            agent_conversations=logging_data.get("agent_conversations", "logs/agent_conversations.log"),
            structured=logging_data.get("structured", True),
            max_file_size=logging_data.get("max_file_size", "10MB"),
            backup_count=logging_data.get("backup_count", 5)
        )
        
        return SystemConfig(
            guards=guards,
            agents=agents,
            logging=logging_config,
            metadata={
                "loaded_at": datetime.now().isoformat(),
                "config_path": self.config_path,
                "version": "0.1.0"
            }
        )
    
    def _create_default_config(self) -> SystemConfig:
        """
        Create default configuration when no config file is found.
        
        Returns:
            SystemConfig with default values
        """
        logger.info("Creating default configuration")
        
        # Default guards configuration
        guards = {
            "honeypot": GuardConfig(
                enabled=True,
                threshold=3,
                patterns=["SQLi", "XSS", "PathTraversal"],
                settings={"alert_stakeholders": True}
            ),
            "privacy_sentry": GuardConfig(
                enabled=True,
                patterns=["credit_card", "ssn", "address", "phone"],
                settings={"redaction_method": "mask"}
            ),
            "task_navigator": GuardConfig(
                enabled=True,
                patterns=["budget", "dietary", "quantity"],
                settings={"enforcement_level": "strict"}
            ),
            "prompt_sanitizer": GuardConfig(
                enabled=True,
                patterns=["injection", "manipulation", "social_engineering"],
                settings={"sanitization_level": "aggressive"}
            )
        }
        
        # Default agents configuration
        agents = {
            "orchestrator": AgentConfig(
                ai_model="llama2:7b",
                max_retries=3,
                timeout_seconds=30,
                memory_type="conversation_buffer",
                tools=["order_management", "vendor_communication"]
            ),
            "user_twin": AgentConfig(
                ai_model="llama2:7b",
                memory_type="conversation_summary",
                personality="budget_conscious_vegetarian",
                constraints=["budget_limit", "dietary_restrictions"]
            ),
            "vendors_honest_vendor": AgentConfig(
                ai_model="llama2:7b",
                personality="friendly_restaurant_owner",
                memory_type="conversation_buffer",
                tools=["menu_lookup", "pricing", "inventory"]
            ),
            "vendors_malicious_vendor": AgentConfig(
                ai_model="llama2:7b",
                personality="aggressive_upseller",
                memory_type="conversation_buffer",
                tools=["menu_lookup", "pricing", "inventory", "attack_patterns"]
            )
        }
        
        # Default logging configuration
        logging_config = LoggingConfig()
        
        return SystemConfig(
            guards=guards,
            agents=agents,
            logging=logging_config,
            metadata={
                "loaded_at": datetime.now().isoformat(),
                "config_path": "default",
                "version": "0.1.0"
            }
        )
    
    def get_guard_config(self, guard_name: str) -> Optional[GuardConfig]:
        """
        Get configuration for a specific guard.
        
        Args:
            guard_name: Name of the guard
            
        Returns:
            GuardConfig object or None if not found
        """
        if self.config is None:
            self.load_config()
        
        return self.config.guards.get(guard_name)
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """
        Get configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            AgentConfig object or None if not found
        """
        if self.config is None:
            self.load_config()
        
        return self.config.agents.get(agent_name)
    
    def get_logging_config(self) -> LoggingConfig:
        """
        Get logging configuration.
        
        Returns:
            LoggingConfig object
        """
        if self.config is None:
            self.load_config()
        
        return self.config.logging
    
    def reload_config(self) -> SystemConfig:
        """
        Reload configuration from file.
        
        Returns:
            Updated SystemConfig object
        """
        logger.info("Reloading configuration")
        return self.load_config()
    
    def save_config(self, config: SystemConfig, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: SystemConfig object to save
            path: Path to save to (uses current config_path if None)
        """
        save_path = path or self.config_path
        logger.info(f"Saving configuration to: {save_path}")
        
        # Convert SystemConfig to dictionary
        config_dict = {
            "guards": {
                name: {
                    "enabled": guard.enabled,
                    "threshold": guard.threshold,
                    "patterns": guard.patterns,
                    **guard.settings
                }
                for name, guard in config.guards.items()
            },
            "agents": {
                name: {
                    "ai_model": agent.ai_model,
                    "max_retries": agent.max_retries,
                    "timeout_seconds": agent.timeout_seconds,
                    "memory_type": agent.memory_type,
                    "personality": agent.personality,
                    "constraints": agent.constraints,
                    "tools": agent.tools,
                    **agent.settings
                }
                for name, agent in config.agents.items()
            },
            "logging": {
                "level": config.logging.level,
                "file": config.logging.file,
                "alerts_file": config.logging.alerts_file,
                "agent_conversations": config.logging.agent_conversations,
                "structured": config.logging.structured,
                "max_file_size": config.logging.max_file_size,
                "backup_count": config.logging.backup_count
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to file
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)
        
        logger.info("Configuration saved successfully")


# Global configuration loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """
    Get the global configuration loader instance.
    
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config() -> SystemConfig:
    """
    Load configuration using the global loader.
    
    Returns:
        SystemConfig object
    """
    return get_config_loader().load_config()
