"""
Unit tests for the configuration loader.

This module tests the configuration loading, validation, and management
functionality of the SafeHive system.
"""

import pytest
import tempfile
import os
import yaml
from pathlib import Path

from safehive.config.config_loader import (
    ConfigLoader, SystemConfig, GuardConfig, AgentConfig, LoggingConfig,
    get_config_loader, load_config
)


class TestConfigLoader:
    """Test the ConfigLoader class functionality."""
    
    def test_config_loader_initialization(self):
        """Test ConfigLoader initialization."""
        loader = ConfigLoader()
        assert loader.config_path is not None
        assert loader.config is None
        assert loader._validation_schema is not None
    
    def test_config_loader_with_custom_path(self):
        """Test ConfigLoader with custom config path."""
        custom_path = "custom_config.yaml"
        loader = ConfigLoader(custom_path)
        assert loader.config_path == custom_path
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        loader = ConfigLoader()
        config = loader._create_default_config()
        
        assert isinstance(config, SystemConfig)
        assert len(config.guards) > 0
        assert len(config.agents) > 0
        assert isinstance(config.logging, LoggingConfig)
        assert "loaded_at" in config.metadata
    
    def test_config_validation_valid_data(self):
        """Test configuration validation with valid data."""
        loader = ConfigLoader()
        valid_config = {
            "guards": {
                "honeypot": {
                    "enabled": True,
                    "threshold": 5,
                    "attacks": ["SQLi", "XSS"]
                }
            },
            "agents": {
                "orchestrator": {
                    "ai_model": "llama2:7b",
                    "max_retries": 3
                }
            },
            "logging": {
                "level": "INFO",
                "file": "test.log"
            }
        }
        
        # Should not raise any exceptions
        loader._validate_config(valid_config)
    
    def test_config_validation_invalid_data(self):
        """Test configuration validation with invalid data."""
        loader = ConfigLoader()
        
        # Test invalid threshold (too high)
        invalid_config = {
            "guards": {
                "honeypot": {
                    "threshold": 150  # Should be max 100
                }
            }
        }
        
        with pytest.raises(ValueError, match="above maximum"):
            loader._validate_config(invalid_config)
        
        # Test invalid log level
        invalid_config = {
            "logging": {
                "level": "INVALID_LEVEL"
            }
        }
        
        with pytest.raises(ValueError, match="Must be one of"):
            loader._validate_config(invalid_config)
    
    def test_config_conversion_to_system_config(self):
        """Test conversion of config data to SystemConfig object."""
        loader = ConfigLoader()
        config_data = {
            "guards": {
                "honeypot": {
                    "enabled": True,
                    "threshold": 3,
                    "attacks": ["SQLi", "XSS"],
                    "alert_stakeholders": True
                }
            },
            "agents": {
                "orchestrator": {
                    "ai_model": "llama2:7b",
                    "max_retries": 3,
                    "timeout_seconds": 30,
                    "memory_type": "conversation_buffer",
                    "tools": ["order_management"]
                }
            },
            "logging": {
                "level": "INFO",
                "file": "test.log",
                "structured": True
            }
        }
        
        system_config = loader._convert_to_system_config(config_data)
        
        assert isinstance(system_config, SystemConfig)
        assert "honeypot" in system_config.guards
        assert system_config.guards["honeypot"].enabled is True
        assert system_config.guards["honeypot"].threshold == 3
        assert "SQLi" in system_config.guards["honeypot"].patterns
        
        assert "orchestrator" in system_config.agents
        assert system_config.agents["orchestrator"].ai_model == "llama2:7b"
        assert system_config.agents["orchestrator"].max_retries == 3
        
        assert system_config.logging.level == "INFO"
        assert system_config.logging.file == "test.log"
        assert system_config.logging.structured is True
    
    def test_load_config_from_file(self):
        """Test loading configuration from a YAML file."""
        # Create temporary config file
        config_data = {
            "guards": {
                "honeypot": {
                    "enabled": True,
                    "threshold": 5,
                    "attacks": ["SQLi", "XSS", "PathTraversal"]
                }
            },
            "agents": {
                "orchestrator": {
                    "ai_model": "llama2:7b",
                    "max_retries": 3
                }
            },
            "logging": {
                "level": "DEBUG",
                "file": "test.log"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            config = loader.load_config()
            
            assert isinstance(config, SystemConfig)
            assert config.guards["honeypot"].threshold == 5
            assert config.logging.level == "DEBUG"
            assert config.metadata["config_path"] == temp_path
            
        finally:
            os.unlink(temp_path)
    
    def test_load_config_file_not_found(self):
        """Test loading configuration when file doesn't exist."""
        loader = ConfigLoader("nonexistent_config.yaml")
        config = loader.load_config()
        
        # Should return default config
        assert isinstance(config, SystemConfig)
        assert config.metadata["config_path"] == "default"
    
    def test_load_config_invalid_yaml(self):
        """Test loading configuration with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            with pytest.raises(yaml.YAMLError):
                loader.load_config()
        finally:
            os.unlink(temp_path)
    
    def test_get_guard_config(self):
        """Test getting specific guard configuration."""
        loader = ConfigLoader()
        config = loader.load_config()
        
        honeypot_config = loader.get_guard_config("honeypot")
        assert isinstance(honeypot_config, GuardConfig)
        assert honeypot_config.enabled is True
        
        # Test non-existent guard
        nonexistent_config = loader.get_guard_config("nonexistent")
        assert nonexistent_config is None
    
    def test_get_agent_config(self):
        """Test getting specific agent configuration."""
        loader = ConfigLoader()
        config = loader.load_config()
        
        orchestrator_config = loader.get_agent_config("orchestrator")
        assert isinstance(orchestrator_config, AgentConfig)
        assert orchestrator_config.ai_model == "llama2:7b"
        
        # Test non-existent agent
        nonexistent_config = loader.get_agent_config("nonexistent")
        assert nonexistent_config is None
    
    def test_get_logging_config(self):
        """Test getting logging configuration."""
        loader = ConfigLoader()
        config = loader.load_config()
        
        logging_config = loader.get_logging_config()
        assert isinstance(logging_config, LoggingConfig)
        assert logging_config.level == "INFO"
    
    def test_save_config(self):
        """Test saving configuration to file."""
        loader = ConfigLoader()
        config = loader.load_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            loader.save_config(config, temp_path)
            
            # Verify file was created and contains valid YAML
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                saved_data = yaml.safe_load(f)
            
            assert "guards" in saved_data
            assert "agents" in saved_data
            assert "logging" in saved_data
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_reload_config(self):
        """Test reloading configuration."""
        loader = ConfigLoader()
        config1 = loader.load_config()
        
        # Modify config and reload
        config2 = loader.reload_config()
        
        assert isinstance(config2, SystemConfig)
        # Should be the same since we're using default config
        assert config2.metadata["loaded_at"] != config1.metadata["loaded_at"]


class TestConfigDataClasses:
    """Test the configuration data classes."""
    
    def test_guard_config(self):
        """Test GuardConfig data class."""
        config = GuardConfig(
            enabled=True,
            threshold=5,
            patterns=["SQLi", "XSS"],
            settings={"test": "value"}
        )
        
        assert config.enabled is True
        assert config.threshold == 5
        assert config.patterns == ["SQLi", "XSS"]
        assert config.settings == {"test": "value"}
    
    def test_agent_config(self):
        """Test AgentConfig data class."""
        config = AgentConfig(
            ai_model="llama2:7b",
            max_retries=3,
            timeout_seconds=30,
            memory_type="conversation_buffer",
            personality="test_personality",
            constraints=["budget"],
            tools=["test_tool"],
            settings={"test": "value"}
        )
        
        assert config.ai_model == "llama2:7b"
        assert config.max_retries == 3
        assert config.timeout_seconds == 30
        assert config.memory_type == "conversation_buffer"
        assert config.personality == "test_personality"
        assert config.constraints == ["budget"]
        assert config.tools == ["test_tool"]
        assert config.settings == {"test": "value"}
    
    def test_logging_config(self):
        """Test LoggingConfig data class."""
        config = LoggingConfig(
            level="DEBUG",
            file="test.log",
            alerts_file="alerts.log",
            agent_conversations="conversations.log",
            structured=True,
            max_file_size="5MB",
            backup_count=3
        )
        
        assert config.level == "DEBUG"
        assert config.file == "test.log"
        assert config.alerts_file == "alerts.log"
        assert config.agent_conversations == "conversations.log"
        assert config.structured is True
        assert config.max_file_size == "5MB"
        assert config.backup_count == 3
    
    def test_system_config(self):
        """Test SystemConfig data class."""
        guards = {"test_guard": GuardConfig()}
        agents = {"test_agent": AgentConfig()}
        logging = LoggingConfig()
        metadata = {"test": "value"}
        
        config = SystemConfig(
            guards=guards,
            agents=agents,
            logging=logging,
            metadata=metadata
        )
        
        assert config.guards == guards
        assert config.agents == agents
        assert config.logging == logging
        assert config.metadata == metadata


class TestGlobalConfigFunctions:
    """Test global configuration functions."""
    
    def test_get_config_loader(self):
        """Test getting global config loader."""
        loader1 = get_config_loader()
        loader2 = get_config_loader()
        
        # Should return the same instance
        assert loader1 is loader2
        assert isinstance(loader1, ConfigLoader)
    
    def test_load_config_global(self):
        """Test global load_config function."""
        config = load_config()
        assert isinstance(config, SystemConfig)


if __name__ == "__main__":
    pytest.main([__file__])
