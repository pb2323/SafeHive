"""
Unit tests for the agent configuration and personality management system.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from safehive.agents.configuration import (
    PersonalityProfile, AgentConfiguration, ConfigurationManager,
    ConfigurationTemplate, SchemaValidator, BusinessRuleValidator,
    PersonalityTrait, ResponseStyle, ConfigurationScope,
    get_configuration_manager, create_configuration_manager
)
from safehive.models.agent_models import AgentType


class TestPersonalityProfile:
    """Test PersonalityProfile class."""
    
    def test_personality_profile_creation(self):
        """Test creating a personality profile."""
        profile = PersonalityProfile()
        
        assert profile.primary_traits == []
        assert profile.secondary_traits == []
        assert profile.response_style == ResponseStyle.CONVERSATIONAL
        assert profile.verbosity_level == 5
        assert profile.formality_level == 5
        assert profile.cooperation_tendency == 0.7
        assert profile.honesty_tendency == 0.8
        assert profile.aggression_level == 0.2
    
    def test_personality_profile_with_traits(self):
        """Test creating a personality profile with traits."""
        profile = PersonalityProfile(
            primary_traits=[PersonalityTrait.FRIENDLY, PersonalityTrait.HELPFUL],
            secondary_traits=[PersonalityTrait.ANALYTICAL],
            response_style=ResponseStyle.PROFESSIONAL,
            verbosity_level=7,
            cooperation_tendency=0.9
        )
        
        assert PersonalityTrait.FRIENDLY in profile.primary_traits
        assert PersonalityTrait.HELPFUL in profile.primary_traits
        assert PersonalityTrait.ANALYTICAL in profile.secondary_traits
        assert profile.response_style == ResponseStyle.PROFESSIONAL
        assert profile.verbosity_level == 7
        assert profile.cooperation_tendency == 0.9
    
    def test_get_trait_strength(self):
        """Test getting trait strength."""
        profile = PersonalityProfile(
            primary_traits=[PersonalityTrait.FRIENDLY],
            secondary_traits=[PersonalityTrait.ANALYTICAL]
        )
        
        assert profile.get_trait_strength(PersonalityTrait.FRIENDLY) == 0.8
        assert profile.get_trait_strength(PersonalityTrait.ANALYTICAL) == 0.5
        assert profile.get_trait_strength(PersonalityTrait.AGGRESSIVE) == 0.0
    
    def test_has_trait(self):
        """Test checking if profile has a trait."""
        profile = PersonalityProfile(
            primary_traits=[PersonalityTrait.FRIENDLY],
            secondary_traits=[PersonalityTrait.ANALYTICAL]
        )
        
        assert profile.has_trait(PersonalityTrait.FRIENDLY) is True
        assert profile.has_trait(PersonalityTrait.ANALYTICAL) is True
        assert profile.has_trait(PersonalityTrait.AGGRESSIVE) is False
    
    def test_add_trait(self):
        """Test adding traits to profile."""
        profile = PersonalityProfile()
        
        # Add primary trait
        profile.add_trait(PersonalityTrait.FRIENDLY, is_primary=True)
        assert PersonalityTrait.FRIENDLY in profile.primary_traits
        assert PersonalityTrait.FRIENDLY not in profile.secondary_traits
        
        # Add secondary trait
        profile.add_trait(PersonalityTrait.ANALYTICAL, is_primary=False)
        assert PersonalityTrait.ANALYTICAL in profile.secondary_traits
        assert PersonalityTrait.ANALYTICAL not in profile.primary_traits
        
        # Move trait from secondary to primary
        profile.add_trait(PersonalityTrait.ANALYTICAL, is_primary=True)
        assert PersonalityTrait.ANALYTICAL in profile.primary_traits
        assert PersonalityTrait.ANALYTICAL not in profile.secondary_traits
    
    def test_remove_trait(self):
        """Test removing traits from profile."""
        profile = PersonalityProfile(
            primary_traits=[PersonalityTrait.FRIENDLY],
            secondary_traits=[PersonalityTrait.ANALYTICAL]
        )
        
        # Remove primary trait
        profile.remove_trait(PersonalityTrait.FRIENDLY)
        assert PersonalityTrait.FRIENDLY not in profile.primary_traits
        
        # Remove secondary trait
        profile.remove_trait(PersonalityTrait.ANALYTICAL)
        assert PersonalityTrait.ANALYTICAL not in profile.secondary_traits
        
        # Remove non-existent trait (should not error)
        profile.remove_trait(PersonalityTrait.AGGRESSIVE)
    
    def test_personality_profile_serialization(self):
        """Test personality profile serialization and deserialization."""
        original_profile = PersonalityProfile(
            primary_traits=[PersonalityTrait.FRIENDLY, PersonalityTrait.HELPFUL],
            secondary_traits=[PersonalityTrait.ANALYTICAL],
            response_style=ResponseStyle.PROFESSIONAL,
            verbosity_level=7,
            formality_level=8,
            cooperation_tendency=0.9,
            honesty_tendency=0.8,
            expertise_level={"security": 0.8, "networking": 0.6}
        )
        
        # Convert to dict
        data = original_profile.to_dict()
        assert data["primary_traits"] == ["friendly", "helpful"]
        assert data["secondary_traits"] == ["analytical"]
        assert data["response_style"] == "professional"
        assert data["verbosity_level"] == 7
        assert data["cooperation_tendency"] == 0.9
        assert data["expertise_level"]["security"] == 0.8
        
        # Convert back from dict
        restored_profile = PersonalityProfile.from_dict(data)
        assert restored_profile.primary_traits == original_profile.primary_traits
        assert restored_profile.secondary_traits == original_profile.secondary_traits
        assert restored_profile.response_style == original_profile.response_style
        assert restored_profile.verbosity_level == original_profile.verbosity_level
        assert restored_profile.cooperation_tendency == original_profile.cooperation_tendency
        assert restored_profile.expertise_level == original_profile.expertise_level


class TestAgentConfiguration:
    """Test AgentConfiguration class."""
    
    def test_agent_configuration_creation(self):
        """Test creating an agent configuration."""
        personality = PersonalityProfile()
        config = AgentConfiguration(
            agent_id="test_agent",
            agent_type=AgentType.USER_TWIN,
            name="Test Agent",
            description="A test agent",
            personality=personality
        )
        
        assert config.agent_id == "test_agent"
        assert config.agent_type == AgentType.USER_TWIN
        assert config.name == "Test Agent"
        assert config.description == "A test agent"
        assert config.personality == personality
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.enable_memory is True
        assert config.version == "1.0.0"
    
    def test_agent_configuration_with_custom_settings(self):
        """Test creating an agent configuration with custom settings."""
        personality = PersonalityProfile(
            primary_traits=[PersonalityTrait.FRIENDLY],
            verbosity_level=8
        )
        
        config = AgentConfiguration(
            agent_id="custom_agent",
            agent_type=AgentType.ORCHESTRATOR,
            name="Custom Agent",
            description="A custom configured agent",
            personality=personality,
            model_name="gpt-4",
            temperature=0.3,
            max_tokens=2000,
            enable_memory=False,
            custom_settings={"custom_param": "value", "another_param": 42}
        )
        
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.3
        assert config.max_tokens == 2000
        assert config.enable_memory is False
        assert config.custom_settings["custom_param"] == "value"
        assert config.custom_settings["another_param"] == 42
    
    def test_update_setting(self):
        """Test updating configuration settings."""
        personality = PersonalityProfile()
        config = AgentConfiguration(
            agent_id="test_agent",
            agent_type=AgentType.USER_TWIN,
            name="Test Agent",
            description="A test agent",
            personality=personality
        )
        
        original_updated_at = config.updated_at
        
        # Update built-in setting
        config.update_setting("temperature", 0.5)
        assert config.temperature == 0.5
        assert config.updated_at > original_updated_at
        
        # Update custom setting
        config.update_setting("custom_param", "new_value")
        assert config.custom_settings["custom_param"] == "new_value"
    
    def test_get_setting(self):
        """Test getting configuration settings."""
        personality = PersonalityProfile()
        config = AgentConfiguration(
            agent_id="test_agent",
            agent_type=AgentType.USER_TWIN,
            name="Test Agent",
            description="A test agent",
            personality=personality,
            temperature=0.8,
            custom_settings={"custom_param": "value"}
        )
        
        # Get built-in setting
        assert config.get_setting("temperature") == 0.8
        assert config.get_setting("max_tokens") == 1000
        
        # Get custom setting
        assert config.get_setting("custom_param") == "value"
        assert config.get_setting("non_existent", "default") == "default"
    
    def test_agent_configuration_serialization(self):
        """Test agent configuration serialization and deserialization."""
        personality = PersonalityProfile(
            primary_traits=[PersonalityTrait.FRIENDLY],
            verbosity_level=6
        )
        
        original_config = AgentConfiguration(
            agent_id="test_agent",
            agent_type=AgentType.USER_TWIN,
            name="Test Agent",
            description="A test agent",
            personality=personality,
            temperature=0.8,
            custom_settings={"test_param": "value"}
        )
        
        # Convert to dict
        data = original_config.to_dict()
        assert data["agent_id"] == "test_agent"
        assert data["agent_type"] == "user_twin"
        assert data["temperature"] == 0.8
        assert data["personality"]["verbosity_level"] == 6
        assert data["custom_settings"]["test_param"] == "value"
        assert "created_at" in data
        assert "updated_at" in data
        
        # Convert back from dict
        restored_config = AgentConfiguration.from_dict(data)
        assert restored_config.agent_id == original_config.agent_id
        assert restored_config.agent_type == original_config.agent_type
        assert restored_config.temperature == original_config.temperature
        assert restored_config.personality.verbosity_level == original_config.personality.verbosity_level
        assert restored_config.custom_settings == original_config.custom_settings


class TestConfigurationValidator:
    """Test configuration validators."""
    
    def test_schema_validator(self):
        """Test schema-based validation."""
        validator = SchemaValidator()
        
        # Valid configuration
        personality = PersonalityProfile()
        valid_config = AgentConfiguration(
            agent_id="test_agent",
            agent_type=AgentType.USER_TWIN,
            name="Test Agent",
            description="A test agent",
            personality=personality,
            temperature=0.7,
            max_tokens=1000
        )
        
        errors = validator.validate(valid_config)
        assert len(errors) == 0
        
        # Invalid configuration (missing required field)
        invalid_config = AgentConfiguration(
            agent_id="",  # Empty agent_id should fail
            agent_type=AgentType.USER_TWIN,
            name="Test Agent",
            description="A test agent",
            personality=personality
        )
        
        errors = validator.validate(invalid_config)
        assert len(errors) > 0
    
    def test_business_rule_validator(self):
        """Test business rule validation."""
        validator = BusinessRuleValidator()
        
        # Valid configuration
        personality = PersonalityProfile(
            aggression_level=0.3,
            cooperation_tendency=0.6
        )
        valid_config = AgentConfiguration(
            agent_id="test_agent",
            agent_type=AgentType.USER_TWIN,
            name="Test Agent",
            description="A test agent",
            personality=personality,
            timeout_seconds=30,
            response_time_preference=2.0,
            max_memory_size=500
        )
        
        errors = validator.validate(valid_config)
        assert len(errors) == 0
        
        # Invalid configuration (contradictory traits)
        invalid_personality = PersonalityProfile(
            aggression_level=0.9,
            cooperation_tendency=0.8
        )
        invalid_config = AgentConfiguration(
            agent_id="test_agent",
            agent_type=AgentType.USER_TWIN,
            name="Test Agent",
            description="A test agent",
            personality=invalid_personality
        )
        
        errors = validator.validate(invalid_config)
        assert len(errors) > 0
        assert any("contradictory" in error.lower() for error in errors)


class TestConfigurationTemplate:
    """Test ConfigurationTemplate class."""
    
    def test_template_creation(self):
        """Test creating a configuration template."""
        template_data = {
            "agent_type": "user_twin",
            "name": "Test Template",
            "description": "A test template",
            "temperature": 0.5,
            "personality": {
                "primary_traits": ["friendly"],
                "verbosity_level": 7
            }
        }
        
        template = ConfigurationTemplate(
            name="test_template",
            description="A test configuration template",
            config_data=template_data
        )
        
        assert template.name == "test_template"
        assert template.description == "A test configuration template"
        assert template.config_data == template_data
    
    def test_create_configuration_from_template(self):
        """Test creating configuration from template."""
        template_data = {
            "agent_type": "user_twin",
            "name": "Template Agent",
            "description": "An agent from template",
            "temperature": 0.5,
            "personality": {
                "primary_traits": ["friendly"],
                "verbosity_level": 7
            }
        }
        
        template = ConfigurationTemplate(
            name="test_template",
            description="A test template",
            config_data=template_data
        )
        
        config = template.create_configuration(
            agent_id="new_agent",
            temperature=0.8,  # Override template value
            custom_param="value"
        )
        
        assert config.agent_id == "new_agent"
        assert config.agent_type == AgentType.USER_TWIN
        assert config.name == "Template Agent"
        assert config.temperature == 0.8  # Override applied
        assert config.personality.verbosity_level == 7  # From template
        assert config.custom_settings["custom_param"] == "value"
    
    def test_template_serialization(self):
        """Test template serialization."""
        template_data = {
            "agent_type": "user_twin",
            "name": "Test Agent",
            "temperature": 0.5
        }
        
        template = ConfigurationTemplate(
            name="test_template",
            description="A test template",
            config_data=template_data
        )
        
        data = template.to_dict()
        assert data["name"] == "test_template"
        assert data["description"] == "A test template"
        assert data["config_data"] == template_data


class TestConfigurationManager:
    """Test ConfigurationManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_configuration_manager_creation(self):
        """Test creating a configuration manager."""
        assert self.config_manager.config_dir == Path(self.temp_dir)
        assert len(self.config_manager._validators) == 2  # Default validators
        assert len(self.config_manager._templates) > 0  # Default templates
    
    def test_add_and_remove_validator(self):
        """Test adding and removing validators."""
        custom_validator = Mock()
        custom_validator.validate.return_value = []
        
        # Add validator
        self.config_manager.add_validator(custom_validator)
        assert custom_validator in self.config_manager._validators
        
        # Remove validator
        self.config_manager.remove_validator(custom_validator)
        assert custom_validator not in self.config_manager._validators
    
    def test_save_and_load_configuration(self):
        """Test saving and loading configurations."""
        personality = PersonalityProfile(
            primary_traits=[PersonalityTrait.FRIENDLY]
        )
        
        config = AgentConfiguration(
            agent_id="test_agent",
            agent_type=AgentType.USER_TWIN,
            name="Test Agent",
            description="A test agent",
            personality=personality,
            temperature=0.8
        )
        
        # Save configuration
        success = self.config_manager.save_configuration(config)
        assert success is True
        
        # Load configuration
        loaded_config = self.config_manager.load_configuration("test_agent")
        assert loaded_config is not None
        assert loaded_config.agent_id == "test_agent"
        assert loaded_config.temperature == 0.8
        assert PersonalityTrait.FRIENDLY in loaded_config.personality.primary_traits
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        personality = PersonalityProfile()
        valid_config = AgentConfiguration(
            agent_id="valid_agent",
            agent_type=AgentType.USER_TWIN,
            name="Valid Agent",
            description="A valid agent",
            personality=personality
        )
        
        errors = self.config_manager.validate_configuration(valid_config)
        assert len(errors) == 0
        
        # Invalid configuration
        invalid_config = AgentConfiguration(
            agent_id="invalid_agent",
            agent_type=AgentType.USER_TWIN,
            name="Invalid Agent",
            description="An invalid agent",
            personality=personality,
            temperature=3.0  # Invalid temperature
        )
        
        errors = self.config_manager.validate_configuration(invalid_config)
        assert len(errors) > 0
    
    def test_update_configuration(self):
        """Test updating configurations."""
        personality = PersonalityProfile()
        config = AgentConfiguration(
            agent_id="update_agent",
            agent_type=AgentType.USER_TWIN,
            name="Update Agent",
            description="An agent to update",
            personality=personality,
            temperature=0.7
        )
        
        # Save initial configuration
        self.config_manager.save_configuration(config)
        
        # Update configuration
        updates = {
            "temperature": 0.9,
            "personality": {
                "verbosity_level": 8
            },
            "custom_setting": "new_value"
        }
        
        success = self.config_manager.update_configuration("update_agent", updates)
        assert success is True
        
        # Verify updates
        updated_config = self.config_manager.load_configuration("update_agent")
        assert updated_config.temperature == 0.9
        assert updated_config.personality.verbosity_level == 8
        assert updated_config.custom_settings["custom_setting"] == "new_value"
    
    def test_delete_configuration(self):
        """Test deleting configurations."""
        personality = PersonalityProfile()
        config = AgentConfiguration(
            agent_id="delete_agent",
            agent_type=AgentType.USER_TWIN,
            name="Delete Agent",
            description="An agent to delete",
            personality=personality
        )
        
        # Save configuration
        self.config_manager.save_configuration(config)
        assert "delete_agent" in self.config_manager.list_configurations()
        
        # Delete configuration
        success = self.config_manager.delete_configuration("delete_agent")
        assert success is True
        assert "delete_agent" not in self.config_manager.list_configurations()
    
    def test_update_callbacks(self):
        """Test configuration update callbacks."""
        callback_called = False
        callback_updates = {}
        
        def test_callback(agent_id: str, updates: dict):
            nonlocal callback_called, callback_updates
            callback_called = True
            callback_updates = updates
        
        # Add callback
        self.config_manager.add_update_callback("test_agent", test_callback)
        
        # Create and save configuration
        personality = PersonalityProfile()
        config = AgentConfiguration(
            agent_id="test_agent",
            agent_type=AgentType.USER_TWIN,
            name="Test Agent",
            description="A test agent",
            personality=personality
        )
        self.config_manager.save_configuration(config)
        
        # Update configuration (should trigger callback)
        updates = {"temperature": 0.9}
        self.config_manager.update_configuration("test_agent", updates)
        
        assert callback_called is True
        assert callback_updates == updates
        
        # Remove callback
        self.config_manager.remove_update_callback("test_agent", test_callback)
        
        # Reset callback state
        callback_called = False
        
        # Update again (should not trigger callback)
        self.config_manager.update_configuration("test_agent", {"temperature": 0.8})
        assert callback_called is False
    
    def test_templates(self):
        """Test configuration templates."""
        # List templates
        templates = self.config_manager.list_templates()
        assert len(templates) > 0
        assert "friendly_helper" in templates
        
        # Get template
        template = self.config_manager.get_template("friendly_helper")
        assert template is not None
        assert template.name == "friendly_helper"
        
        # Create configuration from template
        config = self.config_manager.create_configuration_from_template(
            "friendly_helper",
            "template_agent",
            temperature=0.6
        )
        
        assert config is not None
        assert config.agent_id == "template_agent"
        assert config.agent_type == AgentType.USER_TWIN
        assert config.temperature == 0.6  # Override applied
        
        # Test non-existent template
        config = self.config_manager.create_configuration_from_template(
            "non_existent",
            "test_agent"
        )
        assert config is None
    
    def test_statistics(self):
        """Test configuration manager statistics."""
        # Create some configurations
        personality = PersonalityProfile()
        config1 = AgentConfiguration(
            agent_id="stats_agent_1",
            agent_type=AgentType.USER_TWIN,
            name="Stats Agent 1",
            description="First stats agent",
            personality=personality
        )
        config2 = AgentConfiguration(
            agent_id="stats_agent_2",
            agent_type=AgentType.ORCHESTRATOR,
            name="Stats Agent 2",
            description="Second stats agent",
            personality=personality
        )
        
        self.config_manager.save_configuration(config1)
        self.config_manager.save_configuration(config2)
        
        stats = self.config_manager.get_statistics()
        assert stats["total_configurations"] == 2
        assert stats["total_templates"] > 0
        assert stats["total_validators"] == 2
        assert stats["config_directory"] == self.temp_dir


class TestGlobalConfigurationManager:
    """Test global configuration manager functions."""
    
    def test_get_configuration_manager_singleton(self):
        """Test that get_configuration_manager returns a singleton."""
        manager1 = get_configuration_manager()
        manager2 = get_configuration_manager()
        
        assert manager1 is manager2
    
    def test_create_configuration_manager(self):
        """Test creating a new configuration manager."""
        manager = create_configuration_manager("/tmp/test_configs")
        
        assert isinstance(manager, ConfigurationManager)
        assert manager.config_dir == Path("/tmp/test_configs")


class TestConfigurationIntegration:
    """Integration tests for the configuration system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_configuration_workflow(self):
        """Test complete configuration workflow."""
        # 1. Create personality profile
        personality = PersonalityProfile(
            primary_traits=[PersonalityTrait.FRIENDLY, PersonalityTrait.HELPFUL],
            secondary_traits=[PersonalityTrait.ANALYTICAL],
            response_style=ResponseStyle.PROFESSIONAL,
            verbosity_level=7,
            cooperation_tendency=0.9,
            expertise_level={"security": 0.8, "networking": 0.6}
        )
        
        # 2. Create configuration
        config = AgentConfiguration(
            agent_id="workflow_agent",
            agent_type=AgentType.USER_TWIN,
            name="Workflow Agent",
            description="An agent for workflow testing",
            personality=personality,
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1500,
            enable_memory=True,
            custom_settings={"workflow_param": "test_value"}
        )
        
        # 3. Validate configuration
        errors = self.config_manager.validate_configuration(config)
        assert len(errors) == 0
        
        # 4. Save configuration
        success = self.config_manager.save_configuration(config)
        assert success is True
        
        # 5. Load configuration
        loaded_config = self.config_manager.load_configuration("workflow_agent")
        assert loaded_config is not None
        
        # 6. Verify loaded configuration
        assert loaded_config.agent_id == "workflow_agent"
        assert loaded_config.agent_type == AgentType.USER_TWIN
        assert loaded_config.model_name == "gpt-4"
        assert loaded_config.temperature == 0.7
        assert PersonalityTrait.FRIENDLY in loaded_config.personality.primary_traits
        assert loaded_config.personality.expertise_level["security"] == 0.8
        assert loaded_config.custom_settings["workflow_param"] == "test_value"
        
        # 7. Update configuration
        updates = {
            "temperature": 0.9,
            "personality": {
                "verbosity_level": 8,
                "cooperation_tendency": 0.95
            },
            "custom_settings": {
                "workflow_param": "updated_value",
                "new_param": "new_value"
            }
        }
        
        success = self.config_manager.update_configuration("workflow_agent", updates)
        assert success is True
        
        # 8. Verify updates
        updated_config = self.config_manager.load_configuration("workflow_agent")
        assert updated_config.temperature == 0.9
        assert updated_config.personality.verbosity_level == 8
        assert updated_config.personality.cooperation_tendency == 0.95
        assert updated_config.custom_settings["workflow_param"] == "updated_value"
        assert updated_config.custom_settings["new_param"] == "new_value"
        
        # 9. List configurations
        configs = self.config_manager.list_configurations()
        assert "workflow_agent" in configs
        
        # 10. Delete configuration
        success = self.config_manager.delete_configuration("workflow_agent")
        assert success is True
        assert "workflow_agent" not in self.config_manager.list_configurations()
    
    def test_template_based_configuration_creation(self):
        """Test creating configurations from templates."""
        # Create configuration from template
        config = self.config_manager.create_configuration_from_template(
            "friendly_helper",
            "template_agent",
            name="Custom Template Agent",
            temperature=0.6,
            custom_settings={"template_param": "value"}
        )
        
        assert config is not None
        assert config.agent_id == "template_agent"
        assert config.name == "Custom Template Agent"
        assert config.temperature == 0.6
        assert config.agent_type == AgentType.USER_TWIN
        assert PersonalityTrait.HELPFUL in config.personality.primary_traits
        assert config.custom_settings["template_param"] == "value"
        
        # Save and validate
        errors = self.config_manager.validate_configuration(config)
        assert len(errors) == 0
        
        success = self.config_manager.save_configuration(config)
        assert success is True
    
    def test_error_handling(self):
        """Test error handling in configuration operations."""
        # Test loading non-existent configuration
        config = self.config_manager.load_configuration("non_existent")
        assert config is None
        
        # Test updating non-existent configuration
        success = self.config_manager.update_configuration("non_existent", {"temperature": 0.5})
        assert success is False
        
        # Test deleting non-existent configuration
        success = self.config_manager.delete_configuration("non_existent")
        assert success is False
        
        # Test creating from non-existent template
        config = self.config_manager.create_configuration_from_template(
            "non_existent_template",
            "test_agent"
        )
        assert config is None
        
        # Test invalid configuration data
        personality = PersonalityProfile()
        invalid_config = AgentConfiguration(
            agent_id="invalid_agent",
            agent_type=AgentType.USER_TWIN,
            name="Invalid Agent",
            description="An invalid agent",
            personality=personality,
            temperature=-1.0  # Invalid temperature
        )
        
        errors = self.config_manager.validate_configuration(invalid_config)
        assert len(errors) > 0
        
        # Test saving invalid configuration (should fail validation)
        success = self.config_manager.save_configuration(invalid_config, validate=True)
        assert success is False
