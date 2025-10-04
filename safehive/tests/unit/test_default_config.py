"""
Unit tests for the default configuration file.

This module tests that the default configuration file loads correctly
and contains all expected settings and values.
"""

import pytest
import os
import yaml
from pathlib import Path

from safehive.config.config_loader import ConfigLoader


class TestDefaultConfig:
    """Test the default configuration file."""
    
    def test_default_config_file_exists(self):
        """Test that the default configuration file exists."""
        config_path = "safehive/config/default_config.yaml"
        assert os.path.exists(config_path), f"Default config file not found at {config_path}"
    
    def test_default_config_is_valid_yaml(self):
        """Test that the default configuration file contains valid YAML."""
        config_path = "safehive/config/default_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        assert config_data is not None, "Configuration file is empty or invalid"
        assert isinstance(config_data, dict), "Configuration must be a dictionary"
    
    def test_default_config_has_required_sections(self):
        """Test that the default configuration has all required sections."""
        config_path = "safehive/config/default_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        required_sections = ["guards", "agents", "logging"]
        for section in required_sections:
            assert section in config_data, f"Missing required section: {section}"
    
    def test_guards_configuration(self):
        """Test guards configuration section."""
        config_path = "safehive/config/default_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        guards = config_data["guards"]
        required_guards = ["mcp_server", "privacy_sentry", "task_navigator", "prompt_sanitizer"]
        
        for guard in required_guards:
            assert guard in guards, f"Missing guard configuration: {guard}"
            assert "enabled" in guards[guard], f"Guard {guard} missing 'enabled' setting"
            assert isinstance(guards[guard]["enabled"], bool), f"Guard {guard} 'enabled' must be boolean"
    
    def test_mcp_server_configuration(self):
        """Test MCP server specific configuration."""
        config_path = "safehive/config/default_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        mcp_server = config_data["guards"]["mcp_server"]
        
        # Check required settings
        assert "doordash_api_url" in mcp_server, "MCP server missing 'doordash_api_url' setting"
        assert isinstance(mcp_server["doordash_api_url"], str), "MCP server API URL must be string"
        assert mcp_server["doordash_api_url"].startswith("http"), "MCP server API URL must be valid URL"
        
        assert "sandbox_mode" in mcp_server, "MCP server missing 'sandbox_mode' setting"
        assert isinstance(mcp_server["sandbox_mode"], bool), "MCP server sandbox_mode must be boolean"
        
        assert "order_validation" in mcp_server, "MCP server missing 'order_validation' setting"
        assert isinstance(mcp_server["order_validation"], bool), "MCP server order_validation must be boolean"
        
        assert "retry_attempts" in mcp_server, "MCP server missing 'retry_attempts' setting"
        assert isinstance(mcp_server["retry_attempts"], int), "MCP server retry_attempts must be integer"
        assert mcp_server["retry_attempts"] > 0, "MCP server retry_attempts must be positive"
    
    def test_agents_configuration(self):
        """Test agents configuration section."""
        config_path = "safehive/config/default_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        agents = config_data["agents"]
        required_agents = ["orchestrator", "user_twin", "vendors"]
        
        for agent in required_agents:
            assert agent in agents, f"Missing agent configuration: {agent}"
        
        # Test orchestrator configuration
        orchestrator = agents["orchestrator"]
        assert "ai_model" in orchestrator, "Orchestrator missing 'ai_model' setting"
        assert "max_retries" in orchestrator, "Orchestrator missing 'max_retries' setting"
        assert "timeout_seconds" in orchestrator, "Orchestrator missing 'timeout_seconds' setting"
        assert "memory_type" in orchestrator, "Orchestrator missing 'memory_type' setting"
        assert "tools" in orchestrator, "Orchestrator missing 'tools' setting"
        
        # Test user_twin configuration
        user_twin = agents["user_twin"]
        assert "ai_model" in user_twin, "User twin missing 'ai_model' setting"
        assert "memory_type" in user_twin, "User twin missing 'memory_type' setting"
        assert "personality" in user_twin, "User twin missing 'personality' setting"
        assert "constraints" in user_twin, "User twin missing 'constraints' setting"
        
        # Test vendors configuration
        vendors = agents["vendors"]
        required_vendors = ["honest_vendor", "malicious_vendor"]
        
        for vendor in required_vendors:
            assert vendor in vendors, f"Missing vendor configuration: {vendor}"
            assert "ai_model" in vendors[vendor], f"Vendor {vendor} missing 'ai_model' setting"
            assert "personality" in vendors[vendor], f"Vendor {vendor} missing 'personality' setting"
            assert "memory_type" in vendors[vendor], f"Vendor {vendor} missing 'memory_type' setting"
            assert "tools" in vendors[vendor], f"Vendor {vendor} missing 'tools' setting"
    
    def test_malicious_vendor_configuration(self):
        """Test malicious vendor specific configuration."""
        config_path = "safehive/config/default_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        malicious_vendor = config_data["agents"]["vendors"]["malicious_vendor"]
        
        assert "attack_behaviors" in malicious_vendor, "Malicious vendor missing 'attack_behaviors' setting"
        assert isinstance(malicious_vendor["attack_behaviors"], list), "Attack behaviors must be list"
        assert len(malicious_vendor["attack_behaviors"]) > 0, "Malicious vendor must have at least one attack behavior"
        
        assert "attack_frequency" in malicious_vendor, "Malicious vendor missing 'attack_frequency' setting"
        assert isinstance(malicious_vendor["attack_frequency"], (int, float)), "Attack frequency must be number"
        assert 0 <= malicious_vendor["attack_frequency"] <= 1, "Attack frequency must be between 0 and 1"
    
    def test_logging_configuration(self):
        """Test logging configuration section."""
        config_path = "safehive/config/default_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        logging = config_data["logging"]
        
        assert "level" in logging, "Logging missing 'level' setting"
        assert logging["level"] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], "Invalid logging level"
        
        assert "file" in logging, "Logging missing 'file' setting"
        assert "alerts_file" in logging, "Logging missing 'alerts_file' setting"
        assert "agent_conversations" in logging, "Logging missing 'agent_conversations' setting"
        
        assert "structured" in logging, "Logging missing 'structured' setting"
        assert isinstance(logging["structured"], bool), "Logging structured must be boolean"
        
        assert "max_file_size" in logging, "Logging missing 'max_file_size' setting"
        assert "backup_count" in logging, "Logging missing 'backup_count' setting"
        assert isinstance(logging["backup_count"], int), "Logging backup_count must be integer"
        assert logging["backup_count"] > 0, "Logging backup_count must be positive"
    
    def test_attack_simulation_configuration(self):
        """Test attack simulation configuration section."""
        config_path = "safehive/config/default_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        attack_simulation = config_data["attack_simulation"]
        
        assert "sql_injection" in attack_simulation, "Missing SQL injection configuration"
        assert "xss" in attack_simulation, "Missing XSS configuration"
        assert "path_traversal" in attack_simulation, "Missing path traversal configuration"
        
        # Test SQL injection patterns
        sql_patterns = attack_simulation["sql_injection"]["patterns"]
        assert isinstance(sql_patterns, list), "SQL injection patterns must be list"
        assert len(sql_patterns) > 0, "SQL injection must have at least one pattern"
        
        # Test XSS patterns
        xss_patterns = attack_simulation["xss"]["patterns"]
        assert isinstance(xss_patterns, list), "XSS patterns must be list"
        assert len(xss_patterns) > 0, "XSS must have at least one pattern"
        
        # Test path traversal patterns
        pt_patterns = attack_simulation["path_traversal"]["patterns"]
        assert isinstance(pt_patterns, list), "Path traversal patterns must be list"
        assert len(pt_patterns) > 0, "Path traversal must have at least one pattern"
    
    def test_decoy_data_configuration(self):
        """Test decoy data configuration section."""
        config_path = "safehive/config/default_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        decoy_data = config_data["decoy_data"]
        
        assert "credit_cards" in decoy_data, "Missing credit cards configuration"
        assert "order_history" in decoy_data, "Missing order history configuration"
        assert "customer_profiles" in decoy_data, "Missing customer profiles configuration"
        
        # Test credit cards configuration
        credit_cards = decoy_data["credit_cards"]
        assert "count" in credit_cards, "Credit cards missing 'count' setting"
        assert isinstance(credit_cards["count"], int), "Credit cards count must be integer"
        assert credit_cards["count"] > 0, "Credit cards count must be positive"
        
        assert "types" in credit_cards, "Credit cards missing 'types' setting"
        assert isinstance(credit_cards["types"], list), "Credit cards types must be list"
        assert len(credit_cards["types"]) > 0, "Credit cards must have at least one type"
    
    def test_config_loader_can_load_default_config(self):
        """Test that ConfigLoader can successfully load the default configuration."""
        config_path = "safehive/config/default_config.yaml"
        loader = ConfigLoader(config_path)
        
        # Should not raise any exceptions
        config = loader.load_config()
        
        assert config is not None, "ConfigLoader failed to load default configuration"
        assert len(config.guards) > 0, "No guards loaded from default configuration"
        assert len(config.agents) > 0, "No agents loaded from default configuration"
        assert config.logging is not None, "No logging configuration loaded"
    
    def test_default_config_validation(self):
        """Test that the default configuration passes validation."""
        config_path = "safehive/config/default_config.yaml"
        loader = ConfigLoader(config_path)
        
        # Load and validate configuration
        config = loader.load_config()
        
        # Test that all guards are properly configured
        assert "mcp_server" in config.guards, "MCP server not loaded"
        assert "privacy_sentry" in config.guards, "Privacy sentry guard not loaded"
        assert "task_navigator" in config.guards, "Task navigator guard not loaded"
        assert "prompt_sanitizer" in config.guards, "Prompt sanitizer guard not loaded"
        
        # Test that all agents are properly configured
        assert "orchestrator" in config.agents, "Orchestrator agent not loaded"
        assert "user_twin" in config.agents, "User twin agent not loaded"
        assert "vendors_honest_vendor" in config.agents, "Honest vendor not loaded"
        assert "vendors_malicious_vendor" in config.agents, "Malicious vendor not loaded"
        
        # Test specific configurations
        mcp_config = config.guards["mcp_server"]
        assert mcp_config.enabled is False, "MCP server should be disabled by default for safety"
        assert "doordash_api_url" in mcp_config.settings, "MCP server should have API URL setting"
        
        orchestrator_config = config.agents["orchestrator"]
        assert orchestrator_config.ai_model == "llama2:7b", "Orchestrator should use llama2:7b"
        assert orchestrator_config.max_retries == 3, "Orchestrator max_retries should be 3"
        assert "order_management" in orchestrator_config.tools, "Orchestrator should have order_management tool"


if __name__ == "__main__":
    pytest.main([__file__])
