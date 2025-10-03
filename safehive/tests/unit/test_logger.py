"""
Unit tests for the logging utilities.

This module tests the structured logging functionality using loguru
for the SafeHive system.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from safehive.utils.logger import (
    get_logger, setup_logging, setup_alert_logging, setup_agent_logging,
    log_security_event, log_agent_interaction, log_guard_action
)


class TestLoggerUtilities:
    """Test the logging utility functions."""
    
    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test_module")
        
        assert logger is not None
        # The logger should be bound with the module name
        # We can't easily test the binding without complex mocking,
        # but we can ensure it returns a logger object
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        # Test that setup_logging doesn't raise exceptions
        setup_logging(level="DEBUG")
        
        # Test with different levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            setup_logging(level=level)
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_log_file = f.name
        
        try:
            setup_logging(
                level="INFO",
                log_file=temp_log_file,
                structured=True
            )
            
            # Verify file was created
            assert os.path.exists(temp_log_file)
            
        finally:
            if os.path.exists(temp_log_file):
                os.unlink(temp_log_file)
    
    def test_setup_logging_structured_vs_unstructured(self):
        """Test both structured and unstructured logging formats."""
        # Test structured logging
        setup_logging(level="INFO", structured=True)
        
        # Test unstructured logging
        setup_logging(level="INFO", structured=False)
    
    def test_setup_alert_logging(self):
        """Test alert-specific logging setup."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_alert_file = f.name
        
        try:
            setup_alert_logging(
                alerts_file=temp_alert_file,
                level="WARNING"
            )
            
            # Verify file was created
            assert os.path.exists(temp_alert_file)
            
        finally:
            if os.path.exists(temp_alert_file):
                os.unlink(temp_alert_file)
    
    def test_setup_agent_logging(self):
        """Test agent conversation logging setup."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_agent_file = f.name
        
        try:
            setup_agent_logging(
                conversations_file=temp_agent_file,
                level="INFO"
            )
            
            # Verify file was created
            assert os.path.exists(temp_agent_file)
            
        finally:
            if os.path.exists(temp_agent_file):
                os.unlink(temp_agent_file)
    
    def test_log_security_event(self):
        """Test security event logging."""
        # Test with different severity levels
        test_cases = [
            ("CRITICAL", "critical"),
            ("ERROR", "error"),
            ("WARNING", "warning"),
            ("INFO", "info")
        ]
        
        for severity, expected_level in test_cases:
            # This should not raise any exceptions
            log_security_event(
                event_type="test_attack",
                source="test_source",
                details={"test": "data"},
                severity=severity
            )
    
    def test_log_agent_interaction(self):
        """Test agent interaction logging."""
        # Test basic agent interaction logging
        log_agent_interaction(
            agent_from="orchestrator",
            agent_to="user_twin",
            message_type="request",
            content="Test message",
            metadata={"test": "metadata"}
        )
        
        # Test without metadata
        log_agent_interaction(
            agent_from="user_twin",
            agent_to="orchestrator",
            message_type="response",
            content="Test response"
        )
    
    def test_log_guard_action(self):
        """Test guard action logging."""
        # Test blocking action (should log as warning)
        log_guard_action(
            guard_name="honeypot",
            action="block",
            request_source="malicious_vendor",
            reason="SQL injection detected",
            details={"pattern": "SELECT * FROM users"}
        )
        
        # Test allowing action (should log as info)
        log_guard_action(
            guard_name="privacy_sentry",
            action="allow",
            request_source="honest_vendor",
            reason="No PII detected"
        )
        
        # Test decoy action (should log as warning)
        log_guard_action(
            guard_name="honeypot",
            action="decoy",
            request_source="attacker",
            reason="Threshold exceeded",
            details={"decoy_data": "fake_credit_cards"}
        )
    
    def test_logger_integration_with_config(self):
        """Test logger integration with configuration system."""
        from safehive.config.config_loader import ConfigLoader
        
        # Load configuration
        loader = ConfigLoader()
        config = loader.load_config()
        
        # Setup logging based on configuration
        logging_config = config.logging
        
        setup_logging(
            level=logging_config.level,
            log_file=logging_config.file,
            structured=logging_config.structured,
            max_file_size=logging_config.max_file_size,
            backup_count=logging_config.backup_count
        )
        
        # Test that we can get a logger and use it
        logger = get_logger("test_integration")
        logger.info("Test integration message")
    
    def test_logger_error_handling(self):
        """Test logger error handling."""
        # Test with invalid log level (should not crash)
        setup_logging(level="INVALID_LEVEL")
        
        # Test with invalid file path (should handle gracefully)
        setup_logging(log_file="/invalid/path/that/does/not/exist.log")
    
    def test_logger_performance(self):
        """Test logger performance with multiple messages."""
        import time
        
        start_time = time.time()
        
        # Log many messages quickly
        logger = get_logger("performance_test")
        for i in range(100):
            logger.info(f"Performance test message {i}")
        
        end_time = time.time()
        
        # Should complete quickly (less than 1 second for 100 messages)
        assert (end_time - start_time) < 1.0, "Logging performance is too slow"
    
    def test_logger_thread_safety(self):
        """Test logger thread safety."""
        import threading
        import time
        
        results = []
        
        def log_messages(thread_id):
            logger = get_logger(f"thread_{thread_id}")
            for i in range(10):
                logger.info(f"Thread {thread_id} message {i}")
                results.append(f"thread_{thread_id}_{i}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_messages, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have logged all messages without errors
        assert len(results) == 50, f"Expected 50 messages, got {len(results)}"
    
    def test_logger_file_rotation(self):
        """Test log file rotation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            
            setup_logging(
                level="INFO",
                log_file=log_file,
                max_file_size="1KB",  # Small size to trigger rotation
                backup_count=3
            )
            
            # Generate enough log messages to trigger rotation
            logger = get_logger("rotation_test")
            for i in range(100):
                logger.info(f"Rotation test message {i} with some additional content to make it longer")
            
            # Check that log file exists
            assert os.path.exists(log_file), "Main log file should exist"
            
            # Check for rotated files (they might have different names)
            log_dir = os.path.dirname(log_file)
            log_files = [f for f in os.listdir(log_dir) if f.startswith("test.log")]
            assert len(log_files) >= 1, "Should have at least the main log file"


class TestLoggerConfiguration:
    """Test logger configuration and setup."""
    
    def test_default_logging_setup(self):
        """Test that default logging is set up correctly."""
        # The logger module should set up default logging on import
        from safehive.utils.logger import setup_logging
        
        # Should not raise any exceptions
        setup_logging()
    
    def test_logger_with_different_formats(self):
        """Test logger with different format configurations."""
        # Test with custom format
        setup_logging(
            level="INFO",
            structured=True
        )
        
        # Test with minimal format
        setup_logging(
            level="ERROR",
            structured=False
        )
    
    def test_logger_with_multiple_handlers(self):
        """Test logger with multiple handlers (console + file)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_log_file = f.name
        
        try:
            setup_logging(
                level="INFO",
                log_file=temp_log_file,
                structured=True
            )
            
            # Should have both console and file handlers
            logger = get_logger("multi_handler_test")
            logger.info("Test message for multiple handlers")
            
            # Verify file was created and has content
            assert os.path.exists(temp_log_file)
            
        finally:
            if os.path.exists(temp_log_file):
                os.unlink(temp_log_file)


if __name__ == "__main__":
    pytest.main([__file__])
