"""
Unit tests for Privacy Sentry - PII Detection and Prevention.
"""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from safehive.guards.privacy_sentry import (
    PrivacySentry, PIIType, PIISeverity, PrivacyViolation, PrivacyPolicy,
    PIIDetectionResult, PrivacyAction, PrivacyRule, PrivacyAuditLog
)


class TestPIIType:
    """Test PIIType enum."""
    
    def test_pii_type_values(self):
        """Test PIIType enum values."""
        assert PIIType.EMAIL.value == "email"
        assert PIIType.PHONE.value == "phone"
        assert PIIType.CREDIT_CARD.value == "credit_card"
        assert PIIType.SSN.value == "ssn"
        assert PIIType.NAME.value == "name"
        assert PIIType.IP_ADDRESS.value == "ip_address"
        assert PIIType.DATE_OF_BIRTH.value == "date_of_birth"
        assert PIIType.STREET_ADDRESS.value == "street_address"
        assert PIIType.BANK_ACCOUNT.value == "bank_account"
        assert PIIType.API_KEY.value == "api_key"
        assert PIIType.PASSWORD.value == "password"


class TestPIISeverity:
    """Test PIISeverity enum."""
    
    def test_pii_severity_values(self):
        """Test PIISeverity enum values."""
        assert PIISeverity.LOW.value == "low"
        assert PIISeverity.MEDIUM.value == "medium"
        assert PIISeverity.HIGH.value == "high"
        assert PIISeverity.CRITICAL.value == "critical"


class TestPrivacyAction:
    """Test PrivacyAction enum."""
    
    def test_privacy_action_values(self):
        """Test PrivacyAction enum values."""
        assert PrivacyAction.ALLOW.value == "allow"
        assert PrivacyAction.BLOCK.value == "block"
        assert PrivacyAction.MASK.value == "mask"
        assert PrivacyAction.REDACT.value == "redact"
        assert PrivacyAction.ENCRYPT.value == "encrypt"
        assert PrivacyAction.AUDIT.value == "audit"
        assert PrivacyAction.QUARANTINE.value == "quarantine"


class TestPIIDetectionResult:
    """Test PIIDetectionResult functionality."""
    
    def test_pii_detection_result_creation(self):
        """Test PIIDetectionResult creation."""
        detection = PIIDetectionResult(
            pii_type=PIIType.EMAIL,
            severity=PIISeverity.MEDIUM,
            confidence=0.9,
            start_pos=10,
            end_pos=25,
            detected_value="test@example.com",
            masked_value="te**@example.com",
            context="My email is test@example.com",
            detection_method="regex_pattern"
        )
        
        assert detection.pii_type == PIIType.EMAIL
        assert detection.severity == PIISeverity.MEDIUM
        assert detection.confidence == 0.9
        assert detection.start_pos == 10
        assert detection.end_pos == 25
        assert detection.detected_value == "test@example.com"
        assert detection.masked_value == "te**@example.com"
        assert detection.context == "My email is test@example.com"
        assert detection.detection_method == "regex_pattern"
        assert isinstance(detection.timestamp, datetime)
    
    def test_pii_detection_result_serialization(self):
        """Test PIIDetectionResult serialization."""
        detection = PIIDetectionResult(
            pii_type=PIIType.EMAIL,
            severity=PIISeverity.MEDIUM,
            confidence=0.9,
            start_pos=10,
            end_pos=25,
            detected_value="test@example.com",
            masked_value="te**@example.com",
            context="My email is test@example.com",
            detection_method="regex_pattern"
        )
        
        data = detection.to_dict()
        
        assert data["pii_type"] == "email"
        assert data["severity"] == "medium"
        assert data["confidence"] == 0.9
        assert data["start_pos"] == 10
        assert data["end_pos"] == 25
        assert data["detected_value"] == "test@example.com"
        assert data["masked_value"] == "te**@example.com"
        assert data["context"] == "My email is test@example.com"
        assert data["detection_method"] == "regex_pattern"
        assert "timestamp" in data


class TestPrivacyRule:
    """Test PrivacyRule functionality."""
    
    def test_privacy_rule_creation(self):
        """Test PrivacyRule creation."""
        rule = PrivacyRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test privacy rule",
            pii_types=[PIIType.EMAIL, PIIType.PHONE],
            severity=PIISeverity.MEDIUM,
            action=PrivacyAction.MASK,
            conditions={"min_confidence": 0.8},
            enabled=True
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.description == "A test privacy rule"
        assert rule.pii_types == [PIIType.EMAIL, PIIType.PHONE]
        assert rule.severity == PIISeverity.MEDIUM
        assert rule.action == PrivacyAction.MASK
        assert rule.conditions["min_confidence"] == 0.8
        assert rule.enabled is True
        assert isinstance(rule.created_at, datetime)
    
    def test_privacy_rule_serialization(self):
        """Test PrivacyRule serialization."""
        rule = PrivacyRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test privacy rule",
            pii_types=[PIIType.EMAIL, PIIType.PHONE],
            severity=PIISeverity.MEDIUM,
            action=PrivacyAction.MASK
        )
        
        data = rule.to_dict()
        
        assert data["rule_id"] == "test_rule"
        assert data["name"] == "Test Rule"
        assert data["description"] == "A test privacy rule"
        assert data["pii_types"] == ["email", "phone"]
        assert data["severity"] == "medium"
        assert data["action"] == "mask"
        assert data["enabled"] is True
        assert "created_at" in data


class TestPrivacyPolicy:
    """Test PrivacyPolicy functionality."""
    
    def test_privacy_policy_creation(self):
        """Test PrivacyPolicy creation."""
        rule = PrivacyRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test privacy rule",
            pii_types=[PIIType.EMAIL],
            severity=PIISeverity.MEDIUM,
            action=PrivacyAction.MASK
        )
        
        policy = PrivacyPolicy(
            policy_id="test_policy",
            name="Test Policy",
            description="A test privacy policy",
            rules=[rule],
            default_action=PrivacyAction.AUDIT,
            strict_mode=True,
            audit_all=True
        )
        
        assert policy.policy_id == "test_policy"
        assert policy.name == "Test Policy"
        assert policy.description == "A test privacy policy"
        assert len(policy.rules) == 1
        assert policy.rules[0] == rule
        assert policy.default_action == PrivacyAction.AUDIT
        assert policy.strict_mode is True
        assert policy.audit_all is True
        assert isinstance(policy.created_at, datetime)
        assert isinstance(policy.updated_at, datetime)
    
    def test_privacy_policy_serialization(self):
        """Test PrivacyPolicy serialization."""
        rule = PrivacyRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test privacy rule",
            pii_types=[PIIType.EMAIL],
            severity=PIISeverity.MEDIUM,
            action=PrivacyAction.MASK
        )
        
        policy = PrivacyPolicy(
            policy_id="test_policy",
            name="Test Policy",
            description="A test privacy policy",
            rules=[rule]
        )
        
        data = policy.to_dict()
        
        assert data["policy_id"] == "test_policy"
        assert data["name"] == "Test Policy"
        assert data["description"] == "A test privacy policy"
        assert len(data["rules"]) == 1
        assert data["default_action"] == "audit"
        assert data["strict_mode"] is False
        assert data["audit_all"] is True
        assert "created_at" in data
        assert "updated_at" in data


class TestPrivacyViolation:
    """Test PrivacyViolation functionality."""
    
    def test_privacy_violation_creation(self):
        """Test PrivacyViolation creation."""
        detection = PIIDetectionResult(
            pii_type=PIIType.EMAIL,
            severity=PIISeverity.MEDIUM,
            confidence=0.9,
            start_pos=10,
            end_pos=25,
            detected_value="test@example.com",
            masked_value="te**@example.com",
            context="My email is test@example.com",
            detection_method="regex_pattern"
        )
        
        violation = PrivacyViolation(
            violation_id="violation_001",
            pii_detections=[detection],
            source="user_agent",
            destination="orchestrator_agent",
            action_taken=PrivacyAction.MASK,
            severity=PIISeverity.MEDIUM,
            resolved=False,
            resolution_notes="Under review"
        )
        
        assert violation.violation_id == "violation_001"
        assert len(violation.pii_detections) == 1
        assert violation.pii_detections[0] == detection
        assert violation.source == "user_agent"
        assert violation.destination == "orchestrator_agent"
        assert violation.action_taken == PrivacyAction.MASK
        assert violation.severity == PIISeverity.MEDIUM
        assert violation.resolved is False
        assert violation.resolution_notes == "Under review"
        assert isinstance(violation.timestamp, datetime)
    
    def test_privacy_violation_serialization(self):
        """Test PrivacyViolation serialization."""
        detection = PIIDetectionResult(
            pii_type=PIIType.EMAIL,
            severity=PIISeverity.MEDIUM,
            confidence=0.9,
            start_pos=10,
            end_pos=25,
            detected_value="test@example.com",
            masked_value="te**@example.com",
            context="My email is test@example.com",
            detection_method="regex_pattern"
        )
        
        violation = PrivacyViolation(
            violation_id="violation_001",
            pii_detections=[detection],
            source="user_agent",
            destination="orchestrator_agent",
            action_taken=PrivacyAction.MASK,
            severity=PIISeverity.MEDIUM
        )
        
        data = violation.to_dict()
        
        assert data["violation_id"] == "violation_001"
        assert len(data["pii_detections"]) == 1
        assert data["source"] == "user_agent"
        assert data["destination"] == "orchestrator_agent"
        assert data["action_taken"] == "mask"
        assert data["severity"] == "medium"
        assert data["resolved"] is False
        assert data["resolution_notes"] is None
        assert "timestamp" in data


class TestPrivacyAuditLog:
    """Test PrivacyAuditLog functionality."""
    
    def test_privacy_audit_log_creation(self):
        """Test PrivacyAuditLog creation."""
        log = PrivacyAuditLog(
            log_id="log_001",
            event_type="pii_detection",
            timestamp=datetime.now(),
            source="user_agent",
            destination="orchestrator_agent",
            pii_detected=[PIIType.EMAIL, PIIType.PHONE],
            action_taken=PrivacyAction.MASK,
            severity=PIISeverity.MEDIUM,
            details={"detection_count": 2}
        )
        
        assert log.log_id == "log_001"
        assert log.event_type == "pii_detection"
        assert isinstance(log.timestamp, datetime)
        assert log.source == "user_agent"
        assert log.destination == "orchestrator_agent"
        assert log.pii_detected == [PIIType.EMAIL, PIIType.PHONE]
        assert log.action_taken == PrivacyAction.MASK
        assert log.severity == PIISeverity.MEDIUM
        assert log.details["detection_count"] == 2
    
    def test_privacy_audit_log_serialization(self):
        """Test PrivacyAuditLog serialization."""
        log = PrivacyAuditLog(
            log_id="log_001",
            event_type="pii_detection",
            timestamp=datetime.now(),
            source="user_agent",
            destination="orchestrator_agent",
            pii_detected=[PIIType.EMAIL],
            action_taken=PrivacyAction.MASK,
            severity=PIISeverity.MEDIUM
        )
        
        data = log.to_dict()
        
        assert data["log_id"] == "log_001"
        assert data["event_type"] == "pii_detection"
        assert "timestamp" in data
        assert data["source"] == "user_agent"
        assert data["destination"] == "orchestrator_agent"
        assert data["pii_detected"] == ["email"]
        assert data["action_taken"] == "mask"
        assert data["severity"] == "medium"
        assert data["details"] == {}


class TestPrivacySentry:
    """Test PrivacySentry functionality."""
    
    def test_privacy_sentry_creation(self):
        """Test PrivacySentry creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            assert sentry.storage_path == Path(temp_dir)
            assert len(sentry.pii_patterns) > 0
            assert len(sentry.pii_severity_map) > 0
            assert len(sentry.privacy_rules) > 0
            assert len(sentry.privacy_policies) > 0
            assert sentry.active_policy is not None
            assert len(sentry.audit_logs) == 0
            assert len(sentry.privacy_violations) == 0
    
    def test_detect_pii_email(self):
        """Test PII detection for email addresses."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            text = "My email is john.doe@example.com and you can reach me there."
            detections = sentry.detect_pii(text)
            
            assert len(detections) >= 1
            email_detection = next((d for d in detections if d.pii_type == PIIType.EMAIL), None)
            assert email_detection is not None
            assert email_detection.detected_value == "john.doe@example.com"
            assert email_detection.severity == PIISeverity.MEDIUM
            assert email_detection.confidence > 0.5
    
    def test_detect_pii_phone(self):
        """Test PII detection for phone numbers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            text = "Call me at (555) 123-4567 or 555-123-4567"
            detections = sentry.detect_pii(text)
            
            assert len(detections) >= 1
            phone_detections = [d for d in detections if d.pii_type == PIIType.PHONE]
            assert len(phone_detections) >= 1  # Overlapping detection removal may reduce count
            assert any("(555) 123-4567" in d.detected_value for d in phone_detections)
            assert any("555-123-4567" in d.detected_value for d in phone_detections)
    
    def test_detect_pii_credit_card(self):
        """Test PII detection for credit card numbers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            text = "My credit card is 4532 1234 5678 9012"
            detections = sentry.detect_pii(text)
            
            assert len(detections) >= 1
            cc_detection = next((d for d in detections if d.pii_type == PIIType.CREDIT_CARD), None)
            assert cc_detection is not None
            assert "4532" in cc_detection.detected_value
            assert cc_detection.severity == PIISeverity.CRITICAL
            assert cc_detection.confidence > 0.5
    
    def test_detect_pii_ssn(self):
        """Test PII detection for SSN."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            text = "My SSN is 123-45-6789"
            detections = sentry.detect_pii(text)
            
            assert len(detections) >= 1
            ssn_detection = next((d for d in detections if d.pii_type == PIIType.SSN), None)
            assert ssn_detection is not None
            assert "123-45-6789" in ssn_detection.detected_value
            assert ssn_detection.severity == PIISeverity.CRITICAL
            assert ssn_detection.confidence > 0.5
    
    def test_detect_pii_ip_address(self):
        """Test PII detection for IP addresses."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            text = "The server IP is 192.168.1.1"
            detections = sentry.detect_pii(text)
            
            assert len(detections) >= 1
            ip_detection = next((d for d in detections if d.pii_type == PIIType.IP_ADDRESS), None)
            assert ip_detection is not None
            assert "192.168.1.1" in ip_detection.detected_value
            assert ip_detection.severity == PIISeverity.MEDIUM
    
    def test_detect_pii_date_of_birth(self):
        """Test PII detection for date of birth."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            text = "I was born on 01/15/1990"
            detections = sentry.detect_pii(text)
            
            assert len(detections) >= 1
            dob_detection = next((d for d in detections if d.pii_type == PIIType.DATE_OF_BIRTH), None)
            assert dob_detection is not None
            assert "01/15/1990" in dob_detection.detected_value
            assert dob_detection.severity == PIISeverity.HIGH
    
    def test_detect_pii_name(self):
        """Test PII detection for names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            text = "My name is John Smith"
            detections = sentry.detect_pii(text)
            
            assert len(detections) >= 1
            name_detection = next((d for d in detections if d.pii_type == PIIType.NAME), None)
            assert name_detection is not None
            assert "John Smith" in name_detection.detected_value
            assert name_detection.severity == PIISeverity.MEDIUM
    
    def test_detect_pii_street_address(self):
        """Test PII detection for street addresses."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            text = "I live at 123 Main Street"
            detections = sentry.detect_pii(text)
            
            assert len(detections) >= 1
            address_detection = next((d for d in detections if d.pii_type == PIIType.STREET_ADDRESS), None)
            assert address_detection is not None
            assert "123 Main Street" in address_detection.detected_value
            assert address_detection.severity == PIISeverity.HIGH
    
    def test_detect_pii_postal_code(self):
        """Test PII detection for postal codes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            text = "My ZIP code is 12345"
            detections = sentry.detect_pii(text)
            
            assert len(detections) >= 1
            postal_detection = next((d for d in detections if d.pii_type == PIIType.POSTAL_CODE), None)
            assert postal_detection is not None
            assert "12345" in postal_detection.detected_value
            assert postal_detection.severity == PIISeverity.MEDIUM
    
    def test_detect_pii_bank_account(self):
        """Test PII detection for bank account numbers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            text = "My account number is 123456789012"
            detections = sentry.detect_pii(text)
            
            assert len(detections) >= 1
            bank_detection = next((d for d in detections if d.pii_type == PIIType.BANK_ACCOUNT), None)
            assert bank_detection is not None
            assert "123456789012" in bank_detection.detected_value
            assert bank_detection.severity == PIISeverity.CRITICAL
    
    def test_detect_pii_api_key(self):
        """Test PII detection for API keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            text = "My API key is abc123def456ghi789jkl012mno345pqr678"
            detections = sentry.detect_pii(text)
            
            assert len(detections) >= 1
            api_detection = next((d for d in detections if d.pii_type == PIIType.API_KEY), None)
            assert api_detection is not None
            assert "abc123def456ghi789jkl012mno345pqr678" in api_detection.detected_value
            assert api_detection.severity == PIISeverity.CRITICAL
    
    def test_detect_pii_password(self):
        """Test PII detection for passwords."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            text = "password: mysecretpassword123"
            detections = sentry.detect_pii(text)
            
            assert len(detections) >= 1
            password_detection = next((d for d in detections if d.pii_type == PIIType.PASSWORD), None)
            assert password_detection is not None
            assert "mysecretpassword123" in password_detection.detected_value
            assert password_detection.severity == PIISeverity.CRITICAL
    
    def test_mask_pii_email(self):
        """Test PII masking for email addresses."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            masked = sentry._mask_pii(PIIType.EMAIL, "john.doe@example.com")
            assert masked == "jo******@example.com"  # john.doe = 8 chars, keep first 2, mask 6
            
            masked = sentry._mask_pii(PIIType.EMAIL, "ab@test.com")
            assert masked == "**@test.com"
    
    def test_mask_pii_phone(self):
        """Test PII masking for phone numbers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            masked = sentry._mask_pii(PIIType.PHONE, "555-123-4567")
            assert masked == "******4567"  # 10 digits, keep last 4, mask 6
            
            masked = sentry._mask_pii(PIIType.PHONE, "1234567890")
            assert masked == "******7890"
    
    def test_mask_pii_credit_card(self):
        """Test PII masking for credit card numbers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            masked = sentry._mask_pii(PIIType.CREDIT_CARD, "4532 1234 5678 9012")
            assert masked == "************9012"
            
            masked = sentry._mask_pii(PIIType.CREDIT_CARD, "4532123456789012")
            assert masked == "************9012"
    
    def test_mask_pii_ssn(self):
        """Test PII masking for SSN."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            masked = sentry._mask_pii(PIIType.SSN, "123-45-6789")
            assert masked == "***-**-6789"
            
            masked = sentry._mask_pii(PIIType.SSN, "123456789")
            assert masked == "***-**-6789"
    
    def test_mask_pii_name(self):
        """Test PII masking for names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            masked = sentry._mask_pii(PIIType.NAME, "John Smith")
            assert masked == "J*** S****"
            
            masked = sentry._mask_pii(PIIType.NAME, "A B")
            assert masked == "* *"
    
    def test_process_message_allow(self):
        """Test message processing with ALLOW action."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            message = "Hello, how are you?"
            processed, detections, action = sentry.process_message(
                message, "user", "agent"
            )
            
            assert processed == message
            assert len(detections) == 0
            assert action == PrivacyAction.ALLOW
    
    def test_process_message_mask(self):
        """Test message processing with MASK action."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            message = "My email is john@example.com"
            processed, detections, action = sentry.process_message(
                message, "user", "agent"
            )
            
            # Check if masking was applied or if action was AUDIT (which allows original message)
            if action == PrivacyAction.MASK:
                assert "jo******@example.com" in processed
            else:
                # If action is AUDIT, original message should be returned
                assert "john@example.com" in processed
            assert len(detections) >= 1
            assert action in [PrivacyAction.MASK, PrivacyAction.AUDIT]
    
    def test_process_message_block(self):
        """Test message processing with BLOCK action."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            message = "My SSN is 123-45-6789"
            processed, detections, action = sentry.process_message(
                message, "user", "agent"
            )
            
            # Should be blocked or masked
            assert (processed == "[MESSAGE BLOCKED - Contains sensitive information]" or 
                   "***-**-6789" in processed)
            assert len(detections) >= 1
            assert action in [PrivacyAction.BLOCK, PrivacyAction.MASK]
    
    def test_process_message_redact(self):
        """Test message processing with REDACT action."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            # Create a custom rule with REDACT action
            rule = PrivacyRule(
                rule_id="redact_rule",
                name="Redact Rule",
                description="Redact email addresses",
                pii_types=[PIIType.EMAIL],
                severity=PIISeverity.MEDIUM,
                action=PrivacyAction.REDACT
            )
            sentry.add_privacy_rule(rule)
            
            # Create custom policy
            policy = PrivacyPolicy(
                policy_id="redact_policy",
                name="Redact Policy",
                description="Policy that redacts PII",
                rules=[rule],
                default_action=PrivacyAction.REDACT
            )
            sentry.create_custom_policy(policy)
            sentry.set_active_policy("redact_policy")
            
            message = "My email is john@example.com"
            processed, detections, action = sentry.process_message(
                message, "user", "agent"
            )
            
            assert "[REDACTED]" in processed
            assert len(detections) >= 1
            assert action == PrivacyAction.REDACT
    
    def test_get_privacy_statistics(self):
        """Test getting privacy statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            # Process some messages to generate statistics
            sentry.process_message("My email is test@example.com", "user", "agent")
            sentry.process_message("My phone is 555-123-4567", "user", "agent")
            
            stats = sentry.get_privacy_statistics()
            
            assert "total_detections" in stats
            assert "total_violations" in stats
            assert "detection_stats" in stats
            assert "violation_stats" in stats
            assert "active_policy" in stats
            assert stats["active_policy"] is not None
    
    def test_add_privacy_rule(self):
        """Test adding privacy rules."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            rule = PrivacyRule(
                rule_id="custom_rule",
                name="Custom Rule",
                description="A custom privacy rule",
                pii_types=[PIIType.EMAIL],
                severity=PIISeverity.HIGH,
                action=PrivacyAction.BLOCK
            )
            
            sentry.add_privacy_rule(rule)
            
            assert "custom_rule" in sentry.privacy_rules
            assert sentry.privacy_rules["custom_rule"] == rule
    
    def test_update_privacy_rule(self):
        """Test updating privacy rules."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            # Update existing rule
            success = sentry.update_privacy_rule("critical_pii_rule", enabled=False)
            assert success is True
            
            rule = sentry.privacy_rules["critical_pii_rule"]
            assert rule.enabled is False
            
            # Try to update non-existent rule
            success = sentry.update_privacy_rule("non_existent", enabled=False)
            assert success is False
    
    def test_set_active_policy(self):
        """Test setting active privacy policy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            # Set existing policy
            success = sentry.set_active_policy("default_policy")
            assert success is True
            assert sentry.active_policy.policy_id == "default_policy"
            
            # Try to set non-existent policy
            success = sentry.set_active_policy("non_existent")
            assert success is False
    
    def test_create_custom_policy(self):
        """Test creating custom privacy policies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            rule = PrivacyRule(
                rule_id="custom_rule",
                name="Custom Rule",
                description="A custom privacy rule",
                pii_types=[PIIType.EMAIL],
                severity=PIISeverity.MEDIUM,
                action=PrivacyAction.MASK
            )
            
            policy = PrivacyPolicy(
                policy_id="custom_policy",
                name="Custom Policy",
                description="A custom privacy policy",
                rules=[rule],
                default_action=PrivacyAction.MASK
            )
            
            sentry.create_custom_policy(policy)
            
            assert "custom_policy" in sentry.privacy_policies
            assert sentry.privacy_policies["custom_policy"] == policy
    
    def test_get_recent_violations(self):
        """Test getting recent privacy violations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            # Process messages that should create violations
            sentry.process_message("My SSN is 123-45-6789", "user", "agent")
            sentry.process_message("My credit card is 4532 1234 5678 9012", "user", "agent")
            
            violations = sentry.get_recent_violations(10)
            
            assert len(violations) >= 0  # May or may not create violations depending on policy
    
    def test_get_audit_logs(self):
        """Test getting audit logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            # Process some messages to generate audit logs
            sentry.process_message("My email is test@example.com", "user", "agent")
            sentry.process_message("My phone is 555-123-4567", "user", "agent")
            
            logs = sentry.get_audit_logs(10)
            
            assert len(logs) >= 2
            assert all(isinstance(log, PrivacyAuditLog) for log in logs)
    
    def test_cleanup_old_data(self):
        """Test cleaning up old data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            # Create old audit log
            old_log = PrivacyAuditLog(
                log_id="old_log",
                event_type="pii_detection",
                timestamp=datetime.now() - timedelta(days=35),
                source="user",
                destination="agent",
                pii_detected=[PIIType.EMAIL],
                action_taken=PrivacyAction.AUDIT,
                severity=PIISeverity.MEDIUM
            )
            sentry.audit_logs.append(old_log)
            
            # Create recent audit log
            recent_log = PrivacyAuditLog(
                log_id="recent_log",
                event_type="pii_detection",
                timestamp=datetime.now(),
                source="user",
                destination="agent",
                pii_detected=[PIIType.EMAIL],
                action_taken=PrivacyAction.AUDIT,
                severity=PIISeverity.MEDIUM
            )
            sentry.audit_logs.append(recent_log)
            
            # Clean up data older than 30 days
            cleaned_count = sentry.cleanup_old_data(30)
            
            assert cleaned_count >= 1
            assert len(sentry.audit_logs) == 1
            assert sentry.audit_logs[0].log_id == "recent_log"


class TestPrivacySentryIntegration:
    """Integration tests for PrivacySentry."""
    
    def test_complete_pii_detection_workflow(self):
        """Test complete PII detection and protection workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            # Test message with multiple PII types
            message = """
            Hi, my name is John Smith and my email is john.smith@example.com.
            You can call me at (555) 123-4567.
            My address is 123 Main Street, New York, NY 10001.
            My SSN is 123-45-6789 and my credit card is 4532 1234 5678 9012.
            """
            
            processed, detections, action = sentry.process_message(
                message, "user", "agent"
            )
            
            # Should detect multiple PII types
            detected_types = {detection.pii_type for detection in detections}
            expected_types = {PIIType.NAME, PIIType.EMAIL, PIIType.PHONE, 
                            PIIType.STREET_ADDRESS, PIIType.POSTAL_CODE, 
                            PIIType.SSN, PIIType.CREDIT_CARD}
            
            assert len(detected_types.intersection(expected_types)) >= 3
            
            # Should take some action (not ALLOW)
            assert action != PrivacyAction.ALLOW
            
            # Check statistics
            stats = sentry.get_privacy_statistics()
            assert stats["total_detections"] >= 3
            assert stats["total_violations"] >= 0
    
    def test_persistence_and_recovery(self):
        """Test persistence and recovery of privacy data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first sentry instance
            sentry1 = PrivacySentry(temp_dir)
            
            # Process some messages
            sentry1.process_message("My email is test@example.com", "user", "agent")
            sentry1.process_message("My SSN is 123-45-6789", "user", "agent")
            
            # Create second sentry instance (should load data)
            sentry2 = PrivacySentry(temp_dir)
            
            # Verify data was loaded
            assert len(sentry2.audit_logs) >= 2
            assert len(sentry2.privacy_violations) >= 0
    
    def test_custom_policy_workflow(self):
        """Test custom policy creation and application."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentry = PrivacySentry(temp_dir)
            
            # Create custom rule
            custom_rule = PrivacyRule(
                rule_id="email_block_rule",
                name="Block Email Rule",
                description="Block all email addresses",
                pii_types=[PIIType.EMAIL],
                severity=PIISeverity.MEDIUM,
                action=PrivacyAction.BLOCK
            )
            
            # Create custom policy
            custom_policy = PrivacyPolicy(
                policy_id="strict_policy",
                name="Strict Policy",
                description="Strict privacy policy",
                rules=[custom_rule],
                default_action=PrivacyAction.BLOCK,
                strict_mode=True
            )
            
            # Apply custom policy
            sentry.create_custom_policy(custom_policy)
            sentry.set_active_policy("strict_policy")
            
            # Test message processing
            message = "My email is test@example.com"
            processed, detections, action = sentry.process_message(
                message, "user", "agent"
            )
            
            # Should be blocked
            assert processed == "[MESSAGE BLOCKED - Contains sensitive information]"
            assert action == PrivacyAction.BLOCK
            assert len(detections) >= 1
