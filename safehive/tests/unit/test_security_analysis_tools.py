"""
Unit tests for security analysis tools.
"""

import pytest
import json
from unittest.mock import patch, Mock

from safehive.tools.security_analysis_tools import (
    detect_pii, analyze_attack_patterns, perform_security_assessment, sanitize_data,
    PIIDetectionTool, AttackPatternAnalysisTool, SecurityAssessmentTool, DataSanitizationTool,
    PIIDetectionInput, AttackPatternAnalysisInput, SecurityAssessmentInput, DataSanitizationInput
)


class TestSecurityAnalysisTools:
    """Test security analysis tool functions."""

    def test_detect_pii_credit_card(self):
        """Test PII detection for credit card numbers."""
        text = "My credit card number is 4532-1234-5678-9012"
        result = detect_pii(text, "medium", ["credit_card"])
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["pii_found"] is True
        assert len(result_data["data"]["detected_pii"]) == 1
        assert result_data["data"]["detected_pii"][0]["type"] == "credit_card"

    def test_detect_pii_ssn(self):
        """Test PII detection for SSN."""
        text = "My social security number is 123-45-6789"
        result = detect_pii(text, "medium", ["ssn"])
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["pii_found"] is True
        assert len(result_data["data"]["detected_pii"]) == 1
        assert result_data["data"]["detected_pii"][0]["type"] == "ssn"

    def test_detect_pii_email(self):
        """Test PII detection for email addresses."""
        text = "Please contact me at john.doe@example.com"
        result = detect_pii(text, "medium", ["email"])
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["pii_found"] is True
        assert len(result_data["data"]["detected_pii"]) == 1
        assert result_data["data"]["detected_pii"][0]["type"] == "email"

    def test_detect_pii_phone(self):
        """Test PII detection for phone numbers."""
        text = "Call me at (555) 123-4567 or 555-987-6543"
        result = detect_pii(text, "medium", ["phone"])
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["pii_found"] is True
        assert len(result_data["data"]["detected_pii"]) >= 1

    def test_detect_pii_no_pii(self):
        """Test PII detection with no PII present."""
        text = "This is just a regular message with no sensitive information."
        result = detect_pii(text, "medium")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["pii_found"] is False
        assert len(result_data["data"]["detected_pii"]) == 0

    def test_detect_pii_sensitivity_levels(self):
        """Test PII detection with different sensitivity levels."""
        text = "My number is 1234567890"
        
        # Low sensitivity
        result_low = detect_pii(text, "low", ["phone"])
        data_low = json.loads(result_low)
        
        # High sensitivity
        result_high = detect_pii(text, "high", ["phone"])
        data_high = json.loads(result_high)
        
        # High sensitivity should detect more potential PII
        assert len(data_high["data"]["detected_pii"]) >= len(data_low["data"]["detected_pii"])

    def test_analyze_attack_patterns_sql_injection(self):
        """Test attack pattern analysis for SQL injection."""
        text = "'; DROP TABLE users; --"
        result = analyze_attack_patterns(text, "system_log", "standard")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["attack_patterns_found"] is True
        sql_attacks = [a for a in result_data["data"]["detected_attacks"] if a["attack_type"] == "sql_injection"]
        assert len(sql_attacks) > 0

    def test_analyze_attack_patterns_xss(self):
        """Test attack pattern analysis for XSS."""
        text = "<script>alert('XSS')</script>"
        result = analyze_attack_patterns(text, "user_input", "standard")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["attack_patterns_found"] is True
        xss_attacks = [a for a in result_data["data"]["detected_attacks"] if a["attack_type"] == "xss_attack"]
        assert len(xss_attacks) > 0

    def test_analyze_attack_patterns_social_engineering(self):
        """Test attack pattern analysis for social engineering."""
        text = "URGENT: Click here to verify your account immediately!"
        result = analyze_attack_patterns(text, "vendor_response", "standard")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["attack_patterns_found"] is True
        se_attacks = [a for a in result_data["data"]["detected_attacks"] if a["attack_type"] == "social_engineering"]
        assert len(se_attacks) > 0

    def test_analyze_attack_patterns_no_attacks(self):
        """Test attack pattern analysis with no attacks present."""
        text = "This is a normal, legitimate message with no malicious content."
        result = analyze_attack_patterns(text, "general", "standard")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["attack_patterns_found"] is False
        assert len(result_data["data"]["detected_attacks"]) == 0

    def test_perform_security_assessment_clean_data(self):
        """Test security assessment with clean data."""
        data = {"order_id": "12345", "item": "pizza", "quantity": 2}
        result = perform_security_assessment(data, "comprehensive")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["security_score"] in ["good", "fair"]
        assert result_data["data"]["total_issues"] == 0

    def test_perform_security_assessment_with_pii(self):
        """Test security assessment with PII in data."""
        data = {
            "customer_info": {
                "name": "John Doe",
                "email": "john@example.com",
                "ssn": "123-45-6789"
            },
            "order_details": "Regular order"
        }
        result = perform_security_assessment(data, "comprehensive")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["total_issues"] > 0
        pii_issues = [issue for issue in result_data["data"]["security_issues"] if issue["type"] == "pii_exposure"]
        assert len(pii_issues) > 0

    def test_perform_security_assessment_with_attack_patterns(self):
        """Test security assessment with attack patterns."""
        data = {
            "message": "'; DROP TABLE orders; --",
            "vendor_response": "<script>alert('hack')</script>"
        }
        result = perform_security_assessment(data, "comprehensive")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["total_issues"] > 0
        attack_issues = [issue for issue in result_data["data"]["security_issues"] if issue["type"] == "attack_patterns"]
        assert len(attack_issues) > 0

    def test_sanitize_data_success(self):
        """Test successful data sanitization."""
        data = {
            "customer": {
                "name": "John Doe",
                "email": "john@example.com",
                "ssn": "123-45-6789"
            },
            "order": "pizza order"
        }
        result = sanitize_data(data, "medium", True)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["sanitization_applied"] is True
        assert len(result_data["data"]["sanitization_log"]) > 0

    def test_sanitize_data_high_level(self):
        """Test data sanitization with high level."""
        data = {
            "user_credentials": {
                "username": "john_doe",
                "password": "secret123",
                "email": "john@example.com"
            }
        }
        result = sanitize_data(data, "high", True)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["sanitization_applied"] is True
        sanitized_data = result_data["data"]["sanitized_data"]
        
        # Password should be redacted
        assert sanitized_data["user_credentials"]["password"] == "[REDACTED]"

    def test_sanitize_data_no_sensitive_info(self):
        """Test data sanitization with no sensitive information."""
        data = {"order_id": "12345", "item": "pizza", "quantity": 2}
        result = sanitize_data(data, "medium", True)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["sanitization_applied"] is False
        assert len(result_data["data"]["sanitization_log"]) == 0


class TestSecurityAnalysisToolClasses:
    """Test security analysis tool classes."""

    def test_pii_detection_tool(self):
        """Test PIIDetectionTool class."""
        tool = PIIDetectionTool()
        input_data = PIIDetectionInput(
            text="My credit card is 4532-1234-5678-9012",
            sensitivity_level="medium",
            pii_types=["credit_card"]
        )
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["pii_found"] is True

    def test_attack_pattern_analysis_tool(self):
        """Test AttackPatternAnalysisTool class."""
        tool = AttackPatternAnalysisTool()
        input_data = AttackPatternAnalysisInput(
            text="<script>alert('XSS')</script>",
            context="user_input",
            analysis_depth="standard"
        )
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["attack_patterns_found"] is True

    def test_security_assessment_tool(self):
        """Test SecurityAssessmentTool class."""
        tool = SecurityAssessmentTool()
        input_data = SecurityAssessmentInput(
            data={"ssn": "123-45-6789", "order": "pizza"},
            assessment_type="comprehensive",
            vendor_id="vendor_1"
        )
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "security_score" in result_data["data"]

    def test_data_sanitization_tool(self):
        """Test DataSanitizationTool class."""
        tool = DataSanitizationTool()
        input_data = DataSanitizationInput(
            data={"email": "test@example.com", "ssn": "123-45-6789"},
            sanitization_level="medium",
            preserve_structure=True
        )
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["sanitization_applied"] is True


class TestSecurityAnalysisIntegration:
    """Test integration between security analysis tools."""

    def test_comprehensive_security_workflow(self):
        """Test comprehensive security analysis workflow."""
        # Test data with multiple security issues
        test_data = {
            "customer_info": {
                "name": "John Doe",
                "email": "john@example.com",
                "ssn": "123-45-6789",
                "credit_card": "4532-1234-5678-9012"
            },
            "vendor_message": "URGENT: Click here to verify your account! <script>alert('hack')</script>",
            "order_details": "Regular pizza order"
        }
        
        # 1. Perform security assessment
        assessment_result = perform_security_assessment(test_data, "comprehensive", "vendor_1")
        assessment_data = json.loads(assessment_result)
        assert assessment_data["success"] is True
        assert assessment_data["data"]["total_issues"] > 0
        
        # 2. Sanitize the data
        sanitization_result = sanitize_data(test_data, "high", True)
        sanitization_data = json.loads(sanitization_result)
        assert sanitization_data["success"] is True
        assert sanitization_data["data"]["sanitization_applied"] is True
        
        # 3. Re-assess sanitized data
        sanitized_data = sanitization_data["data"]["sanitized_data"]
        final_assessment = perform_security_assessment(sanitized_data, "comprehensive")
        final_data = json.loads(final_assessment)
        assert final_data["success"] is True
        assert final_data["data"]["total_issues"] < assessment_data["data"]["total_issues"]

    def test_pii_detection_and_sanitization_workflow(self):
        """Test PII detection followed by sanitization workflow."""
        sensitive_text = "Customer John Doe (SSN: 123-45-6789) can be reached at john@example.com or (555) 123-4567"
        
        # 1. Detect PII
        pii_result = detect_pii(sensitive_text, "medium")
        pii_data = json.loads(pii_result)
        assert pii_data["success"] is True
        assert pii_data["data"]["pii_found"] is True
        
        # 2. Sanitize the text (convert to dict format for sanitization)
        text_data = {"message": sensitive_text}
        sanitization_result = sanitize_data(text_data, "medium", True)
        sanitization_data = json.loads(sanitization_result)
        assert sanitization_data["success"] is True
        
        # 3. Verify sanitization worked
        sanitized_text = sanitization_data["data"]["sanitized_data"]["message"]
        assert "123-45-6789" not in sanitized_text  # SSN should be masked
        assert "john@example.com" not in sanitized_text  # Email should be masked

    def test_attack_pattern_detection_workflow(self):
        """Test attack pattern detection workflow."""
        malicious_vendor_message = "URGENT: Click here now! <script>alert('hack')</script> '; DROP TABLE orders; --"
        
        # 1. Analyze attack patterns
        attack_result = analyze_attack_patterns(malicious_vendor_message, "vendor_response", "deep")
        attack_data = json.loads(attack_result)
        assert attack_data["success"] is True
        assert attack_data["data"]["attack_patterns_found"] is True
        
        # 2. Check for specific attack types
        detected_attacks = attack_data["data"]["detected_attacks"]
        attack_types = [attack["attack_type"] for attack in detected_attacks]
        assert "xss_attack" in attack_types
        assert "sql_injection" in attack_types
        assert "social_engineering" in attack_types
        
        # 3. Verify severity assessment
        high_severity_attacks = [a for a in detected_attacks if a["severity"] == "high"]
        assert len(high_severity_attacks) > 0
