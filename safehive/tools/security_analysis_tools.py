"""
Security analysis tools for the SafeHive AI Security Sandbox.

This module provides tools for detecting PII, analyzing attack patterns,
and performing security assessments on vendor communications and data.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import re
from datetime import datetime

from pydantic import Field
from ..utils.logger import get_logger
from .base_tools import BaseSafeHiveTool, ToolInput, ToolOutput, create_tool_output

logger = get_logger(__name__)


# Pydantic models for tool input validation
class PIIDetectionInput(ToolInput):
    """Input for PII detection requests."""
    text: str = Field(description="The text to analyze for PII")
    sensitivity_level: str = Field(default="medium", description="Detection sensitivity (low, medium, high)")
    pii_types: Optional[List[str]] = Field(default=None, description="Specific PII types to detect (credit_card, ssn, email, phone, address)")


class AttackPatternAnalysisInput(ToolInput):
    """Input for attack pattern analysis requests."""
    text: str = Field(description="The text to analyze for attack patterns")
    context: str = Field(default="general", description="Context of the communication (vendor_response, user_input, system_log)")
    analysis_depth: str = Field(default="standard", description="Analysis depth (basic, standard, deep)")


class SecurityAssessmentInput(ToolInput):
    """Input for security assessment requests."""
    data: Dict[str, Any] = Field(description="The data to assess for security issues")
    assessment_type: str = Field(default="comprehensive", description="Type of assessment (basic, comprehensive, deep_scan)")
    vendor_id: Optional[str] = Field(default=None, description="Associated vendor ID if applicable")


class DataSanitizationInput(ToolInput):
    """Input for data sanitization requests."""
    data: Dict[str, Any] = Field(description="The data to sanitize")
    sanitization_level: str = Field(default="medium", description="Sanitization level (low, medium, high)")
    preserve_structure: bool = Field(default=True, description="Whether to preserve data structure")


# PII Detection Patterns
PII_PATTERNS = {
    "credit_card": [
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Standard credit card format
        r'\b\d{13,19}\b'  # Generic 13-19 digit number
    ],
    "ssn": [
        r'\b\d{3}-\d{2}-\d{4}\b',  # Standard SSN format
        r'\b\d{9}\b'  # 9 consecutive digits
    ],
    "email": [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Standard email format
    ],
    "phone": [
        r'\b\d{3}-\d{3}-\d{4}\b',  # Standard phone format
        r'\(\d{3}\)\s+\d{3}-\d{4}',  # Parentheses format with space
        r'\b\d{10}\b'  # 10 consecutive digits
    ],
    "address": [
        r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b',  # Street address
        r'\b[A-Za-z\s]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\s+\d+\b'
    ]
}

# Attack Pattern Detection
ATTACK_PATTERNS = {
    "sql_injection": [
        r'(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)\s+.*(from|into|table|database)',
        r'(?i)(\'|\"|;|--|\/\*|\*\/)',
        r'(?i)(or|and)\s+.*=.*',
        r'(?i)(waitfor|delay|sleep)\s+.*'
    ],
    "xss_attack": [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>'
    ],
    "path_traversal": [
        r'\.\./',
        r'\.\.\\',
        r'/\.\./',
        r'\\\.\.\\',
        r'%2e%2e%2f',
        r'%2e%2e%5c'
    ],
    "social_engineering": [
        r'(?i)(urgent|immediately|asap|limited time|act now|click here|verify|confirm|suspended|locked)',
        r'(?i)(credit card|ssn|social security|password|account|login)',
        r'(?i)(free|win|prize|lottery|congratulations)',
        r'(?i)(phishing|scam|fraud|suspicious)'
    ],
    "data_collection": [
        r'(?i)(provide|enter|input|submit|send).*(personal|private|sensitive|confidential)',
        r'(?i)(information|details|data).*(required|needed|necessary)',
        r'(?i)(form|field|box).*(fill|complete|required)'
    ]
}


def detect_pii(text: str, sensitivity_level: str = "medium", pii_types: List[str] = None) -> str:
    """Detect Personally Identifiable Information (PII) in text.
    
    Args:
        text: The text to analyze
        sensitivity_level: Detection sensitivity level
        pii_types: Specific PII types to detect
        
    Returns:
        A JSON string containing PII detection results
    """
    try:
        if not text or not text.strip():
            return create_tool_output(
                success=True,
                message="No text provided for PII analysis",
                data={"pii_found": False, "detected_pii": []}
            ).to_json()
        
        detected_pii = []
        pii_types_to_check = pii_types or list(PII_PATTERNS.keys())
        
        for pii_type in pii_types_to_check:
            if pii_type not in PII_PATTERNS:
                continue
            
            patterns = PII_PATTERNS[pii_type]
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Mask the detected PII for security
                    masked_value = mask_pii(match.group(), pii_type)
                    
                    detected_pii.append({
                        "type": pii_type,
                        "pattern": pattern,
                        "position": match.start(),
                        "length": len(match.group()),
                        "masked_value": masked_value,
                        "confidence": calculate_confidence(match.group(), pii_type, sensitivity_level)
                    })
        
        # Filter by sensitivity level
        if sensitivity_level == "low":
            detected_pii = [pii for pii in detected_pii if pii["confidence"] > 0.8]
        elif sensitivity_level == "high":
            detected_pii = [pii for pii in detected_pii if pii["confidence"] > 0.3]
        else:  # medium
            detected_pii = [pii for pii in detected_pii if pii["confidence"] > 0.5]
        
        # Remove duplicates
        unique_pii = []
        seen_positions = set()
        for pii in detected_pii:
            position_key = (pii["type"], pii["position"])
            if position_key not in seen_positions:
                unique_pii.append(pii)
                seen_positions.add(position_key)
        
        risk_level = "low"
        if len(unique_pii) > 3:
            risk_level = "high"
        elif len(unique_pii) > 1:
            risk_level = "medium"
        
        result_data = {
            "text_length": len(text),
            "sensitivity_level": sensitivity_level,
            "pii_types_checked": pii_types_to_check,
            "pii_found": len(unique_pii) > 0,
            "detected_pii": unique_pii,
            "pii_count": len(unique_pii),
            "risk_level": risk_level,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"PII detection completed: {len(unique_pii)} PII instances found")
        
        return create_tool_output(
            success=True,
            message=f"PII detection completed: {len(unique_pii)} instances found",
            data=result_data
        ).to_json()
        
    except Exception as e:
        logger.error(f"Failed to detect PII: {e}")
        return create_tool_output(
            success=False,
            message=f"Failed to detect PII: {str(e)}",
            data={"text_length": len(text) if text else 0}
        ).to_json()


def analyze_attack_patterns(text: str, context: str = "general", analysis_depth: str = "standard") -> str:
    """Analyze text for potential attack patterns.
    
    Args:
        text: The text to analyze
        context: Context of the communication
        analysis_depth: Depth of analysis
        
    Returns:
        A JSON string containing attack pattern analysis results
    """
    try:
        if not text or not text.strip():
            return create_tool_output(
                success=True,
                message="No text provided for attack pattern analysis",
                data={"attack_patterns_found": False, "detected_attacks": []}
            ).to_json()
        
        detected_attacks = []
        
        for attack_type, patterns in ATTACK_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    confidence = calculate_attack_confidence(match.group(), attack_type, context)
                    
                    detected_attacks.append({
                        "attack_type": attack_type,
                        "pattern": pattern,
                        "matched_text": match.group(),
                        "position": match.start(),
                        "length": len(match.group()),
                        "confidence": confidence,
                        "severity": get_attack_severity(attack_type, confidence)
                    })
        
        # Filter by analysis depth
        if analysis_depth == "basic":
            detected_attacks = [attack for attack in detected_attacks if attack["confidence"] > 0.7]
        elif analysis_depth == "deep":
            detected_attacks = [attack for attack in detected_attacks if attack["confidence"] > 0.3]
        else:  # standard
            detected_attacks = [attack for attack in detected_attacks if attack["confidence"] > 0.5]
        
        # Remove duplicates and sort by severity
        unique_attacks = []
        seen_positions = set()
        for attack in detected_attacks:
            position_key = (attack["attack_type"], attack["position"])
            if position_key not in seen_positions:
                unique_attacks.append(attack)
                seen_positions.add(position_key)
        
        unique_attacks.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Calculate overall risk assessment
        high_severity_count = len([a for a in unique_attacks if a["severity"] == "high"])
        medium_severity_count = len([a for a in unique_attacks if a["severity"] == "medium"])
        
        overall_risk = "low"
        if high_severity_count > 0:
            overall_risk = "high"
        elif medium_severity_count > 1 or len(unique_attacks) > 2:
            overall_risk = "medium"
        
        result_data = {
            "text_length": len(text),
            "context": context,
            "analysis_depth": analysis_depth,
            "attack_patterns_found": len(unique_attacks) > 0,
            "detected_attacks": unique_attacks,
            "attack_count": len(unique_attacks),
            "high_severity_count": high_severity_count,
            "medium_severity_count": medium_severity_count,
            "overall_risk": overall_risk,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Attack pattern analysis completed: {len(unique_attacks)} patterns detected")
        
        return create_tool_output(
            success=True,
            message=f"Attack pattern analysis completed: {len(unique_attacks)} patterns detected",
            data=result_data
        ).to_json()
        
    except Exception as e:
        logger.error(f"Failed to analyze attack patterns: {e}")
        return create_tool_output(
            success=False,
            message=f"Failed to analyze attack patterns: {str(e)}",
            data={"text_length": len(text) if text else 0}
        ).to_json()


def perform_security_assessment(data: Dict[str, Any], assessment_type: str = "comprehensive", vendor_id: str = None) -> str:
    """Perform a comprehensive security assessment on data.
    
    Args:
        data: The data to assess
        assessment_type: Type of assessment to perform
        vendor_id: Associated vendor ID if applicable
        
    Returns:
        A JSON string containing security assessment results
    """
    try:
        if not data:
            return create_tool_output(
                success=True,
                message="No data provided for security assessment",
                data={"security_issues_found": False, "assessment_results": {}}
            ).to_json()
        
        assessment_results = {
            "data_types": list(data.keys()),
            "data_size": len(str(data)),
            "vendor_id": vendor_id,
            "assessment_type": assessment_type,
            "security_issues": [],
            "risk_factors": [],
            "recommendations": []
        }
        
        # Convert data to string for analysis
        data_text = json.dumps(data)
        
        # PII Detection
        pii_result = detect_pii(data_text, "medium")
        pii_data = json.loads(pii_result)
        if pii_data.get("data", {}).get("pii_found"):
            assessment_results["security_issues"].append({
                "type": "pii_exposure",
                "severity": "high",
                "description": f"PII detected: {pii_data['data']['pii_count']} instances",
                "details": pii_data["data"]["detected_pii"]
            })
            assessment_results["risk_factors"].append("sensitive_data_exposure")
            assessment_results["recommendations"].append("Implement data sanitization and access controls")
        
        # Attack Pattern Analysis
        attack_result = analyze_attack_patterns(data_text, "system_log", assessment_type)
        attack_data = json.loads(attack_result)
        if attack_data.get("data", {}).get("attack_patterns_found"):
            assessment_results["security_issues"].append({
                "type": "attack_patterns",
                "severity": attack_data["data"]["overall_risk"],
                "description": f"Attack patterns detected: {attack_data['data']['attack_count']} patterns",
                "details": attack_data["data"]["detected_attacks"]
            })
            assessment_results["risk_factors"].append("potential_malicious_activity")
            assessment_results["recommendations"].append("Review and validate all external inputs")
        
        # Data Structure Analysis
        if "password" in data_text.lower() or "secret" in data_text.lower():
            assessment_results["security_issues"].append({
                "type": "credential_exposure",
                "severity": "high",
                "description": "Potential credential information detected",
                "details": "Text contains password or secret-related keywords"
            })
            assessment_results["risk_factors"].append("credential_exposure")
            assessment_results["recommendations"].append("Never store or transmit credentials in plain text")
        
        # Vendor-specific risk assessment
        if vendor_id:
            vendor_risk_factors = assess_vendor_risk(vendor_id)
            assessment_results["vendor_risk_assessment"] = vendor_risk_factors
            if vendor_risk_factors.get("overall_risk") == "high":
                assessment_results["risk_factors"].append("high_risk_vendor")
                assessment_results["recommendations"].append("Consider additional security measures for this vendor")
        
        # Calculate overall security score
        total_issues = len(assessment_results["security_issues"])
        high_severity_issues = len([issue for issue in assessment_results["security_issues"] if issue["severity"] == "high"])
        
        if high_severity_issues > 0:
            security_score = "critical"
        elif total_issues > 2:
            security_score = "poor"
        elif total_issues > 0:
            security_score = "fair"
        else:
            security_score = "good"
        
        assessment_results.update({
            "total_issues": total_issues,
            "high_severity_issues": high_severity_issues,
            "security_score": security_score,
            "assessment_timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Security assessment completed: {security_score} score with {total_issues} issues")
        
        return create_tool_output(
            success=True,
            message=f"Security assessment completed: {security_score} security score",
            data=assessment_results
        ).to_json()
        
    except Exception as e:
        logger.error(f"Failed to perform security assessment: {e}")
        return create_tool_output(
            success=False,
            message=f"Failed to perform security assessment: {str(e)}",
            data={"data_keys": list(data.keys()) if data else []}
        ).to_json()


def sanitize_data(data: Dict[str, Any], sanitization_level: str = "medium", preserve_structure: bool = True) -> str:
    """Sanitize data by removing or masking sensitive information.
    
    Args:
        data: The data to sanitize
        sanitization_level: Level of sanitization to apply
        preserve_structure: Whether to preserve the data structure
        
    Returns:
        A JSON string containing sanitized data
    """
    try:
        if not data:
            return create_tool_output(
                success=True,
                message="No data provided for sanitization",
                data={"sanitized_data": {}, "sanitization_applied": False}
            ).to_json()
        
        sanitized_data = data.copy()
        sanitization_log = []
        
        # Convert to string for pattern matching
        data_text = json.dumps(data)
        
        # Detect PII to sanitize
        pii_result = detect_pii(data_text, "low")  # Use low sensitivity to catch more
        pii_data = json.loads(pii_result)
        
        if pii_data.get("data", {}).get("pii_found"):
            for pii_instance in pii_data["data"]["detected_pii"]:
                original_value = data_text[pii_instance["position"]:pii_instance["position"] + pii_instance["length"]]
                sanitized_value = mask_pii(original_value, pii_instance["type"])
                
                # Replace in the JSON string
                data_text = data_text.replace(original_value, sanitized_value)
                sanitization_log.append({
                    "type": pii_instance["type"],
                    "original_position": pii_instance["position"],
                    "sanitization_method": "masking",
                    "confidence": pii_instance["confidence"]
                })
        
        # Parse sanitized data back to dictionary
        try:
            sanitized_dict = json.loads(data_text)
            if preserve_structure:
                sanitized_data = sanitized_dict
        except json.JSONDecodeError:
            # If JSON parsing fails, create a sanitized version manually
            sanitized_data = create_sanitized_version(data, sanitization_level)
        
        # Additional sanitization based on level (work on the dictionary structure)
        if sanitization_level == "high":
            sanitized_data = sanitize_sensitive_keywords(sanitized_data, sanitization_log)
        
        result_data = {
            "original_data_size": len(str(data)),
            "sanitized_data_size": len(str(sanitized_data)),
            "sanitization_level": sanitization_level,
            "preserve_structure": preserve_structure,
            "sanitized_data": sanitized_data,
            "sanitization_log": sanitization_log,
            "sanitization_applied": len(sanitization_log) > 0,
            "sanitization_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Data sanitization completed: {len(sanitization_log)} sanitizations applied")
        
        return create_tool_output(
            success=True,
            message=f"Data sanitization completed: {len(sanitization_log)} sanitizations applied",
            data=result_data
        ).to_json()
        
    except Exception as e:
        logger.error(f"Failed to sanitize data: {e}")
        return create_tool_output(
            success=False,
            message=f"Failed to sanitize data: {str(e)}",
            data={"original_data_keys": list(data.keys()) if data else []}
        ).to_json()


# Helper functions
def mask_pii(value: str, pii_type: str) -> str:
    """Mask PII value based on type."""
    if pii_type == "credit_card":
        if len(value) >= 4:
            return "*" * (len(value) - 4) + value[-4:]
        return "*" * len(value)
    elif pii_type == "ssn":
        return "***-**-" + value[-4:] if len(value) >= 4 else "*" * len(value)
    elif pii_type == "email":
        if "@" in value:
            local, domain = value.split("@", 1)
            return "*" * len(local) + "@" + domain
        return "*" * len(value)
    elif pii_type == "phone":
        if len(value) >= 4:
            return "*" * (len(value) - 4) + value[-4:]
        return "*" * len(value)
    else:
        return "*" * len(value)


def calculate_confidence(value: str, pii_type: str, sensitivity_level: str) -> float:
    """Calculate confidence score for PII detection."""
    base_confidence = 0.5
    
    # Adjust based on value characteristics
    if pii_type == "credit_card":
        if re.match(r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}', value):
            base_confidence = 0.9
    elif pii_type == "ssn":
        if re.match(r'\d{3}-\d{2}-\d{4}', value):
            base_confidence = 0.9
    elif pii_type == "email":
        if "@" in value and "." in value:
            base_confidence = 0.8
    elif pii_type == "phone":
        if re.match(r'\d{3}-\d{3}-\d{4}', value) or re.match(r'\(\d{3}\)\s+\d{3}-\d{4}', value):
            base_confidence = 0.8
    
    # Adjust based on sensitivity level
    if sensitivity_level == "low":
        base_confidence *= 1.2
    elif sensitivity_level == "high":
        base_confidence *= 0.8
    
    return min(1.0, max(0.0, base_confidence))


def calculate_attack_confidence(matched_text: str, attack_type: str, context: str) -> float:
    """Calculate confidence score for attack pattern detection."""
    base_confidence = 0.5
    
    # Adjust based on attack type
    if attack_type == "sql_injection":
        if any(keyword in matched_text.lower() for keyword in ["union", "select", "drop"]):
            base_confidence = 0.8
    elif attack_type == "xss_attack":
        if "<script" in matched_text.lower():
            base_confidence = 0.9
    elif attack_type == "social_engineering":
        if any(keyword in matched_text.lower() for keyword in ["urgent", "click here", "verify"]):
            base_confidence = 0.7
    
    # Adjust based on context
    if context == "vendor_response" and attack_type == "social_engineering":
        base_confidence *= 1.2
    
    return min(1.0, max(0.0, base_confidence))


def get_attack_severity(attack_type: str, confidence: float) -> str:
    """Determine attack severity based on type and confidence."""
    if attack_type in ["sql_injection", "xss_attack"] and confidence > 0.7:
        return "high"
    elif attack_type in ["path_traversal", "data_collection"] and confidence > 0.6:
        return "medium"
    elif confidence > 0.5:
        return "medium"
    else:
        return "low"


def assess_vendor_risk(vendor_id: str) -> Dict[str, Any]:
    """Assess risk level for a specific vendor."""
    # Mock vendor risk assessment
    risk_factors = {
        "vendor_id": vendor_id,
        "overall_risk": "medium",
        "risk_factors": ["unknown_reputation", "new_vendor"],
        "recommendations": ["monitor_communications", "verify_credentials"]
    }
    
    # In a real implementation, this would check against a vendor database
    if "malicious" in vendor_id.lower():
        risk_factors.update({
            "overall_risk": "high",
            "risk_factors": ["known_malicious", "suspicious_behavior", "poor_reputation"],
            "recommendations": ["block_communication", "report_to_security", "enhanced_monitoring"]
        })
    
    return risk_factors


def create_sanitized_version(data: Dict[str, Any], sanitization_level: str) -> Dict[str, Any]:
    """Create a sanitized version of the data structure."""
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, dict):
            sanitized[key] = create_sanitized_version(value, sanitization_level)
        elif isinstance(value, str):
            # Basic string sanitization
            if any(keyword in key.lower() for keyword in ["password", "secret", "key"]):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        else:
            sanitized[key] = value
    return sanitized


def sanitize_sensitive_keywords(data: Dict[str, Any], sanitization_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitize sensitive keywords in the data structure."""
    sanitized = {}
    sensitive_keywords = ["password", "secret", "key", "token", "auth"]
    
    for key, value in data.items():
        if isinstance(value, dict):
            sanitized[key] = sanitize_sensitive_keywords(value, sanitization_log)
        elif isinstance(value, str):
            # Check if the key contains sensitive keywords
            if any(keyword in key.lower() for keyword in sensitive_keywords):
                sanitized[key] = "[REDACTED]"
                sanitization_log.append({
                    "type": "sensitive_keyword",
                    "keyword": key,
                    "sanitization_method": "redaction"
                })
            else:
                sanitized[key] = value
        else:
            sanitized[key] = value
    
    return sanitized


# Tool classes
class PIIDetectionTool(BaseSafeHiveTool):
    name: str = "detect_pii"
    description: str = "Detect Personally Identifiable Information (PII) in text data."
    args_schema: type[PIIDetectionInput] = PIIDetectionInput

    def _execute(self, input_data: PIIDetectionInput) -> str:
        return detect_pii(input_data.text, input_data.sensitivity_level, input_data.pii_types)


class AttackPatternAnalysisTool(BaseSafeHiveTool):
    name: str = "analyze_attack_patterns"
    description: str = "Analyze text for potential attack patterns and malicious content."
    args_schema: type[AttackPatternAnalysisInput] = AttackPatternAnalysisInput

    def _execute(self, input_data: AttackPatternAnalysisInput) -> str:
        return analyze_attack_patterns(input_data.text, input_data.context, input_data.analysis_depth)


class SecurityAssessmentTool(BaseSafeHiveTool):
    name: str = "perform_security_assessment"
    description: str = "Perform a comprehensive security assessment on data and communications."
    args_schema: type[SecurityAssessmentInput] = SecurityAssessmentInput

    def _execute(self, input_data: SecurityAssessmentInput) -> str:
        return perform_security_assessment(input_data.data, input_data.assessment_type, input_data.vendor_id)


class DataSanitizationTool(BaseSafeHiveTool):
    name: str = "sanitize_data"
    description: str = "Sanitize data by removing or masking sensitive information."
    args_schema: type[DataSanitizationInput] = DataSanitizationInput

    def _execute(self, input_data: DataSanitizationInput) -> str:
        return sanitize_data(input_data.data, input_data.sanitization_level, input_data.preserve_structure)


# Convenience function to get all security analysis tools
def get_security_analysis_tools() -> List[BaseSafeHiveTool]:
    """Get all security analysis tools for agent configuration."""
    return [
        PIIDetectionTool(),
        AttackPatternAnalysisTool(),
        SecurityAssessmentTool(),
        DataSanitizationTool()
    ]
