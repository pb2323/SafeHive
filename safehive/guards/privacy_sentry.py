"""
Privacy Sentry - PII Detection and Prevention

This module implements a comprehensive privacy protection system that detects,
classifies, and prevents Personally Identifiable Information (PII) from being
shared inappropriately in agent communications.
"""

import re
import json
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Pattern
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.metrics import record_metric, MetricType

logger = get_logger(__name__)


class PIIType(Enum):
    """Types of Personally Identifiable Information."""
    # Personal Identifiers
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    
    # Financial Information
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    ROUTING_NUMBER = "routing_number"
    
    # Address Information
    STREET_ADDRESS = "street_address"
    POSTAL_CODE = "postal_code"
    CITY = "city"
    STATE = "state"
    COUNTRY = "country"
    
    # Health Information
    MEDICAL_RECORD = "medical_record"
    HEALTH_INSURANCE = "health_insurance"
    MEDICATION = "medication"
    DIAGNOSIS = "diagnosis"
    
    # Biometric Information
    FINGERPRINT = "fingerprint"
    FACE_ID = "face_id"
    VOICE_PRINT = "voice_print"
    
    # Digital Identifiers
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    DEVICE_ID = "device_id"
    USER_ID = "user_id"
    SESSION_ID = "session_id"
    
    # Other Sensitive Information
    DATE_OF_BIRTH = "date_of_birth"
    MOTHER_MAIDEN_NAME = "mother_maiden_name"
    SECURITY_QUESTION = "security_question"
    PASSWORD = "password"
    API_KEY = "api_key"
    TOKEN = "token"


class PIISeverity(Enum):
    """Severity levels for PII detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PrivacyAction(Enum):
    """Actions to take when PII is detected."""
    ALLOW = "allow"
    BLOCK = "block"
    MASK = "mask"
    REDACT = "redact"
    ENCRYPT = "encrypt"
    AUDIT = "audit"
    QUARANTINE = "quarantine"


@dataclass
class PIIDetectionResult:
    """Result of PII detection."""
    pii_type: PIIType
    severity: PIISeverity
    confidence: float
    start_pos: int
    end_pos: int
    detected_value: str
    masked_value: str
    context: str
    detection_method: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pii_type": self.pii_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "detected_value": self.detected_value,
            "masked_value": self.masked_value,
            "context": self.context,
            "detection_method": self.detection_method,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PrivacyViolation:
    """Represents a privacy violation."""
    violation_id: str
    pii_detections: List[PIIDetectionResult]
    source: str
    destination: str
    action_taken: PrivacyAction
    severity: PIISeverity
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "violation_id": self.violation_id,
            "pii_detections": [detection.to_dict() for detection in self.pii_detections],
            "source": self.source,
            "destination": self.destination,
            "action_taken": self.action_taken.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes
        }


@dataclass
class PrivacyRule:
    """Privacy rule for PII handling."""
    rule_id: str
    name: str
    description: str
    pii_types: List[PIIType]
    severity: PIISeverity
    action: PrivacyAction
    conditions: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "pii_types": [pii_type.value for pii_type in self.pii_types],
            "severity": self.severity.value,
            "action": self.action.value,
            "conditions": self.conditions,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class PrivacyPolicy:
    """Privacy policy configuration."""
    policy_id: str
    name: str
    description: str
    rules: List[PrivacyRule]
    default_action: PrivacyAction = PrivacyAction.AUDIT
    strict_mode: bool = False
    audit_all: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "rules": [rule.to_dict() for rule in self.rules],
            "default_action": self.default_action.value,
            "strict_mode": self.strict_mode,
            "audit_all": self.audit_all,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class PrivacyAuditLog:
    """Audit log entry for privacy events."""
    log_id: str
    event_type: str
    timestamp: datetime
    source: str
    destination: str
    pii_detected: List[PIIType]
    action_taken: PrivacyAction
    severity: PIISeverity
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "log_id": self.log_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "destination": self.destination,
            "pii_detected": [pii_type.value for pii_type in self.pii_detected],
            "action_taken": self.action_taken.value,
            "severity": self.severity.value,
            "details": self.details
        }


class PrivacySentry:
    """Privacy protection system for detecting and preventing PII sharing."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_privacy"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # PII detection patterns
        self.pii_patterns: Dict[PIIType, List[Pattern]] = {}
        self.pii_severity_map: Dict[PIIType, PIISeverity] = {}
        
        # Privacy rules and policies
        self.privacy_rules: Dict[str, PrivacyRule] = {}
        self.privacy_policies: Dict[str, PrivacyPolicy] = {}
        self.active_policy: Optional[PrivacyPolicy] = None
        
        # Audit and violation tracking
        self.audit_logs: List[PrivacyAuditLog] = []
        self.privacy_violations: List[PrivacyViolation] = []
        
        # Statistics and metrics
        self.detection_stats: Dict[str, int] = {}
        self.violation_stats: Dict[str, int] = {}
        
        # Initialize default patterns and rules
        self._initialize_pii_patterns()
        self._initialize_default_rules()
        self._load_privacy_data()
        
        logger.info("Privacy Sentry initialized")
    
    def _initialize_pii_patterns(self) -> None:
        """Initialize PII detection patterns."""
        # Email patterns
        email_patterns = [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            re.compile(r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Z|a-z]{2,}\b')
        ]
        self.pii_patterns[PIIType.EMAIL] = email_patterns
        self.pii_severity_map[PIIType.EMAIL] = PIISeverity.MEDIUM
        
        # Phone number patterns
        phone_patterns = [
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # US format
            re.compile(r'\(\d{3}\)\s+\d{3}-\d{4}'),  # US format with parentheses and space
            re.compile(r'\b\+1[-.]?\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # US with country code
            re.compile(r'\b\d{10}\b'),  # 10 digits
            re.compile(r'\b\d{3}\s\d{3}\s\d{4}\b')  # Space separated
        ]
        self.pii_patterns[PIIType.PHONE] = phone_patterns
        self.pii_severity_map[PIIType.PHONE] = PIISeverity.MEDIUM
        
        # Credit card patterns
        credit_card_patterns = [
            re.compile(r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b'),  # 16 digits
            re.compile(r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b'),  # Space separated
            re.compile(r'\b\d{4}-\d{4}-\d{4}-\d{4}\b'),  # Dash separated
        ]
        self.pii_patterns[PIIType.CREDIT_CARD] = credit_card_patterns
        self.pii_severity_map[PIIType.CREDIT_CARD] = PIISeverity.CRITICAL
        
        # SSN patterns
        ssn_patterns = [
            re.compile(r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'),  # Standard SSN format
            re.compile(r'\b\d{3}\s\d{2}\s\d{4}\b'),  # Space separated
            re.compile(r'\b\d{9}\b')  # 9 digits (less specific)
        ]
        self.pii_patterns[PIIType.SSN] = ssn_patterns
        self.pii_severity_map[PIIType.SSN] = PIISeverity.CRITICAL
        
        # IP address patterns
        ip_patterns = [
            re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),  # IPv4
            re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),  # IPv6
        ]
        self.pii_patterns[PIIType.IP_ADDRESS] = ip_patterns
        self.pii_severity_map[PIIType.IP_ADDRESS] = PIISeverity.MEDIUM
        
        # Date of birth patterns
        dob_patterns = [
            re.compile(r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b'),  # MM/DD/YYYY
            re.compile(r'\b(?:0[1-9]|[12][0-9]|3[01])[-/](?:0[1-9]|1[0-2])[-/](?:19|20)\d{2}\b'),  # DD/MM/YYYY
            re.compile(r'\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])\b'),  # YYYY/MM/DD
        ]
        self.pii_patterns[PIIType.DATE_OF_BIRTH] = dob_patterns
        self.pii_severity_map[PIIType.DATE_OF_BIRTH] = PIISeverity.HIGH
        
        # Name patterns (basic)
        name_patterns = [
            re.compile(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b'),  # First Last
            re.compile(r'\b[A-Z][a-z]+\s[A-Z]\.\s[A-Z][a-z]+\b'),  # First M. Last
            re.compile(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+\b'),  # First Middle Last
        ]
        self.pii_patterns[PIIType.NAME] = name_patterns
        self.pii_severity_map[PIIType.NAME] = PIISeverity.MEDIUM
        
        # Street address patterns
        address_patterns = [
            re.compile(r'\b\d+\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b'),
            re.compile(r'\b\d+\s+[A-Za-z0-9\s]+(?:St|Ave|Rd|Dr|Ln|Blvd)\.?\b'),
        ]
        self.pii_patterns[PIIType.STREET_ADDRESS] = address_patterns
        self.pii_severity_map[PIIType.STREET_ADDRESS] = PIISeverity.HIGH
        
        # Postal code patterns
        postal_patterns = [
            re.compile(r'\b\d{5}(?:-\d{4})?\b'),  # US ZIP code
            re.compile(r'\b[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d\b'),  # Canadian postal code
        ]
        self.pii_patterns[PIIType.POSTAL_CODE] = postal_patterns
        self.pii_severity_map[PIIType.POSTAL_CODE] = PIISeverity.MEDIUM
        
        # Bank account patterns
        bank_account_patterns = [
            re.compile(r'\b(?:account|acct)\s*(?:number|#)?\s*:?\s*\d{8,17}\b', re.IGNORECASE),  # With context
            re.compile(r'\b\d{8,17}\b'),  # 8-17 digits (typical bank account length)
        ]
        self.pii_patterns[PIIType.BANK_ACCOUNT] = bank_account_patterns
        self.pii_severity_map[PIIType.BANK_ACCOUNT] = PIISeverity.CRITICAL
        
        # API key patterns
        api_key_patterns = [
            re.compile(r'\b[A-Za-z0-9]{20,}\b'),  # Long alphanumeric strings
            re.compile(r'\b[A-Za-z0-9_-]{32,}\b'),  # With underscores and dashes
        ]
        self.pii_patterns[PIIType.API_KEY] = api_key_patterns
        self.pii_severity_map[PIIType.API_KEY] = PIISeverity.CRITICAL
        
        # Password patterns (basic)
        password_patterns = [
            re.compile(r'\b(?:password|pwd|pass)\s*[:=]\s*[^\s]+\b', re.IGNORECASE),
        ]
        self.pii_patterns[PIIType.PASSWORD] = password_patterns
        self.pii_severity_map[PIIType.PASSWORD] = PIISeverity.CRITICAL
    
    def _initialize_default_rules(self) -> None:
        """Initialize default privacy rules."""
        # Critical PII rule
        critical_rule = PrivacyRule(
            rule_id="critical_pii_rule",
            name="Critical PII Protection",
            description="Block or mask critical PII like SSN, credit cards, bank accounts",
            pii_types=[PIIType.SSN, PIIType.CREDIT_CARD, PIIType.BANK_ACCOUNT, PIIType.API_KEY, PIIType.PASSWORD],
            severity=PIISeverity.CRITICAL,
            action=PrivacyAction.BLOCK
        )
        self.privacy_rules["critical_pii_rule"] = critical_rule
        
        # High severity PII rule
        high_rule = PrivacyRule(
            rule_id="high_pii_rule",
            name="High Severity PII Protection",
            description="Mask high severity PII like addresses and dates of birth",
            pii_types=[PIIType.STREET_ADDRESS, PIIType.DATE_OF_BIRTH],
            severity=PIISeverity.HIGH,
            action=PrivacyAction.MASK
        )
        self.privacy_rules["high_pii_rule"] = high_rule
        
        # Medium severity PII rule
        medium_rule = PrivacyRule(
            rule_id="medium_pii_rule",
            name="Medium Severity PII Protection",
            description="Audit medium severity PII like emails and phone numbers",
            pii_types=[PIIType.EMAIL, PIIType.PHONE, PIIType.NAME, PIIType.POSTAL_CODE, PIIType.IP_ADDRESS],
            severity=PIISeverity.MEDIUM,
            action=PrivacyAction.AUDIT
        )
        self.privacy_rules["medium_pii_rule"] = medium_rule
        
        # Create default policy
        default_policy = PrivacyPolicy(
            policy_id="default_policy",
            name="Default Privacy Policy",
            description="Default privacy policy for SafeHive AI Security Sandbox",
            rules=list(self.privacy_rules.values()),
            default_action=PrivacyAction.AUDIT,
            strict_mode=False,
            audit_all=True
        )
        self.privacy_policies["default_policy"] = default_policy
        self.active_policy = default_policy
    
    def _load_privacy_data(self) -> None:
        """Load privacy data from storage."""
        # Load audit logs
        audit_file = self.storage_path / "audit_logs.json"
        if audit_file.exists():
            try:
                with open(audit_file, 'r') as f:
                    data = json.load(f)
                    for log_data in data:
                        log = self._reconstruct_audit_log(log_data)
                        if log:
                            self.audit_logs.append(log)
                logger.info(f"Loaded {len(self.audit_logs)} audit log entries")
            except Exception as e:
                logger.error(f"Failed to load audit logs: {e}")
        
        # Load violations
        violations_file = self.storage_path / "privacy_violations.json"
        if violations_file.exists():
            try:
                with open(violations_file, 'r') as f:
                    data = json.load(f)
                    for violation_data in data:
                        violation = self._reconstruct_privacy_violation(violation_data)
                        if violation:
                            self.privacy_violations.append(violation)
                logger.info(f"Loaded {len(self.privacy_violations)} privacy violations")
            except Exception as e:
                logger.error(f"Failed to load privacy violations: {e}")
    
    def _save_privacy_data(self) -> None:
        """Save privacy data to storage."""
        # Save audit logs
        audit_file = self.storage_path / "audit_logs.json"
        try:
            data = [log.to_dict() for log in self.audit_logs[-1000:]]  # Keep last 1000 entries
            with open(audit_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved audit logs")
        except Exception as e:
            logger.error(f"Failed to save audit logs: {e}")
        
        # Save violations
        violations_file = self.storage_path / "privacy_violations.json"
        try:
            data = [violation.to_dict() for violation in self.privacy_violations]
            with open(violations_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved privacy violations")
        except Exception as e:
            logger.error(f"Failed to save privacy violations: {e}")
    
    def _reconstruct_audit_log(self, data: Dict[str, Any]) -> Optional[PrivacyAuditLog]:
        """Reconstruct PrivacyAuditLog from stored data."""
        try:
            log = PrivacyAuditLog(
                log_id=data["log_id"],
                event_type=data["event_type"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                source=data["source"],
                destination=data["destination"],
                pii_detected=[PIIType(pii_type) for pii_type in data["pii_detected"]],
                action_taken=PrivacyAction(data["action_taken"]),
                severity=PIISeverity(data["severity"]),
                details=data.get("details", {})
            )
            return log
        except Exception as e:
            logger.error(f"Failed to reconstruct audit log: {e}")
            return None
    
    def _reconstruct_privacy_violation(self, data: Dict[str, Any]) -> Optional[PrivacyViolation]:
        """Reconstruct PrivacyViolation from stored data."""
        try:
            # Reconstruct PII detections
            pii_detections = []
            for detection_data in data.get("pii_detections", []):
                detection = PIIDetectionResult(
                    pii_type=PIIType(detection_data["pii_type"]),
                    severity=PIISeverity(detection_data["severity"]),
                    confidence=detection_data["confidence"],
                    start_pos=detection_data["start_pos"],
                    end_pos=detection_data["end_pos"],
                    detected_value=detection_data["detected_value"],
                    masked_value=detection_data["masked_value"],
                    context=detection_data["context"],
                    detection_method=detection_data["detection_method"],
                    timestamp=datetime.fromisoformat(detection_data["timestamp"])
                )
                pii_detections.append(detection)
            
            violation = PrivacyViolation(
                violation_id=data["violation_id"],
                pii_detections=pii_detections,
                source=data["source"],
                destination=data["destination"],
                action_taken=PrivacyAction(data["action_taken"]),
                severity=PIISeverity(data["severity"]),
                timestamp=datetime.fromisoformat(data["timestamp"]),
                resolved=data.get("resolved", False),
                resolution_notes=data.get("resolution_notes")
            )
            
            return violation
        except Exception as e:
            logger.error(f"Failed to reconstruct privacy violation: {e}")
            return None
    
    def detect_pii(self, text: str, context: Optional[str] = None) -> List[PIIDetectionResult]:
        """Detect PII in text."""
        detections = []
        
        for pii_type, patterns in self.pii_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    detected_value = match.group()
                    confidence = self._calculate_confidence(pii_type, detected_value, text, match.start())
                    
                    if confidence > 0.5:  # Only include high-confidence detections
                        masked_value = self._mask_pii(pii_type, detected_value)
                        
                        detection = PIIDetectionResult(
                            pii_type=pii_type,
                            severity=self.pii_severity_map[pii_type],
                            confidence=confidence,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            detected_value=detected_value,
                            masked_value=masked_value,
                            context=context or text[max(0, match.start()-20):match.end()+20],
                            detection_method="regex_pattern"
                        )
                        detections.append(detection)
        
        # Remove overlapping detections (keep highest confidence)
        detections = self._remove_overlapping_detections(detections)
        
        # Update statistics
        for detection in detections:
            pii_type_key = detection.pii_type.value
            self.detection_stats[pii_type_key] = self.detection_stats.get(pii_type_key, 0) + 1
        
        return detections
    
    def _calculate_confidence(self, pii_type: PIIType, detected_value: str, 
                            full_text: str, position: int) -> float:
        """Calculate confidence score for PII detection."""
        base_confidence = 0.7
        
        # Adjust based on PII type
        if pii_type in [PIIType.SSN, PIIType.CREDIT_CARD, PIIType.BANK_ACCOUNT]:
            base_confidence = 0.9
        elif pii_type in [PIIType.EMAIL, PIIType.PHONE]:
            base_confidence = 0.8
        elif pii_type == PIIType.NAME:
            base_confidence = 0.6  # Names are harder to detect accurately
        
        # Adjust based on context
        context_words = ["name", "email", "phone", "address", "ssn", "credit", "card", "account", "number"]
        context_before = full_text[max(0, position-50):position].lower()
        context_after = full_text[position:position+50].lower()
        
        for word in context_words:
            if word in context_before or word in context_after:
                base_confidence += 0.1
                break
        
        # Adjust based on format validation
        if pii_type == PIIType.EMAIL and "@" in detected_value and "." in detected_value:
            base_confidence += 0.1
        elif pii_type == PIIType.PHONE and len(detected_value.replace("-", "").replace("(", "").replace(")", "").replace(" ", "")) == 10:
            base_confidence += 0.1
        elif pii_type == PIIType.CREDIT_CARD and len(detected_value.replace("-", "").replace(" ", "")) == 16:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _mask_pii(self, pii_type: PIIType, value: str) -> str:
        """Mask PII value based on type."""
        if pii_type == PIIType.EMAIL:
            parts = value.split("@")
            if len(parts) == 2:
                username = parts[0]
                domain = parts[1]
                if len(username) > 2:
                    masked_username = username[:2] + "*" * (len(username) - 2)
                else:
                    masked_username = "*" * len(username)
                return f"{masked_username}@{domain}"
            return "*" * len(value)
        
        elif pii_type == PIIType.PHONE:
            # Keep last 4 digits
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return "*" * (len(digits) - 4) + digits[-4:]
            return "*" * len(value)
        
        elif pii_type == PIIType.CREDIT_CARD:
            # Keep last 4 digits
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return "*" * (len(digits) - 4) + digits[-4:]
            return "*" * len(value)
        
        elif pii_type == PIIType.SSN:
            # Keep last 4 digits
            digits = re.sub(r'\D', '', value)
            if len(digits) == 9:
                return "***-**-" + digits[-4:]
            return "*" * len(value)
        
        elif pii_type == PIIType.NAME:
            # Keep first letter of each word
            words = value.split()
            masked_words = []
            for word in words:
                if len(word) > 1:
                    masked_words.append(word[0] + "*" * (len(word) - 1))
                else:
                    masked_words.append("*")
            return " ".join(masked_words)
        
        else:
            # Default masking - replace with asterisks
            return "*" * len(value)
    
    def _remove_overlapping_detections(self, detections: List[PIIDetectionResult]) -> List[PIIDetectionResult]:
        """Remove overlapping detections, keeping the highest confidence ones."""
        if not detections:
            return detections
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered_detections = []
        for detection in detections:
            # Check if this detection overlaps with any already accepted detection
            overlaps = False
            for accepted in filtered_detections:
                if (detection.start_pos < accepted.end_pos and 
                    detection.end_pos > accepted.start_pos):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def process_message(self, message: str, source: str, destination: str, 
                       context: Optional[str] = None) -> Tuple[str, List[PIIDetectionResult], PrivacyAction]:
        """Process a message for PII detection and protection."""
        # Detect PII
        detections = self.detect_pii(message, context)
        
        if not detections:
            return message, [], PrivacyAction.ALLOW
        
        # Determine action based on policy
        action = self._determine_action(detections, source, destination)
        
        # Process message based on action
        processed_message = self._process_message_by_action(message, detections, action)
        
        # Log the event
        self._log_privacy_event(message, processed_message, detections, source, destination, action)
        
        # Create violation if action is not ALLOW
        if action != PrivacyAction.ALLOW:
            self._create_privacy_violation(detections, source, destination, action)
        
        return processed_message, detections, action
    
    def _determine_action(self, detections: List[PIIDetectionResult], 
                         source: str, destination: str) -> PrivacyAction:
        """Determine the action to take based on detected PII and policy."""
        if not self.active_policy:
            return PrivacyAction.AUDIT
        
        # Find the highest severity detection
        severity_order = {PIISeverity.LOW: 1, PIISeverity.MEDIUM: 2, PIISeverity.HIGH: 3, PIISeverity.CRITICAL: 4}
        max_severity = max(detections, key=lambda d: severity_order[d.severity]).severity
        
        # Find applicable rules
        applicable_rules = []
        for rule in self.active_policy.rules:
            if not rule.enabled:
                continue
            
            # Check if any detected PII types match this rule
            rule_pii_types = set(rule.pii_types)
            detected_pii_types = set(detection.pii_type for detection in detections)
            
            if rule_pii_types.intersection(detected_pii_types):
                applicable_rules.append(rule)
        
        # Sort rules by severity (highest first)
        severity_order = {PIISeverity.LOW: 1, PIISeverity.MEDIUM: 2, PIISeverity.HIGH: 3, PIISeverity.CRITICAL: 4}
        applicable_rules.sort(key=lambda x: severity_order[x.severity], reverse=True)
        
        # Return action from highest severity applicable rule
        if applicable_rules:
            return applicable_rules[0].action
        
        # Return default action
        return self.active_policy.default_action
    
    def _process_message_by_action(self, message: str, detections: List[PIIDetectionResult], 
                                 action: PrivacyAction) -> str:
        """Process message based on the determined action."""
        if action == PrivacyAction.ALLOW:
            return message
        elif action == PrivacyAction.BLOCK:
            return "[MESSAGE BLOCKED - Contains sensitive information]"
        elif action == PrivacyAction.MASK:
            return self._mask_message(message, detections)
        elif action == PrivacyAction.REDACT:
            return self._redact_message(message, detections)
        elif action == PrivacyAction.QUARANTINE:
            return "[MESSAGE QUARANTINED - Requires review]"
        else:  # AUDIT
            return message  # Allow but log
    
    def _mask_message(self, message: str, detections: List[PIIDetectionResult]) -> str:
        """Mask PII in message."""
        # Sort detections by position (reverse order to avoid position shifts)
        detections.sort(key=lambda x: x.start_pos, reverse=True)
        
        masked_message = message
        for detection in detections:
            masked_message = (masked_message[:detection.start_pos] + 
                            detection.masked_value + 
                            masked_message[detection.end_pos:])
        
        return masked_message
    
    def _redact_message(self, message: str, detections: List[PIIDetectionResult]) -> str:
        """Redact PII in message."""
        # Sort detections by position (reverse order to avoid position shifts)
        detections.sort(key=lambda x: x.start_pos, reverse=True)
        
        redacted_message = message
        for detection in detections:
            redacted_message = (redacted_message[:detection.start_pos] + 
                              "[REDACTED]" + 
                              redacted_message[detection.end_pos:])
        
        return redacted_message
    
    def _log_privacy_event(self, original_message: str, processed_message: str,
                          detections: List[PIIDetectionResult], source: str, 
                          destination: str, action: PrivacyAction) -> None:
        """Log privacy event to audit log."""
        log_id = f"log_{int(time.time())}_{hashlib.md5(original_message.encode()).hexdigest()[:8]}"
        
        # Determine severity
        if detections:
            severity_order = {PIISeverity.LOW: 1, PIISeverity.MEDIUM: 2, PIISeverity.HIGH: 3, PIISeverity.CRITICAL: 4}
            severity = max(detections, key=lambda d: severity_order[d.severity]).severity
        else:
            severity = PIISeverity.LOW
        
        audit_log = PrivacyAuditLog(
            log_id=log_id,
            event_type="pii_detection",
            timestamp=datetime.now(),
            source=source,
            destination=destination,
            pii_detected=[detection.pii_type for detection in detections],
            action_taken=action,
            severity=severity,
            details={
                "detection_count": len(detections),
                "message_length": len(original_message),
                "processed_length": len(processed_message),
                "detections": [detection.to_dict() for detection in detections]
            }
        )
        
        self.audit_logs.append(audit_log)
        
        # Record metrics
        record_metric("privacy_sentry.pii_detected", len(detections), MetricType.COUNTER, {
            "action": action.value,
            "severity": severity.value,
            "source": source,
            "destination": destination
        })
        
        # Save audit logs periodically
        if len(self.audit_logs) % 10 == 0:  # Save more frequently for testing
            self._save_privacy_data()
        
        # Also save after each log for testing
        self._save_privacy_data()
    
    def _create_privacy_violation(self, detections: List[PIIDetectionResult], 
                                source: str, destination: str, action: PrivacyAction) -> None:
        """Create a privacy violation record."""
        violation_id = f"violation_{int(time.time())}_{hashlib.md5(f'{source}{destination}'.encode()).hexdigest()[:8]}"
        
        # Determine severity
        if detections:
            severity_order = {PIISeverity.LOW: 1, PIISeverity.MEDIUM: 2, PIISeverity.HIGH: 3, PIISeverity.CRITICAL: 4}
            severity = max(detections, key=lambda d: severity_order[d.severity]).severity
        else:
            severity = PIISeverity.LOW
        
        violation = PrivacyViolation(
            violation_id=violation_id,
            pii_detections=detections,
            source=source,
            destination=destination,
            action_taken=action,
            severity=severity
        )
        
        self.privacy_violations.append(violation)
        
        # Update violation statistics
        severity_key = severity.value
        self.violation_stats[severity_key] = self.violation_stats.get(severity_key, 0) + 1
        
        logger.warning(f"Privacy violation created: {violation_id} - {action.value} action taken")
    
    def get_privacy_statistics(self) -> Dict[str, Any]:
        """Get privacy protection statistics."""
        total_detections = sum(self.detection_stats.values())
        total_violations = len(self.privacy_violations)
        
        # Count by severity
        severity_counts = {}
        for violation in self.privacy_violations:
            severity = violation.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by action
        action_counts = {}
        for violation in self.privacy_violations:
            action = violation.action_taken.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "total_detections": total_detections,
            "total_violations": total_violations,
            "detection_stats": self.detection_stats,
            "violation_stats": self.violation_stats,
            "severity_counts": severity_counts,
            "action_counts": action_counts,
            "active_policy": self.active_policy.name if self.active_policy else None,
            "pii_types_detected": len(self.detection_stats),
            "audit_logs_count": len(self.audit_logs)
        }
    
    def get_recent_violations(self, limit: int = 50) -> List[PrivacyViolation]:
        """Get recent privacy violations."""
        return sorted(self.privacy_violations, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_audit_logs(self, limit: int = 100) -> List[PrivacyAuditLog]:
        """Get recent audit logs."""
        return sorted(self.audit_logs, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def add_privacy_rule(self, rule: PrivacyRule) -> None:
        """Add a new privacy rule."""
        self.privacy_rules[rule.rule_id] = rule
        logger.info(f"Added privacy rule: {rule.name}")
    
    def update_privacy_rule(self, rule_id: str, **updates) -> bool:
        """Update an existing privacy rule."""
        if rule_id not in self.privacy_rules:
            return False
        
        rule = self.privacy_rules[rule_id]
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        logger.info(f"Updated privacy rule: {rule.name}")
        return True
    
    def set_active_policy(self, policy_id: str) -> bool:
        """Set the active privacy policy."""
        if policy_id not in self.privacy_policies:
            return False
        
        self.active_policy = self.privacy_policies[policy_id]
        logger.info(f"Set active privacy policy: {self.active_policy.name}")
        return True
    
    def create_custom_policy(self, policy: PrivacyPolicy) -> None:
        """Create a custom privacy policy."""
        self.privacy_policies[policy.policy_id] = policy
        logger.info(f"Created custom privacy policy: {policy.name}")
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Clean up old audit logs and violations."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up audit logs
        old_logs = [log for log in self.audit_logs if log.timestamp < cutoff_date]
        self.audit_logs = [log for log in self.audit_logs if log.timestamp >= cutoff_date]
        
        # Clean up violations
        old_violations = [violation for violation in self.privacy_violations if violation.timestamp < cutoff_date]
        self.privacy_violations = [violation for violation in self.privacy_violations if violation.timestamp >= cutoff_date]
        
        total_cleaned = len(old_logs) + len(old_violations)
        
        if total_cleaned > 0:
            self._save_privacy_data()
            logger.info(f"Cleaned up {total_cleaned} old privacy records")
        
        return total_cleaned
