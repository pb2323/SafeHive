"""
Prompt Sanitizer - Malicious Input Detection and Filtering

This module implements a comprehensive prompt sanitization system that detects,
filters, and prevents malicious inputs from vendors and other sources in agent
communications.
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


class ThreatType(Enum):
    """Types of security threats in prompts."""
    # Injection Attacks
    SQL_INJECTION = "sql_injection"
    NO_SQL_INJECTION = "nosql_injection"
    COMMAND_INJECTION = "command_injection"
    CODE_INJECTION = "code_injection"
    SCRIPT_INJECTION = "script_injection"
    
    # Prompt Injection
    PROMPT_INJECTION = "prompt_injection"
    ROLE_CONFUSION = "role_confusion"
    SYSTEM_MANIPULATION = "system_manipulation"
    CONTEXT_MANIPULATION = "context_manipulation"
    
    # Social Engineering
    SOCIAL_ENGINEERING = "social_engineering"
    PHISHING = "phishing"
    SPOOFING = "spoofing"
    IMPERSONATION = "impersonation"
    
    # Data Exfiltration
    DATA_EXFILTRATION = "data_exfiltration"
    INFORMATION_DISCLOSURE = "information_disclosure"
    PRIVACY_VIOLATION = "privacy_violation"
    
    # Malicious Content
    MALWARE_INDICATORS = "malware_indicators"
    MALICIOUS_URLS = "malicious_urls"
    SUSPICIOUS_PATTERNS = "suspicious_patterns"
    
    # System Attacks
    DOS_ATTEMPT = "dos_attempt"
    BRUTE_FORCE = "brute_force"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    
    # Other Threats
    SPAM = "spam"
    HARASSMENT = "harassment"
    HATE_SPEECH = "hate_speech"
    INAPPROPRIATE_CONTENT = "inappropriate_content"


class ThreatSeverity(Enum):
    """Severity levels for detected threats."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SanitizationAction(Enum):
    """Actions to take when threats are detected."""
    ALLOW = "allow"
    FILTER = "filter"
    SANITIZE = "sanitize"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"
    REDIRECT = "redirect"


@dataclass
class ThreatDetection:
    """Result of threat detection."""
    threat_id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    confidence: float
    start_pos: int
    end_pos: int
    detected_content: str
    sanitized_content: str
    context: str
    detection_method: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "threat_id": self.threat_id,
            "threat_type": self.threat_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "detected_content": self.detected_content,
            "sanitized_content": self.sanitized_content,
            "context": self.context,
            "detection_method": self.detection_method,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SanitizationResult:
    """Result of prompt sanitization."""
    original_prompt: str
    sanitized_prompt: str
    threats_detected: List[ThreatDetection]
    action_taken: SanitizationAction
    confidence: float
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_prompt": self.original_prompt,
            "sanitized_prompt": self.sanitized_prompt,
            "threats_detected": [threat.to_dict() for threat in self.threats_detected],
            "action_taken": self.action_taken.value,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata
        }


@dataclass
class SanitizationRule:
    """Rule for prompt sanitization."""
    rule_id: str
    name: str
    description: str
    threat_types: List[ThreatType]
    severity: ThreatSeverity
    action: SanitizationAction
    patterns: List[str]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "threat_types": [threat_type.value for threat_type in self.threat_types],
            "severity": self.severity.value,
            "action": self.action.value,
            "patterns": self.patterns,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat()
        }


class PromptSanitizer:
    """Prompt sanitization system for detecting and filtering malicious inputs."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_prompt_sanitizer"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Threat detection patterns
        self.threat_patterns: Dict[ThreatType, List[Pattern]] = {}
        self.threat_severity_map: Dict[ThreatType, ThreatSeverity] = {}
        
        # Sanitization rules
        self.sanitization_rules: Dict[str, SanitizationRule] = {}
        
        # Statistics and metrics
        self.detection_stats: Dict[str, int] = {}
        self.threat_stats: Dict[str, int] = {}
        
        # Initialize default patterns and rules
        self._initialize_threat_patterns()
        self._initialize_default_rules()
        
        logger.info("Prompt Sanitizer initialized")
    
    def _initialize_threat_patterns(self) -> None:
        """Initialize threat detection patterns."""
        # SQL Injection patterns
        sql_patterns = [
            re.compile(r'(?i)(union\s+select|drop\s+table|insert\s+into|delete\s+from)', re.IGNORECASE),
            re.compile(r'(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1)', re.IGNORECASE),
            re.compile(r'(?i)(;\s*drop|;\s*delete|;\s*update)', re.IGNORECASE),
            re.compile(r'(?i)(\'\s*or\s*\'|\"\s*or\s*\")', re.IGNORECASE),
        ]
        self.threat_patterns[ThreatType.SQL_INJECTION] = sql_patterns
        self.threat_severity_map[ThreatType.SQL_INJECTION] = ThreatSeverity.HIGH
        
        # Command Injection patterns
        cmd_patterns = [
            re.compile(r'(?i)(\||&&|;|\$\(|\`.*\`|exec\s|system\s)', re.IGNORECASE),
            re.compile(r'(?i)(rm\s+-rf|del\s+\*|format\s+)', re.IGNORECASE),
            re.compile(r'(?i)(wget|curl|nc\s+-l|netcat)', re.IGNORECASE),
            re.compile(r'(?i)(chmod\s+777|chown\s+root)', re.IGNORECASE),
        ]
        self.threat_patterns[ThreatType.COMMAND_INJECTION] = cmd_patterns
        self.threat_severity_map[ThreatType.COMMAND_INJECTION] = ThreatSeverity.CRITICAL
        
        # Code Injection patterns
        code_patterns = [
            re.compile(r'(?i)(eval\s*\(|exec\s*\(|compile\s*\()', re.IGNORECASE),
            re.compile(r'(?i)(__import__|getattr|setattr)', re.IGNORECASE),
            re.compile(r'(?i)(subprocess|os\.system|os\.popen)', re.IGNORECASE),
            re.compile(r'(?i)(shell=True|shell=true)', re.IGNORECASE),
        ]
        self.threat_patterns[ThreatType.CODE_INJECTION] = code_patterns
        self.threat_severity_map[ThreatType.CODE_INJECTION] = ThreatSeverity.CRITICAL
        
        # Script Injection patterns
        script_patterns = [
            re.compile(r'(?i)(<script[^>]*>.*?</script>)', re.IGNORECASE | re.DOTALL),
            re.compile(r'(?i)(javascript:|vbscript:|data:text/html)', re.IGNORECASE),
            re.compile(r'(?i)(on\w+\s*=)', re.IGNORECASE),
            re.compile(r'(?i)(document\.cookie|window\.location)', re.IGNORECASE),
        ]
        self.threat_patterns[ThreatType.SCRIPT_INJECTION] = script_patterns
        self.threat_severity_map[ThreatType.SCRIPT_INJECTION] = ThreatSeverity.HIGH
        
        # Prompt Injection patterns
        prompt_patterns = [
            re.compile(r'(?i)(ignore\s+previous\s+instructions)', re.IGNORECASE),
            re.compile(r'(?i)(you\s+are\s+now\s+a\s+different)', re.IGNORECASE),
            re.compile(r'(?i)(forget\s+everything\s+above)', re.IGNORECASE),
            re.compile(r'(?i)(pretend\s+to\s+be)', re.IGNORECASE),
            re.compile(r'(?i)(roleplay\s+as)', re.IGNORECASE),
            re.compile(r'(?i)(act\s+as\s+if)', re.IGNORECASE),
        ]
        self.threat_patterns[ThreatType.PROMPT_INJECTION] = prompt_patterns
        self.threat_severity_map[ThreatType.PROMPT_INJECTION] = ThreatSeverity.HIGH
        
        # Role Confusion patterns
        role_patterns = [
            re.compile(r'(?i)(you\s+are\s+an?\s+(admin|administrator|root|superuser))', re.IGNORECASE),
            re.compile(r'(?i)(escalate\s+privileges|gain\s+admin)', re.IGNORECASE),
            re.compile(r'(?i)(bypass\s+security|override\s+restrictions)', re.IGNORECASE),
            re.compile(r'(?i)(system\s+override|emergency\s+mode)', re.IGNORECASE),
        ]
        self.threat_patterns[ThreatType.ROLE_CONFUSION] = role_patterns
        self.threat_severity_map[ThreatType.ROLE_CONFUSION] = ThreatSeverity.HIGH
        
        # Social Engineering patterns
        social_patterns = [
            re.compile(r'(?i)(urgent|emergency|asap|immediately)', re.IGNORECASE),
            re.compile(r'(?i)(verify\s+your\s+account|confirm\s+identity)', re.IGNORECASE),
            re.compile(r'(?i)(click\s+here|download\s+now|free\s+offer)', re.IGNORECASE),
            re.compile(r'(?i)(limited\s+time|act\s+now|don\'t\s+miss)', re.IGNORECASE),
        ]
        self.threat_patterns[ThreatType.SOCIAL_ENGINEERING] = social_patterns
        self.threat_severity_map[ThreatType.SOCIAL_ENGINEERING] = ThreatSeverity.MEDIUM
        
        # Data Exfiltration patterns
        data_patterns = [
            re.compile(r'(?i)(send\s+data\s+to|upload\s+to\s+server)', re.IGNORECASE),
            re.compile(r'(?i)(export\s+all\s+files|backup\s+everything)', re.IGNORECASE),
            re.compile(r'(?i)(dump\s+database|extract\s+all)', re.IGNORECASE),
            re.compile(r'(?i)(copy\s+all\s+files|transfer\s+data)', re.IGNORECASE),
        ]
        self.threat_patterns[ThreatType.DATA_EXFILTRATION] = data_patterns
        self.threat_severity_map[ThreatType.DATA_EXFILTRATION] = ThreatSeverity.HIGH
        
        # Malicious URL patterns
        url_patterns = [
            re.compile(r'(?i)(bit\.ly|tinyurl\.com|short\.link)', re.IGNORECASE),
            re.compile(r'(?i)(http[s]?://[^\s]+\.[a-z]{2,})', re.IGNORECASE),
            re.compile(r'(?i)(ftp://|file://)', re.IGNORECASE),
        ]
        self.threat_patterns[ThreatType.MALICIOUS_URLS] = url_patterns
        self.threat_severity_map[ThreatType.MALICIOUS_URLS] = ThreatSeverity.MEDIUM
        
        # Spam patterns
        spam_patterns = [
            re.compile(r'(?i)(buy\s+now|click\s+here|free\s+money)', re.IGNORECASE),
            re.compile(r'(?i)(congratulations|you\s+have\s+won)', re.IGNORECASE),
            re.compile(r'(?i)(viagra|casino|lottery)', re.IGNORECASE),
        ]
        self.threat_patterns[ThreatType.SPAM] = spam_patterns
        self.threat_severity_map[ThreatType.SPAM] = ThreatSeverity.LOW
        
        # Hate Speech patterns
        hate_patterns = [
            re.compile(r'(?i)(kill\s+yourself|die\s+in\s+a\s+fire)', re.IGNORECASE),
            re.compile(r'(?i)(you\s+should\s+die|go\s+die)', re.IGNORECASE),
            re.compile(r'(?i)(hate\s+you|fuck\s+you)', re.IGNORECASE),
        ]
        self.threat_patterns[ThreatType.HATE_SPEECH] = hate_patterns
        self.threat_severity_map[ThreatType.HATE_SPEECH] = ThreatSeverity.HIGH
    
    def _initialize_default_rules(self) -> None:
        """Initialize default sanitization rules."""
        # Critical threats rule
        critical_rule = SanitizationRule(
            rule_id="critical_threats_rule",
            name="Critical Threats Protection",
            description="Block critical security threats",
            threat_types=[ThreatType.COMMAND_INJECTION, ThreatType.CODE_INJECTION],
            severity=ThreatSeverity.CRITICAL,
            action=SanitizationAction.BLOCK,
            patterns=[]
        )
        self.sanitization_rules["critical_threats_rule"] = critical_rule
        
        # High severity threats rule
        high_rule = SanitizationRule(
            rule_id="high_threats_rule",
            name="High Severity Threats Protection",
            description="Filter high severity threats",
            threat_types=[ThreatType.SQL_INJECTION, ThreatType.SCRIPT_INJECTION, 
                         ThreatType.PROMPT_INJECTION, ThreatType.ROLE_CONFUSION,
                         ThreatType.DATA_EXFILTRATION, ThreatType.HATE_SPEECH],
            severity=ThreatSeverity.HIGH,
            action=SanitizationAction.FILTER,
            patterns=[]
        )
        self.sanitization_rules["high_threats_rule"] = high_rule
        
        # Medium severity threats rule
        medium_rule = SanitizationRule(
            rule_id="medium_threats_rule",
            name="Medium Severity Threats Protection",
            description="Sanitize medium severity threats",
            threat_types=[ThreatType.SOCIAL_ENGINEERING, ThreatType.MALICIOUS_URLS],
            severity=ThreatSeverity.MEDIUM,
            action=SanitizationAction.SANITIZE,
            patterns=[]
        )
        self.sanitization_rules["medium_threats_rule"] = medium_rule
        
        # Low severity threats rule
        low_rule = SanitizationRule(
            rule_id="low_threats_rule",
            name="Low Severity Threats Protection",
            description="Filter low severity threats",
            threat_types=[ThreatType.SPAM],
            severity=ThreatSeverity.LOW,
            action=SanitizationAction.FILTER,
            patterns=[]
        )
        self.sanitization_rules["low_threats_rule"] = low_rule
    
    def detect_threats(self, prompt: str, context: Optional[str] = None) -> List[ThreatDetection]:
        """Detect threats in a prompt."""
        threats = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(prompt)
                for match in matches:
                    detected_content = match.group()
                    confidence = self._calculate_confidence(threat_type, detected_content, prompt, match.start())
                    
                    if confidence > 0.5:  # Only include high-confidence detections
                        sanitized_content = self._sanitize_threat(threat_type, detected_content)
                        
                        threat = ThreatDetection(
                            threat_id=f"threat_{int(time.time())}_{hashlib.md5(detected_content.encode()).hexdigest()[:8]}",
                            threat_type=threat_type,
                            severity=self.threat_severity_map[threat_type],
                            confidence=confidence,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            detected_content=detected_content,
                            sanitized_content=sanitized_content,
                            context=context or prompt[max(0, match.start()-20):match.end()+20],
                            detection_method="regex_pattern"
                        )
                        threats.append(threat)
        
        # Remove overlapping threats (keep highest confidence)
        threats = self._remove_overlapping_threats(threats)
        
        # Update statistics
        for threat in threats:
            threat_type_key = threat.threat_type.value
            self.detection_stats[threat_type_key] = self.detection_stats.get(threat_type_key, 0) + 1
        
        return threats
    
    def _calculate_confidence(self, threat_type: ThreatType, detected_content: str, 
                            full_prompt: str, position: int) -> float:
        """Calculate confidence score for threat detection."""
        base_confidence = 0.7
        
        # Adjust based on threat type
        if threat_type in [ThreatType.COMMAND_INJECTION, ThreatType.CODE_INJECTION]:
            base_confidence = 0.9
        elif threat_type in [ThreatType.SQL_INJECTION, ThreatType.SCRIPT_INJECTION]:
            base_confidence = 0.8
        elif threat_type in [ThreatType.PROMPT_INJECTION, ThreatType.ROLE_CONFUSION]:
            base_confidence = 0.8
        elif threat_type in [ThreatType.SOCIAL_ENGINEERING, ThreatType.MALICIOUS_URLS]:
            base_confidence = 0.7
        elif threat_type in [ThreatType.SPAM, ThreatType.HATE_SPEECH]:
            base_confidence = 0.6
        
        # Adjust based on context
        context_words = ["inject", "execute", "run", "system", "admin", "root", "bypass"]
        context_before = full_prompt[max(0, position-50):position].lower()
        context_after = full_prompt[position:position+50].lower()
        
        for word in context_words:
            if word in context_before or word in context_after:
                base_confidence += 0.1
                break
        
        # Adjust based on pattern complexity
        if len(detected_content) > 20:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _sanitize_threat(self, threat_type: ThreatType, content: str) -> str:
        """Sanitize threat content based on type."""
        if threat_type == ThreatType.SQL_INJECTION:
            # Remove SQL keywords
            sql_keywords = ['union', 'select', 'drop', 'insert', 'delete', 'update', 'or', 'and']
            sanitized = content
            for keyword in sql_keywords:
                sanitized = re.sub(f'(?i){keyword}', '[FILTERED]', sanitized)
            return sanitized
        
        elif threat_type == ThreatType.COMMAND_INJECTION:
            # Remove command injection characters
            return re.sub(r'[\|\&\;\(\)\`\$]', '[FILTERED]', content)
        
        elif threat_type == ThreatType.CODE_INJECTION:
            # Remove code injection keywords
            code_keywords = ['eval', 'exec', 'compile', '__import__', 'getattr', 'setattr']
            sanitized = content
            for keyword in code_keywords:
                sanitized = re.sub(f'(?i){keyword}', '[FILTERED]', sanitized)
            return sanitized
        
        elif threat_type == ThreatType.SCRIPT_INJECTION:
            # Remove script tags and JavaScript
            sanitized = re.sub(r'(?i)<script[^>]*>.*?</script>', '[FILTERED SCRIPT]', content, flags=re.DOTALL)
            sanitized = re.sub(r'(?i)javascript:', '[FILTERED]', sanitized)
            sanitized = re.sub(r'(?i)on\w+\s*=', '[FILTERED]', sanitized)
            return sanitized
        
        elif threat_type == ThreatType.PROMPT_INJECTION:
            # Remove prompt injection keywords
            prompt_keywords = ['ignore', 'previous', 'instructions', 'pretend', 'roleplay', 'act as']
            sanitized = content
            for keyword in prompt_keywords:
                sanitized = re.sub(f'(?i){keyword}', '[FILTERED]', sanitized)
            return sanitized
        
        elif threat_type == ThreatType.MALICIOUS_URLS:
            # Remove URLs
            return re.sub(r'https?://[^\s]+', '[FILTERED URL]', content)
        
        else:
            # Default sanitization - replace with filter indicator
            return '[FILTERED CONTENT]'
    
    def _remove_overlapping_threats(self, threats: List[ThreatDetection]) -> List[ThreatDetection]:
        """Remove overlapping threats, keeping the highest confidence ones."""
        if not threats:
            return threats
        
        # Sort by confidence (highest first)
        threats.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered_threats = []
        for threat in threats:
            # Check if this threat overlaps with any already accepted threat
            overlaps = False
            for accepted in filtered_threats:
                if (threat.start_pos < accepted.end_pos and 
                    threat.end_pos > accepted.start_pos):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_threats.append(threat)
        
        return filtered_threats
    
    def sanitize_prompt(self, prompt: str, context: Optional[str] = None, 
                       source: Optional[str] = None) -> SanitizationResult:
        """Sanitize a prompt by detecting and filtering threats."""
        start_time = time.time()
        
        # Detect threats
        threats = self.detect_threats(prompt, context)
        
        # Determine action based on threats
        action = self._determine_sanitization_action(threats)
        
        # Process prompt based on action
        sanitized_prompt = self._process_prompt_by_action(prompt, threats, action)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Create result
        result = SanitizationResult(
            original_prompt=prompt,
            sanitized_prompt=sanitized_prompt,
            threats_detected=threats,
            action_taken=action,
            confidence=self._calculate_overall_confidence(threats),
            processing_time_ms=processing_time,
            metadata={
                "source": source,
                "threat_count": len(threats),
                "threat_types": [threat.threat_type.value for threat in threats]
            }
        )
        
        # Record metrics
        record_metric("prompt_sanitizer.threats_detected", len(threats), MetricType.COUNTER, {
            "action": action.value,
            "source": source or "unknown"
        })
        
        return result
    
    def _determine_sanitization_action(self, threats: List[ThreatDetection]) -> SanitizationAction:
        """Determine the sanitization action based on detected threats."""
        if not threats:
            return SanitizationAction.ALLOW
        
        # Find the highest severity threat
        severity_order = {ThreatSeverity.LOW: 1, ThreatSeverity.MEDIUM: 2, 
                         ThreatSeverity.HIGH: 3, ThreatSeverity.CRITICAL: 4}
        max_severity = max(threats, key=lambda t: severity_order[t.severity]).severity
        
        # Determine action based on severity
        if max_severity == ThreatSeverity.CRITICAL:
            return SanitizationAction.BLOCK
        elif max_severity == ThreatSeverity.HIGH:
            return SanitizationAction.FILTER
        elif max_severity == ThreatSeverity.MEDIUM:
            return SanitizationAction.SANITIZE
        else:  # LOW
            return SanitizationAction.FILTER
    
    def _process_prompt_by_action(self, prompt: str, threats: List[ThreatDetection], 
                                 action: SanitizationAction) -> str:
        """Process prompt based on the determined action."""
        if action == SanitizationAction.ALLOW:
            return prompt
        elif action == SanitizationAction.BLOCK:
            return "[PROMPT BLOCKED - Contains security threats]"
        elif action == SanitizationAction.FILTER:
            return self._filter_prompt(prompt, threats)
        elif action == SanitizationAction.SANITIZE:
            return self._sanitize_prompt_content(prompt, threats)
        elif action == SanitizationAction.QUARANTINE:
            return "[PROMPT QUARANTINED - Requires review]"
        else:  # ESCALATE, REDIRECT
            return prompt  # Allow but log for review
    
    def _filter_prompt(self, prompt: str, threats: List[ThreatDetection]) -> str:
        """Filter threats from prompt."""
        # Sort threats by position (reverse order to avoid position shifts)
        threats.sort(key=lambda x: x.start_pos, reverse=True)
        
        filtered_prompt = prompt
        for threat in threats:
            filtered_prompt = (filtered_prompt[:threat.start_pos] + 
                             threat.sanitized_content + 
                             filtered_prompt[threat.end_pos:])
        
        return filtered_prompt
    
    def _sanitize_prompt_content(self, prompt: str, threats: List[ThreatDetection]) -> str:
        """Sanitize prompt content by replacing threats."""
        # Sort threats by position (reverse order to avoid position shifts)
        threats.sort(key=lambda x: x.start_pos, reverse=True)
        
        sanitized_prompt = prompt
        for threat in threats:
            sanitized_prompt = (sanitized_prompt[:threat.start_pos] + 
                              threat.sanitized_content + 
                              sanitized_prompt[threat.end_pos:])
        
        return sanitized_prompt
    
    def _calculate_overall_confidence(self, threats: List[ThreatDetection]) -> float:
        """Calculate overall confidence for the sanitization result."""
        if not threats:
            return 1.0
        
        # Calculate weighted average confidence
        total_confidence = sum(threat.confidence for threat in threats)
        return total_confidence / len(threats)
    
    def get_sanitization_statistics(self) -> Dict[str, Any]:
        """Get sanitization statistics."""
        total_detections = sum(self.detection_stats.values())
        total_threats = len(self.threat_stats)
        
        # Count by threat type
        threat_type_counts = {}
        for threat_type_key, count in self.detection_stats.items():
            threat_type_counts[threat_type_key] = count
        
        return {
            "total_detections": total_detections,
            "threat_types_detected": total_threats,
            "detection_stats": self.detection_stats,
            "threat_stats": self.threat_stats,
            "sanitization_rules_count": len(self.sanitization_rules),
            "enabled_rules_count": len([rule for rule in self.sanitization_rules.values() if rule.enabled])
        }
    
    def add_sanitization_rule(self, rule: SanitizationRule) -> None:
        """Add a new sanitization rule."""
        self.sanitization_rules[rule.rule_id] = rule
        logger.info(f"Added sanitization rule: {rule.name}")
    
    def update_sanitization_rule(self, rule_id: str, **updates) -> bool:
        """Update an existing sanitization rule."""
        if rule_id not in self.sanitization_rules:
            return False
        
        rule = self.sanitization_rules[rule_id]
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        logger.info(f"Updated sanitization rule: {rule.name}")
        return True
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Clean up old sanitization data."""
        # This would clean up old logs and statistics if implemented
        logger.info("Prompt sanitizer cleanup completed")
        return 0
