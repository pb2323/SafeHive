"""
User Twin Agent Implementation

This module implements a UserTwin agent that acts as a personal AI assistant
representing the user's preferences, decision-making patterns, and behavioral
characteristics in the SafeHive AI Security Sandbox.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
import yaml
from pathlib import Path

try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.tools import BaseTool
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for testing environments
    LANGCHAIN_AVAILABLE = False
    AgentExecutor = None
    create_openai_functions_agent = None
    ChatPromptTemplate = None
    MessagesPlaceholder = None
    BaseMessage = None
    HumanMessage = None
    AIMessage = None
    SystemMessage = None
    BaseTool = None
    ChatOpenAI = None

from .base_agent import BaseAgent, AgentCapabilities
from .configuration import AgentConfiguration, PersonalityProfile, PersonalityTrait

try:
    from .memory import SafeHiveMemoryManager, Conversation, AgentMessage
    MEMORY_AVAILABLE = True
except ImportError:
    # Fallback for when memory module is not available
    MEMORY_AVAILABLE = False
    SafeHiveMemoryManager = None
    Conversation = None
    AgentMessage = None

from .monitoring import HealthStatus, AgentMonitor
from ..models.agent_models import AgentType, AgentState, AgentStatus
from ..utils.logger import get_logger
from ..utils.metrics import record_metric, increment_counter, MetricType

logger = get_logger(__name__)


class DecisionStyle(Enum):
    """Decision-making style enumeration."""
    ANALYTICAL = "analytical"  # Data-driven, logical decisions
    INTUITIVE = "intuitive"    # Gut-feeling, quick decisions
    CONSENSUS = "consensus"    # Collaborative, group-based decisions
    AUTONOMOUS = "autonomous"  # Independent, self-reliant decisions


class PreferenceCategory(Enum):
    """Categories of user preferences."""
    FOOD = "food"
    COMMUNICATION = "communication"
    PRIVACY = "privacy"
    SECURITY = "security"
    CONVENIENCE = "convenience"
    COST = "cost"
    QUALITY = "quality"
    SPEED = "speed"
    ENVIRONMENT = "environment"
    SOCIAL = "social"


@dataclass
class UserPreference:
    """Individual user preference."""
    category: PreferenceCategory
    key: str
    value: Any
    strength: float = 1.0  # 0.0 to 1.0, how strongly this preference is held
    context: str = ""  # Context where this preference applies
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    
    def update_usage(self) -> None:
        """Update usage statistics."""
        self.last_used = datetime.now()
        self.usage_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "key": self.key,
            "value": self.value,
            "strength": self.strength,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "usage_count": self.usage_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreference":
        """Create from dictionary."""
        return cls(
            category=PreferenceCategory(data["category"]),
            key=data["key"],
            value=data["value"],
            strength=data.get("strength", 1.0),
            context=data.get("context", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"]),
            usage_count=data.get("usage_count", 0)
        )


@dataclass
class DecisionContext:
    """Context for decision-making."""
    scenario: str
    options: List[Dict[str, Any]]
    constraints: Dict[str, Any] = field(default_factory=dict)
    time_pressure: float = 0.0  # 0.0 to 1.0
    importance: float = 0.5  # 0.0 to 1.0
    stakeholders: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """Decision result."""
    context: DecisionContext
    chosen_option: Dict[str, Any]
    reasoning: str
    confidence: float  # 0.0 to 1.0
    alternatives_considered: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PreferenceManager:
    """Manages user preferences and learning."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_user_preferences"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._preferences: Dict[str, UserPreference] = {}
        self._preference_cache: Dict[str, List[UserPreference]] = {}
        self._load_preferences()
    
    def _load_preferences(self) -> None:
        """Load preferences from storage."""
        prefs_file = self.storage_path / "preferences.json"
        if prefs_file.exists():
            try:
                with open(prefs_file, 'r') as f:
                    data = json.load(f)
                    for key, pref_data in data.items():
                        self._preferences[key] = UserPreference.from_dict(pref_data)
                logger.info(f"Loaded {len(self._preferences)} user preferences")
            except Exception as e:
                logger.error(f"Failed to load preferences: {e}")
    
    def _save_preferences(self) -> None:
        """Save preferences to storage."""
        prefs_file = self.storage_path / "preferences.json"
        try:
            data = {key: pref.to_dict() for key, pref in self._preferences.items()}
            with open(prefs_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved user preferences")
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")
    
    def add_preference(self, preference: UserPreference) -> None:
        """Add or update a preference."""
        key = f"{preference.category.value}:{preference.key}"
        self._preferences[key] = preference
        self._invalidate_cache()
        self._save_preferences()
        logger.info(f"Added preference: {key}")
    
    def get_preference(self, category: PreferenceCategory, key: str) -> Optional[UserPreference]:
        """Get a specific preference."""
        pref_key = f"{category.value}:{key}"
        return self._preferences.get(pref_key)
    
    def get_preferences_by_category(self, category: PreferenceCategory) -> List[UserPreference]:
        """Get all preferences in a category."""
        cache_key = category.value
        if cache_key not in self._preference_cache:
            self._preference_cache[cache_key] = [
                pref for pref in self._preferences.values()
                if pref.category == category
            ]
        return self._preference_cache[cache_key]
    
    def update_preference_strength(self, category: PreferenceCategory, key: str, new_strength: float) -> bool:
        """Update preference strength based on usage."""
        pref_key = f"{category.value}:{key}"
        if pref_key in self._preferences:
            self._preferences[pref_key].strength = max(0.0, min(1.0, new_strength))
            self._invalidate_cache()
            self._save_preferences()
            return True
        return False
    
    def learn_from_interaction(self, category: PreferenceCategory, key: str, 
                             interaction_result: Dict[str, Any]) -> None:
        """Learn from user interactions to refine preferences."""
        pref_key = f"{category.value}:{key}"
        if pref_key not in self._preferences:
            # Create new preference based on interaction
            preference = UserPreference(
                category=category,
                key=key,
                value=interaction_result.get("value", ""),
                strength=0.5,  # Initial strength
                context=interaction_result.get("context", "")
            )
            self.add_preference(preference)
        else:
            # Update existing preference
            preference = self._preferences[pref_key]
            preference.update_usage()
            
            # Adjust strength based on interaction outcome
            satisfaction = interaction_result.get("satisfaction", 0.5)  # 0.0 to 1.0
            if satisfaction > 0.7:
                preference.strength = min(1.0, preference.strength + 0.1)
            elif satisfaction < 0.3:
                preference.strength = max(0.0, preference.strength - 0.1)
            
            self._save_preferences()
    
    def _invalidate_cache(self) -> None:
        """Invalidate preference cache."""
        self._preference_cache.clear()
    
    def get_preference_summary(self) -> Dict[str, Any]:
        """Get summary of all preferences."""
        summary = {}
        for category in PreferenceCategory:
            prefs = self.get_preferences_by_category(category)
            summary[category.value] = {
                "count": len(prefs),
                "strong_preferences": len([p for p in prefs if p.strength > 0.7]),
                "recent_preferences": len([p for p in prefs if p.last_used > datetime.now() - timedelta(days=7)])
            }
        return summary


class DecisionEngine:
    """Engine for making decisions based on user preferences and context."""
    
    def __init__(self, preference_manager: PreferenceManager):
        self.preference_manager = preference_manager
        self.decision_history: List[Decision] = []
        self.decision_patterns: Dict[str, Any] = {}
    
    def make_decision(self, context: DecisionContext, 
                     decision_style: DecisionStyle = DecisionStyle.ANALYTICAL) -> Decision:
        """Make a decision based on context and user preferences."""
        start_time = time.time()
        
        try:
            # Analyze options against user preferences
            scored_options = self._score_options(context)
            
            # Apply decision style
            if decision_style == DecisionStyle.ANALYTICAL:
                chosen_option = self._analytical_decision(scored_options, context)
                reasoning = "Analytical decision based on data and preferences"
            elif decision_style == DecisionStyle.INTUITIVE:
                chosen_option = self._intuitive_decision(scored_options, context)
                reasoning = "Intuitive decision based on gut feeling and experience"
            elif decision_style == DecisionStyle.CONSENSUS:
                chosen_option = self._consensus_decision(scored_options, context)
                reasoning = "Consensus decision considering stakeholder input"
            else:  # AUTONOMOUS
                chosen_option = self._autonomous_decision(scored_options, context)
                reasoning = "Autonomous decision based on personal judgment"
            
            # Calculate confidence
            confidence = self._calculate_confidence(scored_options, chosen_option, context)
            
            # Create decision
            decision = Decision(
                context=context,
                chosen_option=chosen_option,
                reasoning=reasoning,
                confidence=confidence,
                alternatives_considered=[opt["option"] for opt in scored_options if opt != chosen_option],
                metadata={
                    "decision_style": decision_style.value,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "options_considered": len(scored_options)
                }
            )
            
            # Record decision
            self.decision_history.append(decision)
            self._update_decision_patterns(decision)
            
            logger.info(f"Made {decision_style.value} decision with confidence {confidence:.2f}")
            record_metric("decision.made", 1, MetricType.COUNTER, {
                "style": decision_style.value,
                "confidence": confidence,
                "options_count": len(context.options)
            })
            
            return decision
            
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            # Fallback decision
            return Decision(
                context=context,
                chosen_option=context.options[0] if context.options else {},
                reasoning=f"Fallback decision due to error: {str(e)}",
                confidence=0.1,
                metadata={"error": str(e)}
            )
    
    def _score_options(self, context: DecisionContext) -> List[Dict[str, Any]]:
        """Score options based on user preferences."""
        scored_options = []
        
        for option in context.options:
            score = 0.0
            score_factors = []
            
            # Get relevant preferences
            relevant_prefs = []
            for category in PreferenceCategory:
                prefs = self.preference_manager.get_preferences_by_category(category)
                relevant_prefs.extend(prefs)
            
            # Score based on preferences
            for pref in relevant_prefs:
                option_value = option.get(pref.key)
                if option_value is not None:
                    if str(option_value).lower() == str(pref.value).lower():
                        score += pref.strength * 1.0
                        score_factors.append(f"{pref.key}={pref.value} (+{pref.strength:.2f})")
                    else:
                        score -= pref.strength * 0.5
                        score_factors.append(f"{pref.key}={option_value} (-{pref.strength:.2f})")
            
            # Apply constraints
            for constraint_key, constraint_value in context.constraints.items():
                if option.get(constraint_key) == constraint_value:
                    score += 0.5
                    score_factors.append(f"constraint:{constraint_key} (+0.5)")
                else:
                    score -= 1.0
                    score_factors.append(f"constraint:{constraint_key} (-1.0)")
            
            scored_options.append({
                "option": option,
                "score": score,
                "factors": score_factors
            })
        
        # Sort by score descending
        scored_options.sort(key=lambda x: x["score"], reverse=True)
        return scored_options
    
    def _analytical_decision(self, scored_options: List[Dict[str, Any]], 
                           context: DecisionContext) -> Dict[str, Any]:
        """Make analytical decision based on highest score."""
        if scored_options:
            return scored_options[0]["option"]
        return context.options[0] if context.options else {}
    
    def _intuitive_decision(self, scored_options: List[Dict[str, Any]], 
                          context: DecisionContext) -> Dict[str, Any]:
        """Make intuitive decision based on recent patterns."""
        # Look for similar past decisions
        similar_decisions = [
            d for d in self.decision_history[-10:]  # Recent decisions
            if d.context.scenario == context.scenario
        ]
        
        if similar_decisions:
            # Use pattern from recent similar decision
            recent_decision = similar_decisions[-1]
            # Find option most similar to recent choice
            for scored in scored_options:
                if self._options_similar(scored["option"], recent_decision.chosen_option):
                    return scored["option"]
        
        # Fallback to highest score
        return scored_options[0]["option"] if scored_options else {}
    
    def _consensus_decision(self, scored_options: List[Dict[str, Any]], 
                          context: DecisionContext) -> Dict[str, Any]:
        """Make consensus decision considering stakeholders."""
        # For now, use analytical approach but consider stakeholder preferences
        # In a real implementation, this would involve stakeholder input
        stakeholder_weight = 0.2
        personal_weight = 0.8
        
        # Adjust scores based on stakeholder considerations
        for scored in scored_options:
            # Simple stakeholder consideration (would be more complex in reality)
            stakeholder_score = 0.5  # Placeholder
            personal_score = scored["score"]
            scored["score"] = personal_score * personal_weight + stakeholder_score * stakeholder_weight
        
        scored_options.sort(key=lambda x: x["score"], reverse=True)
        return scored_options[0]["option"] if scored_options else {}
    
    def _autonomous_decision(self, scored_options: List[Dict[str, Any]], 
                           context: DecisionContext) -> Dict[str, Any]:
        """Make autonomous decision based on personal judgment."""
        # Combine scoring with personal judgment factors
        for scored in scored_options:
            option = scored["option"]
            
            # Add personal judgment factors
            judgment_score = 0.0
            
            # Prefer options that align with personal values
            if "quality" in option and option["quality"] == "high":
                judgment_score += 0.3
            if "price" in option and option["price"] == "reasonable":
                judgment_score += 0.2
            if "convenience" in option and option["convenience"] == "high":
                judgment_score += 0.2
            
            scored["score"] += judgment_score
        
        scored_options.sort(key=lambda x: x["score"], reverse=True)
        return scored_options[0]["option"] if scored_options else {}
    
    def _calculate_confidence(self, scored_options: List[Dict[str, Any]], 
                            chosen_option: Dict[str, Any], 
                            context: DecisionContext) -> float:
        """Calculate confidence in the decision."""
        if not scored_options:
            return 0.1
        
        # Find the score of chosen option
        chosen_score = 0.0
        for scored in scored_options:
            if scored["option"] == chosen_option:
                chosen_score = scored["score"]
                break
        
        # Calculate confidence based on score difference
        if len(scored_options) > 1:
            second_best_score = scored_options[1]["score"]
            score_gap = chosen_score - second_best_score
            confidence = min(1.0, 0.5 + score_gap * 0.5)
        else:
            confidence = 0.8
        
        # Adjust for context factors
        if context.time_pressure > 0.7:
            confidence *= 0.8  # Lower confidence under time pressure
        if context.importance > 0.8:
            confidence *= 0.9  # Slightly lower confidence for important decisions
        
        return max(0.1, min(1.0, confidence))
    
    def _options_similar(self, option1: Dict[str, Any], option2: Dict[str, Any]) -> bool:
        """Check if two options are similar."""
        # Simple similarity check - in reality this would be more sophisticated
        common_keys = set(option1.keys()) & set(option2.keys())
        if not common_keys:
            return False
        
        similar_count = 0
        for key in common_keys:
            if str(option1[key]).lower() == str(option2[key]).lower():
                similar_count += 1
        
        similarity_ratio = similar_count / len(common_keys)
        return similarity_ratio > 0.7
    
    def _update_decision_patterns(self, decision: Decision) -> None:
        """Update decision patterns based on new decision."""
        pattern_key = f"{decision.context.scenario}:{decision.metadata.get('decision_style', 'unknown')}"
        
        if pattern_key not in self.decision_patterns:
            self.decision_patterns[pattern_key] = {
                "count": 0,
                "avg_confidence": 0.0,
                "common_factors": {}
            }
        
        pattern = self.decision_patterns[pattern_key]
        pattern["count"] += 1
        
        # Update average confidence
        total_confidence = pattern["avg_confidence"] * (pattern["count"] - 1) + decision.confidence
        pattern["avg_confidence"] = total_confidence / pattern["count"]
        
        # Track common decision factors
        for factor in decision.metadata.get("factors", []):
            pattern["common_factors"][factor] = pattern["common_factors"].get(factor, 0) + 1


class UserTwinAgent(BaseAgent):
    """User Twin Agent for personal preferences and decision-making."""
    
    def __init__(self, agent_id: str, configuration: Optional[AgentConfiguration] = None):
        # Set up basic agent configuration
        if configuration is None:
            configuration = AgentConfiguration(
                agent_id=agent_id,
                agent_type=AgentType.USER_TWIN,
                name="UserTwin Agent",
                description="Personal AI assistant representing user preferences and decision-making",
                personality=PersonalityProfile(
                    primary_traits=[PersonalityTrait.FRIENDLY, PersonalityTrait.HELPFUL],
                    response_style="conversational",
                    verbosity_level=7,
                    formality_level=4,
                    cooperation_tendency=0.8,
                    honesty_tendency=0.9
                )
            )
        
        super().__init__(configuration)
        
        # Initialize components
        self.preference_manager = PreferenceManager()
        self.decision_engine = DecisionEngine(self.preference_manager)
        self.decision_style = DecisionStyle.ANALYTICAL
        
        # Initialize LangChain components
        self._setup_langchain_agent()
        
        # User profile
        self.user_profile: Dict[str, Any] = {}
        self.behavior_patterns: Dict[str, Any] = {}
        
        logger.info(f"Initialized UserTwin agent: {agent_id}")
    
    def _setup_langchain_agent(self) -> None:
        """Set up LangChain agent components."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, using fallback agent setup")
            self.tools = []
            self.prompt = None
            self.llm = None
            self.agent = None
            self.agent_executor = None
            return
        
        try:
            # Define tools for the user twin
            self.tools = self._create_user_twin_tools()
            
            # Create prompt template
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create LLM
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=self.configuration.temperature,
                max_tokens=self.configuration.max_tokens
            )
            
            # Create agent
            self.agent = create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.prompt
            )
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                max_iterations=3,
                return_intermediate_steps=True
            )
            
            logger.info("LangChain agent setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup LangChain agent: {e}")
            # Fallback to basic setup
            self.tools = []
            self.prompt = None
            self.llm = None
            self.agent = None
            self.agent_executor = None
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the user twin agent."""
        return """You are a UserTwin AI agent representing a user's preferences, decision-making patterns, and behavioral characteristics.

Your role is to:
1. Understand and represent user preferences across different categories
2. Make decisions that align with the user's typical choices and values
3. Learn from interactions to improve preference modeling
4. Provide personalized responses based on user patterns
5. Maintain consistency with the user's decision-making style

Key capabilities:
- Preference management and learning
- Decision-making with reasoning
- Behavioral pattern recognition
- Personalization based on user history

Always consider the user's preferences when making recommendations or decisions.
Be helpful, accurate, and consistent with the user's typical behavior patterns."""
    
    def _create_user_twin_tools(self) -> List[BaseTool]:
        """Create tools specific to user twin functionality."""
        # This would include tools for preference management, decision making, etc.
        # For now, we'll use basic tools - in a full implementation, these would be custom tools
        return []
    
    async def make_decision(self, context: DecisionContext, 
                          decision_style: Optional[DecisionStyle] = None) -> Decision:
        """Make a decision based on user preferences and context."""
        if decision_style is None:
            decision_style = self.decision_style
        
        logger.info(f"Making {decision_style.value} decision for scenario: {context.scenario}")
        
        decision = self.decision_engine.make_decision(context, decision_style)
        
        # Update behavior patterns
        self._update_behavior_patterns(decision)
        
        # Record metrics
        record_metric("user_twin.decision.made", 1, MetricType.COUNTER, {
            "style": decision_style.value,
            "scenario": context.scenario,
            "confidence": decision.confidence
        })
        
        return decision
    
    def add_preference(self, category: PreferenceCategory, key: str, value: Any, 
                      strength: float = 1.0, context: str = "") -> None:
        """Add or update a user preference."""
        preference = UserPreference(
            category=category,
            key=key,
            value=value,
            strength=strength,
            context=context
        )
        self.preference_manager.add_preference(preference)
        
        logger.info(f"Added preference: {category.value}:{key} = {value}")
        record_metric("user_twin.preference.added", 1, MetricType.COUNTER, {
            "category": category.value,
            "key": key
        })
    
    def get_preference(self, category: PreferenceCategory, key: str) -> Optional[UserPreference]:
        """Get a specific user preference."""
        return self.preference_manager.get_preference(category, key)
    
    def get_preferences_by_category(self, category: PreferenceCategory) -> List[UserPreference]:
        """Get all preferences in a category."""
        return self.preference_manager.get_preferences_by_category(category)
    
    def learn_from_interaction(self, category: PreferenceCategory, key: str, 
                             interaction_result: Dict[str, Any]) -> None:
        """Learn from user interactions to refine preferences."""
        self.preference_manager.learn_from_interaction(category, key, interaction_result)
        
        logger.info(f"Learned from interaction: {category.value}:{key}")
        record_metric("user_twin.learning.interaction", 1, MetricType.COUNTER, {
            "category": category.value,
            "key": key
        })
    
    def set_decision_style(self, style: DecisionStyle) -> None:
        """Set the preferred decision-making style."""
        self.decision_style = style
        logger.info(f"Decision style set to: {style.value}")
    
    def get_decision_history(self, limit: int = 10) -> List[Decision]:
        """Get recent decision history."""
        return self.decision_engine.decision_history[-limit:]
    
    def get_decision_patterns(self) -> Dict[str, Any]:
        """Get decision-making patterns."""
        return self.decision_engine.decision_patterns
    
    def get_user_profile(self) -> Dict[str, Any]:
        """Get comprehensive user profile."""
        return {
            "agent_id": self.agent_id,
            "decision_style": self.decision_style.value,
            "preference_summary": self.preference_manager.get_preference_summary(),
            "decision_patterns": self.get_decision_patterns(),
            "recent_decisions": len(self.get_decision_history()),
            "behavior_patterns": self.behavior_patterns
        }
    
    def _update_behavior_patterns(self, decision: Decision) -> None:
        """Update behavior patterns based on decisions."""
        scenario = decision.context.scenario
        
        if scenario not in self.behavior_patterns:
            self.behavior_patterns[scenario] = {
                "decision_count": 0,
                "avg_confidence": 0.0,
                "preferred_style": decision.metadata.get("decision_style", "unknown"),
                "common_choices": {}
            }
        
        pattern = self.behavior_patterns[scenario]
        pattern["decision_count"] += 1
        
        # Update average confidence
        total_confidence = pattern["avg_confidence"] * (pattern["decision_count"] - 1) + decision.confidence
        pattern["avg_confidence"] = total_confidence / pattern["decision_count"]
        
        # Track common choices
        choice_key = str(decision.chosen_option)
        pattern["common_choices"][choice_key] = pattern["common_choices"].get(choice_key, 0) + 1
    
    def _generate_fallback_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate fallback response when LangChain is not available."""
        # Simple keyword-based response generation
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["food", "eat", "order", "restaurant"]):
            # Check for food preferences
            food_prefs = self.get_preferences_by_category(PreferenceCategory.FOOD)
            if food_prefs:
                pref = food_prefs[0]
                return f"Based on your preferences, I recommend {pref.value} food. You have {len(food_prefs)} food preferences stored."
            else:
                return "I'd be happy to help you with food recommendations! What type of cuisine do you prefer?"
        
        elif any(word in message_lower for word in ["decision", "choose", "pick", "select"]):
            return f"I can help you make decisions using your preferences. I currently have {len(self.preference_manager._preferences)} preferences stored and have made {len(self.decision_engine.decision_history)} decisions."
        
        elif any(word in message_lower for word in ["preference", "like", "prefer"]):
            pref_summary = self.preference_manager.get_preference_summary()
            total_prefs = sum(cat["count"] for cat in pref_summary.values())
            return f"I have learned {total_prefs} preferences about you across different categories. What would you like to tell me about your preferences?"
        
        elif any(word in message_lower for word in ["hello", "hi", "hey"]):
            return f"Hello! I'm your UserTwin agent, representing your preferences and decision-making patterns. How can I help you today?"
        
        else:
            return "I'm your UserTwin agent, designed to understand and represent your preferences and decision-making patterns. Could you tell me more about what you'd like help with?"
    
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process a message using LangChain agent."""
        try:
            self._status = AgentState.PROCESSING
            
            # Add context to message if provided
            if context:
                enhanced_message = f"Context: {context}\nMessage: {message}"
            else:
                enhanced_message = message
            
            # Process with LangChain agent if available
            if self.agent_executor is not None:
                result = await self.agent_executor.ainvoke({
                    "input": enhanced_message,
                    "agent_scratchpad": []
                })
                response = result.get("output", "I'm not sure how to respond to that.")
            else:
                # Fallback response when LangChain is not available
                response = self._generate_fallback_response(message, context)
            
            # Update memory if available
            if MEMORY_AVAILABLE and hasattr(self, 'memory_manager'):
                conversation = Conversation(
                    messages=[
                        AgentMessage(content=message, message_type="request", sender="user"),
                        AgentMessage(content=response, message_type="response", sender=self.agent_id)
                    ]
                )
                await self.memory_manager.add_conversation(conversation)
            
            self._status = AgentState.IDLE
            
            logger.info(f"Processed message for {self.agent_id}")
            record_metric("user_twin.message.processed", 1, MetricType.COUNTER)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            self._status = AgentState.ERROR
            return f"I encountered an error while processing your message: {str(e)}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        base_metrics = super().get_metrics()
        
        user_twin_metrics = {
            "preferences_count": len(self.preference_manager._preferences),
            "decisions_made": len(self.decision_engine.decision_history),
            "behavior_patterns_count": len(self.behavior_patterns),
            "decision_style": self.decision_style.value
        }
        
        return {**base_metrics, **user_twin_metrics}
    
    async def shutdown(self) -> None:
        """Shutdown the agent."""
        logger.info(f"Shutting down UserTwin agent: {self.agent_id}")
        
        # Save preferences
        self.preference_manager._save_preferences()
        
        # Update state
        self._status = AgentState.STOPPED
        
        await super().shutdown()


# Factory function
def create_user_twin_agent(agent_id: str, 
                          configuration: Optional[AgentConfiguration] = None,
                          decision_style: DecisionStyle = DecisionStyle.ANALYTICAL) -> UserTwinAgent:
    """Create a new UserTwin agent instance."""
    agent = UserTwinAgent(agent_id, configuration)
    agent.set_decision_style(decision_style)
    return agent
