"""
Unit tests for UserTwin agent implementation.
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import pytest

from safehive.agents.user_twin import (
    DecisionStyle, PreferenceCategory, UserPreference, DecisionContext, Decision,
    PreferenceManager, DecisionEngine, UserTwinAgent, create_user_twin_agent
)
from safehive.agents.configuration import AgentConfiguration, PersonalityProfile, PersonalityTrait
from safehive.models.agent_models import AgentType, AgentState


class TestDecisionStyle:
    """Test decision style enumeration."""
    
    def test_decision_style_values(self):
        """Test decision style enum values."""
        assert DecisionStyle.ANALYTICAL.value == "analytical"
        assert DecisionStyle.INTUITIVE.value == "intuitive"
        assert DecisionStyle.CONSENSUS.value == "consensus"
        assert DecisionStyle.AUTONOMOUS.value == "autonomous"


class TestPreferenceCategory:
    """Test preference category enumeration."""
    
    def test_preference_category_values(self):
        """Test preference category enum values."""
        assert PreferenceCategory.FOOD.value == "food"
        assert PreferenceCategory.COMMUNICATION.value == "communication"
        assert PreferenceCategory.PRIVACY.value == "privacy"
        assert PreferenceCategory.SECURITY.value == "security"
        assert PreferenceCategory.CONVENIENCE.value == "convenience"
        assert PreferenceCategory.COST.value == "cost"
        assert PreferenceCategory.QUALITY.value == "quality"
        assert PreferenceCategory.SPEED.value == "speed"
        assert PreferenceCategory.ENVIRONMENT.value == "environment"
        assert PreferenceCategory.SOCIAL.value == "social"


class TestUserPreference:
    """Test user preference data class."""
    
    def test_user_preference_creation(self):
        """Test user preference creation."""
        preference = UserPreference(
            category=PreferenceCategory.FOOD,
            key="cuisine_type",
            value="italian",
            strength=0.8,
            context="dinner preferences"
        )
        
        assert preference.category == PreferenceCategory.FOOD
        assert preference.key == "cuisine_type"
        assert preference.value == "italian"
        assert preference.strength == 0.8
        assert preference.context == "dinner preferences"
        assert preference.usage_count == 0
        assert isinstance(preference.created_at, datetime)
        assert isinstance(preference.last_used, datetime)
    
    def test_update_usage(self):
        """Test usage update functionality."""
        preference = UserPreference(
            category=PreferenceCategory.FOOD,
            key="cuisine_type",
            value="italian"
        )
        
        initial_count = preference.usage_count
        initial_time = preference.last_used
        
        time.sleep(0.01)  # Small delay to ensure time difference
        preference.update_usage()
        
        assert preference.usage_count == initial_count + 1
        assert preference.last_used > initial_time
    
    def test_preference_serialization(self):
        """Test preference serialization and deserialization."""
        preference = UserPreference(
            category=PreferenceCategory.FOOD,
            key="cuisine_type",
            value="italian",
            strength=0.8,
            context="dinner preferences"
        )
        
        # Test to_dict
        data = preference.to_dict()
        assert data["category"] == "food"
        assert data["key"] == "cuisine_type"
        assert data["value"] == "italian"
        assert data["strength"] == 0.8
        assert data["context"] == "dinner preferences"
        assert "created_at" in data
        assert "last_used" in data
        assert data["usage_count"] == 0
        
        # Test from_dict
        restored_preference = UserPreference.from_dict(data)
        assert restored_preference.category == PreferenceCategory.FOOD
        assert restored_preference.key == "cuisine_type"
        assert restored_preference.value == "italian"
        assert restored_preference.strength == 0.8
        assert restored_preference.context == "dinner preferences"


class TestDecisionContext:
    """Test decision context data class."""
    
    def test_decision_context_creation(self):
        """Test decision context creation."""
        context = DecisionContext(
            scenario="food_ordering",
            options=[
                {"name": "Pizza", "price": 15, "delivery_time": 30},
                {"name": "Burger", "price": 12, "delivery_time": 20}
            ],
            constraints={"max_price": 20},
            time_pressure=0.3,
            importance=0.7
        )
        
        assert context.scenario == "food_ordering"
        assert len(context.options) == 2
        assert context.constraints["max_price"] == 20
        assert context.time_pressure == 0.3
        assert context.importance == 0.7
        assert context.stakeholders == []
        assert context.metadata == {}


class TestDecision:
    """Test decision data class."""
    
    def test_decision_creation(self):
        """Test decision creation."""
        context = DecisionContext(
            scenario="food_ordering",
            options=[{"name": "Pizza"}, {"name": "Burger"}]
        )
        
        decision = Decision(
            context=context,
            chosen_option={"name": "Pizza"},
            reasoning="User prefers Italian food",
            confidence=0.8
        )
        
        assert decision.context == context
        assert decision.chosen_option == {"name": "Pizza"}
        assert decision.reasoning == "User prefers Italian food"
        assert decision.confidence == 0.8
        assert decision.alternatives_considered == []
        assert isinstance(decision.timestamp, datetime)
        assert decision.metadata == {}


class TestPreferenceManager:
    """Test preference manager."""
    
    def test_preference_manager_creation(self):
        """Test preference manager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            assert manager.storage_path == Path(temp_dir)
            assert manager._preferences == {}
            assert manager._preference_cache == {}
    
    def test_add_and_get_preference(self):
        """Test adding and getting preferences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            preference = UserPreference(
                category=PreferenceCategory.FOOD,
                key="cuisine_type",
                value="italian",
                strength=0.8
            )
            
            manager.add_preference(preference)
            
            # Get preference
            retrieved = manager.get_preference(PreferenceCategory.FOOD, "cuisine_type")
            assert retrieved is not None
            assert retrieved.value == "italian"
            assert retrieved.strength == 0.8
    
    def test_get_preferences_by_category(self):
        """Test getting preferences by category."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            # Add preferences in different categories
            food_pref = UserPreference(PreferenceCategory.FOOD, "cuisine", "italian")
            comm_pref = UserPreference(PreferenceCategory.COMMUNICATION, "style", "formal")
            
            manager.add_preference(food_pref)
            manager.add_preference(comm_pref)
            
            # Get food preferences
            food_prefs = manager.get_preferences_by_category(PreferenceCategory.FOOD)
            assert len(food_prefs) == 1
            assert food_prefs[0].key == "cuisine"
            
            # Get communication preferences
            comm_prefs = manager.get_preferences_by_category(PreferenceCategory.COMMUNICATION)
            assert len(comm_prefs) == 1
            assert comm_prefs[0].key == "style"
    
    def test_update_preference_strength(self):
        """Test updating preference strength."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            preference = UserPreference(
                category=PreferenceCategory.FOOD,
                key="cuisine_type",
                value="italian",
                strength=0.5
            )
            
            manager.add_preference(preference)
            
            # Update strength
            success = manager.update_preference_strength(PreferenceCategory.FOOD, "cuisine_type", 0.9)
            assert success
            
            # Check updated strength
            updated = manager.get_preference(PreferenceCategory.FOOD, "cuisine_type")
            assert updated.strength == 0.9
    
    def test_learn_from_interaction(self):
        """Test learning from interactions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            # Learn from positive interaction
            manager.learn_from_interaction(
                PreferenceCategory.FOOD,
                "cuisine_type",
                {"value": "italian", "satisfaction": 0.9, "context": "dinner"}
            )
            
        preference = manager.get_preference(PreferenceCategory.FOOD, "cuisine_type")
        assert preference is not None
        assert preference.value == "italian"
        assert preference.strength >= 0.5  # Should be at least initial 0.5 (could increase)
        
        # Learn from negative interaction
        initial_strength = preference.strength
        manager.learn_from_interaction(
            PreferenceCategory.FOOD,
            "cuisine_type",
            {"value": "italian", "satisfaction": 0.2, "context": "lunch"}
        )
        
        updated_preference = manager.get_preference(PreferenceCategory.FOOD, "cuisine_type")
        assert updated_preference.strength < initial_strength  # Should decrease from initial strength
    
    def test_preference_summary(self):
        """Test getting preference summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            # Add some preferences
            manager.add_preference(UserPreference(PreferenceCategory.FOOD, "cuisine", "italian", 0.8))
            manager.add_preference(UserPreference(PreferenceCategory.FOOD, "spice_level", "medium", 0.6))
            manager.add_preference(UserPreference(PreferenceCategory.COMMUNICATION, "style", "formal", 0.9))
            
            summary = manager.get_preference_summary()
            
            assert "food" in summary
            assert "communication" in summary
            assert summary["food"]["count"] == 2
            assert summary["communication"]["count"] == 1
            assert summary["food"]["strong_preferences"] == 1  # Only cuisine with 0.8


class TestDecisionEngine:
    """Test decision engine."""
    
    def test_decision_engine_creation(self):
        """Test decision engine creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preference_manager = PreferenceManager(temp_dir)
            engine = DecisionEngine(preference_manager)
            
            assert engine.preference_manager == preference_manager
            assert engine.decision_history == []
            assert engine.decision_patterns == {}
    
    def test_make_analytical_decision(self):
        """Test making analytical decision."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preference_manager = PreferenceManager(temp_dir)
            engine = DecisionEngine(preference_manager)
            
            # Add preference
            preference = UserPreference(
                category=PreferenceCategory.FOOD,
                key="cuisine",
                value="italian",
                strength=0.8
            )
            preference_manager.add_preference(preference)
            
            # Create decision context
            context = DecisionContext(
                scenario="food_ordering",
                options=[
                    {"name": "Pizza", "cuisine": "italian", "price": 15},
                    {"name": "Burger", "cuisine": "american", "price": 12}
                ]
            )
            
            # Make decision
            decision = engine.make_decision(context, DecisionStyle.ANALYTICAL)
            
            assert decision.context == context
            assert decision.reasoning == "Analytical decision based on data and preferences"
            assert decision.confidence > 0.0
            assert len(engine.decision_history) == 1
            assert decision.metadata["decision_style"] == "analytical"
    
    def test_make_intuitive_decision(self):
        """Test making intuitive decision."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preference_manager = PreferenceManager(temp_dir)
            engine = DecisionEngine(preference_manager)
            
            # Add some decision history for pattern recognition
            context = DecisionContext(
                scenario="food_ordering",
                options=[{"name": "Pizza"}, {"name": "Burger"}]
            )
            
            # Make first decision
            first_decision = Decision(
                context=context,
                chosen_option={"name": "Pizza"},
                reasoning="Test",
                confidence=0.8
            )
            engine.decision_history.append(first_decision)
            
            # Make intuitive decision
            decision = engine.make_decision(context, DecisionStyle.INTUITIVE)
            
            assert decision.reasoning == "Intuitive decision based on gut feeling and experience"
            assert decision.metadata["decision_style"] == "intuitive"
    
    def test_make_consensus_decision(self):
        """Test making consensus decision."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preference_manager = PreferenceManager(temp_dir)
            engine = DecisionEngine(preference_manager)
            
            context = DecisionContext(
                scenario="food_ordering",
                options=[
                    {"name": "Pizza", "price": 15},
                    {"name": "Burger", "price": 12}
                ],
                stakeholders=["family", "friends"]
            )
            
            decision = engine.make_decision(context, DecisionStyle.CONSENSUS)
            
            assert decision.reasoning == "Consensus decision considering stakeholder input"
            assert decision.metadata["decision_style"] == "consensus"
    
    def test_make_autonomous_decision(self):
        """Test making autonomous decision."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preference_manager = PreferenceManager(temp_dir)
            engine = DecisionEngine(preference_manager)
            
            context = DecisionContext(
                scenario="food_ordering",
                options=[
                    {"name": "Pizza", "quality": "high", "price": "reasonable"},
                    {"name": "Burger", "quality": "medium", "price": "cheap"}
                ]
            )
            
            decision = engine.make_decision(context, DecisionStyle.AUTONOMOUS)
            
            assert decision.reasoning == "Autonomous decision based on personal judgment"
            assert decision.metadata["decision_style"] == "autonomous"
    
    def test_decision_patterns_update(self):
        """Test decision patterns updating."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preference_manager = PreferenceManager(temp_dir)
            engine = DecisionEngine(preference_manager)
            
            context = DecisionContext(
                scenario="food_ordering",
                options=[{"name": "Pizza"}]
            )
            
            # Make multiple decisions
            for _ in range(3):
                engine.make_decision(context, DecisionStyle.ANALYTICAL)
            
            patterns = engine.decision_patterns
            assert len(patterns) == 1
            
            pattern_key = "food_ordering:analytical"
            assert pattern_key in patterns
            assert patterns[pattern_key]["count"] == 3
            assert patterns[pattern_key]["avg_confidence"] > 0.0


class TestUserTwinAgent:
    """Test UserTwin agent."""
    
    def test_user_twin_agent_creation(self):
        """Test UserTwin agent creation."""
        agent = UserTwinAgent("test_user_twin")
        
        assert agent.agent_id == "test_user_twin"
        assert agent.configuration.agent_type == AgentType.USER_TWIN
        assert agent.decision_style == DecisionStyle.ANALYTICAL
        assert isinstance(agent.preference_manager, PreferenceManager)
        assert isinstance(agent.decision_engine, DecisionEngine)
        assert agent.user_profile == {}
        assert agent.behavior_patterns == {}
    
    def test_user_twin_agent_with_configuration(self):
        """Test UserTwin agent with custom configuration."""
        config = AgentConfiguration(
            agent_id="custom_user_twin",
            agent_type=AgentType.USER_TWIN,
            name="Custom UserTwin",
            description="Custom user twin agent",
            personality=PersonalityProfile(
                primary_traits=[PersonalityTrait.CASUAL, PersonalityTrait.HELPFUL],
                response_style="conversational"
            )
        )
        
        agent = UserTwinAgent("custom_user_twin", config)
        
        assert agent.configuration.name == "Custom UserTwin"
        assert agent.configuration.description == "Custom user twin agent"
        assert PersonalityTrait.CASUAL in agent.configuration.personality.primary_traits
    
    @pytest.mark.asyncio
    async def test_make_decision(self):
        """Test making decisions through the agent."""
        agent = UserTwinAgent("test_user_twin")
        
        # Add a preference
        agent.add_preference(
            PreferenceCategory.FOOD,
            "cuisine",
            "italian",
            strength=0.8
        )
        
        # Create decision context
        context = DecisionContext(
            scenario="food_ordering",
            options=[
                {"name": "Pizza", "cuisine": "italian"},
                {"name": "Burger", "cuisine": "american"}
            ]
        )
        
        # Make decision
        decision = await agent.make_decision(context, DecisionStyle.ANALYTICAL)
        
        assert decision.context == context
        assert decision.confidence > 0.0
        assert decision.metadata["decision_style"] == "analytical"
    
    def test_preference_management(self):
        """Test preference management through the agent."""
        agent = UserTwinAgent("test_user_twin")
        
        # Add preference
        agent.add_preference(
            PreferenceCategory.FOOD,
            "cuisine",
            "italian",
            strength=0.8,
            context="dinner"
        )
        
        # Get preference
        preference = agent.get_preference(PreferenceCategory.FOOD, "cuisine")
        assert preference is not None
        assert preference.value == "italian"
        assert preference.strength == 0.8
        
        # Get preferences by category
        food_prefs = agent.get_preferences_by_category(PreferenceCategory.FOOD)
        assert len(food_prefs) == 1
        assert food_prefs[0].key == "cuisine"
    
    def test_learning_from_interaction(self):
        """Test learning from interactions."""
        agent = UserTwinAgent("test_user_twin")
        
        # Learn from interaction
        agent.learn_from_interaction(
            PreferenceCategory.FOOD,
            "cuisine",
            {"value": "italian", "satisfaction": 0.9, "context": "dinner"}
        )
        
        preference = agent.get_preference(PreferenceCategory.FOOD, "cuisine")
        assert preference is not None
        assert preference.value == "italian"
        assert preference.strength > 0.5  # Should be stronger than initial 0.5
    
    def test_decision_style_management(self):
        """Test decision style management."""
        agent = UserTwinAgent("test_user_twin")
        
        # Test default style
        assert agent.decision_style == DecisionStyle.ANALYTICAL
        
        # Change style
        agent.set_decision_style(DecisionStyle.INTUITIVE)
        assert agent.decision_style == DecisionStyle.INTUITIVE
        
        agent.set_decision_style(DecisionStyle.CONSENSUS)
        assert agent.decision_style == DecisionStyle.CONSENSUS
        
        agent.set_decision_style(DecisionStyle.AUTONOMOUS)
        assert agent.decision_style == DecisionStyle.AUTONOMOUS
    
    def test_decision_history(self):
        """Test decision history management."""
        agent = UserTwinAgent("test_user_twin")
        
        # Initially no history
        history = agent.get_decision_history()
        assert len(history) == 0
        
        # Add some decisions (this would normally be done through make_decision)
        context = DecisionContext("test", [{"option": "A"}])
        decision = Decision(context, {"option": "A"}, "test", 0.8)
        agent.decision_engine.decision_history.append(decision)
        
        history = agent.get_decision_history()
        assert len(history) == 1
        
        history = agent.get_decision_history(limit=5)
        assert len(history) == 1
    
    def test_decision_patterns(self):
        """Test decision patterns."""
        agent = UserTwinAgent("test_user_twin")
        
        patterns = agent.get_decision_patterns()
        assert isinstance(patterns, dict)
        assert len(patterns) == 0
    
    def test_user_profile(self):
        """Test user profile generation."""
        agent = UserTwinAgent("test_user_twin")
        
        # Add some preferences and decisions
        agent.add_preference(PreferenceCategory.FOOD, "cuisine", "italian")
        agent.set_decision_style(DecisionStyle.INTUITIVE)
        
        profile = agent.get_user_profile()
        
        assert profile["agent_id"] == "test_user_twin"
        assert profile["decision_style"] == "intuitive"
        assert "preference_summary" in profile
        assert "decision_patterns" in profile
        assert "behavior_patterns" in profile
        assert profile["recent_decisions"] == 0
    
    def test_behavior_patterns_update(self):
        """Test behavior patterns updating."""
        agent = UserTwinAgent("test_user_twin")
        
        # Initially no patterns
        assert len(agent.behavior_patterns) == 0
        
        # Simulate a decision (normally done through make_decision)
        context = DecisionContext("food_ordering", [{"name": "Pizza"}])
        decision = Decision(context, {"name": "Pizza"}, "test", 0.8)
        decision.metadata["decision_style"] = "analytical"
        
        agent._update_behavior_patterns(decision)
        
        assert "food_ordering" in agent.behavior_patterns
        pattern = agent.behavior_patterns["food_ordering"]
        assert pattern["decision_count"] == 1
        assert pattern["avg_confidence"] == 0.8
        assert pattern["preferred_style"] == "analytical"
    
    def test_get_metrics(self):
        """Test getting agent metrics."""
        agent = UserTwinAgent("test_user_twin")
        
        # Add some data
        agent.add_preference(PreferenceCategory.FOOD, "cuisine", "italian")
        
        metrics = agent.get_metrics()
        
        assert "preferences_count" in metrics
        assert "decisions_made" in metrics
        assert "behavior_patterns_count" in metrics
        assert "decision_style" in metrics
        assert metrics["preferences_count"] == 1
        assert metrics["decisions_made"] == 0
        assert metrics["behavior_patterns_count"] == 0
        assert metrics["decision_style"] == "analytical"
    
    @pytest.mark.asyncio
    async def test_process_message(self):
        """Test message processing."""
        agent = UserTwinAgent("test_user_twin")
        
        # Mock the LangChain agent executor
        with patch.object(agent, 'agent_executor') as mock_executor:
            mock_executor.ainvoke = AsyncMock(return_value={"output": "Test response"})
            
            response = await agent.process_message("Hello, what should I eat?")
            
            assert response == "Test response"
            mock_executor.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_message_with_context(self):
        """Test message processing with context."""
        agent = UserTwinAgent("test_user_twin")
        
        context = {"scenario": "food_ordering", "budget": 20}
        
        with patch.object(agent, 'agent_executor') as mock_executor:
            mock_executor.ainvoke = AsyncMock(return_value={"output": "Test response"})
            
            response = await agent.process_message("What should I order?", context)
            
            assert response == "Test response"
            
            # Check that context was included in the message
            call_args = mock_executor.ainvoke.call_args[0][0]
            assert "Context:" in call_args["input"]
            assert "Message:" in call_args["input"]
    
    @pytest.mark.asyncio
    async def test_process_message_error_handling(self):
        """Test message processing error handling."""
        agent = UserTwinAgent("test_user_twin")
        
        with patch.object(agent, 'agent_executor') as mock_executor:
            mock_executor.ainvoke = AsyncMock(side_effect=Exception("Test error"))
            
            response = await agent.process_message("Hello")
            
            assert "error" in response.lower()
            assert agent.status.current_state == AgentState.ERROR
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test agent shutdown."""
        agent = UserTwinAgent("test_user_twin")
        
        # Add some preferences
        agent.add_preference(PreferenceCategory.FOOD, "cuisine", "italian")
        
        with patch.object(agent.preference_manager, '_save_preferences') as mock_save:
            await agent.shutdown()
            
            mock_save.assert_called_once()
            assert agent.status.current_state == AgentState.STOPPED


class TestUserTwinFactory:
    """Test UserTwin factory function."""
    
    def test_create_user_twin_agent(self):
        """Test creating UserTwin agent with factory function."""
        agent = create_user_twin_agent(
            "factory_test",
            decision_style=DecisionStyle.INTUITIVE
        )
        
        assert agent.agent_id == "factory_test"
        assert agent.decision_style == DecisionStyle.INTUITIVE
        assert isinstance(agent, UserTwinAgent)
    
    def test_create_user_twin_agent_with_config(self):
        """Test creating UserTwin agent with custom configuration."""
        config = AgentConfiguration(
            agent_id="factory_test",
            agent_type=AgentType.USER_TWIN,
            name="Factory UserTwin",
            description="Test factory agent",
            personality=PersonalityProfile(
                primary_traits=[PersonalityTrait.CASUAL, PersonalityTrait.HELPFUL],
                response_style="conversational"
            )
        )
        
        agent = create_user_twin_agent(
            "factory_test",
            configuration=config,
            decision_style=DecisionStyle.CONSENSUS
        )
        
        assert agent.configuration.name == "Factory UserTwin"
        assert agent.decision_style == DecisionStyle.CONSENSUS


class TestUserTwinIntegration:
    """Integration tests for UserTwin agent."""
    
    @pytest.mark.asyncio
    async def test_complete_user_twin_workflow(self):
        """Test complete UserTwin workflow."""
        agent = UserTwinAgent("integration_test")
        
        # Set decision style
        agent.set_decision_style(DecisionStyle.ANALYTICAL)
        
        # Add preferences
        agent.add_preference(PreferenceCategory.FOOD, "cuisine", "italian", 0.8)
        agent.add_preference(PreferenceCategory.COST, "budget", "reasonable", 0.7)
        
        # Create decision context
        context = DecisionContext(
            scenario="food_ordering",
            options=[
                {"name": "Pizza", "cuisine": "italian", "price": 15},
                {"name": "Burger", "cuisine": "american", "price": 12},
                {"name": "Sushi", "cuisine": "japanese", "price": 25}
            ],
            constraints={"max_price": 20}
        )
        
        # Make decision
        decision = await agent.make_decision(context)
        
        assert decision.confidence > 0.0
        assert decision.context == context
        assert len(agent.get_decision_history()) == 1
        
        # Learn from interaction
        agent.learn_from_interaction(
            PreferenceCategory.FOOD,
            "cuisine",
            {"value": "italian", "satisfaction": 0.9, "context": "dinner"}
        )
        
        # Check updated preference
        preference = agent.get_preference(PreferenceCategory.FOOD, "cuisine")
        assert preference.strength > 0.8  # Should be stronger after positive interaction
        
        # Get user profile
        profile = agent.get_user_profile()
        assert profile["decision_style"] == "analytical"
        assert profile["recent_decisions"] == 1
        
        # Test message processing
        with patch.object(agent, 'agent_executor') as mock_executor:
            mock_executor.ainvoke = AsyncMock(return_value={"output": "Based on your preferences, I recommend Italian food."})
            
            response = await agent.process_message("What should I eat?")
            assert "Italian food" in response
        
        # Shutdown
        await agent.shutdown()
        assert agent.status.current_state == AgentState.STOPPED
    
    def test_preference_persistence(self):
        """Test preference persistence across agent instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first agent
            agent1 = UserTwinAgent("persistence_test")
            agent1.preference_manager.storage_path = Path(temp_dir)
            
            # Add preference
            agent1.add_preference(PreferenceCategory.FOOD, "cuisine", "italian", 0.8)
            
            # Create second agent with same storage
            agent2 = UserTwinAgent("persistence_test_2")
            agent2.preference_manager.storage_path = Path(temp_dir)
            agent2.preference_manager._load_preferences()
            
            # Check that preference is available in second agent
            preference = agent2.get_preference(PreferenceCategory.FOOD, "cuisine")
            assert preference is not None
            assert preference.value == "italian"
            assert preference.strength == 0.8
