"""
Unit tests for enhanced preference management features.
"""

import asyncio
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
import pytest

from safehive.agents.user_twin import (
    PreferenceCategory, UserPreference, PreferenceConflict, PreferenceManager,
    UserTwinAgent, DecisionStyle
)


class TestPreferenceConflict:
    """Test preference conflict functionality."""
    
    def test_preference_conflict_creation(self):
        """Test preference conflict creation."""
        pref1 = UserPreference(PreferenceCategory.FOOD, "cuisine", "italian", 0.8)
        pref2 = UserPreference(PreferenceCategory.FOOD, "cuisine", "chinese", 0.6)
        
        conflict = PreferenceConflict(pref1, pref2, "competing", 0.7)
        
        assert conflict.preference1 == pref1
        assert conflict.preference2 == pref2
        assert conflict.conflict_type == "competing"
        assert conflict.severity == 0.7
        assert not conflict.resolved
        assert conflict.resolution_method is None
        assert isinstance(conflict.created_at, datetime)
    
    def test_preference_conflict_serialization(self):
        """Test preference conflict serialization."""
        pref1 = UserPreference(PreferenceCategory.FOOD, "cuisine", "italian", 0.8)
        pref2 = UserPreference(PreferenceCategory.FOOD, "cuisine", "chinese", 0.6)
        
        conflict = PreferenceConflict(pref1, pref2, "competing", 0.7)
        conflict.resolved = True
        conflict.resolution_method = "strength_based"
        
        data = conflict.to_dict()
        
        assert data["conflict_type"] == "competing"
        assert data["severity"] == 0.7
        assert data["resolved"] is True
        assert data["resolution_method"] == "strength_based"
        assert "created_at" in data
        assert "preference1_key" in data
        assert "preference2_key" in data


class TestEnhancedPreferenceManager:
    """Test enhanced preference manager functionality."""
    
    def test_preference_conflict_detection(self):
        """Test automatic conflict detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            # Add first preference
            pref1 = UserPreference(
                PreferenceCategory.FOOD, "cuisine", "italian", 0.8, "dinner"
            )
            manager.add_preference(pref1)
            
            # Add conflicting preference
            pref2 = UserPreference(
                PreferenceCategory.FOOD, "cuisine", "chinese", 0.6, "dinner"
            )
            manager.add_preference(pref2)
            
            # Check for conflicts
            conflicts = manager.get_active_conflicts()
            assert len(conflicts) > 0
            assert conflicts[0].conflict_type == "competing"
    
    def test_preference_conflict_resolution_strength_based(self):
        """Test strength-based conflict resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            # Add conflicting preferences
            pref1 = UserPreference(
                PreferenceCategory.FOOD, "cuisine", "italian", 0.8, "dinner"
            )
            pref2 = UserPreference(
                PreferenceCategory.FOOD, "cuisine", "chinese", 0.6, "dinner"
            )
            
            manager.add_preference(pref1)
            manager.add_preference(pref2)
            
            conflicts = manager.get_active_conflicts()
            assert len(conflicts) == 1
            
            # Resolve using strength-based method
            manager.resolve_conflict(conflicts[0], "strength_based")
            
            # Check that stronger preference remains
            remaining_pref = manager.get_preference(PreferenceCategory.FOOD, "cuisine")
            assert remaining_pref is not None
            assert remaining_pref.value == "italian"  # Higher strength
            assert remaining_pref.strength == 0.8
    
    def test_preference_conflict_resolution_recent_based(self):
        """Test recent-based conflict resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            # Add conflicting preferences with different last_used times
            pref1 = UserPreference(
                PreferenceCategory.FOOD, "cuisine", "italian", 0.8, "dinner"
            )
            pref1.last_used = datetime.now() - timedelta(days=2)
            
            pref2 = UserPreference(
                PreferenceCategory.FOOD, "cuisine", "chinese", 0.6, "dinner"
            )
            pref2.last_used = datetime.now() - timedelta(days=1)  # More recent
            
            manager.add_preference(pref1)
            manager.add_preference(pref2)
            
            conflicts = manager.get_active_conflicts()
            
            # Resolve using recent-based method
            manager.resolve_conflict(conflicts[0], "recent_based")
            
            # Check that more recent preference remains
            remaining_pref = manager.get_preference(PreferenceCategory.FOOD, "cuisine")
            assert remaining_pref is not None
            assert remaining_pref.value == "chinese"  # More recent
    
    def test_auto_resolve_conflicts(self):
        """Test automatic conflict resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            # Add multiple conflicting preferences
            manager.add_preference(UserPreference(
                PreferenceCategory.FOOD, "cuisine", "italian", 0.8, "dinner"
            ))
            manager.add_preference(UserPreference(
                PreferenceCategory.FOOD, "cuisine", "chinese", 0.6, "dinner"
            ))
            manager.add_preference(UserPreference(
                PreferenceCategory.COST, "budget", "cheap", 0.7, "lunch"
            ))
            manager.add_preference(UserPreference(
                PreferenceCategory.COST, "budget", "expensive", 0.5, "lunch"
            ))
            
            # Auto-resolve conflicts
            resolved_count = manager.auto_resolve_conflicts()
            assert resolved_count > 0
            
            # Check that conflicts are resolved
            remaining_conflicts = manager.get_active_conflicts()
            assert len(remaining_conflicts) == 0
    
    def test_preference_decay(self):
        """Test temporal preference decay."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            # Add preference with old last_used date
            pref = UserPreference(
                PreferenceCategory.FOOD, "cuisine", "italian", 0.8, "dinner"
            )
            pref.last_used = datetime.now() - timedelta(days=10)
            manager.add_preference(pref)
            
            initial_strength = pref.strength
            
            # Apply decay
            manager.apply_preference_decay()
            
            # Check that strength decreased
            updated_pref = manager.get_preference(PreferenceCategory.FOOD, "cuisine")
            assert updated_pref.strength < initial_strength
            assert updated_pref.strength >= 0.0  # Should not go below 0
    
    def test_preference_history_tracking(self):
        """Test preference change history tracking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            # Add preferences
            manager.add_preference(UserPreference(
                PreferenceCategory.FOOD, "cuisine", "italian", 0.8
            ))
            manager.add_preference(UserPreference(
                PreferenceCategory.FOOD, "spice_level", "medium", 0.6
            ))
            
            # Check history
            history = manager.get_preference_history()
            assert len(history) == 2
            
            # Check history content
            for entry in history:
                assert "timestamp" in entry
                assert "action" in entry
                assert "preference_key" in entry
                assert "value" in entry
                assert "strength" in entry
    
    def test_preference_consistency_validation(self):
        """Test preference consistency validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            # Add various preferences
            manager.add_preference(UserPreference(
                PreferenceCategory.FOOD, "cuisine", "italian", 0.8
            ))
            manager.add_preference(UserPreference(
                PreferenceCategory.FOOD, "spice_level", "medium", 0.2  # Low strength
            ))
            
            # Add old unused preference
            old_pref = UserPreference(
                PreferenceCategory.COST, "budget", "cheap", 0.7
            )
            old_pref.last_used = datetime.now() - timedelta(days=35)
            manager.add_preference(old_pref)
            
            # Validate consistency
            report = manager.validate_preference_consistency()
            
            assert "total_preferences" in report
            assert "active_conflicts" in report
            assert "inconsistencies" in report
            assert "warnings" in report
            assert "suggestions" in report
            
            # Check that warnings are generated
            assert len(report["warnings"]) > 0  # Low strength preference
            assert len(report["suggestions"]) > 0  # Unused preference
    
    def test_contradictory_preference_detection(self):
        """Test detection of contradictory preferences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            # Add contradictory preferences
            manager.add_preference(UserPreference(
                PreferenceCategory.COST, "budget", "cheap", 0.8, "lunch"
            ))
            manager.add_preference(UserPreference(
                PreferenceCategory.COST, "budget", "expensive", 0.6, "dinner"
            ))
            
            conflicts = manager.get_active_conflicts()
            assert len(conflicts) > 0
            
            # Check that contradictory conflict is detected
            contradictory_conflicts = [c for c in conflicts if c.conflict_type == "contradictory"]
            assert len(contradictory_conflicts) > 0
    
    def test_overlapping_context_detection(self):
        """Test detection of overlapping context preferences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PreferenceManager(temp_dir)
            
            # Add preferences with overlapping contexts
            manager.add_preference(UserPreference(
                PreferenceCategory.FOOD, "cuisine", "italian", 0.8, "dinner restaurant"
            ))
            manager.add_preference(UserPreference(
                PreferenceCategory.SPEED, "delivery_time", "fast", 0.7, "dinner delivery"
            ))
            
            conflicts = manager.get_active_conflicts()
            
            # Check for overlapping context conflicts
            overlapping_conflicts = [c for c in conflicts if c.conflict_type == "overlapping"]
            assert len(overlapping_conflicts) > 0


class TestEnhancedUserTwinAgent:
    """Test enhanced UserTwin agent with new preference features."""
    
    def test_enhanced_preference_management(self):
        """Test enhanced preference management in UserTwin agent."""
        agent = UserTwinAgent("test_enhanced_agent")
        
        # Add conflicting preferences
        agent.add_preference(PreferenceCategory.FOOD, "cuisine", "italian", 0.8)
        agent.add_preference(PreferenceCategory.FOOD, "cuisine", "chinese", 0.6)
        
        # Check for conflicts
        conflicts = agent.get_preference_conflicts()
        assert len(conflicts) > 0
        
        # Resolve conflicts
        resolved_count = agent.resolve_preference_conflicts(auto_resolve=True)
        assert resolved_count > 0
        
        # Check that conflicts are resolved
        remaining_conflicts = agent.get_preference_conflicts()
        assert len(remaining_conflicts) == 0
    
    def test_preference_validation(self):
        """Test preference validation in UserTwin agent."""
        agent = UserTwinAgent("test_validation_agent")
        
        # Add various preferences
        agent.add_preference(PreferenceCategory.FOOD, "cuisine", "italian", 0.8)
        agent.add_preference(PreferenceCategory.FOOD, "spice_level", "medium", 0.2)
        
        # Validate preferences
        validation_report = agent.validate_preferences()
        
        assert "total_preferences" in validation_report
        assert "warnings" in validation_report
        assert len(validation_report["warnings"]) > 0  # Low strength warning
    
    def test_preference_history_access(self):
        """Test access to preference history."""
        agent = UserTwinAgent("test_history_agent")
        
        # Add preferences
        agent.add_preference(PreferenceCategory.FOOD, "cuisine", "italian", 0.8)
        agent.add_preference(PreferenceCategory.FOOD, "spice_level", "medium", 0.6)
        
        # Get history
        history = agent.get_preference_history()
        assert len(history) == 2
        
        # Check history content
        for entry in history:
            assert "timestamp" in entry
            assert "action" in entry
            assert "preference_key" in entry
    
    def test_preference_removal(self):
        """Test preference removal functionality."""
        agent = UserTwinAgent("test_removal_agent")
        
        # Add preference
        agent.add_preference(PreferenceCategory.FOOD, "cuisine", "italian", 0.8)
        
        # Verify it exists
        pref = agent.get_preference(PreferenceCategory.FOOD, "cuisine")
        assert pref is not None
        
        # Remove preference
        success = agent.remove_preference(PreferenceCategory.FOOD, "cuisine")
        assert success
        
        # Verify it's removed
        removed_pref = agent.get_preference(PreferenceCategory.FOOD, "cuisine")
        assert removed_pref is None
    
    def test_preference_context_update(self):
        """Test preference context update functionality."""
        agent = UserTwinAgent("test_context_agent")
        
        # Add preference
        agent.add_preference(PreferenceCategory.FOOD, "cuisine", "italian", 0.8, "original context")
        
        # Update context
        success = agent.update_preference_context(
            PreferenceCategory.FOOD, "cuisine", "updated context"
        )
        assert success
        
        # Verify context is updated
        updated_pref = agent.get_preference(PreferenceCategory.FOOD, "cuisine")
        assert updated_pref.context == "updated context"
    
    def test_enhanced_user_profile(self):
        """Test enhanced user profile with preference consistency."""
        agent = UserTwinAgent("test_profile_agent")
        
        # Add preferences
        agent.add_preference(PreferenceCategory.FOOD, "cuisine", "italian", 0.8)
        agent.add_preference(PreferenceCategory.COST, "budget", "moderate", 0.7)
        
        # Get enhanced profile
        profile = agent.get_user_profile()
        
        # Check that preference consistency is included
        assert "preference_consistency" in profile
        consistency = profile["preference_consistency"]
        assert "total_preferences" in consistency
        assert "active_conflicts" in consistency
    
    def test_preference_decay_integration(self):
        """Test preference decay integration with UserTwin agent."""
        agent = UserTwinAgent("test_decay_agent")
        
        # Add preference with old timestamp
        pref = UserPreference(
            PreferenceCategory.FOOD, "cuisine", "italian", 0.8, "dinner"
        )
        pref.last_used = datetime.now() - timedelta(days=5)
        agent.preference_manager.add_preference(pref)
        
        initial_strength = pref.strength
        
        # Apply decay
        agent.apply_preference_decay()
        
        # Check that strength decreased
        updated_pref = agent.get_preference(PreferenceCategory.FOOD, "cuisine")
        assert updated_pref.strength < initial_strength


class TestPreferenceIntegration:
    """Integration tests for enhanced preference management."""
    
    def test_complete_preference_lifecycle(self):
        """Test complete preference lifecycle with conflicts and resolution."""
        agent = UserTwinAgent("test_lifecycle_agent")
        
        # Phase 1: Add initial preferences
        agent.add_preference(PreferenceCategory.FOOD, "cuisine", "italian", 0.8)
        agent.add_preference(PreferenceCategory.COST, "budget", "moderate", 0.7)
        
        # Phase 2: Add conflicting preferences
        agent.add_preference(PreferenceCategory.FOOD, "cuisine", "chinese", 0.6)
        agent.add_preference(PreferenceCategory.COST, "budget", "expensive", 0.5)
        
        # Phase 3: Check for conflicts
        conflicts = agent.get_preference_conflicts()
        assert len(conflicts) >= 2
        
        # Phase 4: Auto-resolve conflicts
        resolved_count = agent.resolve_preference_conflicts(auto_resolve=True)
        assert resolved_count >= 2
        
        # Phase 5: Validate consistency
        validation_report = agent.validate_preferences()
        assert validation_report["active_conflicts"] == 0
        
        # Phase 6: Apply decay and check history
        agent.apply_preference_decay()
        history = agent.get_preference_history()
        assert len(history) >= 4  # At least 4 preference operations
        
        # Phase 7: Get enhanced profile
        profile = agent.get_user_profile()
        assert "preference_consistency" in profile
        assert profile["preference_consistency"]["total_preferences"] > 0
    
    def test_preference_persistence_with_conflicts(self):
        """Test preference persistence with conflicts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first agent
            agent1 = UserTwinAgent("persistence_test_agent")
            agent1.preference_manager.storage_path = Path(temp_dir)
            
            # Add conflicting preferences
            agent1.add_preference(PreferenceCategory.FOOD, "cuisine", "italian", 0.8)
            agent1.add_preference(PreferenceCategory.FOOD, "cuisine", "chinese", 0.6)
            
            # Check conflicts exist
            conflicts1 = agent1.get_preference_conflicts()
            assert len(conflicts1) > 0
            
            # Create second agent with same storage
            agent2 = UserTwinAgent("persistence_test_agent_2")
            agent2.preference_manager.storage_path = Path(temp_dir)
            agent2.preference_manager._load_preferences()
            agent2.preference_manager._load_conflicts()
            
            # Check that conflicts are loaded
            conflicts2 = agent2.get_preference_conflicts()
            assert len(conflicts2) > 0
            
            # Resolve conflicts in second agent
            resolved_count = agent2.resolve_preference_conflicts(auto_resolve=True)
            assert resolved_count > 0
            
            # Verify resolution is persistent
            conflicts_after = agent2.get_preference_conflicts()
            assert len(conflicts_after) == 0
