"""
Unit tests for the Intelligent Order Management System.
"""

import asyncio
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from safehive.agents.intelligent_order_manager import (
    ConstraintType, ReasoningType, OrderConstraint, OrderReasoning,
    OrderOptimizationResult, IntelligentOrderManager
)
from safehive.agents.order_models import Order, OrderItem, Vendor, OrderStatus, OrderType, PaymentStatus
from safehive.agents.user_twin import UserTwinAgent, PreferenceCategory


class TestOrderConstraint:
    """Test OrderConstraint functionality."""
    
    def test_order_constraint_creation(self):
        """Test OrderConstraint creation."""
        constraint = OrderConstraint(
            constraint_type=ConstraintType.BUDGET,
            value=50.0,
            weight=0.9,
            is_hard=True,
            description="Budget limit constraint"
        )
        
        assert constraint.constraint_type == ConstraintType.BUDGET
        assert constraint.value == 50.0
        assert constraint.weight == 0.9
        assert constraint.is_hard is True
        assert constraint.description == "Budget limit constraint"
    
    def test_order_constraint_serialization(self):
        """Test OrderConstraint serialization."""
        constraint = OrderConstraint(
            constraint_type=ConstraintType.DIETARY,
            value="vegetarian",
            weight=0.95,
            is_hard=True,
            description="Vegetarian dietary requirement"
        )
        
        data = constraint.to_dict()
        
        assert data["constraint_type"] == "dietary"
        assert data["value"] == "vegetarian"
        assert data["weight"] == 0.95
        assert data["is_hard"] is True
        assert data["description"] == "Vegetarian dietary requirement"
    
    def test_order_constraint_deserialization(self):
        """Test OrderConstraint deserialization."""
        data = {
            "constraint_type": "time",
            "value": 30,
            "weight": 0.8,
            "is_hard": False,
            "description": "Delivery time preference"
        }
        
        constraint = OrderConstraint.from_dict(data)
        
        assert constraint.constraint_type == ConstraintType.TIME
        assert constraint.value == 30
        assert constraint.weight == 0.8
        assert constraint.is_hard is False
        assert constraint.description == "Delivery time preference"


class TestOrderReasoning:
    """Test OrderReasoning functionality."""
    
    def test_order_reasoning_creation(self):
        """Test OrderReasoning creation."""
        constraint = OrderConstraint(
            constraint_type=ConstraintType.BUDGET,
            value=50.0,
            weight=0.9
        )
        
        reasoning = OrderReasoning(
            reasoning_type=ReasoningType.OPTIMIZATION,
            decision="optimized_order",
            confidence=0.85,
            reasoning_steps=["Step 1", "Step 2"],
            constraints_checked=[constraint],
            alternatives_considered=[{"alt": "option1"}],
            metadata={"key": "value"}
        )
        
        assert reasoning.reasoning_type == ReasoningType.OPTIMIZATION
        assert reasoning.decision == "optimized_order"
        assert reasoning.confidence == 0.85
        assert len(reasoning.reasoning_steps) == 2
        assert len(reasoning.constraints_checked) == 1
        assert len(reasoning.alternatives_considered) == 1
        assert reasoning.metadata["key"] == "value"
    
    def test_order_reasoning_serialization(self):
        """Test OrderReasoning serialization."""
        constraint = OrderConstraint(
            constraint_type=ConstraintType.BUDGET,
            value=50.0,
            weight=0.9
        )
        
        reasoning = OrderReasoning(
            reasoning_type=ReasoningType.OPTIMIZATION,
            decision="optimized_order",
            confidence=0.85,
            reasoning_steps=["Step 1", "Step 2"],
            constraints_checked=[constraint],
            alternatives_considered=[{"alt": "option1"}],
            metadata={"key": "value"}
        )
        
        data = reasoning.to_dict()
        
        assert data["reasoning_type"] == "optimization"
        assert data["decision"] == "optimized_order"
        assert data["confidence"] == 0.85
        assert len(data["reasoning_steps"]) == 2
        assert len(data["constraints_checked"]) == 1
        assert len(data["alternatives_considered"]) == 1
        assert data["metadata"]["key"] == "value"
        assert "timestamp" in data


class TestIntelligentOrderManager:
    """Test IntelligentOrderManager functionality."""
    
    def test_intelligent_order_manager_creation(self):
        """Test IntelligentOrderManager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IntelligentOrderManager(temp_dir)
            
            assert manager.storage_path == Path(temp_dir)
            assert len(manager.constraint_weights) > 0
            assert ConstraintType.BUDGET in manager.constraint_weights
            assert ConstraintType.DIETARY in manager.constraint_weights
            assert len(manager.reasoning_history) == 0
    
    def test_add_constraint(self):
        """Test adding constraints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IntelligentOrderManager(temp_dir)
            
            constraint = OrderConstraint(
                constraint_type=ConstraintType.BUDGET,
                value=50.0,
                weight=0.9,
                description="Test budget constraint"
            )
            
            initial_weight = manager.constraint_weights[ConstraintType.BUDGET]
            manager.add_constraint(constraint)
            
            # Weight should be adjusted based on historical performance
            assert ConstraintType.BUDGET in manager.constraint_weights
    
    @pytest.mark.asyncio
    async def test_optimize_order_basic(self):
        """Test basic order optimization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IntelligentOrderManager(temp_dir)
            
            # Create test vendor
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            # Create test order items
            items = [
                OrderItem(
                    item_id="item_001",
                    name="Test Item 1",
                    quantity=1,
                    unit_price=20.0,
                    total_price=20.0,
                    dietary_requirements=["vegetarian"],
                    allergens=["gluten"]
                )
            ]
            
            # Create test order
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=items,
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=23.0
            )
            
            # Test optimization
            result = await manager.optimize_order(order)
            
            assert result is not None
            assert isinstance(result, OrderOptimizationResult)
            assert result.optimized_order is not None
            assert result.optimization_score >= 0.0
            assert result.optimization_score <= 1.0
            assert len(result.reasoning.reasoning_steps) > 0
    
    @pytest.mark.asyncio
    async def test_optimize_order_with_user_preferences(self):
        """Test order optimization with user preferences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IntelligentOrderManager(temp_dir)
            
            # Create user twin agent with preferences
            user_twin = UserTwinAgent("test_user")
            user_twin.add_preference(PreferenceCategory.COST, "budget", "moderate", 0.8)
            user_twin.add_preference(PreferenceCategory.FOOD, "dietary", "vegetarian", 0.9)
            user_twin.add_preference(PreferenceCategory.SPEED, "delivery_time", "fast", 0.7)
            
            # Create test vendor
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            # Create test order items
            items = [
                OrderItem(
                    item_id="item_001",
                    name="Vegetarian Pizza",
                    quantity=1,
                    unit_price=20.0,
                    total_price=20.0,
                    dietary_requirements=["vegetarian"],
                    allergens=[]
                )
            ]
            
            # Create test order
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=items,
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=23.0
            )
            
            # Test optimization with user preferences
            result = await manager.optimize_order(order, user_twin)
            
            assert result is not None
            assert result.optimized_order is not None
            assert len(result.reasoning.constraints_checked) > 0
            
            # Check that user preference constraints were extracted
            constraint_types = [c.constraint_type for c in result.reasoning.constraints_checked]
            assert ConstraintType.BUDGET in constraint_types
            assert ConstraintType.DIETARY in constraint_types
            assert ConstraintType.TIME in constraint_types
    
    @pytest.mark.asyncio
    async def test_optimize_order_with_custom_constraints(self):
        """Test order optimization with custom constraints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IntelligentOrderManager(temp_dir)
            
            # Create test vendor
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            # Create test order items
            items = [
                OrderItem(
                    item_id="item_001",
                    name="Test Item 1",
                    quantity=1,
                    unit_price=20.0,
                    total_price=20.0
                )
            ]
            
            # Create test order
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=items,
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=23.0
            )
            
            # Create custom constraints
            constraints = [
                OrderConstraint(
                    constraint_type=ConstraintType.BUDGET,
                    value=30.0,
                    weight=0.9,
                    description="Custom budget constraint"
                ),
                OrderConstraint(
                    constraint_type=ConstraintType.TIME,
                    value=30,
                    weight=0.8,
                    description="Custom time constraint"
                )
            ]
            
            # Test optimization with custom constraints
            result = await manager.optimize_order(order, constraints=constraints)
            
            assert result is not None
            assert result.optimized_order is not None
            
            # Check that custom constraints were used
            constraint_types = [c.constraint_type for c in result.reasoning.constraints_checked]
            assert ConstraintType.BUDGET in constraint_types
            assert ConstraintType.TIME in constraint_types
    
    @pytest.mark.asyncio
    async def test_analyze_order_conflicts(self):
        """Test order conflict analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IntelligentOrderManager(temp_dir)
            
            # Create user twin agent with preferences
            user_twin = UserTwinAgent("test_user")
            user_twin.add_preference(PreferenceCategory.FOOD, "dietary", "vegetarian", 0.9)
            user_twin.add_preference(PreferenceCategory.COST, "budget", "cheap", 0.8)
            user_twin.add_preference(PreferenceCategory.SPEED, "delivery_time", "fast", 0.7)
            
            # Create test vendor
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.5,
                delivery_time_minutes=35,  # Slow delivery
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            # Create test order items with potential conflicts
            items = [
                OrderItem(
                    item_id="item_001",
                    name="Beef Burger",  # Non-vegetarian
                    quantity=1,
                    unit_price=25.0,  # Expensive
                    total_price=25.0,
                    dietary_requirements=["meat"],
                    allergens=[]
                )
            ]
            
            # Create test order
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=items,
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=28.0  # Expensive
            )
            
            # Test conflict analysis
            analysis = await manager.analyze_order_conflicts(order, user_twin)
            
            assert analysis is not None
            assert "conflicts" in analysis
            assert "suggestions" in analysis
            assert "conflict_severity" in analysis
            
            # Should detect dietary conflict (beef burger vs vegetarian preference)
            dietary_conflicts = [c for c in analysis["conflicts"] if c["type"] == "dietary_conflict"]
            assert len(dietary_conflicts) > 0
            
            # Should detect budget conflict (expensive order vs cheap preference)
            budget_conflicts = [c for c in analysis["conflicts"] if c["type"] == "budget_conflict"]
            assert len(budget_conflicts) > 0
            
            # Should detect timing conflict (slow delivery vs fast preference)
            timing_conflicts = [c for c in analysis["conflicts"] if c["type"] == "timing_conflict"]
            assert len(timing_conflicts) > 0
            
            # Should have suggestions for resolution
            assert len(analysis["suggestions"]) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_order_conflicts_no_user_preferences(self):
        """Test order conflict analysis without user preferences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IntelligentOrderManager(temp_dir)
            
            # Create test vendor
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            # Create test order items
            items = [
                OrderItem(
                    item_id="item_001",
                    name="Test Item 1",
                    quantity=1,
                    unit_price=20.0,
                    total_price=20.0
                )
            ]
            
            # Create test order
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=items,
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=23.0
            )
            
            # Test conflict analysis without user preferences
            analysis = await manager.analyze_order_conflicts(order)
            
            assert analysis is not None
            assert "conflicts" in analysis
            assert "suggestions" in analysis
            assert "conflict_severity" in analysis
            
            # Should have no conflicts without user preferences
            assert len(analysis["conflicts"]) == 0
            assert analysis["conflict_severity"] == "none"
    
    def test_get_optimization_statistics(self):
        """Test getting optimization statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IntelligentOrderManager(temp_dir)
            
            # Test with no optimization history
            stats = manager.get_optimization_statistics()
            assert stats["total_optimizations"] == 0
            
            # Add some mock reasoning history
            constraint = OrderConstraint(
                constraint_type=ConstraintType.BUDGET,
                value=50.0,
                weight=0.9
            )
            
            reasoning1 = OrderReasoning(
                reasoning_type=ReasoningType.OPTIMIZATION,
                decision="order1",
                confidence=0.85,
                constraints_checked=[constraint]
            )
            
            reasoning2 = OrderReasoning(
                reasoning_type=ReasoningType.OPTIMIZATION,
                decision="order2",
                confidence=0.75,
                constraints_checked=[constraint]
            )
            
            manager.reasoning_history.extend([reasoning1, reasoning2])
            
            # Test statistics with history
            stats = manager.get_optimization_statistics()
            assert stats["total_optimizations"] == 2
            assert stats["average_confidence"] == 0.8
            assert "constraint_satisfaction_rates" in stats
    
    def test_constraint_satisfaction_checking(self):
        """Test constraint satisfaction checking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IntelligentOrderManager(temp_dir)
            
            # Create test vendor
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            # Create test order
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=[],
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=20.0
            )
            
            # Test budget constraint satisfaction
            budget_constraint = OrderConstraint(
                constraint_type=ConstraintType.BUDGET,
                value=30.0,
                weight=0.9
            )
            
            assert manager._constraint_satisfied(order, budget_constraint) is True
            
            budget_constraint.value = 15.0  # Order exceeds budget
            assert manager._constraint_satisfied(order, budget_constraint) is False
            
            # Test time constraint satisfaction
            time_constraint = OrderConstraint(
                constraint_type=ConstraintType.TIME,
                value=30,
                weight=0.8
            )
            
            assert manager._constraint_satisfied(order, time_constraint) is True
            
            time_constraint.value = 20  # Vendor too slow
            assert manager._constraint_satisfied(order, time_constraint) is False
            
            # Test rating constraint satisfaction
            rating_constraint = OrderConstraint(
                constraint_type=ConstraintType.RATING,
                value=4.0,
                weight=0.7
            )
            
            assert manager._constraint_satisfied(order, rating_constraint) is True
            
            rating_constraint.value = 5.0  # Vendor rating too low
            assert manager._constraint_satisfied(order, rating_constraint) is False
    
    def test_vendor_scoring(self):
        """Test vendor scoring based on constraints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IntelligentOrderManager(temp_dir)
            
            # Create test vendor
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="test",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            # Create constraints
            constraints = [
                OrderConstraint(
                    constraint_type=ConstraintType.TIME,
                    value=30,
                    weight=0.8
                ),
                OrderConstraint(
                    constraint_type=ConstraintType.RATING,
                    value=4.0,
                    weight=0.7
                ),
                OrderConstraint(
                    constraint_type=ConstraintType.BUDGET,
                    value=50.0,
                    weight=0.9
                )
            ]
            
            # Test vendor scoring
            score = manager._calculate_vendor_score(vendor, constraints)
            
            assert score >= 0.0
            assert score <= 2.0  # Score should be clamped between 0 and 2
            
            # Test with vendor that doesn't meet constraints
            bad_vendor = Vendor(
                vendor_id="vendor_002",
                name="Bad Vendor",
                cuisine_type="test",
                rating=2.0,  # Low rating
                delivery_time_minutes=60,  # Slow delivery
                minimum_order=15.0,
                delivery_fee=10.0  # Expensive
            )
            
            bad_score = manager._calculate_vendor_score(bad_vendor, constraints)
            assert bad_score < score  # Bad vendor should have lower score


class TestOrderOptimizationResult:
    """Test OrderOptimizationResult functionality."""
    
    def test_order_optimization_result_creation(self):
        """Test OrderOptimizationResult creation."""
        # Create test order
        vendor = Vendor(
            vendor_id="vendor_001",
            name="Test Vendor",
            cuisine_type="test",
            rating=4.5,
            delivery_time_minutes=25,
            minimum_order=15.0,
            delivery_fee=3.0
        )
        
        order = Order(
            order_id="order_001",
            user_id="user_001",
            vendor=vendor,
            items=[],
            order_type=OrderType.DELIVERY,
            status=OrderStatus.PENDING,
            payment_status=PaymentStatus.PENDING,
            total_amount=20.0
        )
        
        # Create reasoning
        reasoning = OrderReasoning(
            reasoning_type=ReasoningType.OPTIMIZATION,
            decision=order,
            confidence=0.85
        )
        
        # Create result
        result = OrderOptimizationResult(
            optimized_order=order,
            optimization_score=0.85,
            reasoning=reasoning,
            improvements=["Improved item selection", "Optimized delivery time"],
            warnings=["Budget constraint warning"],
            metadata={"test": "value"}
        )
        
        assert result.optimized_order == order
        assert result.optimization_score == 0.85
        assert result.reasoning == reasoning
        assert len(result.improvements) == 2
        assert len(result.warnings) == 1
        assert result.metadata["test"] == "value"


class TestIntelligentOrderManagerIntegration:
    """Integration tests for IntelligentOrderManager."""
    
    @pytest.mark.asyncio
    async def test_complete_optimization_workflow(self):
        """Test complete optimization workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = IntelligentOrderManager(temp_dir)
            
            # Create user twin agent with preferences
            user_twin = UserTwinAgent("test_user")
            user_twin.add_preference(PreferenceCategory.FOOD, "dietary", "vegetarian", 0.9)
            user_twin.add_preference(PreferenceCategory.COST, "budget", "moderate", 0.8)
            user_twin.add_preference(PreferenceCategory.SPEED, "delivery_time", "fast", 0.7)
            
            # Create test vendor
            vendor = Vendor(
                vendor_id="vendor_001",
                name="Test Vendor",
                cuisine_type="italian",
                rating=4.5,
                delivery_time_minutes=25,
                minimum_order=15.0,
                delivery_fee=3.0
            )
            
            # Create test order items
            items = [
                OrderItem(
                    item_id="item_001",
                    name="Vegetarian Pizza",
                    quantity=1,
                    unit_price=18.0,
                    total_price=18.0,
                    dietary_requirements=["vegetarian"],
                    allergens=["gluten"]
                ),
                OrderItem(
                    item_id="item_002",
                    name="Caesar Salad",
                    quantity=1,
                    unit_price=12.0,
                    total_price=12.0,
                    dietary_requirements=["vegetarian"],
                    allergens=["dairy"]
                )
            ]
            
            # Create test order
            order = Order(
                order_id="order_001",
                user_id="user_001",
                vendor=vendor,
                items=items,
                order_type=OrderType.DELIVERY,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=33.0
            )
            
            # Step 1: Optimize order
            optimization_result = await manager.optimize_order(order, user_twin)
            
            assert optimization_result is not None
            assert optimization_result.optimized_order is not None
            assert optimization_result.optimization_score > 0.0
            
            # Step 2: Analyze conflicts
            conflict_analysis = await manager.analyze_order_conflicts(
                optimization_result.optimized_order, user_twin
            )
            
            assert conflict_analysis is not None
            assert "conflicts" in conflict_analysis
            assert "suggestions" in conflict_analysis
            
            # Step 3: Check statistics
            stats = manager.get_optimization_statistics()
            assert stats["total_optimizations"] == 1
            assert stats["average_confidence"] == optimization_result.optimization_score
            
            # Step 4: Verify reasoning history
            assert len(manager.reasoning_history) == 1
            assert manager.reasoning_history[0].reasoning_type == ReasoningType.MULTI_OBJECTIVE
    
    @pytest.mark.asyncio
    async def test_persistence_and_recovery(self):
        """Test persistence and recovery of reasoning history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first manager instance
            manager1 = IntelligentOrderManager(temp_dir)
            
            # Add some reasoning history
            constraint = OrderConstraint(
                constraint_type=ConstraintType.BUDGET,
                value=50.0,
                weight=0.9
            )
            
            reasoning = OrderReasoning(
                reasoning_type=ReasoningType.OPTIMIZATION,
                decision="test_order",
                confidence=0.85,
                constraints_checked=[constraint]
            )
            
            manager1.reasoning_history.append(reasoning)
            manager1._save_reasoning_history()
            
            # Create second manager instance (should load history)
            manager2 = IntelligentOrderManager(temp_dir)
            
            assert len(manager2.reasoning_history) == 1
            assert manager2.reasoning_history[0].reasoning_type == ReasoningType.OPTIMIZATION
            assert manager2.reasoning_history[0].confidence == 0.85
            assert len(manager2.reasoning_history[0].constraints_checked) == 1
