"""
Intelligent Order Management System

This module implements intelligent order management with reasoning and constraint checking
for the SafeHive AI Security Sandbox. It provides advanced order optimization, conflict
resolution, and intelligent decision-making capabilities.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from .order_models import Order, OrderItem, Vendor, OrderStatus, OrderType, PaymentStatus
from .user_twin import UserTwinAgent, PreferenceCategory, DecisionContext, DecisionStyle
from ..utils.logger import get_logger
from ..utils.metrics import record_metric, MetricType

logger = get_logger(__name__)


class ConstraintType(Enum):
    """Types of constraints that can be applied to orders."""
    BUDGET = "budget"
    TIME = "time"
    DIETARY = "dietary"
    ALLERGEN = "allergen"
    DISTANCE = "distance"
    AVAILABILITY = "availability"
    RATING = "rating"
    CUSTOM = "custom"


class ReasoningType(Enum):
    """Types of reasoning approaches for order management."""
    OPTIMIZATION = "optimization"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    MULTI_OBJECTIVE = "multi_objective"
    RISK_ASSESSMENT = "risk_assessment"
    USER_PREFERENCE = "user_preference"
    BUSINESS_LOGIC = "business_logic"


@dataclass
class OrderConstraint:
    """Represents a constraint that must be satisfied for an order."""
    constraint_type: ConstraintType
    value: Any
    weight: float = 1.0  # Importance weight (0.0 to 1.0)
    is_hard: bool = True  # Hard constraint (must be satisfied) or soft constraint
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "constraint_type": self.constraint_type.value,
            "value": self.value,
            "weight": self.weight,
            "is_hard": self.is_hard,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderConstraint":
        """Create from dictionary."""
        return cls(
            constraint_type=ConstraintType(data["constraint_type"]),
            value=data["value"],
            weight=data.get("weight", 1.0),
            is_hard=data.get("is_hard", True),
            description=data.get("description", "")
        )


@dataclass
class OrderReasoning:
    """Represents the reasoning process for an order decision."""
    reasoning_type: ReasoningType
    decision: Any
    confidence: float  # 0.0 to 1.0
    reasoning_steps: List[str] = field(default_factory=list)
    constraints_checked: List[OrderConstraint] = field(default_factory=list)
    alternatives_considered: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "reasoning_type": self.reasoning_type.value,
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "constraints_checked": [c.to_dict() for c in self.constraints_checked],
            "alternatives_considered": self.alternatives_considered,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OrderOptimizationResult:
    """Result of order optimization process."""
    optimized_order: Optional[Order]
    optimization_score: float
    reasoning: OrderReasoning
    improvements: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentOrderManager:
    """Intelligent order management with reasoning and constraint checking."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_intelligent_orders"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Reasoning and optimization components
        self.constraint_weights = {
            ConstraintType.BUDGET: 0.9,
            ConstraintType.TIME: 0.8,
            ConstraintType.DIETARY: 0.95,
            ConstraintType.ALLERGEN: 1.0,
            ConstraintType.RATING: 0.7,
            ConstraintType.AVAILABILITY: 1.0,
            ConstraintType.DISTANCE: 0.6,
            ConstraintType.CUSTOM: 0.5
        }
        
        # Reasoning history and learning
        self.reasoning_history: List[OrderReasoning] = []
        self.optimization_patterns: Dict[str, Any] = {}
        self.user_preference_patterns: Dict[str, Any] = {}
        
        # Load historical data
        self._load_reasoning_history()
        
        logger.info("Intelligent Order Manager initialized")
    
    def _load_reasoning_history(self) -> None:
        """Load reasoning history from storage."""
        history_file = self.storage_path / "reasoning_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for reasoning_data in data:
                        reasoning = self._reconstruct_reasoning(reasoning_data)
                        if reasoning:
                            self.reasoning_history.append(reasoning)
                logger.info(f"Loaded {len(self.reasoning_history)} reasoning records")
            except Exception as e:
                logger.error(f"Failed to load reasoning history: {e}")
    
    def _save_reasoning_history(self) -> None:
        """Save reasoning history to storage."""
        history_file = self.storage_path / "reasoning_history.json"
        try:
            data = [reasoning.to_dict() for reasoning in self.reasoning_history]
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved reasoning history")
        except Exception as e:
            logger.error(f"Failed to save reasoning history: {e}")
    
    def _reconstruct_reasoning(self, data: Dict[str, Any]) -> Optional[OrderReasoning]:
        """Reconstruct OrderReasoning from stored data."""
        try:
            constraints = [OrderConstraint.from_dict(c) for c in data.get("constraints_checked", [])]
            return OrderReasoning(
                reasoning_type=ReasoningType(data["reasoning_type"]),
                decision=data["decision"],
                confidence=data["confidence"],
                reasoning_steps=data.get("reasoning_steps", []),
                constraints_checked=constraints,
                alternatives_considered=data.get("alternatives_considered", []),
                metadata=data.get("metadata", {}),
                timestamp=datetime.fromisoformat(data["timestamp"])
            )
        except Exception as e:
            logger.error(f"Failed to reconstruct reasoning: {e}")
            return None
    
    def add_constraint(self, constraint: OrderConstraint) -> None:
        """Add a new constraint to the system."""
        # Update constraint weights based on historical performance
        if constraint.constraint_type in self.constraint_weights:
            # Adaptive weight adjustment based on historical success
            historical_success = self._get_constraint_success_rate(constraint.constraint_type)
            if historical_success > 0.8:
                self.constraint_weights[constraint.constraint_type] = min(1.0, 
                    self.constraint_weights[constraint.constraint_type] + 0.1)
            elif historical_success < 0.5:
                self.constraint_weights[constraint.constraint_type] = max(0.1, 
                    self.constraint_weights[constraint.constraint_type] - 0.1)
        
        logger.info(f"Added constraint: {constraint.constraint_type.value}")
    
    def _get_constraint_success_rate(self, constraint_type: ConstraintType) -> float:
        """Get historical success rate for a constraint type."""
        relevant_reasonings = [r for r in self.reasoning_history 
                             if any(c.constraint_type == constraint_type for c in r.constraints_checked)]
        
        if not relevant_reasonings:
            return 0.5  # Default neutral rate
        
        successful = sum(1 for r in relevant_reasonings if r.confidence > 0.7)
        return successful / len(relevant_reasonings)
    
    async def optimize_order(self, order: Order, user_twin_agent: Optional[UserTwinAgent] = None,
                           constraints: Optional[List[OrderConstraint]] = None) -> OrderOptimizationResult:
        """Intelligently optimize an order using reasoning and constraint checking."""
        logger.info(f"Starting order optimization for order: {order.order_id}")
        
        # Initialize constraints
        if constraints is None:
            constraints = []
        
        # Add user preference constraints if user twin is available
        if user_twin_agent:
            user_constraints = self._extract_user_preference_constraints(user_twin_agent)
            constraints.extend(user_constraints)
        
        # Add business logic constraints
        business_constraints = self._generate_business_logic_constraints(order)
        constraints.extend(business_constraints)
        
        # Perform multi-objective optimization
        optimization_result = await self._perform_multi_objective_optimization(
            order, constraints, user_twin_agent
        )
        
        # Record reasoning
        reasoning = OrderReasoning(
            reasoning_type=ReasoningType.MULTI_OBJECTIVE,
            decision=optimization_result.optimized_order,
            confidence=optimization_result.optimization_score,
            reasoning_steps=optimization_result.improvements,
            constraints_checked=constraints,
            alternatives_considered=optimization_result.metadata.get("alternatives", []),
            metadata=optimization_result.metadata
        )
        
        self.reasoning_history.append(reasoning)
        self._save_reasoning_history()
        
        # Record metrics
        record_metric("intelligent_order.optimization", 1, MetricType.COUNTER, {
            "order_id": order.order_id,
            "optimization_score": optimization_result.optimization_score,
            "constraints_count": len(constraints)
        })
        
        logger.info(f"Order optimization completed with score: {optimization_result.optimization_score}")
        return optimization_result
    
    def _extract_user_preference_constraints(self, user_twin_agent: UserTwinAgent) -> List[OrderConstraint]:
        """Extract constraints from user preferences."""
        constraints = []
        
        # Budget constraints
        cost_prefs = user_twin_agent.get_preferences_by_category(PreferenceCategory.COST)
        for pref in cost_prefs:
            if pref.key == "budget":
                if pref.value == "cheap":
                    constraints.append(OrderConstraint(
                        constraint_type=ConstraintType.BUDGET,
                        value=20.0,
                        weight=pref.strength,
                        description="User prefers budget-friendly options"
                    ))
                elif pref.value == "moderate":
                    constraints.append(OrderConstraint(
                        constraint_type=ConstraintType.BUDGET,
                        value=50.0,
                        weight=pref.strength,
                        description="User prefers moderate pricing"
                    ))
        
        # Dietary constraints
        food_prefs = user_twin_agent.get_preferences_by_category(PreferenceCategory.FOOD)
        for pref in food_prefs:
            if pref.key == "dietary":
                constraints.append(OrderConstraint(
                    constraint_type=ConstraintType.DIETARY,
                    value=pref.value,
                    weight=pref.strength,
                    description=f"User dietary preference: {pref.value}"
                ))
        
        # Time constraints
        speed_prefs = user_twin_agent.get_preferences_by_category(PreferenceCategory.SPEED)
        for pref in speed_prefs:
            if pref.key == "delivery_time":
                if pref.value == "fast":
                    constraints.append(OrderConstraint(
                        constraint_type=ConstraintType.TIME,
                        value=20,  # minutes
                        weight=pref.strength,
                        description="User prefers fast delivery"
                    ))
                elif pref.value == "moderate":
                    constraints.append(OrderConstraint(
                        constraint_type=ConstraintType.TIME,
                        value=40,  # minutes
                        weight=pref.strength,
                        description="User accepts moderate delivery time"
                    ))
        
        return constraints
    
    def _generate_business_logic_constraints(self, order: Order) -> List[OrderConstraint]:
        """Generate business logic constraints for an order."""
        constraints = []
        
        # Vendor availability constraint
        constraints.append(OrderConstraint(
            constraint_type=ConstraintType.AVAILABILITY,
            value=True,
            weight=1.0,
            is_hard=True,
            description="Vendor must be available for orders"
        ))
        
        # Minimum order amount constraint
        constraints.append(OrderConstraint(
            constraint_type=ConstraintType.BUDGET,
            value=order.vendor.minimum_order,
            weight=1.0,
            is_hard=True,
            description="Order must meet vendor minimum amount"
        ))
        
        # Rating constraint (minimum acceptable rating)
        constraints.append(OrderConstraint(
            constraint_type=ConstraintType.RATING,
            value=3.5,  # Minimum acceptable rating
            weight=0.8,
            is_hard=False,
            description="Vendor should have acceptable rating"
        ))
        
        return constraints
    
    async def _perform_multi_objective_optimization(self, order: Order, 
                                                   constraints: List[OrderConstraint],
                                                   user_twin_agent: Optional[UserTwinAgent] = None) -> OrderOptimizationResult:
        """Perform multi-objective optimization of the order."""
        reasoning_steps = []
        improvements = []
        warnings = []
        alternatives = []
        
        # Step 1: Check hard constraints
        reasoning_steps.append("Checking hard constraints...")
        hard_constraint_violations = self._check_hard_constraints(order, constraints)
        
        if hard_constraint_violations:
            warnings.extend([f"Hard constraint violation: {v}" for v in hard_constraint_violations])
            reasoning_steps.append(f"Found {len(hard_constraint_violations)} hard constraint violations")
        
        # Step 2: Optimize item selection
        reasoning_steps.append("Optimizing item selection...")
        optimized_items = self._optimize_item_selection(order.items, constraints, user_twin_agent)
        
        if optimized_items != order.items:
            improvements.append("Optimized item selection based on constraints")
            reasoning_steps.append("Applied item selection optimizations")
        
        # Step 3: Optimize vendor selection (if multiple options)
        reasoning_steps.append("Evaluating vendor alternatives...")
        vendor_alternatives = self._find_vendor_alternatives(order, constraints)
        alternatives.extend(vendor_alternatives)
        
        if vendor_alternatives:
            best_vendor = self._select_best_vendor(vendor_alternatives, constraints)
            if best_vendor != order.vendor:
                improvements.append("Selected better vendor based on constraints")
                reasoning_steps.append(f"Selected vendor: {best_vendor.name}")
        
        # Step 4: Optimize delivery options
        reasoning_steps.append("Optimizing delivery options...")
        optimized_order_type = self._optimize_delivery_type(order, constraints)
        
        if optimized_order_type != order.order_type:
            improvements.append(f"Optimized delivery type: {optimized_order_type.value}")
            reasoning_steps.append(f"Changed delivery type to {optimized_order_type.value}")
        
        # Step 5: Calculate optimization score
        optimization_score = self._calculate_optimization_score(order, constraints, improvements)
        
        # Create optimized order
        optimized_order = Order(
            order_id=order.order_id,
            user_id=order.user_id,
            vendor=order.vendor,  # Could be optimized vendor
            items=optimized_items,
            order_type=optimized_order_type,
            status=order.status,
            payment_status=order.payment_status,
            total_amount=self._calculate_total_amount(optimized_items, order.vendor, optimized_order_type),
            delivery_address=order.delivery_address,
            special_instructions=order.special_instructions,
            estimated_delivery_time=order.estimated_delivery_time,
            metadata={**order.metadata, "optimized": True, "optimization_score": optimization_score}
        )
        
        return OrderOptimizationResult(
            optimized_order=optimized_order,
            optimization_score=optimization_score,
            reasoning=OrderReasoning(
                reasoning_type=ReasoningType.MULTI_OBJECTIVE,
                decision=optimized_order,
                confidence=optimization_score,
                reasoning_steps=reasoning_steps,
                constraints_checked=constraints,
                alternatives_considered=alternatives,
                metadata={"improvements": improvements, "warnings": warnings}
            ),
            improvements=improvements,
            warnings=warnings,
            metadata={"alternatives": alternatives}
        )
    
    def _check_hard_constraints(self, order: Order, constraints: List[OrderConstraint]) -> List[str]:
        """Check hard constraints and return violations."""
        violations = []
        
        for constraint in constraints:
            if not constraint.is_hard:
                continue
            
            if constraint.constraint_type == ConstraintType.BUDGET:
                if order.total_amount > constraint.value:
                    violations.append(f"Order total {order.total_amount} exceeds budget limit {constraint.value}")
            
            elif constraint.constraint_type == ConstraintType.AVAILABILITY:
                if not order.vendor.is_available:
                    violations.append(f"Vendor {order.vendor.name} is not available")
            
            elif constraint.constraint_type == ConstraintType.DIETARY:
                # Check if order items violate dietary constraints
                for item in order.items:
                    if constraint.value.lower() in [d.lower() for d in item.dietary_requirements]:
                        # This is actually good - item matches dietary requirement
                        continue
                    elif constraint.value.lower() == "vegetarian" and "meat" in item.name.lower():
                        violations.append(f"Item {item.name} violates vegetarian constraint")
            
            elif constraint.constraint_type == ConstraintType.ALLERGEN:
                # Check for allergen violations
                for item in order.items:
                    if constraint.value.lower() in [a.lower() for a in item.allergens]:
                        violations.append(f"Item {item.name} contains allergen: {constraint.value}")
        
        return violations
    
    def _optimize_item_selection(self, items: List[OrderItem], constraints: List[OrderConstraint],
                               user_twin_agent: Optional[UserTwinAgent] = None) -> List[OrderItem]:
        """Optimize item selection based on constraints."""
        optimized_items = []
        
        for item in items:
            # Check if item satisfies constraints
            item_satisfies = True
            item_score = 1.0
            
            for constraint in constraints:
                if constraint.constraint_type == ConstraintType.DIETARY:
                    if constraint.value.lower() not in [d.lower() for d in item.dietary_requirements]:
                        item_score *= 0.8  # Reduce score for non-matching dietary requirements
                
                elif constraint.constraint_type == ConstraintType.ALLERGEN:
                    if constraint.value.lower() in [a.lower() for a in item.allergens]:
                        item_satisfies = False
                        break
                
                elif constraint.constraint_type == ConstraintType.BUDGET:
                    if item.unit_price > constraint.value * 0.3:  # Item shouldn't be more than 30% of budget
                        item_score *= 0.7
            
            if item_satisfies and item_score > 0.5:
                optimized_items.append(item)
        
        return optimized_items if optimized_items else items  # Return original if no optimization
    
    def _find_vendor_alternatives(self, order: Order, constraints: List[OrderConstraint]) -> List[Dict[str, Any]]:
        """Find alternative vendors that might better satisfy constraints."""
        alternatives = []
        
        # This would typically integrate with a vendor database
        # For now, we'll simulate with the current vendor and some mock alternatives
        
        current_vendor_score = self._calculate_vendor_score(order.vendor, constraints)
        alternatives.append({
            "vendor": order.vendor,
            "score": current_vendor_score,
            "reason": "Current vendor"
        })
        
        # Mock alternative vendors (in real implementation, this would query a vendor database)
        mock_alternatives = [
            {
                "vendor": Vendor(
                    vendor_id="alt_001",
                    name="Alternative Vendor 1",
                    cuisine_type=order.vendor.cuisine_type,
                    rating=4.5,
                    delivery_time_minutes=order.vendor.delivery_time_minutes - 5,
                    minimum_order=order.vendor.minimum_order,
                    delivery_fee=order.vendor.delivery_fee - 1.0
                ),
                "reason": "Better delivery time and lower fee"
            },
            {
                "vendor": Vendor(
                    vendor_id="alt_002",
                    name="Alternative Vendor 2",
                    cuisine_type=order.vendor.cuisine_type,
                    rating=4.8,
                    delivery_time_minutes=order.vendor.delivery_time_minutes,
                    minimum_order=order.vendor.minimum_order,
                    delivery_fee=order.vendor.delivery_fee
                ),
                "reason": "Higher rating"
            }
        ]
        
        for alt in mock_alternatives:
            score = self._calculate_vendor_score(alt["vendor"], constraints)
            alternatives.append({
                "vendor": alt["vendor"],
                "score": score,
                "reason": alt["reason"]
            })
        
        # Sort by score (highest first)
        alternatives.sort(key=lambda x: x["score"], reverse=True)
        return alternatives
    
    def _calculate_vendor_score(self, vendor: Vendor, constraints: List[OrderConstraint]) -> float:
        """Calculate a score for a vendor based on constraints."""
        score = 1.0
        
        for constraint in constraints:
            weight = self.constraint_weights.get(constraint.constraint_type, 0.5)
            
            if constraint.constraint_type == ConstraintType.TIME:
                if vendor.delivery_time_minutes <= constraint.value:
                    score += weight * 0.2
                else:
                    score -= weight * 0.1
            
            elif constraint.constraint_type == ConstraintType.RATING:
                if vendor.rating >= constraint.value:
                    score += weight * 0.3
                else:
                    score -= weight * 0.2
            
            elif constraint.constraint_type == ConstraintType.BUDGET:
                if vendor.delivery_fee <= constraint.value * 0.1:  # Delivery fee within 10% of budget
                    score += weight * 0.1
        
        return max(0.0, min(2.0, score))  # Clamp between 0 and 2
    
    def _select_best_vendor(self, alternatives: List[Dict[str, Any]], 
                          constraints: List[OrderConstraint]) -> Vendor:
        """Select the best vendor from alternatives."""
        if not alternatives:
            return None
        
        # Return the vendor with the highest score
        best_alternative = alternatives[0]
        return best_alternative["vendor"]
    
    def _optimize_delivery_type(self, order: Order, constraints: List[OrderConstraint]) -> OrderType:
        """Optimize delivery type based on constraints."""
        # Analyze constraints to determine best delivery type
        time_constraint = next((c for c in constraints if c.constraint_type == ConstraintType.TIME), None)
        budget_constraint = next((c for c in constraints if c.constraint_type == ConstraintType.BUDGET), None)
        
        if time_constraint and time_constraint.value <= 15:
            # Very fast delivery needed
            return OrderType.PICKUP
        
        elif budget_constraint and budget_constraint.value < 25:
            # Budget-conscious - avoid delivery fees
            return OrderType.PICKUP if order.vendor.delivery_fee > 3.0 else order.order_type
        
        else:
            # Default to original delivery type
            return order.order_type
    
    def _calculate_optimization_score(self, order: Order, constraints: List[OrderConstraint],
                                    improvements: List[str]) -> float:
        """Calculate overall optimization score."""
        base_score = 0.7  # Base score for any order
        
        # Increase score based on constraint satisfaction
        satisfied_constraints = 0
        total_constraints = len(constraints)
        
        for constraint in constraints:
            if self._constraint_satisfied(order, constraint):
                satisfied_constraints += 1
        
        constraint_score = satisfied_constraints / max(1, total_constraints)
        
        # Increase score based on improvements
        improvement_score = min(0.3, len(improvements) * 0.1)
        
        # Final score
        final_score = base_score + (constraint_score * 0.2) + improvement_score
        return min(1.0, final_score)
    
    def _constraint_satisfied(self, order: Order, constraint: OrderConstraint) -> bool:
        """Check if a constraint is satisfied by the order."""
        if constraint.constraint_type == ConstraintType.BUDGET:
            return order.total_amount <= constraint.value
        
        elif constraint.constraint_type == ConstraintType.TIME:
            return order.vendor.delivery_time_minutes <= constraint.value
        
        elif constraint.constraint_type == ConstraintType.RATING:
            return order.vendor.rating >= constraint.value
        
        elif constraint.constraint_type == ConstraintType.AVAILABILITY:
            return order.vendor.is_available
        
        # Add more constraint types as needed
        return True  # Default to satisfied for unknown constraints
    
    def _calculate_total_amount(self, items: List[OrderItem], vendor: Vendor, 
                              order_type: OrderType) -> float:
        """Calculate total order amount."""
        items_total = sum(item.total_price for item in items)
        
        if order_type == OrderType.DELIVERY:
            return items_total + vendor.delivery_fee
        
        return items_total
    
    async def analyze_order_conflicts(self, order: Order, 
                                    user_twin_agent: Optional[UserTwinAgent] = None) -> Dict[str, Any]:
        """Analyze potential conflicts in an order."""
        logger.info(f"Analyzing order conflicts for order: {order.order_id}")
        
        conflicts = []
        warnings = []
        suggestions = []
        
        # Check for dietary conflicts
        dietary_conflicts = self._check_dietary_conflicts(order, user_twin_agent)
        conflicts.extend(dietary_conflicts)
        
        # Check for budget conflicts
        budget_conflicts = self._check_budget_conflicts(order, user_twin_agent)
        conflicts.extend(budget_conflicts)
        
        # Check for timing conflicts
        timing_conflicts = self._check_timing_conflicts(order, user_twin_agent)
        conflicts.extend(timing_conflicts)
        
        # Generate suggestions for resolution
        if conflicts:
            suggestions = self._generate_conflict_resolution_suggestions(conflicts, order)
        
        analysis_result = {
            "order_id": order.order_id,
            "conflicts": conflicts,
            "warnings": warnings,
            "suggestions": suggestions,
            "conflict_severity": self._calculate_conflict_severity(conflicts),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Record metrics
        record_metric("intelligent_order.conflict_analysis", 1, MetricType.COUNTER, {
            "order_id": order.order_id,
            "conflicts_count": len(conflicts),
            "severity": analysis_result["conflict_severity"]
        })
        
        return analysis_result
    
    def _check_dietary_conflicts(self, order: Order, user_twin_agent: Optional[UserTwinAgent]) -> List[Dict[str, Any]]:
        """Check for dietary preference conflicts."""
        conflicts = []
        
        if not user_twin_agent:
            return conflicts
        
        # Get user dietary preferences
        dietary_prefs = user_twin_agent.get_preferences_by_category(PreferenceCategory.FOOD)
        
        for pref in dietary_prefs:
            if pref.key == "dietary":
                user_dietary = pref.value.lower()
                
                for item in order.items:
                    item_dietary = [d.lower() for d in item.dietary_requirements]
                    
                    if user_dietary == "vegetarian" and ("meat" in item.name.lower() or "beef" in item.name.lower() or "chicken" in item.name.lower() or "pork" in item.name.lower()):
                        conflicts.append({
                            "type": "dietary_conflict",
                            "severity": "high",
                            "description": f"Item '{item.name}' contains meat but user prefers vegetarian",
                            "item": item.name,
                            "user_preference": user_dietary
                        })
                    
                    elif user_dietary == "vegan" and ("dairy" in item.name.lower() or "meat" in item.name.lower() or "beef" in item.name.lower() or "chicken" in item.name.lower() or "pork" in item.name.lower() or "cheese" in item.name.lower()):
                        conflicts.append({
                            "type": "dietary_conflict",
                            "severity": "high",
                            "description": f"Item '{item.name}' may contain animal products but user prefers vegan",
                            "item": item.name,
                            "user_preference": user_dietary
                        })
        
        return conflicts
    
    def _check_budget_conflicts(self, order: Order, user_twin_agent: Optional[UserTwinAgent]) -> List[Dict[str, Any]]:
        """Check for budget preference conflicts."""
        conflicts = []
        
        if not user_twin_agent:
            return conflicts
        
        # Get user budget preferences
        cost_prefs = user_twin_agent.get_preferences_by_category(PreferenceCategory.COST)
        
        for pref in cost_prefs:
            if pref.key == "budget":
                if pref.value == "cheap" and order.total_amount > 25.0:
                    conflicts.append({
                        "type": "budget_conflict",
                        "severity": "medium",
                        "description": f"Order total ${order.total_amount:.2f} may exceed user's budget preference for cheap options",
                        "order_total": order.total_amount,
                        "user_preference": pref.value
                    })
                
                elif pref.value == "moderate" and order.total_amount > 60.0:
                    conflicts.append({
                        "type": "budget_conflict",
                        "severity": "medium",
                        "description": f"Order total ${order.total_amount:.2f} may exceed user's moderate budget preference",
                        "order_total": order.total_amount,
                        "user_preference": pref.value
                    })
        
        return conflicts
    
    def _check_timing_conflicts(self, order: Order, user_twin_agent: Optional[UserTwinAgent]) -> List[Dict[str, Any]]:
        """Check for timing preference conflicts."""
        conflicts = []
        
        if not user_twin_agent:
            return conflicts
        
        # Get user speed preferences
        speed_prefs = user_twin_agent.get_preferences_by_category(PreferenceCategory.SPEED)
        
        for pref in speed_prefs:
            if pref.key == "delivery_time":
                if pref.value == "fast" and order.vendor.delivery_time_minutes > 25:
                    conflicts.append({
                        "type": "timing_conflict",
                        "severity": "low",
                        "description": f"Vendor delivery time {order.vendor.delivery_time_minutes} minutes may be too slow for user's fast delivery preference",
                        "vendor_delivery_time": order.vendor.delivery_time_minutes,
                        "user_preference": pref.value
                    })
        
        return conflicts
    
    def _generate_conflict_resolution_suggestions(self, conflicts: List[Dict[str, Any]], 
                                                order: Order) -> List[Dict[str, Any]]:
        """Generate suggestions for resolving conflicts."""
        suggestions = []
        
        for conflict in conflicts:
            if conflict["type"] == "dietary_conflict":
                suggestions.append({
                    "type": "item_substitution",
                    "description": f"Consider replacing '{conflict['item']}' with a {conflict['user_preference']} alternative",
                    "priority": "high" if conflict["severity"] == "high" else "medium"
                })
            
            elif conflict["type"] == "budget_conflict":
                suggestions.append({
                    "type": "order_optimization",
                    "description": "Consider removing or substituting expensive items to better match budget preferences",
                    "priority": "medium"
                })
            
            elif conflict["type"] == "timing_conflict":
                suggestions.append({
                    "type": "vendor_substitution",
                    "description": "Consider selecting a vendor with faster delivery times",
                    "priority": "low"
                })
        
        return suggestions
    
    def _calculate_conflict_severity(self, conflicts: List[Dict[str, Any]]) -> str:
        """Calculate overall conflict severity."""
        if not conflicts:
            return "none"
        
        high_severity = sum(1 for c in conflicts if c["severity"] == "high")
        medium_severity = sum(1 for c in conflicts if c["severity"] == "medium")
        
        if high_severity > 0:
            return "high"
        elif medium_severity > 1:
            return "medium"
        else:
            return "low"
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get statistics about optimization performance."""
        if not self.reasoning_history:
            return {"total_optimizations": 0}
        
        total_optimizations = len(self.reasoning_history)
        high_confidence = sum(1 for r in self.reasoning_history if r.confidence > 0.8)
        average_confidence = sum(r.confidence for r in self.reasoning_history) / total_optimizations
        
        # Constraint satisfaction statistics
        constraint_stats = {}
        for constraint_type in ConstraintType:
            relevant_reasonings = [r for r in self.reasoning_history 
                                 if any(c.constraint_type == constraint_type for c in r.constraints_checked)]
            if relevant_reasonings:
                satisfaction_rate = sum(1 for r in relevant_reasonings if r.confidence > 0.7) / len(relevant_reasonings)
                constraint_stats[constraint_type.value] = satisfaction_rate
        
        return {
            "total_optimizations": total_optimizations,
            "high_confidence_rate": high_confidence / total_optimizations,
            "average_confidence": average_confidence,
            "constraint_satisfaction_rates": constraint_stats,
            "optimization_patterns": self.optimization_patterns
        }
