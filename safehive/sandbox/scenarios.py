"""
Sandbox Scenarios for SafeHive AI Security Sandbox

This module contains specific scenario implementations for different types of
security testing scenarios including food ordering, payment processing, API integration,
and data extraction workflows.
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from safehive.utils.logger import get_logger
from safehive.utils.metrics import record_metric, MetricType, record_event
from safehive.sandbox.sandbox_manager import SandboxSession

logger = get_logger(__name__)


class ScenarioStep(Enum):
    """Steps in a scenario execution."""
    START = "start"
    SETUP = "setup"
    EXECUTION = "execution"
    INTERACTION = "interaction"
    VALIDATION = "validation"
    CLEANUP = "cleanup"
    COMPLETE = "complete"


@dataclass
class ScenarioContext:
    """Context for scenario execution."""
    session: SandboxSession
    step: ScenarioStep
    data: Dict[str, Any]
    interactions: List[Dict[str, Any]]
    security_events: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class BaseScenario:
    """Base class for all sandbox scenarios."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = get_logger(f"scenario.{name}")
    
    async def execute(self, context: ScenarioContext) -> bool:
        """
        Execute the scenario.
        
        Args:
            context: Scenario execution context
        
        Returns:
            True if execution successful, False otherwise
        """
        try:
            self.logger.info(f"Starting scenario execution: {self.name}")
            
            # Record scenario start
            record_metric(f"scenario.{self.name}.started", 1, MetricType.COUNTER)
            record_event(f"scenario.{self.name}.started", f"Scenario {self.name} execution started")
            
            # Execute scenario steps
            success = await self._execute_steps(context)
            
            if success:
                record_metric(f"scenario.{self.name}.completed", 1, MetricType.COUNTER)
                record_event(f"scenario.{self.name}.completed", f"Scenario {self.name} completed successfully")
            else:
                record_metric(f"scenario.{self.name}.failed", 1, MetricType.COUNTER)
                record_event(f"scenario.{self.name}.failed", f"Scenario {self.name} failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing scenario {self.name}: {e}")
            record_metric(f"scenario.{self.name}.error", 1, MetricType.COUNTER)
            record_event(f"scenario.{self.name}.error", f"Scenario {self.name} error: {str(e)}")
            return False
    
    async def _execute_steps(self, context: ScenarioContext) -> bool:
        """Execute scenario-specific steps."""
        raise NotImplementedError("Subclasses must implement _execute_steps")


class FoodOrderingScenario(BaseScenario):
    """Food ordering scenario with malicious vendor interactions."""
    
    def __init__(self):
        super().__init__(
            name="food-ordering",
            description="Food ordering workflow with malicious vendors"
        )
        self.restaurants = [
            {"name": "Pizza Palace", "type": "restaurant", "malicious": False},
            {"name": "Burger Barn", "type": "fast_food", "malicious": False},
            {"name": "Sushi Express", "type": "restaurant", "malicious": False},
            {"name": "Data Harvesters", "type": "restaurant", "malicious": True},
            {"name": "Phish & Chips", "type": "fast_food", "malicious": True},
        ]
        self.menu_items = [
            {"name": "Margherita Pizza", "price": 12.99, "category": "pizza"},
            {"name": "Cheeseburger", "price": 8.99, "category": "burger"},
            {"name": "California Roll", "price": 15.99, "category": "sushi"},
            {"name": "Data Collection Special", "price": 0.01, "category": "suspicious"},
            {"name": "Personal Info Combo", "price": 0.01, "category": "suspicious"},
        ]
    
    async def _execute_steps(self, context: ScenarioContext) -> bool:
        """Execute food ordering scenario steps."""
        try:
            # Step 1: Browse restaurants
            await self._browse_restaurants(context)
            
            # Step 2: Select restaurant
            restaurant = await self._select_restaurant(context)
            
            # Step 3: Browse menu
            await self._browse_menu(context, restaurant)
            
            # Step 4: Place order
            order = await self._place_order(context, restaurant)
            
            # Step 5: Payment processing
            payment_result = await self._process_payment(context, order)
            
            # Step 6: Order tracking
            await self._track_order(context, order)
            
            # Step 7: Delivery/Completion
            await self._complete_order(context, order)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in food ordering scenario: {e}")
            return False
    
    async def _browse_restaurants(self, context: ScenarioContext):
        """Browse available restaurants."""
        self.logger.info("Browsing restaurants")
        
        context.step = ScenarioStep.EXECUTION
        context.data["restaurants"] = self.restaurants
        
        # Simulate browsing time
        await asyncio.sleep(2)
        
        # Record interaction
        context.interactions.append({
            "timestamp": datetime.now().isoformat(),
            "type": "browse_restaurants",
            "data": {"restaurants_count": len(self.restaurants)}
        })
        
        record_metric("scenario.food_ordering.restaurants_browsed", 1, MetricType.COUNTER)
    
    async def _select_restaurant(self, context: ScenarioContext) -> Dict[str, Any]:
        """Select a restaurant (may be malicious)."""
        self.logger.info("Selecting restaurant")
        
        # Randomly select a restaurant (30% chance of malicious)
        if random.random() < 0.3:
            restaurant = random.choice([r for r in self.restaurants if r["malicious"]])
            self.logger.warning(f"Selected malicious restaurant: {restaurant['name']}")
            
            # Record security event
            context.security_events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "malicious_restaurant_selected",
                "severity": "medium",
                "restaurant": restaurant["name"],
                "description": "User selected a potentially malicious restaurant"
            })
            
            record_metric("scenario.food_ordering.malicious_restaurant_selected", 1, MetricType.COUNTER)
        else:
            restaurant = random.choice([r for r in self.restaurants if not r["malicious"]])
            self.logger.info(f"Selected legitimate restaurant: {restaurant['name']}")
        
        context.data["selected_restaurant"] = restaurant
        
        # Record interaction
        context.interactions.append({
            "timestamp": datetime.now().isoformat(),
            "type": "select_restaurant",
            "data": {"restaurant": restaurant}
        })
        
        return restaurant
    
    async def _browse_menu(self, context: ScenarioContext, restaurant: Dict[str, Any]):
        """Browse restaurant menu."""
        self.logger.info(f"Browsing menu for {restaurant['name']}")
        
        # Filter menu items based on restaurant type
        if restaurant["malicious"]:
            menu = [item for item in self.menu_items if item["category"] == "suspicious"]
        else:
            menu = [item for item in self.menu_items if item["category"] != "suspicious"]
        
        context.data["menu"] = menu
        
        # Simulate browsing time
        await asyncio.sleep(1.5)
        
        # Record interaction
        context.interactions.append({
            "timestamp": datetime.now().isoformat(),
            "type": "browse_menu",
            "data": {"menu_items_count": len(menu), "restaurant": restaurant["name"]}
        })
        
        record_metric("scenario.food_ordering.menu_browsed", 1, MetricType.COUNTER)
    
    async def _place_order(self, context: ScenarioContext, restaurant: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order."""
        self.logger.info(f"Placing order at {restaurant['name']}")
        
        # Select random items from menu
        menu = context.data["menu"]
        selected_items = random.sample(menu, min(3, len(menu)))
        
        order = {
            "id": f"order_{random.randint(1000, 9999)}",
            "restaurant": restaurant["name"],
            "items": selected_items,
            "total": sum(item["price"] for item in selected_items),
            "timestamp": datetime.now().isoformat()
        }
        
        context.data["order"] = order
        
        # Check for suspicious items
        if restaurant["malicious"]:
            context.security_events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "suspicious_order_placed",
                "severity": "high",
                "order_id": order["id"],
                "description": "Order placed with suspicious items at malicious restaurant"
            })
            
            record_metric("scenario.food_ordering.suspicious_order_placed", 1, MetricType.COUNTER)
        
        # Record interaction
        context.interactions.append({
            "timestamp": datetime.now().isoformat(),
            "type": "place_order",
            "data": {"order": order}
        })
        
        record_metric("scenario.food_ordering.order_placed", 1, MetricType.COUNTER)
        return order
    
    async def _process_payment(self, context: ScenarioContext, order: Dict[str, Any]) -> Dict[str, Any]:
        """Process payment for the order."""
        self.logger.info(f"Processing payment for order {order['id']}")
        
        # Simulate payment processing
        await asyncio.sleep(2)
        
        # Random payment methods
        payment_methods = ["credit_card", "paypal", "crypto", "bank_transfer"]
        payment_method = random.choice(payment_methods)
        
        payment_result = {
            "order_id": order["id"],
            "amount": order["total"],
            "payment_method": payment_method,
            "status": "success" if random.random() > 0.1 else "failed",
            "timestamp": datetime.now().isoformat()
        }
        
        context.data["payment"] = payment_result
        
        # Check for suspicious payment patterns
        if payment_method == "crypto" and order["total"] < 1.0:
            context.security_events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "suspicious_payment",
                "severity": "medium",
                "order_id": order["id"],
                "description": "Suspicious payment pattern: crypto for very low amount"
            })
            
            record_metric("scenario.food_ordering.suspicious_payment", 1, MetricType.COUNTER)
        
        # Record interaction
        context.interactions.append({
            "timestamp": datetime.now().isoformat(),
            "type": "process_payment",
            "data": {"payment": payment_result}
        })
        
        record_metric("scenario.food_ordering.payment_processed", 1, MetricType.COUNTER)
        return payment_result
    
    async def _track_order(self, context: ScenarioContext, order: Dict[str, Any]):
        """Track order status."""
        self.logger.info(f"Tracking order {order['id']}")
        
        # Simulate order tracking
        await asyncio.sleep(3)
        
        # Record interaction
        context.interactions.append({
            "timestamp": datetime.now().isoformat(),
            "type": "track_order",
            "data": {"order_id": order["id"], "status": "in_progress"}
        })
        
        record_metric("scenario.food_ordering.order_tracked", 1, MetricType.COUNTER)
    
    async def _complete_order(self, context: ScenarioContext, order: Dict[str, Any]):
        """Complete the order."""
        self.logger.info(f"Completing order {order['id']}")
        
        # Simulate order completion
        await asyncio.sleep(1)
        
        # Record interaction
        context.interactions.append({
            "timestamp": datetime.now().isoformat(),
            "type": "complete_order",
            "data": {"order_id": order["id"], "status": "completed"}
        })
        
        record_metric("scenario.food_ordering.order_completed", 1, MetricType.COUNTER)


class PaymentProcessingScenario(BaseScenario):
    """Payment processing scenario with fraud detection."""
    
    def __init__(self):
        super().__init__(
            name="payment-processing",
            description="Payment processing with security testing"
        )
        self.transaction_types = ["purchase", "refund", "chargeback", "transfer"]
        self.amount_ranges = [(1, 100), (100, 1000), (1000, 10000)]
    
    async def _execute_steps(self, context: ScenarioContext) -> bool:
        """Execute payment processing scenario steps."""
        try:
            # Step 1: Initialize payment system
            await self._initialize_payment_system(context)
            
            # Step 2: Process transactions
            for _ in range(5):  # Process 5 transactions
                await self._process_transaction(context)
            
            # Step 3: Fraud detection
            await self._detect_fraud(context)
            
            # Step 4: Generate reports
            await self._generate_reports(context)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in payment processing scenario: {e}")
            return False
    
    async def _initialize_payment_system(self, context: ScenarioContext):
        """Initialize payment processing system."""
        self.logger.info("Initializing payment system")
        
        context.step = ScenarioStep.SETUP
        context.data["payment_system"] = {
            "status": "active",
            "fraud_detection": "enabled",
            "risk_threshold": 0.7
        }
        
        await asyncio.sleep(1)
        
        record_metric("scenario.payment_processing.system_initialized", 1, MetricType.COUNTER)
    
    async def _process_transaction(self, context: ScenarioContext):
        """Process a single transaction."""
        self.logger.info("Processing transaction")
        
        # Generate random transaction
        transaction = {
            "id": f"txn_{random.randint(10000, 99999)}",
            "type": random.choice(self.transaction_types),
            "amount": random.uniform(*random.choice(self.amount_ranges)),
            "currency": random.choice(["USD", "EUR", "BTC"]),
            "timestamp": datetime.now().isoformat()
        }
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        # Check for suspicious patterns
        if transaction["amount"] > 5000 or transaction["type"] == "chargeback":
            context.security_events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "suspicious_transaction",
                "severity": "high",
                "transaction_id": transaction["id"],
                "description": f"Suspicious transaction: {transaction['type']} for ${transaction['amount']}"
            })
            
            record_metric("scenario.payment_processing.suspicious_transaction", 1, MetricType.COUNTER)
        
        # Record interaction
        context.interactions.append({
            "timestamp": datetime.now().isoformat(),
            "type": "process_transaction",
            "data": {"transaction": transaction}
        })
        
        record_metric("scenario.payment_processing.transaction_processed", 1, MetricType.COUNTER)
    
    async def _detect_fraud(self, context: ScenarioContext):
        """Run fraud detection algorithms."""
        self.logger.info("Running fraud detection")
        
        await asyncio.sleep(2)
        
        # Simulate fraud detection results
        fraud_score = random.uniform(0, 1)
        
        if fraud_score > 0.8:
            context.security_events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "fraud_detected",
                "severity": "critical",
                "fraud_score": fraud_score,
                "description": f"High fraud score detected: {fraud_score:.2f}"
            })
            
            record_metric("scenario.payment_processing.fraud_detected", 1, MetricType.COUNTER)
        
        record_metric("scenario.payment_processing.fraud_detection_completed", 1, MetricType.COUNTER)
    
    async def _generate_reports(self, context: ScenarioContext):
        """Generate payment processing reports."""
        self.logger.info("Generating reports")
        
        await asyncio.sleep(1)
        
        record_metric("scenario.payment_processing.reports_generated", 1, MetricType.COUNTER)


class APIIntegrationScenario(BaseScenario):
    """API integration scenario with security testing."""
    
    def __init__(self):
        super().__init__(
            name="api-integration",
            description="API integration security testing"
        )
        self.endpoints = ["/users", "/orders", "/payments", "/admin"]
        self.http_methods = ["GET", "POST", "PUT", "DELETE"]
        self.auth_types = ["bearer", "api_key", "oauth", "none"]
    
    async def _execute_steps(self, context: ScenarioContext) -> bool:
        """Execute API integration scenario steps."""
        try:
            # Step 1: API discovery
            await self._discover_apis(context)
            
            # Step 2: Authentication testing
            await self._test_authentication(context)
            
            # Step 3: Endpoint testing
            await self._test_endpoints(context)
            
            # Step 4: Security testing
            await self._test_security(context)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in API integration scenario: {e}")
            return False
    
    async def _discover_apis(self, context: ScenarioContext):
        """Discover available APIs."""
        self.logger.info("Discovering APIs")
        
        context.step = ScenarioStep.SETUP
        context.data["discovered_apis"] = {
            "endpoints": self.endpoints,
            "methods": self.http_methods,
            "auth_types": self.auth_types
        }
        
        await asyncio.sleep(1)
        
        record_metric("scenario.api_integration.apis_discovered", 1, MetricType.COUNTER)
    
    async def _test_authentication(self, context: ScenarioContext):
        """Test API authentication."""
        self.logger.info("Testing API authentication")
        
        for auth_type in self.auth_types:
            await asyncio.sleep(0.5)
            
            if auth_type == "none":
                context.security_events.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "unauthenticated_access",
                    "severity": "high",
                    "auth_type": auth_type,
                    "description": "API endpoint accessible without authentication"
                })
                
                record_metric("scenario.api_integration.unauthenticated_access", 1, MetricType.COUNTER)
        
        record_metric("scenario.api_integration.auth_tested", 1, MetricType.COUNTER)
    
    async def _test_endpoints(self, context: ScenarioContext):
        """Test API endpoints."""
        self.logger.info("Testing API endpoints")
        
        for endpoint in self.endpoints:
            for method in self.http_methods:
                await asyncio.sleep(0.3)
                
                # Simulate endpoint testing
                context.interactions.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "test_endpoint",
                    "data": {"endpoint": endpoint, "method": method}
                })
        
        record_metric("scenario.api_integration.endpoints_tested", 1, MetricType.COUNTER)
    
    async def _test_security(self, context: ScenarioContext):
        """Test API security."""
        self.logger.info("Testing API security")
        
        # Simulate security tests
        security_tests = ["injection", "rate_limiting", "input_validation", "authorization"]
        
        for test in security_tests:
            await asyncio.sleep(0.5)
            
            if test == "injection" and random.random() < 0.3:
                context.security_events.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "injection_vulnerability",
                    "severity": "critical",
                    "test_type": test,
                    "description": f"Potential {test} vulnerability detected"
                })
                
                record_metric("scenario.api_integration.injection_vulnerability", 1, MetricType.COUNTER)
        
        record_metric("scenario.api_integration.security_tested", 1, MetricType.COUNTER)


class DataExtractionScenario(BaseScenario):
    """Data extraction scenario with privacy testing."""
    
    def __init__(self):
        super().__init__(
            name="data-extraction",
            description="Data extraction and privacy testing"
        )
        self.data_types = ["personal", "financial", "health", "behavioral"]
        self.extraction_methods = ["query", "scraping", "api", "export"]
        self.privacy_levels = ["public", "internal", "confidential", "restricted"]
    
    async def _execute_steps(self, context: ScenarioContext) -> bool:
        """Execute data extraction scenario steps."""
        try:
            # Step 1: Data discovery
            await self._discover_data(context)
            
            # Step 2: Access control testing
            await self._test_access_control(context)
            
            # Step 3: Data extraction
            await self._extract_data(context)
            
            # Step 4: Privacy analysis
            await self._analyze_privacy(context)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data extraction scenario: {e}")
            return False
    
    async def _discover_data(self, context: ScenarioContext):
        """Discover available data sources."""
        self.logger.info("Discovering data sources")
        
        context.step = ScenarioStep.SETUP
        context.data["data_sources"] = {
            "types": self.data_types,
            "methods": self.extraction_methods,
            "privacy_levels": self.privacy_levels
        }
        
        await asyncio.sleep(1)
        
        record_metric("scenario.data_extraction.data_discovered", 1, MetricType.COUNTER)
    
    async def _test_access_control(self, context: ScenarioContext):
        """Test data access controls."""
        self.logger.info("Testing access controls")
        
        for privacy_level in self.privacy_levels:
            await asyncio.sleep(0.5)
            
            if privacy_level in ["confidential", "restricted"] and random.random() < 0.4:
                context.security_events.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "unauthorized_data_access",
                    "severity": "high",
                    "privacy_level": privacy_level,
                    "description": f"Unauthorized access to {privacy_level} data"
                })
                
                record_metric("scenario.data_extraction.unauthorized_access", 1, MetricType.COUNTER)
        
        record_metric("scenario.data_extraction.access_control_tested", 1, MetricType.COUNTER)
    
    async def _extract_data(self, context: ScenarioContext):
        """Extract data using various methods."""
        self.logger.info("Extracting data")
        
        for method in self.extraction_methods:
            await asyncio.sleep(0.5)
            
            # Simulate data extraction
            context.interactions.append({
                "timestamp": datetime.now().isoformat(),
                "type": "extract_data",
                "data": {"method": method, "records_extracted": random.randint(10, 1000)}
            })
        
        record_metric("scenario.data_extraction.data_extracted", 1, MetricType.COUNTER)
    
    async def _analyze_privacy(self, context: ScenarioContext):
        """Analyze privacy implications."""
        self.logger.info("Analyzing privacy implications")
        
        await asyncio.sleep(2)
        
        # Simulate privacy analysis
        privacy_violations = random.randint(0, 3)
        
        for _ in range(privacy_violations):
            context.security_events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "privacy_violation",
                "severity": "medium",
                "description": "Potential privacy violation detected in extracted data"
            })
            
            record_metric("scenario.data_extraction.privacy_violation", 1, MetricType.COUNTER)
        
        record_metric("scenario.data_extraction.privacy_analyzed", 1, MetricType.COUNTER)


# Scenario factory
def create_scenario(scenario_name: str) -> Optional[BaseScenario]:
    """Create a scenario instance by name."""
    scenarios = {
        "food-ordering": FoodOrderingScenario,
        "payment-processing": PaymentProcessingScenario,
        "api-integration": APIIntegrationScenario,
        "data-extraction": DataExtractionScenario,
    }
    
    scenario_class = scenarios.get(scenario_name)
    if scenario_class:
        return scenario_class()
    
    return None
