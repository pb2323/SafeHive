"""
Unit tests for Sandbox Scenarios

This module contains comprehensive tests for the scenario implementations
including FoodOrderingScenario, PaymentProcessingScenario, APIIntegrationScenario,
and DataExtractionScenario.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from safehive.sandbox.scenarios import (
    BaseScenario, FoodOrderingScenario, PaymentProcessingScenario,
    APIIntegrationScenario, DataExtractionScenario, ScenarioContext,
    ScenarioStep, create_scenario
)
from safehive.sandbox.sandbox_manager import SandboxSession, SandboxScenario


class TestBaseScenario:
    """Test cases for BaseScenario class."""

    def test_base_scenario_creation(self):
        """Test BaseScenario creation."""
        scenario = BaseScenario("test-scenario", "Test scenario description")
        
        assert scenario.name == "test-scenario"
        assert scenario.description == "Test scenario description"
        assert scenario.logger is not None

    @pytest.mark.asyncio
    async def test_base_scenario_execute_not_implemented(self):
        """Test that BaseScenario.execute raises NotImplementedError."""
        scenario = BaseScenario("test-scenario", "Test scenario description")
        
        # Create a mock context
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        with pytest.raises(NotImplementedError):
            await scenario.execute(context)


class TestFoodOrderingScenario:
    """Test cases for FoodOrderingScenario class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scenario = FoodOrderingScenario()

    def test_food_ordering_scenario_initialization(self):
        """Test FoodOrderingScenario initialization."""
        assert self.scenario.name == "food-ordering"
        assert self.scenario.description == "Food ordering workflow with malicious vendors"
        assert len(self.scenario.restaurants) == 5
        assert len(self.scenario.menu_items) == 5
        
        # Check for malicious restaurants
        malicious_restaurants = [r for r in self.scenario.restaurants if r["malicious"]]
        assert len(malicious_restaurants) == 2
        
        # Check for suspicious menu items
        suspicious_items = [item for item in self.scenario.menu_items if item["category"] == "suspicious"]
        assert len(suspicious_items) == 2

    @pytest.mark.asyncio
    async def test_food_ordering_scenario_execution(self):
        """Test FoodOrderingScenario execution."""
        # Create mock session
        mock_session = Mock(spec=SandboxSession)
        mock_session.scenario = Mock()
        mock_session.scenario.name = "food-ordering"
        
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        # Execute scenario
        success = await self.scenario.execute(context)
        
        assert success is True
        assert len(context.interactions) > 0
        assert "restaurants" in context.data
        assert "selected_restaurant" in context.data
        assert "menu" in context.data
        assert "order" in context.data
        assert "payment" in context.data

    @pytest.mark.asyncio
    async def test_browse_restaurants(self):
        """Test browsing restaurants."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._browse_restaurants(context)
        
        assert "restaurants" in context.data
        assert len(context.data["restaurants"]) == 5
        assert len(context.interactions) == 1
        assert context.interactions[0]["type"] == "browse_restaurants"

    @pytest.mark.asyncio
    async def test_select_restaurant(self):
        """Test selecting a restaurant."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={"restaurants": self.scenario.restaurants},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        restaurant = await self.scenario._select_restaurant(context)
        
        assert restaurant is not None
        assert restaurant in self.scenario.restaurants
        assert "selected_restaurant" in context.data
        assert len(context.interactions) == 1
        assert context.interactions[0]["type"] == "select_restaurant"

    @pytest.mark.asyncio
    async def test_browse_menu(self):
        """Test browsing menu."""
        mock_session = Mock(spec=SandboxSession)
        restaurant = {"name": "Test Restaurant", "malicious": False}
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._browse_menu(context, restaurant)
        
        assert "menu" in context.data
        assert len(context.interactions) == 1
        assert context.interactions[0]["type"] == "browse_menu"

    @pytest.mark.asyncio
    async def test_place_order(self):
        """Test placing an order."""
        mock_session = Mock(spec=SandboxSession)
        restaurant = {"name": "Test Restaurant", "malicious": False}
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={"menu": self.scenario.menu_items},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        order = await self.scenario._place_order(context, restaurant)
        
        assert order is not None
        assert "id" in order
        assert "restaurant" in order
        assert "items" in order
        assert "total" in order
        assert "order" in context.data
        assert len(context.interactions) == 1
        assert context.interactions[0]["type"] == "place_order"

    @pytest.mark.asyncio
    async def test_process_payment(self):
        """Test processing payment."""
        mock_session = Mock(spec=SandboxSession)
        order = {
            "id": "test-order-123",
            "restaurant": "Test Restaurant",
            "total": 25.99
        }
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        payment_result = await self.scenario._process_payment(context, order)
        
        assert payment_result is not None
        assert "order_id" in payment_result
        assert "amount" in payment_result
        assert "payment_method" in payment_result
        assert "status" in payment_result
        assert "payment" in context.data
        assert len(context.interactions) == 1
        assert context.interactions[0]["type"] == "process_payment"

    @pytest.mark.asyncio
    async def test_track_order(self):
        """Test tracking order."""
        mock_session = Mock(spec=SandboxSession)
        order = {"id": "test-order-123"}
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._track_order(context, order)
        
        assert len(context.interactions) == 1
        assert context.interactions[0]["type"] == "track_order"

    @pytest.mark.asyncio
    async def test_complete_order(self):
        """Test completing order."""
        mock_session = Mock(spec=SandboxSession)
        order = {"id": "test-order-123"}
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._complete_order(context, order)
        
        assert len(context.interactions) == 1
        assert context.interactions[0]["type"] == "complete_order"

    @pytest.mark.asyncio
    async def test_malicious_restaurant_detection(self):
        """Test detection of malicious restaurants."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={"restaurants": self.scenario.restaurants},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        # Mock random selection to always pick malicious restaurant
        with patch('random.random', return_value=0.1):  # < 0.3 threshold
            restaurant = await self.scenario._select_restaurant(context)
            
            if restaurant["malicious"]:
                assert len(context.security_events) == 1
                assert context.security_events[0]["type"] == "malicious_restaurant_selected"
                assert context.security_events[0]["severity"] == "medium"


class TestPaymentProcessingScenario:
    """Test cases for PaymentProcessingScenario class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scenario = PaymentProcessingScenario()

    def test_payment_processing_scenario_initialization(self):
        """Test PaymentProcessingScenario initialization."""
        assert self.scenario.name == "payment-processing"
        assert self.scenario.description == "Payment processing with security testing"
        assert len(self.scenario.transaction_types) == 4
        assert len(self.scenario.amount_ranges) == 3

    @pytest.mark.asyncio
    async def test_payment_processing_scenario_execution(self):
        """Test PaymentProcessingScenario execution."""
        mock_session = Mock(spec=SandboxSession)
        mock_session.scenario = Mock()
        mock_session.scenario.name = "payment-processing"
        
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        success = await self.scenario.execute(context)
        
        assert success is True
        assert len(context.interactions) > 0
        assert "payment_system" in context.data

    @pytest.mark.asyncio
    async def test_initialize_payment_system(self):
        """Test payment system initialization."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._initialize_payment_system(context)
        
        assert "payment_system" in context.data
        assert context.data["payment_system"]["status"] == "active"
        assert context.data["payment_system"]["fraud_detection"] == "enabled"

    @pytest.mark.asyncio
    async def test_process_transaction(self):
        """Test transaction processing."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._process_transaction(context)
        
        assert len(context.interactions) == 1
        assert context.interactions[0]["type"] == "process_transaction"
        assert "transaction" in context.interactions[0]["data"]

    @pytest.mark.asyncio
    async def test_detect_fraud(self):
        """Test fraud detection."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._detect_fraud(context)
        
        # May or may not detect fraud depending on random values
        # Just ensure the method completes without error
        assert True

    @pytest.mark.asyncio
    async def test_generate_reports(self):
        """Test report generation."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._generate_reports(context)
        
        # Just ensure the method completes without error
        assert True


class TestAPIIntegrationScenario:
    """Test cases for APIIntegrationScenario class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scenario = APIIntegrationScenario()

    def test_api_integration_scenario_initialization(self):
        """Test APIIntegrationScenario initialization."""
        assert self.scenario.name == "api-integration"
        assert self.scenario.description == "API integration security testing"
        assert len(self.scenario.endpoints) == 4
        assert len(self.scenario.http_methods) == 4
        assert len(self.scenario.auth_types) == 4

    @pytest.mark.asyncio
    async def test_api_integration_scenario_execution(self):
        """Test APIIntegrationScenario execution."""
        mock_session = Mock(spec=SandboxSession)
        mock_session.scenario = Mock()
        mock_session.scenario.name = "api-integration"
        
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        success = await self.scenario.execute(context)
        
        assert success is True
        assert len(context.interactions) > 0
        assert "discovered_apis" in context.data

    @pytest.mark.asyncio
    async def test_discover_apis(self):
        """Test API discovery."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._discover_apis(context)
        
        assert "discovered_apis" in context.data
        assert "endpoints" in context.data["discovered_apis"]
        assert "methods" in context.data["discovered_apis"]
        assert "auth_types" in context.data["discovered_apis"]

    @pytest.mark.asyncio
    async def test_test_authentication(self):
        """Test authentication testing."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._test_authentication(context)
        
        # Should detect unauthenticated access
        assert len(context.security_events) == 1
        assert context.security_events[0]["type"] == "unauthenticated_access"
        assert context.security_events[0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_test_endpoints(self):
        """Test endpoint testing."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._test_endpoints(context)
        
        # Should test all endpoint/method combinations
        expected_interactions = len(self.scenario.endpoints) * len(self.scenario.http_methods)
        assert len(context.interactions) == expected_interactions
        
        for interaction in context.interactions:
            assert interaction["type"] == "test_endpoint"
            assert "endpoint" in interaction["data"]
            assert "method" in interaction["data"]

    @pytest.mark.asyncio
    async def test_test_security(self):
        """Test security testing."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._test_security(context)
        
        # May or may not detect vulnerabilities depending on random values
        # Just ensure the method completes without error
        assert True


class TestDataExtractionScenario:
    """Test cases for DataExtractionScenario class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scenario = DataExtractionScenario()

    def test_data_extraction_scenario_initialization(self):
        """Test DataExtractionScenario initialization."""
        assert self.scenario.name == "data-extraction"
        assert self.scenario.description == "Data extraction and privacy testing"
        assert len(self.scenario.data_types) == 4
        assert len(self.scenario.extraction_methods) == 4
        assert len(self.scenario.privacy_levels) == 4

    @pytest.mark.asyncio
    async def test_data_extraction_scenario_execution(self):
        """Test DataExtractionScenario execution."""
        mock_session = Mock(spec=SandboxSession)
        mock_session.scenario = Mock()
        mock_session.scenario.name = "data-extraction"
        
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        success = await self.scenario.execute(context)
        
        assert success is True
        assert len(context.interactions) > 0
        assert "data_sources" in context.data

    @pytest.mark.asyncio
    async def test_discover_data(self):
        """Test data discovery."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._discover_data(context)
        
        assert "data_sources" in context.data
        assert "types" in context.data["data_sources"]
        assert "methods" in context.data["data_sources"]
        assert "privacy_levels" in context.data["data_sources"]

    @pytest.mark.asyncio
    async def test_test_access_control(self):
        """Test access control testing."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._test_access_control(context)
        
        # May detect unauthorized access depending on random values
        # Just ensure the method completes without error
        assert True

    @pytest.mark.asyncio
    async def test_extract_data(self):
        """Test data extraction."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._extract_data(context)
        
        # Should test all extraction methods
        assert len(context.interactions) == len(self.scenario.extraction_methods)
        
        for interaction in context.interactions:
            assert interaction["type"] == "extract_data"
            assert "method" in interaction["data"]
            assert "records_extracted" in interaction["data"]

    @pytest.mark.asyncio
    async def test_analyze_privacy(self):
        """Test privacy analysis."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        await self.scenario._analyze_privacy(context)
        
        # May detect privacy violations depending on random values
        # Just ensure the method completes without error
        assert True


class TestScenarioFactory:
    """Test cases for scenario factory function."""

    def test_create_scenario_valid(self):
        """Test creating valid scenarios."""
        scenarios = ["food-ordering", "payment-processing", "api-integration", "data-extraction"]
        
        for scenario_name in scenarios:
            scenario = create_scenario(scenario_name)
            assert scenario is not None
            assert scenario.name == scenario_name

    def test_create_scenario_invalid(self):
        """Test creating invalid scenario."""
        scenario = create_scenario("nonexistent-scenario")
        assert scenario is None

    def test_create_scenario_types(self):
        """Test that created scenarios are of correct types."""
        food_scenario = create_scenario("food-ordering")
        assert isinstance(food_scenario, FoodOrderingScenario)
        
        payment_scenario = create_scenario("payment-processing")
        assert isinstance(payment_scenario, PaymentProcessingScenario)
        
        api_scenario = create_scenario("api-integration")
        assert isinstance(api_scenario, APIIntegrationScenario)
        
        data_scenario = create_scenario("data-extraction")
        assert isinstance(data_scenario, DataExtractionScenario)


class TestScenarioContext:
    """Test cases for ScenarioContext class."""

    def test_scenario_context_creation(self):
        """Test ScenarioContext creation."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={"test": "value"},
            interactions=[{"type": "test"}],
            security_events=[{"type": "security"}],
            metrics={"metric": 42}
        )
        
        assert context.session == mock_session
        assert context.step == ScenarioStep.START
        assert context.data == {"test": "value"}
        assert context.interactions == [{"type": "test"}]
        assert context.security_events == [{"type": "security"}]
        assert context.metrics == {"metric": 42}

    def test_scenario_context_defaults(self):
        """Test ScenarioContext with default values."""
        mock_session = Mock(spec=SandboxSession)
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        assert context.data == {}
        assert context.interactions == []
        assert context.security_events == []
        assert context.metrics == {}


class TestScenarioIntegration:
    """Integration tests for scenarios."""

    @pytest.mark.asyncio
    async def test_scenario_metrics_recording(self):
        """Test that scenarios record metrics properly."""
        scenario = FoodOrderingScenario()
        mock_session = Mock(spec=SandboxSession)
        mock_session.scenario = Mock()
        mock_session.scenario.name = "food-ordering"
        
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        # Mock the metrics recording
        with patch('safehive.sandbox.scenarios.record_metric') as mock_record_metric:
            with patch('safehive.sandbox.scenarios.record_event') as mock_record_event:
                success = await scenario.execute(context)
                
                assert success is True
                # Should have recorded various metrics
                assert mock_record_metric.call_count > 0
                assert mock_record_event.call_count > 0

    @pytest.mark.asyncio
    async def test_scenario_error_handling(self):
        """Test scenario error handling."""
        scenario = FoodOrderingScenario()
        mock_session = Mock(spec=SandboxSession)
        mock_session.scenario = Mock()
        mock_session.scenario.name = "food-ordering"
        
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        # Mock an error in one of the steps
        with patch.object(scenario, '_browse_restaurants', side_effect=Exception("Test error")):
            success = await scenario.execute(context)
            
            assert success is False

    @pytest.mark.asyncio
    async def test_scenario_interaction_tracking(self):
        """Test that scenarios properly track interactions."""
        scenario = FoodOrderingScenario()
        mock_session = Mock(spec=SandboxSession)
        mock_session.scenario = Mock()
        mock_session.scenario.name = "food-ordering"
        
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        success = await scenario.execute(context)
        
        assert success is True
        assert len(context.interactions) > 0
        
        # Check that interactions have proper structure
        for interaction in context.interactions:
            assert "timestamp" in interaction
            assert "type" in interaction
            assert "data" in interaction

    @pytest.mark.asyncio
    async def test_scenario_security_event_tracking(self):
        """Test that scenarios properly track security events."""
        scenario = FoodOrderingScenario()
        mock_session = Mock(spec=SandboxSession)
        mock_session.scenario = Mock()
        mock_session.scenario.name = "food-ordering"
        
        context = ScenarioContext(
            session=mock_session,
            step=ScenarioStep.START,
            data={},
            interactions=[],
            security_events=[],
            metrics={}
        )
        
        # Mock random selection to trigger security events
        with patch('random.random', return_value=0.1):  # Always trigger malicious selection
            success = await scenario.execute(context)
            
            assert success is True
            # Should have some security events
            assert len(context.security_events) > 0
            
            # Check security event structure
            for event in context.security_events:
                assert "timestamp" in event
                assert "type" in event
                assert "severity" in event
                assert "description" in event
