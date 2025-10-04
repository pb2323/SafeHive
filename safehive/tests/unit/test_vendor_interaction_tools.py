"""
Unit tests for vendor interaction tools.
"""

import pytest
import json
from unittest.mock import patch, Mock

from safehive.tools.vendor_interaction_tools import (
    communicate_with_vendor, negotiate_pricing, check_availability, rate_vendor, get_vendor_info,
    VendorCommunicationTool, PricingNegotiationTool, AvailabilityCheckTool, VendorRatingTool, VendorInfoTool,
    VendorCommunicationInput, PricingNegotiationInput, AvailabilityCheckInput, VendorRatingInput, VendorInfoInput
)


class TestVendorInteractionTools:
    """Test vendor interaction tool functions."""

    def test_communicate_with_vendor_success(self):
        """Test successful vendor communication."""
        result = communicate_with_vendor("vendor_1", "Hello, I'd like to order a pizza", "inquiry")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "Received response from" in result_data["message"]
        assert result_data["data"]["vendor_name"] == "Pizza Palace"
        assert "vendor_response" in result_data["data"]

    def test_communicate_with_vendor_not_found(self):
        """Test vendor communication with non-existent vendor."""
        result = communicate_with_vendor("nonexistent_vendor", "Hello", "general")
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "not found" in result_data["message"]

    def test_communicate_with_malicious_vendor(self):
        """Test communication with malicious vendor."""
        result = communicate_with_vendor("vendor_3", "I'd like to place an order", "general")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "malicious_indicators" in result_data["data"]
        assert len(result_data["data"]["malicious_indicators"]) > 0

    def test_negotiate_pricing_success(self):
        """Test successful pricing negotiation."""
        result = negotiate_pricing("vendor_1", "pizza_1", 10.0, 1, "Bulk order discount")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "Negotiation completed" in result_data["message"]
        assert "negotiation_result" in result_data["data"]
        assert "final_price" in result_data["data"]

    def test_negotiate_pricing_vendor_not_found(self):
        """Test pricing negotiation with non-existent vendor."""
        result = negotiate_pricing("nonexistent_vendor", "item_1", 10.0, 1, "Discount request")
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "not found" in result_data["message"]

    def test_negotiate_pricing_suspicious_vendor(self):
        """Test pricing negotiation with suspicious vendor."""
        result = negotiate_pricing("vendor_3", "item_1", 5.0, 1, "Very low price request")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "suspicious_indicators" in result_data["data"]
        assert len(result_data["data"]["suspicious_indicators"]) > 0

    def test_check_availability_success(self):
        """Test successful availability check."""
        result = check_availability("vendor_1", ["pizza_1", "pizza_2"], "immediate")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "Availability check completed" in result_data["message"]
        assert "items" in result_data["data"]
        assert "pizza_1" in result_data["data"]["items"]

    def test_check_availability_vendor_not_found(self):
        """Test availability check with non-existent vendor."""
        result = check_availability("nonexistent_vendor", ["item_1"], "immediate")
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "not found" in result_data["message"]

    def test_rate_vendor_success(self):
        """Test successful vendor rating."""
        result = rate_vendor("vendor_1", 5, "Excellent service!", "order_123")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "Rating submitted successfully" in result_data["message"]
        assert "review_id" in result_data["data"]
        assert result_data["data"]["rating"] == 5

    def test_rate_vendor_invalid_rating(self):
        """Test vendor rating with invalid rating value."""
        result = rate_vendor("vendor_1", 6, "Great service!")
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "Rating must be between 1 and 5" in result_data["message"]

    def test_rate_vendor_not_found(self):
        """Test vendor rating with non-existent vendor."""
        result = rate_vendor("nonexistent_vendor", 4, "Good service")
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "not found" in result_data["message"]

    def test_get_vendor_info_basic(self):
        """Test getting basic vendor information."""
        result = get_vendor_info("vendor_1", "basic")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "Vendor information retrieved" in result_data["message"]
        assert result_data["data"]["vendor_info"]["name"] == "Pizza Palace"
        assert "rating" in result_data["data"]["vendor_info"]

    def test_get_vendor_info_detailed(self):
        """Test getting detailed vendor information."""
        result = get_vendor_info("vendor_1", "detailed")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "reviews_count" in result_data["data"]["vendor_info"]

    def test_get_vendor_info_reviews(self):
        """Test getting vendor reviews."""
        result = get_vendor_info("vendor_1", "reviews")
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "reviews" in result_data["data"]["vendor_info"]
        assert "average_rating" in result_data["data"]["vendor_info"]

    def test_get_vendor_info_not_found(self):
        """Test getting info for non-existent vendor."""
        result = get_vendor_info("nonexistent_vendor")
        result_data = json.loads(result)
        
        assert result_data["success"] is False
        assert "not found" in result_data["message"]


class TestVendorInteractionToolClasses:
    """Test vendor interaction tool classes."""

    def test_vendor_communication_tool(self):
        """Test VendorCommunicationTool class."""
        tool = VendorCommunicationTool()
        input_data = VendorCommunicationInput(
            vendor_id="vendor_1",
            message="Hello, I need help with my order",
            message_type="inquiry",
            priority="normal"
        )
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["vendor_name"] == "Pizza Palace"

    def test_pricing_negotiation_tool(self):
        """Test PricingNegotiationTool class."""
        tool = PricingNegotiationTool()
        input_data = PricingNegotiationInput(
            vendor_id="vendor_1",
            item_id="pizza_1",
            requested_price=12.0,
            quantity=2,
            justification="Bulk order discount"
        )
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "negotiation_result" in result_data["data"]

    def test_availability_check_tool(self):
        """Test AvailabilityCheckTool class."""
        tool = AvailabilityCheckTool()
        input_data = AvailabilityCheckInput(
            vendor_id="vendor_1",
            items=["pizza_1", "pizza_2"],
            timeframe="immediate"
        )
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "items" in result_data["data"]

    def test_vendor_rating_tool(self):
        """Test VendorRatingTool class."""
        tool = VendorRatingTool()
        input_data = VendorRatingInput(
            vendor_id="vendor_1",
            rating=4,
            feedback="Good service and fast delivery",
            order_id="test_order_123"
        )
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert "review_id" in result_data["data"]

    def test_vendor_info_tool(self):
        """Test VendorInfoTool class."""
        tool = VendorInfoTool()
        input_data = VendorInfoInput(
            vendor_id="vendor_1",
            info_type="basic"
        )
        
        result = tool._execute(input_data)
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["data"]["vendor_info"]["name"] == "Pizza Palace"


class TestVendorInteractionIntegration:
    """Test integration between vendor interaction tools."""

    def test_vendor_discovery_and_rating_workflow(self):
        """Test complete vendor discovery and rating workflow."""
        # 1. Get vendor information
        info_result = get_vendor_info("vendor_1", "basic")
        info_data = json.loads(info_result)
        assert info_data["success"] is True
        
        # 2. Communicate with vendor
        comm_result = communicate_with_vendor("vendor_1", "I'd like to know your specialties", "inquiry")
        comm_data = json.loads(comm_result)
        assert comm_data["success"] is True
        
        # 3. Check availability
        avail_result = check_availability("vendor_1", ["pizza_1"], "immediate")
        avail_data = json.loads(avail_result)
        assert avail_data["success"] is True
        
        # 4. Negotiate pricing
        neg_result = negotiate_pricing("vendor_1", "pizza_1", 11.0, 1, "Regular customer")
        neg_data = json.loads(neg_result)
        assert neg_data["success"] is True
        
        # 5. Rate the vendor
        rating_result = rate_vendor("vendor_1", 5, "Excellent negotiation and communication", "order_456")
        rating_data = json.loads(rating_result)
        assert rating_data["success"] is True

    def test_malicious_vendor_detection_workflow(self):
        """Test detection of malicious vendor behavior."""
        # 1. Get vendor information (should show malicious indicators)
        info_result = get_vendor_info("vendor_3", "detailed")
        info_data = json.loads(info_result)
        assert info_data["success"] is True
        
        # 2. Communicate with malicious vendor
        comm_result = communicate_with_vendor("vendor_3", "I need help with my order", "inquiry")
        comm_data = json.loads(comm_result)
        assert comm_data["success"] is True
        assert len(comm_data["data"]["malicious_indicators"]) > 0
        
        # 3. Try pricing negotiation (should trigger suspicious indicators)
        neg_result = negotiate_pricing("vendor_3", "item_1", 3.0, 1, "Very low price")
        neg_data = json.loads(neg_result)
        assert neg_data["success"] is True
        assert len(neg_data["data"]["suspicious_indicators"]) > 0

    def test_vendor_rating_impact_on_reputation(self):
        """Test how vendor ratings impact reputation over time."""
        # Get initial rating
        initial_info = get_vendor_info("vendor_2", "basic")
        initial_data = json.loads(initial_info)
        initial_rating = initial_data["data"]["vendor_info"]["rating"]
        
        # Submit a good rating
        rating_result = rate_vendor("vendor_2", 5, "Excellent service improvement!", "order_789")
        rating_data = json.loads(rating_result)
        assert rating_data["success"] is True
        
        # Check if rating improved
        new_info = get_vendor_info("vendor_2", "basic")
        new_data = json.loads(new_info)
        new_rating = new_data["data"]["vendor_info"]["rating"]
        
        # Rating should have improved or stayed the same
        assert new_rating >= initial_rating
