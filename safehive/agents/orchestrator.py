"""
Orchestrator Agent Implementation

This module implements an OrchestratorAgent that acts as an AI assistant
for coordinating food ordering workflows, managing vendor communications,
and orchestrating the overall food ordering process in the SafeHive AI Security Sandbox.
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
from .user_twin import UserTwinAgent, PreferenceCategory

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


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PREPARING = "preparing"
    READY = "ready"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"  # Added this status


class OrderType(Enum):
    """Order type enumeration."""
    DINE_IN = "dine_in"
    TAKEAWAY = "takeaway"
    DELIVERY = "delivery"
    PICKUP = "pickup"


class PaymentStatus(Enum):
    """Payment status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


@dataclass
class OrderItem:
    """Individual order item."""
    item_id: str
    name: str
    quantity: int
    unit_price: float
    total_price: float
    special_instructions: str = ""
    dietary_requirements: List[str] = field(default_factory=list)
    allergens: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "name": self.name,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "total_price": self.total_price,
            "special_instructions": self.special_instructions,
            "dietary_requirements": self.dietary_requirements,
            "allergens": self.allergens
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderItem":
        """Create from dictionary."""
        return cls(
            item_id=data["item_id"],
            name=data["name"],
            quantity=data["quantity"],
            unit_price=data["unit_price"],
            total_price=data["total_price"],
            special_instructions=data.get("special_instructions", ""),
            dietary_requirements=data.get("dietary_requirements", []),
            allergens=data.get("allergens", [])
        )


@dataclass
class Vendor:
    """Vendor information."""
    vendor_id: str
    name: str
    cuisine_type: str
    rating: float
    delivery_time_minutes: int
    minimum_order: float
    delivery_fee: float
    is_available: bool = True
    specialties: List[str] = field(default_factory=list)
    contact_info: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "vendor_id": self.vendor_id,
            "name": self.name,
            "cuisine_type": self.cuisine_type,
            "rating": self.rating,
            "delivery_time_minutes": self.delivery_time_minutes,
            "minimum_order": self.minimum_order,
            "delivery_fee": self.delivery_fee,
            "is_available": self.is_available,
            "specialties": self.specialties,
            "contact_info": self.contact_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Vendor":
        """Create from dictionary."""
        return cls(
            vendor_id=data["vendor_id"],
            name=data["name"],
            cuisine_type=data["cuisine_type"],
            rating=data["rating"],
            delivery_time_minutes=data["delivery_time_minutes"],
            minimum_order=data["minimum_order"],
            delivery_fee=data["delivery_fee"],
            is_available=data.get("is_available", True),
            specialties=data.get("specialties", []),
            contact_info=data.get("contact_info", {})
        )


@dataclass
class Order:
    """Complete order information."""
    order_id: str
    user_id: str
    vendor: Vendor
    items: List[OrderItem]
    order_type: OrderType
    status: OrderStatus
    payment_status: PaymentStatus
    total_amount: float
    delivery_address: Optional[str] = None
    special_instructions: str = ""
    estimated_delivery_time: Optional[datetime] = None
    actual_delivery_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "order_id": self.order_id,
            "user_id": self.user_id,
            "vendor": self.vendor.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "order_type": self.order_type.value,
            "status": self.status.value,
            "payment_status": self.payment_status.value,
            "total_amount": self.total_amount,
            "delivery_address": self.delivery_address,
            "special_instructions": self.special_instructions,
            "estimated_delivery_time": self.estimated_delivery_time.isoformat() if self.estimated_delivery_time else None,
            "actual_delivery_time": self.actual_delivery_time.isoformat() if self.actual_delivery_time else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        """Create from dictionary."""
        return cls(
            order_id=data["order_id"],
            user_id=data["user_id"],
            vendor=Vendor.from_dict(data["vendor"]),
            items=[OrderItem.from_dict(item) for item in data["items"]],
            order_type=OrderType(data["order_type"]),
            status=OrderStatus(data["status"]),
            payment_status=PaymentStatus(data["payment_status"]),
            total_amount=data["total_amount"],
            delivery_address=data.get("delivery_address"),
            special_instructions=data.get("special_instructions", ""),
            estimated_delivery_time=datetime.fromisoformat(data["estimated_delivery_time"]) if data.get("estimated_delivery_time") else None,
            actual_delivery_time=datetime.fromisoformat(data["actual_delivery_time"]) if data.get("actual_delivery_time") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {})
        )


class OrderManager:
    """Manages orders and order-related operations."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_orders"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._orders: Dict[str, Order] = {}
        self._order_history: List[Dict[str, Any]] = []
        self._load_orders()
    
    def _load_orders(self) -> None:
        """Load orders from storage."""
        orders_file = self.storage_path / "orders.json"
        if orders_file.exists():
            try:
                with open(orders_file, 'r') as f:
                    data = json.load(f)
                    for order_id, order_data in data.items():
                        self._orders[order_id] = Order.from_dict(order_data)
                logger.info(f"Loaded {len(self._orders)} orders")
            except Exception as e:
                logger.error(f"Failed to load orders: {e}")
    
    def _save_orders(self) -> None:
        """Save orders to storage."""
        orders_file = self.storage_path / "orders.json"
        try:
            data = {order_id: order.to_dict() for order_id, order in self._orders.items()}
            with open(orders_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved orders")
        except Exception as e:
            logger.error(f"Failed to save orders: {e}")
    
    def create_order(self, order: Order) -> bool:
        """Create a new order."""
        try:
            self._orders[order.order_id] = order
            
            # Record order history
            self._order_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "create",
                "order_id": order.order_id,
                "user_id": order.user_id,
                "vendor_id": order.vendor.vendor_id,
                "total_amount": order.total_amount,
                "status": order.status.value
            })
            
            self._save_orders()
            logger.info(f"Created order: {order.order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID."""
        return self._orders.get(order_id)
    
    def update_order_status(self, order_id: str, new_status: OrderStatus) -> bool:
        """Update order status."""
        order = self._orders.get(order_id)
        if order:
            old_status = order.status
            order.status = new_status
            order.updated_at = datetime.now()
            
            # Record status change
            self._order_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "status_update",
                "order_id": order_id,
                "old_status": old_status.value,
                "new_status": new_status.value
            })
            
            self._save_orders()
            logger.info(f"Updated order {order_id} status: {old_status.value} -> {new_status.value}")
            return True
        return False
    
    def update_payment_status(self, order_id: str, new_payment_status: PaymentStatus) -> bool:
        """Update payment status."""
        order = self._orders.get(order_id)
        if order:
            old_status = order.payment_status
            order.payment_status = new_payment_status
            order.updated_at = datetime.now()
            
            # Record payment status change
            self._order_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "payment_update",
                "order_id": order_id,
                "old_status": old_status.value,
                "new_status": new_payment_status.value
            })
            
            self._save_orders()
            logger.info(f"Updated order {order_id} payment status: {old_status.value} -> {new_payment_status.value}")
            return True
        return False
    
    def get_orders_by_user(self, user_id: str) -> List[Order]:
        """Get all orders for a user."""
        return [order for order in self._orders.values() if order.user_id == user_id]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get all orders with a specific status."""
        return [order for order in self._orders.values() if order.status == status]
    
    def get_order_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get order history."""
        return self._order_history[-limit:] if self._order_history else []
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order statistics."""
        if not self._orders:
            return {"total_orders": 0}
        
        total_orders = len(self._orders)
        status_counts = {}
        payment_status_counts = {}
        total_revenue = 0.0
        completed_orders_count = 0
        
        for order in self._orders.values():
            # Status counts
            status = order.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Payment status counts
            payment_status = order.payment_status.value
            payment_status_counts[payment_status] = payment_status_counts.get(payment_status, 0) + 1
            
            # Revenue (only completed orders)
            if order.payment_status == PaymentStatus.COMPLETED:
                total_revenue += order.total_amount
                completed_orders_count += 1
        
        return {
            "total_orders": total_orders,
            "status_counts": status_counts,
            "payment_status_counts": payment_status_counts,
            "total_revenue": total_revenue,
            "average_order_value": total_revenue / completed_orders_count if completed_orders_count > 0 else 0.0
        }


class VendorManager:
    """Manages vendor information and operations."""
    
    def __init__(self, storage_path: str = "/tmp/safehive_vendors"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._vendors: Dict[str, Vendor] = {}
        self._load_vendors()
        self._initialize_default_vendors()
    
    def _load_vendors(self) -> None:
        """Load vendors from storage."""
        vendors_file = self.storage_path / "vendors.json"
        if vendors_file.exists():
            try:
                with open(vendors_file, 'r') as f:
                    data = json.load(f)
                    for vendor_id, vendor_data in data.items():
                        self._vendors[vendor_id] = Vendor.from_dict(vendor_data)
                logger.info(f"Loaded {len(self._vendors)} vendors")
            except Exception as e:
                logger.error(f"Failed to load vendors: {e}")
    
    def _save_vendors(self) -> None:
        """Save vendors to storage."""
        vendors_file = self.storage_path / "vendors.json"
        try:
            data = {vendor_id: vendor.to_dict() for vendor_id, vendor in self._vendors.items()}
            with open(vendors_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved vendors")
        except Exception as e:
            logger.error(f"Failed to save vendors: {e}")
    
    def _initialize_default_vendors(self) -> None:
        """Initialize default vendors if none exist."""
        if not self._vendors:
            default_vendors = [
                Vendor(
                    vendor_id="vendor_001",
                    name="Mario's Italian Bistro",
                    cuisine_type="italian",
                    rating=4.8,
                    delivery_time_minutes=25,
                    minimum_order=15.0,
                    delivery_fee=3.0,
                    specialties=["pizza", "pasta", "risotto"],
                    contact_info={"phone": "+1-555-0123", "email": "orders@marios.com"}
                ),
                Vendor(
                    vendor_id="vendor_002",
                    name="Spicy Thai Palace",
                    cuisine_type="thai",
                    rating=4.6,
                    delivery_time_minutes=20,
                    minimum_order=12.0,
                    delivery_fee=2.5,
                    specialties=["curry", "pad thai", "tom yum"],
                    contact_info={"phone": "+1-555-0124", "email": "orders@spicythai.com"}
                ),
                Vendor(
                    vendor_id="vendor_003",
                    name="Burger Express",
                    cuisine_type="american",
                    rating=4.4,
                    delivery_time_minutes=15,
                    minimum_order=10.0,
                    delivery_fee=2.0,
                    specialties=["burgers", "fries", "milkshakes"],
                    contact_info={"phone": "+1-555-0125", "email": "orders@burgerexpress.com"}
                ),
                Vendor(
                    vendor_id="vendor_004",
                    name="Sushi Zen",
                    cuisine_type="japanese",
                    rating=4.9,
                    delivery_time_minutes=30,
                    minimum_order=20.0,
                    delivery_fee=4.0,
                    specialties=["sushi", "sashimi", "ramen"],
                    contact_info={"phone": "+1-555-0126", "email": "orders@sushizen.com"}
                )
            ]
            
            for vendor in default_vendors:
                self._vendors[vendor.vendor_id] = vendor
            
            self._save_vendors()
            logger.info(f"Initialized {len(default_vendors)} default vendors")
    
    def get_vendor(self, vendor_id: str) -> Optional[Vendor]:
        """Get vendor by ID."""
        return self._vendors.get(vendor_id)
    
    def get_vendors_by_cuisine(self, cuisine_type: str) -> List[Vendor]:
        """Get vendors by cuisine type."""
        return [vendor for vendor in self._vendors.values() 
                if vendor.cuisine_type.lower() == cuisine_type.lower() and vendor.is_available]
    
    def get_all_vendors(self) -> List[Vendor]:
        """Get all available vendors."""
        return [vendor for vendor in self._vendors.values() if vendor.is_available]
    
    def update_vendor_availability(self, vendor_id: str, is_available: bool) -> bool:
        """Update vendor availability."""
        vendor = self._vendors.get(vendor_id)
        if vendor:
            vendor.is_available = is_available
            self._save_vendors()
            logger.info(f"Updated vendor {vendor_id} availability: {is_available}")
            return True
        return False
    
    def search_vendors(self, query: str) -> List[Vendor]:
        """Search vendors by name, cuisine, or specialties."""
        query_lower = query.lower()
        results = []
        
        for vendor in self._vendors.values():
            if not vendor.is_available:
                continue
                
            # Search in name, cuisine, and specialties
            if (query_lower in vendor.name.lower() or 
                query_lower in vendor.cuisine_type.lower() or
                any(query_lower in specialty.lower() for specialty in vendor.specialties)):
                results.append(vendor)
        
        # Sort by rating (highest first)
        results.sort(key=lambda v: v.rating, reverse=True)
        return results


class OrchestratorAgent(BaseAgent):
    """Orchestrator Agent for food ordering coordination."""
    
    def __init__(self, agent_id: str, configuration: Optional[AgentConfiguration] = None):
        # Set up basic agent configuration
        if configuration is None:
            configuration = AgentConfiguration(
                agent_id=agent_id,
                agent_type=AgentType.ORCHESTRATOR,
                name="Orchestrator Agent",
                description="AI assistant for coordinating food ordering workflows",
                personality=PersonalityProfile(
                    primary_traits=[PersonalityTrait.SYSTEMATIC, PersonalityTrait.HELPFUL],
                    response_style="professional",
                    verbosity_level=6,
                    formality_level=7,
                    cooperation_tendency=0.9,
                    honesty_tendency=0.95
                )
            )
        
        super().__init__(configuration)
        
        # Initialize components
        self.order_manager = OrderManager()
        self.vendor_manager = VendorManager()
        self.user_twin_agent: Optional[UserTwinAgent] = None
        
        # Initialize LangChain components
        self._setup_langchain_agent()
        
        # Order workflow state
        self.current_orders: Dict[str, Order] = {}
        self.order_workflows: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized Orchestrator agent: {agent_id}")
    
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
            # Define tools for the orchestrator
            self.tools = self._create_orchestrator_tools()
            
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
                max_iterations=5,
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
        """Get system prompt for the orchestrator agent."""
        return """You are an Orchestrator AI agent responsible for coordinating food ordering workflows.

Your role is to:
1. Help users find and select appropriate vendors based on their preferences
2. Coordinate the ordering process with vendors
3. Manage order status updates and tracking
4. Handle payment processing and confirmations
5. Resolve order-related issues and conflicts
6. Provide excellent customer service throughout the ordering process

Key capabilities:
- Vendor discovery and recommendation
- Order creation and management
- Payment processing coordination
- Order tracking and status updates
- Customer support and issue resolution

Always be helpful, professional, and efficient in coordinating the food ordering process.
Prioritize user satisfaction and ensure smooth order fulfillment."""
    
    def _create_orchestrator_tools(self) -> List[BaseTool]:
        """Create tools specific to orchestrator functionality."""
        # This would include tools for order management, vendor communication, etc.
        # For now, we'll use basic tools - in a full implementation, these would be custom tools
        return []
    
    def set_user_twin_agent(self, user_twin_agent: UserTwinAgent) -> None:
        """Set the user twin agent for preference-based recommendations."""
        self.user_twin_agent = user_twin_agent
        logger.info(f"Connected to UserTwin agent: {user_twin_agent.agent_id}")
    
    async def search_vendors(self, query: str, user_preferences: Optional[Dict[str, Any]] = None) -> List[Vendor]:
        """Search for vendors based on query and user preferences."""
        logger.info(f"Searching vendors with query: {query}")
        
        # Get basic search results
        vendors = self.vendor_manager.search_vendors(query)
        
        # Apply user preferences if available
        if user_preferences and self.user_twin_agent:
            vendors = self._apply_user_preferences_to_vendors(vendors, user_preferences)
        
        # Record metrics
        record_metric("orchestrator.vendor.search", 1, MetricType.COUNTER, {
            "query": query,
            "results_count": len(vendors)
        })
        
        return vendors
    
    def _apply_user_preferences_to_vendors(self, vendors: List[Vendor], 
                                         user_preferences: Dict[str, Any]) -> List[Vendor]:
        """Apply user preferences to vendor recommendations."""
        if not self.user_twin_agent:
            return vendors
        
        # Get user food preferences
        food_prefs = self.user_twin_agent.get_preferences_by_category(PreferenceCategory.FOOD)
        cost_prefs = self.user_twin_agent.get_preferences_by_category(PreferenceCategory.COST)
        
        # Score vendors based on preferences
        scored_vendors = []
        for vendor in vendors:
            score = 0.0
            
            # Check cuisine preference
            for pref in food_prefs:
                if pref.key == "cuisine" and vendor.cuisine_type.lower() in pref.value.lower():
                    score += pref.strength * 10
            
            # Check cost preferences
            for pref in cost_prefs:
                if pref.key == "budget":
                    if pref.value == "cheap" and vendor.delivery_fee <= 2.0:
                        score += pref.strength * 5
                    elif pref.value == "moderate" and 2.0 < vendor.delivery_fee <= 4.0:
                        score += pref.strength * 5
                    elif pref.value == "expensive" and vendor.delivery_fee > 4.0:
                        score += pref.strength * 5
            
            # Add rating bonus
            score += vendor.rating * 2
            
            scored_vendors.append((vendor, score))
        
        # Sort by score and return vendors
        scored_vendors.sort(key=lambda x: x[1], reverse=True)
        return [vendor for vendor, score in scored_vendors]
    
    async def create_order(self, user_id: str, vendor_id: str, items: List[Dict[str, Any]], 
                          order_type: OrderType = OrderType.DELIVERY,
                          delivery_address: Optional[str] = None,
                          special_instructions: str = "") -> Optional[Order]:
        """Create a new order."""
        try:
            # Get vendor information
            vendor = self.vendor_manager.get_vendor(vendor_id)
            if not vendor or not vendor.is_available:
                logger.error(f"Vendor {vendor_id} not found or unavailable")
                return None
            
            # Create order items
            order_items = []
            total_amount = 0.0
            
            for item_data in items:
                item = OrderItem(
                    item_id=item_data.get("item_id", f"item_{len(order_items) + 1}"),
                    name=item_data["name"],
                    quantity=item_data["quantity"],
                    unit_price=item_data["unit_price"],
                    total_price=item_data["quantity"] * item_data["unit_price"],
                    special_instructions=item_data.get("special_instructions", ""),
                    dietary_requirements=item_data.get("dietary_requirements", []),
                    allergens=item_data.get("allergens", [])
                )
                order_items.append(item)
                total_amount += item.total_price
            
            # Add delivery fee if applicable
            if order_type == OrderType.DELIVERY:
                total_amount += vendor.delivery_fee
            
            # Check minimum order
            if total_amount < vendor.minimum_order:
                logger.warning(f"Order total {total_amount} below minimum {vendor.minimum_order}")
                return None
            
            # Create order
            order = Order(
                order_id=f"order_{int(time.time())}_{user_id}",
                user_id=user_id,
                vendor=vendor,
                items=order_items,
                order_type=order_type,
                status=OrderStatus.PENDING,
                payment_status=PaymentStatus.PENDING,
                total_amount=total_amount,
                delivery_address=delivery_address,
                special_instructions=special_instructions,
                estimated_delivery_time=datetime.now() + timedelta(minutes=vendor.delivery_time_minutes)
            )
            
            # Save order
            if self.order_manager.create_order(order):
                self.current_orders[order.order_id] = order
                
                # Record metrics
                record_metric("orchestrator.order.created", 1, MetricType.COUNTER, {
                    "vendor_id": vendor_id,
                    "order_type": order_type.value,
                    "total_amount": total_amount
                })
                
                logger.info(f"Created order: {order.order_id}")
                return order
            
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
        
        return None
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get current order status."""
        order = self.order_manager.get_order(order_id)
        if order:
            return {
                "order_id": order.order_id,
                "status": order.status.value,
                "payment_status": order.payment_status.value,
                "vendor_name": order.vendor.name,
                "estimated_delivery_time": order.estimated_delivery_time.isoformat() if order.estimated_delivery_time else None,
                "actual_delivery_time": order.actual_delivery_time.isoformat() if order.actual_delivery_time else None,
                "total_amount": order.total_amount,
                "items": [item.to_dict() for item in order.items]
            }
        return None
    
    async def update_order_status(self, order_id: str, new_status: OrderStatus) -> bool:
        """Update order status."""
        success = self.order_manager.update_order_status(order_id, new_status)
        if success:
            # Update current orders cache
            if order_id in self.current_orders:
                self.current_orders[order_id].status = new_status
                self.current_orders[order_id].updated_at = datetime.now()
            
            # Record metrics
            record_metric("orchestrator.order.status_updated", 1, MetricType.COUNTER, {
                "order_id": order_id,
                "new_status": new_status.value
            })
            
            logger.info(f"Updated order {order_id} status to {new_status.value}")
        
        return success
    
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
                response = result.get("output", "I'm not sure how to help with that.")
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
            record_metric("orchestrator.message.processed", 1, MetricType.COUNTER)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            self._status = AgentState.ERROR
            return f"I encountered an error while processing your request: {str(e)}"
    
    def _generate_fallback_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate fallback response when LangChain is not available."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["search", "find", "vendor", "restaurant"]):
            return "I can help you find vendors and restaurants. I have access to several vendors including Italian, Thai, American, and Japanese cuisine options. What type of food are you looking for?"
        
        elif any(word in message_lower for word in ["order", "place", "buy"]):
            return "I can help you place orders. To get started, please specify which vendor you'd like to order from and what items you'd like to order."
        
        elif any(word in message_lower for word in ["status", "track", "where"]):
            return "I can help you track your orders. Please provide your order ID and I'll check the current status for you."
        
        elif any(word in message_lower for word in ["help", "assistance"]):
            return "I'm your food ordering assistant. I can help you search for vendors, place orders, track order status, and resolve any issues. How can I assist you today?"
        
        else:
            return "I'm your food ordering orchestrator. I can help you find vendors, place orders, and track your food deliveries. What would you like to do?"
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order statistics."""
        return self.order_manager.get_order_statistics()
    
    def get_vendor_statistics(self) -> Dict[str, Any]:
        """Get vendor statistics."""
        vendors = self.vendor_manager.get_all_vendors()
        if not vendors:
            return {"total_vendors": 0}
        
        cuisine_counts = {}
        total_rating = 0.0
        
        for vendor in vendors:
            cuisine = vendor.cuisine_type
            cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1
            total_rating += vendor.rating
        
        return {
            "total_vendors": len(vendors),
            "cuisine_counts": cuisine_counts,
            "average_rating": total_rating / len(vendors),
            "available_vendors": len([v for v in vendors if v.is_available])
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        base_metrics = super().get_metrics()
        
        orchestrator_metrics = {
            "total_orders": len(self.order_manager._orders),
            "active_orders": len(self.current_orders),
            "total_vendors": len(self.vendor_manager._vendors),
            "available_vendors": len(self.vendor_manager.get_all_vendors())
        }
        
        return {**base_metrics, **orchestrator_metrics}
    
    async def shutdown(self) -> None:
        """Shutdown the agent."""
        logger.info(f"Shutting down Orchestrator agent: {self.agent_id}")
        
        # Update state
        self._status = AgentState.STOPPED
        
        await super().shutdown()


# Factory function
def create_orchestrator_agent(agent_id: str, 
                             configuration: Optional[AgentConfiguration] = None) -> OrchestratorAgent:
    """Create a new Orchestrator agent instance."""
    agent = OrchestratorAgent(agent_id, configuration)
    return agent
