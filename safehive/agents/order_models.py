"""
Order-related data models for the SafeHive AI Security Sandbox.

This module contains the core data models used across the order management system
to avoid circular import issues.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


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
    COMPLETED = "completed"


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
