"""
Metrics Collection and Management for SafeHive AI Security Sandbox

This module provides comprehensive metrics collection, storage, and analysis
capabilities for monitoring system performance, security events, and agent behavior.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict, deque
import threading
from enum import Enum

from safehive.utils.logger import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    EVENT = "event"


@dataclass
class MetricData:
    """Individual metric data point."""
    name: str
    value: Union[int, float, str]
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['metric_type'] = self.metric_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricData':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['metric_type'] = MetricType(data['metric_type'])
        return cls(**data)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    metric_type: MetricType
    count: int = 0
    total: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    avg_value: float = 0.0
    last_value: Optional[Union[int, float, str]] = None
    last_updated: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def update(self, metric: MetricData):
        """Update summary with new metric data."""
        self.count += 1
        self.last_updated = metric.timestamp
        self.last_value = metric.value
        self.tags.update(metric.tags)

        if isinstance(metric.value, (int, float)):
            self.total += metric.value
            
            if self.min_value is None or metric.value < self.min_value:
                self.min_value = metric.value
            if self.max_value is None or metric.value > self.max_value:
                self.max_value = metric.value
            
            self.avg_value = self.total / self.count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.last_updated:
            data['last_updated'] = self.last_updated.isoformat()
        data['metric_type'] = self.metric_type.value
        return data


class MetricsCollector:
    """
    Thread-safe metrics collector for SafeHive system.
    
    Collects, stores, and provides summary statistics for various system metrics
    including performance, security events, and agent behavior.
    """
    
    def __init__(self, max_metrics: int = 10000, retention_hours: int = 24):
        """
        Initialize metrics collector.
        
        Args:
            max_metrics: Maximum number of metrics to keep in memory
            retention_hours: Hours to retain metrics data
        """
        self.max_metrics = max_metrics
        self.retention_hours = retention_hours
        self.retention_delta = timedelta(hours=retention_hours)
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._metrics: deque = deque(maxlen=max_metrics)
        self._summaries: Dict[str, MetricSummary] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._events: List[MetricData] = []
        
        # System metrics
        self._start_time = datetime.now()
        self._total_metrics_collected = 0
        
        logger.info(f"Metrics collector initialized with max_metrics={max_metrics}, retention_hours={retention_hours}")

    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = datetime.now() - self.retention_delta
        
        with self._lock:
            # Clean up metrics deque
            while self._metrics and self._metrics[0].timestamp < cutoff_time:
                self._metrics.popleft()
            
            # Clean up events
            self._events = [event for event in self._events if event.timestamp >= cutoff_time]
            
            # Clean up timer data
            for timer_name in list(self._timers.keys()):
                self._timers[timer_name] = [
                    value for value in self._timers[timer_name]
                    if value >= (datetime.now() - self.retention_delta).timestamp()
                ]
                if not self._timers[timer_name]:
                    del self._timers[timer_name]

    def record_metric(
        self,
        name: str,
        value: Union[int, float, str],
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional tags for categorization
            metadata: Optional metadata
        """
        if tags is None:
            tags = {}
        if metadata is None:
            metadata = {}
        
        metric = MetricData(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags,
            metadata=metadata
        )
        
        with self._lock:
            # Add to metrics deque
            self._metrics.append(metric)
            
            # Update summary
            if name not in self._summaries:
                self._summaries[name] = MetricSummary(
                    name=name,
                    metric_type=metric_type
                )
            self._summaries[name].update(metric)
            
            # Update specific metric type storage
            if metric_type == MetricType.COUNTER:
                self._counters[name] += int(value) if isinstance(value, (int, float)) else 1
            elif metric_type == MetricType.GAUGE:
                self._gauges[name] = float(value) if isinstance(value, (int, float)) else 0.0
            elif metric_type == MetricType.TIMER:
                if isinstance(value, (int, float)):
                    self._timers[name].append(float(value))
            elif metric_type == MetricType.EVENT:
                self._events.append(metric)
            
            self._total_metrics_collected += 1
            
            # Periodic cleanup
            if self._total_metrics_collected % 100 == 0:
                self._cleanup_old_metrics()
        
        logger.debug(f"Recorded metric: {name}={value} (type={metric_type.value})")

    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)

    def set_gauge(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        self.record_metric(name, value, MetricType.GAUGE, tags)

    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric (duration in seconds)."""
        self.record_metric(name, duration, MetricType.TIMER, tags)

    def record_event(self, name: str, description: str, tags: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record an event metric."""
        event_metadata = {"description": description}
        if metadata:
            event_metadata.update(metadata)
        self.record_metric(name, description, MetricType.EVENT, tags, event_metadata)

    def get_metric_summary(self, name: str) -> Optional[MetricSummary]:
        """Get summary for a specific metric."""
        with self._lock:
            return self._summaries.get(name)

    def get_all_summaries(self) -> Dict[str, MetricSummary]:
        """Get all metric summaries."""
        with self._lock:
            return self._summaries.copy()

    def get_counters(self) -> Dict[str, int]:
        """Get all counter values."""
        with self._lock:
            return self._counters.copy()

    def get_gauges(self) -> Dict[str, float]:
        """Get all gauge values."""
        with self._lock:
            return self._gauges.copy()

    def get_timer_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get timer statistics for a specific timer."""
        with self._lock:
            if name not in self._timers or not self._timers[name]:
                return None
            
            values = self._timers[name]
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "total": sum(values)
            }

    def get_recent_events(self, limit: int = 100) -> List[MetricData]:
        """Get recent events."""
        with self._lock:
            return sorted(self._events, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-level statistics."""
        with self._lock:
            uptime = datetime.now() - self._start_time
            return {
                "uptime_seconds": uptime.total_seconds(),
                "uptime_human": str(uptime).split('.')[0],  # Remove microseconds
                "total_metrics_collected": self._total_metrics_collected,
                "active_metrics": len(self._summaries),
                "active_counters": len(self._counters),
                "active_gauges": len(self._gauges),
                "active_timers": len(self._timers),
                "recent_events": len(self._events),
                "memory_usage_mb": len(self._metrics) * 0.001,  # Rough estimate
                "start_time": self._start_time.isoformat()
            }

    def export_metrics(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format ("json", "dict")
        
        Returns:
            Exported metrics data
        """
        with self._lock:
            export_data = {
                "system_stats": self.get_system_stats(),
                "summaries": {name: summary.to_dict() for name, summary in self._summaries.items()},
                "counters": self._counters,
                "gauges": self._gauges,
                "timer_stats": {
                    name: self.get_timer_stats(name) 
                    for name in self._timers.keys()
                },
                "recent_events": [event.to_dict() for event in self.get_recent_events(50)],
                "export_timestamp": datetime.now().isoformat()
            }
            
            if format == "json":
                return json.dumps(export_data, indent=2)
            else:
                return export_data

    def save_metrics(self, filepath: Union[str, Path]) -> bool:
        """Save metrics to file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                f.write(self.export_metrics("json"))
            
            logger.info(f"Metrics saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save metrics to {filepath}: {e}")
            return False

    def load_metrics(self, filepath: Union[str, Path]) -> bool:
        """Load metrics from file."""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.warning(f"Metrics file {filepath} does not exist")
                return False
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Note: This is a simplified load - in production you might want
            # to restore the full metrics history
            logger.info(f"Metrics loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load metrics from {filepath}: {e}")
            return False

    def clear_metrics(self) -> None:
        """Clear all metrics data."""
        with self._lock:
            self._metrics.clear()
            self._summaries.clear()
            self._counters.clear()
            self._gauges.clear()
            self._timers.clear()
            self._events.clear()
            self._total_metrics_collected = 0
            self._start_time = datetime.now()
        
        logger.info("All metrics cleared")


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def record_metric(
    name: str,
    value: Union[int, float, str],
    metric_type: MetricType = MetricType.GAUGE,
    tags: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Record a metric using the global collector."""
    get_metrics_collector().record_metric(name, value, metric_type, tags, metadata)


def increment_counter(name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
    """Increment a counter using the global collector."""
    get_metrics_collector().increment_counter(name, value, tags)


def set_gauge(name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
    """Set a gauge using the global collector."""
    get_metrics_collector().set_gauge(name, value, tags)


def record_timer(name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record a timer using the global collector."""
    get_metrics_collector().record_timer(name, duration, tags)


def record_event(name: str, description: str, tags: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Record an event using the global collector."""
    get_metrics_collector().record_event(name, description, tags, metadata)


def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of all metrics."""
    collector = get_metrics_collector()
    return {
        "system_stats": collector.get_system_stats(),
        "summaries": collector.get_all_summaries(),
        "counters": collector.get_counters(),
        "gauges": collector.get_gauges(),
        "recent_events": collector.get_recent_events(20)
    }


def export_metrics(format: str = "json") -> Union[str, Dict[str, Any]]:
    """Export metrics using the global collector."""
    return get_metrics_collector().export_metrics(format)


def save_metrics(filepath: Union[str, Path]) -> bool:
    """Save metrics using the global collector."""
    return get_metrics_collector().save_metrics(filepath)


def clear_metrics() -> None:
    """Clear all metrics using the global collector."""
    get_metrics_collector().clear_metrics()
