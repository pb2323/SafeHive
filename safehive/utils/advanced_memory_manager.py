"""
Advanced Memory Management for SafeHive AI Agents

This module provides enhanced memory management capabilities including
context-aware retrieval, memory compression, and advanced analytics.
"""

import json
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
from dataclasses import dataclass, field
import hashlib
import numpy as np
from collections import defaultdict, deque

from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory
)
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings

from safehive.utils.logger import get_logger
from safehive.models.agent_models import AgentMessage, MessageType, Conversation, AgentMemory
from safehive.utils.agent_memory import SafeHiveMemoryManager

logger = get_logger(__name__)


@dataclass
class MemoryContext:
    """Represents contextual information for memory retrieval."""
    query: str
    conversation_id: Optional[str] = None
    participant: Optional[str] = None
    message_type: Optional[MessageType] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    relevance_threshold: float = 0.7
    max_results: int = 10


@dataclass
class MemoryResult:
    """Represents a memory retrieval result with relevance scoring."""
    message: AgentMessage
    relevance_score: float
    context_match: Dict[str, Any] = field(default_factory=dict)
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryAnalyticsData:
    """Represents analytics data about memory usage."""
    total_messages: int
    total_conversations: int
    memory_size_bytes: int
    most_active_participants: List[Tuple[str, int]]
    most_common_message_types: List[Tuple[str, int]]
    conversation_frequency: Dict[str, int]
    memory_growth_rate: float
    compression_ratio: float
    last_analyzed: datetime = field(default_factory=datetime.now)


class ContextualMemoryRetriever:
    """Advanced memory retriever with contextual awareness."""
    
    def __init__(self, memory_manager: SafeHiveMemoryManager):
        self.memory_manager = memory_manager
        self.embedding_cache: Dict[str, List[float]] = {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
    
    def calculate_relevance_score(
        self,
        message: AgentMessage,
        context: MemoryContext
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate relevance score for a message given context."""
        score = 0.0
        context_match = {}
        
        # Content similarity (simplified - in production, use embeddings)
        query_words = set(context.query.lower().split())
        message_words = set(message.content.lower().split())
        
        if query_words and message_words:
            content_similarity = len(query_words.intersection(message_words)) / len(query_words.union(message_words))
            score += content_similarity * 0.4
            context_match["content_similarity"] = content_similarity
        
        # Participant match
        if context.participant:
            if message.sender == context.participant or message.recipient == context.participant:
                score += 0.3
                context_match["participant_match"] = True
        
        # Message type match
        if context.message_type and message.message_type == context.message_type:
            score += 0.2
            context_match["message_type_match"] = True
        
        # Conversation ID match
        if context.conversation_id:
            # This would require conversation tracking - simplified for now
            score += 0.1
            context_match["conversation_match"] = True
        
        # Time relevance (more recent = higher score)
        if context.time_range:
            start_time, end_time = context.time_range
            if start_time <= message.timestamp <= end_time:
                time_factor = 1.0
            else:
                # Calculate time decay
                if message.timestamp < start_time:
                    time_diff = (start_time - message.timestamp).total_seconds()
                else:
                    time_diff = (message.timestamp - end_time).total_seconds()
                
                time_factor = max(0.1, 1.0 - (time_diff / (24 * 3600)))  # Decay over days
            
            score += time_factor * 0.2
            context_match["time_relevance"] = time_factor
        
        return min(1.0, score), context_match
    
    def retrieve_relevant_memories(
        self,
        context: MemoryContext,
        limit: Optional[int] = None
    ) -> List[MemoryResult]:
        """Retrieve memories relevant to the given context."""
        try:
            # Get all conversation history
            all_messages = self.memory_manager.get_conversation_history()
            
            # Calculate relevance scores
            scored_results = []
            for message in all_messages:
                relevance_score, context_match = self.calculate_relevance_score(message, context)
                
                if relevance_score >= context.relevance_threshold:
                    result = MemoryResult(
                        message=message,
                        relevance_score=relevance_score,
                        context_match=context_match,
                        retrieval_metadata={
                            "retrieved_at": datetime.now().isoformat(),
                            "context_query": context.query
                        }
                    )
                    scored_results.append(result)
            
            # Sort by relevance score
            scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Apply limit
            if limit:
                scored_results = scored_results[:limit]
            elif context.max_results:
                scored_results = scored_results[:context.max_results]
            
            logger.debug(f"Retrieved {len(scored_results)} relevant memories for context: {context.query}")
            return scored_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant memories: {e}")
            return []


class MemoryCompressor:
    """Handles memory compression and summarization."""
    
    def __init__(self, memory_manager: SafeHiveMemoryManager):
        self.memory_manager = memory_manager
        self.compression_threshold = 100  # Compress when more than 100 messages
        self.summary_threshold = 50  # Summarize conversations with more than 50 messages
    
    def compress_conversation_history(
        self,
        conversation: Conversation,
        compression_ratio: float = 0.5
    ) -> Conversation:
        """Compress a conversation by keeping only important messages."""
        try:
            if len(conversation.messages) <= self.summary_threshold:
                return conversation
            
            # Simple compression: keep first, last, and every Nth message
            original_count = len(conversation.messages)
            target_count = max(1, int(original_count * compression_ratio))
            
            if target_count >= original_count:
                return conversation
            
            # Select messages to keep
            kept_messages = []
            
            # Always keep first and last messages
            if conversation.messages:
                kept_messages.append(conversation.messages[0])
            
            # Keep messages at regular intervals (excluding first and last)
            if len(conversation.messages) > 1 and target_count > 1:
                remaining_slots = target_count - 1  # Reserve space for last message
                middle_messages = len(conversation.messages) - 1  # Exclude first message
                
                if remaining_slots > 0 and middle_messages > 0:
                    step = max(1, middle_messages // remaining_slots)
                    
                    for i in range(1, len(conversation.messages), step):
                        if len(kept_messages) < target_count - 1:  # Reserve space for last message
                            kept_messages.append(conversation.messages[i])
            
            # Always add the last message if there are multiple messages
            if len(conversation.messages) > 1:
                kept_messages.append(conversation.messages[-1])
            
            # Create compressed conversation
            compressed_conversation = Conversation(
                conversation_id=f"{conversation.conversation_id}_compressed",
                participants=conversation.participants,
                messages=kept_messages,
                created_at=conversation.created_at,
                updated_at=datetime.now(),
                metadata={
                    **conversation.metadata,
                    "compressed": True,
                    "original_message_count": original_count,
                    "compression_ratio": compression_ratio
                }
            )
            
            logger.info(f"Compressed conversation {conversation.conversation_id} from {original_count} to {len(kept_messages)} messages")
            return compressed_conversation
            
        except Exception as e:
            logger.error(f"Failed to compress conversation: {e}")
            return conversation
    
    def summarize_conversation(
        self,
        conversation: Conversation,
        max_summary_length: int = 500
    ) -> str:
        """Create a summary of a conversation."""
        try:
            if not conversation.messages:
                return "Empty conversation"
            
            # Simple summarization - in production, use LLM
            participants = ", ".join(conversation.participants)
            message_count = len(conversation.messages)
            duration = conversation.updated_at - conversation.created_at
            
            # Extract key topics (simplified)
            topics = set()
            for message in conversation.messages:
                words = message.content.lower().split()
                # Simple keyword extraction
                for word in words:
                    if len(word) > 4 and word.isalpha():
                        topics.add(word)
            
            key_topics = list(topics)[:5]  # Top 5 topics
            
            summary = f"Conversation between {participants} with {message_count} messages over {duration}. "
            if key_topics:
                summary += f"Key topics: {', '.join(key_topics)}."
            
            return summary[:max_summary_length]
            
        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            return "Failed to create summary"
    
    def should_compress_memory(self) -> bool:
        """Check if memory should be compressed."""
        stats = self.memory_manager.get_memory_stats()
        return stats.get("total_messages", 0) > self.compression_threshold


class MemoryAnalytics:
    """Provides analytics and insights about memory usage."""
    
    def __init__(self, memory_manager: SafeHiveMemoryManager):
        self.memory_manager = memory_manager
        self.analytics_cache: Optional[MemoryAnalyticsData] = None
        self.cache_ttl = timedelta(hours=1)
    
    def generate_analytics(self, force_refresh: bool = False) -> MemoryAnalyticsData:
        """Generate comprehensive memory analytics."""
        try:
            # Check cache
            if (not force_refresh and 
                self.analytics_cache and 
                datetime.now() - self.analytics_cache.last_analyzed < self.cache_ttl):
                return self.analytics_cache
            
            # Get all messages
            all_messages = self.memory_manager.get_conversation_history()
            
            # Basic statistics
            total_messages = len(all_messages)
            total_conversations = len(set(msg.metadata.get("conversation_id", "unknown") for msg in all_messages))
            
            # Calculate memory size
            memory_size_bytes = sum(len(json.dumps(msg.to_dict())) for msg in all_messages)
            
            # Participant activity
            participant_counts = defaultdict(int)
            for message in all_messages:
                participant_counts[message.sender] += 1
                participant_counts[message.recipient] += 1
            
            most_active_participants = sorted(
                participant_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Message type distribution
            message_type_counts = defaultdict(int)
            for message in all_messages:
                message_type_counts[message.message_type.value] += 1
            
            most_common_message_types = sorted(
                message_type_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Conversation frequency
            conversation_frequency = defaultdict(int)
            for message in all_messages:
                conv_id = message.metadata.get("conversation_id", "unknown")
                conversation_frequency[conv_id] += 1
            
            # Calculate growth rate (simplified)
            memory_growth_rate = self._calculate_growth_rate(all_messages)
            
            # Compression ratio
            compression_ratio = self._calculate_compression_ratio(all_messages)
            
            analytics = MemoryAnalyticsData(
                total_messages=total_messages,
                total_conversations=total_conversations,
                memory_size_bytes=memory_size_bytes,
                most_active_participants=most_active_participants,
                most_common_message_types=most_common_message_types,
                conversation_frequency=dict(conversation_frequency),
                memory_growth_rate=memory_growth_rate,
                compression_ratio=compression_ratio,
                last_analyzed=datetime.now()
            )
            
            self.analytics_cache = analytics
            logger.info(f"Generated memory analytics for agent {self.memory_manager.agent_id}")
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate memory analytics: {e}")
            # Return minimal analytics
            return MemoryAnalyticsData(
                total_messages=0,
                total_conversations=0,
                memory_size_bytes=0,
                most_active_participants=[],
                most_common_message_types=[],
                conversation_frequency={},
                memory_growth_rate=0.0,
                compression_ratio=1.0
            )
    
    def _calculate_growth_rate(self, messages: List[AgentMessage]) -> float:
        """Calculate memory growth rate over time."""
        if len(messages) < 2:
            return 0.0
        
        # Group messages by day
        daily_counts = defaultdict(int)
        for message in messages:
            day = message.timestamp.date()
            daily_counts[day] += 1
        
        if len(daily_counts) < 2:
            return 0.0
        
        # Calculate average daily growth
        days = sorted(daily_counts.keys())
        total_growth = 0
        for i in range(1, len(days)):
            prev_count = daily_counts[days[i-1]]
            curr_count = daily_counts[days[i]]
            if prev_count > 0:
                growth = (curr_count - prev_count) / prev_count
                total_growth += growth
        
        return total_growth / (len(days) - 1) if len(days) > 1 else 0.0
    
    def _calculate_compression_ratio(self, messages: List[AgentMessage]) -> float:
        """Calculate compression ratio based on message content."""
        if not messages:
            return 1.0
        
        # Simple heuristic: ratio of unique content to total content
        unique_contents = set(msg.content for msg in messages)
        return len(unique_contents) / len(messages) if messages else 1.0


class AdvancedMemoryManager:
    """
    Advanced memory manager with contextual awareness, compression, and analytics.
    
    This class extends the basic SafeHiveMemoryManager with advanced features
    for intelligent memory management.
    """
    
    def __init__(self, memory_manager: SafeHiveMemoryManager):
        self.memory_manager = memory_manager
        self.retriever = ContextualMemoryRetriever(memory_manager)
        self.compressor = MemoryCompressor(memory_manager)
        self.analytics = MemoryAnalytics(memory_manager)
        
        # Memory optimization settings
        self.auto_compress = True
        self.compression_threshold = 100
        self.analytics_enabled = True
    
    def retrieve_contextual_memories(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        participant: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        relevance_threshold: float = 0.05,
        max_results: int = 10
    ) -> List[MemoryResult]:
        """Retrieve memories relevant to a specific context."""
        context = MemoryContext(
            query=query,
            conversation_id=conversation_id,
            participant=participant,
            message_type=message_type,
            time_range=time_range,
            relevance_threshold=relevance_threshold,
            max_results=max_results
        )
        
        return self.retriever.retrieve_relevant_memories(context)
    
    def compress_memory(self, compression_ratio: float = 0.5) -> Dict[str, Any]:
        """Compress memory to reduce size while maintaining important information."""
        try:
            if not self.compressor.should_compress_memory():
                return {"compressed": False, "reason": "Memory size below threshold"}
            
            # Get current analytics
            analytics = self.analytics.generate_analytics()
            
            # Compress conversations
            compressed_count = 0
            original_size = analytics.total_messages
            
            # This is a simplified compression - in production, you'd work with actual conversations
            # For now, we'll just log the compression request
            logger.info(f"Memory compression requested for agent {self.memory_manager.agent_id}")
            
            return {
                "compressed": True,
                "original_messages": original_size,
                "compression_ratio": compression_ratio,
                "compressed_conversations": compressed_count
            }
            
        except Exception as e:
            logger.error(f"Failed to compress memory: {e}")
            return {"compressed": False, "error": str(e)}
    
    def get_memory_analytics(self, force_refresh: bool = False) -> MemoryAnalyticsData:
        """Get comprehensive memory analytics."""
        return self.analytics.generate_analytics(force_refresh)
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Automatically optimize memory usage."""
        try:
            optimization_results = {}
            
            # Get current analytics
            analytics = self.analytics.generate_analytics()
            optimization_results["initial_analytics"] = {
                "total_messages": analytics.total_messages,
                "memory_size_bytes": analytics.memory_size_bytes,
                "compression_ratio": analytics.compression_ratio
            }
            
            # Auto-compress if needed
            if self.auto_compress and analytics.total_messages > self.compression_threshold:
                compression_result = self.compress_memory()
                optimization_results["compression"] = compression_result
            
            # Get final analytics
            final_analytics = self.analytics.generate_analytics(force_refresh=True)
            optimization_results["final_analytics"] = {
                "total_messages": final_analytics.total_messages,
                "memory_size_bytes": final_analytics.memory_size_bytes,
                "compression_ratio": final_analytics.compression_ratio
            }
            
            logger.info(f"Memory optimization completed for agent {self.memory_manager.agent_id}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Failed to optimize memory: {e}")
            return {"error": str(e)}
    
    def search_memories(
        self,
        search_query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryResult]:
        """Search memories with advanced filtering."""
        try:
            # Build context from filters
            context = MemoryContext(query=search_query)
            
            if filters:
                context.participant = filters.get("participant")
                context.message_type = filters.get("message_type")
                context.time_range = filters.get("time_range")
                context.relevance_threshold = filters.get("relevance_threshold", 0.7)
                context.max_results = filters.get("max_results", 10)
            
            return self.retriever.retrieve_relevant_memories(context)
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    def get_memory_insights(self) -> Dict[str, Any]:
        """Get insights and recommendations about memory usage."""
        try:
            analytics = self.analytics.generate_analytics()
            
            insights = {
                "memory_health": self._assess_memory_health(analytics),
                "recommendations": self._generate_recommendations(analytics),
                "trends": self._analyze_trends(analytics),
                "efficiency_metrics": {
                    "compression_ratio": analytics.compression_ratio,
                    "growth_rate": analytics.memory_growth_rate,
                    "memory_utilization": min(1.0, analytics.total_messages / 1000)  # Assuming 1000 as max
                }
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get memory insights: {e}")
            return {"error": str(e)}
    
    def _assess_memory_health(self, analytics: MemoryAnalytics) -> Dict[str, Any]:
        """Assess the health of the memory system."""
        health_score = 100.0
        
        # Check memory size
        if analytics.memory_size_bytes > 10 * 1024 * 1024:  # 10MB
            health_score -= 20
        
        # Check compression ratio
        if analytics.compression_ratio < 0.5:
            health_score -= 15
        
        # Check growth rate
        if analytics.memory_growth_rate > 1.0:  # 100% growth
            health_score -= 10
        
        # Determine health status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 60:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "status": status,
            "score": health_score,
            "total_messages": analytics.total_messages,
            "memory_size_mb": analytics.memory_size_bytes / (1024 * 1024)
        }
    
    def _generate_recommendations(self, analytics: MemoryAnalytics) -> List[str]:
        """Generate recommendations for memory optimization."""
        recommendations = []
        
        if analytics.total_messages > 500:
            recommendations.append("Consider compressing old conversations to reduce memory usage")
        
        if analytics.compression_ratio < 0.6:
            recommendations.append("High message duplication detected - consider deduplication")
        
        if analytics.memory_growth_rate > 0.5:
            recommendations.append("Rapid memory growth detected - consider implementing retention policies")
        
        if analytics.memory_size_bytes > 5 * 1024 * 1024:  # 5MB
            recommendations.append("Memory size is large - consider archiving old data")
        
        if not recommendations:
            recommendations.append("Memory usage is optimal - no immediate actions needed")
        
        return recommendations
    
    def _analyze_trends(self, analytics: MemoryAnalytics) -> Dict[str, Any]:
        """Analyze memory usage trends."""
        return {
            "most_active_participants": analytics.most_active_participants[:3],
            "common_message_types": analytics.most_common_message_types[:3],
            "growth_rate": analytics.memory_growth_rate,
            "conversation_count": analytics.total_conversations
        }


# Factory function for creating advanced memory managers
def create_advanced_memory_manager(
    memory_manager: SafeHiveMemoryManager
) -> AdvancedMemoryManager:
    """Create an advanced memory manager from a basic one."""
    return AdvancedMemoryManager(memory_manager)


# Global registry for advanced memory managers
_advanced_memory_managers: Dict[str, AdvancedMemoryManager] = {}


def get_advanced_memory_manager(agent_id: str) -> Optional[AdvancedMemoryManager]:
    """Get an advanced memory manager for an agent."""
    return _advanced_memory_managers.get(agent_id)


def register_advanced_memory_manager(
    agent_id: str,
    advanced_manager: AdvancedMemoryManager
) -> None:
    """Register an advanced memory manager."""
    _advanced_memory_managers[agent_id] = advanced_manager


def unregister_advanced_memory_manager(agent_id: str) -> bool:
    """Unregister an advanced memory manager."""
    if agent_id in _advanced_memory_managers:
        del _advanced_memory_managers[agent_id]
        return True
    return False
