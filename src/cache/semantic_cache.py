"""
Semantic Cache for AInstein - Intelligent query result caching using embeddings.

This cache uses cosine similarity to match semantically similar queries,
dramatically reducing response time and API costs for ADR analysis.
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    query: str
    query_embedding: np.ndarray
    result: Any
    timestamp: datetime
    hit_count: int = 0
    analysis_type: Optional[str] = None
    confidence: float = 0.0
    
    def is_expired(self, ttl_minutes: int = 60) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() - self.timestamp > timedelta(minutes=ttl_minutes)


class SemanticADRCache:
    """
    Semantic caching for ADR analysis queries.
    
    Features:
    - Similarity-based matching (not just exact match)
    - TTL-based expiration
    - Hit rate tracking
    - Analysis type categorization
    """
    
    def __init__(
        self, 
        similarity_threshold: float = 0.85,
        ttl_minutes: int = 60,
        max_entries: int = 1000,
        classifier=None
    ):
        """
        Initialize semantic cache.
        
        Args:
            similarity_threshold: Minimum cosine similarity for cache hit (0.85 = 85%)
            ttl_minutes: Time-to-live for cache entries
            max_entries: Maximum cache size before LRU eviction
            classifier: Optional function to classify query types. If None, uses default.
        """
        self.similarity_threshold = similarity_threshold
        self.ttl_minutes = ttl_minutes
        self.max_entries = max_entries
        self.classifier = classifier or self._default_classifier
        
        # Cache storage
        self.entries: List[CacheEntry] = []
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'avg_similarity': 0.0
        }
        
        # Analysis-specific caches for better organization
        self.analysis_caches = {
            'decision_drivers': [],
            'patterns': [],
            'compliance': [],
            'summary': [],
            'general': []
        }
        
        logger.info(f"SemanticADRCache initialized (threshold={similarity_threshold})")

    def _default_classifier(self, query: str) -> str:
        """
        Simple default classifier based on keywords.
        Can be overridden by passing a custom classifier to __init__.
        """
        query_lower = query.lower()
        
        # Check for ADR-specific patterns first
        if 'adr' in query_lower or 'decision record' in query_lower:
            # Decision driver queries
            if any(word in query_lower for word in ['why', 'driver', 'reason', 'rationale', 'motivation', 'drove', 'caused']):
                return 'decision_drivers'
            # Pattern queries
            elif any(word in query_lower for word in ['pattern', 'trend', 'common', 'recurring', 'similar', 'theme']):
                return 'patterns'
            # Compliance queries
            elif any(word in query_lower for word in ['compliance', 'standard', 'regulation', 'togaf', 'requirement']):
                return 'compliance'
            # Summary queries
            elif any(word in query_lower for word in ['summary', 'overview', 'list', 'all', 'show me', 'what are']):
                return 'summary'
        
        # Default to general for anything else
        return 'general'
    
    def get(
        self, 
        query: str, 
        query_embedding: np.ndarray,
        analysis_type: Optional[str] = None
    ) -> Optional[Any]:
        """
        Retrieve cached result for semantically similar query.
        
        Args:
            query: The query text
            query_embedding: Query embedding vector (normalized)
            analysis_type: Type of analysis for better matching
            
        Returns:
            Cached result if found, None otherwise
        """
        start_time = time.time()
        
        # Choose appropriate cache pool
        cache_pool = (self.analysis_caches.get(analysis_type, []) 
                     if analysis_type else self.entries)
        
        # Remove expired entries
        cache_pool = [e for e in cache_pool if not e.is_expired(self.ttl_minutes)]
        
        if not cache_pool:
            self.stats['misses'] += 1
            return None
        
        # Find best match using cosine similarity
        best_match = None
        best_similarity = 0.0
        
        for entry in cache_pool:
            # Compute cosine similarity (vectors are pre-normalized)
            similarity = np.dot(query_embedding, entry.query_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        # Check if similarity exceeds threshold
        if best_match and best_similarity >= self.similarity_threshold:
            # Update statistics
            self.stats['hits'] += 1
            self.stats['avg_similarity'] = (
                (self.stats['avg_similarity'] * (self.stats['hits'] - 1) + best_similarity) 
                / self.stats['hits']
            )
            
            # Update hit count
            best_match.hit_count += 1
            
            # Log cache hit
            logger.info(
                f"🎯 Cache HIT: similarity={best_similarity:.3f}, "
                f"type={analysis_type}, time={time.time()-start_time:.3f}s"
            )
            
            return best_match.result
        
        self.stats['misses'] += 1
        logger.debug(f"Cache MISS: best_similarity={best_similarity:.3f}")
        return None
    
    def put(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: Any,
        analysis_type: Optional[str] = None,
        confidence: float = 0.0
    ) -> None:
        """
        Store result in cache.
        
        Args:
            query: The query text
            query_embedding: Query embedding vector (normalized)
            result: Result to cache
            analysis_type: Type of analysis
            confidence: Confidence score of result
        """
        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            result=result,
            timestamp=datetime.now(),
            analysis_type=analysis_type,
            confidence=confidence
        )
        
        # Add to appropriate cache
        if analysis_type and analysis_type in self.analysis_caches:
            cache_pool = self.analysis_caches[analysis_type]
        else:
            cache_pool = self.entries
        
        cache_pool.append(entry)
        
        # LRU eviction if needed
        if len(cache_pool) > self.max_entries:
            # Sort by hit_count and timestamp, remove least used
            cache_pool.sort(key=lambda x: (x.hit_count, x.timestamp))
            removed = cache_pool.pop(0)
            self.stats['evictions'] += 1
            logger.debug(f"Evicted cache entry: {removed.query[:50]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_queries = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_queries if total_queries > 0 else 0
        
        return {
            'hit_rate': f"{hit_rate:.2%}",
            'total_hits': self.stats['hits'],
            'total_misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'avg_similarity': f"{self.stats['avg_similarity']:.3f}",
            'cache_sizes': {
                k: len(v) for k, v in self.analysis_caches.items()
            }
        }
    
    def clear(self, analysis_type: Optional[str] = None) -> None:
        """Clear cache (all or specific type)."""
        if analysis_type:
            self.analysis_caches[analysis_type] = []
            logger.info(f"Cleared {analysis_type} cache")
        else:
            self.entries = []
            for key in self.analysis_caches:
                self.analysis_caches[key] = []
            logger.info("Cleared all caches")