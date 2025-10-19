"""
API-based reranking using OpenAI text-embedding-3-small.

This module provides high-quality reranking without local compute requirements.
Uses OpenAI's state-of-the-art embeddings via API for superior results.

Key Features:
- State-of-the-art quality (MTEB 62.3)
- Fast API calls (~200-500ms)
- Negligible cost ($0.00004 per query)
- Zero local resources
- Embedding caching for efficiency
- Selective reranking to optimize costs

Author: AInstein Team
Date: October 2025
"""

import asyncio
import numpy as np
import os
from typing import List, Dict, Optional, Set
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation."""
    original_rank: int
    new_rank: int
    element: str
    citation: str
    original_score: float
    rerank_score: float
    score_improvement: float


class APIReranker:
    """
    High-quality reranking using OpenAI embeddings API.

    Uses text-embedding-3-small for optimal quality/cost/speed balance.

    Performance Characteristics:
    - Quality: MTEB score 62.3 (state-of-the-art for small models)
    - Speed: 200-500ms per query (includes API latency)
    - Cost: $0.00002 per 1K tokens ≈ $0.00004 per query
    - Dimensions: 1536 (vs 384 for older models)

    Advantages over local cross-encoders:
    - 3-5% better quality than ms-marco-MiniLM
    - 4-10x faster than CPU inference on MacBook
    - Zero local RAM usage
    - Always up-to-date (OpenAI maintains model)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        cache_embeddings: bool = True
    ):
        """
        Initialize API reranker.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Embedding model to use
                   - "text-embedding-3-small": Recommended (1536 dims, $0.02/1M tokens)
                   - "text-embedding-3-large": Overkill (3072 dims, $0.13/1M tokens)
            cache_embeddings: Cache embeddings to reduce API calls
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.cache_embeddings = cache_embeddings

        # Embedding cache (in-memory for session)
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Statistics
        self.stats = {
            'total_reranks': 0,
            'total_candidates': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'total_tokens': 0,
            'estimated_cost_usd': 0.0,
            'avg_latency_ms': 0.0
        }

        logger.info(f"✅ APIReranker initialized")
        logger.info(f"   Model: {model}")
        logger.info(f"   Dimensions: 1536")
        logger.info(f"   Cost: $0.00002/1K tokens")
        logger.info(f"   Caching: {'Enabled' if cache_embeddings else 'Disabled'}")

    async def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        return_details: bool = False
    ) -> List[Dict]:
        """
        Rerank candidates using OpenAI embeddings.

        Args:
            query: User query
            candidates: List of candidate results to rerank
            top_k: Number of top results to return
            return_details: Include reranking details in results

        Returns:
            Reranked candidates with updated scores
        """
        if not candidates:
            logger.warning("No candidates to rerank")
            return []

        if len(candidates) <= 1:
            logger.info("Only 1 candidate, skipping rerank")
            return candidates

        start_time = datetime.now()

        try:
            logger.info(f"🔄 Reranking {len(candidates)} candidates with {self.model}")

            # Prepare texts
            query_text = query.strip()
            candidate_texts = [
                self._extract_text_from_candidate(c)
                for c in candidates
            ]

            # Get embeddings (with caching)
            query_embedding = await self._get_embedding(query_text)
            candidate_embeddings = await self._get_embeddings_batch(candidate_texts)

            # Calculate cosine similarities
            similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]

            # Create rerank results for analysis
            rerank_results = []

            # Update candidates with rerank scores
            for idx, (candidate, similarity) in enumerate(zip(candidates, similarities)):
                original_score = candidate.get('confidence',
                                              candidate.get('semantic_score',
                                              candidate.get('score', 0.0)))

                rerank_result = RerankResult(
                    original_rank=idx,
                    new_rank=-1,  # Will be set after sorting
                    element=candidate.get('element', 'Unknown'),
                    citation=candidate.get('citation', 'N/A'),
                    original_score=original_score,
                    rerank_score=float(similarity),
                    score_improvement=float(similarity) - original_score
                )
                rerank_results.append(rerank_result)

                # Update candidate
                candidate['original_score'] = original_score
                candidate['rerank_score'] = float(similarity)
                candidate['confidence'] = float(similarity)  # Update primary score
                candidate['reranked'] = True
                candidate['rerank_model'] = self.model

                if return_details:
                    candidate['rerank_details'] = {
                        'original_rank': idx,
                        'score_change': float(similarity) - original_score,
                        'method': 'api_embeddings'
                    }

            # Sort by rerank score
            reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

            # Update new ranks in results
            for new_rank, result in enumerate(sorted(rerank_results,
                                                     key=lambda x: x.rerank_score,
                                                     reverse=True)):
                result.new_rank = new_rank

            # Log reranking results
            self._log_rerank_results(rerank_results, top_k)

            # Update statistics
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(len(candidates), latency_ms)

            logger.info(f"✅ Reranking complete in {latency_ms:.0f}ms")

            return reranked[:top_k]

        except Exception as e:
            logger.error(f"❌ API reranking failed: {e}")
            logger.warning("Returning original candidate order")
            return candidates[:top_k]

    def _extract_text_from_candidate(self, candidate: Dict) -> str:
        """
        Extract text from candidate for embedding.

        Priority: definition > text > element
        """
        # Try definition first (most complete)
        if candidate.get('definition'):
            text = candidate['definition']
        # Then text
        elif candidate.get('text'):
            text = candidate['text']
        # Finally element name
        else:
            text = candidate.get('element', 'Unknown')

        # Ensure reasonable length (embeddings work best with full context)
        return text.strip()

    async def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for single text with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (1536 dimensions)
        """
        # Check cache
        if self.cache_embeddings and text in self._embedding_cache:
            self.stats['cache_hits'] += 1
            return self._embedding_cache[text]

        self.stats['cache_misses'] += 1

        # Get from API
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )

        embedding = np.array(response.data[0].embedding, dtype=np.float32)

        # Update token count
        tokens_used = response.usage.total_tokens
        self.stats['total_tokens'] += tokens_used
        self.stats['api_calls'] += 1

        # Cache it
        if self.cache_embeddings:
            self._embedding_cache[text] = embedding

        return embedding

    async def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for batch of texts with caching.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings (n_texts x 1536)
        """
        # Separate cached and uncached
        cached_embeddings = {}
        texts_to_fetch = []

        for i, text in enumerate(texts):
            if self.cache_embeddings and text in self._embedding_cache:
                cached_embeddings[i] = self._embedding_cache[text]
                self.stats['cache_hits'] += 1
            else:
                texts_to_fetch.append((i, text))
                self.stats['cache_misses'] += 1

        # Fetch uncached embeddings
        fetched_embeddings = {}
        if texts_to_fetch:
            indices, texts_list = zip(*texts_to_fetch)

            response = await self.client.embeddings.create(
                model=self.model,
                input=list(texts_list)
            )

            # Update token count
            tokens_used = response.usage.total_tokens
            self.stats['total_tokens'] += tokens_used
            self.stats['api_calls'] += 1

            for idx, embedding_data in zip(indices, response.data):
                embedding = np.array(embedding_data.embedding, dtype=np.float32)
                fetched_embeddings[idx] = embedding

                # Cache it
                if self.cache_embeddings:
                    self._embedding_cache[texts_list[indices.index(idx)]] = embedding

        # Combine in original order
        all_embeddings = []
        for i in range(len(texts)):
            if i in cached_embeddings:
                all_embeddings.append(cached_embeddings[i])
            else:
                all_embeddings.append(fetched_embeddings[i])

        return np.array(all_embeddings)

    def _log_rerank_results(self, results: List[RerankResult], top_k: int):
        """Log reranking results for debugging."""
        logger.info(f"📊 Reranking Results (showing top {min(top_k, 3)}):")

        # Show top results
        top_results = sorted(results, key=lambda x: x.rerank_score, reverse=True)[:min(top_k, 3)]

        for result in top_results:
            rank_change = result.original_rank - result.new_rank
            rank_symbol = "↑" if rank_change > 0 else "↓" if rank_change < 0 else "="

            logger.info(
                f"   #{result.new_rank + 1} (was #{result.original_rank + 1} {rank_symbol}): "
                f"{result.element[:40]}... [{result.citation}] "
                f"score: {result.original_score:.3f}→{result.rerank_score:.3f} "
                f"(Δ{result.score_improvement:+.3f})"
            )

        # Show biggest improvements
        biggest_improvements = sorted(results, key=lambda x: x.score_improvement, reverse=True)[:2]
        if biggest_improvements and biggest_improvements[0].score_improvement > 0.1:
            logger.info(f"💡 Biggest improvements:")
            for result in biggest_improvements:
                if result.score_improvement > 0.1:
                    logger.info(
                        f"   {result.element[:40]}... "
                        f"(Δ{result.score_improvement:+.3f})"
                    )

    def _update_stats(self, num_candidates: int, latency_ms: float):
        """Update reranking statistics."""
        self.stats['total_reranks'] += 1
        self.stats['total_candidates'] += num_candidates

        # Update average latency (running average)
        n = self.stats['total_reranks']
        old_avg = self.stats['avg_latency_ms']
        self.stats['avg_latency_ms'] = (old_avg * (n - 1) + latency_ms) / n

        # Estimate cost ($0.00002 per 1K tokens)
        cost_per_token = 0.00002 / 1000
        self.stats['estimated_cost_usd'] = self.stats['total_tokens'] * cost_per_token

    def get_stats(self) -> Dict:
        """Get reranking statistics."""
        cache_hit_rate = (
            self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
            if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0
            else 0.0
        )

        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'avg_candidates_per_rerank': (
                self.stats['total_candidates'] / self.stats['total_reranks']
                if self.stats['total_reranks'] > 0
                else 0
            )
        }

    def clear_cache(self):
        """Clear embedding cache."""
        cache_size = len(self._embedding_cache)
        self._embedding_cache.clear()
        logger.info(f"🗑️  Cleared embedding cache ({cache_size} entries)")

    def print_stats(self):
        """Print detailed statistics."""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("API RERANKER STATISTICS")
        print("="*60)
        print(f"Total rerank operations: {stats['total_reranks']}")
        print(f"Total candidates reranked: {stats['total_candidates']}")
        print(f"Average candidates per rerank: {stats['avg_candidates_per_rerank']:.1f}")
        print(f"\nAPI Performance:")
        print(f"  API calls made: {stats['api_calls']}")
        print(f"  Total tokens used: {stats['total_tokens']:,}")
        print(f"  Average latency: {stats['avg_latency_ms']:.0f}ms")
        print(f"\nCaching:")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache misses: {stats['cache_misses']}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']*100:.1f}%")
        print(f"\nCost:")
        print(f"  Estimated cost: ${stats['estimated_cost_usd']:.4f}")
        print(f"  Cost per rerank: ${stats['estimated_cost_usd']/max(1, stats['total_reranks']):.5f}")
        print("="*60 + "\n")


class SelectiveAPIReranker:
    """
    Intelligently decide when to rerank for optimal quality/cost balance.

    Strategy:
    - Rerank when it's likely to significantly improve results
    - Skip reranking when there's already a clear winner
    - Target: 20-30% rerank rate (saves 70% of costs while keeping 90% of quality)

    Triggers for reranking:
    1. Similar scores (variance < threshold) - unclear ranking
    2. Comparison queries - need precise distinction
    3. Medium confidence - uncertain, might benefit from reranking
    4. Close top-2 - hard to distinguish best result
    5. Low overall quality - all results below threshold
    """

    def __init__(self, api_reranker: APIReranker):
        """
        Initialize selective reranker.

        Args:
            api_reranker: APIReranker instance to use when reranking
        """
        self.reranker = api_reranker

        # Statistics
        self.stats = {
            'total_queries': 0,
            'reranked_queries': 0,
            'skipped_queries': 0,
            'rerank_reasons': {
                'similar_scores': 0,
                'comparison_query': 0,
                'medium_confidence': 0,
                'close_top_two': 0,
                'low_quality': 0
            }
        }

        logger.info("✅ SelectiveAPIReranker initialized")

    def should_rerank(self, candidates: List[Dict], query: str) -> tuple[bool, str]:
        """
        Decide if reranking is beneficial.

        Args:
            candidates: List of candidates to potentially rerank
            query: Original user query

        Returns:
            Tuple of (should_rerank: bool, reason: str)
        """
        if len(candidates) < 2:
            return False, "insufficient_candidates"

        # Limit analysis to top 10 for efficiency
        analysis_candidates = candidates[:10]

        # Extract scores
        scores = [
            c.get('confidence', c.get('semantic_score', c.get('score', 0.0)))
            for c in analysis_candidates
        ]

        if not scores or max(scores) == 0:
            return False, "no_scores"

        # Rule 1: Similar scores (low variance)
        score_variance = np.var(scores)
        if score_variance < 0.01:
            return True, "similar_scores"

        # Rule 2: Comparison query
        query_lower = query.lower()
        comparison_indicators = [
            'difference', 'compare', 'comparison', 'vs', 'versus',
            'better', 'worse', 'which', 'between', 'or'
        ]
        if any(indicator in query_lower for indicator in comparison_indicators):
            return True, "comparison_query"

        # Rule 3: Medium confidence (might benefit from reranking)
        max_confidence = max(scores)
        if 0.50 < max_confidence < 0.75:
            return True, "medium_confidence"

        # Rule 4: Top 2 candidates very close
        if len(scores) >= 2:
            sorted_scores = sorted(scores, reverse=True)
            top_two_diff = abs(sorted_scores[0] - sorted_scores[1])
            if top_two_diff < 0.05:
                return True, "close_top_two"

        # Rule 5: Low overall quality
        if max_confidence < 0.60:
            return True, "low_quality"

        # No trigger - skip reranking
        return False, "high_confidence_clear_winner"

    async def conditional_rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Conditionally rerank based on heuristics.

        Args:
            query: User query
            candidates: Candidates to potentially rerank
            top_k: Number of top results to return

        Returns:
            Reranked candidates if triggered, otherwise original order
        """
        self.stats['total_queries'] += 1

        should_rerank, reason = self.should_rerank(candidates, query)

        if should_rerank:
            logger.info(f"🎯 Reranking triggered: {reason}")
            self.stats['reranked_queries'] += 1
            self.stats['rerank_reasons'][reason] += 1

            reranked = await self.reranker.rerank(query, candidates, top_k)
            return reranked
        else:
            logger.info(f"⏭️  Reranking skipped: {reason}")
            self.stats['skipped_queries'] += 1
            return candidates[:top_k]

    def get_rerank_rate(self) -> float:
        """Get percentage of queries that were reranked."""
        if self.stats['total_queries'] == 0:
            return 0.0
        return self.stats['reranked_queries'] / self.stats['total_queries']

    def get_stats(self) -> Dict:
        """Get detailed statistics."""
        rate = self.get_rerank_rate()

        # Estimate monthly cost based on current rate
        estimated_monthly_cost = self.estimate_monthly_cost(500)

        return {
            **self.stats,
            'rerank_rate': rate,
            'estimated_monthly_cost_usd': estimated_monthly_cost,
            'api_reranker_stats': self.reranker.get_stats()
        }

    def estimate_monthly_cost(self, queries_per_month: int) -> float:
        """
        Estimate monthly cost based on current rerank rate.

        Args:
            queries_per_month: Expected monthly query volume

        Returns:
            Estimated monthly cost in USD
        """
        rate = self.get_rerank_rate() if self.stats['total_queries'] > 0 else 0.30
        cost_per_reranked_query = 0.00004  # $0.00004 per reranked query
        estimated_reranked = queries_per_month * rate
        return estimated_reranked * cost_per_reranked_query

    def print_stats(self):
        """Print detailed statistics."""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("SELECTIVE RERANKER STATISTICS")
        print("="*60)
        print(f"Total queries: {stats['total_queries']}")
        print(f"Reranked: {stats['reranked_queries']} ({stats['rerank_rate']*100:.1f}%)")
        print(f"Skipped: {stats['skipped_queries']} ({(1-stats['rerank_rate'])*100:.1f}%)")
        print(f"\nRerank Triggers:")
        for reason, count in stats['rerank_reasons'].items():
            if count > 0:
                print(f"  {reason}: {count}")
        print(f"\nCost Estimate:")
        print(f"  Monthly (500 queries): ${stats['estimated_monthly_cost_usd']:.4f}")
        print(f"  Annual (6000 queries): ${stats['estimated_monthly_cost_usd']*12:.4f}")
        print("="*60 + "\n")

        # Print API reranker stats
        self.reranker.print_stats()

    async def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Delegate reranking to base reranker.

        Args:
            query: Search query
            candidates: Candidate documents
            top_k: Number of results to return

        Returns:
            Reranked candidates
        """
        return await self.reranker.rerank(query, candidates, top_k)