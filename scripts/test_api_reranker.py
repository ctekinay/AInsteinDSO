#!/usr/bin/env python3
"""
Test script for API reranker.

Tests:
1. Basic reranking functionality
2. Selective reranking triggers
3. Cost estimation
4. Performance benchmarking
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.api_reranker import APIReranker, SelectiveAPIReranker


async def test_basic_reranking():
    """Test basic reranking functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic Reranking")
    print("="*60)

    # Initialize
    reranker = APIReranker()

    # Test query
    query = "What is reactive power?"

    # Mock candidates (simulating retrieval results)
    candidates = [
        {
            'element': 'Active power',
            'definition': 'The real component of the apparent power at fundamental frequency, expressed in watts or multiples thereof.',
            'citation': 'eurlex:631-20',
            'confidence': 0.65
        },
        {
            'element': 'Reactive power',
            'definition': 'The imaginary component of the apparent power at fundamental frequency, usually expressed in kilovar or megavar.',
            'citation': 'eurlex:631-28',
            'confidence': 0.63
        },
        {
            'element': 'Apparent power',
            'definition': 'The product of RMS voltage and RMS current, expressed in volt-amperes.',
            'citation': 'eurlex:631-25',
            'confidence': 0.60
        },
        {
            'element': 'Power factor',
            'definition': 'The ratio of active power to apparent power.',
            'citation': 'eurlex:631-30',
            'confidence': 0.58
        }
    ]

    print(f"\n📝 Query: {query}")
    print(f"📦 Candidates: {len(candidates)}")
    print(f"\n🔍 Original ranking:")
    for i, c in enumerate(candidates, 1):
        print(f"   #{i}: {c['element']} (score: {c['confidence']:.3f}) [{c['citation']}]")

    # Rerank
    print(f"\n🔄 Reranking with text-embedding-3-small...")
    reranked = await reranker.rerank(query, candidates, top_k=4)

    print(f"\n✅ Reranked results:")
    for i, c in enumerate(reranked, 1):
        orig_score = c['original_score']
        new_score = c['rerank_score']
        change = new_score - orig_score
        symbol = "↑" if change > 0 else "↓" if change < 0 else "="
        print(f"   #{i}: {c['element']} (score: {orig_score:.3f}→{new_score:.3f} {symbol}) [{c['citation']}]")

    # Print stats
    reranker.print_stats()

    return reranker


async def test_selective_reranking():
    """Test selective reranking logic."""
    print("\n" + "="*60)
    print("TEST 2: Selective Reranking")
    print("="*60)

    # Initialize
    api_reranker = APIReranker()
    selective = SelectiveAPIReranker(api_reranker)

    # Test cases
    test_cases = [
        {
            'query': 'What is the difference between active and reactive power?',
            'expected_rerank': True,
            'expected_reason': 'comparison_query'
        },
        {
            'query': 'What is reactive power?',
            'candidates': [
                {'element': 'Reactive power', 'citation': 'eurlex:631-28', 'confidence': 0.95},
                {'element': 'Active power', 'citation': 'eurlex:631-20', 'confidence': 0.45}
            ],
            'expected_rerank': False,
            'expected_reason': 'high_confidence_clear_winner'
        },
        {
            'query': 'Tell me about power systems',
            'candidates': [
                {'element': 'Active power', 'citation': 'eurlex:631-20', 'confidence': 0.65},
                {'element': 'Reactive power', 'citation': 'eurlex:631-28', 'confidence': 0.63},
                {'element': 'Apparent power', 'citation': 'eurlex:631-25', 'confidence': 0.62}
            ],
            'expected_rerank': True,
            'expected_reason': 'medium_confidence'
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        query = test_case['query']
        candidates = test_case.get('candidates', [
            {'element': f'Concept {j}', 'citation': f'test:00{j}', 'confidence': 0.6}
            for j in range(1, 4)
        ])

        print(f"\n📝 Test case {i}: {query}")

        should_rerank, reason = selective.should_rerank(candidates, query)

        print(f"   Decision: {'RERANK' if should_rerank else 'SKIP'} (reason: {reason})")
        print(f"   Expected: {'RERANK' if test_case['expected_rerank'] else 'SKIP'} (reason: {test_case['expected_reason']})")

        if should_rerank == test_case['expected_rerank']:
            print(f"   ✅ PASS")
        else:
            print(f"   ❌ FAIL")

    # Print stats
    selective.print_stats()

    return selective


async def test_cost_estimation():
    """Test cost estimation."""
    print("\n" + "="*60)
    print("TEST 3: Cost Estimation")
    print("="*60)

    # Initialize
    api_reranker = APIReranker()
    selective = SelectiveAPIReranker(api_reranker)

    # Simulate different query volumes
    query_volumes = [100, 500, 1000, 5000, 10000]

    print(f"\n💰 Cost estimates (assuming 30% rerank rate):")
    print(f"\n{'Queries/Month':<15} {'Monthly Cost':<15} {'Annual Cost':<15}")
    print("-" * 45)

    for volume in query_volumes:
        monthly = selective.estimate_monthly_cost(volume)
        annual = monthly * 12
        print(f"{volume:<15,} ${monthly:<14.4f} ${annual:<14.2f}")

    print(f"\n📊 For reference:")
    print(f"   - Your pilot (500 queries/month): ${selective.estimate_monthly_cost(500):.4f}/month")
    print(f"   - Production (10k queries/month): ${selective.estimate_monthly_cost(10000):.4f}/month")


async def test_performance_benchmark():
    """Benchmark reranking performance."""
    print("\n" + "="*60)
    print("TEST 4: Performance Benchmark")
    print("="*60)

    import time

    # Initialize
    reranker = APIReranker()

    # Test with different candidate counts
    candidate_counts = [5, 10, 20]

    print(f"\n⏱️  Performance benchmarks:")
    print(f"\n{'Candidates':<12} {'Latency (ms)':<15} {'Tokens':<10}")
    print("-" * 37)

    for count in candidate_counts:
        # Create mock candidates
        candidates = [
            {
                'element': f'Concept {i}',
                'definition': f'This is a detailed definition for concept {i} with sufficient text to simulate real candidates.',
                'citation': f'test:00{i}',
                'confidence': 0.6 - (i * 0.05)
            }
            for i in range(count)
        ]

        # Measure latency
        start = time.time()
        await reranker.rerank("Test query for benchmarking", candidates, top_k=5)
        latency_ms = (time.time() - start) * 1000

        # Get token usage
        tokens = reranker.stats['total_tokens']

        print(f"{count:<12} {latency_ms:<14.0f}ms {tokens:<10,}")

    print(f"\n✅ Average latency: {reranker.stats['avg_latency_ms']:.0f}ms")
    print(f"✅ Total cost: ${reranker.stats['estimated_cost_usd']:.6f}")


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("API RERANKER TEST SUITE")
    print("="*60)
    print("Testing text-embedding-3-small reranking")
    print("="*60)

    try:
        # Run tests
        await test_basic_reranking()
        await test_selective_reranking()
        await test_cost_estimation()
        await test_performance_benchmark()

        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())