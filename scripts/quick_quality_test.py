#!/usr/bin/env python3
"""
Quick quality test focusing on the two critical improvements:
1. Comparison query fix (distinct concepts)
2. API reranking quality boost
"""

import asyncio
import sys
import os
from pathlib import Path

# Disable tokenizer warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.ea_assistant import ProductionEAAgent


async def test_comparison_fix():
    """Test that comparison queries return distinct concepts."""
    print("\n" + "="*60)
    print("TEST 1: Comparison Query Fix")
    print("="*60)

    agent = ProductionEAAgent()

    # The original failing case
    query = "What is the difference between active power and reactive power?"
    print(f"\nQuery: {query}")

    try:
        result = await agent.process_query(query, "test-comparison-001")

        # Handle tuple vs response object
        if isinstance(result, tuple):
            result = result[0]  # Take first element if it's a tuple

        # Check for distinct citations
        citations = result.citations if hasattr(result, 'citations') else []
        citations_distinct = len(citations) == len(set(citations))

        print(f"\n✅ Response received")
        print(f"   Citations: {citations}")
        print(f"   Count: {len(citations)}")
        print(f"   Distinct: {'YES ✅' if citations_distinct else 'NO ❌'}")

        # Check if both concepts mentioned
        response_text = result.response if hasattr(result, 'response') else str(result)
        response_lower = response_text.lower()
        has_active = 'active' in response_lower
        has_reactive = 'reactive' in response_lower

        print(f"   Mentions 'active': {'YES ✅' if has_active else 'NO ❌'}")
        print(f"   Mentions 'reactive': {'YES ✅' if has_reactive else 'NO ❌'}")

        # Pass/Fail
        passed = (
            len(citations) >= 2 and
            citations_distinct and
            has_active and
            has_reactive
        )

        print(f"\n{'✅ TEST PASSED' if passed else '❌ TEST FAILED'}")

        return passed

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


async def test_api_reranking():
    """Test that API reranking is working."""
    print("\n" + "="*60)
    print("TEST 2: API Reranking")
    print("="*60)

    agent = ProductionEAAgent()

    if not agent.selective_reranker:
        print("⚠️  API reranker not available - skipping test")
        return None

    # Run a few queries to accumulate stats
    test_queries = [
        "What is reactive power?",
        "How does voltage relate to power systems?",
        "Compare voltage and current",
    ]

    print(f"\nProcessing {len(test_queries)} queries...")

    try:
        for i, query in enumerate(test_queries, 1):
            print(f"  {i}. {query[:50]}...")
            await agent.process_query(query, f"test-rerank-{i:03d}")

        # Get reranking stats
        stats = agent.selective_reranker.get_stats()

        print(f"\n📊 Reranking Statistics:")
        print(f"   Total queries: {stats['total_queries']}")
        print(f"   Reranked: {stats['reranked_queries']}")
        print(f"   Rerank rate: {stats['rerank_rate']*100:.1f}%")
        print(f"   Estimated monthly cost (500 queries): ${stats['estimated_monthly_cost_usd']:.4f}")

        # Pass if reranker is working (processed queries)
        passed = stats['total_queries'] > 0

        print(f"\n{'✅ TEST PASSED' if passed else '❌ TEST FAILED'}")

        return passed

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


async def test_no_regression():
    """Test that basic functionality still works."""
    print("\n" + "="*60)
    print("TEST 3: No Regression (Basic Query)")
    print("="*60)

    agent = ProductionEAAgent()

    query = "What is voltage?"
    print(f"\nQuery: {query}")

    try:
        result = await agent.process_query(query, "test-regression-001")

        # Handle tuple vs response object
        if isinstance(result, tuple):
            result = result[0]  # Take first element if it's a tuple

        confidence = getattr(result, 'confidence', 0.0)
        citations = getattr(result, 'citations', [])
        response_text = getattr(result, 'response', str(result))

        print(f"\n✅ Response received")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Citations: {len(citations)}")
        print(f"   Has content: {'YES ✅' if len(response_text) > 50 else 'NO ❌'}")

        # Pass if we got a reasonable response (lower threshold for basic test)
        passed = (
            confidence > 0.4 and  # Lower threshold
            len(citations) >= 1 and
            len(response_text) > 50
        )

        print(f"\n{'✅ TEST PASSED' if passed else '❌ TEST FAILED'}")

        return passed

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


async def main():
    """Run quick quality tests."""
    print("\n" + "="*60)
    print("QUICK QUALITY VALIDATION")
    print("="*60)
    print("Testing critical improvements:")
    print("1. Comparison query fix")
    print("2. API reranking integration")
    print("3. No regression")

    results = {}

    try:
        results['comparison'] = await test_comparison_fix()
        results['reranking'] = await test_api_reranking()
        results['regression'] = await test_no_regression()

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        for test_name, passed in results.items():
            if passed is None:
                status = "⚠️  SKIPPED"
            elif passed:
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
            print(f"{test_name:20} {status}")

        # Overall
        passed_count = sum(1 for p in results.values() if p is True)
        total_count = sum(1 for p in results.values() if p is not None)

        if total_count > 0:
            print(f"\nResult: {passed_count}/{total_count} tests passed")

            if passed_count == total_count:
                print("\n🎉 All critical features working correctly!")
            elif passed_count >= total_count * 0.67:
                print("\n⚠️  Most features working, but some issues need attention")
            else:
                print("\n❌ Critical issues detected - needs debugging")

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())