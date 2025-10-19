#!/usr/bin/env python3
"""
Integration test for EA Assistant with API reranking.

Tests:
1. API reranking is properly initialized
2. Reranking works in semantic enhancement
3. Reranking works in comparison queries
4. Graceful degradation when reranking fails
5. Can be disabled via environment variable
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.ea_assistant import ProductionEAAgent


async def test_initialization():
    """Test that API reranker initializes correctly."""
    print("\n" + "="*60)
    print("TEST 1: Initialization")
    print("="*60)

    # Check if OpenAI API key is set
    if not os.environ.get('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set")
        print("   Set it with: export OPENAI_API_KEY=your_key")
        return False

    # Initialize agent
    print("\n🔄 Initializing ProductionEAAgent...")
    agent = ProductionEAAgent(
        llm_provider='groq',
        vocab_path='config/vocabularies.json',
        models_path='data/models',
        docs_path='data/docs'
    )

    # Check if reranker is available
    if agent.api_reranker:
        print("✅ API reranker initialized successfully")
        print(f"   Model: {agent.api_reranker.model}")
        return True
    else:
        print("❌ API reranker not initialized")
        if agent.selective_reranker:
            print("   Selective reranker available but base reranker missing")
        return False


async def test_semantic_enhancement_with_reranking():
    """Test semantic enhancement with API reranking."""
    print("\n" + "="*60)
    print("TEST 2: Semantic Enhancement with Reranking")
    print("="*60)

    agent = ProductionEAAgent()

    if not agent.selective_reranker:
        print("⚠️  API reranker not available, skipping test")
        return

    # Test query that should trigger semantic search
    query = "What is power quality in electrical systems?"

    print(f"\n📝 Query: {query}")
    print("🔄 Processing query...")

    try:
        response = await agent.process_query(query, session_id="test-rerank-001")

        print(f"\n✅ Response received")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Citations: {len(response.citations)}")
        print(f"   Requires review: {response.requires_human_review}")

        # Check reranking stats
        if agent.selective_reranker:
            stats = agent.selective_reranker.get_stats()
            print(f"\n📊 Reranking Stats:")
            print(f"   Total queries: {stats['total_queries']}")
            print(f"   Reranked: {stats['reranked_queries']}")
            print(f"   Rerank rate: {stats['rerank_rate']*100:.1f}%")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_comparison_query_with_reranking():
    """Test comparison query with API reranking."""
    print("\n" + "="*60)
    print("TEST 3: Comparison Query with Reranking")
    print("="*60)

    agent = ProductionEAAgent()

    if not agent.selective_reranker:
        print("⚠️  API reranker not available, skipping test")
        return

    # Test comparison query
    query = "What is the difference between active power and reactive power?"

    print(f"\n📝 Query: {query}")
    print("🔄 Processing comparison query...")

    try:
        response = await agent.process_query(query, session_id="test-comparison-001")

        print(f"\n✅ Response received")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Citations: {response.citations}")
        print(f"   Number of citations: {len(response.citations)}")

        # Validate distinct citations
        if len(response.citations) >= 2:
            if len(set(response.citations)) == len(response.citations):
                print("   ✅ All citations are distinct")
            else:
                print("   ❌ Found duplicate citations!")
                return False

        # Check if response mentions both concepts
        response_lower = response.response.lower()
        if 'active' in response_lower and 'reactive' in response_lower:
            print("   ✅ Response mentions both concepts")
        else:
            print("   ⚠️  Response may not mention both concepts")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_graceful_degradation():
    """Test that system works even if reranking fails."""
    print("\n" + "="*60)
    print("TEST 4: Graceful Degradation")
    print("="*60)

    # Temporarily set invalid API key
    original_key = os.environ.get('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = 'invalid_key_for_testing'

    try:
        agent = ProductionEAAgent()

        # Should initialize without reranker
        if agent.api_reranker:
            print("⚠️  Reranker initialized with invalid key (unexpected)")
        else:
            print("✅ Agent initialized without reranker (expected)")

        # Should still process queries
        query = "What is voltage?"
        print(f"\n📝 Testing query without reranker: {query}")

        response = await agent.process_query(query, session_id="test-degradation-001")

        print(f"✅ Query processed successfully without reranker")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Citations: {len(response.citations)}")

        return True

    except Exception as e:
        print(f"❌ Graceful degradation failed: {e}")
        return False

    finally:
        # Restore original key
        if original_key:
            os.environ['OPENAI_API_KEY'] = original_key
        else:
            del os.environ['OPENAI_API_KEY']


async def test_disable_via_env_var():
    """Test that reranking can be disabled via environment variable."""
    print("\n" + "="*60)
    print("TEST 5: Disable via Environment Variable")
    print("="*60)

    # Set env var to disable
    os.environ['ENABLE_API_RERANKING'] = 'false'

    try:
        agent = ProductionEAAgent()

        if agent.api_reranker:
            print("❌ Reranker initialized despite ENABLE_API_RERANKING=false")
            return False
        else:
            print("✅ Reranker disabled by environment variable")
            return True

    finally:
        # Re-enable
        os.environ['ENABLE_API_RERANKING'] = 'true'


async def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("INTEGRATION TEST SUITE - API RERANKING")
    print("="*60)

    results = []

    # Run tests
    results.append(("Initialization", await test_initialization()))
    results.append(("Semantic Enhancement", await test_semantic_enhancement_with_reranking()))
    results.append(("Comparison Query", await test_comparison_query_with_reranking()))
    results.append(("Graceful Degradation", await test_graceful_degradation()))
    results.append(("Disable via Env Var", await test_disable_via_env_var()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! API reranking is properly integrated.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")


if __name__ == "__main__":
    asyncio.run(main())