#!/usr/bin/env python3
"""
End-to-end test for API reranker integration with EA assistant.

Tests the complete pipeline with API reranking enabled.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.ea_assistant import ProductionEAAgent


async def test_ea_with_reranker():
    """Test EA assistant with API reranking."""
    print("=" * 60)
    print("EA ASSISTANT + API RERANKER TEST")
    print("=" * 60)

    try:
        # Initialize EA assistant
        print("🔧 Initializing EA Assistant...")
        agent = ProductionEAAgent(
            kg_path="data/energy_knowledge_graph.ttl",
            models_path="data/models",
            docs_path="data/docs",
            vocab_path="config/vocabularies.json",
            llm_provider="groq"
        )
        print("✅ EA Assistant initialized")

        # Verify API reranker is available
        if agent.api_reranker:
            print(f"✅ API reranker is available: {type(agent.api_reranker).__name__}")
        else:
            print("❌ API reranker not available")
            return

        # Test queries
        test_queries = [
            "What is the difference between active and reactive power?",  # Should trigger reranking
            "What is reactive power?",  # May skip reranking (high confidence)
            "Tell me about power systems",  # Should trigger reranking (medium confidence)
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}: {query}")
            print(f"{'='*60}")

            try:
                # Process query
                response = await agent.process_query(query)

                # Check response
                if response and 'response' in response:
                    print(f"✅ Response received: {len(response['response'])} characters")
                    print(f"📝 Preview: {response['response'][:200]}...")

                    # Check if reranking occurred
                    if 'retrieval_context' in response:
                        context = response['retrieval_context']
                        api_reranked = context.get('api_reranked', False)
                        print(f"🔄 API reranked: {api_reranked}")

                        if api_reranked:
                            print("   ✅ API reranking was applied")
                        else:
                            print("   ⏭️  API reranking was skipped")

                    # Show stats if available
                    if agent.api_reranker:
                        agent.api_reranker.print_stats()

                else:
                    print("❌ No response received")

            except Exception as e:
                print(f"❌ Test {i} failed: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*60}")
        print("✅ END-TO-END TESTING COMPLETED")
        print(f"{'='*60}")

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_ea_with_reranker())