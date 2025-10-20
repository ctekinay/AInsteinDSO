#!/usr/bin/env python3
"""
Comprehensive test script to demonstrate EA Assistant tracing.

This script shows exactly which modules are called and in what order
when you prompt the EA Assistant, with detailed timing and flow information.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).parent / "src"))

from src.agent.ea_assistant import ProductionEAAgent
from src.utils.trace import get_tracer

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Reduce noise from other loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
tracer = get_tracer()


async def test_query_flow(query: str):
    """Test a single query and show the complete trace."""

    print(f"\n{'='*80}")
    print(f"🔍 TRACING QUERY: '{query}'")
    print(f"{'='*80}")

    try:
        # Initialize the EA Assistant
        print("📝 Initializing EA Assistant...")

        agent = ProductionEAAgent()
        print("✅ EA Assistant initialized successfully")

        # Process the query with full tracing
        print(f"\n🚀 Processing query: '{query}'")
        print("-" * 60)

        response = await agent.process_query(query)

        print("-" * 60)
        print("✅ Query processing completed!")

        # Show the results
        print(f"\n📋 RESULTS:")
        print(f"   Route: {response.route}")
        print(f"   Response length: {len(response.response)} characters")
        print(f"   Citations found: {len(response.citations)}")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Requires review: {response.requires_human_review}")
        print(f"   TOGAF phase: {response.togaf_phase}")

        print(f"\n📄 RESPONSE:")
        print(f"   {response.response[:200]}...")

        print(f"\n📚 CITATIONS:")
        for i, citation in enumerate(response.citations[:3], 1):
            print(f"   {i}. {citation}")

        if len(response.citations) > 3:
            print(f"   ... and {len(response.citations) - 3} more")

    except Exception as e:
        print(f"❌ Error during query processing: {e}")
        import traceback
        traceback.print_exc()


async def run_trace_tests():
    """Run multiple test queries to demonstrate different flows."""

    print("🔍 EA ASSISTANT TRACING DEMONSTRATION")
    print("=" * 80)
    print("This script shows exactly which modules are called when")
    print("you prompt the EA Assistant system.")
    print("=" * 80)

    # Test queries for different routing scenarios
    test_queries = [
        "What is an asset in Alliander terms?",  # Should route to structured_model
        "How do I create a TOGAF Phase B architecture?",  # Should route to togaf_method
        "What are the latest regulations for grid operators?",  # Should route to unstructured_docs
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 TEST {i}/{len(test_queries)}")
        await test_query_flow(query)

        if i < len(test_queries):
            print(f"\n⏳ Waiting 2 seconds before next test...")
            await asyncio.sleep(2)

    print(f"\n🎉 ALL TESTS COMPLETED")
    print("=" * 80)
    print("💡 You can now see the exact flow of modules called!")
    print("   - Check the console output above for detailed traces")
    print("   - Each function call is logged with timing information")
    print("   - Routing decisions are clearly visible")
    print("   - Knowledge retrieval steps are tracked")
    print("=" * 80)


async def demo_single_query():
    """Demo with a single query for focused tracing."""

    print("🔍 SINGLE QUERY TRACING DEMO")
    print("=" * 60)

    # Use a query that will show the full pipeline
    query = "What is an asset in energy systems?"

    await test_query_flow(query)


if __name__ == "__main__":
    print("🚀 Starting EA Assistant Tracing Demo...")

    # Choose demo mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        asyncio.run(demo_single_query())
    else:
        asyncio.run(run_trace_tests())