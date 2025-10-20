#!/usr/bin/env python3
"""
Debug script to check if OptimizedKGLoader is loading the full knowledge graph correctly.
"""

import asyncio
import time
import logging
from pathlib import Path
from src.knowledge.kg_loader import KnowledgeGraphLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_kg_loading():
    """Test knowledge graph loading and statistics."""

    print("🔍 Testing KnowledgeGraphLoader...")

    # Initialize loader
    kg_path = Path("data/energy_knowledge_graph.ttl")
    loader = KnowledgeGraphLoader(kg_path)

    print(f"📁 Knowledge graph file: {kg_path}")
    print(f"📊 File size: {kg_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Test initial load
    print("\n🚀 Starting load...")
    start_time = time.time()
    loader.load()
    initial_load_time = time.time() - start_time

    print(f"⏱️  Initial load completed in {initial_load_time * 1000:.0f}ms")

    # Check initial stats
    initial_stats = loader.get_graph_stats()
    print(f"📈 Initial stats: {initial_stats}")

    # Wait for background load to complete
    print("\n⏳ Waiting for background load to complete...")
    wait_time = 0
    max_wait = 30  # 30 seconds max

    while not loader.is_full_graph_loaded() and wait_time < max_wait:
        await asyncio.sleep(1)
        wait_time += 1
        print(f"   Waiting... {wait_time}s")

    if loader.is_full_graph_loaded():
        print("✅ Full graph loaded successfully!")
    else:
        print("❌ Full graph failed to load within 30 seconds")
        return

    # Check final stats
    final_stats = loader.get_graph_stats()
    print(f"📊 Final stats: {final_stats}")

    # Test a simple query
    print("\n🔍 Testing query with 'asset'...")
    test_terms = ["asset"]
    results = loader.query_definitions(test_terms)

    print(f"📋 Query results for 'asset': {len(results)} results")
    for i, result in enumerate(results[:3]):  # Show first 3
        print(f"   {i+1}. {result['label']}: {result['definition'][:100]}...")
        print(f"      Citation: {result['citation_id']}, Score: {result['score']}")

    # Test vocabulary hydration
    print("\n🔤 Testing vocabulary hydration...")
    iec_terms, entsoe_terms = loader.hydrate_vocabularies()
    print(f"📚 Hydrated vocabularies: {len(iec_terms)} IEC terms, {len(entsoe_terms)} ENTSOE terms")

    # Show some examples
    if iec_terms:
        print(f"   IEC examples: {iec_terms[:5]}")
    if entsoe_terms:
        print(f"   ENTSOE examples: {entsoe_terms[:5]}")

    print("\n✅ Knowledge graph loading test completed!")

if __name__ == "__main__":
    asyncio.run(test_kg_loading())