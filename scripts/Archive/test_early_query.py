#!/usr/bin/env python3
"""
Test script to simulate early query scenario (like UI server startup).
"""

import time
import logging
from pathlib import Path
from src.knowledge.kg_loader import KnowledgeGraphLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_early_query():
    """Test querying immediately after load() like UI server does."""

    print("🚀 Testing early query scenario (simulating UI server)...")

    # Initialize loader
    kg_path = Path("data/energy_knowledge_graph.ttl")
    loader = KnowledgeGraphLoader(kg_path)

    # Load immediately
    print("📂 Starting load...")
    loader.load()

    # Query IMMEDIATELY (like UI server does)
    print("⚡ Querying IMMEDIATELY for 'asset' (before background load completes)...")
    start_time = time.time()

    results = loader.query_definitions(["asset"])

    query_time = time.time() - start_time

    print(f"⏱️  Query completed in {query_time:.2f}s")
    print(f"📋 Results: {len(results)} definitions found")

    if results:
        print("✅ SUCCESS: Query waited for graph loading and returned results!")
        for i, result in enumerate(results[:2]):
            print(f"   {i+1}. {result['label']}: {result['definition'][:80]}...")
    else:
        print("❌ FAILED: Query returned empty results")

if __name__ == "__main__":
    test_early_query()