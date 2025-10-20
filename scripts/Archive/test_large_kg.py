#!/usr/bin/env python3
"""
Test the optimized KG loader with the large 132,532 triple knowledge graph.
"""

import time
import logging
from pathlib import Path
from src.knowledge.kg_loader import KnowledgeGraphLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_large_kg():
    """Test loading and querying the large knowledge graph."""
    logger.info("Testing KnowledgeGraphLoader with large knowledge graph...")

    # Initialize loader
    kg_path = Path("data/energy_knowledge_graph.ttl")
    loader = KnowledgeGraphLoader(kg_path)

    # Test lazy loading
    start_time = time.time()
    loader.load()
    init_time = time.time() - start_time
    logger.info(f"Lazy initialization completed in {init_time:.3f}s")

    # Test immediate query (before full load)
    start_time = time.time()
    early_result = loader.load_on_demand(["power", "grid"])
    early_time = time.time() - start_time
    logger.info(f"Early query completed in {early_time:.3f}s with {len(early_result)} results")

    # Wait for full graph to load
    logger.info("Waiting for full graph to load...")
    while not loader.is_full_graph_loaded():
        time.sleep(1)
        logger.info("Still loading...")

    # Test full graph query
    start_time = time.time()
    full_result = loader.load_on_demand(["transformer", "breaker", "generator"])
    full_time = time.time() - start_time
    logger.info(f"Full query completed in {full_time:.3f}s with {len(full_result)} results")

    # Test vocabulary hydration
    start_time = time.time()
    iec_terms, entsoe_terms = loader.hydrate_vocabularies()
    hydration_time = time.time() - start_time
    logger.info(f"Vocabulary hydration completed in {hydration_time:.3f}s")
    logger.info(f"Extracted {len(iec_terms)} IEC terms, {len(entsoe_terms)} ENTSOE terms")

    # Get statistics
    stats = loader.get_graph_stats()
    logger.info(f"Graph statistics: {stats}")

    # Test performance with cached queries
    start_time = time.time()
    cached_result = loader.load_on_demand(["power", "grid"])  # Same as first query
    cached_time = time.time() - start_time
    logger.info(f"Cached query completed in {cached_time:.3f}s (should be much faster)")

    logger.info("Large knowledge graph test completed successfully!")
    return True

if __name__ == "__main__":
    test_large_kg()