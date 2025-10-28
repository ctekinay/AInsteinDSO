#!/usr/bin/env python3
"""Verify ADRs are properly included in embeddings."""

import logging
from pathlib import Path
from src.agents.ea_assistant import ProductionEAAgent

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def verify_adr_embeddings():
    """Check if ADRs are in the embedding cache."""
    
    print("\n🔍 Verifying ADR Embeddings...\n")
    
    # Initialize agent
    agent = ProductionEAAgent()
    
    # Check ADR indexer
    if agent.adr_indexer:
        adr_count = len(agent.adr_indexer.adrs)
        print(f"✅ ADR Indexer: {adr_count} ADRs loaded")
        for adr in agent.adr_indexer.adrs[:3]:  # Show first 3
            print(f"   - ADR-{adr.number}: {adr.title[:50]}...")
    else:
        print("❌ ADR Indexer not initialized")
        return
    
    # Check embeddings
    if agent.embedding_agent and agent.embedding_agent.embeddings:
        metadata = agent.embedding_agent.embeddings.get('metadata', [])
        
        # Count by source
        source_counts = {}
        for m in metadata:
            source = m.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"\n📊 Embedding Statistics:")
        print(f"   Total embeddings: {len(metadata)}")
        for source, count in sorted(source_counts.items()):
            status = "✅" if count > 0 else "❌"
            print(f"   {status} {source}: {count}")
        
        # Check ADR specifics
        adr_embeddings = [m for m in metadata if m.get('source') == 'adr']
        if adr_embeddings:
            print(f"\n✅ ADR Embeddings Found: {len(adr_embeddings)}")
            for m in adr_embeddings[:3]:  # Show first 3
                print(f"   - ADR-{m.get('adr_number')}: {m.get('title', 'N/A')[:50]}...")
        else:
            print("\n❌ NO ADR EMBEDDINGS FOUND!")
            print("   Run: rm -rf data/embeddings/*.pkl")
            print("   Then restart the application to rebuild embeddings")
    else:
        print("❌ Embedding agent not initialized")
    
    # Test a query
    print("\n🧪 Testing ADR Query...")
    try:
        result = agent.process_query_sync("What does ADR 0025 say about interfaces?")
        if result:
            print(f"✅ Query successful: {result.get('response', '')[:200]}...")
        else:
            print("❌ Query failed")
    except Exception as e:
        print(f"❌ Query error: {e}")

if __name__ == "__main__":
    verify_adr_embeddings()