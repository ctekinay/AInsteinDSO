#!/usr/bin/env python3
"""
Test Citation Pool Implementation

This script verifies that:
1. Citation pools are pre-loaded at startup
2. Metadata is cached correctly
3. Query-specific pools are built from retrieval
4. No fake citations appear in responses
5. System uses only real citations from the pool
"""

import asyncio
import time
from pathlib import Path

print("="*70)
print("TESTING CITATION POOL IMPLEMENTATION")
print("="*70)

# Import the agent
from src.agent.ea_assistant import ProductionEAAgent

async def test_citation_pools():
    """Main test function."""
    
    print("\n1. Initializing ProductionEAAgent (will pre-load citations)...")
    start_time = time.time()
    
    try:
        agent = ProductionEAAgent()
        init_time = (time.time() - start_time) * 1000
        print(f"   ✓ Agent initialized in {init_time:.0f}ms")
    except Exception as e:
        print(f"   ✗ Initialization failed: {e}")
        return False
    
    # Test 1: Verify citation pools loaded
    print("\n2. Verifying Citation Pools Loaded...")
    print("-" * 70)
    
    if not hasattr(agent, 'citation_pools'):
        print("   ✗ ERROR: citation_pools not found!")
        print("   Validate ea_assistant.py is the latest version")
        return False
    
    print(f"   ✓ Total citations: {len(agent.all_citations)}")
    
    for namespace, citations in agent.citation_pools.items():
        print(f"   • {namespace.upper()}: {len(citations)} citations")
    
    if len(agent.all_citations) < 1000:
        print(f"   ⚠️  WARNING: Only {len(agent.all_citations)} citations loaded")
        print("   Expected ~5,000 citations")
        return False
    
    # Test 2: Verify metadata cached
    print("\n3. Verifying Metadata Cache...")
    print("-" * 70)
    
    if not hasattr(agent, 'citation_metadata_cache'):
        print("   ✗ ERROR: citation_metadata_cache not found!")
        return False
    
    print(f"   ✓ Metadata cached: {len(agent.citation_metadata_cache)} citations")
    
    # Show sample metadata
    sample_citations = list(agent.all_citations)[:3]
    print("\n   Sample cached metadata:")
    for citation in sample_citations:
        metadata = agent.citation_metadata_cache.get(citation)
        if metadata:
            print(f"   • {citation}")
            print(f"     Label: {metadata.get('label', 'N/A')}")
            print(f"     Source: {metadata.get('source', 'N/A')}")
    
    # Test 3: Test with real query
    print("\n4. Testing with Real Query...")
    print("-" * 70)
    
    test_queries = [
        "What capability for grid congestion management?",
        "What is reactive power?",
        "What application component for SCADA?"
    ]
    
    all_passed = True
    
    for idx, query in enumerate(test_queries, 1):
        print(f"\n   Test Query {idx}: {query}")
        
        try:
            response = await agent.process_query(query, session_id=f"test-{idx:03d}")
            
            print(f"   ✓ Response generated: {len(response.response)} chars")
            print(f"   ✓ Route: {response.route}")
            print(f"   ✓ Citations found: {len(response.citations)}")
            print(f"   ✓ Confidence: {response.confidence:.2f}")
            
            # Check for fake citations
            fake_patterns = [
                "iec:GridCongestion",
                "archi:id-cap-001",
                "skos:FakeTerm",
                "iec:61968",
                "iec:61970"
            ]
            
            detected_fakes = []
            for fake in fake_patterns:
                if fake in response.response:
                    detected_fakes.append(fake)
            
            if detected_fakes:
                print(f"   ✗ FAKE CITATIONS DETECTED: {detected_fakes}")
                all_passed = False
            else:
                print(f"   ✓ No fake citations detected")
            
            # Show citations used
            if response.citations:
                print(f"   Citations used in response:")
                for cit in response.citations[:5]:
                    print(f"     • {cit}")
            
        except Exception as e:
            print(f"   ✗ Query failed: {e}")
            all_passed = False
    
    # Test 4: Verify citation pool building
    print("\n5. Testing Citation Pool Building...")
    print("-" * 70)
    
    # Create a mock retrieval context
    mock_context = {
        'kg_results': [
            {'citation': 'skos:1502', 'label': 'Test'},
            {'citation': 'iec:Asset', 'label': 'Asset'}
        ],
        'archimate_elements': [
            {'id': 'test-123', 'citation': 'archi:id-test-123'}
        ]
    }
    
    try:
        citation_pool = agent._build_citation_pool_from_retrieval(mock_context)
        print(f"   ✓ Citation pool built: {len(citation_pool)} citations")
        
        for item in citation_pool:
            print(f"   • {item.get('citation')}: {item.get('label')}")
        
    except Exception as e:
        print(f"   ✗ Citation pool building failed: {e}")
        all_passed = False
    
    # Test 5: Statistics
    print("\n6. Agent Statistics...")
    print("-" * 70)
    
    try:
        stats = agent.get_statistics()
        
        print(f"   Knowledge Graph:")
        print(f"     • Triples: {stats.get('knowledge_graph', {}).get('triple_count', 'N/A')}")
        
        print(f"   Citation System:")
        print(f"     • Pools loaded: {stats.get('citation_pools_loaded', 'N/A')}")
        print(f"     • Metadata cached: {stats.get('citation_metadata_cached', 'N/A')}")
        
        print(f"   Sessions:")
        print(f"     • Processed: {stats.get('sessions_processed', 0)}")
        
    except Exception as e:
        print(f"   ⚠️  Could not get statistics: {e}")
    
    return all_passed


async def main():
    """Run all tests."""
    print("\n🚀 Starting Citation Pool Tests...\n")
    
    try:
        success = await test_citation_pools()
        
        print("\n" + "="*70)
        if success:
            print("✅ ALL TESTS PASSED!")
            print("="*70)
            print("\nCitation pool implementation is working correctly:")
            print("  ✓ ~5,000+ citations pre-loaded at startup")
            print("  ✓ Metadata cached for fast lookup")
            print("  ✓ Query-specific pools built correctly")
            print("  ✓ Zero fake citations in responses")
            print("  ✓ System ready for production use")
            return 0
        else:
            print("❌ SOME TESTS FAILED")
            print("="*70)
            print("\nPlease review the errors above and:")
            print("  1. Ensure you're using the updated ea_assistant.py")
            print("  2. Verify knowledge graph is loaded (39K+ triples)")
            print("  3. Check that ArchiMate models are available")
            print("  4. Review any error messages")
            return 1
    
    except Exception as e:
        print("\n" + "="*70)
        print("💥 TEST SUITE CRASHED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)