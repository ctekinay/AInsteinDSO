#!/usr/bin/env python3
import asyncio
import time
from pathlib import Path
from src.agent.ea_assistant import ProductionEAAgent
from src.knowledge.kg_loader import KnowledgeGraphLoader

async def test_real_queries():
    """Test if queries actually return real data from our sources."""
    
    print("=== REAL QUERY TEST ===\n")
    
    # First check KG directly
    print("1. Testing Knowledge Graph directly...")
    kg = KnowledgeGraphLoader(Path("data/energy_knowledge_graph.ttl"))
    kg.load()
    
    # Wait for load
    for i in range(20):
        if kg.is_full_graph_loaded():
            print(f"   KG loaded after {i*0.5}s")
            break
        time.sleep(0.5)
    
    # Test query for asset
    asset_results = kg.query_definitions(["asset"])
    print(f"   KG returned {len(asset_results)} results for 'asset'")
    
    if asset_results:
        first = asset_results[0]
        print(f"   First result: {first['label']} (citation: {first['citation_id']})")
        print(f"   Has definition: {'Yes' if first.get('definition') else 'No'}")
    
    # Now test full pipeline
    print("\n2. Testing Full Pipeline...")
    agent = ProductionEAAgent()
    
    # Wait for everything to load
    await asyncio.sleep(5)
    
    # Test queries that MUST work if system is functioning
    test_cases = [
        ("What is reactive power?", "Should find IEC definition"),
        ("Show capabilities", "Should find ArchiMate capabilities"),
        ("What is TOGAF Phase B?", "Should find TOGAF method info")
    ]
    
    results = []
    
    for query, expected in test_cases:
        print(f"\n   Query: {query}")
        print(f"   Expected: {expected}")
        
        try:
            response = await agent.process_query(query)
            
            # Check if we got a real response or fallback
            if "external:llm" in str(response.citations):
                print(f"   ❌ FAILED: Fell back to LLM")
                results.append(False)
            elif not response.citations:
                print(f"   ❌ FAILED: No citations at all")
                results.append(False)
            elif response.response and len(response.response) > 50:
                print(f"   ✅ SUCCESS: Got response with citations: {response.citations[:2]}")
                print(f"      Preview: {response.response[:100]}...")
                results.append(True)
            else:
                print(f"   ⚠️ SUSPICIOUS: Response too short or empty")
                results.append(False)
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append(False)
    
    # Summary
    print(f"\n=== RESULTS ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if sum(results) == 0:
        print("❌ SYSTEM IS NOT WORKING - All queries failed")
        return False
    elif sum(results) < len(results):
        print("⚠️ SYSTEM PARTIALLY WORKING - Some queries failed")
        return False
    else:
        print("✅ SYSTEM WORKING - All queries succeeded")
        return True

if __name__ == "__main__":
    result = asyncio.run(test_real_queries())
    exit(0 if result else 1)