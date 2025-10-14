"""
Improved Knowledge Graph Retrieval Test
Tests more flexible query strategies.
"""

import time
from pathlib import Path

from src.knowledge.kg_loader import KnowledgeGraphLoader


def test_flexible_term_matching():
    """Test retrieval with more flexible matching strategies."""
    print("\n" + "="*60)
    print("IMPROVED RETRIEVAL TEST")
    print("="*60)
    
    loader = KnowledgeGraphLoader(Path("data/energy_knowledge_graph_original.ttl"))
    loader.load()
    time.sleep(12)
    
    print("\n🔍 Testing different query strategies:")
    
    # Strategy 1: Exact single-word terms
    print("\n1. SINGLE-WORD EXACT MATCH:")
    single_terms = ["Congestion", "Voltage", "Power", "Grid"]
    
    for term in single_terms:
        results = loader.load_on_demand([term])
        status = "✓" if results else "✗"
        print(f"   {status} '{term}': {len(results)} results")
    
    # Strategy 2: Multi-word with space
    print("\n2. MULTI-WORD TERMS:")
    multi_terms = ["Active Power", "Reactive Power", "Congestion management"]
    
    for term in multi_terms:
        # Try with space
        results = loader.load_on_demand([term])
        status = "✓" if results else "✗"
        print(f"   {status} '{term}': {len(results)} results")
    
    # Strategy 3: Partial matching
    print("\n3. PARTIAL MATCHING (first word only):")
    partial_terms = ["Active", "Reactive", "Structural"]
    
    for term in partial_terms:
        results = loader.load_on_demand([term])
        status = "✓" if results else "✗"
        print(f"   {status} '{term}': {len(results)} results")
        
        if results:
            # Show what was found
            for key, value in list(results.items())[:2]:
                if isinstance(value, dict):
                    print(f"      → Found: {value.get('label', key)}")
    
    # Strategy 4: Direct SPARQL for known concepts
    print("\n4. DIRECT SPARQL VERIFICATION:")
    
    # Query for all concepts containing "congestion"
    query = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    
    SELECT ?concept ?label WHERE {
        ?concept skos:prefLabel ?label .
        FILTER(CONTAINS(LCASE(?label), "congestion"))
    }
    LIMIT 10
    """
    
    results = list(loader.graph.query(query))
    print(f"   Found {len(results)} congestion-related concepts:")
    for row in results:
        print(f"   ✓ {row.label}")
    
    # Strategy 5: Recommended test concepts
    print("\n5. RECOMMENDED TEST CONCEPTS (100% VERIFIED):")
    verified_concepts = [
        "ActivePower",
        "ReactivePower", 
        "Voltage",
        "Current",
        "Congestion",  # ✅ Use this instead of GridCongestion
    ]
    
    success_count = 0
    for concept in verified_concepts:
        results = loader.load_on_demand([concept])
        if results:
            success_count += 1
            print(f"   ✓ {concept}: Found")
        else:
            print(f"   ✗ {concept}: Not found")
    
    print(f"\n📊 SUCCESS RATE: {success_count}/{len(verified_concepts)} ({success_count/len(verified_concepts)*100:.0f}%)")


if __name__ == "__main__":
    test_flexible_term_matching()