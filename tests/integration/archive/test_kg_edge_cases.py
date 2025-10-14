"""
Edge case tests to verify the coding agent actually fixed the issues
and isn't just making tests pass with fake data.
"""

import asyncio
import sys
import hashlib
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge.kg_loader import KnowledgeGraphLoader
from src.agent.ea_assistant import ProductionEAAgent


def test_actual_asset_definition_content():
    """Verify the ACTUAL content of Asset definition matches TTL file."""
    print("\n=== EDGE CASE 1: Verify Actual Asset Definition ===")
    
    from pathlib import Path
    kg_path = Path(__file__).parent.parent / "data" / "energy_knowledge_graph.ttl"
    kg = KnowledgeGraphLoader(kg_path)
    kg.load()
    
    import time
    for i in range(20):
        if kg.is_full_graph_loaded():
            break
        time.sleep(0.5)
    
    results = kg.query_definitions(["asset"])
    
    asset_found = False
    for result in results:
        if result['label'].lower() == 'asset':
            asset_found = True
            definition = result.get('definition', '')
            print(f"Found Asset with definition: {definition}")
            
            # Check for the EXACT definition from your TTL
            expected_phrases = [
                "Entity of value",
                "individuals or organizations"
            ]
            
            for phrase in expected_phrases:
                assert phrase in definition, f"Definition should contain '{phrase}' from actual TTL"
            
            # Make sure it's NOT a generic placeholder
            bad_phrases = [
                "physical or logical component",
                "grid infrastructure",
                "energy distribution"
            ]
            
            for bad in bad_phrases:
                assert bad not in definition, f"Definition contains suspicious generic text: '{bad}'"
            
            print(f"✓ Actual definition verified: {definition[:100]}...")
            break
    
    assert asset_found, "Must find exact 'Asset' concept, not just AssetManagement"
    return True


def test_random_nonexistent_terms():
    """Test with random terms that definitely don't exist."""
    print("\n=== EDGE CASE 2: Random Non-Existent Terms ===")
    
    from pathlib import Path
    kg_path = Path(__file__).parent.parent / "data" / "energy_knowledge_graph.ttl"
    kg = KnowledgeGraphLoader(kg_path)
    kg.load()
    
    import time
    for i in range(20):
        if kg.is_full_graph_loaded():
            break
        time.sleep(0.5)
    
    # Generate truly random terms that can't be hardcoded
    random_terms = [
        f"xyz{random.randint(10000, 99999)}",
        f"test_{hashlib.md5(str(random.random()).encode()).hexdigest()[:8]}",
        "definitelynotinknowledgegraph123"
    ]
    
    for term in random_terms:
        print(f"  Testing random term: {term}")
        results = kg.query_definitions([term])
        
        assert len(results) == 0, f"Random term '{term}' should return no results, got {len(results)}"
        
        # Make sure it's not returning fake data
        for r in results:
            assert term not in r['label'].lower(), "Should not fabricate results for non-existent terms"
    
    print("✓ Non-existent terms correctly return empty")
    return True


def test_citation_format_consistency():
    """Verify citations follow consistent format from actual URIs."""
    print("\n=== EDGE CASE 3: Citation Format Validation ===")
    
    from pathlib import Path
    kg_path = Path(__file__).parent.parent / "data" / "energy_knowledge_graph.ttl"
    kg = KnowledgeGraphLoader(kg_path)
    kg.load()
    
    import time
    for i in range(20):
        if kg.is_full_graph_loaded():
            break
        time.sleep(0.5)
    
    # Test multiple real terms
    test_terms = ["asset", "power", "management", "grid"]
    all_citations = []
    
    for term in test_terms:
        results = kg.query_definitions([term])
        for result in results[:3]:
            citation = result.get('citation_id', '')
            if citation:
                all_citations.append(citation)
                print(f"  Term '{term}' → Citation: {citation}")
                
                # Citations should NOT be generic placeholders
                assert not citation.startswith('archi:id-'), f"Suspicious ArchiMate citation: {citation}"
                assert not citation == 'iec:61968', f"Suspicious generic IEC citation: {citation}"
                
                # Should match actual namespace patterns from TTL
                valid_patterns = [
                    'default',  # Your TTL uses default1:, default8:, etc
                    'BSI',
                    'IEC',
                    'CIM'
                ]
                
                has_valid_pattern = any(pattern in citation for pattern in valid_patterns)
                assert has_valid_pattern, f"Citation '{citation}' doesn't match expected patterns"
    
    # Check we got diverse citations, not all the same
    unique_citations = set(all_citations)
    assert len(unique_citations) > 1, "Should have diverse citations, not all the same"
    
    print(f"✓ Found {len(unique_citations)} unique citation formats")
    return True


async def test_different_queries_different_results():
    """Verify different queries return different, appropriate results."""
    print("\n=== EDGE CASE 4: Query Differentiation ===")
    
    from pathlib import Path
    kg_path = Path(__file__).parent.parent / "data" / "energy_knowledge_graph.ttl"
    models_path = Path(__file__).parent.parent / "data" / "models"
    vocab_path = Path(__file__).parent.parent / "config" / "vocabularies.json"

    agent = ProductionEAAgent(
        kg_path=str(kg_path),
        models_path=str(models_path),
        vocab_path=str(vocab_path)
    )
    
    await asyncio.sleep(3)
    
    queries = [
        "What is an asset?",
        "What is power?",
        "What is a grid?"
    ]
    
    all_responses = []
    all_citations = []
    
    for query in queries:
        try:
            response = await agent.process_query(query)
            all_responses.append(response.response[:200])
            all_citations.append(response.citations)
            
            print(f"\nQuery: {query}")
            print(f"Response preview: {response.response[:100]}...")
            print(f"Citations: {response.citations}")
            
            # Each query should have relevant content
            key_term = query.split()[-1].rstrip('?').lower()
            assert key_term in response.response.lower(), f"Response should mention '{key_term}'"
            
        except Exception as e:
            print(f"Error processing '{query}': {e}")
            return False
    
    # Responses should be different for different queries
    for i in range(len(all_responses)):
        for j in range(i+1, len(all_responses)):
            similarity = len(set(all_responses[i].split()) & set(all_responses[j].split()))
            total_words = len(set(all_responses[i].split()) | set(all_responses[j].split()))
            similarity_ratio = similarity / total_words if total_words > 0 else 1
            
            assert similarity_ratio < 0.8, f"Responses too similar ({similarity_ratio:.1%})"
    
    print("✓ Different queries produce appropriately different responses")
    return True


def test_sparql_with_special_characters():
    """Test SPARQL doesn't break with special characters."""
    print("\n=== EDGE CASE 5: Special Characters in Queries ===")
    
    from pathlib import Path
    kg_path = Path(__file__).parent.parent / "data" / "energy_knowledge_graph.ttl"
    kg = KnowledgeGraphLoader(kg_path)
    kg.load()
    
    import time
    for i in range(20):
        if kg.is_full_graph_loaded():
            break
        time.sleep(0.5)
    
    # Terms with characters that could break SPARQL
    tricky_terms = [
        "asset's",
        "asset-management",
        "power/energy",
        "grid (electric)",
        '"quoted"',
        "100%"
    ]
    
    for term in tricky_terms:
        print(f"  Testing term with special chars: {term}")
        try:
            results = kg.query_definitions([term])
            print(f"    → Returned {len(results)} results without error")
            # Should handle gracefully, not crash
        except Exception as e:
            print(f"    ✗ Failed on '{term}': {e}")
            return False
    
    print("✓ Special characters handled gracefully")
    return True


def test_verify_no_hardcoded_responses():
    """Verify the system isn't using hardcoded responses."""
    print("\n=== EDGE CASE 6: No Hardcoded Responses ===")
    
    from pathlib import Path
    kg_path = Path(__file__).parent.parent / "data" / "energy_knowledge_graph.ttl"
    kg = KnowledgeGraphLoader(kg_path)
    kg.load()
    
    import time
    for i in range(20):
        if kg.is_full_graph_loaded():
            break
        time.sleep(0.5)
    
    # Check source code for hardcoded responses
    kg_loader_path = Path(__file__).parent.parent / "src" / "knowledge" / "kg_loader.py"
    
    if kg_loader_path.exists():
        with open(kg_loader_path, 'r') as f:
            source = f.read()
            
            # Check for suspicious hardcoded values
            suspicious_patterns = [
                'archi:id-cap-001',
                'iec:61968',
                '"Grid Congestion Management"',
                'infrastructure element'
            ]
            
            for pattern in suspicious_patterns:
                assert pattern not in source, f"Found suspicious hardcoded value: {pattern}"
    
    print("✓ No hardcoded responses detected")
    return True


def run_edge_case_tests():
    """Run all edge case tests."""
    print("=" * 60)
    print("EDGE CASE VALIDATION TEST SUITE")
    print("=" * 60)
    
    test_results = []
    
    test_results.append(("Actual Asset Definition", test_actual_asset_definition_content()))
    test_results.append(("Random Terms", test_random_nonexistent_terms()))
    test_results.append(("Citation Format", test_citation_format_consistency()))
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    test_results.append(("Query Differentiation", loop.run_until_complete(test_different_queries_different_results())))
    
    loop.close()
    
    test_results.append(("Special Characters", test_sparql_with_special_characters()))
    test_results.append(("No Hardcoded", test_verify_no_hardcoded_responses()))
    
    print("\n" + "=" * 60)
    print("EDGE CASE TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    failed = len(test_results) - passed
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\n⚠️ EDGE CASES FAILED - The implementation may be faking results!")
        print("The coding agent needs to fix the actual issues, not just make tests pass.")
    else:
        print("\n✅ ALL EDGE CASES PASSED - Implementation appears genuine!")
    
    return failed == 0


if __name__ == "__main__":
    success = run_edge_case_tests()
    sys.exit(0 if success else 1)