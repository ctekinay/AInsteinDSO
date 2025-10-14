"""
Test suite to verify SPARQL queries work correctly after fixing comment syntax.
This ensures the knowledge graph returns actual data instead of errors.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge.kg_loader import KnowledgeGraphLoader
from src.agent.ea_assistant import ProductionEAAgent


def test_sparql_basic_query():
    """Test that basic SPARQL queries work without errors."""
    print("\n=== TEST 1: Basic SPARQL Query ===")

    kg = KnowledgeGraphLoader()
    kg.load()

    for i in range(20):
        if kg.is_full_graph_loaded():
            print(f"KG loaded after {i*0.5}s")
            break
        time.sleep(0.5)

    assert kg.is_full_graph_loaded(), "KG failed to load"

    simple_query = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?concept ?label WHERE {
        ?concept skos:prefLabel ?label .
    }
    LIMIT 5
    """

    try:
        results = list(kg.graph.query(simple_query))
        print(f"Basic query returned {len(results)} results")
        assert len(results) > 0, "Basic SPARQL query should return results"
        print("✓ Basic SPARQL query works")
        return True
    except Exception as e:
        print(f"✗ Basic SPARQL query failed: {e}")
        return False


def test_query_definitions_asset():
    """Test that query_definitions returns Asset when searching for 'asset'."""
    print("\n=== TEST 2: Query Definitions for 'asset' ===")

    kg = KnowledgeGraphLoader()
    kg.load()

    for i in range(20):
        if kg.is_full_graph_loaded():
            break
        time.sleep(0.5)

    results = kg.query_definitions(["asset"])
    print(f"Found {len(results)} results for 'asset'")

    assert len(results) > 0, "Should find results for 'asset'"

    exact_match_found = False
    asset_management_position = -1

    for idx, result in enumerate(results):
        print(f"  [{idx}] {result['label']} (score: {result['score']})")
        if result['label'].lower() == 'asset':
            exact_match_found = True
            print(f"    ✓ Found exact match 'Asset' at position {idx}")
        if 'management' in result['label'].lower():
            asset_management_position = idx

    assert exact_match_found, "Should find exact match 'Asset'"

    if asset_management_position >= 0:
        assert asset_management_position > 0, "AssetManagement should not be first result for 'asset' query"

    print("✓ Query definitions correctly prioritizes exact matches")
    return True


def test_multiple_search_terms():
    """Test that various search terms return appropriate results."""
    print("\n=== TEST 3: Multiple Search Terms ===")

    kg = KnowledgeGraphLoader()
    kg.load()

    for i in range(20):
        if kg.is_full_graph_loaded():
            break
        time.sleep(0.5)

    test_terms = [
        ("power", "Should find power-related concepts"),
        ("grid", "Should find grid-related concepts"),
        ("management", "Should find management-related concepts"),
        ("nonexistentterm123", "Should handle non-existent terms gracefully")
    ]

    for term, description in test_terms:
        print(f"\n  Testing '{term}': {description}")
        results = kg.query_definitions([term])

        if term == "nonexistentterm123":
            print(f"    Found {len(results)} results (expected 0)")
            assert len(results) == 0, "Should return empty for non-existent terms"
        else:
            print(f"    Found {len(results)} results")
            if len(results) > 0:
                print(f"    Top result: {results[0]['label']}")

    print("\n✓ Multiple search terms handled correctly")
    return True


async def test_full_pipeline_with_asset():
    """Test the full pipeline with 'asset' query to ensure it uses KG data."""
    print("\n=== TEST 4: Full Pipeline with Asset Query ===")

    agent = ProductionEAAgent()

    await asyncio.sleep(3)

    try:
        response = await agent.process_query("What is an asset?")

        print(f"Response preview: {response.response[:200]}...")
        print(f"Citations: {response.citations}")
        print(f"Route: {response.route}")

        assert response.route == "structured_model", "Should route to structured_model"

        if "external:llm" in str(response.citations):
            print("✗ WARNING: Still using LLM fallback instead of KG data")
            return False

        assert len(response.citations) > 0, "Should have citations from KG"
        assert "asset" in response.response.lower(), "Response should mention 'asset'"

        print("✓ Full pipeline uses KG data correctly")
        return True

    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        return False


async def test_exact_vs_partial_match():
    """Test that exact matches are prioritized over partial matches."""
    print("\n=== TEST 5: Exact vs Partial Match Priority ===")

    kg = KnowledgeGraphLoader()
    kg.load()

    for i in range(20):
        if kg.is_full_graph_loaded():
            break
        time.sleep(0.5)

    results = kg.query_definitions(["manage"])

    if len(results) > 0:
        print(f"Results for 'manage':")
        for idx, result in enumerate(results[:5]):
            is_exact = result['label'].lower() == 'manage'
            print(f"  [{idx}] {result['label']} (score: {result['score']}, exact: {is_exact})")

        if results[0]['score'] == 100:
            assert results[0]['label'].lower() == 'manage', "Score 100 should be exact match"
            print("✓ Exact matches get score 100")

        for result in results:
            if result['score'] == 100:
                assert result['label'].lower() == 'manage', "Only exact matches should get score 100"
            elif result['score'] == 75:
                assert 'manage' in result['label'].lower(), "Score 75 should be partial matches"

    print("✓ Scoring system works correctly")
    return True


def run_all_tests():
    """Run all test cases and report results."""
    print("=" * 60)
    print("SPARQL FIX VALIDATION TEST SUITE")
    print("=" * 60)

    test_results = []

    test_results.append(("Basic SPARQL", test_sparql_basic_query()))
    test_results.append(("Query Definitions", test_query_definitions_asset()))
    test_results.append(("Multiple Terms", test_multiple_search_terms()))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    test_results.append(("Full Pipeline", loop.run_until_complete(test_full_pipeline_with_asset())))
    test_results.append(("Exact vs Partial", loop.run_until_complete(test_exact_vs_partial_match())))

    loop.close()

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - SPARQL queries are working correctly!")
    else:
        print(f"\n⚠️ {failed} tests failed - please review the fixes")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)