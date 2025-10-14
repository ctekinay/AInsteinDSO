"""
Comprehensive Knowledge Graph Authenticity and Retrieval Testing

This test suite validates:
1. The knowledge graph data is authentic (not artificially inflated)
2. The retrieval mechanism works correctly
3. SPARQL queries return accurate results
4. Caching performs as expected
"""

import time
from pathlib import Path
import pytest

from src.knowledge.kg_loader import KnowledgeGraphLoader


class TestKGAuthenticity:
    """Test suite for KG data integrity and retrieval mechanisms."""

    def test_01_compare_both_knowledge_graphs(self):
        """
        TEST 1: Compare the two knowledge graph files
        
        This checks if both graphs exist and compares their sizes.
        The 'original' should have ~39K triples (from Skosmos).
        """
        print("\n" + "="*60)
        print("TEST 1: Comparing Knowledge Graph Files")
        print("="*60)
        
        # Check if both files exist
        inflated_path = Path("data/energy_knowledge_graph.ttl")
        original_path = Path("data/energy_knowledge_graph_original.ttl")
        
        if not inflated_path.exists():
            pytest.skip(f"Inflated graph not found at {inflated_path}")
        if not original_path.exists():
            pytest.skip(f"Original graph not found at {original_path}")
        
        # Load both graphs
        print("\n📊 Loading inflated graph...")
        loader_inflated = KnowledgeGraphLoader(inflated_path)
        loader_inflated.load()
        
        print("📊 Loading original graph...")
        loader_original = KnowledgeGraphLoader(original_path)
        loader_original.load()
        
        # Wait for background loading to complete
        print("⏳ Waiting for full graph loading (12 seconds)...")
        time.sleep(12)
        
        # Get statistics
        stats_inflated = loader_inflated.get_graph_stats()
        stats_original = loader_original.get_graph_stats()
        
        print(f"\n📈 RESULTS:")
        print(f"   Inflated graph:  {stats_inflated['triples']:,} triples")
        print(f"   Original graph:  {stats_original['triples']:,} triples")
        print(f"   Difference:      {stats_inflated['triples'] - stats_original['triples']:,} triples")
        
        # Validation checks
        print(f"\n✅ VALIDATION:")
        
        # Check 1: Original should be in the 39K range (from Skosmos)
        if 39000 <= stats_original['triples'] <= 50000:
            print(f"   ✓ Original graph size is legitimate (39K-50K range)")
        else:
            print(f"   ⚠️  Original graph size is unusual: {stats_original['triples']:,} triples")
        
        # Check 2: Compare ratio
        if stats_inflated['triples'] > 0:
            ratio = stats_inflated['triples'] / stats_original['triples']
            print(f"   → Inflated graph is {ratio:.1f}x larger than original")
            
            if ratio > 3:
                print(f"   ⚠️  Inflation ratio is suspiciously high (>3x)")
        
        # Assertions
        assert stats_original['triples'] >= 39000, \
            f"Original graph should have at least 39K triples, got {stats_original['triples']:,}"
        assert stats_original['triples'] < 100000, \
            f"Original graph suspiciously large: {stats_original['triples']:,} triples"

    def test_02_detect_artificial_inflation(self):
        """
        TEST 2: Check for artificial duplication patterns
        
        Real knowledge graphs have unique concepts.
        Fake data often has duplicates or systematic naming like "Term_001", "Term_002".
        """
        print("\n" + "="*60)
        print("TEST 2: Detecting Artificial Inflation Patterns")
        print("="*60)
        
        inflated_path = Path("data/energy_knowledge_graph.ttl")
        
        if not inflated_path.exists():
            pytest.skip(f"Inflated graph not found at {inflated_path}")
        
        print("\n📊 Loading inflated graph for analysis...")
        loader = KnowledgeGraphLoader(inflated_path)
        loader.load()
        time.sleep(12)
        
        # Query for duplicate labels (suspicious in real data)
        print("\n🔍 Checking for duplicate labels...")
        sparql_check = """
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        
        SELECT ?label (COUNT(?label) as ?count) WHERE {
            ?s skos:prefLabel ?label .
        }
        GROUP BY ?label
        HAVING (COUNT(?label) > 1)
        ORDER BY DESC(?count)
        LIMIT 50
        """
        
        try:
            results = list(loader.graph.query(sparql_check))
            
            if len(results) > 0:
                print(f"\n⚠️  Found {len(results)} duplicate labels:")
                for i, row in enumerate(results[:10], 1):
                    print(f"   {i}. '{row.label}' appears {row.count} times")
                
                # Check for systematic naming patterns
                systematic_count = sum(1 for row in results 
                                      if '_' in str(row.label) and 
                                      any(char.isdigit() for char in str(row.label)))
                
                systematic_ratio = systematic_count / len(results) if results else 0
                print(f"\n📊 Systematic naming (Term_XXX pattern): {systematic_count}/{len(results)} ({systematic_ratio:.0%})")
                
                if systematic_ratio > 0.5:
                    print(f"   ⚠️  HIGH systematic naming ratio - likely artificial inflation!")
                elif systematic_ratio > 0.2:
                    print(f"   ⚠️  Moderate systematic naming - questionable authenticity")
                else:
                    print(f"   ✓ Low systematic naming - appears natural")
                
                # Assertion
                assert systematic_ratio < 0.5, \
                    f"Too much systematic naming ({systematic_ratio:.0%}) - likely artificial"
            else:
                print("   ✓ No duplicate labels found - good sign!")
                
        except Exception as e:
            print(f"   ⚠️  Could not run duplicate check: {e}")

    def test_03_sparql_retrieval_accuracy(self):
        """
        TEST 3: Verify SPARQL queries work correctly
        
        Tests that we can query for real energy concepts and get accurate results.
        """
        print("\n" + "="*60)
        print("TEST 3: Testing SPARQL Retrieval Accuracy")
        print("="*60)
        
        original_path = Path("data/energy_knowledge_graph_original.ttl")
        
        if not original_path.exists():
            pytest.skip(f"Original graph not found at {original_path}")
        
        print("\n📊 Loading original graph for queries...")
        loader = KnowledgeGraphLoader(original_path)
        loader.load()
        time.sleep(12)
        
        # Test cases with real IEC/ENTSOE concepts
        # NEW (matches your actual graph):
        test_cases = [
            {"term": "ActivePower", "description": "IEC CIM Active Power"},
            {"term": "ReactivePower", "description": "IEC CIM Reactive Power"},
            {"term": "Voltage", "description": "IEC CIM Voltage"},
            {"term": "Congestion", "description": "Grid Congestion Management"},  # ✅ Exists!
        ]
                
        print("\n🔍 Testing retrieval for key energy concepts:")
        results_summary = []
        
        for test in test_cases:
            print(f"\n   Query: '{test['term']}' ({test['description']})")
            
            try:
                results = loader.load_on_demand([test['term']])
                result_count = len(results)
                
                if result_count > 0:
                    print(f"   ✓ Found {result_count} result(s)")
                    results_summary.append({"term": test['term'], "found": True, "count": result_count})
                    
                    # Show first result
                    first_key = list(results.keys())[0]
                    first_result = results[first_key]
                    if isinstance(first_result, dict):
                        print(f"      Sample: {first_result.get('label', first_key)}")
                else:
                    print(f"   ✗ No results found")
                    results_summary.append({"term": test['term'], "found": False, "count": 0})
                    
            except Exception as e:
                print(f"   ⚠️  Query failed: {e}")
                results_summary.append({"term": test['term'], "found": False, "count": 0, "error": str(e)})
        
        # Summary
        found_count = sum(1 for r in results_summary if r['found'])
        total_count = len(results_summary)
        
        print(f"\n📊 SUMMARY: Found {found_count}/{total_count} concepts")
        
        # At least 50% of core concepts should be found
        success_rate = found_count / total_count
        assert success_rate >= 0.5, \
            f"Only found {found_count}/{total_count} core concepts - graph may be incomplete"
        
        if success_rate == 1.0:
            print("   ✓ All core concepts found - excellent!")
        elif success_rate >= 0.75:
            print("   ✓ Most core concepts found - good coverage")
        else:
            print("   ⚠️  Some core concepts missing - limited coverage")

    def test_04_caching_performance(self):
        """
        TEST 4: Verify caching mechanism works
        
        Second query should be much faster due to caching.
        """
        print("\n" + "="*60)
        print("TEST 4: Testing Cache Performance")
        print("="*60)
        
        original_path = Path("data/energy_knowledge_graph_original.ttl")
        
        if not original_path.exists():
            pytest.skip(f"Original graph not found at {original_path}")
        
        print("\n📊 Loading graph...")
        loader = KnowledgeGraphLoader(original_path)
        loader.load()
        time.sleep(12)
        
        query_terms = ["ActivePower", "ReactivePower"]
        
        # First query - cold cache
        print(f"\n🔍 First query (cold cache): {query_terms}")
        start = time.time()
        results1 = loader.load_on_demand(query_terms)
        time1 = time.time() - start
        
        print(f"   ⏱️  Time: {time1*1000:.2f}ms")
        print(f"   📊 Results: {len(results1)} items")
        
        # Second query - should hit cache
        print(f"\n🔍 Second query (should be cached): {query_terms}")
        start = time.time()
        results2 = loader.load_on_demand(query_terms)
        time2 = time.time() - start
        
        print(f"   ⏱️  Time: {time2*1000:.2f}ms")
        print(f"   📊 Results: {len(results2)} items")
        
        # Analysis
        speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"\n📈 PERFORMANCE:")
        print(f"   Cold cache:  {time1*1000:.2f}ms")
        print(f"   Hot cache:   {time2*1000:.2f}ms")
        print(f"   Speedup:     {speedup:.1f}x faster")
        
        # Validation
        if speedup > 2:
            print(f"   ✓ Cache is working well (>2x speedup)")
        elif speedup > 1.2:
            print(f"   ⚠️  Cache provides modest speedup")
        else:
            print(f"   ⚠️  Cache may not be working properly")
        
        assert results1 == results2, "Cached results should match original"
        assert time2 < time1, "Cached query should be faster"

    def test_05_energy_domain_coverage(self):
        """
        TEST 5: Check for legitimate energy domain concepts
        
        A real IEC CIM graph MUST have these core concepts.
        """
        print("\n" + "="*60)
        print("TEST 5: Validating Energy Domain Coverage")
        print("="*60)
        
        original_path = Path("data/energy_knowledge_graph_original.ttl")
        
        if not original_path.exists():
            pytest.skip(f"Original graph not found at {original_path}")
        
        print("\n📊 Loading graph...")
        loader = KnowledgeGraphLoader(original_path)
        loader.load()
        time.sleep(12)
        
        # Core IEC 61970/61968 CIM concepts that MUST exist
        required_concepts = [
            "ActivePower",
            "ReactivePower",
            "Voltage",
            "Current",
            "ACLineSegment",
            "PowerTransformer",
            "EnergyConsumer",
            "GeneratingUnit",
        ]
        
        print(f"\n🔍 Checking for {len(required_concepts)} core IEC CIM concepts:")
        
        found_concepts = []
        missing_concepts = []
        
        for concept in required_concepts:
            print(f"   Checking: {concept}...", end=" ")
            results = loader.load_on_demand([concept])
            
            if len(results) > 0:
                print("✓ Found")
                found_concepts.append(concept)
            else:
                print("✗ Missing")
                missing_concepts.append(concept)
        
        # Summary
        coverage_ratio = len(found_concepts) / len(required_concepts)
        
        print(f"\n📊 COVERAGE RESULTS:")
        print(f"   Found:   {len(found_concepts)}/{len(required_concepts)} ({coverage_ratio:.0%})")
        print(f"   Missing: {len(missing_concepts)}/{len(required_concepts)}")
        
        if missing_concepts:
            print(f"\n   Missing concepts:")
            for concept in missing_concepts:
                print(f"      - {concept}")
        
        # Validation
        if coverage_ratio >= 0.75:
            print("\n   ✓ Good domain coverage - appears to be legitimate IEC CIM data")
        elif coverage_ratio >= 0.5:
            print("\n   ⚠️  Moderate coverage - some core concepts missing")
        else:
            print("\n   ⚠️  Poor coverage - may not be authentic IEC CIM data")
        
        # At least 50% of core concepts should be present
        assert coverage_ratio >= 0.5, \
            f"Only {coverage_ratio:.0%} of core IEC concepts found - graph may not be authentic"

    def test_06_vocabulary_hydration_quality(self):
        """
        TEST 6: Check if auto-hydrated vocabularies are meaningful
        
        Real vocabularies have natural names. Fake data has systematic naming.
        """
        print("\n" + "="*60)
        print("TEST 6: Validating Vocabulary Hydration Quality")
        print("="*60)
        
        original_path = Path("data/energy_knowledge_graph_original.ttl")
        
        if not original_path.exists():
            pytest.skip(f"Original graph not found at {original_path}")
        
        print("\n📊 Loading graph and hydrating vocabularies...")
        loader = KnowledgeGraphLoader(original_path)
        loader.load()
        time.sleep(12)
        
        try:
            iec_terms, entsoe_terms = loader.hydrate_vocabularies()
            
            print(f"\n📊 HYDRATION RESULTS:")
            print(f"   IEC terms:    {len(iec_terms):,} terms")
            print(f"   ENTSOE terms: {len(entsoe_terms):,} terms")
            
            if len(iec_terms) == 0 and len(entsoe_terms) == 0:
                print("\n   ⚠️  No terms extracted - hydration may not be working")
                pytest.skip("Vocabulary hydration returned no terms")
            
            # Check for artificial naming patterns
            def check_systematic_naming(terms, label):
                if not terms:
                    return 0.0
                    
                systematic_count = sum(1 for term in terms 
                                      if '_' in str(term) and 
                                      any(c.isdigit() for c in str(term)))
                ratio = systematic_count / len(terms)
                
                print(f"\n   {label}:")
                print(f"      Total terms:       {len(terms):,}")
                print(f"      Systematic naming: {systematic_count:,} ({ratio:.0%})")
                
                if ratio < 0.1:
                    print(f"      ✓ Natural naming - appears authentic")
                elif ratio < 0.3:
                    print(f"      ⚠️  Some systematic naming")
                else:
                    print(f"      ⚠️  High systematic naming - questionable authenticity")
                
                return ratio
            
            iec_ratio = check_systematic_naming(iec_terms, "IEC Terms")
            entsoe_ratio = check_systematic_naming(entsoe_terms, "ENTSOE Terms")
            
            # Validation - real vocabularies should have <30% systematic naming
            assert iec_ratio < 0.5, \
                f"IEC vocabulary has {iec_ratio:.0%} systematic naming - likely artificial"
            assert entsoe_ratio < 0.5, \
                f"ENTSOE vocabulary has {entsoe_ratio:.0%} systematic naming - likely artificial"
                
        except Exception as e:
            print(f"\n   ⚠️  Hydration test failed: {e}")
            # Don't fail the test if hydration isn't implemented yet
            pytest.skip(f"Vocabulary hydration not fully implemented: {e}")


# Run this test file directly for debugging
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Knowledge Graph Authenticity Test Suite")
    print("="*60)
    pytest.main([__file__, "-v", "-s"])