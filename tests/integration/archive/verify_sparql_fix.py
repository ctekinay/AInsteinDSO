#!/usr/bin/env python3
"""
Quick Verification Script for SPARQL Fix

Run this BEFORE and AFTER applying the fix to compare results.
"""

from pathlib import Path
import time

print("="*70)
print("SPARQL FIX VERIFICATION")
print("="*70)

# Load the knowledge graph
print("\n1. Loading Knowledge Graph...")
from src.knowledge.kg_loader import KnowledgeGraphLoader

kg = KnowledgeGraphLoader(Path("data/energy_knowledge_graph.ttl"))
kg.load()

print("   Waiting for graph to load...")
for i in range(20):
    if kg.is_full_graph_loaded():
        print(f"   ✓ Graph loaded after {i*0.5:.1f}s")
        break
    time.sleep(0.5)
else:
    print("   ⚠️  Timeout waiting for graph")

stats = kg.get_graph_stats()
print(f"\n   Graph has {stats['triples']:,} triples")

# Test 1: Direct triple iteration (should always work)
print("\n2. Testing Direct Triple Iteration...")
try:
    count = 0
    preflabel_count = 0
    
    for subj, pred, obj in kg.graph:
        count += 1
        if 'prefLabel' in str(pred):
            preflabel_count += 1
        
        if count >= 1000:  # Sample first 1000
            break
    
    print(f"   ✓ Successfully iterated {count} triples")
    print(f"   ✓ Found {preflabel_count} prefLabel predicates in sample")
    
except Exception as e:
    print(f"   ✗ Direct iteration failed: {e}")

# Test 2: Citation existence checks
print("\n3. Testing Citation Existence...")

test_citations = [
    ("skos:Asset", True, "Common SKOS term"),
    ("iec:ActivePower", True, "IEC power term"),
    ("skos:FakeTerm", False, "Non-existent term"),
    ("iec:GridCongestion", False, "Possibly fake term"),
]

results = {"passed": 0, "failed": 0}

for citation, expected, description in test_citations:
    try:
        result = kg.citation_exists(citation)
        
        if result == expected:
            status = "✓ PASS"
            results["passed"] += 1
        else:
            status = "✗ FAIL"
            results["failed"] += 1
        
        print(f"   {status}: {citation} -> {result} (expected {expected})")
        print(f"           {description}")
        
    except Exception as e:
        print(f"   ✗ ERROR: {citation} raised {type(e).__name__}: {e}")
        results["failed"] += 1

# Test 3: Get all citations by namespace
print("\n4. Testing Citation Extraction...")

namespaces = ["skos", "iec", "entsoe", "eurlex"]
total_citations = 0

for namespace in namespaces:
    try:
        citations = kg.get_all_valid_citations(namespace=namespace)
        total_citations += len(citations)
        
        if citations:
            print(f"   ✓ {namespace.upper()}: {len(citations)} citations")
            # Show first 3 examples
            for cit in citations[:3]:
                metadata = kg.get_citation_metadata(cit)
                if metadata:
                    print(f"      • {cit}: {metadata.get('label', 'N/A')}")
        else:
            print(f"   ⚠️  {namespace.upper()}: 0 citations (may indicate problem)")
            
    except Exception as e:
        print(f"   ✗ {namespace.upper()}: Error - {type(e).__name__}")

# Test 4: Citation metadata retrieval
print("\n5. Testing Metadata Retrieval...")

metadata_tests = ["skos:Asset", "iec:ActivePower"]

for citation in metadata_tests:
    try:
        metadata = kg.get_citation_metadata(citation)
        
        if metadata:
            print(f"   ✓ {citation}:")
            print(f"      Label: {metadata.get('label', 'N/A')}")
            print(f"      Definition: {metadata.get('definition', 'N/A')[:50]}...")
        else:
            print(f"   ⚠️  {citation}: No metadata found")
            
    except Exception as e:
        print(f"   ✗ {citation}: {type(e).__name__}: {e}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
Citation Tests:
  • Passed: {results['passed']}/{results['passed'] + results['failed']}
  • Failed: {results['failed']}/{results['passed'] + results['failed']}

Citation Extraction:
  • Total citations found: {total_citations}
  • Namespaces checked: {len(namespaces)}

Expected Results:
  BEFORE FIX: ~0 citations, many SPARQL errors
  AFTER FIX: 800+ citations, no SPARQL errors
""")

if results["failed"] == 0 and total_citations > 0:
    print("✅ ALL TESTS PASSED - Fix is working!")
elif results["failed"] == 0 and total_citations == 0:
    print("⚠️  Tests passed but no citations found - fix may be incomplete")
else:
    print("❌ TESTS FAILED - Fix needs more work")

print("="*70)