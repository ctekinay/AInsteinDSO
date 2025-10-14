#!/usr/bin/env python3
"""
Test the corrected namespace patterns against actual data

This verifies the new patterns will find real citations.
"""

from pathlib import Path
import time

print("="*70)
print("TESTING CORRECTED NAMESPACE PATTERNS")
print("="*70)

from src.knowledge.kg_loader import KnowledgeGraphLoader

kg = KnowledgeGraphLoader(Path("data/energy_knowledge_graph.ttl"))
kg.load()

print("\nWaiting for graph to load...")
for i in range(20):
    if kg.is_full_graph_loaded():
        print(f"✓ Graph loaded after {i*0.5:.1f}s")
        break
    time.sleep(0.5)

# Test with actual URIs we found
print("\n" + "="*70)
print("TESTING CITATION EXTRACTION WITH ACTUAL URIS")
print("="*70)

# These are REAL URIs from the inspection
test_cases = [
    {
        "uri": "https://vocabs.alliander.com/def/ppt/1502",
        "expected_citation": "skos:1502",
        "label": "Klimaatneutrale elektriciteit"
    },
    {
        "uri": "http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/DCLine",
        "expected_citation": "iec:DCLine",
        "label": "DCLine"
    },
    {
        "uri": "http://data.europa.eu/eli/terms/1485-10",
        "expected_citation": "eurlex:1485-10",
        "label": "Reserve providing unit"
    },
    {
        "uri": "http://iec.ch/TC57/IEC61968/Asset",
        "expected_citation": "iec:Asset",
        "label": "Asset"
    },
    {
        "uri": "http://vocabs.alliander.com/terms/66",
        "expected_citation": "skos:66",
        "label": "RegelStation"
    }
]

print("\n1. Testing _extract_citation_id() logic:")
print("-" * 70)

for test in test_cases:
    uri = test["uri"]
    expected = test["expected_citation"]
    
    # Manually test extraction logic
    result = None
    
    patterns = [
        ("skos", [
            "https://vocabs.alliander.com/def/ppt/",
            "http://vocabs.alliander.com/terms/"
        ]),
        ("iec", [
            "http://iec.ch/TC57/ns/CIM/",
            "http://iec.ch/TC57/IEC61968/",
            "http://iec.ch/TC57/IEC61970/"
        ]),
        ("eurlex", [
            "http://data.europa.eu/eli/terms/"
        ]),
    ]
    
    for prefix, base_uris in patterns:
        for base_uri in base_uris:
            if uri.startswith(base_uri):
                local_part = uri[len(base_uri):]
                if '/' in local_part:
                    local_part = local_part.split('/')[-1]
                local_part = local_part.split('#')[-1]
                result = f"{prefix}:{local_part}"
                break
        if result:
            break
    
    status = "✓" if result == expected else "✗"
    print(f"{status} {uri}")
    print(f"   Expected: {expected}")
    print(f"   Got:      {result}")
    print()

# Test actual citation existence after applying fix
print("\n2. Testing citation_exists() with corrected patterns:")
print("-" * 70)

# Note: Test cases include both expected to exist and not exist
test_citations = [
    ("skos:1502", True, "Klimaatneutrale elektriciteit"),
    ("iec:DCLine", True, "DCLine"),
    ("eurlex:1485-10", True, "Reserve providing unit"),
    ("iec:Asset", True, "Asset"),
    ("skos:FakeTerm", False, "Non-existent"),
]

print("NOTE: These tests will PASS after applying the corrected code to kg_loader.py\n")

for citation, expected, description in test_citations:
    result = kg.citation_exists(citation)
    status = "✓" if result == expected else "⚠️"
    print(f"{status} {citation}: {result} (expected {expected}) - {description}")

# Test citation extraction
print("\n3. Testing get_all_valid_citations():")
print("-" * 70)

for namespace in ["skos", "iec", "eurlex", "entsoe", "lido"]:
    citations = kg.get_all_valid_citations(namespace=namespace)
    print(f"{namespace.upper()}: {len(citations)} citations")
    
    if citations and len(citations) > 0:
        print(f"   Examples: {citations[:3]}")

print("\n" + "="*70)
print("EXPECTED RESULTS POST-FIX (FOR VALIDATION):")
print("="*70)
print("""
Before Fix:
  SKOS: 0 citations
  IEC: 0 citations
  EUR-LEX: 562 citations

After Fix:
  SKOS: 3000+ citations (Alliander vocabulary)
  IEC: 400+ citations (IEC standards)
  EUR-LEX: 562 citations (already working)
  ENTSOE: 100+ citations (grid standards)
  LIDO: 250+ citations (Dutch regulations)
""")
print("="*70)