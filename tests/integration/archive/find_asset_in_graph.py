#!/usr/bin/env python3
"""
Find where "Asset" actually exists in the knowledge graph
"""

from pathlib import Path
import time

print("="*70)
print("SEARCHING FOR 'ASSET' IN KNOWLEDGE GRAPH")
print("="*70)

from src.knowledge.kg_loader import KnowledgeGraphLoader

kg = KnowledgeGraphLoader(Path("data/energy_knowledge_graph.ttl"))
kg.load()

print("\nWaiting for graph to load...")
for i in range(20):
    if kg.is_full_graph_loaded():
        print(f"✓ Graph loaded after {i*0.5:.1f}s\n")
        break
    time.sleep(0.5)

# Search for "Asset" in URIs and labels
print("Searching for 'Asset' (case-insensitive)...\n")

found_in_uris = []
found_in_labels = []

for subj, pred, obj in kg.graph:
    subj_str = str(subj)
    pred_str = str(pred)
    obj_str = str(obj)
    
    # Check in subject URIs
    if 'asset' in subj_str.lower():
        found_in_uris.append(subj_str)
    
    # Check in labels
    if 'prefLabel' in pred_str and 'asset' in obj_str.lower():
        found_in_labels.append({
            'uri': subj_str,
            'label': obj_str
        })

# Remove duplicates
found_in_uris = list(set(found_in_uris))

print(f"Found {len(found_in_uris)} URIs containing 'Asset':")
print("-" * 70)
for uri in found_in_uris[:10]:
    print(f"  • {uri}")
    
    # Try to extract citation
    if 'iec.ch' in uri.lower():
        # Extract IEC citation
        if '/IEC61968/' in uri:
            local = uri.split('/IEC61968/')[-1]
            print(f"    → Citation: iec:{local}")
        elif '/ns/CIM/' in uri:
            local = uri.split('/')[-1]
            print(f"    → Citation: iec:{local}")

print(f"\nFound {len(found_in_labels)} labels containing 'Asset':")
print("-" * 70)
for item in found_in_labels[:10]:
    print(f"  • {item['label']}")
    print(f"    URI: {item['uri']}")
    
    # Determine namespace
    if 'vocabs.alliander.com' in item['uri']:
        print(f"    → This is SKOS (Alliander vocabulary)")
    elif 'iec.ch' in item['uri']:
        print(f"    → This is IEC")
    elif 'europa.eu' in item['uri']:
        print(f"    → This is EUR-LEX")

# Test actual citations
print("\n" + "="*70)
print("TESTING ACTUAL CITATIONS")
print("="*70)

test_citations = [
    "iec:Asset",
    "skos:Asset",
]

for citation in test_citations:
    exists = kg.citation_exists(citation)
    print(f"\n{citation}: {exists}")
    
    if exists:
        metadata = kg.get_citation_metadata(citation)
        if metadata:
            print(f"  Label: {metadata.get('label')}")
            print(f"  URI: {metadata.get('uri')}")

# Check what SKOS terms are actually available
print("\n" + "="*70)
print("SAMPLE SKOS TERMS (first 20)")
print("="*70)

skos_citations = kg.get_all_valid_citations(namespace="skos")
for citation in skos_citations[:20]:
    metadata = kg.get_citation_metadata(citation)
    if metadata:
        print(f"  {citation}: {metadata.get('label')}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
The test is looking for 'skos:Asset' but:
1. 'Asset' is an IEC term, not SKOS
2. SKOS namespace is Alliander's Dutch vocabulary
3. Should use 'iec:Asset' instead

The test expectations need to be updated to use terms that
actually exist in each namespace.
""")