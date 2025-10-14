#!/usr/bin/env python3
"""
Inspect actual URI patterns in the knowledge graph

This will show us what the REAL URIs look like so we can fix the citation extraction.
"""

from pathlib import Path
import time
from collections import defaultdict

print("="*70)
print("KNOWLEDGE GRAPH URI PATTERN INSPECTION")
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

print(f"\nGraph has {len(kg.graph):,} triples")

# Collect sample URIs by predicate type
samples = {
    'prefLabel': [],
    'definition': [],
    'subjects': defaultdict(list)
}

print("\n" + "="*70)
print("ANALYZING URI PATTERNS")
print("="*70)

count = 0
namespace_counts = defaultdict(int)

for subj, pred, obj in kg.graph:
    subj_str = str(subj)
    pred_str = str(pred)
    
    # Track namespace usage
    if '#' in subj_str:
        namespace = subj_str.split('#')[0] + '#'
        namespace_counts[namespace] += 1
    elif '/' in subj_str:
        parts = subj_str.split('/')
        if len(parts) > 3:
            namespace = '/'.join(parts[:-1]) + '/'
            namespace_counts[namespace] += 1
    
    # Collect samples with prefLabel
    if 'prefLabel' in pred_str:
        if len(samples['prefLabel']) < 20:
            samples['prefLabel'].append({
                'subject': subj_str,
                'predicate': pred_str,
                'object': str(obj)
            })
    
    # Collect samples with definition
    if 'definition' in pred_str:
        if len(samples['definition']) < 10:
            samples['definition'].append({
                'subject': subj_str,
                'predicate': pred_str,
                'object': str(obj)[:100] + '...'
            })
    
    count += 1
    if count >= 5000:  # Sample first 5000 triples
        break

# Display namespace statistics
print("\n1. NAMESPACE USAGE (top 10):")
print("-" * 70)
for namespace, count in sorted(namespace_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"   {count:5d} triples: {namespace}")

# Display prefLabel samples
print("\n2. SAMPLE CONCEPTS WITH prefLabel (first 20):")
print("-" * 70)
for i, sample in enumerate(samples['prefLabel'], 1):
    print(f"\n   [{i}] Subject URI:")
    print(f"       {sample['subject']}")
    print(f"       Label: {sample['object']}")
    
    # Try to identify namespace
    if 'skos' in sample['subject'].lower():
        print(f"       → Appears to be SKOS")
    elif 'iec' in sample['subject'].lower():
        print(f"       → Appears to be IEC")
    elif 'entsoe' in sample['subject'].lower():
        print(f"       → Appears to be ENTSOE")
    elif 'eurlex' in sample['subject'].lower() or 'europa' in sample['subject'].lower():
        print(f"       → Appears to be EUR-LEX")

# Display definition samples
print("\n3. SAMPLE CONCEPTS WITH definition (first 10):")
print("-" * 70)
for i, sample in enumerate(samples['definition'], 1):
    print(f"\n   [{i}] {sample['subject']}")
    print(f"       Definition: {sample['object']}")

# Check for specific terms we're looking for
print("\n4. SEARCHING FOR SPECIFIC TERMS:")
print("-" * 70)

search_terms = ['Asset', 'ActivePower', 'GridCongestion', 'Transformer']

for term in search_terms:
    print(f"\n   Searching for '{term}':")
    found = False
    
    for subj, pred, obj in kg.graph:
        subj_str = str(subj)
        obj_str = str(obj)
        
        # Case-insensitive search in subject URIs
        if term.lower() in subj_str.lower():
            print(f"      ✓ Found in subject: {subj_str}")
            found = True
            break
        
        # Case-insensitive search in labels
        if 'prefLabel' in str(pred) and term.lower() in obj_str.lower():
            print(f"      ✓ Found in label: {obj_str}")
            print(f"        Subject URI: {subj_str}")
            found = True
            break
    
    if not found:
        print(f"      ✗ Not found")

# Analysis summary
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print(f"""
Total Namespaces Found: {len(namespace_counts)}
Most Common Namespace: {max(namespace_counts.items(), key=lambda x: x[1])[0]}

Concepts with prefLabel: {len(samples['prefLabel'])}
Concepts with definition: {len(samples['definition'])}

NEXT STEPS:
1. Review the actual URI patterns above
2. Update _extract_citation_id() to match these patterns
3. Update namespace_map in citation methods
4. Re-run verification
""")

print("="*70)