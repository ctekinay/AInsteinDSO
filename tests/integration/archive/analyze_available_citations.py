#!/usr/bin/env python3
"""
Comprehensive Analysis of Available Citations

This script analyzes all available citations in:
1. Knowledge Graph (SKOS, IEC, ENTSOE, EUR-LEX)
2. ArchiMate Models
3. Generates a reference guide for developers

Run this to know exactly what citations are available for use.
"""

from pathlib import Path
import time
import json
from collections import defaultdict

print("="*70)
print("COMPREHENSIVE CITATION ANALYSIS")
print("="*70)

# ============================================================================
# PART 1: Knowledge Graph Citations
# ============================================================================

print("\n📚 PART 1: Analyzing Knowledge Graph Citations...")
print("-" * 70)

from src.knowledge.kg_loader import KnowledgeGraphLoader

kg = KnowledgeGraphLoader(Path("data/energy_knowledge_graph.ttl"))
kg.load()

print("Waiting for graph to load...")
for i in range(20):
    if kg.is_full_graph_loaded():
        print(f"✓ Graph loaded after {i*0.5:.1f}s")
        break
    time.sleep(0.5)
else:
    print("⚠️  Graph still loading after 10s")

# Get statistics
stats = kg.get_graph_stats()
print(f"\nGraph Statistics:")
print(f"  Total triples: {stats['triples']:,}")
print(f"  Load time: {stats['load_time_ms']:.0f}ms")

# Get all citations by namespace
namespaces = ["skos", "iec", "entsoe", "eurlex", "lido"]
citation_catalog = {}

for namespace in namespaces:
    print(f"\n{namespace.upper()} Citations:")
    citations = kg.get_all_valid_citations(namespace=namespace)
    citation_catalog[namespace] = citations
    
    print(f"  Found {len(citations)} citations")
    
    if citations:
        print(f"  First 10 examples:")
        for citation in citations[:10]:
            metadata = kg.get_citation_metadata(citation)
            if metadata:
                label = metadata.get('label', 'N/A')
                print(f"    • {citation}: {label}")
    else:
        print(f"    (No {namespace} citations found)")

# ============================================================================
# PART 2: ArchiMate Model Citations
# ============================================================================

print("\n" + "="*70)
print("🏛️  PART 2: Analyzing ArchiMate Model Citations...")
print("-" * 70)

from src.archimate.parser import ArchiMateParser

parser = ArchiMateParser()

# Load all available models
models_dir = Path("data/models")
model_files = list(models_dir.glob("*.xml"))

print(f"\nFound {len(model_files)} ArchiMate model files:")
for model_file in model_files:
    print(f"  • {model_file.name}")

archimate_citations = {}

for model_file in model_files:
    print(f"\nLoading: {model_file.name}")
    try:
        parser.load_model(str(model_file))
        citations = parser.get_valid_citations()
        archimate_citations[model_file.name] = citations
        
        print(f"  ✓ Found {len(citations)} elements")
        
        # Group by element type
        elements_by_type = defaultdict(list)
        for citation in citations[:100]:  # Sample first 100
            element = parser.get_element_by_citation(citation)
            if element:
                elements_by_type[element.type].append(element)
        
        print(f"  Element types found:")
        for elem_type, elements in sorted(elements_by_type.items()):
            print(f"    • {elem_type}: {len(elements)} elements")
            
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")

# ============================================================================
# PART 3: Citation Usage Examples
# ============================================================================

print("\n" + "="*70)
print("💡 PART 3: Citation Usage Examples")
print("-" * 70)

print("\n1. Knowledge Graph Citation Examples:")
print("\n   CORRECT Usage:")
for namespace, citations in citation_catalog.items():
    if citations:
        example = citations[0]
        metadata = kg.get_citation_metadata(example)
        if metadata:
            print(f"   • {example}: {metadata.get('label', 'N/A')}")

print("\n   INCORRECT (Fake) Citations - NEVER USE:")
print("   • iec:GridCongestion  ← Does not exist in graph")
print("   • skos:FakeTerm  ← Invented citation")
print("   • iec:61968  ← Standard number, not a concept")

print("\n2. ArchiMate Citation Examples:")
if archimate_citations:
    first_model = list(archimate_citations.keys())[0]
    citations = archimate_citations[first_model]
    
    print(f"\n   CORRECT Usage (from {first_model}):")
    for citation in citations[:5]:
        element = parser.get_element_by_citation(citation)
        if element:
            print(f"   • {citation}")
            print(f"     Name: {element.name}, Type: {element.type}")

print("\n   INCORRECT (Fake) Citations - NEVER USE:")
print("   • archi:id-cap-001  ← Generic ID, not from actual model")
print("   • archi:id-asset-management  ← Human-readable IDs not used")

# ============================================================================
# PART 4: Save Citation Reference
# ============================================================================

print("\n" + "="*70)
print("💾 PART 4: Saving Citation Reference")
print("-" * 70)

reference = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "knowledge_graph": {
        "total_triples": stats['triples'],
        "namespaces": {}
    },
    "archimate_models": {}
}

# Add KG citations
for namespace, citations in citation_catalog.items():
    reference["knowledge_graph"]["namespaces"][namespace] = {
        "count": len(citations),
        "examples": citations[:20]  # First 20 examples
    }

# Add ArchiMate citations
for model_name, citations in archimate_citations.items():
    reference["archimate_models"][model_name] = {
        "count": len(citations),
        "examples": citations[:20]  # First 20 examples
    }

# Save to file
output_file = Path("docs/AVAILABLE_CITATIONS.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(reference, f, indent=2)

print(f"\n✓ Citation reference saved to: {output_file}")

# ============================================================================
# PART 5: Summary Statistics
# ============================================================================

print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

total_kg_citations = sum(len(cits) for cits in citation_catalog.values())
total_archimate_citations = sum(len(cits) for cits in archimate_citations.values())

print(f"""
Knowledge Graph Citations:
  • SKOS: {len(citation_catalog.get('skos', []))} citations
  • IEC: {len(citation_catalog.get('iec', []))} citations
  • ENTSOE: {len(citation_catalog.get('entsoe', []))} citations
  • EUR-LEX: {len(citation_catalog.get('eurlex', []))} citations
  • LIDO: {len(citation_catalog.get('lido', []))} citations
  • TOTAL: {total_kg_citations} citations

ArchiMate Model Citations:
  • Models loaded: {len(archimate_citations)}
  • Total elements: {total_archimate_citations}

GRAND TOTAL: {total_kg_citations + total_archimate_citations} valid citations available
""")

print("="*70)
print("✓ Analysis complete!")
print("="*70)
print(f"\nNext steps:")
print(f"  1. Review {output_file} for complete citation list")
print(f"  2. Use only these citations in responses")
print(f"  3. Update prompts to include citation examples")
print(f"  4. Test citation validation with test_citation_exists.py")