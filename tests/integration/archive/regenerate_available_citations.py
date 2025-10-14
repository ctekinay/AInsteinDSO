#!/usr/bin/env python3
"""
Regenerate AVAILABLE_CITATIONS.json with correct citation data

This creates the authoritative reference document with all 5,265+ valid citations.
"""

from pathlib import Path
import time
import json
from datetime import datetime
from collections import defaultdict

print("="*70)
print("REGENERATING AVAILABLE_CITATIONS.json")
print("="*70)

from src.knowledge.kg_loader import KnowledgeGraphLoader
from src.archimate.parser import ArchiMateParser

# Load Knowledge Graph
print("\n1. Loading Knowledge Graph...")
kg = KnowledgeGraphLoader(Path("data/energy_knowledge_graph.ttl"))
kg.load()

print("   Waiting for graph to load...")
for i in range(20):
    if kg.is_full_graph_loaded():
        print(f"   ✓ Graph loaded after {i*0.5:.1f}s")
        break
    time.sleep(0.5)

stats = kg.get_graph_stats()
print(f"   Graph has {stats['triples']:,} triples")

# Extract all citations by namespace
print("\n2. Extracting Knowledge Graph Citations...")

namespaces = {
    "skos": "Alliander Dutch energy vocabulary",
    "iec": "IEC International energy standards", 
    "entsoe": "ENTSOE European grid operator standards",
    "eurlex": "EUR-LEX European Union energy regulations",
    "lido": "LIDO Dutch government energy regulations"
}

kg_data = {}
total_kg_citations = 0

for namespace, description in namespaces.items():
    print(f"   Extracting {namespace.upper()} citations...")
    
    citations = kg.get_all_valid_citations(namespace=namespace)
    
    # Get metadata for first 50 examples
    examples = []
    for citation in citations[:50]:
        metadata = kg.get_citation_metadata(citation)
        if metadata:
            examples.append({
                "citation": citation,
                "label": metadata.get('label', 'N/A'),
                "definition": metadata.get('definition', 'N/A')[:100] + '...' if metadata.get('definition') else None
            })
    
    kg_data[namespace] = {
        "count": len(citations),
        "description": description,
        "examples": examples
    }
    
    total_kg_citations += len(citations)
    print(f"      ✓ Found {len(citations)} citations")

# Extract ArchiMate citations
print("\n3. Extracting ArchiMate Model Citations...")

parser = ArchiMateParser()
models_dir = Path("data/models")
model_files = list(models_dir.glob("*.xml"))

archimate_data = {}
total_archimate_citations = 0

for model_file in model_files:
    print(f"   Loading {model_file.name}...")
    
    try:
        parser.load_model(str(model_file))
        citations = parser.get_valid_citations()
        
        # Get element details for examples
        examples = []
        element_types = defaultdict(int)
        
        for citation in citations[:50]:
            element = parser.get_element_by_citation(citation)
            if element:
                examples.append({
                    "citation": citation,
                    "name": element.name,
                    "type": element.type,
                    "layer": element.layer
                })
                element_types[element.type] += 1
        
        archimate_data[model_file.name] = {
            "count": len(citations),
            "examples": examples,
            "element_types": dict(element_types)
        }
        
        total_archimate_citations += len(citations)
        print(f"      ✓ Found {len(citations)} elements")
        
    except Exception as e:
        print(f"      ✗ Failed: {e}")

# Build complete reference document
print("\n4. Building Complete Reference Document...")

reference = {
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "summary": {
        "total_citations": total_kg_citations + total_archimate_citations,
        "knowledge_graph_citations": total_kg_citations,
        "archimate_citations": total_archimate_citations
    },
    "knowledge_graph": {
        "total_triples": stats['triples'],
        "load_time_ms": stats['load_time_ms'],
        "namespaces": kg_data
    },
    "archimate_models": archimate_data,
    "usage_notes": {
        "citation_formats": {
            "skos": "skos:1502, skos:220, etc.",
            "iec": "iec:Asset, iec:DCLine, etc.",
            "entsoe": "entsoe:ReconciliationResponsible, etc.",
            "eurlex": "eurlex:944-2, eurlex:1485-10, etc.",
            "lido": "lido:108, lido:93, etc.",
            "archimate": "archi:id-id-{uuid}"
        },
        "fake_citation_examples": [
            "iec:GridCongestion (does not exist)",
            "archi:id-cap-001 (wrong format)",
            "skos:FakeTerm (invented)",
            "iec:61968 (standard number, not concept)"
        ],
        "validation_rules": [
            "All citations must exist in knowledge sources",
            "Use citation_exists() to validate before use",
            "Build citation pool from retrieval context",
            "LLM must only use citations from provided pool"
        ]
    }
}

# Save to file
output_file = Path("docs/AVAILABLE_CITATIONS.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(reference, f, indent=2, ensure_ascii=False)

print(f"   ✓ Saved to {output_file}")

# Also create a human-readable markdown version
print("\n5. Creating Human-Readable Markdown Guide...")

md_content = f"""# Available Citations Reference

**Generated:** {reference['generated_at']}

## Summary

- **Total Citations:** {reference['summary']['total_citations']:,}
- **Knowledge Graph:** {reference['summary']['knowledge_graph_citations']:,} citations
- **ArchiMate Models:** {reference['summary']['archimate_citations']:,} citations

---

## Knowledge Graph Citations ({reference['summary']['knowledge_graph_citations']:,} total)

"""

for namespace, data in kg_data.items():
    md_content += f"\n### {namespace.upper()} - {data['description']}\n"
    md_content += f"**Count:** {data['count']:,} citations\n\n"
    md_content += "**Examples:**\n"
    
    for example in data['examples'][:10]:
        md_content += f"- `{example['citation']}`: {example['label']}\n"
        if example.get('definition'):
            md_content += f"  - *{example['definition']}*\n"
    
    md_content += "\n"

md_content += f"""---

## ArchiMate Model Citations ({reference['summary']['archimate_citations']:,} total)

"""

for model_name, data in archimate_data.items():
    md_content += f"\n### {model_name}\n"
    md_content += f"**Count:** {data['count']:,} elements\n\n"
    
    if data.get('element_types'):
        md_content += "**Element Types:**\n"
        for elem_type, count in sorted(data['element_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
            md_content += f"- {elem_type}: {count}\n"
        md_content += "\n"
    
    md_content += "**Example Citations:**\n"
    for example in data['examples'][:5]:
        md_content += f"- `{example['citation']}`\n"
        md_content += f"  - Name: {example['name']}\n"
        md_content += f"  - Type: {example['type']}\n"
        md_content += f"  - Layer: {example['layer']}\n"

md_content += """
---

## Citation Formats

### Valid Formats

- **SKOS (Alliander):** `skos:1502`, `skos:220`
- **IEC Standards:** `iec:Asset`, `iec:DCLine`
- **ENTSOE Grid:** `entsoe:ReconciliationResponsible`
- **EUR-LEX:** `eurlex:944-2`, `eurlex:1485-10`
- **LIDO Dutch:** `lido:108`, `lido:93`
- **ArchiMate:** `archi:id-id-{uuid}`

### Invalid (Fake) Citations - NEVER USE

- ❌ `iec:GridCongestion` - Does not exist in graph
- ❌ `archi:id-cap-001` - Wrong format (not a real UUID)
- ❌ `skos:FakeTerm` - Invented citation
- ❌ `iec:61968` - Standard number, not a concept

---

## Validation Rules

1. **All citations must exist** in knowledge sources (KG or ArchiMate)
2. **Use `citation_exists()`** to validate before use
3. **Build citation pool** from retrieval context
4. **LLM must only use** citations from provided pool
5. **No fabricated citations** - zero tolerance policy

---

## Usage Examples

### Python

```python
from src.knowledge.kg_loader import KnowledgeGraphLoader
from pathlib import Path

# Load KG
kg = KnowledgeGraphLoader(Path("data/energy_knowledge_graph.ttl"))
kg.load()

# Validate citation
exists = kg.citation_exists("iec:Asset")  # True
fake = kg.citation_exists("iec:GridCongestion")  # False

# Get metadata
metadata = kg.get_citation_metadata("iec:Asset")
print(metadata['label'])  # "Asset Info"

# Get all citations for a namespace
iec_citations = kg.get_all_valid_citations(namespace="iec")
print(f"Found {len(iec_citations)} IEC citations")
```

### In Templates

```python
response = f\"\"\"
Based on [skos:1502] (Klimaatneutrale elektriciteit), the system 
should use [iec:Asset] for asset management and follow [eurlex:944-2] 
regulations for distribution.
\"\"\"
```

---

*This document is auto-generated. Run `python regenerate_available_citations.py` to update.*
"""

md_file = Path("docs/AVAILABLE_CITATIONS.md")
with open(md_file, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"   ✓ Saved to {md_file}")

# Print summary
print("\n" + "="*70)
print("COMPLETE CITATION REFERENCE GENERATED")
print("="*70)

print(f"""
Files Created:
  1. {output_file} (machine-readable JSON)
  2. {md_file} (human-readable guide)

Citation Breakdown:
  • SKOS (Alliander): {kg_data['skos']['count']:,} citations
  • IEC (Standards): {kg_data['iec']['count']:,} citations
  • EUR-LEX (EU): {kg_data['eurlex']['count']:,} citations
  • ENTSOE (Grid): {kg_data['entsoe']['count']:,} citations
  • LIDO (Dutch): {kg_data['lido']['count']:,} citations
  • ArchiMate: {total_archimate_citations:,} elements

Total: {reference['summary']['total_citations']:,} valid citations

These files can be used by:
  • Development tools for citation guidance
  • LLM prompts to constrain citation usage
  • Documentation and training materials
  • Quality assurance and testing
""")

print("="*70)
print("✅ Citation reference is now up to date!")
print("="*70)