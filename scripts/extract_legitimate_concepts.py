#!/usr/bin/env python3
"""
Extract legitimate concepts from knowledge graph.

This script creates a high-quality knowledge graph containing only
legitimate domain concepts without synthetic numbered variants.
"""

import re
import logging
import time
from pathlib import Path
from typing import Set

try:
    from rdflib import Graph, Namespace, URIRef
    from rdflib.namespace import SKOS, RDF
    RDFLIB_AVAILABLE = True
except ImportError:
    print("ERROR: rdflib not available. Install with: pip install rdflib")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Patterns that indicate synthetic concepts
SYNTHETIC_PATTERNS = [
    r'_\d{3,4}$',  # Ends with _000 to _9999
]

def is_legitimate_concept(concept_uri: str) -> bool:
    """
    Determine if a concept is legitimate (not auto-generated).

    Legitimate concepts:
    - IEC standard terms without numbered suffixes
    - ENTSOE terms without numbered suffixes
    - Domain-specific terms with semantic meaning
    """
    # Remove synthetic numbered patterns
    for pattern in SYNTHETIC_PATTERNS:
        if re.search(pattern, concept_uri):
            return False

    # Extract the concept name (part after namespace)
    if ':' in concept_uri:
        namespace, concept_name = concept_uri.split(':', 1)
    else:
        return False

    # Legitimate if it's a semantic term from authoritative namespaces
    authoritative_namespaces = ['iec', 'entsoe', 'skos']

    if namespace in authoritative_namespaces:
        # Must not have numbered patterns and should be semantic
        if not re.search(r'_\d', concept_name):
            # Check if it's a meaningful concept name
            semantic_patterns = [
                r'^[A-Z][a-zA-Z]+$',  # CamelCase like ActivePower
                r'^[a-z]+[A-Z][a-zA-Z]*$',  # camelCase like gridCongestion
                r'^[A-Za-z]+$',  # Simple terms like Bay, Breaker
            ]

            for pattern in semantic_patterns:
                if re.match(pattern, concept_name):
                    return True

    return False

def extract_legitimate_graph(input_path: str, output_path: str) -> None:
    """Extract only legitimate concepts to a new graph."""
    logger.info(f"Loading knowledge graph from {input_path}")

    # Load original graph
    graph = Graph()
    graph.parse(input_path, format='turtle')
    logger.info(f"Loaded {len(graph):,} triples")

    # Create new graph for legitimate concepts
    clean_graph = Graph()

    # Copy namespaces
    for prefix, namespace in graph.namespaces():
        clean_graph.bind(prefix, namespace)

    # Get all concepts
    all_concepts = set(graph.subjects(RDF.type, SKOS.Concept))
    logger.info(f"Found {len(all_concepts):,} concepts")

    # Debug: show first few concepts
    sample_concepts = list(all_concepts)[:10]
    logger.info(f"Sample concepts: {[str(c) for c in sample_concepts]}")

    # Filter to legitimate concepts
    legitimate_concepts = set()
    debug_count = 0
    for concept in all_concepts:
        concept_str = str(concept)
        is_legit = is_legitimate_concept(concept_str)
        if is_legit:
            legitimate_concepts.add(concept)
        elif debug_count < 5:  # Debug first few
            logger.info(f"Rejected: {concept_str} (reason: pattern check)")
            debug_count += 1

    logger.info(f"Identified {len(legitimate_concepts):,} legitimate concepts")

    # Copy all triples related to legitimate concepts
    copied_triples = 0
    for subj, pred, obj in graph:
        # Include if subject is legitimate concept or if it's metadata about legitimate concepts
        if (subj in legitimate_concepts or
            (pred in [SKOS.definition, SKOS.prefLabel, SKOS.inScheme, RDF.type] and subj in legitimate_concepts)):
            clean_graph.add((subj, pred, obj))
            copied_triples += 1

    logger.info(f"Copied {copied_triples:,} triples for legitimate concepts")

    # Add some essential vocabulary structure
    essential_triples = [
        # Add scheme definitions
        ("http://energy.example.org/grid#GridEquipment", RDF.type, "http://www.w3.org/2004/02/skos/core#ConceptScheme"),
        ("http://energy.example.org/power#PowerSystems", RDF.type, "http://www.w3.org/2004/02/skos/core#ConceptScheme"),
    ]

    for subj, pred, obj in essential_triples:
        try:
            clean_graph.add((URIRef(subj), URIRef(pred), URIRef(obj)))
        except:
            pass  # Skip if invalid

    # Save clean graph
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving clean graph to {output_path}")
    clean_graph.serialize(destination=str(output_path), format='turtle')

    logger.info(f"Final graph: {len(clean_graph):,} triples")

    # Calculate reduction
    reduction = (len(graph) - len(clean_graph)) / len(graph)
    logger.info(f"Size reduction: {reduction:.1%}")

if __name__ == "__main__":
    extract_legitimate_graph(
        "data/energy_knowledge_graph.ttl",
        "data/energy_knowledge_graph_legitimate.ttl"
    )