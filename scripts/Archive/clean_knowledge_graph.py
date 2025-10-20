#!/usr/bin/env python3
"""
Knowledge Graph Cleaner - Remove synthetic data pollution.

This script identifies and removes auto-generated synthetic concepts that were
artificially inflating the knowledge graph from legitimate 39,100+ triples
to 132,532 through systematic numbered variants.

REMOVES:
- Numbered concept patterns: Category_0000 through Category_9999
- Identical template definitions: "comprehensive energy domain coverage"
- Auto-generated variants with minimal semantic value

PRESERVES:
- Authoritative IEC standard references
- Legitimate ENTSOE vocabulary terms
- Domain-specific concepts with unique definitions
"""

import re
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter

try:
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import SKOS, RDF
    RDFLIB_AVAILABLE = True
except ImportError:
    print("ERROR: rdflib not available. Install with: pip install rdflib")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Synthetic patterns to identify and remove
SYNTHETIC_PATTERNS = [
    r'\w+_\d{4}',  # Category_0000 through Category_9999
    r'\w+_\d{3}',  # Category_000 through Category_999
]

# Template definitions that indicate auto-generation
TEMPLATE_DEFINITIONS = [
    "comprehensive energy domain coverage",
    "Grid Access asset \\d+",
    "Regulation aspect \\d+",
    "Transmission element \\d+",
    "Distribution component \\d+",
    "Generation unit \\d+",
]

# Authoritative prefixes to ALWAYS preserve
AUTHORITATIVE_PREFIXES = [
    "http://iec.ch/TC57/CIM#",
    "http://entsoe.eu/CIM/",
    "http://www.w3.org/2004/02/skos/core#",
]

class KnowledgeGraphCleaner:
    """
    Removes synthetic data pollution from knowledge graphs.

    Implements quality gates to ensure we preserve legitimate domain knowledge
    while removing auto-generated template content.
    """

    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.stats = {
            'total_triples': 0,
            'removed_synthetic': 0,
            'removed_template': 0,
            'preserved_authoritative': 0,
            'final_triples': 0,
        }

    def load_graph(self) -> Graph:
        """Load the knowledge graph from file."""
        logger.info(f"Loading knowledge graph from {self.input_path}")
        start_time = time.time()

        graph = Graph()
        graph.parse(self.input_path, format='turtle')

        load_time = time.time() - start_time
        self.stats['total_triples'] = len(graph)

        logger.info(f"Loaded {len(graph):,} triples in {load_time:.2f}s")
        return graph

    def identify_synthetic_concepts(self, graph: Graph) -> Set[URIRef]:
        """Identify synthetic concepts using pattern matching."""
        synthetic_concepts = set()

        # Get all subjects that are concepts
        concepts = set(graph.subjects(RDF.type, SKOS.Concept))

        for concept in concepts:
            concept_str = str(concept)

            # Check for numbered patterns
            for pattern in SYNTHETIC_PATTERNS:
                if re.search(pattern, concept_str):
                    synthetic_concepts.add(concept)
                    break

        logger.info(f"Identified {len(synthetic_concepts):,} synthetic concepts")
        return synthetic_concepts

    def identify_template_definitions(self, graph: Graph) -> Set[URIRef]:
        """Identify concepts with template definitions."""
        template_concepts = set()
        definition_counts = Counter()

        # Count definition frequencies
        for subj, pred, obj in graph.triples((None, SKOS.definition, None)):
            if isinstance(obj, Literal):
                definition_counts[str(obj)] += 1

        # Find template definitions (used more than threshold)
        template_definitions = {
            defn for defn, count in definition_counts.items()
            if count > 10  # More than 10 identical definitions = template
        }

        # Find concepts with template definitions
        for subj, pred, obj in graph.triples((None, SKOS.definition, None)):
            if isinstance(obj, Literal) and str(obj) in template_definitions:
                template_concepts.add(subj)

        logger.info(f"Identified {len(template_concepts):,} concepts with template definitions")
        logger.info(f"Template definitions: {list(template_definitions)[:3]}...")

        return template_concepts

    def is_authoritative_concept(self, concept: URIRef) -> bool:
        """Check if concept comes from authoritative source."""
        concept_str = str(concept)

        # NEVER preserve numbered synthetic patterns, even if from "authoritative" namespaces
        for pattern in SYNTHETIC_PATTERNS:
            if re.search(pattern, concept_str):
                return False

        # Only preserve if from authoritative namespace AND not synthetic pattern
        for prefix in AUTHORITATIVE_PREFIXES:
            if concept_str.startswith(prefix):
                return True

        # Additional checks for legitimate concepts (only if not synthetic pattern)
        if any(term in concept_str.lower() for term in [
            'gridcongestion', 'loadflow', 'transformer', 'substation',
            'activepower', 'reactivepower', 'conductor', 'breaker'
        ]):
            return True

        return False

    def clean_graph(self, graph: Graph) -> Graph:
        """Remove synthetic and template concepts while preserving authoritative content."""
        logger.info("Starting knowledge graph cleaning...")

        # Identify concepts to remove
        synthetic_concepts = self.identify_synthetic_concepts(graph)
        template_concepts = self.identify_template_definitions(graph)

        # Combine removal sets
        concepts_to_remove = synthetic_concepts | template_concepts

        # Preserve authoritative concepts even if they match patterns
        authoritative_concepts = {
            concept for concept in concepts_to_remove
            if self.is_authoritative_concept(concept)
        }
        concepts_to_remove -= authoritative_concepts

        logger.info(f"Removing {len(concepts_to_remove):,} concepts")
        logger.info(f"Preserving {len(authoritative_concepts):,} authoritative concepts")

        # Create clean graph
        clean_graph = Graph()

        # Copy namespaces
        for prefix, namespace in graph.namespaces():
            clean_graph.bind(prefix, namespace)

        # Copy triples that don't involve removed concepts
        removed_count = 0
        for subj, pred, obj in graph:
            if subj in concepts_to_remove or obj in concepts_to_remove:
                removed_count += 1
                continue

            clean_graph.add((subj, pred, obj))

        self.stats['removed_synthetic'] = len(synthetic_concepts - authoritative_concepts)
        self.stats['removed_template'] = len(template_concepts - authoritative_concepts)
        self.stats['preserved_authoritative'] = len(authoritative_concepts)
        self.stats['final_triples'] = len(clean_graph)

        logger.info(f"Removed {removed_count:,} triples")
        logger.info(f"Clean graph: {len(clean_graph):,} triples")

        return clean_graph

    def validate_quality_gates(self, clean_graph: Graph) -> bool:
        """Validate the cleaned graph meets quality requirements."""
        logger.info("Validating quality gates...")

        # Check definition uniqueness
        definitions = []
        for subj, pred, obj in clean_graph.triples((None, SKOS.definition, None)):
            if isinstance(obj, Literal):
                definitions.append(str(obj))

        unique_ratio = len(set(definitions)) / len(definitions) if definitions else 1.0

        # Check synthetic ratio
        concepts = list(clean_graph.subjects(RDF.type, SKOS.Concept))
        synthetic_count = sum(
            1 for concept in concepts
            if any(re.search(pattern, str(concept)) for pattern in SYNTHETIC_PATTERNS)
        )
        synthetic_ratio = synthetic_count / len(concepts) if concepts else 0.0

        # Quality gates
        quality_gates = {
            'unique_definitions': unique_ratio >= 0.8,  # 80% unique definitions
            'low_synthetic': synthetic_ratio <= 0.1,   # <10% synthetic patterns
            'reasonable_size': 30000 <= len(clean_graph) <= 50000,  # Reasonable size
        }

        logger.info(f"Quality metrics:")
        logger.info(f"  Definition uniqueness: {unique_ratio:.2%}")
        logger.info(f"  Synthetic ratio: {synthetic_ratio:.2%}")
        logger.info(f"  Graph size: {len(clean_graph):,} triples")

        all_passed = all(quality_gates.values())
        logger.info(f"Quality gates: {'PASSED' if all_passed else 'FAILED'}")

        return all_passed

    def save_clean_graph(self, clean_graph: Graph) -> None:
        """Save the cleaned graph to output file."""
        logger.info(f"Saving clean graph to {self.output_path}")

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as turtle format
        clean_graph.serialize(destination=str(self.output_path), format='turtle')

        logger.info(f"Saved {len(clean_graph):,} triples")

    def print_summary(self) -> None:
        """Print cleaning summary statistics."""
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH CLEANING SUMMARY")
        print("="*60)
        print(f"Original triples:           {self.stats['total_triples']:,}")
        print(f"Removed synthetic:          {self.stats['removed_synthetic']:,}")
        print(f"Removed template defs:      {self.stats['removed_template']:,}")
        print(f"Preserved authoritative:    {self.stats['preserved_authoritative']:,}")
        print(f"Final triples:              {self.stats['final_triples']:,}")

        reduction = (self.stats['total_triples'] - self.stats['final_triples']) / self.stats['total_triples']
        print(f"Size reduction:             {reduction:.1%}")
        print("="*60)

    def run(self) -> bool:
        """Execute the complete cleaning process."""
        try:
            # Load original graph
            graph = self.load_graph()

            # Clean the graph
            clean_graph = self.clean_graph(graph)

            # Validate quality
            quality_passed = self.validate_quality_gates(clean_graph)

            if quality_passed:
                # Save clean graph
                self.save_clean_graph(clean_graph)
                self.print_summary()
                return True
            else:
                logger.error("Quality gates failed - not saving cleaned graph")
                return False

        except Exception as e:
            logger.error(f"Cleaning failed: {e}")
            return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Clean synthetic data from knowledge graph")
    parser.add_argument(
        "--input",
        default="data/energy_knowledge_graph.ttl",
        help="Input knowledge graph file"
    )
    parser.add_argument(
        "--output",
        default="data/energy_knowledge_graph_clean.ttl",
        help="Output cleaned knowledge graph file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze without saving output"
    )

    args = parser.parse_args()

    cleaner = KnowledgeGraphCleaner(args.input, args.output)

    if args.dry_run:
        logger.info("DRY RUN - Analysis only")
        graph = cleaner.load_graph()
        synthetic = cleaner.identify_synthetic_concepts(graph)
        template = cleaner.identify_template_definitions(graph)

        print(f"\nAnalysis Results:")
        print(f"Total concepts: {len(list(graph.subjects(RDF.type, SKOS.Concept)))}")
        print(f"Synthetic patterns: {len(synthetic)}")
        print(f"Template definitions: {len(template)}")
        print(f"Would remove: {len(synthetic | template)} concepts")
    else:
        success = cleaner.run()
        exit(0 if success else 1)


if __name__ == "__main__":
    main()