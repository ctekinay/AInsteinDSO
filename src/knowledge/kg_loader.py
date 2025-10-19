"""
Optimized Knowledge Graph loader with lazy loading and caching.

FIXED VERSION:
- Proper citation ID extraction with prefix preservation (skos:, iec:, entsoe:)
- Robust SPARQL queries that avoid postParse2() errors
- Fallback to Python filtering when SPARQL fails

This addresses the performance issues with large knowledge graphs by implementing
lazy loading, background graph loading, and intelligent caching.
"""

import json
import logging
import time
import threading
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from rdflib import RDF

# Initialize RDFLIB_AVAILABLE at module level
RDFLIB_AVAILABLE = False

if TYPE_CHECKING:
    from rdflib import Graph as RDFGraph
    from rdflib import Namespace, URIRef
    from rdflib.query import ResultRow
    RDFLIB_AVAILABLE = True
else:
    try:
        from rdflib import Graph as RDFGraph
        from rdflib import Namespace, URIRef
        from rdflib.query import ResultRow
        RDFLIB_AVAILABLE = True
    except ImportError:
        # Create runtime fallbacks for when rdflib is not available
        RDFGraph = None  # type: ignore

        def _namespace_fallback(x: str) -> str:
            return x

        Namespace = _namespace_fallback  # type: ignore
        URIRef = None  # type: ignore
        ResultRow = None  # type: ignore

from src.exceptions.exceptions import KnowledgeGraphError
from src.utils.trace import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer()

# SKOS vocabulary predicates (not concept URIs)
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
# Actual concept namespaces (used in citation_exists, etc.)
ALLIANDER_SKOS = Namespace("https://vocabs.alliander.com/def/ppt/")
IEC = Namespace("http://iec.ch/TC57/CIM#")
ENTSOE = Namespace("http://entsoe.eu/CIM/SchemaExtension/3/1#")

MINIMUM_TRIPLES = 39100  # Original requirement for legitimate domain knowledge


class KnowledgeGraphLoader:
    """Knowledge Graph loader with lazy loading and caching."""

    def __init__(self, kg_path: Optional[Path] = None):
        """
        Initialize the Knowledge Graph loader.

        Args:
            kg_path: Path to the knowledge graph TTL file (string or Path object).
                    Defaults to data/energy_knowledge_graph.ttl
        """
        if kg_path is None:
            self.kg_path = Path("data/energy_knowledge_graph.ttl")
        elif isinstance(kg_path, str):
            self.kg_path = Path(kg_path)
        else:
            self.kg_path = kg_path

        self.graph: Optional[RDFGraph] = None
        self.cache: Dict[str, Any] = {}
        self.iec_terms: Dict[str, str] = {}
        self.entsoe_terms: Dict[str, str] = {}
        self.eurlex_terms: Dict[str, str] = {}
        self.load_time_ms: float = 0
        self._query_cache: Dict[str, Any] = {}  # Query result cache
        self._loading = False
        self._load_lock = threading.Lock()
        self._full_graph_loaded = False

    def load(self) -> None:
        """
        Initialize lazy loading - start background load of full graph.
        Returns immediately with basic graph for immediate queries.

        OPTIMIZED: Adds KG caching for faster subsequent loads.
        """
        if not self.kg_path.exists():
            raise KnowledgeGraphError(f"Knowledge graph not found at {self.kg_path}")

        # Check cache first
        cache_file = Path('data/.kg_cache.pkl')
        if cache_file.exists():
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if cache_data.get('kg_path') == str(self.kg_path):
                        self.graph = cache_data.get('graph')
                        self.iec_terms = cache_data.get('iec_terms', {})
                        self.entsoe_terms = cache_data.get('entsoe_terms', {})
                        self.eurlex_terms = cache_data.get('eurlex_terms', {})
                        self._full_graph_loaded = True
                        logger.info(f"✅ Loaded KG from cache (instant) - {len(self.graph)} triples")
                        return
            except Exception as e:
                logger.debug(f"Cache load failed: {e}")

        # Initialize basic graph for immediate use
        if RDFLIB_AVAILABLE and RDFGraph:
            self.graph = RDFGraph()
        logger.info("Starting background load of full knowledge graph...")

        # Start background thread to load full graph
        threading.Thread(target=self._load_full_graph, daemon=True).start()

    def _load_full_graph(self) -> None:
        """Load full knowledge graph in background thread."""
        with self._load_lock:
            if self._full_graph_loaded or self._loading:
                return

            self._loading = True

        try:
            start_time = time.time()
            logger.info("Loading full knowledge graph...")

            if not RDFLIB_AVAILABLE or not RDFGraph:
                logger.error("rdflib not available, cannot load graph")
                return

            full_graph = RDFGraph()
            full_graph.parse(self.kg_path, format="turtle")

            # Replace graph atomically
            with self._load_lock:
                self.graph = full_graph
                self._full_graph_loaded = True
                self.load_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Full knowledge graph loaded in {self.load_time_ms:.0f}ms "
                f"with {len(self.graph)} triples"
            )

            # Extract terms after full load
            self._extract_terms()

            # Save to cache for faster future loads
            try:
                import pickle
                cache_file = Path('data/.kg_cache.pkl')
                cache_file.parent.mkdir(exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'kg_path': str(self.kg_path),
                        'graph': self.graph,
                        'iec_terms': self.iec_terms,
                        'entsoe_terms': self.entsoe_terms,
                        'eurlex_terms': self.eurlex_terms
                    }, f)
                logger.info("✅ KG cached for fast future loads")
            except Exception as e:
                logger.debug(f"Cache save failed: {e}")

        except Exception as e:
            logger.error(f"Failed to load full knowledge graph: {e}")
            with self._load_lock:
                self._loading = False

    def load_on_demand(self, query_terms: List[str]) -> Dict[str, Any]:
        """
        Load only relevant subgraph for query with caching.

        Args:
            query_terms: Terms to query for

        Returns:
            Dictionary of relevant concepts
        """
        from src.monitoring.performance_slos import (
            get_performance_monitor, ComponentType
        )

        monitor = get_performance_monitor()
        context = {
            "query_terms_count": len(query_terms),
            "cold_start": not self._full_graph_loaded
        }

        with monitor.measure_operation(
            ComponentType.KNOWLEDGE_GRAPH, "load_on_demand", context
        ):
            # Create cache key from query terms
            cache_key = hashlib.md5(
                '|'.join(sorted(query_terms)).encode()
            ).hexdigest()

            if cache_key in self.cache:
                logger.debug(f"Cache hit for query terms: {query_terms}")
                return self.cache[cache_key]

            # Execute query and cache result
            result = self._query_cached(query_terms)
            self.cache[cache_key] = result

            logger.debug(f"Cached result for query terms: {query_terms}")
            return result
        
    def extract_terms(self) -> None:
        """
        Extract IEC, ENTSOE, and EUR-LEX terms from the graph.

        Uses SPARQL queries to extract terms with their labels.
        """
        from src.exceptions.exceptions import KnowledgeGraphError

        if not self.graph:
            raise KnowledgeGraphError("Graph must be loaded before extracting terms")

        self.iec_terms = self._extract_domain_terms("IEC")
        self.entsoe_terms = self._extract_domain_terms("ENTSOE")
        self.eurlex_terms = self._extract_domain_terms("EUR-LEX")

        logger.info(
            f"Extracted {len(self.iec_terms)} IEC terms, "
            f"{len(self.entsoe_terms)} ENTSOE terms, "
            f"{len(self.eurlex_terms)} EUR-LEX terms"
        )


    def _extract_vocabulary_name_from_uri(self, uri: str) -> Optional[str]:
        """
        Extract vocabulary name from a URI using pattern matching.
        
        Maps URI patterns to the actual vocabulary names from Alliander.esa structure.
        
        Args:
            uri: Full URI string
            
        Returns:
            Vocabulary name or None
        """
        uri_lower = uri.lower()
        
        # LEGAL & REGULATION
        if 'eurlex' in uri_lower or 'europa.eu/eli' in uri_lower:
            return "EUR-Lex Regulation"
        elif 'acer' in uri_lower:
            return "EUR Energy Regulators (ACER)"
        elif 'wetten.nl' in uri_lower or ('dutch' in uri_lower and 'regulation' in uri_lower):
            return "Dutch Regulation Electricity"
        
        # ENERGY MANAGEMENT SYSTEM
        elif 'modulair' in uri_lower or 'msb' in uri_lower:
            return "Alliander Modulair Station Building Blocks"
        elif 'pas1879' in uri_lower or ('bsi' in uri_lower and 'energy' in uri_lower):
            return "PAS1879 - Energy smart appliances (BSI)"
        elif 'entso-e' in uri_lower or 'entsoe' in uri_lower:
            if 'market' in uri_lower or 'role' in uri_lower:
                return "Harmonized Electricity Market Role Model (ENTSO-E)"
            return "ENTSO-E Standards"
        elif 'alliander' in uri_lower and 'energy' in uri_lower:
            return "Alliander Energy System"
        
        # IEC STANDARDS (most detailed mapping)
        elif '61968' in uri_lower or '/iec61968/' in uri_lower:
            return "IEC 61968 - Meters, Assets and Work"
        
        elif '61970' in uri_lower or '/iec61970/' in uri_lower or 'tc57/ns/cim' in uri_lower:
            # Check for specific CGMES packages in URI
            if 'coreequipment' in uri_lower or '/equipment-eu' in uri_lower:
                return "IEC 61970 - Core Equipment (CGMES)"
            elif 'diagram' in uri_lower:
                return "IEC 61970 - Diagram Layout (CGMES)"
            elif 'boundary' in uri_lower:
                return "IEC 61970 - Equipment Boundary (CGMES)"
            elif 'geographical' in uri_lower or 'location' in uri_lower:
                return "IEC 61970 - Geographical Location (CGMES)"
            elif 'operation' in uri_lower or '/operation-eu' in uri_lower:
                return "IEC 61970 - Operation (CGMES)"
            elif 'shortcircuit' in uri_lower:
                return "IEC 61970 - Short Circuit (CGMES)"
            elif 'statevariables' in uri_lower:
                return "IEC 61970 - State Variables (CGMES)"
            elif 'steadystatehypothesis' in uri_lower:
                return "IEC 61970 - Steady State Hypothesis (CGMES)"
            elif 'topology' in uri_lower:
                return "IEC 61970 - Topology (CGMES)"
            else:
                return "IEC 61970 - Common Information Model"
        
        elif '62325' in uri_lower:
            return "IEC 62325 - Market Model"
        
        elif '62746' in uri_lower:
            return "IEC 62746 - Demand Site Resource"
        
        # ARCHITECTURE
        elif 'ai' in uri_lower and ('ontology' in uri_lower or 'artificial' in uri_lower):
            return "Alliander Artificial Intelligence Ontology"
        elif 'archimate' in uri_lower or 'archi:' in uri_lower:
            return "ArchiMate Model"
        
        # LEGACY / OUT OF DATE
        elif 'confluence' in uri_lower:
            return "Alliander Found on Confluence - Glossary (Out of Date)"
        elif 'poolparty' in uri_lower:
            return "Alliander Poolparty - Glossary (Out of Date)"
        
        # Generic IEC fallback
        elif 'iec' in uri_lower:
            return "IEC Standards"
        
        return None

    def _extract_domain_terms(self, domain: str) -> Dict[str, str]:
        """
        Extract terms for a specific domain using SPARQL.

        FIXED: Uses simple query + Python filtering to avoid SPARQL errors.

        Args:
            domain: Domain identifier (IEC, ENTSOE, EUR-LEX)

        Returns:
            Dictionary mapping concept URIs to their preferred labels
        """
        domain_patterns = {
            "IEC": "iec.ch",
            "ENTSOE": "entsoe.eu",
            "EUR-LEX": "europa.eu"
        }

        pattern = domain_patterns.get(domain, domain.lower())
        terms = {}

        try:
            # Simple query without FILTER - filter in Python instead
            query = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT ?concept ?label WHERE {
                ?concept skos:prefLabel ?label .
            }
            """

            results = self.graph.query(query)
            
            # Filter in Python
            for row in results:
                concept = str(row[0])
                label = str(row[1])
                
                # Check if concept URI contains the pattern
                if pattern.lower() in concept.lower():
                    terms[concept] = label
            
            logger.debug(f"Extracted {len(terms)} terms for domain {domain}")

        except Exception as e:
            logger.error(f"Failed to extract {domain} terms: {e}")

        return terms

    def get_vocabulary_membership(self, citation_id: str) -> Optional[str]:
        """
        Determine which vocabulary/subdictionary a concept belongs to.
        
        This queries the KG directly to find vocabulary membership based on:
        1. Concept scheme (skos:inScheme)
        2. URI patterns
        3. Collection membership
        
        Args:
            citation_id: Citation ID (e.g., 'skos:1502', 'iec:ActivePower')
            
        Returns:
            Vocabulary name or None if not found
        """
        if not self.graph or not self._full_graph_loaded:
            return None
        
        # Get the full URI for this citation
        metadata = self.get_citation_metadata(citation_id)
        if not metadata:
            return None
        
        concept_uri = metadata.get("uri")
        if not concept_uri:
            return None
        
        try:
            # Convert URI string to URIRef for SPARQL
            from rdflib import URIRef
            concept = URIRef(concept_uri)
            
            # Method 1: Check skos:inScheme (if vocabulary uses SKOS properly)
            for s, p, o in self.graph.triples((concept, None, None)):
                pred_str = str(p)
                if 'inScheme' in pred_str or 'scheme' in pred_str.lower():
                    scheme_uri = str(o)
                    
                    # Extract vocabulary name from scheme URI
                    vocab_name = self._extract_vocabulary_name_from_uri(scheme_uri)
                    if vocab_name:
                        logger.info(f"Found vocabulary '{vocab_name}' for {citation_id} via skos:inScheme")
                        return vocab_name
            
            # Method 2: Check skos:member or skos:Collection membership
            for s, p, o in self.graph.triples((None, None, concept)):
                pred_str = str(p)
                if 'member' in pred_str.lower():
                    collection_uri = str(s)
                    
                    vocab_name = self._extract_vocabulary_name_from_uri(collection_uri)
                    if vocab_name:
                        logger.info(f"Found vocabulary '{vocab_name}' for {citation_id} via collection membership")
                        return vocab_name
            
            # Method 3: Analyze URI pattern itself
            vocab_name = self._extract_vocabulary_name_from_uri(concept_uri)
            if vocab_name:
                logger.info(f"Found vocabulary '{vocab_name}' for {citation_id} via URI pattern")
                return vocab_name
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to determine vocabulary membership for {citation_id}: {e}")
            return None

    def save_vocabularies(self, output_path: Optional[Path] = None) -> None:
        """
        Save extracted vocabularies to JSON file.

        Args:
            output_path: Path to save vocabularies.
                        Defaults to config/vocabularies.json
        """
        output_path = output_path or Path("config/vocabularies.json")

        # Load existing vocabularies if file exists
        existing_vocab = {}
        if output_path.exists():
            with open(output_path, "r") as f:
                existing_vocab = json.load(f)

        # Update with extracted terms
        extracted_terms = {
            "extracted_terms": {
                "iec_terms": self.iec_terms,
                "entsoe_terms": self.entsoe_terms,
                "eurlex_terms": self.eurlex_terms,
                "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "triple_count": len(self.graph) if self.graph else 0,
            }
        }

        # Merge with existing vocabulary structure
        updated_vocab = {**existing_vocab, **extracted_terms}

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(updated_vocab, f, indent=2)

        logger.info(f"Saved vocabularies to {output_path}")
        
    def _query_cached(self, terms: List[str]) -> Dict[str, Any]:
        """
        Execute SPARQL query for terms with caching.
        
        FIXED: Uses simple query + Python filtering to avoid SPARQL errors.
        """
        if not self.graph:
            return {}

        concepts = {}

        try:
            # Simple query without complex FILTER
            sparql_query = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX iec: <http://iec.ch/TC57/CIM#>
            PREFIX entsoe: <http://entsoe.eu/CIM/SchemaExtension/3/1#>

            SELECT ?concept ?label ?definition WHERE {
                ?concept skos:prefLabel ?label .
                OPTIONAL { ?concept skos:definition ?definition }
            }
            """

            results = self.graph.query(sparql_query)
            
            # Filter in Python
            for term in terms:
                term_lower = term.lower()
                
                for row in results:
                    if hasattr(row, 'concept') and hasattr(row, 'label'):
                        label = str(row.label)
                        label_lower = label.lower()
                        
                        # Check if term matches
                        if term_lower in label_lower:
                            concept_uri = str(row.concept)
                            concept_key = (
                                concept_uri.split('#')[-1]
                                if '#' in concept_uri
                                else concept_uri
                            )
                            concepts[concept_key] = {
                                'label': label,
                                'definition': (
                                    str(row.definition)
                                    if hasattr(row, 'definition') and row.definition
                                    else ''
                                ),
                                'uri': concept_uri
                            }
        
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)

        return concepts

    def _extract_terms(self) -> None:
        """
        Extract domain terms from the loaded graph for vocabulary hydration.
        
        FIXED: Uses simple queries + Python filtering.
        """
        if not self.graph:
            return

        try:
            # Simple query for all concepts
            simple_query = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT DISTINCT ?s ?label WHERE {
                ?s skos:prefLabel ?label .
            }
            """

            results = list(self.graph.query(simple_query))
            
            # Filter for IEC terms in Python
            self.iec_terms = {
                str(row.label): str(row.label)
                for row in results 
                if hasattr(row, 'label') and hasattr(row, 's') 
                and 'iec' in str(row.s).lower()
            }

            # Filter for ENTSOE terms in Python
            self.entsoe_terms = {
                str(row.label): str(row.label)
                for row in results 
                if hasattr(row, 'label') and hasattr(row, 's') 
                and 'entsoe' in str(row.s).lower()
            }

            logger.info(
                f"Extracted {len(self.iec_terms)} IEC terms, "
                f"{len(self.entsoe_terms)} ENTSOE terms"
            )

        except Exception as e:
            logger.error(f"Failed to extract terms: {e}", exc_info=True)

    def query_definitions(self, terms: List[str], trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph for definitions with enhanced ranking.
        
        FIXED VERSION:
        - Uses simple SPARQL query to get ALL concepts
        - Filters in Python to avoid postParse2() errors
        - Proper citation ID extraction with prefix preservation

        Args:
            terms: List of terms to search for

        Returns:
            List of concept dictionaries with scores and citations
        """
        if trace_id:
            tracer.trace_info(trace_id, "kg_loader", "query_definitions",
                            terms=terms, terms_count=len(terms))

        if not self._full_graph_loaded:
            logger.info("Graph not fully loaded yet, waiting up to 10 seconds...")
            if trace_id:
                tracer.trace_info(trace_id, "kg_loader", "wait_for_graph",
                                status="waiting", max_wait_seconds=10)

            wait_time = 0.0
            max_wait = 10.0

            while not self._full_graph_loaded and wait_time < max_wait:
                time.sleep(0.5)
                wait_time += 0.5

            if not self._full_graph_loaded:
                logger.warning(f"Graph failed to load within {max_wait}s, returning empty results")
                if trace_id:
                    tracer.trace_info(trace_id, "kg_loader", "wait_for_graph",
                                    status="timeout", wait_time=wait_time)
                return []
            else:
                logger.info(f"Graph loaded successfully after {wait_time}s wait")
                if trace_id:
                    tracer.trace_info(trace_id, "kg_loader", "wait_for_graph",
                                    status="success", wait_time=wait_time)

        if not self.graph:
            logger.warning("Graph object is None, returning empty results")
            if trace_id:
                tracer.trace_info(trace_id, "kg_loader", "query_definitions",
                                error="graph_object_none")
            return []

        stop_words = {
            'what', 'is', 'an', 'a', 'the', 'and', 'or', 'but', 'in', 'on',
            'at', 'to', 'for', 'of', 'with', 'by', 'are', 'were', 'was', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
            'terms?', 'terms'
        }

        filtered_terms = []
        for term in terms:
            clean_term = term.lower().strip('.,!?;:()[]{}"\'-')
            if clean_term not in stop_words and len(clean_term) > 2:
                filtered_terms.append(clean_term)

        logger.info(f"Filtered terms from {terms} to {filtered_terms}")

        if not filtered_terms:
            logger.warning("No meaningful terms found after filtering stop words")
            return []

        all_results = []

        # CRITICAL FIX: Use simple query + Python filtering to avoid SPARQL parsing errors
        try:
            # ALTERNATIVE APPROACH: Iterate through graph triples directly
            # This completely bypasses SPARQL parsing issues
            logger.info("Using direct triple iteration to avoid SPARQL parsing errors")
            
            all_concepts = []
            
            # Iterate through all triples to find skos:Concept instances
            for subj, pred, label in self.graph.triples((None, SKOS.prefLabel, None)):
                    concept_uri = str(subj)
                    label = str(label)
                    
                    # Find definition
                    definition = ""
                    for s, p, def_obj in self.graph.triples((subj, SKOS.definition, None)):
                        definition = str(def_obj)
                        break
                    
                    all_concepts.append({
                        'concept_uri': concept_uri,
                        'label': label,
                        'definition': definition
                    })
            
            logger.info(f"Found {len(all_concepts)} concepts via direct triple iteration")
            
            # Filter concepts by search terms in Python
            for term in filtered_terms:
                term_lower = term.lower()
                
                for concept in all_concepts:
                    label = concept['label']
                    label_lower = label.lower()
                    
                    # Check if this concept matches the current term
                    if term_lower in label_lower:
                        concept_uri = concept['concept_uri']
                        definition = concept['definition']
                        
                        # Calculate score
                        exact_match = label_lower == term_lower
                        score = 100.0 if exact_match else 75.0
                        
                        # Extract proper citation ID (CRITICAL FIX)
                        citation_id = self._extract_citation_id(concept_uri)
                        
                        all_results.append({
                            "concept": concept_uri,
                            "label": label,
                            "definition": definition,
                            "score": score,
                            "citation_id": citation_id,
                            "type": "Knowledge Graph Concept",
                            "term_matched": term
                        })
            
            logger.info(f"Direct iteration found {len(all_results)} matching results")
            
        except Exception as e:
            logger.error(f"Direct triple iteration failed: {e}", exc_info=True)
            return []

        # Sort by score (exact matches first) and label
        all_results.sort(key=lambda x: (-x['score'], x['label']))

        # Return top 10 results
        top_results = all_results[:10]
        logger.info(f"Returning top {len(top_results)} results for terms: {filtered_terms}")
        
        return top_results

    # CORRECTED VERSION for src/knowledge/kg_loader.py
    # Based on actual URI patterns found in the knowledge graph
    #
    # KEY FINDINGS:
    # - IEC uses: http://iec.ch/TC57/ns/CIM/ (NOT http://iec.ch/TC57/CIM#)
    # - SKOS concepts are in: https://vocabs.alliander.com/def/ppt/ (Alliander's own vocab)
    # - EUR-LEX uses: http://data.europa.eu/eli/terms/
    # - ENTSOE uses: http://vocabs.entsoe.com/terms/
    # - LIDO uses: http://linkeddata.overheid.nl/terms/ (Dutch government data)

    def citation_exists(self, citation_id: str) -> bool:
        """
        Check if citation ID exists in knowledge graph.
        
        CORRECTED VERSION: Uses actual namespace patterns from the graph.
        
        Args:
            citation_id: Citation in format 'skos:Asset', 'iec:ActivePower', etc.
            
        Returns:
            True if citation exists in graph, False otherwise
        """
        if not self.graph or not self._full_graph_loaded:
            logger.warning("Graph not loaded, cannot validate citation")
            return False
        
        # Extract namespace and local part
        if ":" not in citation_id:
            return False
        
        namespace, local_part = citation_id.split(":", 1)
        
        # CORRECTED: Use actual namespace patterns from the graph
        namespace_patterns = {
            # Alliander SKOS
            "skos": [
                "https://vocabs.alliander.com/def/ppt/",
                "http://vocabs.alliander.com/terms/"
            ],
            # IEC Standards (split by series for correct citation format)
            "iec61968": [
                "http://iec.ch/TC57/IEC61968/"
            ],
            "iec61970": [
                "http://iec.ch/TC57/IEC61970/",
                "http://iec.ch/TC57/ns/CIM/"
            ],
            "iec62325": [
                "http://iec.ch/TC57/IEC62325/"
            ],
            "iec62746": [
                "http://iec.ch/TC57/IEC62746/"
            ],
            # ENTSO-E
            "entsoe": [
                "http://vocabs.entsoe.com/terms/",
                "http://entsoe.eu/"
            ],
            # EUR-LEX Regulation
            "eurlex": [
                "http://data.europa.eu/eli/terms/"
            ],
            # Dutch Government
            "lido": [
                "http://linkeddata.overheid.nl/terms/"
            ],
            "dutch": [
                "http://wetten.nl/terms/"
            ],
            # ACER
            "acer": [
                "http://acer.europa.eu/terms/"
            ],
            # Alliander specific
            "modulair": [
                "http://vocabs.alliander.com/modulair/"
            ],
            "aiontology": [
                "http://vocabs.alliander.com/ai/"
            ],
            # ArchiMate
            "archi": [
                "http://www.archimatetool.com/"
            ],
            # BSI / PAS1879
            "pas1879": [
            "http://linkeddata.bsigroup.com/pas/1879/",
            "http://bsi-group.com/pas1879/"
            ]
        }
        patterns = namespace_patterns.get(namespace)
        if not patterns:
            logger.debug(f"Unknown namespace: {namespace}")
            return False


        try:
            # OPTIMIZED: Only check subjects with prefLabels
            for subj in self.graph.subjects(SKOS.prefLabel, None):
                subj_str = str(subj)
            
                for pattern in patterns:
                    if subj_str.startswith(pattern):
                        # CRITICAL FIX: Extract URI local part properly
                        uri_local_part = subj_str[len(pattern):]
                        
                        if '/' in uri_local_part:
                            uri_local_part = uri_local_part.split('/')[-1]
                        
                        if '#' in uri_local_part:
                            uri_local_part = uri_local_part.split('#')[-1]
                        
                        # EXACT match
                        if uri_local_part.lower() == local_part.lower():
                            logger.debug(f"Citation {citation_id} found: {subj_str}")
                            return True
            
            logger.debug(f"Citation {citation_id} not found in graph")
            return False

        except Exception as e:
            logger.error(f"Citation existence check failed for {citation_id}: {e}")
            return False

    def get_all_vocabularies(self) -> Dict[str, List[str]]:
        """
        Get a list of all vocabularies in the KG with their concept counts.

        Returns:
            Dictionary mapping vocabulary names to lists of concept URIs
        """
        if not self.graph or not self._full_graph_loaded:
            return {}

        vocabularies = {}

        try:
            # Iterate through all concepts
            for subj, pred, obj in self.graph:
                if 'prefLabel' in str(pred):
                    concept_uri = str(subj)
                    citation_id = self._extract_citation_id(concept_uri)

                    if citation_id:
                        # Determine vocabulary
                        vocab_name = self._extract_vocabulary_name_from_uri(concept_uri)

                        if vocab_name:
                            if vocab_name not in vocabularies:
                                vocabularies[vocab_name] = []
                            vocabularies[vocab_name].append(citation_id)

            # Sort by size
            vocabularies = dict(sorted(
                vocabularies.items(),
                key=lambda x: len(x[1]),
                reverse=True
            ))

            logger.info(f"Found {len(vocabularies)} vocabularies in KG:")
            for vocab, concepts in vocabularies.items():
                logger.info(f"  - {vocab}: {len(concepts)} concepts")

            return vocabularies

        except Exception as e:
            logger.error(f"Failed to get all vocabularies: {e}")
            return {}

    # Add to kg_loader.py after line 800 (after get_all_valid_citations)
    def get_all_concepts_with_definitions(self) -> List[Tuple[str, str, str]]:
        """
        Extract all concepts with labels and definitions from the graph.
        
        Returns:
            List of tuples: (concept_uri, label, definition)
        """
        if not self.graph or not self._full_graph_loaded:
            logger.warning("Graph not loaded, cannot extract concepts")
            return []
        
        concepts = []
        
        try:
            # Iterate through all subjects with prefLabel
            for subj in self.graph.subjects(SKOS.prefLabel, None):
                concept_uri = str(subj)
                
                # Get label
                label = None
                for s, p, o in self.graph.triples((subj, SKOS.prefLabel, None)):
                    label = str(o)
                    break
                
                # Get definition
                definition = None
                for s, p, o in self.graph.triples((subj, SKOS.definition, None)):
                    definition = str(o)
                    break
                
                if label:
                    concepts.append((concept_uri, label, definition))
            
            logger.info(f"Extracted {len(concepts)} concepts with definitions")
            return concepts
            
        except Exception as e:
            logger.error(f"Failed to extract concepts: {e}")
            return []

    def get_all_valid_citations(self, namespace: str = None) -> List[str]:
        """
        Get list of all valid citation IDs from knowledge graph.

        CORRECTED VERSION: Uses actual namespace patterns.

        Args:
            namespace: Optional namespace filter ('skos', 'iec', 'entsoe', 'eurlex', 'lido')

        Returns:
            List of citation IDs
        """
        if not self.graph or not self._full_graph_loaded:
            logger.warning("Graph not loaded, cannot get citations")
            return []

        # CORRECTED: Use actual namespace patterns
        namespace_patterns = {
            "skos": [
                "https://vocabs.alliander.com/def/ppt/",
                "http://vocabs.alliander.com/terms/"
            ],
            "iec": [
                "http://iec.ch/TC57/ns/CIM/",
                "http://iec.ch/TC57/IEC61968/",
                "http://iec.ch/TC57/IEC61970/"
            ],
            "entsoe": [
                "http://vocabs.entsoe.com/terms/"
            ],
            "eurlex": [
                "http://data.europa.eu/eli/terms/"
            ],
            "lido": [
                "http://linkeddata.overheid.nl/terms/"
            ]
        }

        citations = []
        seen_subjects = set()

        try:
            logger.info("Extracting citations using optimized subject iteration...")

            # FIXED: Iterate through subjects with prefLabels (optimized)
            for subj in self.graph.subjects(SKOS.prefLabel, None):
                if subj not in seen_subjects:
                    subj_str = str(subj)
                    seen_subjects.add(subj)

                    # Extract citation ID from URI
                    citation_id = self._extract_citation_id(subj_str)

                    if citation_id:
                        # Filter by namespace if specified
                        if namespace:
                            if citation_id.startswith(f"{namespace}:"):
                                citations.append(citation_id)
                        else:
                            citations.append(citation_id)

            logger.info(f"Extracted {len(citations)} citations" + 
                    (f" for namespace '{namespace}'" if namespace else ""))
            return citations

        except Exception as e:
            logger.error(f"Failed to get valid citations: {e}")
            return []

    def _extract_citation_id(self, uri: str) -> Optional[str]:
        """
        Extract citation ID EXACTLY as it appears in Turtle syntax.
        
        This preserves the format from your RDF files:
        - eurlex:631-20 (not eurlex:ActivePower)
        - iec61968:Asset (not iec:Asset)
        - etc.
        
        Examples:
            http://data.europa.eu/eli/terms/631-20 -> eurlex:631-20
            http://iec.ch/TC57/IEC61968/Asset -> iec61968:Asset
            http://iec.ch/TC57/IEC61970/CIM/CoreEquipment-EU/DCLine -> iec61970:DCLine
            https://vocabs.alliander.com/def/ppt/1502 -> skos:1502
        
        Args:
            uri: Full URI string
        
        Returns:
            Citation ID in format matching Turtle syntax or None
        """
        
        # ============= EUR-LEX REGULATION =============
        if "data.europa.eu/eli/terms/" in uri:
            # Format: http://data.europa.eu/eli/terms/631-20
            local_part = uri.split("/terms/")[-1]
            return f"eurlex:{local_part}"
        
        # ============= IEC STANDARDS (Multiple Series) =============
        # IEC 61968 - Meters, Assets and Work
        elif "iec.ch/TC57/IEC61968/" in uri:
            # Format: http://iec.ch/TC57/IEC61968/Asset
            local_part = uri.split("/IEC61968/")[-1]
            # Handle nested paths
            if '/' in local_part:
                local_part = local_part.split('/')[-1]
            local_part = local_part.split('#')[-1]
            return f"iec61968:{local_part}"
        
        # IEC 61970 - Common Information Model (CGMES sub-standards)
        elif "iec.ch/TC57/IEC61970/" in uri or "iec.ch/TC57/ns/CIM/" in uri:
            # Format: http://iec.ch/TC57/IEC61970/CIM/CoreEquipment-EU/DCLine
            # or: http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/DCLine
            if "/IEC61970/" in uri:
                local_part = uri.split("/IEC61970/")[-1]
            else:
                local_part = uri.split("/CIM/")[-1]
            
            # Handle nested paths (get last segment)
            if '/' in local_part:
                local_part = local_part.split('/')[-1]
            local_part = local_part.split('#')[-1]
            return f"iec61970:{local_part}"
        
        # IEC 62325 - Market Model
        elif "iec.ch/TC57/IEC62325/" in uri:
            local_part = uri.split("/IEC62325/")[-1]
            if '/' in local_part:
                local_part = local_part.split('/')[-1]
            local_part = local_part.split('#')[-1]
            return f"iec62325:{local_part}"
        
        # IEC 62746 - Demand Site Resource
        elif "iec.ch/TC57/IEC62746/" in uri:
            local_part = uri.split("/IEC62746/")[-1]
            if '/' in local_part:
                local_part = local_part.split('/')[-1]
            local_part = local_part.split('#')[-1]
            return f"iec62746:{local_part}"
        
        # ============= ENTSO-E =============
        elif "entsoe.eu" in uri.lower() or "entso-e" in uri.lower() or "vocabs.entsoe.com" in uri:
            local_part = uri.split("/")[-1]
            local_part = local_part.split('#')[-1]
            return f"entsoe:{local_part}"
        
        # ============= ALLIANDER VOCABULARIES =============
        elif "vocabs.alliander.com" in uri:
            local_part = uri.split("/")[-1]
            local_part = local_part.split('#')[-1]
            
            # Detect specific Alliander sub-vocabularies
            if "modulair" in uri.lower() or "msb" in uri.lower():
                return f"modulair:{local_part}"
            elif "ai" in uri.lower() and "ontology" in uri.lower():
                return f"aiontology:{local_part}"
            else:
                return f"skos:{local_part}"
        
        # ============= ACER (EU Energy Regulators) =============
        elif "acer.europa.eu" in uri.lower() or "acer-remit" in uri.lower():
            local_part = uri.split("/")[-1]
            local_part = local_part.split('#')[-1]
            return f"acer:{local_part}"
        
        # ============= DUTCH REGULATION =============
        elif "wetten.nl" in uri.lower() or "dutch" in uri.lower() or "linkeddata.overheid.nl" in uri:
            local_part = uri.split("/")[-1]
            local_part = local_part.split('#')[-1]
            return f"dutch:{local_part}"
        
        # ============= PAS1879 / BSI =============
        elif "pas1879" in uri.lower() or "bsi-group" in uri.lower() or "linkeddata.bsigroup.com" in uri.lower():
            # Format: http://linkeddata.bsigroup.com/pas/1879/Asset
            local_part = uri.split("/")[-1]
            local_part = local_part.split('#')[-1]
            return f"pas1879:{local_part}"
        
        # ============= ARCHIMATE =============
        elif "archi" in uri.lower() or "archimate" in uri.lower():
            local_part = uri.split("/")[-1]
            local_part = local_part.split('#')[-1]
            return f"archi:{local_part}"
        
        # ============= LEGACY / OUT OF DATE =============
        elif "confluence.alliander" in uri.lower():
            local_part = uri.split("/")[-1]
            local_part = local_part.split('#')[-1]
            return f"confluence:{local_part}"

        elif "poolparty.alliander" in uri.lower():
            local_part = uri.split("/")[-1]
            local_part = local_part.split('#')[-1]
            return f"poolparty:{local_part}"

        # ============= LIDO (Dutch Government) =============
        elif "linkeddata.overheid.nl" in uri or "linkeddata/terms/" in uri.lower():
            local_part = uri.split("/")[-1]
            local_part = local_part.split('#')[-1]
            return f"lido:{local_part}"

        # ============= FALLBACK =============
        else:
            # Try to preserve namespace from URI
            local_part = uri.split("/")[-1]
            local_part = local_part.split('#')[-1]

            if local_part and "//" in uri:
                domain = uri.split("//")[1].split("/")[0]
                prefix = domain.split('.')[0]
                return f"{prefix}:{local_part}"

            return None
    def get_citation_metadata(self, citation_id: str) -> Optional[Dict]:
        """
        Get metadata for a citation (label, definition, type).

        CORRECTED VERSION: Uses actual namespace patterns with exact matching.

        Args:
            citation_id: Citation ID (e.g., 'skos:1502', 'iec:DCLine')

        Returns:
            Dictionary with metadata or None if not found
        """
        if not self.citation_exists(citation_id):
            return None
        
        # Extract namespace and local part
        namespace, local_part = citation_id.split(":", 1)
        
        # CORRECTED: Use actual namespace patterns
        namespace_patterns = {
            # Alliander SKOS
            "skos": [
                "https://vocabs.alliander.com/def/ppt/",
                "http://vocabs.alliander.com/terms/"
            ],
            # IEC Standards (split by series for correct citation format)
            "iec61968": [
                "http://iec.ch/TC57/IEC61968/"
            ],
            "iec61970": [
                "http://iec.ch/TC57/IEC61970/",
                "http://iec.ch/TC57/ns/CIM/"
            ],
            "iec62325": [
                "http://iec.ch/TC57/IEC62325/"
            ],
            "iec62746": [
                "http://iec.ch/TC57/IEC62746/"
            ],
            # ENTSO-E
            "entsoe": [
                "http://vocabs.entsoe.com/terms/",
                "http://entsoe.eu/"
            ],
            # EUR-LEX Regulation
            "eurlex": [
                "http://data.europa.eu/eli/terms/"
            ],
            # Dutch Government
            "lido": [
                "http://linkeddata.overheid.nl/terms/"
            ],
            "dutch": [
                "http://wetten.nl/terms/"
            ],
            # ACER
            "acer": [
                "http://acer.europa.eu/terms/"
            ],
            # Alliander specific
            "modulair": [
                "http://vocabs.alliander.com/modulair/"
            ],
            "aiontology": [
                "http://vocabs.alliander.com/ai/"
            ],
            # ArchiMate
            "archi": [
                "http://www.archimatetool.com/"
            ],
            # BSI / PAS1879
            "pas1879": [
            "http://linkeddata.bsigroup.com/pas/1879/",
            "http://bsi-group.com/pas1879/"
            ]
        }
        
        patterns = namespace_patterns.get(namespace)
        if not patterns:
            return None
        
        try:
            # OPTIMIZED: Only iterate through subjects with prefLabel
            # This is much faster than iterating ALL triples
            for subj in self.graph.subjects(SKOS.prefLabel, None):
                subj_str = str(subj)
                
                # Check if subject matches any namespace pattern
                for pattern in patterns:
                    if subj_str.startswith(pattern):
                        # CRITICAL FIX: Extract the URI's local part properly
                        # and do EXACT matching instead of substring matching
                        uri_local_part = subj_str[len(pattern):]
                        
                        # Handle nested paths in IEC URIs (e.g., /CoreEquipment-EU/DCLine)
                        if '/' in uri_local_part:
                            uri_local_part = uri_local_part.split('/')[-1]
                        
                        # Handle fragments (e.g., #DCLine)
                        if '#' in uri_local_part:
                            uri_local_part = uri_local_part.split('#')[-1]
                        
                        # EXACT case-insensitive match (not substring!)
                        if uri_local_part.lower() == local_part.lower():
                            # Found exact match! Now extract metadata
                            label = None
                            definition = None
                            
                            # Get all properties of this subject
                            for s, p, o in self.graph.triples((subj, None, None)):
                                pred_str = str(p)
                                
                                if 'prefLabel' in pred_str:
                                    label = str(o)
                                elif 'definition' in pred_str:
                                    definition = str(o)
                            
                            if label:  # Only return if we found at least a label
                                return {
                                    "citation_id": citation_id,
                                    "uri": subj_str,
                                    "label": label,
                                    "definition": definition
                                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get citation metadata for {citation_id}: {e}")
            return None

    def hydrate_vocabularies(self) -> Tuple[List[str], List[str]]:
        """
        Auto-hydrate vocabularies from SKOS + the knowledge graph.
        
        FIXED: Gracefully handles SPARQL errors, returns empty if query fails.

        Returns:
            Tuple of (iec_terms, entsoe_terms) lists
        """
        iec_terms: List[str] = []
        entsoe_terms: List[str] = []

        if not self.graph:
            logger.warning("Graph not loaded, returning empty vocabularies")
            return iec_terms, entsoe_terms

        try:
            # Try to get all concepts - if this fails, just return empty
            # This is not critical for the system to work
            simple_query = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT DISTINCT ?s ?label WHERE {
                ?s skos:prefLabel ?label .
            }
            """

            logger.info("Attempting to hydrate vocabularies from knowledge graph...")
            results = list(self.graph.query(simple_query))
            
            # Filter in Python
            for row in results:
                if hasattr(row, 'label') and hasattr(row, 's'):
                    label = str(row.label)
                    concept = str(row.s).lower()
                    label_lower = label.lower()
                    
                    # IEC/Energy terms
                    if ('iec' in concept or 
                        'power' in label_lower or 
                        'grid' in label_lower or 
                        'energy' in label_lower):
                        if label not in iec_terms:
                            iec_terms.append(label)
                    
                    # ENTSOE terms
                    if ('entsoe' in concept or 
                        'market' in label_lower or 
                        'bidding' in label_lower):
                        if label not in entsoe_terms:
                            entsoe_terms.append(label)

            logger.info(
                f"Successfully hydrated vocabularies: {len(iec_terms)} IEC terms, "
                f"{len(entsoe_terms)} ENTSOE terms"
            )

        except Exception as e:
            # This is non-critical - vocabulary hydration is optional
            logger.warning(f"Failed to hydrate vocabularies (non-critical): {e}")
            logger.info("Continuing without vocabulary hydration - core functionality will still work")

        return iec_terms, entsoe_terms

    def get_concepts_by_prefix(self, prefix: str) -> Dict[str, str]:
        """
        Get all concepts with a specific prefix.
        
        FIXED: Uses simple query + Python filtering.
        """
        if not self.graph:
            return {}

        concepts = {}
        
        try:
            # Simple query without FILTER
            sparql_query = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT ?concept ?label WHERE {
                ?concept skos:prefLabel ?label .
            }
            """

            results = self.graph.query(sparql_query)
            
            # Filter in Python
            for row in results:
                if hasattr(row, 'concept') and hasattr(row, 'label'):
                    concept_uri = str(row.concept)
                    
                    # Check if starts with prefix
                    if concept_uri.startswith(prefix):
                        concept_id = (
                            concept_uri.split('#')[-1]
                            if '#' in concept_uri
                            else concept_uri
                        )
                        concepts[concept_id] = str(row.label)

        except Exception as e:
            logger.error(f"Failed to get concepts by prefix {prefix}: {e}", exc_info=True)

        return concepts

    def is_full_graph_loaded(self) -> bool:
        """Check if the full graph is loaded."""
        with self._load_lock:
            return self._full_graph_loaded

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded graph."""
        if not self.graph:
            return {"loaded": False, "triples": 0}

        return {
            "loaded": True,
            "triples": len(self.graph),
            "full_graph_loaded": self._full_graph_loaded,
            "load_time_ms": self.load_time_ms,
            "cache_size": len(self.cache),
            "iec_terms": len(self.iec_terms),
            "entsoe_terms": len(self.entsoe_terms)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the knowledge graph for backward compatibility.

        Returns:
            Dictionary with comprehensive graph statistics
        """
        if not self.graph:
            return {
                "loaded": False,
                "total_triples": 0,
                "namespaces": {},
                "top_predicates": {},
                "iec_terms_count": 0,
                "entsoe_terms_count": 0,
                "eurlex_terms_count": 0
            }

        # Get namespace information
        namespaces = {}
        for prefix, namespace in self.graph.namespaces():
            namespaces[str(prefix)] = str(namespace)

        # Get top predicates
        predicate_query = """
        SELECT ?p (COUNT(*) as ?count) WHERE {
            ?s ?p ?o .
        } GROUP BY ?p ORDER BY DESC(?count) LIMIT 10
        """

        top_predicates = {}
        try:
            for row in self.graph.query(predicate_query):
                predicate = str(row[0])
                count = int(row[1])
                top_predicates[predicate] = count
        except Exception:
            # If query fails, continue with empty predicates
            pass

        return {
            "loaded": True,
            "total_triples": len(self.graph),
            "full_graph_loaded": self._full_graph_loaded,
            "load_time_ms": self.load_time_ms,
            "cache_size": len(self.cache),
            "namespaces": namespaces,
            "top_predicates": top_predicates,
            "iec_terms_count": len(self.iec_terms),
            "entsoe_terms_count": len(self.entsoe_terms),
            "eurlex_terms_count": len(self.eurlex_terms)
        }

    def query(self, sparql_query: str) -> List[Dict[str, Any]]:
        """
        Execute SPARQL query on the graph.

        Args:
            sparql_query: SPARQL query string

        Returns:
            List of result dictionaries

        Raises:
            KnowledgeGraphError: If graph not loaded or query fails
        """
        from src.exceptions.exceptions import KnowledgeGraphError, PerformanceError

        if not self.graph:
            raise KnowledgeGraphError("Graph not loaded. Call load() first.")

        start_time = time.perf_counter()

        try:
            query_result = self.graph.query(sparql_query)
            results = []

            # Convert query results to list of dictionaries
            for row in query_result:
                result_dict = {}
                # For SELECT queries, we have variable names and can access them by index
                for i, value in enumerate(row):
                    # Try to get variable name if available
                    try:
                        var_name = str(query_result.vars[i]) if hasattr(query_result, 'vars') else f"var_{i}"
                    except (IndexError, AttributeError):
                        var_name = f"var_{i}"

                    result_dict[var_name] = str(value) if value else None
                results.append(result_dict)

            query_time_ms = (time.perf_counter() - start_time) * 1000

            # Check performance target (300ms for SPARQL queries)
            if query_time_ms > 300:
                raise PerformanceError(f"Query exceeded 300ms threshold: {query_time_ms:.1f}ms")

            return results

        except Exception as e:
            if isinstance(e, PerformanceError):
                raise
            raise KnowledgeGraphError(f"SPARQL query failed: {e}")

    def _validate_graph(self) -> None:
        """
        Validate the loaded graph meets minimum requirements.

        Raises:
            KnowledgeGraphError: If graph validation fails
        """
        from src.exceptions.exceptions import KnowledgeGraphError

        if not self.graph:
            raise KnowledgeGraphError("No graph loaded")

        triple_count = len(self.graph)

        # Minimum 39100 triples for production
        min_triples = 39100
        if triple_count < min_triples:
            raise KnowledgeGraphError(
                f"Graph has {triple_count} triples, minimum {min_triples} required"
            )