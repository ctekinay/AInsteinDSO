"""
Query Router - Critical input protection layer that prevents vector fallback.

This router enforces STRICT priority-based routing to ensure structured data
is checked FIRST before falling back to embeddings. This is essential for:
1. Performance - structured queries are faster than vector search
2. Accuracy - domain-specific data beats general embeddings
3. Grounding - structured sources provide better citations

ROUTING PRIORITY (STRICT ORDER):
0. ADR queries → unstructured_docs (HIGHEST PRIORITY - LLM-detected)
1. IEC/Energy + ArchiMate terms → structured_model (knowledge graph)
2. TOGAF ADM phases + viewpoints → togaf_method (TOGAF patterns)
3. Everything else → unstructured_docs (vector search)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, TYPE_CHECKING, Any, Tuple

# Use TYPE_CHECKING to avoid circular imports while keeping type hints
if TYPE_CHECKING:
    from src.knowledge.kg_loader import KnowledgeGraphLoader

from src.utils.trace import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer()

class QueryRouter:
    """
    High-performance query router with strict priority-based routing.

    Prevents unnecessary vector searches by checking structured vocabularies first.
    Performance target: < 50ms per routing decision.
    """

    def __init__(self, vocab_path: str = "config/vocabularies.json", kg_loader: Optional["KnowledgeGraphLoader"] = None):
        """
        Initialize router with domain vocabularies.

        Args:
            vocab_path: Path to vocabularies configuration file
            kg_loader: Optional knowledge graph loader for auto-hydration
        """
        self.vocab_path = Path(vocab_path)
        self.vocabularies = {}
        self.routing_terms = {}
        self.performance_target_ms = 50
        self.kg_loader = kg_loader
        self.auto_hydrated = False

        self._load_vocabularies()

        # Auto-hydrate vocabularies from knowledge graph if available
        if self.kg_loader:
            # Wait for KG to load before hydrating
            import time
            logger.info("Waiting for knowledge graph to load...")

            for i in range(10):
                if self.kg_loader.is_full_graph_loaded():
                    logger.info(f"KG loaded after {i*0.5}s")
                    break
                time.sleep(0.5)

            self._auto_hydrate_vocabularies()

        logger.info(f"QueryRouter initialized with {len(self.routing_terms)} term categories")

    def _load_vocabularies(self) -> None:
        """Load domain vocabularies from configuration file."""
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                self.vocabularies = json.load(f)

            # Extract routing terms for performance
            self.routing_terms = self.vocabularies.get("routing_terms", {})

            # Convert to lowercase sets for fast lookup
            self.iec_energy_terms = set(
                term.lower() for term in self.routing_terms.get("iec_energy", [])
            )
            self.togaf_archimate_terms = set(
                term.lower() for term in self.routing_terms.get("togaf_archimate", [])
            )

            # Performance target (50ms for fast routing decisions)
            self.performance_target_ms = 50

            logger.info(f"Loaded {len(self.iec_energy_terms)} IEC/energy terms")
            logger.info(f"Loaded {len(self.togaf_archimate_terms)} TOGAF/ArchiMate terms")

        except Exception as e:
            logger.error(f"Failed to load vocabularies from {self.vocab_path}: {e}")
            # Initialize empty sets to prevent crashes
            self.iec_energy_terms = set()
            self.togaf_archimate_terms = set()
            raise

    def route(self, query: str, trace_id: Optional[str] = None) -> str:
        """
        Route query to appropriate knowledge source.
        
        Priority:
        0. LLM-detected ADR/document queries → unstructured_docs
        1. IEC/Energy + ArchiMate terms → structured_model
        2. TOGAF ADM + viewpoints → togaf_method
        3. Everything else → unstructured_docs
        """
        from src.monitoring.performance_slos import get_performance_monitor, ComponentType

        monitor = get_performance_monitor()
        with monitor.measure_operation(ComponentType.QUERY_ROUTER, "route_query"):
            start_time = time.perf_counter()

            if trace_id:
                tracer.trace_info(trace_id, "query_router", "route",
                                query=query, query_length=len(query))

            if not query or not query.strip():
                logger.warning("Empty query provided, routing to unstructured_docs")
                return "unstructured_docs"

            query_lower = query.lower()

            #✅ PRIORITY 0: ALWAYS use LLM classification (it's fast and accurate)
            # The LLM understands semantic intent better than keyword matching
            try:
                intent, confidence = self._classify_query_intent(query)
                
                # Route based on LLM classification
                if intent in ["adr_query", "document_query", "system_query"] and confidence > 0.7:
                    route = "unstructured_docs"
                    logger.info(f"Query routed to {route} (LLM: {intent}, conf: {confidence:.2f})")
                    if trace_id:
                        tracer.trace_info(trace_id, "query_router", "route_decision",
                                        route=route, reason=f"llm_{intent}", confidence=confidence)
                    self._log_performance(start_time, route)
                    return route
                
                # Log low-confidence classifications for analysis
                if confidence < 0.7:
                    logger.debug(f"Low-confidence classification: {intent} ({confidence:.2f}), continuing to term-based routing")

            except Exception as e:
                logger.warning(f"LLM classification failed: {e}, falling back to term-based routing")

            # PRIORITY 1: Check for IEC/Energy + ArchiMate terms
            if trace_id:
                tracer.trace_info(trace_id, "query_router", "check_structured_terms",
                                query_lower=query_lower[:50])

            if self._contains_structured_terms(query_lower):
                route = "structured_model"
                logger.info(f"Query routed to {route} (contains IEC/ArchiMate terms)")
                if trace_id:
                    tracer.trace_info(trace_id, "query_router", "route_decision",
                                    route=route, reason="structured_terms_found")
                self._log_performance(start_time, route)
                return route

            # PRIORITY 2: Check for TOGAF ADM phases + viewpoints
            if trace_id:
                tracer.trace_info(trace_id, "query_router", "check_togaf_terms",
                                query_lower=query_lower[:50])

            if self._contains_togaf_terms(query_lower):
                route = "togaf_method"
                logger.info(f"Query routed to {route} (contains TOGAF terms)")
                if trace_id:
                    tracer.trace_info(trace_id, "query_router", "route_decision",
                                    route=route, reason="togaf_terms_found")
                self._log_performance(start_time, route)
                return route

            # PRIORITY 3: Default to unstructured search
            route = "unstructured_docs"
            logger.info(f"Query routed to {route} (no domain-specific terms found)")
            if trace_id:
                tracer.trace_info(trace_id, "query_router", "route_decision",
                                route=route, reason="default_fallback")
            self._log_performance(start_time, route)
            return route

    def _classify_query_intent(self, query: str) -> Tuple[str, float]:
        """
        Use LLM to classify query intent for routing decisions.
        
        This uses a synchronous approach compatible with the router's sync context.
        
        Args:
            query: User query
            
        Returns:
            (intent_type, confidence)
        """
        try:
            import os
            import httpx
            import json
            
            # Determine which provider to use
            provider = os.getenv('LLM_PROVIDER', 'groq')
            
            if provider == 'groq':
                api_key = os.getenv('GROQ_API_KEY_1')
                api_url = "https://api.groq.com/openai/v1/chat/completions"
                model = "llama-3.3-70b-versatile"
            elif provider == 'openai':
                api_key = os.getenv('OPENAI_API_KEY')
                api_url = "https://api.openai.com/v1/chat/completions"
                model = "gpt-4o-mini"  # Use cheaper model for classification
            else:
                # For ollama or other providers, skip LLM classification
                return "other", 0.0
            
            if not api_key:
                return "other", 0.0
            
            classification_prompt = f"""Classify this user query into ONE category and provide your confidence level:

    **Categories:**

    1. **adr_query** - User wants content/details of a specific ADR
    Examples: "show adr 0025", "details of ADR on interfaces", "what does ADR 25 say"
    
    2. **system_query** - User asks about the system itself (counts, lists, what exists)
    Examples: "how many ADRs do we have", "list all ADRs", "what ADRs exist", "show me all decisions"
    
    3. **document_query** - User wants content from a document/file
    Examples: "show the PDF", "what's in the document"
    
    4. **definition_query** - User wants a definition/explanation of a concept
    Examples: "what is reactive power", "explain demand response", "define X"
    
    5. **other** - None of the above

    **Query to classify:**
    "{query}"

    **Instructions:**
    - Analyze the query semantics carefully
    - Choose the MOST appropriate category
    - Confidence must be between 0.0 (uncertain) and 1.0 (very certain)
    - Base confidence on how well the query matches category patterns
    - Respond with ONLY valid JSON (no markdown code blocks, no explanation text)

    **Response format:**
    {{"type": "category_name", "confidence": 0.XX}}

    **Example valid responses:**
    {{"type": "adr_query", "confidence": 0.95}}
    {{"type": "system_query", "confidence": 0.88}}
    {{"type": "definition_query", "confidence": 0.75}}
    {{"type": "other", "confidence": 0.60}}"""

            # Make synchronous HTTP request
            with httpx.Client(timeout=5.0) as client:
                response = client.post(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": classification_prompt}],
                        "temperature": 0.1,  # Low temperature for consistent classification
                        "max_tokens": 100
                    }
                )
            
            if response.status_code != 200:
                logger.warning(f"LLM API returned {response.status_code}")
                return "other", 0.0
            
            result_data = response.json()
            content = result_data['choices'][0]['message']['content'].strip()
            
            # Parse JSON response (remove markdown if present)
            import re
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = content.strip()
            
            result = json.loads(content)
            intent = result.get('type', 'other')
            confidence = float(result.get('confidence', 0.5))
            
            # Validate confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            logger.info(f"🤖 LLM classified '{query[:50]}...' as: {intent} (confidence: {confidence:.2f})")
            
            return intent, confidence
            
        except Exception as e:
            logger.warning(f"LLM intent classification failed: {e}")
            return "other", 0.0

    def _might_be_document_query(self, query_lower: str) -> bool:
        """
        STAGE 1: Quick pre-filter to avoid calling LLM for every query.
        
        Only returns True if query MIGHT be about documents/ADRs.
        This is just a performance optimization - false positives are OK.
        The LLM will do the accurate classification in stage 2.
        
        Performance: < 1ms (simple keyword check)
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            True if query might be about documents/ADRs (needs LLM classification)
        """
        # Simple keyword check (fast, no LLM needed)
        document_keywords = [
            'adr', 'decision', 'document', 'file', 'record',
            'details', 'show', 'explain', 'tell me about',
            'latest', 'recent', 'our', 'content of',
            'what does', 'summary of', 'describe'
        ]
        
        # Also check for number patterns that might be ADR numbers
        import re
        if re.search(r'\b\d{4}\b', query_lower):  # 4-digit numbers like "0025"
            return True
        
        return any(keyword in query_lower for keyword in document_keywords)

     
    def _contains_structured_terms(self, query_lower: str) -> bool:
        """
        Check if query contains IEC/Energy or ArchiMate terms.

        Args:
            query_lower: Lowercase query string

        Returns:
            True if structured terms found
        """
        # Check for direct term matches (case insensitive)
        for term in self.iec_energy_terms:
            if term.lower() in query_lower:
                logger.debug(f"Found IEC/energy term: {term}")
                return True

        # Check for ArchiMate element types in TOGAF/ArchiMate terms
        archimate_elements = {
            "actor", "role", "process", "service", "capability",
            "component", "interface", "node", "device"
        }

        for element in archimate_elements:
            if element in query_lower and element in self.togaf_archimate_terms:
                logger.debug(f"Found ArchiMate element: {element}")
                return True

        # Check for IEC standard mentions
        iec_patterns = ["iec ", "iec61968", "iec61970", "iec 61968", "iec 61970"]
        for pattern in iec_patterns:
            if pattern in query_lower:
                logger.debug(f"Found IEC pattern: {pattern}")
                return True

        return False
        

    def _contains_togaf_terms(self, query_lower: str) -> bool:
        """
        Check if query contains TOGAF ADM phases or viewpoints.

        Args:
            query_lower: Lowercase query string

        Returns:
            True if TOGAF terms found
        """
        # Check for ADM phases
        adm_phases = ["phase a", "phase b", "phase c", "phase d", "adm"]
        for phase in adm_phases:
            if phase in query_lower:
                logger.debug(f"Found TOGAF ADM phase: {phase}")
                return True

        # Check for TOGAF-specific terms
        togaf_terms = ["viewpoint", "architecture vision", "business architecture",
                      "application architecture", "technology architecture"]
        for term in togaf_terms:
            if term in query_lower:
                logger.debug(f"Found TOGAF term: {term}")
                return True

        # Check for explicit TOGAF mentions
        if "togaf" in query_lower:
            logger.debug("Found explicit TOGAF mention")
            return True

        return False

    def _log_performance(self, start_time: float, route: str) -> None:
        """
        Log routing performance and warn if target exceeded.

        Args:
            start_time: Start timestamp from time.perf_counter()
            route: Routing destination
        """
        duration_ms = (time.perf_counter() - start_time) * 1000

        if duration_ms > self.performance_target_ms:
            logger.warning(
                f"Router performance: {duration_ms:.1f}ms (target: {self.performance_target_ms}ms) "
                f"for route: {route}"
            )
        else:
            logger.debug(f"Router performance: {duration_ms:.1f}ms → {route}")

    def get_routing_stats(self) -> Dict:
        """
        Get router configuration and statistics.

        Returns:
            Dictionary with router statistics
        """
        return {
            "vocabularies_loaded": bool(self.routing_terms),
            "iec_energy_terms_count": len(self.iec_energy_terms),
            "togaf_archimate_terms_count": len(self.togaf_archimate_terms),
            "performance_target_ms": self.performance_target_ms,
            "vocab_file": str(self.vocab_path),
            "auto_hydrated": self.auto_hydrated,
            "kg_loader_available": self.kg_loader is not None,
            "routing_priorities": [
                "1. IEC/Energy + ArchiMate → structured_model",
                "2. TOGAF ADM + viewpoints → togaf_method",
                "3. Everything else → unstructured_docs"
            ]
        }

    def explain_routing(self, query: str) -> Dict:
        """
        Explain routing decision for a given query (for debugging).

        Args:
            query: Query to analyze

        Returns:
            Dictionary with routing explanation
        """
        query_lower = query.lower()

        explanation = {
            "query": query,
            "route": self.route(query),
            "analysis": {
                "iec_energy_matches": [],
                "archimate_matches": [],
                "togaf_matches": [],
                "decision_reason": ""
            }
        }

        # Find IEC/Energy matches
        for term in self.iec_energy_terms:
            if term in query_lower:
                explanation["analysis"]["iec_energy_matches"].append(term)

        # Find ArchiMate matches
        archimate_elements = {"actor", "role", "process", "service", "capability",
                            "component", "interface", "node", "device"}
        for element in archimate_elements:
            if element in query_lower and element in self.togaf_archimate_terms:
                explanation["analysis"]["archimate_matches"].append(element)

        # Find TOGAF matches
        togaf_patterns = ["phase a", "phase b", "phase c", "phase d", "adm",
                         "viewpoint", "togaf", "architecture vision"]
        for pattern in togaf_patterns:
            if pattern in query_lower:
                explanation["analysis"]["togaf_matches"].append(pattern)

        # Set decision reason
        if explanation["analysis"]["iec_energy_matches"] or explanation["analysis"]["archimate_matches"]:
            explanation["analysis"]["decision_reason"] = "Contains IEC/Energy or ArchiMate terms"
        elif explanation["analysis"]["togaf_matches"]:
            explanation["analysis"]["decision_reason"] = "Contains TOGAF ADM or viewpoint terms"
        else:
            explanation["analysis"]["decision_reason"] = "No domain-specific terms found"

        return explanation

    def _auto_hydrate_vocabularies(self) -> None:
        """
        Auto-hydrate vocabularies from knowledge graph using SPARQL queries.

        This dynamically extracts domain terms from the loaded knowledge graph
        to expand the static vocabulary lists with real ontological data.
        """
        if not self.kg_loader:
            logger.warning("No KG loader available for auto-hydration")
            return

        try:
            logger.info("Auto-hydrating vocabularies from knowledge graph...")

            # Extract IEC/Energy and ENTSOE terms from KG
            iec_terms, entsoe_terms = self.kg_loader.hydrate_vocabularies()

            # Convert to lowercase sets and merge with existing terms
            kg_iec_terms = set(term.lower() for term in iec_terms)
            kg_entsoe_terms = set(term.lower() for term in entsoe_terms)

            # Merge with existing vocabulary terms
            self.iec_energy_terms.update(kg_iec_terms)
            self.iec_energy_terms.update(kg_entsoe_terms)  # Add ENTSOE to IEC/energy terms

            # Add common energy patterns from knowledge graph
            energy_patterns = {
                "active power", "reactive power", "apparent power",
                "voltage", "current", "frequency", "phase angle",
                "conductor", "breaker", "transformer", "substation",
                "distribution", "transmission", "generation",
                "load", "feeder", "busbar", "switch",
                "protection", "relay", "measurement",
                "scada", "ems", "dms", "gis"
            }
            self.iec_energy_terms.update(energy_patterns)

            self.auto_hydrated = True

            logger.info(
                f"Auto-hydration complete: {len(self.iec_energy_terms)} IEC/energy terms, "
                f"{len(self.togaf_archimate_terms)} TOGAF/ArchiMate terms"
            )

        except Exception as e:
            logger.error(f"Failed to auto-hydrate vocabularies: {e}")
            # Continue with existing vocabularies

    def hydrate_from_kg(self, kg_loader: "KnowledgeGraphLoader") -> None:
        """
        Manually hydrate vocabularies from a knowledge graph loader.

        Args:
            kg_loader: Knowledge graph loader instance
        """
        self.kg_loader = kg_loader
        self._auto_hydrate_vocabularies()

    def reload_vocabularies(self) -> None:
        """Reload vocabularies from file and re-hydrate from KG (useful for updates)."""
        logger.info("Reloading vocabularies...")
        self._load_vocabularies()

        # Re-run auto-hydration if KG loader is available
        if self.kg_loader:
            self._auto_hydrate_vocabularies()

        logger.info("Vocabularies reloaded successfully")