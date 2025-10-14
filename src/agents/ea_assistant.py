"""
Production EA Agent - Full 4R+G+C Pipeline Integration.

REDESIGNED VERSION:
- Simplified retrieval logic (no complex definition query special-casing)
- Enhanced template fallback (domain-aware, multi-source synthesis)
- Unified behavior across all query types
- Better traceability and maintainability
- ENHANCED: Citation authenticity validation to prevent fake citations
- NEW: Pre-loaded citation pools with metadata for fast validation

This is the main agent that integrates all safety components into a production-ready
pipeline for Enterprise Architecture assistance. It implements the complete
Reflect → Route → Retrieve → Refine → Ground → Critic → Validate flow.

The agent ensures:
1. Queries are routed to the right knowledge source (Router)
2. Responses are grounded in real citations (GroundingCheck)
3. Citations are validated for authenticity (CitationValidator)
4. Citation pools constrain LLM to real citations only
5. Confidence is assessed with human review when needed (Critic)
6. TOGAF alignment is validated (ArchiMateParser)
7. Full audit trail is maintained for accountability
"""

import asyncio
import json
import logging
import os
import time
import re

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
from rdflib import URIRef, Graph, Namespace
from src.knowledge.kg_loader import KnowledgeGraphLoader
from src.archimate.parser import ArchiMateParser, ArchiMateElement
from src.routing.query_router import QueryRouter
from src.documents.pdf_indexer import PDFIndexer, DocumentChunk
from src.safety.grounding import GroundingCheck
from src.safety.citation_validator import CitationValidator
from src.validation.critic import Critic, CriticAssessment
from src.llm.factory import create_llm_provider
from src.llm.prompts import EAPromptTemplate
from src.exceptions.exceptions import UngroundedReplyError, LowConfidenceError, FakeCitationError
from src.utils.trace import get_tracer
from src.agents.session_manager import SessionManager, ConversationTurn

# Configuration constants - replace all hardcoded values
from src.config.constants import (
    CONFIDENCE,
    SEMANTIC_CONFIG,
    RANKING_CONFIG,
    CONTEXT_CONFIG,
    LANG_CONFIG,
    PERFORMANCE_CONFIG,
    COMPARISON_TERM_STOP_WORDS,
    QUERY_TERM_STOP_WORDS,
    COMPARISON_PATTERNS,
)


logger = logging.getLogger(__name__)
tracer = get_tracer()

try:
    from src.agents.embedding_agent import EmbeddingAgent
    EMBEDDING_AVAILABLE = True
    from src.agents.llm_council import LLMCouncil
    
except (ImportError, RuntimeError) as e:
    logger.warning(f"EmbeddingAgent not available: {e}")
    logger.warning("Semantic search disabled. Install sentence-transformers to enable.")
    EmbeddingAgent = None
    EMBEDDING_AVAILABLE = False

def is_definition_query(query_text: str) -> bool:
    """
    Detect definition queries.
    
    Used for prioritizing Knowledge Graph results which contain definitions.
    """
    if not query_text:
        return False
    q_lower = query_text.strip().lower()
    
    patterns = [
        r'^(what\s+is|what\s+are)\s+',
        r'^(define|definition\s+of)\s+',
        r'^(how\s+do\s+we\s+define|how\s+would\s+we\s+define)\s+',
        r'\b(meaning\s+of|means)\b'
    ]

    return any(re.search(pattern, q_lower) for pattern in patterns)

@dataclass
class PhaseTrace:
    """Tracks execution of a single pipeline phase"""
    phase_name: str
    start_time: float
    end_time: float = 0
    duration_ms: float = 0
    status: str = "running"
    details: Dict[str, Any] = field(default_factory=dict)
    sub_steps: List[Dict[str, Any]] = field(default_factory=list)
    
    def complete(self, status: str = "completed"):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        
    def add_detail(self, key: str, value: Any):
        self.details[key] = value
        
    def add_substep(self, description: str, result: Any = None):
        self.sub_steps.append({
            "description": description,
            "result": result,
            "timestamp": time.time()
        })


@dataclass
class PipelineTrace:
    """Complete trace of pipeline execution"""
    session_id: str
    query: str
    phases: List[PhaseTrace] = field(default_factory=list)
    total_duration_ms: float = 0
    start_time: float = field(default_factory=time.time)
    
    def add_phase(self, phase_name: str) -> PhaseTrace:
        phase = PhaseTrace(phase_name=phase_name, start_time=time.time())
        self.phases.append(phase)
        return phase
    
    def finalize(self):
        self.total_duration_ms = (time.time() - self.start_time) * 1000
        
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "total_duration_ms": self.total_duration_ms,
            "phases": [
                {
                    "name": p.phase_name,
                    "duration_ms": p.duration_ms,
                    "status": p.status,
                    "details": p.details,
                    "sub_steps": p.sub_steps
                }
                for p in self.phases
            ]
        }

@dataclass
class PipelineResponse:
    """Structured response from the EA assistant pipeline."""
    query: str
    response: str
    route: str
    citations: List[str]
    confidence: float
    requires_human_review: bool
    togaf_phase: Optional[str]
    archimate_elements: List[Dict]
    processing_time_ms: float
    session_id: str
    timestamp: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ProductionEAAgent:
    """
    Production-ready Enterprise Architecture assistant with full safety pipeline.

    Integrates all components:
    - Knowledge Graph (39,100+ energy triples)
    - ArchiMate Parser (real model elements)
    - Query Router (structured-first routing)
    - Grounding Check (citation enforcement)
    - Citation Validator (authenticity validation)
    - Citation Pools (pre-loaded for fast validation) - NEW
    - Critic (confidence assessment)
    - TOGAF Validator (phase alignment)
    """

    def __init__(self, kg_path: str = "data/energy_knowledge_graph.ttl",
                 models_path: str = "data/models/",
                 docs_path: str = "data/docs/",
                 vocab_path: str = "config/vocabularies.json",
                 llm_provider: str = "groq"):
        """
        Initialize the production EA agent with all safety components including LLM.

        Args:
            kg_path: Path to knowledge graph TTL file
            models_path: Directory containing ArchiMate models
            docs_path: Directory containing PDF documents
            vocab_path: Path to routing vocabularies
            llm_provider: LLM provider to use (groq, openai, ollama)
        """
        logger.info("Initializing Production EA Agent with citation validation...")

        # Initialize LLM Council for dual validation (OpenAI + Groq)
        self.llm_council = LLMCouncil(
            primary_api_key=os.getenv("OPENAI_API_KEY"),
            validator_api_key=os.getenv("GROQ_API_KEY"),
            use_openai_primary=True,
            primary_model=os.getenv("OPENAI_MODEL", "gpt-5"),
            validator_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        )
        logger.info(f"LLM Council initialized: Primary=OpenAI GPT-5, Validator=Groq Llama-3.3")

        # Initialize knowledge graph with lazy loading
        self.kg_loader = KnowledgeGraphLoader(Path(kg_path))
        self.kg_loader.load()  # This starts background loading
        logger.info("Knowledge Graph initialized with lazy loading")
        
        # Initialize session manager for conversation memory
        self.session_manager = SessionManager(
            max_history=10,
            session_timeout_minutes=30
        )
        logger.info("✅ Session manager initialized for conversation memory")

        # Initialize ArchiMate parser
        self.archimate_parser = ArchiMateParser()
        self._load_archimate_models(models_path)

        # Initialize PDF document indexer
        self.pdf_indexer = PDFIndexer(docs_path)
        try:
            self.pdf_indexer.load_or_create_index()
            logger.info("PDF document indexer initialized successfully")
        except Exception as e:
            logger.warning(f"PDF indexer initialization failed: {e}")
            self.pdf_indexer = None
            
        # Semantic search enhancement (optional)
        self.embedding_agent = None
        if EMBEDDING_AVAILABLE and not os.environ.get('DISABLE_EMBEDDING_AGENT'):
            try:
                self.embedding_agent = EmbeddingAgent(
                    kg_loader=self.kg_loader,
                    archimate_parser=self.archimate_parser,
                    pdf_indexer=self.pdf_indexer,
                    embedding_model="all-MiniLM-L6-v2",  # Fast, good quality
                    cache_dir="data/embeddings"
                )
                logger.info("Embedding agent initialized for semantic search fallback")
            except Exception as e:
                logger.warning(f"Could not initialize embedding agent: {e}")
                logger.warning("Continuing without semantic search capability")
                self.embedding_agent = None
        else:
            logger.info("Embedding agent disabled or not available - semantic search not available")

        # Initialize query router with KG auto-hydration
        self.router = QueryRouter(vocab_path, kg_loader=self.kg_loader)
        
        # Initialize citation validator
        self.citation_validator = CitationValidator(
            kg_loader=self.kg_loader,
            archimate_parser=self.archimate_parser,
            pdf_indexer=self.pdf_indexer
        )
        logger.info("Citation validator initialized for authenticity checks")
        
        # Initialize grounding check WITH citation validator
        self.grounder = GroundingCheck(citation_validator=self.citation_validator)
        logger.info("Grounding check initialized with citation validator")
        
        # Initialize critic
        self.critic = Critic()

        # Initialize LLM provider (will be set async)
        self.llm_provider = None
        self.llm_provider_name = llm_provider
        self.prompt_template = EAPromptTemplate()

        # Context store for audit trail
        self.context_store = {}

        # ============= CITATION POOL LOADING (NEW) =============
        logger.info("Pre-loading citation pools for fast validation...")
        pool_start_time = time.time()

        # 1. Load all KG citations by namespace
        self.citation_pools = {}
        for namespace in ["skos", "iec", "entsoe", "eurlex", "lido"]:
            citations = self.kg_loader.get_all_valid_citations(namespace=namespace)
            self.citation_pools[namespace] = set(citations)
            logger.info(f"  Loaded {len(citations)} {namespace.upper()} citations")

        # 2. Load all ArchiMate citations
        archimate_citations = self.archimate_parser.get_valid_citations()
        self.citation_pools['archimate'] = set(archimate_citations)
        logger.info(f"  Loaded {len(archimate_citations)} ArchiMate citations")

        # 3. Create combined pool
        self.all_citations = set()
        for citations in self.citation_pools.values():
            self.all_citations.update(citations)

        # 4. Cache metadata for each citation
        self.citation_metadata_cache = {}
        logger.info("  Caching citation metadata...")

        for citation in self.all_citations:
            metadata = None
            
            if citation.startswith("archi:"):
                element = self.archimate_parser.get_element_by_citation(citation)
                if element:
                    metadata = {
                        "citation": citation,
                        "label": element.name,
                        "type": element.type,
                        "layer": element.layer,
                        "source": "archimate"
                    }
            else:
                # KG citation
                kg_metadata = self.kg_loader.get_citation_metadata(citation)
                if kg_metadata:
                    definition = kg_metadata.get('definition', '')
                    if definition and len(definition) > 100:
                        definition = definition[:100] + '...'
                    
                    metadata = {
                        "citation": citation,
                        "label": kg_metadata.get('label'),
                        "definition": definition,
                        "source": "knowledge_graph"
                    }
            
            if metadata:
                self.citation_metadata_cache[citation] = metadata

        pool_load_time = (time.time() - pool_start_time) * 1000
        logger.info(f"Citation pools ready in {pool_load_time:.0f}ms: "
                   f"{len(self.all_citations)} citations, "
                   f"{len(self.citation_metadata_cache)} with metadata")
        # ============= END CITATION POOL LOADING =============

        logger.info("Production EA Agent initialized with citation authenticity validation")

    async def _initialize_llm(self) -> None:
        """Initialize LLM provider if not already initialized."""
        if self.llm_provider is None and self.llm_provider_name:
            try:
                self.llm_provider = await create_llm_provider(self.llm_provider_name)
                if self.llm_provider:
                    logger.info(f"LLM provider {self.llm_provider_name} initialized successfully")
                else:
                    logger.info(f"LLM provider {self.llm_provider_name} unavailable, using template fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM provider {self.llm_provider_name}: {e}")
                logger.info("Using template fallback for safety")
                self.llm_provider = None

    def _load_archimate_models(self, models_path: str) -> None:
        """
        Load all ArchiMate models from directory.

        Args:
            models_path: Directory containing .archimate files
        """
        models_dir = Path(models_path)
        if not models_dir.exists():
            logger.warning(f"Models directory not found: {models_path}")
            return

        # Load both .archimate and .xml ArchiMate model files
        model_files = list(models_dir.glob("*.archimate")) + list(models_dir.glob("*.xml"))
        for model_file in model_files:
            # Skip .DS_Store and other non-model files
            if model_file.name.startswith('.'):
                continue

            success = self.archimate_parser.load_model(str(model_file))
            if success:
                logger.info(f"Loaded ArchiMate model: {model_file.name}")
            else:
                logger.warning(f"Failed to load model: {model_file.name}")
                
    async def _detect_language_with_llm(self, text: str) -> str:
        """
        Detect language of text using LLM intelligence.
        
        Uses the LLM Council's primary LLM for fast, accurate language detection.
        Falls back to simple heuristic if LLM unavailable.
        
        Returns 'en' for English, 'nl' for Dutch, 'unknown' for unclear.
        """
        if not text or len(text) < 5:
            return "unknown"
        
        # Try LLM-based detection first
        if self.llm_council:
            try:
                prompt = f"""Identify the primary language of this text. Respond with ONLY one word:
    - "english" if the text is primarily in English
    - "dutch" if the text is primarily in Dutch/Nederlands  
    - "unknown" if unclear or mixed

    Text: "{text[:200]}"

    Language:"""
                
                response = await self.llm_council.primary_llm.generate(
                    prompt=prompt,
                    system_prompt="You are a language detection assistant. Respond with only one word: english, dutch, or unknown.",
                    temperature=LANG_CONFIG.LLM_TEMPERATURE,
                    max_tokens=LANG_CONFIG.LLM_MAX_TOKENS
                )
                
                language = response.content.strip().lower()
                
                if "english" in language or language == "en":
                    return "en"
                elif "dutch" in language or "nederlands" in language or language == "nl":
                    return "nl"
                else:
                    return "unknown"
                    
            except Exception as e:
                logger.warning(f"LLM language detection failed: {e}, falling back to heuristic")
        
        # Fallback: Simple character-based heuristic (no word lists!)
        # Dutch has more diacritics and specific character patterns
        text_lower = text.lower()
        
        # Dutch-specific character patterns
        dutch_chars = text_lower.count('ij') + text_lower.count('oe') + text_lower.count('ui')
        dutch_chars += text_lower.count('ë') + text_lower.count('ï') + text_lower.count('ü')
        
        # Check for Dutch compound words (lots of long words)
        words = text_lower.split()
        long_words = sum(1 for w in words if len(w) > LANG_CONFIG.LONG_WORD_MIN_LENGTH)
        
        # Simple heuristic
        if dutch_chars > LANG_CONFIG.DUTCH_CHAR_THRESHOLD or (long_words > len(words) * LANG_CONFIG.LONG_WORD_PERCENTAGE and len(words) > 3):
            return "nl"
        elif any(word in text_lower for word in ['the ', ' and ', ' for ', ' power', ' grid']):
            return "en"
        else:
            return "unknown"

    def _build_citation_pool_from_retrieval(self, retrieval_context: Dict) -> List[Dict]:
        """
        Build query-specific citation pool from retrieval results with metadata.

        This extracts citations from retrieval results and enriches them with
        labels and definitions for the LLM to use.

        Args:
            retrieval_context: Dict with kg_results, archimate_elements, etc.

        Returns:
            List of citation dicts with metadata
        """
        citation_set = set()

        # Extract from KG results
        kg_results = retrieval_context.get('kg_results', [])
        for result in kg_results:
            if not isinstance(result, dict):
                continue
            citation = self._extract_citation(result)
            if citation != "unknown":
                citation_set.add(citation)
        
        # Extract from ArchiMate elements
        archimate_elements = retrieval_context.get('archimate_elements', [])
        for element in archimate_elements:
            if not isinstance(element, dict):
                continue
            
            citation = self._extract_citation(element)
            
            # If no citation found, try to construct from element ID
            if citation == "unknown":
                element_id = element.get('id')
                if element_id and isinstance(element_id, str):
                    citation = f"archi:id-{element_id}"
            
            # Add if valid
            if citation != "unknown":
                citation_set.add(citation)
        
        # Extract from document results
        doc_results = retrieval_context.get('document_results', [])
        for doc in doc_results:
            if not isinstance(doc, dict):
                continue
            citation = self._extract_citation(doc)
            if citation != "unknown":
                citation_set.add(citation)
        
        # Extract from candidates (fallback)
        candidates = retrieval_context.get('candidates', [])
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            citation = self._extract_citation(candidate)
            if citation != "unknown":
                citation_set.add(citation)
        
        # Build enriched pool with metadata
        enriched_pool = []
        for citation in citation_set:
            # CRITICAL: Double-check citation is valid string
            if not citation or not isinstance(citation, str):
                logger.warning(f"Skipping invalid citation in pool: {repr(citation)}")
                continue
                
            metadata = self.citation_metadata_cache.get(citation)
            if metadata:
                enriched_pool.append(metadata)
            else:
                # Fallback for citations not in cache
                enriched_pool.append({
                    "citation": citation,
                    "label": "Unknown",
                    "source": "retrieval"
                })
        
        logger.info(f"Built citation pool: {len(enriched_pool)} citations from retrieval")
        return enriched_pool
    
    def _extract_dynamic_domain_context(self, primary_result: Dict) -> Dict:
        """
        Extract domain context from the primary result by querying KG metadata.
        
        This uses the actual vocabulary structure from the KG rather than 
        hardcoded namespace mappings.
        
        Args:
            primary_result: The top-ranked candidate with citation info
            
        Returns:
            Dictionary with dynamic domain context
        """
        citation = primary_result.get("citation") or primary_result.get("citation_id")
        
        # Default context
        context = {
            "domain": "Energy Distribution Systems (Alliander DSO)",
            "standards": [],
            "methodology": "TOGAF ADM"
        }
        
        if not citation or not self.kg_loader:
            return context
        
        # Get full metadata from KG to determine vocabulary
        metadata = self.kg_loader.get_citation_metadata(citation)
        
        if not metadata:
            # Fallback to generic context
            context["standards"] = ["Energy Domain Standards"]
            return context
        
        # Extract URI to determine vocabulary membership
        uri = metadata.get("uri", "")
        
        # Map URI patterns to actual vocabularies from your KG structure
        # Based on the Alliander.esa Vocabularies structure
        
        # LEGAL & REGULATION vocabularies
        if any(pattern in uri.lower() for pattern in ['regulation', 'eurlex', 'acer', 'dutch']):
            if 'eurlex' in uri.lower():
                context["standards"] = ["EUR-Lex Regulation"]
            elif 'acer' in uri.lower():
                context["standards"] = ["EUR Energy Regulators (ACER)"]
            elif 'dutch' in uri.lower() or 'nl' in uri.lower():
                context["standards"] = ["Dutch Regulation Electricity"]
            else:
                context["standards"] = ["Legal & Regulatory Standards"]
        
        # ENERGY MANAGEMENT SYSTEM vocabularies
        elif any(pattern in uri.lower() for pattern in ['alliander', 'modulair', 'pas1879', 'bsi']):
            if 'modulair' in uri.lower() or 'msb' in uri.lower():
                context["standards"] = ["Alliander Modulair Station Building Blocks"]
            elif 'pas1879' in uri.lower() or 'bsi' in uri.lower():
                context["standards"] = ["PAS1879 - Energy smart appliances (BSI)"]
            else:
                context["standards"] = ["Alliander Energy System"]
        
        # Harmonized Electricity Market
        elif 'entso-e' in uri.lower() or 'entsoe' in uri.lower():
            context["standards"] = ["Harmonized Electricity Market Role Model (ENTSO-E)"]
        
        # INTERNATIONAL ELECTROTECHNICAL COMMISSION vocabularies
        elif 'iec' in uri.lower() or any(std in uri.lower() for std in ['61968', '61970', '62325', '62746']):
            # Determine specific IEC standard from URI
            if '61968' in uri.lower():
                context["standards"] = ["IEC 61968 - Meters, Assets and Work"]
            elif '61970' in uri.lower():
                # Check for specific 61970 sub-standards
                if 'cgmes' in uri.lower():
                    if 'core' in uri.lower():
                        context["standards"] = ["IEC 61970 - Core Equipment (CGMES)"]
                    elif 'diagram' in uri.lower():
                        context["standards"] = ["IEC 61970 - Diagram Layout (CGMES)"]
                    elif 'equipment' in uri.lower() and 'boundary' in uri.lower():
                        context["standards"] = ["IEC 61970 - Equipment Boundary (CGMES)"]
                    elif 'geographical' in uri.lower() or 'location' in uri.lower():
                        context["standards"] = ["IEC 61970 - Geographical Location (CGMES)"]
                    elif 'operation' in uri.lower():
                        context["standards"] = ["IEC 61970 - Operation (CGMES)"]
                    elif 'short' in uri.lower() and 'circuit' in uri.lower():
                        context["standards"] = ["IEC 61970 - Short Circuit (CGMES)"]
                    elif 'state' in uri.lower() and 'variables' in uri.lower():
                        context["standards"] = ["IEC 61970 - State Variables (CGMES)"]
                    elif 'steady' in uri.lower() and 'hypothesis' in uri.lower():
                        context["standards"] = ["IEC 61970 - Steady State Hypothesis (CGMES)"]
                    elif 'topology' in uri.lower():
                        context["standards"] = ["IEC 61970 - Topology (CGMES)"]
                    else:
                        context["standards"] = ["IEC 61970 - Common Information Model (CGMES)"]
                else:
                    context["standards"] = ["IEC 61970 - Common Information Model"]
            elif '62325' in uri.lower():
                context["standards"] = ["IEC 62325 - Market Model"]
            elif '62746' in uri.lower():
                context["standards"] = ["IEC 62746 - Demand Site Resource"]
            else:
                context["standards"] = ["IEC Standards"]
        
        # ARCHITECTURE vocabulary
        elif 'ai' in uri.lower() and 'ontology' in uri.lower():
            context["standards"] = ["Alliander Artificial Intelligence Ontology"]
        
        # Generic fallback - try to extract vocabulary name from URI
        else:
            # Try to extract meaningful vocabulary name from URI segments
            uri_segments = uri.replace('http://', '').replace('https://', '').split('/')
            
            # Look for meaningful segments
            for segment in uri_segments:
                if len(segment) > 5 and segment not in ['www', 'com', 'org', 'net']:
                    # Capitalize and use as standard name
                    vocab_name = segment.replace('-', ' ').replace('_', ' ').title()
                    context["standards"] = [vocab_name]
                    break
            
            # Ultimate fallback
            if not context["standards"]:
                context["standards"] = ["Energy Domain Standards"]
        
        return context

    async def process_query(
        self, 
        query: str, 
        session_id: str = None,
        use_conversation_context: bool = True
    ) -> Tuple[PipelineResponse, PipelineTrace]:
        """
        Process a query through the full 4R+G+C pipeline with complete tracing.

        ENHANCED: Now supports follow-up queries with conversation memory.

        Pipeline steps:
        1. INITIALIZATION: Create session and audit trail
        2. CONVERSATION_CONTEXT: Build context from history (NEW)
        3. REFLECT: Analyze query intent
        4. ROUTE: Determine knowledge source
        5. RETRIEVE: Get relevant knowledge
        6. BUILD_CITATION_POOL: Extract valid citations
        7. REFINE: Generate response candidates
        8. GROUND: Enforce citations with authenticity validation
        9. CRITIC: Assess confidence
        10. VALIDATE: Check TOGAF alignment
        11. RESPONSE_ASSEMBLY: Build final response

        Args:
            query: User query string
            session_id: Optional session ID for tracking (enables conversation memory)
            use_conversation_context: Whether to use conversation history for follow-ups

        Returns:
            Tuple of (PipelineResponse, PipelineTrace)
        """
        start_time = time.perf_counter()

        if not session_id:
            session_id = str(uuid4())[:8]

        # Initialize trace
        trace = PipelineTrace(session_id=session_id, query=query)

        # Start comprehensive tracing (keep for backwards compatibility)
        trace_id = tracer.start_trace(session_id)

        # Initialize audit trail for this query
        audit_trail = {
            "session_id": session_id,
            "trace_id": trace_id,
            "query": query,
            "original_query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "steps": []
        }

        tracer.trace_info(
            trace_id, "ea_assistant", "process_query",
            query=query, session_id=session_id
        )

        try:
            # ========== PHASE 1: INITIALIZATION 🚀 ==========
            init_phase = trace.add_phase("INITIALIZATION")
            init_phase.add_detail("session_id", session_id)
            init_phase.add_detail("trace_id", trace_id)
            init_phase.add_detail("timestamp", datetime.utcnow().isoformat())
            init_phase.add_substep("Created session ID", session_id)
            init_phase.add_substep("Started trace ID", trace_id)
            init_phase.add_substep("Initialized audit trail", "Active")
            
            # Check session history
            session_history = self.session_manager.get_history(session_id)
            init_phase.add_detail("session_turns", len(session_history))
            if session_history:
                init_phase.add_substep("Session history found", f"{len(session_history)} previous turns")
            
            init_phase.complete()

            audit_trail["steps"].append({
                "step": "INITIALIZATION",
                "action": "System initialized",
                "session_turns": len(session_history),
                "timestamp": datetime.utcnow().isoformat()
            })

            # ========== PHASE 2: CONVERSATION_CONTEXT 💬 ==========
            enhanced_query = query
            is_followup = False
            conversation_context = ""
            
            if use_conversation_context and session_history:
                context_phase = trace.add_phase("CONVERSATION_CONTEXT")
                
                # Get conversation context
                conversation_context, is_followup = self.session_manager.get_context_for_query(
                    session_id, query
                )
                
                context_phase.add_detail("is_followup", is_followup)
                context_phase.add_detail("previous_turns", len(session_history))
                
                if is_followup:
                    enhanced_query = conversation_context
                    context_phase.add_detail("context_length", len(conversation_context))
                    context_phase.add_detail("enhanced_query_length", len(enhanced_query))
                    context_phase.add_substep("Follow-up detected", "Context enhanced")
                    context_phase.add_substep("Previous concepts", 
                        ", ".join(session_history[-1].key_concepts[:3]) if session_history[-1].key_concepts else "None")
                    
                    logger.info(f"Follow-up query detected - enhanced with {len(session_history)} previous turns")
                    
                    audit_trail["conversation_context"] = {
                        "is_followup": True,
                        "context_length": len(conversation_context),
                        "previous_turns": len(session_history)
                    }
                else:
                    context_phase.add_substep("Standalone query", "No context enhancement needed")
                    audit_trail["conversation_context"] = {
                        "is_followup": False
                    }
                
                context_phase.complete()
            
            # Update audit trail
            audit_trail["enhanced_query"] = enhanced_query if is_followup else query
            audit_trail["is_followup"] = is_followup

            # ========== PHASE 3: REFLECT 🤔 ==========
            reflect_phase = trace.add_phase("REFLECT")
            
            with tracer.trace_function(trace_id, "ea_assistant", "reflect", query=query):
                # Analyze query intent
                query_intent = "definition" if is_definition_query(query) else "general"
                
                # Enhanced intent for follow-ups
                if is_followup:
                    if self.session_manager._is_comparison_query(query):
                        query_intent = "comparison"
                        reflect_phase.add_substep("Comparison query detected", "Will compare concepts")
                
                reflect_phase.add_detail("query_intent", query_intent)
                reflect_phase.add_detail("query_length", len(query))
                reflect_phase.add_detail("is_followup", is_followup)
                reflect_phase.add_substep("Analyzed query intent", query_intent)
                
                # Extract key terms
                key_terms = self._extract_query_terms(enhanced_query if is_followup else query)
                reflect_phase.add_detail("key_terms", key_terms[:5])
                reflect_phase.add_substep("Extracted key terms", f"{len(key_terms)} terms")
                
                audit_trail["steps"].append({
                    "step": "REFLECT",
                    "action": "Analyzing query intent",
                    "query_intent": query_intent,
                    "is_followup": is_followup,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            reflect_phase.complete()

            # ========== PHASE 4: ROUTE 🧭 ==========
            route_phase = trace.add_phase("ROUTE")
            
            with tracer.trace_function(trace_id, "query_router", "route", query=query):
                route = self.router.route(enhanced_query if is_followup else query, trace_id=trace_id)
                route_phase.add_detail("route", route)
                
                if is_followup:
                    route_phase.add_detail("routing_mode", "context_enhanced")
                
                # Get routing reasoning
                routing_reason = self.router.get_last_routing_reason() if hasattr(self.router, 'get_last_routing_reason') else "Route determined"
                route_phase.add_detail("reasoning", routing_reason)
                route_phase.add_substep(f"Routed to: {route}", routing_reason)
                
                audit_trail["steps"].append({
                    "step": "ROUTE",
                    "result": route,
                    "routing_mode": "context_enhanced" if is_followup else "standard",
                    "timestamp": datetime.utcnow().isoformat()
                })
                logger.info(f"Query routed to: {route}")
            
            route_phase.complete()

            # ========== PHASE 5: RETRIEVE 📚 ==========
            retrieve_phase = trace.add_phase("RETRIEVE")

            with tracer.trace_function(trace_id, "ea_assistant", "retrieve_knowledge", route=route):
                
                # Start with original query (enhanced_query is for LLM context, not retrieval)
                retrieval_query = query
                
                # 🎯 ENHANCED: For follow-up queries, add terms from session history
                if is_followup and session_history and len(session_history) >= 1:
                    print(f"🎯 RETRIEVAL ENHANCEMENT FOR FOLLOW-UP:")
                    print(f"🎯   Original query: {query}")
                    
                    # Extract terms from previous queries
                    prev_terms = []
                    for turn in session_history[-2:]:  # Last 2 turns
                        turn_terms = self._extract_query_terms(turn.query)
                        prev_terms.extend(turn_terms)
                        print(f"🎯   Previous query: {turn.query}")
                        print(f"🎯   Extracted terms: {turn_terms[:5]}")
                    
                    # Remove duplicates and common words
                    prev_terms = list(set(prev_terms))
                    # Filter out generic terms
                    prev_terms = [t for t in prev_terms if len(t) > 4 and 'what' not in t and 'is' not in t]
                    
                    print(f"🎯   Filtered previous terms: {prev_terms[:10]}")
                    
                    # Combine with current query - add key terms
                    key_prev_terms = [t for t in prev_terms if 'power' in t or 'active' in t or 'reactive' in t][:3]
                    retrieval_query = query + " " + " ".join(key_prev_terms)
                    print(f"🎯   Final retrieval query: {retrieval_query}")
                
                retrieval_context = await self._retrieve_knowledge(
                    retrieval_query,
                    route, 
                    trace_id
                )
                
                # Add conversation context to retrieval context
                if is_followup:
                    retrieval_context["conversation_context"] = conversation_context
                    retrieval_context["is_followup"] = True
                    retrieval_context["session_history"] = [
                        {"query": t.query, "response": t.response[:200]}
                        for t in session_history[-2:]
                    ]
                
                total_candidates = len(retrieval_context.get("candidates", []))
                retrieve_phase.add_detail("total_candidates", total_candidates)
                retrieve_phase.add_detail("route", route)
                
                if is_followup:
                    retrieve_phase.add_detail("retrieval_mode", "context_enhanced")
                    retrieve_phase.add_substep("Using conversation context", "Enhanced retrieval")
                
                # 🎯 DEBUG: Show what was retrieved
                if is_followup:
                    print(f"🎯 RETRIEVAL RESULTS:")
                    print(f"🎯   Total candidates: {total_candidates}")
                    for i, c in enumerate(retrieval_context.get("candidates", [])[:5]):
                        print(f"🎯   [{i}] {c.get('element', 'N/A')} - {c.get('citation', 'N/A')}")
                
                if route == "structured_model":
                    kg_count = len(retrieval_context.get("kg_results", []))
                    archimate_count = len(retrieval_context.get("archimate_elements", []))
                    
                    retrieve_phase.add_detail("kg_results_count", kg_count)
                    retrieve_phase.add_detail("archimate_results_count", archimate_count)
                    retrieve_phase.add_substep("Querying Knowledge Graph", "Started")
                    retrieve_phase.add_substep("KG Results Found", kg_count)
                    retrieve_phase.add_substep("Querying ArchiMate Models", "Started")
                    retrieve_phase.add_substep("ArchiMate Elements Found", archimate_count)
                elif route == "togaf_method":
                    togaf_count = len(retrieval_context.get("togaf_docs", []))
                    retrieve_phase.add_detail("togaf_docs_count", togaf_count)
                    retrieve_phase.add_substep("TOGAF Documents Found", togaf_count)
                elif route == "unstructured_docs":
                    doc_count = len(retrieval_context.get("document_chunks", []))
                    retrieve_phase.add_detail("document_chunks_count", doc_count)
                    retrieve_phase.add_substep("Document Chunks Found", doc_count)
                
                # Semantic enhancement check
                if retrieval_context.get("semantic_enhanced"):
                    retrieve_phase.add_detail("semantic_enhanced", True)
                    retrieve_phase.add_substep("Semantic search enhancement", "Applied")
                
                audit_trail["steps"].append({
                    "step": "RETRIEVE",
                    "items_retrieved": total_candidates,
                    "retrieval_mode": "context_enhanced" if is_followup else "standard",
                    "timestamp": datetime.utcnow().isoformat()
                })

            retrieve_phase.complete()

            # Out-of-scope guard (early return with trace finalization)
            if route == "unstructured_docs" and not retrieval_context.get("candidates"):
                polite = (
                    "This question appears outside the Alliander energy/enterprise architecture scope. "
                    "I answer domain questions grounded in our SKOS dictionary, IEC standards, and ArchiMate models. "
                    "Try asking, for example: 'What is reactive power?' or 'Show capabilities related to congestion management.'"
                )
                
                # Add to audit trail
                audit_trail["steps"].append({
                    "step": "FILTER",
                    "action": "out_of_scope_decline",
                    "timestamp": datetime.utcnow().isoformat()
                })
                self.context_store[session_id] = audit_trail
                
                # Finalize trace before returning
                trace.finalize()
                
                return PipelineResponse(
                    query=query,
                    response=polite,
                    route=route,
                    citations=[],
                    confidence=0.0,
                    requires_human_review=False,
                    togaf_phase=None,
                    archimate_elements=[],
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                    session_id=session_id,
                    timestamp=datetime.utcnow().isoformat()
                ), trace

            # ========== PHASE 6: BUILD CITATION POOL 🎯 ==========
            citation_phase = trace.add_phase("BUILD_CITATION_POOL")
            
            with tracer.trace_function(trace_id, "citation_validator", "build_pool",
                                    candidates_count=len(retrieval_context.get("candidates", []))):
                citation_pool = self._build_citation_pool_from_retrieval(retrieval_context)
                
                citation_phase.add_detail("citation_pool_size", len(citation_pool))
                
                # Get citation sources
                sources = list(set(c.get("source", "unknown") for c in citation_pool))
                citation_phase.add_detail("citation_sources", sources)
                citation_phase.add_substep("Extracted citations from retrieval", len(citation_pool))
                
                # Show sample citations
                sample_citations = [c["citation"] for c in citation_pool[:5]]
                citation_phase.add_detail("sample_citations", sample_citations)
                citation_phase.add_substep("Sample citations", sample_citations)
                
                audit_trail["steps"].append({
                    "step": "BUILD_CITATION_POOL",
                    "pool_size": len(citation_pool),
                    "sample_citations": sample_citations,
                    "timestamp": datetime.utcnow().isoformat()
                })
                logger.info(f"Citation pool created: {len(citation_pool)} valid citations")
            
            citation_phase.complete()

            # ========== PHASE 7: REFINE ✨ ==========
            refine_phase = trace.add_phase("REFINE")

            with tracer.trace_function(trace_id, "ea_assistant", "refine_response",
                                    context_items=len(retrieval_context.get("candidates", [])),
                                    citation_pool_size=len(citation_pool)):
                
                # Determine refinement method
                await self._initialize_llm()
                
                if self.llm_provider or self.llm_council:
                    refine_phase.add_substep("Using LLM Council", "Dual validation enabled")
                    refine_phase.add_detail("llm_method", "council")
                else:
                    refine_phase.add_substep("Using Template Fallback", "No LLM available")
                    refine_phase.add_detail("llm_method", "template")
                
                # 🔍 DEBUG: Log follow-up detection
                print(f"🔍 REFINE PHASE DEBUG:")
                print(f"🔍   is_followup: {is_followup}")
                print(f"🔍   query: {query}")
                print(f"🔍   session_history length: {len(session_history)}")
                
                # Check if this is a comparison query requiring special handling
                if is_followup:
                    is_comparison = self.session_manager._is_comparison_query(query)
                    print(f"🔍   _is_comparison_query result: {is_comparison}")
                
                if is_followup and self.session_manager._is_comparison_query(query):
                    print("🔍 ✅ COMPARISON DETECTED - calling _refine_response_for_comparison")
                    refine_phase.add_substep("Using comparison refinement", "Enhanced for follow-up")
                    refine_phase.add_detail("refinement_mode", "comparison")
                    
                    # Use comparison-focused refinement
                    response, candidates = await self._refine_response_for_comparison(
                        query,
                        conversation_context,
                        retrieval_context,
                        citation_pool,
                        trace_id
                    )
                    print(f"🔍 Comparison method returned response with {len(response)} chars")
                    
                    # 🎯 ADD THIS DEBUG - Show first 500 chars of response
                    print(f"🎯 RESPONSE PREVIEW:")
                    print(f"🎯 {response[:500]}")
                    print(f"🎯 Citation count in response: {response.count('[')}")
                else:
                    print(f"🔍 ❌ NOT A COMPARISON - calling standard _refine_response")  # ← CHANGED
                    print(f"🔍   Reason: is_followup={is_followup}, is_comparison={'unknown' if not is_followup else self.session_manager._is_comparison_query(query)}")  # ← CHANGED
                    refine_phase.add_detail("refinement_mode", "standard")
                    
                    # Standard refinement
                    response, candidates = await self._refine_response(
                        enhanced_query if is_followup else query,
                        retrieval_context,
                        citation_pool,
                        trace_id
                    )
                    print(f"🔍 Standard method returned response with {len(response)} chars")  # ← CHANGED
                
                refine_phase.add_detail("response_length", len(response))
                refine_phase.add_detail("candidates_generated", len(candidates))
                refine_phase.add_substep("Generated response", f"{len(response)} chars")
                
                audit_trail["steps"].append({
                    "step": "REFINE",
                    "candidates_generated": len(candidates),
                    "response_length": len(response),
                    "refinement_mode": "comparison" if (is_followup and self.session_manager._is_comparison_query(query)) else "standard",
                    "timestamp": datetime.utcnow().isoformat()
                })

            refine_phase.complete()
            
            # ========== PHASE 8: GROUND 🔒 ==========
            ground_phase = trace.add_phase("GROUND")
            citation_pool_ids = [c['citation'] for c in citation_pool]

            # CRITICAL: Determine response type based on data availability
            has_kg_data = len(retrieval_context.get("candidates", [])) > 0

            # 🎯 NEW LOGIC: Distinguish between retrieval and synthesis
            is_synthesis_query = (
                is_followup and 
                self.session_manager._is_comparison_query(query)
            )

            if is_synthesis_query:
                # TYPE A: LLM SYNTHESIS - Citations optional/not applicable
                ground_phase.add_substep("LLM synthesis response", "Citations not required")
                
                # Extract citations if present (for reference), but don't enforce
                citations = self.grounder._extract_existing_citations(response)
                
                # If no citations, that's fine - it's synthesized reasoning
                if not citations:
                    citations = []  # Empty is OK for synthesis
                    ground_phase.add_detail("synthesis_mode", "no_citations_required")
                
                ground_phase.add_detail("response_type", "llm_synthesis")
                ground_phase.add_detail("validation", "content_based")
                ground_phase.complete()
                
                audit_trail["steps"].append({
                    "step": "GROUND",
                    "response_type": "llm_synthesis",
                    "citations": citations,
                    "validation_mode": "relaxed",
                    "timestamp": datetime.utcnow().isoformat()
                })

            elif has_kg_data:
                # TYPE B: DIRECT KG RETRIEVAL - Citations REQUIRED
                ground_phase.add_substep("Grounding KG-based response", "Citations required")
                
                # Extract citations from response
                found_citations = self.grounder._extract_existing_citations(response)
                ground_phase.add_detail("citations_found", len(found_citations))
                
                # Validate citations
                grounding_result = self.grounder.assert_citations(
                    response,
                    retrieval_context,
                    citation_pool_ids,
                    trace_id
                )
                
                citations = grounding_result.get("citations", [])
                if not citations:
                    citations = found_citations
                
                # ENFORCE: KG responses MUST have citations
                if len(citations) < 1:
                    ground_phase.add_detail("error", "Insufficient citations")
                    ground_phase.complete("failed")
                    raise UngroundedReplyError(
                        message="KG-based response lacks required citations",
                        required_prefixes=["skos:", "eurlex:", "iec:", "entsoe:", "archi:"],
                        suggestions=[c for c in citation_pool_ids[:5]] if citation_pool_ids else []
                    )
                
                ground_phase.add_detail("response_type", "kg_retrieval")
                ground_phase.add_detail("grounding_status", "PASSED")
                ground_phase.complete()

            else:
                # TYPE C: NO DATA - Pure LLM fallback
                ground_phase.add_substep("No KG data - LLM fallback", "No citations available")
                
                citations = []  # No KG data = no citations
                
                # Add disclaimer to response
                response = f"""{response}

            ---

            **⚠️ Note:** This response was generated by AI reasoning as no information was found in our knowledge bases.
            """
                
                ground_phase.add_detail("response_type", "llm_fallback")
                ground_phase.complete()

            # ========== PHASE 9: CRITIC 🎓 ==========
            critic_phase = trace.add_phase("CRITIC")

            with tracer.trace_function(trace_id, "critic", "assess",
                                    candidate_count=len(candidates)):
                assessment = self.critic.assess(candidates, retrieval_context)

                critic_phase.add_detail("confidence", assessment.confidence)
                critic_phase.add_detail("requires_review", assessment.requires_human_review)
                critic_phase.add_detail("top_suggestions_count", len(assessment.top_suggestions))

                # Show confidence calculation breakdown
                confidence_breakdown = {
                    "candidate_count": len(candidates),
                    "top_suggestions": [
                        {
                            "element": s["element"],
                            "confidence": s["confidence"]
                        }
                        for s in assessment.top_suggestions
                    ],
                    "average_confidence": assessment.confidence,
                    "calculation": f"Average of top {len(assessment.top_suggestions)} suggestions"
                }
                critic_phase.add_detail("confidence_breakdown", confidence_breakdown)
                critic_phase.add_substep(
                    "Confidence calculation",
                    f"Average of {len(assessment.top_suggestions)} top suggestions: {assessment.confidence:.2f}"
                )

                if assessment.confidence >= CONFIDENCE.HIGH_CONFIDENCE_THRESHOLD:
                    critic_phase.add_substep("✓ High confidence", f"{assessment.confidence:.2f}")
                elif assessment.confidence >= CONFIDENCE.MEDIUM_CONFIDENCE_THRESHOLD:
                    critic_phase.add_substep("⚠ Medium confidence", f"{assessment.confidence:.2f}")
                else:
                    critic_phase.add_substep("⚠ Low confidence - flagging for review", f"{assessment.confidence:.2f}")

                audit_trail["steps"].append({
                    "step": "CRITIC",
                    "confidence": assessment.confidence,
                    "requires_review": assessment.requires_human_review,
                    "timestamp": datetime.utcnow().isoformat()
                })

            critic_phase.complete()

            # ========== PHASE 10: VALIDATE_TOGAF ✅ ==========
            togaf_phase = None
            archimate_elements = []

            if route == "structured_model":
                validate_phase = trace.add_phase("VALIDATE_TOGAF")

                with tracer.trace_function(trace_id, "archimate_parser", "validate_togaf",
                                        route=route):
                    togaf_phase, archimate_elements = await self._validate_togaf_alignment(
                        candidates, retrieval_context
                    )

                    validate_phase.add_detail("togaf_phase", togaf_phase or "N/A")
                    validate_phase.add_detail("validated_elements_count", len(archimate_elements))

                    if togaf_phase:
                        validate_phase.add_substep(f"TOGAF Phase: {togaf_phase}",
                                                f"{len(archimate_elements)} elements")
                    else:
                        validate_phase.add_substep("No TOGAF phase determined", "N/A")

                    audit_trail["steps"].append({
                        "step": "VALIDATE",
                        "togaf_phase": togaf_phase,
                        "elements_validated": len(archimate_elements),
                        "timestamp": datetime.utcnow().isoformat()
                    })

                validate_phase.complete()

            # ========== PHASE 11: RESPONSE_ASSEMBLY 📦 ==========
            assembly_phase = trace.add_phase("RESPONSE_ASSEMBLY")

            # Calculate processing time
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            assembly_phase.add_detail("processing_time_ms", processing_time_ms)
            assembly_phase.add_detail("response_components", {
                "route": route,
                "citations_count": len(citations),
                "confidence": assessment.confidence,
                "togaf_phase": togaf_phase,
                "requires_review": assessment.requires_human_review,
                "is_followup": is_followup
            })
            assembly_phase.add_substep("Assembled final response", f"{processing_time_ms:.0f}ms total")
            
            # Build final response
            pipeline_response = PipelineResponse(
                query=query,
                response=response,
                route=route,
                citations=citations,
                confidence=assessment.confidence,
                requires_human_review=assessment.requires_human_review,
                togaf_phase=togaf_phase,
                archimate_elements=[{"id": e.get("id"), "name": e.get("element")} 
                                for e in archimate_elements] if archimate_elements else [],
                processing_time_ms=processing_time_ms,
                session_id=session_id,
                timestamp=datetime.utcnow().isoformat()
            )
            
            assembly_phase.complete()
            
            # ========== ADD TO SESSION HISTORY 💾 ==========
            if use_conversation_context:
                self.session_manager.add_turn(
                    session_id=session_id,
                    query=query,
                    response=response,
                    citations=citations,
                    confidence=assessment.confidence,
                    route=route,
                    processing_time_ms=processing_time_ms
                )
                logger.info(f"✅ Added turn to session {session_id} (total: {len(session_history) + 1} turns)")
            
            # Store audit trail
            self.context_store[session_id] = audit_trail
            
            # Print comprehensive trace report (existing tracer)
            tracer.print_trace_report(trace_id)
            
            # Finalize trace
            trace.finalize()
            
            logger.info(f"Pipeline completed in {processing_time_ms:.1f}ms "
                    f"(confidence: {assessment.confidence:.2f}, "
                    f"review: {assessment.requires_human_review}, "
                    f"citations: {len(citations)}, "
                    f"followup: {is_followup})")

            return pipeline_response, trace

        except UngroundedReplyError as e:
            logger.error(f"Grounding violation: {e}")
            trace.finalize()
            raise
        except FakeCitationError as e:
            logger.error(f"Fake citation error: {e}")
            trace.finalize()
            raise
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            audit_trail["steps"].append({
                "step": "ERROR",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            self.context_store[session_id] = audit_trail
            trace.finalize()
            raise

    def _build_citation_pool(self, retrieval_context: Dict, trace_id: str = None) -> List[str]:
        """
        Build pool of valid citations from retrieval context.
        
        This is CRITICAL - only citations in this pool can be used by LLM.
        Prevents fake citation generation by limiting LLM to real citations.
        
        Args:
            retrieval_context: Dictionary containing retrieval results
            trace_id: Optional trace ID for logging
            
        Returns:
            List of valid citation IDs
        """
        if trace_id:
            tracer.trace_info(trace_id, "agent", "citation_pool_build_start",
                            context_keys=list(retrieval_context.keys()))
        
        citation_pool = self.citation_validator.get_citation_pool(retrieval_context, trace_id)
        
        logger.info(f"Citation pool created: {len(citation_pool)} valid citations")
        
        if len(citation_pool) == 0:
            logger.warning("Citation pool is EMPTY - no valid citations available for LLM")
        elif len(citation_pool) < 3:
            logger.warning(f"Citation pool is small ({len(citation_pool)}) - limited citations available")
        
        if trace_id:
            tracer.trace_info(trace_id, "agent", "citation_pool_build_complete",
                            pool_size=len(citation_pool),
                            sample=citation_pool[:10])
        
        return citation_pool

    async def _retrieve_knowledge(self, query: str, route: str, trace_id: Optional[str] = None) -> Dict:
        """
        SIMPLIFIED UNIFIED RETRIEVAL - No complex conditional logic.
        
        Priority order (always the same):
        1. Knowledge Graph (SPARQL) - for definitions and concepts
        2. ArchiMate Models - for architectural elements
        3. PDF Documents - for unstructured content
        
        Args:
            query: User query
            route: Route destination from router
            trace_id: Optional trace ID for logging
            
        Returns:
            Dictionary with unified retrieval context
        """
        retrieval_context = {
            "route": route,
            "candidates": [],
            "kg_results": [],
            "archimate_elements": [],
            "togaf_docs": [],
            "togaf_context": {}
            # domain_context will be added dynamically in response building
        }
        
        # Extract query terms (simplified)
        query_terms = self._extract_query_terms(query)
        logger.info(f"Extracted query terms: {query_terms}")
        
            # 🎯 NEW: For comparison queries, extract terms from previous turns
        if hasattr(self, 'session_manager'):
            session_history = self.session_manager.get_history(trace_id or 'default')
            if session_history and len(session_history) >= 2:
                # Get terms from last 2 queries
                prev_query_1 = session_history[-1].query if len(session_history) >= 1 else ""
                prev_query_2 = session_history[-2].query if len(session_history) >= 2 else ""
                
                prev_terms = self._extract_query_terms(prev_query_1 + " " + prev_query_2)
                
                # Combine with current terms
                combined_terms = list(set(query_terms + prev_terms))
                logger.info(f"🎯 Added {len(prev_terms)} terms from session history")
                logger.info(f"🎯 Combined terms: {combined_terms[:10]}")
                
                query_terms = combined_terms
        
        if route == "structured_model":
            # STEP 1: Query Knowledge Graph (ALWAYS FIRST)
            kg_candidates = await self._query_knowledge_graph(query_terms, trace_id)
            logger.info(f"KG returned {len(kg_candidates)} candidates")
            
            # Store for citation pool extraction
            retrieval_context["kg_results"] = kg_candidates
            
            # STEP 2: Query ArchiMate Models (ALWAYS SECOND)
            archimate_candidates = await self._query_archimate_models(query_terms, trace_id)
            logger.info(f"ArchiMate returned {len(archimate_candidates)} candidates")
            
            # Store for citation pool extraction
            retrieval_context["archimate_elements"] = archimate_candidates
            
            # Combine with KG first (higher priority)
            retrieval_context["candidates"] = kg_candidates + archimate_candidates
            
            # Add TOGAF context
            retrieval_context["togaf_context"] = self._get_togaf_context(query, archimate_candidates)
            
        elif route == "togaf_method":
            # TOGAF methodology guidance
            phase_context = self._get_togaf_phase_guidance(query)
            togaf_candidate = {
                "element": f"TOGAF ADM {phase_context['phase']}",
                "type": "Methodology",
                "citation_id": f"togaf:adm:{phase_context['phase_letter']}",
                "confidence": CONFIDENCE.TOGAF_DOCUMENTATION,
                "definition": phase_context["description"],
                "source": "TOGAF 9.2 Standard",
                "priority": "togaf"
            }
            # THESE LINES MUST BE HERE - RIGHT AFTER CREATING togaf_candidate
            retrieval_context["candidates"] = [togaf_candidate]
            retrieval_context["togaf_docs"] = [togaf_candidate]
            retrieval_context["togaf_context"] = phase_context
            
        elif route == "unstructured_docs":
            # PDF document search
            if self.pdf_indexer:
                doc_candidates = await self._query_documents(query_terms, trace_id)
                retrieval_context["candidates"] = doc_candidates
                retrieval_context["document_chunks"] = doc_candidates
                logger.info(f"Documents returned {len(doc_candidates)} candidates")
        
        # PHASE 2: SEMANTIC ENHANCEMENT (NEW - always run when available)
        enable_semantic = os.getenv("ENABLE_SEMANTIC_ENHANCEMENT", "true").lower() == "true"
        if self.embedding_agent and enable_semantic:
            semantic_start = time.time()
            try:
                semantic_candidates = await self._semantic_enhancement(
                    query,
                    retrieval_context["candidates"]
                )
                retrieval_context["candidates"].extend(semantic_candidates)
                retrieval_context["semantic_enhanced"] = True

                semantic_duration = (time.time() - semantic_start) * 1000
                logger.info(f"Added {len(semantic_candidates)} semantic candidates")
                logger.info(f"Semantic enhancement took {semantic_duration:.2f}ms")
            except Exception as e:
                logger.warning(f"Semantic enhancement failed: {e}")
                retrieval_context["semantic_enhanced"] = False

        # PHASE 3: Context expansion for follow-ups
        session_id = trace_id or 'default'
        if hasattr(self, 'session_manager') and len(self.session_manager.get_history(session_id)) > 0:
            context_start = time.time()
            try:
                context_candidates = await self._context_expansion(query, session_id)
                retrieval_context["candidates"].extend(context_candidates)

                context_duration = (time.time() - context_start) * 1000
                logger.info(f"Added {len(context_candidates)} context candidates")
                logger.info(f"Context expansion took {context_duration:.2f}ms")
            except Exception as e:
                logger.warning(f"Context expansion failed: {e}")

        # PHASE 4: Ranking and deduplication (NEW - needed!)
        rank_start = time.time()
        original_count = len(retrieval_context["candidates"])
        retrieval_context["candidates"] = self._rank_and_deduplicate(
            retrieval_context["candidates"],
            query
        )
        final_count = len(retrieval_context["candidates"])
        rank_duration = (time.time() - rank_start) * 1000
        logger.info(f"Ranking and deduplication: {original_count} → {final_count} candidates in {rank_duration:.2f}ms")
        
        # Log final candidate count - MOVED OUTSIDE of semantic block
        total_candidates = len(retrieval_context.get("candidates", []))
        logger.info(f"Total candidates for refinement: {total_candidates}")
        
        if trace_id:
            tracer.trace_info(trace_id, "ea_assistant", "retrieve_complete",
                            total_candidates=total_candidates,
                            kg_count=len(retrieval_context.get("kg_results", [])),
                            archimate_count=len(retrieval_context.get("archimate_elements", [])))
        
        # ALWAYS return retrieval_context
        return retrieval_context

    def _extract_query_terms(self, query: str) -> List[str]:
        """
        SIMPLIFIED term extraction.
        
        Extract meaningful terms while preserving multi-word concepts.
        No complex conditional logic - same approach for all queries.
        """
        query_lower = query.lower()
        
        # Stop words to filter out
        stop_words = QUERY_TERM_STOP_WORDS
        
        words = query_lower.split()
        terms = []
        
        # Extract 2-word phrases
        for i in range(len(words) - 1):
            if words[i] not in stop_words or words[i+1] not in stop_words:
                two_word = f"{words[i]} {words[i+1]}"
                if len(two_word) > 5:
                    terms.append(two_word)
        
        # Extract 3-word phrases
        for i in range(len(words) - 2):
            three_word = f"{words[i]} {words[i+1]} {words[i+2]}"
            if len(three_word) > 10:
                terms.append(three_word)
        
        # Add individual significant words
        for word in words:
            if word not in stop_words and len(word) > 3:
                terms.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms
    
    # Place them near the _extract_query_terms() method

    def _extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key terms from query for reflection phase.
        
        This is a simplified version focused on meaningful terms,
        filtering out stop words and short words.
        
        Args:
            query: User query string
            
        Returns:
            List of key terms
        """
        import re
        
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Common stop words to filter
        stopwords = {
            'what', 'is', 'the', 'a', 'an', 'how', 'do', 'does', 'in', 'for',
            'of', 'to', 'from', 'with', 'at', 'by', 'on', 'this', 'that',
            'can', 'could', 'should', 'would', 'will', 'and', 'or', 'but'
        }
        
        # Extract meaningful terms (longer than 3 chars, not stop words)
        key_terms = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Also try to capture important 2-word phrases
        phrases = []
        for i in range(len(words) - 1):
            if words[i] not in stopwords or words[i+1] not in stopwords:
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 7:  # Meaningful phrase length
                    phrases.append(phrase)
        
        # Combine and deduplicate
        all_terms = key_terms + phrases
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in all_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms[:10]  # Limit to 10 most relevant terms

    async def _query_knowledge_graph(self, query_terms: List[str], trace_id: Optional[str] = None) -> List[Dict]:
        """
        Query knowledge graph for concepts and definitions.

        Returns list of candidate dictionaries with proper citations.
        Format includes citation_id for citation pool extraction.
        """
        kg_candidates = []
        seen_citations = set()  # Track duplicates

        if not self.kg_loader or not self.kg_loader.is_full_graph_loaded():
            logger.warning("Knowledge graph not loaded, skipping KG query")
            return kg_candidates

        try:
            kg_results = self.kg_loader.query_definitions(query_terms, trace_id=trace_id)

            for kg_result in kg_results:
                citation_id = kg_result["citation_id"]
                
                # Skip duplicates
                if citation_id in seen_citations:
                    logger.debug(f"Skipping duplicate citation: {citation_id}")
                    continue
                
                seen_citations.add(citation_id)  # Track this citation
                
                # Calculate confidence based on match quality and definition presence
                has_definition = bool(kg_result.get("definition"))
                base_confidence = CONFIDENCE.KG_WITH_DEFINITION if has_definition else CONFIDENCE.KG_WITHOUT_DEFINITION
                relevance = kg_result.get("score", 75) / 100.0

                candidate = {
                    "element": kg_result["label"],
                    "type": "Knowledge Graph Concept",
                    "citation": citation_id,
                    "citation_id": citation_id,  # Keep both for backwards compatibility
                    "confidence": base_confidence * relevance,
                    "definition": kg_result.get("definition", ""),
                    "source": "SKOS/IEC/ENTSOE",
                    "priority": "knowledge_graph"
                }
                kg_candidates.append(candidate)

        except Exception as e:
            logger.error(f"KG query failed: {e}", exc_info=True)

        return kg_candidates

    async def _query_archimate_models(self, query_terms: List[str], trace_id: Optional[str] = None) -> List[Dict]:
        """
        Query ArchiMate models for architectural elements.

        Returns list of candidate dictionaries with proper citations.
        Format includes both 'id' and 'citation_id' for citation pool extraction.
        """
        archimate_candidates = []

        try:
            elements = self.archimate_parser.get_citation_candidates(query_terms)

            for element in elements[:10]:  # Limit to top 10
                citation_id = element.get_citation_id()
                element_id = element.id

                candidate = {
                    "element": element.name,
                    "type": element.type,
                    "layer": element.layer,
                    "id": element_id,
                    "citation": citation_id,
                    "citation_id": citation_id,
                    "confidence": self._calculate_element_confidence(element, query_terms),
                    "togaf_phase": self._get_togaf_phase_for_layer(element.layer),
                    "source": "ArchiMate Model",
                    "priority": "archimate"
                }
                archimate_candidates.append(candidate)

        except Exception as e:
            logger.error(f"ArchiMate query failed: {e}", exc_info=True)

        return archimate_candidates

    async def _query_documents(self, query_terms: List[str], trace_id: Optional[str] = None) -> List[Dict]:
        """
        Query PDF documents for relevant content.

        Returns list of candidate dictionaries with proper citations.
        Format includes 'doc_id' for citation pool extraction.
        """
        doc_candidates = []

        if not self.pdf_indexer:
            return doc_candidates

        try:
            chunks = self.pdf_indexer.search_documents(query_terms, max_results=5)

            for chunk in chunks:
                if hasattr(chunk, 'doc_id') and chunk.doc_id and hasattr(chunk, 'page_number'):
                    citation_id = f"doc:{chunk.doc_id}:page{chunk.page_number}"

                    candidate = {
                        "element": chunk.title,
                        "type": "Document",
                        "doc_id": chunk.doc_id,
                        "citation": citation_id,
                        "citation_id": citation_id,
                        "confidence": CONFIDENCE.DOCUMENT_CHUNKS,
                        "definition": chunk.content[:200] + "...",
                        "source": "PDF Documents",
                        "priority": "document"
                    }
                    doc_candidates.append(candidate)

        except Exception as e:
            logger.error(f"Document query failed: {e}", exc_info=True)

        return doc_candidates

    def _calculate_element_confidence(self, element: ArchiMateElement, query_terms: List[str]) -> float:
        """Calculate confidence score for an ArchiMate element based on query relevance."""
        base_confidence = CONFIDENCE.ARCHIMATE_ELEMENTS
        element_name_lower = element.name.lower()

        # Boost for exact matches
        for term in query_terms:
            if term in element_name_lower:
                base_confidence += CONFIDENCE.EXACT_TERM_MATCH_BONUS

        # Boost for energy domain terms
        energy_terms = ["congestion", "grid", "scada", "monitoring", "management", "power", "reactive", "capability"]
        for term in energy_terms:
            if term in element_name_lower:
                base_confidence += CONFIDENCE.PARTIAL_TERM_MATCH_BONUS

        return min(base_confidence, CONFIDENCE.KG_WITH_DEFINITION)

    def _get_togaf_phase_for_layer(self, layer: str) -> str:
        """Get appropriate TOGAF phase for ArchiMate layer."""
        layer_phase_mapping = {
            "Business": "Phase B",
            "Application": "Phase C",
            "Technology": "Phase D",
            "Strategy": "Phase A",
            "Physical": "Phase D"
        }
        return layer_phase_mapping.get(layer, "Unknown")

    def _get_togaf_context(self, query: str, candidates: List[Dict]) -> Dict:
        """Determine TOGAF context based on query and candidates."""
        query_lower = query.lower()

        # Determine phase from query keywords
        if "business" in query_lower or "capability" in query_lower or "process" in query_lower:
            primary_phase = "Phase B"
            guidance = "Focus on business architecture and capabilities"
        elif "application" in query_lower or "system" in query_lower or "component" in query_lower:
            primary_phase = "Phase C"
            guidance = "Focus on application architecture and data"
        elif "technology" in query_lower or "infrastructure" in query_lower:
            primary_phase = "Phase D"
            guidance = "Focus on technology architecture"
        else:
            primary_phase = "Phase B"  # Default
            guidance = "General business architecture guidance"

        # Determine involved layers from candidate dictionaries
        layers = []
        if candidates:
            for candidate in candidates:
                if isinstance(candidate, dict) and "layer" in candidate:
                    layer = candidate.get("layer")
                    if layer and layer not in layers:
                        layers.append(layer)

        return {
            "primary_phase": primary_phase,
            "involved_layers": layers,
            "adm_guidance": guidance
        }

    def _get_togaf_phase_guidance(self, query: str) -> Dict:
        """Get specific TOGAF phase guidance based on query."""
        query_lower = query.lower()

        if "phase b" in query_lower or "business architecture" in query_lower:
            return {
                "phase": "Phase B",
                "phase_letter": "B",
                "description": "Business Architecture - develop business architecture views",
                "deliverables": ["Business Architecture Document", "Capability Assessment"]
            }
        elif "phase c" in query_lower or "application" in query_lower:
            return {
                "phase": "Phase C",
                "phase_letter": "C",
                "description": "Information Systems Architecture - application and data architecture",
                "deliverables": ["Application Architecture Document", "Data Architecture Document"]
            }
        elif "phase d" in query_lower or "technology" in query_lower:
            return {
                "phase": "Phase D",
                "phase_letter": "D",
                "description": "Technology Architecture - define technology platform",
                "deliverables": ["Technology Architecture Document", "Technology Standards"]
            }
        else:
            return {
                "phase": "Phase B",
                "phase_letter": "B",
                "description": "Business Architecture - general business guidance",
                "deliverables": ["Architecture Vision", "Business Principles"]
            }

    async def _refine_response(
        self,
        query: str,
        retrieval_context: Dict,
        citation_pool: List[Dict],
        trace_id: str = None
    ) -> Tuple[str, List[Dict]]:
        """
        SIMPLIFIED response refinement WITH CITATION POOL.

        Strategy:
        1. Try LLM if available (pass citation pool to constrain citations)
        2. Fall back to enhanced template
        3. No complex conditional logic - same flow for all query types

        Args:
            query: User query
            retrieval_context: Retrieval context with candidates
            citation_pool: List of citation dicts with metadata
            trace_id: Optional trace ID for logging

        Returns:
            Tuple of (response_text, critic_candidates)
        """
        candidates = retrieval_context.get("candidates", [])

        # Handle no candidates case
        if not candidates:
            raise UngroundedReplyError(
                f"No grounded information found for query: '{query}'. "
                f"This query could not be answered using Alliander's knowledge base, "
                f"ArchiMate models, or available documents. Consider rephrasing with "
                f"architecture-specific terms or consult external resources.",
                required_prefixes=[]
            )

        # Sort candidates by priority and confidence (UNIFIED SORTING)
        sorted_candidates = self._sort_candidates(candidates, query)

        # Try LLM first
        try:
            await self._initialize_llm()

            if self.llm_provider:
                format_type = self._determine_response_format(query)
                response = await self._generate_llm_response(
                    query,
                    retrieval_context,
                    citation_pool,  # Pass enriched citation pool
                    format_type,
                    trace_id
                )
                return response, self._prepare_critic_candidates(sorted_candidates)

        except Exception as e:
            logger.warning(f"LLM generation failed: {e}. Using template fallback.")

        # Fall back to enhanced template
        response = await self._generate_enhanced_template_response(query, retrieval_context, sorted_candidates)
        return response, self._prepare_critic_candidates(sorted_candidates)

    async def _refine_response_for_comparison(
        self,
        original_query: str,
        conversation_context: str,
        retrieval_context: Dict,
        citation_pool: List[Dict],
        trace_id: Optional[str] = None
    ) -> Tuple[str, List[Dict]]:
        """Enhanced refinement for comparison queries with conversation context."""
        
        logger.info("🎯 _refine_response_for_comparison CALLED")

        candidates = retrieval_context.get("candidates", [])
        logger.info(f"🎯 Candidates count: {len(candidates)}")
        
        # 🔍 DEBUG: Show what candidates we have
        print(f"🔍 CANDIDATES DEBUG:")
        for i, c in enumerate(candidates[:5]):
            print(f"🔍   [{i}] element: {c.get('element', 'N/A')}")
            print(f"🔍       citation: {c.get('citation', 'N/A')}")
            print(f"🔍       definition: {c.get('definition', 'N/A')[:80]}...")
        
        # Build structured comparison at all times
        if len(candidates) >= 2:
            logger.info("🎯 Building comparison for 2+ candidates")

            # Validate and select distinct candidates for comparison
            c1, c2 = await self._validate_comparison_candidates(candidates, original_query)
            logger.info(f"🎯 Selected candidates after validation: {c1.get('element', 'N/A')} vs {c2.get('element', 'N/A')}")
            
            label1 = c1.get('element', 'Concept 1')
            label2 = c2.get('element', 'Concept 2')
            def1 = c1.get('definition') or 'Definition not available'
            def2 = c2.get('definition') or 'Definition not available'
            cite1 = c1.get('citation', 'N/A')
            cite2 = c2.get('citation', 'N/A')
            
            logger.info(f"🎯 Comparing: {label1} vs {label2}")
            logger.info(f"🎯 Citations: {cite1}, {cite2}")
            
            # Build response with EXPLICIT citations in brackets
            response = f"""**Comparison: {label1} vs {label2}**

    **{label1}** [{cite1}]:
    {def1}

    **{label2}** [{cite2}]:
    {def2}

    ---

    **Key Differences** [{cite1}][{cite2}]:

    From the definitions above:
    - **{label1}**: {def1[:100] if len(def1) > 100 else def1} [{cite1}]
    - **{label2}**: {def2[:100] if len(def2) > 100 else def2} [{cite2}]

    Both concepts are related to power systems. The {label1.lower()} [{cite1}] represents one aspect, while {label2.lower()} [{cite2}] represents another complementary aspect.

    ---

    **References:**
    - {label1}: [{cite1}]
    - {label2}: [{cite2}]
    """
            
            logger.info(f"🎯 Built response with {len(response)} chars")
            logger.info(f"🎯 Response contains citations: {cite1}, {cite2}")
            
            return response, candidates
        
        elif len(candidates) == 1:  # ← This must be at same level as 'if'
            logger.info("🎯 Building comparison for 1 candidate")
            
            c = candidates[0]
            label = c.get('element', 'Concept')
            definition = c.get('definition', '') or 'Definition not available'
            citation = c.get('citation', 'N/A')
            
            response = f"""**{label}** [{citation}]:

    {definition}

    *Based on previous discussion in this session.*

    **Reference:** [{citation}]
    """
            logger.info(f"🎯 Built single concept response")
            return response, candidates
        
        # LAST RESORT: If no candidates at all
        logger.error("🎯 NO CANDIDATES - This should not happen for comparison!")
        logger.error(f"🎯 Query: {original_query}")
        logger.error(f"🎯 Retrieval context keys: {retrieval_context.keys()}")
        
        # Don't call _refine_response - build emergency response
        if citation_pool:
            emergency_response = f"""Unable to compare concepts with limited data.

    **Available references:**
    - [{citation_pool[0].get('citation', 'N/A')}]
    """
            return emergency_response, []
        
        # Absolute last resort
        return "Unable to generate comparison with available data.", []

    def _sort_candidates(self, candidates: List[Dict], query: str) -> List[Dict]:
        """
        UNIFIED candidate sorting - same logic for all query types.

        Priority:
        1. Knowledge Graph with definitions (highest)
        2. Knowledge Graph without definitions
        3. ArchiMate elements
        4. Documents

        Within each priority tier, sort by confidence.
        """
        def sort_key(candidate: Dict) -> Tuple:
            priority = candidate.get("priority", "")
            has_definition = bool(candidate.get("definition"))
            confidence = candidate.get("confidence", 0)

            # Assign priority scores
            if priority == "knowledge_graph" and has_definition:
                priority_score = RANKING_CONFIG.PRIORITY_SCORE_DEFINITION * 10  # Scale up for this function
            elif priority == "knowledge_graph":
                priority_score = RANKING_CONFIG.PRIORITY_SCORE_NORMAL * 10
            elif priority == "archimate":
                priority_score = RANKING_CONFIG.PRIORITY_SCORE_CONTEXT * 10
            elif priority == "togaf":
                priority_score = (RANKING_CONFIG.PRIORITY_SCORE_NORMAL + RANKING_CONFIG.PRIORITY_SCORE_CONTEXT) * 5  # 700 equivalent
            elif priority == "document":
                priority_score = RANKING_CONFIG.PRIORITY_SCORE_CONTEXT * 7  # 420 equivalent
            else:
                priority_score = RANKING_CONFIG.PRIORITY_SCORE_FALLBACK * 4  # 200 equivalent

            # Return Tuple for sorting (higher priority first, then higher confidence)
            return (priority_score + confidence, confidence)

        return sorted(candidates, key=sort_key, reverse=True)

    def _determine_response_format(self, query: str) -> str:
        """Determine appropriate response format based on query intent."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["recommend", "should", "use", "choose"]):
            return "recommendation"
        elif any(word in query_lower for word in ["analyze", "assessment", "evaluate"]):
            return "analysis"
        elif any(word in query_lower for word in ["how", "implement", "guide", "steps"]):
            return "guidance"
        else:
            return "recommendation"  # Default

    async def _generate_llm_response(self, query, retrieval_context, citation_pool, format_type, trace_id):
        # Use LLM Council instead of single LLM
        council_response = await self.llm_council.get_validated_response(
            query=query,
            context=retrieval_context,
            citation_pool=citation_pool
        )
        return council_response.content

    async def _generate_enhanced_template_response(self, query: str, retrieval_context: Dict,
                                                sorted_candidates: List[Dict]) -> str:

        """
        ENHANCED template response generation.

        Creates domain-aware, multi-source synthesis without LLM.
        Much richer than the old version.

        NOTE: This is now async because it calls async language detection.
        """
        if not sorted_candidates:
            return "Unable to find relevant information in Alliander's knowledge base."

        # Detect query type for appropriate response structure
        is_definition = is_definition_query(query)

        if is_definition:
            return await self._build_definition_response(sorted_candidates, retrieval_context)
        else:
            return await self._build_recommendation_response(sorted_candidates, retrieval_context, query)  # ← Add await

    async def _build_definition_response(self, candidates: List[Dict], context: Dict) -> str:
        """
        Build comprehensive definition response.

        SIMPLIFIED:
        - Shows only: definition, URI, vocabulary name
        - No assumptions about metadata fields
        - Distinguishes "Related Concepts" from "Other Results"
        """
        primary = candidates[0]

        lines = []

        # Primary definition
        element = primary.get("element", "Unknown")
        definition = primary.get("definition", "")

        # Better citation extraction
        citation = self._extract_citation(primary, fallback="unknown")

        
        lines.append(f"**{element}**")
        lines.append("")
        
        if definition:
            lines.append(f"{definition}")
            lines.append("")
        
        # Get vocabulary name and URI
        if self.kg_loader and citation != "unknown":
            metadata = self.kg_loader.get_citation_metadata(citation)
            if metadata:
                uri = metadata.get("uri", "")
                
                # Get vocabulary name
                vocab_name = self.kg_loader.get_vocabulary_membership(citation)
                if not vocab_name:
                    vocab_name = self.kg_loader._extract_vocabulary_name_from_uri(uri)
                
                # Show simple reference section
                lines.append("---")
                lines.append("")
                lines.append("**Reference:**")
                
                if vocab_name:
                    lines.append(f"• **Vocabulary:** {vocab_name}")
                
                if uri:
                    lines.append(f"• **URI:** {uri}")
                
                lines.append(f"• **Citation:** `{citation}`")
                lines.append("")
        else:
            # Fallback if no metadata
            source = primary.get("source", "")
            if source:
                lines.append(f"*Source: {source} `{citation}`*")
                lines.append("")
        
        # Detect query language
        query_language = await self._detect_language_with_llm(element + " " + definition)
        
        # Check for explicit SKOS relations (skos:related, skos:broader, skos:narrower)
        has_explicit_relations = self._has_explicit_relations(primary, citation)
        
        if has_explicit_relations:
            lines.append("---")
            lines.append("")
            lines.append("**Related Concepts:**")
            lines.append("*Concepts explicitly linked in the vocabulary*")
            lines.append("")

            for relation in has_explicit_relations[:3]:
                lines.append(f"• **{relation['label']}**: {relation.get('definition', 'No definition available')} `{relation['citation']}`")

            lines.append("")

        # ✅ FIXED: Proper filtering and deduplication
        primary_term = primary.get("element", "").lower()
        main_words = primary_term.split()[:2]  # Get first 2 words, e.g., ["reactive", "power"]
        
        seen_citations = {citation}  # ✅ Don't duplicate primary
        other_results = []
        
        for c in candidates[1:10]:
            if not c.get("definition"):
                continue
            
            candidate_citation = self._extract_citation(c, fallback="unknown")
            
            # ✅ Skip duplicates by citation
            if candidate_citation in seen_citations:
                continue
            
            candidate_term = c.get("element", "").lower()
            
            # ✅ Check if related to main query term
            is_related = any(word in candidate_term for word in main_words if len(word) > 3)
            
            if not is_related:
                continue
            
            # ✅ Check language match
            candidate_text = c.get("element", "") + " " + c.get("definition", "")
            candidate_lang = await self._detect_language_with_llm(candidate_text)
            
            if candidate_lang != query_language and candidate_lang != "unknown":
                continue
            
            # ✅ Add to results
            other_results.append(c)
            seen_citations.add(candidate_citation)
            
            if len(other_results) >= 2:  # ✅ Limit to 2
                break

        if other_results:
            lines.append("---")
            lines.append("")
            lines.append("**Other Results Found:**")
            lines.append("*Additional entries matching your search*")
            lines.append("")

            for other in other_results:
                other_name = other.get("element", "Unknown")
                other_def = other.get("definition", "")
                other_cit = self._extract_citation(other, fallback="unknown")

                # Truncate long definitions
                if len(other_def) > 100:
                    other_def = other_def[:100] + "..."

                lines.append(f"• **{other_name}**: {other_def} `{other_cit}`")

            lines.append("")

        return "\n".join(lines)

    def _has_explicit_relations(self, primary: Dict, citation: Optional[str] = None) -> List[Dict]:
        """
        Check if a concept has explicit SKOS relations in the KG.

        Args:
            primary: Primary candidate dict
            citation: Optional citation string (will extract from primary if not provided)

        Returns list of explicitly related concepts (skos:related, skos:broader, skos:narrower).
        """
        # Extract citation if not provided
        if not citation:
            citation = self._extract_citation(primary, fallback="unknown")

        if not self.kg_loader or not citation or citation == "unknown":
            return []

        try:
            # Get metadata with URI
            metadata = self.kg_loader.get_citation_metadata(citation)
            if not metadata:
                return []

            concept_uri = metadata.get("uri")
            if not concept_uri:
                return []

            # Query KG for SKOS relations
            concept = URIRef(concept_uri)

            related_concepts = []

            # Check for skos:related, skos:broader, skos:narrower
            for s, p, o in self.kg_loader.graph.triples((concept, None, None)):
                pred_str = str(p).lower()

                # Look for SKOS relation predicates
                if any(rel in pred_str for rel in ['related', 'broader', 'narrower']):
                    related_uri = str(o)

                    # Get label and definition for related concept
                    related_label = None
                    related_def = None

                    for s2, p2, o2 in self.kg_loader.graph.triples((o, None, None)):
                        pred2_str = str(p2)
                        if 'prefLabel' in pred2_str:
                            related_label = str(o2)
                        elif 'definition' in pred2_str:
                            related_def = str(o2)

                    if related_label:
                        related_citation = self.kg_loader._extract_citation_id(related_uri)
                        related_concepts.append({
                            "label": related_label,
                            "definition": related_def,
                            "citation": related_citation,
                            "relation_type": pred_str
                        })

            return related_concepts[:5]  # Limit to 5

        except Exception as e:
            logger.warning(f"Could not check for explicit relations: {e}")
            return []

    def _extract_citation(self, item: Dict, fallback: str = "unknown") -> str:
        """
        Extract citation from a dictionary with consistent fallback logic.

        Args:
            item: Dictionary that may contain citation
            fallback: Default value if no citation found

        Returns:
            Citation string or fallback value
        """
        # Try common citation field names in priority order
        citation = (
            item.get("citation") or
            item.get("citation_id") or
            item.get("citation_ref") or
            item.get("id")  # Last resort for some ArchiMate elements
        )

        # Validate it's actually a string and not empty
        if citation and isinstance(citation, str) and citation.strip():
            return citation.strip()

        return fallback

    async def _build_recommendation_response(self, candidates: List[Dict], context: Dict, query: str) -> str:
        """
        Build comprehensive recommendation response.

        Synthesizes multiple sources with TOGAF alignment and domain awareness.
        """
        primary = candidates[0]

        lines = []
        lines.append("**Architectural Recommendation**")
        lines.append("")

        # Primary recommendation
        element = primary.get("element", "Unknown")
        citation = primary.get("citation", "unknown")
        elem_type = primary.get("type", "Element")
        confidence = primary.get("confidence", 0)

        lines.append(f"**Recommended Element:** {element}")
        lines.append(f"• Type: {elem_type}")
        lines.append(f"• Citation: [{citation}]")
        lines.append(f"• Confidence: {confidence:.0%}")

        if primary.get("layer"):
            lines.append(f"• Layer: {primary['layer']}")

        if primary.get("definition"):
            lines.append("")
            lines.append(f"**Definition:** {primary['definition']}")

        lines.append("")

        # TOGAF alignment
        togaf_context = context.get("togaf_context", {})
        if togaf_context:
            lines.append("---")
            lines.append("")
            lines.append("**TOGAF Alignment:**")
            phase = togaf_context.get("primary_phase", "Phase B")
            lines.append(f"• Primary Phase: {phase}")
            if togaf_context.get("adm_guidance"):
                lines.append(f"• Guidance: {togaf_context['adm_guidance']}")
            lines.append("")

        # Alternative options
        if len(candidates) > 1:
            lines.append("---")
            lines.append("")
            lines.append("**Alternative Options:**")
            lines.append("")
            for idx, alt in enumerate(candidates[1:4], 2):
                alt_name = alt.get("element", "Unknown")
                alt_cit = alt.get("citation", "unknown")
                alt_type = alt.get("type", "Element")
                lines.append(f"{idx}. **{alt_name}** [{alt_cit}]")
                lines.append(f"   Type: {alt_type}, Confidence: {alt.get('confidence', 0):.0%}")
            lines.append("")

        # Implementation guidance
        lines.append("---")
        lines.append("")
        lines.append("**Implementation Guidance:**")
        lines.append("")
        lines.append("1. **Modeling:** Document this element in your ArchiMate models using the Archi tool")
        lines.append("2. **Standards Compliance:** Ensure alignment with IEC 61968/61970 standards")
        lines.append("3. **TOGAF Process:** Follow ADM methodology for implementation")
        lines.append("4. **Version Control:** Maintain models in GitHub repositories")

        # Domain-specific guidance
        domain_context = context.get("domain_context", {})
        if domain_context:
            lines.append("")
            lines.append("**Domain Considerations:**")
            lines.append(f"• Domain: {domain_context.get('domain', 'Energy Systems')}")
            standards = domain_context.get("standards", [])
            if standards:
                lines.append(f"• Required Standards: {', '.join(standards[:2])}")
            lines.append("• Regulatory Context: Dutch DSO regulations apply")

        return "\n".join(lines)

    def _prepare_critic_candidates(self, sorted_candidates: List[Dict]) -> List[Dict]:
        """Prepare candidates for critic assessment."""
        critic_candidates = []
        for candidate in sorted_candidates:
            critic_candidates.append({
                "element": candidate["element"],
                "confidence": candidate.get("confidence", 0.5),
                "citations": [candidate.get("citation", "unknown")]
            })
        return critic_candidates

    async def _validate_togaf_alignment(self, candidates: List[Dict],
                                   retrieval_context: Dict) -> Tuple[Optional[str], List[Dict]]:
        """
        Validate TOGAF phase alignment for architectural elements.

        Args:
            candidates: Response candidates
            retrieval_context: Retrieval context

        Returns:
            Tuple of (togaf_phase, validated_elements)
        """
        togaf_phase = None
        validated_elements = []

        for candidate in candidates:
            # FIXED: Handle None citations safely
            citations_list = candidate.get("citations", [])

            # Skip if no citations or citations is None
            if not citations_list:
                continue

            # Get first non-None citation
            citation = None
            for c in citations_list:
                if c and isinstance(c, str):
                    citation = c
                    break

            # Skip if no valid citation found
            if not citation:
                continue

            # Only validate ArchiMate citations
            if citation.startswith("archi:"):
                element = self.archimate_parser.validate_citation(citation)

                if element:
                    if element.layer == "Business":
                        phase = "Phase B"
                    elif element.layer == "Application":
                        phase = "Phase C"
                    elif element.layer == "Technology":
                        phase = "Phase D"
                    else:
                        phase = "Unknown"

                    is_aligned = self.archimate_parser.validate_togaf_alignment(element, phase)

                    validated_elements.append({
                        "element": element.name,
                        "type": element.type,
                        "layer": element.layer,
                        "phase": phase,
                        "aligned": is_aligned,
                        "citation": citation
                    })

                    if not togaf_phase and is_aligned:
                        togaf_phase = phase

        return togaf_phase, validated_elements

    def get_audit_trail(self, session_id: str) -> Optional[Dict]:
        """Get audit trail for a session."""
        return self.context_store.get(session_id)

    def get_statistics(self) -> Dict:
        """Get agent statistics."""
        kg_stats = {"triple_count": len(self.kg_loader.graph)} if self.kg_loader.graph else {}
        model_stats = self.archimate_parser.get_model_summary()

        stats = {
            "knowledge_graph": kg_stats,
            "archimate_models": model_stats,
            "router_config": self.router.get_routing_stats(),
            "critic_config": self.critic.get_assessment_stats(),
            "citation_validator_available": self.citation_validator is not None,
            "citation_pools_loaded": len(self.all_citations),
            "citation_metadata_cached": len(self.citation_metadata_cache),
            "sessions_processed": len(self.context_store)
        }

        # Add embedding stats only if available
        if self.embedding_agent:
            stats["embedding_agent"] = {
                "available": True,
                "embeddings_loaded": len(self.embedding_agent.embeddings.get('texts', [])) if hasattr(self.embedding_agent, 'embeddings') else 0,
                "cache_location": str(self.embedding_agent.cache_dir) if hasattr(self.embedding_agent, 'cache_dir') else None
            }
        else:
            stats["embedding_agent"] = {
                "available": False,
                "reason": "sentence-transformers not installed or disabled"
            }

        return stats

    def _extract_comparison_terms(self, query: str) -> List[str]:
        """Extract comparison terms from a query."""
        if not query:
            return []

        # Common comparison patterns - fixed to be non-greedy and more specific
        comparison_patterns = COMPARISON_PATTERNS

        terms = []
        query_lower = query.lower().strip()

        for pattern in comparison_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    for term in match:
                        if term and term.strip():
                            cleaned_term = self._clean_comparison_term(term.strip())
                            if cleaned_term:
                                terms.append(cleaned_term)
                elif match and match.strip():
                    cleaned_term = self._clean_comparison_term(match.strip())
                    if cleaned_term:
                        terms.append(cleaned_term)

        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)

        return unique_terms

    def _clean_comparison_term(self, term: str) -> str:
        """Clean up extracted comparison term."""
        if not term:
            return ""

        # Remove common stop words
        stop_words = COMPARISON_TERM_STOP_WORDS
        words = term.split()
        cleaned_words = [w for w in words if w.lower() not in stop_words]
        cleaned = ' '.join(cleaned_words)

        # Remove extra whitespace and punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'[^\w\s\-]', '', cleaned)

        return cleaned.strip()

    async def _validate_comparison_candidates(self, candidates: List[Dict], query: str) -> Tuple[Dict, Dict]:
        """Ensure we have two distinct concepts for comparison."""
        if len(candidates) < 2:
            # Not enough candidates for comparison
            return candidates[0] if candidates else {}, {}

        comparison_terms = self._extract_comparison_terms(query)

        if len(comparison_terms) != 2:
            # Fallback to first two candidates
            return candidates[0], candidates[1]

        term1_candidates = []
        term2_candidates = []

        for candidate in candidates:
            element_lower = candidate.get('element', '').lower()
            definition_lower = candidate.get('definition', '').lower()

            # Check which term this candidate matches
            term1_match = (comparison_terms[0].lower() in element_lower or
                          comparison_terms[0].lower() in definition_lower)
            term2_match = (comparison_terms[1].lower() in element_lower or
                          comparison_terms[1].lower() in definition_lower)

            # Only add to one list, prioritizing exact match in element name
            if term1_match and not term2_match:
                term1_candidates.append(candidate)
            elif term2_match and not term1_match:
                term2_candidates.append(candidate)
            elif term1_match and term2_match:
                # If both match, skip ambiguous candidates
                continue

        # Return best matches or fallback to semantic search
        if term1_candidates and term2_candidates:
            return term1_candidates[0], term2_candidates[0]
        else:
            # Use semantic fallback if available
            if self.embedding_agent:
                return await self._semantic_comparison_fallback(query, candidates)
            else:
                # Final fallback to first two distinct candidates
                return candidates[0], candidates[1] if len(candidates) > 1 else candidates[0]

    async def _semantic_comparison_fallback(self, query: str, existing_candidates: List[Dict]) -> Tuple[Dict, Dict]:
        """Use semantic search to find distinct concepts for comparison."""

        comparison_terms = self._extract_comparison_terms(query)

        if len(comparison_terms) != 2:
            return existing_candidates[0], existing_candidates[1] if len(existing_candidates) > 1 else existing_candidates[0]

        # Search for each term separately using embedding agent
        try:
            concept1_results = self.embedding_agent.semantic_search(
                comparison_terms[0],
                top_k=SEMANTIC_CONFIG.TOP_K_COMPARISON,
                min_score=SEMANTIC_CONFIG.MIN_SCORE_COMPARISON
            )
            concept2_results = self.embedding_agent.semantic_search(
                comparison_terms[1],
                top_k=SEMANTIC_CONFIG.TOP_K_COMPARISON,
                min_score=SEMANTIC_CONFIG.MIN_SCORE_COMPARISON
            )

            # Convert to candidate format
            candidate1 = self._semantic_result_to_candidate(concept1_results[0]) if concept1_results else existing_candidates[0]
            candidate2 = self._semantic_result_to_candidate(concept2_results[0]) if concept2_results else (existing_candidates[1] if len(existing_candidates) > 1 else existing_candidates[0])

            return candidate1, candidate2

        except Exception as e:
            logger.warning(f"Semantic comparison fallback failed: {e}")
            # Final fallback to existing candidates
            return existing_candidates[0], existing_candidates[1] if len(existing_candidates) > 1 else existing_candidates[0]

    def _semantic_result_to_candidate(self, result) -> Dict:
        """Convert EmbeddingResult to candidate format."""
        if not result:
            return {}

        return {
            "element": result.text[:100] if hasattr(result, 'text') else "Semantic Match",
            "type": "Semantic Match",
            "citation": getattr(result, 'citation', None) or f"semantic:{getattr(result, 'source', 'unknown')}",
            "confidence": getattr(result, 'score', 0.5),
            "definition": getattr(result, 'text', ''),
            "source": f"Semantic Search ({getattr(result, 'source', 'unknown')})",
            "priority": "normal",
            "semantic_score": getattr(result, 'score', 0.5)
        }

    async def _semantic_enhancement(self, query: str, structured_results: List[Dict]) -> List[Dict]:
        """Use embeddings to find semantically related concepts."""

        semantic_candidates = []

        # Get semantic matches with quality threshold
        semantic_results = self.embedding_agent.semantic_search(
            query,
            top_k=SEMANTIC_CONFIG.TOP_K_PRIMARY,
            min_score=SEMANTIC_CONFIG.MIN_SCORE_PRIMARY
        )

        # Track seen concepts to avoid duplicates
        seen_citations = {c.get('citation') for c in structured_results if c.get('citation')}

        for result in semantic_results:
            # Skip if duplicate of structured result
            result_citation = getattr(result, 'citation', None) or f"semantic:{getattr(result, 'source', 'unknown')}"
            if result_citation in seen_citations:
                continue

            # Only add high-quality semantic results
            result_score = getattr(result, 'score', 0.0)
            if result_score >= SEMANTIC_CONFIG.MIN_SCORE_PRIMARY:
                candidate = {
                    "element": getattr(result, 'text', '')[:100],
                    "type": "Semantic Enhancement",
                    "citation": result_citation,
                    "confidence": result_score,
                    "definition": getattr(result, 'text', ''),
                    "source": f"Semantic ({result_score:.2f})",
                    "priority": "context",  # Lower priority than structured
                    "semantic_score": result_score
                }
                semantic_candidates.append(candidate)

        # Limit to top 3 semantic enhancements to avoid overwhelming
        return semantic_candidates[:SEMANTIC_CONFIG.MAX_SEMANTIC_CANDIDATES]

    async def _context_expansion(self, query: str, session_id: str) -> List[Dict]:
        """Expand context using conversation history."""

        context_candidates = []

        # Get recent conversation history
        session_data = self.session_manager.get_session_data(session_id)
        if not session_data or len(session_data.get("messages", [])) < 2:
            return context_candidates

        # Extract concepts from last 3 turns (not 4 - reduce noise)
        previous_concepts = []
        for message in session_data["messages"][-3:]:
            if message["type"] == "user":
                concepts = self._extract_query_terms(message["content"])
                previous_concepts.extend(concepts)

        # Deduplicate and limit
        previous_concepts = list(set(previous_concepts))[:5]

        if previous_concepts and self.embedding_agent:
            # Create context-enhanced query
            enhanced_query = f"{query} {' '.join(previous_concepts[:3])}"

            related_results = self.embedding_agent.semantic_search(
                enhanced_query,
                top_k=SEMANTIC_CONFIG.TOP_K_CONTEXT,
                min_score=SEMANTIC_CONFIG.MIN_SCORE_CONTEXT
            )

            for result in related_results:
                candidate = {
                    "element": getattr(result, 'text', '')[:100],
                    "type": "Context Enhancement",
                    "citation": getattr(result, 'citation', None) or "context:history",
                    "confidence": getattr(result, 'score', 0.5),
                    "definition": getattr(result, 'text', ''),
                    "source": f"History ({getattr(result, 'source', 'unknown')})",
                    "priority": "context"
                }
                context_candidates.append(candidate)

        return context_candidates

    def _rank_and_deduplicate(self, candidates: List[Dict], query: str) -> List[Dict]:
        """Rank and deduplicate candidates by relevance and source priority."""

        # Remove exact duplicates by citation
        seen_citations = set()
        unique_candidates = []

        for candidate in candidates:
            citation = candidate.get('citation')
            if citation and citation not in seen_citations:
                seen_citations.add(citation)
                unique_candidates.append(candidate)
            elif not citation:
                unique_candidates.append(candidate)

        # Assign priority scores
        def get_priority_score(candidate):
            priority = candidate.get('priority', 'normal')
            type_name = candidate.get('type', '')
            confidence = candidate.get('confidence', 0.5)
            semantic_score = candidate.get('semantic_score', 0)

            # Priority scoring
            priority_map = {
                'definition': RANKING_CONFIG.PRIORITY_SCORE_DEFINITION,
                'normal': RANKING_CONFIG.PRIORITY_SCORE_NORMAL,
                'context': RANKING_CONFIG.PRIORITY_SCORE_CONTEXT
            }
            base_score = priority_map.get(priority, RANKING_CONFIG.PRIORITY_SCORE_FALLBACK)

            # Add confidence/semantic bonus
            bonus = (confidence * RANKING_CONFIG.CONFIDENCE_BONUS_MULTIPLIER) if confidence > RANKING_CONFIG.CONFIDENCE_BONUS_THRESHOLD else (semantic_score * RANKING_CONFIG.CONFIDENCE_BONUS_MULTIPLIER)

            return base_score + bonus

        # Sort by priority score
        ranked = sorted(unique_candidates, key=get_priority_score, reverse=True)

        # Limit total candidates to prevent overwhelming LLM
        return ranked[:RANKING_CONFIG.MAX_TOTAL_CANDIDATES]

    async def cleanup(self):
        """Cleanup async resources."""
        if self.llm_provider:
            # Add cleanup if provider has it
            pass