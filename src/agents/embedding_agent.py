"""
Embedding Agent - Adds semantic search capabilities when KG/ArchiMate fail.

This agent creates and manages embeddings for all knowledge sources,
providing context-aware fallback when structured queries return nothing.
"""
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Lazy load sentence-transformers to avoid import hanging
SENTENCE_TRANSFORMERS_AVAILABLE = False
SentenceTransformer = None

def _lazy_load_sentence_transformers():
    """Lazy load sentence-transformers library."""
    global SENTENCE_TRANSFORMERS_AVAILABLE, SentenceTransformer
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            from sentence_transformers import SentenceTransformer as ST
            SentenceTransformer = ST
            SENTENCE_TRANSFORMERS_AVAILABLE = True
            logger.debug("✅ Sentence transformers loaded successfully")
        except ImportError as e:
            logger.debug(f"Sentence transformers not available: {e}")
            SENTENCE_TRANSFORMERS_AVAILABLE = False

# Fallback to OpenAI if available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

@dataclass
class EmbeddingResult:
    """Result from semantic search."""
    text: str
    score: float
    source: str  # 'kg', 'archimate', 'togaf', 'custom'
    metadata: Dict
    citation: Optional[str] = None


class EmbeddingAgent:
    """
    Manages embeddings for semantic search across all knowledge sources.
    
    Features:
    - Creates embeddings from KG, ArchiMate, and TOGAF docs
    - Provides semantic search when structured queries fail
    - Caches embeddings for performance
    - Supports both local (sentence-transformers) and API (OpenAI) embeddings
    """
    
    def __init__(
        self,
        kg_loader=None,
        archimate_parser=None,
        pdf_indexer=None,
        embedding_model: str = "all-MiniLM-L6-v2",
        cache_dir: str = "data/embeddings",
        use_openai: bool = False,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize embedding agent.
        
        FIXED: Properly loads sentence-transformers before checking availability.
        
        Args:
            kg_loader: KnowledgeGraphLoader instance
            archimate_parser: ArchiMateParser instance
            pdf_indexer: PDFIndexer instance
            embedding_model: Model name for sentence-transformers
            cache_dir: Directory for caching embeddings
            use_openai: Whether to use OpenAI embeddings instead of local
            openai_api_key: OpenAI API key if use_openai=True
        """
        self.kg_loader = kg_loader
        self.archimate_parser = archimate_parser
        self.pdf_indexer = pdf_indexer
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.use_openai = use_openai and OPENAI_AVAILABLE
        
        if self.use_openai:
            if openai_api_key:
                openai.api_key = openai_api_key
            self.model = None
            logger.info("✅ Using OpenAI embeddings")
        else:
            # CRITICAL FIX: Load sentence-transformers BEFORE checking availability
            logger.info("Attempting to load sentence-transformers...")
            _lazy_load_sentence_transformers()
            
            if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
                try:
                    logger.info(f"Initializing sentence-transformers model: {embedding_model}")
                    self.model = SentenceTransformer(embedding_model)
                    logger.info(f"✅ Embedding agent initialized with sentence-transformers ({embedding_model})")
                except Exception as e:
                    logger.error(f"Failed to load sentence-transformers model: {e}")
                    raise RuntimeError(f"Failed to load sentence-transformers: {e}")
            else:
                error_msg = "No embedding backend available. Install sentence-transformers or openai"
                logger.error(error_msg)
                logger.error("Install with: pip install sentence-transformers torch")
                raise RuntimeError(error_msg)
        
        # Load or create embeddings
        logger.info("Loading or creating embeddings...")
        self.embeddings = self._load_or_create_embeddings()
        logger.info("✅ Embedding agent fully initialized")
        
    def _load_or_create_embeddings(self) -> Dict:
        """Load cached embeddings or create new ones."""
        cache_file = self.cache_dir / "embeddings.pkl"
        
        if cache_file.exists():
            logger.info("Loading cached embeddings...")
            try:
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                    logger.info(f"✅ Loaded {len(embeddings.get('texts', []))} cached embeddings")
                    return embeddings
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}, creating new ones")
        
        logger.info("Creating new embeddings from knowledge sources...")
        embeddings = self._create_embeddings()
        
        # Cache for future use
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"✅ Cached {len(embeddings['texts'])} embeddings to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")
        
        return embeddings
    
    def _create_embeddings(self) -> Dict:
        """Create embeddings from all knowledge sources."""
        texts = []
        metadata = []
        citations = []
        
        # 1. Extract from Knowledge Graph
        if self.kg_loader:
            logger.info("Extracting KG concepts...")
            try:
                kg_texts = self._extract_kg_texts()
                for text, meta, citation in kg_texts:
                    texts.append(text)
                    metadata.append(meta)
                    citations.append(citation)
            except Exception as e:
                logger.warning(f"Failed to extract KG texts: {e}")
        
        # 2. Extract from ArchiMate
        if self.archimate_parser:
            logger.info("Extracting ArchiMate elements...")
            try:
                archimate_texts = self._extract_archimate_texts()
                for text, meta, citation in archimate_texts:
                    texts.append(text)
                    metadata.append(meta)
                    citations.append(citation)
            except Exception as e:
                logger.warning(f"Failed to extract ArchiMate texts: {e}")
        
        # 3. Extract from TOGAF PDFs
        if self.pdf_indexer:
            logger.info("Extracting TOGAF documents...")
            try:
                pdf_texts = self._extract_pdf_texts()
                for text, meta, citation in pdf_texts:
                    texts.append(text)
                    metadata.append(meta)
                    citations.append(citation)
            except Exception as e:
                logger.warning(f"Failed to extract PDF texts: {e}")
        
        # 4. Add domain-specific context
        logger.info("Adding domain contexts...")
        domain_contexts = self._get_domain_contexts()
        for text, meta in domain_contexts:
            texts.append(text)
            metadata.append(meta)
            citations.append(None)
        
        if not texts:
            logger.warning("No texts extracted for embeddings, creating minimal set")
            texts = ["Alliander DSO energy systems"]
            metadata = [{'source': 'fallback'}]
            citations = [None]
        
        logger.info(f"Creating embeddings for {len(texts)} texts...")
        
        # Create embeddings
        try:
            if self.use_openai:
                vectors = self._create_openai_embeddings(texts)
            else:
                vectors = self.model.encode(texts, show_progress_bar=True)
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise
        
        logger.info(f"✅ Created {len(vectors)} embeddings")
        
        return {
            'texts': texts,
            'vectors': vectors,
            'metadata': metadata,
            'citations': citations,
            'created_at': datetime.utcnow().isoformat()
        }
    
    def _extract_kg_texts(self) -> List[Tuple[str, Dict, str]]:
        """Extract texts from knowledge graph."""
        texts = []
        
        # Get all concepts with definitions
        try:
            concepts = self.kg_loader.get_all_concepts_with_definitions()
        except Exception as e:
            logger.warning(f"Failed to get KG concepts: {e}")
            return texts
        
        for concept_uri, label, definition in concepts:
            # Create searchable text
            text = f"{label}: {definition}" if definition else label
            
            # Extract namespace for citation
            if "vocabs.alliander.com" in concept_uri:
                citation = f"skos:{concept_uri.split('/')[-1]}"
            elif "iec.ch" in concept_uri:
                citation = f"iec:{concept_uri.split('/')[-1]}"
            elif "entsoe" in concept_uri:
                citation = f"entsoe:{concept_uri.split('/')[-1]}"
            elif "europa.eu" in concept_uri:
                citation = f"eurlex:{concept_uri.split('/')[-1]}"
            else:
                citation = f"external:{concept_uri.split('/')[-1]}"
            
            metadata = {
                'source': 'knowledge_graph',
                'uri': concept_uri,
                'label': label,
                'type': 'concept'
            }
            
            texts.append((text, metadata, citation))
        
        logger.info(f"✅ Extracted {len(texts)} KG concepts")
        return texts
    
    def _extract_archimate_texts(self) -> List[Tuple[str, Dict, str]]:
        """Extract texts from ArchiMate models."""
        texts = []
        
        if not hasattr(self.archimate_parser, 'elements') or not self.archimate_parser.elements:
            logger.warning("No ArchiMate elements loaded")
            return texts
        
        for elem_id, element in self.archimate_parser.elements.items():
            # Create searchable text
            text = f"{element.name}: {element.documentation}" if element.documentation else element.name
            text = f"[{element.type}] {text}"
            
            citation = f"archi:id-{elem_id}"
            metadata = {
                'source': 'archimate',
                'id': elem_id,
                'name': element.name,
                'type': element.type,
                'layer': element.layer
            }
            
            texts.append((text, metadata, citation))
        
        logger.info(f"✅ Extracted {len(texts)} ArchiMate elements")
        return texts
    
    def _extract_pdf_texts(self) -> List[Tuple[str, Dict, str]]:
        """Extract texts from PDF documents."""
        texts = []
        
        # Get all document chunks
        try:
            chunks = self.pdf_indexer.get_all_chunks()
        except Exception as e:
            logger.warning(f"Failed to get PDF chunks: {e}")
            return texts
        
        for chunk in chunks:
            text = chunk.content  # Fixed: use 'content' not 'text'
            citation = f"doc:{chunk.doc_id}:{chunk.page_number}"
            metadata = {
                'source': 'pdf',
                'doc_id': chunk.doc_id,
                'page': chunk.page_number,
                'doc_type': chunk.doc_type
            }
            
            texts.append((text, metadata, citation))
        
        logger.info(f"✅ Extracted {len(texts)} PDF chunks")
        return texts
    
    def _get_domain_contexts(self) -> List[Tuple[str, Dict]]:
        """Add domain-specific context for better retrieval."""
        contexts = [
            (
                "Alliander is a Distribution System Operator (DSO) in the Netherlands responsible for "
                "managing electricity and gas distribution networks. Key responsibilities include grid "
                "maintenance, congestion management, and facilitating energy transition.",
                {'source': 'domain', 'type': 'organization'}
            ),
            (
                "Grid congestion occurs when electricity demand exceeds network capacity. Solutions include "
                "demand response, flexible connections, and grid reinforcement. IEC 61968 standards provide "
                "data models for congestion management systems.",
                {'source': 'domain', 'type': 'concept'}
            ),
            (
                "TOGAF ADM phases for energy systems: Preliminary (establish architecture capability), "
                "Architecture Vision (define energy transition goals), Business Architecture (DSO processes), "
                "Information Systems (IEC CIM models), Technology Architecture (SCADA, DMS, GIS integration).",
                {'source': 'domain', 'type': 'methodology'}
            ),
            (
                "ArchiMate modeling for DSO: Business layer (market processes, regulatory compliance), "
                "Application layer (grid management systems, customer portals), Technology layer "
                "(substations, smart meters, communication infrastructure).",
                {'source': 'domain', 'type': 'modeling'}
            ),
            (
                "Energy transition requires modeling renewable integration, prosumer management, "
                "flexibility markets, and sector coupling. Key standards: IEC 61970 (EMS-API), "
                "IEC 61968 (DMS interfaces), IEC 62325 (energy market communications).",
                {'source': 'domain', 'type': 'standards'}
            )
        ]
        
        return contexts
    
    def _create_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using OpenAI API."""
        import openai
        
        embeddings = []
        batch_size = 100  # OpenAI limit
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = openai.Embedding.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            batch_embeddings = [item['embedding'] for item in response['data']]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
        source_filter: Optional[List[str]] = None
    ) -> List[EmbeddingResult]:
        """
        Perform semantic search across all embedded content.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            source_filter: Optional list of sources to search ('kg', 'archimate', 'pdf', 'domain')
            
        Returns:
            List of EmbeddingResult objects sorted by relevance
        """
        # Create query embedding
        if self.use_openai:
            import openai
            response = openai.Embedding.create(
                input=[query],
                model="text-embedding-ada-002"
            )
            query_vector = np.array(response['data'][0]['embedding'])
        else:
            query_vector = self.model.encode([query])[0]
        
        # Calculate similarities
        vectors = self.embeddings['vectors']
        if isinstance(vectors, list):
            vectors = np.array(vectors)
        
        # Cosine similarity
        similarities = np.dot(vectors, query_vector) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
        )
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get extra for filtering
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            
            # Apply minimum score threshold
            if score < min_score:
                continue
            
            metadata = self.embeddings['metadata'][idx]
            
            # Apply source filter
            if source_filter and metadata['source'] not in source_filter:
                continue
            
            result = EmbeddingResult(
                text=self.embeddings['texts'][idx],
                score=score,
                source=metadata['source'],
                metadata=metadata,
                citation=self.embeddings['citations'][idx]
            )
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        logger.info(f"Semantic search for '{query}' returned {len(results)} results")
        return results
    
    def get_enhanced_context(
        self,
        query: str,
        kg_results: List[Dict],
        archimate_results: List[Dict]
    ) -> Dict:
        """
        Enhance context with semantic search when structured results are insufficient.
        
        Args:
            query: Original query
            kg_results: Results from KG query
            archimate_results: Results from ArchiMate query
            
        Returns:
            Enhanced context dictionary
        """
        context = {
            'kg_results': kg_results,
            'archimate_results': archimate_results,
            'semantic_results': [],
            'needs_llm_fallback': False
        }
        
        # Check if we have enough structured results
        total_structured = len(kg_results) + len(archimate_results)
        
        if total_structured < 2:
            logger.info("Insufficient structured results, adding semantic search...")
            
            # Perform semantic search
            semantic_results = self.semantic_search(
                query,
                top_k=5,
                min_score=0.4
            )
            
            context['semantic_results'] = [
                {
                    'text': r.text,
                    'score': r.score,
                    'source': r.source,
                    'citation': r.citation,
                    'metadata': r.metadata
                }
                for r in semantic_results
            ]
            
            # If still no good results, flag for LLM fallback
            if len(semantic_results) < 2:
                context['needs_llm_fallback'] = True
                logger.warning("Even semantic search returned limited results, LLM fallback recommended")
        
        return context
    
    def refresh_embeddings(self):
        """Refresh embeddings by recreating from current knowledge sources."""
        logger.info("Refreshing embeddings...")
        
        # Clear cache
        cache_file = self.cache_dir / "embeddings.pkl"
        if cache_file.exists():
            cache_file.unlink()
        
        # Recreate
        self.embeddings = self._load_or_create_embeddings()
        logger.info("✅ Embeddings refreshed successfully")