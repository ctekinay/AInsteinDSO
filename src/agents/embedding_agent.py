# src/agents/embedding_agent.py
"""
Embedding Agent - Adds semantic search capabilities when KG/ArchiMate fail.

This agent creates and manages embeddings for all knowledge sources,
providing context-aware fallback when structured queries return nothing.
"""
from __future__ import annotations

import os
import pickle
import json
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
            from sentence_transformers import SentenceTransformer as ST  # type: ignore
            SentenceTransformer = ST
            SENTENCE_TRANSFORMERS_AVAILABLE = True
            logger.debug("✅ sentence-transformers loaded")
        except ImportError as e:
            logger.debug(f"sentence-transformers not available: {e}")
            SENTENCE_TRANSFORMERS_AVAILABLE = False


try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

@dataclass
class EmbeddingResult:
    """Result from semantic search."""
    text: str
    score: float
    source: str  # 'knowledge_graph', 'archimate', 'pdf', 'domain'
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
        openai_api_key: Optional[str] = None,
        auto_refresh: Optional[bool] = None,
    ):
        """
        Initialize embedding agent.

        Args:
            auto_refresh: If True, automatically refresh embeddings when source files change.
                          If None, uses EMBEDDING_AUTO_REFRESH env var.
                          Defaults to False for production safety.
        """
        self.kg_loader = kg_loader
        self.archimate_parser = archimate_parser
        self.pdf_indexer = pdf_indexer
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if auto_refresh is None:
            auto_refresh_env = os.getenv("EMBEDDING_AUTO_REFRESH", "false").strip().lower()
            self.auto_refresh = auto_refresh_env in ("true", "1", "yes", "y", "on")
            logger.info(f"Auto-refresh setting from env: {auto_refresh_env} -> {self.auto_refresh}")
        else:
            self.auto_refresh = auto_refresh
            logger.info(f"Auto-refresh explicitly set to: {self.auto_refresh}")

        # Initialize embedding model and model configuration
        self.use_openai = use_openai and OPENAI_AVAILABLE
        self.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

        if self.use_openai:
            # prefer explicit arg, else env var OPENAI_API_KEY
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set but use_openai=True")
            self.openai_client = OpenAI(api_key=api_key)
            self.model = None
            logger.info(f"✅ Using OpenAI embeddings ({self.openai_embedding_model})")
        else:
            logger.info("Attempting to load sentence-transformers...")
            _lazy_load_sentence_transformers()
            if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
                logger.info(f"Initializing sentence-transformers model: {embedding_model}")
                self.model = SentenceTransformer(embedding_model)
                logger.info(f"✅ Embedding agent initialized with sentence-transformers ({embedding_model})")
            else:
                error_msg = "No embedding backend available. Install sentence-transformers or set use_openai=True"
                logger.error(error_msg)
                logger.error("Install with: pip install sentence-transformers torch")
                raise RuntimeError(error_msg)

        logger.info(f"Auto-refresh: {'ENABLED' if self.auto_refresh else 'DISABLED'}")

        # Load or create embeddings (single implementation; removed duplicate)
        logger.info("Loading or creating embeddings...")
        self.embeddings = self._load_or_create_embeddings()
        logger.info("✅ Embedding agent fully initialized")

    # ---------- Caching / refresh ----------

    def _get_source_file_info(self) -> Dict[str, float]:
        """
        Get modification times of all source files.
        Returns dict mapping filename → mtime.
        """
        file_info: Dict[str, float] = {}

        # Knowledge Graph TTL file
        if self.kg_loader and hasattr(self.kg_loader, 'kg_path'):
            kg_path = Path(self.kg_loader.kg_path)
            if kg_path.exists():
                file_info['kg_ttl'] = kg_path.stat().st_mtime

        # ArchiMate model files
        if self.archimate_parser and hasattr(self.archimate_parser, 'model_paths'):
            for model_path in getattr(self.archimate_parser, 'model_paths', []):
                path = Path(model_path)
                if path.exists():
                    file_info[f"archimate_{path.name}"] = path.stat().st_mtime

        # PDF documents
        if self.pdf_indexer and hasattr(self.pdf_indexer, 'docs_path'):
            docs_dir = Path(self.pdf_indexer.docs_path)
            if docs_dir.exists():
                for pdf in docs_dir.glob("*.pdf"):
                    file_info[f"pdf_{pdf.name}"] = pdf.stat().st_mtime

        return file_info

    def _load_or_create_embeddings(self) -> Dict:
        """Load cached embeddings or create new ones with optional auto-refresh."""
        cache_file = self.cache_dir / "embeddings.pkl"
        metadata_file = self.cache_dir / "embeddings_metadata.json"

        last_source_mtimes: Dict[str, float] = {}
        if metadata_file.exists() and self.auto_refresh:
            try:
                with open(metadata_file, 'r') as f:
                    meta = json.load(f)
                    last_source_mtimes = meta.get('source_file_mtimes', {})
            except Exception as e:
                logger.warning(f"Failed to load embedding metadata: {e}")

        current_source_mtimes = self._get_source_file_info()

        should_refresh = False
        if self.auto_refresh and cache_file.exists():
            for fname, current_mtime in current_source_mtimes.items():
                last_mtime = last_source_mtimes.get(fname)
                if last_mtime is None or current_mtime > last_mtime:
                    logger.info(f"Source file changed: {fname}")
                    should_refresh = True
                    break

        if not cache_file.exists() or should_refresh:
            logger.info("Creating new embeddings (files changed or cache missing)...")
            embeddings = self._create_embeddings()

            # Cache them
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(embeddings, f)
                logger.info(f"✅ Cached embeddings to {cache_file}")

                metadata = {
                    'created_at': datetime.utcnow().isoformat(),
                    'source_file_mtimes': current_source_mtimes,
                    'embedding_count': len(embeddings['texts']),
                }
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"✅ Saved metadata to {metadata_file}")
            except Exception as e:
                logger.warning(f"Failed to cache embeddings: {e}")

            return embeddings

        logger.info("Loading cached embeddings...")
        try:
            with open(cache_file, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"✅ Loaded {len(embeddings.get('texts', []))} cached embeddings")
            return embeddings
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}, creating new ones")
            return self._create_embeddings()

    def refresh_embeddings(self):
        """Force refresh embeddings by recreating from current knowledge sources."""
        logger.info("🔄 Manually refreshing embeddings...")

        cache_file = self.cache_dir / "embeddings.pkl"
        metadata_file = self.cache_dir / "embeddings_metadata.json"

        for file in (cache_file, metadata_file):
            try:
                if file.exists():
                    file.unlink()
                    logger.debug(f"Deleted {file}")
            except Exception as e:
                logger.warning(f"Could not delete {file}: {e}")

        self.embeddings = self._load_or_create_embeddings()
        logger.info("✅ Embeddings manually refreshed successfully")

    # ---------- Extraction ----------

    def _create_embeddings(self) -> Dict:
        """Create embeddings from all knowledge sources."""
        texts: List[str] = []
        metadata: List[Dict] = []
        citations: List[Optional[str]] = []

        # 1. Knowledge Graph
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

        # 2. ArchiMate
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

        # 3. PDFs
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

        # 4. Domain contexts
        logger.info("Adding domain contexts...")
        for text, meta in self._get_domain_contexts():
            texts.append(text)
            metadata.append(meta)
            citations.append(None)

        if not texts:
            logger.warning("No texts extracted for embeddings, creating minimal set")
            texts = ["Alliander DSO energy systems"]
            metadata = [{'source': 'fallback'}]
            citations = [None]

        logger.info(f"Creating embeddings for {len(texts)} texts...")
        try:
            if self.use_openai:
                vectors = self._create_openai_embeddings(texts)
            else:
                # Ensure numpy ndarray output
                vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)  # type: ignore
                vectors = np.asarray(vectors, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise

        logger.info(f"✅ Created {len(vectors)} embeddings")

        return {
            'texts': texts,
            'vectors': vectors,
            'metadata': metadata,
            'citations': citations,
            'created_at': datetime.utcnow().isoformat(),
        }

    def _extract_kg_texts(self) -> List[Tuple[str, Dict, str]]:
        """Extract texts from knowledge graph."""
        rows: List[Tuple[str, Dict, str]] = []
        try:
            concepts = self.kg_loader.get_all_concepts_with_definitions()
        except Exception as e:
            logger.warning(f"Failed to get KG concepts: {e}")
            return rows

        for concept_uri, label, definition in concepts:
            text = f"{label}: {definition}" if definition else label
            # naive prefix mapping (kept as-is)
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

            rows.append((text, {
                'source': 'knowledge_graph',
                'uri': concept_uri,
                'label': label,
                'type': 'concept',
            }, citation))

        logger.info(f"✅ Extracted {len(rows)} KG concepts")
        return rows

    def _extract_archimate_texts(self) -> List[Tuple[str, Dict, str]]:
        """Extract texts from ArchiMate models."""
        rows: List[Tuple[str, Dict, str]] = []

        if not hasattr(self.archimate_parser, 'elements') or not self.archimate_parser.elements:
            logger.warning("No ArchiMate elements loaded")
            return rows

        for elem_id, element in self.archimate_parser.elements.items():
            name = getattr(element, "name", "") or getattr(element, "properties", {}).get("name") or elem_id
            doc = getattr(element, "documentation", "")
            typ = getattr(element, "type", "Element")
            layer = getattr(element, "layer", None)

            text = f"[{typ}] {name}" if not doc else f"[{typ}] {name}: {doc}"
            citation = f"archi:id-{elem_id}"

            rows.append((text, {
                'source': 'archimate',
                'id': elem_id,
                'name': name,
                'type': typ,
                'layer': layer,
            }, citation))

        logger.info(f"✅ Extracted {len(rows)} ArchiMate elements")
        return rows

    def _extract_pdf_texts(self) -> List[Tuple[str, Dict, str]]:
        """Extract texts from PDF documents."""
        rows: List[Tuple[str, Dict, str]] = []
        try:
            chunks = self.pdf_indexer.get_all_chunks()
        except Exception as e:
            logger.warning(f"Failed to get PDF chunks: {e}")
            return rows

        for chunk in chunks:
            text = getattr(chunk, "content", None) or getattr(chunk, "text", "")
            citation = f"doc:{chunk.doc_id}:{chunk.page_number}"
            rows.append((text, {
                'source': 'pdf',
                'doc_id': chunk.doc_id,
                'page': chunk.page_number,
                'doc_type': getattr(chunk, "doc_type", None),
            }, citation))

        logger.info(f"✅ Extracted {len(rows)} PDF chunks")
        return rows

    def _get_domain_contexts(self) -> List[Tuple[str, Dict]]:
        """Add domain-specific context for better retrieval."""
        return [
            (
                "Alliander is a Distribution System Operator (DSO) in the Netherlands responsible for "
                "managing electricity and gas distribution networks. Key responsibilities include grid "
                "maintenance, congestion management, and facilitating energy transition.",
                {'source': 'domain', 'type': 'organization'},
            ),
            (
                "Grid congestion occurs when electricity demand exceeds network capacity. Solutions include "
                "demand response, flexible connections, and grid reinforcement. IEC 61968 standards provide "
                "data models for congestion management systems.",
                {'source': 'domain', 'type': 'concept'},
            ),
            (
                "TOGAF ADM phases for energy systems: Preliminary (establish architecture capability), "
                "Architecture Vision (define energy transition goals), Business Architecture (DSO processes), "
                "Information Systems (IEC CIM models), Technology Architecture (SCADA, DMS, GIS integration).",
                {'source': 'domain', 'type': 'methodology'},
            ),
            (
                "ArchiMate modeling for DSO: Business layer (market processes, regulatory compliance), "
                "Application layer (grid management systems, customer portals), Technology layer "
                "(substations, smart meters, communication infrastructure).",
                {'source': 'domain', 'type': 'modeling'},
            ),
            (
                "Energy transition requires modeling renewable integration, prosumer management, "
                "flexibility markets, and sector coupling. Key standards: IEC 61970 (EMS-API), "
                "IEC 61968 (DMS interfaces), IEC 62325 (energy market communications).",
                {'source': 'domain', 'type': 'standards'},
            ),
        ]

    # ---------- OpenAI backend ----------

    def _create_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using OpenAI API (v1+ client)."""
        resp = self.openai_client.embeddings.create(
            model=self.openai_embedding_model,
            input=texts
        )
        vectors = [item.embedding for item in resp.data]
        return np.array(vectors, dtype=np.float32)

    # ---------- Search ----------

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
        source_filter: Optional[List[str]] = None,
    ) -> List[EmbeddingResult]:
        """
        Perform semantic search across all embedded content.
        """
        # Create query embedding
        if self.use_openai:
            response = openai.Embedding.create(  # type: ignore[attr-defined]
                input=[query],
                model=self.openai_embedding_model,
            )
            query_vector = np.asarray(response['data'][0]['embedding'], dtype=np.float32)  # type: ignore[index]
        else:
            query_vector = self.model.encode([query], convert_to_numpy=True)[0]  # type: ignore
            query_vector = np.asarray(query_vector, dtype=np.float32)

        vectors = self.embeddings['vectors']
        if isinstance(vectors, list):
            vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.size == 0:
            logger.warning("No stored vectors available for search")
            return []

        if vectors.shape[1] != query_vector.shape[0]:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"stored={vectors.shape[1]} vs query={query_vector.shape[0]}. "
                "This usually happens after changing the embedding model. "
                "Run refresh_embeddings() to rebuild with the current model."
            )

        # Cosine similarity
        denom = (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector))
        # Avoid divide-by-zero
        denom = np.where(denom == 0.0, 1e-12, denom)
        similarities = np.dot(vectors, query_vector) / denom

        # Get top results
        top_indices = np.argsort(similarities)[::-1][: max(top_k * 2, top_k)]

        results: List[EmbeddingResult] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                continue

            meta = self.embeddings['metadata'][idx]
            if source_filter and meta.get('source') not in source_filter:
                continue

            results.append(EmbeddingResult(
                text=self.embeddings['texts'][idx],
                score=score,
                source=meta.get('source', 'unknown'),
                metadata=meta,
                citation=self.embeddings['citations'][idx],
            ))

            if len(results) >= top_k:
                break

        logger.info(f"Semantic search for '{query}' returned {len(results)} results")
        return results

    def get_enhanced_context(
        self,
        query: str,
        kg_results: List[Dict],
        archimate_results: List[Dict],
    ) -> Dict:
        """
        Enhance context with semantic search when structured results are insufficient.
        """
        context = {
            'kg_results': kg_results,
            'archimate_results': archimate_results,
            'semantic_results': [],
            'needs_llm_fallback': False,
        }

        total_structured = len(kg_results) + len(archimate_results)
        if total_structured < 2:
            logger.info("Insufficient structured results, adding semantic search...")
            semantic_results = self.semantic_search(query, top_k=5, min_score=0.4)
            context['semantic_results'] = [
                {
                    'text': r.text,
                    'score': r.score,
                    'source': r.source,
                    'citation': r.citation,
                    'metadata': r.metadata,
                }
                for r in semantic_results
            ]
            if len(semantic_results) < 2:
                context['needs_llm_fallback'] = True
                logger.warning("Even semantic search returned limited results, LLM fallback recommended")

        return context

    # ---------- Stats ----------

    def stats(self) -> Dict:
        """Return simple stats for monitoring."""
        return {
            "total_embeddings": len(self.embeddings.get('texts', [])),
            "sources": {
                "knowledge_graph": sum(1 for m in self.embeddings.get('metadata', []) if m.get('source') == 'knowledge_graph'),
                "archimate": sum(1 for m in self.embeddings.get('metadata', []) if m.get('source') == 'archimate'),
                "pdf": sum(1 for m in self.embeddings.get('metadata', []) if m.get('source') == 'pdf'),
                "domain": sum(1 for m in self.embeddings.get('metadata', []) if m.get('source') == 'domain'),
                "fallback": sum(1 for m in self.embeddings.get('metadata', []) if m.get('source') == 'fallback'),
            },
            "created_at": self.embeddings.get("created_at"),
        }