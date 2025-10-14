"""
PDF Document Indexer for EA Assistant

This module provides PDF parsing and text indexing capabilities for TOGAF
and ArchiMate research documents. It creates searchable indexes without
requiring external vector databases.
"""

import json
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a PDF document."""
    doc_id: str
    chunk_id: str
    title: str
    content: str
    page_number: int
    chunk_index: int
    keywords: List[str]
    doc_type: str  # "togaf_concepts" or "archimate_research"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

class PDFIndexer:
    """
    Simple PDF indexer that creates searchable text chunks without vector embeddings.

    Uses text-based search with keyword matching, suitable for offline deployment
    without requiring LLM APIs for embeddings.
    """

    def __init__(self, docs_path: str = "data/docs/", cache_path: str = "data/document_index.json"):
        """
        Initialize the PDF indexer.

        Args:
            docs_path: Directory containing PDF documents
            cache_path: Path to store the document index cache
        """
        self.docs_path = Path(docs_path)
        self.cache_path = Path(cache_path)
        self.document_chunks: List[DocumentChunk] = []
        self.keyword_index: Dict[str, List[str]] = {}  # keyword -> chunk_ids
        self.loaded = False

    def load_or_create_index(self) -> None:
        """Load existing index or create new one from PDFs."""
        if self.cache_path.exists():
            try:
                self._load_cached_index()
                logger.info(f"Loaded document index with {len(self.document_chunks)} chunks")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}, rebuilding...")

        self._build_index_from_pdfs()
        self._save_index()

    def _load_cached_index(self) -> None:
        """Load cached document index."""
        with open(self.cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.document_chunks = [
            DocumentChunk(**chunk_data) for chunk_data in data['chunks']
        ]
        self.keyword_index = data['keyword_index']
        self.loaded = True
    
    def citation_exists(self, citation: str) -> bool:
        """
        Check if citation exists in document index.
        
        Args:
            citation: Citation like 'togaf:concepts:001', 'archimate:research:005'
            
        Returns:
            True if citation exists in indexed documents
        """
        # Extract document ID pattern
        if citation.startswith("togaf:concepts:"):
            doc_pattern = citation.replace("togaf:concepts:", "togaf_concepts_")
        elif citation.startswith("archimate:research:"):
            doc_pattern = citation.replace("archimate:research:", "archimate_research_")
        elif citation.startswith("doc:"):
            # Format: doc:source:chunk_id
            parts = citation.split(":")
            if len(parts) >= 2:
                doc_pattern = parts[1]
            else:
                return False
        else:
            return False
        
        # Check if pattern exists in indexed documents
        if hasattr(self, 'documents') and self.documents:
            for doc in self.documents:
                if doc_pattern in doc.get('doc_id', ''):
                    return True
        
        return False
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """
        Return all indexed document chunks.
        
        Used by EmbeddingAgent to create embeddings from document content.
        
        Returns:
            List of all DocumentChunk objects
        """
        if not self.loaded:
            logger.warning("Index not loaded, loading now...")
            self.load_or_create_index()
        
        return self.document_chunks

    def _build_index_from_pdfs(self) -> None:
        """Build document index from PDF files."""
        logger.info("Building document index from PDFs...")

        # Process TOGAF concepts document
        togaf_path = self.docs_path / "Core concepts TOGAF.pdf"
        if togaf_path.exists():
            togaf_chunks = self._process_togaf_pdf(togaf_path)
            self.document_chunks.extend(togaf_chunks)
            logger.info(f"Processed TOGAF document: {len(togaf_chunks)} chunks")

        # Process ArchiMate research document
        research_path = self.docs_path / "2501.03566v1.pdf"
        if research_path.exists():
            research_chunks = self._process_research_pdf(research_path)
            self.document_chunks.extend(research_chunks)
            logger.info(f"Processed ArchiMate research: {len(research_chunks)} chunks")

        # Build keyword index
        self._build_keyword_index()
        self.loaded = True

        logger.info(f"Document index built: {len(self.document_chunks)} total chunks")

    def _process_togaf_pdf(self, pdf_path: Path) -> List[DocumentChunk]:
        """Process TOGAF concepts PDF using actual PDF parsing."""
        chunks = []

        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    text_content = page.extract_text()

                    if text_content.strip():
                        # Extract title from first line or use page-based title
                        lines = text_content.strip().split('\n')
                        title = lines[0] if lines else f"TOGAF Page {page_num + 1}"

                        chunk = DocumentChunk(
                            doc_id="togaf_concepts",
                            chunk_id=f"togaf_concepts_{page_num:03d}",
                            title=title[:100],  # Limit title length
                            content=text_content,
                            page_number=page_num + 1,
                            chunk_index=page_num,
                            keywords=self._extract_keywords(text_content),
                            doc_type="togaf_concepts"
                        )
                        chunks.append(chunk)

        except ImportError:
            logger.warning("PyPDF2 not available. Install with: pip install PyPDF2")
            return []
        except Exception as e:
            logger.warning(f"Failed to process TOGAF PDF {pdf_path}: {e}")
            return []

        logger.info(f"Extracted {len(chunks)} chunks from TOGAF PDF")
        return chunks

    def _process_research_pdf(self, pdf_path: Path) -> List[DocumentChunk]:
        """Process ArchiMate research PDF using actual PDF parsing."""
        chunks = []

        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    text_content = page.extract_text()

                    if text_content.strip():
                        # Extract title from first line or use page-based title
                        lines = text_content.strip().split('\n')
                        title = lines[0] if lines else f"Research Page {page_num + 1}"

                        chunk = DocumentChunk(
                            doc_id="archimate_research",
                            chunk_id=f"archimate_research_{page_num:03d}",
                            title=title[:100],  # Limit title length
                            content=text_content,
                            page_number=page_num + 1,
                            chunk_index=page_num,
                            keywords=self._extract_keywords(text_content),
                            doc_type="archimate_research"
                        )
                        chunks.append(chunk)

        except ImportError:
            logger.warning("PyPDF2 not available. Install with: pip install PyPDF2")
            return []
        except Exception as e:
            logger.warning(f"Failed to process research PDF {pdf_path}: {e}")
            return []

        logger.info(f"Extracted {len(chunks)} chunks from research PDF")
        return chunks

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for indexing."""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())

        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'with', 'this', 'that', 'can', 'will',
            'from', 'they', 'been', 'have', 'were', 'said', 'each', 'which',
            'their', 'time', 'but', 'all', 'any', 'may', 'use', 'her', 'him',
            'his', 'how', 'its', 'our', 'out', 'day', 'get', 'has', 'had',
            'way', 'too', 'old', 'see', 'now', 'man', 'two', 'new', 'who',
            'oil', 'sit', 'set', 'run', 'eat', 'far', 'sea', 'eye', 'ago'
        }

        keywords = [word for word in words if word not in stop_words]
        return list(set(keywords))  # Remove duplicates

    def _build_keyword_index(self) -> None:
        """Build keyword index for fast text search."""
        self.keyword_index = {}

        for chunk in self.document_chunks:
            for keyword in chunk.keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(chunk.chunk_id)

    def _save_index(self) -> None:
        """Save document index to cache file."""
        try:
            data = {
                'chunks': [chunk.to_dict() for chunk in self.document_chunks],
                'keyword_index': self.keyword_index,
                'version': '1.0',
                'created_at': str(Path().cwd())
            }

            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Document index saved to {self.cache_path}")

        except Exception as e:
            logger.error(f"Failed to save document index: {e}")

    def search_documents(self, query_terms: List[str], max_results: int = 5) -> List[DocumentChunk]:
        """
        Search documents using keyword matching.

        Args:
            query_terms: List of search terms
            max_results: Maximum number of results to return

        Returns:
            List of relevant document chunks
        """
        if not self.loaded:
            self.load_or_create_index()

        # Filter out stop words and clean punctuation (same as KG)
        stop_words = {'what', 'is', 'an', 'a', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'are', 'were', 'was', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}

        # Add domain-specific filtering for non-architecture terms
        non_architecture_terms = {'cat', 'dog', 'animal', 'car', 'house', 'food', 'weather', 'sports', 'music', 'movie', 'game', 'book', 'color', 'number'}

        filtered_terms = []
        for term in query_terms:
            # Clean punctuation and lowercase
            clean_term = term.lower().strip('.,!?;:()[]{}"\'-')

            # Skip stop words, very short terms, and non-architecture terms
            if (clean_term not in stop_words and
                clean_term not in non_architecture_terms and
                len(clean_term) > 2):

                # Additional check: must contain architecture-related patterns
                architecture_patterns = ['arch', 'togaf', 'business', 'application', 'technology', 'model', 'process', 'service', 'component', 'layer', 'capability', 'system', 'data', 'infrastructure', 'governance', 'enterprise', 'framework', 'method', 'pattern', 'view', 'element']

                # Allow terms that match architecture patterns OR are longer technical terms
                if (any(pattern in clean_term for pattern in architecture_patterns) or
                    len(clean_term) > 6):  # Longer terms more likely to be technical
                    filtered_terms.append(clean_term)

        logger.info(f"PDF search filtered terms from {query_terms} to {filtered_terms}")

        if not filtered_terms:
            logger.info("No architecture-relevant terms found after filtering")
            return []

        # Score chunks based on keyword matches
        chunk_scores: Dict[str, float] = {}

        for term in filtered_terms:
            term_lower = term.lower()

            # Direct keyword matches
            if term_lower in self.keyword_index:
                for chunk_id in self.keyword_index[term_lower]:
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + 1.0

            # Partial keyword matches
            for keyword in self.keyword_index:
                if term_lower in keyword or keyword in term_lower:
                    for chunk_id in self.keyword_index[keyword]:
                        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + 0.5

        # Apply relevance threshold to filter out irrelevant results
        MIN_RELEVANCE_SCORE = 1.0  # Require at least one exact keyword match

        # Filter chunks by minimum relevance score
        relevant_chunks = [(chunk_id, score) for chunk_id, score in chunk_scores.items()
                          if score >= MIN_RELEVANCE_SCORE]

        if not relevant_chunks:
            logger.info(f"No relevant documents found for query terms: {query_terms} (scores below {MIN_RELEVANCE_SCORE})")
            return []  # Return empty for irrelevant queries

        # Get top chunks by score
        sorted_chunks = sorted(relevant_chunks, key=lambda x: x[1], reverse=True)
        top_chunk_ids = [chunk_id for chunk_id, _ in sorted_chunks[:max_results]]

        # Return corresponding chunks
        results = []
        chunk_lookup = {chunk.chunk_id: chunk for chunk in self.document_chunks}

        for chunk_id in top_chunk_ids:
            if chunk_id in chunk_lookup:
                results.append(chunk_lookup[chunk_id])

        logger.info(f"Document search for {query_terms}: {len(results)} relevant results (filtered from {len(chunk_scores)} total)")
        return results

    def get_document_stats(self) -> Dict:
        """Get statistics about the document index."""
        if not self.loaded:
            return {"error": "Index not loaded"}

        togaf_chunks = [c for c in self.document_chunks if c.doc_type == "togaf_concepts"]
        research_chunks = [c for c in self.document_chunks if c.doc_type == "archimate_research"]

        return {
            "total_chunks": len(self.document_chunks),
            "togaf_chunks": len(togaf_chunks),
            "research_chunks": len(research_chunks),
            "total_keywords": len(self.keyword_index),
            "documents_indexed": len(set(c.doc_id for c in self.document_chunks))
        }