"""
Citation Validator - Bidirectional citation authenticity validation.

This module implements the critical fix for fake citation prevention by validating
that ALL citations in responses actually exist in knowledge sources (KG, ArchiMate, TOGAF).

CRITICAL REQUIREMENT:
- Citations must exist in knowledge sources, not just match format patterns
- Zero tolerance for hallucinated citations like "archi:id-cap-001", "iec:GridCongestion"
- Builds citation pool from retrieval context for LLM to use
"""

import logging
import re
from typing import Dict, List, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from src.knowledge.kg_loader import KnowledgeGraphLoader
    from src.archimate.parser import ArchiMateParser
    from src.documents.pdf_indexer import PDFIndexer

from src.utils.trace import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer()


@dataclass
class CitationMetadata:
    """Metadata for a validated citation."""
    citation_id: str
    source_type: str  # "knowledge_graph", "archimate", "togaf_doc"
    label: Optional[str] = None
    definition: Optional[str] = None
    namespace: Optional[str] = None
    exists: bool = False


class CitationValidator:
    """
    Validates citation authenticity against knowledge sources.
    
    This is the critical component that prevents fake citation generation by:
    1. Building a pool of valid citations from retrieval context
    2. Validating each citation in responses exists in knowledge sources
    3. Rejecting responses with any fabricated citations
    """
    
    def __init__(
        self,
        kg_loader: 'KnowledgeGraphLoader' = None,
        archimate_parser: 'ArchiMateParser' = None,
        pdf_indexer: 'PDFIndexer' = None
    ):
        """
        Initialize citation validator with knowledge sources.
        
        Args:
            kg_loader: Knowledge graph loader for SKOS/IEC/ENTSOE citations
            archimate_parser: ArchiMate parser for archi:id-* citations
            pdf_indexer: PDF indexer for TOGAF document citations
        """
        self.kg_loader = kg_loader
        self.archimate_parser = archimate_parser
        self.pdf_indexer = pdf_indexer
        
        # Cache for performance
        self._validation_cache: Dict[str, bool] = {}
        self._metadata_cache: Dict[str, CitationMetadata] = {}
        
        logger.info("CitationValidator initialized - authenticity validation active")
    
    def validate_citation_exists(self, citation: str, trace_id: str = None) -> bool:
        """
        Check if citation exists in any knowledge source.
        
        This is the core validation method that prevents fake citations.
        
        Args:
            citation: Citation ID to validate (e.g., "skos:Asset", "archi:id-123")
            trace_id: Optional trace ID for logging
            
        Returns:
            True if citation exists in knowledge sources, False otherwise
        """
        # Check cache first
        if citation in self._validation_cache:
            if trace_id:
                tracer.trace_info(trace_id, "citation_validator", "cache_hit",
                                citation=citation, exists=self._validation_cache[citation])
            return self._validation_cache[citation]
        
        exists = False
        
        # Determine namespace and validate accordingly
        if citation.startswith("skos:") or citation.startswith("iec:") or \
           citation.startswith("entsoe:") or citation.startswith("lido:") or \
           citation.startswith("eurlex:"):
            # Validate against knowledge graph
            exists = self._validate_kg_citation(citation, trace_id)
            
        elif citation.startswith("archi:id-"):
            # Validate against ArchiMate models
            exists = self._validate_archimate_citation(citation, trace_id)
            
        elif citation.startswith("togaf:") or citation.startswith("archimate:research:") or \
             citation.startswith("doc:"):
            # Validate against document index
            exists = self._validate_document_citation(citation, trace_id)
            
        elif citation.startswith("external:"):
            # External citations are always considered valid (with caution)
            exists = True
            logger.warning(f"External citation allowed: {citation}")
        
        else:
            logger.warning(f"Unknown citation namespace: {citation}")
            exists = False
        
        # Cache result
        self._validation_cache[citation] = exists
        
        if trace_id:
            tracer.trace_info(trace_id, "citation_validator", "validation_complete",
                            citation=citation, exists=exists)
        
        return exists
    
    def _validate_kg_citation(self, citation: str, trace_id: str = None) -> bool:
        """Validate citation exists in knowledge graph."""
        if not self.kg_loader:
            logger.warning("KG loader not available, cannot validate citation")
            return False
        
        try:
            exists = self.kg_loader.citation_exists(citation)
            logger.debug(f"KG citation validation: {citation} -> {exists}")
            return exists
        except Exception as e:
            logger.error(f"KG citation validation failed for {citation}: {e}")
            return False
    
    def _validate_archimate_citation(self, citation: str, trace_id: str = None) -> bool:
        """Validate citation exists in ArchiMate models."""
        if not self.archimate_parser:
            logger.warning("ArchiMate parser not available, cannot validate citation")
            return False
        
        try:
            exists = self.archimate_parser.citation_exists(citation)
            logger.debug(f"ArchiMate citation validation: {citation} -> {exists}")
            return exists
        except Exception as e:
            logger.error(f"ArchiMate citation validation failed for {citation}: {e}")
            return False
    
    def _validate_document_citation(self, citation: str, trace_id: str = None) -> bool:
        """Validate citation exists in document index."""
        if not self.pdf_indexer:
            logger.warning("PDF indexer not available, cannot validate citation")
            return False
        
        try:
            # Extract document ID from citation
            # Format: togaf:concepts:001, archimate:research:005, doc:togaf:001
            exists = self.pdf_indexer.citation_exists(citation)
            logger.debug(f"Document citation validation: {citation} -> {exists}")
            return exists
        except Exception as e:
            logger.error(f"Document citation validation failed for {citation}: {e}")
            return False
    
    def get_citation_pool(self, retrieval_context: Dict, trace_id: str = None) -> List[str]:
        """
        Extract all valid citations from retrieval context.
        
        This builds the pool of citations that the LLM is allowed to use.
        Note: Only citations that actually exist in knowledge sources.
        
        Args:
            retrieval_context: Dictionary containing retrieval results
            trace_id: Optional trace ID for logging
            
        Returns:
            List of valid citation IDs
        """
        if trace_id:
            tracer.trace_info(trace_id, "citation_validator", "pool_extraction_start",
                            context_keys=list(retrieval_context.keys()))
        
        citation_pool: Set[str] = set()
        
        # Extract from KG results
        if "kg_results" in retrieval_context:
            for result in retrieval_context["kg_results"]:
                if isinstance(result, dict) and "citation_id" in result:
                    citation_pool.add(result["citation_id"])
        
        # Extract from ArchiMate elements
        if "archimate_elements" in retrieval_context:
            for element in retrieval_context["archimate_elements"]:
                if isinstance(element, dict) and "id" in element:
                    citation_pool.add(f"archi:id-{element['id']}")
        
        # Extract from TOGAF documents
        if "togaf_docs" in retrieval_context:
            for doc in retrieval_context["togaf_docs"]:
                if isinstance(doc, dict) and "citation_id" in doc:
                    citation_pool.add(doc["citation_id"])
        
        # Extract from document chunks
        if "document_chunks" in retrieval_context:
            for chunk in retrieval_context["document_chunks"]:
                if isinstance(chunk, dict) and "doc_id" in chunk:
                    citation_pool.add(f"doc:{chunk['doc_id']}")
        
        # Convert to sorted list
        citation_pool_list = sorted(list(citation_pool))
        
        logger.info(f"Citation pool extracted: {len(citation_pool_list)} valid citations")
        
        if trace_id:
            tracer.trace_info(trace_id, "citation_validator", "pool_extraction_complete",
                            pool_size=len(citation_pool_list),
                            sample_citations=citation_pool_list[:10])
        
        return citation_pool_list
    
    def validate_response_citations(
        self,
        response: str,
        citation_pool: List[str],
        trace_id: str = None
    ) -> Dict:
        """
        Validate ALL citations in response exist in citation pool.
        
        This is the final enforcement gate that rejects responses with fake citations.
        
        Args:
            response: Generated response text
            citation_pool: List of valid citations from retrieval context
            trace_id: Optional trace ID for logging
            
        Returns:
            Dictionary with validation results:
            - valid: Boolean indicating if all citations are valid
            - fake_citations: List of fake citations found
            - valid_citations: List of valid citations found
            - message: Human-readable validation message
        """
        if trace_id:
            tracer.trace_info(trace_id, "citation_validator", "response_validation_start",
                            pool_size=len(citation_pool))
        
        # Extract all citations from response
        citation_patterns = {
            r"archi:id-[a-zA-Z0-9\-_]+",
            r"skos:[a-zA-Z0-9\-_]+",
            r"iec:[a-zA-Z0-9\-_\.]+",
            r"togaf:adm:[a-zA-Z0-9\-_]+",
            r"togaf:concepts:[0-9]{3}",
            r"archimate:research:[0-9]{3}",
            r"entsoe:[a-zA-Z0-9\-_]+",
            r"lido:[a-zA-Z0-9\-_]+",
            r"doc:[a-zA-Z0-9\-_]+:[0-9]{3}",
            r"external:[a-zA-Z0-9\-_]+:[a-zA-Z0-9\-_]+"
        }
        
        found_citations: Set[str] = set()
        for pattern in citation_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            found_citations.update(matches)
        
        # Classify citations
        valid_citations = []
        fake_citations = []
        
        citation_pool_set = set(citation_pool)
        
        for citation in found_citations:
            if citation in citation_pool_set:
                valid_citations.append(citation)
            else:
                # Double-check with existence validation
                if self.validate_citation_exists(citation, trace_id):
                    valid_citations.append(citation)
                    logger.warning(f"Citation not in pool but exists: {citation}")
                else:
                    fake_citations.append(citation)
        
        # Determine validation result
        is_valid = len(fake_citations) == 0
        
        result = {
            "valid": is_valid,
            "fake_citations": fake_citations,
            "valid_citations": valid_citations,
            "total_citations": len(found_citations),
            "message": self._create_validation_message(is_valid, fake_citations, valid_citations)
        }
        
        if trace_id:
            tracer.trace_info(trace_id, "citation_validator", "response_validation_complete",
                            valid=is_valid, fake_count=len(fake_citations),
                            valid_count=len(valid_citations))
        
        if not is_valid:
            logger.error(f"FAKE CITATIONS DETECTED: {fake_citations}")
        
        return result
    
    def _create_validation_message(
        self,
        is_valid: bool,
        fake_citations: List[str],
        valid_citations: List[str]
    ) -> str:
        """Create human-readable validation message."""
        if is_valid:
            if valid_citations:
                return f"✅ All {len(valid_citations)} citations validated successfully"
            else:
                return "⚠️  No citations found in response"
        else:
            return (f"❌ VALIDATION FAILED: {len(fake_citations)} fake citation(s) detected: "
                   f"{', '.join(fake_citations[:5])}")
    
    def get_citation_metadata(self, citation: str, trace_id: str = None) -> Optional[CitationMetadata]:
        """
        Get metadata for a citation from knowledge sources.
        
        Args:
            citation: Citation ID
            trace_id: Optional trace ID for logging
            
        Returns:
            CitationMetadata object if found, None otherwise
        """
        # Check cache
        if citation in self._metadata_cache:
            return self._metadata_cache[citation]
        
        metadata = None
        
        # Fetch from appropriate source
        if citation.startswith("skos:") or citation.startswith("iec:"):
            metadata = self._get_kg_metadata(citation)
        elif citation.startswith("archi:id-"):
            metadata = self._get_archimate_metadata(citation)
        elif citation.startswith("togaf:") or citation.startswith("doc:"):
            metadata = self._get_document_metadata(citation)
        
        # Cache result
        if metadata:
            self._metadata_cache[citation] = metadata
        
        return metadata
    
    def _get_kg_metadata(self, citation: str) -> Optional[CitationMetadata]:
        """Get metadata from knowledge graph."""
        if not self.kg_loader:
            return None
        
        try:
            metadata_dict = self.kg_loader.get_citation_metadata(citation)
            if metadata_dict:
                return CitationMetadata(
                    citation_id=citation,
                    source_type="knowledge_graph",
                    label=metadata_dict.get("label"),
                    definition=metadata_dict.get("definition"),
                    namespace=citation.split(":")[0],
                    exists=True
                )
        except Exception as e:
            logger.error(f"Failed to get KG metadata for {citation}: {e}")
        
        return None
    
    def _get_archimate_metadata(self, citation: str) -> Optional[CitationMetadata]:
        """Get metadata from ArchiMate models."""
        if not self.archimate_parser:
            return None
        
        try:
            element = self.archimate_parser.get_element_by_citation(citation)
            if element:
                return CitationMetadata(
                    citation_id=citation,
                    source_type="archimate",
                    label=element.name,
                    definition=element.documentation,
                    namespace="archi",
                    exists=True
                )
        except Exception as e:
            logger.error(f"Failed to get ArchiMate metadata for {citation}: {e}")
        
        return None
    
    def _get_document_metadata(self, citation: str) -> Optional[CitationMetadata]:
        """Get metadata from document index."""
        if not self.pdf_indexer:
            return None
        
        try:
            # Implementation depends on PDF indexer structure
            return CitationMetadata(
                citation_id=citation,
                source_type="togaf_doc",
                namespace=citation.split(":")[0],
                exists=True
            )
        except Exception as e:
            logger.error(f"Failed to get document metadata for {citation}: {e}")
        
        return None
    
    def clear_cache(self):
        """Clear validation and metadata caches."""
        self._validation_cache.clear()
        self._metadata_cache.clear()
        logger.info("Citation validator caches cleared")