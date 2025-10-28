"""
Configuration Constants for AInstein EA Assistant

IMPORTANT: These values are empirically derived from prototype testing.
They should be validated with production data and adjusted accordingly.

Version: 1.0.0
Last Updated: 2025-10-14
Review Frequency: Quarterly or after major data updates
Production Validated: NO - Prototype values only
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

# ============================================================================
# VERSION METADATA
# ============================================================================

CONFIG_VERSION = "1.0.0"
CONFIG_LAST_UPDATED = "2025-10-14"
CALIBRATION_DATASET_SIZE = 50  # Small prototype dataset
PRODUCTION_VALIDATED = False

# ============================================================================
# CONFIDENCE SCORING CONSTANTS
# ============================================================================
# NOTE: These thresholds determine when human review is required.
# Current values based on prototype testing with ~50 queries.
# REQUIRES PRODUCTION VALIDATION with minimum 1000 queries.

@dataclass
class ConfidenceThresholds:
    """
    Confidence score thresholds for quality assessment.
    
    Usage in code:
        from src.config.constants import CONFIDENCE
        if score >= CONFIDENCE.HIGH_CONFIDENCE_THRESHOLD:
            # No review needed
    
    Calibration Notes:
        - Current thresholds are ESTIMATES from prototype
        - Need precision/recall analysis on production data
        - Monitor false positive/negative rates
        - Adjust based on business requirements (quality vs speed)
    """
    
    # Knowledge Graph confidence scores
    # Based on: Structured data with full definitions = highest quality
    KG_WITH_DEFINITION: float = 0.95
    KG_WITHOUT_DEFINITION: float = 0.75
    
    # Other source confidence scores
    # Based on: Source type and information completeness
    TOGAF_DOCUMENTATION: float = 0.85
    ARCHIMATE_ELEMENTS: float = 0.75
    DOCUMENT_CHUNKS: float = 0.70
    SEMANTIC_FALLBACK: float = 0.50
    
    # Confidence bonuses for match quality
    # Based on: Query term matching in results
    EXACT_TERM_MATCH_BONUS: float = 0.10
    PARTIAL_TERM_MATCH_BONUS: float = 0.05
    
    # Review requirement thresholds
    # Based on: Desired quality vs throughput tradeoff
    HIGH_CONFIDENCE_THRESHOLD: float = 0.75  # Above: no review needed
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.50  # Above: optional review
    LOW_CONFIDENCE_THRESHOLD: float = 0.50  # Below: mandatory review
    
    def __post_init__(self):
        """Validate thresholds are within valid ranges."""
        for field_name, value in self.__dict__.items():
            if isinstance(value, float):
                if not 0 <= value <= 1:
                    raise ValueError(f"Confidence value {field_name}={value} outside range [0,1]")

# Global instance
CONFIDENCE = ConfidenceThresholds()

# ============================================================================
# SEMANTIC ENHANCEMENT CONSTANTS
# ============================================================================
# NOTE: These control semantic search quality and quantity.
# Values tuned for all-MiniLM-L6-v2 embedding model.
# REQUIRES RE-CALIBRATION if changing embedding model.

@dataclass
class SemanticEnhancementConfig:
    """
    Configuration for semantic search enhancement.
    
    Usage in code:
        from src.config.constants import SEMANTIC_CONFIG
        results = embedding_agent.semantic_search(
            query, 
            top_k=SEMANTIC_CONFIG.TOP_K_PRIMARY,
            min_score=SEMANTIC_CONFIG.MIN_SCORE_PRIMARY
        )
    
    Calibration Notes:
        - Thresholds based on cosine similarity (0-1 range)
        - Current values optimized for all-MiniLM-L6-v2
        - Higher min_score = higher precision, lower recall
        - Tune based on precision@k and recall@k metrics
    """
    
    # Minimum similarity scores (cosine similarity 0-1)
    # Based on: Empirical testing to balance precision/recall
    MIN_SCORE_PRIMARY: float = 0.40  # Main semantic search threshold
    MIN_SCORE_CONTEXT: float = 0.45  # Higher threshold for context (reduce noise)
    MIN_SCORE_COMPARISON: float = 0.45  # Higher threshold for comparison queries
    
    # Result limits
    # Based on: Avoid overwhelming LLM while providing context
    TOP_K_PRIMARY: int = 5  # Max results for main semantic search
    TOP_K_CONTEXT: int = 2  # Max results for context expansion (reduce noise)
    TOP_K_COMPARISON: int = 3  # Max results per comparison term
    
    # Candidate limits (post-filtering)
    # Based on: Balance between context richness and LLM token limits
    MAX_SEMANTIC_CANDIDATES: int = 3  # Limit semantic enhancements
    
    # Feature flag
    ENABLED_BY_DEFAULT: bool = True  # Can be disabled via env var
    
    # Embedding model metadata
    # NOTE: When use_openai=True, text-embedding-3-small (1536-dim) is used
    # This fallback is for local-only deployments
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # ✅ Updated default
    EMBEDDING_DIMENSION: int = 1536  # ✅ Updated for OpenAI model
    FALLBACK_MODEL: str = "all-MiniLM-L6-v2"  # Local fallback
    FALLBACK_DIMENSION: int = 384  # Local fallback dimensio
    
    def __post_init__(self):
        """Validate configuration values."""
        assert 0 <= self.MIN_SCORE_PRIMARY <= 1, "MIN_SCORE_PRIMARY must be in [0,1]"
        assert 0 <= self.MIN_SCORE_CONTEXT <= 1, "MIN_SCORE_CONTEXT must be in [0,1]"
        assert self.TOP_K_PRIMARY > 0, "TOP_K_PRIMARY must be positive"
        assert self.MAX_SEMANTIC_CANDIDATES > 0, "MAX_SEMANTIC_CANDIDATES must be positive"

# Global instance
SEMANTIC_CONFIG = SemanticEnhancementConfig()

# ============================================================================
# RANKING AND PRIORITIZATION CONSTANTS
# ============================================================================
# NOTE: These determine how candidates are ranked and selected.
# Scoring system designed to prioritize: structured > semantic > context

@dataclass
class RankingConfig:
    """
    Configuration for candidate ranking and prioritization.
    
    Usage in code:
        from src.config.constants import RANKING_CONFIG
        priority_score = RANKING_CONFIG.PRIORITY_SCORE_DEFINITION
    
    Calibration Notes:
        - Priority scores are RELATIVE weights (not absolute)
        - Current scale (100/80/60) is arbitrary
        - Should validate with ranking metrics (MRR, NDCG)
        - Adjust based on which sources users actually prefer
    """
    
    # Base priority scores (higher = more important)
    # Based on: Assumed reliability of different sources
    PRIORITY_SCORE_DEFINITION: int = 100  # Highest: KG with full definition
    PRIORITY_SCORE_NORMAL: int = 80  # Standard: KG, ArchiMate, structured
    PRIORITY_SCORE_CONTEXT: int = 60  # Lower: conversation context, semantic
    PRIORITY_SCORE_FALLBACK: int = 50  # Lowest: unknown source
    
    # Confidence multiplier
    # Based on: Amplify differences between high/low confidence
    CONFIDENCE_BONUS_MULTIPLIER: int = 20  # Multiply confidence by this
    CONFIDENCE_BONUS_THRESHOLD: float = 0.5  # Only apply bonus above this
    
    # Result limits
    # Based on: LLM context window and quality considerations
    MAX_TOTAL_CANDIDATES: int = 10  # Limit to top 10 to avoid LLM overwhelm
    MIN_STRUCTURED_RESULTS: int = 3  # Threshold for triggering semantic fallback
    
    def __post_init__(self):
        """Validate ranking configuration."""
        assert self.PRIORITY_SCORE_DEFINITION > self.PRIORITY_SCORE_NORMAL
        assert self.PRIORITY_SCORE_NORMAL > self.PRIORITY_SCORE_CONTEXT
        assert self.MAX_TOTAL_CANDIDATES > 0
        assert 0 <= self.CONFIDENCE_BONUS_THRESHOLD <= 1

# Global instance
RANKING_CONFIG = RankingConfig()

# ============================================================================
# CONTEXT EXPANSION CONSTANTS
# ============================================================================
# NOTE: Controls conversation memory and follow-up handling.

@dataclass
class ContextExpansionConfig:
    """
    Configuration for conversation context expansion.
    
    Usage in code:
        from src.config.constants import CONTEXT_CONFIG
        history = session_manager.get_history(
            max_turns=CONTEXT_CONFIG.MAX_HISTORY_TURNS
        )
    
    Calibration Notes:
        - History depth affects conversation coherence
        - More history = better context but more noise
        - Current values based on informal testing
        - Should monitor conversation quality metrics
    """
    
    # History depth
    # Based on: Balance between context and noise
    MAX_HISTORY_TURNS: int = 3  # Look back at last N conversation turns
    MAX_CONCEPTS_FROM_HISTORY: int = 5  # Max concepts to extract from history
    
    # Query term filtering
    # Based on: Meaningful term length for energy domain
    MIN_TERM_LENGTH: int = 4  # Ignore terms shorter than this
    
    # Follow-up detection
    # Based on: Similarity threshold for detecting follow-ups
    FOLLOWUP_SIMILARITY_THRESHOLD: float = 0.6  # Cosine similarity threshold
    
    def __post_init__(self):
        """Validate context configuration."""
        assert self.MAX_HISTORY_TURNS > 0
        assert self.MIN_TERM_LENGTH > 0
        assert 0 <= self.FOLLOWUP_SIMILARITY_THRESHOLD <= 1

# Global instance
CONTEXT_CONFIG = ContextExpansionConfig()

# ============================================================================
# CITATION EXTRACTION CONSTANTS
# ============================================================================
# NOTE: Stop words and patterns for citation term cleaning.

# Stop words for comparison term cleaning
# Based on: Common English words that don't contribute to meaning
COMPARISON_TERM_STOP_WORDS: List[str] = [
    'the', 'a', 'an', 'is', 'are', 'what', 'which', 
    'difference', 'in', 'of', 'to', 'from', 'with'
]

# Stop words for general query term extraction
# Based on: Common English query patterns
QUERY_TERM_STOP_WORDS: Set[str] = {
    'what', 'is', 'the', 'a', 'an', 'of', 'for', 'in', 'on', 'at',
    'to', 'how', 'why', 'when', 'where', 'are', 'do', 'does'
}

ANALYSIS_PATTERNS = {
    'decision_drivers': {
        'keywords': ['driver', 'why', 'reason', 'rationale', 'motivation', 'cause'],
        'patterns': [r'why.*decision', r'what.*drove', r'reason.*for'],
        'weight': 1.0
    },
    'patterns': {
        'keywords': ['pattern', 'trend', 'common', 'recurring', 'similarity'],
        'patterns': [r'common.*across', r'pattern.*in'],
        'weight': 0.9
    },
    # Easy to add new types without touching code
}

# Citation pattern prefixes
# ============================================================================
# CITATION PREFIXES - Complete List
# ============================================================================
# All valid citation prefixes used across the knowledge base.
# These are checked during grounding validation to ensure responses
# are properly cited from authoritative sources.
#
# Last Updated: 2025-10-20 (expanded from prototype)
# Citation Counts: 3,970 total (skos: 3249, eurlex: 562, entsoe: 58, iec: 19, archi: 82)

# Primary citation prefixes (required for grounding validation)
REQUIRED_CITATION_PREFIXES: List[str] = [
    # ─────────────────────────────────────────────────────────────────────
    # ALLIANDER VOCABULARIES (3,249+ citations)
    # ─────────────────────────────────────────────────────────────────────
    "skos:",              # Alliander SKOS Vocabulary (PRIMARY - 3,249 citations)
    "aiontology:",        # Alliander Artificial Intelligence Ontology
    "modulair:",          # Alliander Modulair Station Building Blocks
    "msb:",              # Modulair Station Building (alternative prefix)
    
    # ─────────────────────────────────────────────────────────────────────
    # IEC STANDARDS - International Electrotechnical Commission (19+ citations)
    # ─────────────────────────────────────────────────────────────────────
    "iec:",              # Generic IEC (19 citations)
    "iec61968:",         # IEC 61968 - Meters, Assets and Work
    "iec61970:",         # IEC 61970 - Common Information Model (CIM/CGMES)
    "iec62325:",         # IEC 62325 - Market Model
    "iec62746:",         # IEC 62746 - Demand Site Resource
    
    # ─────────────────────────────────────────────────────────────────────
    # EUROPEAN STANDARDS & REGULATION (620+ citations)
    # ─────────────────────────────────────────────────────────────────────
    "eurlex:",           # EUR-Lex Regulation (PRIMARY - 562 citations) ⚠️ CRITICAL
    "acer:",             # EUR Energy Regulators (ACER)
    "entsoe:",           # Harmonized Electricity Market Role Model (ENTSO-E - 58 citations)
    
    # ─────────────────────────────────────────────────────────────────────
    # DUTCH GOVERNMENT & REGULATION
    # ─────────────────────────────────────────────────────────────────────
    "dutch:",            # Dutch Regulation Electricity
    "lido:",             # Linked Data Overheid (Dutch Government - 0 citations currently)
    
    # ─────────────────────────────────────────────────────────────────────
    # BRITISH STANDARDS
    # ─────────────────────────────────────────────────────────────────────
    "pas1879:",          # PAS1879 - Energy smart appliances (BSI)
    "bsi:",              # British Standards Institution (alternative prefix)
    
    # ─────────────────────────────────────────────────────────────────────
    # ARCHITECTURE & METHODOLOGY (82+ citations)
    # ─────────────────────────────────────────────────────────────────────
    "archi:",            # ArchiMate Model elements (82 citations)
    "archi:id-",         # ArchiMate element IDs (specific format)
    "togaf:adm:",        # TOGAF Architecture Development Method
    "togaf:concepts:",   # TOGAF Concepts (format: togaf:concepts:001)
    "archimate:research:", # ArchiMate Research (format: archimate:research:001)
    
    # ─────────────────────────────────────────────────────────────────────
    # DOCUMENTS & EXTERNAL SOURCES
    # ─────────────────────────────────────────────────────────────────────
    "doc:",              # PDF Documents (format: doc:filename:page123)
    "adr:",              # ArchiMate Decision Records (format: adr:record-name)
    "external:",         # External sources (format: external:source:identifier)
    
    # ─────────────────────────────────────────────────────────────────────
    # LEGACY (Deprecated but still in knowledge base)
    # ─────────────────────────────────────────────────────────────────────
    "confluence:",       # Alliander Confluence - Glossary (Out of Date)
    "poolparty:",        # Alliander Poolparty - Glossary (Out of Date)
]

# ============================================================================
# CITATION PREFIX CATEGORIES (for reporting and validation)
# ============================================================================

CITATION_PREFIX_CATEGORIES: Dict[str, List[str]] = {
    "alliander": ["skos:", "aiontology:", "modulair:", "msb:"],
    "iec_standards": ["iec:", "iec61968:", "iec61970:", "iec62325:", "iec62746:"],
    "european_regulation": ["eurlex:", "acer:", "entsoe:"],
    "dutch_regulation": ["dutch:", "lido:"],
    "british_standards": ["pas1879:", "bsi:"],
    "architecture": ["archi:", "archi:id-", "togaf:adm:", "togaf:concepts:", "archimate:research:", "adr:"],
    "documents": ["doc:", "external:"],
    "legacy": ["confluence:", "poolparty:"]
}

# ============================================================================
# HIGH-PRIORITY PREFIXES (for ranking/sorting)
# ============================================================================
# Prefixes that should be prioritized in responses (most authoritative)

HIGH_PRIORITY_PREFIXES: List[str] = [
    "skos:",        # Alliander primary vocabulary (3,249 citations)
    "eurlex:",      # EU Regulation (562 citations)
    "iec61968:",    # IEC Meters & Assets
    "iec61970:",    # IEC CIM/CGMES
    "entsoe:",      # European grid standards (58 citations)
    "archi:",       # ArchiMate models (82 citations)
]

# ============================================================================
# CITATION VALIDATION HELPERS
# ============================================================================

def is_valid_citation_prefix(citation: str) -> bool:
    """
    Check if citation has a valid prefix.
    
    Args:
        citation: Citation string to validate (e.g., "eurlex:631-28")
        
    Returns:
        True if citation starts with a valid prefix
        
    Example:
        >>> is_valid_citation_prefix("eurlex:631-28")
        True
        >>> is_valid_citation_prefix("fake:123")
        False
    """
    if not citation or not isinstance(citation, str):
        return False
    
    citation_lower = citation.lower()
    return any(citation_lower.startswith(prefix.lower()) 
               for prefix in REQUIRED_CITATION_PREFIXES)
    
def _classify_analysis_type(self, query: str) -> str:
    """
    Rule-based classification using external configuration.
    Easily extensible via config changes.
    """
    query_lower = query.lower()
    scores = {}
    
    for analysis_type, config in ANALYSIS_PATTERNS.items():
        score = 0.0
        
        # Check keywords
        for keyword in config.get('keywords', []):
            if keyword in query_lower:
                score += config.get('weight', 1.0)
        
        # Check regex patterns
        for pattern in config.get('patterns', []):
            if re.search(pattern, query_lower):
                score += config.get('weight', 1.0) * 1.5
        
        scores[analysis_type] = score
    
    # Return highest scoring type, or 'general' if no matches
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0] if max(scores.values()) > 0 else 'general'
    return 'general'

def get_citation_category(citation: str) -> str:
    """
    Get category for a citation prefix.
    
    Args:
        citation: Citation string (e.g., "iec61968:Asset")
        
    Returns:
        Category name or "unknown" if not found
        
    Example:
        >>> get_citation_category("iec61968:Asset")
        'iec_standards'
        >>> get_citation_category("eurlex:631-28")
        'european_regulation'
    """
    if not citation or not isinstance(citation, str):
        return "unknown"
    
    citation_lower = citation.lower()
    for category, prefixes in CITATION_PREFIX_CATEGORIES.items():
        if any(citation_lower.startswith(prefix.lower()) for prefix in prefixes):
            return category
    
    return "unknown"

def is_high_priority_citation(citation: str) -> bool:
    """
    Check if citation is from a high-priority source.
    
    High-priority sources are the most authoritative and should be
    preferred when multiple sources are available.
    
    Args:
        citation: Citation string (e.g., "skos:220")
        
    Returns:
        True if citation is from a high-priority source
        
    Example:
        >>> is_high_priority_citation("skos:220")
        True
        >>> is_high_priority_citation("poolparty:x")
        False
    """
    if not citation or not isinstance(citation, str):
        return False
    
    citation_lower = citation.lower()
    return any(citation_lower.startswith(prefix.lower()) 
               for prefix in HIGH_PRIORITY_PREFIXES)

def get_citation_source_name(citation: str) -> str:
    """
    Get human-readable source name for a citation.
    
    Args:
        citation: Citation string (e.g., "eurlex:631-28")
        
    Returns:
        Human-readable source name
        
    Example:
        >>> get_citation_source_name("eurlex:631-28")
        'EUR-Lex Regulation'
        >>> get_citation_source_name("skos:220")
        'Alliander SKOS Vocabulary'
    """
    if not citation or not isinstance(citation, str):
        return "Unknown Source"
    
    citation_lower = citation.lower()
    
    # Map prefixes to human-readable names
    source_names = {
        "skos:": "Alliander SKOS Vocabulary",
        "eurlex:": "EUR-Lex Regulation",
        "entsoe:": "ENTSO-E Harmonized Market Model",
        "iec61968:": "IEC 61968 - Meters, Assets and Work",
        "iec61970:": "IEC 61970 - Common Information Model",
        "iec62325:": "IEC 62325 - Market Model",
        "iec62746:": "IEC 62746 - Demand Site Resource",
        "iec:": "IEC Standards",
        "archi:": "ArchiMate Model",
        "togaf:adm:": "TOGAF ADM",
        "dutch:": "Dutch Regulation",
        "acer:": "ACER (EU Energy Regulators)",
        "pas1879:": "PAS1879 - Energy Smart Appliances",
        "doc:": "PDF Document",
        "confluence:": "Alliander Confluence (Legacy)",
        "poolparty:": "Alliander Poolparty (Legacy)",
    }
    
    for prefix, name in source_names.items():
        if citation_lower.startswith(prefix):
            return name
    
    return "Unknown Source"

# ============================================================================
# CITATION STATISTICS
# ============================================================================
# NOTE: Based on system initialization logs (2025-10-20)

CITATION_STATISTICS: Dict[str, int] = {
    "skos": 3249,        # Alliander SKOS Vocabulary
    "eurlex": 562,       # EUR-Lex Regulation
    "entsoe": 58,        # ENTSO-E Market Model
    "iec": 19,           # IEC Generic
    "archi": 82,         # ArchiMate Models
    "lido": 0,           # Dutch Government (empty)
    "total": 3970        # Total citations
}

# ============================================================================
# PERFORMANCE CONSTANTS
# ============================================================================

@dataclass
class PerformanceConfig:
    """Performance-related constants and timeouts."""
    
    # Timeouts (milliseconds)
    SEMANTIC_SEARCH_TIMEOUT_MS: int = 200  # Max time for semantic search
    KG_QUERY_TIMEOUT_S: int = 10  # Max time for SPARQL query
    
    # Logging precision
    TIME_PRECISION_DECIMALS: int = 2  # Decimal places for timing logs
    
    # Cache sizes
    KG_QUERY_CACHE_SIZE: int = 1000  # Max cached SPARQL queries
    CITATION_METADATA_CACHE_SIZE: int = 10000  # Max cached citation metadata

# Global instance
PERFORMANCE_CONFIG = PerformanceConfig()

# ============================================================================
# LANGUAGE DETECTION CONSTANTS
# ============================================================================
# NOTE: Currently supports English and Dutch only.
# LIMITATION: Hardcoded for these languages, needs generalization.

@dataclass
class LanguageDetectionConfig:
    """
    Language detection configuration.
    
    KNOWN LIMITATIONS:
        - Only supports English and Dutch
        - Heuristics are language-specific
        - Needs proper i18n framework for production
    """
    
    # LLM parameters (if using LLM for detection)
    LLM_TEMPERATURE: float = 0.1  # Low temperature for deterministic detection
    LLM_MAX_TOKENS: int = 10  # Very short response needed
    
    # Heuristic thresholds (fallback if LLM unavailable)
    # WARNING: These are ARBITRARY and LANGUAGE-SPECIFIC
    DUTCH_CHAR_THRESHOLD: int = 2  # Minimum Dutch-specific chars to detect
    LONG_WORD_PERCENTAGE: float = 0.2  # % of long words to suggest Dutch
    LONG_WORD_MIN_LENGTH: int = 12  # What constitutes a "long word"
    
    # Supported languages
    SUPPORTED_LANGUAGES: List[str] = field(default_factory=lambda: ['en', 'nl'])

# Global instance
LANG_CONFIG = LanguageDetectionConfig()

# ============================================================================
# COMPARISON QUERY PATTERNS
# ============================================================================
# NOTE: Regex patterns for extracting comparison terms.
# LIMITATION: English-only patterns, needs i18n support.

# Comparison query detection patterns
# Based on: Common English comparison formats
COMPARISON_PATTERNS: List[str] = [
    # "difference between X and Y"
    r'(?:difference|differences)\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)',
    # "X vs Y" or "X versus Y"  
    r'(.+?)\s+(?:vs|versus|compared\s+to|vs\.)\s+(.+?)(?:\s+in\s+|\?|$)',
    # "compare X with/and/to Y"
    r'(?:compare|comparison)\s+(?:of\s+)?(?:the\s+)?(.+?)\s+(?:with|and|to)\s+(.+?)(?:\?|$)',
    # "X or Y"
    r'(.+?)\s+or\s+(.+?)(?:\s+-\s+|\?|$)',
]

# ============================================================================
# API RERANKING CONFIGURATION
# ============================================================================
# NOTE: API reranking using OpenAI text-embedding-3-small
# Can be disabled via ENABLE_API_RERANKING=false environment variable

@dataclass
class APIRerankingConfig:
    """
    Configuration for API-based reranking.

    Usage in code:
        from src.config.constants import API_RERANKING_CONFIG
        if API_RERANKING_CONFIG.ENABLED:
            # Use API reranking

    Calibration Notes:
        - Model: text-embedding-3-small (OpenAI, Jan 2024)
        - Cost: ~$0.00004 per reranked query
        - Expected rerank rate: 20-30% of queries
        - Monthly cost (500 queries): ~$0.006-0.012
    """

    # Feature flag
    ENABLED: bool = True  # Can be overridden by env var

    # Model configuration
    MODEL: str = "text-embedding-3-small"

    # When to trigger reranking (SelectiveAPIReranker thresholds)
    MIN_CANDIDATES_FOR_RERANKING: int = 2

    # Score variance threshold (trigger if variance < this)
    SIMILAR_SCORES_VARIANCE_THRESHOLD: float = 0.01

    # Confidence thresholds for triggering
    MEDIUM_CONFIDENCE_MIN: float = 0.50
    MEDIUM_CONFIDENCE_MAX: float = 0.75
    LOW_QUALITY_THRESHOLD: float = 0.60
    CLOSE_TOP_TWO_THRESHOLD: float = 0.05

    # Caching
    CACHE_EMBEDDINGS: bool = True

    # Comparison query indicators (case-insensitive)
    COMPARISON_INDICATORS: List[str] = field(default_factory=lambda: [
        'difference', 'compare', 'comparison', 'vs', 'versus',
        'better', 'worse', 'which', 'between', 'or'
    ])

    def __post_init__(self):
        """Validate configuration and check environment overrides."""
        import os

        # Check environment variable override
        env_enabled = os.environ.get('ENABLE_API_RERANKING', '').lower()
        if env_enabled in ('false', '0', 'no', 'off'):
            self.ENABLED = False
        elif env_enabled in ('true', '1', 'yes', 'on'):
            self.ENABLED = True

        # Validate thresholds
        assert 0 <= self.SIMILAR_SCORES_VARIANCE_THRESHOLD <= 1
        assert 0 <= self.MEDIUM_CONFIDENCE_MIN < self.MEDIUM_CONFIDENCE_MAX <= 1
        assert 0 <= self.LOW_QUALITY_THRESHOLD <= 1
        assert 0 <= self.CLOSE_TOP_TWO_THRESHOLD <= 1

# Global instance
API_RERANKING_CONFIG = APIRerankingConfig()

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_all_constants() -> bool:
    """
    Validate that all constants are within reasonable ranges.
    
    This runs automatically on import to catch configuration errors early.
    
    Returns:
        True if all validations pass
        
    Raises:
        ValueError: If any constant is invalid
    """
    try:
        # Validate confidence thresholds
        CONFIDENCE.__post_init__()
        
        # Validate semantic config
        SEMANTIC_CONFIG.__post_init__()
        
        # Validate ranking config
        RANKING_CONFIG.__post_init__()
        
        # Validate context config
        CONTEXT_CONFIG.__post_init__()

        # Validate API reranking config
        API_RERANKING_CONFIG.__post_init__()

        # Additional cross-validation
        assert SEMANTIC_CONFIG.MIN_SCORE_CONTEXT >= SEMANTIC_CONFIG.MIN_SCORE_PRIMARY, \
            "Context threshold should be >= primary threshold"
        
        assert RANKING_CONFIG.PRIORITY_SCORE_DEFINITION > RANKING_CONFIG.PRIORITY_SCORE_CONTEXT, \
            "Definition priority should be > context priority"
        
        # ✅ NEW: Validate citation prefixes
        assert len(REQUIRED_CITATION_PREFIXES) > 0, \
            "REQUIRED_CITATION_PREFIXES cannot be empty"
        
        assert "eurlex:" in REQUIRED_CITATION_PREFIXES, \
            "eurlex: prefix must be in REQUIRED_CITATION_PREFIXES (562 citations)"
        
        assert "skos:" in REQUIRED_CITATION_PREFIXES, \
            "skos: prefix must be in REQUIRED_CITATION_PREFIXES (3249 citations)"
        
        # Validate helper functions work
        assert is_valid_citation_prefix("eurlex:631-28") == True, \
            "Citation validation helper failed"
        
        assert get_citation_category("eurlex:631-28") == "european_regulation", \
            "Citation category helper failed"
        
        return True
        
    except (AssertionError, ValueError) as e:
        raise ValueError(f"Configuration validation failed: {e}")

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def get_config_summary() -> Dict[str, any]:
    """
    Get summary of current configuration for logging/debugging.
    
    Returns:
        Dictionary with configuration summary
    """
    return {
        "version": CONFIG_VERSION,
        "last_updated": CONFIG_LAST_UPDATED,
        "production_validated": PRODUCTION_VALIDATED,
        "calibration_dataset_size": CALIBRATION_DATASET_SIZE,
        "confidence": {
            "high_threshold": CONFIDENCE.HIGH_CONFIDENCE_THRESHOLD,
            "kg_with_def": CONFIDENCE.KG_WITH_DEFINITION,
            "review_threshold": CONFIDENCE.HIGH_CONFIDENCE_THRESHOLD
        },
        "semantic": {
            "enabled": SEMANTIC_CONFIG.ENABLED_BY_DEFAULT,
            "min_score": SEMANTIC_CONFIG.MIN_SCORE_PRIMARY,
            "top_k": SEMANTIC_CONFIG.TOP_K_PRIMARY,
            "model": SEMANTIC_CONFIG.EMBEDDING_MODEL
        },
        "ranking": {
            "max_candidates": RANKING_CONFIG.MAX_TOTAL_CANDIDATES,
            "priority_scale": f"{RANKING_CONFIG.PRIORITY_SCORE_DEFINITION}/"
                            f"{RANKING_CONFIG.PRIORITY_SCORE_NORMAL}/"
                            f"{RANKING_CONFIG.PRIORITY_SCORE_CONTEXT}"
        },
        "context": {
            "max_history": CONTEXT_CONFIG.MAX_HISTORY_TURNS,
            "max_concepts": CONTEXT_CONFIG.MAX_CONCEPTS_FROM_HISTORY
        },
        "warnings": [
            "Configuration based on PROTOTYPE testing only",
            "REQUIRES production validation before deployment",
            f"Calibrated on only {CALIBRATION_DATASET_SIZE} queries",
            "Monitor quality metrics and adjust as needed"
        ]
    }

# ============================================================================
# AUTO-VALIDATION ON IMPORT
# ============================================================================

# Validate configuration when module is imported
validate_all_constants()

# Log configuration summary on import
import logging
logger = logging.getLogger(__name__)
logger.info("Configuration constants loaded and validated")
logger.debug(f"Config summary: {get_config_summary()}")
