"""
Configuration Constants for AInstein EA Assistant

IMPORTANT: These values are empirically derived from prototype testing.
They should be validated with production data and adjusted accordingly.

Version: 1.0.0
Last Updated: 2025-10-14
Review Frequency: Quarterly or after major data updates
Production Validated: NO - Prototype values only
"""

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
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
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

# Citation pattern prefixes
# Based on: All supported citation namespaces
REQUIRED_CITATION_PREFIXES: List[str] = [
    "archi:id-",
    "skos:",
    "iec:",
    "togaf:adm:",
    "togaf:concepts:",
    "archimate:research:",
    "entsoe:",
    "lido:",
    "doc:",
    "external:"
]

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
        API_RERANKING_CONFIG.__post_init__()  # ← ADD THIS LINE

        # Additional cross-validation
        assert SEMANTIC_CONFIG.MIN_SCORE_CONTEXT >= SEMANTIC_CONFIG.MIN_SCORE_PRIMARY, \
            "Context threshold should be >= primary threshold"
        
        assert RANKING_CONFIG.PRIORITY_SCORE_DEFINITION > RANKING_CONFIG.PRIORITY_SCORE_CONTEXT, \
            "Definition priority should be > context priority"
        
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
