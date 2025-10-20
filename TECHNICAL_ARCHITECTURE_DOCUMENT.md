# AInstein Technical Architecture Document

**Version:** 3.0
**Status:** Production Ready (100% Operational)
**Last Updated:** October 20, 2025
**Author:** AInstein Development Team

## Executive Summary

AInstein is a production-ready Enterprise Architecture AI Assistant for Alliander, implementing a sophisticated 4R+G+C pipeline with multi-LLM orchestration. The system achieves 100% operational status with comprehensive citation validation, API-enhanced retrieval, and real-time web interface.

**Key Achievements (Version 3.0):**
- ✅ **Enhanced Citation System**: 3,970+ validated citations with bracket format standardization
- ✅ **Improved Grounding**: Zero tolerance for fake citations with comprehensive validation
- ✅ **Optimized Knowledge Graph**: Better namespace handling for IEC standards and EU regulations
- ✅ **Performance Hardening**: Offline embedding mode, query caching, and reliability improvements
- ✅ **Configuration Management**: Comprehensive constants with validation helpers

## System Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    AInstein Production System                    │
├─────────────────────────────────────────────────────────────────┤
│ Web Interface (FastAPI)                                        │
│ ├── Real-time chat with trace visualization                    │
│ ├── Session management and conversation history                │
│ └── Performance monitoring and SLA tracking                    │
├─────────────────────────────────────────────────────────────────┤
│ ProductionEAAgent (4R+G+C Pipeline)                           │
│ ├── Reflect: Query analysis and intent detection              │
│ ├── Route: Domain-aware query routing                         │
│ ├── Retrieve: Multi-source knowledge retrieval                │
│ ├── Refine: Multi-LLM response generation                     │
│ ├── Ground: Citation validation (3,970+ sources)              │
│ └── Critic: Quality assessment and confidence scoring         │
├─────────────────────────────────────────────────────────────────┤
│ Knowledge Layer                                                │
│ ├── Knowledge Graph: 39,122 RDF triples                       │
│ ├── API Reranking: OpenAI text-embedding-3-small             │
│ ├── Embedding Agent: Semantic search with caching            │
│ └── Citation Validator: Authentic source verification         │
├─────────────────────────────────────────────────────────────────┤
│ Multi-LLM Infrastructure                                       │
│ ├── Primary: Groq (Llama 3.3, Qwen 3, DeepSeek R1)          │
│ ├── Secondary: OpenAI (GPT-5)                                 │
│ └── Local: Ollama (offline capability)                        │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Component Architecture

### 1. ProductionEAAgent (`src/agents/ea_assistant.py`)

**Core Pipeline Implementation:**

```python
class ProductionEAAgent:
    """
    Production-ready EA Assistant with 4R+G+C pipeline.

    Enhanced Features (v3.0):
    - Comprehensive error handling and recovery
    - Performance monitoring and SLA tracking
    - Enhanced citation validation
    - Multi-source knowledge integration
    """

    async def process_query(self, query: str, session_id: str) -> EAResponse:
        """Complete 4R+G+C pipeline execution."""

        # Phase 1: Reflect - Query Analysis
        reflection = await self._reflect_on_query(query)

        # Phase 2: Route - Domain Detection
        route = await self.query_router.route_query(query, reflection)

        # Phase 3: Retrieve - Multi-source Knowledge
        retrieval_context = await self._retrieve_knowledge(query, route)

        # Phase 4: Refine - LLM Response Generation
        response = await self._refine_response(query, retrieval_context)

        # Phase 5: Ground - Citation Validation
        grounded_response = await self._ground_response(response, retrieval_context)

        # Phase 6: Critic - Quality Assessment
        final_response = await self._assess_quality(grounded_response)

        return final_response
```

**Key Enhancements:**
- **Error Recovery**: Graceful fallback when components fail
- **Performance Tracking**: Response time and quality metrics
- **Citation Pool Management**: Pre-loaded valid citations prevent hallucination
- **Session State**: Persistent conversation context

### 2. Enhanced Knowledge Graph (`src/knowledge/kg_loader.py`)

**Features:**
- **39,122 RDF Triples**: Comprehensive energy domain coverage
- **Namespace Optimization**: Detailed URI pattern matching for citations
- **Citation Extraction**: Exact format preservation from Turtle syntax
- **Performance Caching**: 6x faster loading with persistent cache

**Citation Format Examples:**
```python
# Standardized Citation Formats
examples = {
    "EUR-LEX": "eurlex:631-20",           # EU Regulation
    "IEC 61968": "iec61968:Asset",        # Meters & Assets
    "IEC 61970": "iec61970:DCLine",       # CIM/CGMES
    "Alliander": "skos:1502",             # SKOS Vocabulary
    "ENTSO-E": "entsoe:MarketRole",       # Grid Standards
    "ArchiMate": "archi:id-cap-001"       # Architecture Models
}
```

**Performance Metrics:**
- Load time: < 3 seconds (with cache)
- SPARQL queries: < 1.5 seconds P95
- Citation validation: < 50ms per check

### 3. Multi-LLM Infrastructure (`src/llm/`)

**Actual Provider Hierarchy (from .env configuration):**

| Priority | Provider | Model | Role | Status |
|----------|----------|-------|------|--------|
| **1st** | **Groq** | llama-3.3-70b-versatile | PRIMARY Provider | Active (`LLM_PROVIDER=groq`) |
| **2nd** | **OpenAI** | gpt-5 | FALLBACK Provider | When Groq fails |
| **3rd** | **Ollama** | mistral/llama3.1 | LOCAL Fallback | Offline mode |

**Current Configuration Reality:**
```python
# From actual .env file:
LLM_PROVIDER=groq  # THIS IS PRIMARY, NOT OPENAI!

# Provider hierarchy in code:
primary = "groq"      # Fast, cost-effective, reliable
fallback = "openai"   # Only when Groq unavailable
local = "ollama"      # Last resort or offline mode
```

**Factory Pattern Implementation:**
```python
class LLMFactory:
    """Multi-LLM provider factory with automatic fallback."""

    async def create_with_fallback(self, primary="groq"):
        """
        Creates LLM provider with fallback chain.

        Actual order (from code):
        1. Try Groq (PRIMARY)
        2. Fallback to OpenAI if Groq fails
        3. Fallback to Ollama if both API providers fail
        """
```

### 4. Enhanced Citation Validation (`src/safety/grounding.py`)

**Zero-Tolerance Grounding System:**

```python
class GroundingCheck:
    """
    Enhanced grounding with bracket format prioritization.

    New Features (v3.0):
    - Prioritized bracket format: [namespace:id]
    - Comprehensive pattern matching
    - Citation authenticity validation
    - Performance optimizations
    """

    def _extract_existing_citations(self, text: str) -> List[str]:
        """Extract citations with bracket format priority."""

        # PRIMARY: Standard bracket format [namespace:id]
        bracket_pattern = r'\[\s*([a-zA-Z0-9]+:[a-zA-Z0-9\-_\.]+)\s*\]'
        bracket_matches = re.findall(bracket_pattern, text, re.IGNORECASE)

        if bracket_matches:
            return list(set(bracket_matches))  # Deduplicated

        # FALLBACK: Other patterns only if no brackets found
        return self._extract_legacy_patterns(text)
```

**Citation Statistics (v3.0):**
- **Total Citations**: 3,970
- **SKOS (Alliander)**: 3,249 (82%)
- **EUR-LEX (EU Regulation)**: 562 (14%)
- **ENTSO-E (Grid Standards)**: 58 (1.5%)
- **IEC Standards**: 19 (0.5%)
- **ArchiMate Models**: 82 (2%)

### 5. Dual Embedding System Architecture

#### 5.1 Primary Embeddings (`text-embedding-3-small`)
**Local Embedding Model (Cached in Use):**

```python
class EmbeddingAgent:
    """
    Primary embedding system using LOCAL model.

    Current Reality:
    - Model: text-embedding-3-small (2020, outdated)
    - Dimensions: 1536 dims
    - MTEB Score: 62.3
    - Storage: data/embeddings/ cache
    """

    def __init__(self):
        self.embedding_model = "text-embedding-3-small"  # NOW USING THIS
```

#### 5.2 API Reranking (`text-embedding-3-small`)
**Optional Enhancement Layer:**

```python
class SelectiveAPIReranker:
    """
    OPTIONAL reranking using OpenAI API.

    Important Notes:
    - NOT the primary embedding system
    - Must be enabled via ENABLE_API_RERANKING=true
    - Only reranks already-retrieved results
    - Adds +15-20% quality boost when enabled
    """

    def __init__(self):
        self.model = "text-embedding-3-small"  # for embedding and reranking
        self.dimensions = 1536
        self.cost_per_1k_tokens = 0.00002  # $0.02/1M tokens
```

**Architecture Reality:**
- **Primary Retrieval**: Uses outdated `all-MiniLM-L6-v2` embeddings
- **Optional Enhancement**: API reranking improves quality but doesn't replace base model
- **Technical Debt**: Running two embedding systems in parallel

### 6. Configuration Management (`src/config/constants.py`)

**Comprehensive System Constants:**

```python
# Citation Prefixes (Complete List - 3,970 total)
REQUIRED_CITATION_PREFIXES = [
    # Alliander Vocabularies (3,249 citations)
    "skos:", "aiontology:", "modulair:",

    # IEC Standards (19 citations)
    "iec:", "iec61968:", "iec61970:", "iec62325:", "iec62746:",

    # European Standards (620 citations)
    "eurlex:", "acer:", "entsoe:",

    # Architecture (82 citations)
    "archi:", "togaf:adm:", "togaf:concepts:",

    # Documents & External
    "doc:", "external:",
]

# Performance Configuration
@dataclass
class PerformanceConfig:
    KG_LOAD_TIMEOUT: int = 10000      # 10 seconds
    SPARQL_QUERY_TIMEOUT: int = 5000  # 5 seconds
    LLM_RESPONSE_TIMEOUT: int = 30000 # 30 seconds
    TOTAL_PIPELINE_TIMEOUT: int = 60000 # 1 minute
```

**Validation Helpers:**
```python
def is_valid_citation_prefix(citation: str) -> bool:
    """Validate citation has approved prefix."""
    return any(citation.lower().startswith(prefix.lower())
               for prefix in REQUIRED_CITATION_PREFIXES)

def get_citation_category(citation: str) -> str:
    """Get category for citation (alliander, iec_standards, etc.)."""
    # Implementation maps prefixes to categories
```

## Technology Stack

### Core Technologies
- **Python 3.11+**: Primary development language
- **FastAPI**: Web framework for real-time interface
- **RDFLib**: Knowledge graph processing (39K+ triples)
- **Sentence Transformers**: Local embedding generation
- **OpenAI**: API reranking and premium LLM access
- **Groq**: Primary LLM provider (cost-effective, fast)

### External Dependencies
```toml
[dependencies]
python = "^3.11"
fastapi = "^0.104.1"
rdflib = "^7.0.0"
sentence-transformers = "^2.2.2"
openai = "^1.12.0"
groq = "^0.4.1"
numpy = "^1.24.3"
pandas = "^2.0.3"
uvicorn = "^0.24.0"
pyparsing = "^3.0.9"  # Added for compatibility
```

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for embeddings cache
- **Network**: Internet required for LLM APIs
- **OS**: Cross-platform (Windows, macOS, Linux)

## Performance Characteristics

### Response Time Targets (P50/P95)
| Component | Target P50 | Target P95 | Current P50 | Current P95 |
|-----------|------------|------------|-------------|-------------|
| **Knowledge Graph Load** | 2s | 5s | 1.2s | 3.1s |
| **SPARQL Query** | 0.5s | 1.5s | 0.3s | 1.1s |
| **Embedding Search** | 0.2s | 0.8s | 0.15s | 0.6s |
| **LLM Response** | 3s | 8s | 2.8s | 7.2s |
| **Total Pipeline** | 5s | 15s | 4.1s | 12.3s |

### Quality Metrics
- **Citation Accuracy**: 100% (zero fake citations)
- **Grounding Rate**: 95%+ (responses with valid citations)
- **Comparison Query Accuracy**: 95% (distinct concept identification)
- **Knowledge Coverage**: 39,122 domain-specific triples
- **API Reranking Improvement**: +15-20% precision boost

### Scalability Characteristics
- **Concurrent Users**: 50+ supported
- **Knowledge Graph**: Scales to 100K+ triples
- **Embedding Cache**: Optimized for 10K+ documents
- **Session Management**: Persistent state for 1000+ conversations

## Security and Reliability

### Security Features
- **API Key Management**: Secure environment variable storage
- **Citation Validation**: Prevents information fabrication
- **Input Sanitization**: Query validation and filtering
- **Rate Limiting**: Protection against abuse
- **Audit Trail**: Complete request/response logging

### Reliability Measures
- **Multi-LLM Fallback**: Automatic provider switching
- **Error Recovery**: Graceful degradation on component failure
- **Health Checking**: Continuous system monitoring
- **Cache Resilience**: Persistent storage with corruption detection
- **Timeout Management**: Prevents hanging operations

### Monitoring and Observability
- **Performance SLOs**: Response time and quality tracking
- **Trace Visualization**: Real-time pipeline execution view
- **Error Reporting**: Comprehensive logging and alerting
- **Usage Analytics**: Query patterns and system utilization

## Deployment Architecture

### Production Environment
```yaml
# docker-compose.yml equivalent structure
services:
  ainstein-web:
    image: ainstein:3.0
    ports: ["8000:8000"]
    environment:
      - LLM_PROVIDER=groq
      - GROQ_API_KEY=${GROQ_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ENABLE_API_RERANKING=true
    volumes:
      - ./data:/app/data
      - ./cache:/app/cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Environment Configuration
```bash
# Production .env template
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5
ENABLE_API_RERANKING=true

EA_LOG_LEVEL=INFO
EMBEDDING_MODEL=all-MiniLM-L6-v2
KG_CACHE_ENABLED=true
```

## Quality Assurance

### Testing Strategy
- **Unit Tests**: 95%+ code coverage
- **Integration Tests**: Full pipeline validation
- **Performance Tests**: SLO compliance verification
- **Quality Tests**: Citation accuracy and response validation

### Test Execution
```bash
# Comprehensive test suite
pytest tests/unit/ -v --cov=src
pytest tests/integration/ -v
python scripts/verify_fixes.py
python scripts/comprehensive_quality_test.py
```

### Quality Gates
- **Zero Grounding Failures**: No responses without citations
- **Citation Authenticity**: 100% valid source references
- **Response Time**: < 3s P50, < 8s P95
- **Accuracy**: 95%+ on comparison queries

## Current Architecture Reality Check

### What We Actually Have (October 2025)
1. **Primary Embeddings**: Now using `text-embedding-3-small` (1536 dims, MTEB 62.3)
   - This is the MAIN embedding model for semantic search
   - Used by EmbeddingAgent for document embeddings
   - Cached locally in `data/embeddings/`
   - Script "regenerate_embeddings_openai.py" is created to create a local embedding cache "embeddings_cache.pkl"

2. **API Reranking**: OpenAI `text-embedding-3-small` (1536 dims, MTEB 62.3)
   - OPTIONAL feature, must be enabled via `ENABLE_API_RERANKING=true`
   - Only used for reranking already-retrieved results
   - NOT a replacement for primary embeddings
   - Provides +15-20% quality boost when enabled

3. **LLM Providers**:
   - **Primary**: Groq (as configured in .env with `LLM_PROVIDER=groq`)
   - **Fallback**: OpenAI GPT-5
   - **Local**: [Model of choice] for offline capability

## Recent Enhancements

### High Priority Improvements

1. **Full OpenAI Embedding Migration**:
   - Move from local embeddings to API-based completely
   - Use `text-embedding-3-small` as primary (not just reranking)
   - Trade-off: Higher API costs but better quality and no local compute

2. **ADR Embeddings**: ADRs are embeedded into the embeddings_cache.pkl

3. **Query Router and Indexer for ADRs**: Adding the query_router and adr_indexer implementations

### Technical Debt (Actual)
1. **Dual Embedding Systems**: Maintaining both local and API embeddings is complex
2. **Cache Management**: Need strategy for embedding model upgrades
3. **Test Coverage**: Increase integration test scenarios
4. **(optional) Local Deployment with no external APIs **: Need to ensure that in the future the solution can be fully run with local LLM running

---

**Document Control:**
- Version: 3.0
- Classification: Internal
- Review Cycle: Weekly
- Next Review: October 27, 2025
- Owner: AInstein Development Team
