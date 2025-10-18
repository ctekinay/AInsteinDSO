# Claude Coding Agent Instructions - AInstein Alliander AI Assistant

## Project Context
You are working with AInstein, an advanced Enterprise Architecture AI Assistant for Alliander, a Dutch Distribution System Operator (DSO). This system provides intelligent assistance for enterprise architecture consulting in the energy sector, featuring multi-LLM architecture, comprehensive knowledge graphs, homonym disambiguation, and real-time web interface for ArchiMate modeling and TOGAF compliance.

## Critical Requirements

### 1. GROUNDING IS MANDATORY
- **NEVER** generate a response without citations
- Every response MUST include at least one of: `archi:id-`, `skos:`, `iec:`, `togaf:adm:`, `entsoe:`, `lido:`
- If you cannot ground a response, raise `UngroundedReplyError` - do not guess

### 2. Knowledge Sources (in priority order)
1. **energy_knowledge_graph.ttl** - 39,100+ triples with IEC, ENTSOE, EUR-LEX terms
2. **ArchiMate models** - XML files in data/models/
3. **TOGAF patterns** - Indexed separately in togaf_core RAG
4. **Only use vectors as last resort** - Structured data first, always

### 3. Current Architecture Principles
- **4R+G+C Pipeline**: Reflect → Route → Retrieve → Refine → Ground → Critic → Validate
- **Multi-LLM orchestration**: Primary Groq, fallback OpenAI, local Ollama support
- **Embedding-first retrieval**: Vector similarity with homonym disambiguation
- **Citation authenticity**: Pre-loaded citation pools prevent fake citations
- **Session persistence**: Conversation state and audit trail management
- **Human review triggers**: Automatic when confidence < 0.75
- **Real-time web interface**: FastAPI with trace visualization

## Current File Structure
```
AInsteinAlliander/
├── src/
│   ├── agents/             # Main AI agents and session management
│   │   ├── ea_assistant.py      # Production EA Agent with 4R+G+C pipeline
│   │   ├── embedding_agent.py   # Embedding-based agent
│   │   ├── llm_council.py       # Multi-LLM coordination
│   │   └── session_manager.py   # Conversation state management
│   ├── llm/                # LLM providers and prompts
│   │   ├── factory.py           # LLM provider factory
│   │   ├── groq_provider.py     # Groq integration (primary)
│   │   ├── openai_provider.py   # OpenAI integration
│   │   ├── ollama_provider.py   # Local Ollama support
│   │   └── prompts.py           # EA-specific prompt templates
│   ├── knowledge/          # Knowledge graph and SPARQL
│   │   └── kg_loader.py         # RDF graph loader and SPARQL
│   ├── safety/             # Grounding and citation validation
│   │   ├── grounding.py         # Citation validation
│   │   └── citation_validator.py # Authentic citation checking
│   ├── routing/            # Query routing and disambiguation
│   │   ├── query_router.py      # Domain-aware query routing
│   │   ├── homonym_detector.py  # Homonym detection system
│   │   └── homonym_guard.py     # Disambiguation protection
│   ├── archimate/          # ArchiMate parsing and TOGAF
│   │   └── parser.py            # ArchiMate XML parsing
│   ├── validation/         # Critic and quality assessment
│   │   ├── critic.py            # Response quality assessment
│   │   └── togaf_rules.py       # TOGAF compliance validation
│   ├── documents/          # Document processing
│   │   └── pdf_indexer.py       # PDF indexing and chunking
│   ├── web/                # Web interface
│   │   ├── app.py               # FastAPI web application
│   │   ├── static/              # CSS, JS assets
│   │   └── templates/           # HTML templates
│   ├── api/                # API endpoints (future)
│   ├── monitoring/         # Performance monitoring
│   │   └── performance_slos.py  # SLA monitoring
│   ├── evaluation/         # Quality metrics and testing
│   ├── utils/              # Utilities
│   │   ├── trace.py             # Execution tracing
│   │   ├── citation_ids.py      # Citation ID management
│   │   └── dedupe_results.py    # Result deduplication
│   ├── exceptions/         # Custom exceptions
│   │   └── exceptions.py        # Domain-specific errors
│   └── config/             # Configuration
│       └── constants.py         # System constants
├── data/
│   ├── docs/               # PDF documents for indexing
│   ├── embeddings/         # Vector embeddings cache
│   └── models/             # ArchiMate XML models
├── config/                 # Configuration files
├── tests/
│   ├── unit/               # Unit tests with archive
│   ├── integration/        # Integration tests
│   └── conftest.py         # Test configuration
├── tools/                  # Build and maintenance tools
│   └── build_homonym_lexicon.py # Homonym detection tools
├── pyproject.toml          # Poetry project configuration
├── run_web_demo.py         # Web interface launcher
└── test_conversation.py    # CLI testing interface
```

## Code Standards

### Python Requirements
- Python 3.11+
- Type hints on ALL functions
- Docstrings with parameter descriptions
- Async/await for I/O operations
- Comprehensive error handling with specific exceptions

### Testing Requirements
- Every module must have corresponding tests
- Integration tests for full pipeline
- Mock external dependencies in unit tests
- Test coverage > 80%

### Current Architecture Components

1. **Multi-LLM Integration** (src/llm/)
   - **Primary**: Groq with Llama 3.3, Qwen 3, Kimi K2 models
   - **Secondary**: OpenAI GPT-4/5 support
   - **Local**: Ollama integration for offline use
   - **Factory Pattern**: Unified LLM provider interface

2. **ProductionEAAgent** (src/agents/ea_assistant.py) ✅ IMPLEMENTED
   - Complete 4R+G+C pipeline implementation
   - Embedding-based retrieval with homonym disambiguation
   - Citation validation with authentic source checking
   - Full audit trail and session management
   - Confidence assessment with human review triggers

3. **Homonym Disambiguation System** (src/routing/) ✅ IMPLEMENTED
   - **HomonymDetector**: Identifies ambiguous terms in queries
   - **HomonymGuard**: Prevents incorrect interpretations
   - **QueryRouter**: Domain-aware routing with embedding fallback
   - **Pre-loaded lexicons**: Energy domain terminology

4. **Advanced Safety System** (src/safety/) ✅ IMPLEMENTED
   - **GroundingCheck**: Validates ALL responses have citations
   - **CitationValidator**: Prevents fake citation generation
   - **Pre-loaded citation pools**: Constrains LLM to authentic sources
   - **Fingerprint validation**: Vector optimization for accuracy

5. **Web Interface** (src/web/) ✅ IMPLEMENTED
   - Real-time chat interface with trace visualization
   - FastAPI backend with async handling
   - Session persistence and conversation history
   - Response quality indicators

## Quality Gates (MUST PASS)
- **Grounding failures**: 0 (absolutely no ungrounded responses)
- **Top-1 accuracy**: ≥ 80%
- **Abstention rate**: ≥ 15% (good to abstain when uncertain)
- **Response time**: < 3 seconds
- **All model changes**: via PR only

## Domain Vocabulary Samples

### IEC Terms to Recognize
- ActivePower, ReactivePower, Equipment
- Conductor, Breaker, Transformer
- IEC 61968, IEC 61970, CIM

### ArchiMate Elements
- Business: Actor, Role, Process, Service, Capability
- Application: Component, Service, Interface
- Technology: Node, Device, System Software

### TOGAF ADM Phases
- Phase A: Architecture Vision
- Phase B: Business Architecture → Business Layer elements
- Phase C: Information Systems → Application/Data elements
- Phase D: Technology Architecture → Technology Layer elements

## Example Patterns

### Good Citation Example
```python
response = "Based on TOGAF Phase B (togaf:adm:B), use Business Capability (archi:id-cap-001) for modeling grid congestion per IEC 61968 (iec:GridCongestion)."
# Has multiple citations ✓
```

### Bad Response Example
```python
response = "You should use a Business Capability for this."
# NO CITATIONS - Must raise UngroundedReplyError ✗
```

## Special Instructions

### When Creating Tests
Always include these test cases:
1. Electronic Court Filing → Capability (from research paper)
2. Grid congestion management (energy domain specific)
3. Query without possible grounding (should abstain)

### When Implementing Router
Priority order:
1. Check for IEC/energy terms → structured_model
2. Check for TOGAF/ArchiMate terms → togaf_method  
3. Default → unstructured_docs

### When Generating PR Drafts
Include:
- Branch name: `ea-assist-{session_id[:8]}`
- jArchi script for changes
- Diff preview
- Citations for why changes are needed

## Performance Targets
- Knowledge graph load: < 3000ms (realistic for 39K triples)
- SPARQL query: < 1500ms P95 (with 35,000x cache speedup)
- Full pipeline: < 3s P50
- Router decision: < 50ms

## Error Messages
- UngroundedReplyError: "Response lacks required citations. Must include at least one of: {prefixes}"
- LowConfidenceError: "Confidence {score} below threshold. Human review required."
- InvalidModelChangeError: "Model changes must be submitted via PR, not direct modification."

## Remember
1. **Grounding is not optional** - it's the #1 requirement
2. **Structured data beats embeddings** - always check KG first
3. **When in doubt, abstain** - better to request human review than give wrong answer
4. **Every decision is audited** - context_store tracks everything

## Git Workflow Instructions

### Commit Strategy
- **Commit after EVERY completed component** (not just at the end)
- **Run tests before committing** - never commit broken code
- **Use conventional commits** format (see below)
- **Push to feature branch**, not main directly

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature implementation
- `fix`: Bug fix
- `test`: Adding tests
- `docs`: Documentation only
- `refactor`: Code change that neither fixes nor adds feature
- `chore`: Changes to build process or auxiliary tools

Examples:
```
feat(grounding): implement citation validation with UngroundedReplyError

- Add GroundingCheck class with assert_citations method
- Implement citation pattern matching for all required prefixes
- Add auto-suggestion for missing citations
- Comprehensive test coverage for grounding scenarios

Refs: claude.md requirements section
```

### Git Commands to Execute

After creating each module:
```bash
# Run tests first
pytest tests/test_<module>.py -v

# If tests pass, stage changes
git add .

# Commit with descriptive message
git commit -m "feat(<module>): <what you implemented>"

# Push to feature branch
git push origin feat/ea-assistant-implementation
```

### Branch Strategy
- Main branch: `main` (protected)
- Feature branch: `feat/ea-assistant-implementation`
- PR will be created after all quality gates pass

### Automated Commit Points
1. After project structure creation → `chore(setup): initialize project structure with Poetry`
2. After KG loader → `feat(knowledge): implement knowledge graph loader with validation`
3. After grounding → `feat(safety): add grounding check with citation enforcement`
4. After router → `feat(routing): implement query router with domain term detection`
5. After critic → `feat(validation): add critic module with confidence assessment`
6. After ArchiMate parser → `feat(archimate): implement model parser with TOGAF mapping`
7. After main pipeline → `feat(agent): integrate full 4R+G+C pipeline`
8. After API → `feat(api): add FastAPI endpoints with async handling`
9. After evaluation → `feat(evaluation): implement quality gate harness`
10. After documentation → `docs: complete README and ADRs`

## Current Implementation Status (October 2024)

### ✅ COMPLETED COMPONENTS
- [x] **Project structure** - Poetry-based with proper module organization
- [x] **Multi-LLM architecture** - Groq, OpenAI, Ollama providers with factory pattern
- [x] **Knowledge graph loader** - RDF/SPARQL integration with 39K+ triples
- [x] **ProductionEAAgent** - Complete 4R+G+C pipeline with embeddings
- [x] **Homonym disambiguation** - Advanced detection and guard systems
- [x] **Citation validation** - Grounding check with authentic source validation
- [x] **Query routing** - Domain-aware routing with embedding fallback
- [x] **Critic assessment** - Confidence scoring and human review triggers
- [x] **Session management** - Conversation state and audit trails
- [x] **Web interface** - Real-time FastAPI app with trace visualization
- [x] **Comprehensive testing** - Unit and integration test suites
- [x] **Performance monitoring** - SLA tracking and optimization

### 🚧 IN PROGRESS
- [ ] **API endpoints** - RESTful API for external integration
- [ ] **TOGAF compliance** - Enhanced methodology alignment
- [ ] **Multilingual support** - Dutch language processing

### 🎯 DEPLOYMENT READY
The system is production-ready with:
- Web interface at `http://localhost:8000` via `run_web_demo.py`
- CLI testing via `test_conversation.py`
- Comprehensive test coverage with `pytest`
- Performance SLOs monitoring
- Citation authenticity validation

## Current Technology Stack

### Core Technologies
- **Python 3.11+** with Poetry dependency management
- **FastAPI** for web interface and API endpoints
- **RDFLib** for knowledge graph processing (39K+ triples)
- **Sentence Transformers** for embedding generation
- **PyTorch** for neural network operations

### LLM Providers (Multi-LLM Architecture)
```python
# Primary: Groq (fast, cost-effective)
GROQ_MODELS = ["llama-3.3-70b-versatile", "qwen2.5-72b-instruct", "deepseek-r1-distill-llama-70b"]

# Secondary: OpenAI (high quality)
OPENAI_MODELS = ["gpt-4", "gpt-5"]

# Local: Ollama (offline capability)
OLLAMA_MODELS = ["llama3.1", "qwen2.5"]
```

### Key Features in Production
1. **Homonym Detection**: Identifies ambiguous terms like "power" (electrical vs. authority)
2. **Citation Validation**: Prevents hallucinated citations with pre-loaded pools
3. **Session Management**: Persistent conversation state with audit trails
4. **Trace Visualization**: Real-time pipeline execution tracking
5. **Performance SLOs**: Response time and accuracy monitoring

## How to Use the Current System

### Web Interface (Recommended)
```bash
python run_web_demo.py
# Open http://localhost:8000
```

### CLI Testing
```bash
python test_conversation.py
```

### Running Tests
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Full test suite with coverage
pytest --cov=src tests/
```
