# AInstein AI Assistant - Advanced Workflow

## Production 4R+G+C Pipeline with Multi-LLM Architecture

The AInstein system implements a sophisticated pipeline for enterprise architecture assistance with homonym disambiguation, multi-LLM orchestration, and real-time web interface.

## Core Pipeline: Reflect → Route → Retrieve → Refine → Ground → Critic → Validate

### 1. **REFLECT**: Advanced Query Analysis with Homonym Detection
   - **Homonym Detection**: Identify ambiguous terms (e.g., "power" = electrical vs. authority)
   - **Domain Classification**: Energy terms (IEC, ENTSOE), ArchiMate elements, TOGAF phases
   - **Intent Analysis**: Definition queries, comparison requests, architecture modeling
   - **Context Expansion**: Use conversation history for disambiguation
   - **Language Detection**: English/Dutch processing capability

### 2. **ROUTE**: Intelligent Source Selection with Fallback Strategy
   - **Primary**: Embedding-based retrieval with semantic similarity
   - **Homonym Guard**: Prevent incorrect interpretations using pre-loaded lexicons
   - **Multi-source orchestration**:
     - Knowledge Graph (39K+ triples) → IEC/ENTSOE definitions
     - ArchiMate Models → Enterprise architecture patterns
     - PDF Documents → Detailed specifications
     - TOGAF Framework → Methodology guidance
   - **Adaptive routing**: Switch sources based on query confidence

### 3. **RETRIEVE**: 4-Phase Enhanced Retrieval System
   - **Phase 1**: Direct knowledge graph lookup (SPARQL)
   - **Phase 2**: Semantic enhancement with embeddings (min_score=0.40)
   - **Phase 3**: Context expansion using conversation history
   - **Phase 4**: Ranking and deduplication of candidates
   - **Vector similarity**: Sentence Transformers with PyTorch optimization
   - **Citation pools**: Pre-loaded authentic citations constrain LLM responses

### 4. **REFINE**: Multi-LLM Synthesis and Enhancement
   - **Primary LLM**: Groq (Llama 3.3, Qwen 3, Kimi K2) for speed and cost-effectiveness
   - **Fallback LLM**: OpenAI (GPT-4/5) for complex reasoning
   - **Local LLM**: Ollama for offline/privacy requirements
   - **LLM Council**: Coordinate multiple models for optimal response
   - **Domain synthesis**: Merge multiple sources with energy expertise
   - **Quality thresholds**: Filter noise with confidence scoring

### 5. **GROUND**: Citation Authenticity and Validation
   - **Mandatory grounding**: Every response MUST include valid citations
   - **Citation patterns**: `archi:id-`, `skos:`, `iec:`, `togaf:adm:`, `entsoe:`, `lido:`, `doc:`, `external:`
   - **Authenticity validation**: Prevent hallucinated citations with pre-loaded pools
   - **Fingerprint validation**: Vector optimization for citation accuracy
   - **Auto-suggestion**: Recommend valid citations when missing
   - **Exception handling**: Raise `UngroundedReplyError` if no valid citations found

### 6. **CRITIC**: Advanced Confidence Assessment and Quality Control
   - **Relevance scoring**: Semantic similarity and domain alignment
   - **Contradiction detection**: Cross-reference multiple sources
   - **Confidence calculation**: Weighted scoring across multiple factors
   - **Human review triggers**: Automatic escalation when confidence < 0.75
   - **Quality gates**: Top-1 accuracy ≥ 80%, abstention rate ≥ 15%
   - **Performance SLOs**: Response time < 3 seconds, grounding failures = 0

### 7. **VALIDATE**: TOGAF Compliance and Architecture Governance
   - **Phase appropriateness**: Validate recommendations against TOGAF ADM phases
   - **Layer consistency**: Ensure Business/Application/Technology alignment
   - **ArchiMate compliance**: Verify element relationships and patterns
   - **Architecture principles**: Check adherence to enterprise standards
   - **Change management**: Generate PR drafts for model modifications

## Advanced Features

### Multi-LLM Orchestration
```python
# Primary: Groq (fast, cost-effective)
GROQ_MODELS = ["llama-3.3-70b-versatile", "qwen2.5-72b-instruct", "deepseek-r1-distill-llama-70b"]

# Secondary: OpenAI (high quality)
OPENAI_MODELS = ["gpt-4", "gpt-5"]

# Local: Ollama (offline capability)
OLLAMA_MODELS = ["llama3.1", "qwen2.5"]
```

### Session Management and Audit Trail
- **Conversation state**: Persistent session tracking across interactions
- **Context store**: Audit trail for accountability and learning
- **Session IDs**: UUID-based tracking for debugging and analytics
- **Turn management**: Structured conversation history with metadata

### Real-time Web Interface
- **FastAPI backend**: Async handling with uvicorn
- **Trace visualization**: Real-time pipeline execution monitoring
- **Response quality indicators**: Confidence scores and citation validation
- **Interactive chat**: Session persistence with conversation history

### Performance Optimization
- **Embedding cache**: Persistent vector storage for fast retrieval
- **SPARQL optimization**: 35,000x cache speedup for knowledge graph queries
- **Async processing**: Non-blocking I/O for web interface
- **Background tasks**: Long-running operations with progress tracking

## Deployment and Usage

### Production Deployment
```bash
# Web interface (recommended)
python run_web_demo.py
# Access: http://localhost:8000

# CLI testing
python test_conversation.py

# Test suite
pytest tests/ --cov=src
```

### Quality Assurance
- **Comprehensive testing**: Unit and integration test suites
- **Performance monitoring**: SLA tracking with alerts
- **Citation validation**: Authentic source verification
- **Error handling**: Graceful degradation with informative messages

This workflow represents the current production implementation of AInstein, featuring advanced AI capabilities while maintaining enterprise-grade safety and compliance standards.