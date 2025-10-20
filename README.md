# AInstein AI Assistant for Alliander

**Advanced Enterprise Architecture AI Assistant for Energy Systems**

An intelligent AI assistant designed for enterprise architecture consulting in the energy sector, featuring multi-LLM architecture, comprehensive knowledge graphs, and real-time web interface.

## 🎯 Production Status (October 2025)

**Version 3.0** - Production Ready with ADR Support

- ✅ **Architecture Decision Records** - 13 ADRs integrated with intelligent routing
- ✅ **LLM-Based Query Routing** - Semantic understanding of user intent
- ✅ **Enhanced Citations** - 3,970+ sources including ADR references
- ✅ **API Reranking** - OpenAI text-embedding-3-small for +15-20% quality boost
- ✅ **Performance Optimized** - 6x faster KG loading, 3x faster initialization
- ✅ **Intelligent Filtering** - Preserves ADR and document content while filtering noise

## 🌟 Features

- **Multi-LLM Architecture** - Groq (Llama 3.3), OpenAI (GPT-5), Ollama support
- **Knowledge Graph Integration** - 39,122 energy domain triples with IEC standards
- **Advanced Retrieval** - API reranking with OpenAI embeddings for superior quality
- **Citation Validation** - Grounded responses with authentic source verification
- **Real-time Web Interface** - Interactive chat with trace visualization
- **Enterprise Architecture** - ArchiMate model parsing and TOGAF methodology
- **Performance Monitoring** - SLA tracking with response time < 3 seconds

## 🚀 Installation on Your Computer

### Prerequisites

- **Python 3.11+** (Python 3.11 or 3.12 recommended)
- **Git**
- **API Keys** (Groq and/or OpenAI)

### Step 1: Download the Project

```bash
git clone https://github.com/ctekinay/AInsteinDSO.git
cd AInsteinDSO
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Create your `.env` file:
```bash
touch .env
```

Edit the `.env` file with your API keys:
```env
# Primary provider - Groq (recommended, fast and cost-effective)
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Optional - OpenAI for API reranking (enhanced quality)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5
ENABLE_API_RERANKING=true

# Performance settings
EA_LOG_LEVEL=INFO
```

### Step 4: Get Your API Keys

**For Groq (Recommended):**
1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up for free
3. Go to the API Keys section
4. Create a new API key
5. Copy it to your `.env` file

**For OpenAI (Optional):**
1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create API key
3. Copy it to your `.env` file

### Step 5: Start the Application

```bash
python run_web_demo.py
```

That's it! Open your browser to [http://localhost:8000](http://localhost:8000)

## 💬 How to Use

1. **Open your web browser** to http://localhost:8000
2. **Ask questions** about energy systems, enterprise architecture, and technical concepts
3. **Get cited responses** with references from authoritative sources
4. **View processing trace** to see the 4R+G+C pipeline in action

**Example questions:**
- "What is the difference between active power and reactive power?"
- "How does IEC 61968 relate to asset management?"
- "Show me ADR 0025" (Architecture Decision Record queries)
- "What does our decision on demand response say?"
- "Tell me about our architectural choice on interfaces"
- "How many ADRs do we have?" (System queries)
- "What is a Business Capability in ArchiMate?"

## 🔧 Configuration Options

Edit your `.env` file to customize:

```env
# Choose your AI provider
LLM_PROVIDER=groq              # or: openai, ollama

# Groq settings (fast, cost-effective)
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.3           # 0.0 = focused, 1.0 = creative

# OpenAI settings (high quality)
OPENAI_MODEL=gpt-5
OPENAI_TEMPERATURE=0.3
ENABLE_API_RERANKING=true      # Enhanced retrieval quality

# Performance
EA_LOG_LEVEL=INFO              # DEBUG for detailed logs
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Local embedding model
```

## 🐛 Common Issues & Solutions

**"Module not found" errors:**
```bash
# Make sure you activated the virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Then reinstall
pip install -r requirements.txt
```

**"API key invalid" errors:**
- Check your `.env` file has the correct API keys
- Make sure there are no extra spaces or quotes
- Verify the API key is active on the provider's website

**"Port 8000 already in use":**
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9  # macOS/Linux
# or use a different port
python run_web_demo.py --port 8001
```

**Slow responses:**
- Switch to `llama-3.1-8b-instant` in your `.env` for faster responses
- Check your internet connection
- Try lowering `GROQ_MAX_TOKENS` to 512

## 🧪 Testing the Installation

Run these tests to verify everything works:

```bash
# Test basic functionality
python test_conversation.py

# Test system integrations
python scripts/verify_fixes.py

# Test quality improvements
python scripts/quick_quality_test.py
```

## 📁 Project Structure

```
AInsteinDSO/
├── src/
│   ├── agents/             # AI agents and session management
│   ├── llm/                # Multi-LLM providers (Groq, OpenAI, Ollama)
│   ├── knowledge/          # Knowledge graph processing
│   ├── retrieval/          # API reranking and embeddings
│   ├── safety/             # Citation validation and grounding
│   ├── routing/            # LLM-based intelligent query routing
│   ├── documents/          # ADR indexer and document processing
│   ├── web/                # FastAPI web interface
│   └── config/             # System configuration
├── data/
│   ├── adrs/               # Architecture Decision Records (13 ADRs)
│   ├── energy_knowledge_graph.ttl  # 39K+ energy domain triples
│   ├── embeddings/         # Vector cache for semantic search
│   └── models/             # ArchiMate XML models
├── scripts/                # Quality tests and verification tools
├── tests/                  # Comprehensive test suite
├── run_web_demo.py         # Start the web interface
├── .env                    # Your API keys (create this)
└── requirements.txt        # Python packages needed
```

## 🤝 Need Help?

- **Check the logs** - Look at the terminal output for error messages and debug functionality
- **Verify your setup** - Make sure Python 3.11+, API keys are correct
- **Check web interface** - Open http://localhost:8000 to see trace visualization

## 📊 System Architecture

**Enhanced 4R+G+C Pipeline:**
1. **Reflect** - Query analysis and intent detection
2. **Route** - LLM-based intelligent routing with ADR support
3. **Retrieve** - Multi-source knowledge (KG + ADRs + API reranking)
4. **Refine** - Multi-LLM response generation with enhanced prompts
5. **Ground** - Citation validation (3,970+ sources including ADRs)
6. **Critic** - Quality assessment and confidence scoring

**Current Performance:**
- Response time: < 3 seconds P50
- Citation accuracy: 100% (no fake citations)
- Knowledge coverage: 39,122 energy domain triples + 13 ADRs
- Quality improvement: +15-20% with API reranking
- ADR integration: Direct number matching and content search

---

**Built for Enterprise Architecture in Energy Systems**
*Production Ready - Version 3.0 with ADR Support - October 2025*
