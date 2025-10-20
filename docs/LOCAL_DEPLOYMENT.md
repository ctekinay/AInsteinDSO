# Local Deployment Guide - AInstein Alliander EA Assistant

This guide helps you run the AInstein EA Assistant completely offline using your own local Large Language Model (LLM), with no external API calls.

## 🎯 What This Achieves

- **100% Local Processing**: All AI processing happens on your machine
- **No Internet Required**: Works completely offline (after initial setup)
- **Model Agnostic**: Use ANY local LLM server that supports OpenAI-compatible APIs
- **Full Feature Set**: Complete EA Assistant functionality with local embeddings

## 🔧 Quick Setup (5 Minutes)

### 1. Copy Configuration Template
```bash
cp .env.local .env
```

### 2. Configure Your Local LLM
Edit `.env` to point to your LLM server:

```bash
# Example for Ollama
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.3:70b

# Example for LM Studio
LLM_PROVIDER=custom
OPENAI_BASE_URL=http://localhost:1234
OPENAI_MODEL=your-model-name

# Example for text-generation-webui
LLM_PROVIDER=custom
OPENAI_BASE_URL=http://localhost:5000
OPENAI_MODEL=your-model-name
```

### 3. Verify Setup
```bash
python scripts/check_local_setup.py
```

### 4. Start EA Assistant
```bash
python run_web_demo.py
```

Open http://localhost:8000 in your browser.

## 📋 Supported LLM Servers

The EA Assistant works with ANY LLM server that provides OpenAI-compatible API endpoints. Popular options include:

### Ollama (Recommended for Beginners)
```bash
# Install and start Ollama
ollama pull llama3.3:70b
ollama serve

# Configure in .env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.3:70b
```

### LM Studio
```bash
# Start LM Studio server on port 1234
# Configure in .env
LLM_PROVIDER=custom
OPENAI_BASE_URL=http://localhost:1234
OPENAI_MODEL=your-loaded-model
```

### text-generation-webui (oobabooga)
```bash
# Start with OpenAI API extension
python server.py --api --listen

# Configure in .env
LLM_PROVIDER=custom
OPENAI_BASE_URL=http://localhost:5000
OPENAI_MODEL=your-model
```

### vLLM
```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server --model your-model

# Configure in .env
LLM_PROVIDER=custom
OPENAI_BASE_URL=http://localhost:8000
OPENAI_MODEL=your-model
```

### llama.cpp Server
```bash
# Start llama.cpp server
./server -m model.gguf --port 8080

# Configure in .env
LLM_PROVIDER=custom
OPENAI_BASE_URL=http://localhost:8080
OPENAI_MODEL=your-model
```

## 🔍 Technical Requirements

### What EA Assistant Expects from Your LLM Server

Your local LLM server must provide these OpenAI-compatible endpoints:

#### 1. Chat Completions (Required)
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "your-model",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.3,
  "max_tokens": 4096
}
```

**Response Format:**
```json
{
  "choices": [{"message": {"content": "response"}}],
  "usage": {"total_tokens": 123}
}
```

#### 2. Model Listing (Optional)
```http
GET /v1/models
```

**Response Format:**
```json
{
  "data": [{"id": "model-name", "object": "model"}]
}
```

### Model Recommendations

For optimal EA Assistant performance:

- **Minimum**: 7B parameters (Llama 3.1 7B, Mistral 7B)
- **Recommended**: 13B+ parameters (Llama 3.3 70B, Qwen 3 32B)
- **Memory**: 8GB+ RAM for 7B models, 16GB+ for larger models
- **Context**: 4K+ token context window

## ⚙️ Configuration Reference

### Core Local Settings
```bash
# Force local-only operation
USE_OPENAI_EMBEDDINGS=false          # Use local embedding model
ENABLE_API_RERANKING=false           # Disable API-based reranking
EMBEDDING_MODEL=all-MiniLM-L6-v2     # Local embedding model

# LLM Provider Selection
LLM_PROVIDER=ollama                  # or: openai, custom
```

### Provider-Specific Settings

#### Ollama Configuration
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.3:70b
OLLAMA_MAX_TOKENS=4096
OLLAMA_TEMPERATURE=0.3
OLLAMA_TIMEOUT=60
```

#### Custom OpenAI-Compatible Server
```bash
LLM_PROVIDER=custom
OPENAI_BASE_URL=http://localhost:1234
OPENAI_MODEL=your-model-name
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.3
OPENAI_TIMEOUT=30
```

### Performance Tuning
```bash
# Optimize for local deployment
LLM_FALLBACK_ENABLED=true            # Enable fallback mechanisms
LLM_HEALTH_CHECK_TIMEOUT=10          # Health check timeout
EMBEDDING_CACHE_DIR=data/embeddings  # Cache embeddings locally
EMBEDDING_AUTO_REFRESH=false         # Disable auto-refresh
```

## 🚀 Testing Your Setup

### Automated Validation
```bash
# Run comprehensive setup check
python scripts/check_local_setup.py
```

This script validates:
- ✅ Environment configuration
- ✅ Local-only settings
- ✅ LLM endpoint connectivity
- ✅ Required data files
- ✅ Python dependencies
- ✅ Embedding model functionality
- ✅ End-to-end integration

### Manual Testing

#### 1. Test LLM Connectivity
```bash
# For Ollama
curl http://localhost:11434/api/version

# For OpenAI-compatible
curl http://localhost:1234/v1/models
```

#### 2. Test EA Assistant
```bash
# CLI interface
python test_conversation.py

# Web interface
python run_web_demo.py
# Open http://localhost:8000
```

#### 3. Verify Local-Only Operation
Check that these commands work WITHOUT internet:
- Disconnect from internet
- Ask EA Assistant: "What is a business capability?"
- Should receive proper response with citations

## 🔧 Troubleshooting

### Common Issues

#### "Cannot connect to LLM server"
```bash
# Check if your LLM server is running
curl http://localhost:11434/api/version  # Ollama
curl http://localhost:1234/v1/models     # Others

# Verify port and URL in .env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434   # Check this URL
```

#### "Model not found"
```bash
# For Ollama - pull the model
ollama pull llama3.3:70b

# For others - check model name matches what's loaded
curl http://localhost:1234/v1/models
```

#### "Embedding model download failed"
```bash
# Ensure internet connection for first download
# Model will be cached locally afterward
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

#### "Knowledge graph not found"
```bash
# Verify data files exist
ls -la data/energy_knowledge_graph.ttl
ls -la config/vocabularies.json

# If missing, contact system administrator
```

### Performance Optimization

#### Memory Issues
```bash
# Reduce model size
OLLAMA_MODEL=llama3.1:7b  # Instead of 70b

# Limit token generation
OLLAMA_MAX_TOKENS=2048    # Instead of 4096
```

#### Slow Response Times
```bash
# Increase timeouts
OLLAMA_TIMEOUT=120        # Instead of 60

# Enable caching
EMBEDDING_CACHE_DIR=data/embeddings
```

#### High CPU Usage
```bash
# Limit concurrent processing
export TOKENIZERS_PARALLELISM=false

# Use GPU acceleration (if available)
# Configure in your LLM server settings
```

## 🏗️ Architecture Overview

### Local Deployment Stack
```
┌─────────────────────────────────────┐
│           Web Browser               │
│        http://localhost:8000        │
└─────────────────────────────────────┘
                    │
┌─────────────────────────────────────┐
│        EA Assistant WebUI           │
│      (FastAPI + HTML/CSS/JS)        │
└─────────────────────────────────────┘
                    │
┌─────────────────────────────────────┐
│      ProductionEAAgent              │
│     (4R+G+C Pipeline)               │
│  Reflect→Route→Retrieve→Refine      │
│       →Ground→Critic                │
└─────────────────────────────────────┘
                    │
┌─────────────────────────────────────┐
│         Your Local LLM              │
│    (Ollama/LM Studio/etc.)          │
│   OpenAI-Compatible API             │
└─────────────────────────────────────┘
```

### Data Flow
1. **User Query** → Web interface
2. **4R+G+C Pipeline** → Process query locally
3. **Knowledge Retrieval** → Local embeddings + knowledge graph
4. **LLM Generation** → Your local model
5. **Citation Validation** → Grounding check
6. **Response** → Back to user

### Key Components Running Locally
- **Knowledge Graph**: 39,100+ energy domain triples
- **Embeddings**: Sentence transformers (all-MiniLM-L6-v2)
- **ArchiMate Parser**: XML model processing
- **TOGAF Rules**: Architecture methodology validation
- **Homonym Detection**: Domain-specific disambiguation
- **Citation Validation**: Authentic source verification

## 📚 Additional Resources

### Model Recommendations
- **Llama 3.3 70B**: Best overall performance for EA tasks
- **Qwen 3 32B**: Good balance of speed and quality
- **Mistral 7B**: Lightweight option for resource-constrained setups
- **CodeLlama**: Good for technical architecture discussions

### Hardware Requirements
- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB+ RAM, 8+ CPU cores
- **GPU**: Optional but recommended for larger models
- **Storage**: 10GB+ for models and embeddings cache

### External Documentation
- [Ollama Documentation](https://ollama.ai/docs)
- [LM Studio Guide](https://lmstudio.ai/docs)
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui)
- [OpenAI API Compatibility](https://platform.openai.com/docs/api-reference)

## 🤝 Support

If you encounter issues:

1. **Run diagnostics**: `python scripts/check_local_setup.py`
2. **Check logs**: Look for errors in console output
3. **Verify configuration**: Ensure .env matches your LLM server
4. **Test connectivity**: Use curl to test your LLM endpoints

For EA Assistant specific issues, check the main README.md and technical documentation.

---

**Happy local AI deployment! 🎉**

*This setup gives you a powerful, private, and fully autonomous Enterprise Architecture AI assistant running entirely on your infrastructure.*