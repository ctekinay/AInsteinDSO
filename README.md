# AInstein AI Assistant for Alliander

**Advanced Enterprise Architecture AI Assistant for Energy Systems**

An intelligent AI assistant designed specifically for enterprise architecture consulting in the energy sector, featuring multi-LLM architecture, comprehensive knowledge graphs, and real-time web interface.

## 🌟 Features

### Core Capabilities
- **Multi-LLM Architecture** - Groq integration with Llama 3.3, Qwen 3, Kimi K2 models
- **Knowledge Graph Integration** - 39,000+ energy domain triples with IEC standards
- **Citation Validation** - Grounded responses with authentic source verification
- **Real-time Web Interface** - Interactive chat with trace visualization
- **Session Management** - Conversation tracking and context maintenance
- **TOGAF Methodology** - Enterprise architecture framework integration
- **Multilingual Support** - English and Dutch language detection

### Technical Features
- **LLM Council Architecture** - Dual validation for enhanced accuracy
- **ArchiMate Model Support** - IEC 61968 and enterprise architecture models
- **Comprehensive Testing** - Unit, integration, and performance testing
- **Production-Ready** - Security-compliant configuration and deployment
- **Async Architecture** - High-performance async/await patterns
- **Monitoring & Tracing** - Complete pipeline observability

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │────│  EA Assistant    │────│  Knowledge Graph│
│   (FastAPI)     │    │  (Multi-LLM)     │    │  (39K+ triples) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────┴────────┐             │
         │              │   LLM Council   │             │
         │              │ (Dual Validation)│            │
         │              └─────────────────┘             │
         │                       │                       │
    ┌────┴─────┐        ┌───────┴────────┐       ┌─────┴──────┐
    │ Session  │        │ Citation       │       │ ArchiMate  │
    │ Manager  │        │ Validator      │       │ Models     │
    └──────────┘        └────────────────┘       └────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (recommended 3.11 or 3.12)
- **Git** for repository management
- **API Keys** for LLM providers (Groq recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/ctekinay/AInsteinDSO.git
cd AInsteinDSO
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Alternative with Poetry (recommended):**

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your configuration
nano .env  # or your preferred editor
```

**Required Configuration (.env file):**

```env
# LLM Provider Configuration
LLM_PROVIDER=groq

# Groq Configuration (Recommended)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MODEL_PRIMARY=llama-3.3-70b-versatile
GROQ_MODEL_VALIDATOR=qwen/qwen3-32b
GROQ_MAX_TOKENS=4096
GROQ_TEMPERATURE=0.3
GROQ_TIMEOUT=30
GROQ_RETRIES=3

# Optional: OpenAI Fallback
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

# Performance Settings
LLM_FALLBACK_ENABLED=true
EA_LOG_LEVEL=INFO
```

### 4. Get API Keys

#### Groq (Recommended - Fast & Cost-effective)
1. Visit [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Generate a new API key
5. Copy the key to your `.env` file

#### OpenAI (Optional Fallback)
1. Visit [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create an API key
3. Add to `.env` file for fallback support

### 5. Launch the Application

```bash
# Start the web interface
python run_web_demo.py

# Or using the CLI
python cli.py
```

The web interface will be available at:
- **Main Interface:** http://localhost:8000
- **API Documentation:** http://localhost:8000/api/docs
- **Health Check:** http://localhost:8000/health

## 📖 Usage Guide

### Web Interface

1. **Open Browser** - Navigate to http://localhost:8000
2. **Start Chatting** - Ask energy architecture questions
3. **View Citations** - See authentic source references
4. **Trace Pipeline** - Monitor processing steps in real-time

**Example Queries:**
```
"What is reactive power in electrical systems?"
"How do I model a transformer in ArchiMate?"
"Explain the IEC 61968 standard for grid management"
"What are the TOGAF phases for energy architecture?"
```

### Command Line Interface

```bash
# Interactive CLI mode
python cli.py

# Single query
python cli.py --query "What is active power?"

# Specify model
python cli.py --model llama-3.3-70b-versatile --query "Explain grid topology"
```

### Integration Testing

```bash
# Run comprehensive tests
python test_integration.py

# Test conversation flow
python test_conversation.py

# Performance testing
pytest tests/integration/ -v
```

## 🔧 Configuration Options

### LLM Providers

**Groq (Recommended)**
- Fast inference with open-source models
- Cost-effective for production use
- Supports Llama 3.3, Qwen 3, Mixtral models

**OpenAI**
- High-quality responses with GPT models
- Good fallback option
- Higher cost per token

**Ollama (Local)**
- Run models locally for privacy
- No API costs
- Requires powerful hardware

### Model Selection

| Model | Provider | Best For | Context Window |
|-------|----------|----------|----------------|
| `llama-3.3-70b-versatile` | Groq | General EA consulting | 131K |
| `qwen/qwen3-32b` | Groq | Technical validation | 131K |
| `moonshotai/kimi-k2-instruct` | Groq | Complex reasoning | 131K |
| `gpt-4` | OpenAI | High-quality responses | 128K |
| `mistral` | Ollama | Local deployment | 32K |

### Performance Tuning

```env
# Adjust these settings in .env for optimal performance

# Response quality vs speed
GROQ_TEMPERATURE=0.3      # Lower = more focused
GROQ_MAX_TOKENS=4096      # Longer responses

# Reliability
GROQ_RETRIES=3            # Auto-retry on failures
GROQ_TIMEOUT=30           # Request timeout (seconds)

# Features
LLM_FALLBACK_ENABLED=true # Auto-fallback to backup provider
EA_LOG_LEVEL=INFO         # Logging detail (DEBUG/INFO/WARNING)
```

## 🧪 Testing

### Run All Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Full test suite
pytest -v

# With coverage
pytest --cov=src tests/ --cov-report=html
```

### Test Specific Components

```bash
# Test knowledge graph integration
python -m pytest tests/unit/test_kg_loader.py

# Test grounding and citations
python -m pytest tests/integration/test_grounding.py

# Test LLM providers
python -m pytest tests/unit/test_llm_providers.py
```

## 🐛 Troubleshooting

### Common Issues

**1. API Key Errors**
```
Error: Invalid API key
```
**Solution:** Verify your API keys in `.env` file and ensure they're active.

**2. Knowledge Graph Loading**
```
Error: Failed to load knowledge graph
```
**Solution:** Check that `data/energy_knowledge_graph.ttl` exists and is readable.

**3. Port Already in Use**
```
Error: Port 8000 is already in use
```
**Solution:**
```bash
# Use different port
python run_web_demo.py --port 8001

# Or kill existing process
lsof -ti:8000 | xargs kill -9
```

**4. Memory Issues with Large Models**
```
Error: Out of memory
```
**Solution:** Switch to smaller models or increase system memory.

### Debug Mode

```bash
# Enable debug logging
export EA_LOG_LEVEL=DEBUG
python run_web_demo.py

# Run diagnostics
python tests/integration/archive/diagnose.py
```

### Performance Issues

**Slow Responses:**
- Switch to `llama-3.1-8b-instant` for faster responses
- Reduce `GROQ_MAX_TOKENS` in `.env`
- Check internet connection for API calls

**High Memory Usage:**
- Restart the application periodically
- Use smaller knowledge graph subsets for testing
- Monitor with `htop` or Task Manager

## 🚀 Production Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t ainstein-assistant .

# Run container
docker run -p 8000:8000 --env-file .env ainstein-assistant
```

### Cloud Deployment

**Recommended Platforms:**
- **AWS EC2** - Full control, cost-effective
- **Google Cloud Run** - Serverless, auto-scaling
- **Heroku** - Simple deployment
- **DigitalOcean App Platform** - Managed hosting

**Resource Requirements:**
- **CPU:** 2+ cores recommended
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 2GB for application + models
- **Network:** Stable internet for API calls

### Environment Variables for Production

```env
# Production settings
EA_LOG_LEVEL=WARNING
LLM_FALLBACK_ENABLED=true
GROQ_TIMEOUT=60
GROQ_RETRIES=5

# Security
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
SECRET_KEY=your-secret-key-here

# Database (if using persistent sessions)
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

## 📚 API Documentation

### REST API Endpoints

**Health Check**
```http
GET /health
```

**Get Statistics**
```http
GET /api/stats
```

**Export Conversation**
```http
POST /api/export/{session_id}?format=markdown
```

### WebSocket API

**Connect to Chat**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/session-123');
ws.send(JSON.stringify({
    "message": "What is reactive power?"
}));
```

**Response Format**
```json
{
    "type": "assistant",
    "content": "Response text...",
    "confidence": 0.85,
    "citations": ["eurlex:631-28"],
    "requires_human_review": false,
    "processing_time_ms": 1250
}
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/ctekinay/AInsteinDSO.git
cd AInsteinDSO

# Install development dependencies
poetry install --with dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Code Standards

- **Python 3.11+** with type hints
- **Black** for code formatting
- **pytest** for testing
- **Comprehensive docstrings**
- **Async/await patterns** for I/O operations

### Adding New Features

1. **Create feature branch** - `git checkout -b feature/new-feature`
2. **Add tests** - Write tests first (TDD approach)
3. **Implement feature** - Follow existing patterns
4. **Update documentation** - Include usage examples
5. **Submit PR** - Include test results and documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Alliander** - Energy sector domain expertise
- **Groq** - Fast LLM inference platform
- **Meta** - Llama model family
- **IEC Standards** - International energy standards
- **TOGAF** - Enterprise architecture methodology
- **ArchiMate** - Architecture modeling language

## 📧 Support

For issues, questions, or contributions:

- **GitHub Issues:** [Report bugs or request features](https://github.com/ctekinay/AInsteinDSO/issues)
- **Documentation:** This README and inline code documentation
- **Community:** Check existing issues and discussions

---

**Built with ❤️ for Enterprise Architecture in Energy Systems**

*AInstein AI Assistant - Intelligent, Grounded, Production-Ready*