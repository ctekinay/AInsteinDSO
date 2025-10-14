# AInstein AI Assistant for Alliander

**Advanced Enterprise Architecture AI Assistant for Energy Systems**

An intelligent AI assistant designed for enterprise architecture consulting in the energy sector, featuring multi-LLM architecture, comprehensive knowledge graphs, and real-time web interface.

## 🌟 Features

- **Multi-LLM Architecture** - Groq integration with Llama 3.3, Qwen 3, Kimi K2 models
- **Knowledge Graph Integration** - 39,000+ energy domain triples with IEC standards
- **Citation Validation** - Grounded responses with authentic source verification
- **Real-time Web Interface** - Interactive chat with trace visualization
- **TOGAF Methodology** - Enterprise architecture framework integration
- **Multilingual Support** - English and Dutch language detection

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

Copy the example configuration file:
```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:
```env
# Primary provider - Groq (recommended, fast and free tier available)
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here

# Optional - OpenAI (backup)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5
```

### Step 4: Get Your API Keys

**For Groq (Recommended):**
1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up for free
3. Go to API Keys section
4. Create new API key
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
2. **Ask questions** about energy systems, enterprise architecture, or electrical engineering
3. **Get cited responses** with authentic sources
4. **View processing trace** to see how the AI works

**Example questions:**
- "What is reactive power in electrical systems?"
- "How do I model a transformer in ArchiMate?"
- "Explain the IEC 61968 standard"
- "What are the TOGAF phases for energy architecture?"

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

# Performance
EA_LOG_LEVEL=INFO              # DEBUG for detailed logs
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

Run this to test everything works:
```bash
python test_integration.py
```

## 📁 Project Structure

```
AInsteinDSO/
├── src/                    # Main application code
├── data/                   # Knowledge graphs and models
├── config/                 # Configuration files
├── tests/                  # Test files
├── run_web_demo.py         # Start the web interface
├── cli.py                  # Command line interface
├── .env                    # Your API keys (create this)
└── requirements.txt        # Python packages needed
```

## 🤝 Need Help?

- **Check the logs** - Look at the terminal output for error messages
- **Verify your setup** - Make sure Python 3.11+, API keys are correct
- **Try simple test** - Run `python cli.py` first to test basic functionality
- **Open an issue** - Report problems on GitHub

---

**Built for Enterprise Architecture in Energy Systems**

*Simple Python installation - no Docker needed!*