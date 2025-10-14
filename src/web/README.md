# Alliander EA Assistant Web Interface

Modern chat interface showcasing the safety features and professional capabilities of the Enterprise Architecture AI Assistant.

## Features

### 🎨 Modern Design
- **Glassmorphism effects** with energy sector branding
- **Dark/light theme toggle** for different preferences
- **Responsive design** works on desktop, tablet, and mobile
- **Alliander colors** (blue, green, teal) for professional appearance

### 🔒 Safety Features Showcase
- **Citation pills** showing source references (clickable badges)
- **Confidence indicators** with color-coded bars (green >75%, yellow 50-75%, red <50%)
- **Human review banners** when confidence falls below threshold
- **Grounding status** visible for each response
- **Real-time safety validation** with WebSocket communication

### 💬 Chat Features
- **Real-time messaging** via WebSocket
- **Typing indicators** during processing
- **Message bubbles** (user right, assistant left)
- **Copy buttons** for individual messages
- **Export conversation** as Markdown or JSON
- **Session management** with unique IDs

### 🔗 Citation System
- **Clickable citation badges**: `[iec:xxx]` `[archi:id-xxx]` `[togaf:adm:xxx]`
- **Hover tooltips** showing source information
- **Copy to clipboard** functionality
- **Source validation** ensuring all responses are grounded

## Quick Start

### Option 1: Using the Demo Script
```bash
python run_web_demo.py
```

### Option 2: Direct Start
```bash
# Install dependencies (if not already installed)
pip install fastapi uvicorn jinja2

# Start the web server
python -m src.web.app
```

### Option 3: Using Poetry
```bash
poetry install
poetry run python -m src.web.app
```

## Access Points

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/health
- **WebSocket**: ws://localhost:8000/ws/{session_id}

## Sample Queries for Demo

Try these queries to showcase different features:

### 1. Energy Domain Queries
```
What capability should I use for grid congestion management?
```
**Expected**: Citations like `iec:GridCongestion`, `archi:id-cap-001`

### 2. ArchiMate Modeling
```
How do I model SCADA systems in ArchiMate?
```
**Expected**: TOGAF citations, confidence indicators

### 3. TOGAF Methodology
```
What TOGAF phase covers business architecture?
```
**Expected**: `togaf:adm:PhaseB` citations

### 4. Standards References
```
Show me IEC standards for power measurement
```
**Expected**: Multiple IEC citations, high confidence

## Architecture

```
src/web/
├── app.py              # FastAPI application with WebSocket
├── templates/
│   └── chat.html       # Single-page chat interface
├── static/             # Static assets (minimal)
└── README.md          # This file
```

### Key Components

1. **FastAPI Backend** (`app.py`)
   - WebSocket for real-time chat
   - REST endpoints for export/stats
   - Integration with ProductionEAAgent
   - Session management

2. **Chat Interface** (`templates/chat.html`)
   - Modern CSS with glassmorphism
   - JavaScript WebSocket client
   - Real-time UI updates
   - Theme switching

3. **Safety Integration**
   - Grounding check visualization
   - Confidence assessment display
   - Citation badge rendering
   - Human review alerts

## Customization

### Theme Colors
Modify CSS variables in `chat.html`:
```css
:root {
    --primary-blue: #0066cc;    /* Alliander primary */
    --secondary-green: #00a86b; /* Energy green */
    --accent-teal: #20b2aa;     /* Tech accent */
}
```

### Citation Styles
Update `.citation-pill` CSS class for different badge appearances.

### Confidence Thresholds
Modify confidence levels in JavaScript:
```javascript
if (confidence >= 0.75) {        // High confidence (green)
    fill.classList.add('high');
} else if (confidence >= 0.5) {  // Medium confidence (yellow)
    fill.classList.add('medium');
} else {                         // Low confidence (red)
    fill.classList.add('low');
}
```

## Production Deployment

### Environment Variables
```bash
export EA_ASSISTANT_HOST=0.0.0.0
export EA_ASSISTANT_PORT=8000
export EA_ASSISTANT_LOG_LEVEL=info
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install fastapi uvicorn jinja2
EXPOSE 8000
CMD ["python", "-m", "src.web.app"]
```

### Security Considerations
- Enable HTTPS in production
- Configure CORS for specific domains
- Add authentication if needed
- Rate limiting for WebSocket connections

## Troubleshooting

### Common Issues

1. **Dependencies Missing**
   ```
   pip install fastapi uvicorn jinja2
   ```

2. **Knowledge Graph Not Found**
   - Ensure `data/energy_knowledge_graph.ttl` exists
   - Demo will work with mock responses if KG missing

3. **Port Already in Use**
   - Change port in `app.py`: `uvicorn.run(..., port=8001)`

4. **WebSocket Connection Failed**
   - Check firewall settings
   - Ensure no proxy blocking WebSocket

### Debug Mode
Start with debug logging:
```bash
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from src.web.app import main
main()
"
```

## Features Demonstrated

### For Stakeholders
- **Professional UI** matching enterprise standards
- **Real-time interaction** with immediate feedback
- **Safety-first approach** with visible citations
- **Quality indicators** showing confidence levels

### For Technical Teams
- **Modern web stack** (FastAPI, WebSocket, responsive CSS)
- **Clean architecture** with separated concerns
- **Production-ready** error handling and logging
- **Extensible design** for additional features

## Integration Points

The web interface integrates with:
- **ProductionEAAgent** for query processing
- **QueryRouter** for intelligent routing
- **GroundingCheck** for citation validation
- **Critic** for confidence assessment
- **Knowledge Graph** for domain expertise

This showcases the complete EA Assistant pipeline in an accessible, stakeholder-friendly format.