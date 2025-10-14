#!/usr/bin/env python3
"""
FastAPI Web Interface for Alliander EA Assistant

Modern chat interface with WebSocket support, showcasing safety features:
- Real-time grounding enforcement with citations
- Confidence assessment with visual indicators
- Human review alerts for low confidence responses
- Professional energy sector UI with Alliander branding
- FULL PIPELINE TRACE VISUALIZATION
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import EA Assistant components FIRST
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.agents.ea_assistant import ProductionEAAgent
    from src.exceptions.exceptions import UngroundedReplyError, LowConfidenceError, FakeCitationError
    EA_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: EA Agent not available: {e}")
    EA_AGENT_AVAILABLE = False

# NOW force reload after first import
import importlib
if EA_AGENT_AVAILABLE:
    importlib.reload(sys.modules['src.agent.ea_assistant'])
    print("✅ Reloaded ea_assistant module with fixes")
    # Re-import after reload
    from src.agents.ea_assistant import ProductionEAAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app with energy sector branding
app = FastAPI(
    title="Alliander EA Assistant Web Interface",
    description="Enterprise Architecture AI Assistant for Energy Systems - Debug Mode",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Static files and templates
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Global EA Assistant instance - INITIALIZE ONLY ONCE
ea_agent: Optional[ProductionEAAgent] = None
if EA_AGENT_AVAILABLE:
    try:
        ea_agent = ProductionEAAgent()
        logger.info("✅ EA Assistant initialized successfully with trace support")
    except Exception as e:
        logger.error(f"❌ Failed to initialize EA Assistant: {e}")
        ea_agent = None

# Active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_data: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.session_data[session_id] = {
            "messages": [],
            "traces": [],  # NEW: Store traces
            "created_at": datetime.now(),
            "total_queries": 0
        }
        logger.info(f"🔌 WebSocket connected: {session_id}")

    def disconnect(self, websocket: WebSocket, session_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if session_id in self.session_data:
            del self.session_data[session_id]
        logger.info(f"🔌 WebSocket disconnected: {session_id}")

    async def send_message(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "title": "Alliander EA Assistant - Debug Mode",
        "agent_available": ea_agent is not None
    })


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy" if ea_agent else "unavailable",
        "timestamp": datetime.now().isoformat(),
        "ea_agent_available": ea_agent is not None,
        "active_connections": len(manager.active_connections),
        "trace_enabled": True
    }


@app.get("/api/stats")
async def get_stats():
    """Get system statistics for dashboard."""
    if not ea_agent:
        return {"error": "EA Agent not available"}

    try:
        # Get comprehensive statistics
        stats = ea_agent.get_statistics()

        return {
            "system": {
                "knowledge_graph_loaded": stats.get("knowledge_graph", {}).get("triple_count", 0) > 0,
                "citation_pools_loaded": stats.get("citation_pools_loaded", 0),
                "active_sessions": len(manager.session_data),
                "trace_enabled": True
            },
            "performance": {
                "citation_validator_available": stats.get("citation_validator_available", False),
                "llm_council_available": hasattr(ea_agent, 'llm_council') and ea_agent.llm_council is not None
            },
            "details": stats
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": str(e)}


@app.post("/api/export/{session_id}")
async def export_conversation(session_id: str, format: str = "markdown"):
    """Export conversation history with trace data."""
    if session_id not in manager.session_data:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    session = manager.session_data[session_id]
    messages = session["messages"]
    traces = session.get("traces", [])

    if format == "markdown":
        # Generate markdown export with trace info
        md_content = f"# EA Assistant Conversation - Debug Mode\n\n"
        md_content += f"**Session:** {session_id}\n"
        md_content += f"**Date:** {session['created_at'].strftime('%Y-%m-%d %H:%M:%S')}\n"
        md_content += f"**Total Queries:** {session['total_queries']}\n\n"
        md_content += "---\n\n"

        for idx, msg in enumerate(messages):
            if msg["type"] == "user":
                md_content += f"## User Query\n\n{msg['content']}\n\n"
            elif msg["type"] == "assistant":
                md_content += f"## EA Assistant Response\n\n"
                md_content += f"**Confidence:** {msg.get('confidence', 'N/A')}\n"
                md_content += f"**Route:** {msg.get('route', 'N/A')}\n"
                md_content += f"**Processing Time:** {msg.get('processing_time_ms', 0):.0f}ms\n\n"
                md_content += f"{msg['content']}\n\n"

                if msg.get('citations'):
                    md_content += f"**Citations:** {', '.join(msg['citations'])}\n\n"

                if msg.get('requires_human_review'):
                    md_content += f"⚠️ **Human Review Required**\n\n"
                
                # Add trace summary if available
                if idx < len(traces):
                    trace = traces[idx]
                    md_content += f"### Pipeline Trace\n\n"
                    md_content += f"Total Duration: {trace.get('total_duration_ms', 0):.0f}ms\n\n"
                    for phase in trace.get('phases', []):
                        md_content += f"- **{phase['name']}**: {phase['duration_ms']:.0f}ms [{phase['status']}]\n"
                    md_content += "\n"

                md_content += "---\n\n"

        return {
            "content": md_content,
            "filename": f"ea_assistant_chat_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        }
    
    elif format == "json":
        # Export as JSON with full trace data
        export_data = {
            "session_id": session_id,
            "created_at": session['created_at'].isoformat(),
            "total_queries": session['total_queries'],
            "messages": messages,
            "traces": traces
        }
        
        return {
            "content": json.dumps(export_data, indent=2),
            "filename": f"ea_assistant_chat_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }

    return JSONResponse({"error": "Unsupported format"}, status_code=400)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat with FULL TRACE SUPPORT.
    
    This handles the tuple return from process_query() and sends:
    1. The main response message
    2. A separate trace message with pipeline details
    """
    await manager.connect(websocket, session_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            user_query = message_data.get("message", "").strip()
            if not user_query:
                continue

            # Store user message
            user_message = {
                "type": "user",
                "content": user_query,
                "timestamp": datetime.now().isoformat()
            }
            manager.session_data[session_id]["messages"].append(user_message)
            manager.session_data[session_id]["total_queries"] += 1

            # Send typing indicator
            await manager.send_message(websocket, {
                "type": "typing",
                "content": "EA Assistant is processing your query..."
            })

            try:
                if ea_agent:
                    # Process query through EA Assistant
                    logger.info(f"Processing query: {user_query}")

                    try:
                        # ============================================
                        # CRITICAL: Unpack tuple return
                        # ============================================
                        response, trace = await ea_agent.process_query(user_query, session_id)

                        # Build response message
                        response_message = {
                            "type": "assistant",
                            "content": response.response,
                            "timestamp": datetime.now().isoformat(),
                            "confidence": round(response.confidence, 3),
                            "citations": response.citations,
                            "requires_human_review": response.requires_human_review,
                            "grounding_status": "grounded" if response.citations else "ungrounded",
                            "route": response.route,
                            "togaf_phase": response.togaf_phase,
                            "archimate_elements": response.archimate_elements,
                            "processing_time_ms": response.processing_time_ms
                        }
                        
                        # Send the main response
                        await manager.send_message(websocket, response_message)
                        
                        # ============================================
                        # NEW: Send trace data separately
                        # ============================================
                        trace_message = {
                            "type": "trace",
                            "trace": trace.to_dict(),
                            "timestamp": datetime.now().isoformat()
                        }
                        await manager.send_message(websocket, trace_message)
                        
                        # Store both response and trace
                        manager.session_data[session_id]["messages"].append(response_message)
                        manager.session_data[session_id]["traces"].append(trace.to_dict())
                        
                        logger.info(f"✅ Query processed successfully in {response.processing_time_ms:.0f}ms")
                        logger.info(f"   Phases: {len(trace.phases)}, Citations: {len(response.citations)}, Confidence: {response.confidence:.2f}")

                    except FakeCitationError as e:
                        # Handle fake citation detection
                        logger.error(f"❌ FAKE CITATIONS DETECTED: {e.fake_citations}")
                        
                        response_message = {
                            "type": "error",
                            "content": (
                                "🚫 Citation Validation Failed\n\n"
                                "The system detected fabricated citations in the response. "
                                "This query requires human expert review.\n\n"
                                f"Technical details: {len(e.fake_citations)} fake citation(s) detected. "
                                f"Valid pool contained {len(e.valid_pool)} authentic citations."
                            ),
                            "timestamp": datetime.now().isoformat(),
                            "error_type": "fake_citation",
                            "details": {
                                "fake_citations": e.fake_citations,
                                "valid_pool_size": len(e.valid_pool)
                            }
                        }
                        await manager.send_message(websocket, response_message)

                    except UngroundedReplyError as e:
                        # Handle ungrounded response
                        logger.error(f"❌ UNGROUNDED REPLY: {e}")
                        
                        response_message = {
                            "type": "error",
                            "content": f"⚠️ Unable to provide grounded response: {str(e)}",
                            "timestamp": datetime.now().isoformat(),
                            "error_type": "ungrounded_reply"
                        }
                        await manager.send_message(websocket, response_message)

                    except LowConfidenceError as e:
                        # Handle low confidence
                        logger.warning(f"⚠️ LOW CONFIDENCE: {e}")
                        
                        response_message = {
                            "type": "assistant",
                            "content": (
                                "I have limited confidence in this response. "
                                "Human review is recommended.\n\n"
                                f"Reason: {str(e)}"
                            ),
                            "timestamp": datetime.now().isoformat(),
                            "confidence": 0.4,
                            "requires_human_review": True,
                            "error_type": "low_confidence"
                        }
                        await manager.send_message(websocket, response_message)

                else:
                    # Fallback when EA Agent not available
                    response_message = {
                        "type": "error",
                        "content": (
                            "EA Assistant is currently not available. "
                            f"Your query '{user_query}' has been received but cannot be processed. "
                            "Please ensure the knowledge graph is properly loaded."
                        ),
                        "timestamp": datetime.now().isoformat(),
                        "confidence": 0.0,
                        "requires_human_review": True,
                        "citations": [],
                        "grounding_status": "ungrounded",
                        "error_type": "agent_unavailable"
                    }
                    await manager.send_message(websocket, response_message)

            except Exception as e:
                logger.error(f"❌ Error processing query: {e}", exc_info=True)
                
                response_message = {
                    "type": "error",
                    "content": f"System error: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "error_type": "system_error"
                }
                await manager.send_message(websocket, response_message)

    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket, session_id)


def main():
    """Run the web server."""
    print("🚀 Starting Alliander EA Assistant Web Interface")
    print("=" * 70)
    print("📡 Available at: http://localhost:8000")
    print("📊 API docs at: http://localhost:8000/api/docs")
    print("🔌 WebSocket at: ws://localhost:8000/ws/{session_id}")
    print("🔍 TRACE MODE: ENABLED")
    print("=" * 70)

    uvicorn.run(
        app,  # Pass app directly
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()