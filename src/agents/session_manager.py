"""
Session Manager - Maintains conversation context for follow-up queries.

This module enables the EA Assistant to understand follow-up questions and
maintain conversation context across multiple turns.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single conversation turn with full context."""
    query: str
    response: str
    citations: List[str]
    timestamp: datetime
    confidence: float
    route: str = "unknown"
    processing_time_ms: float = 0.0
    key_concepts: List[str] = field(default_factory=list)


class SessionManager:
    """
    Manages conversation sessions with context memory.
    
    Features:
    - Conversation history tracking
    - Follow-up query detection
    - Context building for enhanced understanding
    - Automatic session cleanup
    """
    
    def __init__(self, max_history: int = 5, session_timeout_minutes: int = 30):
        """
        Initialize session manager.
        
        Args:
            max_history: Maximum number of turns to remember per session
            session_timeout_minutes: Auto-clear sessions after this time
        """
        self.sessions: Dict[str, List[ConversationTurn]] = {}
        self.max_history = max_history
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        logger.info(f"SessionManager initialized (max_history={max_history}, timeout={session_timeout_minutes}m)")
    
    def add_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        citations: List[str],
        confidence: float,
        route: str = "unknown",
        processing_time_ms: float = 0.0
    ):
        """
        Add a conversation turn to session history.
        
        Args:
            session_id: Session identifier
            query: User query
            response: System response
            citations: List of citations used
            confidence: Response confidence score
            route: Query routing decision
            processing_time_ms: Processing time
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            logger.info(f"Created new session: {session_id}")
        
        # Extract key concepts from query and citations
        key_concepts = self._extract_key_concepts(query, citations)
        
        turn = ConversationTurn(
            query=query,
            response=response,
            citations=citations,
            timestamp=datetime.utcnow(),
            confidence=confidence,
            route=route,
            processing_time_ms=processing_time_ms,
            key_concepts=key_concepts
        )
        
        self.sessions[session_id].append(turn)
        
        # Trim history if too long
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
            logger.debug(f"Trimmed session {session_id} to {self.max_history} turns")
        
        logger.debug(f"Added turn to session {session_id} (total: {len(self.sessions[session_id])} turns)")
    
    def get_history(self, session_id: str) -> List[ConversationTurn]:
        """Get conversation history for session."""
        # Clean up old sessions first
        self._cleanup_old_sessions()
        
        return self.sessions.get(session_id, [])
    
    def get_context_for_query(self, session_id: str, current_query: str) -> Tuple[str, bool]:
        """
        Build context string for current query based on history.
        
        This is the KEY method that enables follow-up understanding.
        
        Args:
            session_id: Session identifier
            current_query: Current user query
            
        Returns:
            Tuple of (context_string, is_followup)
        """
        history = self.get_history(session_id)
        if not history:
            return "", False
        
        # Detect if this is a follow-up question
        is_followup = self._is_followup_query(current_query, history)
        
        if not is_followup:
            return "", False
        
        logger.info(f"Detected follow-up query in session {session_id}")
        
        # Build rich context from recent turns
        context_lines = ["### CONVERSATION CONTEXT ###"]
        context_lines.append("Previous discussion in this session:")
        context_lines.append("")
        
        # Include last 3 turns for context
        recent_turns = history[-3:]
        
        for i, turn in enumerate(recent_turns, 1):
            context_lines.append(f"**Previous Q{i}:** {turn.query}")
            
            # Include concise response summary
            response_summary = self._summarize_response(turn.response)
            context_lines.append(f"**Previous A{i}:** {response_summary}")
            
            # Include key concepts
            if turn.key_concepts:
                context_lines.append(f"**Concepts discussed:** {', '.join(turn.key_concepts[:5])}")
            
            # Include important citations
            if turn.citations:
                context_lines.append(f"**Citations used:** {', '.join(turn.citations[:3])}")
            
            context_lines.append("")
        
        context_lines.append("### CURRENT QUERY ###")
        context_lines.append(f"**Current question:** {current_query}")
        context_lines.append("")
        
        # Add comparison instructions if detected
        if self._is_comparison_query(current_query):
            context_lines.append("**Note:** This is a comparison query. Please:")
            context_lines.append("1. Define each concept clearly")
            context_lines.append("2. Explain key differences")
            context_lines.append("3. Show relationships between concepts")
            context_lines.append("4. Use citations for all information")
            context_lines.append("")
        
        return "\n".join(context_lines), True
    
    def _extract_key_concepts(self, query: str, citations: List[str]) -> List[str]:
        """Extract key concepts from query and citations."""
        concepts = []
        
        # Extract from citations
        for citation in citations[:5]:  # Top 5 citations
            if ':' in citation:
                # Extract the concept part after the prefix
                concept = citation.split(':')[-1]
                concepts.append(concept)
        
        # Extract important words from query (simple approach)
        query_words = query.lower().split()
        important_words = [
            word for word in query_words
            if len(word) > 4 and word not in {
                'what', 'which', 'where', 'when', 'about', 'between',
                'difference', 'power', 'should', 'would', 'could'
            }
        ]
        concepts.extend(important_words[:3])
        
        return concepts
    
    def _summarize_response(self, response: str, max_length: int = 200) -> str:
        """Summarize a response for context."""
        # Remove markdown formatting
        clean = response.replace('**', '').replace('##', '').replace('###', '')
        
        # Take first paragraph or max_length chars
        first_para = clean.split('\n\n')[0]
        
        if len(first_para) > max_length:
            return first_para[:max_length] + "..."
        
        return first_para
    
    def _is_followup_query(self, query: str, history: List[ConversationTurn]) -> bool:
        """
        Detect if query is a follow-up question.
        
        Uses multiple signals:
        1. Linguistic patterns
        2. Concept overlap with history
        3. Query structure
        """
        query_lower = query.lower()
        
        # Strong follow-up indicators
        strong_patterns = [
            "difference between",
            "compare",
            "versus",
            "vs",
            "how does it differ",
            "in contrast to",
            "compared to",
            "what about",
            "and what is",
            "how about",
            "what's the difference"
        ]
        
        if any(pattern in query_lower for pattern in strong_patterns):
            return True
        
        # Weak follow-up indicators (need concept overlap)
        weak_patterns = [
            "also",
            "additionally",
            "furthermore",
            "similarly",
            "on the other hand",
            "however",
            "but",
            "and"
        ]
        
        has_weak_pattern = any(pattern in query_lower for pattern in weak_patterns)
        
        # Check concept overlap with recent history
        if has_weak_pattern and history:
            recent_concepts = []
            for turn in history[-2:]:
                recent_concepts.extend(turn.key_concepts)
            
            # Check if current query mentions previous concepts
            query_words = set(query_lower.split())
            concept_overlap = any(
                concept.lower() in query_words
                for concept in recent_concepts
            )
            
            if concept_overlap:
                return True
        
        return False
    
    def _is_comparison_query(self, query: str) -> bool:
        """Detect if query is asking for comparison."""
        query_lower = query.lower()
        
        comparison_patterns = [
            "difference between",
            "compare",
            "versus",
            "vs",
            "differ from",
            "contrast",
            "compared to",
            "what's the difference",
            "how do they differ"
        ]
        
        return any(pattern in query_lower for pattern in comparison_patterns)
    
    def _cleanup_old_sessions(self):
        """Remove sessions that have timed out."""
        now = datetime.utcnow()
        to_remove = []
        
        for session_id, turns in self.sessions.items():
            if not turns:
                to_remove.append(session_id)
                continue
            
            last_activity = turns[-1].timestamp
            if now - last_activity > self.session_timeout:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
    
    def clear_session(self, session_id: str):
        """Clear a specific session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session."""
        history = self.get_history(session_id)
        
        if not history:
            return {
                "exists": False,
                "turns": 0
            }
        
        total_citations = sum(len(t.citations) for t in history)
        avg_confidence = sum(t.confidence for t in history) / len(history)
        avg_processing_time = sum(t.processing_time_ms for t in history) / len(history)
        
        duration_minutes = 0.0
        if len(history) > 1:
            duration_minutes = (history[-1].timestamp - history[0].timestamp).total_seconds() / 60
        
        return {
            "exists": True,
            "turns": len(history),
            "avg_confidence": round(avg_confidence, 2),
            "avg_processing_time_ms": round(avg_processing_time, 1),
            "total_citations": total_citations,
            "duration_minutes": round(duration_minutes, 1),
            "last_activity": history[-1].timestamp.isoformat(),
            "key_concepts_discussed": list(set(
                concept
                for turn in history
                for concept in turn.key_concepts
            ))[:10]
        }
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all active session IDs."""
        self._cleanup_old_sessions()
        return list(self.sessions.keys())