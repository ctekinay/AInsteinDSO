# Embedding Integration Fix Plan

**Branch:** `esa--embeddings-fix`
**Priority:** HIGH
**Estimated Effort:** 2-3 hours
**Risk Level:** MEDIUM (touching core pipeline)

---

## Problem Statement

Based on comprehensive system analysis, two critical issues prevent quality responses:

1. **Comparison Query Bug**: System compares "active power vs active power" instead of "active vs reactive power"
2. **Embedding Underutilization**: Semantic search only used as fallback, missing context enhancement opportunities

**Root Causes Identified:**
- Citation extraction regex fails on comparison response format
- Knowledge graph queries timeout (10s limit)
- Embeddings treated as secondary instead of primary context source
- Duplicate candidate retrieval for comparison queries

---

## Implementation Strategy

### Phase 1: Critical Bug Fixes (30 minutes)
**Goal**: Fix comparison queries to work correctly

### Phase 2: Embedding Integration Enhancement (90 minutes)
**Goal**: Make embeddings primary for context, not just fallback

### Phase 3: Validation and Testing (30 minutes)
**Goal**: Ensure fixes work without breaking existing functionality

---

## Detailed Implementation Plan

### PHASE 1: Critical Bug Fixes

#### 1.1 Fix Citation Extraction for Comparison Responses
**File**: `src/safety/grounding.py`
**Issue**: Regex cannot extract citations from format `**Active power** [eurlex:631-20]:`

**Current Regex:**
```python
citation_patterns = [
    r'\b(archi:id-[a-zA-Z0-9\-]+)',
    r'\b(skos:[a-zA-Z0-9\-:]+)',
    # ...
]
```

**Fix**: Add pattern for embedded citations in comparison format
```python
citation_patterns = [
    # Existing patterns...
    r'\[([a-zA-Z0-9\-:]+)\]',  # NEW: Embedded citations in brackets
    r'\*\*.*?\*\*\s*\[([a-zA-Z0-9\-:]+)\]',  # NEW: Bold text with citation
]
```

**Risk**: LOW - Adding patterns, not changing existing ones
**Test**: Verify comparison responses pass grounding check

#### 1.2 Fix Duplicate Candidate Retrieval for Comparisons
**File**: `src/agents/ea_assistant.py`
**Method**: `_refine_response_for_comparison()`

**Current Issue**: Takes candidates[0] and candidates[1] without validation

**Fix**: Add distinct concept validation
```python
def _validate_comparison_candidates(self, candidates: List[Dict], query: str) -> Tuple[Dict, Dict]:
    """Ensure we have two distinct concepts for comparison."""

    # Extract comparison terms from query
    comparison_terms = self._extract_comparison_terms(query)

    # Find candidates matching each term
    term1_candidates = []
    term2_candidates = []

    for candidate in candidates:
        element = candidate.get('element', '').lower()
        for term in comparison_terms:
            if term.lower() in element and candidate not in term1_candidates:
                term1_candidates.append(candidate)
            elif len(comparison_terms) > 1 and comparison_terms[1].lower() in element:
                term2_candidates.append(candidate)

    # Return best match for each term or use embeddings as fallback
    if len(term1_candidates) > 0 and len(term2_candidates) > 0:
        return term1_candidates[0], term2_candidates[0]
    else:
        # Use semantic search to find distinct concepts
        return self._semantic_comparison_fallback(query, candidates)
```

**Risk**: MEDIUM - Changes comparison logic
**Test**: "difference between active and reactive power" returns distinct concepts

#### 1.3 Add Comparison Term Extraction
**File**: `src/agents/ea_assistant.py`
**New Method**: `_extract_comparison_terms()`

```python
def _extract_comparison_terms(self, query: str) -> List[str]:
    """Extract the two terms being compared from query."""
    query_lower = query.lower()

    # Patterns for "A and B", "A vs B", "between A and B"
    patterns = [
        r'between\s+(.+?)\s+and\s+(.+?)(?:\?|$)',
        r'(.+?)\s+(?:vs|versus)\s+(.+?)(?:\?|$)',
        r'(.+?)\s+and\s+(.+?)(?:\?|$)',
        r'compare\s+(.+?)\s+(?:to|with)\s+(.+?)(?:\?|$)'
    ]

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            term1 = match.group(1).strip()
            term2 = match.group(2).strip()

            # Clean up terms (remove stop words, etc.)
            term1 = self._clean_comparison_term(term1)
            term2 = self._clean_comparison_term(term2)

            return [term1, term2]

    return []
```

**Risk**: LOW - New functionality, doesn't affect existing code
**Test**: Correctly extracts "active power" and "reactive power"

### PHASE 2: Embedding Integration Enhancement

#### 2.1 Make Embeddings Primary for Context Enhancement
**File**: `src/agents/ea_assistant.py`
**Method**: `_retrieve_knowledge()`

**Current Logic**: Use embeddings only if structured results < 3
**New Logic**: Always use embeddings for context enhancement

**Implementation**:
```python
async def _retrieve_knowledge_enhanced(self, query: str, session_id: str = None, trace_id: str = None) -> Dict:
    """Enhanced retrieval with embeddings as primary context source."""

    retrieval_context = {"candidates": [], "sources": []}

    # PHASE 1: Structured retrieval (existing logic)
    structured_results = await self._structured_retrieval(query)
    retrieval_context["candidates"].extend(structured_results)

    # PHASE 2: Semantic enhancement (NEW - always run)
    if self.embedding_agent:
        semantic_results = await self._semantic_enhancement(query, structured_results)
        retrieval_context["candidates"].extend(semantic_results)
        retrieval_context["semantic_enhanced"] = True

    # PHASE 3: Context expansion for follow-ups (ENHANCED)
    if session_id:
        context_results = await self._context_expansion(query, session_id)
        retrieval_context["candidates"].extend(context_results)

    # PHASE 4: Ranking and deduplication
    retrieval_context["candidates"] = self._rank_and_deduplicate(
        retrieval_context["candidates"], query
    )

    return retrieval_context
```

**Risk**: MEDIUM - Changes core retrieval logic
**Test**: Verify improved context without breaking existing queries

#### 2.2 Add Semantic Query Expansion
**File**: `src/agents/ea_assistant.py`
**New Method**: `_semantic_enhancement()`

```python
async def _semantic_enhancement(self, query: str, structured_results: List[Dict]) -> List[Dict]:
    """Use embeddings to find semantically related concepts."""

    semantic_candidates = []

    # Get semantic matches
    semantic_results = self.embedding_agent.semantic_search(
        query,
        top_k=5,
        min_score=0.3  # Lower threshold for more context
    )

    # Convert to candidate format
    for result in semantic_results:
        # Skip if we already have this concept from structured search
        if not self._is_duplicate_concept(result, structured_results):
            candidate = {
                "element": result.text[:100],
                "type": "Semantic Enhancement",
                "citation": result.citation or "semantic:context",
                "confidence": result.score,
                "definition": result.text,
                "source": f"Semantic Search ({result.source})",
                "priority": "context"  # Different from main results
            }
            semantic_candidates.append(candidate)

    return semantic_candidates
```

**Risk**: LOW - Additive functionality
**Test**: Verify semantic results enhance context without overwhelming

#### 2.3 Improve Comparison with Semantic Fallback
**File**: `src/agents/ea_assistant.py`
**New Method**: `_semantic_comparison_fallback()`

```python
async def _semantic_comparison_fallback(self, query: str, existing_candidates: List[Dict]) -> Tuple[Dict, Dict]:
    """Use semantic search to find distinct concepts for comparison."""

    # Extract comparison terms
    comparison_terms = self._extract_comparison_terms(query)

    if len(comparison_terms) != 2:
        # If we can't extract terms, use first two candidates
        return existing_candidates[0], existing_candidates[1] if len(existing_candidates) > 1 else existing_candidates[0]

    # Search for each term separately using embeddings
    concept1_results = self.embedding_agent.semantic_search(comparison_terms[0], top_k=3, min_score=0.4)
    concept2_results = self.embedding_agent.semantic_search(comparison_terms[1], top_k=3, min_score=0.4)

    # Convert best results to candidate format
    candidate1 = self._semantic_result_to_candidate(concept1_results[0]) if concept1_results else existing_candidates[0]
    candidate2 = self._semantic_result_to_candidate(concept2_results[0]) if concept2_results else existing_candidates[1] if len(existing_candidates) > 1 else existing_candidates[0]

    return candidate1, candidate2
```

**Risk**: MEDIUM - Fallback mechanism for comparisons
**Test**: "difference between active and reactive power" finds distinct concepts via embeddings

#### 2.4 Add Context Expansion for Follow-up Queries
**File**: `src/agents/ea_assistant.py`
**New Method**: `_context_expansion()`

```python
async def _context_expansion(self, query: str, session_id: str) -> List[Dict]:
    """Expand context using conversation history and semantic similarity."""

    context_candidates = []

    # Get conversation history
    session_data = self.session_manager.get_session_data(session_id)
    if not session_data or len(session_data.get("messages", [])) < 2:
        return context_candidates

    # Extract concepts from previous queries
    previous_concepts = []
    for message in session_data["messages"][-4:]:  # Last 4 messages
        if message["type"] == "user":
            concepts = self._extract_query_terms(message["content"])
            previous_concepts.extend(concepts)

    # Use embeddings to find related concepts
    if previous_concepts and self.embedding_agent:
        related_query = f"{query} {' '.join(previous_concepts[:3])}"
        related_results = self.embedding_agent.semantic_search(
            related_query,
            top_k=3,
            min_score=0.35
        )

        # Add as context candidates
        for result in related_results:
            candidate = {
                "element": result.text[:100],
                "type": "Context Enhancement",
                "citation": result.citation or "context:related",
                "confidence": result.score,
                "definition": result.text,
                "source": f"Conversation Context ({result.source})",
                "priority": "context"
            }
            context_candidates.append(candidate)

    return context_candidates
```

**Risk**: LOW - Enhancement for follow-up queries
**Test**: Follow-up questions get better context from conversation history

### PHASE 3: Validation and Testing

#### 3.1 Comprehensive Test Cases

**Test Case 1: Basic Comparison Query**
```
Input: "What is the difference between active power and reactive power?"
Expected: Two distinct concepts with proper definitions and citations
Validation: Response contains both "active power" AND "reactive power" definitions
```

**Test Case 2: Embedding Enhancement**
```
Input: "What is voltage stability?"
Expected: Main definition + related concepts from embeddings
Validation: Response enhanced with semantically related context
```

**Test Case 3: Follow-up Context**
```
Input 1: "What is reactive power?"
Input 2: "How does it affect grid stability?"
Expected: Second response uses context from first query
Validation: Better quality response due to conversation context
```

**Test Case 4: Citation Grounding**
```
Input: Any comparison query
Expected: All responses pass grounding check
Validation: No UngroundedReplyError exceptions
```

#### 3.2 Performance Validation

**Metrics to Monitor:**
- Response time: Should not increase by more than 1 second
- Citation extraction: 100% success rate for comparison responses
- Semantic enhancement: Activated for all queries when embedding_agent available
- Context expansion: Activated for follow-up queries

**Performance Thresholds:**
- Total response time: < 5 seconds for comparison queries
- Embedding search: < 200ms
- Citation extraction: < 100ms
- Memory usage: No significant increase

#### 3.3 Rollback Plan

**If Issues Occur:**
1. **Immediate**: Disable semantic enhancement via environment variable
2. **Short-term**: Revert to original comparison logic
3. **Long-term**: Branch rollback to main

**Rollback Triggers:**
- Response time > 8 seconds
- Citation extraction failures
- System errors or crashes
- Unacceptable response quality

---

## Implementation Checklist

### Pre-Implementation
- [ ] Backup current working state
- [ ] Set up test environment
- [ ] Prepare test queries and expected results

### Phase 1: Critical Bug Fixes
- [ ] Update citation extraction regex
- [ ] Add comparison term extraction method
- [ ] Implement distinct candidate validation
- [ ] Test comparison queries work correctly

### Phase 2: Embedding Enhancement
- [ ] Refactor retrieval logic for embedding priority
- [ ] Implement semantic enhancement method
- [ ] Add semantic comparison fallback
- [ ] Add context expansion for follow-ups
- [ ] Implement ranking and deduplication

### Phase 3: Validation
- [ ] Run comprehensive test suite
- [ ] Validate performance metrics
- [ ] Check citation grounding success
- [ ] Test edge cases and error scenarios

### Post-Implementation
- [ ] Monitor performance in development
- [ ] Document changes and new behavior
- [ ] Prepare deployment notes

---

## Risk Mitigation

### High-Risk Changes
1. **Citation extraction modification**: Thoroughly test all citation patterns
2. **Core retrieval logic changes**: Maintain backward compatibility
3. **Comparison logic overhaul**: Ensure existing queries still work

### Mitigation Strategies
1. **Comprehensive testing**: Test all query types before deployment
2. **Feature flags**: Allow disabling new features via environment variables
3. **Monitoring**: Track performance and error rates closely
4. **Rollback plan**: Quick revert capability if issues arise

### Quality Assurance
1. **Unit tests**: For all new methods
2. **Integration tests**: For complete pipeline
3. **Performance tests**: Response time validation
4. **User acceptance tests**: Real-world query scenarios

---

## Expected Outcomes

### Immediate Improvements
- **Comparison queries work correctly**: "active vs reactive power" returns distinct concepts
- **Citation grounding succeeds**: No more UngroundedReplyError for comparisons
- **Better semantic understanding**: Embeddings enhance all responses

### Long-term Benefits
- **Higher response quality**: Semantic context improves relevance
- **Better conversation flow**: Context expansion improves follow-ups
- **Increased user satisfaction**: More accurate and helpful responses
- **Reduced human review needs**: Higher confidence scores

### Success Metrics
- **Comparison query success rate**: 95%+ (currently ~10%)
- **Citation extraction success**: 100% (currently fails for comparisons)
- **Response quality improvement**: Measurable via user feedback
- **Semantic enhancement activation**: 100% when embedding_agent available

---

This implementation plan balances immediate bug fixes with strategic embedding enhancement while maintaining system safety and performance requirements.